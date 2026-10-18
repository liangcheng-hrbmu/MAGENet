"""Training objectives, optimization and checkpoint selection."""
from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Sequence
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import random
from torch.utils.data import DataLoader, TensorDataset
import time
from pathlib import Path
import pandas as pd
from .config import load_config, DEFAULT_FLAGS
from .data import KnowledgeGraph, load_fold
from .builder import build_model
from .checkpoint import save_model
from .evaluate import predict, score


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _loader(x: Any, y: np.ndarray, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(np.asarray(x, dtype=np.float32)),
        torch.from_numpy(np.asarray(y, dtype=np.float32)),
    )
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        generator=torch.Generator().manual_seed(int(seed)),
        pin_memory=torch.cuda.is_available(),
        num_workers=0,
    )


def _criterion(y: np.ndarray, device: torch.device) -> torch.nn.Module:
    positive = float(np.asarray(y).sum())
    return torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor((len(y) - positive) / positive, device=device)
    )


def _clone_state(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


@torch.no_grad()
def _warm_rms(
    model: nn.Module,
    x: Any,
    y: np.ndarray,
    batch_size: int,
) -> None:
    model.train()
    device = next(model.parameters()).device
    for bx, _ in _loader(x, y, batch_size, False, 1):
        model.update_control_rms(bx.to(device, non_blocking=True))


def deterministic_keep(
    batch: int,
    metabolites: int,
    fraction: float,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    keep = torch.rand(int(batch), int(metabolites), generator=generator) >= float(fraction)
    fully_kept = keep.all(1)
    if bool(fully_kept.any()):
        rows = torch.nonzero(fully_kept, as_tuple=False).flatten()
        columns = torch.arange(len(rows)) % int(metabolites)
        keep[rows, columns] = False
    return keep.float().to(device)


def deterministic_residual_keep(
    batch: int,
    probability: float,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    """Sample an independently reproducible per-cell residual DropPath mask."""
    probability = float(probability)
    if not 0.0 <= probability < 1.0:
        raise ValueError("residual DropPath probability must lie in [0, 1)")
    if probability == 0.0:
        return torch.ones(int(batch), device=device)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    keep = torch.rand(int(batch), generator=generator) >= probability
    return keep.float().to(device)


def masked_reconstruction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    keep: torch.Tensor,
) -> torch.Tensor:
    missing = (1.0 - keep)[:, :, None].expand_as(target)
    error = F.smooth_l1_loss(prediction, target.detach(), reduction="none")
    return (error * missing).sum() / missing.sum().clamp_min(1.0)


def pairwise_auc_surrogate(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    maximum_samples: int,
    temperature: float,
) -> torch.Tensor:
    """Smooth pairwise ranking loss aligned with ROC-AUC selection."""
    positive = logits[labels > 0.5][: int(maximum_samples)]
    negative = logits[labels <= 0.5][: int(maximum_samples)]
    if not len(positive) or not len(negative):
        return logits.new_zeros(())
    margin = (positive[:, None] - negative[None, :]) / float(temperature)
    return F.softplus(-margin).mean()


def class_balanced_counterfactual_gain_loss(
    full_logit: torch.Tensor,
    reference_logit: torch.Tensor,
    labels: torch.Tensor,
    *,
    margin: float,
) -> torch.Tensor:
    """Require a module's counterfactual logit change to support the label.

    The class weights prevent the positive-heavy tuning data from teaching a
    one-directional virtual contribution.  Callers route this auxiliary loss
    only to the module whose counterfactual effect is being specialized.
    """
    if full_logit.shape != reference_logit.shape or full_logit.shape != labels.shape:
        raise ValueError("full_logit, reference_logit, and labels must have one shape")
    if float(margin) < 0.0:
        raise ValueError("margin must be nonnegative")
    positive = labels > 0.5
    count = labels.new_tensor(float(labels.numel()))
    positive_count = positive.sum().to(labels).clamp_min(1.0)
    negative_count = (~positive).sum().to(labels).clamp_min(1.0)
    weights = torch.where(
        positive,
        count / (2.0 * positive_count),
        count / (2.0 * negative_count),
    )
    weights = weights / weights.mean().clamp_min(1e-12)
    signed_gain = (2.0 * labels - 1.0) * (full_logit - reference_logit)
    return (weights * F.relu(float(margin) - signed_gain)).mean()


def message_energy_q_penalty(
    model: nn.Module,
    output: Mapping[str, Any],
    *,
    target: str = "neutral",
    layers: Sequence[str] = ("gm", "mm"),
) -> torch.Tensor:
    """Penalize Full-Q structure in the messages that are actually propagated.

    Capacity orthogonality alone can be bypassed by increasing an encoder row
    or the shared value map.  This statistic includes the source activation,
    value transform, and sample-wise capacity, while remaining independent of
    labels and validation data.
    """
    if target not in ("neutral", "fixed_source_prior"):
        raise ValueError(
            "message_energy_q_target must be 'neutral' or 'fixed_source_prior'"
        )
    selected_layers = tuple(str(value) for value in layers)
    if not selected_layers or not set(selected_layers).issubset({"gm", "mm"}):
        raise ValueError("layers must be a nonempty subset of ('gm', 'mm')")
    auxiliary = output["auxiliary"]
    penalties = []
    for layer, relation, source, relation_aux in (
        ("gm", model.gm, auxiliary["gene_state"], auxiliary["gm"]),
        ("mm", model.mm, auxiliary["full_metabolite_state"], auxiliary["mm"]),
    ):
        if layer not in selected_layers:
            continue
        basis = relation.source_basis
        if basis.shape[1] <= 1:
            continue
        capacity = relation_aux["source_capacity"]
        if capacity.ndim == 1:
            capacity = capacity[None, :].expand(source.shape[0], -1)
        value = relation.value(source) * capacity[:, :, None]
        log_energy = 0.5 * torch.log(
            value.float().square().mean(dim=(0, 2)).clamp_min(1e-8)
        )
        expected = torch.zeros_like(log_energy)
        if target == "fixed_source_prior":
            if not hasattr(relation, "fixed_source_log_capacity"):
                raise ValueError(
                    "fixed_source_prior target requires an opportunity-neutral relation"
                )
            expected = relation.fixed_source_log_capacity.detach().to(log_energy)
        deviation = log_energy - expected
        centered = deviation - deviation.mean()
        standardized = centered / centered.square().mean().sqrt().clamp_min(1e-6)
        coefficients = (
            standardized.double() @ basis[:, 1:].double()
        ) / math.sqrt(float(len(standardized)))
        penalties.append(coefficients.square().mean().to(source))
    if not penalties:
        return next(model.parameters()).new_zeros(())
    return torch.stack(penalties).mean()


def fit_model(
    model: nn.Module,
    fold_data: Any,
    config: Mapping[str, Any],
    seed: int,
) -> tuple[list[Dict[str, Any]], Dict[str, Any]]:
    _set_seed(seed)
    device = next(model.parameters()).device
    selected = {id(value) for value in model.trainable_parameters()}
    for parameter in model.parameters():
        parameter.requires_grad_(id(parameter) in selected)
    groups = model.parameter_groups(
        float(config["base_learning_rate"]),
        source_learning_rate=float(config["source_learning_rate"]),
        edge_learning_rate=float(config["edge_learning_rate"]),
        virtual_learning_rate=float(config["virtual_learning_rate"]),
        residual_learning_rate=float(config["residual_learning_rate"]),
    )
    optimizer = torch.optim.AdamW(groups, weight_decay=float(config["weight_decay"]))
    ema_decay = float(config.get("ema_decay", 0.0))
    if not 0.0 <= ema_decay < 1.0:
        raise ValueError("ema_decay must lie in [0, 1)")
    ema_parameters = {
        name: value.detach().clone()
        for name, value in model.named_parameters() if value.requires_grad
    } if ema_decay > 0.0 else {}
    minimum_lrs = [
        float(config["scheduler_min_lr"]) * float(group["lr"]) / float(config["base_learning_rate"])
        for group in groups
    ]
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=float(config["scheduler_factor"]),
        patience=int(config["scheduler_patience"]),
        threshold=1e-4,
        min_lr=minimum_lrs,
    )
    criterion = _criterion(fold_data.train_y, device)
    model.reset_control_rms()
    _warm_rms(model, fold_data.train_x, fold_data.train_y, int(config["batch_size"]))
    model.freeze_source_references()
    initial = predict(
        model, fold_data.early_x, fold_data.early_y, int(config["batch_size"])
    )
    initial_score = score(fold_data.early_y, initial)
    best_score = float(initial_score["J"])
    best_state = _clone_state(model)
    best_epoch = 0
    stale = 0
    gradient_max = {str(group["name"]): 0.0 for group in groups}
    history: list[Dict[str, Any]] = []
    started = time.perf_counter()
    clean_total = sum(float(config[name]) for name in (
        "clean_final_weight", "clean_knowledge_weight", "clean_real_weight",
        "neutral_knowledge_weight", "masked_knowledge_weight",
    ))

    for epoch in range(1, int(config["epochs"]) + 1):
        model.train()
        components: Dict[str, list[float]] = {
        "objective": [], "final": [], "knowledge": [], "real": [],
            "neutral_knowledge": [], "masked_knowledge": [],
            "masked_consistency": [], "reconstruction": [], "ranking": [],
            "residual_penalty": [],
            "virtual_counterfactual_gain": [],
            "virtual_counterfactual_gain_weight": [],
            "message_energy_q": [],
            "gm_message_energy_q": [],
            "mm_message_energy_q": [],
            "residual_keep_fraction": [],
        }
        for batch_number, (bx, by) in enumerate(_loader(
            fold_data.train_x,
            fold_data.train_y,
            int(config["batch_size"]),
            True,
            seed + epoch,
        )):
            bx = bx.to(device, non_blocking=True)
            by = by.to(device, non_blocking=True)
            if bool(config.get("rms_update_during_training", True)) and batch_number % 4 == 0:
                model.update_control_rms(bx)
            optimizer.zero_grad(set_to_none=True)
            clean = model(bx, return_real=True, return_aux=True)
            residual_keep = deterministic_residual_keep(
                len(bx),
                float(config.get("residual_droppath_probability", 0.0)),
                seed + 300000 * epoch + batch_number,
                device,
            )
            training_final = clean["final"]
            if clean["residual"] is not None and float(
                config.get("residual_droppath_probability", 0.0)
            ) > 0.0:
                keep_probability = 1.0 - float(
                    config["residual_droppath_probability"]
                )
                training_final = clean["knowledge"] + (
                    model.residual_fusion_weight
                    * clean["residual"]
                    * residual_keep
                    / keep_probability
                )
            final_loss = criterion(training_final, by)
            knowledge_loss = criterion(clean["knowledge"], by)
            real_loss = criterion(clean["real"], by)
            neutral_loss = torch.zeros((), device=device)
            if float(config["neutral_knowledge_weight"]) > 0.0:
                neutral = model(
                    bx,
                    include_attention=False,
                    include_virtual=False,
                    include_residual=False,
                    return_real=False,
                )
                neutral_loss = criterion(neutral["knowledge"], by)
            keep = deterministic_keep(
                len(bx),
                model.nodes["metabolites"],
                float(config["mask_fraction"]),
                seed + 100000 * epoch + batch_number,
                device,
            )
            # The residual is deliberately absent from every missing-state task.
            masked = model(
                bx,
                metabolite_keep=keep,
                include_residual=False,
                return_real=False,
                return_aux=True,
            )
            masked_loss = criterion(masked["knowledge"], by)
            consistency = torch.zeros((), device=device)
            if float(config["masked_consistency_weight"]) > 0.0:
                consistency = F.mse_loss(
                    torch.sigmoid(masked["knowledge"]),
                    torch.sigmoid(clean["knowledge"]).detach(),
                )
            reconstruction = torch.zeros((), device=device)
            if model.virtual_enabled:
                if bool(config.get("detached_clean_reconstruction", False)):
                    reconstruction_input = clean["auxiliary"][
                        "metabolite_pre_dropout_state"
                    ].detach()
                    reconstruction_prediction, _ = model.virtual_context.predict(
                        reconstruction_input, keep
                    )
                    reconstruction = masked_reconstruction_loss(
                        reconstruction_prediction, reconstruction_input, keep
                    )
                else:
                    reconstruction = masked_reconstruction_loss(
                        masked["auxiliary"]["virtual"]["virtual_prediction"],
                        clean["auxiliary"]["real_metabolite_state"],
                        keep,
                    )
            objective = (
                float(config["clean_final_weight"]) * final_loss
                + float(config["clean_knowledge_weight"]) * knowledge_loss
                + float(config["clean_real_weight"]) * real_loss
                + float(config["neutral_knowledge_weight"]) * neutral_loss
                + float(config["masked_knowledge_weight"]) * masked_loss
            ) / clean_total
            objective = objective + float(config["masked_consistency_weight"]) * consistency
            ranking = torch.zeros((), device=device)
            if float(config.get("auc_surrogate_weight", 0.0)) > 0.0:
                ranking = (
                    0.75 * pairwise_auc_surrogate(
                        clean["final"], by,
                        maximum_samples=int(config["auc_surrogate_samples"]),
                        temperature=float(config["auc_surrogate_temperature"]),
                    )
                    + 0.25 * pairwise_auc_surrogate(
                        clean["real"], by,
                        maximum_samples=int(config["auc_surrogate_samples"]),
                        temperature=float(config["auc_surrogate_temperature"]),
                    )
                )
                objective = objective + float(config["auc_surrogate_weight"]) * ranking
            if model.virtual_enabled:
                objective = objective + float(config["virtual_reconstruction_weight"]) * reconstruction
            residual_penalty = torch.zeros((), device=device)
            if clean["residual"] is not None:
                residual_penalty = clean["residual"].square().mean()
                objective = objective + float(
                    config.get("residual_logit_penalty_weight", 0.0)
                ) * residual_penalty
            regularization = model.regularization()
            if model.attention_enabled:
                objective = objective + float(config["attention_regularization_weight"]) * regularization["attention"]
            energy_q = torch.zeros((), device=device)
            gm_energy_q = torch.zeros((), device=device)
            mm_energy_q = torch.zeros((), device=device)
            shared_energy_weight = float(
                config.get("message_energy_q_penalty_weight", 0.0)
            )
            gm_registered_weight = config.get("gm_message_energy_q_penalty_weight")
            mm_registered_weight = config.get("mm_message_energy_q_penalty_weight")
            if gm_registered_weight is None and mm_registered_weight is None:
                # Use the shared penalty equally across layers.
                gm_energy_weight = 0.5 * shared_energy_weight
                mm_energy_weight = 0.5 * shared_energy_weight
            else:
                gm_energy_weight = (
                    shared_energy_weight
                    if gm_registered_weight is None
                    else float(gm_registered_weight)
                )
                mm_energy_weight = (
                    shared_energy_weight
                    if mm_registered_weight is None
                    else float(mm_registered_weight)
                )
            if (
                model.attention_enabled
                and model.source_enabled
                and max(gm_energy_weight, mm_energy_weight) > 0.0
            ):
                energy_target = str(
                    config.get("message_energy_q_target", "neutral")
                )
                if gm_energy_weight > 0.0:
                    gm_energy_q = message_energy_q_penalty(
                        model, clean, target=energy_target, layers=("gm",)
                    )
                    objective = objective + gm_energy_weight * gm_energy_q
                if mm_energy_weight > 0.0:
                    mm_energy_q = message_energy_q_penalty(
                        model, clean, target=energy_target, layers=("mm",)
                    )
                    objective = objective + mm_energy_weight * mm_energy_q
                energy_q = 0.5 * (gm_energy_q + mm_energy_q)
            if model.virtual_enabled:
                objective = objective + float(config["virtual_regularization_weight"]) * regularization["virtual"]
            virtual_gain = torch.zeros((), device=device)
            virtual_gain_weight = 0.0
            virtual_gain_gradients: Sequence[Optional[torch.Tensor]] = ()
            registered_virtual_gain_weight = float(
                config.get("virtual_counterfactual_gain_weight", 0.0)
            )
            if model.virtual_enabled and registered_virtual_gain_weight > 0.0:
                warmup_epochs = int(
                    config.get("virtual_counterfactual_gain_warmup_epochs", 1)
                )
                if warmup_epochs < 1:
                    raise ValueError(
                        "virtual_counterfactual_gain_warmup_epochs must be positive"
                    )
                virtual_gain_weight = registered_virtual_gain_weight * min(
                    float(epoch) / float(warmup_epochs), 1.0
                )
                virtual_gain = class_balanced_counterfactual_gain_loss(
                    clean["knowledge"],
                    clean["real"].detach(),
                    by,
                    margin=float(
                        config.get("virtual_counterfactual_gain_margin", 0.01)
                    ),
                )
                virtual_parameters = tuple(
                    value for value in model.virtual_parameters() if value.requires_grad
                )
                virtual_gain_gradients = torch.autograd.grad(
                    virtual_gain_weight * virtual_gain,
                    virtual_parameters,
                    retain_graph=True,
                    allow_unused=True,
                )
            objective.backward()
            if virtual_gain_gradients:
                for parameter, gradient in zip(
                    virtual_parameters, virtual_gain_gradients
                ):
                    if gradient is None:
                        continue
                    if parameter.grad is None:
                        parameter.grad = gradient
                    else:
                        parameter.grad.add_(gradient)
            for group in groups:
                gradients = [
                    value.grad.detach().abs().max()
                    for value in group["params"] if value.grad is not None
                ]
                if gradients:
                    maximum = float(torch.stack(gradients).max().cpu())
                    if not np.isfinite(maximum):
                        raise RuntimeError("Non-finite gradient in %s" % group["name"])
                    gradient_max[str(group["name"])] = max(
                        gradient_max[str(group["name"])] , maximum
                    )
            torch.nn.utils.clip_grad_norm_(
                [value for group in groups for value in group["params"]],
                float(config["gradient_clip_norm"]),
            )
            optimizer.step()
            if ema_parameters:
                with torch.no_grad():
                    for name, value in model.named_parameters():
                        if name in ema_parameters:
                            ema_parameters[name].mul_(ema_decay).add_(
                                value.detach(), alpha=1.0 - ema_decay
                            )
            for name, value in (
                ("objective", objective), ("final", final_loss),
                ("knowledge", knowledge_loss), ("real", real_loss),
                ("neutral_knowledge", neutral_loss),
                ("masked_knowledge", masked_loss),
                ("masked_consistency", consistency),
                ("reconstruction", reconstruction),
                ("ranking", ranking),
                ("residual_penalty", residual_penalty),
                ("virtual_counterfactual_gain", virtual_gain),
                (
                    "virtual_counterfactual_gain_weight",
                    objective.new_tensor(virtual_gain_weight),
                ),
                ("message_energy_q", energy_q),
                ("gm_message_energy_q", gm_energy_q),
                ("mm_message_energy_q", mm_energy_q),
                ("residual_keep_fraction", residual_keep.mean()),
            ):
                components[name].append(float(value.detach().cpu()))

        current_parameters: Dict[str, torch.Tensor] = {}
        if ema_parameters:
            with torch.no_grad():
                for name, value in model.named_parameters():
                    if name in ema_parameters:
                        current_parameters[name] = value.detach().clone()
                        value.copy_(ema_parameters[name])
        early_output = predict(
            model, fold_data.early_x, fold_data.early_y, int(config["batch_size"])
        )
        early = score(fold_data.early_y, early_output)
        evaluated_state = _clone_state(model)
        if current_parameters:
            with torch.no_grad():
                for name, value in model.named_parameters():
                    if name in current_parameters:
                        value.copy_(current_parameters[name])
        scheduler.step(float(early["J"]))
        row: Dict[str, Any] = {
            "epoch": epoch,
            "early_J": float(early["J"]),
            "early_final_auc": float(early["final"]["auc"]),
            "early_real_auc": float(early["real"]["auc"]),
        }
        for name, values in components.items():
            row["train_" + name] = float(np.mean(values))
        for group in optimizer.param_groups:
            row["learning_rate_" + str(group["name"])] = float(group["lr"])
        history.append(row)
        print(
            "  epoch=%d validation_auc=%.4f" % (epoch, early["final"]["auc"]),
            flush=True,
        )
        if float(early["J"]) > best_score + 1e-4:
            best_score = float(early["J"])
            best_state = evaluated_state
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if epoch >= int(config["minimum_epochs"]) and stale >= int(config["patience"]):
            break
    model.load_state_dict(best_state)
    missing_gradient = [name for name, value in gradient_max.items() if value <= 0.0]
    if missing_gradient:
        raise RuntimeError("No nonzero gradient reached: %s" % ", ".join(missing_gradient))
    return history, {
        "best_epoch": int(best_epoch),
        "best_early_J": float(best_score),
        "epochs_ran": int(len(history)),
        "training_seconds": float(time.perf_counter() - started),
        "gradient_abs_max_by_group": gradient_max,
        "joint_optimizer_from_epoch1": True,
        "checkpoint_parameter_ema_decay": ema_decay,
        "frozen_training_stages": 0,
        "trainable_parameter_count": int(sum(
            value.numel() for group in groups for value in group["params"]
        )),
        "total_parameter_count": int(sum(value.numel() for value in model.parameters())),
    }


def train(graph_file, fold_file, output, config=None, device="cpu", seed=1001,
          training_seed=1101, threads=8):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(threads)
    config = load_config(config)
    graph = KnowledgeGraph.load(graph_file)
    fold, arrays = load_fold(fold_file, graph)
    _set_seed(seed)
    model = build_model(graph, config, DEFAULT_FLAGS, arrays["gm_source_basis"],
                        arrays["mm_source_basis"]).to(device)
    history, fit = fit_model(model, fold, config, seed=training_seed)
    save_model(output / "model.pt", model, config, fit=fit, seed=seed,
               training_seed=training_seed, scaler={k: arrays[k] for k in ("mean", "sd")})
    pd.DataFrame(history)[["epoch", "early_final_auc", "early_real_auc"]].rename(
        columns={"early_final_auc": "validation_auc", "early_real_auc": "knowledge_validation_auc"}
    ).to_csv(output / "history.tsv", sep="\t", index=False, float_format="%.4f")
    print("Saved", output / "model.pt")
    return model, fit
