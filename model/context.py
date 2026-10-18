from __future__ import annotations
import math
from typing import Any, Dict, Optional
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .components import _logit

def _dense_binary(value: Any) -> np.ndarray:
    if hasattr(value, "toarray"):
        value = value.toarray()
    result = (np.asarray(value) > 0).astype(np.float32)
    if result.ndim != 2:
        raise ValueError("Topology masks must be two-dimensional")
    return result


def build_topology_peer_matrix(
    gm_mask: Any,
    mm_mask: Any,
    *,
    top_k: int = 8,
    shuffle: bool = False,
    seed: int = 1001,
) -> tuple[np.ndarray, Dict[str, Any]]:
    """Build target-specific metabolite peers from binary topology only.

    Common genes and microbes are inverse-degree weighted before cosine
    similarity, so ubiquitous neighbors contribute less to peer selection.
    """
    gm = _dense_binary(gm_mask)
    mm = _dense_binary(mm_mask)
    if gm.shape[1] != mm.shape[0]:
        raise ValueError("GM targets must match MM sources")
    metabolites = gm.shape[1]
    if not 1 <= int(top_k) < metabolites:
        raise ValueError("top_k must be in [1, metabolites - 1]")

    gene_degree = gm.sum(1).clip(min=1.0)
    microbe_degree = mm.sum(0).clip(min=1.0)
    gene_fingerprint = gm.T / np.sqrt(gene_degree)[None, :]
    microbe_fingerprint = mm / np.sqrt(microbe_degree)[None, :]
    fingerprint = np.concatenate([gene_fingerprint, microbe_fingerprint], axis=1)
    norm = np.sqrt(np.square(fingerprint).sum(1, keepdims=True)).clip(min=1e-12)
    normalized = fingerprint / norm
    similarity = normalized @ normalized.T
    np.fill_diagonal(similarity, 0.0)
    similarity = np.maximum(similarity, 0.0)

    peers = np.zeros_like(similarity, dtype=np.float32)
    all_indices = np.arange(metabolites)
    for row in range(metabolites):
        candidates = all_indices[all_indices != row]
        candidate_similarity = similarity[row, candidates]
        positions = np.argpartition(candidate_similarity, -int(top_k))[-int(top_k):]
        selected = candidates[positions]
        values = np.maximum(similarity[row, selected], 1e-8)
        if float(values.sum()) <= 0.0:
            selected = np.asarray(
                [value for value in range(metabolites) if value != row][: int(top_k)],
                dtype=np.int64,
            )
            values = np.ones(len(selected), dtype=np.float32)
        peers[row, selected] = values
    peers /= peers.sum(1, keepdims=True).clip(min=1e-12)

    permutation = np.arange(metabolites)
    if shuffle:
        permutation = np.random.default_rng(int(seed)).permutation(metabolites)
        peers = peers[permutation][:, permutation]
    np.fill_diagonal(peers, 0.0)
    peers /= peers.sum(1, keepdims=True).clip(min=1e-12)
    metadata = {
        "method": "inverse_degree_weighted_topology_cosine_topk",
        "top_k": int(top_k),
        "uses_labels": False,
        "shuffle": bool(shuffle),
        "shuffle_seed": int(seed) if shuffle else None,
        "self_weight_max": float(np.diag(peers).max()),
        "row_sum_max_error": float(np.abs(peers.sum(1) - 1.0).max()),
        "nonzero_min": int((peers > 0).sum(1).min()),
        "nonzero_max": int((peers > 0).sum(1).max()),
    }
    return peers.astype(np.float32), metadata


class TopologyPeerVirtualContext(nn.Module):
    """One self-excluding topology-peer virtual state per real metabolite."""

    def __init__(
        self,
        peer_matrix: Any,
        width: int,
        *,
        maximum_clean_scale: float = 0.20,
        clean_scale_init: float = 0.05,
        clean_mode: str = "multiplicative",
        targetwise_clean_scale: bool = False,
        samplewise_clean_gate: bool = False,
        samplewise_clean_gate_mode: str = "unbounded_logit",
        maximum_clean_gate_ratio: float = 4.0,
        learnable_peer_allocation: bool = False,
        maximum_peer_ratio: float = 4.0,
        samplewise_peer_attention: bool = False,
        peer_attention_strength_init: float = 0.25,
        peer_attention_query_mode: str = "target",
    ) -> None:
        super().__init__()
        peers = torch.as_tensor(np.asarray(peer_matrix), dtype=torch.float32)
        if peers.ndim != 2 or peers.shape[0] != peers.shape[1]:
            raise ValueError("peer_matrix must be square")
        if float(torch.diag(peers).abs().max()) > 0.0:
            raise ValueError("Virtual peers must exclude the target itself")
        if float((peers.double().sum(1) - 1.0).abs().max()) > 1e-6:
            raise ValueError("Virtual peer rows must sum to one")
        if not 0.0 < float(clean_scale_init) < float(maximum_clean_scale):
            raise ValueError("clean_scale_init must lie inside (0, maximum_clean_scale)")
        self.register_buffer("peer_matrix", peers)
        self.register_buffer("peer_support", peers.gt(0).float(), persistent=False)
        self.peer_value = nn.Linear(int(width), int(width), bias=False)
        nn.init.eye_(self.peer_value.weight)
        if clean_mode not in (
            "multiplicative", "convex", "completion_only", "orthogonal_innovation"
        ):
            raise ValueError(
                "clean_mode must be 'multiplicative', 'convex', "
                "'completion_only', or 'orthogonal_innovation'"
            )
        if float(maximum_peer_ratio) <= 1.0:
            raise ValueError("maximum_peer_ratio must exceed one")
        self.clean_mode = str(clean_mode)
        self.targetwise_clean_scale = bool(targetwise_clean_scale)
        self.samplewise_clean_gate = bool(samplewise_clean_gate)
        if str(samplewise_clean_gate_mode) not in (
            "unbounded_logit", "centered_tanh"
        ):
            raise ValueError(
                "samplewise_clean_gate_mode must be 'unbounded_logit' or "
                "'centered_tanh'"
            )
        if float(maximum_clean_gate_ratio) <= 1.0:
            raise ValueError("maximum_clean_gate_ratio must exceed one")
        self.samplewise_clean_gate_mode = str(samplewise_clean_gate_mode)
        self.maximum_clean_gate_ratio = float(maximum_clean_gate_ratio)
        self.maximum_clean_gate_log = 0.5 * math.log(
            self.maximum_clean_gate_ratio
        )
        self.learnable_peer_allocation = bool(learnable_peer_allocation)
        self.samplewise_peer_attention = bool(samplewise_peer_attention)
        if str(peer_attention_query_mode) not in (
            "target", "peer_summary", "missing_peer_summary"
        ):
            raise ValueError(
                "peer_attention_query_mode must be 'target', 'peer_summary', "
                "or 'missing_peer_summary'"
            )
        self.peer_attention_query_mode = str(peer_attention_query_mode)
        scale_shape = (peers.shape[0], int(width)) if targetwise_clean_scale else ()
        self.maximum_clean_scale = float(maximum_clean_scale)
        self.clean_scale_logit = nn.Parameter(torch.full(
            scale_shape,
            _logit(float(clean_scale_init) / float(maximum_clean_scale)),
            dtype=torch.float32,
        ))
        if self.samplewise_clean_gate:
            # Preserve the initialization stream of every downstream module so
            # gated and ungated factorial conditions differ only by this gate.
            with torch.random.fork_rng(devices=[]):
                self.clean_gate = nn.Linear(3 * int(width), 1)
            nn.init.zeros_(self.clean_gate.weight)
            nn.init.zeros_(self.clean_gate.bias)
        else:
            self.clean_gate = None
        if self.learnable_peer_allocation:
            self.peer_logit = nn.Parameter(torch.zeros_like(peers))
        else:
            self.register_buffer("peer_logit", torch.zeros_like(peers), persistent=False)
        self.maximum_peer_log = math.log(float(maximum_peer_ratio))
        if not 0.0 < float(peer_attention_strength_init) < 1.0:
            raise ValueError("peer_attention_strength_init must lie in (0, 1)")
        self.peer_attention_strength_logit = (
            nn.Parameter(torch.tensor(
                _logit(float(peer_attention_strength_init)), dtype=torch.float32
            ))
            if self.samplewise_peer_attention
            else None
        )

    def clean_scale(self) -> torch.Tensor:
        return self.maximum_clean_scale * torch.sigmoid(self.clean_scale_logit)

    def effective_clean_scale(
        self,
        reference: torch.Tensor,
        prediction: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a bounded clean-context scale conditioned on this sample."""
        if self.clean_gate is None:
            scale = self.clean_scale()
            if scale.ndim == 2:
                scale = scale[None, :, :]
            return scale, reference.new_zeros(reference.shape[:2])
        features = torch.cat([
            F.layer_norm(reference, (reference.shape[-1],)),
            F.layer_norm(prediction, (prediction.shape[-1],)),
            F.layer_norm(prediction - reference, (reference.shape[-1],)),
        ], dim=-1)
        gate_score = self.clean_gate(features).squeeze(-1)
        if self.samplewise_clean_gate_mode == "centered_tanh":
            centered = gate_score - gate_score.mean(1, keepdim=True)
            gate_score = self.maximum_clean_gate_log * torch.tanh(centered)
        base_logit = self.clean_scale_logit
        if base_logit.ndim == 2:
            base_logit = base_logit[None, :, :]
        scale = self.maximum_clean_scale * torch.sigmoid(
            base_logit + gate_score[:, :, None]
        )
        return scale, gate_score

    def peer_allocation(self) -> torch.Tensor:
        if not self.learnable_peer_allocation:
            return self.peer_matrix
        residual = self.maximum_peer_log * torch.tanh(self.peer_logit)
        weights = self.peer_matrix * torch.exp(residual) * self.peer_support
        return weights / weights.sum(1, keepdim=True).clamp_min(1e-12)

    def peer_attention_strength(self) -> torch.Tensor:
        if self.peer_attention_strength_logit is None:
            return self.peer_matrix.new_zeros(())
        return torch.sigmoid(self.peer_attention_strength_logit)

    def sample_peer_allocation(
        self,
        real_metabolite: torch.Tensor,
        keep: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select topology-supported peers from the observed state of one cell."""
        observed = real_metabolite * keep[:, :, None].to(real_metabolite)
        weights = self.peer_allocation()[None, :, :] * keep[:, None, :].to(
            real_metabolite
        )
        if self.samplewise_peer_attention:
            if self.peer_attention_query_mode in (
                "peer_summary", "missing_peer_summary"
            ):
                base_route = weights / weights.sum(2, keepdim=True).clamp_min(1e-8)
                query_state = torch.einsum(
                    "bmn,bnd->bmd", base_route, observed
                )
            else:
                query_state = observed
            query = F.normalize(query_state, p=2.0, dim=-1, eps=1e-8)
            key = F.normalize(observed, p=2.0, dim=-1, eps=1e-8)
            similarity = torch.einsum("bmd,bnd->bmn", query, key)
            bounded_log_multiplier = (
                0.5
                * self.maximum_peer_log
                * self.peer_attention_strength()
                * torch.tanh(similarity)
            )
            if self.peer_attention_query_mode == "missing_peer_summary":
                # Preserve the fixed topology route for observed metabolites;
                # use dynamic peer selection only for the completion task.
                bounded_log_multiplier = bounded_log_multiplier * (
                    1.0 - keep[:, :, None].to(bounded_log_multiplier)
                )
            weights = weights * torch.exp(bounded_log_multiplier) * self.peer_support
        normalizer = weights.sum(2, keepdim=True)
        available = normalizer[:, :, 0] > 1e-8
        return weights / normalizer.clamp_min(1e-8), available

    def predict(self, real_metabolite: torch.Tensor, keep: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if keep.shape != real_metabolite.shape[:2]:
            raise ValueError("keep must have shape [batch, metabolites]")
        weights, available = self.sample_peer_allocation(real_metabolite, keep)
        peer = torch.einsum("bmn,bnd->bmd", weights, real_metabolite)
        prediction = torch.tanh(self.peer_value(peer))
        prediction = prediction * available[:, :, None]
        return prediction, available

    def forward(
        self,
        real_metabolite: torch.Tensor,
        keep: Optional[torch.Tensor] = None,
        *,
        enabled: bool = True,
        return_aux: bool = False,
    ):
        if keep is None:
            keep = torch.ones(
                real_metabolite.shape[:2],
                dtype=real_metabolite.dtype,
                device=real_metabolite.device,
            )
        keep = keep.to(real_metabolite)
        observed = real_metabolite * keep[:, :, None]
        if enabled:
            prediction, available = self.predict(real_metabolite, keep)
            scale, gate_score = self.effective_clean_scale(observed, prediction)
            if self.clean_mode == "completion_only":
                modulation = torch.ones_like(real_metabolite)
                full = observed + (1.0 - keep[:, :, None]) * prediction
            elif self.clean_mode == "orthogonal_innovation":
                reference = observed
                coefficient = (prediction * reference).sum(-1, keepdim=True) / (
                    reference.square().sum(-1, keepdim=True).clamp_min(1e-12)
                )
                innovation = prediction - coefficient * reference
                coefficient = (innovation * reference).sum(-1, keepdim=True) / (
                    reference.square().sum(-1, keepdim=True).clamp_min(1e-12)
                )
                innovation = innovation - coefficient * reference
                reference_norm = reference.norm(dim=-1, keepdim=True)
                innovation_norm = innovation.norm(dim=-1, keepdim=True)
                innovation = innovation * torch.minimum(
                    torch.ones_like(innovation_norm),
                    reference_norm / innovation_norm.clamp_min(1e-8),
                ).detach()
                context = keep[:, :, None] * scale * innovation
                modulation = torch.ones_like(real_metabolite)
                full = (
                    observed
                    + context
                    + (1.0 - keep[:, :, None]) * prediction
                )
            elif self.clean_mode == "convex":
                modulation = 1.0 - scale
                full = (
                    observed
                    + keep[:, :, None] * scale * (prediction - observed)
                    + (1.0 - keep[:, :, None]) * prediction
                )
            else:
                modulation = 1.0 + scale * torch.tanh(prediction)
                full = observed * modulation + (1.0 - keep[:, :, None]) * prediction
        else:
            prediction = torch.zeros_like(real_metabolite)
            available = torch.zeros_like(keep, dtype=torch.bool)
            modulation = torch.ones_like(real_metabolite)
            full = observed
            scale = self.clean_scale()
            gate_score = real_metabolite.new_zeros(real_metabolite.shape[:2])
        if not return_aux:
            return full
        return full, {
            "virtual_prediction": prediction,
            "available_peer": available,
            "modulation": modulation,
            "context_effect": full - observed,
            "peer_innovation": (
                innovation
                if enabled and self.clean_mode == "orthogonal_innovation"
                else torch.zeros_like(real_metabolite)
            ),
            "clean_scale": float(self.clean_scale().detach().mean().cpu()),
            "effective_clean_scale": scale,
            "sample_gate_score": gate_score,
            "clean_additive_path": self.clean_mode == "orthogonal_innovation",
        }

    def regularization(self) -> torch.Tensor:
        identity = torch.eye(
            self.peer_value.weight.shape[0],
            dtype=self.peer_value.weight.dtype,
            device=self.peer_value.weight.device,
        )
        prior = self.peer_matrix.clamp_min(1e-12)
        route = self.peer_allocation().clamp_min(1e-12)
        peer_kl = (
            route * (route.log() - prior.log()) * self.peer_support
        ).sum() / self.peer_matrix.shape[0]
        gate_penalty = (
            self.clean_scale_logit.new_zeros(())
            if self.clean_gate is None
            else self.clean_gate.weight.square().mean()
        )
        peer_attention_penalty = (
            self.clean_scale_logit.new_zeros(())
            if self.peer_attention_strength_logit is None
            else self.peer_attention_strength_logit.square()
        )
        return self.clean_scale().square().mean() + 0.001 * gate_penalty + 0.01 * (
            self.peer_value.weight - identity
        ).square().mean() + 0.01 * peer_kl + 1e-5 * peer_attention_penalty

    @torch.no_grad()
    def audit(self) -> Dict[str, Any]:
        route = self.peer_allocation()
        scale = self.clean_scale()
        return {
            "peer_row_sum_max_error": float(
                (route.double().sum(1) - 1.0).abs().max().cpu()
            ),
            "unsupported_peer_weight_max_abs": float(
                (route * (1.0 - self.peer_support)).abs().max().cpu()
            ),
            "self_weight_max_abs": float(torch.diag(route).abs().max().cpu()),
            "minimum_peers": int((self.peer_support > 0).sum(1).min().cpu()),
            "maximum_peers": int((self.peer_support > 0).sum(1).max().cpu()),
            "clean_scale": float(scale.mean().cpu()),
            "clean_scale_min": float(scale.min().cpu()),
            "clean_scale_max": float(scale.max().cpu()),
            "maximum_clean_scale": self.maximum_clean_scale,
            "clean_mode": self.clean_mode,
            "clean_identity": self.clean_mode == "completion_only",
            "targetwise_clean_scale": self.targetwise_clean_scale,
            "samplewise_clean_gate": self.samplewise_clean_gate,
            "samplewise_clean_gate_mode": self.samplewise_clean_gate_mode,
            "maximum_clean_gate_ratio": self.maximum_clean_gate_ratio,
            "learnable_peer_allocation": self.learnable_peer_allocation,
            "samplewise_peer_attention": self.samplewise_peer_attention,
            "peer_attention_strength": float(
                self.peer_attention_strength().detach().cpu()
            ),
            "peer_attention_query_mode": self.peer_attention_query_mode,
            "clean_additive_path": self.clean_mode == "orthogonal_innovation",
            "direct_gene_to_virtual_edges": 0,
            "direct_virtual_to_microbe_edges": 0,
            "derived_metabolite_context_edges": int(
                self.peer_support.sum().cpu()
            ),
            "introduces_topology_derived_metabolite_context_dependencies": True,
        }

    @torch.no_grad()
    def audit_sample(
        self,
        real_metabolite: torch.Tensor,
        keep: torch.Tensor,
    ) -> Dict[str, float]:
        route, available = self.sample_peer_allocation(real_metabolite, keep)
        active_support = (
            self.peer_support[None, :, :] * keep[:, None, :].to(route)
        ).bool()
        expected_sum = available.to(route)
        row_sum_error = (route.double().sum(2) - expected_sum.double()).abs().max()
        unsupported = (route * (~active_support).to(route)).abs().max()
        diagonal = torch.diagonal(route, dim1=1, dim2=2).abs().max()
        prior = self.peer_allocation()[None, :, :].expand_as(route)
        relative = route / prior.clamp_min(1e-12)
        row_maximum = torch.where(
            active_support, relative, torch.zeros_like(relative)
        ).amax(2)
        row_minimum = torch.where(
            active_support, relative, torch.full_like(relative, float("inf"))
        ).amin(2)
        valid = available & torch.isfinite(row_minimum)
        ratio = row_maximum[valid] / row_minimum[valid].clamp_min(1e-12)
        maximum_ratio = float(ratio.max().cpu()) if bool(valid.any()) else 1.0
        ratio_error = max(maximum_ratio - math.exp(self.maximum_peer_log), 0.0)
        return {
            "sample_peer_row_sum_max_error": float(row_sum_error.cpu()),
            "sample_unsupported_peer_weight_max_abs": float(unsupported.cpu()),
            "sample_self_weight_max_abs": float(diagonal.cpu()),
            "sample_peer_multiplier_ratio": maximum_ratio,
            "configured_maximum_sample_peer_multiplier_ratio": math.exp(
                self.maximum_peer_log
            ),
            "sample_peer_multiplier_ratio_bound_error": ratio_error,
        }
