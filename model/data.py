"""Ordered binary priors, expression validation and training-only preprocessing."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import numpy as np
from typing import Any, Dict, Sequence
import torch
from typing import Tuple
from .context import _dense_binary as dense_binary


@dataclass
class KnowledgeGraph:
    genes: list
    metabolites: list
    microbes: list
    gm: np.ndarray
    mm: np.ndarray

    def __post_init__(self):
        for field in ("genes", "metabolites", "microbes"):
            values = [str(v) for v in getattr(self, field)]
            if not values or len(values) != len(set(values)):
                raise ValueError(field + " must contain unique ordered identifiers")
            setattr(self, field, values)
        self.gm = np.asarray(self.gm, dtype=np.float32)
        self.mm = np.asarray(self.mm, dtype=np.float32)
        if self.gm.shape != (len(self.genes), len(self.metabolites)):
            raise ValueError("GM shape does not match ordered labels")
        if self.mm.shape != (len(self.metabolites), len(self.microbes)):
            raise ValueError("MM shape does not match ordered labels")
        for mask in (self.gm, self.mm):
            if not np.isin(mask, [0, 1]).all():
                raise ValueError("Knowledge masks must be binary")
            if (mask.sum(0) == 0).any() or (mask.sum(1) == 0).any():
                raise ValueError("Each graph node must have support in its relation")

    def as_tuple(self):
        return self.genes, self.metabolites, self.microbes, self.gm, self.mm

    def payload(self):
        return {"genes": np.asarray(self.genes), "metabolites": np.asarray(self.metabolites),
                "microbes": np.asarray(self.microbes), "gm": self.gm, "mm": self.mm}

    def fingerprint(self):
        digest = hashlib.sha256(json.dumps([self.genes, self.metabolites, self.microbes]).encode())
        digest.update(self.gm.astype(np.uint8).tobytes())
        digest.update(self.mm.astype(np.uint8).tobytes())
        return digest.hexdigest()

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **self.payload())

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as a:
            return cls(**{k: a[k].copy() for k in ("genes", "metabolites", "microbes", "gm", "mm")})


Q_PROFILES = ("full", "local", "topology", "intercept", "none")


def topology_covariates(gm_mask: Any, mm_mask: Any) -> Dict[str, np.ndarray]:
    """Compute label-free opportunity covariates for both relation layers."""
    gm = (dense_binary(gm_mask) > 0).astype(np.int32)
    mm = (dense_binary(mm_mask) > 0).astype(np.int32)
    if gm.shape[1] != mm.shape[0]:
        raise ValueError("GM targets must equal MM sources")
    complete = gm @ mm
    gene_degree = gm.sum(axis=1).astype(np.float32)
    gene_two_hop = (gm @ mm.sum(axis=1)).astype(np.float32)
    gene_reachability = (complete > 0).sum(axis=1).astype(np.float32)
    metabolite_in = gm.sum(axis=0).astype(np.float32)
    metabolite_out = mm.sum(axis=1).astype(np.float32)
    return {
        "gene_degree": gene_degree,
        "gene_two_hop_paths": gene_two_hop,
        "gene_reachability": gene_reachability,
        "metabolite_in_degree": metabolite_in,
        "metabolite_out_degree": metabolite_out,
        "metabolite_complete_path_count": metabolite_in * metabolite_out,
    }


def orthonormal_basis(columns: list[np.ndarray]) -> np.ndarray:
    """Return an orthonormal basis for intercept and log-scaled covariates."""
    if not columns:
        raise ValueError("At least one covariate column is required")
    observations = len(columns[0])
    vectors = [np.ones(observations, dtype=np.float64)]
    for column in columns:
        values = np.asarray(column, dtype=np.float64)
        if values.shape != (observations,):
            raise ValueError("Every basis covariate must contain one value per source")
        if np.any(values < 0) or not np.isfinite(values).all():
            raise ValueError("Basis covariates must be finite and nonnegative")
        values = np.log1p(values)
        standard_deviation = float(values.std())
        if standard_deviation > 1e-10:
            vectors.append((values - values.mean()) / standard_deviation)
    design = np.column_stack(vectors)
    left, singular, _ = np.linalg.svd(design, full_matrices=False)
    tolerance = max(design.shape) * np.finfo(np.float64).eps * max(float(singular.max()), 1.0)
    rank = max(1, int(np.sum(singular > tolerance)))
    # SVD determines the column space robustly when covariates are duplicated;
    # the final QR gives the explicit Q used by the projection formula.
    q, _ = np.linalg.qr(left[:, :rank], mode="reduced")
    return q.astype(np.float32)


def _normalise_q_profile(value: Any) -> str:
    aliases = {
        "intercept-only": "intercept",
        "intercept_only": "intercept",
        "no_q": "none",
        "no-q": "none",
    }
    profile = aliases.get(str(value).strip().lower(), str(value).strip().lower())
    if profile not in Q_PROFILES:
        raise ValueError("q_profile must be one of %s" % (", ".join(Q_PROFILES),))
    return profile


def _validate_rms(values: Sequence[float], length: int, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32)
    if result.shape != (length,):
        raise ValueError("%s must have shape (%d,)" % (name, length))
    if np.any(result < 0) or not np.isfinite(result).all():
        raise ValueError("%s must be finite and nonnegative" % name)
    return result


def _intercept_basis(n_sources: int) -> np.ndarray:
    # ``orthonormal_basis`` always prepends an intercept.  A constant dummy
    # column is rank-deficient and therefore leaves precisely that intercept.
    return orthonormal_basis([np.zeros(int(n_sources), dtype=np.float32)])


def _profile_basis(columns: Sequence[np.ndarray], n_sources: int) -> np.ndarray:
    if not columns:
        return _intercept_basis(n_sources)
    return orthonormal_basis([np.asarray(column, dtype=np.float32) for column in columns])


def build_q_bases(
    gm_mask: Any,
    mm_mask: Any,
    gene_raw_rms: Sequence[float],
    metabolite_hidden_rms: Sequence[float],
    q_profile: str = "full",
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Build the registered fold-local Q profile for GM and MM sources."""
    profile = _normalise_q_profile(q_profile)
    covariates = topology_covariates(gm_mask, mm_mask)
    gene_rms = _validate_rms(
        gene_raw_rms, len(covariates["gene_degree"]), "gene_raw_rms"
    )
    metabolite_rms = _validate_rms(
        metabolite_hidden_rms,
        len(covariates["metabolite_in_degree"]),
        "metabolite_hidden_rms",
    )
    gm_by_profile = {
        "full": (
            ("degree", covariates["gene_degree"]),
            ("two_hop_paths", covariates["gene_two_hop_paths"]),
            ("reachability", covariates["gene_reachability"]),
            ("raw_expression_rms", gene_rms),
        ),
        "local": (
            ("degree", covariates["gene_degree"]),
            ("raw_expression_rms", gene_rms),
        ),
        "topology": (
            ("degree", covariates["gene_degree"]),
            ("two_hop_paths", covariates["gene_two_hop_paths"]),
            ("reachability", covariates["gene_reachability"]),
        ),
        "intercept": (),
        "none": (),
    }
    mm_by_profile = {
        "full": (
            ("in_degree", covariates["metabolite_in_degree"]),
            ("out_degree", covariates["metabolite_out_degree"]),
            ("complete_path_count", covariates["metabolite_complete_path_count"]),
            ("hidden_metabolite_rms", metabolite_rms),
        ),
        "local": (
            ("in_degree", covariates["metabolite_in_degree"]),
            ("out_degree", covariates["metabolite_out_degree"]),
            ("hidden_metabolite_rms", metabolite_rms),
        ),
        "topology": (
            ("in_degree", covariates["metabolite_in_degree"]),
            ("out_degree", covariates["metabolite_out_degree"]),
            ("complete_path_count", covariates["metabolite_complete_path_count"]),
        ),
        "intercept": (),
        "none": (),
    }
    gm_selected = gm_by_profile[profile]
    mm_selected = mm_by_profile[profile]
    gm_basis = _profile_basis(
        [value for _, value in gm_selected], len(covariates["gene_degree"])
    )
    mm_basis = _profile_basis(
        [value for _, value in mm_selected], len(covariates["metabolite_in_degree"])
    )
    gm64 = gm_basis.astype(np.float64)
    mm64 = mm_basis.astype(np.float64)
    metadata = {
        "q_profile": profile,
        "basis_type": "intercept_plus_log1p_standardized_selected_covariates_qr",
        "gm_columns": [name for name, _ in gm_selected],
        "mm_columns": [name for name, _ in mm_selected],
        "gm_shape": list(gm_basis.shape),
        "mm_shape": list(mm_basis.shape),
        "gm_orthonormal_error": float(
            np.max(np.abs(gm64.T @ gm64 - np.eye(gm_basis.shape[1])))
        ),
        "mm_orthonormal_error": float(
            np.max(np.abs(mm64.T @ mm64 - np.eye(mm_basis.shape[1])))
        ),
        "gene_raw_rms_sha256": hashlib.sha256(gene_rms.tobytes()).hexdigest(),
        "metabolite_hidden_rms_sha256": hashlib.sha256(metabolite_rms.tobytes()).hexdigest(),
    }
    return gm_basis, mm_basis, metadata


@dataclass
class FoldData:
    train_x: np.ndarray
    train_y: np.ndarray
    early_x: np.ndarray
    early_y: np.ndarray


def validate_xy(x, y, genes):
    if x.ndim != 2 or x.shape[1] != genes or len(x) != len(y) or y.ndim != 1:
        raise ValueError("Expected x[cells,genes] and y[cells]")
    if not len(x) or not np.isfinite(x).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("Expression must be finite; labels must be 0/1")


def load_fold(path, graph):
    with np.load(path, allow_pickle=False) as a:
        if "graph_fingerprint" not in a or str(a["graph_fingerprint"]) != graph.fingerprint():
            raise ValueError("Fold and graph identifier/order fingerprints differ")
        values = {k: a[k].copy() for k in a.files}
    for prefix in ("train", "early", "test"):
        if prefix + "_x" in values:
            validate_xy(values[prefix + "_x"], values[prefix + "_y"], len(graph.genes))
            if prefix != "test" and len(np.unique(values[prefix + "_y"])) != 2:
                raise ValueError(prefix + " split must contain both response classes")
    fold = FoldData(**{k: values[k] for k in ("train_x", "train_y", "early_x", "early_y")})
    return fold, values


def fold_bases(train_x, raw_train_rms, graph, batch_size=4096):
    """Build Full-Q covariates using training cells only."""
    gm = graph.gm.copy()
    gm /= gm.sum(1, keepdims=True).clip(min=1.0)
    route = torch.as_tensor(gm)
    total = torch.zeros(len(graph.metabolites), dtype=torch.float64)
    for start in range(0, len(train_x), batch_size):
        block = torch.as_tensor(train_x[start:start + batch_size], dtype=torch.float32)
        total += torch.tanh(block @ route).double().square().sum(0)
    hidden_rms = (total / len(train_x)).sqrt().float().numpy()
    return build_q_bases(graph.gm, graph.mm, raw_train_rms, hidden_rms, q_profile="full")


def prepare(expression_file, graph_file, output):
    graph = KnowledgeGraph.load(graph_file)
    with np.load(expression_file, allow_pickle=False) as a:
        x, y = a["x"].astype(np.float32), a["y"].copy()
        if a["genes"].astype(str).tolist() != graph.genes:
            raise ValueError("Expression gene order must exactly match graph.genes")
        indices = {name: a[name + "_index"].copy() for name in ("train", "early", "test")
                   if name + "_index" in a}
    validate_xy(x, y, len(graph.genes))
    y = y.astype(np.int64)
    if not {"train", "early"}.issubset(indices):
        raise ValueError("Supply train_index and early_index explicitly")
    for name, index in indices.items():
        if (index.ndim != 1 or not np.issubdtype(index.dtype, np.integer)
                or not len(index) or index.min() < 0 or index.max() >= len(x)):
            raise ValueError("Invalid split indices: " + name)
    all_indices = np.concatenate(list(indices.values()))
    if len(np.unique(all_indices)) != len(all_indices):
        raise ValueError("Split indices overlap or contain duplicate cells")
    train = x[indices["train"]]
    mean = train.mean(0).astype(np.float32)
    variance = np.maximum((train * train).mean(0) - mean.astype(np.float64) ** 2, 1e-6)
    sd = np.sqrt(variance).astype(np.float32)
    values = dict(mean=mean, sd=sd, graph_fingerprint=np.asarray(graph.fingerprint()))
    for name, index in indices.items():
        values[name + "_x"] = ((x[index] - mean) / sd).astype(np.float32)
        values[name + "_y"] = y[index]
        values[name + "_index"] = index
    raw_rms = np.sqrt(np.square(train.astype(np.float64)).mean(0)).astype(np.float32)
    gm, mm, _ = fold_bases(values["train_x"], raw_rms, graph)
    values.update(gm_source_basis=gm, mm_source_basis=mm, raw_train_rms=raw_rms)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **values)
    return output
