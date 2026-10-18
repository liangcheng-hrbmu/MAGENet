"""Prediction and classification metrics."""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence
import numpy as np
import torch
from torch import nn
from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score,
    balanced_accuracy_score, f1_score, log_loss, brier_score_loss)


def classification_metrics(labels: Sequence[int], probability: Sequence[float]) -> Dict[str, float]:
    y = np.asarray(labels, dtype=np.int8)
    p = np.asarray(probability, dtype=np.float64)
    prediction = p >= 0.5
    return {
        "auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
    }


@torch.no_grad()
def predict(
    model: nn.Module,
    x: Any,
    y: np.ndarray,
    batch_size: int,
    *,
    kwargs: Optional[Mapping[str, Any]] = None,
    keep: Optional[np.ndarray] = None,
    use_knowledge: bool = False,
) -> Dict[str, np.ndarray]:
    model.eval()
    device = next(model.parameters()).device
    values: Dict[str, list[np.ndarray]] = {"final": [], "knowledge": [], "real": []}
    options = dict(kwargs or {})
    for start in range(0, len(y), int(batch_size)):
        stop = min(start + int(batch_size), len(y))
        bx = torch.as_tensor(np.asarray(x[start:stop], dtype=np.float32), device=device)
        local = dict(options)
        if keep is not None:
            local["metabolite_keep"] = torch.as_tensor(
                keep[start:stop], dtype=torch.float32, device=device
            )
        output = model(bx, return_real=True, **local)
        for name in values:
            value = output[name]
            if value is not None:
                values[name].append(torch.sigmoid(value).cpu().numpy())
    result = {
        name: np.concatenate(parts) if parts else np.asarray([], dtype=np.float32)
        for name, parts in values.items()
    }
    if use_knowledge:
        result["final"] = result["knowledge"]
    return result


def score(labels: np.ndarray, output: Mapping[str, np.ndarray]) -> Dict[str, Any]:
    final = classification_metrics(labels, output["final"])
    real_values = output["real"] if len(output["real"]) else output["knowledge"]
    real = classification_metrics(labels, real_values)
    return {
        "final": final,
        "real": real,
        "J": 0.75 * float(final["auc"]) + 0.25 * float(real["auc"]),
    }
