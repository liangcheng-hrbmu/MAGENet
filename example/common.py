"""Shared input and runtime handling for the examples."""
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch

from model.checkpoint import load_model
from model.data import validate_xy

ROOT = Path(__file__).resolve().parents[1]


def device_name(value):
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if value.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; use --device cpu")
    return value


def load_expression(path):
    with np.load(path, allow_pickle=False) as a:
        data = {k: a[k].copy() for k in a.files}
    validate_xy(data["x"], data["y"], len(data["genes"]))
    return data


def analysis_input(checkpoint, expression, cells, seed, device):
    model, saved = load_model(checkpoint, device=device, dtype=torch.float64)
    data = load_expression(expression)
    if data["genes"].astype(str).tolist() != model.graph.genes:
        raise ValueError("Expression genes/order differ from the checkpoint")
    pool = data["test_index"]
    if cells < 2 or cells % 2:
        raise ValueError("--cells must be an even number of at least two")
    rng = np.random.default_rng(seed)
    chosen = []
    for label in (0, 1):
        candidates = pool[data["y"][pool] == label]
        if len(candidates) < cells // 2:
            raise ValueError("Not enough test cells for balanced analysis")
        chosen.extend(rng.choice(candidates, cells // 2, replace=False))
    index = np.sort(chosen)
    scaler = saved["scaler"]
    x = (data["x"][index].astype(np.float64) - scaler["mean"]) / scaler["sd"]
    return model, x, data["y"][index], data["cell_ids"][index]
