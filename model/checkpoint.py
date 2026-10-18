from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import torch

from .config import DEFAULT_FLAGS
from .data import KnowledgeGraph
from .builder import build_model

ARCHITECTURE = "MAGENet"


def read_checkpoint(path):
    options = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        options["weights_only"] = False
    return torch.load(path, **options)


def load_model(path, *, graph=None, device="cpu", dtype=torch.float32):
    saved = read_checkpoint(path)
    if saved.get("architecture") != ARCHITECTURE:
        raise ValueError("Checkpoint is not MAGENet")
    if graph is None:
        if "graph" not in saved:
            raise ValueError("Checkpoint requires an explicit ordered graph")
        graph = KnowledgeGraph(**saved["graph"])
    if "graph_fingerprint" in saved and saved["graph_fingerprint"] != graph.fingerprint():
        raise ValueError("Checkpoint and graph identifiers/order differ")
    state = saved["state_dict"]
    for name, mask in (("gm", graph.gm), ("mm", graph.mm)):
        if not np.array_equal(state[name + ".mask"].cpu().numpy(), mask):
            raise ValueError("Checkpoint support differs from the supplied graph: " + name)
    model = build_model(graph, saved["config"], saved.get("flags", DEFAULT_FLAGS),
                        state["gm.source_basis"].cpu().numpy(), state["mm.source_basis"].cpu().numpy())
    model = model.to(dtype=dtype)
    model.load_state_dict(state, strict=True)
    model.virtual_context.peer_support.copy_(model.virtual_context.peer_matrix.gt(0))
    return model.to(device).eval(), saved


def save_model(path, model, config, *, flags=None, scaler=None, **metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(architecture=ARCHITECTURE, config=dict(config),
                    flags=dict(flags or DEFAULT_FLAGS), graph=model.graph.payload(),
                    graph_fingerprint=model.graph.fingerprint(), scaler=scaler,
                    state_dict={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                    **metadata), path)
