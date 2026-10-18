"""Evaluate a saved model on the fixed test cells."""
import argparse
import json
from pathlib import Path
import pandas as pd
from .common import ROOT, device_name, load_expression
import numpy as np
import torch
from model.checkpoint import load_model
from model.evaluate import predict, classification_metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=ROOT / "outputs/demo/model.pt")
    p.add_argument("--data", type=Path, default=ROOT / "data/example.npz")
    p.add_argument("--output", type=Path)
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=128)
    a = p.parse_args()
    torch.set_num_threads(4)
    model, saved = load_model(a.checkpoint, device=device_name(a.device))
    data = load_expression(a.data)
    if data["genes"].astype(str).tolist() != model.graph.genes:
        raise ValueError("Expression genes/order differ from checkpoint")
    index = data["test_index"]
    scaler = saved["scaler"]
    x = ((data["x"][index] - scaler["mean"]) / scaler["sd"]).astype(np.float32)
    y = data["y"][index]
    result = predict(model, x, y, a.batch_size)
    metrics = classification_metrics(y, result["final"])
    metrics.update(n_test=int(len(y)), n_positive=int(y.sum()), threshold=0.5,
                   split_unit="cell", positive_class="source patient is an ICB responder")
    output = a.output or a.checkpoint.parent / "test"
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(dict(cell_id=data["cell_ids"][index], label=y,
                      probability=result["final"], prediction=(result["final"] >= .5).astype(int))).to_csv(
                          output / "predictions.tsv", sep="\t", index=False)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    for key in ("auc", "accuracy", "f1"):
        print(f"{key.upper()}: {metrics[key]:.4f}")
    print("Saved", output)


if __name__ == "__main__":
    main()
