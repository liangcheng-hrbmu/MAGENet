"""Compute PCS for every prior-supported gene–metabolite–microbe path."""
import argparse
import json
import time
from pathlib import Path
from .common import ROOT, device_name, analysis_input
import torch
from model.paths import compute_pcs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=ROOT / "outputs/demo/model.pt")
    p.add_argument("--data", type=Path, default=ROOT / "data/example.npz")
    p.add_argument("--output", type=Path)
    p.add_argument("--cells", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    a = p.parse_args()
    torch.set_num_threads(4)
    device = device_name(a.device)
    model, x, y, cell_ids = analysis_input(a.checkpoint, a.data, a.cells, a.seed, device)
    started = time.perf_counter()
    paths, qa = compute_pcs(model, x, y, batch_size=a.batch_size)
    output = a.output or a.checkpoint.parent / "paths"
    output.mkdir(parents=True, exist_ok=True)
    paths.to_csv(output / "pcs_all.tsv.gz", sep="\t", index=False)
    paths.head(100).to_csv(output / "pcs_top100.tsv", sep="\t", index=False)
    qa.update(cells=len(x), cell_ids=cell_ids.tolist(), seed=a.seed, paths=len(paths),
              seconds=time.perf_counter() - started, score="path contribution score")
    (output / "analysis.json").write_text(json.dumps(qa, indent=2) + "\n")
    print(f"Saved {len(paths):,} paths; reconstruction error: {qa['path_sum_max_abs_error']:.3e}")


if __name__ == "__main__":
    main()
