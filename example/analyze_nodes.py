"""Rank gene, metabolite and microbe nodes using gradients and perturbations."""
import argparse
import json
import time
from pathlib import Path
from .common import ROOT, device_name, analysis_input
import numpy as np
import torch
from model.attribution import gene_probability_sensitivity, finite_difference_gene
from model.attribution import gradient, reduce_gradient, perturbation
from model.attribution import consensus_table


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=ROOT / "outputs/demo/model.pt")
    p.add_argument("--data", type=Path, default=ROOT / "data/example.npz")
    p.add_argument("--output", type=Path)
    p.add_argument("--cells", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    a = p.parse_args()
    torch.set_num_threads(4)
    device = device_name(a.device)
    model, x, y, cell_ids = analysis_input(a.checkpoint, a.data, a.cells, a.seed, device)
    bx = torch.as_tensor(x, dtype=torch.float64, device=device)
    by = torch.as_tensor(y, dtype=torch.float64, device=device)
    output = a.output or a.checkpoint.parent / "nodes"
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    gene_score, _, _, qa = gene_probability_sensitivity(model, bx, batch=a.batch_size)
    other = reduce_gradient(model, gradient(model, bx, by, batch=a.batch_size))
    scores = dict(gene=gene_score, metabolite=other["metabolite"], microbe=other["microbe"])
    names = dict(gene=model.graph.genes, metabolite=model.graph.metabolites, microbe=model.graph.microbes)
    for layer in names:
        finite = []
        for i in range(len(names[layer])):
            if layer == "gene":
                value = float(np.abs(finite_difference_gene(model, bx, i, batch=a.batch_size)).mean())
            else:
                value, _ = perturbation(model, bx, layer, i, batch=a.batch_size)
            finite.append(value)
            if (i + 1) % 100 == 0:
                print(f"{layer}: {i + 1}/{len(names[layer])}", flush=True)
        table = consensus_table(names[layer], scores[layer], finite)
        table.to_csv(output / f"{layer}_ranking.tsv", sep="\t", index=False)
        print("Saved", layer, "ranking", flush=True)
    metadata = dict(cells=len(x), cell_ids=cell_ids.tolist(), seed=a.seed,
                    epsilon=1e-5, seconds=time.perf_counter() - started,
                    gene_gradient="mean absolute per-cell probability directional derivative",
                    other_gradient="mean absolute incoming-parameter gradient of mean unweighted BCE",
                    perturbation="mean absolute central finite difference of probability",
                    consensus="mean of descending gradient and perturbation ranks", **qa)
    (output / "analysis.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
