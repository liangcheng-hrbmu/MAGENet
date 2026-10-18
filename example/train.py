"""Train MAGENet using the supplied training and validation splits."""
import argparse
from pathlib import Path
import tempfile
from .common import ROOT, device_name
from model.data import prepare
from model.training import train


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=ROOT / "data/example.npz")
    p.add_argument("--output", type=Path, default=ROOT / "outputs/demo")
    p.add_argument("--device", default="auto")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=1001)
    a = p.parse_args()
    device = device_name(a.device)
    if a.output.exists():
        p.error("Output directory already exists; choose a new run directory")
    with tempfile.TemporaryDirectory(prefix="magenet_") as temporary:
        fold = prepare(a.data, a.data, Path(temporary) / "fold.npz")
        train(a.data, fold, a.output, device=device, seed=a.seed,
              training_seed=a.seed + 100, threads=a.threads)
    print("Next: python -m example.test --data", a.data, "--checkpoint", a.output / "model.pt")


if __name__ == "__main__":
    main()
