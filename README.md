# MAGENet

MAGENet is an interpretable, knowledge-guided neural network for investigating microbe–metabolite–gene associations from immune-cell transcriptomes. It learns expression patterns associated with the clinical immune checkpoint blockade (ICB) response of source patients. Structural attention regulates propagation through a biological knowledge network, while virtual metabolite context and an expression residual branch provide complementary information. Gradient attribution, perturbation analysis and path contribution scores (PCS) identify candidate nodes and associations.

## Installation

Run from the repository root:

```bash
conda env create -f environment.yml
conda activate magenet
```

The examples use a GPU when available. Add `--device cpu` to run on a CPU. The bundled example requires no additional downloads.

## Usage

```bash
# Train and save the selected model.
python -m example.train

# Evaluate the saved model on the test split.
python -m example.test

# Rank genes, metabolites and microbes.
python -m example.analyze_nodes

# Score candidate microbe–metabolite–gene paths.
python -m example.analyze_paths
```

Training writes `outputs/demo/model.pt` and `history.tsv`. The checkpoint includes the model, biological network and training-derived normalization. Testing writes predictions and metrics to `outputs/demo/test/`. Node analysis writes three ranking tables to `outputs/demo/nodes/`. Path analysis writes the complete PCS table and its top 100 entries to `outputs/demo/paths/`.

Node and path examples use small, balanced subsets of test cells; increase `--cells` for a larger analysis. Node scores describe predictive contributions, and PCS ranks knowledge-supported associations by their response-group contribution contrast. Metabolite and microbial representations are inferred from the host expression input and prior network.

To create another run, use `--output outputs/my_run` when training and `--checkpoint outputs/my_run/model.pt` in subsequent commands. Training requires a new output directory. Each command provides `--help`.

## Example data

`data/example.npz` contains 8,000 BCC immune cells from GSE123813: 4,000 from ICB responders and 4,000 from non-responders. The fixed training, validation and test splits contain 5,600, 1,200 and 1,200 cells, respectively, with balanced labels in each split. Splitting is at the cell level, with source patients shared across splits.

The example achieved **test AUC 0.8890, accuracy 0.8133 and F1 0.8161**. These metrics describe the bundled cell-level evaluation.

The same file contains ordered node identifiers and binary gene–metabolite and metabolite–microbe masks. See [data/DATASETS.md](data/DATASETS.md) for source and download links.

## Input format

For another dataset, save a NumPy `.npz` file with the following arrays. Gene expression must be normalized before input; training fits the subsequent standardization using training cells only.

| Key | Contents |
|---|---|
| `x` | Numeric expression matrix: cells × genes. |
| `y` | One label per cell: `1` for a responder source patient, `0` for a non-responder. |
| `genes`, `metabolites`, `microbes` | Unique, ordered string identifiers; bundled metabolite IDs are PubChem CIDs. |
| `gm` | Binary mask: genes × metabolites. |
| `mm` | Binary mask: metabolites × microbes. |
| `cell_ids` | One string identifier per cell. |
| `train_index`, `early_index`, `test_index` | Disjoint integer row indices for training, validation and testing. |
| `patient_ids`, `cell_types` | Optional per-cell metadata, included in the example. |

Expression columns and `gm` rows must follow `genes`; `gm` columns and `mm` rows must follow `metabolites`; `mm` columns must follow `microbes`. A mask value of `1` permits a connection and `0` excludes it. Mask values encode association support; propagation coefficients are learned during training.

Pass the same input file to all four commands, for example:

```bash
python -m example.train --data my_data.npz --output outputs/my_run
python -m example.test --data my_data.npz --checkpoint outputs/my_run/model.pt
python -m example.analyze_nodes --data my_data.npz --checkpoint outputs/my_run/model.pt
python -m example.analyze_paths --data my_data.npz --checkpoint outputs/my_run/model.pt
```

## Files

```text
MAGENet/
├── README.md                  Usage and input format
├── environment.yml            Dependencies
├── .gitignore                 Generated-file exclusions
├── data/
│   ├── example.npz            Expression, labels, splits and binary masks
│   └── DATASETS.md            Dataset and association download links
├── model/
│   ├── __init__.py            Model exports
│   ├── network.py             MAGENet architecture
│   ├── builder.py             Model construction
│   ├── components.py          Gene encoding, normalization and readout
│   ├── attention.py           Source capacity and edge allocation
│   ├── context.py             Virtual metabolite context
│   ├── residual.py            Expression residual branch
│   ├── config.py              Internal defaults
│   ├── data.py                Input validation and preprocessing
│   ├── training.py            Optimization and checkpoint selection
│   ├── checkpoint.py          Model saving and loading
│   ├── evaluate.py            Prediction and evaluation
│   ├── attribution.py         Node attribution and ranking
│   └── paths.py               Path messages and PCS
└── example/
    ├── __init__.py            Example package
    ├── common.py              Shared input handling
    ├── train.py               Training example
    ├── test.py                Testing example
    ├── analyze_nodes.py       Node analysis example
    └── analyze_paths.py       Path analysis example
```
