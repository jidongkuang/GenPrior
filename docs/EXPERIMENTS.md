# Tables 1-3 Experiments

This guide describes the GenPrior release protocol. The YAML files contain all
run-specific settings so that model selection and evaluation remain explicit.

## Environment

```bash
conda env create -f environment.yml
conda activate genprior
cd genprior/models/shiftgcn/temporal_shift/cuda
python setup.py build_ext --inplace
cd ../../../../../
```

The reference software versions are recorded in `environment.yml`. A CUDA 11.8
compatible driver and CUDA compiler are required for the Shift-GCN extension.
All commands below assume execution from the repository root.

## Required Assets

The source repository intentionally excludes large and third-party assets.
Download URLs and SHA256 checksums are forthcoming; do not substitute files
from unrelated experiments.

```text
skeleton_features/
  synse_features/
    ntu60_seen55_unseen5.npz
    ntu60_seen48_unseen12.npz
    ntu120_seen110_unseen10.npz
    ntu120_seen96_unseen24.npz
  sa_dvae_features/
    ntu60_seen55_unseen5_split{1,2,3}.npz
    ntu120_seen110_unseen10_split{1,2,3}.npz
    pku51_seen46_unseen5_split{1,2,3}.npz
data/
  generated_motions.pkl
checkpoints/
  skeleton_encoder.pt
  table1/
    {ntu60_55_5,ntu60_48_12,ntu120_110_10,ntu120_96_24}/best_gpr.pt
  table3/
    {ntu60_55_5,ntu60_48_12,ntu120_110_10,ntu120_96_24}/best_gzsl.pt
```

The text encoder must resolve from each config's `text_encoder_name_or_path`
value.

## Common Command

An individual training run uses:

```bash
python train.py \
  --config configs/table1/ntu60_xsub_unseen5.yaml \
  --device 0 \
  --output-dir outputs/example
```

Command-line values override YAML values. Each output directory contains a log
and the selected checkpoints. ZSL training saves `best_base.pt` and
`best_gpr.pt`; GZSL training saves `best_gzsl.pt`. Existing output directories
are not reused by the training scripts.


## Table 1: Fixed-Split ZSL

```bash
bash scripts/run_table1.sh 0 outputs/table1
```

| Config | Paper accuracy |
|---|---:|
| `configs/table1/ntu60_xsub_unseen5.yaml` | 87.98 |
| `configs/table1/ntu60_xsub_unseen12.yaml` | 73.43 |
| `configs/table1/ntu120_xsub_unseen10.yaml` | 90.44 |
| `configs/table1/ntu120_xsub_unseen24.yaml` | 81.39 |

## Table 2: Random-Split ZSL

```bash
bash scripts/run_table2.sh 0 outputs/table2
```

| Dataset | Paper split values | Paper mean |
|---|---|---:|
| NTU RGB+D 60 | 94.31, 97.81, 96.81 | 96.3 |
| NTU RGB+D 120 | 82.49, 86.95, 84.69 | 84.7 |
| PKU-MMD | 57.61, 80.26, 86.20 | 74.7 |

The script trains all nine YAML configurations sequentially. Compute each
dataset mean from the three split results.

## Table 3: Generalized ZSL

```bash
bash scripts/run_table3.sh 0 outputs/table3
```

The script trains a GZSL checkpoint for each fixed split and then runs
`scripts/evaluate_gzsl.py` with the procedure defined by its `configs/table3/`
YAML. The evaluator always applies GPR.

| Split | Evaluation procedure | Paper S / U / H |
|---|---|---:|
| NTU RGB+D 60, 55 / 5 | calibration-split round 2 | 73.5 / 80.0 / 76.6 |
| NTU RGB+D 60, 48 / 12 | full-set round 1 | 65.5 / 61.4 / 63.4 |
| NTU RGB+D 120, 110 / 10 | full-set round 1 | 67.1 / 75.2 / 70.9 |
| NTU RGB+D 120, 96 / 24 | full-set round 1 | 56.5 / 55.4 / 55.9 |

These choices are explicit in the Table 3 configurations. In particular,
NTU60 55 / 5 uses the evaluator's calibration-based split, refines
the two subsets separately, and merges their second-round predictions. The
other three splits report first-round GPR on the full test set.

## Reporting Checklist

For each reported run, retain:

- the complete command and YAML snapshot;
- seed, GPU model, CUDA, PyTorch, and Transformers versions;
- checkpoint iteration and SHA256;
- reported metrics from each test-set evaluation; and
- GPR support ratio, prior fusion weight, and GZSL calibration factor.

Expected numerical variation should be assessed only after the official asset
manifest is available. The VAE path samples latent noise, so deterministic
flags and seed recording reduce but may not eliminate GPU-level variation.
