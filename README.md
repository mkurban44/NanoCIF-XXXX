## Scripts

| Script | Description |
|--------|-------------|
| `10_convert_nanocif.py` | Convert relaxed nanoparticle structures into NanoCIF format |
| `11_train_tokenizer.py` | Train the BPE tokenizer (512 vocab) |
| `12_train_model.py` | Train the autoregressive GPT model |
| `13_generate_evaluate.py` | Generate nanoparticles and compute evaluation metrics |
| `14_postprocess_relax.py` | DFTB+/PTBP post-processing and relaxation |
| `15_augment_data.py` | Generate rotated training sequences (SO(3) augmentation) |
| `16_plot_neurips_figures.py` | Reproduce all paper figures |
| `17_analyze_failures.py` | Failure analysis and adaptive post-processing recovery |

## Data

The `nanocif/` directory contains:

- `train.txt`, `val.txt`, `test.txt` — dataset splits (80/10/10 by composition)
- `all.txt` — full dataset (2,488 structures)
- `train_aug.txt` — 10× augmented training data (21,340 sequences)
- `train_aug_5x.txt` — 5× augmented training data
- `tokenizer.json` — trained BPE tokenizer
- `files/` — individual NanoCIF files

## Models

- `model/best_model.pt` — best checkpoint for the main model (10× radial, epoch 42, val_loss 0.965)
- `model/training_history.json` — full training history
- `model_5x/best_model.pt` — 5× augmentation baseline
- `model_5x/training_history.json` — 5× training history

Intermediate checkpoints are omitted to keep the repository lightweight.

## Results

The `generated/` directory contains:

- `random_baseline_results.json` — random coordinate placement baseline results
- `generated_nanocifs.txt` — 500 unconditionally generated structures
- `postprocess_results.json` — baseline DFTB+ relaxation (313/500 converged)
- `improved_results.json` — adaptive timeout recovery (+148 structures)
- `retry_timeout_results.json` — extended timeout retry (+10 structures)
- `retry_final_results.json` — final SCC tuning retry (+25 structures)

Final pipeline: 499/500 converged (99.8%), 482/499 geometrically valid (96.6%).

## Ablations

The `ablations/` directory contains evaluation summaries for ablation experiments reported in Table 1:

- `random_baseline_results.json` — random coordinate placement baseline (10.2% validity)
- `noclass_evaluation_summary.json` — 10× radial, no class label (67.8% validity)
- `random_order_evaluation_summary.json` — 10× augmentation, random atom ordering (68.0% validity)

## Reproducing the Pipeline

```bash
# 1. Convert dataset to NanoCIF
python scripts/10_convert_nanocif.py --base .

# 2. Augment training data (10× SO(3) rotations)
python scripts/15_augment_data.py --base . --n_aug 10

# 3. Train tokenizer
python scripts/11_train_tokenizer.py --base .

# 4. Train model (requires GPU)
python scripts/12_train_model.py --base . --epochs 50 --batch_size 8

# 5. Generate 500 structures
python scripts/13_generate_evaluate.py --base . --n_samples 500

# 6. DFTB+ post-processing (requires DFTB+/PTBP)
python scripts/14_postprocess_relax.py --base . --workers 10

# 7. Failure analysis and adaptive recovery
python scripts/17_analyze_failures.py --base . --mode analyze
python scripts/17_analyze_failures.py --base . --mode rerun --workers 10

# 8. Generate figures
python scripts/16_plot_neurips_figures.py --base .
```

## Requirements

- Python 3.9+
- PyTorch 1.13+
- HuggingFace Tokenizers
- NumPy, SciPy, Pandas, Matplotlib
- DFTB+ with PTBP parameter set (for post-processing only)

## Hardware

- Model training: single NVIDIA V100 GPU (16 GB), ~4 hours
- Post-processing: multi-core CPU, ~5 hours for 500 structures
- Training was performed on the TRUBA HPC cluster

## License

Code: MIT | Data: CC-BY-4.0
