# NanoCIF: Autoregressive Generation of Alloy Nanoparticle Structures via Text-Based Representation

Anonymous repository for the NeurIPS 2026 submission.

## Overview

This repository contains the code, data, trained models, and result files needed to reproduce the main experiments in the paper:

**NanoCIF: Autoregressive Generation of Alloy Nanoparticle Structures via Text-Based Representation**

The repository includes:

- NanoCIF conversion scripts
- data augmentation scripts
- tokenizer training
- autoregressive model training
- unconditional generation and evaluation
- DFTB/PTBP post-processing and relaxation
- figure generation scripts
- failure-analysis scripts
- trained model checkpoints
- NanoCIF dataset splits
- generated samples and evaluation summaries

## Repository structure

- `scripts/` — preprocessing, training, generation, post-processing, plotting, and analysis scripts
- `nanocif/` — tokenizer, dataset splits, and NanoCIF files
- `model/` — main model checkpoint and training history
- `model_5x/` — 5x augmentation baseline checkpoint and training history
- `generated/` — main generated samples and post-processing results
- `generated_5x/` — 5x augmentation baseline generation outputs

## Main scripts

- `10_convert_nanocif.py` — convert relaxed nanoparticle structures into NanoCIF format
- `11_train_tokenizer.py` — train the BPE tokenizer
- `12_train_model.py` — train the autoregressive GPT model
- `13_generate_evaluate.py` — generate nanoparticles and compute generation metrics
- `14_postprocess_relax.py` — run DFTB/PTBP post-processing and relaxation
- `15_augment_data.py` — generate rotated training sequences
- `16_plot_neurips_figures.py` — reproduce paper figures
- `17_analyze_failures.py` — analyze post-processing failures and adaptive recovery

## Data

The `nanocif/` directory contains:

- `train`, `val`, `test` — dataset splits
- `all` — full dataset listing
- `train_aug` — augmented training data used for the main model
- `train_aug_5x` — augmented training data used for the 5x baseline
- `tokenizer.json` — trained BPE tokenizer
- `files/` — NanoCIF files for individual nanoparticle structures

## Models

The repository provides:

- `model/best_model.pt` — best checkpoint for the main model
- `model/training_history.json` — training history for the main model
- `model_5x/best_model.pt` — best checkpoint for the 5x augmentation baseline
- `model_5x/training_history.json` — training history for the 5x baseline

Intermediate checkpoints are omitted to keep the repository lightweight.

## Results

The `generated/` and `generated_5x/` directories contain generated NanoCIF samples and summary JSON files for evaluation and post-processing.

Important files include:

- `generated/generated_nanocifs`
- `generated/improved_results.json`
- `generated/postprocess_results.json`
- `generated/retry_timeout_results.json`
- `generated/retry_final_results.json`
- `generated_5x/generated_nanocifs`
- `generated_5x/evaluation_summary.json`

## Reproducing the pipeline

Typical execution order:

```bash
python scripts/10_convert_nanocif.py
python scripts/15_augment_data.py
python scripts/11_train_tokenizer.py
python scripts/12_train_model.py
python scripts/13_generate_evaluate.py
python scripts/14_postprocess_relax.py
python scripts/17_analyze_failures.py
python scripts/16_plot_neurips_figures.py
