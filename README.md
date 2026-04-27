# NanoCIF: Autoregressive Generation of Alloy Nanoparticle Structures via Text-Based Representation

Anonymous repository for the NeurIPS 2026 submission.

## Overview

This repository contains the code, data, and result files needed to reproduce the main experiments in the paper:

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
- NanoCIF dataset splits
- generated samples and evaluation summaries
- ablation outputs for augmentation, random-order, and no-class variants

## Repository structure

- `scripts/` — preprocessing, training, generation, post-processing, plotting, and analysis scripts
- `nanocif/` — tokenizer, dataset splits, and NanoCIF files
- `model/` — training history for the main model
- `model_5x/` — training history for the 5× augmentation baseline
- `generated/` — main generated samples and post-processing results
- `generated_5x/` — 5× augmentation baseline generation outputs
- `ablations/` — evaluation outputs for additional ablations such as random-order and no-class variants

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

- `train.txt`, `val.txt`, `test.txt` — dataset splits
- `all.txt` — full dataset listing
- `train_aug.txt` — augmented training data used for the main model
- `train_aug_5x.txt` — augmented training data used for the 5× baseline
- `tokenizer.json` — trained BPE tokenizer
- `files/` — NanoCIF files for individual nanoparticle structures

## Models

To keep the repository lightweight, large checkpoint files are omitted. The repository instead provides:

- `model/training_history.json` — training history for the main model
- `model_5x/training_history.json` — training history for the 5× augmentation baseline

The training scripts and data needed to reproduce these models are included in the repository.

## Results

The repository includes generated NanoCIF samples and summary JSON files used for the main paper results, post-processing analysis, and ablations.

Important files include:

- `generated/generated_nanocifs.txt`
- `generated/evaluation_summary.json`
- `generated/postprocess_results.json`
- `generated/improved_results.json`
- `generated/retry_final_results.json`
- `generated/random_baseline_results.json`
- `generated_5x/generated_nanocifs.txt`
- `generated_5x/evaluation_summary.json`
- `ablations/random_order/evaluation_summary.json`
- `ablations/no_class/evaluation_summary.json`

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
