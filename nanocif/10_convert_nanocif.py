#!/usr/bin/env python3
"""
NanoGenLM → NanoCIF Converter
Converts relaxed nanoparticle structures to NanoCIF text format for LLM training.

Usage: python3 10_convert_nanocif.py --base .
Output: nanocif/ directory with .nanocif files + train/val/test splits

NanoCIF Format:
    data_<formula>
    _class <perovskite|heusler|hydride>
    _radius <5|6>
    _natoms <N>
    _elements <El1> <El2> ...
    _composition <El1>:<frac1> <El2>:<frac2> ...
    loop_
    _atom_type _x _y _z
    <El> <x> <y> <z>
    ...
    [atoms sorted by distance from centre of mass]
"""
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter


def read_gen_file(filepath):
    """Read DFTB+ .gen file and return elements and coordinates."""
    with open(filepath, 'r') as f:
        lines = f.readlines()

    first = lines[0].split()
    n_atoms = int(first[0])
    geo_type = first[1].strip()  # C=cluster, S=supercell

    elements = lines[1].split()

    coords = []
    atom_types = []
    for i in range(2, 2 + n_atoms):
        parts = lines[i].split()
        type_idx = int(parts[1]) - 1  # 1-indexed in gen format
        atom_types.append(elements[type_idx])
        coords.append([float(parts[2]), float(parts[3]), float(parts[4])])

    return atom_types, np.array(coords)


def to_nanocif(formula, cls, radius, atom_types, coords, precision=2):
    """Convert a nanoparticle structure to NanoCIF text format."""
    n_atoms = len(atom_types)

    # Centre of mass (unweighted — geometric centre)
    com = coords.mean(axis=0)
    coords_centered = coords - com

    # Sort by distance from centre (radial ordering: core → surface)
    distances = np.linalg.norm(coords_centered, axis=1)
    sort_idx = np.argsort(distances)
    atom_types_sorted = [atom_types[i] for i in sort_idx]
    coords_sorted = coords_centered[sort_idx]

    # Unique elements (sorted by atomic number for consistency)
    from collections import OrderedDict
    unique_elements = list(OrderedDict.fromkeys(atom_types_sorted))

    # Composition fractions
    elem_counts = Counter(atom_types_sorted)
    comp_parts = []
    for el in unique_elements:
        frac = elem_counts[el] / n_atoms
        comp_parts.append(f"{el}:{frac:.2f}")

    # Build NanoCIF string
    lines = []
    lines.append(f"data_{formula}")
    lines.append(f"_class {cls}")
    lines.append(f"_radius {radius}")
    lines.append(f"_natoms {n_atoms}")
    lines.append(f"_elements {' '.join(unique_elements)}")
    lines.append(f"_composition {' '.join(comp_parts)}")
    lines.append("loop_")
    lines.append("_atom_type _x _y _z")

    fmt = f"{{}} {{:.{precision}f}} {{:.{precision}f}} {{:.{precision}f}}"
    for el, (x, y, z) in zip(atom_types_sorted, coords_sorted):
        lines.append(fmt.format(el, x, y, z))

    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=".")
    parser.add_argument("--dataset", default="neurips",
                        choices=["nature_comm", "neurips"],
                        help="Which dataset to convert")
    parser.add_argument("--precision", type=int, default=2,
                        help="Decimal places for coordinates")
    parser.add_argument("--split", default="0.8,0.1,0.1",
                        help="Train/val/test split ratios")
    args = parser.parse_args()

    base = Path(args.base)
    precision = args.precision
    split_ratios = [float(x) for x in args.split.split(',')]

    # Load dataset
    if args.dataset == "nature_comm":
        csv_path = base / "dataset" / "nature_comm_dataset.csv"
    else:
        csv_path = base / "dataset" / "neurips_dataset.csv"

    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found")
        return

    df = pd.read_csv(csv_path)
    print(f"Loaded: {len(df)} records from {csv_path}")

    # Output directory
    outdir = base / "nanocif"
    outdir.mkdir(exist_ok=True)

    # Convert each structure
    nanocif_data = []  # (formula, class, radius, nanocif_text)
    success = 0
    failed = 0

    for idx, row in df.iterrows():
        formula = row['formula']
        cls = row['class']
        radius = int(row['R'])

        # Find geo_end.gen file
        geo_path = base / "nanoparticles" / cls / formula / f"R{radius:02d}" / "geo_end.gen"
        if not geo_path.exists():
            # Try alternative path format
            geo_path = base / "nanoparticles" / cls / formula / f"R0{radius}" / "geo_end.gen"
        if not geo_path.exists():
            failed += 1
            continue

        try:
            atom_types, coords = read_gen_file(geo_path)
            nanocif_text = to_nanocif(formula, cls, radius, atom_types, coords, precision)
            nanocif_data.append({
                'formula': formula,
                'class': cls,
                'radius': radius,
                'text': nanocif_text,
                'n_atoms': len(atom_types),
            })
            success += 1
        except Exception as e:
            print(f"  ERROR: {formula} R={radius}: {e}")
            failed += 1

        if (idx + 1) % 500 == 0:
            print(f"  Processed {idx + 1}/{len(df)}...")

    print(f"\nConverted: {success}, Failed: {failed}")

    # Save individual .nanocif files
    files_dir = outdir / "files"
    files_dir.mkdir(exist_ok=True)
    for item in nanocif_data:
        fname = f"{item['formula']}_R{item['radius']}.nanocif"
        (files_dir / fname).write_text(item['text'])

    print(f"Individual files saved to {files_dir}/ ({len(nanocif_data)} files)")

    # Train/val/test split (by composition, not by file — same formula stays together)
    formulas = sorted(set(item['formula'] for item in nanocif_data))
    np.random.seed(42)
    np.random.shuffle(formulas)

    n = len(formulas)
    n_train = int(n * split_ratios[0])
    n_val = int(n * split_ratios[1])

    train_formulas = set(formulas[:n_train])
    val_formulas = set(formulas[n_train:n_train + n_val])
    test_formulas = set(formulas[n_train + n_val:])

    splits = {'train': [], 'val': [], 'test': []}
    for item in nanocif_data:
        if item['formula'] in train_formulas:
            splits['train'].append(item['text'])
        elif item['formula'] in val_formulas:
            splits['val'].append(item['text'])
        else:
            splits['test'].append(item['text'])

    # Save split files (one nanocif per line, separated by blank line)
    separator = "\n\n"
    for split_name, texts in splits.items():
        out_path = outdir / f"{split_name}.txt"
        out_path.write_text(separator.join(texts))
        print(f"  {split_name}: {len(texts)} structures → {out_path}")

    # Save combined file for tokenizer training
    all_path = outdir / "all.txt"
    all_path.write_text(separator.join(item['text'] for item in nanocif_data))
    print(f"  all: {len(nanocif_data)} structures → {all_path}")

    # Statistics
    print(f"\n{'='*50}")
    print(f"NanoCIF Dataset Statistics")
    print(f"{'='*50}")
    print(f"Total structures: {len(nanocif_data)}")
    print(f"Unique compositions: {len(formulas)}")
    print(f"Train: {len(splits['train'])} ({len(train_formulas)} compositions)")
    print(f"Val:   {len(splits['val'])} ({len(val_formulas)} compositions)")
    print(f"Test:  {len(splits['test'])} ({len(test_formulas)} compositions)")

    # Token length estimate
    lengths = [len(item['text']) for item in nanocif_data]
    print(f"\nNanoCIF text length (chars):")
    print(f"  Mean: {np.mean(lengths):.0f}")
    print(f"  Min:  {np.min(lengths)}")
    print(f"  Max:  {np.max(lengths)}")
    print(f"  Total: {sum(lengths) / 1e6:.2f} M chars")

    # Atom count distribution
    atom_counts = [item['n_atoms'] for item in nanocif_data]
    print(f"\nAtom count:")
    print(f"  Mean: {np.mean(atom_counts):.0f}")
    print(f"  Min:  {np.min(atom_counts)}")
    print(f"  Max:  {np.max(atom_counts)}")

    print(f"\nDone!")


if __name__ == "__main__":
    main()
