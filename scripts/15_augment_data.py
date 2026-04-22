#!/usr/bin/env python3
import argparse
import numpy as np
from pathlib import Path

def random_rotation_matrix():
    rng = np.random.default_rng()
    z = rng.standard_normal((3, 3))
    q, r = np.linalg.qr(z)
    d = np.diag(np.sign(np.diag(r)))
    return q @ d

def augment_nanocif(text, precision=2):
    lines = text.strip().split('\n')
    header_lines = []
    atom_lines = []
    in_loop = False
    header_done = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('loop_'):
            in_loop = True
            header_lines.append(line)
            continue
        if stripped.startswith('_atom_type'):
            header_lines.append(line)
            header_done = True
            continue
        if in_loop and header_done and stripped:
            atom_lines.append(stripped)
        else:
            header_lines.append(line)
    if not atom_lines:
        return text
    elements = []
    coords = []
    for aline in atom_lines:
        parts = aline.split()
        if len(parts) >= 4:
            elements.append(parts[0])
            coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if not coords:
        return text
    coords = np.array(coords)
    com = coords.mean(axis=0)
    coords_centered = coords - com
    R = random_rotation_matrix()
    coords_rotated = (R @ coords_centered.T).T
    distances = np.linalg.norm(coords_rotated, axis=1)
    sort_idx = np.argsort(distances)
    elements_sorted = [elements[i] for i in sort_idx]
    coords_sorted = coords_rotated[sort_idx]
    fmt = f"{{}} {{:.{precision}f}} {{:.{precision}f}} {{:.{precision}f}}"
    new_atom_lines = []
    for el, (x, y, z) in zip(elements_sorted, coords_sorted):
        new_atom_lines.append(fmt.format(el, x, y, z))
    return '\n'.join(header_lines + new_atom_lines)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=".")
    parser.add_argument("--n_aug", type=int, default=10)
    parser.add_argument("--precision", type=int, default=2)
    args = parser.parse_args()
    base = Path(args.base)
    nanocif_dir = base / "nanocif"
    train_path = nanocif_dir / "train.txt"
    if not train_path.exists():
        print(f"ERROR: {train_path} not found")
        return
    text = train_path.read_text()
    structures = [s.strip() for s in text.split("\n\n") if s.strip()]
    n_orig = len(structures)
    print(f"Original training structures: {n_orig}")
    augmented = list(structures)
    for i in range(args.n_aug):
        for s in structures:
            aug = augment_nanocif(s, args.precision)
            augmented.append(aug)
        print(f"  Round {i+1}/{args.n_aug} done ({len(augmented)} total)")
    np.random.seed(42)
    np.random.shuffle(augmented)
    out_path = nanocif_dir / "train_aug.txt"
    out_path.write_text("\n\n".join(augmented))
    print(f"\nAugmented: {len(augmented)} structures ({len(augmented)/n_orig:.0f}x)")
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()
