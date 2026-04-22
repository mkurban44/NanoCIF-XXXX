#!/usr/bin/env python3
"""
NanoCIF Generation & Evaluation
Usage: python3 13_generate_evaluate.py --base . --n_samples 100
Requirements: pip install torch tokenizers ase
"""
import argparse
import json
import re
import numpy as np
from pathlib import Path
from collections import Counter

import torch


def load_model(model_dir, device):
    """Load trained model and tokenizer."""
    from tokenizers import Tokenizer

    checkpoint = torch.load(model_dir / "best_model.pt", map_location=device)
    config = checkpoint['config']

    # Import model class
    from importlib.util import spec_from_file_location, module_from_spec
    # We need the model class — inline it here for simplicity
    import torch.nn as nn

    class CausalSelfAttention(nn.Module):
        def __init__(self, n_embd, n_head, block_size, dropout=0.1):
            super().__init__()
            self.n_head = n_head
            self.head_dim = n_embd // n_head
            self.qkv = nn.Linear(n_embd, 3 * n_embd)
            self.proj = nn.Linear(n_embd, n_embd)
            self.attn_drop = nn.Dropout(dropout)
            self.proj_drop = nn.Dropout(dropout)
            self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size))
                                 .view(1, 1, block_size, block_size))

        def forward(self, x):
            B, T, C = x.shape
            qkv = self.qkv(x).reshape(B, T, 3, self.n_head, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            att = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
            att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
            att = torch.softmax(att, dim=-1)
            att = self.attn_drop(att)
            y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
            return self.proj_drop(self.proj(y))

    class Block(nn.Module):
        def __init__(self, n_embd, n_head, block_size, dropout=0.1):
            super().__init__()
            self.ln1 = nn.LayerNorm(n_embd)
            self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
            self.ln2 = nn.LayerNorm(n_embd)
            self.mlp = nn.Sequential(
                nn.Linear(n_embd, 4 * n_embd), nn.GELU(),
                nn.Linear(4 * n_embd, n_embd), nn.Dropout(dropout))

        def forward(self, x):
            x = x + self.attn(self.ln1(x))
            x = x + self.mlp(self.ln2(x))
            return x

    class NanoCIFGPT(nn.Module):
        def __init__(self, vocab_size, block_size, n_embd=256, n_head=8, n_layer=6, dropout=0.1):
            super().__init__()
            self.block_size = block_size
            self.tok_emb = nn.Embedding(vocab_size, n_embd)
            self.pos_emb = nn.Embedding(block_size, n_embd)
            self.drop = nn.Dropout(dropout)
            self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
            self.ln_f = nn.LayerNorm(n_embd)
            self.head = nn.Linear(n_embd, vocab_size, bias=False)
            self.tok_emb.weight = self.head.weight

        def forward(self, idx, targets=None):
            B, T = idx.shape
            pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)
            x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
            x = self.blocks(x)
            x = self.ln_f(x)
            logits = self.head(x)
            loss = None
            if targets is not None:
                loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=0)
            return logits, loss

        @torch.no_grad()
        def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, eos_token=2):
            for _ in range(max_new_tokens):
                idx_cond = idx[:, -self.block_size:]
                logits, _ = self(idx_cond)
                logits = logits[:, -1, :] / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float('-inf')
                probs = torch.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                idx = torch.cat((idx, idx_next), dim=1)
                if idx_next.item() == eos_token:
                    break
            return idx

    model = NanoCIFGPT(**config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Model loaded (epoch {checkpoint['epoch']}, val_loss={checkpoint['val_loss']:.4f})")

    return model, config


def parse_nanocif(text):
    """Parse a NanoCIF string and return metadata + coordinates."""
    result = {
        'valid': False,
        'formula': None,
        'cls': None,
        'radius': None,
        'natoms_declared': None,
        'elements_declared': [],
        'atoms': [],
        'coords': [],
    }

    lines = text.strip().split('\n')
    in_loop = False

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.startswith('data_'):
            result['formula'] = line[5:]
        elif line.startswith('_class '):
            result['cls'] = line[7:]
        elif line.startswith('_radius '):
            try:
                result['radius'] = int(line[8:])
            except:
                pass
        elif line.startswith('_natoms '):
            try:
                result['natoms_declared'] = int(line[8:])
            except:
                pass
        elif line.startswith('_elements '):
            result['elements_declared'] = line[10:].split()
        elif line.startswith('loop_'):
            in_loop = True
        elif line.startswith('_atom_type'):
            continue
        elif in_loop:
            parts = line.split()
            if len(parts) >= 4:
                try:
                    el = parts[0]
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    result['atoms'].append(el)
                    result['coords'].append([x, y, z])
                except (ValueError, IndexError):
                    continue

    # Validation
    natoms_actual = len(result['atoms'])
    if (result['formula'] and
        result['cls'] in ['perovskite', 'heusler', 'hydride'] and
        result['radius'] in [5, 6] and
        natoms_actual >= 3):
        result['valid'] = True

    result['natoms_actual'] = natoms_actual
    result['coords'] = np.array(result['coords']) if result['coords'] else np.array([])

    return result


def compute_structural_metrics(parsed):
    """Compute structural plausibility metrics."""
    if not parsed['valid'] or len(parsed['coords']) < 2:
        return None

    coords = parsed['coords']
    n = len(coords)

    # Nearest-neighbour distances
    from scipy.spatial.distance import cdist
    dists = cdist(coords, coords)
    np.fill_diagonal(dists, np.inf)
    nn_dists = dists.min(axis=1)

    # Radius of gyration
    com = coords.mean(axis=0)
    rg = np.sqrt(np.mean(np.sum((coords - com) ** 2, axis=1)))

    return {
        'n_atoms': n,
        'nn_mean': nn_dists.mean(),
        'nn_min': nn_dists.min(),
        'nn_max': nn_dists.max(),
        'nn_std': nn_dists.std(),
        'rg': rg,
        'physically_valid': nn_dists.min() > 0.8,  # no atomic overlap
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=".")
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--mode", default="unconditional",
                        choices=["unconditional", "conditional"])
    parser.add_argument("--condition_class", default=None,
                        help="Material class for conditional generation")
    parser.add_argument("--condition_formula", default=None,
                        help="Formula for conditional generation")
    args = parser.parse_args()

    base = Path(args.base)
    model_dir = base / "model"
    nanocif_dir = base / "nanocif"
    gen_dir = base / "generated"
    gen_dir.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load tokenizer
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(nanocif_dir / "tokenizer.json"))
    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")

    # Load model
    model, config = load_model(model_dir, device)

    # Load test set for comparison
    test_text = (nanocif_dir / "test.txt").read_text()
    test_structures = [s.strip() for s in test_text.split("\n\n") if s.strip()]
    test_parsed = [parse_nanocif(s) for s in test_structures]

    print(f"\n{'='*60}")
    print(f"Generating {args.n_samples} structures...")
    print(f"Mode: {args.mode}, T={args.temperature}, top_k={args.top_k}")
    print(f"{'='*60}")

    generated_texts = []
    generated_parsed = []

    for i in range(args.n_samples):
        # Build prompt
        if args.mode == "conditional" and args.condition_class:
            prompt = f"data_"
            if args.condition_formula:
                prompt = f"data_{args.condition_formula}\n_class {args.condition_class}"
            else:
                prompt = f"data_\n_class {args.condition_class}"
            prompt_tokens = tokenizer.encode(prompt).ids
            input_ids = [bos_id] + prompt_tokens
        else:
            input_ids = [bos_id]

        idx = torch.tensor([input_ids], dtype=torch.long, device=device)
        output = model.generate(idx, max_new_tokens=config['block_size'] - len(input_ids),
                               temperature=args.temperature, top_k=args.top_k, eos_token=eos_id)

        text = tokenizer.decode(output[0].tolist())
        # Clean up
        text = text.replace("<bos>", "").replace("<eos>", "").replace("<pad>", "").strip()
        generated_texts.append(text)

        parsed = parse_nanocif(text)
        generated_parsed.append(parsed)

        if (i + 1) % 20 == 0:
            n_valid = sum(1 for p in generated_parsed if p['valid'])
            print(f"  Generated {i+1}/{args.n_samples} | Valid: {n_valid}/{i+1}")

    # ================================================================
    # EVALUATION
    # ================================================================
    print(f"\n{'='*60}")
    print("EVALUATION")
    print(f"{'='*60}")

    # 1. Validity rate
    n_valid = sum(1 for p in generated_parsed if p['valid'])
    print(f"\n1. Validity")
    print(f"   Parseable & valid: {n_valid}/{args.n_samples} ({100*n_valid/args.n_samples:.1f}%)")

    valid_parsed = [p for p in generated_parsed if p['valid']]

    # 2. Class distribution
    print(f"\n2. Class distribution")
    cls_counts = Counter(p['cls'] for p in valid_parsed)
    for cls, count in sorted(cls_counts.items()):
        print(f"   {cls}: {count} ({100*count/max(len(valid_parsed),1):.1f}%)")

    # 3. Structural plausibility
    print(f"\n3. Structural plausibility")
    metrics = [compute_structural_metrics(p) for p in valid_parsed]
    metrics = [m for m in metrics if m is not None]

    if metrics:
        nn_means = [m['nn_mean'] for m in metrics]
        nn_mins = [m['nn_min'] for m in metrics]
        rgs = [m['rg'] for m in metrics]
        n_physical = sum(1 for m in metrics if m['physically_valid'])

        print(f"   Physically valid (d_min > 0.8 Å): {n_physical}/{len(metrics)} ({100*n_physical/len(metrics):.1f}%)")
        print(f"   NN mean: {np.mean(nn_means):.3f} ± {np.std(nn_means):.3f} Å")
        print(f"   NN min:  {np.mean(nn_mins):.3f} ± {np.std(nn_mins):.3f} Å")
        print(f"   Rg:      {np.mean(rgs):.3f} ± {np.std(rgs):.3f} Å")

    # 4. Uniqueness
    print(f"\n4. Uniqueness")
    unique_formulas = set(p['formula'] for p in valid_parsed if p['formula'])
    print(f"   Unique formulas: {len(unique_formulas)}/{len(valid_parsed)}")

    # 5. Novelty (not in training set)
    train_text = (nanocif_dir / "train.txt").read_text()
    train_structures = [s.strip() for s in train_text.split("\n\n") if s.strip()]
    train_formulas = set()
    for s in train_structures:
        for line in s.split('\n'):
            if line.startswith('data_'):
                train_formulas.add(line[5:])

    novel = [f for f in unique_formulas if f not in train_formulas]
    print(f"   Novel formulas (not in train): {len(novel)}/{len(unique_formulas)}")

    # 6. Comparison with test set
    print(f"\n5. Comparison with test set")
    test_metrics = [compute_structural_metrics(p) for p in test_parsed if p['valid']]
    test_metrics = [m for m in test_metrics if m is not None]
    if test_metrics and metrics:
        print(f"   {'Metric':<20s} {'Generated':>12s} {'Test set':>12s}")
        print(f"   {'─'*44}")
        print(f"   {'⟨d_NN⟩ (Å)':<20s} {np.mean(nn_means):>12.3f} {np.mean([m['nn_mean'] for m in test_metrics]):>12.3f}")
        print(f"   {'d_NN,min (Å)':<20s} {np.mean(nn_mins):>12.3f} {np.mean([m['nn_min'] for m in test_metrics]):>12.3f}")
        print(f"   {'Rg (Å)':<20s} {np.mean(rgs):>12.3f} {np.mean([m['rg'] for m in test_metrics]):>12.3f}")
        print(f"   {'N_atoms':<20s} {np.mean([m['n_atoms'] for m in metrics]):>12.1f} {np.mean([m['n_atoms'] for m in test_metrics]):>12.1f}")

    # Save generated structures
    out_path = gen_dir / "generated_nanocifs.txt"
    out_path.write_text("\n\n".join(generated_texts))
    print(f"\nGenerated structures saved to {out_path}")

    # Save individual valid files
    valid_dir = gen_dir / "valid_nanocifs"
    valid_dir.mkdir(exist_ok=True)
    for i, (text, parsed) in enumerate(zip(generated_texts, generated_parsed)):
        if parsed['valid']:
            fname = f"{parsed['formula']}_R{parsed['radius']}_{i:04d}.nanocif"
            (valid_dir / fname).write_text(text)

    # Save evaluation summary
    summary = {
        'n_generated': args.n_samples,
        'n_valid': n_valid,
        'validity_rate': n_valid / args.n_samples,
        'n_physically_valid': n_physical if metrics else 0,
        'physical_validity_rate': n_physical / len(metrics) if metrics else 0,
        'n_unique_formulas': len(unique_formulas),
        'n_novel_formulas': len(novel),
        'novelty_rate': len(novel) / max(len(unique_formulas), 1),
        'class_distribution': dict(cls_counts),
        'temperature': args.temperature,
        'top_k': args.top_k,
    }
    with open(gen_dir / "evaluation_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Evaluation summary saved to {gen_dir}/evaluation_summary.json")

    print(f"\nDone!")


if __name__ == "__main__":
    main()
