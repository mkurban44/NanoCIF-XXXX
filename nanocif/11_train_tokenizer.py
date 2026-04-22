#!/usr/bin/env python3
"""
NanoCIF Tokenizer Training
Usage: python3 11_train_tokenizer.py --base .
Requirements: pip install tokenizers
"""
import argparse
from pathlib import Path
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=".")
    parser.add_argument("--vocab_size", type=int, default=512,
                        help="Vocabulary size (small corpus → small vocab)")
    args = parser.parse_args()

    base = Path(args.base)
    nanocif_dir = base / "nanocif"
    corpus_path = nanocif_dir / "all.txt"

    if not corpus_path.exists():
        print(f"ERROR: {corpus_path} not found. Run 10_convert_nanocif.py first.")
        return

    print(f"Training BPE tokenizer on {corpus_path}")
    print(f"Vocab size: {args.vocab_size}")

    # Initialize BPE tokenizer
    tokenizer = Tokenizer(models.BPE())

    # Pre-tokenizer: split on whitespace and newlines
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # Decoder
    tokenizer.decoder = decoders.ByteLevel()

    # Post-processor
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    # Special tokens
    special_tokens = ["<pad>", "<bos>", "<eos>", "<unk>"]

    # Trainer
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=special_tokens,
        min_frequency=2,
        show_progress=True,
    )

    # Train
    tokenizer.train([str(corpus_path)], trainer)

    # Save
    tok_path = nanocif_dir / "tokenizer.json"
    tokenizer.save(str(tok_path))
    print(f"\nTokenizer saved to {tok_path}")

    # Stats
    print(f"Vocab size: {tokenizer.get_vocab_size()}")

    # Test encoding
    test_text = """data_BaTiO3
_class perovskite
_radius 6
_natoms 54
_elements Ba Ti O
_composition Ba:0.20 Ti:0.20 O:0.60
loop_
_atom_type _x _y _z
Ba 0.00 0.00 0.00
Ti -1.95 0.00 0.00
O -0.97 -0.97 0.00"""

    encoded = tokenizer.encode(test_text)
    print(f"\nTest encoding:")
    print(f"  Text length: {len(test_text)} chars")
    print(f"  Token count: {len(encoded.ids)}")
    print(f"  Tokens/char ratio: {len(encoded.ids)/len(test_text):.2f}")
    print(f"  First 20 tokens: {encoded.ids[:20]}")
    print(f"  Decoded back: {tokenizer.decode(encoded.ids)[:200]}...")

    # Full corpus stats
    full_text = corpus_path.read_text()
    structures = full_text.split("\n\n")
    token_lengths = []
    for s in structures:
        if s.strip():
            enc = tokenizer.encode(s.strip())
            token_lengths.append(len(enc.ids))

    import numpy as np
    token_lengths = np.array(token_lengths)
    print(f"\nCorpus token statistics:")
    print(f"  Structures: {len(token_lengths)}")
    print(f"  Mean tokens: {token_lengths.mean():.0f}")
    print(f"  Min tokens: {token_lengths.min()}")
    print(f"  Max tokens: {token_lengths.max()}")
    print(f"  Total tokens: {token_lengths.sum():,}")
    print(f"  p95 tokens: {np.percentile(token_lengths, 95):.0f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
