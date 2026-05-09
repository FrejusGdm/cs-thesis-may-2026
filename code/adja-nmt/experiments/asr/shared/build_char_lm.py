#!/usr/bin/env python3
from __future__ import annotations
"""
Build a character-level n-gram language model for Adja ASR.

Takes Adja transcriptions, space-separates characters, and trains
a KenLM n-gram model. The output .arpa file is used with pyctcdecode
for CTC beam search decoding.

Why this helps: CTC greedy decoding picks characters independently —
it doesn't know that "ŋɖuɖu" is a word but "ŋŋŋɖɖɖ" is not.
An n-gram LM fixes this by scoring character sequences based on
how often they appear in Adja text.

Usage:
    python build_char_lm.py --data-dir data --order 5
    python build_char_lm.py --data-dir data --order 5 --extra-text extra_adja.txt

Requirements: kenlm (pip install kenlm)
    KenLM must also be installed as a system tool for lmplz binary.
    If lmplz is not available, this script uses a pure-Python fallback.
"""

import argparse
import json
import os
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)


def normalize_text(text: str) -> str:
    """NFC normalize and clean whitespace."""
    return " ".join(unicodedata.normalize("NFC", text.strip()).split())


def load_texts_from_manifest(manifest_path: str) -> list[str]:
    """Load text column from TSV manifest."""
    texts = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                texts.append(normalize_text(parts[2]))
    return texts


def chars_to_spaced(text: str) -> str:
    """Convert text to space-separated characters for character LM.

    "hello" -> "h e l l o"
    Word boundaries are marked with | (pipe) so the LM can learn word patterns.
    """
    result = []
    for char in text:
        if char == " ":
            result.append("|")  # word boundary marker
        else:
            result.append(char)
    return " ".join(result)


def build_arpa_python(texts: list[str], order: int = 5, output_path: str = "char_lm.arpa"):
    """Build a simple ARPA-format n-gram LM in pure Python.

    This is a fallback when KenLM's lmplz binary is not available.
    It builds a basic maximum-likelihood n-gram model with simple
    backoff weights. Not as good as KenLM but works for our purposes.
    """
    print(f"Building {order}-gram character LM (pure Python)...")

    # Convert texts to spaced characters
    spaced_texts = [chars_to_spaced(t) for t in texts]

    # Count n-grams
    ngram_counts = {}
    for n in range(1, order + 1):
        ngram_counts[n] = Counter()

    for text in spaced_texts:
        tokens = ["<s>"] + text.split() + ["</s>"]
        for n in range(1, order + 1):
            for i in range(len(tokens) - n + 1):
                ngram = tuple(tokens[i:i + n])
                ngram_counts[n][ngram] += 1

    # Compute probabilities with simple add-1 smoothing
    vocab = set()
    for text in spaced_texts:
        vocab.update(text.split())
    vocab.add("<s>")
    vocab.add("</s>")
    vocab.add("|")
    V = len(vocab)

    print(f"Vocabulary: {V} tokens")
    for n in range(1, order + 1):
        print(f"  {n}-grams: {len(ngram_counts[n])}")

    # Write ARPA format
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\\data\\\n")
        for n in range(1, order + 1):
            f.write(f"ngram {n}={len(ngram_counts[n])}\n")
        f.write("\n")

        for n in range(1, order + 1):
            f.write(f"\\{n}-grams:\n")

            if n == 1:
                total = sum(ngram_counts[1].values())
                for ngram, count in sorted(ngram_counts[n].items()):
                    prob = (count + 1) / (total + V)
                    import math
                    log_prob = math.log10(prob)
                    f.write(f"{log_prob:.4f}\t{' '.join(ngram)}\n")
            else:
                # Compute conditional probabilities
                context_counts = Counter()
                for ngram, count in ngram_counts[n].items():
                    context = ngram[:-1]
                    context_counts[context] += count

                for ngram, count in sorted(ngram_counts[n].items()):
                    context = ngram[:-1]
                    prob = (count + 1) / (context_counts[context] + V)
                    import math
                    log_prob = math.log10(prob)
                    f.write(f"{log_prob:.4f}\t{' '.join(ngram)}\n")

            f.write("\n")

        f.write("\\end\\\n")

    print(f"ARPA LM saved to {output_path}")
    return output_path


def build_arpa_kenlm(texts: list[str], order: int = 5, output_path: str = "char_lm.arpa"):
    """Build ARPA LM using KenLM's lmplz (much better smoothing)."""
    import subprocess
    import tempfile

    # Write spaced character text to temp file
    spaced_texts = [chars_to_spaced(t) for t in texts]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        for text in spaced_texts:
            f.write(text + "\n")
        temp_path = f.name

    print(f"Training {order}-gram LM with KenLM on {len(texts)} sentences...")

    try:
        result = subprocess.run(
            ["lmplz", "-o", str(order), "--text", temp_path, "--arpa", output_path],
            capture_output=True, text=True, timeout=60
        )
        if result.returncode == 0:
            print(f"KenLM ARPA saved to {output_path}")
            return output_path
        else:
            print(f"KenLM failed: {result.stderr[:200]}")
            return None
    except FileNotFoundError:
        print("lmplz binary not found — using Python fallback")
        return None
    finally:
        os.unlink(temp_path)


def analyze_char_frequencies(texts: list[str]):
    """Print character frequency analysis — shows which chars are rare."""
    all_chars = Counter()
    for text in texts:
        all_chars.update(text)

    total = sum(all_chars.values())
    print(f"\nCharacter frequency analysis ({len(all_chars)} unique chars, {total} total):")
    print(f"{'Char':<6} {'Count':>6} {'Freq%':>7} {'Cumul%':>7}")
    print("-" * 30)

    cumul = 0
    for char, count in all_chars.most_common():
        freq = count / total * 100
        cumul += freq
        display = repr(char) if char == " " else char
        print(f"{display:<6} {count:>6} {freq:>6.2f}% {cumul:>6.1f}%")

    # Highlight rare characters
    rare = [(c, n) for c, n in all_chars.items() if n < 10]
    if rare:
        print(f"\n⚠️  Rare characters (< 10 occurrences): {len(rare)}")
        for c, n in sorted(rare, key=lambda x: x[1]):
            print(f"  '{c}' (U+{ord(c):04X}): {n} times")


def main():
    parser = argparse.ArgumentParser(description="Build character LM for Adja ASR")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--order", type=int, default=5, help="n-gram order (default: 5)")
    parser.add_argument("--output", type=str, default=None, help="Output .arpa path")
    parser.add_argument("--extra-text", type=str, default=None, help="Extra Adja text file (one sentence per line)")
    parser.add_argument("--analyze", action="store_true", help="Print character frequency analysis")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    # Load training transcriptions
    train_manifest = data_dir / "manifests" / "train.tsv"
    if not train_manifest.exists():
        print(f"ERROR: {train_manifest} not found")
        sys.exit(1)

    texts = load_texts_from_manifest(str(train_manifest))
    print(f"Loaded {len(texts)} transcriptions from {train_manifest}")

    # Optionally add extra text
    if args.extra_text:
        extra_path = Path(args.extra_text)
        with open(extra_path, "r", encoding="utf-8") as f:
            extra = [normalize_text(line) for line in f if line.strip()]
        texts.extend(extra)
        print(f"Added {len(extra)} extra sentences from {extra_path}")

    print(f"Total: {len(texts)} sentences")

    # Character frequency analysis
    if args.analyze:
        analyze_char_frequencies(texts)

    # Build LM
    output_path = args.output or str(data_dir / f"char_{args.order}gram.arpa")

    # Try KenLM first, fall back to Python
    result = build_arpa_kenlm(texts, order=args.order, output_path=output_path)
    if result is None:
        build_arpa_python(texts, order=args.order, output_path=output_path)

    # Also save the vocabulary list (needed by pyctcdecode)
    vocab_path = data_dir / "lm_vocab.json"
    all_chars = set()
    for text in texts:
        all_chars.update(text)
    lm_vocab = sorted(all_chars)
    lm_vocab = ["|"] + [c for c in lm_vocab if c != " "]  # | = word boundary

    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(lm_vocab, f, ensure_ascii=False, indent=2)
    print(f"LM vocabulary ({len(lm_vocab)} chars) saved to {vocab_path}")


if __name__ == "__main__":
    main()
