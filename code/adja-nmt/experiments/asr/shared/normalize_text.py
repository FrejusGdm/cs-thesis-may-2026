#!/usr/bin/env python3
from __future__ import annotations
"""
Text normalization for Adja ASR experiments.

Applies Unicode NFC normalization and consistent whitespace handling.
Critical for Adja because:
- Tone marks (é, è, ẽ) can be composed or decomposed Unicode
- Special characters (ɛ, ɔ, ŋ, ɖ) must be preserved
- Inconsistent normalization = inflated/deflated WER/CER

Adapted from the NMT project's unicode-normalization/normalize_unicode.py.

Usage:
    python normalize_text.py --input transcripts.txt --output normalized.txt
    python normalize_text.py --input manifests/train.tsv --output manifests/train_normalized.tsv --tsv
"""

import argparse
import sys
import unicodedata

sys.stdout.reconfigure(line_buffering=True)


def normalize_nfc(text: str) -> str:
    """Normalize text to Unicode NFC form.

    NFC = Canonical Composition. Ensures that characters like
    é (e + combining acute) are stored as single codepoints where possible.
    This is critical for consistent CER computation.
    """
    return unicodedata.normalize("NFC", text)


def clean_whitespace(text: str) -> str:
    """Normalize whitespace: strip, collapse multiple spaces."""
    return " ".join(text.split())


def normalize_for_asr(text: str) -> str:
    """Full normalization pipeline for ASR transcripts.

    Steps:
    1. Unicode NFC normalization
    2. Whitespace normalization
    3. Lowercase (optional — configurable, off by default for Adja)

    Does NOT:
    - Remove tone marks or diacritics
    - Remove special Gbe characters (ɛ, ɔ, ŋ, ɖ)
    - Remove punctuation (let the model handle it or remove separately)
    """
    text = normalize_nfc(text)
    text = clean_whitespace(text)
    return text


def check_normalization_changes(original: str, normalized: str) -> list[dict]:
    """Report character-level changes from normalization."""
    changes = []
    if original != normalized:
        orig_chars = list(original)
        norm_chars = list(normalized)

        # Report codepoint differences
        orig_points = set((i, c, hex(ord(c)), unicodedata.name(c, "UNKNOWN")) for i, c in enumerate(orig_chars))
        norm_points = set((i, c, hex(ord(c)), unicodedata.name(c, "UNKNOWN")) for i, c in enumerate(norm_chars))

        if len(orig_chars) != len(norm_chars):
            changes.append({
                "type": "length_change",
                "original_len": len(orig_chars),
                "normalized_len": len(norm_chars),
            })

    return changes


def normalize_manifest(input_path: str, output_path: str, text_col: int = 2):
    """Normalize the text column in a TSV manifest."""
    changes_count = 0
    total = 0

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        header = fin.readline()
        fout.write(header)

        for line in fin:
            total += 1
            parts = line.rstrip("\n").split("\t")
            if len(parts) > text_col:
                original = parts[text_col]
                normalized = normalize_for_asr(original)
                if original != normalized:
                    changes_count += 1
                parts[text_col] = normalized
            fout.write("\t".join(parts) + "\n")

    print(f"Normalized {changes_count}/{total} lines in {input_path}")
    return changes_count, total


def normalize_text_file(input_path: str, output_path: str):
    """Normalize a plain text file (one line per utterance)."""
    changes_count = 0
    total = 0

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line in fin:
            total += 1
            original = line.rstrip("\n")
            normalized = normalize_for_asr(original)
            if original != normalized:
                changes_count += 1
            fout.write(normalized + "\n")

    print(f"Normalized {changes_count}/{total} lines in {input_path}")
    return changes_count, total


def main():
    parser = argparse.ArgumentParser(description="Normalize text for Adja ASR")
    parser.add_argument("--input", required=True, help="Input file path")
    parser.add_argument("--output", required=True, help="Output file path")
    parser.add_argument("--tsv", action="store_true", help="Input is TSV manifest (normalize text column)")
    parser.add_argument("--text-col", type=int, default=2, help="Column index for text in TSV (0-indexed, default=2)")
    args = parser.parse_args()

    if args.tsv:
        normalize_manifest(args.input, args.output, args.text_col)
    else:
        normalize_text_file(args.input, args.output)


if __name__ == "__main__":
    main()
