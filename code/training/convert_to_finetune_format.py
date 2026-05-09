"""
convert_to_finetune_format.py — Convert TSV parallel data to fine-tuning formats.

Reads train/val TSV pairs and outputs:
  - OpenAI JSONL (chat completion format for GPT-4o-mini fine-tuning)
  - Gemini JSONL (chat format for Vertex AI supervised tuning)

Usage:
    # Convert a single condition:
    python convert_to_finetune_format.py \
        --data-dir experiments/data/splits \
        --experiment exp1 \
        --condition RANDOM-10K \
        --output-dir experiments/data/finetune_formats

    # Convert all conditions (batch mode):
    python convert_to_finetune_format.py \
        --data-dir experiments/data/splits \
        --batch \
        --output-dir experiments/data/finetune_formats

Output structure:
    {output_dir}/{experiment}/{condition}/openai_train.jsonl
    {output_dir}/{experiment}/{condition}/openai_val.jsonl
    {output_dir}/{experiment}/{condition}/gemini_train.jsonl
    {output_dir}/{experiment}/{condition}/gemini_val.jsonl
"""

import argparse
import json
import os
import re
import sys
import typing as tp
import unicodedata
from datetime import datetime, timezone

from sacremoses import MosesPunctNormalizer

# ---------------------------------------------------------------------------
# Text preprocessing (same as hf_job_train.py for fair comparison)
# ---------------------------------------------------------------------------

_mpn = MosesPunctNormalizer(lang="en")
_mpn.substitutions = [(re.compile(r), sub) for r, sub in _mpn.substitutions]


def _get_non_printing_char_replacer(replace_by: str = " ") -> tp.Callable[[str], str]:
    non_printable_map = {
        ord(c): replace_by
        for c in (chr(i) for i in range(sys.maxunicode + 1))
        if unicodedata.category(c) in {"C", "Cc", "Cf", "Cs", "Co", "Cn"}
    }
    def replace_non_printing_char(line: str) -> str:
        return line.translate(non_printable_map)
    return replace_non_printing_char


_replace_nonprint = _get_non_printing_char_replacer(" ")


def preproc(text: str) -> str:
    clean = _mpn.normalize(text)
    clean = _replace_nonprint(clean)
    clean = unicodedata.normalize("NFKC", clean)
    return clean


# ---------------------------------------------------------------------------
# Format converters
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = "You are a French to Adja translator. Translate the French text to Adja. Output only the Adja translation, nothing else."


def to_openai_jsonl(pairs: list[tuple[str, str]]) -> list[str]:
    """Convert (src, tgt) pairs to OpenAI chat fine-tuning JSONL lines."""
    lines = []
    for src, tgt in pairs:
        obj = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": src},
                {"role": "assistant", "content": tgt},
            ]
        }
        lines.append(json.dumps(obj, ensure_ascii=False))
    return lines


def to_gemini_jsonl(pairs: list[tuple[str, str]]) -> list[str]:
    """Convert (src, tgt) pairs to Gemini/Vertex AI supervised tuning JSONL.

    Vertex AI expects JSONL with 'contents' field matching the Gemini API format:
    {"contents": [{"role": "user", "parts": [{"text": "..."}]},
                   {"role": "model", "parts": [{"text": "..."}]}]}

    The system instruction is prepended to the user message.
    """
    lines = []
    for src, tgt in pairs:
        obj = {
            "systemInstruction": {
                "parts": [{"text": SYSTEM_PROMPT}]
            },
            "contents": [
                {"role": "user", "parts": [{"text": src}]},
                {"role": "model", "parts": [{"text": tgt}]},
            ]
        }
        lines.append(json.dumps(obj, ensure_ascii=False))
    return lines


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_tsv_pairs(filepath: str) -> list[tuple[str, str]]:
    pairs = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                pairs.append((preproc(parts[0]), preproc(parts[1])))
    return pairs


def write_jsonl(lines: list[str], filepath: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def convert_condition(data_dir: str, experiment: str, condition: str, output_dir: str):
    """Convert a single experiment/condition."""
    train_path = os.path.join(data_dir, experiment, condition, "train.tsv")
    val_path = os.path.join(data_dir, experiment, condition, "val.tsv")

    if not os.path.exists(train_path):
        print(f"  [SKIP] {experiment}/{condition}: train.tsv not found")
        return False

    train_pairs = load_tsv_pairs(train_path)
    val_pairs = load_tsv_pairs(val_path) if os.path.exists(val_path) else []

    out_dir = os.path.join(output_dir, experiment, condition)

    # OpenAI format
    write_jsonl(to_openai_jsonl(train_pairs), os.path.join(out_dir, "openai_train.jsonl"))
    if val_pairs:
        write_jsonl(to_openai_jsonl(val_pairs), os.path.join(out_dir, "openai_val.jsonl"))

    # Gemini format
    write_jsonl(to_gemini_jsonl(train_pairs), os.path.join(out_dir, "gemini_train.jsonl"))
    if val_pairs:
        write_jsonl(to_gemini_jsonl(val_pairs), os.path.join(out_dir, "gemini_val.jsonl"))

    print(f"  [OK] {experiment}/{condition}: {len(train_pairs)} train, {len(val_pairs)} val")
    return True


def find_all_conditions(data_dir: str) -> list[tuple[str, str]]:
    """Walk data_dir and find all experiment/condition directories with train.tsv."""
    conditions = []
    for root, dirs, files in os.walk(data_dir):
        if "train.tsv" in files:
            rel = os.path.relpath(root, data_dir)
            parts = rel.split(os.sep)
            if len(parts) >= 2:
                experiment = os.sep.join(parts[:-1])
                condition = parts[-1]
                conditions.append((experiment, condition))
    return sorted(conditions)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Convert TSV data to fine-tuning formats")
    parser.add_argument("--data-dir", required=True, help="Root directory with experiment/condition/train.tsv")
    parser.add_argument("--output-dir", required=True, help="Output directory for JSONL files")
    parser.add_argument("--experiment", help="Single experiment to convert")
    parser.add_argument("--condition", help="Single condition to convert")
    parser.add_argument("--batch", action="store_true", help="Convert all conditions found in data-dir")
    args = parser.parse_args()

    print(f"Convert to fine-tuning formats")
    print(f"  Data dir:   {args.data_dir}")
    print(f"  Output dir: {args.output_dir}")
    print(f"  Date:       {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print()

    if args.batch:
        conditions = find_all_conditions(args.data_dir)
        print(f"Found {len(conditions)} conditions in batch mode:")
        ok = 0
        for exp, cond in conditions:
            if convert_condition(args.data_dir, exp, cond, args.output_dir):
                ok += 1
        print(f"\nConverted {ok}/{len(conditions)} conditions.")
    elif args.experiment and args.condition:
        convert_condition(args.data_dir, args.experiment, args.condition, args.output_dir)
    else:
        parser.error("Specify --experiment + --condition, or use --batch")


if __name__ == "__main__":
    main()
