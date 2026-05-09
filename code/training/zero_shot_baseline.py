"""
zero_shot_baseline.py — Zero-shot French→Adja translation baselines using LLM APIs.

Sends test sentences to OpenAI (GPT-4.1) and/or Gemini (2.5 Flash) with a
translation prompt — no fine-tuning, just the base model. Computes BLEU/chrF/
chrF++/TER on the same shared test set used by all other experiments.

Saves test_metrics.json (compatible with collect_hf_results.py) and
predictions.tsv (src\tref\tpred) for manual inspection.

Prerequisites:
    pip install openai google-genai sacrebleu sacremoses python-dotenv

Environment variables:
    OPENAI_API_KEY   — for --platform openai
    GOOGLE_API_KEY   — for --platform gemini

Usage:
    # Quick sanity check (5 sentences):
    python zero_shot_baseline.py --platform openai --limit 5

    # Full run, both platforms:
    python zero_shot_baseline.py --platform both

    # Custom model:
    python zero_shot_baseline.py --platform openai --model gpt-4o
"""

import argparse
import json
import os
import re
import sys
import time
import typing as tp
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

try:
    from dotenv import load_dotenv
    # Search same locations as launch_api_finetune.sh
    _script_dir = Path(__file__).resolve().parent
    _project_root = _script_dir.parent.parent
    for _envfile in [_project_root / ".env", _script_dir / "hpc" / ".env", _script_dir / ".env"]:
        if _envfile.exists():
            load_dotenv(_envfile)
            break
except ImportError:
    pass

import sacrebleu
from sacremoses import MosesPunctNormalizer

# ---------------------------------------------------------------------------
# Text preprocessing (same as hf_job_train.py / openai_finetune.py)
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
# Constants
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = "You are a French to Adja translator. Translate the French text to Adja. Output only the Adja translation, nothing else."
DEFAULT_OPENAI_MODEL = "gpt-4.1-2025-04-14"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"


# ---------------------------------------------------------------------------
# Load test data
# ---------------------------------------------------------------------------

def load_test(test_tsv: str, limit: int = 0) -> tuple[list[str], list[str]]:
    """Load test pairs from TSV. Returns (sources, references)."""
    sources, references = [], []
    with open(test_tsv, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                sources.append(preproc(parts[0]))
                references.append(preproc(parts[1]))
    if limit > 0:
        sources = sources[:limit]
        references = references[:limit]
    return sources, references


# ---------------------------------------------------------------------------
# OpenAI zero-shot
# ---------------------------------------------------------------------------

def run_openai(sources: list[str], model: str) -> list[str]:
    """Translate all sources using OpenAI chat completions."""
    from openai import OpenAI
    client = OpenAI()

    predictions = []
    for i, src in enumerate(sources):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": src},
                ],
                temperature=0,
                max_tokens=256,
            )
            pred = response.choices[0].message.content.strip()
            predictions.append(pred)
        except Exception as e:
            print(f"  Warning: API error for '{src[:50]}...': {e}")
            predictions.append("")

        if (i + 1) % 50 == 0 or (i + 1) == len(sources):
            print(f"  Progress: {i + 1}/{len(sources)}", end="\r")

    print()
    return predictions


# ---------------------------------------------------------------------------
# Gemini zero-shot
# ---------------------------------------------------------------------------

def run_gemini(sources: list[str], model: str) -> list[str]:
    """Translate all sources using Gemini generate_content."""
    from google import genai

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: Set GOOGLE_API_KEY env var for Gemini")
        sys.exit(1)
    client = genai.Client(api_key=api_key)

    predictions = []
    for i, src in enumerate(sources):
        try:
            response = client.models.generate_content(
                model=model,
                contents=src,
                config={
                    "system_instruction": SYSTEM_PROMPT,
                    "temperature": 0,
                    "max_output_tokens": 256,
                },
            )
            pred = response.text.strip() if response.text else ""
            predictions.append(pred)
        except Exception as e:
            print(f"  Warning: API error for '{src[:50]}...': {e}")
            predictions.append("")

        # Gemini rate limiting — 1s between calls
        if i < len(sources) - 1:
            time.sleep(1)

        if (i + 1) % 50 == 0 or (i + 1) == len(sources):
            print(f"  Progress: {i + 1}/{len(sources)}", end="\r")

    print()
    return predictions


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(predictions: list[str], references: list[str]) -> dict:
    bleu = sacrebleu.corpus_bleu(predictions, [references])
    chrf = sacrebleu.corpus_chrf(predictions, [references])
    chrfpp = sacrebleu.corpus_chrf(predictions, [references], word_order=2)
    ter = sacrebleu.corpus_ter(predictions, [references])
    return {
        "test_bleu": bleu.score,
        "test_chrf": chrf.score,
        "test_chrfpp": chrfpp.score,
        "test_ter": ter.score,
        "test_bleu_signature": str(bleu),
        "test_n_samples": len(predictions),
    }


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def save_results(output_dir: str, predictions: list[str], sources: list[str],
                 references: list[str], metrics: dict, metadata: dict):
    os.makedirs(output_dir, exist_ok=True)

    # test_metrics.json
    result = {**metrics, **metadata}
    with open(os.path.join(output_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    # predictions.tsv (src\tref\tpred)
    with open(os.path.join(output_dir, "predictions.tsv"), "w", encoding="utf-8") as f:
        for src, ref, pred in zip(sources, references, predictions):
            f.write(f"{src}\t{ref}\t{pred}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    default_test = str(project_root / "experiments" / "data" / "splits" / "shared" / "test.tsv")
    default_results = str(project_root / "results")

    parser = argparse.ArgumentParser(
        description="Zero-shot French→Adja translation baselines (no fine-tuning)")
    parser.add_argument("--platform", required=True, choices=["openai", "gemini", "both"],
                        help="Which API to use")
    parser.add_argument("--test-tsv", default=default_test,
                        help=f"Test set TSV (default: {default_test})")
    parser.add_argument("--output-dir", default=None,
                        help="Results directory (default: results/<platform>/zero-shot/)")
    parser.add_argument("--model", default=None,
                        help="Override model name (default: platform-specific)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit to first N test sentences (0 = all)")
    args = parser.parse_args()

    platforms = ["openai", "gemini"] if args.platform == "both" else [args.platform]

    # Load test data once
    sources, references = load_test(args.test_tsv, args.limit)
    print(f"Loaded {len(sources)} test sentences from {args.test_tsv}")
    if args.limit:
        print(f"  (limited to first {args.limit})")

    for platform in platforms:
        if platform == "openai":
            model = args.model or DEFAULT_OPENAI_MODEL
        else:
            model = args.model or DEFAULT_GEMINI_MODEL

        output_dir = args.output_dir or os.path.join(default_results, platform, "zero-shot")

        print(f"\n{'=' * 60}")
        print(f"Zero-Shot Baseline: {platform.upper()}")
        print(f"  Model:      {model}")
        print(f"  Test size:  {len(sources)}")
        print(f"  Output:     {output_dir}")
        print(f"  Date:       {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
        print(f"{'=' * 60}\n")

        start_time = time.time()

        if platform == "openai":
            predictions = run_openai(sources, model)
        else:
            predictions = run_gemini(sources, model)

        elapsed = time.time() - start_time

        metrics = compute_metrics(predictions, references)
        metadata = {
            "inference_time_seconds": elapsed,
            "experiment": "baseline",
            "condition": "ZERO-SHOT",
            "seed": 0,
            "model": model,
            "base_model": model,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "platform": f"{platform}-zero-shot",
            "train_size": 0,
            "system_prompt": SYSTEM_PROMPT,
        }

        save_results(output_dir, predictions, sources, references, metrics, metadata)

        print(f"Results ({platform}):")
        print(f"  BLEU:   {metrics['test_bleu']:.1f}")
        print(f"  chrF:   {metrics['test_chrf']:.1f}")
        print(f"  chrF++: {metrics['test_chrfpp']:.1f}")
        print(f"  TER:    {metrics['test_ter']:.1f}")
        print(f"  Time:   {elapsed:.0f}s")
        print(f"  Saved:  {output_dir}/test_metrics.json")

    print("\nDone.")


if __name__ == "__main__":
    main()
