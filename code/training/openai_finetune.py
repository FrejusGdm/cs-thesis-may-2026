"""
openai_finetune.py — Fine-tune GPT-4o-mini on French→Adja translation data.

Uploads training JSONL, creates fine-tuning job, monitors to completion,
runs inference on the test set, and computes BLEU/chrF/chrF++/TER using
the same SacreBLEU pipeline as hf_job_train.py.

Saves test_metrics.json in the same format for easy comparison with HF results.

Prerequisites:
    pip install openai sacrebleu sacremoses

Environment variables:
    OPENAI_API_KEY  — OpenAI API key (or set in .env)

Usage:
    # Single condition:
    python openai_finetune.py \
        --train-jsonl experiments/data/finetune_formats/exp1/RANDOM-10K/openai_train.jsonl \
        --val-jsonl experiments/data/finetune_formats/exp1/RANDOM-10K/openai_val.jsonl \
        --test-tsv experiments/data/splits/shared/test.tsv \
        --experiment exp1 \
        --condition RANDOM-10K \
        --seed 42 \
        --output-dir results/gpt4omini/exp1/RANDOM-10K/seed42

    # The script handles: upload → fine-tune → wait → evaluate → save metrics
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

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import sacrebleu
from openai import OpenAI
from sacremoses import MosesPunctNormalizer

# ---------------------------------------------------------------------------
# Text preprocessing (same as hf_job_train.py)
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
BASE_MODEL = "gpt-4.1-2025-04-14"
POLL_INTERVAL = 30  # seconds between status checks


# ---------------------------------------------------------------------------
# Fine-tuning pipeline
# ---------------------------------------------------------------------------

def upload_file(client: OpenAI, filepath: str, purpose: str = "fine-tune") -> str:
    """Upload a JSONL file and return the file ID."""
    print(f"  Uploading {filepath}...")
    with open(filepath, "rb") as f:
        response = client.files.create(file=f, purpose=purpose)
    print(f"  File ID: {response.id}")
    return response.id


def create_finetune_job(client: OpenAI, train_file_id: str,
                        val_file_id: str = None, suffix: str = None) -> str:
    """Create a fine-tuning job and return the job ID."""
    kwargs = {
        "training_file": train_file_id,
        "model": BASE_MODEL,
    }
    if val_file_id:
        kwargs["validation_file"] = val_file_id
    if suffix:
        kwargs["suffix"] = suffix

    job = client.fine_tuning.jobs.create(**kwargs)
    print(f"  Fine-tuning job created: {job.id}")
    print(f"  Status: {job.status}")
    return job.id


def wait_for_completion(client: OpenAI, job_id: str) -> dict:
    """Poll until the fine-tuning job completes. Returns the final job object."""
    print(f"\n  Waiting for job {job_id} to complete...")
    start = time.time()

    while True:
        job = client.fine_tuning.jobs.retrieve(job_id)
        elapsed = time.time() - start

        if job.status in ("succeeded", "failed", "cancelled"):
            print(f"\n  Job {job.status} after {elapsed:.0f}s")
            if job.status == "failed":
                print(f"  Error: {job.error}")
                sys.exit(1)
            if job.status == "cancelled":
                print("  Job was cancelled.")
                sys.exit(1)
            return job

        print(f"  [{elapsed:.0f}s] Status: {job.status}", end="\r")
        time.sleep(POLL_INTERVAL)


def evaluate_test(client: OpenAI, model_id: str, test_tsv: str,
                  batch_size: int = 20) -> dict:
    """Run inference on test set and compute metrics."""
    print(f"\n  Evaluating on test set with model: {model_id}")

    # Load test pairs
    sources, references = [], []
    with open(test_tsv, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                sources.append(preproc(parts[0]))
                references.append(preproc(parts[1]))

    print(f"  Test samples: {len(sources)}")

    # Generate translations in batches
    predictions = []
    for i in range(0, len(sources), batch_size):
        batch = sources[i:i + batch_size]
        for src in batch:
            try:
                response = client.chat.completions.create(
                    model=model_id,
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

        done = min(i + batch_size, len(sources))
        print(f"  Progress: {done}/{len(sources)}", end="\r")

    print()

    # Compute metrics (same as hf_job_train.py)
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
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="OpenAI GPT-4o-mini fine-tuning for French→Adja NMT")
    parser.add_argument("--train-jsonl", required=True, help="OpenAI-format training JSONL")
    parser.add_argument("--val-jsonl", default=None, help="OpenAI-format validation JSONL")
    parser.add_argument("--test-tsv", required=True, help="Test set TSV (fr\\tadja)")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", required=True, help="Directory for test_metrics.json")
    parser.add_argument("--suffix", default=None, help="Model suffix for identification")
    args = parser.parse_args()

    print("=" * 60)
    print("OpenAI Fine-Tuning: French → Adja")
    print(f"  Experiment:  {args.experiment}")
    print(f"  Condition:   {args.condition}")
    print(f"  Seed:        {args.seed}")
    print(f"  Base model:  {BASE_MODEL}")
    print(f"  Train file:  {args.train_jsonl}")
    print(f"  Val file:    {args.val_jsonl or 'none'}")
    print(f"  Test file:   {args.test_tsv}")
    print(f"  Date:        {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 60)

    client = OpenAI()
    start_time = time.time()

    # Step 1: Upload files
    print("\nStep 1: Upload training data")
    train_file_id = upload_file(client, args.train_jsonl)
    val_file_id = upload_file(client, args.val_jsonl) if args.val_jsonl else None

    # Step 2: Create fine-tuning job
    print("\nStep 2: Create fine-tuning job")
    suffix = args.suffix or f"adja-{args.experiment}-{args.condition}-s{args.seed}"
    # OpenAI suffix max 18 chars
    suffix = suffix[:18]
    job_id = create_finetune_job(client, train_file_id, val_file_id, suffix)

    # Step 3: Wait for completion
    print("\nStep 3: Wait for training")
    job = wait_for_completion(client, job_id)
    fine_tuned_model = job.fine_tuned_model
    print(f"  Fine-tuned model: {fine_tuned_model}")

    training_time = time.time() - start_time

    # Step 4: Evaluate on test set
    print("\nStep 4: Evaluate on test set")
    test_metrics = evaluate_test(client, fine_tuned_model, args.test_tsv)

    # Add metadata (same fields as hf_job_train.py for compatibility)
    test_metrics["training_time_seconds"] = training_time
    test_metrics["experiment"] = args.experiment
    test_metrics["condition"] = args.condition
    test_metrics["seed"] = args.seed
    test_metrics["model"] = fine_tuned_model
    test_metrics["base_model"] = BASE_MODEL
    test_metrics["train_file"] = args.train_jsonl
    test_metrics["openai_job_id"] = job_id
    test_metrics["timestamp"] = datetime.now(timezone.utc).isoformat()
    test_metrics["platform"] = "openai"

    # Count training pairs
    with open(args.train_jsonl) as f:
        test_metrics["train_size"] = sum(1 for _ in f)

    print(f"\nTest metrics:")
    print(f"  BLEU:   {test_metrics['test_bleu']:.1f}")
    print(f"  chrF:   {test_metrics['test_chrf']:.1f}")
    print(f"  chrF++: {test_metrics['test_chrfpp']:.1f}")
    print(f"  TER:    {test_metrics['test_ter']:.1f}")

    # Step 5: Save results
    os.makedirs(args.output_dir, exist_ok=True)
    result_path = os.path.join(args.output_dir, "test_metrics.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    print(f"\nResults saved to: {result_path}")
    print("Done.")


if __name__ == "__main__":
    main()
