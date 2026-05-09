"""
gemini_finetune.py — Fine-tune Gemini Flash on French→Adja translation data.

Uses the Google Gen AI Python SDK (google-genai) with Vertex AI for supervised
tuning. Uploads training data, creates a tuning job, waits for completion,
runs inference, and computes BLEU/chrF/chrF++/TER.

Saves test_metrics.json in the same format as hf_job_train.py and openai_finetune.py.

Prerequisites:
    pip install google-genai sacrebleu sacremoses

    # Authentication — one of:
    export GOOGLE_API_KEY="..."          # For Google AI Studio (Gemini API)
    # OR
    gcloud auth application-default login  # For Vertex AI

Usage:
    python gemini_finetune.py \
        --train-jsonl experiments/data/finetune_formats/exp1/RANDOM-10K/gemini_train.jsonl \
        --test-tsv experiments/data/splits/shared/test.tsv \
        --experiment exp1 \
        --condition RANDOM-10K \
        --seed 42 \
        --output-dir results/gemini/exp1/RANDOM-10K/seed42 \
        --use-vertex \
        --project YOUR_GCP_PROJECT \
        --location us-central1

    # Or with Google AI Studio (if tuning is available):
    python gemini_finetune.py \
        --train-jsonl ... \
        --test-tsv ... \
        --experiment exp1 \
        --condition RANDOM-10K \
        --seed 42 \
        --output-dir results/gemini/exp1/RANDOM-10K/seed42
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
BASE_MODEL = "gemini-2.5-flash"
POLL_INTERVAL = 60  # seconds between status checks


# ---------------------------------------------------------------------------
# Gemini fine-tuning pipeline
# ---------------------------------------------------------------------------

def create_client(args):
    """Create the google.genai client based on mode (Vertex or AI Studio)."""
    from google import genai

    if args.use_vertex:
        client = genai.Client(
            vertexai=True,
            project=args.project,
            location=args.location,
        )
        print(f"  Using Vertex AI (project={args.project}, location={args.location})")
    else:
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            print("ERROR: Set GOOGLE_API_KEY env var or use --use-vertex")
            sys.exit(1)
        client = genai.Client(api_key=api_key)
        print("  Using Google AI Studio (Gemini API)")

    return client


def upload_and_tune(client, train_jsonl: str, display_name: str, args) -> dict:
    """Upload training data and start a tuning job."""
    from google.genai import types

    if args.use_vertex:
        # For Vertex AI, we need to upload JSONL to GCS first, or use inline data
        # If a GCS URI is provided, use it directly; otherwise upload to a temp bucket
        if args.gcs_uri:
            training_dataset = types.TuningDataset(gcs_uri=args.gcs_uri)
            print(f"  Using GCS training data: {args.gcs_uri}")
        else:
            # Read training examples from local JSONL and pass as inline dataset
            examples = []
            with open(train_jsonl, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        examples.append(json.loads(line))
            training_dataset = types.TuningDataset(
                gcs_uri=args.gcs_uri if args.gcs_uri else None
            )
            # Vertex AI requires GCS URI for training data
            print("ERROR: Vertex AI tuning requires --gcs-uri for training data.")
            print("  Upload your JSONL to GCS first:")
            print(f"    gsutil cp {train_jsonl} gs://YOUR_BUCKET/adja-nmt/")
            sys.exit(1)
    else:
        # Google AI Studio — can use inline training data
        examples = []
        with open(train_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    examples.append(json.loads(line))

        training_dataset = types.TuningDataset(
            examples=[
                types.TuningExample(
                    text_input=ex["contents"][0]["parts"][0]["text"],
                    output=ex["contents"][1]["parts"][0]["text"],
                )
                for ex in examples
            ]
        )
        print(f"  Loaded {len(examples)} training examples from {train_jsonl}")

    config = types.CreateTuningJobConfig(
        epoch_count=args.epochs,
        tuned_model_display_name=display_name,
    )

    print(f"  Creating tuning job (base_model={BASE_MODEL}, epochs={args.epochs})...")
    tuning_job = client.tunings.tune(
        base_model=BASE_MODEL,
        training_dataset=training_dataset,
        config=config,
    )
    print(f"  Tuning job: {tuning_job.name}")
    return tuning_job


def wait_for_tuning(client, tuning_job) -> dict:
    """Poll until tuning job completes."""
    print(f"\n  Waiting for tuning job to complete...")
    start = time.time()

    completed_states = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED"}

    while True:
        job = client.tunings.get(name=tuning_job.name)
        state = str(job.state) if hasattr(job, "state") else "UNKNOWN"
        elapsed = time.time() - start

        if state in completed_states:
            print(f"\n  Job {state} after {elapsed:.0f}s")
            if state == "JOB_STATE_FAILED":
                print(f"  Error: {getattr(job, 'error', 'unknown')}")
                sys.exit(1)
            if state == "JOB_STATE_CANCELLED":
                sys.exit(1)
            return job

        print(f"  [{elapsed:.0f}s] Status: {state}", end="\r")
        time.sleep(POLL_INTERVAL)


def evaluate_test(client, model_id: str, test_tsv: str) -> dict:
    """Run inference on test set and compute metrics."""
    print(f"\n  Evaluating with model: {model_id}")

    sources, references = [], []
    with open(test_tsv, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                sources.append(preproc(parts[0]))
                references.append(preproc(parts[1]))

    print(f"  Test samples: {len(sources)}")

    predictions = []
    for i, src in enumerate(sources):
        try:
            response = client.models.generate_content(
                model=model_id,
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

        if (i + 1) % 50 == 0:
            print(f"  Progress: {i + 1}/{len(sources)}", end="\r")

    print()

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
    parser = argparse.ArgumentParser(description="Gemini fine-tuning for French→Adja NMT")
    parser.add_argument("--train-jsonl", required=True, help="Gemini-format training JSONL")
    parser.add_argument("--test-tsv", required=True, help="Test set TSV (fr\\tadja)")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", required=True, help="Directory for test_metrics.json")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    # Vertex AI options
    parser.add_argument("--use-vertex", action="store_true", help="Use Vertex AI (requires GCP project)")
    parser.add_argument("--project", default=None, help="GCP project ID (for Vertex AI)")
    parser.add_argument("--location", default="us-central1", help="GCP region (for Vertex AI)")
    parser.add_argument("--gcs-uri", default=None, help="GCS URI for training data (Vertex AI)")
    args = parser.parse_args()

    print("=" * 60)
    print("Gemini Fine-Tuning: French → Adja")
    print(f"  Experiment:  {args.experiment}")
    print(f"  Condition:   {args.condition}")
    print(f"  Seed:        {args.seed}")
    print(f"  Base model:  {BASE_MODEL}")
    print(f"  Train file:  {args.train_jsonl}")
    print(f"  Test file:   {args.test_tsv}")
    print(f"  Epochs:      {args.epochs}")
    print(f"  Date:        {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 60)

    start_time = time.time()

    # Step 1: Create client
    print("\nStep 1: Initialize client")
    client = create_client(args)

    # Step 2: Create tuning job
    print("\nStep 2: Create tuning job")
    display_name = f"adja-{args.experiment}-{args.condition}-s{args.seed}"
    tuning_job = upload_and_tune(client, args.train_jsonl, display_name, args)

    # Step 3: Wait for completion
    print("\nStep 3: Wait for training")
    completed_job = wait_for_tuning(client, tuning_job)

    # Get tuned model endpoint
    tuned_model = None
    if hasattr(completed_job, "tuned_model") and completed_job.tuned_model:
        tuned_model = getattr(completed_job.tuned_model, "endpoint", None) or \
                      getattr(completed_job.tuned_model, "model", None) or \
                      str(completed_job.tuned_model)
    if not tuned_model:
        print("ERROR: Could not determine tuned model endpoint")
        sys.exit(1)

    print(f"  Tuned model: {tuned_model}")
    training_time = time.time() - start_time

    # Step 4: Evaluate on test set
    print("\nStep 4: Evaluate on test set")
    test_metrics = evaluate_test(client, tuned_model, args.test_tsv)

    # Add metadata
    test_metrics["training_time_seconds"] = training_time
    test_metrics["experiment"] = args.experiment
    test_metrics["condition"] = args.condition
    test_metrics["seed"] = args.seed
    test_metrics["model"] = tuned_model
    test_metrics["base_model"] = BASE_MODEL
    test_metrics["train_file"] = args.train_jsonl
    test_metrics["tuning_job_name"] = tuning_job.name
    test_metrics["epochs"] = args.epochs
    test_metrics["timestamp"] = datetime.now(timezone.utc).isoformat()
    test_metrics["platform"] = "gemini-vertex" if args.use_vertex else "gemini-ai-studio"

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
