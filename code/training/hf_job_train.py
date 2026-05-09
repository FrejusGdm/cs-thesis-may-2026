# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch",
#     "transformers==4.44.2",
#     "sentencepiece",
#     "protobuf",
#     "sacrebleu",
#     "huggingface-hub",
#     "accelerate",
#     "sacremoses",
#     "rouge-score",
# ]
# ///
"""
hf_job_train.py — Self-contained training script for HuggingFace Jobs.

Downloads data from HF Hub, fine-tunes NLLB-200 or mBART-50 with a custom
training loop (Adafactor + constant-with-warmup), evaluates on the shared
test set, and uploads test_metrics.json to a private HF results repo.

Supports multiple model families:
  - NLLB: uses fra_Latn/ewe_Latn lang tokens (default)
  - mBART-50: uses fr_XX lang tokens, auto-detected from model name

Uses a custom `aj_Latn` language token initialized from a similar language's
embeddings (or randomly if SIMILAR_LANG=none).

Environment variables (set via --env in hf jobs run):
    EXPERIMENT    — e.g. "exp1", "exp2", "baselines", "ablations"
    CONDITION     — e.g. "RANDOM-10K", "STRUCTURED-4K-ONLY"
    SEED          — e.g. "42"
    MODEL         — e.g. "facebook/nllb-200-distilled-600M" (default)
    DATASET_REPO  — HF dataset repo with splits (e.g. "username/adja-nmt-splits")
    RESULTS_REPO  — HF model repo for results (e.g. "username/adja-nmt-results")
    SIMILAR_LANG  — lang token to copy embeddings from, or "none" for random init
    SRC_LANG      — source language token (auto-detected from MODEL if not set)

The script:
  1. Downloads train.tsv, val.tsv from {EXPERIMENT}/{CONDITION}/
  2. Downloads shared/test.tsv
  3. Registers aj_Latn token, initializes from Ewe embeddings
  4. Fine-tunes with Adafactor + early stopping on val chrF
  5. Generates test translations and computes BLEU/chrF/chrF++/TER
  6. Uploads test_metrics.json to RESULTS_REPO at {EXPERIMENT}/{CONDITION}/seed{SEED}/
"""

import gc
import json
import os
import random
import re
import sys
import time
import typing as tp
import unicodedata
from datetime import datetime, timezone

# Force line-buffered stdout so print() appears in real-time in container logs
sys.stdout.reconfigure(line_buffering=True)

import math

import numpy as np
import sacrebleu
import torch
from huggingface_hub import HfApi, hf_hub_download
from rouge_score import rouge_scorer as rouge_scorer_lib
from sacremoses import MosesPunctNormalizer
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    get_constant_schedule_with_warmup,
    set_seed,
)
from transformers.optimization import Adafactor


# ---------------------------------------------------------------------------
# Config from environment
# ---------------------------------------------------------------------------

EXPERIMENT = os.environ.get("EXPERIMENT", "exp1")
CONDITION = os.environ.get("CONDITION", "RANDOM-10K")
SEED = int(os.environ.get("SEED", "42"))
MODEL_NAME = os.environ.get("MODEL", "facebook/nllb-200-distilled-600M")
DATASET_REPO = os.environ["DATASET_REPO"]
RESULTS_REPO = os.environ["RESULTS_REPO"]

# Training hyperparameters (can be overridden via env vars)
LEARNING_RATE = float(os.environ.get("LR", "1e-4"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "16"))
MAX_EPOCHS = int(os.environ.get("MAX_EPOCHS", "50"))
PATIENCE = int(os.environ.get("PATIENCE", "10"))
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "128"))
WARMUP_STEPS = int(os.environ.get("WARMUP_STEPS", "500"))
NUM_BEAMS = int(os.environ.get("NUM_BEAMS", "5"))
EVAL_STEPS = int(os.environ.get("EVAL_STEPS", "200"))
LOG_STEPS = int(os.environ.get("LOG_STEPS", "50"))
SAVE_CHECKPOINT = os.environ.get("SAVE_CHECKPOINT", "1") == "1"

# Auto-detect lang token format based on model family
_model_lower = MODEL_NAME.lower()
if "mbart" in _model_lower:
    _src_default, _sim_default = "fr_XX", "fr_XX"
else:  # NLLB and others
    _src_default, _sim_default = "fra_Latn", "ewe_Latn"

SRC_LANG = os.environ.get("SRC_LANG", _src_default)
TGT_LANG = os.environ.get("TGT_LANG", "aj_Latn")  # custom token for all models
SIMILAR_LANG = os.environ.get("SIMILAR_LANG", _sim_default)


# ---------------------------------------------------------------------------
# Text preprocessing (from NLLB pretraining pipeline)
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
    """Normalize text the same way NLLB was pretrained."""
    clean = _mpn.normalize(text)
    clean = _replace_nonprint(clean)
    clean = unicodedata.normalize("NFKC", clean)
    return clean


# ---------------------------------------------------------------------------
# Language token setup
# ---------------------------------------------------------------------------

def fix_tokenizer(tokenizer, new_lang="aj_Latn"):
    """Add a new language code token to the tokenizer in a version-safe way."""
    extra = list(getattr(tokenizer, "additional_special_tokens", []))
    if new_lang not in extra:
        tokenizer.add_special_tokens({"additional_special_tokens": extra + [new_lang]})

    new_id = tokenizer.convert_tokens_to_ids(new_lang)

    if hasattr(tokenizer, "lang_code_to_id") and isinstance(tokenizer.lang_code_to_id, dict):
        tokenizer.lang_code_to_id[new_lang] = new_id
    if hasattr(tokenizer, "id_to_lang_code") and isinstance(tokenizer.id_to_lang_code, dict):
        tokenizer.id_to_lang_code[new_id] = new_lang

    init = getattr(tokenizer, "init_kwargs", {})
    if isinstance(init.get("lang_code_to_id", None), dict):
        init["lang_code_to_id"][new_lang] = new_id
    if isinstance(init.get("id_to_lang_code", None), dict):
        init["id_to_lang_code"][str(new_id)] = new_lang

    return new_id


def init_lang_token(model, tokenizer, new_lang="aj_Latn", similar_lang="ewe_Latn"):
    """Register aj_Latn and initialize from a similar lang or randomly."""
    fix_tokenizer(tokenizer, new_lang)
    model.resize_token_embeddings(len(tokenizer))

    added_token_id = tokenizer.convert_tokens_to_ids(new_lang)
    assert added_token_id != tokenizer.unk_token_id, f"{new_lang} was not registered"

    if similar_lang.lower() == "none":
        print(f"Initialized {new_lang} with random embedding (no similar lang)")
    else:
        similar_lang_id = tokenizer.convert_tokens_to_ids(similar_lang)
        assert similar_lang_id != tokenizer.unk_token_id, f"{similar_lang} not found in tokenizer"

        with torch.no_grad():
            emb_in = model.get_input_embeddings().weight
            emb_in[added_token_id].copy_(emb_in[similar_lang_id])

            out_head = model.get_output_embeddings()
            if out_head is not None and hasattr(out_head, "weight") and out_head.weight.shape[0] == emb_in.shape[0]:
                out_head.weight[added_token_id].copy_(out_head.weight[similar_lang_id])

        print(f"Initialized {new_lang} from {similar_lang} (id {added_token_id} <- {similar_lang_id})")

    model.config.forced_bos_token_id = None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_pairs(file_path):
    """Load parallel sentence pairs from a TSV file, with preprocessing."""
    pairs = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                pairs.append((preproc(parts[0]), preproc(parts[1])))
    return pairs


def get_batch(data, batch_size):
    """Sample a random batch of (src, tgt) pairs."""
    batch = random.choices(data, k=batch_size)
    src_texts = [p[0] for p in batch]
    tgt_texts = [p[1] for p in batch]
    return src_texts, tgt_texts


# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------

def download_data():
    """Download train/val/test TSV files from HF Hub."""
    token = os.environ.get("HF_TOKEN")
    data_dir = "/tmp/data"
    os.makedirs(data_dir, exist_ok=True)

    train_path = f"{EXPERIMENT}/{CONDITION}/train.tsv"
    val_path = f"{EXPERIMENT}/{CONDITION}/val.tsv"
    test_path = "shared/test.tsv"
    structured_path = "shared/structured_train.tsv"

    print(f"Downloading data from {DATASET_REPO}...")
    print(f"  Train: {train_path}")
    print(f"  Val: {val_path}")
    print(f"  Test: {test_path}")

    train_file = hf_hub_download(
        repo_id=DATASET_REPO, filename=train_path,
        repo_type="dataset", token=token, local_dir=data_dir,
    )
    val_file = hf_hub_download(
        repo_id=DATASET_REPO, filename=val_path,
        repo_type="dataset", token=token, local_dir=data_dir,
    )
    test_file = hf_hub_download(
        repo_id=DATASET_REPO, filename=test_path,
        repo_type="dataset", token=token, local_dir=data_dir,
    )

    # Download structured data for subset evaluation
    structured_file = None
    try:
        structured_file = hf_hub_download(
            repo_id=DATASET_REPO, filename=structured_path,
            repo_type="dataset", token=token, local_dir=data_dir,
        )
    except Exception:
        print("  Note: structured_train.tsv not found, subset eval will be skipped")

    return train_file, val_file, test_file, structured_file


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def evaluate_val(model, tokenizer, val_data, device, num_samples=200):
    """Compute val loss, BLEU, and chrF on a sample of the validation set."""
    model.eval()
    val_losses = []
    all_preds = []
    all_refs = []

    forced_bos = tokenizer.convert_tokens_to_ids(TGT_LANG)

    # Compute val loss on sampled batches
    n_batches = max(1, min(len(val_data), num_samples) // BATCH_SIZE)
    sample_data = random.sample(val_data, min(len(val_data), n_batches * BATCH_SIZE))

    with torch.no_grad():
        for i in range(0, len(sample_data), BATCH_SIZE):
            batch = sample_data[i:i + BATCH_SIZE]
            batch_src = [p[0] for p in batch]
            batch_tgt = [p[1] for p in batch]

            tokenizer.src_lang = SRC_LANG
            tokenizer.tgt_lang = TGT_LANG
            x = tokenizer(batch_src, return_tensors="pt", padding=True,
                          truncation=True, max_length=MAX_LENGTH).to(device)

            with tokenizer.as_target_tokenizer():
                y = tokenizer(batch_tgt, return_tensors="pt", padding=True,
                              truncation=True, max_length=MAX_LENGTH).to(device)

            labels = y.input_ids.clone()
            labels[labels == tokenizer.pad_token_id] = -100

            loss = model(**x, labels=labels).loss
            val_losses.append(loss.item())

            # Generate translations for metrics
            generated = model.generate(
                **x,
                forced_bos_token_id=forced_bos,
                max_length=MAX_LENGTH,
                num_beams=NUM_BEAMS,
                length_penalty=1.0,
            )
            decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
            all_preds.extend(decoded)
            all_refs.extend(batch_tgt)

    avg_loss = np.mean(val_losses)
    bleu = sacrebleu.corpus_bleu(all_preds, [all_refs]).score
    chrf = sacrebleu.corpus_chrf(all_preds, [all_refs]).score

    model.train()
    return avg_loss, bleu, chrf


# ---------------------------------------------------------------------------
# Test evaluation
# ---------------------------------------------------------------------------

def _compute_subset_metrics(predictions, references):
    """Compute BLEU, chrF, chrF++ for a subset of predictions/references."""
    if not predictions:
        return {}
    bleu = sacrebleu.corpus_bleu(predictions, [references])
    chrf = sacrebleu.corpus_chrf(predictions, [references])
    chrfpp = sacrebleu.corpus_chrf(predictions, [references], word_order=2)
    return {"bleu": bleu.score, "chrf": chrf.score, "chrfpp": chrfpp.score,
            "n_samples": len(predictions)}


def _load_structured_sources(structured_file):
    """Load the set of French source sentences from the structured data."""
    sources = set()
    with open(structured_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                sources.add(preproc(parts[0]))
    return sources


def evaluate_test(model, tokenizer, test_file, structured_file=None):
    """Generate translations on test set and compute metrics."""
    model.eval()
    device = model.device

    sources, references = [], []
    with open(test_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                sources.append(preproc(parts[0]))
                references.append(preproc(parts[1]))

    predictions = []
    tokenizer.src_lang = SRC_LANG
    forced_bos = tokenizer.convert_tokens_to_ids(TGT_LANG)

    for i in range(0, len(sources), BATCH_SIZE):
        batch_src = sources[i:i + BATCH_SIZE]
        inputs = tokenizer(
            batch_src, return_tensors="pt", max_length=MAX_LENGTH,
            truncation=True, padding=True,
        ).to(device)
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                forced_bos_token_id=forced_bos,
                max_length=MAX_LENGTH,
                num_beams=NUM_BEAMS,
                length_penalty=1.0,
            )
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        predictions.extend(decoded)

    bleu = sacrebleu.corpus_bleu(predictions, [references])
    chrf = sacrebleu.corpus_chrf(predictions, [references])
    chrfpp = sacrebleu.corpus_chrf(predictions, [references], word_order=2)
    ter = sacrebleu.corpus_ter(predictions, [references])

    # ROUGE-L (token-level F1, averaged over sentences, scaled to 0–100)
    scorer = rouge_scorer_lib.RougeScorer(["rougeL"], use_stemmer=False)
    rouge_l_scores = [
        scorer.score(ref, pred)["rougeL"].fmeasure
        for pred, ref in zip(predictions, references)
    ]
    rouge_l = sum(rouge_l_scores) / len(rouge_l_scores) * 100.0

    # Perplexity: evaluate cross-entropy loss (NLL) on target sequences
    total_nll = 0.0
    total_tokens = 0
    for i in range(0, len(sources), BATCH_SIZE):
        batch_src = sources[i:i + BATCH_SIZE]
        batch_ref = references[i:i + BATCH_SIZE]
        tokenizer.src_lang = SRC_LANG
        src_enc = tokenizer(
            batch_src, return_tensors="pt", max_length=MAX_LENGTH,
            truncation=True, padding=True,
        ).to(device)
        # Tokenize targets with target language (deprecated but works in 4.44.2)
        tokenizer.src_lang = TGT_LANG
        tgt_enc = tokenizer(
            batch_ref, return_tensors="pt", max_length=MAX_LENGTH,
            truncation=True, padding=True,
        ).to(device)
        tokenizer.src_lang = SRC_LANG  # restore
        labels = tgt_enc["input_ids"].clone()
        labels[labels == tokenizer.pad_token_id] = -100
        with torch.no_grad():
            out = model(**src_enc, labels=labels)
        # out.loss is mean NLL per non-pad token; recover sum
        n_tokens = (labels != -100).sum().item()
        total_nll += out.loss.item() * n_tokens
        total_tokens += n_tokens
    perplexity = math.exp(total_nll / total_tokens) if total_tokens > 0 else float("inf")

    result = {
        "test_bleu": bleu.score,
        "test_chrf": chrf.score,
        "test_chrfpp": chrfpp.score,
        "test_ter": ter.score,
        "test_rougeL": rouge_l,
        "test_perplexity": perplexity,
        "test_bleu_signature": str(bleu),
        "test_n_samples": len(predictions),
    }

    # Subset evaluation: structured vs Tatoeba/random
    if structured_file and os.path.exists(structured_file):
        struct_sources = _load_structured_sources(structured_file)
        struct_preds, struct_refs = [], []
        random_preds, random_refs = [], []
        for src, pred, ref in zip(sources, predictions, references):
            if src in struct_sources:
                struct_preds.append(pred)
                struct_refs.append(ref)
            else:
                random_preds.append(pred)
                random_refs.append(ref)

        struct_metrics = _compute_subset_metrics(struct_preds, struct_refs)
        random_metrics = _compute_subset_metrics(random_preds, random_refs)

        for k, v in struct_metrics.items():
            result[f"subset_structured_{k}"] = v
        for k, v in random_metrics.items():
            result[f"subset_tatoeba_{k}"] = v

        print(f"  Subset eval: {len(struct_preds)} structured, {len(random_preds)} tatoeba")
        if struct_metrics:
            print(f"    Structured — BLEU: {struct_metrics['bleu']:.1f}, chrF++: {struct_metrics['chrfpp']:.1f}")
        if random_metrics:
            print(f"    Tatoeba    — BLEU: {random_metrics['bleu']:.1f}, chrF++: {random_metrics['chrfpp']:.1f}")

    return result


# ---------------------------------------------------------------------------
# Upload results
# ---------------------------------------------------------------------------

def upload_results(metrics):
    """Upload test_metrics.json to the HF results repo."""
    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)

    try:
        api.create_repo(RESULTS_REPO, repo_type="dataset", private=True, exist_ok=True)
    except Exception as e:
        print(f"Note: repo creation returned: {e}")

    # Model slug prefix: non-default models get their own namespace to avoid overwrites
    _model_slug = ""
    if "1.3B" in MODEL_NAME or "1.3b" in MODEL_NAME:
        _model_slug = "nllb-1.3b/"
    elif "mbart" in MODEL_NAME.lower():
        _model_slug = "mbart/"
    # 600M (default) → no prefix, backward compatible

    result_path = f"{_model_slug}{EXPERIMENT}/{CONDITION}/seed{SEED}/test_metrics.json"
    metrics_json = json.dumps(metrics, indent=2)

    print(f"Uploading results to {RESULTS_REPO}/{result_path}...")
    api.upload_file(
        path_or_fileobj=metrics_json.encode("utf-8"),
        path_in_repo=result_path,
        repo_id=RESULTS_REPO,
        repo_type="dataset",
    )
    print("Upload complete.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print(f"HF Jobs Training Script (custom loop)")
    print(f"  Experiment: {EXPERIMENT}")
    print(f"  Condition:  {CONDITION}")
    print(f"  Seed:       {SEED}")
    print(f"  Model:      {MODEL_NAME}")
    print(f"  Dataset:    {DATASET_REPO}")
    print(f"  Results:    {RESULTS_REPO}")
    print(f"  LR:         {LEARNING_RATE}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Max epochs: {MAX_EPOCHS}")
    print(f"  Warmup:     {WARMUP_STEPS} steps")
    print(f"  Eval every: {EVAL_STEPS} steps")
    print(f"  Patience:   {PATIENCE} evals")
    print(f"  TGT_LANG:   {TGT_LANG}")
    print("=" * 60)

    set_seed(SEED)

    # Download data
    train_file, val_file, test_file, structured_file = download_data()

    # Load model + tokenizer
    print(f"\nLoading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    # Register aj_Latn token and init from Ewe
    init_lang_token(model, tokenizer, TGT_LANG, SIMILAR_LANG)

    # Load data
    train_data = load_pairs(train_file)
    val_data = load_pairs(val_file)
    print(f"  Train: {len(train_data)} pairs")
    print(f"  Val:   {len(val_data)} pairs")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"  Device: {device}")

    # Optimizer + scheduler
    optimizer = Adafactor(
        [p for p in model.parameters() if p.requires_grad],
        scale_parameter=False,
        relative_step=False,
        lr=LEARNING_RATE,
        clip_threshold=1.0,
        weight_decay=1e-3,
    )

    max_steps = (len(train_data) // BATCH_SIZE) * MAX_EPOCHS
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS)

    print(f"\n  Max training steps: {max_steps}")
    print(f"\nStarting training...")

    # Training loop
    model.train()
    start_time = time.time()
    losses = []
    best_chrf = -1.0
    best_model_state = None
    evals_without_improvement = 0
    actual_steps = 0

    for step in range(1, max_steps + 1):
        batch_src, batch_tgt = get_batch(train_data, BATCH_SIZE)

        try:
            tokenizer.src_lang = SRC_LANG
            tokenizer.tgt_lang = TGT_LANG
            x = tokenizer(batch_src, return_tensors="pt", padding=True,
                          truncation=True, max_length=MAX_LENGTH).to(device)

            with tokenizer.as_target_tokenizer():
                y = tokenizer(batch_tgt, return_tensors="pt", padding=True,
                              truncation=True, max_length=MAX_LENGTH).to(device)

            y.input_ids[y.input_ids == tokenizer.pad_token_id] = -100

            loss = model(**x, labels=y.input_ids).loss
            loss.backward()
            losses.append(loss.item())

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        except RuntimeError as e:
            optimizer.zero_grad(set_to_none=True)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"Step {step} | RuntimeError: {e}")
            continue

        actual_steps = step

        # Logging
        if step % LOG_STEPS == 0:
            avg_loss = np.mean(losses[-LOG_STEPS:])
            elapsed = time.time() - start_time
            print(f"Step {step}/{max_steps} | loss: {avg_loss:.4f} | "
                  f"elapsed: {elapsed:.0f}s")

        # Validation
        if step % EVAL_STEPS == 0:
            val_loss, val_bleu, val_chrf = evaluate_val(
                model, tokenizer, val_data, device
            )
            print(f"  Val loss: {val_loss:.4f} | Val BLEU: {val_bleu:.1f} | "
                  f"Val chrF: {val_chrf:.1f}")

            if val_chrf > best_chrf:
                best_chrf = val_chrf
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                evals_without_improvement = 0
                print(f"  -> New best chrF: {best_chrf:.1f}")
            else:
                evals_without_improvement += 1
                print(f"  -> No improvement ({evals_without_improvement}/{PATIENCE})")

            if evals_without_improvement >= PATIENCE:
                print(f"\nEarly stopping at step {step} (patience={PATIENCE})")
                break

            model.train()

    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.0f}s ({training_time / 3600:.1f}h)")
    print(f"  Actual steps: {actual_steps}")
    print(f"  Best val chrF: {best_chrf:.1f}")

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model.to(device)
        print("Restored best model checkpoint.")

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_metrics = evaluate_test(model, tokenizer, test_file, structured_file)
    test_metrics["training_time_seconds"] = training_time
    test_metrics["experiment"] = EXPERIMENT
    test_metrics["condition"] = CONDITION
    test_metrics["seed"] = SEED
    test_metrics["model"] = MODEL_NAME
    test_metrics["train_size"] = len(train_data)
    test_metrics["val_size"] = len(val_data)
    test_metrics["learning_rate"] = LEARNING_RATE
    test_metrics["batch_size"] = BATCH_SIZE
    test_metrics["max_epochs"] = MAX_EPOCHS
    test_metrics["actual_steps"] = actual_steps
    test_metrics["best_val_chrf"] = best_chrf
    test_metrics["warmup_steps"] = WARMUP_STEPS
    test_metrics["src_lang"] = SRC_LANG
    test_metrics["tgt_lang"] = TGT_LANG
    test_metrics["similar_lang"] = SIMILAR_LANG
    test_metrics["timestamp"] = datetime.now(timezone.utc).isoformat()

    print(f"\nTest metrics:")
    print(f"  BLEU:   {test_metrics['test_bleu']:.1f}")
    print(f"  chrF:   {test_metrics['test_chrf']:.1f}")
    print(f"  chrF++: {test_metrics['test_chrfpp']:.1f}")
    print(f"  TER:    {test_metrics['test_ter']:.1f}")

    # Upload results
    upload_results(test_metrics)

    # Save and upload model checkpoint
    if SAVE_CHECKPOINT:
        checkpoint_dir = "/tmp/checkpoint"
        print(f"\nSaving model checkpoint to {checkpoint_dir}...")
        model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)

        token = os.environ.get("HF_TOKEN")
        api = HfApi(token=token)

        _model_slug = ""
        if "1.3B" in MODEL_NAME or "1.3b" in MODEL_NAME:
            _model_slug = "nllb-1.3b/"
        elif "mbart" in MODEL_NAME.lower():
            _model_slug = "mbart/"

        ckpt_repo_path = f"{_model_slug}{EXPERIMENT}/{CONDITION}/seed{SEED}/checkpoint"
        print(f"Uploading checkpoint to {RESULTS_REPO}/{ckpt_repo_path}...")
        api.upload_folder(
            folder_path=checkpoint_dir,
            path_in_repo=ckpt_repo_path,
            repo_id=RESULTS_REPO,
            repo_type="dataset",
        )
        print("Checkpoint upload complete.")

    print("\nDone.")


if __name__ == "__main__":
    main()
