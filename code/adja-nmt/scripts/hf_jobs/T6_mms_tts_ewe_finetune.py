#!/usr/bin/env python3
from __future__ import annotations

"""
T6: Fine-tune MMS-TTS-Ewe on Adja via ylacombe/finetune-hf-vits on HF Jobs.

Created: 2026-04-18
Author: Adja speech exploration (Josue Godeme)

Why T6:
- T1 Sesame CSM LoRA r=32 plateaued at eval loss 6.488 — generated audio is noise.
  Hypothesis from `ideas/tts-models-to-try.md`: the pretraining prior is wrong
  (CSM is English-centric). Adja and Ewe are both Gbe-family Niger-Congo tonal
  languages with a shared phoneme inventory (ɛ, ɔ, ŋ, ɖ, tone system).
- `facebook/mms-tts-ewe` is a VITS-based TTS model already trained on Ewe. Starting
  from a model that already "speaks" a Gbe-family language is our bet for turning
  1.7h of Adja data into intelligible speech.

Why ylacombe/finetune-hf-vits:
- MMS-TTS / VITS fine-tuning requires a GAN-aware trainer (generator + discriminator),
  which `transformers.Trainer` alone cannot drive. ylacombe's repo is the de-facto
  community trainer for HF VITS/MMS models. README:
  https://github.com/ylacombe/finetune-hf-vits/blob/main/README.md
- Their `finetune_mms.json` (Gujarati MMS) is our template.

Container: pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel  (same as T1)
Runtime:   HF Jobs l40sx1 (48GB, $1.80/hr)
Licence:   CC-BY-NC 4.0 (MMS base) — research-only, NOT for commercial deployment.

Flow (executed in one HF job):
1. install dependencies (torch is provided by the container)
2. clone ylacombe/finetune-hf-vits, install its requirements
3. build the monotonic_align cython extension
4. convert the MMS-TTS-Ewe discriminator (required for fine-tuning)
5. prepare Adja dataset:
   - load `JosueG/adja-tts-orpheus`
   - NFC-normalise text (critical for ɛ, ɔ, ŋ, ɖ, tone marks)
   - 80/10/10 seed=42 split (same protocol as every other experiment in this repo)
   - resample audio to 16 kHz (MMS-TTS native sample rate)
   - push to `JosueG/adja-tts-mms-ready` (private) so ylacombe's trainer can
     load it via `datasets.load_dataset`
6. write a JSON config mirroring `finetune_mms.json` but pointing at our
   converted-discriminator model + prepared dataset
7. `accelerate launch run_vits_finetuning.py <config.json>`
8. generate sample Adja wavs using `transformers.pipeline("text-to-speech", ...)`
9. push model, metrics, and wavs to `JosueG/adja-tts-results/<prefix>/`

Usage on HF Jobs:
    SCRIPT_B64=$(base64 < scripts/hf_jobs/T6_mms_tts_ewe_finetune.py)
    hf jobs run pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel \
      --flavor l40sx1 --secrets HF_TOKEN --timeout 4h -d \
      -- bash -c "echo '$SCRIPT_B64' | base64 -d > /tmp/T6.py && python /tmp/T6.py --push-to-hub"

Assumptions (2026-04-18):
- ylacombe's entry point is `run_vits_finetuning.py` launched via `accelerate launch`
  (verified against the repo's main README 2026-04-18).
- The MMS-TTS-Ewe tokenizer handles raw Unicode Adja characters after NFC normalize.
  If we hit "uroman required" errors for Ewe, a fallback uroman step needs adding.
- 1.7h of Adja audio is sufficient to fine-tune VITS per community reports
  (ylacombe README claims 80-150 samples suffice for 20-min runs).
- Discriminator conversion for Ewe works via the
  `convert_original_discriminator_checkpoint.py --language_code ewe` helper.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import unicodedata
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)


YLACOMBE_REPO = "https://github.com/ylacombe/finetune-hf-vits.git"
YLACOMBE_DIR = Path("/tmp/finetune-hf-vits")
CONVERTED_MODEL_DIR = Path("/tmp/mms-ewe-train-with-disc")
MMS_LANGUAGE_CODE = "ewe"  # ISO 639-3 — facebook/mms-tts-ewe base
TARGET_SR = 16000  # MMS-TTS native sample rate


# ---------------------------------------------------------------------------
# Arg parsing
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="T6 MMS-TTS-Ewe -> Adja fine-tune via ylacombe/finetune-hf-vits")
    parser.add_argument("--dataset", default="JosueG/adja-tts-orpheus", help="Source private HF dataset")
    parser.add_argument("--prepped-dataset-repo", default="JosueG/adja-tts-mms-ready",
                        help="Repo we push the preprocessed Adja dataset to so ylacombe's trainer can load it")
    parser.add_argument("--output-dir", default="/tmp/mms_ewe_adja_output", help="Local training output")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16, help="Per-device train batch; VITS is small so 16 is fine on L40S")
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="2e-5 matches ylacombe's MMS example")
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--max-eval-samples", type=int, default=25)
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--results-repo", default="JosueG/adja-tts-results")
    parser.add_argument("--results-prefix", default="T6_mms_ewe_20ep_2026-04-18")
    parser.add_argument("--sample-text",
                        default="Nye ŋkɔ nyé Tom",
                        help="Sample Adja text to synthesize once training finishes")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------
def run(cmd: str, check: bool = True, cwd: str | None = None) -> None:
    """Run a shell command, echo it, exit if fails when `check`."""
    print(f"$ {cmd}", flush=True)
    rc = subprocess.call(cmd, shell=True, cwd=cwd)
    if check and rc != 0:
        raise SystemExit(f"Command failed (rc={rc}): {cmd}")


def install_env() -> None:
    """Install container-side deps. Torch is provided by the pytorch image."""
    # The pytorch container does NOT ship git. ylacombe clone needs it.
    run("apt-get update -q && apt-get install -y -q git build-essential", check=False)
    # Pin transformers to 4.51.3 — newer versions removed `send_example_telemetry` from
    # transformers.utils, which ylacombe's run_vits_finetuning.py imports at startup.
    # Pin matplotlib < 3.8 — ylacombe's eval plotting calls FigureCanvasAgg.tostring_rgb,
    # which was removed in matplotlib 3.8. Pinning to 3.7.x keeps that API available.
    run("pip install -q 'transformers==4.51.3' 'datasets[audio]>=3.4.1,<4.0.0' "
        "'accelerate>=0.24.1' 'matplotlib<3.8' wandb tensorboard Cython")
    run("pip install -q soundfile librosa numpy scipy sentencepiece protobuf "
        "'huggingface_hub>=0.34.0' hf_transfer unidecode")
    # torchaudio is pre-installed and matches container torch — don't touch it.


def clone_and_build_ylacombe() -> None:
    if YLACOMBE_DIR.exists():
        shutil.rmtree(YLACOMBE_DIR)
    run(f"git clone --depth 1 {YLACOMBE_REPO} {YLACOMBE_DIR}")
    run(f"pip install -q -r {YLACOMBE_DIR}/requirements.txt")
    # Patch: newer transformers' VitsConfig (loaded from facebook/mms-tts-ewe) doesn't
    # expose `pad_token_id` on the config object, so ylacombe's line
    # `nn.Embedding(vocab_size, hidden_size, config.pad_token_id)` crashes. Default
    # to 0 via getattr, matching the historical behaviour.
    run(
        "sed -i 's/nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)/"
        "nn.Embedding(config.vocab_size, config.hidden_size, getattr(config, \"pad_token_id\", 0))/g' "
        f"{YLACOMBE_DIR}/utils/modeling_vits_training.py"
    )
    # Build the cython monotonic alignment search — mandatory per ylacombe README.
    ma_parent = YLACOMBE_DIR / "monotonic_align"
    ma_inner = ma_parent / "monotonic_align"
    ma_inner.mkdir(parents=True, exist_ok=True)
    run("python setup.py build_ext --inplace", cwd=str(ma_parent))


def convert_discriminator() -> None:
    """One-time step per language: produces a VITS checkpoint with a trainable
    discriminator, rooted at /tmp/mms-ewe-train-with-disc."""
    if CONVERTED_MODEL_DIR.exists():
        shutil.rmtree(CONVERTED_MODEL_DIR)
    CONVERTED_MODEL_DIR.mkdir(parents=True)
    run(
        f"python {YLACOMBE_DIR}/convert_original_discriminator_checkpoint.py "
        f"--language_code {MMS_LANGUAGE_CODE} "
        f"--pytorch_dump_folder_path {CONVERTED_MODEL_DIR}"
    )


# ---------------------------------------------------------------------------
# Dataset prep + push
# ---------------------------------------------------------------------------
def nfc(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text.strip()).split())


def prepare_and_push_dataset(args: argparse.Namespace, token: str) -> str:
    """Load Adja dataset, NFC-normalise text, 80/10/10 split seed 42, resample to
    16 kHz, push to `prepped-dataset-repo` so ylacombe's trainer can load it."""
    from datasets import Audio, DatasetDict, load_dataset

    print(f"Loading source dataset: {args.dataset}")
    ds = load_dataset(args.dataset, token=token, split="train")
    print(f"  {len(ds)} utterances")

    print("NFC-normalising text + adding speaker_id=0 column")
    ds = ds.map(
        lambda ex: {"text": nfc(ex["text"]), "speaker_id": 0},
        desc="Normalize text",
    )

    print(f"Casting audio to {TARGET_SR} Hz")
    ds = ds.cast_column("audio", Audio(sampling_rate=TARGET_SR))

    print(f"Splitting 80/10/10 seed={args.seed}")
    split1 = ds.train_test_split(test_size=0.1, seed=args.seed)
    split2 = split1["train"].train_test_split(test_size=0.1 / 0.9, seed=args.seed)
    prepped = DatasetDict({
        "train": split2["train"],
        "dev": split2["test"],
        "test": split1["test"],
    })
    print(f"  train={len(prepped['train'])} dev={len(prepped['dev'])} test={len(prepped['test'])}")

    print(f"Pushing to {args.prepped_dataset_repo} (private)")
    prepped.push_to_hub(args.prepped_dataset_repo, token=token, private=True)
    return args.prepped_dataset_repo


# ---------------------------------------------------------------------------
# Training config (mirrors ylacombe's finetune_mms.json schema)
# ---------------------------------------------------------------------------
def write_training_config(args: argparse.Namespace, prepped_repo: str) -> Path:
    config = {
        "project_name": f"mms_ewe_adja_{args.num_epochs}ep",
        "push_to_hub": False,
        "report_to": ["tensorboard"],
        "overwrite_output_dir": True,
        "output_dir": args.output_dir,

        "model_name_or_path": str(CONVERTED_MODEL_DIR),

        "dataset_name": prepped_repo,
        "audio_column_name": "audio",
        "text_column_name": "text",
        "train_split_name": "train",
        "eval_split_name": "dev",
        "speaker_id_column_name": "speaker_id",
        "override_speaker_embeddings": True,
        # No speaker filter — single-speaker dataset, all rows have speaker_id=0.

        "full_generation_sample_text": args.sample_text,

        "max_duration_in_seconds": 20,
        "min_duration_in_seconds": 1.0,
        "max_tokens_length": 500,

        "preprocessing_num_workers": 4,

        # Training hyperparams — mirror ylacombe's Gujarati MMS example
        "do_train": True,
        "num_train_epochs": args.num_epochs,
        "gradient_accumulation_steps": args.grad_accum,
        "gradient_checkpointing": False,
        "per_device_train_batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "adam_beta1": 0.8,
        "adam_beta2": 0.99,
        "warmup_ratio": 0.01,
        "group_by_length": False,

        "do_eval": True,
        "eval_steps": args.eval_steps,
        "per_device_eval_batch_size": args.batch_size,
        "max_eval_samples": args.max_eval_samples,
        "do_step_schedule_per_epoch": True,

        # GAN loss weights — ylacombe Gujarati defaults
        "weight_disc": 3,
        "weight_fmaps": 1,
        "weight_gen": 1,
        "weight_kl": 1.5,
        "weight_duration": 1,
        "weight_mel": 35,

        "fp16": True,
        "seed": args.seed,
    }

    cfg_path = Path(args.output_dir) / "training_config.json"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote training config: {cfg_path}")
    return cfg_path


# ---------------------------------------------------------------------------
# Training launch
# ---------------------------------------------------------------------------
def launch_training(cfg_path: Path) -> None:
    run("accelerate config default", check=False)
    entry = YLACOMBE_DIR / "run_vits_finetuning.py"
    assert entry.exists(), f"ylacombe entry script missing: {entry}"
    run(f"accelerate launch {entry} {cfg_path}")


# ---------------------------------------------------------------------------
# Post-training: generate samples + push everything to the Hub
# ---------------------------------------------------------------------------
def generate_samples(args: argparse.Namespace) -> Path:
    """After training, run a few inference passes and dump .wavs into a folder."""
    import scipy.io.wavfile
    from transformers import pipeline

    model_dir = Path(args.output_dir)
    print(f"Loading fine-tuned model from: {model_dir}")
    synth = pipeline("text-to-speech", model=str(model_dir), device=0)

    out_dir = model_dir / "generated"
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = [
        args.sample_text,
        "Tom trɔ yi kpanŋkɔ wezexu",
        "Ele lɔ awu ehoci lɔ sa",
        "Ɛ yi gbɛ̀",
        "Mì ɖo alɔ ji",
    ]
    results = []
    for i, text in enumerate(samples):
        text_n = nfc(text)
        try:
            out = synth(text_n)
            wav_path = out_dir / f"plain_{i:02d}.wav"
            scipy.io.wavfile.write(str(wav_path), rate=out["sampling_rate"], data=out["audio"][0])
            dur = len(out["audio"][0]) / out["sampling_rate"]
            results.append({"text": text_n, "file": wav_path.name, "duration_sec": round(dur, 2)})
            print(f"  [{i}] '{text_n}' -> {dur:.1f}s")
        except Exception as exc:
            print(f"  [{i}] FAILED: {exc}")
            results.append({"text": text_n, "error": str(exc)})

    (model_dir / "generated_samples.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n"
    )
    return out_dir


def push_results(args: argparse.Namespace, token: str, elapsed_min: float) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(args.results_repo, private=True, exist_ok=True)

    model_dir = Path(args.output_dir)
    prefix = args.results_prefix

    metrics = {
        "experiment": "T6",
        "path": "mms_tts_ewe_adja_via_ylacombe",
        "created_date": "2026-04-18",
        "seed": args.seed,
        "source_dataset": args.dataset,
        "prepped_dataset_repo": args.prepped_dataset_repo,
        "base_model": f"facebook/mms-tts-{MMS_LANGUAGE_CODE}",
        "converted_base_model_dir": str(CONVERTED_MODEL_DIR),
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "learning_rate": args.learning_rate,
        "eval_steps": args.eval_steps,
        "max_eval_samples": args.max_eval_samples,
        "training_time_min": round(elapsed_min, 1),
        "sample_rate": TARGET_SR,
    }
    (model_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n"
    )

    api.upload_file(
        path_or_fileobj=(model_dir / "metrics.json").read_bytes(),
        path_in_repo=f"{prefix}/metrics.json",
        repo_id=args.results_repo,
        token=token,
    )
    for sub in ("generated",):
        sub_path = model_dir / sub
        if not sub_path.exists():
            continue
        api.upload_folder(
            folder_path=str(sub_path),
            path_in_repo=f"{prefix}/{sub}",
            repo_id=args.results_repo,
            token=token,
        )
    # ylacombe's trainer writes the final model directly into `output_dir`.
    # Upload the flat model files (config, safetensors, tokenizer) under prefix/model/.
    # We exclude the "generated" subdir since it's uploaded separately.
    try:
        api.upload_folder(
            folder_path=str(model_dir),
            path_in_repo=f"{prefix}/model",
            repo_id=args.results_repo,
            token=token,
            ignore_patterns=["generated/*", "*.tmp", "checkpoint-*", "runs/*"],
        )
    except Exception as exc:
        print(f"Model upload best-effort: {exc}")

    for name in ("training_config.json", "generated_samples.json"):
        fp = model_dir / name
        if fp.exists():
            api.upload_file(
                path_or_fileobj=fp.read_bytes(),
                path_in_repo=f"{prefix}/{name}",
                repo_id=args.results_repo,
                token=token,
            )
    print(f"Pushed to https://huggingface.co/{args.results_repo}/tree/main/{prefix}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is required")

    print("=" * 60)
    print("T6 MMS-TTS-Ewe -> Adja fine-tune (2026-04-18)")
    print("=" * 60)

    install_env()

    import torch
    print(f"torch={torch.__version__}, CUDA={torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()

    print("===== STAGE 1: clone + build ylacombe =====")
    clone_and_build_ylacombe()

    print("===== STAGE 2: convert Ewe discriminator =====")
    convert_discriminator()

    print("===== STAGE 3: prepare + push Adja dataset =====")
    prepped_repo = prepare_and_push_dataset(args, token)

    print("===== STAGE 4: write training config =====")
    cfg_path = write_training_config(args, prepped_repo)

    print("===== STAGE 5: accelerate launch =====")
    t0 = time.time()
    launch_training(cfg_path)
    elapsed_min = (time.time() - t0) / 60.0
    print(f"Training finished in {elapsed_min:.1f} min")

    print("===== STAGE 6: generate Adja samples =====")
    generate_samples(args)

    if args.push_to_hub:
        print("===== STAGE 7: push results to Hub =====")
        push_results(args, token, elapsed_min)

    print("T6 MMS-TTS-Ewe finetune complete.")


if __name__ == "__main__":
    main()
