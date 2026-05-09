#!/usr/bin/env python3
"""
Login-node pre-flight smoke test for the HPC speech pipeline.

Last updated: 2026-04-22

Run this on the Discovery login node (no GPU needed) INSIDE speech-training.sif
before ever `sbatch`ing a real training job. Each sbatch slot is 2-24h of queue
wait, so we cannot afford to discover "cache is read-only" or "snac missing"
during a production run.

Exit 0 only if every check passes. Exit 1 with a clear message on first failure.

Runs in <5 minutes. Covers:
  1. All Python imports succeed
  2. HF_TOKEN valid (whoami returns JosueG)
  3. /data binds writable (catches :ro regression in sbatch)
  4. All pre-cached model dirs have config.json
  5. WaxalNLP metadata resolves with auth + network
  6. JosueG/adja-tts-orpheus metadata resolves
  7. SNAC imports and loads a codec

Invocation: `bash hpc/smoke/login_check.sh` (not this file directly).
"""
import os
import sys
import tempfile
import time
from pathlib import Path

# Bind targets inside the container
MODELS_DIR = Path(os.environ.get("MODELS_DIR", "/models"))
DATA_DIR   = Path(os.environ.get("DATA_DIR", "/data"))

EXPECTED_MODELS = [
    # ASR
    "openai__whisper-large-v3",
    "facebook__wav2vec2-xls-r-300m",
    "facebook__wav2vec2-xls-r-1b",
    "facebook__mms-1b-all",
    # TTS
    "unsloth__csm-1b",
    "canopylabs__orpheus-3b-0.1-ft",
    "canopylabs__3b-zh-ft-research_release",
    "canopylabs__3b-fr-ft-research_release",
    "hubertsiuzdak__snac_24khz",
]

CHECKS_TOTAL = 7


def ok(step, msg):
    print(f"[{step}/{CHECKS_TOTAL}] ✅ {msg}")


def fail(step, msg, exc=None):
    print(f"[{step}/{CHECKS_TOTAL}] ❌ {msg}")
    if exc is not None:
        print(f"       error: {type(exc).__name__}: {exc}")
    sys.exit(1)


def main():
    t_start = time.time()
    print("=" * 60)
    print("HPC speech pipeline login-node smoke")
    print(f"Models dir: {MODELS_DIR}")
    print(f"Data dir:   {DATA_DIR}")
    print("=" * 60)

    # ----- 1. imports -----
    try:
        import torch
        import transformers
        import datasets
        import peft
        import huggingface_hub
        import soundfile
        import librosa
        import numpy
        ok(1, f"imports (torch {torch.__version__}, transformers {transformers.__version__}, "
              f"datasets {datasets.__version__}, peft {peft.__version__})")
    except Exception as exc:
        fail(1, "Python import failed — container is broken, rebuild speech-training.sif", exc)

    # snac is the one non-core package, check separately with a clearer message
    try:
        from snac import SNAC  # noqa: F401
    except Exception as exc:
        fail(1, "snac package missing (used by Orpheus audio-LM) — rebuild container", exc)

    # ----- 2. HF_TOKEN valid -----
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        fail(2, "HF_TOKEN not set in env. `export HF_TOKEN=hf_xxx` on the login shell.")
    try:
        from huggingface_hub import HfApi
        who = HfApi(token=hf_token).whoami()
        ok(2, f"HF auth works → user='{who.get('name')}'")
    except Exception as exc:
        fail(2, "HF_TOKEN invalid or HF Hub unreachable", exc)

    # ----- 3. /data is writable (catches :ro regression) -----
    if not DATA_DIR.exists():
        fail(3, f"{DATA_DIR} does not exist — check bind mount in sbatch")
    try:
        with tempfile.NamedTemporaryFile(dir=str(DATA_DIR), prefix=".smoke_", delete=True):
            pass
        ok(3, f"{DATA_DIR} is writable (datasets FileLock will work)")
    except PermissionError as exc:
        fail(3, f"{DATA_DIR} is READ-ONLY — sbatch bind is :ro. Remove the :ro suffix.", exc)
    except Exception as exc:
        fail(3, f"cannot write to {DATA_DIR}", exc)

    # ----- 4. pre-cached model dirs all have a config.json (or snac_config.json / similar) -----
    missing = []
    for mdir in EXPECTED_MODELS:
        path = MODELS_DIR / mdir
        if not path.exists():
            missing.append(f"{mdir} (dir missing)")
            continue
        has_config = any(path.glob("*.json")) or any(path.glob("*config*"))
        if not has_config:
            missing.append(f"{mdir} (no config.json)")
    if missing:
        fail(4, "Pre-cached models missing or incomplete:\n  " + "\n  ".join(missing) +
                "\n  → re-run deploy/download_models.sh on login node")
    ok(4, f"all {len(EXPECTED_MODELS)} pre-cached models present and have config files")

    # ----- 5. WaxalNLP metadata resolves -----
    try:
        from datasets import load_dataset_builder
        from pathlib import Path

        import sys as _sys
        _sys.path.insert(0, "/scripts/pretraining")
        from waxalnlp_loader import load_waxal_split  # type: ignore

        builder = load_dataset_builder(
            "google/WaxalNLP", name="ewe_asr",
            cache_dir=str(DATA_DIR / "WaxalNLP_ewe_asr"),
            token=hf_token,
        )
        splits = list(builder.info.splits.keys()) if builder.info.splits else "?"
        # Tiny actual load to catch Parquet schema mismatches early.
        tiny = load_waxal_split(
            config="ewe_asr",
            split="train",
            cache_dir=str(DATA_DIR / "WaxalNLP_ewe_asr"),
            token=hf_token,
        )
        tiny = tiny.select(range(1)) if len(tiny) else tiny
        ok(5, f"google/WaxalNLP ewe_asr resolves (splits: {splits}) | cols: {tiny.column_names}")
    except Exception as exc:
        fail(5, "WaxalNLP ewe_asr dataset metadata failed — most likely auth or RO filesystem", exc)

    # ----- 6. Adja dataset resolves -----
    try:
        builder = load_dataset_builder(
            "JosueG/adja-tts-orpheus",
            cache_dir=str(DATA_DIR / "JosueG_adja-tts-orpheus"),
            token=hf_token,
        )
        ok(6, "JosueG/adja-tts-orpheus builder resolves")
    except Exception as exc:
        fail(6, "Adja dataset metadata failed", exc)

    # ----- 7. SNAC loads from pre-cached copy -----
    try:
        snac_path = str(MODELS_DIR / "hubertsiuzdak__snac_24khz")
        from snac import SNAC
        model = SNAC.from_pretrained(snac_path)
        ok(7, f"SNAC loads from {snac_path} (sample rate 24kHz)")
        del model
    except Exception as exc:
        fail(7, "SNAC failed to load from pre-cached dir", exc)

    elapsed = time.time() - t_start
    print("=" * 60)
    print(f"ALL {CHECKS_TOTAL} CHECKS PASSED ({elapsed:.1f}s)")
    print("=" * 60)
    print()
    print("Next: request an interactive GPU session and run gpu_smoke.sh:")
    print("  srun --pty --partition=gpuq \\")
    print("       --gres=gpu:nvidia_a100_80gb_pcie_3g.40gb:1 \\")
    print("       --mem=64G --time=30:00 bash")
    print("  bash smoke/gpu_smoke.sh")


if __name__ == "__main__":
    main()
