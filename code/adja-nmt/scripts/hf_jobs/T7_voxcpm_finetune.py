#!/usr/bin/env python3
from __future__ import annotations

"""
T7: Fine-tune VoxCPM on Adja via Hugging Face Jobs.

This launcher is self-contained so it can be submitted directly to HF Jobs.
It mirrors the upstream VoxCPM SFT/LoRA training contract:

- dataset materialized to JSONL manifests
- upstream `scripts/train_voxcpm_finetune.py`
- LoRA by default, full fine-tune via flag

References:
  - https://github.com/OpenBMB/VoxCPM
  - upstream docs/finetune.md
"""

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import time
import unicodedata
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)


UPSTREAM_REPO = "https://github.com/OpenBMB/VoxCPM.git"
UPSTREAM_DIR = Path("/tmp/VoxCPM")
LOCAL_DATASET_DIR = Path("/tmp/voxcpm_adja_data")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="T7 VoxCPM fine-tune on Adja")
    parser.add_argument("--model-name", default="openbmb/VoxCPM2")
    parser.add_argument("--dataset", default="JosueG/adja-tts-orpheus")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--full-finetune", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--num-iters", type=int, default=1000)
    parser.add_argument("--valid-interval", type=int, default=25)
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--results-repo", default="JosueG/adja-tts-results")
    parser.add_argument("--results-prefix", default="T7_voxcpm")
    parser.add_argument("--output-dir", default="/tmp/t7_voxcpm_output")
    return parser.parse_args()


def run(cmd: str, cwd: str | None = None) -> None:
    print(f"$ {cmd}")
    subprocess.check_call(cmd, shell=True, cwd=cwd)


def pip_install(packages: list[str]) -> None:
    quoted = " ".join(f'"{pkg}"' for pkg in packages)
    run(f"python3 -m pip install -q {quoted}")


def install_env() -> None:
    run("apt-get update -q && apt-get install -y -q git ffmpeg")
    pip_install(
        [
            "huggingface_hub>=0.34.0",
            "datasets>=3.4.1,<4.0.0",
            "soundfile",
            "librosa",
            "numpy",
        ]
    )


def clone_upstream() -> None:
    if UPSTREAM_DIR.exists():
        shutil.rmtree(UPSTREAM_DIR)
    run(f'git clone --depth 1 "{UPSTREAM_REPO}" "{UPSTREAM_DIR}"')
    run("python3 -m pip install -q .", cwd=str(UPSTREAM_DIR))


def nfc(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text.strip()).split())


def materialize_dataset(args: argparse.Namespace, token: str) -> tuple[Path, Path]:
    import soundfile as sf
    from datasets import load_dataset

    if LOCAL_DATASET_DIR.exists():
        shutil.rmtree(LOCAL_DATASET_DIR)
    wav_dir = LOCAL_DATASET_DIR / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(args.dataset, token=token, split="train")
    split1 = ds.train_test_split(test_size=0.1, seed=args.seed)
    split2 = split1["train"].train_test_split(test_size=0.1 / 0.9, seed=args.seed)
    train_ds = split2["train"]
    dev_ds = split2["test"]
    if args.dry_run:
        train_ds = train_ds.select(range(min(8, len(train_ds))))
        dev_ds = dev_ds.select(range(min(3, len(dev_ds))))

    def write_manifest(split, manifest_path: Path) -> None:
        rows = []
        for idx, example in enumerate(split):
            text = nfc(example["text"])
            audio = example["audio"]["array"]
            sr = example["audio"]["sampling_rate"]
            wav_path = wav_dir / f"{manifest_path.stem}_{idx:06d}.wav"
            sf.write(str(wav_path), audio, sr)
            rows.append({"audio": str(wav_path), "text": text, "dataset_id": 0})
        manifest_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )

    train_manifest = LOCAL_DATASET_DIR / "train.jsonl"
    val_manifest = LOCAL_DATASET_DIR / "validation.jsonl"
    write_manifest(train_ds, train_manifest)
    write_manifest(dev_ds, val_manifest)
    return train_manifest, val_manifest


def download_base_model(model_name: str) -> Path:
    from huggingface_hub import snapshot_download

    local_dir = Path("/tmp/voxcpm_base")
    if local_dir.exists():
        shutil.rmtree(local_dir)
    snapshot_download(repo_id=model_name, local_dir=str(local_dir))
    return local_dir


def read_model_sample_rate(model_dir: Path) -> int:
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    audio_vae_cfg = config.get("audio_vae_config", {})
    return int(audio_vae_cfg.get("sample_rate", 16000))


def build_train_config(
    args: argparse.Namespace,
    model_dir: Path,
    train_manifest: Path,
    val_manifest: Path,
    sample_rate: int,
) -> Path:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_dir = output_dir / "checkpoints"
    tb_dir = output_dir / "tensorboard"

    dry_num_iters = 2 if args.dry_run else args.num_iters
    dry_valid_interval = 1 if args.dry_run else args.valid_interval
    dry_save_interval = 1 if args.dry_run else args.save_interval

    config = {
        "pretrained_path": str(model_dir),
        "train_manifest": str(train_manifest),
        "val_manifest": str(val_manifest),
        "sample_rate": sample_rate,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum,
        "num_workers": 2,
        "num_iters": dry_num_iters,
        "log_interval": 1 if args.dry_run else 5,
        "valid_interval": dry_valid_interval,
        "save_interval": dry_save_interval,
        "learning_rate": args.learning_rate if not args.full_finetune else min(args.learning_rate, 1e-5),
        "weight_decay": 0.01,
        "warmup_steps": 1 if args.dry_run else 20,
        "max_steps": dry_num_iters,
        "max_batch_tokens": 0,
        "save_path": str(save_dir),
        "tensorboard": str(tb_dir),
        "lambdas": {"loss/diff": 1.0, "loss/stop": 1.0},
    }
    if not args.full_finetune:
        config["lora"] = {
            "enable_lm": True,
            "enable_dit": True,
            "enable_proj": False,
            "r": 16 if args.dry_run else 32,
            "alpha": 16,
            "dropout": 0.0,
            "target_modules_lm": ["q_proj", "v_proj", "k_proj", "o_proj"],
            "target_modules_dit": ["q_proj", "v_proj", "k_proj", "o_proj"],
        }
        config["hf_model_id"] = args.model_name
        config["distribute"] = True

    cfg_path = output_dir / "config.json"
    cfg_path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return cfg_path


def collect_metrics(args: argparse.Namespace, elapsed_min: float, sample_rate: int) -> Path:
    output_dir = Path(args.output_dir)
    ckpt_dir = output_dir / "checkpoints"
    latest = ckpt_dir / "latest"
    latest_exists = latest.exists()
    metrics = {
        "experiment": "T7",
        "model_name": args.model_name,
        "training_mode": "full_finetune" if args.full_finetune else "lora",
        "dataset": args.dataset,
        "seed": args.seed,
        "dry_run": args.dry_run,
        "sample_rate": sample_rate,
        "elapsed_min": round(elapsed_min, 1),
        "checkpoints_dir": str(ckpt_dir),
        "latest_exists": latest_exists,
        "checkpoint_files": sorted(p.name for p in latest.iterdir()) if latest_exists else [],
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return metrics_path


def generate_audio_samples(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir)
    samples_dir = output_dir / "generated_audio"
    samples_dir.mkdir(parents=True, exist_ok=True)

    latest_ckpt = output_dir / "checkpoints" / "latest"
    lora_cfg_path = latest_ckpt / "lora_config.json"
    if not lora_cfg_path.exists():
        raise FileNotFoundError(f"Missing LoRA config at {lora_cfg_path}")

    lora_info = json.loads(lora_cfg_path.read_text(encoding="utf-8"))
    base_model = lora_info.get("base_model")
    if not base_model:
        raise RuntimeError("base_model missing from lora_config.json")

    texts = [
        "Nye ŋkɔ nyé Tom",
        "ŋɖuɖu lɔwo nuɔn",
        "Eshilɔ ɖote yi mi jaja a ?",
    ]
    output_stem = samples_dir / "sample"
    infer_script = UPSTREAM_DIR / "scripts" / "test_voxcpm_lora_infer.py"
    if not infer_script.exists():
        raise FileNotFoundError(f"Missing upstream inference script: {infer_script}")

    for idx, text in enumerate(texts):
        run(
            "python3 "
            f"\"{infer_script}\" "
            f"--lora_ckpt \"{latest_ckpt}\" "
            f"--text {json.dumps(text)} "
            f"--output \"{output_stem}_{idx:02d}.wav\" "
            "--cfg_value 2.0 "
            "--inference_timesteps 10 "
            "--max_len 600",
            cwd=str(UPSTREAM_DIR),
        )
    manifest = []
    for wav_path in sorted(samples_dir.glob("*.wav")):
        manifest.append({"file": wav_path.name})
    if not manifest:
        raise RuntimeError(f"No audio samples were written under {samples_dir}")
    (samples_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return samples_dir


def push_results(args: argparse.Namespace, token: str, metrics_path: Path) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(args.results_repo, private=True, exist_ok=True)
    prefix = args.results_prefix.rstrip("/")
    api.upload_file(
        path_or_fileobj=metrics_path.read_bytes(),
        path_in_repo=f"{prefix}/metrics.json",
        repo_id=args.results_repo,
        token=token,
    )
    output_dir = Path(args.output_dir)
    ckpt_dir = output_dir / "checkpoints"
    if ckpt_dir.exists():
        try:
            api.upload_folder(
                folder_path=str(ckpt_dir),
                path_in_repo=f"{prefix}/checkpoints",
                repo_id=args.results_repo,
                token=token,
                ignore_patterns=["**/optimizer.pth", "**/scheduler.pth"],
            )
        except Exception as exc:
            print(f"Checkpoint upload best-effort: {exc}")
    samples_dir = output_dir / "generated_audio"
    if samples_dir.exists():
        try:
            api.upload_folder(
                folder_path=str(samples_dir),
                path_in_repo=f"{prefix}/generated_audio",
                repo_id=args.results_repo,
                token=token,
            )
        except Exception as exc:
            print(f"Generated audio upload best-effort: {exc}")


def main() -> None:
    args = parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is required")

    random.seed(args.seed)

    print("=" * 60)
    print("T7 VoxCPM fine-tune on Adja")
    print("=" * 60)
    print(f"Model: {args.model_name}")
    print(f"Dry run: {args.dry_run}")
    print(f"Mode: {'full fine-tune' if args.full_finetune else 'LoRA'}")
    print()

    install_env()
    clone_upstream()
    train_manifest, val_manifest = materialize_dataset(args, token)
    model_dir = download_base_model(args.model_name)
    sample_rate = read_model_sample_rate(model_dir)
    cfg_path = build_train_config(args, model_dir, train_manifest, val_manifest, sample_rate)

    train_script = UPSTREAM_DIR / "scripts" / "train_voxcpm_finetune.py"
    assert train_script.exists(), f"Missing upstream script: {train_script}"

    t0 = time.time()
    run(f'python3 "{train_script}" --config_path "{cfg_path}"', cwd=str(UPSTREAM_DIR))
    elapsed_min = (time.time() - t0) / 60.0

    try:
        generate_audio_samples(args)
    except Exception as exc:
        print(f"Warning: generated audio export failed: {exc}")

    metrics_path = collect_metrics(args, elapsed_min, sample_rate)
    print(metrics_path.read_text(encoding="utf-8"))

    if args.push_to_hub:
        push_results(args, token, metrics_path)

    print("T7 complete.")


if __name__ == "__main__":
    main()
