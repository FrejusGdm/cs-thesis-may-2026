#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml

sys.stdout.reconfigure(line_buffering=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


FINETUNING_ROOT = Path(__file__).resolve().parents[2]
VENDOR_DIR = FINETUNING_ROOT / "vendor" / "qwen3_tts_official"


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    flattened = {}
    for section in raw.values():
        if isinstance(section, dict):
            flattened.update(section)
    return flattened


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Qwen3-TTS Fine-tuning")
    parser.add_argument("--config", type=str)
    parser.add_argument("--init_model_path", type=str, default=None)
    parser.add_argument("--train_jsonl", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--speaker_name", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--attn_implementation", type=str, default=None)
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_model_id", type=str, default=None)
    return parser.parse_args()


def merge_config(args: argparse.Namespace) -> dict:
    config = {}
    if args.config:
        config = load_config(args.config)

    overrides = {
        "init_model_path": args.init_model_path,
        "train_jsonl": args.train_jsonl,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "num_epochs": args.num_epochs,
        "speaker_name": args.speaker_name,
        "output_dir": args.output_dir,
        "attn_implementation": args.attn_implementation,
    }
    for key, value in overrides.items():
        if value is not None:
            config[key] = value

    config.setdefault("init_model_path", "Qwen/Qwen3-TTS-12Hz-0.6B-Base")
    config.setdefault("train_jsonl", "")
    config.setdefault("batch_size", 2)
    config.setdefault("learning_rate", 2e-5)
    config.setdefault("num_epochs", 3)
    config.setdefault("speaker_name", "my_speaker")
    config.setdefault("output_dir", "./tts_output")
    config.setdefault("attn_implementation", "flash_attention_2")
    return config


def find_latest_checkpoint(output_dir: Path) -> Path | None:
    pattern = re.compile(r"^checkpoint-epoch-(\d+)$")
    best: tuple[int, Path] | None = None
    for path in output_dir.iterdir():
        match = pattern.match(path.name)
        if not match or not path.is_dir():
            continue
        epoch = int(match.group(1))
        if best is None or epoch > best[0]:
            best = (epoch, path)
    return best[1] if best else None


def build_vendor_command(config: dict, attn_implementation: str) -> list[str]:
    return [
        sys.executable,
        str(VENDOR_DIR / "sft_12hz.py"),
        "--init_model_path",
        config["init_model_path"],
        "--output_model_path",
        config["output_dir"],
        "--train_jsonl",
        config["train_jsonl"],
        "--batch_size",
        str(config["batch_size"]),
        "--lr",
        str(config["learning_rate"]),
        "--num_epochs",
        str(config["num_epochs"]),
        "--speaker_name",
        config["speaker_name"],
        "--attn_implementation",
        attn_implementation,
    ]


def run_vendor_train(config: dict) -> None:
    requested_attn = config["attn_implementation"]
    attempts = [requested_attn]
    if requested_attn == "flash_attention_2":
        attempts.append("eager")

    last_error: subprocess.CalledProcessError | None = None
    for attn_implementation in attempts:
        command = build_vendor_command(config, attn_implementation)
        logger.info("Running vendored Qwen3-TTS trainer with attn_implementation=%s", attn_implementation)
        try:
            subprocess.check_call(command, cwd=str(VENDOR_DIR))
            return
        except subprocess.CalledProcessError as exc:
            last_error = exc
            if attn_implementation != attempts[-1]:
                logger.warning("Vendored trainer failed with %s. Retrying with eager attention.", attn_implementation)
            else:
                break

    if last_error is not None:
        raise last_error


def upload_latest_checkpoint(output_dir: Path, hub_model_id: str) -> None:
    from huggingface_hub import HfApi

    latest = find_latest_checkpoint(output_dir)
    if latest is None:
        raise FileNotFoundError(f"No checkpoint found in {output_dir}")

    api = HfApi()
    api.upload_folder(
        folder_path=str(latest),
        repo_id=hub_model_id,
        repo_type="model",
    )
    logger.info("Uploaded %s to https://huggingface.co/%s", latest, hub_model_id)


def main() -> None:
    args = parse_args()
    config = merge_config(args)

    train_jsonl = Path(config["train_jsonl"]).resolve()
    if not train_jsonl.exists():
        raise SystemExit(f"Training file not found: {train_jsonl}")
    config["train_jsonl"] = str(train_jsonl)

    output_dir = Path(config["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config["output_dir"] = str(output_dir)

    logger.info("Qwen3-TTS thin wrapper")
    for key, value in sorted(config.items()):
        logger.info("  %s: %s", key, value)

    unused_keys = sorted(
        set(load_config(args.config).keys()) - {
            "init_model_path",
            "train_jsonl",
            "batch_size",
            "learning_rate",
            "num_epochs",
            "speaker_name",
            "output_dir",
            "attn_implementation",
        }
    ) if args.config else []
    if unused_keys:
        logger.info("Ignoring config keys not used by the vendored official trainer: %s", ", ".join(unused_keys))

    run_vendor_train(config)

    latest = find_latest_checkpoint(output_dir)
    if latest:
        logger.info("Latest checkpoint: %s", latest)

    if args.push_to_hub and args.hub_model_id:
        upload_latest_checkpoint(output_dir, args.hub_model_id)


if __name__ == "__main__":
    main()
