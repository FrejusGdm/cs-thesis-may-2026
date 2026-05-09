#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#   "datasets",
#   "soundfile",
#   "librosa",
#   "numpy",
#   "huggingface-hub",
#   "torch==2.8.0",
#   "torchaudio==2.8.0",
#   "fairseq2==0.6.0",
#   "pyarrow",
#   "pandas",
#   "numba",
#   "polars>=1.29.0",
#   "retrying",
#   "xxhash",
#   "tensorboard",
#   "torchcodec",
# ]
# ///
from __future__ import annotations

"""
Self-contained OmniASR fine-tuning launcher for Hugging Face Jobs.

This script follows the repo's working Omni HF pattern:
- keep the script self-contained for `hf jobs uv run`
- reuse the existing Adja split convention used by the other HF launchers
- materialize a manifest dataset on the fly
- download the official Meta OmniASR source tree for the training recipe
- avoid collisions by requiring explicit experiment/result identifiers

The script is intentionally conservative:
- smoke runs use small sample caps and short schedules
- model outputs and logs are uploaded to distinct Hub locations
- the official recipe is used for the actual fine-tune path
"""

import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import unicodedata
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

sys.stdout.reconfigure(line_buffering=True)


class TeeStream:
    def __init__(self, *streams: Any) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


@dataclass(frozen=True)
class FineTuneDefaults:
    model_card: str
    tokenizer_ref: str
    lr: float
    max_audio_len: int
    max_num_elements: int
    grad_accumulation: int
    default_flavor: str
    default_timeout: str


FAMILY_DEFAULTS: dict[str, FineTuneDefaults] = {
    "ctc": FineTuneDefaults(
        model_card="omniASR_CTC_7B_v2",
        tokenizer_ref="omniASR_tokenizer_written_v2",
        lr=1e-5,
        max_audio_len=30 * 16_000,
        max_num_elements=30 * 16_000,
        grad_accumulation=8,
        default_flavor="a100-large",
        default_timeout="8h",
    ),
    "llm": FineTuneDefaults(
        model_card="omniASR_LLM_7B_v2",
        tokenizer_ref="omniASR_tokenizer_written_v2",
        lr=5e-6,
        max_audio_len=10 * 16_000,
        max_num_elements=10 * 16_000,
        grad_accumulation=16,
        default_flavor="a100-large",
        default_timeout="10h",
    ),
}


def require_env(name: str, default: str | None = None, required: bool = True) -> str:
    value = os.environ.get(name, default)
    if required and not value:
        raise SystemExit(f"{name} is required")
    return value or ""


def env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def setup_tee_logging(log_path: Path) -> tuple[Any, Any, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("a", encoding="utf-8")
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = TeeStream(original_stdout, log_handle)
    sys.stderr = TeeStream(original_stderr, log_handle)
    return log_handle, original_stdout, original_stderr


def restore_tee_logging(log_handle: Any, original_stdout: Any, original_stderr: Any) -> None:
    sys.stdout.flush()
    sys.stderr.flush()
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    log_handle.close()


def run_command(command: list[str], cwd: Path | None = None) -> None:
    print(f"$ {' '.join(command)}")
    subprocess.run(command, cwd=str(cwd) if cwd else None, check=True)


def normalize_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text.strip()).split())


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def ensure_upstream_repo(upstream_dir: Path, ref: str) -> Path:
    if upstream_dir.exists():
        print(f"Using cached OmniASR source tree: {upstream_dir}")
        return upstream_dir

    archive_path = upstream_dir.parent / f"omnilingual-asr-{ref}.tar.gz"
    url = f"https://github.com/facebookresearch/omnilingual-asr/archive/refs/heads/{ref}.tar.gz"
    print(f"Downloading OmniASR source archive from {url}")
    upstream_dir.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, archive_path)

    with tarfile.open(archive_path, "r:gz") as handle:
        handle.extractall(upstream_dir.parent)

    extracted_dir = upstream_dir.parent / f"omnilingual-asr-{ref}"
    if not extracted_dir.exists():
        raise FileNotFoundError(f"Expected extracted source tree at {extracted_dir}")

    extracted_dir.rename(upstream_dir)
    return upstream_dir


def ensure_editable_install(upstream_dir: Path) -> None:
    uv = shutil.which("uv")
    if uv:
        command = [uv, "pip", "install", "--python", sys.executable, "--no-deps", "-e", str(upstream_dir)]
    else:
        command = [sys.executable, "-m", "pip", "install", "--no-deps", "-e", str(upstream_dir)]
    run_command(command)


def create_splits(dataset, seed: int):
    split1 = dataset.train_test_split(test_size=0.1, seed=seed)
    split2 = split1["train"].train_test_split(test_size=0.1 / 0.9, seed=seed)
    return {"train": split2["train"], "dev": split2["test"], "test": split1["test"]}


def cap_split(split, limit: int):
    if limit <= 0 or len(split) <= limit:
        return split
    return split.select(range(limit))


def materialize_manifest_dataset(
    *,
    dataset_id: str,
    token: str,
    workspace_dir: Path,
    seed: int,
    max_train_samples: int,
    max_dev_samples: int,
    max_test_samples: int,
) -> dict[str, Any]:
    import librosa
    from datasets import load_dataset

    dataset = load_dataset(dataset_id, token=token, split="train")
    splits = create_splits(dataset, seed)
    splits["train"] = cap_split(splits["train"], max_train_samples)
    splits["dev"] = cap_split(splits["dev"], max_dev_samples)
    splits["test"] = cap_split(splits["test"], max_test_samples)

    audio_root = workspace_dir / "manifest_dataset" / "audio"
    manifest_dir = workspace_dir / "manifest_dataset" / "manifests"
    audio_root.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "dataset_id": dataset_id,
        "seed": seed,
        "audio_root": str(audio_root),
        "manifest_dir": str(manifest_dir),
        "splits": {},
    }

    for split_name, split_data in splits.items():
        split_audio_dir = audio_root / split_name
        split_audio_dir.mkdir(parents=True, exist_ok=True)
        tsv_lines = [str(audio_root)]
        wrd_lines: list[str] = []
        total_audio_sec = 0.0

        for index in range(len(split_data)):
            sample = split_data[index]
            audio = np.asarray(sample["audio"]["array"], dtype=np.float32)
            sampling_rate = int(sample["audio"]["sampling_rate"])
            if audio.ndim > 1:
                audio = np.mean(audio, axis=-1)
            if sampling_rate != 16_000:
                audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=16_000)
                sampling_rate = 16_000

            rel_path = Path(split_name) / f"{split_name}_{index:05d}.wav"
            abs_path = audio_root / rel_path
            sf.write(abs_path, audio, sampling_rate)

            num_frames = int(audio.shape[0])
            total_audio_sec += num_frames / 16_000.0
            tsv_lines.append(f"{rel_path.as_posix()}\t{num_frames}")
            wrd_lines.append(normalize_text(sample["text"]))

        write_text(manifest_dir / f"{split_name}.tsv", "\n".join(tsv_lines) + "\n")
        write_text(manifest_dir / f"{split_name}.wrd", "\n".join(wrd_lines) + "\n")

        summary["splits"][split_name] = {
            "num_records": len(split_data),
            "total_audio_sec": round(total_audio_sec, 3),
        }

    write_json(workspace_dir / "manifest_dataset" / "summary.json", summary)
    return summary


def write_dataset_card(
    *,
    upstream_dir: Path,
    dataset_name: str,
    manifest_dir: Path,
    tokenizer_ref: str,
) -> Path:
    card_dir = upstream_dir / "src" / "omnilingual_asr" / "cards" / "datasets"
    card_path = card_dir / f"{dataset_name}.yaml"
    card = (
        f"name: {dataset_name}\n"
        f"dataset_family: manifest_asr_dataset\n"
        f"dataset_config:\n"
        f"  data: {manifest_dir}\n"
        f"tokenizer_ref: {tokenizer_ref}\n"
    )
    write_text(card_path, card)
    return card_path


def render_train_config(
    *,
    family: str,
    model_card: str,
    dataset_name: str,
    tokenizer_ref: str,
    learning_rate: float,
    freeze_encoder_for_n_steps: int,
    max_audio_len: int,
    max_num_elements: int,
    grad_accumulation: int,
    num_steps: int,
    validate_every_n_steps: int,
    publish_metrics_every_n_steps: int,
    checkpoint_every_n_steps: int,
    seed: int,
    use_fsdp: bool,
) -> str:
    trainer_lines = [
        "trainer:",
    ]
    if use_fsdp:
        trainer_lines.extend(
            [
                '  data_parallelism: "fsdp"',
                "  fsdp:",
                '    granularity: "stack"',
                '    version: "v1"',
                "    fp32_reduce: false",
            ]
        )
    trainer_lines.extend(
        [
            f"  freeze_encoder_for_n_steps: {freeze_encoder_for_n_steps}",
            "  mixed_precision:",
            '    dtype: "torch.bfloat16"',
            "  grad_accumulation:",
            f"    num_batches: {grad_accumulation}",
        ]
    )

    return "\n".join(
        [
            f'model:',
            f'  name: "{model_card}"',
            "",
            "dataset:",
            f'  name: "{dataset_name}"',
            '  train_split: "train"',
            '  valid_split: "dev"',
            '  storage_mode: "MANIFEST"',
            '  task_mode: "ASR"',
            "  manifest_storage_config:",
            "    read_text: true",
            "  asr_task_config:",
            "    min_audio_len: 0",
            f"    max_audio_len: {max_audio_len}",
            f"    max_num_elements: {max_num_elements}",
            "    batch_shuffle_window: 1",
            "    normalize_audio: true",
            "    example_shuffle_window: 1",
            "",
            "tokenizer:",
            f'  name: "{tokenizer_ref}"',
            "",
            "optimizer:",
            "  config:",
            f"    lr: {learning_rate}",
            "",
            *trainer_lines,
            "",
            "regime:",
            f"  num_steps: {num_steps}",
            "  validate_after_n_steps: 1",
            f"  validate_every_n_steps: {validate_every_n_steps}",
            f"  checkpoint_every_n_steps: {checkpoint_every_n_steps}",
            f"  publish_metrics_every_n_steps: {publish_metrics_every_n_steps}",
            "",
            "common:",
            f"  seed: {seed}",
            "",
        ]
    )


def upload_file_if_exists(api, repo_id: str, local_path: Path, path_in_repo: str, repo_type: str = "dataset") -> None:
    if not local_path.exists():
        return
    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type=repo_type,
    )


def upload_folder_if_exists(api, repo_id: str, local_path: Path, path_in_repo: str | None = None, repo_type: str = "model") -> None:
    if not local_path.exists():
        return
    api.upload_folder(
        folder_path=str(local_path),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type=repo_type,
    )


def gather_output_snapshot(output_dir: Path) -> dict[str, Any]:
    snapshot: dict[str, Any] = {"exists": output_dir.exists(), "files": []}
    if not output_dir.exists():
        return snapshot
    for path in sorted(output_dir.rglob("*")):
        if path.is_file():
            snapshot["files"].append(str(path.relative_to(output_dir)))
    return snapshot


def main() -> None:
    family = require_env("MODEL_FAMILY", required=True).strip().lower()
    if family not in FAMILY_DEFAULTS:
        raise SystemExit(f"MODEL_FAMILY must be one of {sorted(FAMILY_DEFAULTS)}, got {family!r}")
    defaults = FAMILY_DEFAULTS[family]

    token = require_env("HF_TOKEN", required=True)
    dataset_id = require_env("HF_DATASET_ID", default="JosueG/adja-tts-orpheus")
    experiment_id = require_env("EXPERIMENT_ID", required=True)
    output_repo_id = require_env("OUTPUT_REPO_ID", required=True)
    results_repo_id = require_env("RESULTS_REPO_ID", default="JosueG/adja-asr-results")
    upstream_ref = require_env("UPSTREAM_REF", default="main")
    model_card = require_env("MODEL_CARD", default=defaults.model_card)
    tokenizer_ref = require_env("TOKENIZER_REF", default=defaults.tokenizer_ref)
    smoke_run = env_flag("SMOKE_RUN", default=False)
    seed = int(require_env("SEED", default="42"))
    workspace_dir = Path(require_env("WORKSPACE_DIR", default=f"/tmp/{experiment_id}")).resolve()
    use_torchrun = env_flag("USE_TORCHRUN", default=False)
    use_fsdp = env_flag("USE_FSDP", default=False)
    nproc_per_node = int(require_env("NPROC_PER_NODE", default="1"))
    learning_rate = float(require_env("LEARNING_RATE", default=str(defaults.lr)))
    freeze_encoder_for_n_steps = int(require_env("FREEZE_ENCODER_FOR_N_STEPS", default="0"))

    if smoke_run:
        max_train_samples = int(require_env("MAX_TRAIN_SAMPLES", default="64"))
        max_dev_samples = int(require_env("MAX_DEV_SAMPLES", default="8"))
        max_test_samples = int(require_env("MAX_TEST_SAMPLES", default="8"))
        num_steps = int(require_env("NUM_STEPS", default="4"))
        validate_every_n_steps = int(require_env("VALIDATE_EVERY_N_STEPS", default="1"))
        publish_metrics_every_n_steps = int(require_env("PUBLISH_METRICS_EVERY_N_STEPS", default="1"))
        checkpoint_every_n_steps = int(require_env("CHECKPOINT_EVERY_N_STEPS", default="2"))
        max_audio_len = int(require_env("MAX_AUDIO_LEN", default=str(min(defaults.max_audio_len, 8 * 16_000))))
        max_num_elements = int(require_env("MAX_NUM_ELEMENTS", default=str(min(defaults.max_num_elements, 8 * 16_000))))
        grad_accumulation = int(require_env("GRAD_ACCUMULATION", default=str(defaults.grad_accumulation)))
    else:
        max_train_samples = int(require_env("MAX_TRAIN_SAMPLES", default="0"))
        max_dev_samples = int(require_env("MAX_DEV_SAMPLES", default="0"))
        max_test_samples = int(require_env("MAX_TEST_SAMPLES", default="0"))
        num_steps = int(require_env("NUM_STEPS", default="400"))
        validate_every_n_steps = int(require_env("VALIDATE_EVERY_N_STEPS", default="25"))
        publish_metrics_every_n_steps = int(require_env("PUBLISH_METRICS_EVERY_N_STEPS", default="10"))
        checkpoint_every_n_steps = int(require_env("CHECKPOINT_EVERY_N_STEPS", default="50"))
        max_audio_len = int(require_env("MAX_AUDIO_LEN", default=str(defaults.max_audio_len)))
        max_num_elements = int(require_env("MAX_NUM_ELEMENTS", default=str(defaults.max_num_elements)))
        grad_accumulation = int(require_env("GRAD_ACCUMULATION", default=str(defaults.grad_accumulation)))

    workspace_dir.mkdir(parents=True, exist_ok=True)
    log_path = workspace_dir / "job.log"
    log_handle, original_stdout, original_stderr = setup_tee_logging(log_path)

    from huggingface_hub import HfApi

    api = HfApi(token=token)

    try:
        print(f"Experiment ID: {experiment_id}")
        print(f"Family: {family}")
        print(f"Model card: {model_card}")
        print(f"Smoke run: {smoke_run}")
        print(f"Workspace: {workspace_dir}")

        import torch

        print(f"Torch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        for index in range(torch.cuda.device_count()):
            print(f"  GPU[{index}]: {torch.cuda.get_device_name(index)}")

        upstream_dir = ensure_upstream_repo(workspace_dir / "upstream" / "omnilingual-asr", upstream_ref)
        ensure_editable_install(upstream_dir)

        dataset_summary = materialize_manifest_dataset(
            dataset_id=dataset_id,
            token=token,
            workspace_dir=workspace_dir,
            seed=seed,
            max_train_samples=max_train_samples,
            max_dev_samples=max_dev_samples,
            max_test_samples=max_test_samples,
        )
        manifest_dir = Path(dataset_summary["manifest_dir"])

        dataset_name = f"{experiment_id.lower()}_dataset"
        dataset_card_path = write_dataset_card(
            upstream_dir=upstream_dir,
            dataset_name=dataset_name,
            manifest_dir=manifest_dir,
            tokenizer_ref=tokenizer_ref,
        )

        config_text = render_train_config(
            family=family,
            model_card=model_card,
            dataset_name=dataset_name,
            tokenizer_ref=tokenizer_ref,
            learning_rate=learning_rate,
            freeze_encoder_for_n_steps=freeze_encoder_for_n_steps,
            max_audio_len=max_audio_len,
            max_num_elements=max_num_elements,
            grad_accumulation=grad_accumulation,
            num_steps=num_steps,
            validate_every_n_steps=validate_every_n_steps,
            publish_metrics_every_n_steps=publish_metrics_every_n_steps,
            checkpoint_every_n_steps=checkpoint_every_n_steps,
            seed=seed,
            use_fsdp=use_fsdp,
        )
        config_path = workspace_dir / "train_config.yaml"
        write_text(config_path, config_text)
        print(f"Wrote config: {config_path}")
        print(config_text)

        output_dir = workspace_dir / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)

        if use_torchrun:
            command = [
                "torchrun",
                "--standalone",
                "--nproc-per-node",
                str(nproc_per_node),
                "-m",
                "workflows.recipes.wav2vec2.asr",
                str(output_dir),
                "--config-file",
                str(config_path),
            ]
        else:
            command = [
                sys.executable,
                "-m",
                "workflows.recipes.wav2vec2.asr",
                str(output_dir),
                "--config-file",
                str(config_path),
            ]

        train_started_at = time.time()
        run_command(command, cwd=upstream_dir)
        train_elapsed_sec = round(time.time() - train_started_at, 1)
        print(f"Training finished in {train_elapsed_sec}s")

        output_snapshot = gather_output_snapshot(output_dir)
        payload = {
            "experiment": experiment_id,
            "family": family,
            "model_card": model_card,
            "tokenizer_ref": tokenizer_ref,
            "dataset_id": dataset_id,
            "dataset_summary": dataset_summary,
            "smoke_run": smoke_run,
            "config_path": str(config_path),
            "dataset_card_path": str(dataset_card_path),
            "train_elapsed_sec": train_elapsed_sec,
            "output_snapshot": output_snapshot,
        }
        payload_path = workspace_dir / "summary.json"
        write_json(payload_path, payload)

        api.create_repo(results_repo_id, private=True, exist_ok=True, repo_type="dataset")
        api.create_repo(output_repo_id, private=True, exist_ok=True, repo_type="model")
        upload_file_if_exists(api, results_repo_id, log_path, f"{experiment_id}/job.log", repo_type="dataset")
        upload_file_if_exists(api, results_repo_id, payload_path, f"{experiment_id}/summary.json", repo_type="dataset")
        upload_file_if_exists(api, results_repo_id, config_path, f"{experiment_id}/train_config.yaml", repo_type="dataset")
        upload_file_if_exists(api, results_repo_id, workspace_dir / "manifest_dataset" / "summary.json", f"{experiment_id}/dataset_summary.json", repo_type="dataset")

        model_card_text = "\n".join(
            [
                f"# {experiment_id}",
                "",
                f"- Model family: `{family}`",
                f"- Base checkpoint: `{model_card}`",
                f"- Dataset: `{dataset_id}`",
                f"- Smoke run: `{smoke_run}`",
                f"- Training wall time (sec): `{train_elapsed_sec}`",
                "",
                "Artifacts were produced by `scripts/hf_jobs/omni_asr_finetune.py`.",
                "",
            ]
        )
        model_card_path = workspace_dir / "README.md"
        write_text(model_card_path, model_card_text)
        upload_file_if_exists(api, output_repo_id, model_card_path, "README.md", repo_type="model")
        upload_folder_if_exists(api, output_repo_id, output_dir, repo_type="model")

        print("Uploaded results and output snapshots to Hugging Face.")
        print(json.dumps(payload, indent=2))
    finally:
        restore_tee_logging(log_handle, original_stdout, original_stderr)


if __name__ == "__main__":
    main()
