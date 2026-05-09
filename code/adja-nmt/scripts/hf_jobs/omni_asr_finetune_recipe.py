# /// script
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
#   "pyyaml",
# ]
# ///
"""
Run OmniASR fine-tuning with the official fairseq2 recipe on Adja data.

This script is intended for Hugging Face Jobs (`hf jobs uv run`) and supports
both recipe variants through one code path:
  - FT_MODE=ctc
  - FT_MODE=llm
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import unicodedata
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import yaml
from datasets import load_dataset
from huggingface_hub import HfApi

sys.stdout.reconfigure(line_buffering=True)


def run_cmd(cmd: list[str], cwd: Path | None = None) -> None:
    print(f"+ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def dump_torchrun_failure_logs(log_dir: Path) -> None:
    if not log_dir.exists():
        print(f"torchrun log dir not found: {log_dir}")
        return

    error_files = sorted(log_dir.rglob("error.json"))
    stderr_files = sorted(log_dir.rglob("stderr.log"))

    if error_files:
        print("torchrun error.json files:")
        for path in error_files:
            try:
                content = path.read_text(encoding="utf-8", errors="replace").strip()
            except OSError as exc:
                print(f"- {path}: failed to read ({exc})")
                continue
            if content:
                print(f"--- {path} ---")
                print(content)

    if stderr_files:
        print("torchrun stderr tails:")
        for path in stderr_files:
            try:
                lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError as exc:
                print(f"- {path}: failed to read ({exc})")
                continue
            if lines:
                print(f"--- {path} (last 120 lines) ---")
                print("\n".join(lines[-120:]))


def detect_visible_gpu_count() -> int:
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if cuda_visible and cuda_visible.lower() not in {"void", "none"}:
        device_tokens = [token.strip() for token in cuda_visible.split(",") if token.strip()]
        if device_tokens:
            return len(device_tokens)

    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return 1

    try:
        output = subprocess.check_output(
            [nvidia_smi, "--query-gpu=index", "--format=csv,noheader"],
            text=True,
        )
    except Exception as exc:  # pragma: no cover - best effort runtime probe
        print(f"GPU count probe failed: {exc}")
        return 1

    gpu_lines = [line for line in output.splitlines() if line.strip()]
    return max(len(gpu_lines), 1)


def ensure_system_libsndfile() -> None:
    apt_get = shutil.which("apt-get")
    if not apt_get:
        print("apt-get not found; cannot install libsndfile1 automatically.")
        return
    run_cmd([apt_get, "update"])
    run_cmd([apt_get, "install", "-y", "libsndfile1"])


def patch_manifest_storage_for_lang(repo_dir: Path) -> None:
    manifest_storage_path = (
        repo_dir
        / "src"
        / "omnilingual_asr"
        / "datasets"
        / "storage"
        / "manifest_storage.py"
    )
    content = manifest_storage_path.read_text(encoding="utf-8")
    if "def read_lang_file(" in content:
        return

    old_block = """        if self.config.read_text:
            tsv_pipeline = ManifestStorage.read_tsv_file(
                manifest_dir=self._manifest_dir, split=split
            ).and_return()
            wrd_pipeline = ManifestStorage.read_wrd_file(
                manifest_dir=self._manifest_dir, split=split
            ).and_return()

            builder = DataPipeline.zip([tsv_pipeline, wrd_pipeline], flatten=True)
        else:
            builder = ManifestStorage.read_tsv_file(
                manifest_dir=self._manifest_dir, split=split
            )
"""
    new_block = """        if self.config.read_text:
            tsv_pipeline = ManifestStorage.read_tsv_file(
                manifest_dir=self._manifest_dir, split=split
            ).and_return()
            wrd_pipeline = ManifestStorage.read_wrd_file(
                manifest_dir=self._manifest_dir, split=split
            ).and_return()

            lang_file = self._manifest_dir.joinpath(f\"{split}.lang\")
            if lang_file.exists():
                lang_pipeline = ManifestStorage.read_lang_file(
                    manifest_dir=self._manifest_dir, split=split
                ).and_return()
                builder = DataPipeline.zip(
                    [tsv_pipeline, wrd_pipeline, lang_pipeline], flatten=True
                )
            else:
                builder = DataPipeline.zip([tsv_pipeline, wrd_pipeline], flatten=True)
        else:
            builder = ManifestStorage.read_tsv_file(
                manifest_dir=self._manifest_dir, split=split
            )
"""
    if old_block not in content:
        raise RuntimeError("Failed to patch manifest storage block for language support.")
    content = content.replace(old_block, new_block)

    insertion_anchor = """    def read_wrd_file(manifest_dir: Path, split: str) -> DataPipelineBuilder:
        \"\"\"Read WRD file containing text transcriptions.\"\"\"
        wrd_file = manifest_dir.joinpath(f\"{split}.wrd\")

        return read_text(wrd_file, key=\"text\", rtrim=True, memory_map=True)
"""
    insertion_text = """    @staticmethod
    def read_lang_file(manifest_dir: Path, split: str) -> DataPipelineBuilder:
        \"\"\"Read language tags file containing one language code per row.\"\"\"
        lang_file = manifest_dir.joinpath(f\"{split}.lang\")

        return read_text(lang_file, key=\"lang\", rtrim=True, memory_map=True)

"""
    if insertion_anchor not in content:
        raise RuntimeError("Failed to locate read_wrd_file anchor for lang helper insertion.")
    content = content.replace(insertion_anchor, insertion_anchor + "\n" + insertion_text)
    manifest_storage_path.write_text(content, encoding="utf-8")


def install_omnilingual_editable(repo_dir: Path) -> None:
    uv_bin = shutil.which("uv")
    if uv_bin:
        run_cmd(
            [
                uv_bin,
                "pip",
                "install",
                "--python",
                sys.executable,
                "--no-deps",
                "-e",
                ".",
            ],
            cwd=repo_dir,
        )
        return

    try:
        run_cmd([sys.executable, "-m", "ensurepip", "--upgrade"], cwd=repo_dir)
    except subprocess.CalledProcessError:
        # Some managed images disable ensurepip; fall back to pip and let it error clearly.
        pass

    run_cmd([sys.executable, "-m", "pip", "install", "--no-deps", "-e", "."], cwd=repo_dir)


def normalize_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text.strip()).split())


def edit_distance(ref: list[str], hyp: list[str]) -> int:
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j - 1],
                    dp[i][j - 1],
                    dp[i - 1][j],
                )
    return dp[n][m]


def cer(refs: list[str], hyps: list[str]) -> float:
    edits = sum(edit_distance(list(r), list(h)) for r, h in zip(refs, hyps))
    total = sum(len(r) for r in refs)
    return round((edits / max(total, 1)) * 100.0, 2)


def wer(refs: list[str], hyps: list[str]) -> float:
    edits = sum(edit_distance(r.split(), h.split()) for r, h in zip(refs, hyps))
    total = sum(len(r.split()) for r in refs)
    return round((edits / max(total, 1)) * 100.0, 2)


def compute_eval_metrics_from_transcriptions(
    ref_path: Path,
    hyp_path: Path,
    split_sizes: dict[str, int],
    split_order: list[str],
) -> dict[str, object]:
    refs = [line.rstrip("\n") for line in ref_path.read_text(encoding="utf-8").splitlines()]
    hyps = [line.rstrip("\n") for line in hyp_path.read_text(encoding="utf-8").splitlines()]
    pair_count = min(len(refs), len(hyps))
    refs = refs[:pair_count]
    hyps = hyps[:pair_count]

    metrics: dict[str, object] = {}
    cursor = 0
    for split in split_order:
        expected = split_sizes.get(split, 0)
        end = cursor + expected
        split_refs = refs[cursor:end]
        split_hyps = hyps[cursor:end]
        metrics[split] = {
            "num_samples": len(split_refs),
            "wer": wer(split_refs, split_hyps),
            "cer": cer(split_refs, split_hyps),
        }
        cursor = end

    metrics["_meta"] = {
        "num_ref_lines": len(refs),
        "num_hyp_lines": len(hyps),
        "expected_total": sum(split_sizes.get(split, 0) for split in split_order),
        "used_pairs": pair_count,
    }
    return metrics


def locate_transcription_files(train_output_dir: Path) -> tuple[Path | None, Path | None]:
    """
    Locate rank-0 transcription files produced by fairseq2 validation.

    Depending on fairseq2 workspace layout, files can be written either directly
    under `train_output_dir/transcriptions` or under nested `ws_*/transcriptions`.
    """
    direct_ref = train_output_dir / "transcriptions" / "rank_0.ref.txt"
    direct_hyp = train_output_dir / "transcriptions" / "rank_0.hyp.txt"
    if direct_ref.exists() and direct_hyp.exists():
        return direct_ref, direct_hyp

    for ref_path in sorted(train_output_dir.rglob("rank_0.ref.txt")):
        hyp_path = ref_path.with_name("rank_0.hyp.txt")
        if hyp_path.exists():
            return ref_path, hyp_path

    return None, None


def parse_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default))
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got: {raw}") from exc


def parse_float_env(name: str, default: float) -> float:
    raw = os.environ.get(name, str(default))
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float, got: {raw}") from exc


def parse_bool_env(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def main() -> None:
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN is required.")

    ft_mode = os.environ.get("FT_MODE", "ctc").strip().lower()
    if ft_mode not in {"ctc", "llm"}:
        raise ValueError(f"FT_MODE must be one of [ctc, llm], got: {ft_mode}")

    seed = parse_int_env("SEED", 42)
    dataset_id = os.environ.get("DATASET_ID", "JosueG/adja-tts-orpheus")
    max_train_samples = parse_int_env("MAX_TRAIN_SAMPLES", 320)
    max_dev_samples = parse_int_env("MAX_DEV_SAMPLES", 80)
    max_test_samples = parse_int_env("MAX_TEST_SAMPLES", max_dev_samples)
    max_audio_sec = parse_float_env("MAX_AUDIO_SEC", 30.0)
    num_steps = parse_int_env("NUM_STEPS", 120)
    min_audio_len = parse_int_env("MIN_AUDIO_LEN", 32000)
    prep_only = parse_bool_env("PREP_ONLY", False)
    valid_splits_raw = os.environ.get("VALID_SPLITS", "").strip()
    valid_splits = [split.strip() for split in valid_splits_raw.split(",") if split.strip()]
    valid_splits_csv = ",".join(valid_splits) if valid_splits else None
    validate_after_n_steps = (
        parse_int_env("VALIDATE_AFTER_N_STEPS", num_steps) if valid_splits else 0
    )
    validate_every_n_steps = (
        parse_int_env("VALIDATE_EVERY_N_STEPS", num_steps) if valid_splits else 20
    )
    checkpoint_every_n_steps = parse_int_env("CHECKPOINT_EVERY_N_STEPS", 50)
    publish_metrics_every_n_steps = parse_int_env("PUBLISH_METRICS_EVERY_N_STEPS", 10)
    save_model_only_raw = os.environ.get("SAVE_MODEL_ONLY", "").strip().lower()
    save_model_only: bool | str = False
    if save_model_only_raw in {"1", "true", "yes", "y", "on"}:
        save_model_only = True
    elif save_model_only_raw in {"all_but_last"}:
        # Supported in newer fairseq2; older versions may ignore/raise.
        save_model_only = "all_but_last"
    output_root = Path(os.environ.get("OUTPUT_ROOT", "/tmp/omni_finetune_outputs"))
    results_repo = os.environ.get("RESULTS_REPO", "JosueG/adja-asr-results")
    experiment_prefix = os.environ.get("EXPERIMENT_PREFIX", "Omni_FT")
    repo_url = os.environ.get(
        "OMNI_REPO_URL", "https://github.com/facebookresearch/omnilingual-asr.git"
    )
    clone_depth = parse_int_env("CLONE_DEPTH", 1)

    mode_defaults = {
        "ctc": {
            "model_name": os.environ.get("MODEL_NAME", "omniASR_CTC_300M_v2"),
            "tokenizer_name": "omniASR_tokenizer_written_v2",
            "batch_size": parse_int_env("BATCH_SIZE", 2),
            "max_num_elements": parse_int_env("MAX_NUM_ELEMENTS", 320000),
            "grad_accum": parse_int_env("GRAD_ACCUM", 4),
            "data_parallelism": os.environ.get("DATA_PARALLELISM", "ddp"),
        },
        "llm": {
            "model_name": os.environ.get("MODEL_NAME", "omniASR_LLM_300M_v2"),
            "tokenizer_name": "omniASR_tokenizer_written_v2",
            "batch_size": parse_int_env("BATCH_SIZE", 1),
            "max_num_elements": parse_int_env("MAX_NUM_ELEMENTS", 160000),
            "grad_accum": parse_int_env("GRAD_ACCUM", 8),
            "data_parallelism": os.environ.get("DATA_PARALLELISM", "fsdp"),
            "language_code": os.environ.get("LANGUAGE_CODE", "ajg_Latn"),
        },
    }
    mode_config = mode_defaults[ft_mode]
    effective_max_audio_sec = (
        parse_float_env("LLM_MAX_AUDIO_SEC", min(max_audio_sec, 8.0))
        if ft_mode == "llm"
        else max_audio_sec
    )

    ts = int(time.time())
    run_name = f"{experiment_prefix}_{ft_mode}_{ts}"
    work_dir = output_root / run_name
    repo_dir = work_dir / "omnilingual-asr"
    manifests_dir = work_dir / "manifests"
    audio_root = work_dir / "audio"
    train_output_dir = work_dir / "recipe_output"
    config_path = work_dir / f"{ft_mode}_finetune.yaml"
    run_summary_path = work_dir / "run_summary.json"

    for path in (manifests_dir, audio_root, train_output_dir):
        path.mkdir(parents=True, exist_ok=True)

    print(
        json.dumps(
            {
                "mode": ft_mode,
                "dataset_id": dataset_id,
                "num_steps": num_steps,
                "prep_only": prep_only,
                "max_train_samples": max_train_samples,
                "max_dev_samples": max_dev_samples,
                "max_test_samples": max_test_samples,
                "max_audio_sec": effective_max_audio_sec,
                "min_audio_len": min_audio_len,
                "model_name": mode_config["model_name"],
                "valid_splits": valid_splits,
            },
            indent=2,
        )
    )

    print("Loading dataset from Hugging Face...")
    ds = load_dataset(dataset_id, token=hf_token, split="train")
    split1 = ds.train_test_split(test_size=0.1, seed=seed)
    split2 = split1["train"].train_test_split(test_size=0.1 / 0.9, seed=seed)

    train_data = split2["train"]
    dev_data = split2["test"]
    test_data = split1["test"]

    if max_train_samples > 0:
        train_data = train_data.select(range(min(max_train_samples, len(train_data))))
    if max_dev_samples > 0:
        dev_data = dev_data.select(range(min(max_dev_samples, len(dev_data))))
    if max_test_samples > 0:
        test_data = test_data.select(range(min(max_test_samples, len(test_data))))

    print(
        f"Prepared splits: train={len(train_data)} dev={len(dev_data)} test={len(test_data)}"
    )

    def write_manifest(split_name: str, split_data) -> dict[str, int]:
        split_audio_root = audio_root / split_name
        split_audio_root.mkdir(parents=True, exist_ok=True)

        tsv_path = manifests_dir / f"{split_name}.tsv"
        wrd_path = manifests_dir / f"{split_name}.wrd"

        total_samples = 0
        kept_samples = 0
        dropped_short_audio = 0
        dropped_empty_text = 0
        lang_path = manifests_dir / f"{split_name}.lang"
        with (
            tsv_path.open("w", encoding="utf-8") as tsv_fp,
            wrd_path.open("w", encoding="utf-8") as wrd_fp,
            lang_path.open("w", encoding="utf-8") as lang_fp,
        ):
            tsv_fp.write(f"{audio_root}\n")

            for idx, sample in enumerate(split_data):
                total_samples += 1
                audio = np.asarray(sample["audio"]["array"], dtype=np.float32)
                sr = int(sample["audio"]["sampling_rate"])
                text = normalize_text(sample["text"])

                if sr != 16000:
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

                if effective_max_audio_sec > 0:
                    max_samples = int(effective_max_audio_sec * 16000)
                    if len(audio) > max_samples:
                        audio = audio[:max_samples]

                if len(audio) < min_audio_len:
                    dropped_short_audio += 1
                    continue
                if not text:
                    dropped_empty_text += 1
                    continue

                rel_audio = Path(split_name) / f"{idx:06d}.wav"
                abs_audio = audio_root / rel_audio
                sf.write(abs_audio, audio, 16000)

                tsv_fp.write(f"{rel_audio}\t{len(audio)}\n")
                wrd_fp.write(f"{text}\n")
                lang_fp.write(f"{mode_config.get('language_code', 'ajg_Latn')}\n")
                kept_samples += 1

        return {
            "total": total_samples,
            "kept": kept_samples,
            "dropped_short_audio": dropped_short_audio,
            "dropped_empty_text": dropped_empty_text,
            "min_audio_len": min_audio_len,
        }

    train_counts = write_manifest("train", train_data)
    dev_counts = write_manifest("dev", dev_data)
    test_counts = write_manifest("test", test_data)
    split_counts = {
        "train": train_counts,
        "dev": dev_counts,
        "test": test_counts,
    }
    print(f"Manifest stats: {split_counts}")
    if (
        train_counts["kept"] == 0
        or dev_counts["kept"] == 0
        or test_counts["kept"] == 0
    ):
        raise RuntimeError("Manifest generation produced empty split(s).")
    for split in valid_splits:
        if split not in split_counts:
            raise ValueError(
                f"VALID_SPLITS contains unsupported split '{split}'. Supported: train,dev,test."
            )
        if split_counts[split]["kept"] == 0:
            raise RuntimeError(
                f"VALID_SPLITS includes '{split}' but that split has zero examples after filtering."
            )

    print("Cloning omnilingual-asr...")
    run_cmd(
        [
            "git",
            "clone",
            "--depth",
            str(clone_depth),
            repo_url,
            str(repo_dir),
        ]
    )

    print("Installing omnilingual-asr package in editable mode (no deps)...")
    install_omnilingual_editable(repo_dir)
    patch_manifest_storage_for_lang(repo_dir)
    print("Ensuring system dependency libsndfile is present...")
    ensure_system_libsndfile()

    dataset_root_for_card = manifests_dir

    dataset_card_name = f"omni_manifest_adja_{ft_mode}_{ts}"
    dataset_card_path = (
        repo_dir
        / "src"
        / "omnilingual_asr"
        / "cards"
        / "datasets"
        / f"{dataset_card_name}.yaml"
    )
    dataset_card_path.write_text(
        "\n".join(
            [
                f"name: {dataset_card_name}",
                "dataset_family: manifest_asr_dataset",
                "dataset_config:",
                f"  data: {dataset_root_for_card}",
                f"tokenizer_ref: {mode_config['tokenizer_name']}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    recipe_cfg = {
        "model": {"name": mode_config["model_name"]},
        "dataset": {
            "name": dataset_card_name,
            "train_split": "train",
            "valid_split": valid_splits_csv,
            "storage_mode": "MANIFEST",
            "task_mode": "ASR",
            "manifest_storage_config": {
                "read_text": True,
                "sync_mode": "UNTIL_LAST",
            },
            "asr_task_config": {
                "batching_strategy": "STATIC",
                "batch_size": mode_config["batch_size"],
                "min_audio_len": min_audio_len,
                "max_audio_len": int(effective_max_audio_sec * 16000)
                if effective_max_audio_sec > 0
                else 480000,
                "max_num_elements": mode_config["max_num_elements"],
                "batch_shuffle_window": 16,
                "example_shuffle_window": 16,
                "normalize_audio": True,
                "max_num_batches": None,
            },
        },
        "tokenizer": {"name": mode_config["tokenizer_name"]},
        "trainer": {
            "data_parallelism": mode_config["data_parallelism"],
            "fsdp": {
                "granularity": "stack",
                "version": "v1",
                "fp32_reduce": False,
            },
            "freeze_encoder_for_n_steps": 0,
            "mixed_precision": {"dtype": "torch.bfloat16"},
            "grad_accumulation": {"num_batches": mode_config["grad_accum"]},
        },
        "optimizer": {"config": {"lr": 1e-5 if ft_mode == "ctc" else 5e-6}},
        "regime": {
            "num_steps": num_steps,
            "validate_after_n_steps": validate_after_n_steps,
            "validate_every_n_steps": validate_every_n_steps,
            "checkpoint_every_n_steps": checkpoint_every_n_steps,
            "publish_metrics_every_n_steps": publish_metrics_every_n_steps,
            "save_model_only": save_model_only,
            "score_metric": "wer",
        },
        "common": {
            "seed": seed,
            "metric_recorders": {
                "tensorboard": {"enabled": False},
                "wandb": {"enabled": False},
            },
        },
    }

    config_path.write_text(yaml.safe_dump(recipe_cfg, sort_keys=False), encoding="utf-8")
    print(f"Config written to: {config_path}")
    print(config_path.read_text(encoding="utf-8"))

    if prep_only:
        print("PREP_ONLY=true -> skipping recipe training execution.")
    else:
        print("Ensuring system dependency libsndfile1 is installed...")
        ensure_system_libsndfile()
        print("Starting Omni recipe training...")
        gpu_count = detect_visible_gpu_count()
        print(f"Detected {gpu_count} visible GPU(s) for training.")
        torchrun_log_dir = train_output_dir / "torchrun_logs"
        if gpu_count > 1 and mode_config["data_parallelism"] in {"ddp", "fsdp"}:
            print(
                f"Launching distributed recipe with torch.distributed.run across {gpu_count} processes."
            )
            train_cmd = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                f"--nproc_per_node={gpu_count}",
                "--log-dir",
                str(torchrun_log_dir),
                "--redirects",
                "3",
                "--tee",
                "3",
                "--module",
                "workflows.recipes.wav2vec2.asr",
                str(train_output_dir),
                "--config-file",
                str(config_path),
            ]
        else:
            train_cmd = [
                sys.executable,
                "-m",
                "workflows.recipes.wav2vec2.asr",
                str(train_output_dir),
                "--config-file",
                str(config_path),
            ]
        print(f"+ {' '.join(train_cmd)}")
        train_result = subprocess.run(train_cmd, cwd=str(repo_dir), check=False)
        if train_result.returncode != 0:
            if gpu_count > 1 and mode_config["data_parallelism"] in {"ddp", "fsdp"}:
                dump_torchrun_failure_logs(torchrun_log_dir)
            raise subprocess.CalledProcessError(train_result.returncode, train_cmd)

    eval_metrics: dict[str, object] | None = None
    ref_path, hyp_path = locate_transcription_files(train_output_dir)
    if not prep_only and valid_splits and ref_path is not None and hyp_path is not None:
        eval_metrics = compute_eval_metrics_from_transcriptions(
            ref_path=ref_path,
            hyp_path=hyp_path,
            split_sizes={split: split_counts[split]["kept"] for split in valid_splits},
            split_order=valid_splits,
        )
        print("Final eval metrics (computed from validation transcriptions):")
        print(json.dumps(eval_metrics, indent=2))
    elif not prep_only and valid_splits:
        print("Validation transcriptions were not found; eval_metrics remains null.")

    summary = {
        "experiment": run_name,
        "ft_mode": ft_mode,
        "dataset_id": dataset_id,
        "model_name": mode_config["model_name"],
        "tokenizer_name": mode_config["tokenizer_name"],
        "num_steps": num_steps,
        "prep_only": prep_only,
        "train_kept": train_counts["kept"],
        "dev_kept": dev_counts["kept"],
        "test_kept": test_counts["kept"],
        "valid_splits": valid_splits,
        "eval_metrics": eval_metrics,
        "manifests_dir": str(manifests_dir),
        "config_path": str(config_path),
        "train_output_dir": str(train_output_dir),
    }
    run_summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

    print("Uploading run summary to results repo...")
    api = HfApi(token=hf_token)
    api.create_repo(results_repo, private=True, exist_ok=True)
    api.upload_file(
        path_or_fileobj=run_summary_path.read_bytes(),
        path_in_repo=f"{run_name}/run_summary.json",
        repo_id=results_repo,
        token=hf_token,
    )
    api.upload_file(
        path_or_fileobj=config_path.read_bytes(),
        path_in_repo=f"{run_name}/recipe_config.yaml",
        repo_id=results_repo,
        token=hf_token,
    )
    if eval_metrics is not None:
        eval_metrics_path = work_dir / "eval_metrics.json"
        eval_metrics_path.write_text(json.dumps(eval_metrics, indent=2), encoding="utf-8")
        api.upload_file(
            path_or_fileobj=eval_metrics_path.read_bytes(),
            path_in_repo=f"{run_name}/eval_metrics.json",
            repo_id=results_repo,
            token=hf_token,
        )
        api.upload_file(
            path_or_fileobj=ref_path.read_bytes(),
            path_in_repo=f"{run_name}/transcriptions_rank_0.ref.txt",
            repo_id=results_repo,
            token=hf_token,
        )
        api.upload_file(
            path_or_fileobj=hyp_path.read_bytes(),
            path_in_repo=f"{run_name}/transcriptions_rank_0.hyp.txt",
            repo_id=results_repo,
            token=hf_token,
        )
    print(f"Uploaded summary artifacts to {results_repo}/{run_name}/")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FATAL: {exc}")
        raise
