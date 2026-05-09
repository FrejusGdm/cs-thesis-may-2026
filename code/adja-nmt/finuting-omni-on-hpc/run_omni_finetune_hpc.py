#!/usr/bin/env python3
"""
Run OmniASR fine-tuning on HPC with local manifest data.

Designed for Dartmouth Discovery / SLURM workflows:
- uses absolute paths (no dirname("$0") assumptions)
- supports dry-run checks before burning GPU queue time
- works with manifests already prepared on HPC storage
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


def run_cmd(cmd: list[str], cwd: Path | None = None, dry_run: bool = False) -> None:
    print(f"+ {' '.join(cmd)}")
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OmniASR fine-tune runner for HPC.")
    parser.add_argument("--omni-repo-dir", required=True, type=Path)
    parser.add_argument("--manifest-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--ft-mode", choices=["ctc", "llm"], required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--tokenizer-name", default="omniASR_tokenizer_written_v2")
    parser.add_argument("--language-code", default="ajg_Latn")
    parser.add_argument("--num-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-num-elements", type=int, default=100000)
    parser.add_argument("--grad-accum", type=int, default=32)
    parser.add_argument("--data-parallelism", default="fsdp")
    parser.add_argument("--max-audio-sec", type=float, default=8.0)
    parser.add_argument("--min-audio-len", type=int, default=1600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--valid-splits", default="dev,test")
    parser.add_argument("--validate-after-n-steps", type=int, default=None)
    parser.add_argument("--validate-every-n-steps", type=int, default=None)
    parser.add_argument("--checkpoint-every-n-steps", type=int, default=50)
    parser.add_argument("--publish-metrics-every-n-steps", type=int, default=10)
    parser.add_argument(
        "--save-model-only",
        choices=["false", "true", "all_but_last"],
        default="false",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def parse_save_model_only(raw: str) -> bool | str:
    if raw == "true":
        return True
    if raw == "all_but_last":
        return "all_but_last"
    return False


def ensure_paths(args: argparse.Namespace) -> None:
    required_files = [
        args.manifest_dir / "train.tsv",
        args.manifest_dir / "train.wrd",
        args.manifest_dir / "dev.tsv",
        args.manifest_dir / "dev.wrd",
        args.manifest_dir / "test.tsv",
        args.manifest_dir / "test.wrd",
    ]
    if args.ft_mode == "llm":
        required_files.extend(
            [
                args.manifest_dir / "train.lang",
                args.manifest_dir / "dev.lang",
                args.manifest_dir / "test.lang",
            ]
        )

    missing = [str(p) for p in required_files if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing manifest files:\n" + "\n".join(f"- {m}" for m in missing)
        )

    if not (args.omni_repo_dir / "workflows" / "recipes" / "wav2vec2" / "asr").exists():
        raise FileNotFoundError(
            f"Omni repo not found or invalid: {args.omni_repo_dir}. "
            "Expected workflows/recipes/wav2vec2/asr."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)


def patch_manifest_storage_for_lang(repo_dir: Path) -> None:
    path = (
        repo_dir
        / "src"
        / "omnilingual_asr"
        / "datasets"
        / "storage"
        / "manifest_storage.py"
    )
    content = path.read_text(encoding="utf-8")
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

    anchor = """    def read_wrd_file(manifest_dir: Path, split: str) -> DataPipelineBuilder:
        \"\"\"Read WRD file containing text transcriptions.\"\"\"
        wrd_file = manifest_dir.joinpath(f\"{split}.wrd\")

        return read_text(wrd_file, key=\"text\", rtrim=True, memory_map=True)
"""
    insert = """    @staticmethod
    def read_lang_file(manifest_dir: Path, split: str) -> DataPipelineBuilder:
        \"\"\"Read language tags file containing one language code per row.\"\"\"
        lang_file = manifest_dir.joinpath(f\"{split}.lang\")

        return read_text(lang_file, key=\"lang\", rtrim=True, memory_map=True)

"""
    if anchor not in content:
        raise RuntimeError("Failed to locate read_wrd_file anchor for lang helper insertion.")
    content = content.replace(anchor, anchor + "\n" + insert)
    path.write_text(content, encoding="utf-8")


def build_recipe_config(args: argparse.Namespace) -> dict[str, Any]:
    validate_after = (
        args.validate_after_n_steps
        if args.validate_after_n_steps is not None
        else args.num_steps
    )
    validate_every = (
        args.validate_every_n_steps
        if args.validate_every_n_steps is not None
        else args.num_steps
    )
    valid_splits = [s.strip() for s in args.valid_splits.split(",") if s.strip()]
    valid_splits_csv = ",".join(valid_splits) if valid_splits else None

    dataset_card_name = "omni_manifest_hpc_local"

    recipe_cfg: dict[str, Any] = {
        "model": {"name": args.model_name},
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
                "batch_size": args.batch_size,
                "min_audio_len": args.min_audio_len,
                "max_audio_len": int(args.max_audio_sec * 16000),
                "max_num_elements": args.max_num_elements,
                "batch_shuffle_window": 16,
                "example_shuffle_window": 16,
                "normalize_audio": True,
                "max_num_batches": None,
            },
        },
        "tokenizer": {"name": args.tokenizer_name},
        "trainer": {
            "data_parallelism": args.data_parallelism,
            "fsdp": {
                "granularity": "stack",
                "version": "v1",
                "fp32_reduce": False,
            },
            "freeze_encoder_for_n_steps": 0,
            "mixed_precision": {"dtype": "torch.bfloat16"},
            "grad_accumulation": {"num_batches": args.grad_accum},
        },
        "optimizer": {
            "config": {"lr": 1e-5 if args.ft_mode == "ctc" else 5e-6}
        },
        "regime": {
            "num_steps": args.num_steps,
            "validate_after_n_steps": validate_after,
            "validate_every_n_steps": validate_every,
            "checkpoint_every_n_steps": args.checkpoint_every_n_steps,
            "publish_metrics_every_n_steps": args.publish_metrics_every_n_steps,
            "save_model_only": parse_save_model_only(args.save_model_only),
            "score_metric": "wer",
        },
        "common": {
            "seed": args.seed,
            "metric_recorders": {
                "tensorboard": {"enabled": False},
                "wandb": {"enabled": False},
            },
        },
    }

    if args.ft_mode == "llm":
        recipe_cfg["model"]["model_arch"] = {
            "model_type": "LLM_ASR_LID",
            "language_column_name": "lang",
        }
        recipe_cfg["dataset"]["manifest_storage_config"]["read_lang"] = True

    return recipe_cfg


def write_dataset_card(repo_dir: Path, manifest_dir: Path, tokenizer_name: str) -> Path:
    card_name = "omni_manifest_hpc_local"
    card_path = (
        repo_dir
        / "src"
        / "omnilingual_asr"
        / "cards"
        / "datasets"
        / f"{card_name}.yaml"
    )
    card_path.write_text(
        "\n".join(
            [
                f"name: {card_name}",
                "dataset_family: manifest_asr_dataset",
                "dataset_config:",
                f"  data: {manifest_dir}",
                f"tokenizer_ref: {tokenizer_name}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return card_path


def print_run_summary(args: argparse.Namespace, config_path: Path) -> None:
    summary = {
        "ft_mode": args.ft_mode,
        "model_name": args.model_name,
        "language_code": args.language_code,
        "manifest_dir": str(args.manifest_dir),
        "output_dir": str(args.output_dir),
        "config_path": str(config_path),
        "dry_run": args.dry_run,
    }
    print(json.dumps(summary, indent=2))


def main() -> None:
    args = parse_args()
    ensure_paths(args)

    if args.ft_mode == "llm":
        patch_manifest_storage_for_lang(args.omni_repo_dir)

    _dataset_card = write_dataset_card(
        args.omni_repo_dir, args.manifest_dir, args.tokenizer_name
    )
    recipe_cfg = build_recipe_config(args)

    config_path = args.output_dir / f"{args.ft_mode}_finetune_hpc.yaml"
    config_path.write_text(yaml.safe_dump(recipe_cfg, sort_keys=False), encoding="utf-8")
    print(f"Config written to: {config_path}")
    print(config_path.read_text(encoding="utf-8"))

    print_run_summary(args, config_path)
    if args.dry_run:
        print("Dry-run mode: skipping fairseq2 recipe execution.")
        return

    run_cmd(
        [
            sys.executable,
            "-m",
            "workflows.recipes.wav2vec2.asr",
            str(args.output_dir),
            "--config-file",
            str(config_path),
        ],
        cwd=args.omni_repo_dir,
        dry_run=False,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FATAL: {exc}")
        raise
