#!/usr/bin/env python3
from __future__ import annotations

"""
Submit Qwen3 Adja HF Jobs from the local repo workspace.

Prepared for: Josue Godeme
Requested by: Josue Godeme

This helper matches the repo's standard HF Jobs submission pattern:
- submit `scripts/hf_jobs/*.py` directly with `hf jobs uv run`
- rely on the uploaded local workspace context
- do not clone GitHub inside the HF container

Examples:
    python scripts/hf_jobs/submit_qwen3_jobs.py --modality both --smoke
    python scripts/hf_jobs/submit_qwen3_jobs.py --modality both --smoke --execute
    python scripts/hf_jobs/submit_qwen3_jobs.py --modality tts --pilot --execute
    python scripts/hf_jobs/submit_qwen3_jobs.py --modality asr --pilot --execute --detach
    python scripts/hf_jobs/submit_qwen3_jobs.py --modality tts --smoke --tts-model-id Qwen/Qwen3-TTS-12Hz-1.7B-Base --tts-output-repo-id JosueG/qwen3-adja-tts-1p7b --tts-flavor l40sx1 --execute --detach
"""

import argparse
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class JobSpec:
    name: str
    script_path: str
    flavor: str
    timeout: str
    env: dict[str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit Qwen3 Adja HF Jobs")
    parser.add_argument("--modality", choices=["asr", "tts", "both"], default="both")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--smoke", action="store_true", help="Submit smoke runs (default)")
    mode_group.add_argument("--pilot", action="store_true", help="Submit pilot runs")
    parser.add_argument("--namespace", default=os.environ.get("HF_NAMESPACE", ""), help="HF namespace for the jobs")
    parser.add_argument("--python", default="3.11", help="Python version passed to `hf jobs uv run`")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset-id", default="JosueG/adja-tts-orpheus")
    parser.add_argument("--results-repo-id", default="JosueG/qwen3-adja-results")
    parser.add_argument("--asr-model-id", default="Qwen/Qwen3-ASR-0.6B")
    parser.add_argument("--tts-model-id", default="Qwen/Qwen3-TTS-12Hz-0.6B-Base")
    parser.add_argument("--asr-output-repo-id", default="JosueG/qwen3-adja-asr-0p6b")
    parser.add_argument("--tts-output-repo-id", default="JosueG/qwen3-adja-tts-0p6b")
    parser.add_argument("--asr-flavor", default=None, help="Override the HF flavor for ASR")
    parser.add_argument("--tts-flavor", default=None, help="Override the HF flavor for TTS")
    parser.add_argument("--asr-timeout", default=None, help="Override the HF timeout for ASR")
    parser.add_argument("--tts-timeout", default=None, help="Override the HF timeout for TTS")
    parser.add_argument("--execute", action="store_true", help="Run the submission commands instead of printing them")
    parser.add_argument("--detach", action="store_true", help="Submit detached HF jobs")
    return parser.parse_args()


def build_specs(args: argparse.Namespace) -> list[JobSpec]:
    smoke = not args.pilot
    specs: list[JobSpec] = []

    if args.modality in {"asr", "both"}:
        specs.append(
            JobSpec(
                name="qwen3-asr-adja",
                script_path="scripts/hf_jobs/qwen3_asr_adja.py",
                flavor=args.asr_flavor or "a10g-large",
                timeout=args.asr_timeout or "4h",
                env={
                    "HF_DATASET_ID": args.dataset_id,
                    "MODEL_ID": args.asr_model_id,
                    "OUTPUT_REPO_ID": args.asr_output_repo_id,
                    "RESULTS_REPO_ID": args.results_repo_id,
                    "SMOKE_RUN": "1" if smoke else "0",
                    "WORKSPACE_DIR": "/tmp/qwen3_adja_asr",
                    "SEED": str(args.seed),
                    "PYTHONUNBUFFERED": "1",
                },
            )
        )

    if args.modality in {"tts", "both"}:
        specs.append(
            JobSpec(
                name="qwen3-tts-adja",
                script_path="scripts/hf_jobs/qwen3_tts_adja.py",
                flavor=args.tts_flavor or "l40sx1",
                timeout=args.tts_timeout or ("4h" if smoke else "8h"),
                env={
                    "HF_DATASET_ID": args.dataset_id,
                    "MODEL_ID": args.tts_model_id,
                    "OUTPUT_REPO_ID": args.tts_output_repo_id,
                    "RESULTS_REPO_ID": args.results_repo_id,
                    "SMOKE_RUN": "1" if smoke else "0",
                    "WORKSPACE_DIR": "/tmp/qwen3_adja_tts",
                    "SEED": str(args.seed),
                    "PYTHONUNBUFFERED": "1",
                },
            )
        )

    return specs


def build_command(spec: JobSpec, args: argparse.Namespace) -> list[str]:
    command = [
        "hf",
        "jobs",
        "uv",
        "run",
        "--flavor",
        spec.flavor,
        "--timeout",
        spec.timeout,
        "--python",
        args.python,
        "--secrets",
        "HF_TOKEN",
    ]
    if args.detach:
        command.append("--detach")
    if args.namespace:
        command.extend(["--namespace", args.namespace])
    for key, value in spec.env.items():
        command.extend(["-e", f"{key}={value}"])
    command.append(spec.script_path)
    return command


def main() -> None:
    args = parse_args()
    specs = build_specs(args)

    if not specs:
        raise SystemExit("No jobs selected")

    print(f"Repo root: {REPO_ROOT}")
    for spec in specs:
        command = build_command(spec, args)
        print()
        print(f"[{spec.name}]")
        print(shlex.join(command))
        if args.execute:
            subprocess.check_call(command, cwd=str(REPO_ROOT))


if __name__ == "__main__":
    main()
