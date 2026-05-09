#!/usr/bin/env python3
from __future__ import annotations

"""
Submit OmniASR 7B Adja fine-tuning jobs via `hf jobs uv run`.

This helper keeps CTC and LLM runs in separate namespaces to avoid clobbering
older Omni experiments.
"""

import argparse
import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class JobSpec:
    name: str
    family: str
    flavor: str
    timeout: str
    env: dict[str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit OmniASR 7B Adja jobs")
    parser.add_argument("--family", choices=["ctc", "llm", "both"], default="both")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--smoke", action="store_true", help="Run short validation jobs")
    mode_group.add_argument("--pilot", action="store_true", help="Run longer pilot jobs")
    parser.add_argument("--namespace", default=os.environ.get("HF_NAMESPACE", ""))
    parser.add_argument("--python", default="3.11")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset-id", default="JosueG/adja-tts-orpheus")
    parser.add_argument("--results-repo-id", default="JosueG/adja-asr-results")
    parser.add_argument("--ctc-model-card", default="omniASR_CTC_7B_v2")
    parser.add_argument("--llm-model-card", default="omniASR_LLM_7B_v2")
    parser.add_argument("--ctc-output-repo-id", default="JosueG/omniASR-ctc-7b-v2-adja")
    parser.add_argument("--llm-output-repo-id", default="JosueG/omniASR-llm-7b-v2-adja")
    parser.add_argument("--tag", default="2026-04-19-a1", help="Unique suffix to avoid overwriting prior runs")
    parser.add_argument("--ctc-flavor", default=None)
    parser.add_argument("--llm-flavor", default=None)
    parser.add_argument("--ctc-timeout", default=None)
    parser.add_argument("--llm-timeout", default=None)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--detach", action="store_true")
    return parser.parse_args()


def build_specs(args: argparse.Namespace) -> list[JobSpec]:
    smoke = not args.pilot
    specs: list[JobSpec] = []

    if args.family in {"ctc", "both"}:
        experiment_id = f"Omni_FT_CTC_7B_v2_{'smoke' if smoke else 'pilot'}_{args.tag}"
        specs.append(
            JobSpec(
                name="omni-ft-ctc-7b-v2",
                family="ctc",
                flavor=args.ctc_flavor or "a100-large",
                timeout=args.ctc_timeout or ("3h" if smoke else "8h"),
                env={
                    "MODEL_FAMILY": "ctc",
                    "HF_DATASET_ID": args.dataset_id,
                    "MODEL_CARD": args.ctc_model_card,
                    "OUTPUT_REPO_ID": args.ctc_output_repo_id,
                    "RESULTS_REPO_ID": args.results_repo_id,
                    "EXPERIMENT_ID": experiment_id,
                    "SMOKE_RUN": "1" if smoke else "0",
                    "SEED": str(args.seed),
                    "WORKSPACE_DIR": f"/tmp/{experiment_id.lower()}",
                    "PYTHONUNBUFFERED": "1",
                },
            )
        )

    if args.family in {"llm", "both"}:
        experiment_id = f"Omni_FT_LLM_7B_v2_{'smoke' if smoke else 'pilot'}_{args.tag}"
        specs.append(
            JobSpec(
                name="omni-ft-llm-7b-v2",
                family="llm",
                flavor=args.llm_flavor or "a100-large",
                timeout=args.llm_timeout or ("4h" if smoke else "10h"),
                env={
                    "MODEL_FAMILY": "llm",
                    "HF_DATASET_ID": args.dataset_id,
                    "MODEL_CARD": args.llm_model_card,
                    "OUTPUT_REPO_ID": args.llm_output_repo_id,
                    "RESULTS_REPO_ID": args.results_repo_id,
                    "EXPERIMENT_ID": experiment_id,
                    "SMOKE_RUN": "1" if smoke else "0",
                    "SEED": str(args.seed),
                    "WORKSPACE_DIR": f"/tmp/{experiment_id.lower()}",
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
    command.append("scripts/hf_jobs/omni_asr_finetune.py")
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
