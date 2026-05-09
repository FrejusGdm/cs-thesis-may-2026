#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINING_REQUIREMENTS = REPO_ROOT / "experiments" / "finetuning-qwen3" / "training" / "hf" / "requirements.txt"


def require_env(name: str, default: str | None = None, required: bool = True) -> str:
    value = os.environ.get(name, default)
    if required and not value:
        raise SystemExit(f"{name} is required")
    return value or ""


def install_requirements(extra_packages: list[str] | None = None) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(TRAINING_REQUIREMENTS)])
    if extra_packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", *extra_packages])


def run_logged(command: list[str], log_path: Path, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)

    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=str(cwd or REPO_ROOT),
            env=merged_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log_handle.write(line)
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def subset_jsonl(source: Path, destination: Path, limit: int) -> Path:
    records = read_jsonl(source)[:limit]
    write_jsonl(destination, records)
    return destination


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def find_latest_checkpoint(output_dir: Path) -> Path | None:
    patterns = [
        re.compile(r"^checkpoint-(\d+)$"),
        re.compile(r"^checkpoint-epoch-(\d+)$"),
    ]
    best: tuple[int, Path] | None = None
    if not output_dir.exists():
        return None

    for path in output_dir.iterdir():
        if not path.is_dir():
            continue
        for pattern in patterns:
            match = pattern.match(path.name)
            if not match:
                continue
            step = int(match.group(1))
            if best is None or step > best[0]:
                best = (step, path)
            break
    return best[1] if best else None


def ensure_repo(api, repo_id: str, repo_type: str = "model") -> None:
    api.create_repo(repo_id, private=True, exist_ok=True, repo_type=repo_type)


def upload_file_if_exists(api, repo_id: str, local_path: Path, path_in_repo: str, repo_type: str = "model") -> None:
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
        repo_id=repo_id,
        path_in_repo=path_in_repo,
        repo_type=repo_type,
    )
