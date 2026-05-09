#!/usr/bin/env python3
"""Duplicate the Adja speech dataset into the canonical thesis dataset repo.

Default mode is dry-run. Pass --apply to create/upload.

This script reads from JosueG/adja-tts-mms-ready and writes only to a new target
repo. It never mutates the source repo.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import HfApi, create_repo


SOURCE_REPO = "JosueG/adja-tts-mms-ready"
DEFAULT_TARGET_REPO = "JosueG/cs-thesis-may-2026-data"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-repo", default=SOURCE_REPO)
    parser.add_argument("--target-repo", default=DEFAULT_TARGET_REPO)
    parser.add_argument("--private", action="store_true", default=True)
    parser.add_argument("--public", action="store_false", dest="private")
    parser.add_argument("--apply", action="store_true", help="Actually create/upload the target repo")
    args = parser.parse_args()

    print(f"Source repo: {args.source_repo}")
    print(f"Target repo: {args.target_repo}")
    print(f"Visibility on create: {'private' if args.private else 'public'}")

    ds = load_dataset(args.source_repo)
    print(ds)

    manifest = {
        "source_repo": args.source_repo,
        "target_repo": args.target_repo,
        "component": "speech/adja-asr-tts",
        "duplicated_at": datetime.now(timezone.utc).isoformat(),
        "splits": {split: int(len(part)) for split, part in ds.items()},
        "schema": {split: list(part.features.keys()) for split, part in ds.items()},
        "policy": "source repo is read-only; target repo is the canonical thesis duplicate",
    }
    print(manifest)

    if not args.apply:
        print("Dry run only. Re-run with --apply after README/card review.")
        return

    api = HfApi()
    create_repo(args.target_repo, repo_type="dataset", private=args.private, exist_ok=True)

    # Store speech as a config name so future text components can coexist.
    ds.push_to_hub(
        args.target_repo,
        config_name="speech_adja_asr_tts",
        commit_message="Add duplicated Adja ASR/TTS speech dataset component",
    )

    tmp = Path("hf_release_manifest_speech_adja_asr_tts.json")
    tmp.write_text(__import__("json").dumps(manifest, indent=2, sort_keys=True) + "\n")
    api.upload_file(
        path_or_fileobj=str(tmp),
        path_in_repo="manifests/speech_adja_asr_tts_manifest.json",
        repo_id=args.target_repo,
        repo_type="dataset",
        commit_message="Add speech dataset duplication manifest",
    )
    tmp.unlink(missing_ok=True)

    card = Path("data/hf_cards/cs-thesis-may-2026-data.README.md")
    if card.exists():
        api.upload_file(
            path_or_fileobj=str(card),
            path_in_repo="README.md",
            repo_id=args.target_repo,
            repo_type="dataset",
            commit_message="Add canonical thesis dataset card",
        )


if __name__ == "__main__":
    main()
