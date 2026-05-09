#!/usr/bin/env python3
"""Publish canonical public ASR/TTS result reporting repos.

Default mode is dry-run. Pass --apply to create/update the Hugging Face dataset
repos and add them to the thesis collection.

This publishes reports, metrics, selected predictions, and manifests only. It
does not upload checkpoints or mutate old mixed experiment/result repos.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi, create_repo


RESULT_REPOS = {
    "asr-results": "JosueG/cs-thesis-may-2026-asr-results",
    "tts-results": "JosueG/cs-thesis-may-2026-tts-results",
}

COLLECTION_SLUG = "JosueG/cs-thesis-may-2026-69ff55e45e0d5b0eb7fa4344"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exports-dir", type=Path, default=Path("results/canonical_exports"))
    parser.add_argument("--private", action="store_true", help="Create result repos as private instead of public")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    api = HfApi()

    for export_name, repo_id in RESULT_REPOS.items():
        folder = args.exports_dir / export_name
        if not folder.exists():
            raise SystemExit(f"Missing export folder: {folder}")

        print(f"Result export ready: {repo_id} <- {folder}")
        if args.apply:
            create_repo(repo_id, repo_type="dataset", private=args.private, exist_ok=True)
            api.upload_folder(
                folder_path=str(folder),
                repo_id=repo_id,
                repo_type="dataset",
                commit_message="Publish CS thesis result reporting export",
            )
            api.add_collection_item(COLLECTION_SLUG, repo_id, "dataset", exists_ok=True)

    if not args.apply:
        print("Dry run only. Re-run with --apply to create/update result repos and collection entries.")


if __name__ == "__main__":
    main()
