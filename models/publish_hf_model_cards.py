#!/usr/bin/env python3
"""Publish thesis-final model cards and update the HF collection.

Default mode is dry-run. Pass --apply to upload README cards, make the selected
private thesis-final model repos public, and update the collection.

This script intentionally does not copy checkpoints, rename repos, or edit
non-target repos.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi


TARGET_MODELS = [
    "JosueG/adja-nmt-nllb-600m-forward-r10ks4k-seed42",
    "JosueG/adja-nmt-nllb-600m-reverse-r10ks4k-seed42",
    "JosueG/wav2vec2-xlsr-adja-c4v2",
    "JosueG/whisper-ewe-adja-e4v4",
    "JosueG/spark-tts-adja-t3",
]

MAKE_PUBLIC = [
    "JosueG/adja-nmt-nllb-600m-forward-r10ks4k-seed42",
    "JosueG/adja-nmt-nllb-600m-reverse-r10ks4k-seed42",
    "JosueG/spark-tts-adja-t3",
]

COLLECTION_TITLE = "CS Thesis May 2026"
COLLECTION_NAMESPACE = "JosueG"
GITHUB_RELEASE_URL = "https://github.com/FrejusGdm/cs-thesis-may-2026"

COLLECTION_DATASETS = [
    "JosueG/adja-speech-asr-tts",
    "JosueG/french-adja-parallel-corpus",
]

COLLECTION_DESCRIPTION = "Adja MT, ASR, TTS datasets and models for Josue Godeme's May 2026 CS thesis. GitHub: github.com/FrejusGdm/cs-thesis-may-2026"


def card_path(cards_dir: Path, repo_id: str) -> Path:
    return cards_dir / f"{repo_id.split('/', 1)[1]}.README.md"


def find_or_create_collection(api: HfApi, apply: bool):
    if not apply:
        print(f"Collection target: {COLLECTION_NAMESPACE}/{COLLECTION_TITLE}")
        return None

    matches = [
        collection
        for collection in api.list_collections(owner=COLLECTION_NAMESPACE, limit=100)
        if collection.title == COLLECTION_TITLE
    ]
    if matches:
        collection = matches[0]
        print(f"Collection exists: {collection.slug}")
        collection = api.update_collection_metadata(
            collection.slug,
            description=COLLECTION_DESCRIPTION,
            private=False,
        )
        return collection

    print(f"Collection missing: {COLLECTION_NAMESPACE}/{COLLECTION_TITLE}")

    collection = api.create_collection(
        COLLECTION_TITLE,
        namespace=COLLECTION_NAMESPACE,
        description=COLLECTION_DESCRIPTION,
        private=False,
        exists_ok=True,
    )
    print(f"Created collection: {collection.slug}")
    return collection


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cards-dir", type=Path, default=Path("models/hf_cards"))
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    api = HfApi()

    for repo_id in TARGET_MODELS:
        readme = card_path(args.cards_dir, repo_id)
        if not readme.exists():
            raise SystemExit(f"Missing model card file: {readme}")

        print(f"Card ready: {repo_id} <- {readme}")
        if args.apply:
            api.upload_file(
                path_or_fileobj=str(readme),
                path_in_repo="README.md",
                repo_id=repo_id,
                repo_type="model",
                commit_message="Update model card for CS thesis release",
            )

    for repo_id in MAKE_PUBLIC:
        print(f"Visibility target public: {repo_id}")
        if args.apply:
            api.update_repo_settings(repo_id=repo_id, repo_type="model", private=False)

    collection = find_or_create_collection(api, args.apply)
    if collection is not None:
        for repo_id in TARGET_MODELS:
            print(f"Collection model item: {repo_id}")
            if args.apply:
                api.add_collection_item(collection.slug, repo_id, "model", exists_ok=True)
        for repo_id in COLLECTION_DATASETS:
            print(f"Collection dataset item: {repo_id}")
            if args.apply:
                api.add_collection_item(collection.slug, repo_id, "dataset", exists_ok=True)

    if not args.apply:
        print("Dry run only. Re-run with --apply to publish cards, visibility, and collection updates.")


if __name__ == "__main__":
    main()
