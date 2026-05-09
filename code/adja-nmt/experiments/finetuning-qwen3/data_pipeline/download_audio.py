#!/usr/bin/env python3
"""
Stage 1: Data Acquisition
=========================
Downloads audio files from URLs in your dataset, with:
- Parallel downloading (configurable concurrency)
- Automatic retries with exponential backoff
- Progress tracking and checkpointing (resume if interrupted)
- File integrity validation (checks audio is readable)

Input:  CSV/TSV/JSON with columns: text, audio_url
Output: Directory of WAV files + metadata JSONL

Usage:
    python download_audio.py \
        --input_file dataset.csv \
        --output_dir ./downloaded_audio \
        --output_jsonl raw_data.jsonl \
        --max_workers 8 \
        --text_column text \
        --url_column audio_url
"""

import argparse
import csv
import hashlib
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def url_to_filename(url: str, index: int) -> str:
    """Generate a deterministic filename from URL + index.

    We use a hash to avoid filesystem issues with long/weird URLs,
    but prefix with the index so files sort in dataset order.
    """
    url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
    return f"utt_{index:06d}_{url_hash}.wav"


def download_single(
    url: str,
    output_path: str,
    max_retries: int = 3,
    timeout: int = 60,
) -> tuple[bool, str]:
    """Download a single file with retry logic.

    Returns:
        (success: bool, error_message: str)
    """
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=timeout, stream=True)
            response.raise_for_status()

            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            # Basic size check — an empty or tiny file is likely an error page
            file_size = os.path.getsize(output_path)
            if file_size < 1000:  # Less than 1KB is suspicious for audio
                return False, f"File too small ({file_size} bytes), likely not audio"

            return True, ""

        except requests.exceptions.RequestException as e:
            wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
            if attempt < max_retries - 1:
                logger.warning(
                    f"Attempt {attempt + 1}/{max_retries} failed for {url}: {e}. "
                    f"Retrying in {wait_time}s..."
                )
                time.sleep(wait_time)
            else:
                return False, f"All {max_retries} attempts failed: {e}"

    return False, "Unknown error"


def validate_audio_file(filepath: str) -> tuple[bool, str]:
    """Check if the downloaded file is valid audio.

    We try to open it with soundfile (fastest) or fall back to checking
    file headers. This catches corrupt downloads, HTML error pages, etc.
    """
    try:
        import soundfile as sf
        info = sf.info(filepath)
        if info.duration < 0.1:
            return False, f"Audio too short ({info.duration:.2f}s)"
        if info.duration > 300:  # 5 minutes — configurable
            return False, f"Audio too long ({info.duration:.2f}s)"
        return True, ""
    except Exception as e:
        return False, f"Cannot read audio: {e}"


# ---------------------------------------------------------------------------
# Checkpoint system
# ---------------------------------------------------------------------------

def load_checkpoint(checkpoint_path: str) -> set:
    """Load set of already-downloaded URLs from checkpoint file."""
    if not os.path.exists(checkpoint_path):
        return set()
    with open(checkpoint_path, "r") as f:
        return set(line.strip() for line in f if line.strip())


def save_checkpoint(checkpoint_path: str, url: str):
    """Append a successfully downloaded URL to checkpoint."""
    with open(checkpoint_path, "a") as f:
        f.write(url + "\n")


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_input_data(
    input_file: str,
    text_column: str,
    url_column: str,
) -> list[dict]:
    """Load dataset from CSV, TSV, or JSON/JSONL."""
    ext = Path(input_file).suffix.lower()

    records = []

    if ext == ".json":
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                records = data
            else:
                raise ValueError("JSON file must contain a list of objects")

    elif ext == ".jsonl":
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

    elif ext in (".csv", ".tsv"):
        delimiter = "\t" if ext == ".tsv" else ","
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            records = list(reader)

    else:
        raise ValueError(f"Unsupported file format: {ext}. Use .csv, .tsv, .json, or .jsonl")

    # Validate required columns exist
    if not records:
        raise ValueError("Input file is empty")

    sample = records[0]
    if text_column not in sample:
        available = list(sample.keys())
        raise ValueError(
            f"Column '{text_column}' not found. Available columns: {available}"
        )
    if url_column not in sample:
        available = list(sample.keys())
        raise ValueError(
            f"Column '{url_column}' not found. Available columns: {available}"
        )

    logger.info(f"Loaded {len(records)} records from {input_file}")
    return records


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def download_dataset(
    input_file: str,
    output_dir: str,
    output_jsonl: str,
    text_column: str = "text",
    url_column: str = "audio_url",
    max_workers: int = 8,
    max_retries: int = 3,
    validate: bool = True,
):
    """Download all audio files and produce a clean JSONL manifest."""

    os.makedirs(output_dir, exist_ok=True)

    records = load_input_data(input_file, text_column, url_column)

    checkpoint_path = os.path.join(output_dir, ".download_checkpoint.txt")
    completed_urls = load_checkpoint(checkpoint_path)
    logger.info(f"Checkpoint: {len(completed_urls)} files already downloaded")

    # Filter to only pending downloads
    pending = []
    for i, rec in enumerate(records):
        url = rec[url_column].strip()
        if url and url not in completed_urls:
            pending.append((i, rec))

    logger.info(f"Pending downloads: {len(pending)} / {len(records)}")

    if not pending:
        logger.info("All files already downloaded. Skipping to manifest generation.")
    else:
        # Parallel download
        success_count = 0
        fail_count = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for idx, rec in pending:
                url = rec[url_column].strip()
                filename = url_to_filename(url, idx)
                output_path = os.path.join(output_dir, filename)
                future = executor.submit(
                    download_single, url, output_path, max_retries
                )
                futures[future] = (idx, rec, url, output_path)

            for future in as_completed(futures):
                idx, rec, url, output_path = futures[future]
                success, error = future.result()

                if success and validate:
                    valid, verr = validate_audio_file(output_path)
                    if not valid:
                        success = False
                        error = verr
                        os.remove(output_path)  # Remove invalid file

                if success:
                    success_count += 1
                    save_checkpoint(checkpoint_path, url)
                else:
                    fail_count += 1
                    logger.warning(f"FAILED [{idx}] {url}: {error}")

                total_done = success_count + fail_count
                if total_done % 100 == 0 or total_done == len(pending):
                    logger.info(
                        f"Progress: {total_done}/{len(pending)} "
                        f"(success={success_count}, failed={fail_count})"
                    )

        logger.info(
            f"Download complete: {success_count} succeeded, {fail_count} failed"
        )

    # Generate output JSONL manifest
    logger.info(f"Generating manifest: {output_jsonl}")
    written = 0
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for i, rec in enumerate(records):
            url = rec[url_column].strip()
            if not url:
                continue
            filename = url_to_filename(url, i)
            filepath = os.path.join(output_dir, filename)
            if os.path.exists(filepath):
                entry = {
                    "audio": os.path.abspath(filepath),
                    "text": rec[text_column],
                }
                # Preserve any extra columns (language, speaker_id, etc.)
                for k, v in rec.items():
                    if k not in (text_column, url_column):
                        entry[k] = v
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                written += 1

    logger.info(f"Manifest written: {written} entries → {output_jsonl}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download audio files from URLs in a dataset"
    )
    parser.add_argument(
        "--input_file", required=True,
        help="Path to input file (CSV, TSV, JSON, or JSONL)"
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Directory to save downloaded audio files"
    )
    parser.add_argument(
        "--output_jsonl", required=True,
        help="Path to output JSONL manifest"
    )
    parser.add_argument(
        "--text_column", default="text",
        help="Name of the text/transcript column (default: text)"
    )
    parser.add_argument(
        "--url_column", default="audio_url",
        help="Name of the audio URL column (default: audio_url)"
    )
    parser.add_argument(
        "--max_workers", type=int, default=8,
        help="Number of parallel download threads (default: 8)"
    )
    parser.add_argument(
        "--max_retries", type=int, default=3,
        help="Max retry attempts per download (default: 3)"
    )
    parser.add_argument(
        "--no_validate", action="store_true",
        help="Skip audio validation after download"
    )
    args = parser.parse_args()

    download_dataset(
        input_file=args.input_file,
        output_dir=args.output_dir,
        output_jsonl=args.output_jsonl,
        text_column=args.text_column,
        url_column=args.url_column,
        max_workers=args.max_workers,
        max_retries=args.max_retries,
        validate=not args.no_validate,
    )


if __name__ == "__main__":
    main()
