#!/usr/bin/env python3
"""
Dataset Validation
==================
Validates a prepared JSONL dataset before training.
Catches common problems that would cause training to crash or produce poor results.

Checks:
1. All audio files exist and are readable
2. All required fields are present
3. Audio duration is within expected range
4. Text is non-empty and properly formatted
5. (TTS) ref_audio exists
6. (ASR) language prefix is correctly formatted
7. Sample rate matches target

Usage:
    python validate_dataset.py \
        --input_jsonl splits/train.jsonl \
        --mode tts \
        --expected_sr 24000
"""

import argparse
import json
import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def validate_record_tts(rec: dict, index: int, expected_sr: int) -> list[str]:
    """Validate a single TTS training record."""
    errors = []

    # Check required fields
    for field in ("audio", "text"):
        if field not in rec:
            errors.append(f"[{index}] Missing field: {field}")

    # Check audio file exists
    audio_path = rec.get("audio", "")
    if audio_path and not os.path.exists(audio_path):
        errors.append(f"[{index}] Audio file not found: {audio_path}")
    elif audio_path:
        try:
            import soundfile as sf
            info = sf.info(audio_path)
            if info.samplerate != expected_sr:
                errors.append(
                    f"[{index}] Wrong sample rate: {info.samplerate} "
                    f"(expected {expected_sr})"
                )
            if info.duration < 0.3:
                errors.append(f"[{index}] Audio too short: {info.duration:.2f}s")
            if info.duration > 60:
                errors.append(f"[{index}] Audio very long: {info.duration:.2f}s (may cause OOM)")
            if info.channels != 1:
                errors.append(f"[{index}] Not mono: {info.channels} channels")
        except Exception as e:
            errors.append(f"[{index}] Cannot read audio: {e}")

    # Check text
    text = rec.get("text", "")
    if not text or not text.strip():
        errors.append(f"[{index}] Empty text")

    # Check ref_audio (TTS-specific)
    ref_audio = rec.get("ref_audio", "")
    if ref_audio and not os.path.exists(ref_audio):
        errors.append(f"[{index}] ref_audio not found: {ref_audio}")

    return errors


def validate_record_asr(rec: dict, index: int, expected_sr: int) -> list[str]:
    """Validate a single ASR training record."""
    errors = []

    for field in ("audio", "text"):
        if field not in rec:
            errors.append(f"[{index}] Missing field: {field}")

    # Check audio
    audio_path = rec.get("audio", "")
    if audio_path and not os.path.exists(audio_path):
        errors.append(f"[{index}] Audio file not found: {audio_path}")
    elif audio_path:
        try:
            import soundfile as sf
            info = sf.info(audio_path)
            if info.samplerate != expected_sr:
                errors.append(
                    f"[{index}] Wrong sample rate: {info.samplerate} "
                    f"(expected {expected_sr})"
                )
        except Exception as e:
            errors.append(f"[{index}] Cannot read audio: {e}")

    # Check ASR text format
    text = rec.get("text", "")
    if not text:
        errors.append(f"[{index}] Empty text")
    elif "<asr_text>" in text:
        # If already formatted for ASR, validate the prefix
        if not text.startswith("language "):
            errors.append(f"[{index}] ASR text missing 'language' prefix")
    # If plain text (not yet formatted), that's fine — process_text.py handles it

    return errors


def validate_dataset(
    input_jsonl: str,
    mode: str = "tts",
    expected_sr: int = 24000,
    max_errors: int = 50,
):
    """Validate an entire dataset JSONL."""
    records = []
    with open(input_jsonl, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.error(f"Line {line_num}: Invalid JSON: {e}")

    logger.info(f"Validating {len(records)} records (mode={mode}, expected_sr={expected_sr})")

    all_errors = []
    validate_fn = validate_record_tts if mode == "tts" else validate_record_asr

    for i, rec in enumerate(records):
        errors = validate_fn(rec, i, expected_sr)
        all_errors.extend(errors)
        if len(all_errors) >= max_errors:
            logger.warning(f"Stopping early after {max_errors} errors")
            break

    # Report
    logger.info("=" * 60)
    if all_errors:
        logger.error(f"VALIDATION FAILED: {len(all_errors)} errors found")
        for err in all_errors[:max_errors]:
            logger.error(f"  {err}")
        if len(all_errors) > max_errors:
            logger.error(f"  ... and {len(all_errors) - max_errors} more")
        return False
    else:
        logger.info("VALIDATION PASSED: All records are valid")

        # Print dataset summary
        if records:
            texts = [r.get("text", "") for r in records]
            avg_text_len = sum(len(t) for t in texts) / len(texts)
            logger.info(f"  Records:          {len(records)}")
            logger.info(f"  Avg text length:  {avg_text_len:.0f} chars")

            # Check for duplicate texts (potential data quality issue)
            unique_texts = set(texts)
            if len(unique_texts) < len(texts):
                dupes = len(texts) - len(unique_texts)
                logger.warning(
                    f"  Duplicate texts:  {dupes} "
                    "(consider deduplicating for better training quality)"
                )

        return True


def main():
    parser = argparse.ArgumentParser(description="Validate dataset for Qwen3 fine-tuning")
    parser.add_argument("--input_jsonl", required=True, help="JSONL file to validate")
    parser.add_argument("--mode", choices=["tts", "asr"], default="tts", help="Validation mode")
    parser.add_argument(
        "--expected_sr", type=int, default=24000,
        help="Expected sample rate (24000 for TTS, 16000 for ASR)"
    )
    parser.add_argument("--max_errors", type=int, default=50, help="Stop after this many errors")
    args = parser.parse_args()

    valid = validate_dataset(
        input_jsonl=args.input_jsonl,
        mode=args.mode,
        expected_sr=args.expected_sr,
        max_errors=args.max_errors,
    )

    sys.exit(0 if valid else 1)


if __name__ == "__main__":
    main()
