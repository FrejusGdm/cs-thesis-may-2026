#!/usr/bin/env python3
"""
Prepare OmniASR manifest files from an HPC-local metadata TSV.

Input TSV must be tab-delimited with header columns:
  - split: train|dev|test
  - text: transcription
  - one of:
      * audio_path (absolute path), or
      * audio_relpath (relative path) + --audio-root
  - optional: lang or language (falls back to --default-lang)

Output files in --output-manifest-dir:
  - train.tsv / train.wrd / train.lang
  - dev.tsv   / dev.wrd   / dev.lang
  - test.tsv  / test.wrd  / test.lang
  - manifest_stats.json
"""

from __future__ import annotations

import argparse
import csv
import json
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf


VALID_SPLITS = {"train", "dev", "test"}


@dataclass
class Sample:
    split: str
    audio_path: Path
    text: str
    lang: str
    num_frames: int


def normalize_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text.strip()).split())


def resolve_audio_path(row: dict[str, str], audio_root: Path | None) -> Path:
    audio_path = (row.get("audio_path") or "").strip()
    audio_relpath = (row.get("audio_relpath") or "").strip()

    if audio_path:
        path = Path(audio_path)
        if not path.is_absolute():
            raise ValueError(f"audio_path must be absolute, got: {audio_path}")
        return path

    if audio_relpath:
        if audio_root is None:
            raise ValueError("audio_relpath provided, but --audio-root was not set.")
        return audio_root / audio_relpath

    raise ValueError("Each row must provide either audio_path or audio_relpath.")


def read_samples(
    input_tsv: Path,
    audio_root: Path | None,
    default_lang: str,
    min_audio_samples: int,
    expected_sample_rate: int,
) -> tuple[list[Sample], dict[str, dict[str, int]]]:
    samples: list[Sample] = []
    stats: dict[str, dict[str, int]] = {
        split: {
            "total": 0,
            "kept": 0,
            "dropped_short_audio": 0,
            "dropped_empty_text": 0,
            "dropped_bad_sample_rate": 0,
        }
        for split in VALID_SPLITS
    }

    with input_tsv.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp, delimiter="\t")
        required = {"split", "text"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("Input TSV must include header columns: split, text.")

        for row_idx, row in enumerate(reader, start=2):
            split = (row.get("split") or "").strip()
            if split not in VALID_SPLITS:
                raise ValueError(f"Row {row_idx}: invalid split '{split}'.")

            stats[split]["total"] += 1

            text = normalize_text(row.get("text") or "")
            if not text:
                stats[split]["dropped_empty_text"] += 1
                continue

            audio_path = resolve_audio_path(row, audio_root)
            if not audio_path.exists():
                raise FileNotFoundError(f"Row {row_idx}: audio file not found: {audio_path}")

            info = sf.info(str(audio_path))
            num_frames = int(info.frames)
            sr = int(info.samplerate)
            if expected_sample_rate > 0 and sr != expected_sample_rate:
                stats[split]["dropped_bad_sample_rate"] += 1
                continue
            if num_frames < min_audio_samples:
                stats[split]["dropped_short_audio"] += 1
                continue

            lang = (row.get("lang") or row.get("language") or "").strip() or default_lang

            samples.append(
                Sample(
                    split=split,
                    audio_path=audio_path,
                    text=text,
                    lang=lang,
                    num_frames=num_frames,
                )
            )
            stats[split]["kept"] += 1

    return samples, stats


def write_manifest_files(samples: list[Sample], output_manifest_dir: Path) -> None:
    output_manifest_dir.mkdir(parents=True, exist_ok=True)

    # Use "/" as manifest root to support any absolute audio location.
    manifest_root = Path("/")
    per_split = {split: [s for s in samples if s.split == split] for split in VALID_SPLITS}

    for split, split_samples in per_split.items():
        tsv_path = output_manifest_dir / f"{split}.tsv"
        wrd_path = output_manifest_dir / f"{split}.wrd"
        lang_path = output_manifest_dir / f"{split}.lang"

        with (
            tsv_path.open("w", encoding="utf-8") as tsv_fp,
            wrd_path.open("w", encoding="utf-8") as wrd_fp,
            lang_path.open("w", encoding="utf-8") as lang_fp,
        ):
            tsv_fp.write(f"{manifest_root}\n")
            for sample in split_samples:
                # Example: <HPC_WORKDIR> -> dartfs/.../audio.wav
                rel_to_root = sample.audio_path.as_posix().lstrip("/")
                tsv_fp.write(f"{rel_to_root}\t{sample.num_frames}\n")
                wrd_fp.write(f"{sample.text}\n")
                lang_fp.write(f"{sample.lang}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare OmniASR manifest files from local TSV.")
    parser.add_argument("--input-tsv", type=Path, required=True, help="Absolute path to metadata TSV.")
    parser.add_argument(
        "--output-manifest-dir",
        type=Path,
        required=True,
        help="Absolute output directory for train/dev/test manifest files.",
    )
    parser.add_argument(
        "--audio-root",
        type=Path,
        default=None,
        help="Absolute audio root used when metadata rows provide audio_relpath.",
    )
    parser.add_argument(
        "--default-lang",
        type=str,
        default="ajg_Latn",
        help="Fallback language tag when row lang is missing.",
    )
    parser.add_argument(
        "--min-audio-samples",
        type=int,
        default=0,
        help="Drop audio shorter than this number of samples.",
    )
    parser.add_argument(
        "--expected-sample-rate",
        type=int,
        default=16000,
        help="Drop rows not matching this sample rate. Set 0 to disable check.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input_tsv.is_absolute():
        raise ValueError("--input-tsv must be an absolute path.")
    if not args.output_manifest_dir.is_absolute():
        raise ValueError("--output-manifest-dir must be an absolute path.")
    if args.audio_root is not None and not args.audio_root.is_absolute():
        raise ValueError("--audio-root must be an absolute path when provided.")

    samples, stats = read_samples(
        input_tsv=args.input_tsv,
        audio_root=args.audio_root,
        default_lang=args.default_lang,
        min_audio_samples=args.min_audio_samples,
        expected_sample_rate=args.expected_sample_rate,
    )

    write_manifest_files(samples, args.output_manifest_dir)

    stats_path = args.output_manifest_dir / "manifest_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps({"output_manifest_dir": str(args.output_manifest_dir), "stats": stats}, indent=2))


if __name__ == "__main__":
    main()
