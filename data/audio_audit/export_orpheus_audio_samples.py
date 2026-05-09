#!/usr/bin/env python3
"""Export a few Adja speech dataset rows to WAV for listening and diagnostics."""

from __future__ import annotations

import argparse
import csv
import math
import wave
from pathlib import Path
from typing import Any

import numpy as np
from datasets import load_dataset


def as_mono_float(audio: dict[str, Any]) -> tuple[np.ndarray, int]:
    if isinstance(audio, dict) and "array" in audio:
        array = np.asarray(audio["array"])
        sampling_rate = int(audio["sampling_rate"])
    elif isinstance(audio, dict) and "bytes" in audio:
        raise SystemExit(
            "This row stores encoded audio bytes. Install a compatible datasets audio decoder "
            "or materialize array/sampling_rate columns before running the audit."
        )
    else:
        raise SystemExit(f"Unsupported audio payload type: {type(audio)!r}")
    if array.ndim == 2:
        if array.shape[0] <= 2:
            array = array.mean(axis=0)
        else:
            array = array.mean(axis=1)
    array = np.asarray(array, dtype=np.float32)
    return array, sampling_rate


def stats_for(array: np.ndarray, sampling_rate: int) -> dict[str, str]:
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        finite = np.array([0.0], dtype=np.float32)
    peak = float(np.max(np.abs(finite)))
    rms = float(math.sqrt(float(np.mean(np.square(finite)))))
    return {
        "sampling_rate": str(sampling_rate),
        "duration_seconds": f"{len(array) / sampling_rate:.3f}" if sampling_rate else "0.000",
        "dtype_after_decode": str(array.dtype),
        "shape_after_decode": "x".join(str(part) for part in array.shape),
        "min": f"{float(np.min(finite)):.8f}",
        "max": f"{float(np.max(finite)):.8f}",
        "peak": f"{peak:.8f}",
        "rms": f"{rms:.8f}",
    }


def scale_for_wav(array: np.ndarray) -> tuple[np.ndarray, str]:
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return np.zeros_like(array, dtype=np.float32), "silent_or_non_finite"
    peak = float(np.max(np.abs(finite)))
    if peak == 0.0:
        return array.astype(np.float32), "zero_peak"
    if peak <= 1.0:
        return array.astype(np.float32), "already_float_minus1_to_1"
    if peak <= 65536.0:
        return (array / 65536.0).astype(np.float32), "integer_like_divide_by_65536"
    return (array / peak * 0.98).astype(np.float32), "peak_normalized_to_0p98"


def write_wav(path: Path, array: np.ndarray, sampling_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    scaled, _ = scale_for_wav(array)
    clipped = np.clip(scaled, -1.0, 1.0)
    pcm16 = (clipped * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sampling_rate)
        handle.writeframes(pcm16.tobytes())


def export_dataset(repo: str, config: str | None, split: str, output_dir: Path, prefix: str, max_samples: int, indices: list[int]) -> list[dict[str, str]]:
    kwargs = {}
    if config:
        kwargs["name"] = config
    dataset = load_dataset(repo, split=split, **kwargs)

    rows = []
    selected = indices if indices else list(range(min(max_samples, len(dataset))))
    for index in selected[:max_samples]:
        row = dataset[int(index)]
        if "audio" not in row:
            raise SystemExit(f"{repo}/{split} row {index} has no audio column")
        array, sampling_rate = as_mono_float(row["audio"])
        wav_path = output_dir / prefix / f"{prefix}_{int(index):05d}.wav"
        write_wav(wav_path, array, sampling_rate)
        record = {
            "repo": repo,
            "config": config or "",
            "split": split,
            "index": str(int(index)),
            "wav": str(wav_path),
            "text": str(row.get("text", ""))[:300],
        }
        record.update(stats_for(array, sampling_rate))
        _, scaling = scale_for_wav(array)
        record["wav_scaling"] = scaling
        rows.append(record)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="JosueG/adja-speech-asr-tts")
    parser.add_argument("--config", default="adja_speech_orpheus_48khz")
    parser.add_argument("--split", default="train")
    parser.add_argument("--compare-repo")
    parser.add_argument("--compare-config")
    parser.add_argument("--compare-split", default="train")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=5)
    parser.add_argument("--indices", default="", help="Comma-separated row indices")
    args = parser.parse_args()

    indices = [int(part) for part in args.indices.split(",") if part.strip()]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = export_dataset(args.repo, args.config or None, args.split, output_dir, "orpheus", args.max_samples, indices)
    if args.compare_repo:
        rows.extend(export_dataset(
            args.compare_repo,
            args.compare_config or None,
            args.compare_split,
            output_dir,
            "compare",
            args.max_samples,
            indices,
        ))

    manifest = output_dir / "audio_audit_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} WAV files and manifest: {manifest}")


if __name__ == "__main__":
    main()
