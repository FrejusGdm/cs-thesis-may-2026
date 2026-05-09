#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import unicodedata
from pathlib import Path
from typing import Iterable

import librosa
import numpy as np
import soundfile as sf
from datasets import load_dataset

sys.stdout.reconfigure(line_buffering=True)


TEST_RATIO = 0.1
DEV_RATIO = 0.1


def normalize_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text.strip()).split())


def create_splits(dataset, seed: int):
    split1 = dataset.train_test_split(test_size=TEST_RATIO, seed=seed)
    split2 = split1["train"].train_test_split(
        test_size=DEV_RATIO / (1 - TEST_RATIO),
        seed=seed,
    )
    return {
        "train": split2["train"],
        "val": split2["test"],
        "test": split1["test"],
    }


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if audio.ndim > 1:
        audio = np.mean(audio, axis=-1)
    if orig_sr != target_sr:
        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
    return np.asarray(audio, dtype=np.float32)


def asr_text(text: str) -> str:
    return f"language None<asr_text>{text}"


def dump_jsonl(records: Iterable[dict], output_path: Path) -> int:
    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def export_split(split_data, split_name: str, base_dir: Path, mode: str, target_sr: int) -> dict:
    wav_dir = base_dir / "wavs" / split_name
    wav_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    total_duration = 0.0
    for index in range(len(split_data)):
        sample = split_data[index]
        audio = np.asarray(sample["audio"]["array"], dtype=np.float32)
        sr = int(sample["audio"]["sampling_rate"])
        audio = resample_audio(audio, sr, target_sr)

        utt_id = f"{split_name}_{index:05d}"
        wav_path = wav_dir / f"{utt_id}.wav"
        sf.write(wav_path, audio, target_sr)

        text = normalize_text(sample["text"])
        duration = len(audio) / float(target_sr)
        total_duration += duration

        if mode == "asr":
            record = {
                "audio": str(wav_path),
                "text": asr_text(text),
            }
            jsonl_name = f"{split_name}.jsonl"
        else:
            record = {
                "audio": str(wav_path),
                "text": text,
            }
            jsonl_name = f"{split_name}_raw.jsonl"

        records.append(record)

    output_path = base_dir / jsonl_name
    count = dump_jsonl(records, output_path)
    return {
        "jsonl": str(output_path),
        "num_records": count,
        "total_duration_sec": round(total_duration, 3),
        "target_sample_rate": target_sr,
    }


def materialize(dataset_id: str, token: str, workspace_dir: Path, mode: str, seed: int) -> dict:
    dataset = load_dataset(dataset_id, token=token, split="train")
    splits = create_splits(dataset, seed=seed)

    summary = {
        "dataset_id": dataset_id,
        "seed": seed,
        "mode": mode,
        "splits": {},
    }

    targets = []
    if mode in {"asr", "both"}:
        targets.append(("asr", workspace_dir / "asr", 16000))
    if mode in {"tts", "both"}:
        targets.append(("tts", workspace_dir / "tts", 24000))

    for target_mode, base_dir, target_sr in targets:
        base_dir.mkdir(parents=True, exist_ok=True)
        target_summary = {}
        for split_name, split_data in splits.items():
            target_summary[split_name] = export_split(
                split_data,
                split_name=split_name,
                base_dir=base_dir,
                mode=target_mode,
                target_sr=target_sr,
            )
        summary["splits"][target_mode] = target_summary

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize the Adja HF dataset into local ASR/TTS JSONL files and WAVs."
    )
    parser.add_argument("--dataset_id", default="JosueG/adja-tts-orpheus")
    parser.add_argument("--token_env_var", default="HF_TOKEN")
    parser.add_argument("--workspace_dir", required=True)
    parser.add_argument("--mode", choices=["asr", "tts", "both"], default="both")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    token = os.environ.get(args.token_env_var)
    if not token:
        raise SystemExit(f"{args.token_env_var} is required to access the private dataset")

    workspace_dir = Path(args.workspace_dir).resolve()
    workspace_dir.mkdir(parents=True, exist_ok=True)

    summary = materialize(
        dataset_id=args.dataset_id,
        token=token,
        workspace_dir=workspace_dir,
        mode=args.mode,
        seed=args.seed,
    )

    summary_path = workspace_dir / "materialization_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
