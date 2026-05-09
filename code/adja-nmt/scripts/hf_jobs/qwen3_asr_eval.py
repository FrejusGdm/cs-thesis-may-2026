#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#     "torch==2.5.1",
#     "torchaudio==2.5.1",
#     "qwen-asr==0.0.6",
#     "datasets>=2.18.0",
#     "librosa>=0.10.0",
#     "soundfile>=0.12.0",
#     "numpy>=1.24.0",
#     "huggingface-hub>=0.34.0",
# ]
# ///
from __future__ import annotations

"""
Qwen3-ASR held-out test evaluation for Hugging Face Jobs.

Prepared for: Josue Godeme
Requested by: Josue Godeme

This script computes actual test-set WER/CER for the fine-tuned Qwen3-ASR
checkpoint using the same held-out split seed as training.
"""

import json
import os
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np

sys.stdout.reconfigure(line_buffering=True)


def require_env(name: str, default: str | None = None, required: bool = True) -> str:
    value = os.environ.get(name, default)
    if required and not value:
        raise SystemExit(f"{name} is required")
    return value or ""


def normalize_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text.strip()).split())


def create_splits(dataset, seed: int):
    split1 = dataset.train_test_split(test_size=0.1, seed=seed)
    split2 = split1["train"].train_test_split(test_size=0.1 / 0.9, seed=seed)
    return {"train": split2["train"], "val": split2["test"], "test": split1["test"]}


_PUNCT_TO_STRIP = r"""[.,!?;:()\[\]"'«»“”‘’]"""


def normalize_for_metrics(text: str) -> str:
    import re

    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    text = re.sub(_PUNCT_TO_STRIP, "", text)
    return " ".join(text.split())


def edit_distance(ref: list[str], hyp: list[str]) -> tuple[int, int, int, int]:
    n, m = len(ref), len(hyp)
    dp = [[(0, 0, 0)] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = (0, 0, i)
    for j in range(1, m + 1):
        dp[0][j] = (0, j, 0)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                continue
            s, ins, d = dp[i - 1][j - 1]
            sub_cost = (s + 1, ins, d)
            s, ins, d = dp[i][j - 1]
            ins_cost = (s, ins + 1, d)
            s, ins, d = dp[i - 1][j]
            del_cost = (s, ins, d + 1)
            dp[i][j] = min([sub_cost, ins_cost, del_cost], key=lambda x: sum(x))
    s, i, d = dp[n][m]
    return s, i, d, s + i + d


def compute_word_metrics(references: list[str], hypotheses: list[str], normalize: bool) -> dict[str, Any]:
    total_edits = total_words = total_sub = total_ins = total_del = 0
    for ref_text, hyp_text in zip(references, hypotheses):
        if normalize:
            ref_text = normalize_for_metrics(ref_text)
            hyp_text = normalize_for_metrics(hyp_text)
        ref_words = ref_text.split()
        hyp_words = hyp_text.split()
        s, i, d, edits = edit_distance(ref_words, hyp_words)
        total_edits += edits
        total_sub += s
        total_ins += i
        total_del += d
        total_words += len(ref_words)
    wer = (total_edits / total_words * 100.0) if total_words else 0.0
    return {
        "wer": round(wer, 2),
        "total_edits": total_edits,
        "total_ref_words": total_words,
        "substitutions": total_sub,
        "insertions": total_ins,
        "deletions": total_del,
    }


def compute_char_metrics(references: list[str], hypotheses: list[str], normalize: bool) -> dict[str, Any]:
    total_edits = total_chars = total_sub = total_ins = total_del = 0
    for ref_text, hyp_text in zip(references, hypotheses):
        if normalize:
            ref_text = normalize_for_metrics(ref_text)
            hyp_text = normalize_for_metrics(hyp_text)
        ref_chars = list(ref_text)
        hyp_chars = list(hyp_text)
        s, i, d, edits = edit_distance(ref_chars, hyp_chars)
        total_edits += edits
        total_sub += s
        total_ins += i
        total_del += d
        total_chars += len(ref_chars)
    cer = (total_edits / total_chars * 100.0) if total_chars else 0.0
    return {
        "cer": round(cer, 2),
        "total_edits": total_edits,
        "total_ref_chars": total_chars,
        "substitutions": total_sub,
        "insertions": total_ins,
        "deletions": total_del,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


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


def to_16k_waveform(audio: dict[str, Any]) -> np.ndarray:
    import librosa

    waveform = np.asarray(audio["array"], dtype=np.float32)
    sample_rate = int(audio["sampling_rate"])
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=0)
    if sample_rate != 16000:
        waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=16000)
    return np.asarray(waveform, dtype=np.float32)


def materialize_eval_audio(test_split, workspace_dir: Path) -> list[str]:
    import soundfile as sf

    audio_dir = workspace_dir / "eval_audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for index, sample in enumerate(test_split):
        wav_path = audio_dir / f"{index:04d}.wav"
        waveform = to_16k_waveform(sample["audio"])
        sf.write(str(wav_path), waveform, 16000, format="WAV")
        paths.append(str(wav_path))
    return paths


def main() -> None:
    from datasets import load_dataset
    from huggingface_hub import HfApi
    from qwen_asr import Qwen3ASRModel
    import torch

    hf_token = require_env("HF_TOKEN")
    dataset_id = require_env("HF_DATASET_ID", default="JosueG/adja-tts-orpheus", required=False)
    model_id = require_env("MODEL_ID", default="JosueG/qwen3-adja-asr-0p6b", required=False)
    results_repo_id = require_env("RESULTS_REPO_ID")
    workspace_dir = Path(require_env("WORKSPACE_DIR", default="/tmp/qwen3_asr_eval", required=False)).resolve()
    seed = int(require_env("SEED", default="42", required=False))

    dataset = load_dataset(dataset_id, token=hf_token, split="train")
    test_split = create_splits(dataset, seed)["test"]
    references = [normalize_text(sample["text"]) for sample in test_split]
    audio_inputs = materialize_eval_audio(test_split, workspace_dir)

    model = Qwen3ASRModel.from_pretrained(
        model_id,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="cuda:0" if torch.cuda.is_available() else "cpu",
    )

    hypotheses: list[str] = []
    started = time.time()
    for index, audio in enumerate(audio_inputs, start=1):
        result = model.transcribe(audio=audio)[0]
        hypotheses.append(normalize_text(result.text if hasattr(result, "text") else str(result)))
        if index % 20 == 0:
            print(f"Decoded {index}/{len(audio_inputs)} test utterances")

    elapsed = time.time() - started
    raw_wer = compute_word_metrics(references, hypotheses, normalize=False)
    raw_cer = compute_char_metrics(references, hypotheses, normalize=False)
    norm_wer = compute_word_metrics(references, hypotheses, normalize=True)
    norm_cer = compute_char_metrics(references, hypotheses, normalize=True)

    preview = []
    for idx in range(min(10, len(references))):
        preview.append(
            {
                "index": idx,
                "reference": references[idx],
                "prediction": hypotheses[idx],
            }
        )

    summary = {
        "dataset_id": dataset_id,
        "model_id": model_id,
        "seed": seed,
        "num_test_utterances": len(references),
        "elapsed_sec": round(elapsed, 2),
        "raw": {
            "wer": raw_wer["wer"],
            "cer": raw_cer["cer"],
            "wer_details": raw_wer,
            "cer_details": raw_cer,
        },
        "normalized": {
            "wer": norm_wer["wer"],
            "cer": norm_cer["cer"],
            "wer_details": norm_wer,
            "cer_details": norm_cer,
        },
        "preview": preview,
    }

    metrics_path = workspace_dir / "artifacts" / "qwen3_asr_eval" / "metrics.json"
    write_json(metrics_path, summary)

    api = HfApi(token=hf_token)
    ensure_repo(api, results_repo_id)
    upload_file_if_exists(api, results_repo_id, metrics_path, "qwen3_asr_adja/eval_test/metrics.json")

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
