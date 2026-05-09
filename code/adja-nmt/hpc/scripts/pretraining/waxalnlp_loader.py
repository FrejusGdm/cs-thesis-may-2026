#!/usr/bin/env python3
from __future__ import annotations

from typing import Optional, Sequence


def get_waxal_features(*, dataset: str, config: str, token: Optional[str], cache_dir: str):
    """
    Return the declared Features for a WaxalNLP config (e.g., ewe_asr / ewe_tts).

    We use the builder metadata as the source of truth, then optionally extend it
    in load_waxal_split() to tolerate stray Pandas index columns in Parquet.
    """
    from datasets import load_dataset_builder

    builder = load_dataset_builder(
        dataset,
        name=config,
        cache_dir=cache_dir,
        token=token,
    )
    return builder.info.features


def safe_load_audio_array(audio, *, target_sr: int):
    """
    Decode an HF Audio example into a 1D float32 waveform at target_sr.

    This is intentionally more robust than datasets' built-in Audio decode on HPC
    because some WaxalNLP `unlabeled` audio blobs can fail libsndfile decoding.

    Returns None when decoding fails.
    """
    import io

    import numpy as np

    if audio is None:
        return None

    # If the dataset already decoded it for us, honor it.
    if isinstance(audio, dict) and audio.get("array") is not None:
        arr = np.asarray(audio["array"], dtype=np.float32)
        sr = int(audio.get("sampling_rate") or target_sr)
        if sr != target_sr:
            import librosa

            arr = librosa.resample(arr, orig_sr=sr, target_sr=target_sr)
        return arr

    if not isinstance(audio, dict):
        return None

    import os

    path = audio.get("path")
    raw = audio.get("bytes")

    # Prefer file path decode (can fall back to audioread via librosa).
    if path and os.path.exists(path):
        try:
            import soundfile as sf

            data, sr = sf.read(path, dtype="float32", always_2d=False)
            if hasattr(data, "ndim") and data.ndim > 1:
                data = np.asarray(data, dtype=np.float32).mean(axis=-1)
            else:
                data = np.asarray(data, dtype=np.float32)
            if int(sr) != target_sr:
                import librosa

                data = librosa.resample(data, orig_sr=int(sr), target_sr=target_sr)
            return np.asarray(data, dtype=np.float32)
        except Exception:
            try:
                import librosa

                data, _ = librosa.load(path, sr=target_sr, mono=True)
                return np.asarray(data, dtype=np.float32)
            except Exception:
                return None

    # If a path string exists but isn't a real file (WaxalNLP bytes-backed MP3),
    # fall through to the bytes decoder below.

    # Bytes decode (WaxalNLP unlabeled frequently uses bytes-backed MP3 with
    # non-existent paths). Prefer ffmpeg pipe to avoid temp files.
    if raw:
        # First try libsndfile; it can handle wav/flac/ogg but usually not mp3.
        try:
            import soundfile as sf

            data, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
            data = np.asarray(data, dtype=np.float32)
            if data.ndim > 1:
                data = data.mean(axis=-1)
            if int(sr) != target_sr:
                import librosa

                data = librosa.resample(data, orig_sr=int(sr), target_sr=target_sr)
            return np.asarray(data, dtype=np.float32)
        except Exception:
            pass

        # ffmpeg handles mp3 robustly. Decode to raw float32 PCM mono at target_sr.
        try:
            import subprocess

            proc = subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    "pipe:0",
                    "-vn",
                    "-ac",
                    "1",
                    "-ar",
                    str(int(target_sr)),
                    "-f",
                    "f32le",
                    "pipe:1",
                ],
                input=raw,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            if proc.returncode != 0 or not proc.stdout:
                return None
            data = np.frombuffer(proc.stdout, dtype=np.float32)
            if data.size == 0:
                return None
            return data
        except Exception:
            return None

    return None


def _is_schema_cast_error(exc: BaseException) -> bool:
    msg = str(exc)
    if "column names don't match" in msg:
        return True
    if "Couldn't cast" in msg:
        return True
    try:
        from datasets.table import CastError  # type: ignore
    except Exception:
        CastError = None  # type: ignore
    if CastError is not None and isinstance(exc, CastError):
        return True
    cause = getattr(exc, "__cause__", None)
    return bool(cause) and _is_schema_cast_error(cause)


def load_waxal_split(
    *,
    dataset: str = "google/WaxalNLP",
    config: str,
    split: str,
    cache_dir: str,
    token: Optional[str],
):
    """
    Load a split from WaxalNLP robustly.

    WaxalNLP shards can include a Pandas index column `__index_level_0__` that is
    not present in the dataset's declared Features. This triggers a CastError in
    `datasets` when generating the split. We first try to load with an augmented
    Features schema that includes the index column, and fall back to the builder
    Features when the error isn't a schema mismatch (or upstream removes it).
    """
    from datasets import Features, Value, load_dataset

    builder_features = get_waxal_features(
        dataset=dataset,
        config=config,
        token=token,
        cache_dir=cache_dir,
    )

    # Copy into a new Features object so we never mutate builder_features.
    plus_index = Features(dict(builder_features))
    if "__index_level_0__" not in plus_index:
        plus_index["__index_level_0__"] = Value("int64")

    try:
        ds = load_dataset(
            dataset,
            name=config,
            split=split,
            cache_dir=cache_dir,
            token=token,
            features=plus_index,
        )
    except Exception as exc:
        # Only retry when we are confident it's the schema mismatch issue.
        if not _is_schema_cast_error(exc):
            raise
        ds = load_dataset(
            dataset,
            name=config,
            split=split,
            cache_dir=cache_dir,
            token=token,
            features=builder_features,
        )

    if "__index_level_0__" in getattr(ds, "column_names", []):
        ds = ds.remove_columns("__index_level_0__")
    return ds


def detect_transcript_col(ds, candidates: Sequence[str] = ("transcription", "text", "sentence")) -> str:
    cols = list(getattr(ds, "column_names", []))
    for c in candidates:
        if c in cols:
            print(f"[WaxalNLP] transcript column: '{c}' (candidates={list(candidates)})")
            return c
    raise ValueError(f"Could not find transcript column in {cols}. Tried: {list(candidates)}")
