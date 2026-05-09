#!/usr/bin/env python3
"""
Stage 2: Audio Processing
=========================
Converts downloaded audio to the exact format required by Qwen3-TTS / Qwen3-ASR.

Processing steps:
1. Convert to WAV (PCM 16-bit or 32-bit float)
2. Resample to target sample rate (24kHz for TTS, 16kHz for ASR)
3. Convert to mono
4. Trim leading/trailing silence
5. Normalize volume (peak normalization)
6. Filter by duration (remove too short / too long)
7. Generate quality report

Input:  JSONL from Stage 1 (with "audio" paths)
Output: JSONL with processed audio paths + quality stats

Usage:
    # For TTS:
    python process_audio.py \
        --input_jsonl raw_data.jsonl \
        --output_dir ./processed_audio_tts \
        --output_jsonl processed_tts.jsonl \
        --target_sr 24000 \
        --min_duration 0.5 \
        --max_duration 30.0

    # For ASR:
    python process_audio.py \
        --input_jsonl raw_data.jsonl \
        --output_dir ./processed_audio_asr \
        --output_jsonl processed_asr.jsonl \
        --target_sr 16000 \
        --min_duration 0.5 \
        --max_duration 300.0
"""

import argparse
import json
import logging
import os
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Audio processing functions
# ---------------------------------------------------------------------------

def load_audio(filepath: str, target_sr: int) -> tuple[np.ndarray, int]:
    """Load audio file and resample to target sample rate.

    librosa.load() handles:
    - Any format ffmpeg can read (wav, mp3, flac, ogg, m4a, etc.)
    - Automatic mono conversion
    - Automatic resampling
    """
    audio, sr = librosa.load(filepath, sr=target_sr, mono=True)
    return audio, sr


def trim_silence(
    audio: np.ndarray,
    sr: int,
    top_db: int = 25,
    frame_length: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """Remove leading and trailing silence.

    top_db controls the threshold — audio below this many dB from the
    peak is considered silence. 25 dB is a good default that removes
    obvious silence without cutting speech.

    WHY THIS MATTERS: Silence at the start/end wastes model capacity.
    The model would learn to predict long runs of zero-energy tokens,
    which doesn't help it learn your language's phonology.
    """
    trimmed, _ = librosa.effects.trim(
        audio,
        top_db=top_db,
        frame_length=frame_length,
        hop_length=hop_length,
    )
    return trimmed


def normalize_volume(audio: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    """Peak-normalize audio to a target level.

    WHY THIS MATTERS: If your recordings have wildly different volumes,
    the model sees different energy patterns for the same phonemes.
    Normalization ensures consistent input, making training more stable.

    target_peak=0.95 leaves a tiny headroom to avoid clipping after any
    downstream processing (like mel-spectrogram extraction).
    """
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio * (target_peak / peak)
    return audio


def compute_audio_stats(audio: np.ndarray, sr: int) -> dict:
    """Compute quality metrics for an audio file."""
    duration = len(audio) / sr
    rms = float(np.sqrt(np.mean(audio ** 2)))
    peak = float(np.max(np.abs(audio)))
    # Simple signal-to-noise ratio estimate (ratio of RMS to silence floor)
    silence_floor = np.percentile(np.abs(audio), 5)
    snr_estimate = 20 * np.log10(rms / max(silence_floor, 1e-10))

    return {
        "duration_s": round(duration, 3),
        "rms": round(rms, 6),
        "peak": round(peak, 6),
        "snr_estimate_db": round(snr_estimate, 1),
    }


# ---------------------------------------------------------------------------
# Full processing pipeline
# ---------------------------------------------------------------------------

def process_single_audio(
    input_path: str,
    output_path: str,
    target_sr: int = 24000,
    trim: bool = True,
    normalize: bool = True,
    min_duration: float = 0.5,
    max_duration: float = 30.0,
) -> tuple[bool, dict]:
    """Process a single audio file.

    Returns:
        (accepted: bool, stats: dict)
    """
    try:
        # Load and resample
        audio, sr = load_audio(input_path, target_sr)

        # Trim silence
        if trim:
            audio = trim_silence(audio, sr)

        # Check duration
        duration = len(audio) / sr
        if duration < min_duration:
            return False, {"reason": f"too_short ({duration:.2f}s < {min_duration}s)"}
        if duration > max_duration:
            return False, {"reason": f"too_long ({duration:.2f}s > {max_duration}s)"}

        # Normalize volume
        if normalize:
            audio = normalize_volume(audio)

        # Compute quality stats
        stats = compute_audio_stats(audio, sr)

        # Save as WAV (PCM 16-bit for compatibility)
        sf.write(output_path, audio, sr, subtype="PCM_16")
        stats["output_path"] = output_path

        return True, stats

    except Exception as e:
        return False, {"reason": f"processing_error: {e}"}


def process_dataset(
    input_jsonl: str,
    output_dir: str,
    output_jsonl: str,
    target_sr: int = 24000,
    trim: bool = True,
    normalize: bool = True,
    min_duration: float = 0.5,
    max_duration: float = 30.0,
):
    """Process all audio files in a dataset JSONL."""
    os.makedirs(output_dir, exist_ok=True)

    # Load input manifest
    records = []
    with open(input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    logger.info(f"Processing {len(records)} audio files → {output_dir}")

    accepted = 0
    rejected = 0
    reject_reasons = {}
    all_stats = []

    with open(output_jsonl, "w", encoding="utf-8") as out_f:
        for i, rec in enumerate(records):
            input_path = rec["audio"]
            filename = f"processed_{i:06d}.wav"
            output_path = os.path.join(output_dir, filename)

            success, stats = process_single_audio(
                input_path=input_path,
                output_path=output_path,
                target_sr=target_sr,
                trim=trim,
                normalize=normalize,
                min_duration=min_duration,
                max_duration=max_duration,
            )

            if success:
                accepted += 1
                entry = {
                    "audio": os.path.abspath(output_path),
                    "text": rec["text"],
                    "duration_s": stats["duration_s"],
                }
                # Carry forward extra fields
                for k, v in rec.items():
                    if k not in ("audio", "text"):
                        entry[k] = v
                out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                all_stats.append(stats)
            else:
                rejected += 1
                reason = stats.get("reason", "unknown")
                reject_reasons[reason] = reject_reasons.get(reason, 0) + 1

            if (i + 1) % 100 == 0:
                logger.info(f"Progress: {i + 1}/{len(records)} (accepted={accepted})")

    # Print summary
    logger.info("=" * 60)
    logger.info("PROCESSING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total input:  {len(records)}")
    logger.info(f"Accepted:     {accepted}")
    logger.info(f"Rejected:     {rejected}")

    if reject_reasons:
        logger.info("Rejection reasons:")
        for reason, count in sorted(reject_reasons.items(), key=lambda x: -x[1]):
            logger.info(f"  {reason}: {count}")

    if all_stats:
        durations = [s["duration_s"] for s in all_stats]
        logger.info(f"Duration stats:")
        logger.info(f"  Min:    {min(durations):.2f}s")
        logger.info(f"  Max:    {max(durations):.2f}s")
        logger.info(f"  Mean:   {np.mean(durations):.2f}s")
        logger.info(f"  Total:  {sum(durations) / 3600:.2f} hours")

    # Save quality report
    report_path = os.path.join(output_dir, "quality_report.json")
    report = {
        "total_input": len(records),
        "accepted": accepted,
        "rejected": rejected,
        "reject_reasons": reject_reasons,
        "duration_stats": {
            "min_s": round(min(durations), 2) if durations else 0,
            "max_s": round(max(durations), 2) if durations else 0,
            "mean_s": round(float(np.mean(durations)), 2) if durations else 0,
            "total_hours": round(sum(durations) / 3600, 2) if durations else 0,
        },
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Quality report: {report_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Process audio files for Qwen3 TTS/ASR fine-tuning"
    )
    parser.add_argument("--input_jsonl", required=True, help="Input JSONL from download stage")
    parser.add_argument("--output_dir", required=True, help="Directory for processed audio")
    parser.add_argument("--output_jsonl", required=True, help="Output JSONL manifest")
    parser.add_argument(
        "--target_sr", type=int, default=24000,
        help="Target sample rate (24000 for TTS, 16000 for ASR)"
    )
    parser.add_argument("--min_duration", type=float, default=0.5, help="Min duration in seconds")
    parser.add_argument("--max_duration", type=float, default=30.0, help="Max duration in seconds")
    parser.add_argument("--no_trim", action="store_true", help="Skip silence trimming")
    parser.add_argument("--no_normalize", action="store_true", help="Skip volume normalization")
    args = parser.parse_args()

    process_dataset(
        input_jsonl=args.input_jsonl,
        output_dir=args.output_dir,
        output_jsonl=args.output_jsonl,
        target_sr=args.target_sr,
        trim=not args.no_trim,
        normalize=not args.no_normalize,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
    )


if __name__ == "__main__":
    main()
