#!/usr/bin/env python3
"""
ASR Inference Test
==================
Test your fine-tuned Qwen3-ASR model by transcribing audio.

Usage:
    python test_asr.py \
        --model_path ./asr_output/checkpoint-200 \
        --audio_path test_audio.wav

    # Batch test with a directory:
    python test_asr.py \
        --model_path ./asr_output/checkpoint-200 \
        --audio_dir ./test_audios/

    # Compare with base model:
    python test_asr.py \
        --model_path Qwen/Qwen3-ASR-1.7B \
        --audio_path test_audio.wav
"""

import argparse
import glob
import os

import torch


def main():
    parser = argparse.ArgumentParser(description="Test Qwen3-ASR inference")
    parser.add_argument(
        "--model_path", required=True,
        help="Path to fine-tuned checkpoint or HF model ID"
    )
    parser.add_argument("--audio_path", default=None, help="Single audio file to transcribe")
    parser.add_argument("--audio_dir", default=None, help="Directory of audio files to transcribe")
    parser.add_argument("--device", default="cuda:0", help="Device")
    args = parser.parse_args()

    if not args.audio_path and not args.audio_dir:
        print("ERROR: Provide --audio_path or --audio_dir")
        return

    print(f"Loading model: {args.model_path}")
    from qwen_asr import Qwen3ASRModel

    model = Qwen3ASRModel.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        device_map=args.device,
    )

    # Collect audio files
    audio_files = []
    if args.audio_path:
        audio_files.append(args.audio_path)
    if args.audio_dir:
        for ext in ("*.wav", "*.mp3", "*.flac", "*.ogg"):
            audio_files.extend(glob.glob(os.path.join(args.audio_dir, ext)))

    print(f"Transcribing {len(audio_files)} file(s)...")
    print("-" * 60)

    for audio_path in sorted(audio_files):
        results = model.transcribe(audio=audio_path)

        filename = os.path.basename(audio_path)
        lang = results[0].language if hasattr(results[0], "language") else "?"
        text = results[0].text if hasattr(results[0], "text") else str(results[0])

        print(f"[{filename}] ({lang}) {text}")

    print("-" * 60)
    print("Done!")


if __name__ == "__main__":
    main()
