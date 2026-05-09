#!/usr/bin/env python3
"""
TTS Inference Test
==================
Test your fine-tuned Qwen3-TTS model by generating speech.

Usage:
    python test_tts.py \
        --model_path ./tts_output/best \
        --speaker_name my_speaker \
        --text "Hello, this is a test of my fine-tuned voice." \
        --output_wav test_output.wav

    # Compare with base model:
    python test_tts.py \
        --model_path Qwen/Qwen3-TTS-12Hz-0.6B-Base \
        --text "Hello, this is the base model." \
        --output_wav base_output.wav \
        --clone_ref path/to/reference.wav
"""

import argparse
import os

import soundfile as sf
import torch


def main():
    parser = argparse.ArgumentParser(description="Test Qwen3-TTS inference")
    parser.add_argument(
        "--model_path", required=True,
        help="Path to fine-tuned checkpoint or HF model ID"
    )
    parser.add_argument("--text", required=True, help="Text to synthesize")
    parser.add_argument("--output_wav", default="tts_output.wav", help="Output WAV path")
    parser.add_argument("--speaker_name", default=None, help="Speaker name (for fine-tuned models)")
    parser.add_argument("--clone_ref", default=None, help="Reference audio for voice cloning (base model)")
    parser.add_argument("--device", default="cuda:0", help="Device")
    args = parser.parse_args()

    print(f"Loading model: {args.model_path}")
    from qwen_tts import Qwen3TTSModel

    # Determine attention implementation
    attn = "flash_attention_2"
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        attn = "eager"
        print("FlashAttention not available, using eager attention")

    model = Qwen3TTSModel.from_pretrained(
        args.model_path,
        device_map=args.device,
        dtype=torch.bfloat16,
        attn_implementation=attn,
    )

    print(f"Generating speech for: '{args.text}'")

    if args.speaker_name:
        # Fine-tuned model with custom voice
        wavs, sr = model.generate_custom_voice(
            text=args.text,
            speaker=args.speaker_name,
        )
    elif args.clone_ref:
        # Base model with voice cloning
        wavs, sr = model.generate_voice_clone(
            text=args.text,
            ref_audio=args.clone_ref,
        )
    else:
        print("WARNING: No --speaker_name or --clone_ref specified.")
        print("Using default generation (may not have a specific voice).")
        wavs, sr = model.generate(text=args.text)

    # Save output
    if isinstance(wavs, torch.Tensor):
        wavs = wavs.cpu().numpy()

    # Handle batch output (take first sample)
    if wavs.ndim > 1:
        wavs = wavs[0]

    sf.write(args.output_wav, wavs, sr)
    duration = len(wavs) / sr
    print(f"Saved: {args.output_wav} ({duration:.2f}s, {sr}Hz)")
    print("Done! Listen to the output to evaluate quality.")


if __name__ == "__main__":
    main()
