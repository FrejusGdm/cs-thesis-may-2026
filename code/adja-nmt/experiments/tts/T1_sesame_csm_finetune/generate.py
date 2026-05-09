#!/usr/bin/env python3
from __future__ import annotations

"""
Waveform generation utility for a fine-tuned Sesame CSM adapter.

This is the notebook-accurate inference path:
- load `unsloth/csm-1b`
- attach the LoRA adapter with PEFT
- generate waveform audio with `model.generate(output_audio=True)`
- optionally use reference audio for speaker conditioning
"""

import argparse
import unicodedata
from pathlib import Path

import numpy as np
import soundfile as sf


def normalize_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text.strip()).split())


def linear_resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return audio.astype(np.float32)
    duration = len(audio) / orig_sr
    target_length = int(duration * target_sr)
    indices = np.linspace(0, len(audio) - 1, target_length)
    return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)


def load_audio(path: str, target_sr: int) -> np.ndarray:
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return linear_resample(audio.astype(np.float32), sr, target_sr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Adja speech with a fine-tuned CSM adapter")
    parser.add_argument("--adapter-dir", required=True, help="Path to the saved LoRA adapter directory")
    parser.add_argument("--base-model", default="unsloth/csm-1b", help="Base CSM model id")
    parser.add_argument("--text", required=True, help="Target text to synthesize")
    parser.add_argument("--output", default="output.wav", help="Output waveform path")
    parser.add_argument("--reference-audio", default=None, help="Optional reference waveform for speaker conditioning")
    parser.add_argument("--reference-text", default=None, help="Transcript for the reference audio")
    parser.add_argument("--max-new-tokens", type=int, default=125, help="Max audio tokens to generate")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoProcessor, CsmForConditionalGeneration

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for waveform generation")

    device = torch.device("cuda")
    target_sr = 24000
    text = normalize_text(args.text)

    print(f"Loading base model: {args.base_model}")
    base_model = CsmForConditionalGeneration.from_pretrained(
        args.base_model,
        torch_dtype=torch.float32,
    ).to(device)
    model = PeftModel.from_pretrained(base_model, args.adapter_dir).to(device)
    processor = AutoProcessor.from_pretrained(args.base_model)
    model.eval()

    if args.reference_audio:
        if not args.reference_text:
            raise SystemExit("--reference-text is required when --reference-audio is provided")
        ref_audio = load_audio(args.reference_audio, target_sr)
        conversation = [
            {
                "role": "0",
                "content": [
                    {"type": "text", "text": normalize_text(args.reference_text)},
                    {"type": "audio", "path": ref_audio},
                ],
            },
            {"role": "0", "content": [{"type": "text", "text": text}]},
        ]
        model_inputs = processor.apply_chat_template(
            conversation,
            tokenize=True,
            return_dict=True,
            common_kwargs={"return_tensors": "pt"},
        ).to(device)
    else:
        model_inputs = processor(
            f"[0]{text}",
            add_special_tokens=True,
            return_tensors="pt",
        ).to(device)

    print(f"Generating waveform for: {text}")
    with torch.no_grad():
        audio_values = model.generate(
            **model_inputs,
            max_new_tokens=args.max_new_tokens,
            output_audio=True,
        )

    audio = audio_values[0].to(torch.float32).cpu().numpy()
    if audio.size == 0:
        raise RuntimeError("Generation returned an empty waveform")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), audio, target_sr)
    print(f"Saved waveform to {output_path} ({len(audio) / target_sr:.2f}s)")


if __name__ == "__main__":
    main()
