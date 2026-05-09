#!/usr/bin/env python3
from __future__ import annotations

"""
T2 canonical: Fine-tune Orpheus 3B TTS on Adja via HuggingFace Jobs.

Built on the Kinyarwanda-Orpheus Unsloth Colab notebook
(experiments/tts/T2_orpheus_finetune/T2_adja_orpheus_finetune_reference.py)
and shaped for HF Jobs using the T1 Sesame CSM canonical path as a template.

Why Orpheus after T1 plateaued at eval loss 6.488 with LoRA r=32:
    - 3.8B Llama backbone (vs CSM's 1B) — more capacity for new phoneme mappings
    - Canopy Labs publishes a pre-fine-tuned French variant (3b-fr-ft-research_release)
      and German (3b-de-ft-research_release). Reddit anecdote reports successful
      Kazakh adaptation via LoRA on ~350h + 70/30 target/English mix.
    - SNAC 24kHz audio codec tokenises speech into 7-token frames and the model
      generates them autoregressively like text — clean LM-style training loop.

Data-scale caveat: Adja is ~1.7h (~1277 utterances). Kazakh success used 350h.
If T2 plateaus like T1 did, the bottleneck is data, not capacity — pivot to
T6 (MMS-TTS-Ewe) per the hard-stop rule in T1's README.

Operational shape:
    - Container: pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel
    - GPU: L40S 48GB for LoRA (default); A100 80GB for --full-finetune
    - Vanilla HF (AutoModelForCausalLM + peft.LoraConfig) — Unsloth left as
      optional fallback. T1's README documents why vanilla is the stable default.
    - Loss is next-token cross-entropy over the combined text + SNAC audio token
      stream (there is no separate audio-head; Orpheus is just Llama with a
      vocabulary extended to include the SNAC codec tokens).

References:
    - Orpheus TTS: https://github.com/canopyai/Orpheus-TTS
    - SNAC codec: https://github.com/hubertsiuzdak/snac (arXiv:2410.00037)
    - LoRA: https://arxiv.org/abs/2106.09685
    - French Orpheus variant: https://huggingface.co/canopylabs/3b-fr-ft-research_release
    - Upstream Unsloth notebook: references/unsloth-tts-notebooks/Orpheus_3B_TTS.ipynb
    - Reddit precedent (Kazakh 350h, Finnish 7000h): r/LocalLLaMA Aug 2025 thread

For a 2-minute sanity check before submitting a long run, use
scripts/hf_jobs/T2_orpheus_vanilla.py.
"""

import argparse
import json
import os
import random
import subprocess
import sys
import time
import unicodedata
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)


# Orpheus token layout (from the upstream Unsloth notebook, lines 297-311).
# Reserved slots inside the extended Llama vocabulary.
TOKENISER_LENGTH = 128256
START_OF_TEXT = 128000
END_OF_TEXT = 128009
START_OF_SPEECH = TOKENISER_LENGTH + 1    # 128257
END_OF_SPEECH = TOKENISER_LENGTH + 2      # 128258
START_OF_HUMAN = TOKENISER_LENGTH + 3     # 128259
END_OF_HUMAN = TOKENISER_LENGTH + 4       # 128260
START_OF_AI = TOKENISER_LENGTH + 5        # 128261
END_OF_AI = TOKENISER_LENGTH + 6          # 128262
PAD_TOKEN = TOKENISER_LENGTH + 7          # 128263
AUDIO_TOKENS_START = TOKENISER_LENGTH + 10  # 128266


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="T2: full Orpheus 3B training for Adja TTS")
    parser.add_argument(
        "--base-model",
        default="canopylabs/orpheus-3b-0.1-ft",
        help="English (default), French: canopylabs/3b-fr-ft-research_release, "
        "German: canopylabs/3b-de-ft-research_release",
    )
    parser.add_argument("--dataset", default="JosueG/adja-tts-orpheus", help="Private HF dataset id")
    parser.add_argument("--output-dir", default="/tmp/orpheus_adja_full", help="Local output directory")
    parser.add_argument("--seed", type=int, default=42, help="Split + training seed (matches rest of repo)")
    parser.add_argument("--max-steps", type=int, default=-1, help="Hard step cap (default -1 = use --num-epochs)")
    parser.add_argument("--num-epochs", type=int, default=20, help="Max epochs (early stopping usually fires first)")
    parser.add_argument("--batch-size", type=int, default=1, help="Per-device batch size")
    parser.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation steps (effective batch = bs * grad_accum)")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="LR (2e-4 LoRA / 5e-5 full fine-tune)")
    parser.add_argument("--lora-r", type=int, default=64, help="LoRA rank (ignored if --full-finetune). 64 > T1's 32 because backbone is 3.8x bigger; 512 from upstream notebook is overkill.")
    parser.add_argument("--full-finetune", action="store_true", help="Skip LoRA, update all 3.8B params. Requires A100 80GB.")
    parser.add_argument("--eval-steps", type=int, default=50, help="Evaluate on dev set every N steps")
    parser.add_argument("--early-stopping-patience", type=int, default=5, help="Stop if eval_loss doesn't improve for N evals")
    parser.add_argument("--max-seq-length", type=int, default=2048, help="Orpheus max context (matches upstream notebook)")
    parser.add_argument("--dry-run", action="store_true", help="5 samples, 2 steps, single eval — matches T1's pipeline-validation discipline")
    parser.add_argument("--push-to-hub", action="store_true", help="Upload metrics, adapter, and audio samples")
    parser.add_argument("--results-repo", default="JosueG/adja-tts-results", help="HF repo for outputs")
    parser.add_argument("--results-prefix", default="T2_orpheus", help="Subdirectory inside the results repo")
    return parser.parse_args()


def pip_install(packages: list[str], no_deps: bool = False) -> None:
    cmd = [sys.executable, "-m", "pip", "install", "-q"]
    if no_deps:
        cmd.append("--no-deps")
    cmd.extend(packages)
    subprocess.check_call(cmd)


def install_env() -> None:
    # Core pins come from the upstream Unsloth Orpheus notebook. Kept as exact
    # versions because Orpheus's dtype/tokeniser assumptions move with
    # transformers releases.
    pip_install(
        [
            "transformers==4.56.2",
            "peft>=0.11.0,<0.16.0",
            "accelerate",
            "datasets>=3.4.1,<4.0.0",
            "huggingface_hub>=0.34.0",
            "hf_transfer",
            "soundfile",
            "librosa",
            "numpy",
            "scipy",
            "sentencepiece",
            "protobuf",
            "bitsandbytes",
            "torchaudio",
        ]
    )
    pip_install(["trl==0.22.2"], no_deps=True)
    pip_install(["snac"])


def normalize_text(text: str) -> str:
    """NFC normalize — critical for Adja tone marks (ɛ, ɔ, ŋ, ɖ, é, è)."""
    return " ".join(unicodedata.normalize("NFC", text.strip()).split())


def build_snac_codes(audio_array, snac_model, orig_sr: int, device: str) -> list[int] | None:
    """
    Encode a single waveform into flat Orpheus SNAC token ids.

    Output convention (7 tokens per SNAC frame, interleaved across the 3 codec
    layers, matches the upstream notebook lines 248-276):
        [l1, l2+4096, l3+2*4096, l3+3*4096, l2+4*4096, l3+5*4096, l3+6*4096]
    All ids are offset by AUDIO_TOKENS_START so they land in the reserved part
    of the extended Llama vocabulary.
    """
    import numpy as np
    import torch
    import torchaudio.transforms as T

    if isinstance(audio_array, list):
        waveform = np.asarray(audio_array, dtype=np.float32)
    else:
        waveform = np.asarray(audio_array, dtype=np.float32)

    tensor = torch.from_numpy(waveform).unsqueeze(0).to(dtype=torch.float32)
    if orig_sr != 24000:
        tensor = T.Resample(orig_freq=orig_sr, new_freq=24000)(tensor)
    tensor = tensor.unsqueeze(0).to(device)

    with torch.inference_mode():
        codes = snac_model.encode(tensor)

    flat: list[int] = []
    n_frames = codes[0].shape[1]
    for i in range(n_frames):
        flat.append(codes[0][0][i].item() + AUDIO_TOKENS_START)
        flat.append(codes[1][0][2 * i].item() + AUDIO_TOKENS_START + 4096)
        flat.append(codes[2][0][4 * i].item() + AUDIO_TOKENS_START + 2 * 4096)
        flat.append(codes[2][0][4 * i + 1].item() + AUDIO_TOKENS_START + 3 * 4096)
        flat.append(codes[1][0][2 * i + 1].item() + AUDIO_TOKENS_START + 4 * 4096)
        flat.append(codes[2][0][4 * i + 2].item() + AUDIO_TOKENS_START + 5 * 4096)
        flat.append(codes[2][0][4 * i + 3].item() + AUDIO_TOKENS_START + 6 * 4096)
    return flat


def dedupe_frames(codes: list[int]) -> list[int]:
    """Drop consecutive frames that start with the same layer-1 token. From upstream notebook lines 316-336."""
    if len(codes) % 7 != 0:
        raise ValueError(f"SNAC code list length {len(codes)} is not divisible by 7")
    kept = codes[:7]
    for i in range(7, len(codes), 7):
        if codes[i] != kept[-7]:
            kept.extend(codes[i : i + 7])
    return kept


def redistribute_codes(flat: list[int]):
    """Undo build_snac_codes — split 7-per-frame stream back into SNAC's 3 layers for decoding."""
    import torch
    layer_1, layer_2, layer_3 = [], [], []
    for i in range((len(flat) + 1) // 7):
        base = 7 * i
        layer_1.append(flat[base] - AUDIO_TOKENS_START)
        layer_2.append(flat[base + 1] - AUDIO_TOKENS_START - 4096)
        layer_3.append(flat[base + 2] - AUDIO_TOKENS_START - 2 * 4096)
        layer_3.append(flat[base + 3] - AUDIO_TOKENS_START - 3 * 4096)
        layer_2.append(flat[base + 4] - AUDIO_TOKENS_START - 4 * 4096)
        layer_3.append(flat[base + 5] - AUDIO_TOKENS_START - 5 * 4096)
        layer_3.append(flat[base + 6] - AUDIO_TOKENS_START - 6 * 4096)
    return [
        torch.tensor(layer_1).unsqueeze(0),
        torch.tensor(layer_2).unsqueeze(0),
        torch.tensor(layer_3).unsqueeze(0),
    ]


def main() -> None:
    args = parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is required")

    print("Installing environment...")
    install_env()
    print("Environment ready.\n")

    import numpy as np
    import soundfile as sf
    import torch
    from datasets import Audio, load_dataset
    from huggingface_hub import HfApi
    from peft import LoraConfig, get_peft_model
    from snac import SNAC
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    print("=" * 60)
    print("T2 CANONICAL: Orpheus 3B TTS on Adja")
    print("=" * 60)
    print(f"Base model: {args.base_model}")
    print(f"Dataset: {args.dataset}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Seed: {args.seed}")
    mode = "FULL fine-tune" if args.full_finetune else f"LoRA r={args.lora_r}"
    print(f"Mode: {mode}")
    if args.dry_run:
        print("** DRY-RUN: 5 samples, 2 steps, single eval **")
    steps_desc = f"{args.max_steps} hard-capped" if args.max_steps > 0 else f"up to {args.num_epochs} epochs"
    print(f"Training: {steps_desc}, batch {args.batch_size} x grad_accum {args.grad_accum} = {args.batch_size * args.grad_accum} effective")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Eval every {args.eval_steps} steps, early stop patience {args.early_stopping_patience}")
    print()

    # ===== STAGE 1: Dataset =====
    print("===== STAGE 1: Dataset =====")
    ds = load_dataset(args.dataset, token=token, split="train")
    # 80/10/10 split, seed=42 — matches the rest of the repo (CLAUDE.md + T1).
    split1 = ds.train_test_split(test_size=0.1, seed=args.seed)
    split2 = split1["train"].train_test_split(test_size=0.1 / 0.9, seed=args.seed)
    train_ds = split2["train"]
    dev_ds = split2["test"]
    test_ds = split1["test"]
    orig_sr = train_ds[0]["audio"]["sampling_rate"]
    print(f"Original audio sampling rate: {orig_sr} Hz (will resample to 24000)")
    # Keep raw sampling rate until SNAC encoding — the resample runs inside build_snac_codes
    # so we avoid double-resampling via cast_column.
    if args.dry_run:
        train_ds = train_ds.select(range(min(5, len(train_ds))))
        dev_ds = dev_ds.select(range(min(2, len(dev_ds))))
        test_ds = test_ds.select(range(min(2, len(test_ds))))
    print(f"Train samples: {len(train_ds)}")
    print(f"Dev samples: {len(dev_ds)}")
    print(f"Test samples: {len(test_ds)}\n")

    # ===== STAGE 2: Model + tokenizer =====
    # Base model kept in fp32 during load (Trainer handles bf16 via autocast) —
    # same pattern as T1's canonical CSM script, avoids dtype mismatches during
    # index_put_-style ops in the codec-token forward path.
    print("===== STAGE 2: Model =====")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, token=token)
    # Llama tokenizer ships without a pad token. Point `pad_token` at EOS for
    # DataCollatorForLanguageModeling (which inspects the string attribute),
    # then override the numeric id to Orpheus's reserved pad slot (128263).
    # Pad positions are masked to -100 in labels, so the string representation
    # never matters for loss — only the id does, and Orpheus never generates PAD.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = PAD_TOKEN
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float32,
        token=token,
    ).to("cuda")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Base model params: {total_params / 1e6:.1f}M")

    if args.full_finetune:
        training_mode = "full_finetune"
        print("Mode: FULL fine-tune (all 3.8B params trainable, no LoRA)")
    else:
        training_mode = f"lora_r{args.lora_r}"
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_r,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable: {trainable / 1e6:.2f}M / {total_params / 1e6:.1f}M ({100 * trainable / total_params:.2f}%)\n")

    # ===== STAGE 3: SNAC encode + tokenize =====
    # Orpheus training is single-sequence: [SOH] <text_ids> [EOT] [EOH] [SOA] [SOS] <snac_codes> [EOS] [EOA]
    # Labels = input_ids (causal LM). Loss is cross-entropy over the whole
    # sequence — the model learns to emit SNAC codes *conditional on the text*.
    print("===== STAGE 3: SNAC encode + tokenize =====")
    snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to("cuda")
    snac_model.eval()

    dropped = {"empty_text": 0, "short_audio": 0, "encode_fail": 0, "too_long": 0}

    def preprocess_example(example: dict) -> dict | None:
        text = normalize_text(example.get("text") or "")
        if not text:
            dropped["empty_text"] += 1
            return None
        audio_payload = example.get("audio") or {}
        audio_array = audio_payload.get("array")
        if audio_array is None or len(audio_array) < 10000:
            dropped["short_audio"] += 1
            return None
        try:
            flat_codes = build_snac_codes(audio_array, snac_model, orig_sr, device="cuda")
        except Exception as exc:
            print(f"  Skip '{text[:40]}': encode failed ({exc})")
            dropped["encode_fail"] += 1
            return None
        if not flat_codes:
            dropped["encode_fail"] += 1
            return None
        flat_codes = dedupe_frames(flat_codes)

        text_ids = tokenizer.encode(text, add_special_tokens=True)
        text_ids.append(END_OF_TEXT)
        input_ids = (
            [START_OF_HUMAN]
            + text_ids
            + [END_OF_HUMAN]
            + [START_OF_AI]
            + [START_OF_SPEECH]
            + flat_codes
            + [END_OF_SPEECH]
            + [END_OF_AI]
        )
        if len(input_ids) > args.max_seq_length:
            dropped["too_long"] += 1
            return None
        return {
            "input_ids": input_ids,
            "labels": input_ids,
            "attention_mask": [1] * len(input_ids),
        }

    def process(split, name: str):
        out = split.map(
            preprocess_example,
            remove_columns=split.column_names,
            desc=f"Preprocessing {name}",
        )
        # HF Datasets replaces None with an all-None row; filter those out.
        out = out.filter(lambda ex: ex.get("input_ids") is not None)
        return out

    processed_train = process(train_ds, "train")
    processed_dev = process(dev_ds, "dev")

    print(f"Preprocessed: train={len(processed_train)} dev={len(processed_dev)}")
    print(f"Dropped — empty_text:{dropped['empty_text']} short_audio:{dropped['short_audio']} encode_fail:{dropped['encode_fail']} too_long:{dropped['too_long']}\n")

    # Free SNAC VRAM before model training (move to CPU; we'll move back for
    # generation). Keeps headroom on L40S for the 3.8B backbone.
    snac_model.to("cpu")
    torch.cuda.empty_cache()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ===== STAGE 4: Train =====
    print("===== STAGE 4: Train =====")
    # DataCollatorForLanguageModeling with mlm=False handles left-padding and
    # aligns labels with input_ids while masking pad positions (-100).
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    if args.dry_run:
        eval_steps = 2
        max_steps = 2
        num_epochs = 1
        patience = 1
    else:
        eval_steps = args.eval_steps
        max_steps = args.max_steps
        num_epochs = args.num_epochs
        patience = args.early_stopping_patience

    training_args = TrainingArguments(
        output_dir=str(output_dir / "trainer_output"),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_steps=10,
        num_train_epochs=num_epochs,
        max_steps=max_steps,  # -1 means use num_train_epochs
        learning_rate=args.learning_rate,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=5,
        optim="adamw_torch" if args.full_finetune else "adamw_8bit",
        weight_decay=0.001,
        lr_scheduler_type="cosine",
        seed=args.seed,
        report_to="none",
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False,
        # See T1 fix #3 in T1_sesame_csm_finetune/README.md: PeftModel hides the
        # base forward signature, so Trainer needs explicit label_names.
        label_names=["labels"],
        # gradient_checkpointing OFF — T1 fix #2 (PEFT + checkpointing breaks grad flow).
    )

    trainer = Trainer(
        model=model,
        train_dataset=processed_train,
        eval_dataset=processed_dev,
        data_collator=collator,
        args=training_args,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)],
    )

    t0 = time.time()
    trainer_stats = trainer.train()
    elapsed = time.time() - t0
    train_loss = float(trainer_stats.metrics.get("train_loss", 0.0))
    peak_mem = round(torch.cuda.max_memory_reserved() / 1e9, 2)

    eval_history = [log for log in trainer.state.log_history if "eval_loss" in log]
    best_eval_loss = min((log["eval_loss"] for log in eval_history), default=None)
    final_eval = trainer.evaluate()
    final_eval_loss = float(final_eval.get("eval_loss", 0.0))

    print(f"\nTraining complete in {elapsed / 60:.1f} min")
    print(f"Train loss: {train_loss:.4f}")
    if best_eval_loss is not None:
        print(f"Best dev loss: {best_eval_loss:.4f}")
    else:
        print("No eval history")
    print(f"Final dev loss: {final_eval_loss:.4f} (model reloaded at best checkpoint)")
    print(f"Peak VRAM: {peak_mem} GB\n")

    # ===== STAGE 5: Generate =====
    # Move SNAC back to GPU (or keep on CPU — SNAC decode is small, CPU is fine).
    print("===== STAGE 5: Generate =====")
    generated_dir = output_dir / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    snac_model.to("cpu")

    prompt_pool: list[str] = []
    seen: set[str] = set()
    for i in range(min(5, len(test_ds))):
        candidate = normalize_text(test_ds[i]["text"])
        if candidate and candidate not in seen:
            seen.add(candidate)
            prompt_pool.append(candidate)

    generated: list[dict] = []

    start_tok = torch.tensor([[START_OF_HUMAN]], dtype=torch.int64)
    end_toks = torch.tensor([[END_OF_TEXT, END_OF_HUMAN]], dtype=torch.int64)

    for idx, text in enumerate(prompt_pool):
        try:
            text_ids = tokenizer(text, return_tensors="pt").input_ids
            prompt_ids = torch.cat([start_tok, text_ids, end_toks], dim=1).to(model.device)
            attn = torch.ones_like(prompt_ids)
            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids=prompt_ids,
                    attention_mask=attn,
                    max_new_tokens=1200,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.95,
                    repetition_penalty=1.1,
                    num_return_sequences=1,
                    eos_token_id=END_OF_SPEECH,
                    use_cache=True,
                )
            # Take tokens after the last START_OF_SPEECH marker, drop END_OF_SPEECH padding.
            row = generated_ids[0].tolist()
            if START_OF_SPEECH in row:
                row = row[len(row) - 1 - row[::-1].index(START_OF_SPEECH):][1:]
            row = [t for t in row if t != END_OF_SPEECH]
            row = row[: (len(row) // 7) * 7]  # trim to whole 7-token frames
            if not row:
                raise RuntimeError("generation produced no audio frames")
            codes = redistribute_codes(row)
            with torch.inference_mode():
                waveform = snac_model.decode(codes)
            audio = waveform.detach().squeeze().cpu().numpy().astype(np.float32)
            if audio.size == 0:
                raise RuntimeError("empty waveform")
            path = generated_dir / f"plain_{idx:02d}.wav"
            sf.write(str(path), audio, 24000)
            generated.append(
                {
                    "type": "plain",
                    "text": text,
                    "file": path.name,
                    "duration_sec": round(float(audio.size) / 24000, 2),
                }
            )
            print(f"  plain[{idx}]: '{text[:40]}' -> {audio.size / 24000:.1f}s")
        except Exception as exc:
            print(f"  plain[{idx}] FAILED: {exc}")
            generated.append({"type": "plain", "text": text, "error": str(exc)})

    # ===== STAGE 6: Save adapter + metrics =====
    adapter_dir = output_dir / "adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    results = {
        "experiment": "T2",
        "path": "vanilla_orpheus_peft",
        "base_model": args.base_model,
        "training_mode": training_mode,
        "seed": args.seed,
        "dataset": args.dataset,
        "max_steps": max_steps,
        "num_epochs": num_epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "effective_batch_size": args.batch_size * args.grad_accum,
        "learning_rate": args.learning_rate,
        "lora_r": None if args.full_finetune else args.lora_r,
        "eval_steps": eval_steps,
        "early_stopping_patience": patience,
        "train_loss": round(train_loss, 4),
        "best_eval_loss": round(best_eval_loss, 4) if best_eval_loss is not None else None,
        "final_eval_loss": round(final_eval_loss, 4),
        "global_step_at_stop": trainer.state.global_step,
        "epochs_completed": round(trainer.state.epoch, 2) if trainer.state.epoch is not None else None,
        "training_time_min": round(elapsed / 60, 1),
        "peak_vram_gb": peak_mem,
        "trainable_params_M": round(trainable / 1e6, 2),
        "total_params_M": round(total_params / 1e6, 1),
        "n_train_samples": len(processed_train),
        "n_dev_samples": len(processed_dev),
        "n_test_samples": len(test_ds),
        "dropped": dropped,
        "gpu": torch.cuda.get_device_name(0),
        "generated": generated,
        "eval_history": eval_history,
        "dry_run": args.dry_run,
    }

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nSaved metrics to {metrics_path}")

    # ===== STAGE 7: Push to Hub =====
    if args.push_to_hub:
        print("\n===== STAGE 7: Push to Hub =====")
        api = HfApi(token=token)
        api.create_repo(args.results_repo, private=True, exist_ok=True)
        prefix = args.results_prefix.rstrip("/")
        api.upload_file(
            path_or_fileobj=metrics_path.read_bytes(),
            path_in_repo=f"{prefix}/metrics.json",
            repo_id=args.results_repo,
            token=token,
        )
        api.upload_folder(
            folder_path=str(adapter_dir),
            path_in_repo=f"{prefix}/adapter",
            repo_id=args.results_repo,
            token=token,
        )
        api.upload_folder(
            folder_path=str(generated_dir),
            path_in_repo=f"{prefix}/generated_audio",
            repo_id=args.results_repo,
            token=token,
        )
        print(f"Uploaded to https://huggingface.co/{args.results_repo} under {prefix}/")

    print("\nT2 canonical training finished.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        raise
