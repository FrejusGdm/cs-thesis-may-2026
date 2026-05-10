#!/usr/bin/env python3
# /// script
# dependencies = ["torch==2.5.1", "torchaudio==2.5.1", "transformers==4.52.3", "peft>=0.11.0,<0.16.0", "accelerate", "datasets>=3.4.1,<4.0.0", "soundfile", "librosa", "numpy", "scipy", "huggingface-hub>=0.34.0", "hf_transfer", "sentencepiece", "protobuf", "torchcodec", "bitsandbytes"]
# ///
from __future__ import annotations
"""
T1-tokfix: Test the 'Llama BPE tokenizer is the bottleneck' hypothesis for CSM.

Last updated: 2026-04-21

Sesame CSM uses the Llama 3.2 BPE tokenizer, which byte-fragments Adja
diacritics (ɛ ɔ ŋ ɖ + tone marks). Hypothesis: expanding the tokenizer with
native Adja tokens + resizing embeddings lets the LM learn phoneme→audio
mapping from real characters, not byte fragments. Identical to T1 except
for the tokenizer surgery before training.

If this beats vanilla T1 dev loss → the tokenizer was the culprit and we can
recover CSM. If it does not beat T1 → some other layer is also broken.

References:
  - tokenizer fragmentation diagnosis: learnings-from-the-past/training-gotchas.md
  - Three-layer TTS architecture: ideas/tts-models-guide.md
  - Sesame CSM: https://github.com/SesameAILabs/csm
"""

import argparse, json, os, random, sys, time, unicodedata
from pathlib import Path
sys.stdout.reconfigure(line_buffering=True)


# Adja character set: empirically derived from 455k lines of Adja text across:
#   - <LOCAL_PATH>
#   - <LOCAL_PATH>
# Normalized with NFKC before counting. Ordered by descending frequency; includes
# every character seen ≥ 20 times in the corpus, plus all combining diacritics.
# Rare noise (typos like small-caps ᴐ, Greek ε) is excluded because add_tokens on
# them doesn't help the LM and wastes embedding rows.
ADJA_CHARS = [
    # Core Gbe-family phonemic characters
    "ɔ", "ɛ", "ɖ", "ŋ", "Ŋ", "Ɖ", "Ɔ", "Ɛ",
    # French-inherited accented vowels (very common in Adja borrowings + orthography)
    "é", "à", "è", "ê", "ó", "á", "ç", "ú", "ì", "ù", "ò", "ô", "â", "î", "û",
    "œ", "Ç", "É", "À", "Ê", "Á", "Ò", "Ô", "È", "ï", "ë", "í",
    # Additional African Latin characters
    "ɣ", "Ɣ", "ɤ", "ʒ", "Ʒ", "ƒ", "ɲ",
    # Tone-marked and caron vowels (less frequent but present)
    "ǒ", "ǎ", "ǔ", "ǹ", "ń", "ĩ", "ă", "ī", "ŭ", "ö", "õ", "ĭ", "ū", "Ô",
    # Combining diacritics (important: NFKC often keeps these as standalone chars)
    "\u0300",  # combining grave   (8006 occurrences)
    "\u0301",  # combining acute  (20421 occurrences)
    "\u0302",  # combining circumflex
    "\u0303",  # combining tilde
    "\u0304",  # combining macron
    "\u030C",  # combining caron
    "\u0330",  # combining tilde below
    # Special phonetic markers sometimes seen in Adja corpora
    "ː", "ˈ",
    # Frequent bigrams (Adja orthography uses these as digraphs)
    "kp", "gb", "ts", "dz", "ny", "ŋk", "ɖe", "ɖi",
]


def parse_args():
    p = argparse.ArgumentParser(description="T1-tokfix: CSM + expanded tokenizer → Adja")
    p.add_argument("--dataset", default="JosueG/adja-tts-orpheus")
    p.add_argument("--base-model", default="unsloth/csm-1b")
    p.add_argument("--output-dir", default="/tmp/csm_tokfix")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--num-epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--full-finetune", action="store_true")
    p.add_argument("--eval-steps", type=int, default=50)
    p.add_argument("--early-stopping-patience", type=int, default=5)
    p.add_argument("--push-to-hub", action="store_true")
    p.add_argument("--results-repo", default="JosueG/adja-tts-results")
    p.add_argument("--results-prefix", default="T1_csm_tokfix")
    return p.parse_args()


def normalize_text(text):
    # NFKC: matches the normalization used to derive ADJA_CHARS from the full
    # 455k-line prior-project corpus. NFKC is a superset of NFC — any text that
    # round-trips correctly under NFC also round-trips under NFKC.
    return " ".join(unicodedata.normalize("NFKC", text.strip()).split())


def normalize_waveform_range(audio):
    import numpy as np

    wav = np.asarray(audio, dtype=np.float32)
    if wav.ndim > 1:
        wav = wav.mean(axis=-1)
    if wav.size == 0:
        return wav
    max_abs = float(np.max(np.abs(wav)))
    if max_abs > 2.0:
        scale = 65536.0 if max_abs <= 65536.0 * 1.1 else max_abs
        wav = wav / scale
    return wav.astype(np.float32, copy=False)


def prepare_audio_array(audio_info, target_sr=24000):
    audio_array = normalize_waveform_range(audio_info["array"])
    sampling_rate = int(audio_info.get("sampling_rate") or target_sr)
    if sampling_rate != target_sr:
        import torch
        import torchaudio.functional as F

        tensor = torch.from_numpy(audio_array).unsqueeze(0)
        audio_array = F.resample(tensor, orig_freq=sampling_rate, new_freq=target_sr).squeeze(0).numpy()
        audio_array = normalize_waveform_range(audio_array)
    return audio_array


def target_sample_count(audio_info, target_sr=24000):
    sampling_rate = int(audio_info.get("sampling_rate") or target_sr)
    return int(round(len(audio_info["array"]) * target_sr / sampling_rate))


def _manual_resize_token_embeddings(model, new_vocab_size: int, old_vocab_size: int) -> None:
    import torch
    import torch.nn as nn

    def replace_module(root: nn.Module, dotted_name: str, new_module: nn.Module) -> None:
        parent = root
        parts = dotted_name.split(".")
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], new_module)

    embed_candidates: list[tuple[int, str, nn.Embedding]] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Embedding):
            score = 0
            if module.num_embeddings == old_vocab_size:
                score += 1000
            if any(k in name.lower() for k in ("token", "embed", "word", "vocab")):
                score += 50
            score += int(module.embedding_dim / 8)
            embed_candidates.append((score, name, module))

    if not embed_candidates:
        raise RuntimeError("Could not find an nn.Embedding to resize for tokenizer expansion")

    _, embed_name, old_embed = max(embed_candidates, key=lambda t: t[0])
    if old_embed.num_embeddings != old_vocab_size:
        raise RuntimeError(
            f"Best embedding candidate ({embed_name}) has num_embeddings={old_embed.num_embeddings}, expected {old_vocab_size}"
        )

    device = old_embed.weight.device
    dtype = old_embed.weight.dtype
    new_embed = nn.Embedding(new_vocab_size, old_embed.embedding_dim, device=device, dtype=dtype)
    nn.init.normal_(new_embed.weight, mean=0.0, std=0.02)
    new_embed.weight.data[:old_vocab_size].copy_(old_embed.weight.data)
    replace_module(model, embed_name, new_embed)

    # Resize a matching lm head if present (often nn.Linear(hidden, vocab)).
    head_candidates: list[tuple[int, str, nn.Linear]] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.out_features == old_vocab_size:
            score = 0
            if any(k in name.lower() for k in ("lm_head", "output", "classifier", "head")):
                score += 100
            score += int(module.in_features / 8)
            head_candidates.append((score, name, module))
    if head_candidates:
        _, head_name, old_head = max(head_candidates, key=lambda t: t[0])
        new_head = nn.Linear(old_head.in_features, new_vocab_size, bias=old_head.bias is not None, device=device, dtype=dtype)
        nn.init.normal_(new_head.weight, mean=0.0, std=0.02)
        new_head.weight.data[:old_vocab_size].copy_(old_head.weight.data)
        if old_head.bias is not None:
            new_head.bias.data.zero_()
            new_head.bias.data[:old_vocab_size].copy_(old_head.bias.data)
        replace_module(model, head_name, new_head)

        # If the original was likely tied, re-tie by reference when possible.
        try:
            if getattr(new_head, "bias", None) is None and new_head.weight.shape == new_embed.weight.shape:
                new_head.weight = new_embed.weight
        except Exception:
            pass

    # Keep config consistent for downstream save/load.
    if getattr(model, "config", None) is not None and hasattr(model.config, "vocab_size"):
        model.config.vocab_size = new_vocab_size
    if hasattr(model, "tie_weights"):
        try:
            model.tie_weights()
        except Exception:
            pass


def main():
    args = parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN required")

    import numpy as np
    import soundfile as sf
    import torch
    from datasets import Audio, load_dataset
    from huggingface_hub import HfApi
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoProcessor, CsmForConditionalGeneration,
                              EarlyStoppingCallback, Trainer, TrainingArguments)

    assert torch.cuda.is_available(), "CUDA required"
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    print(f"GPU: {torch.cuda.get_device_name(0)}\n")

    # ===== Dataset =====
    ds = load_dataset(args.dataset, token=token, split="train")
    split1 = ds.train_test_split(test_size=0.1, seed=args.seed)
    split2 = split1["train"].train_test_split(test_size=0.1/0.9, seed=args.seed)
    train_ds = split2["train"]
    dev_ds = split2["test"]
    test_ds = split1["test"]
    print(f"Train={len(train_ds)} Dev={len(dev_ds)} Test={len(test_ds)}\n")

    # ===== Model + TOKENIZER EXPANSION =====
    model = CsmForConditionalGeneration.from_pretrained(args.base_model, torch_dtype=torch.float32).to("cuda")
    processor = AutoProcessor.from_pretrained(args.base_model)

    # Diagnostic: before expansion, check how Adja chars tokenize
    sample_adja = "Mì ɖo alɔ ji ŋkɔ nyé Tom ɛnyi wɛ dze è"
    tokens_before = processor.tokenizer.tokenize(sample_adja)
    vocab_before = len(processor.tokenizer)
    print(f"[DIAGNOSTIC] Before expansion:")
    print(f"  vocab size: {vocab_before}")
    print(f"  '{sample_adja}' → {tokens_before}\n")

    # The expansion — test the hypothesis.
    # We add every ADJA_CHARS entry; the tokenizer de-dupes automatically and only
    # registers genuinely new tokens.
    n_added = processor.tokenizer.add_tokens(ADJA_CHARS)
    vocab_after = len(processor.tokenizer)
    # Critical: resize the model's embedding matrix so the new token IDs map to real vectors.
    # Without this, any new token would index past the embedding table and error at forward.
    #
    # Why the config.vocab_size dance: CsmForConditionalGeneration uses
    # config.vocab_size for the AUDIO codebook backbone loss (tensor last-dim = 2051,
    # a Mimi codebook + specials), NOT the text vocab. Both HF's resize_token_embeddings
    # and our _manual_resize_token_embeddings mutate config.vocab_size to vocab_after,
    # which makes the backbone_loss crash at `logits.view(-1, vocab_size)`. We save
    # the original value and restore it after any resize path runs.
    _orig_cfg_vocab = getattr(model.config, "vocab_size", None)
    try:
        model.resize_token_embeddings(vocab_after)
    except Exception as exc:
        print(f"[WARN] resize_token_embeddings failed ({type(exc).__name__}: {exc}). Falling back to manual resize.")
        _manual_resize_token_embeddings(model, new_vocab_size=vocab_after, old_vocab_size=vocab_before)
    if _orig_cfg_vocab is not None and getattr(model.config, "vocab_size", None) != _orig_cfg_vocab:
        print(f"[TOKFIX] Restoring config.vocab_size {model.config.vocab_size} → {_orig_cfg_vocab} "
              f"(CSM backbone_loss uses audio codebook vocab, not text vocab; text embedding rows remain at {vocab_after})")
        model.config.vocab_size = _orig_cfg_vocab

    tokens_after = processor.tokenizer.tokenize(sample_adja)
    print(f"[DIAGNOSTIC] After expansion:")
    print(f"  added: {n_added} new tokens, vocab {vocab_before} → {vocab_after}")
    print(f"  '{sample_adja}' → {tokens_after}\n")

    total_params = sum(p.numel() for p in model.parameters())
    if args.full_finetune:
        training_mode = "full_finetune"
    else:
        training_mode = f"lora_r{args.lora_r}"
        model = get_peft_model(model, LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_r,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0, bias="none",
        ))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable: {trainable / 1e6:.2f}M / {total_params / 1e6:.1f}M\n")

    # ===== Preprocess =====
    MAX_AUDIO_SAMPLES = 240001

    def filter_by_length(ds, name):
        n_before = len(ds)
        ds_filtered = ds.filter(lambda ex: target_sample_count(ex["audio"]) <= MAX_AUDIO_SAMPLES,
                                desc=f"Filtering over-long {name}")
        if len(ds_filtered) < n_before:
            print(f"  {name}: dropped {n_before - len(ds_filtered)} clips > 10s")
        return ds_filtered

    train_ds = filter_by_length(train_ds, "train")
    dev_ds = filter_by_length(dev_ds, "dev")

    def preprocess_example(example):
        text = normalize_text(example["text"])
        audio_array = prepare_audio_array(example["audio"], target_sr=24000)
        conversation = [{
            "role": "0",
            "content": [
                {"type": "text", "text": text},
                {"type": "audio", "path": audio_array},
            ],
        }]
        try:
            model_inputs = processor.apply_chat_template(
                conversation, tokenize=True, return_dict=True, output_labels=True,
                text_kwargs={"padding": "max_length", "max_length": 256,
                             "pad_to_multiple_of": 8, "padding_side": "right"},
                audio_kwargs={"sampling_rate": 24000, "max_length": 240001, "padding": "max_length"},
                common_kwargs={"return_tensors": "pt"},
            )
        except Exception as exc:
            print(f"  Skip '{text[:40]}': {exc}")
            return None
        required = ["input_ids", "attention_mask", "labels", "input_values", "input_values_cutoffs"]
        if any(k not in model_inputs for k in required):
            return None
        return {k: model_inputs[k][0] for k in required}

    processed_train = train_ds.map(preprocess_example, remove_columns=train_ds.column_names, desc="Preprocessing train")
    processed_train = processed_train.filter(lambda x: x.get("input_ids") is not None)
    processed_dev = dev_ds.map(preprocess_example, remove_columns=dev_ds.column_names, desc="Preprocessing dev")
    processed_dev = processed_dev.filter(lambda x: x.get("input_ids") is not None)
    print(f"Preprocessed: train={len(processed_train)} dev={len(processed_dev)}\n")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir / "trainer_output"),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_steps=10, num_train_epochs=args.num_epochs, max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        fp16=not torch.cuda.is_bf16_supported(), bf16=torch.cuda.is_bf16_supported(),
        logging_steps=5, optim="adamw_torch", weight_decay=0.001,
        lr_scheduler_type="cosine", seed=args.seed, report_to="none",
        eval_strategy="steps", eval_steps=args.eval_steps,
        save_strategy="steps", save_steps=args.eval_steps, save_total_limit=3,
        load_best_model_at_end=True, metric_for_best_model="eval_loss",
        greater_is_better=False, remove_unused_columns=False, label_names=["labels"],
    )
    trainer = Trainer(
        model=model, train_dataset=processed_train, eval_dataset=processed_dev,
        args=training_args,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )
    t0 = time.time()
    stats = trainer.train()
    elapsed = time.time() - t0
    train_loss = float(stats.metrics.get("train_loss", 0.0))
    peak_mem = round(torch.cuda.max_memory_reserved() / 1e9, 2)
    eval_history = [l for l in trainer.state.log_history if "eval_loss" in l]
    best_eval = min((l["eval_loss"] for l in eval_history), default=None)
    print(f"Done {elapsed/60:.1f}min | train_loss={train_loss:.4f} | best_eval={best_eval}\n")

    # ===== Generate =====
    model.eval()
    generated = []
    generated_dir = output_dir / "generated"
    generated_dir.mkdir(exist_ok=True)
    for i in range(min(5, len(test_ds))):
        text = normalize_text(test_ds[i]["text"])
        try:
            plain_inputs = processor(f"[0]{text}", add_special_tokens=True, return_tensors="pt").to(model.device)
            with torch.no_grad():
                audio_values = model.generate(**plain_inputs, max_new_tokens=125, output_audio=True)
            audio = audio_values[0].to(torch.float32).cpu().numpy()
            if audio.size == 0:
                raise RuntimeError("empty waveform")
            path = generated_dir / f"adja_{i:02d}.wav"
            sf.write(str(path), audio, 24000)
            generated.append({"text": text, "file": path.name, "duration_sec": round(len(audio)/24000, 2)})
        except Exception as exc:
            generated.append({"text": text, "error": str(exc)})

    adapter_dir = output_dir / "adapter"
    model.save_pretrained(str(adapter_dir))
    processor.save_pretrained(str(adapter_dir))

    results = {
        "experiment": "T1_csm_tokfix",
        "base_model": args.base_model,
        "training_mode": training_mode,
        "tokenizer_expansion": {
            "chars_attempted": len(ADJA_CHARS),
            "chars_actually_added": n_added,
            "vocab_before": vocab_before,
            "vocab_after": vocab_after,
            "tokens_before": tokens_before,
            "tokens_after": tokens_after,
        },
        "train_loss": round(train_loss, 4),
        "best_eval_loss": round(best_eval, 4) if best_eval else None,
        "training_time_min": round(elapsed/60, 1),
        "peak_vram_gb": peak_mem,
        "gpu": torch.cuda.get_device_name(0),
        "generated": generated,
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if args.push_to_hub:
        api = HfApi(token=token)
        prefix = args.results_prefix
        api.create_repo(args.results_repo, private=True, exist_ok=True)
        api.upload_file(path_or_fileobj=metrics_path.read_bytes(),
                        path_in_repo=f"{prefix}/metrics.json",
                        repo_id=args.results_repo, token=token)
        api.upload_folder(folder_path=str(adapter_dir),
                          path_in_repo=f"{prefix}/adapter",
                          repo_id=args.results_repo, token=token)
        api.upload_folder(folder_path=str(generated_dir),
                          path_in_repo=f"{prefix}/generated_audio",
                          repo_id=args.results_repo, token=token)
        print(f"Results: https://huggingface.co/{args.results_repo}/tree/main/{prefix}")


if __name__ == "__main__":
    main()
