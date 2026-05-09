#!/usr/bin/env python3
from __future__ import annotations
"""
Audio-LM pretraining: CSM backbone on Mimi codes from unlabeled Ewe → fine-tune.

Last updated: 2026-04-21

Pipeline (two stages):
  Stage A: Encode 183k unlabeled Ewe clips into Mimi codes (no text). Train CSM's
           LM backbone on pure next-token prediction over the flattened code stream.
           The LM learns acoustic prior for Gbe-family speech without any transcripts.
  Stage B: Supervised fine-tune on labeled Ewe TTS + Adja TTS data, using the
           standard (text, audio) conversation format.

Gated on: Mimi reconstruction test passing locally (see
  experiments/tts/T1_sesame_csm_finetune/mimi_reconstruction_test.py).
If the codec loses tonal information at encode-decode, this track is dead and
we should fall back to audio_lm_orpheus_ewe.py (SNAC) or tokenizer expansion.

References:
  - Mimi codec: https://arxiv.org/abs/2410.00037
  - Sesame CSM: https://github.com/SesameAILabs/csm
  - Audio LM pretraining (AudioLM, VALL-E): https://arxiv.org/abs/2209.03143

Runtime (A100 80GB): Stage A ~24h for 183k utts, Stage B ~6-8h for ~17k labeled utts.
"""
import argparse, json, os, random, sys, time, unicodedata
from pathlib import Path
sys.stdout.reconfigure(line_buffering=True)

from waxalnlp_loader import load_waxal_split, safe_load_audio_array


def parse_args():
    p = argparse.ArgumentParser(description="CSM audio-LM pretraining on Ewe")
    p.add_argument("--models-dir", default="/models")
    p.add_argument("--data-dir", default="/data")
    p.add_argument("--output-dir", default="/results/audiolm_csm_ewe")
    p.add_argument("--base", default="unsloth/csm-1b")
    p.add_argument("--mimi-repo", default="kyutai/mimi",
                   help="Mimi codec repo (separate from CSM)")
    p.add_argument("--stage", choices=["pretrain", "finetune", "both"], default="both")
    p.add_argument("--pretrain-epochs", type=int, default=3)
    p.add_argument("--finetune-epochs", type=int, default=20)
    p.add_argument("--pretrain-batch", type=int, default=4)
    p.add_argument("--pretrain-lr", type=float, default=1e-4)
    p.add_argument("--finetune-lr", type=float, default=5e-5)
    p.add_argument("--max-audio-sec", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model-key", default="csm-1b")
    p.add_argument("--preprocess-proc", type=int, default=min(8, (os.cpu_count() or 1)),
                   help="Processes for dataset preprocessing (decode). "
                        "Falls back to 1 if multiprocessing fails.")
    p.add_argument("--smoke", action="store_true",
                   help="Tiny dry run: 4 samples, 2 steps, Stage A only, no save.")
    return p.parse_args()


def resolve_base(args, repo):
    safe = repo.replace("/", "__")
    candidate = Path(args.models_dir) / safe
    return str(candidate) if candidate.exists() else repo


def main():
    args = parse_args()
    hf_token = os.environ.get("HF_TOKEN")
    import numpy as np
    import torch
    from datasets import Audio, Features, Sequence, Value, load_dataset
    from transformers import (AutoProcessor, CsmForConditionalGeneration,
                              EarlyStoppingCallback, Trainer, TrainingArguments)

    assert torch.cuda.is_available(), "CUDA required"
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pretrain_out = output_dir / "stage_a_pretrain"
    finetune_out = output_dir / "stage_b_finetune"

    base_path = resolve_base(args, args.base)
    mimi_path = resolve_base(args, args.mimi_repo)
    print(f"CSM base: {base_path}\nMimi:     {mimi_path}")
    print(f"GPU: {torch.cuda.get_device_name(0)}\n")

    if args.smoke:
        print(">>> SMOKE MODE: 4 samples, 2 steps, Stage A only, no save <<<\n")
        args.stage = "pretrain"
        args.pretrain_epochs = 1

    # --------------------------------------------------------------
    # Stage A — pretrain CSM LM on Mimi codes from unlabeled Ewe
    # --------------------------------------------------------------
    if args.stage in ("pretrain", "both"):
        print("=" * 60)
        print("STAGE A — CSM LM on Mimi codes (unlabeled Ewe)")
        print("=" * 60)

        # Load unlabeled Ewe
        wax_cache = Path(args.data_dir) / "WaxalNLP_ewe_asr"
        unlab = load_waxal_split(
            config="ewe_asr",
            split="unlabeled",
            cache_dir=str(wax_cache),
            token=hf_token,
        )
        if args.smoke:
            unlab = unlab.select(range(4))
        max_samples = int(args.max_audio_sec * 24000)
        # Avoid eager Audio decoding: some unlabeled blobs fail libsndfile.
        unlab = unlab.cast_column("audio", Audio(decode=False))
        n_total = len(unlab)
        print(f"Unlabeled Ewe utts (raw): {n_total}")

        # Load CSM + Mimi encoder (CSM ships with Mimi internally in its processor)
        model = CsmForConditionalGeneration.from_pretrained(base_path, torch_dtype=torch.float32).to("cuda")
        processor = AutoProcessor.from_pretrained(base_path)

        # Build audio-only training examples using conversation-style inputs with
        # a placeholder text "[0]" so the LM sees only audio tokens as labels.
        keys = ["input_ids", "attention_mask", "labels", "input_values", "input_values_cutoffs"]
        out_features = Features({
            "_ok": Value("bool"),
            "input_ids": Sequence(Value("int64")),
            "attention_mask": Sequence(Value("int64")),
            "labels": Sequence(Value("int64")),
            # CSM processor returns per-example audio as [channels, samples].
            # Keep the nested shape so Arrow does not try to cast a list to float.
            "input_values": Sequence(Sequence(Value("float32"))),
            "input_values_cutoffs": Sequence(Value("int64")),
        })

        def _to_numpy(value):
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            return np.asarray(value)

        def _int_vector(value):
            return _to_numpy(value).astype(np.int64, copy=False).reshape(-1).tolist()

        def _float32_matrix(value):
            arr = _to_numpy(value).astype(np.float32, copy=False)
            if arr.ndim == 0:
                arr = arr.reshape(1, 1)
            elif arr.ndim == 1:
                arr = arr.reshape(1, -1)
            elif arr.ndim > 2:
                arr = arr.reshape(-1, arr.shape[-1])
            return arr.tolist()

        def empty_row(ok: bool):
            return {
                "_ok": ok,
                "input_ids": [],
                "attention_mask": [],
                "labels": [],
                "input_values": [],
                "input_values_cutoffs": [],
            }

        def encode_audio_only(ex):
            try:
                arr = safe_load_audio_array(ex.get("audio"), target_sr=24000)
                if arr is None or len(arr) > max_samples:
                    return empty_row(False)
                conv = [{
                    "role": "0",
                    "content": [
                        {"type": "text", "text": " "},  # minimal text; LM learns audio
                        {"type": "audio", "path": arr},
                    ],
                }]
                m = processor.apply_chat_template(
                    conv, tokenize=True, return_dict=True, output_labels=True,
                    text_kwargs={"padding": "max_length", "max_length": 64,
                                 "pad_to_multiple_of": 8, "padding_side": "right"},
                    audio_kwargs={"sampling_rate": 24000, "max_length": 240001, "padding": "max_length"},
                    common_kwargs={"return_tensors": "pt"},
                )
                if not all(k in m for k in keys):
                    return empty_row(False)

                return {
                    "_ok": True,
                    "input_ids": _int_vector(m["input_ids"][0]),
                    "attention_mask": _int_vector(m["attention_mask"][0]),
                    "labels": _int_vector(m["labels"][0]),
                    "input_values": _float32_matrix(m["input_values"][0]),
                    "input_values_cutoffs": _int_vector(m["input_values_cutoffs"][0]),
                }
            except Exception:
                return empty_row(False)

        try:
            encoded = unlab.map(
                encode_audio_only,
                remove_columns=unlab.column_names,
                features=out_features,
                num_proc=max(1, int(args.preprocess_proc)),
                desc="decode+encode",
            )
        except Exception as exc:
            print(f"[warn] preprocessing multiprocessing failed ({exc}); retrying with --preprocess-proc 1",
                  file=__import__('sys').stderr)
            encoded = unlab.map(
                encode_audio_only,
                remove_columns=unlab.column_names,
                features=out_features,
                num_proc=1,
                desc="decode+encode",
            )
        encoded = encoded.filter(lambda x: x["_ok"], desc="drop bad/long")
        encoded = encoded.remove_columns(["_ok"])
        n_ok = len(encoded)
        print(f"Usable unlabeled samples: {n_ok}/{n_total}")
        if n_ok == 0:
            try:
                a = unlab[0].get("audio") if n_total else None
                raw = (a or {}).get("bytes") or b""
                magic = raw[:8]
                path = (a or {}).get("path")
                keys = list((a or {}).keys()) if isinstance(a, dict) else []
                print(f"[fatal] decoded 0/{n_total}. audio keys={keys} path={path!r} bytes_magic={magic!r}",
                      file=__import__('sys').stderr)
            except Exception:
                print(f"[fatal] decoded 0/{n_total}. Could not inspect first example.",
                      file=__import__('sys').stderr)
            raise SystemExit("No usable unlabeled audio after decoding; check ffmpeg bytes decode.")

        training_args = TrainingArguments(
            output_dir=str(pretrain_out / "trainer"),
            per_device_train_batch_size=args.pretrain_batch,
            num_train_epochs=args.pretrain_epochs,
            max_steps=2 if args.smoke else -1,
            learning_rate=args.pretrain_lr,
            warmup_steps=0 if args.smoke else 500,
            lr_scheduler_type="linear",
            fp16=not torch.cuda.is_bf16_supported(), bf16=torch.cuda.is_bf16_supported(),
            logging_steps=1 if args.smoke else 100,
            save_strategy="no" if args.smoke else "epoch", save_total_limit=2,
            seed=args.seed, report_to="none", remove_unused_columns=False,
            label_names=["labels"],
            dataloader_num_workers=0 if args.smoke else 4,
        )
        trainer = Trainer(model=model, train_dataset=encoded, args=training_args)
        t0 = time.time()
        trainer.train()
        elapsed = time.time() - t0
        if args.smoke:
            peak_gb = torch.cuda.max_memory_reserved() / 1e9
            print(f"SMOKE PASS: audio_lm_csm_ewe | peak VRAM {peak_gb:.1f}GB | {elapsed:.1f}s")
            return
        model.save_pretrained(str(pretrain_out / "backbone"))
        processor.save_pretrained(str(pretrain_out / "backbone"))
        print(f"Stage A done: {elapsed/60:.1f}min\n")

    # --------------------------------------------------------------
    # Stage B — supervised fine-tune on labeled Ewe + Adja
    # --------------------------------------------------------------
    if args.stage in ("finetune", "both"):
        print("=" * 60)
        print("STAGE B — CSM supervised fine-tune (Ewe + Adja)")
        print("=" * 60)

        start_path = str(pretrain_out / "backbone") if args.stage == "both" else base_path

        # Ewe TTS + Adja TTS combined
        from datasets import concatenate_datasets
        ewe = load_waxal_split(
            config="ewe_tts",
            split="train",
            cache_dir=str(Path(args.data_dir) / "WaxalNLP_ewe_tts"),
            token=hf_token,
        )
        ewe = ewe.cast_column("audio", Audio(sampling_rate=24000))
        if "sentence" in ewe.column_names and "text" not in ewe.column_names:
            ewe = ewe.rename_column("sentence", "text")
        adja = load_dataset("JosueG/adja-tts-orpheus", split="train",
                            cache_dir=str(Path(args.data_dir) / "JosueG_adja-tts-orpheus"),
                            token=hf_token)
        adja = adja.cast_column("audio", Audio(sampling_rate=24000))

        combined = concatenate_datasets([
            ewe.remove_columns([c for c in ewe.column_names if c not in ("audio", "text")]),
            adja.remove_columns([c for c in adja.column_names if c not in ("audio", "text")]),
        ])
        split = combined.train_test_split(test_size=0.1, seed=args.seed)
        train_ds, dev_ds = split["train"], split["test"]

        model = CsmForConditionalGeneration.from_pretrained(start_path, torch_dtype=torch.float32).to("cuda")
        processor = AutoProcessor.from_pretrained(start_path)

        def normalize(t):
            return " ".join(unicodedata.normalize("NFKC", t.strip()).split())

        def prep(ex):
            text = normalize(ex["text"])
            conv = [{
                "role": "0",
                "content": [{"type": "text", "text": text},
                            {"type": "audio", "path": ex["audio"]["array"]}],
            }]
            try:
                m = processor.apply_chat_template(
                    conv, tokenize=True, return_dict=True, output_labels=True,
                    text_kwargs={"padding": "max_length", "max_length": 256,
                                 "pad_to_multiple_of": 8, "padding_side": "right"},
                    audio_kwargs={"sampling_rate": 24000, "max_length": 240001, "padding": "max_length"},
                    common_kwargs={"return_tensors": "pt"},
                )
                keys = ["input_ids", "attention_mask", "labels", "input_values", "input_values_cutoffs"]
                return {k: m[k][0] for k in keys} if all(k in m for k in keys) else None
            except Exception:
                return None

        train_p = train_ds.filter(lambda ex: len(ex["audio"]["array"]) <= 240001)
        train_p = train_p.map(prep, remove_columns=train_p.column_names, desc="prep train")
        train_p = train_p.filter(lambda x: x.get("input_ids") is not None)
        dev_p = dev_ds.filter(lambda ex: len(ex["audio"]["array"]) <= 240001)
        dev_p = dev_p.map(prep, remove_columns=dev_p.column_names, desc="prep dev")
        dev_p = dev_p.filter(lambda x: x.get("input_ids") is not None)

        training_args = TrainingArguments(
            output_dir=str(finetune_out / "trainer"),
            per_device_train_batch_size=4, per_device_eval_batch_size=4,
            gradient_accumulation_steps=4,
            num_train_epochs=args.finetune_epochs,
            learning_rate=args.finetune_lr,
            warmup_steps=100, lr_scheduler_type="cosine",
            fp16=not torch.cuda.is_bf16_supported(), bf16=torch.cuda.is_bf16_supported(),
            logging_steps=20, eval_strategy="steps", eval_steps=50,
            save_strategy="steps", save_steps=50, save_total_limit=3,
            load_best_model_at_end=True, metric_for_best_model="eval_loss",
            greater_is_better=False, seed=args.seed, report_to="none",
            remove_unused_columns=False, label_names=["labels"],
        )
        trainer = Trainer(
            model=model, args=training_args,
            train_dataset=train_p, eval_dataset=dev_p,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
        )
        t0 = time.time()
        stats = trainer.train()
        elapsed = time.time() - t0
        train_loss = float(stats.metrics.get("train_loss", 0.0))
        eval_history = [l for l in trainer.state.log_history if "eval_loss" in l]
        best_eval = min((l["eval_loss"] for l in eval_history), default=None)

        model.save_pretrained(str(finetune_out / "model"))
        processor.save_pretrained(str(finetune_out / "model"))

        results = {
            "experiment": "audiolm_csm_ewe",
            "stage": args.stage,
            "pretrain_epochs": args.pretrain_epochs,
            "finetune_epochs": args.finetune_epochs,
            "train_loss": round(train_loss, 4),
            "best_eval_loss": round(best_eval, 4) if best_eval else None,
            "finetune_time_min": round(elapsed/60, 1),
            "gpu": torch.cuda.get_device_name(0),
            "peak_vram_gb": round(torch.cuda.max_memory_reserved()/1e9, 2),
        }
        (output_dir / "metrics.json").write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
        print(f"All done. Metrics saved.")


if __name__ == "__main__":
    main()
