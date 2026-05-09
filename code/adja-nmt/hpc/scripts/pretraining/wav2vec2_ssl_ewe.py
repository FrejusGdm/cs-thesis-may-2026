#!/usr/bin/env python3
from __future__ import annotations
"""
wav2vec 2.0 self-supervised continue-pretraining on unlabeled Ewe → Adja fine-tune.

Last updated: 2026-04-21

Two-stage pipeline driven by the --stage flag (or run both with --stage both):
  Stage A: continue-pretrain the wav2vec 2.0 encoder on unlabeled Ewe audio
           (WaxalNLP ewe_asr `unlabeled` split, ~183k utterances) using the
           standard contrastive SSL objective. This adapts the encoder to Gbe-family
           phonology without requiring transcripts.
  Stage B: supervised CTC fine-tune on Adja (JosueG/adja-tts-orpheus), starting
           from the Stage A encoder.

Hypothesis: the SSL phase makes the encoder sensitive to Adja's phonemic
distinctions (ATR vowels, tones, nasals), so the supervised phase in Stage B
converges with far less labeled data than starting from an English-pretrained
encoder.

References:
  - wav2vec 2.0: https://arxiv.org/abs/2006.11477
  - XLS-R: https://arxiv.org/abs/2111.09296
  - SSL for low-resource ASR: https://arxiv.org/abs/2211.04546 (survey)

Runtime (A100 80GB):
  - Stage A: ~24-36h for 183k utts, 3 epochs (model-dependent)
  - Stage B: ~4-8h for 1.6k Adja utts, 30 epochs with early stopping
"""
import argparse, json, os, random, sys, time, unicodedata
from pathlib import Path
sys.stdout.reconfigure(line_buffering=True)

from waxalnlp_loader import load_waxal_split, safe_load_audio_array


def parse_args():
    p = argparse.ArgumentParser(description="wav2vec 2.0 SSL on Ewe → Adja")
    p.add_argument("--model-key", default="xlsr-300m",
                   help="Logical name for --models-dir lookup")
    p.add_argument("--base", default="facebook/wav2vec2-xls-r-300m",
                   help="Local model dir name or HF repo id")
    p.add_argument("--models-dir", default="/models",
                   help="Cluster-local models dir (read-only inside container)")
    p.add_argument("--data-dir", default="/data",
                   help="Cluster-local data dir with pre-cached HF datasets")
    p.add_argument("--output-dir", default="/results/ssl_xlsr_ewe")
    p.add_argument("--stage", choices=["pretrain", "finetune", "both"], default="both")
    p.add_argument("--pretrain-epochs", type=int, default=3)
    p.add_argument("--finetune-epochs", type=int, default=30)
    p.add_argument("--pretrain-batch-size", type=int, default=8)
    p.add_argument("--pretrain-grad-accum", type=int, default=1)
    p.add_argument("--finetune-batch-size", type=int, default=8)
    p.add_argument("--finetune-grad-accum", type=int, default=1)
    p.add_argument("--pretrain-lr", type=float, default=5e-5)
    p.add_argument("--finetune-lr", type=float, default=3e-5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-audio-sec", type=float, default=15.0)
    p.add_argument("--early-stopping-patience", type=int, default=10)
    p.add_argument("--preprocess-proc", type=int, default=min(8, (os.cpu_count() or 1)),
                   help="Processes for dataset preprocessing (decode/feature). "
                        "Falls back to 1 if multiprocessing fails.")
    p.add_argument("--smoke", action="store_true",
                   help="Tiny dry run: 4 samples, 2 steps, Stage A only, no save. "
                        "Use this inside gpu_smoke.sh before real sbatch.")
    return p.parse_args()


def resolve_base(args):
    """Turn HF repo-id or local safe-name into an absolute path."""
    safe = args.base.replace("/", "__")
    candidate = Path(args.models_dir) / safe
    if candidate.exists():
        return str(candidate)
    # Fall back to direct repo id (will need HF_HUB_OFFLINE=0 on cluster)
    return args.base


def main():
    args = parse_args()
    # HF_TOKEN required for gated WaxalNLP and private JosueG/adja-tts-orpheus.
    # `datasets` reads env var automatically in recent versions but be explicit.
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("WARNING: HF_TOKEN not set; gated/private datasets will 401.",
              file=__import__("sys").stderr)
    import numpy as np
    import torch
    from datasets import Audio, Features, Sequence, Value, load_dataset
    from transformers import (AutoFeatureExtractor, AutoModelForPreTraining,
                              AutoProcessor, EarlyStoppingCallback, Trainer,
                              TrainingArguments, Wav2Vec2FeatureExtractor,
                              Wav2Vec2ForCTC, Wav2Vec2ForPreTraining,
                              Wav2Vec2Processor)

    assert torch.cuda.is_available(), "CUDA required"
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    base_path = resolve_base(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"[{args.model_key}] base={base_path}")
    print(f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {vram_gb:.1f}GB\n")

    if vram_gb < 60:
        old_pre_bs, old_pre_ga = args.pretrain_batch_size, args.pretrain_grad_accum
        old_ft_bs, old_ft_ga = args.finetune_batch_size, args.finetune_grad_accum
        args.pretrain_batch_size = 1
        args.pretrain_grad_accum = max(args.pretrain_grad_accum, old_pre_bs)
        args.finetune_batch_size = min(args.finetune_batch_size, 2)
        args.finetune_grad_accum = max(args.finetune_grad_accum, max(1, old_ft_bs // args.finetune_batch_size))
        print(f"[auto-fit] low VRAM detected ({vram_gb:.1f}GB): "
              f"pretrain batch {old_pre_bs}->{args.pretrain_batch_size}, "
              f"pretrain grad_accum {old_pre_ga}->{args.pretrain_grad_accum}, "
              f"finetune batch {old_ft_bs}->{args.finetune_batch_size}, "
              f"finetune grad_accum {old_ft_ga}->{args.finetune_grad_accum}")

    pretrain_out = output_dir / "stage_a_pretrain"
    finetune_out = output_dir / "stage_b_finetune"

    def _float32_list(values):
        return np.asarray(values, dtype=np.float32).reshape(-1).tolist()

    # ==================================================================
    # STAGE A: SSL pretraining on unlabeled Ewe
    # ==================================================================
    # Smoke mode overrides: run a minimal Stage A dry run, skip Stage B.
    if args.smoke:
        print(">>> SMOKE MODE: 4 samples, 2 steps, Stage A only, no save <<<\n")
        args.stage = "pretrain"
        args.pretrain_epochs = 1
        args.pretrain_batch_size = 1 if vram_gb < 60 else 2
        args.pretrain_grad_accum = 1

    if args.stage in ("pretrain", "both"):
        print("=" * 60)
        print("STAGE A — SSL continue-pretraining on unlabeled Ewe")
        print("=" * 60)

        # Load unlabeled Ewe. This split is audio-only, no transcripts.
        wax_cache = Path(args.data_dir) / "WaxalNLP_ewe_asr"
        unlabeled = load_waxal_split(
            config="ewe_asr",
            split="unlabeled",
            cache_dir=str(wax_cache),
            token=hf_token,
        )
        if args.smoke:
            unlabeled = unlabeled.select(range(4))
        # Avoid eager Audio decoding: some unlabeled blobs fail libsndfile.
        unlabeled = unlabeled.cast_column("audio", Audio(decode=False))
        max_samples = int(args.max_audio_sec * 16000)
        n_total = len(unlabeled)
        print(f"Unlabeled Ewe utterances (raw): {n_total}")

        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(base_path)
        model = Wav2Vec2ForPreTraining.from_pretrained(base_path)
        model.gradient_checkpointing_enable()
        model.to("cuda")

        out_features = Features({
            "keep": Value("bool"),
            "input_values": Sequence(Value("float32")),
        })

        def prep_ssl(ex):
            arr = safe_load_audio_array(ex.get("audio"), target_sr=16000)
            if arr is None:
                return {"keep": False, "input_values": []}
            if len(arr) > max_samples:
                arr = arr[:max_samples]
            iv = feature_extractor(np.asarray(arr, dtype=np.float32), sampling_rate=16000).input_values[0]
            return {"keep": True, "input_values": _float32_list(iv)}

        try:
            encoded = unlabeled.map(
                prep_ssl,
                remove_columns=unlabeled.column_names,
                features=out_features,
                num_proc=max(1, int(args.preprocess_proc)),
                desc="decode+feature",
            )
        except Exception as exc:
            print(f"[warn] preprocessing multiprocessing failed ({exc}); retrying with --preprocess-proc 1",
                  file=__import__('sys').stderr)
            encoded = unlabeled.map(
                prep_ssl,
                remove_columns=unlabeled.column_names,
                features=out_features,
                num_proc=1,
                desc="decode+feature",
            )
        encoded = encoded.filter(lambda ex: ex["keep"], desc="drop bad/long")
        encoded = encoded.remove_columns(["keep"])
        n_ok = len(encoded)
        print(f"Usable unlabeled utterances: {n_ok}/{n_total}")
        if n_ok == 0:
            try:
                a = unlabeled[0].get("audio") if n_total else None
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

        # Quantizer + contrastive loss is what Wav2Vec2ForPreTraining.forward handles when
        # mask_time_indices + sampled_negative_indices are passed. HF has a helper:
        from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices
        from transformers.models.wav2vec2.modeling_wav2vec2 import _sample_negative_indices

        def ssl_collate(batch):
            feats = [torch.tensor(b["input_values"], dtype=torch.float32) for b in batch]
            maxlen = max(f.size(0) for f in feats)
            padded = torch.zeros(len(feats), maxlen)
            attention_mask = torch.zeros(len(feats), maxlen, dtype=torch.long)
            for i, f in enumerate(feats):
                padded[i, :f.size(0)] = f
                attention_mask[i, :f.size(0)] = 1
            # Mask 6.5% of time-steps with span 10 (wav2vec 2.0 default)
            feat_shape = model._get_feat_extract_output_lengths(attention_mask.sum(-1)).tolist()
            mask_time_indices = _compute_mask_indices(
                shape=(len(feats), int(max(feat_shape))),
                mask_prob=0.065, mask_length=10,
                attention_mask=None,
            )
            sampled_neg = _sample_negative_indices(
                features_shape=(len(feats), int(max(feat_shape))),
                num_negatives=100,
                mask_time_indices=mask_time_indices,
            )
            return {
                "input_values": padded,
                "attention_mask": attention_mask,
                "mask_time_indices": torch.tensor(mask_time_indices, dtype=torch.long),
                "sampled_negative_indices": torch.tensor(sampled_neg, dtype=torch.long),
            }

        training_args = TrainingArguments(
            output_dir=str(pretrain_out / "trainer"),
            per_device_train_batch_size=args.pretrain_batch_size,
            num_train_epochs=args.pretrain_epochs,
            gradient_accumulation_steps=1 if args.smoke else args.pretrain_grad_accum,
            max_steps=2 if args.smoke else -1,
            learning_rate=args.pretrain_lr,
            warmup_ratio=0.1, lr_scheduler_type="linear",
            fp16=not torch.cuda.is_bf16_supported(), bf16=torch.cuda.is_bf16_supported(),
            logging_steps=1 if args.smoke else 50,
            save_strategy="no" if args.smoke else "epoch", save_total_limit=2,
            seed=args.seed, report_to="none", remove_unused_columns=False,
            dataloader_num_workers=0 if args.smoke else 4,
            gradient_checkpointing=True,
        )
        trainer = Trainer(model=model, train_dataset=encoded, args=training_args,
                          data_collator=ssl_collate)
        t0 = time.time()
        trainer.train()
        elapsed = time.time() - t0
        if args.smoke:
            peak_gb = torch.cuda.max_memory_reserved() / 1e9
            print(f"SMOKE PASS: {args.model_key} | peak VRAM {peak_gb:.1f}GB | {elapsed:.1f}s")
            return
        model.save_pretrained(str(pretrain_out / "encoder"))
        feature_extractor.save_pretrained(str(pretrain_out / "encoder"))
        print(f"Stage A done: {elapsed/60:.1f}min\n")

    # ==================================================================
    # STAGE B: supervised CTC fine-tune on Adja
    # ==================================================================
    if args.stage in ("finetune", "both"):
        print("=" * 60)
        print("STAGE B — supervised CTC on Adja")
        print("=" * 60)

        encoder_path = str(pretrain_out / "encoder") if args.stage == "both" else base_path

        adja_cache = Path(args.data_dir) / "JosueG_adja-tts-orpheus"
        adja = load_dataset("JosueG/adja-tts-orpheus", split="train",
                            cache_dir=str(adja_cache), token=hf_token)
        s1 = adja.train_test_split(test_size=0.1, seed=args.seed)
        s2 = s1["train"].train_test_split(test_size=0.1/0.9, seed=args.seed)
        train_ds = s2["train"].cast_column("audio", Audio(sampling_rate=16000))
        dev_ds = s2["test"].cast_column("audio", Audio(sampling_rate=16000))
        test_ds = s1["test"].cast_column("audio", Audio(sampling_rate=16000))

        # Build Adja char vocab for CTC
        def normalize(t):
            return " ".join(unicodedata.normalize("NFC", t.strip()).split())

        all_chars = set()
        for row in train_ds:
            all_chars.update(normalize(row["text"]))
        vocab = {"[PAD]": 0, "[UNK]": 1, "|": 2}
        for c in sorted(all_chars):
            if c == " ":
                continue
            vocab[c] = len(vocab)
        print(f"CTC vocab size: {len(vocab)}")

        from transformers import Wav2Vec2CTCTokenizer
        vocab_path = finetune_out / "vocab.json"
        finetune_out.mkdir(parents=True, exist_ok=True)
        vocab_path.write_text(json.dumps(vocab, ensure_ascii=False))
        tokenizer = Wav2Vec2CTCTokenizer(str(vocab_path), unk_token="[UNK]",
                                         pad_token="[PAD]", word_delimiter_token="|")
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(encoder_path)
        processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

        model = Wav2Vec2ForCTC.from_pretrained(
            encoder_path, vocab_size=len(vocab),
            pad_token_id=tokenizer.pad_token_id,
            ctc_loss_reduction="mean",
        ).to("cuda")
        model.freeze_feature_encoder()

        def prep(ex):
            arr = np.asarray(ex["audio"]["array"], dtype=np.float32)
            text = normalize(ex["text"]).replace(" ", "|")
            iv = feature_extractor(arr, sampling_rate=16000).input_values[0]
            labels = tokenizer(text).input_ids
            return {"input_values": _float32_list(iv), "labels": [int(x) for x in labels]}

        train_p = train_ds.map(prep, remove_columns=train_ds.column_names, desc="prep train")
        dev_p = dev_ds.map(prep, remove_columns=dev_ds.column_names, desc="prep dev")

        def ctc_collate(batch):
            ivs = [torch.tensor(b["input_values"], dtype=torch.float32) for b in batch]
            labels = [torch.tensor(b["labels"], dtype=torch.long) for b in batch]
            max_iv = max(i.size(0) for i in ivs)
            max_lab = max(l.size(0) for l in labels)
            iv_pad = torch.zeros(len(ivs), max_iv)
            attn = torch.zeros(len(ivs), max_iv, dtype=torch.long)
            lab_pad = torch.full((len(labels), max_lab), -100, dtype=torch.long)
            for i, (iv, lab) in enumerate(zip(ivs, labels)):
                iv_pad[i, :iv.size(0)] = iv
                attn[i, :iv.size(0)] = 1
                lab_pad[i, :lab.size(0)] = lab
            return {"input_values": iv_pad, "attention_mask": attn, "labels": lab_pad}

        training_args = TrainingArguments(
            output_dir=str(finetune_out / "trainer"),
            per_device_train_batch_size=args.finetune_batch_size,
            per_device_eval_batch_size=args.finetune_batch_size,
            gradient_accumulation_steps=1 if args.smoke else args.finetune_grad_accum,
            num_train_epochs=args.finetune_epochs,
            learning_rate=args.finetune_lr,
            warmup_steps=500, lr_scheduler_type="linear",
            fp16=not torch.cuda.is_bf16_supported(), bf16=torch.cuda.is_bf16_supported(),
            logging_steps=20, eval_strategy="epoch", save_strategy="epoch",
            save_total_limit=3, load_best_model_at_end=True,
            metric_for_best_model="eval_loss", greater_is_better=False,
            seed=args.seed, report_to="none", remove_unused_columns=False,
            dataloader_num_workers=4, gradient_checkpointing=True,
        )
        trainer = Trainer(
            model=model, args=training_args,
            train_dataset=train_p, eval_dataset=dev_p, data_collator=ctc_collate,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
        )
        t0 = time.time()
        stats = trainer.train()
        ft_elapsed = time.time() - t0
        train_loss = float(stats.metrics.get("train_loss", 0.0))
        eval_history = [l for l in trainer.state.log_history if "eval_loss" in l]
        best_eval = min((l["eval_loss"] for l in eval_history), default=None)

        model.save_pretrained(str(finetune_out / "model"))
        processor.save_pretrained(str(finetune_out / "model"))

        results = {
            "experiment": f"ssl_{args.model_key}_ewe",
            "base": args.base,
            "pretrain_epochs": args.pretrain_epochs,
            "finetune_epochs": args.finetune_epochs,
            "train_loss": round(train_loss, 4),
            "best_eval_loss": round(best_eval, 4) if best_eval else None,
            "finetune_time_min": round(ft_elapsed/60, 1),
            "vocab_size": len(vocab),
            "gpu": torch.cuda.get_device_name(0),
            "peak_vram_gb": round(torch.cuda.max_memory_reserved()/1e9, 2),
        }
        metrics_path = output_dir / "metrics.json"
        metrics_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
        print(f"All done. Metrics → {metrics_path}")


if __name__ == "__main__":
    main()
