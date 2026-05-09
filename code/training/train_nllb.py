"""
train_nllb.py — Fine-tune NLLB-200 for French -> Adja translation.

Supports NLLB-200-distilled-600M, NLLB-200-1.3B, and mBART-50.

Usage:
    python train_nllb.py \
        --train-file ./splits/exp1/RANDOM-10K/train.tsv \
        --val-file ./splits/exp1/RANDOM-10K/val.tsv \
        --test-file ./splits/shared/test.tsv \
        --output-dir ./results/exp1/RANDOM-10K/seed42 \
        --model facebook/nllb-200-distilled-600M \
        --seed 42
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)


class TranslationDataset(Dataset):
    """Simple parallel translation dataset from TSV file (src\ttgt per line)."""

    def __init__(self, file_path, tokenizer, max_length=128, src_lang="fra_Latn",
                 tgt_lang="fon_Latn"):
        self.examples = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) >= 2:
                    self.examples.append((parts[0], parts[1]))

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        src, tgt = self.examples[idx]

        self.tokenizer.src_lang = self.src_lang
        model_inputs = self.tokenizer(
            src, max_length=self.max_length, truncation=True, padding=False,
        )

        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                tgt, max_length=self.max_length, truncation=True, padding=False,
            )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


def compute_metrics_factory(tokenizer, tgt_lang="fon_Latn"):
    """Create compute_metrics function with access to tokenizer."""

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        # Replace -100 with pad token id
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Compute chrF using sacrebleu
        try:
            import sacrebleu
            chrf = sacrebleu.corpus_chrf(decoded_preds, [decoded_labels])
            bleu = sacrebleu.corpus_bleu(decoded_preds, [decoded_labels])
            return {
                "bleu": bleu.score,
                "chrf": chrf.score,
                "bleu_signature": str(bleu),
            }
        except ImportError:
            raise ImportError(
                "sacrebleu is required for metric computation. "
                "Install with: pip install sacrebleu"
            )

    return compute_metrics


def translate_test_set(model, tokenizer, test_file, output_file, src_lang, tgt_lang,
                       max_length=128, num_beams=5):
    """Generate translations for the test set and save to file."""
    model.eval()
    device = model.device

    sources = []
    references = []
    with open(test_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                sources.append(parts[0])
                references.append(parts[1])

    predictions = []
    tokenizer.src_lang = src_lang
    forced_bos = tokenizer.convert_tokens_to_ids(tgt_lang)
    if forced_bos == tokenizer.unk_token_id or forced_bos == 0:
        raise ValueError(
            f"Language code '{tgt_lang}' not found in tokenizer vocabulary. "
            f"Check that this is a valid NLLB language tag (e.g., 'fon_Latn')."
        )

    batch_size = 16
    for i in range(0, len(sources), batch_size):
        batch_src = sources[i:i + batch_size]
        inputs = tokenizer(
            batch_src, return_tensors="pt", max_length=max_length,
            truncation=True, padding=True,
        ).to(device)

        with torch.no_grad():
            generated = model.generate(
                **inputs,
                forced_bos_token_id=forced_bos,
                max_length=max_length,
                num_beams=num_beams,
                length_penalty=1.0,
            )

        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        predictions.extend(decoded)

    # Save predictions
    with open(output_file, "w", encoding="utf-8") as f:
        for src, ref, pred in zip(sources, references, predictions):
            f.write(f"{src}\t{ref}\t{pred}\n")

    return sources, references, predictions


def main():
    parser = argparse.ArgumentParser(description="Fine-tune NLLB-200 for Fr->Adja")
    parser.add_argument("--train-file", required=True, help="Training TSV (src\\ttgt)")
    parser.add_argument("--val-file", required=True, help="Validation TSV")
    parser.add_argument("--test-file", help="Test TSV (optional, for final evaluation)")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--model", default="facebook/nllb-200-distilled-600M")
    parser.add_argument("--src-lang", default="fra_Latn")
    parser.add_argument("--tgt-lang", default="fon_Latn", help="Target lang code (Fon as Adja proxy)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--num-beams", type=int, default=5)
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--eval-steps", type=int, default=200)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Save config for reproducibility
    config = vars(args)
    config["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    config["torch_version"] = torch.__version__
    try:
        import transformers
        config["transformers_version"] = transformers.__version__
    except Exception:
        pass
    try:
        import sacrebleu
        config["sacrebleu_version"] = sacrebleu.__version__
    except Exception:
        pass

    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)

    print("Loading datasets...")
    train_dataset = TranslationDataset(
        args.train_file, tokenizer, args.max_length, args.src_lang, args.tgt_lang
    )
    val_dataset = TranslationDataset(
        args.val_file, tokenizer, args.max_length, args.src_lang, args.tgt_lang
    )
    print(f"  Train: {len(train_dataset)} pairs")
    print(f"  Val: {len(val_dataset)} pairs")

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.max_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="linear",
        label_smoothing_factor=args.label_smoothing,
        weight_decay=0.01,
        fp16=args.fp16 and torch.cuda.is_available(),
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="chrf",
        greater_is_better=True,
        predict_with_generate=True,
        generation_max_length=args.max_length,
        generation_num_beams=args.num_beams,
        seed=args.seed,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_factory(tokenizer, args.tgt_lang),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )

    print("Starting training...")
    start_time = time.time()
    train_result = trainer.train()
    training_time = time.time() - start_time

    print(f"Training completed in {training_time:.0f}s ({training_time/3600:.1f}h)")

    # Save training results
    metrics = train_result.metrics
    metrics["training_time_seconds"] = training_time
    trainer.save_metrics("train", metrics)
    trainer.save_model(os.path.join(args.output_dir, "best_model"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "best_model"))

    # Evaluate on validation
    val_metrics = trainer.evaluate()
    trainer.save_metrics("eval", val_metrics)
    print(f"Val metrics: {val_metrics}")

    # Generate translations on test set if provided
    if args.test_file and os.path.exists(args.test_file):
        print("Translating test set...")
        pred_file = os.path.join(args.output_dir, "predictions.tsv")
        sources, references, predictions = translate_test_set(
            model, tokenizer, args.test_file, pred_file,
            args.src_lang, args.tgt_lang, args.max_length, args.num_beams
        )
        print(f"Predictions saved to {pred_file}")

        # Compute test metrics
        try:
            import sacrebleu
            bleu = sacrebleu.corpus_bleu(predictions, [references])
            chrf = sacrebleu.corpus_chrf(predictions, [references])
            chrfpp = sacrebleu.corpus_chrf(predictions, [references], word_order=2)
            ter = sacrebleu.corpus_ter(predictions, [references])

            test_metrics = {
                "test_bleu": bleu.score,
                "test_chrf": chrf.score,
                "test_chrfpp": chrfpp.score,
                "test_ter": ter.score,
                "test_bleu_signature": str(bleu),
                "test_n_samples": len(predictions),
            }

            with open(os.path.join(args.output_dir, "test_metrics.json"), "w") as f:
                json.dump(test_metrics, f, indent=2)
            print(f"Test metrics: BLEU={bleu.score:.1f}, chrF={chrf.score:.1f}, "
                  f"chrF++={chrfpp.score:.1f}, TER={ter.score:.1f}")
        except ImportError:
            print("WARNING: sacrebleu not installed, skipping test metrics")

    print(f"\nAll outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
