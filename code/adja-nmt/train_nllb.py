#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "transformers==4.44.2",
#   "datasets>=2.21.0",
#   "sacrebleu>=2.4.2",
#   "torch>=2.2.0",
#   "tqdm>=4.66.0",
#   "numpy>=1.26.0",
#   "huggingface_hub>=0.24.0",
#   "trackio>=0.0.20",
# ]
# ///

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import sacrebleu
import torch
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    Adafactor,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    get_constant_schedule_with_warmup,
)

DATASET_ID = "JosueG/adja-fr-mt-acl-paper-private"
MODEL_ID = "facebook/nllb-200-distilled-600M"
SRC_LANG = "fra_Latn"
TGT_LANG = "aj_Latn"
CUSTOM_TOKEN = "aj_Latn"
DONOR_TOKEN = "ewe_Latn"
SEED = 42


@dataclass
class Batch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--hf-token", type=str, default=os.getenv("HF_TOKEN", ""))
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--max-source-length", type=int, default=128)
    p.add_argument("--max-target-length", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--trackio-name", type=str, default="nllb-fra-aj-smoke")
    p.add_argument("--trackio-project", type=str, default="adja-mt")
    p.add_argument("--trackio-entity", type=str, default=os.getenv("TRACKIO_ENTITY", "JosueG"))
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def nfc(text: Any) -> str:
    return str(text).strip().replace("\u00a0", " ")


def load_data(token: str):
    ds = load_dataset(DATASET_ID, token=token or None)
    return ds["train"], ds["validation"], ds["test"]


def get_trackio_module():
    try:
        import trackio
        return trackio
    except Exception:
        try:
            import wandb as trackio  # fallback for W&B-compatible Trackio API
            return trackio
        except Exception:
            return None


def build_model_and_tokenizer(token: str):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=token or None, src_lang=SRC_LANG, tgt_lang=TGT_LANG)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID, token=token or None)
    if CUSTOM_TOKEN not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": [CUSTOM_TOKEN]})
        model.resize_token_embeddings(len(tokenizer))
    fix_tokenizer(model, tokenizer)
    return model, tokenizer


def fix_tokenizer(model, tokenizer):
    token_id = tokenizer.convert_tokens_to_ids(CUSTOM_TOKEN)
    donor_id = tokenizer.convert_tokens_to_ids(DONOR_TOKEN)
    emb = model.get_input_embeddings().weight.data
    if token_id is None or donor_id is None:
        return
    if token_id >= emb.shape[0] or donor_id >= emb.shape[0]:
        return
    with torch.no_grad():
        emb[token_id].copy_(emb[donor_id].clone())


def encode_examples(examples, tokenizer, max_source_length: int, max_target_length: int):
    sources = [nfc(x) for x in examples["fra_Latn"]]
    targets = [nfc(x) for x in examples["aj_Latn"]]
    model_inputs = tokenizer(sources, max_length=max_source_length, truncation=True)
    labels = tokenizer(text_target=targets, max_length=max_target_length, truncation=True)
    label_ids = []
    for seq in labels["input_ids"]:
        label_ids.append([tok if tok != tokenizer.pad_token_id else -100 for tok in seq])
    model_inputs["labels"] = label_ids
    return model_inputs


def collate_fn(features: List[Dict[str, Any]], tokenizer) -> Batch:
    input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
    attention_mask = [torch.tensor(f["attention_mask"], dtype=torch.long) for f in features]
    labels = [torch.tensor(f["labels"], dtype=torch.long) for f in features]
    return Batch(
        input_ids=pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id),
        attention_mask=pad_sequence(attention_mask, batch_first=True, padding_value=0),
        labels=pad_sequence(labels, batch_first=True, padding_value=-100),
    )


def make_loader(ds, tokenizer, batch_size, shuffle, num_workers, max_source_length, max_target_length):
    cols = ["fra_Latn", "aj_Latn"]
    ds = ds.remove_columns([c for c in ds.column_names if c not in cols])
    ds = ds.map(
        lambda ex: encode_examples(ex, tokenizer, max_source_length, max_target_length),
        batched=True,
        remove_columns=ds.column_names,
        desc="Tokenizing",
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda feats: collate_fn(feats, tokenizer),
    )


def evaluate(model, tokenizer, dataloader, device) -> Dict[str, Any]:
    model.eval()
    preds: List[str] = []
    refs: List[str] = []
    losses: List[float] = []
    bos_id = tokenizer.convert_tokens_to_ids(TGT_LANG)
    for batch in dataloader:
        batch = Batch(**{k: v.to(device) for k, v in batch.__dict__.items()})
        with torch.no_grad():
            outputs = model(input_ids=batch.input_ids, attention_mask=batch.attention_mask, labels=batch.labels)
            losses.append(outputs.loss.item())
            generated = model.generate(
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
                forced_bos_token_id=bos_id,
                max_new_tokens=128,
            )
        decoded_preds = tokenizer.batch_decode(generated, skip_special_tokens=True)
        label_ids = batch.labels.clone()
        label_ids[label_ids == -100] = tokenizer.pad_token_id
        decoded_refs = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        preds.extend([nfc(x) for x in decoded_preds])
        refs.extend([nfc(x) for x in decoded_refs])
    chrf = sacrebleu.corpus_chrf(preds, [refs]).score
    bleu = sacrebleu.corpus_bleu(preds, [refs], tokenize="flores200").score
    return {"loss": float(np.mean(losses)) if losses else math.nan, "chrf": chrf, "spbleu": bleu, "preds": preds, "refs": refs}


def maybe_init_trackio(name: str, project: str, entity: str):
    trackio = get_trackio_module()
    if trackio is None:
        return None
    try:
        return trackio.init(project=project, name=name, entity=entity)
    except Exception:
        return None


def log_trackio(run, metrics: Dict[str, Any], step: int):
    if run is None:
        return
    try:
        run.log({**metrics, "step": step})
    except Exception:
        pass


def main() -> None:
    args = parse_args()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds, val_ds, test_ds = load_data(args.hf_token)
    model, tokenizer = build_model_and_tokenizer(args.hf_token)
    model.to(device)
    model.train()

    train_loader = make_loader(train_ds, tokenizer, args.batch_size, True, args.num_workers, args.max_source_length, args.max_target_length)
    val_loader = make_loader(val_ds, tokenizer, args.batch_size, False, args.num_workers, args.max_source_length, args.max_target_length)
    test_loader = make_loader(test_ds, tokenizer, args.batch_size, False, args.num_workers, args.max_source_length, args.max_target_length)

    optimizer = Adafactor(
        model.parameters(),
        lr=args.lr,
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
        clip_threshold=1.0,
        decay_rate=-0.8,
        beta1=None,
        weight_decay=0.0,
        eps=(1e-30, 1e-3),
    )
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    trackio_run = maybe_init_trackio(args.trackio_name, args.trackio_project, args.trackio_entity)

    best_chrf = -1.0
    best_step = -1
    global_step = 0
    train_iter = iter(train_loader)
    pbar = tqdm(total=args.max_steps, disable=True)
    while global_step < args.max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        batch = Batch(**{k: v.to(device) for k, v in batch.__dict__.items()})
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32):
            outputs = model(input_ids=batch.input_ids, attention_mask=batch.attention_mask, labels=batch.labels)
            loss = outputs.loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        global_step += 1
        pbar.update(1)
        if global_step == 1 or global_step % 5 == 0:
            print(json.dumps({"step": global_step, "train_loss": float(loss.item()), "lr": scheduler.get_last_lr()[0]}))
            log_trackio(trackio_run, {"train/loss": float(loss.item()), "train/lr": scheduler.get_last_lr()[0]}, global_step)
        if global_step % args.eval_every == 0 or (args.smoke and global_step == args.max_steps):
            metrics = evaluate(model, tokenizer, val_loader, device)
            print(json.dumps({"step": global_step, "val_loss": metrics["loss"], "val_chrf": metrics["chrf"], "val_spbleu": metrics["spbleu"]}))
            log_trackio(trackio_run, {"val/loss": metrics["loss"], "val/chrf": metrics["chrf"], "val/spbleu": metrics["spbleu"]}, global_step)
            if metrics["chrf"] > best_chrf:
                best_chrf = metrics["chrf"]
                best_step = global_step
                torch.save(model.state_dict(), out_dir / "best_model.pt")
                tokenizer.save_pretrained(out_dir / "tokenizer")
            if args.smoke:
                break

    test_metrics = evaluate(model, tokenizer, test_loader, device)
    payload = {
        "dataset": DATASET_ID,
        "model": MODEL_ID,
        "src_lang": SRC_LANG,
        "tgt_lang": TGT_LANG,
        "custom_token": CUSTOM_TOKEN,
        "donor_token": DONOR_TOKEN,
        "best_val_chrf": best_chrf,
        "best_step": best_step,
        "test_loss": test_metrics["loss"],
        "test_chrf": test_metrics["chrf"],
        "test_spbleu": test_metrics["spbleu"],
        "max_steps": args.max_steps,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "eval_every": args.eval_every,
        "smoke": args.smoke,
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    with open(out_dir / "samples.json", "w", encoding="utf-8") as f:
        json.dump({"preds": test_metrics["preds"][:10], "refs": test_metrics["refs"][:10]}, f, ensure_ascii=False, indent=2)
    if trackio_run is not None:
        try:
            trackio_run.log({"test/chrf": test_metrics["chrf"], "test/spbleu": test_metrics["spbleu"]})
            trackio_run.finish()
        except Exception:
            pass
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
