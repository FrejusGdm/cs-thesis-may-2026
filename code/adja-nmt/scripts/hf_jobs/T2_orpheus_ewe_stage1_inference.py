#!/usr/bin/env python3
# /// script
# dependencies = ["torch==2.6.0", "torchaudio==2.6.0", "transformers==4.56.2", "peft>=0.11.0,<0.16.0", "accelerate", "datasets>=3.4.1,<4.0.0", "huggingface-hub>=0.34.0", "hf_transfer", "soundfile", "librosa", "numpy", "scipy", "sentencepiece", "protobuf", "bitsandbytes", "snac"]
# ///
from __future__ import annotations
"""
Orpheus Ewe Stage 1 — inference-only.

Stage 1 training scripts (`T2_orpheus_ewe_stage1.py`) intentionally skip audio
generation to keep Stage 1 wall time short — generation was deferred to Stage 2.
But that left us with no way to listening-test the Ewe checkpoint itself, which
we need to judge whether the Gbe-family Stage 1 actually produced intelligible
speech (the way CSM's Stage 1 audibly did).

This script just loads a Stage 1 checkpoint, runs generation on 5 held-out Ewe
test-split sentences from WaxalNLP, and pushes the wavs to Hub for listening.

Matches the generation path inside `T2_orpheus_ewe_adja_stage2.py` (same prompt
format, same SNAC decode, same hyperparameters) — identical model behaviour, just
without the fine-tuning step.

Usage (HF Jobs):

  hf jobs uv run --flavor l40sx1 --timeout 1h --python 3.11 --secrets HF_TOKEN \
    -d scripts/hf_jobs/T2_orpheus_ewe_stage1_inference.py -- \
    --stage1-checkpoint JosueG/adja-tts-checkpoints/T2_orpheus_en_ewe_stage1 \
    --checkpoint-tag en \
    --results-prefix T2_orpheus_en_ewe_stage1_inference \
    --push-to-hub
"""
import argparse, json, os, sys, time
from pathlib import Path
sys.stdout.reconfigure(line_buffering=True)

TOKENISER_LENGTH = 128256
END_OF_TEXT      = 128009
START_OF_SPEECH  = TOKENISER_LENGTH + 1
END_OF_SPEECH    = TOKENISER_LENGTH + 2
START_OF_HUMAN   = TOKENISER_LENGTH + 3
END_OF_HUMAN     = TOKENISER_LENGTH + 4
START_OF_AI      = TOKENISER_LENGTH + 5
END_OF_AI        = TOKENISER_LENGTH + 6
PAD_TOKEN        = TOKENISER_LENGTH + 7
AUDIO_TOKENS_START = TOKENISER_LENGTH + 10


def parse_args():
    p = argparse.ArgumentParser(description="Orpheus Ewe Stage 1 inference-only")
    p.add_argument("--stage1-checkpoint",
                   default="JosueG/adja-tts-checkpoints/T2_orpheus_en_ewe_stage1",
                   help="Hub repo/subfolder path to Stage 1 merged model")
    p.add_argument("--checkpoint-tag", default="en", help="en|fr|zh — naming only")
    p.add_argument("--dataset", default="google/WaxalNLP")
    p.add_argument("--dataset-config", default="ewe_tts")
    p.add_argument("--num-samples", type=int, default=5)
    p.add_argument("--output-dir", default="/tmp/orpheus_ewe_stage1_inference")
    p.add_argument("--push-to-hub", action="store_true")
    p.add_argument("--results-repo", default="JosueG/adja-tts-results")
    p.add_argument("--results-prefix", default="T2_orpheus_en_ewe_stage1_inference")
    return p.parse_args()


def redistribute_codes(flat):
    import torch
    l1, l2, l3 = [], [], []
    for i in range((len(flat)+1)//7):
        b = 7*i
        l1.append(flat[b]   - AUDIO_TOKENS_START)
        l2.append(flat[b+1] - AUDIO_TOKENS_START - 4096)
        l3.append(flat[b+2] - AUDIO_TOKENS_START - 2*4096)
        l3.append(flat[b+3] - AUDIO_TOKENS_START - 3*4096)
        l2.append(flat[b+4] - AUDIO_TOKENS_START - 4*4096)
        l3.append(flat[b+5] - AUDIO_TOKENS_START - 5*4096)
        l3.append(flat[b+6] - AUDIO_TOKENS_START - 6*4096)
    return [torch.tensor(l1).unsqueeze(0), torch.tensor(l2).unsqueeze(0), torch.tensor(l3).unsqueeze(0)]


def main():
    args = parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN required")

    import numpy as np, soundfile as sf, torch
    from datasets import load_dataset
    from huggingface_hub import HfApi, snapshot_download
    from snac import SNAC
    from transformers import AutoModelForCausalLM, AutoTokenizer

    assert torch.cuda.is_available(), "CUDA required"
    print(f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB\n")

    # Download Stage 1 checkpoint
    stage1_repo, stage1_subdir = args.stage1_checkpoint.rsplit("/", 1)
    print(f"Downloading Stage 1 checkpoint from {args.stage1_checkpoint}...")
    snapshot_download(stage1_repo, local_dir="/tmp/stage1_orpheus",
                      allow_patterns=f"{stage1_subdir}/*", token=token)
    stage1_local = f"/tmp/stage1_orpheus/{stage1_subdir}"

    # Load Ewe test texts
    test_ds = load_dataset(args.dataset, name=args.dataset_config, split="test", token=token)
    text_col = "text" if "text" in test_ds.column_names else "sentence"
    print(f"Ewe test split: {len(test_ds)} rows; using column '{text_col}'")

    # Load model + tokenizer + SNAC codec
    print(f"Loading Stage 1 model from {stage1_local}...")
    tokenizer = AutoTokenizer.from_pretrained(stage1_local, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = PAD_TOKEN
    model = AutoModelForCausalLM.from_pretrained(stage1_local, torch_dtype=torch.bfloat16, token=token).to("cuda").eval()
    snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to("cpu").eval()
    print(f"Loaded {sum(p.numel() for p in model.parameters())/1e9:.2f}B params\n")

    output_dir = Path(args.output_dir)
    generated_dir = output_dir / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    # Generate
    start_tok = torch.tensor([[START_OF_HUMAN]], dtype=torch.int64)
    end_toks = torch.tensor([[END_OF_TEXT, END_OF_HUMAN]], dtype=torch.int64)
    generated = []
    n = min(args.num_samples, len(test_ds))
    print(f"Generating {n} Ewe samples...\n")
    for i in range(n):
        text = test_ds[i][text_col]
        t0 = time.time()
        try:
            ids = tokenizer(text, return_tensors="pt").input_ids
            prompt = torch.cat([start_tok, ids, end_toks], dim=1).to(model.device)
            with torch.no_grad():
                out = model.generate(
                    input_ids=prompt, attention_mask=torch.ones_like(prompt),
                    max_new_tokens=1200, do_sample=True, temperature=0.6,
                    top_p=0.95, repetition_penalty=1.1, eos_token_id=END_OF_SPEECH,
                )
            row = out[0].tolist()
            if START_OF_SPEECH in row:
                row = row[len(row) - 1 - row[::-1].index(START_OF_SPEECH):][1:]
            row = [t for t in row if t != END_OF_SPEECH]
            row = row[:(len(row)//7)*7]
            if not row:
                raise RuntimeError("no audio frames")
            codes = redistribute_codes(row)
            snac_model.to("cuda")
            with torch.inference_mode():
                wav = snac_model.decode([c.to("cuda") for c in codes]).detach().squeeze().cpu().numpy().astype(np.float32)
            snac_model.to("cpu")
            p = generated_dir / f"ewe_{i:02d}.wav"
            sf.write(str(p), wav, 24000)
            gen_time = time.time() - t0
            generated.append({
                "text": text, "file": p.name,
                "duration_sec": round(float(wav.size)/24000, 2),
                "gen_time_sec": round(gen_time, 1),
            })
            print(f"  [{i+1}/{n}] '{text[:60]}' -> {p.name} ({wav.size/24000:.2f}s audio, {gen_time:.1f}s gen)")
        except Exception as e:
            print(f"  [{i+1}/{n}] ERROR on '{text[:60]}': {e}")
            generated.append({"text": text, "error": str(e)})

    # Save + push
    results = {
        "experiment": f"T2_orpheus_{args.checkpoint_tag}_ewe_stage1_inference",
        "stage1_checkpoint": args.stage1_checkpoint,
        "checkpoint_tag": args.checkpoint_tag,
        "dataset": f"{args.dataset}/{args.dataset_config}",
        "split": "test",
        "num_samples": n,
        "gpu": torch.cuda.get_device_name(0),
        "generated": generated,
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nMetrics saved to {metrics_path}")

    if args.push_to_hub:
        api = HfApi(token=token)
        api.create_repo(args.results_repo, private=True, exist_ok=True)
        api.upload_file(path_or_fileobj=metrics_path.read_bytes(),
                        path_in_repo=f"{args.results_prefix}/metrics.json",
                        repo_id=args.results_repo, token=token)
        api.upload_folder(folder_path=str(generated_dir),
                          path_in_repo=f"{args.results_prefix}/generated",
                          repo_id=args.results_repo, token=token)
        print(f"Results: https://huggingface.co/{args.results_repo}/tree/main/{args.results_prefix}")


if __name__ == "__main__":
    main()
