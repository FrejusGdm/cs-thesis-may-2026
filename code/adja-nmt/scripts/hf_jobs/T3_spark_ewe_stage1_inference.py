#!/usr/bin/env python3
# /// script
# dependencies = ["torch==2.6.0", "torchaudio==2.6.0", "torchvision==0.21.0", "numpy", "sentencepiece", "protobuf", "datasets>=3.4.1,<4.0.0", "huggingface-hub>=0.34.0", "hf_transfer", "soundfile", "librosa", "bitsandbytes", "accelerate", "xformers", "peft", "trl==0.22.2", "triton", "transformers==4.56.2", "omegaconf", "einx", "einops", "torchcodec", "setuptools", "wheel", "pillow", "psutil", "msgspec", "tyro", "cut_cross_entropy"]
# ///
from __future__ import annotations
"""
Spark Ewe Stage 1 — inference-only.

Stage 1 training script (`T3_spark_ewe_stage1.py`) saves the merged model checkpoint
but does not generate audio samples. This inference-only job loads the Stage 1
checkpoint and generates Ewe speech from 5 held-out WaxalNLP Ewe test sentences
via Spark's BiCodec detokenizer, so we can listen to the Stage 1 output the same
way we did with CSM-Ewe Stage 1.

Mirrors the generation path inside `T3_spark_tts_finetune.py` (same prompt format,
same semantic+global token regex extraction, same BiCodec detokenize).

Usage (HF Jobs):

  hf jobs uv run --flavor l40sx1 --timeout 1h --python 3.11 --secrets HF_TOKEN \
    -d scripts/hf_jobs/T3_spark_ewe_stage1_inference.py -- \
    --stage1-checkpoint JosueG/adja-tts-checkpoints/T3_spark_ewe_stage1 \
    --results-prefix T3_spark_ewe_stage1_inference \
    --push-to-hub
"""
import argparse, json, os, re, subprocess, sys, time, unicodedata
from pathlib import Path
sys.stdout.reconfigure(line_buffering=True)

# Same libcuda / python-dev prep as T3_spark_ewe_stage1 (needed for Unsloth+Triton).
for _libdir in ("/usr/lib64-nvidia", "/usr/lib/x86_64-linux-gnu", "/lib/x86_64-linux-gnu"):
    _src = f"{_libdir}/libcuda.so.1"
    _dst = f"{_libdir}/libcuda.so"
    if os.path.exists(_src):
        if not os.path.exists(_dst):
            try: os.symlink(_src, _dst)
            except OSError: pass
        os.environ["LIBRARY_PATH"] = _libdir + os.pathsep + os.environ.get("LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = _libdir + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
        break

os.environ["UNSLOTH_DISABLE_AUTO_UPDATES"] = "1"
print("===== Installing dependencies =====")
os.system("apt-get update -q && apt-get install -y -q git python3-dev libpython3.11-dev >/dev/null 2>&1")


def uv_pip_install_no_deps(packages):
    subprocess.check_call(["uv", "pip", "install", "-q", "--no-deps", *packages])


uv_pip_install_no_deps(["unsloth_zoo", "unsloth"])
uv_pip_install_no_deps(["trl==0.22.2"])
os.system("git clone --depth 1 https://github.com/SparkAudio/Spark-TTS /tmp/Spark-TTS")
print("Dependencies installed.\n")


def parse_args():
    p = argparse.ArgumentParser(description="Spark Ewe Stage 1 inference-only")
    p.add_argument("--stage1-checkpoint",
                   default="JosueG/adja-tts-checkpoints/T3_spark_ewe_stage1",
                   help="Hub repo/subfolder path to Stage 1 merged model")
    p.add_argument("--dataset", default="google/WaxalNLP")
    p.add_argument("--dataset-config", default="ewe_tts")
    p.add_argument("--num-samples", type=int, default=5)
    p.add_argument("--output-dir", default="/tmp/spark_ewe_stage1_inference")
    p.add_argument("--push-to-hub", action="store_true")
    p.add_argument("--results-repo", default="JosueG/adja-tts-results")
    p.add_argument("--results-prefix", default="T3_spark_ewe_stage1_inference")
    return p.parse_args()


def main():
    args = parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN required")

    import numpy as np
    import soundfile as sf
    import torch
    from datasets import load_dataset
    from huggingface_hub import HfApi, snapshot_download

    sys.path.insert(0, "/tmp/Spark-TTS")
    from sparktts.models.audio_tokenizer import BiCodecTokenizer

    from unsloth import FastModel

    assert torch.cuda.is_available(), "CUDA required"
    print(f"torch={torch.__version__}, GPU={torch.cuda.get_device_name(0)}, VRAM={torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB\n")

    # Download Stage 1 checkpoint
    stage1_repo, stage1_subdir = args.stage1_checkpoint.rsplit("/", 1)
    print(f"Downloading Stage 1 checkpoint from {args.stage1_checkpoint}...")
    snapshot_download(stage1_repo, local_dir="/tmp/stage1_spark",
                      allow_patterns=f"{stage1_subdir}/*", token=token)
    stage1_local = f"/tmp/stage1_spark/{stage1_subdir}"

    # BiCodec needs the full Spark-TTS-0.5B directory (not just LLM subdir) for its audio assets
    snapshot_download("unsloth/Spark-TTS-0.5B", local_dir="/tmp/Spark-TTS-0.5B")

    # Load Spark LLM from Stage 1 merged checkpoint (bf16 to fit alongside BiCodec on L40S)
    print(f"Loading Stage 1 Spark LLM from {stage1_local}...")
    model, tokenizer = FastModel.from_pretrained(
        model_name=stage1_local,
        max_seq_length=2048,
        dtype=torch.bfloat16,
        full_finetuning=False,
        load_in_4bit=False,
    )
    FastModel.for_inference(model)
    print(f"Loaded. Loading BiCodec tokenizer...")
    audio_tokenizer = BiCodecTokenizer("/tmp/Spark-TTS-0.5B", "cuda")

    # Load Ewe test sentences
    test_ds = load_dataset(args.dataset, name=args.dataset_config, split="test", token=token)
    text_col = "text" if "text" in test_ds.column_names else "sentence"
    print(f"Ewe test split: {len(test_ds)} rows; using '{text_col}'\n")

    output_dir = Path(args.output_dir)
    generated_dir = output_dir / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    generated = []
    sample_rate = audio_tokenizer.config.get("sample_rate", 16000)
    n = min(args.num_samples, len(test_ds))
    print(f"Generating {n} Ewe samples at {sample_rate} Hz...\n")
    for i in range(n):
        text = unicodedata.normalize("NFC", test_ds[i][text_col])
        t0 = time.time()
        try:
            prompt = "".join(["<|task_tts|>", "<|start_content|>", text, "<|end_content|>", "<|start_global_token|>"])
            inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
            gen_ids = model.generate(
                **inputs, max_new_tokens=2048,
                do_sample=True, temperature=0.8, top_k=50, top_p=1.0,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
            gen_trimmed = gen_ids[:, inputs.input_ids.shape[1]:]
            gen_text = tokenizer.batch_decode(gen_trimmed, skip_special_tokens=False)[0]

            semantic_matches = re.findall(r"<\|bicodec_semantic_(\d+)\|>", gen_text)
            global_matches = re.findall(r"<\|bicodec_global_(\d+)\|>", gen_text)
            if not semantic_matches:
                raise RuntimeError("no semantic tokens in generated output")
            pred_semantic = torch.tensor([int(t) for t in semantic_matches]).long().unsqueeze(0)
            pred_global = (torch.tensor([int(t) for t in global_matches]).long().unsqueeze(0).unsqueeze(0)
                           if global_matches else torch.zeros((1, 1, 1), dtype=torch.long))
            wav_np = audio_tokenizer.detokenize(pred_global.to("cuda").squeeze(0), pred_semantic.to("cuda"))
            wav_path = generated_dir / f"ewe_{i:02d}.wav"
            sf.write(str(wav_path), wav_np, sample_rate)
            gen_time = time.time() - t0
            generated.append({
                "text": text, "file": wav_path.name,
                "duration_sec": round(len(wav_np)/sample_rate, 2),
                "semantic_tokens": len(semantic_matches),
                "global_tokens": len(global_matches),
                "gen_time_sec": round(gen_time, 1),
            })
            print(f"  [{i+1}/{n}] '{text[:60]}' -> {wav_path.name} ({len(wav_np)/sample_rate:.2f}s, {len(semantic_matches)} sem, {len(global_matches)} glob, {gen_time:.1f}s)")
        except Exception as e:
            print(f"  [{i+1}/{n}] ERROR on '{text[:60]}': {e}")
            generated.append({"text": text, "error": str(e)})

    # Metrics + push (wrapped so audio always pushes even if metrics fails)
    results = {
        "experiment": "T3_spark_ewe_stage1_inference",
        "stage1_checkpoint": args.stage1_checkpoint,
        "dataset": f"{args.dataset}/{args.dataset_config}",
        "split": "test",
        "num_samples": n,
        "sample_rate": sample_rate,
        "gpu": torch.cuda.get_device_name(0),
        "generated": generated,
    }
    metrics_path = output_dir / "metrics.json"
    try:
        metrics_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    except Exception as exc:
        print(f"[WARN] metrics.json write failed ({type(exc).__name__}: {exc}). Continuing so audio still pushes.")

    if args.push_to_hub:
        api = HfApi(token=token)
        try:
            api.create_repo(args.results_repo, private=True, exist_ok=True)
        except Exception as exc:
            print(f"[WARN] create_repo failed: {exc}")
        # Audio first
        try:
            api.upload_folder(folder_path=str(generated_dir),
                              path_in_repo=f"{args.results_prefix}/generated",
                              repo_id=args.results_repo, token=token)
            print(f"[push] generated -> {args.results_prefix}/generated")
        except Exception as exc:
            print(f"[WARN] generated upload failed: {exc}")
        try:
            if metrics_path.exists():
                api.upload_file(path_or_fileobj=metrics_path.read_bytes(),
                                path_in_repo=f"{args.results_prefix}/metrics.json",
                                repo_id=args.results_repo, token=token)
                print(f"[push] metrics.json")
        except Exception as exc:
            print(f"[WARN] metrics.json upload failed: {exc}")
        print(f"Results: https://huggingface.co/{args.results_repo}/tree/main/{args.results_prefix}")


if __name__ == "__main__":
    main()
