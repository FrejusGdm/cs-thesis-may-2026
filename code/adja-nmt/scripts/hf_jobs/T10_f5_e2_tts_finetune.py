#!/usr/bin/env python3
# /// script
# dependencies = ["torch==2.5.1", "torchaudio==2.5.1", "huggingface-hub>=0.34.0", "datasets>=3.4.1,<4.0.0", "soundfile", "librosa", "numpy", "pyarrow", "cached_path", "safetensors"]
# ///
from __future__ import annotations

"""
T10/T11: Fine-tune F5-TTS or E2-TTS on Adja via Hugging Face Jobs.

The upstream repo expects a preprocessed dataset bundle:
  - raw.arrow
  - duration.json
  - vocab.txt

This launcher:
  1. clones the upstream repo
  2. materializes Adja into the upstream dataset format
  3. expands pretrained text embeddings when Adja adds new symbols
  4. runs the upstream finetune CLI for a small real smoke or a longer run
"""

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import time
import unicodedata
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)


UPSTREAM_REPO = "https://github.com/SWivid/F5-TTS.git"
UPSTREAM_DIR = Path("/tmp/F5-TTS")
DATA_ROOT = Path("/tmp/f5_adja")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="T10/T11 F5/E2 fine-tune on Adja")
    parser.add_argument("--model-family", default="F5TTS_Base", choices=["F5TTS_v1_Base", "F5TTS_Base", "E2TTS_Base"])
    parser.add_argument("--dataset", default="JosueG/adja-tts-orpheus")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--batch-size-per-gpu", type=int, default=1200)
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--results-repo", default="JosueG/adja-tts-results")
    parser.add_argument("--results-prefix", default="T10_f5")
    parser.add_argument("--output-dir", default="/tmp/t10_f5_output")
    return parser.parse_args()


def run(cmd: str, cwd: str | None = None) -> None:
    print(f"$ {cmd}")
    subprocess.check_call(cmd, shell=True, cwd=cwd)

def install_env() -> None:
    run("apt-get update -q && apt-get install -y -q git ffmpeg")


def clone_upstream() -> None:
    if UPSTREAM_DIR.exists():
        shutil.rmtree(UPSTREAM_DIR)
    run(f'git clone --depth 1 "{UPSTREAM_REPO}" "{UPSTREAM_DIR}"')
    run("uv pip install -q -e .", cwd=str(UPSTREAM_DIR))


def nfc(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text.strip()).split())


def materialize_dataset(args: argparse.Namespace, token: str) -> tuple[Path, list[str]]:
    import soundfile as sf
    from datasets import load_dataset

    target_dir = DATA_ROOT / "Adja_char"
    if target_dir.exists():
        shutil.rmtree(target_dir)
    wav_dir = target_dir / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(args.dataset, token=token, split="train")
    split1 = ds.train_test_split(test_size=0.1, seed=args.seed)
    split2 = split1["train"].train_test_split(test_size=0.1 / 0.9, seed=args.seed)
    train_ds = split2["train"]
    if args.dry_run:
        train_ds = train_ds.select(range(min(12, len(train_ds))))

    metadata_path = target_dir / "metadata.csv"
    rows: list[str] = []
    vocab = {" "}
    kept = 0
    for idx, example in enumerate(train_ds):
        text = nfc(example["text"])
        if not text:
            continue
        wav_path = wav_dir / f"clip_{idx:06d}.wav"
        audio = example["audio"]["array"]
        sr = example["audio"]["sampling_rate"]
        sf.write(str(wav_path), audio, sr)
        rows.append(f"{wav_path}|{text}")
        vocab.update(text)
        kept += 1

    metadata_path.write_text("audio_file|text\n" + "\n".join(rows) + "\n", encoding="utf-8")
    return target_dir, sorted(vocab)


def preprocess_f5_dataset(dataset_dir: Path) -> Path:
    prep_script = UPSTREAM_DIR / "src" / "f5_tts" / "train" / "datasets" / "prepare_csv_wavs.py"
    run(f'python3 "{prep_script}" "{dataset_dir / "metadata.csv"}" "{dataset_dir}" --workers 4')
    return dataset_dir


def base_checkpoint_uri(model_family: str) -> str:
    if model_family == "F5TTS_v1_Base":
        return "hf://SWivid/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors"
    if model_family == "F5TTS_Base":
        return "hf://SWivid/F5-TTS/F5TTS_Base/model_1200000.pt"
    return "hf://SWivid/E2-TTS/E2TTS_Base/model_1200000.pt"


def model_name_for_results(model_family: str) -> str:
    return "T10" if model_family.startswith("F5") else "T11"


def ensure_vocab_extension(dataset_dir: Path, model_family: str) -> tuple[Path | None, list[str]]:
    from cached_path import cached_path
    from safetensors.torch import load_file, save_file
    import torch

    vocab_path = dataset_dir / "vocab.txt"
    dataset_vocab = [line.rstrip("\n") for line in vocab_path.read_text(encoding="utf-8").splitlines() if line != ""]
    dataset_vocab_set = set(dataset_vocab)

    base_ckpt = Path(str(cached_path(base_checkpoint_uri(model_family))))
    if base_ckpt.suffix == ".safetensors":
        ckpt = load_file(str(base_ckpt), device="cpu")
        ema_sd = ckpt
    else:
        ckpt = torch.load(str(base_ckpt), map_location="cpu", weights_only=False)
        ema_sd = ckpt["ema_model_state_dict"]

    embed_key = "ema_model.transformer.text_embed.text_embed.weight"
    old_embed = ema_sd[embed_key]
    vocab_old = old_embed.size(0)

    # Upstream uses index 0 as unknown / space in the char vocab. We extend only
    # when the dataset vocab is larger than the released checkpoint vocabulary.
    missing_count = max(0, len(dataset_vocab_set) - vocab_old)
    missing_symbols: list[str] = []
    if missing_count <= 0:
        return None, missing_symbols

    seen = set()
    base_vocab_file = dataset_dir / "vocab.txt"
    for symbol in dataset_vocab:
        if symbol not in seen:
            seen.add(symbol)
            missing_symbols.append(symbol)
    missing_symbols = missing_symbols[vocab_old:]

    torch.manual_seed(666)
    new_embed = torch.zeros((vocab_old + missing_count, old_embed.size(1)))
    new_embed[:vocab_old] = old_embed
    new_embed[vocab_old:] = torch.randn((missing_count, old_embed.size(1)))
    ema_sd[embed_key] = new_embed

    ckpt_dir = Path("/tmp/f5_ckpts") / dataset_dir.name.replace("_char", "")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    new_ckpt = ckpt_dir / f"pretrained_{base_ckpt.name}"
    if new_ckpt.suffix == ".safetensors":
        save_file(ema_sd, str(new_ckpt))
    else:
        torch.save(ckpt, str(new_ckpt))
    return new_ckpt, missing_symbols


def run_training(args: argparse.Namespace, dataset_dir: Path, pretrain_ckpt: Path | None) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "ckpts" / dataset_dir.name.replace("_char", "")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    data_mirror_dir = UPSTREAM_DIR / "data" / dataset_dir.name
    if data_mirror_dir.exists():
        shutil.rmtree(data_mirror_dir)
    data_mirror_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(dataset_dir, data_mirror_dir)
    os.chdir(UPSTREAM_DIR)
    sys.path.insert(0, str(UPSTREAM_DIR / "src"))

    from cached_path import cached_path
    from f5_tts.model import CFM, DiT, Trainer, UNetT
    from f5_tts.model.dataset import load_dataset
    from f5_tts.model.utils import get_tokenizer
    import torch

    if args.model_family == "F5TTS_v1_Base":
        model_cls = DiT
        model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
    elif args.model_family == "F5TTS_Base":
        model_cls = DiT
        model_cfg = dict(
            dim=1024,
            depth=22,
            heads=16,
            ff_mult=2,
            text_dim=512,
            text_mask_padding=False,
            conv_layers=4,
            pe_attn_head=1,
        )
    else:
        model_cls = UNetT
        model_cfg = dict(
            dim=1024,
            depth=24,
            heads=16,
            ff_mult=4,
            text_mask_padding=False,
            pe_attn_head=1,
        )

    vocab_char_map, vocab_size = get_tokenizer(str(dataset_dir / "vocab.txt"), "custom")
    mel_spec_kwargs = dict(
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        n_mel_channels=100,
        target_sample_rate=24000,
        mel_spec_type="vocos",
    )
    model = CFM(
        transformer=model_cls(**model_cfg, text_num_embeds=vocab_size, mel_dim=100),
        mel_spec_kwargs=mel_spec_kwargs,
        vocab_char_map=vocab_char_map,
    )

    # ema-pytorch deepcopies the model by default, but the current F5 stack is not
    # deepcopy-safe in this environment. Pass an explicit fresh module with the same
    # architecture so the upstream Trainer can still manage EMA state.
    ema_model = CFM(
        transformer=model_cls(**model_cfg, text_num_embeds=vocab_size, mel_dim=100),
        mel_spec_kwargs=mel_spec_kwargs,
        vocab_char_map=vocab_char_map,
    )

    pretrain_path = str(pretrain_ckpt) if pretrain_ckpt is not None else str(cached_path(base_checkpoint_uri(args.model_family)))
    if not pretrain_path.startswith(str(ckpt_dir)):
        file_checkpoint = os.path.basename(pretrain_path)
        if not file_checkpoint.startswith("pretrained_"):
            file_checkpoint = "pretrained_" + file_checkpoint
        copied_ckpt = ckpt_dir / file_checkpoint
        if not copied_ckpt.exists():
            shutil.copy2(pretrain_path, copied_ckpt)
            print("copy checkpoint for finetune")
        pretrain_path = str(copied_ckpt)

    trainer = Trainer(
        model,
        1 if args.dry_run else args.epochs,
        args.learning_rate,
        num_warmup_updates=10 if args.dry_run else 200,
        save_per_updates=2 if args.dry_run else 200,
        keep_last_n_checkpoints=2,
        checkpoint_path=str(ckpt_dir),
        batch_size_per_gpu=600 if args.dry_run else args.batch_size_per_gpu,
        batch_size_type="frame",
        max_samples=8 if args.dry_run else args.max_samples,
        grad_accumulation_steps=args.grad_accum,
        max_grad_norm=1.0,
        logger=None,
        last_per_updates=2 if args.dry_run else 100,
        ema_kwargs={"ema_model": ema_model},
    )

    # The upstream CustomDataset path currently appends `_{tokenizer}` internally, so
    # pass the base dataset stem without the trailing `_char` suffix to avoid ending up
    # at `Adja_char_char`.
    train_dataset = load_dataset(
        dataset_dir.name.replace("_char", ""),
        "char",
        dataset_type="CustomDataset",
        mel_spec_kwargs=mel_spec_kwargs,
    )
    # Seed the checkpoint dir with the pretrained weights so the upstream Trainer
    # load_checkpoint() path behaves exactly like finetune_cli.py.
    trainer.train(train_dataset, resumable_with_seed=666)


def generate_audio_samples(
    args: argparse.Namespace,
    dataset_dir: Path,
    token: str,
) -> tuple[Path, list[dict]]:
    import soundfile as sf
    import torch
    from datasets import load_dataset as hf_load_dataset

    sys.path.insert(0, str(UPSTREAM_DIR / "src"))
    from f5_tts.infer.utils_infer import infer_process, load_model, load_vocoder
    from f5_tts.model import DiT, UNetT

    output_dir = Path(args.output_dir)
    gen_dir = output_dir / "generated_audio"
    gen_dir.mkdir(parents=True, exist_ok=True)

    ckpt_dir = output_dir / "ckpts" / dataset_dir.name.replace("_char", "")
    candidates = sorted(ckpt_dir.glob("model_last.pt")) + sorted(ckpt_dir.glob("update_*.pt"))
    if not candidates:
        print("No checkpoint found — skipping audio generation")
        return gen_dir, []
    latest_ckpt = candidates[-1]
    print(f"Loading checkpoint for inference: {latest_ckpt}")

    if args.model_family in ("F5TTS_v1_Base", "F5TTS_Base"):
        model_cls = DiT
        model_cfg = dict(
            dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512,
            text_mask_padding=False, conv_layers=4, pe_attn_head=1,
        )
    else:
        model_cls = UNetT
        model_cfg = dict(dim=1024, depth=24, heads=16, ff_mult=4, text_mask_padding=False, pe_attn_head=1)

    vocab_path = str(dataset_dir / "vocab.txt")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_model(
        model_cls, model_cfg, str(latest_ckpt),
        mel_spec_type="vocos", vocab_file=vocab_path, use_ema=True, device=device,
    )
    vocoder = load_vocoder(vocoder_name="vocos", is_local=False, device=device)

    ds = hf_load_dataset(args.dataset, token=token, split="train")
    test_ds = ds.train_test_split(test_size=0.1, seed=args.seed)["test"]
    test_ds = test_ds.select(range(min(6, len(test_ds))))

    ref = test_ds[0]
    ref_wav_path = str(gen_dir / "ref.wav")
    sf.write(ref_wav_path, ref["audio"]["array"], ref["audio"]["sampling_rate"])
    ref_text = nfc(ref["text"])
    print(f"Reference voice: '{ref_text[:60]}'")

    generated = []
    for i in range(1, len(test_ds)):
        example = test_ds[i]
        gen_text = nfc(example["text"])
        out_path = str(gen_dir / f"sample_{i:02d}.wav")
        try:
            audio, sr, _ = infer_process(
                ref_audio=ref_wav_path,
                ref_text=ref_text,
                gen_text=gen_text,
                model_obj=model,
                vocoder=vocoder,
                cross_fade_duration=0.15,
                speed=1.0,
            )
            sf.write(out_path, audio, sr)
            dur = len(audio) / sr
            generated.append({"text": gen_text, "file": f"sample_{i:02d}.wav", "duration_sec": round(dur, 2)})
            print(f"  [{i}] '{gen_text[:40]}' -> {dur:.1f}s")
        except Exception as e:
            print(f"  [{i}] FAILED: {e}")
            generated.append({"text": gen_text, "error": str(e)})

    return gen_dir, generated


def collect_metrics(
    args: argparse.Namespace,
    dataset_dir: Path,
    elapsed_min: float,
    missing_symbols: list[str],
    pretrain_ckpt: Path | None,
    generated: list[dict],
) -> Path:
    output_dir = Path(args.output_dir)
    ckpt_dir = output_dir / "ckpts" / dataset_dir.name.replace("_char", "")
    metrics = {
        "experiment": model_name_for_results(args.model_family),
        "model_family": args.model_family,
        "dataset": args.dataset,
        "seed": args.seed,
        "dry_run": args.dry_run,
        "elapsed_min": round(elapsed_min, 1),
        "dataset_dir": str(dataset_dir),
        "vocab_path": str(dataset_dir / "vocab.txt"),
        "pretrain_checkpoint": str(pretrain_ckpt) if pretrain_ckpt is not None else None,
        "missing_symbols_extended": missing_symbols,
        "checkpoint_dir_exists": ckpt_dir.exists(),
        "checkpoint_files": sorted(str(p.relative_to(ckpt_dir)) for p in ckpt_dir.rglob("*") if p.is_file()) if ckpt_dir.exists() else [],
        "generated_audio": generated,
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return metrics_path


def push_results(args: argparse.Namespace, token: str, metrics_path: Path) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(args.results_repo, private=True, exist_ok=True)
    prefix = args.results_prefix.rstrip("/")
    api.upload_file(
        path_or_fileobj=metrics_path.read_bytes(),
        path_in_repo=f"{prefix}/metrics.json",
        repo_id=args.results_repo,
        token=token,
    )
    ckpt_dir = Path(args.output_dir) / "ckpts"
    if ckpt_dir.exists():
        try:
            api.upload_folder(
                folder_path=str(ckpt_dir),
                path_in_repo=f"{prefix}/checkpoints",
                repo_id=args.results_repo,
                token=token,
            )
        except Exception as exc:
            print(f"Checkpoint upload best-effort: {exc}")
    gen_dir = Path(args.output_dir) / "generated_audio"
    if gen_dir.exists():
        try:
            api.upload_folder(
                folder_path=str(gen_dir),
                path_in_repo=f"{prefix}/generated_audio",
                repo_id=args.results_repo,
                token=token,
            )
        except Exception as exc:
            print(f"Audio upload best-effort: {exc}")


def main() -> None:
    args = parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is required")

    random.seed(args.seed)

    print("=" * 60)
    print("T10/T11 F5/E2 fine-tune on Adja")
    print("=" * 60)
    print(f"Model family: {args.model_family}")
    print(f"Dry run: {args.dry_run}")
    print()

    install_env()
    clone_upstream()
    dataset_dir, _ = materialize_dataset(args, token)
    preprocess_f5_dataset(dataset_dir)
    pretrain_ckpt, missing_symbols = ensure_vocab_extension(dataset_dir, args.model_family)

    t0 = time.time()
    run_training(args, dataset_dir, pretrain_ckpt)
    elapsed_min = (time.time() - t0) / 60.0

    gen_dir, generated = generate_audio_samples(args, dataset_dir, token)

    metrics_path = collect_metrics(args, dataset_dir, elapsed_min, missing_symbols, pretrain_ckpt, generated)
    print(metrics_path.read_text(encoding="utf-8"))

    if args.push_to_hub:
        push_results(args, token, metrics_path)

    print("T10/T11 complete.")


if __name__ == "__main__":
    main()
