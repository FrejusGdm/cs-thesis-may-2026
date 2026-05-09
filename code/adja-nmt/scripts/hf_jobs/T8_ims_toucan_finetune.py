#!/usr/bin/env python3
# /// script
# dependencies = ["torch==2.5.1", "torchaudio==2.5.1", "huggingface-hub>=0.34.0", "datasets>=3.4.1,<4.0.0", "soundfile", "librosa", "numpy", "scipy", "matplotlib>=3.7,<3.10", "torch_complex~=0.4.3", "epitran==1.24", "tqdm~=4.64.1", "praat-parselmouth~=0.4.2", "pypinyin~=0.47.1", "pyloudnorm~=0.1.0", "cvxopt~=1.3.0", "phonemizer~=3.2.1", "wandb~=0.13.5", "speechbrain==0.5.13", "dragonmapper~=0.2.6", "alias_free_torch~=0.0.6", "dotwiz==0.4.0", "transphone==1.5.3", "phonepiece==1.4.2", "geopy==2.4.1", "einops==0.7.0", "pandas~=1.5.0", "rich~=13.4.2", "PyYAML~=6.0", "imageio~=2.34.0", "pykakasi~=2.2.1", "jamo~=0.4.1", "g2pk~=0.9.4", "pykan~=0.2.6", "networkx~=3.3", "scikit-learn~=1.5.0"]
# ///
from __future__ import annotations

"""
T8: Fine-tune IMS-Toucan on Adja via Hugging Face Jobs.

This launcher is intentionally self-contained and mirrors the upstream
fine-tuning building blocks instead of relying on local helper modules.

Goal of the smoke:
  - prove Adja (`ajg`) text frontend works
  - prove corpus preprocessing succeeds
  - prove Toucan enters the real training loop
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


UPSTREAM_REPO = "https://github.com/DigitalPhonetics/IMS-Toucan.git"
UPSTREAM_BRANCH = "MassiveScaleToucan"
UPSTREAM_DIR = Path("/tmp/IMS-Toucan")
DATA_DIR = Path("/tmp/ims_toucan_adja")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="T8 IMS-Toucan fine-tune on Adja")
    parser.add_argument("--dataset", default="JosueG/adja-tts-orpheus")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--language-code", default="ajg")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--output-dir", default="/tmp/t8_ims_toucan_output")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--results-repo", default="JosueG/adja-tts-results")
    parser.add_argument("--results-prefix", default="T8_ims_toucan")
    return parser.parse_args()


def run(cmd: str, cwd: str | None = None) -> None:
    print(f"$ {cmd}")
    subprocess.check_call(cmd, shell=True, cwd=cwd)

def install_env() -> None:
    run(
        "apt-get update -q && apt-get install -y -q "
        "git ffmpeg espeak-ng libsndfile1 libasound-dev libportaudio2 libsqlite3-dev"
    )
    os.environ["MKL_THREADING_LAYER"] = "GNU"
    os.environ["OMP_NUM_THREADS"] = "1"


def clone_upstream() -> None:
    if UPSTREAM_DIR.exists():
        shutil.rmtree(UPSTREAM_DIR)
    run(
        f'git clone --depth 1 --branch "{UPSTREAM_BRANCH}" "{UPSTREAM_REPO}" "{UPSTREAM_DIR}"'
    )
    # Tiny Adja smoke runs can produce empty worker shards in the upstream aligner
    # cache builder, which then index path_list[0] and crash. Clamp the worker
    # count so every spawned process gets at least one path.
    run(
        "python3 - <<'PY'\n"
        "from pathlib import Path\n"
        "path = Path('/tmp/IMS-Toucan/Modules/Aligner/CodecAlignerDataset.py')\n"
        "text = path.read_text(encoding='utf-8')\n"
        "old = \"\"\"        key_splits = list()\\n"
        "        process_list = list()\\n"
        "        for i in range(loading_processes):\\n"
        "            key_splits.append(\\n"
        "                key_list[i * len(key_list) // loading_processes:(i + 1) * len(key_list) // loading_processes])\\n"
        "        for key_split in key_splits:\\n"
        "\"\"\"\n"
        "new = \"\"\"        key_splits = list()\\n"
        "        process_list = list()\\n"
        "        loading_processes = max(1, min(loading_processes, len(key_list)))\\n"
        "        for i in range(loading_processes):\\n"
        "            key_split = key_list[i * len(key_list) // loading_processes:(i + 1) * len(key_list) // loading_processes]\\n"
        "            if key_split:\\n"
        "                key_splits.append(key_split)\\n"
        "        for key_split in key_splits:\\n"
        "\"\"\"\n"
        "if old not in text:\n"
        "    raise SystemExit('expected CodecAlignerDataset split block not found')\n"
        "path.write_text(text.replace(old, new), encoding='utf-8')\n"
        "print('patched', path)\n"
        "PY",
        cwd=str(UPSTREAM_DIR),
    )
    # SpeechBrain's ECAPA helper currently expects repo files that no longer
    # exist on the Hub. For startup validation we only need consistent speaker
    # embeddings, not perfect speaker identity modeling, so fall back to zero
    # vectors if the external speaker encoder cannot be fetched.
    run(
        "python3 - <<'PY'\n"
        "from pathlib import Path\n"
        "path = Path('/tmp/IMS-Toucan/Modules/Aligner/CodecAlignerDataset.py')\n"
        "text = path.read_text(encoding='utf-8')\n"
        "old = '''        # add speaker embeddings\\n"
        "        self.speaker_embeddings = list()\\n"
        "        speaker_embedding_func_ecapa = EncoderClassifier.from_hparams(source=\"speechbrain/spkrec-ecapa-voxceleb\",\\n"
        "                                                                      run_opts={\"device\": str(device)},\\n"
        "                                                                      savedir=os.path.join(MODEL_DIR, \"Embedding\", \"speechbrain_speaker_embedding_ecapa\"))\\n"
        "        with torch.inference_mode():\\n"
        "            for wave in tqdm(norm_waves):\\n"
        "                self.speaker_embeddings.append(speaker_embedding_func_ecapa.encode_batch(wavs=wave.to(device).unsqueeze(0)).squeeze().cpu())\\n"
        "''' \n"
        "new = '''        # add speaker embeddings\\n"
        "        self.speaker_embeddings = list()\\n"
        "        speaker_embedding_func_ecapa = None\\n"
        "        try:\\n"
        "            speaker_embedding_func_ecapa = EncoderClassifier.from_hparams(source=\"speechbrain/spkrec-ecapa-voxceleb\",\\n"
        "                                                                          run_opts={\"device\": str(device)},\\n"
        "                                                                          savedir=os.path.join(MODEL_DIR, \"Embedding\", \"speechbrain_speaker_embedding_ecapa\"))\\n"
        "        except Exception as exc:\\n"
        "            print(f\"Speaker embedding fallback active: {exc}\")\\n"
        "        with torch.inference_mode():\\n"
        "            for wave in tqdm(norm_waves):\\n"
        "                if speaker_embedding_func_ecapa is None:\\n"
        "                    self.speaker_embeddings.append(torch.zeros(192))\\n"
        "                else:\\n"
        "                    self.speaker_embeddings.append(speaker_embedding_func_ecapa.encode_batch(wavs=wave.to(device).unsqueeze(0)).squeeze().cpu())\\n"
        "''' \n"
        "if old not in text:\n"
        "    raise SystemExit('expected speaker embedding block not found')\n"
        "path.write_text(text.replace(old, new), encoding='utf-8')\n"
        "print('patched', path)\n"
        "PY",
        cwd=str(UPSTREAM_DIR),
    )


def ensure_transphone_assets() -> None:
    # Toucan falls back to Transphone for ISO codes outside its explicit eSpeak
    # map. Prefetch the model once in the parent process so the multiprocessing
    # cache builder does not race while downloading/extracting the same files.
    from transphone.g2p import read_g2p

    model = read_g2p(device="cpu")
    model_dir = Path(model.model_path)
    required = [model_dir / "grapheme.vocab", model_dir / "phoneme.vocab"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Transphone assets missing after prefetch: {missing}")
    print(f"Transphone assets ready under {model_dir}")


def ensure_phonepiece_assets() -> None:
    # Toucan's fallback frontend also consults PhonePiece inventories. Just like
    # Transphone, prefetch them once up front so worker processes don't race on
    # first access to the shared model directory.
    from phonepiece.bin.download_model import download_model
    from phonepiece.config import PhonePieceConfig

    download_model("latest")
    model_dir = Path(PhonePieceConfig.data_path) / "model" / "latest"
    required = [model_dir / "eng" / "phone.txt", model_dir / "eng" / "phoneme.txt"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"PhonePiece assets missing after prefetch: {missing}")
    print(f"PhonePiece assets ready under {model_dir}")


def patch_huggingface_hub_compat() -> None:
    # Older speechbrain releases still call hf_hub_download(use_auth_token=...).
    # Newer huggingface_hub removed that keyword in favor of token=..., so adapt
    # the call site in-process rather than downgrading the whole Hub stack.
    import huggingface_hub

    original = huggingface_hub.hf_hub_download

    def compat_hf_hub_download(*args, use_auth_token=None, **kwargs):
        if use_auth_token is not None and "token" not in kwargs:
            kwargs["token"] = use_auth_token
        return original(*args, **kwargs)

    huggingface_hub.hf_hub_download = compat_hf_hub_download


def nfc(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text.strip()).split())


def materialize_adja(args: argparse.Namespace, token: str) -> tuple[dict[str, str], Path]:
    import soundfile as sf
    from datasets import load_dataset

    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)
    wav_dir = DATA_DIR / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(args.dataset, token=token, split="train")
    split1 = ds.train_test_split(test_size=0.1, seed=args.seed)
    split2 = split1["train"].train_test_split(test_size=0.1 / 0.9, seed=args.seed)
    train_ds = split2["train"]
    dev_ds = split2["test"]
    test_ds = split1["test"]

    if args.dry_run:
        train_ds = train_ds.select(range(min(12, len(train_ds))))
        dev_ds = dev_ds.select(range(min(3, len(dev_ds))))
        test_ds = test_ds.select(range(min(3, len(test_ds))))

    def export_split(split, name: str) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for idx, example in enumerate(split):
            text = nfc(example["text"])
            if not text:
                continue
            audio = example["audio"]["array"]
            sr = example["audio"]["sampling_rate"]
            wav_path = wav_dir / f"{name}_{idx:06d}.wav"
            sf.write(str(wav_path), audio, sr)
            mapping[str(wav_path)] = text
        return mapping

    payload = {
        "train": export_split(train_ds, "train"),
        "dev": export_split(dev_ds, "dev"),
        "test": export_split(test_ds, "test"),
    }
    payload_path = DATA_DIR / "path_to_transcripts.json"
    payload_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload, payload_path


def run_training(args: argparse.Namespace, payload: dict[str, dict[str, str]]) -> Path:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Upstream imports happen only after the repo is cloned and its requirements
    # are installed.
    sys.path.insert(0, str(UPSTREAM_DIR))
    os.chdir(UPSTREAM_DIR)
    os.environ["MKL_THREADING_LAYER"] = "GNU"
    os.environ["OMP_NUM_THREADS"] = "1"

    from huggingface_hub import hf_hub_download
    from Modules.ToucanTTS.ToucanTTS import ToucanTTS
    from Modules.ToucanTTS.toucantts_train_loop_arbiter import train_loop
    from Utility.corpus_preparation import prepare_tts_corpus
    from Utility.storage_config import MODEL_DIR, PREPROCESSING_DIR

    print(f"IMS-Toucan MODEL_DIR={MODEL_DIR} PREPROCESSING_DIR={PREPROCESSING_DIR}")
    ensure_transphone_assets()
    ensure_phonepiece_assets()
    patch_huggingface_hub_compat()

    train_set = prepare_tts_corpus(
        transcript_dict=payload["train"],
        corpus_dir=str(DATA_DIR / "cache_train"),
        lang=args.language_code,
        fine_tune_aligner=not args.dry_run,
    )
    sampler = [__import__("torch").utils.data.RandomSampler(train_set)]
    model = ToucanTTS()

    checkpoint = hf_hub_download(cache_dir=MODEL_DIR, repo_id="Flux9665/ToucanTTS", filename="ToucanTTS.pt")

    steps = 2 if args.dry_run else args.steps
    batch_size = min(args.batch_size, 2) if args.dry_run else args.batch_size
    warmup = 1 if args.dry_run else args.warmup_steps

    train_loop(
        net=model,
        datasets=[train_set],
        train_samplers=sampler,
        gpu_count=1,
        device=__import__("torch").device("cuda"),
        save_directory=str(output_dir),
        path_to_checkpoint=checkpoint,
        lr=args.learning_rate,
        resume=False,
        warmup_steps=warmup,
        use_wandb=False,
        batch_size=batch_size,
        eval_lang=args.language_code,
        fine_tune=True,
        steps=steps,
    )
    return output_dir


def collect_metrics(args: argparse.Namespace, payload_path: Path, elapsed_min: float) -> Path:
    output_dir = Path(args.output_dir)
    checkpoint_files = []
    if output_dir.exists():
        checkpoint_files = sorted(p.name for p in output_dir.iterdir())
    metrics = {
        "experiment": "T8",
        "dataset": args.dataset,
        "language_code": args.language_code,
        "dry_run": args.dry_run,
        "elapsed_min": round(elapsed_min, 1),
        "output_dir": str(output_dir),
        "checkpoint_files": checkpoint_files,
        "path_to_transcripts_json": str(payload_path),
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
    out_dir = Path(args.output_dir)
    if out_dir.exists():
        try:
            api.upload_folder(
                folder_path=str(out_dir),
                path_in_repo=f"{prefix}/output",
                repo_id=args.results_repo,
                token=token,
            )
        except Exception as exc:
            print(f"Best-effort upload warning: {exc}")


def main() -> None:
    args = parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is required")

    random.seed(args.seed)

    print("=" * 60)
    print("T8 IMS-Toucan fine-tune on Adja")
    print("=" * 60)
    print(f"Language code: {args.language_code}")
    print(f"Dry run: {args.dry_run}")
    print()

    install_env()
    clone_upstream()
    payload, payload_path = materialize_adja(args, token)

    t0 = time.time()
    run_training(args, payload)
    elapsed_min = (time.time() - t0) / 60.0

    metrics_path = collect_metrics(args, payload_path, elapsed_min)
    print(metrics_path.read_text(encoding="utf-8"))

    if args.push_to_hub:
        push_results(args, token, metrics_path)

    print("T8 complete.")


if __name__ == "__main__":
    main()
