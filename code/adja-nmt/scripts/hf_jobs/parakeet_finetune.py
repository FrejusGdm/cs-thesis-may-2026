# /// script
# dependencies = [
#     "setuptools>=68",
#     "torch==2.5.1",
#     "torchaudio==2.5.1",
#     "datasets",
#     "soundfile",
#     "librosa",
#     "numpy<2",
#     "pandas",
#     "huggingface-hub",
#     "jiwer",
#     "lightning>=2.2,<2.4",
#     "nemo_toolkit[asr]==2.0.0",
#     "sentencepiece",
# ]
# ///
# Note: `setuptools` is listed explicitly because `lightning>=2.2` does
# `import pkg_resources` at top-of-module. uv's Python 3.11 image does NOT
# ship setuptools by default, so omitting it produces:
#   ModuleNotFoundError: No module named 'pkg_resources'
# at the very first `import lightning.pytorch as pl`. (Reproduced on HF Job
# 69e37e2ecd8c002f31dfe8fc, 2026-04-18.)
"""
Parakeet-TDT 0.6B v3 fine-tuning for Adja ASR on HuggingFace Jobs.
Self-contained: pulls the dataset, materializes WAVs + NeMo manifests under
/tmp, runs NeMo finetune + eval, and uploads metrics to the Hub.

Adapted for HF Jobs from the original Dartmouth HPC code in
experiments/asr/D5_parakeet_finetune/hpc/. The HPC code remains the source
of truth for SLURM/Apptainer execution; this script reproduces the same
training/eval logic in a single self-contained launcher.

Env vars:
  HF_TOKEN        (required, even for DRY_RUN — dataset is private)
  EXP_ID          (default D5)
  MODEL_NAME      (default nvidia/parakeet-tdt-0.6b-v3)
  MAX_EPOCHS      (default 20)
  BATCH_SIZE      (default 8)
  LR              (default 1e-4)
  WARMUP_STEPS    (default 500)
  EVAL_BATCH_SIZE (default 16)
  RNNT_LOSS_NAME  (default: tdt_pytorch under UV, else tdt)
                  tdt_pytorch is NeMo's pure-PyTorch TDT fallback (slow,
                  debug-only); use for UV smoke. tdt is the normal fused
                  numba loss, needs NVCC + libnvvm at runtime — use on a
                  CUDA devel Docker image (pytorch:2.5.1-cuda12.1-cudnn9-devel).
  DRY_RUN         (default 0; "1" -> 4 samples, 1 epoch, no Hub upload)
  RETRAIN_TOKENIZER (default 0; "1" -> train a fresh SentencePiece BPE
                  tokenizer on Adja text + train transcripts, then swap
                  via `model.change_vocabulary(...)` before training. The
                  pretrained Parakeet-TDT v3 tokenizer is English-only and
                  emits `⁇` unk tokens for Adja specials ɛ ɔ ŋ ɖ.)
  VOCAB_SIZE      (default 1024; new tokenizer vocab size — small because
                  our corpus is ~14k sentences; setting too large yields
                  sparse merges and rarer subwords.)
  TOKENIZER_CORPUS_REPO (default JosueG/adja-text-corpus; private Hub
                  dataset holding `extra_adja_text.txt`, the 12k-sentence
                  Adja LM corpus merged from prior character-LM work.)

Smoke on UV (cheap validation, fallback loss):
    hf jobs uv run --flavor a10g-small --timeout 2h --secrets HF_TOKEN \\
        -p 3.11 -e DRY_RUN=1 -e EXP_ID=D5_smoke \\
        -e RNNT_LOSS_NAME=tdt_pytorch \\
        scripts/hf_jobs/parakeet_finetune.py

Pilot on Docker (real run, real loss): see D5 README — uses
`hf jobs run pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel bash -lc '...'`
with SCRIPT_URL extracted from the successful smoke via `hf jobs inspect`.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import unicodedata
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

# `lightning>=2.2`'s top-level `__init__` does `import pkg_resources`, which
# lives in `setuptools`. uv's python:3.11-bookworm image does NOT preinstall
# setuptools, so we install it to an explicit `--target` dir and prepend to
# sys.path. Pin <80: setuptools >=81 dropped the bundled `pkg_resources`
# module, so `pip install setuptools==82` still leaves `import pkg_resources`
# broken (verified on HF Job 69e3874a).
BOOTSTRAP_TARGET = "/tmp/_bootstrap_extras"
_need_install = []
try:
    import pkg_resources  # noqa: F401
except ModuleNotFoundError:
    _need_install.append("setuptools<80")

if _need_install:
    print("Bootstrapping", _need_install, "to", BOOTSTRAP_TARGET, "...")
    subprocess.check_call(
        ["uv", "pip", "install", "--target", BOOTSTRAP_TARGET, *_need_install]
    )
    sys.path.insert(0, BOOTSTRAP_TARGET)
    import pkg_resources  # noqa: F401

# ---------------------------------------------------------------- Config
EXP_ID = os.environ.get("EXP_ID", "D5")
MODEL_NAME = os.environ.get("MODEL_NAME", "nvidia/parakeet-tdt-0.6b-v3")
DATASET_ID = os.environ.get("DATASET_ID", "JosueG/adja-tts-orpheus")
RESULTS_REPO = os.environ.get("RESULTS_REPO", "JosueG/adja-asr-results")
RESULTS_SUBDIR = f"{EXP_ID}_parakeet"
SEED = int(os.environ.get("SEED", "42"))
LR = float(os.environ.get("LR", "1e-4"))
WARMUP_STEPS = int(os.environ.get("WARMUP_STEPS", "500"))
MAX_EPOCHS = int(os.environ.get("MAX_EPOCHS", "20"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "8"))
EVAL_BATCH_SIZE = int(os.environ.get("EVAL_BATCH_SIZE", "16"))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", "1e-3"))
ACCUMULATE_GRAD_BATCHES = int(os.environ.get("ACCUMULATE_GRAD_BATCHES", "1"))
DRY_RUN = os.environ.get("DRY_RUN", "0") == "1"

# tdt_pytorch = pure-PyTorch fallback (slow, debug-only per NeMo docs, but
#   avoids the numba NVVM path entirely — used on the uv smoke image).
# tdt         = NeMo's normal fused TDT loss, needs NVCC + libnvvm at
#   runtime — use on a CUDA devel Docker image (the pilot path).
# Default: tdt_pytorch when running on HF's uv runtime (which sets
# UV_SCRIPT_URL), else tdt.
_default_loss = "tdt_pytorch" if os.environ.get("UV_SCRIPT_URL") else "tdt"
RNNT_LOSS_NAME = os.environ.get("RNNT_LOSS_NAME", _default_loss)

RETRAIN_TOKENIZER = os.environ.get("RETRAIN_TOKENIZER", "0") == "1"
VOCAB_SIZE = int(os.environ.get("VOCAB_SIZE", "1024"))
TOKENIZER_CORPUS_REPO = os.environ.get(
    "TOKENIZER_CORPUS_REPO", "JosueG/adja-text-corpus"
)
TOKENIZER_CORPUS_FILE = os.environ.get(
    "TOKENIZER_CORPUS_FILE", "extra_adja_text.txt"
)

WORKSPACE = Path(os.environ.get("WORKSPACE", "/tmp/parakeet_adja"))
WAV_DIR = WORKSPACE / "wav"
MANIFEST_DIR = WORKSPACE / "manifests"
OUTPUT_DIR = WORKSPACE / "output"
for d in (WAV_DIR, MANIFEST_DIR, OUTPUT_DIR):
    d.mkdir(parents=True, exist_ok=True)

token = os.environ.get("HF_TOKEN")
if not token:
    raise SystemExit(
        "HF_TOKEN is required (dataset is private). Pass via --secrets HF_TOKEN or env."
    )

print(f"{EXP_ID}: {MODEL_NAME}")
if DRY_RUN:
    print("*** DRY_RUN=1: 4 samples, 1 epoch, no Hub upload ***")
    MAX_EPOCHS = 1


# ---------------------------------------------------------------- Materialize HF dataset to WAV + NeMo manifests
def normalize_text(t: str) -> str:
    """NFC normalize per project convention."""
    return " ".join(unicodedata.normalize("NFC", t.strip()).split())


def materialize_split(hf_split, split_name: str) -> Path:
    """Write WAVs (16kHz mono) and a NeMo JSON-lines manifest."""
    import numpy as np
    import soundfile as sf
    import librosa

    split_wav_dir = WAV_DIR / split_name
    split_wav_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = MANIFEST_DIR / f"{split_name}_manifest.json"

    n = 0
    with manifest_path.open("w", encoding="utf-8") as fout:
        for i, sample in enumerate(hf_split):
            audio = np.asarray(sample["audio"]["array"], dtype=np.float32)
            sr = int(sample["audio"]["sampling_rate"])
            if sr != 16000:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                sr = 16000

            wav_path = split_wav_dir / f"{split_name}_{i:06d}.wav"
            sf.write(str(wav_path), audio, sr, subtype="PCM_16")

            duration = len(audio) / sr
            entry = {
                "audio_filepath": str(wav_path),
                "text": normalize_text(sample["text"]),
                "duration": round(duration, 4),
            }
            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
            n += 1

    print(f"  {split_name}: {n} utts -> {manifest_path}")
    return manifest_path


print("Loading dataset ...")
from datasets import load_dataset

ds = load_dataset(DATASET_ID, token=token, split="train")
split1 = ds.train_test_split(test_size=0.1, seed=SEED)
split2 = split1["train"].train_test_split(test_size=0.1 / 0.9, seed=SEED)
splits = {
    "train": split2["train"],
    "dev": split2["test"],
    "test": split1["test"],
}
if DRY_RUN:
    splits = {k: v.select(range(min(4, len(v)))) for k, v in splits.items()}
print(f"Splits: train={len(splits['train'])} dev={len(splits['dev'])} test={len(splits['test'])}")

print("Materializing WAVs + manifests ...")
manifest_paths = {name: materialize_split(split, name) for name, split in splits.items()}


# ---------------------------------------------------------------- NeMo training
print("Importing NeMo ...")
import torch
import numpy as np
import pandas as pd
import jiwer
# NeMo 2.0's `modelPT.__init__` does an `isinstance(trainer,
# pytorch_lightning.Trainer)` check and rejects `lightning.pytorch.Trainer`
# even though they're the same class re-exported. Use the legacy namespace.
# Verified on HF Job 69e38874: lightning.pytorch.Trainer -> ValueError.
import pytorch_lightning as pl
from omegaconf import OmegaConf, DictConfig

import nemo.collections.asr as nemo_asr
import numba, lightning, nemo  # noqa: E402
from nemo.utils.exp_manager import exp_manager  # noqa: E402

print(
    f"[versions] torch={torch.__version__} nemo={nemo.__version__} "
    f"lightning={lightning.__version__} numba={numba.__version__}"
)
print(
    f"[config] RNNT_LOSS_NAME={RNNT_LOSS_NAME} DRY_RUN={DRY_RUN} "
    f"RETRAIN_TOKENIZER={RETRAIN_TOKENIZER} VOCAB_SIZE={VOCAB_SIZE}"
)

device_count = torch.cuda.device_count()
print(f"CUDA devices: {device_count}")
NUM_GPUS = max(1, device_count)

exp_dir = OUTPUT_DIR / "nemo_experiments"
exp_dir.mkdir(parents=True, exist_ok=True)

trainer = pl.Trainer(
    devices=NUM_GPUS,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    max_epochs=MAX_EPOCHS,
    accumulate_grad_batches=ACCUMULATE_GRAD_BATCHES,
    precision="bf16-mixed" if torch.cuda.is_available() else "32",
    log_every_n_steps=10,
    enable_checkpointing=False,
    logger=False,
    num_sanity_val_steps=2 if not DRY_RUN else 0,
)

exp_manager(
    trainer,
    DictConfig(
        {
            "exp_dir": str(exp_dir),
            "name": f"parakeet_tdt_{EXP_ID}",
            "checkpoint_callback_params": {
                "monitor": "val_wer",
                "mode": "min",
                "save_top_k": 1,
                "always_save_nemo": True,
            },
            "create_tensorboard_logger": False,
        }
    ),
)

print(f"Loading pretrained {MODEL_NAME} ...")
asr_model = nemo_asr.models.ASRModel.from_pretrained(
    model_name=MODEL_NAME,
    trainer=trainer,
)


def _swap_rnnt_loss(model, loss_name: str) -> None:
    """Swap only the training loss on a pretrained NeMo RNN-T / TDT model.

    tdt_pytorch avoids numba's NVVM codepath entirely (slow, debug-only per
    NeMo docs). tdt is the normal fused loss (needs NVCC at runtime).
    Decoding is left untouched so `transcribe()` uses the pretrained TDT
    decode setup.
    """
    from nemo.collections.asr.losses.rnnt import RNNTLoss

    cfg_loss = (
        OmegaConf.to_container(model.cfg.loss, resolve=True)
        if "loss" in model.cfg
        else {}
    )
    original_name = cfg_loss.get("loss_name", "<unknown>")
    tdt_kwargs = cfg_loss.get("tdt_kwargs", {}) or {}

    if loss_name == "tdt_pytorch":
        # NeMo's tdt_pytorch impl only uses `durations` + `sigma` from the
        # TDT kwargs; drop the rest (e.g. omega, numba-specific flags).
        effective_kwargs = {
            k: tdt_kwargs[k] for k in ("durations", "sigma") if k in tdt_kwargs
        }
    else:
        effective_kwargs = dict(tdt_kwargs)

    # For TDT models, `joint.num_classes_with_blank` = vocab + 1 + num_durations.
    # The label logits produced by the joint have last dim = vocab + 1, so the
    # blank index (which tdt_pytorch uses as `self.blank = num_classes`) must
    # be `vocab = num_classes_with_blank - 1 - num_durations`. Forgetting the
    # duration subtraction triggered IndexError: index 8197 out of bounds for
    # dim 3 size 8193 on HF Job 69e399bccd8c002f31dfea19 (Parakeet-TDT: vocab
    # 8192, 5 durations -> num_classes_with_blank=8198).
    num_durations = (
        len(effective_kwargs.get("durations", []))
        if loss_name.startswith("tdt")
        else 0
    )
    num_classes = model.joint.num_classes_with_blank - 1 - num_durations

    print(f"[loss-swap] original={original_name} -> selected={loss_name}")
    print(f"[loss-swap] effective_kwargs={effective_kwargs}")
    print(
        f"[loss-swap] num_classes_with_blank={model.joint.num_classes_with_blank} "
        f"num_durations={num_durations} -> num_classes={num_classes}"
    )

    new_loss = RNNTLoss(
        num_classes=num_classes,
        loss_name=loss_name,
        loss_kwargs=effective_kwargs,
    )
    model.loss = new_loss
    if hasattr(model.joint, "set_loss"):
        model.joint.set_loss(new_loss)


def _retrain_tokenizer_and_swap(
    model,
    train_manifest_path: Path,
    corpus_repo: str,
    corpus_file: str,
    vocab_size: int,
    out_dir: Path,
    hf_token: str,
) -> None:
    """Train a fresh SentencePiece BPE tokenizer on Adja text and swap it
    onto the pretrained model via `change_vocabulary`.

    Why: Parakeet-TDT 0.6B v3's pretrained English SentencePiece cannot emit
    Adja specials (ɛ ɔ ŋ ɖ + tone marks), so every test hypothesis
    saturates with `⁇` and corpus-WER becomes NaN (observed on pilot v2,
    job 69e406c9ac288e522d8efe49). `change_vocabulary` keeps the
    FastConformer encoder weights and rebuilds the joint + decoder
    embedding for the new vocab.

    Corpus: `{corpus_repo}/{corpus_file}` is the 12k-sentence Adja LM corpus
    curated during prior character-LM work, MERGED with the training split's
    transcripts (~1.3k lines) for ~13.3k total SP training sentences. NFC
    normalization is applied before training.
    """
    import sentencepiece as spm
    from huggingface_hub import hf_hub_download

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[tokenizer] Downloading corpus {corpus_repo}/{corpus_file} ...")
    lm_corpus_path = hf_hub_download(
        repo_id=corpus_repo,
        filename=corpus_file,
        repo_type="dataset",
        token=hf_token,
    )

    combined_path = out_dir / "spm_corpus.txt"
    n_lm = n_train = 0
    with combined_path.open("w", encoding="utf-8") as fout:
        with open(lm_corpus_path, "r", encoding="utf-8") as fin:
            for line in fin:
                t = unicodedata.normalize("NFC", line.strip())
                if t:
                    fout.write(t + "\n")
                    n_lm += 1
        with open(train_manifest_path, "r", encoding="utf-8") as fin:
            for line in fin:
                entry = json.loads(line)
                t = unicodedata.normalize("NFC", entry["text"].strip())
                if t:
                    fout.write(t + "\n")
                    n_train += 1
    total = n_lm + n_train
    print(f"[tokenizer] Corpus: {n_lm} LM lines + {n_train} train transcripts = {total}")

    model_prefix = str(out_dir / "tokenizer")
    print(
        f"[tokenizer] Training SentencePiece BPE "
        f"(vocab_size={vocab_size}, character_coverage=1.0) ..."
    )
    spm.SentencePieceTrainer.train(
        input=str(combined_path),
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="bpe",
        # char_coverage=1.0 is essential for low-resource langs with rare
        # diacritics — 0.9995 (Parakeet-EN default) drops ŋ/ɖ.
        character_coverage=1.0,
        bos_id=-1,
        eos_id=-1,
        pad_id=-1,
        unk_id=0,
        # Corpus is already NFC; avoid SP re-normalization overriding it.
        normalization_rule_name="identity",
    )

    model_path = Path(model_prefix + ".model")
    if not model_path.exists():
        raise RuntimeError(f"SentencePiece did not write {model_path}")

    # Some NeMo versions also read a `vocab.txt` next to `tokenizer.model`.
    vocab_src = Path(model_prefix + ".vocab")
    vocab_dst = out_dir / "vocab.txt"
    with vocab_src.open("r", encoding="utf-8") as fin, vocab_dst.open("w", encoding="utf-8") as fout:
        for line in fin:
            fout.write(line.split("\t", 1)[0] + "\n")

    before = model.joint.num_classes_with_blank
    print(f"[change_vocabulary] old num_classes_with_blank={before}")
    model.change_vocabulary(
        new_tokenizer_dir=str(out_dir),
        new_tokenizer_type="bpe",
    )
    after = model.joint.num_classes_with_blank
    print(f"[change_vocabulary] new num_classes_with_blank={after}")


if RETRAIN_TOKENIZER:
    _retrain_tokenizer_and_swap(
        asr_model,
        train_manifest_path=manifest_paths["train"],
        corpus_repo=TOKENIZER_CORPUS_REPO,
        corpus_file=TOKENIZER_CORPUS_FILE,
        vocab_size=VOCAB_SIZE,
        out_dir=WORKSPACE / "tokenizer_adja",
        hf_token=token,
    )
else:
    print("[tokenizer] RETRAIN_TOKENIZER=0, keeping pretrained tokenizer.")

# NOTE: _swap_rnnt_loss MUST run after change_vocabulary, because
# change_vocabulary rebuilds model.joint (new num_classes_with_blank) and
# we need the swapped loss to see the new shapes.
_swap_rnnt_loss(asr_model, RNNT_LOSS_NAME)

# --- Wire datasets ---
OmegaConf.set_struct(asr_model.cfg, False)
for ds_name, manifest in (("train_ds", manifest_paths["train"]), ("validation_ds", manifest_paths["dev"])):
    ds_cfg = asr_model.cfg[ds_name]
    ds_cfg.manifest_filepath = str(manifest)
    ds_cfg.batch_size = BATCH_SIZE
    ds_cfg.num_workers = 2
    ds_cfg.pin_memory = True
    ds_cfg.max_duration = 20.0
    ds_cfg.min_duration = 0.1
    ds_cfg.is_tarred = False
    ds_cfg.use_lhotse = False
    ds_cfg.pretokenize = False
asr_model.cfg.train_ds.shuffle = True
OmegaConf.set_struct(asr_model.cfg, True)

asr_model.setup_training_data(asr_model.cfg.train_ds)
asr_model.setup_validation_data(asr_model.cfg.validation_ds)

# --- Optimizer / scheduler ---
n_train = sum(1 for _ in open(manifest_paths["train"]))
steps_per_epoch = max(1, n_train // (BATCH_SIZE * ACCUMULATE_GRAD_BATCHES))
max_steps = max(steps_per_epoch * MAX_EPOCHS, 1)
print(f"steps_per_epoch={steps_per_epoch}, max_steps={max_steps}")

asr_model.setup_optimization(
    OmegaConf.create(
        {
            "name": "adamw",
            "lr": LR,
            "betas": [0.9, 0.98],
            "weight_decay": WEIGHT_DECAY,
            "sched": {
                "name": "CosineAnnealing",
                "warmup_steps": min(WARMUP_STEPS, max(max_steps // 10, 1)),
                "max_steps": max_steps,
                "min_lr": 1e-6,
            },
        }
    )
)

print("Training ...")
t_start = time.time()
trainer.fit(asr_model)
train_elapsed = time.time() - t_start
print(f"Training done in {train_elapsed/60:.1f} min")

# --- Resolve best checkpoint ---
best_ckpt = None
if trainer.checkpoint_callback and trainer.checkpoint_callback.best_model_path:
    best_ckpt = trainer.checkpoint_callback.best_model_path
else:
    import glob

    candidates = [
        f for f in glob.glob(str(exp_dir / "**" / "*.ckpt"), recursive=True)
        if "-last" not in f
    ]
    if candidates:
        best_ckpt = sorted(candidates)[0]
print(f"Best checkpoint: {best_ckpt}")

if best_ckpt:
    ckpt = torch.load(best_ckpt, map_location="cpu", weights_only=False)
    asr_model.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)


# ---------------------------------------------------------------- Evaluation
print("Evaluating on test split ...")
asr_model.eval()
if torch.cuda.is_available():
    asr_model = asr_model.cuda()

audio_paths, references = [], []
with open(manifest_paths["test"], "r", encoding="utf-8") as fh:
    for line in fh:
        entry = json.loads(line)
        audio_paths.append(entry["audio_filepath"])
        references.append(entry["text"])

with torch.no_grad():
    raw = asr_model.transcribe(audio_paths, batch_size=EVAL_BATCH_SIZE, return_hypotheses=False)

if isinstance(raw, (list, tuple)) and raw and isinstance(raw[0], (list, tuple)):
    predictions = [str(h) for h in raw[0]]
elif isinstance(raw, (list, tuple)) and raw and hasattr(raw[0], "text"):
    predictions = [h.text for h in raw]
else:
    predictions = [str(h) for h in raw]

wer_xform = jiwer.Compose([jiwer.ToLowerCase(), jiwer.RemoveMultipleSpaces(), jiwer.Strip(), jiwer.ReduceToListOfListOfWords()])
cer_xform = jiwer.Compose([jiwer.ToLowerCase(), jiwer.RemoveMultipleSpaces(), jiwer.Strip(), jiwer.ReduceToListOfListOfChars()])

per_sentence = []
all_wers, all_cers = [], []
for ref, hyp, ap in zip(references, predictions, audio_paths):
    ref_c, hyp_c = ref.strip(), (hyp or "").strip()
    if not ref_c and not hyp_c:
        sw, sc = 0.0, 0.0
    elif not ref_c:
        sw, sc = 1.0, 1.0
    else:
        try:
            sw = jiwer.wer(ref_c, hyp_c, truth_transform=wer_xform, hypothesis_transform=wer_xform)
        except Exception:
            sw = 1.0
        try:
            sc = jiwer.cer(ref_c, hyp_c, truth_transform=cer_xform, hypothesis_transform=cer_xform)
        except Exception:
            sc = 1.0
    all_wers.append(sw)
    all_cers.append(sc)
    per_sentence.append({"audio_filepath": ap, "reference": ref_c, "prediction": hyp_c, "wer": round(sw, 6), "cer": round(sc, 6)})

results_csv = OUTPUT_DIR / "test_results.csv"
pd.DataFrame(per_sentence).to_csv(results_csv, index=False, encoding="utf-8")

try:
    corpus_wer = jiwer.wer(references, predictions, truth_transform=wer_xform, hypothesis_transform=wer_xform)
except Exception:
    corpus_wer = float("nan")
try:
    corpus_cer = jiwer.cer(references, predictions, truth_transform=cer_xform, hypothesis_transform=cer_xform)
except Exception:
    corpus_cer = float("nan")

summary_lines = [
    "=" * 60,
    f"  EVALUATION SUMMARY — {EXP_ID} / {MODEL_NAME}",
    "=" * 60,
    f"Test sentences: {len(all_wers)}",
    f"Median WER:  {float(np.median(all_wers))*100:.2f}%",
    f"Median CER:  {float(np.median(all_cers))*100:.2f}%",
    f"Mean   WER:  {float(np.mean(all_wers))*100:.2f}%",
    f"Mean   CER:  {float(np.mean(all_cers))*100:.2f}%",
    f"Corpus WER:  {corpus_wer*100:.2f}%",
    f"Corpus CER:  {corpus_cer*100:.2f}%",
    "=" * 60,
]
summary_path = OUTPUT_DIR / "evaluation_summary.txt"
summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
for line in summary_lines:
    print(line)

# Print 3 decode samples
for i in range(min(3, len(per_sentence))):
    print(f"  REF: {per_sentence[i]['reference']}")
    print(f"  HYP: {per_sentence[i]['prediction']}")

metrics = {
    "experiment": EXP_ID,
    "model": MODEL_NAME,
    "dataset": DATASET_ID,
    "n_train": len(splits["train"]),
    "n_dev": len(splits["dev"]),
    "n_test": len(splits["test"]),
    "epochs": MAX_EPOCHS,
    "batch_size": BATCH_SIZE,
    "learning_rate": LR,
    "warmup_steps": WARMUP_STEPS,
    "best_checkpoint": best_ckpt,
    "median_wer": float(np.median(all_wers)),
    "median_cer": float(np.median(all_cers)),
    "mean_wer": float(np.mean(all_wers)),
    "mean_cer": float(np.mean(all_cers)),
    "corpus_wer": float(corpus_wer) if corpus_wer == corpus_wer else None,
    "corpus_cer": float(corpus_cer) if corpus_cer == corpus_cer else None,
    "train_minutes": round(train_elapsed / 60, 1),
}
metrics_path = OUTPUT_DIR / "metrics.json"
metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

if DRY_RUN:
    print("*** DRY_RUN complete — skipping Hub upload ***")
    sys.exit(0)

# ---------------------------------------------------------------- Push to Hub
print(f"Uploading to {RESULTS_REPO}/{RESULTS_SUBDIR}/ ...")
from huggingface_hub import HfApi

api = HfApi(token=token)
try:
    api.create_repo(RESULTS_REPO, private=True, exist_ok=True)
except Exception:
    pass

for path, repo_path in [
    (metrics_path, f"{RESULTS_SUBDIR}/metrics.json"),
    (summary_path, f"{RESULTS_SUBDIR}/evaluation_summary.txt"),
    (results_csv, f"{RESULTS_SUBDIR}/test_results.csv"),
]:
    api.upload_file(
        path_or_fileobj=str(path),
        path_in_repo=repo_path,
        repo_id=RESULTS_REPO,
        token=token,
    )

# Try to upload the .nemo checkpoint if exp_manager wrote one
nemo_files = list(exp_dir.rglob("*.nemo"))
if nemo_files:
    best_nemo = max(nemo_files, key=lambda p: p.stat().st_mtime)
    try:
        api.upload_file(
            path_or_fileobj=str(best_nemo),
            path_in_repo=f"{RESULTS_SUBDIR}/best_model.nemo",
            repo_id=RESULTS_REPO,
            token=token,
        )
        print(f"Uploaded {best_nemo}")
    except Exception as e:
        print(f"Skipping .nemo upload: {e}")

print(f"Done. Results: {RESULTS_REPO}/{RESULTS_SUBDIR}/")
