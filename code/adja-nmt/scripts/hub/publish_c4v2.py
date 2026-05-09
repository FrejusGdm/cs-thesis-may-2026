#!/usr/bin/env python3
"""
Publish C4v2 (XLS-R 300M CTC fine-tuned on Adja) as a deployable public model
repo. Reconstructs the missing tokenizer + processor files from the same
corpus + same logic that the original training script used, so that the
HF Serverless Inference API can decode CTC outputs into Adja text.

Why this is non-trivial
-----------------------
The original training script
[scripts/hf_jobs/ctc_finetune.py](../hf_jobs/ctc_finetune.py:53-64)
built its CTC vocab DETERMINISTICALLY from the NFC-normalized characters
of `JosueG/adja-tts-orpheus`, with the special-token prefix:

    {<blank>: 0, <pad>: 1, <unk>: 2, ...sorted_chars}

…then trained a `Wav2Vec2ForCTC` head with that vocab. **It never saved a
HuggingFace tokenizer**, only `ctc_vocab.json` (a Python dict) and the model
weights. As a result `JosueG/adja-asr-results/C4v2/best_model/` contains
`config.json` + `model.safetensors` but NO tokenizer, no
preprocessor_config.json, no special_tokens_map.json.

This script rebuilds those files identically (same dataset, same NFC
normalization, same sort order, same special-token prefix), wraps them in
the HuggingFace `Wav2Vec2CTCTokenizer` + `Wav2Vec2FeatureExtractor` +
`Wav2Vec2Processor` triad, and pushes everything to the destination repo
as a clean serverless-compatible artifact.

Source:      JosueG/adja-asr-results (model repo, subfolder C4v2/best_model/)
Destination: JosueG/wav2vec2-xlsr-adja-c4v2   [public]

Usage:
    HF_TOKEN=$HF_TOKEN python scripts/hub/publish_c4v2.py
    python scripts/hub/publish_c4v2.py --dry-run   # rebuild + verify only

Verification:
    --dry-run runs the entire reconstruction locally, prints the vocab size
    and a sample greedy-decode output on a synthetic batch (no API hits, no
    Hub writes). Use this to confirm the tokenizer assembles before pushing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import unicodedata
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)


SOURCE_REPO = "JosueG/adja-asr-results"
SOURCE_PATH = "C4v2/best_model"
SOURCE_REPO_TYPE = "model"

DEFAULT_DEST = "JosueG/wav2vec2-xlsr-adja-c4v2"

DATASET_ID = "JosueG/adja-tts-orpheus"
SEED = 42  # matches scripts/hf_jobs/ctc_finetune.py


def normalize_text(t: str) -> str:
    """Mirror scripts/hf_jobs/ctc_finetune.py:54.

    NFC + strip + collapse whitespace. Preserves Adja special chars and
    tone marks because both NFC and `str.split` are character-preserving.
    """
    return " ".join(unicodedata.normalize("NFC", t.strip()).split())


def build_vocab() -> dict[str, int]:
    """Reconstruct the EXACT CTC vocabulary used to train C4v2.

    From scripts/hf_jobs/ctc_finetune.py:53-64 :

        all_texts = [normalize_text(s["text"]) for s in ds]
        chars = set()
        for t in all_texts: chars.update(t)
        vocab = sorted(chars)
        char2idx = {"<blank>": 0, "<pad>": 1, "<unk>": 2}
        for i, c in enumerate(vocab, start=len(char2idx)): char2idx[c] = i

    `ds` here is `load_dataset(DATASET_ID, token=token, split="train")`
    BEFORE the train/dev/test split, so the vocab is over the entire corpus
    — i.e. deterministic given the dataset commit and `normalize_text`.
    """
    from datasets import load_dataset

    token = os.environ.get("HF_TOKEN")
    print(f"  Loading {DATASET_ID} train split (full, for vocab)…")
    ds = load_dataset(DATASET_ID, token=token, split="train")
    print(f"  {len(ds)} samples loaded")

    chars: set[str] = set()
    for sample in ds:
        chars.update(normalize_text(sample["text"]))

    vocab = sorted(chars)
    char2idx = {"<blank>": 0, "<pad>": 1, "<unk>": 2}
    for i, c in enumerate(vocab, start=len(char2idx)):
        char2idx[c] = i
    return char2idx


def build_processor(vocab: dict[str, int], out_dir: Path) -> None:
    """Materialize a `Wav2Vec2Processor` matching the vocab.

    The HF inference pipeline for `automatic-speech-recognition` on
    `Wav2Vec2ForCTC` consumes `Wav2Vec2Processor.from_pretrained(repo)` and
    uses `processor.batch_decode(...)` to map argmax token ids to text.
    `decode()` skips `pad_token_id` (the CTC blank) and `unk_token`.

    Important: in this CTC scheme the *blank* token is at index 0, while
    the script-time `<pad>` is at index 1 (used only to signal label padding
    during training). For the runtime tokenizer we want CTC blank skipping,
    so we set `pad_token = "<blank>"` (index 0) — that mirrors the runtime
    behaviour of the original `greedy_decode` in ctc_finetune.py:170.
    """
    from transformers import (
        Wav2Vec2CTCTokenizer,
        Wav2Vec2FeatureExtractor,
        Wav2Vec2Processor,
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    # vocab.json — id -> char mapping in HF format
    vocab_path = out_dir / "vocab.json"
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    # CTC tokenizer.
    #
    # Special-token wiring:
    #   pad_token = "<blank>" (id 0)  -> stripped during CTC decode (the blank).
    #   unk_token = "<unk>"   (id 2)  -> auto-skipped with skip_special_tokens=True.
    #   additional_special_tokens=["<pad>"] (id 1) -> ALSO skipped with
    #     skip_special_tokens=True, so the model's id-1 emissions don't leak
    #     into the decoded text as a literal "<pad>" string.
    #
    # The third bullet was missing in the first publish (2026-04-28); the
    # endpoint then returned hypotheses like "M<pad>i<pad>a <pad>kpɔ..." with
    # CER >100%. Verified with the post-process cleanup in
    # experiments/tts/eval/reverse_wer.py:_clean_hyp.
    tokenizer = Wav2Vec2CTCTokenizer(
        vocab_file=str(vocab_path),
        unk_token="<unk>",
        pad_token="<blank>",
        word_delimiter_token=" ",
        do_lower_case=False,
        replace_word_delimiter_char=" ",
        additional_special_tokens=["<pad>"],
    )

    # XLS-R feature extractor — standard 16kHz settings, matches training
    feat = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16000,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True,
    )

    processor = Wav2Vec2Processor(feature_extractor=feat, tokenizer=tokenizer)
    processor.save_pretrained(str(out_dir))
    print(f"  Processor written to {out_dir}")


def patch_config(out_dir: Path, vocab_size: int) -> None:
    """Make sure the model `config.json` agrees with the rebuilt tokenizer.

    The source C4v2 config.json should already have vocab_size == vocab_size
    we just rebuilt (115). Any mismatch indicates the dataset has shifted
    since training (e.g. new chars added). We assert and bail loudly if so.
    """
    cfg_path = out_dir / "config.json"
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    src_vocab_size = cfg.get("vocab_size")
    if src_vocab_size != vocab_size:
        print(f"  WARN: config.vocab_size={src_vocab_size} but rebuilt vocab has {vocab_size} entries")
        print("        The dataset may have changed since training. Aborting publish.")
        sys.exit(2)
    # Pad token id alignment: training script set model.config.pad_token_id to <pad>=1.
    # For runtime, it's harmless because Wav2Vec2ForCTC decoding uses tokenizer's
    # pad_token_id (the CTC blank, index 0). Leave config as-is for fidelity.


MODEL_CARD = """\
---
language:
- aj
license: apache-2.0
tags:
- automatic-speech-recognition
- wav2vec2
- ctc
- low-resource
- adja
- gbe
base_model: facebook/wav2vec2-xls-r-300m
datasets:
- JosueG/adja-tts-orpheus
metrics:
- wer
- cer
pipeline_tag: automatic-speech-recognition
---

# Wav2Vec2 XLS-R 300M → Adja CTC (C4v2)

Wav2Vec2 XLS-R 300M (`facebook/wav2vec2-xls-r-300m`) fine-tuned on ~1.36 h
of Adja audio from `JosueG/adja-tts-orpheus`, with a Connectionist Temporal
Classification head over a 115-entry character-level vocabulary derived
deterministically from the training corpus.

This is the second of two ASR architectures fine-tuned for Adja in
[Adja-NMT](https://github.com/anthropics/adja-nmt). It complements the
Whisper-Ewe→Adja pathway (E4v4) with an architecturally distinct
self-supervised + CTC pipeline — useful for two-ASR averaging in TTS
reverse-WER evaluation.

## Reported metrics (test set, NFC-normalized)

| WER    | CER    | Best epoch |
|--------|--------|------------|
| 72.39% | 25.05% | 48         |

A char-level n-gram LM rescore (`pyctcdecode`, α=0.67, β=-0.29, Optuna)
brings normalized WER to 70.76% / CER 22.67% in the
[`D4_C4v2_lm_optuna`](https://huggingface.co/JosueG/adja-asr-results/tree/main/D4_C4v2_lm_optuna)
record but is **not served** here — the HF Serverless Inference API does
not run pyctcdecode, so this repo uses greedy CTC decoding.

## Tokenizer reconstruction

The training script
([`scripts/hf_jobs/ctc_finetune.py`](https://github.com/anthropics/adja-nmt/blob/main/scripts/hf_jobs/ctc_finetune.py#L53-L64))
saved only the model weights, not the HuggingFace tokenizer. The tokenizer
in this repo was reconstructed by
[`scripts/hub/publish_c4v2.py`](https://github.com/anthropics/adja-nmt/blob/main/scripts/hub/publish_c4v2.py),
re-running the same logic against the same dataset:

```
chars = set()
for sample in load_dataset("JosueG/adja-tts-orpheus", split="train"):
    chars.update(unicodedata.normalize("NFC", sample["text"].strip()))
vocab = sorted(chars)
char2idx = {"<blank>": 0, "<pad>": 1, "<unk>": 2,
            **{c: i for i, c in enumerate(vocab, start=3)}}
```

The result is a 115-entry vocab. The `Wav2Vec2CTCTokenizer` is configured
with `pad_token="<blank>"` so that the decoder skips the CTC blank during
greedy decoding — matching the runtime behaviour of the original
`greedy_decode` in the training script.

## Inference

```python
from transformers import pipeline
asr = pipeline("automatic-speech-recognition",
               model="JosueG/wav2vec2-xlsr-adja-c4v2")
print(asr("path/to/adja_audio.wav")["text"])
```

Or via the HF Serverless Inference API:

```python
from huggingface_hub import InferenceClient
client = InferenceClient(model="JosueG/wav2vec2-xlsr-adja-c4v2")
hyp = client.automatic_speech_recognition(open("clip.wav", "rb").read())
print(hyp.text)
```

## Intended use

Adja → text. Suitable for:
- TTS reverse-WER auto-evaluation as an architectural counterpart to E4v4
  (different error modes → averaging produces a more robust ranking).
- Research baselines on Adja low-resource ASR.

NOT suitable for production transcription — 25% CER is a research-grade
floor, not a deployable transcript.

## References

- Conneau et al., 2020. *Unsupervised Cross-lingual Representation Learning for Speech Recognition*. [arXiv:2006.13979](https://arxiv.org/abs/2006.13979)
- Babu et al., 2021. *XLS-R: Self-supervised Cross-lingual Speech Representation Learning at Scale*. [arXiv:2111.09296](https://arxiv.org/abs/2111.09296)
- Graves et al., 2006. *Connectionist Temporal Classification*. https://www.cs.toronto.edu/~graves/icml_2006.pdf
- Adja-NMT exploration: [github.com/anthropics/adja-nmt](https://github.com/anthropics/adja-nmt) (registry: `experiments/registry.md`).

## Curated under

[`JosueG/adja-asr-best`](https://huggingface.co/collections/JosueG/adja-asr-best) — the project's curated best-of Adja ASR models.
"""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dest", default=DEFAULT_DEST, help=f"Destination model repo (default: {DEFAULT_DEST})")
    p.add_argument("--private", action="store_true", help="Create dest as private (default public — required by serverless API)")
    p.add_argument("--dry-run", action="store_true", help="Reconstruct + verify only, do not push")
    args = p.parse_args()

    token = os.environ.get("HF_TOKEN")
    from huggingface_hub import HfApi, snapshot_download

    api = HfApi(token=token)

    print("[1/4] Snapshot C4v2 weights from results repo…")
    with tempfile.TemporaryDirectory(prefix="c4v2-publish-") as tmpdir:
        local_dir = snapshot_download(
            repo_id=SOURCE_REPO,
            repo_type=SOURCE_REPO_TYPE,
            allow_patterns=[f"{SOURCE_PATH}/*"],
            local_dir=tmpdir,
        )
        artifact_dir = Path(local_dir) / SOURCE_PATH
        files = sorted(p.name for p in artifact_dir.iterdir() if p.is_file())
        print(f"      Got {len(files)} files: {files}")
        if "model.safetensors" not in files or "config.json" not in files:
            sys.exit("FATAL: source artifact missing config.json or model.safetensors")

        print("[2/4] Rebuild CTC vocabulary from JosueG/adja-tts-orpheus")
        vocab = build_vocab()
        print(f"      Vocab size: {len(vocab)} (special: <blank>={vocab['<blank>']}, <pad>={vocab['<pad>']}, <unk>={vocab['<unk>']})")
        # First few non-special tokens for human verification
        sample_chars = [k for k in vocab if k not in {"<blank>", "<pad>", "<unk>"}][:25]
        print(f"      First 25 chars: {sample_chars}")

        print("[3/4] Materialize Wav2Vec2Processor + verify config alignment")
        build_processor(vocab, artifact_dir)
        patch_config(artifact_dir, len(vocab))
        readme = artifact_dir / "README.md"
        readme.write_text(MODEL_CARD, encoding="utf-8")

        if args.dry_run:
            print("\n*** dry-run complete — no Hub writes ***")
            print(f"Local artifact at {artifact_dir}")
            return

        print(f"[4/4] Create + upload to {args.dest}")
        api.create_repo(args.dest, repo_type="model", private=args.private, exist_ok=True)
        api.upload_folder(
            folder_path=str(artifact_dir),
            repo_id=args.dest,
            repo_type="model",
            commit_message="Publish C4v2 (XLS-R CTC) with reconstructed tokenizer for serverless inference",
        )

    print(f"\nPublished: https://huggingface.co/{args.dest}")


if __name__ == "__main__":
    main()
