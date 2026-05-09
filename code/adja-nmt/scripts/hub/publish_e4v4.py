#!/usr/bin/env python3
"""
Publish E4v4 (Whisper-Ewe → Adja, EOS-mask fix) as a clean public model
repo so the HF Serverless Inference API can serve it.

Source:      JosueG/adja-asr-results (model repo, subfolder E4v4/best_model/)
Destination: JosueG/whisper-ewe-adja-e4v4   [public]

Why this exists
---------------
E4v4 lives inside an *experiment-results* repo under E4v4/best_model/, which
is fine for record-keeping but has two limitations for serving:
  1. The HF Serverless Inference API resolves models by repo root, not by
     subfolder. A consumer can't say "use E4v4/best_model" via /api-inference.
  2. Mixing 30+ experiment subfolders in one repo is noisy for downstream
     users — they want one clean repo per model.

This script downloads the E4v4/best_model/ subfolder, writes a model card
documenting the architecture / metrics / known caveats, and pushes
everything to a fresh public repo. Reverse-WER auto-eval
(`experiments/tts/eval/reverse_wer.py`) then targets the new repo directly.

Usage:
    HF_TOKEN=$HF_TOKEN python scripts/hub/publish_e4v4.py
    # or with explicit destination
    python scripts/hub/publish_e4v4.py --dest JosueG/whisper-ewe-adja-e4v4

Idempotent: re-running uploads only files whose hash changed.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)


SOURCE_REPO = "JosueG/adja-asr-results"
SOURCE_PATH = "E4v4/best_model"
SOURCE_REPO_TYPE = "model"

DEFAULT_DEST = "JosueG/whisper-ewe-adja-e4v4"


MODEL_CARD = """\
---
language:
- aj
license: apache-2.0
tags:
- automatic-speech-recognition
- whisper
- low-resource
- adja
- ewe
- gbe
base_model: dodziraynard/whisper-small-ee
datasets:
- JosueG/adja-tts-orpheus
metrics:
- wer
- cer
pipeline_tag: automatic-speech-recognition
---

# Whisper-Ewe → Adja (E4v4)

Whisper-small fine-tuned on Adja ASR via a two-stage Gbe-family transfer:
upstream Ewe Whisper (`dodziraynard/whisper-small-ee`, WER 37.48% on Ewe)
fine-tuned on ~1.36 h of Adja audio from `JosueG/adja-tts-orpheus`.

This is the **EOS-mask-fix** revision of the original E4 experiment. The
training collator masks padding via `attention_mask`, not `labels ==
pad_token_id` — the latter erases the real EOS in Whisper's tokenizer where
`pad_token == eos_token` and triggers infinite-hallucination loops at
inference. See [docs/whisper-training-gotchas.md](https://github.com/anthropics/adja-nmt/blob/main/docs/whisper-training-gotchas.md)
for the bug analysis.

## Reported metrics (test set, NFC-normalized)

| WER    | CER    | Best epoch |
|--------|--------|------------|
| 83.61% | 37.18% | 20         |

Best epoch (20) selected by dev CER. The model overfits past epoch 25
(train loss → 0.0005, dev CER plateaus around 50%). For production-style
Adja ASR, the better point on the learning curve is the original E4 run
(CER 24.90% at epoch 50) — but **those weights were never uploaded** and
the reproduction attempt as `E4_v5` is in progress (HF Job
`69f0f1c8d2c8bd8662bd235c`).

## Intended use

Adja → text. Suitable for:
- **TTS reverse-WER auto-evaluation** (the original motivation for
  publishing this checkpoint as a model repo).
- Research baselines on Adja low-resource ASR.
- Bootstrapping data-augmentation pipelines for Adja.

NOT suitable for production transcription — 37% CER is too noisy for human
consumption. Use as a *relative* signal across TTS systems, not an absolute
quality judgment of any single utterance.

## Inference

```python
from transformers import pipeline
asr = pipeline("automatic-speech-recognition",
               model="JosueG/whisper-ewe-adja-e4v4")
result = asr("path/to/adja_audio.wav")
print(result["text"])
```

Or via the HF Serverless Inference API:

```python
from huggingface_hub import InferenceClient
client = InferenceClient(model="JosueG/whisper-ewe-adja-e4v4")
hyp = client.automatic_speech_recognition(open("clip.wav", "rb").read())
print(hyp.text)
```

## Training data

`JosueG/adja-tts-orpheus` (private), 80/10/10 train/dev/test split with
seed=42 (enforced by `experiments/asr/shared/data_prep.py`). NFC
normalization applied to all text — preserves Adja special characters
ɛ, ɔ, ŋ, ɖ and tone marks.

Training-time hyperparameters (from
[experiments/asr/E4_whisper_ewe_finetune/config.yaml](https://github.com/anthropics/adja-nmt/blob/main/experiments/asr/E4_whisper_ewe_finetune/config.yaml)):

| Setting | Value |
|---|---|
| Base | `dodziraynard/whisper-small-ee` |
| LR | 1e-5 |
| Warmup steps | 500 |
| Max epochs | 50 (early-stop best at 20) |
| Batch size | 8 (effective 32 with grad-accum 4) |
| Patience | 20 |
| Seed | 42 |

## References

- Radford et al., *Robust Speech Recognition via Large-Scale Weak Supervision*. https://cdn.openai.com/papers/whisper.pdf
- Dodzi Raynard, [`dodziraynard/whisper-small-ee`](https://huggingface.co/dodziraynard/whisper-small-ee) — upstream Ewe Whisper.
- Adja-NMT exploration: [github.com/anthropics/adja-nmt](https://github.com/anthropics/adja-nmt) (registry: `experiments/registry.md`).

## Curated under

[`JosueG/adja-asr-best`](https://huggingface.co/collections/JosueG/adja-asr-best) — the project's curated set of best-of Adja ASR models.
"""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dest", default=DEFAULT_DEST, help=f"Destination model repo (default: {DEFAULT_DEST})")
    p.add_argument("--private", action="store_true", help="Create dest as private (default public — required by serverless API)")
    args = p.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        # huggingface_hub also reads ~/.cache/huggingface/token, so we don't
        # hard-fail. We just warn.
        print("WARN: HF_TOKEN not in env; relying on cached login")

    from huggingface_hub import HfApi, snapshot_download

    api = HfApi(token=token)

    print(f"[1/3] Snapshot {SOURCE_REPO}/{SOURCE_PATH}/")
    with tempfile.TemporaryDirectory(prefix="e4v4-publish-") as tmpdir:
        local_dir = snapshot_download(
            repo_id=SOURCE_REPO,
            repo_type=SOURCE_REPO_TYPE,
            allow_patterns=[f"{SOURCE_PATH}/*"],
            local_dir=tmpdir,
        )
        artifact_dir = Path(local_dir) / SOURCE_PATH
        files = sorted(p.name for p in artifact_dir.iterdir() if p.is_file())
        print(f"      Got {len(files)} files: {files}")

        readme = artifact_dir / "README.md"
        readme.write_text(MODEL_CARD, encoding="utf-8")
        print("      README.md written")

        print(f"[2/3] Create {args.dest} (private={args.private})")
        api.create_repo(args.dest, repo_type="model", private=args.private, exist_ok=True)

        print(f"[3/3] Upload {artifact_dir} -> {args.dest}")
        api.upload_folder(
            folder_path=str(artifact_dir),
            repo_id=args.dest,
            repo_type="model",
            commit_message="Publish E4v4 (Whisper-Ewe→Adja, EOS-mask fix) for serverless inference",
        )

    print(f"\nPublished: https://huggingface.co/{args.dest}")
    print("Inference (after a brief warm-up):")
    print(f'  curl -X POST -H "Authorization: Bearer $HF_TOKEN" \\')
    print(f"       --data-binary @clip.wav \\")
    print(f"       https://api-inference.huggingface.co/models/{args.dest}")


if __name__ == "__main__":
    main()
