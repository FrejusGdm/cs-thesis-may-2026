#!/usr/bin/env python3
"""
Publish a Spark TTS Adja fine-tune as a deployable Hugging Face model repo.

Source layout
-------------
The fine-tuned LLM half of Spark lives under
`JosueG/adja-tts-results/<run>/adapter/` (Qwen2 0.5B with the BPE
tokenizer that was used at training time). The audio-tokenizer half
(BiCodec encoder/decoder + the small wav2vec2-large-xlsr-53 used for
prompt encoding) lives at `unsloth/Spark-TTS-0.5B/{BiCodec,
wav2vec2-large-xlsr-53}/`.

This script combines both halves into a single public repo with the
layout that `Spark-TTS/cli/SparkTTS.py:SparkTTS(model_dir)` expects
when invoked as a library:

    <dest-repo>/
        LLM/                     # adapter/ from the fine-tune
        BiCodec/                 # from unsloth base
        wav2vec2-large-xlsr-53/  # from unsloth base
        config.yaml              # from unsloth base
        handler.py               # custom inference handler (see scripts/hub/spark_handler.py)
        requirements.txt         # what the endpoint container installs
        README.md                # model card

Default destination is the canonical `JosueG/spark-tts-adja-t3`. A different
Spark variant can be picked with `--source-run`.

Usage
-----
    HF_TOKEN=$HF_TOKEN python scripts/hub/publish_spark_t3.py
    python scripts/hub/publish_spark_t3.py --source-run T3_spark_20ep_earlystop_2026-04-18
    python scripts/hub/publish_spark_t3.py --dry-run     # local assemble, no Hub push

Verification
------------
The script prints the path of the local-assembled artifact and the
hub URL on push. Before deploying to an Inference Endpoint, **run
`scripts/hub/spark_handler.py` locally against the assembled folder
to confirm the handler can synthesize one Adja sample from text-only
input** (no prompt audio). Spark needs `Spark-TTS/` checked out at the
project root for the `BiCodecTokenizer`; the publisher copies that
folder into the destination repo so the handler is self-contained.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)


SOURCE_REPO = "JosueG/adja-tts-results"
SOURCE_REPO_TYPE = "model"
DEFAULT_SOURCE_RUN = "T3"

BASE_REPO = "unsloth/Spark-TTS-0.5B"
BASE_FILES = [
    "BiCodec/config.yaml",
    "BiCodec/model.safetensors",
    "wav2vec2-large-xlsr-53/config.json",
    "wav2vec2-large-xlsr-53/preprocessor_config.json",
    "wav2vec2-large-xlsr-53/pytorch_model.bin",
    "config.yaml",
]

DEFAULT_DEST = "JosueG/spark-tts-adja-t3"


REQUIREMENTS_TXT = """\
torch==2.5.1
torchaudio==2.5.1
transformers==4.49.0
sentencepiece>=0.2.0
soundfile>=0.12.0
librosa>=0.10.0
omegaconf>=2.3.0
einops>=0.8.0
einx>=0.3.0
"""
# 2026-05-07: einx added — sparktts.modules.fsq.residual_fsq imports
# `from einx import get_at` which is not part of einops; endpoint container
# failed with ModuleNotFoundError without this line.


def model_card(source_run: str, dest: str) -> str:
    return f"""\
---
language:
- aj
license: apache-2.0
tags:
- text-to-speech
- spark-tts
- low-resource
- adja
- gbe
base_model: unsloth/Spark-TTS-0.5B
datasets:
- JosueG/adja-tts-orpheus
pipeline_tag: text-to-speech
---

# Spark TTS Adja ({source_run})

A fine-tune of `unsloth/Spark-TTS-0.5B` on the `JosueG/adja-tts-orpheus`
Adja TTS dataset (~1.6 hours of speech). This is the only TTS architecture
in the Adja-NMT exploration that produced **intelligible Adja** with
this amount of data; CSM 1B, Orpheus 3B, F5-TTS, E2-TTS, MMS-TTS-Ewe,
and VoxCPM all collapse to noise on direct Adja fine-tuning.

The Spark architecture combines a Qwen2 0.5B language model that emits
discrete BiCodec tokens with a BiCodec encoder/decoder. The multilingual
acoustic priors of XLS-R-53 (carried into the BiCodec semantic
tokenizer) plus the Qwen2 BPE tokenizer give Spark Adja phoneme
coverage out of the box; the fine-tune only has to learn the mapping
from Adja text to BiCodec semantic and global token sequences.

## Reverse-WER auto-eval

Scored against two Adja ASRs deployed as Hugging Face Inference
Endpoints (XLS-R + character CTC and Whisper-small Ewe-to-Adja
transfer):

| ASR              | Normalized CER | Normalized WER |
|------------------|----------------|----------------|
| XLS-R + CTC      | 36.14 %        | 72.73 %        |
| Whisper-Ewe-Adja | 28.92 %        | 59.09 %        |

Within ~10 points of the CTC ASR's own published test CER on real
Adja speech (25.05 %). See `results/reverse-wer-summary.md` in the
`adja-nmt` project for the 22-run table this was selected from.

## Inference

This repo ships with a custom `handler.py` that loads
`Spark-TTS/cli/SparkTTS.py:SparkTTS` and exposes a `text_to_speech`
interface for Hugging Face Inference Endpoints. Use a GPU instance
(`nvidia-l4` x1 or larger) for acceptable latency. Sample rate is
16 kHz at the BiCodec output.

```python
import requests
from huggingface_hub import get_token

resp = requests.post(
    ENDPOINT_URL,
    json={{"inputs": "Tɛnigbe ciyi vayi de ŋweba"}},
    headers={{"Authorization": f"Bearer {{get_token()}}"}},
    timeout=180,
)
open("output.wav", "wb").write(resp.content)
```

## Training data

`JosueG/adja-tts-orpheus` (private). 80/10/10 train/dev/test split with
seed 42. NFC normalization on all text. Source run subfolder:
`{source_run}` in `JosueG/adja-tts-results` (model side).

## Curated under

[`JosueG/adja-tts-best`](https://huggingface.co/collections/JosueG/adja-tts-best) — the project's curated best-of Adja TTS models.
"""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source-run", default=DEFAULT_SOURCE_RUN,
                   help=f"Subfolder under {SOURCE_REPO} containing the adapter/ "
                        f"(default: {DEFAULT_SOURCE_RUN})")
    p.add_argument("--dest", default=DEFAULT_DEST,
                   help=f"Destination model repo (default: {DEFAULT_DEST})")
    p.add_argument("--private", action="store_true",
                   help="Create dest as private (default public).")
    p.add_argument("--dry-run", action="store_true",
                   help="Assemble the artifact locally and stop. No Hub push.")
    args = p.parse_args()

    token = os.environ.get("HF_TOKEN")

    from huggingface_hub import HfApi, snapshot_download

    api = HfApi(token=token)

    # Spark-TTS source repo must be present locally — the handler imports from it.
    spark_src = Path(__file__).resolve().parents[2] / "Spark-TTS"
    if not spark_src.exists():
        sys.exit(
            f"FATAL: {spark_src} not found. The publisher copies the Spark-TTS "
            f"library into the destination repo so the handler can import "
            f"`sparktts.models.audio_tokenizer.BiCodecTokenizer`. Clone "
            f"https://github.com/SparkAudio/Spark-TTS into the project root, "
            f"or check it out at HEAD if it was already a submodule."
        )
    handler_src = Path(__file__).parent / "spark_handler.py"
    if not handler_src.exists():
        sys.exit(f"FATAL: {handler_src} not found")

    print(f"[1/5] Snapshot adapter from {SOURCE_REPO}/{args.source_run}/adapter/")
    with tempfile.TemporaryDirectory(prefix="spark-publish-") as tmpdir:
        adapter_dir = Path(snapshot_download(
            repo_id=SOURCE_REPO,
            repo_type=SOURCE_REPO_TYPE,
            allow_patterns=[f"{args.source_run}/adapter/*"],
            local_dir=tmpdir,
        )) / args.source_run / "adapter"
        if not adapter_dir.exists():
            sys.exit(f"FATAL: adapter/ not found at {adapter_dir}")

        print(f"[2/5] Snapshot BiCodec + wav2vec2 from {BASE_REPO}")
        base_dir = Path(snapshot_download(
            repo_id=BASE_REPO,
            repo_type="model",
            allow_patterns=BASE_FILES,
            local_dir=tmpdir + "/base",
        ))

        # Assemble the destination layout
        out_dir = Path(tmpdir) / "assembled"
        out_dir.mkdir(parents=True)
        print(f"[3/5] Assemble destination layout at {out_dir}")
        # LLM/ <- adapter
        shutil.copytree(adapter_dir, out_dir / "LLM")
        # BiCodec/ + wav2vec2-large-xlsr-53/ + config.yaml from base
        for sub in ("BiCodec", "wav2vec2-large-xlsr-53"):
            src_sub = base_dir / sub
            if src_sub.exists():
                shutil.copytree(src_sub, out_dir / sub)
        if (base_dir / "config.yaml").exists():
            shutil.copy(base_dir / "config.yaml", out_dir / "config.yaml")
        # Spark-TTS library
        shutil.copytree(spark_src / "sparktts", out_dir / "sparktts")
        if (spark_src / "cli").exists():
            shutil.copytree(spark_src / "cli", out_dir / "cli")

        # 2026-05-07: Patch audio_tokenizer.py to handle Git-LFS stubs.
        # HF Inference Endpoints clone the repo but do not always resolve LFS
        # pointers; pytorch_model.bin may arrive as a 135-byte stub.  We detect
        # a dangling stub (file size < 1 MB) and fall back to loading wav2vec2
        # directly from `facebook/wav2vec2-large-xlsr-53` via the Hub.
        _at_path = out_dir / "sparktts" / "models" / "audio_tokenizer.py"
        _at_code = _at_path.read_text(encoding="utf-8")
        _at_old = (
            '        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(\n'
            '            f"{self.model_dir}/wav2vec2-large-xlsr-53"\n'
            '        )\n'
            '        self.feature_extractor = Wav2Vec2Model.from_pretrained(\n'
            '            f"{self.model_dir}/wav2vec2-large-xlsr-53"\n'
            '        ).to(self.device)\n'
            '        self.feature_extractor.config.output_hidden_states = True'
        )
        _at_new = (
            '        import os as _os\n'
            '        _wav2vec2_local = f"{self.model_dir}/wav2vec2-large-xlsr-53"\n'
            '        _bin = _os.path.join(_wav2vec2_local, "pytorch_model.bin")\n'
            '        _safetensors = _os.path.join(_wav2vec2_local, "model.safetensors")\n'
            '        _has_real_weights = (\n'
            '            (_os.path.isfile(_bin) and _os.path.getsize(_bin) > 1_000_000)\n'
            '            or _os.path.isfile(_safetensors)\n'
            '        )\n'
            '        _wav2vec2_src = _wav2vec2_local if _has_real_weights else "facebook/wav2vec2-large-xlsr-53"\n'
            '        if not _has_real_weights:\n'
            '            print(\n'
            '                f"[BiCodecTokenizer] wav2vec2 LFS stub detected; loading from Hub", flush=True\n'
            '            )\n'
            '        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(_wav2vec2_src)\n'
            '        self.feature_extractor = Wav2Vec2Model.from_pretrained(_wav2vec2_src).to(self.device)\n'
            '        self.feature_extractor.config.output_hidden_states = True'
        )
        if _at_old in _at_code:
            _at_path.write_text(_at_code.replace(_at_old, _at_new), encoding="utf-8")
            print("  [patch] audio_tokenizer.py: LFS-stub fallback applied")
        # handler.py + requirements.txt + README
        shutil.copy(handler_src, out_dir / "handler.py")
        (out_dir / "requirements.txt").write_text(REQUIREMENTS_TXT, encoding="utf-8")
        (out_dir / "README.md").write_text(model_card(args.source_run, args.dest), encoding="utf-8")

        files = sorted(p.relative_to(out_dir) for p in out_dir.rglob("*") if p.is_file())
        print(f"[4/5] Local artifact ready: {len(files)} files")
        for f in files[:25]:
            print(f"      {f}")
        if len(files) > 25:
            print(f"      ... and {len(files) - 25} more")

        if args.dry_run:
            persisted = Path(f"/tmp/spark-publish-{args.source_run}")
            if persisted.exists():
                shutil.rmtree(persisted)
            shutil.copytree(out_dir, persisted)
            print(f"\n*** dry-run: artifact persisted at {persisted} ***")
            print("Test the handler locally with:")
            print(f'  cd {persisted} && python -c "import handler; h = handler.EndpointHandler(\\".\\"); print(h({{\\"inputs\\": \\"Tɛnigbe ciyi vayi de ŋweba\\"}}))"')
            return

        print(f"[5/5] Create + upload to {args.dest}")
        api.create_repo(args.dest, repo_type="model", private=args.private, exist_ok=True)
        api.upload_folder(
            folder_path=str(out_dir),
            repo_id=args.dest,
            repo_type="model",
            commit_message=f"Publish Spark TTS Adja ({args.source_run})",
        )

    print(f"\nPublished: https://huggingface.co/{args.dest}")
    print()
    print("Next: deploy as a dedicated Inference Endpoint with a custom handler.")
    print("  python scripts/hub/manage_endpoints.py up", args.dest)
    print("  (...switch flavor to nvidia-l4 x1 in the HF UI before going live;")
    print("   the default intel-spr CPU is too slow for Spark inference.)")


if __name__ == "__main__":
    main()
