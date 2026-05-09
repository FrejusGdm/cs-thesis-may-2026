#!/usr/bin/env python3
from __future__ import annotations
"""
MMS-1B SSL continue-pretraining on unlabeled Ewe → Adja CTC fine-tune.

Last updated: 2026-04-21

MMS is architecturally wav2vec 2.0, so this is a thin wrapper around
wav2vec2_ssl_ewe.py with facebook/mms-1b-all as the --base model.

Reference: Massively Multilingual Speech (MMS) — https://arxiv.org/abs/2305.13516
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

# Re-export main so the SLURM launcher can run this file directly.
from wav2vec2_ssl_ewe import main as _main


def main():
    # If --base was not supplied, set MMS default.
    if "--base" not in sys.argv:
        sys.argv += ["--base", "facebook/mms-1b-all"]
    if "--model-key" not in sys.argv:
        sys.argv += ["--model-key", "mms-1b"]
    _main()


if __name__ == "__main__":
    main()
