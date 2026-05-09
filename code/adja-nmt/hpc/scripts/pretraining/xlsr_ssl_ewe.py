#!/usr/bin/env python3
from __future__ import annotations
"""
XLS-R SSL continue-pretraining on unlabeled Ewe → Adja CTC fine-tune.

Last updated: 2026-04-21

XLS-R (facebook/wav2vec2-xls-r-{300m,1b,2b}) is the multilingual wav2vec 2.0
variant pretrained on 128 languages. Thin wrapper around wav2vec2_ssl_ewe.py;
user wants this particularly highlighted — it's arguably the current SOTA
starting point for low-resource SSL ASR.

Reference: XLS-R — https://arxiv.org/abs/2111.09296
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from wav2vec2_ssl_ewe import main as _main


def main():
    if "--base" not in sys.argv:
        sys.argv += ["--base", "facebook/wav2vec2-xls-r-300m"]
    if "--model-key" not in sys.argv:
        sys.argv += ["--model-key", "xlsr-300m"]
    _main()


if __name__ == "__main__":
    main()
