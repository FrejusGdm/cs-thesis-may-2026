#!/usr/bin/env python3
from __future__ import annotations

"""
Deprecated entrypoint for Sesame CSM training.

This file used to follow a generic text-only SFT flow, which is not valid for CSM.
Use one of these instead:

- Vendor source notebook:
  references/unsloth-tts-notebooks/Sesame_CSM_1B_TTS.ipynb
- Adja runbook:
  experiments/tts/T1_sesame_csm_finetune/UPSTREAM_NOTEBOOK_RUNBOOK.md
- Canonical script mirror:
  scripts/hf_jobs/T1_sesame_csm_finetune.py
"""


raise SystemExit(
    "Deprecated CSM trainer. Use the notebook at "
    "references/unsloth-tts-notebooks/Sesame_CSM_1B_TTS.ipynb "
    "with the Adja runbook at experiments/tts/T1_sesame_csm_finetune/UPSTREAM_NOTEBOOK_RUNBOOK.md, "
    "or use the canonical script mirror at scripts/hf_jobs/T1_sesame_csm_finetune.py."
)
