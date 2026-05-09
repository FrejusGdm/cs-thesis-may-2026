#!/usr/bin/env python3
from __future__ import annotations

"""
Deprecated CSM launcher.

This file previously tracked an older Unsloth CSM path which still used
`use_gradient_checkpointing="unsloth"` and diverged from the local notebook fix.

Use:
- references/unsloth-tts-notebooks/Sesame_CSM_1B_TTS.ipynb
- experiments/tts/T1_sesame_csm_finetune/UPSTREAM_NOTEBOOK_RUNBOOK.md
- scripts/hf_jobs/T1_sesame_csm_finetune.py
"""


raise SystemExit(
    "Deprecated T1 launcher. Use the vendor notebook at "
    "references/unsloth-tts-notebooks/Sesame_CSM_1B_TTS.ipynb with the Adja runbook at "
    "experiments/tts/T1_sesame_csm_finetune/UPSTREAM_NOTEBOOK_RUNBOOK.md, "
    "or use scripts/hf_jobs/T1_sesame_csm_finetune.py."
)
