# OmniASR 7B Fine-tuning Log

This directory tracks the 2026-04-19 OmniASR 7B Adja fine-tuning work.

Scope:

- `omniASR_CTC_7B_v2`
- `omniASR_LLM_7B_v2`

Launchers:

- `scripts/hf_jobs/omni_asr_finetune.py`
- `scripts/hf_jobs/submit_omni_jobs.py`

Collision policy:

- do not reuse old Omni result prefixes
- do not share output repos between CTC and LLM
- every submission tag should be unique (`2026-04-19-a1`, `2026-04-19-a2`, ...)

Expected artifact locations:

- results repo: `JosueG/adja-asr-results/Omni_FT_*`
- model repos:
  - `JosueG/omniASR-ctc-7b-v2-adja`
  - `JosueG/omniASR-llm-7b-v2-adja`
