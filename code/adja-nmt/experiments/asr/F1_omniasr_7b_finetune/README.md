# F1: OmniASR 7B Fine-tuning on Adja

Purpose: run the largest released Meta OmniASR checkpoints on Adja in a clean,
experiment-specific workspace without overwriting the earlier Omni zero-shot or
ICL artifacts.

This folder tracks two related runs:

- `omniASR_CTC_7B_v2`
- `omniASR_LLM_7B_v2`

The execution path is the repo-native Hugging Face Jobs launcher:

- submit helper: `scripts/hf_jobs/submit_omni_jobs.py`
- job script: `scripts/hf_jobs/omni_asr_finetune.py`

Operational notes:

- The job script is self-contained for `hf jobs uv run`.
- It materializes the Adja dataset into Omni's manifest format at runtime.
- It downloads the official `facebookresearch/omnilingual-asr` source tree so
  the upstream fairseq2 recipe can run without us vendoring the whole repo.
- Result identifiers must include a unique tag to avoid collisions with prior
  Omni runs.

Default launch commands:

```bash
python scripts/hf_jobs/submit_omni_jobs.py --family both --smoke
python scripts/hf_jobs/submit_omni_jobs.py --family both --smoke --execute --detach
python scripts/hf_jobs/submit_omni_jobs.py --family both --pilot --execute --detach
```

Initial namespace plan:

- CTC output repo: `JosueG/omniASR-ctc-7b-v2-adja`
- LLM output repo: `JosueG/omniASR-llm-7b-v2-adja`
- Results repo: `JosueG/adja-asr-results`
- Results prefixes:
  - `Omni_FT_CTC_7B_v2_*`
  - `Omni_FT_LLM_7B_v2_*`
