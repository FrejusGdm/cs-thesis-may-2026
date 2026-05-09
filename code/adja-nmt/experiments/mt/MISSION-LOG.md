# Mission Log — NLLB chrF mission

## 2026-04-28

### Mission intake
- Read `experiments/mt/MISSION.md`, `CLAUDE.md`, `learnings-from-the-past/training-gotchas.md`, and `learnings-from-the-past/hf-jobs-gotchas.md`.
- Verified target dataset `JosueG/adja-fr-mt-acl-paper-private` via `hf_inspect_dataset`.
  - Splits: train / validation / test
  - Schema: `fra_Latn`, `aj_Latn`
  - Sample rows look aligned French→Adja and suitable for seq2seq translation.
- Confirmed the mission guardrails:
  - max 6 HF jobs total
  - GPU jobs on `a10g-large` or better if needed
  - no model weights pushed to Hub
  - update `experiments/registry.md` and `results/run-ledger.md` after each run
  - write `MISSION-SUMMARY.md` at stop condition

### Research notes
- Papers / findings pointed to three promising hypotheses:
  1. related-language / target-token initialization for the new `aj_Latn` token
  2. low-resource fine-tuning with heavy Adja upsampling and careful LR control
  3. decode-time beam / length-penalty tuning on a fixed checkpoint
- Most actionable decode-time result: beam and length penalty matter; moderate beams with normalization often beat greedy or very wide beams.
- Most actionable token-init result: initializing a new language token from a related African language embedding is a strong prior.

### HF Jobs submissions
- Intended to use stable script URL:
  `https://huggingface.co/datasets/JosueG/adja-mt-scripts-private/resolve/main/train_nllb.py`

### Failed launch diagnosis
- First 3 GPU submissions were missing required env var `RESULTS_REPO`.
- Error from job logs:
  `KeyError: 'RESULTS_REPO'`
- This is a submission/config bug, not a model or dataset issue.

### Exact job IDs tried
- `69f0e8c9d2c8bd8662bd22e2` — failed, missing `RESULTS_REPO`
- `69f0e8e1d70108f37ace11cd` — failed, missing `RESULTS_REPO`
- `69f0e8f8d2c8bd8662bd22e4` — running at time of inspection, but logs show the same missing-env bug path if not corrected

### Current conclusion
- I have a valid path forward, but the three submitted jobs were misconfigured because `RESULTS_REPO` was not passed in the job env/secret map.
- Next step: resubmit with full env vars, then monitor the first real hypothesis run and stop if needed after the mission stop conditions are met.
