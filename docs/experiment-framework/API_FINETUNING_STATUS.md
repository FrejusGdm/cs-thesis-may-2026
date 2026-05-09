# API Fine-Tuning Status & Plan

Last updated: 2026-02-14

## Current Status

### OpenAI (gpt-4.1-2025-04-14)
- **Completed**: 1/45 core-tier jobs (`exp1/STRUCTURED-2K/seed42` -> BLEU 15.2, chrF 26.1)
- **Running**: STOPPED. Paused pending budget approval.
- **Budget remaining**: ~$28 on the OpenAI account
- **Problem**: Most jobs exceed remaining quota (RANDOM-10K alone costs $35/job)

### Gemini
- **AI Studio**: NOT AVAILABLE. Google dropped fine-tuning support from AI Studio (May 2025).
- **Vertex AI**: Only option. Requires GCP project + billing + GCS bucket.
- **Status**: Not started, pending setup.

---

## Budget Request for Professor

### What we need to run

**Core tier** = the minimum set of experiments for the paper:
- 6 main conditions (exp1) x 5 random seeds = 30 jobs
- 3 baseline conditions x 5 random seeds = 15 jobs
- **Total: 45 jobs per platform**

### OpenAI costs (fine-tuning gpt-4.1)

| Condition | Train examples | Cost per job | Jobs | Subtotal |
|-----------|---------------|-------------|------|----------|
| STRUCTURED-2K | 1,800 | $8.81 | 4 remaining | $35 |
| LENGTH-STRATIFIED | 1,800 | $8.81 | 5 | $44 |
| TF-IDF-DIVERSE | 1,800 | $8.81 | 5 | $44 |
| VOCAB-MAXIMIZED | 1,800 | $8.81 | 5 | $44 |
| STRUCTURED-4K-ONLY | 3,116 | $15.25 | 5 | $76 |
| RANDOM-4K | 3,600 | $17.62 | 5 | $88 |
| RANDOM-10K | 7,200 | $35.24 | 5 | $176 |
| RANDOM-6K_STRUCTURED-4K | 8,516 | $41.67 | 5 | $208 |
| RANDOM-10K_STRUCTURED-4K | 10,316 | $50.49 | 5 | $252 |
| **44 remaining jobs** | | | | **~$970** |

Already spent: ~$9 (1 completed job). Already in account: ~$28.

**Option A — gpt-4.1 (full-size model):** Add ~$940 to OpenAI account. Total ~$970.
**Option B — gpt-4o-mini (smaller, 8x cheaper):** Add ~$95. Total ~$120-150.
**Option C — Run only cheap conditions:** STRUCTURED-2K + baselines + STRUCTURED-4K + RANDOM-4K = 29 jobs, ~$290 with gpt-4.1 or ~$36 with gpt-4o-mini. Skips the 3 large-data conditions (15 jobs).

### Gemini / Vertex AI costs

- Vertex AI fine-tuning: ~$2-5 per 1M training tokens (much cheaper than OpenAI)
- Estimated total for 45 core-tier jobs: **~$50-100**
- Requires: GCP project with billing, GCS bucket, `gcloud` auth
- Free trial: Google Cloud offers $300 in free credits for new accounts

### Inference costs (test evaluation)

Each job also runs inference on 1,455 test sentences after training:
- OpenAI gpt-4.1: ~$0.50-1.00 per job in inference -> ~$22-45 total
- OpenAI gpt-4o-mini: ~$0.05-0.10 per job -> ~$2-5 total
- Gemini: ~$0.01-0.05 per job -> negligible

### Summary of options

| Option | OpenAI model | OpenAI cost | Gemini cost | Total |
|--------|-------------|-------------|-------------|-------|
| Full (both platforms, gpt-4.1) | gpt-4.1 | ~$1,000 | ~$75 | ~$1,075 |
| Full (both platforms, gpt-4o-mini) | gpt-4o-mini | ~$150 | ~$75 | ~$225 |
| OpenAI only, gpt-4o-mini | gpt-4o-mini | ~$150 | $0 | ~$150 |
| Cheap conditions only, gpt-4.1 | gpt-4.1 | ~$290 | $0 | ~$290 |

### Why we need multiple seeds and conditions

- **5 seeds per condition**: Required for statistical significance (std dev, confidence intervals). Reviewers will reject single-seed results.
- **9 conditions**: Tests the core hypothesis (structured data > random data at same size) plus baselines and controls.
- **Multiple platforms**: Shows findings generalize beyond one model architecture.

---

## TODO

### Blocked on budget approval
- [ ] Get professor approval on budget (which option above?)
- [ ] Add OpenAI credits based on chosen option
- [ ] Re-run OpenAI launcher (skip logic auto-resumes):
  ```bash
  bash experiments/training/launch_api_finetune.sh --platform openai --tier core
  ```

### Gemini / Vertex AI setup (when ready)
- [ ] Set up GCP project with billing (or use $300 free trial)
- [ ] Create GCS bucket for training data
- [ ] Run `gcloud auth application-default login`
- [ ] Fix `gemini_finetune.py` BASE_MODEL (`gemini-2.5-flash` is invalid for tuning)
  - Likely need `models/gemini-2.0-flash-001` or check Vertex AI docs
- [ ] Upload JSONL files to GCS
- [ ] Launch:
  ```bash
  bash experiments/training/launch_api_finetune.sh \
      --platform gemini --tier core \
      --gcp-project YOUR_PROJECT --gcs-bucket YOUR_BUCKET
  ```

### If budget is tight — model switch
- [ ] To use gpt-4o-mini instead, change `BASE_MODEL` in `openai_finetune.py`:
  ```python
  BASE_MODEL = "gpt-4o-mini-2024-07-18"  # instead of "gpt-4.1-2025-04-14"
  ```
- [ ] Delete existing STRUCTURED-2K/seed42 result to re-run with new model (for consistency)

### After core tier completes
- [ ] Collect results across all platforms
- [ ] Compare NLLB vs GPT-4.1/4o-mini vs Gemini across conditions
- [ ] Generate paper figures

---

## What's already working

- All training data is converted to JSONL format at `experiments/data/finetune_formats/`
- Launcher script handles: job dispatch, skip-if-done, logging, .env auto-loading
- One successful proof-of-concept: OpenAI STRUCTURED-2K/seed42 -> BLEU 15.2
- Re-running the launcher after adding credits picks up where it left off

## Files Reference

| File | Purpose |
|------|---------|
| `experiments/training/openai_finetune.py` | Single OpenAI fine-tune + eval job |
| `experiments/training/gemini_finetune.py` | Single Gemini fine-tune + eval job |
| `experiments/training/launch_api_finetune.sh` | Orchestrator (all conditions/seeds, skip logic) |
| `experiments/training/convert_to_finetune_format.py` | Convert splits to JSONL |
| `experiments/data/finetune_formats/` | Generated JSONL files |
| `results/openai/` | OpenAI results (test_metrics.json per condition/seed) |
| `results/gemini/` | Gemini results (when ready) |
| `launch_api_finetune.log` | Launcher log (all runs appended) |
