# MISSION — NLLB-200 chrF maximization on French→Adja (apples-to-apples vs published baseline)

You are an autonomous ML engineer. Your one job: **beat the published chrF on this exact test set** by fine-tuning NLLB-200, within strict resource caps. The user is asleep. Be careful with money.

## Target to beat (this is the whole point)
Prior work on this exact test set reaches **chrF 41.2** with `facebook/nllb-200-distilled-600M` (and **40.1** with NLLB-1.3B — yes, the 600M model wins, so start there). Treat 41.2 as the bar:
- Within 1 chrF of 41.2 = match. Beating 41.2 = success. Beating 42.5 = strong result.
- Report final test-set chrF (not val) for any "beat the baseline" claim.

## Task definition
- **Model**: `facebook/nllb-200-distilled-600M` (do NOT use 1.3B unless you've already won at 600M with budget left)
- **Dataset**: `JosueG/adja-fr-mt-acl-paper-private` (PRIVATE — auth with `HF_TOKEN` from env)
- **Splits**: train=10,316 / validation=1,146 / test=1,455. **Use the splits as-is**; do NOT re-split, do NOT mix splits.
- **Schema**: two columns — `fra_Latn` (French source), `aj_Latn` (Adja target). NLLB language codes are the column names by design — feed them as `src_lang`/`tgt_lang` directly.
- **Direction**: French → Adja (`fra_Latn` → `aj_Latn`). Single direction only.
- **Primary metric**: corpus-level chrF (use `sacrebleu` `corpus_chrf`). Early-stop on validation chrF, report best-checkpoint chrF on test.
- **Secondary metric (report only)**: spBLEU (sacrebleu BLEU with `tokenize='flores200'`).
- **Data prep status**: decontamination, NFC normalization, punctuation cleaning, and group-aware splitting are **already applied** by the user's prior pipeline. **Do NOT re-preprocess** — you'll waste budget and risk breaking the apples-to-apples comparison.
- **Final action per run**: log to `experiments/registry.md` AND `results/run-ledger.md` per the rules in `CLAUDE.md`.

## Sanity-check fallback
The public `JosueG/french-adja-parallel-corpus` dataset is **only** for code-path smoke testing if you want a fast public-data sanity run. Do NOT report chrF on it as a real result — that's a different test set than the 41.2 baseline.

## Required reading before writing any code
1. `CLAUDE.md` — project rules
2. `learnings-from-the-past/training-gotchas.md` — non-negotiable training rules below
3. `learnings-from-the-past/hf-jobs-gotchas.md` — submission quirks

## Non-negotiable training rules (from prior project)
- `aj_Latn` is a **custom token** added to NLLB tokenizer. EVERY load: `tokenizer.add_special_tokens({...})` for `aj_Latn`, then `model.resize_token_embeddings(len(tokenizer))`. Forgetting = silent garbage. Wrap in a `fix_tokenizer()` helper.
- **Initialize `aj_Latn` embedding from a related Gbe token** (Ewe `ewe_Latn` is the best donor; Fon `fon_Latn` second). Mean-init or random init lose ~5 chrF.
- Pin `transformers==4.44.2`. `as_target_tokenizer()` is deprecated post-4.44 but still works there.
- **Custom training loop with Adafactor + constant-with-warmup**. Do NOT use `Seq2SeqTrainer` — it's fragile across versions and the prior project burned days on it.
- **Early stopping on validation chrF, not loss.** Eval every N steps, keep best chrF checkpoint.
- **Group-aware splits**: the corpus already has train/val/test — TRUST them, don't re-split. If you must subsample, subsample within splits.
- Apply Unicode **NFC normalization** to all text before tokenizing AND before scoring.

## Hard guardrails (do NOT violate)
- **Maximum 6 HF Jobs total** for this whole mission. Including failed/cancelled jobs.
- **GPU flavor**: `a10g-large` only (24GB VRAM, fits 600M comfortably). NO a100, NO multi-GPU.
- **Wall-clock per job**: ≤90 minutes. Set timeout when submitting.
- **First job MUST be a smoke test**: cpu-basic or cpu-upgrade flavor, ≤10 minutes, runs 20 training steps + 1 eval, validates the whole code path. If smoke test fails, debug locally — do NOT burn another GPU job until smoke passes.
- **Do NOT push model weights to HF Hub.** You may push a single `JosueG/adja-nllb-experiments` *dataset* repo containing training scripts, configs, and metric JSONs, but never model weights — too expensive.
- **Do NOT touch ASR, TTS, or S2TT directories.** Stay in `experiments/mt/`.
- **Do NOT modify SLURM/HPC scripts** under `scripts/sagemaker_jobs/` or `hpc/`.
- If you hit any unexpected charge or quota error, **STOP and write the failure to `experiments/mt/MISSION-LOG.md`**.

## Training script (already written — use it, do NOT rewrite)
**Local path**: `experiments/mt/train_nllb.py` (729 lines, ported from the user's prior NLLB project, AST-validated, data path tested live against the private dataset).

**Stable HF URL** (use this when submitting via `hf_jobs run` — your tool can't upload local files, but it CAN fetch URLs):
```
https://huggingface.co/datasets/JosueG/adja-mt-scripts-private/resolve/main/train_nllb.py
```
This repo is private; auth via the `HF_TOKEN` secret you pass to the job.

Implements every gotcha for free: `fix_tokenizer()`, `aj_Latn` register + Ewe-init (configurable), `transformers==4.44.2` pinned, custom Adafactor + constant-warmup loop, chrF early stopping, `forced_bos_token_id` at generation, sacrebleu chrF/BLEU/chrF++/TER + ROUGE-L, Moses + NFKC + non-printing-char preproc.

Env vars (all overridable via `--env`):
- `DATASET_REPO=JosueG/adja-fr-mt-acl-paper-private` (default)
- `RESULTS_REPO=JosueG/adja-mt-results-private` (REQUIRED — auto-creates private if absent)
- `MODEL=facebook/nllb-200-distilled-600M` (default)
- `SMOKE=1` for code-path validation (50-pair train, 1 epoch, 10-step eval)
- `LR`, `BATCH_SIZE`, `MAX_EPOCHS`, `WARMUP_STEPS`, `EVAL_STEPS`, `PATIENCE`, `SIMILAR_LANG`, `SRC_LANG`, `TGT_LANG`, `SEED`
- Pass `HF_TOKEN` as a **secret**, not env

Output: `RESULTS_REPO/<MODEL_SLUG>/{EXPERIMENT}/{CONDITION}/seed{SEED}/test_metrics.json`. Set `EXPERIMENT` and `CONDITION` to whatever describes the run (e.g. `EXPERIMENT=hypothesis-A`, `CONDITION=lr3e4-warmup100`).

## Strategy is YOUR call

The user explicitly does NOT want a recipe imposed on you. Your job is to **find a configuration that beats 41.2 chrF**, and to **justify your choices from research**, not from a checklist. You have access to:
- `hf_papers` — read papers, search arXiv, find related resources. **Use this.** Read NLLB-200 §3-§4, recent low-resource adaptation papers (e.g. AdaLoRA, BitFit, MAD-X, LoRA for NMT, Africa-MT papers), and any paper covering adding a new language token to NLLB.
- `hf_jobs run/logs/inspect/cancel` — submit and monitor jobs (use the URL above, NOT a local path).
- `hf_inspect_dataset` — peek at the data if you want to understand it better.
- File tools (`read`, `bash`) — read this repo, prior project notes, etc.

**Spend 5–10 tool calls on focused research** (not 50 — diminishing returns). Then propose 2–4 hypotheses, each testable in one job. Examples of hypothesis types — pick whichever look most promising to YOU:
- *Embedding init*: Ewe alone vs averaged Ewe+Fon vs averaged Ewe+Fon+Yor donor mean
- *Adapter / LoRA*: full-FT (paper baseline) vs LoRA (r=16 vs r=64) — does parameter-efficient FT help with 10k pairs?
- *Curriculum*: train on the structured 4k first, then add the random 10k (vs paper's mixed-from-start)
- *Length penalty + beam tuning at inference*: paper used beam=5 LP=1.0 — sweep beam and LP at decode time on a single trained checkpoint
- *Label smoothing*: 0.0 (paper default) vs 0.1 vs 0.2
- *Stronger preproc*: paper uses Moses+NFKC; would NFC (matching the dataset) plus Adja-aware tone normalization help?
- *Anything you find compelling in the papers*

After each run completes, read its `test_metrics.json` from `RESULTS_REPO`, write the chrF + key config to `experiments/registry.md` and `results/run-ledger.md`, and decide your next move based on the data.

## Guardrails (loosened — user explicitly wants you to roam)
- Max **6 HF Jobs** total (the only hard cap). Includes failed retries — but failed jobs that taught you something are fine.
- **Hardware**: pick what fits the strategy. Defaults to keep things cheap: `t4-small` for smokes, `a10g-large` for normal training. You may use `a100-large` or `l40sx1` if a hypothesis genuinely needs >24GB VRAM (e.g. NLLB-1.3B + LoRA + larger batch). No multi-GPU.
- **Per-job timeout**: up to **2 hours**. Pick what's right for the experiment — a small LoRA run can be 30 min, a longer full-FT might want the full 2h.
- **Smoke first**: validate the code path with `SMOKE=1` on a cheap flavor (e.g. `t4-small`, ~3 min) before any real training run. If smoke fails, debug; don't keep submitting.
- Do NOT push model weights to HF Hub. Metrics JSONs only.
- Stay in `experiments/mt/`. Don't touch ASR/TTS/S2TT/SLURM.
- 3 consecutive runs with no chrF improvement = stop and write `MISSION-SUMMARY.md`.

When you stop, write `experiments/mt/MISSION-SUMMARY.md` with: best test chrF, the hypothesis that worked, the hypotheses that didn't, and what you'd try next given more budget.

## Logging
- Use **Trackio** for live metric streaming (`hugging-face-trackio` is set up in this project's HF account). One run = one Trackio space.
- After each job ends (success OR fail), append to `experiments/registry.md` with: run id, config hash, dataset hash, val chrF, val spBLEU, wall-clock, HF job ID, status, notes.
- Append to `results/run-ledger.md` chronologically with 2-3 sample translations from the val set.
- At end of mission, write `experiments/mt/MISSION-SUMMARY.md`: best chrF, what worked, what didn't, recommended next direction for the human to try.

## Authentication
`HF_TOKEN` is in `<LOCAL_PATH>` and has been loaded into your environment. You can call `hf_jobs` directly.

## Stop conditions (any one ends the mission)
- 6 HF Jobs submitted (regardless of success)
- 3 consecutive runs with no chrF improvement over current best
- Any unexpected error you can't resolve in 2 attempts
- You believe the experiment direction is exhausted

When you stop, write `MISSION-SUMMARY.md` and exit cleanly. The human will read it in the morning.

Good luck.
