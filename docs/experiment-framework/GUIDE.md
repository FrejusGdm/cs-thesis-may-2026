# Experiment Framework Guide

How to navigate this framework, where to start, and how to run experiments across platforms.

---

## Reading Order

If you're opening this repo for the first time:

1. **This file** — you're here, keep going
2. **README.md** — master experiment tracker, all conditions and their status
3. **BASELINES.md** — how each baseline is constructed (with code snippets)
4. **ABLATIONS.md** — what each ablation tests and what to look for
5. **REVIEWER_DEFENSE.md** — anticipated reviewer objections and preemptive responses
6. **configs/base_config.yaml** — shared hyperparameters for all experiments

---

## Directory Map

```
experiments/
├── GUIDE.md                ← YOU ARE HERE
├── README.md               # Master tracker: all experiments, conditions, results tables
├── EXPERIMENT_LOG.md       # Running log — paste results here as runs complete
├── BASELINES.md            # 7 baseline recipes with Python code
├── ABLATIONS.md            # 4 ablation studies with analysis guidance
├── REVIEWER_DEFENSE.md     # 10 anticipated objections + responses
├── requirements.txt        # pip dependencies for all scripts
│
├── configs/                # YAML configs (what to run)
│   ├── base_config.yaml        # Shared hyperparameters
│   ├── experiment1_primary.yaml # Exp 1 conditions
│   ├── experiment2_scaling.yaml # Exp 2 conditions
│   ├── experiment3_curriculum.yaml
│   ├── experiment4_crossverb.yaml
│   ├── experiment5_architecture.yaml
│   ├── baselines.yaml
│   └── ablations.yaml
│
├── data/                   # Data preparation (run BEFORE training)
│   ├── prepare_splits.py       # 80-10-10 train/val/test splits (group-aware)
│   ├── prepare_baselines.py    # Construct baseline training sets
│   ├── prepare_ablations.py    # Construct ablation subsets
│   └── audit_random.py         # Quality audit for 10K random data
│
├── training/               # Training scripts (run ON GPU platforms)
│   ├── train_nllb.py           # Fine-tune NLLB-200 (single run)
│   ├── run_experiment.py       # Run one condition across all seeds
│   └── run_all_experiments.sh  # Orchestrate all ~300 runs
│
├── evaluation/             # Evaluation scripts (run AFTER training)
│   ├── evaluate.py             # Compute BLEU, chrF, COMET, BERTScore
│   ├── significance.py         # Statistical significance tests
│   └── collect_results.py      # Aggregate results into tables
│
├── analysis/               # Analysis scripts
│   └── data_statistics.py      # Vocab overlap, TTR, entropy
│
├── templates/              # Templates for paper
│   ├── experiment_card.md      # Per-experiment report template
│   ├── results_table.md        # LaTeX table templates (6 tables)
│   ├── human_eval_form.md      # Scoring sheet for Adja evaluators
│   └── error_analysis_form.md  # Error categorization form
│
├── results/                # Raw results (created by training scripts)
│   └── .gitkeep
│
└── notebooks/              # Your Colab/Jupyter notebooks (put them here)
    └── (bring your NLLB notebook here)
```

---

## Getting Started (Step by Step)

### Step 0: Install dependencies

```bash
pip install -r experiments/requirements.txt
```

### Step 1: Bring in your data

You need two CSV files with columns `french` and `adja_translation`:

```
your_data/
├── structured_run1.csv      # ~1,982 sentences (Code-Run-1)
├── structured_run2.csv      # ~2,284 sentences (Code-Run-2)
└── tatoeba_random_10k.csv   # ~10,000 random Tatoeba sentences
```

If your column names differ, use `--src-col` and `--tgt-col` flags on all scripts.

### Step 2: Audit the random data (optional but recommended)

```bash
python experiments/data/audit_random.py \
    --input your_data/tatoeba_random_10k.csv \
    --output your_data/tatoeba_random_clean.csv
```

This flags duplicates, empty translations, and outliers. If you have your own cleaning scripts, run those first, then run this audit on the output.

### Step 3: Create train/val/test splits

This is the most critical step. It creates 80-10-10 splits with group-aware splitting for structured data (no minimal-pair leakage).

```bash
python experiments/data/prepare_splits.py \
    --structured-run1 your_data/structured_run1.csv \
    --structured-run2 your_data/structured_run2.csv \
    --random your_data/tatoeba_random_10k.csv \
    --output-dir experiments/data/splits \
    --val-ratio 0.1 \
    --test-ratio 0.1 \
    --seed 42
```

This creates:
- `splits/shared/test.tsv` — shared test set (ALL experiments use this)
- `splits/exp1/RANDOM-10K/train.tsv`, `val.tsv` — per-condition splits
- `splits/exp2/...` — scaling curve splits
- `splits/manifest.json` — record of what was created

### Step 4: Create baseline and ablation splits

```bash
# Baselines (from random data train portion)
python experiments/data/prepare_baselines.py \
    --random experiments/data/splits/random_train.csv \
    --output-dir experiments/data/splits/baselines

# Ablations (from structured data train portion)
python experiments/data/prepare_ablations.py \
    --structured experiments/data/splits/structured_train.csv \
    --output-dir experiments/data/splits/ablations
```

### Step 5: Run training (on GPU platform)

See "Multi-Platform Workflow" below for how to do this on Colab/HF/HPC.

```bash
# Single condition, single seed:
python experiments/training/train_nllb.py \
    --train-file experiments/data/splits/exp1/RANDOM-10K/train.tsv \
    --val-file experiments/data/splits/exp1/RANDOM-10K/val.tsv \
    --test-file experiments/data/splits/shared/test.tsv \
    --output-dir experiments/results/exp1/RANDOM-10K/seed42 \
    --seed 42

# Single condition, all 5 seeds:
python experiments/training/run_experiment.py \
    --experiment exp1 --condition RANDOM-10K \
    --splits-dir experiments/data/splits \
    --results-dir experiments/results \
    --test-file experiments/data/splits/shared/test.tsv
```

### Step 6: Evaluate

```bash
# Full metrics for one condition (all seeds):
python experiments/evaluation/evaluate.py \
    --predictions-dir experiments/results/exp1/RANDOM-10K/ \
    --output experiments/results/exp1/RANDOM-10K/aggregated_metrics.json

# Significance tests between two conditions:
python experiments/evaluation/significance.py \
    --dir-a experiments/results/exp1/RANDOM-10K/ \
    --dir-b experiments/results/exp1/STRUCTURED-4K-ONLY/ \
    --output experiments/results/comparisons/random_vs_structured.json
```

### Step 7: Collect all results

```bash
python experiments/evaluation/collect_results.py \
    --results-dir experiments/results \
    --output-dir experiments/results/summary
```

This generates `results_summary.md` (paste into README) and `all_results.csv` (for plotting).

---

## Multi-Platform Workflow

You'll run experiments on Colab, HuggingFace, and Dartmouth HPC. Here's how the pieces fit.

### What to upload to each platform

Upload these files/folders:
```
experiments/training/train_nllb.py          # The training script
experiments/data/splits/shared/test.tsv     # Shared test set
experiments/data/splits/exp1/CONDITION/     # Train + val for the condition you're running
experiments/requirements.txt                # Dependencies
```

Or if using your Colab notebook, upload:
```
notebooks/your_nllb_notebook.ipynb
experiments/data/splits/shared/test.tsv
experiments/data/splits/exp1/CONDITION/train.tsv
experiments/data/splits/exp1/CONDITION/val.tsv
```

### What to download back

After each run, download:
```
results/CONDITION/seedN/
├── config.json           # Hyperparameters used
├── test_metrics.json     # BLEU, chrF, etc.
├── predictions.tsv       # src \t ref \t pred (for significance tests)
├── train_results.json    # Training loss curve
└── best_model/           # (optional: only if you want to keep the checkpoint)
```

Place these into:
```
experiments/results/exp1/CONDITION/seedN/
```

### Platform-specific notes

**Google Colab:**
- Upload splits via Colab file browser or mount Google Drive
- `!pip install -r requirements.txt` in first cell
- Either use `train_nllb.py` directly or adapt your existing notebook
- Download results folder when done

**HuggingFace:**
- Push splits to a private HF dataset
- Use `train_nllb.py` in a HF Space or training job
- Download results via HF API

**Dartmouth HPC (SLURM):**
- `scp` splits to HPC
- Create a SLURM job script wrapping `run_experiment.py`
- `scp` results back when done

### Integrating your Colab notebook

1. Put your notebook in `experiments/notebooks/`
2. Make sure it reads from TSV files (the format `prepare_splits.py` outputs)
3. Make sure it saves predictions as `predictions.tsv` with format: `source\treference\tprediction`
4. Make sure it saves metrics as `test_metrics.json`
5. The evaluation and significance scripts will work with these outputs regardless of whether they came from `train_nllb.py` or your notebook

### Integrating your cleaning scripts

1. Put your cleaning scripts in `experiments/data/`
2. Run them BEFORE `prepare_splits.py`
3. Feed the cleaned output into `prepare_splits.py --random your_cleaned_data.csv`

---

## Critical Path (What Blocks What)

```
[Your raw data files]
        │
        ▼
[audit_random.py] ──→ optional RANDOM-CLEAN
        │
        ▼
[prepare_splits.py] ──→ shared test set + all train/val splits
        │
        ├──→ [prepare_baselines.py] ──→ baseline splits
        │
        ├──→ [prepare_ablations.py] ──→ ablation splits
        │
        ▼
[train_nllb.py / your notebook] ──→ predictions.tsv + test_metrics.json
        │                               (run on Colab / HF / HPC)
        ▼
[evaluate.py] ──→ full_metrics.json (BLEU, chrF, COMET, BERTScore)
        │
        ▼
[significance.py] ──→ pairwise significance tests
        │
        ▼
[collect_results.py] ──→ summary tables for paper
```

**Nothing can start until `prepare_splits.py` runs.** That's your day-one task.

---

## Experiment Phases

Run experiments in this order (each phase builds on the previous):

| Phase | What to run | ~Runs | Priority |
|-------|-------------|-------|----------|
| **1. Core** | Experiment 1 (6 conditions x 5 seeds) + Experiment 2 scaling curves | ~135 | Do this first |
| **2. Ablations** | Module, pronoun, verb, minimal-pair ablations | ~85 | Explains WHY structured works |
| **3. Extensions** | Cross-verb generalization, architecture comparison, curriculum ordering | ~56 | Strengthens the paper |

Start with Phase 1. You can begin writing the paper after Phase 1 results are in — Phases 2-3 fill in the analysis sections.

---

## Quick Reference: Common Commands

```bash
# Audit random data
python experiments/data/audit_random.py --input DATA.csv --output CLEAN.csv

# Create all splits
python experiments/data/prepare_splits.py --structured-run1 R1.csv --structured-run2 R2.csv --random RAND.csv --output-dir experiments/data/splits

# Train one condition (all seeds)
python experiments/training/run_experiment.py --experiment exp1 --condition RANDOM-10K --splits-dir experiments/data/splits --results-dir experiments/results --test-file experiments/data/splits/shared/test.tsv

# Evaluate one condition
python experiments/evaluation/evaluate.py --predictions-dir experiments/results/exp1/RANDOM-10K/ --output experiments/results/exp1/RANDOM-10K/metrics.json --skip-comet

# Compare two conditions
python experiments/evaluation/significance.py --dir-a experiments/results/exp1/RANDOM-10K/ --dir-b experiments/results/exp1/STRUCTURED-4K-ONLY/

# Aggregate everything
python experiments/evaluation/collect_results.py --results-dir experiments/results --output-dir experiments/results/summary

# Data statistics
python experiments/analysis/data_statistics.py --structured STRUCT.csv --random RAND.csv --test TEST.csv --output stats.json
```
