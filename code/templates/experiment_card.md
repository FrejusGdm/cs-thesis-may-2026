# Experiment Card: [EXPERIMENT_NAME]

## Metadata
- **Date started:** YYYY-MM-DD
- **Date completed:** YYYY-MM-DD
- **Experimenter:** [name]
- **Compute:** [GPU type, platform (Colab/HF/Dartmouth HPC)]
- **Git hash:** [commit hash at time of experiment]

## Hypothesis
[What are you testing? One clear sentence.]

## Conditions

| Condition | Data | Size | Description |
|-----------|------|------|-------------|
| | | | |
| | | | |

## Configuration

```yaml
model: facebook/nllb-200-distilled-600M
source_lang: fra_Latn
target_lang: fon_Latn  # proxy for Adja
learning_rate: 2e-5
lr_schedule: linear_warmup_10pct_then_linear_decay
batch_size: 16
max_epochs: 50
early_stopping_patience: 10
early_stopping_metric: eval_chrf
label_smoothing: 0.1
beam_size: 5
length_penalty: 1.0
max_seq_length: 128
seeds: [42, 123, 456, 789, 1024]
```

## Results

### Automatic Metrics (mean +/- std across 5 seeds)

| Condition | BLEU | chrF | chrF++ | TER | COMET | BERTScore (F1) |
|-----------|------|------|--------|-----|-------|----------------|
| | | | | | | |
| | | | | | | |

### Per-Seed Breakdown (for reproducibility)

| Condition | Seed | BLEU | chrF | COMET |
|-----------|------|------|------|-------|
| | 42 | | | |
| | 123 | | | |
| | 456 | | | |
| | 789 | | | |
| | 1024 | | | |

### Statistical Significance (paired bootstrap, p < 0.05 with Bonferroni)

| Comparison | BLEU diff | chrF diff | p-value (BLEU) | p-value (chrF) | Cohen's d | Significant? |
|------------|-----------|-----------|----------------|----------------|-----------|-------------|
| | | | | | | |

### Human Evaluation (if applicable)

| Condition | Adequacy (1-5) | Fluency (1-5) | Pairwise preference |
|-----------|---------------|---------------|---------------------|
| | | | |

## Analysis
[What did you learn? Does the hypothesis hold?]

## In-Distribution vs Out-of-Distribution Breakdown

| Condition | chrF (in-dist) | chrF (out-of-dist) | chrF (overall) |
|-----------|----------------|--------------------| --------------|
| | | | |

## Surprises / Issues
[Anything unexpected? Failures? Things to investigate?]

## Next Steps
[What experiment should follow from these results?]
