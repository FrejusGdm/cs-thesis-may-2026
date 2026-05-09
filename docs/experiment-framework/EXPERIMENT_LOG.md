# Experiment Log

Running log of all experiment runs. Add entries as experiments complete.

---

## Entry Template

Copy this block for each completed run:

```
### [DATE] — Exp[N]: [EXPERIMENT_NAME] — [CONDITION]

**Seed:** [42/123/456/789/1024]
**Data:** [description, size]
**Model:** [model name, HuggingFace ID or path]
**Compute:** [GPU type, platform (Colab/HF/Dartmouth HPC), training time]
**Config:** [path to config file used]

**Data split:**
- Train: [N] sentences
- Validation: [N] sentences
- Test: HELDOUT-DIVERSE (300)

**Results:**
| Metric | Value |
|--------|-------|
| BLEU   |       |
| chrF   |       |
| chrF++ |       |
| TER    |       |
| COMET  |       |
| BERTScore (F1) | |

**SacreBLEU signature:** `BLEU+case.mixed+numrefs.1+smooth.exp+tok.13a+version.X.X.X`

**Training curve:**
- Best epoch: [N]
- Best val chrF: [value]
- Training stopped at epoch: [N] (early stopping / max epochs)

**Notes:**
- [Observations, issues, surprises]

**Reproducibility:**
- Git hash: [commit hash]
- Library versions: transformers=X.X.X, sacrebleu=X.X.X, torch=X.X.X
- Random seed verified: [yes/no]

**Checkpoint:** [path to saved model]
**Output:** [path to generated translations]
**Predictions file:** [path to raw predictions TSV]
```

---

## Log Entries

(Add entries below as experiments are completed)

---
