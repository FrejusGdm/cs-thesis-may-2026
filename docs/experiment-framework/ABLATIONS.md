# Ablation Studies

3 seeds per ablation condition unless noted otherwise.
All ablations use NLLB-200-distilled-600M with shared hyperparameters.

---

## Ablation 1: Module Ablations (Leave-One-Out)

**Question:** Which modules contribute most to translation quality?

| Condition | Modules Included | Approx Size | BLEU | chrF | COMET | Status |
|-----------|-----------------|-------------|------|------|-------|--------|
| FULL | M1+M2+M3+M4+M5 | ~4,266 | — | — | — | TODO |
| -NEGATION | M1+M3+M4+M5 | ~3,362 | — | — | — | TODO |
| -PAST | M1+M2+M4+M5 | ~3,362 | — | — | — | TODO |
| -FUTURE | M1+M2+M3+M5 | ~3,362 | — | — | — | TODO |
| -QUESTIONS | M1+M2+M3+M4 | ~3,616 | — | — | — | TODO |
| BASE-ONLY | M1 only | ~904 | — | — | — | TODO |

**What to look for:**
- If -QUESTIONS drops most, question formation is the hardest for the model to learn independently
- If BASE-ONLY is close to FULL, the transformations are indeed redundant (Reviewer Objection 8)
- If BASE-ONLY << FULL, minimal pairs provide genuine learning signal

## Ablation 1b: Size-Controlled Module Ablation

**Question:** Same as above, but controlling for total data size (1,000 sentences each).

This isolates module *diversity* from module *quantity*. Each condition has exactly 1,000 sentences, distributed across available modules.

| Condition | Modules | Sentences per Module | Size | BLEU | chrF | Status |
|-----------|---------|---------------------|------|------|------|--------|
| FULL-1K | M1+M2+M3+M4+M5 | 200 each | 1,000 | — | — | TODO |
| -NEG-1K | M1+M3+M4+M5 | 250 each | 1,000 | — | — | TODO |
| -PAST-1K | M1+M2+M4+M5 | 250 each | 1,000 | — | — | TODO |
| -FUT-1K | M1+M2+M3+M5 | 250 each | 1,000 | — | — | TODO |
| -QUEST-1K | M1+M2+M3+M4 | 250 each | 1,000 | — | — | TODO |
| BASE-1K | M1 only | 1,000 | 1,000 | — | — | TODO |

## Ablation 2: Pronoun Coverage

**Question:** How many pronoun forms are needed for the model to generalize?

All conditions use the full structured dataset but filtered to include only the specified pronouns.

| Condition | Pronouns | Approx Size | BLEU | chrF | Status |
|-----------|----------|-------------|------|------|--------|
| ALL-8 | je, tu, il, elle, nous, vous, ils, elles | ~4,266 | — | — | TODO |
| REDUCED-4 | je, tu, il, nous | ~2,133 | — | — | TODO |
| SINGULAR-3 | je, tu, il | ~1,600 | — | — | TODO |
| MINIMAL-1 | je | ~533 | — | — | TODO |

**What to look for:**
- Steep drop from ALL-8 to REDUCED-4 suggests pronoun diversity is critical
- Small drop suggests the model generalizes well from limited pronoun exposure
- Compare MINIMAL-1 to RANDOM at equal size for information density

## Ablation 3: Verb Diversity

**Question:** How many verb types are needed?

For 5-VERBS and 3-VERBS, run 3 random subsets each and report mean/std.

| Condition | Verb Count | Which Verbs | BLEU | chrF | Status |
|-----------|-----------|-------------|------|------|--------|
| 10-VERBS | All 10 | Full vocabulary | — | — | TODO |
| 5-VERBS-a | 5 | Random subset a | — | — | TODO |
| 5-VERBS-b | 5 | Random subset b | — | — | TODO |
| 5-VERBS-c | 5 | Random subset c | — | — | TODO |
| 3-VERBS-a | 3 | Random subset a | — | — | TODO |
| 3-VERBS-b | 3 | Random subset b | — | — | TODO |
| 3-VERBS-c | 3 | Random subset c | — | — | TODO |
| 1-VERB | 1 | manger only | — | — | TODO |

**What to look for:**
- How fast does performance degrade with fewer verbs?
- Is there a "sweet spot" (e.g., 5 verbs captures 90% of the benefit)?
- Does variance across random subsets tell us about verb choice sensitivity?

## Ablation 4: Minimal-Pair Structure

**Question:** Does the contrastive structure (linking M2-M5 to M1 base sentences) matter, or would randomly shuffled transformations work equally well?

| Condition | Description | BLEU | chrF | Status |
|-----------|-------------|------|------|--------|
| PAIRS-INTACT | Original: each M2-M5 sentence linked to its M1 base | — | — | TODO |
| PAIRS-BROKEN | Same sentences, but shuffle M2-M5 independently (break base_sentence_id) | — | — | TODO |

**Construction of PAIRS-BROKEN:**
```python
# Shuffle each module's sentences independently, breaking alignment
for module in ["M2", "M3", "M4", "M5"]:
    module_mask = df["module"].str.startswith(f"module{module[1]}")
    shuffled = df.loc[module_mask, "french"].sample(frac=1, random_state=42).values
    df.loc[module_mask, "french"] = shuffled
```

**What to look for:**
- If PAIRS-INTACT >> PAIRS-BROKEN, contrastive structure provides genuine signal
- If equal, the model doesn't leverage minimal pairs (the sentences themselves matter, not their pairing)

---

## Summary Table (filled after all ablations complete)

| Ablation | Key Finding | Implication |
|----------|-------------|-------------|
| Module leave-one-out | | |
| Module size-controlled | | |
| Pronoun coverage | | |
| Verb diversity | | |
| Minimal-pair structure | | |

## Analysis Notes

(Add observations as ablations are completed)
