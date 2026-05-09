# Experiment Tracker: Data Composition vs. Quantity in Low-Resource NMT

## Research Question

Does data structure matter more than quantity in extremely low-resource NMT (French -> Adja)?

**Central claim to validate:** 10K random alone -> BLEU 2-3; 10K random + ~4K structured -> BLEU ~20. Is this driven by the structured data's composition, or simply by having more data (14K vs 10K)?

## Data Inventory

| Dataset | Size | Status | Description |
|---------|------|--------|-------------|
| Tatoeba Random (10K) | ~10,000 | Translated | Random French sentences with Adja translations |
| Structured Run-1 | ~1,982 | Translated | 10 verbs (manger, boire, voir, aller, venir, faire, avoir, prendre, donner, vouloir) |
| Structured Run-2 | ~2,284 | Translated | 10 verbs (aimer, acheter, chercher, trouver, parler, savoir, mettre, laisser, apporter, montrer) |
| Structured Combined | ~4,266 | Translated | Both runs merged |
| Fr-Fon (English2Gbe) | ~53,000 | Available | Cross-lingual transfer source |
| Fr-Ewe (English2Gbe) | ~23,000 | Available | Cross-lingual transfer source |

### Test Set

All experiments share a single held-out test set created by `prepare_splits.py` via **80-10-10 group-aware splitting**:

| Split | Ratio | Source | Description |
|-------|-------|--------|-------------|
| Train | 80% | Structured + Random | Per-experiment subsets drawn from this pool |
| Val | 10% | Structured + Random | Per-condition validation |
| Test | 10% | Structured + Random | Shared across ALL experiments |

**Group-aware protocol:** For structured data, splits are made at the `base_sentence_id` group level — if a Module 1 base sentence is in test, ALL its transformations (M2-M5) go to test too. This prevents minimal-pair leakage between train and test.

**Verification:** `prepare_splits.py` runs a leakage check after splitting and will error if any `base_sentence_id` group appears in more than one split.

| Optional Test Set | Size | Status | Description |
|-------------------|------|--------|-------------|
| FLORES-PROXY | varies | Available | FLORES-200 devtest for Fon/Ewe as proxy |

---

## Experiments Overview

### Experiment 1: Primary Hypothesis — Does Structured Data Drive the BLEU ~20?

**Key test:** RANDOM-10K vs RANDOM-6K+STRUCTURED-4K — both 10K total. If mixed wins, it's composition, not size.

| Condition | Data | Size | BLEU | chrF | COMET | Status |
|-----------|------|------|------|------|-------|--------|
| RANDOM-10K | All 10K Tatoeba | 10,000 | — | — | — | TODO |
| RANDOM-6K + STRUCTURED-4K | 6K random + 4K structured | 10,000 | — | — | — | TODO |
| RANDOM-10K + STRUCTURED-4K | 10K + 4K structured (full data) | ~14,000 | — | — | — | TODO |
| STRUCTURED-4K-ONLY | 4K structured only | ~4,266 | — | — | — | TODO |
| RANDOM-4K | 4K random sample | 4,000 | — | — | — | TODO |
| STRUCTURED-2K | 2K structured subset | 2,000 | — | — | — | TODO |

**Critical comparison:** RANDOM-10K vs RANDOM-6K+STRUCTURED-4K (same total size, different composition).

### Experiment 2: Scaling Curves

| Type | Sizes | Seeds | Status |
|------|-------|-------|--------|
| Structured only | 200, 500, 1000, 2000, 3000, 4000 | 5 each | TODO |
| Random only | 200, 500, 1000, 2000, 4000, 6000, 8000, 10000 | 5 each | TODO |
| Combined additive (6K random + X structured) | X = 500, 1000, 2000, 4000 | 5 each | TODO |
| Combined replacement ((10K-X) random + X structured) | X = 500, 1000, 2000, 4000 | 5 each | TODO |

**Tests:** (a) steeper BLEU-per-sentence for structured, (b) whether replacing random with structured at fixed total size helps, (c) diminishing returns.

### Experiment 3: Curriculum-Ordered vs. Shuffled Training

| Condition | Description | Seeds | Status |
|-----------|-------------|-------|--------|
| CURRICULUM | Present M1 first, add modules progressively | 5 | TODO |
| SHUFFLED | Same data, random order | 5 | TODO |
| COMPETENCE | Platanios-style competence scheduling | 5 | TODO |

### Experiment 4: Cross-Verb Generalization

| Condition | Train | Test | Seeds | Status |
|-----------|-------|------|-------|--------|
| Run1->Run2 | Run-1 verbs (~1,982) | Run-2 verbs | 5 | TODO |
| Run2->Run1 | Run-2 verbs (~2,284) | Run-1 verbs | 5 | TODO |
| Combined | Both runs | Held-out from both | 5 | TODO |

### Experiment 5: Fine-Tuning vs. From Scratch

| Model | Params | Seeds | Status |
|-------|--------|-------|--------|
| NLLB-200-distilled-600M | 600M | 5 | TODO |
| NLLB-200-1.3B | 1.3B | 5 | TODO |
| mBART-50 | ~600M | 5 | TODO |
| Transformer-base (scratch) | ~65M | 5 | TODO |
| Transformer-tiny (scratch) | ~15M | 5 | TODO |

### Experiment 6: Data Augmentation & Transfer Learning

**Back-translation:**
| Condition | Description | Seeds | Status |
|-----------|-------------|-------|--------|
| STRUCTURED + BT-5K | Structured + 5K back-translated pairs | 3 | TODO |
| RANDOM + BT-5K | Random + 5K back-translated pairs | 3 | TODO |
| Iterative BT (3 rounds) | Track convergence over BT iterations | 3 | TODO |

**Cross-lingual transfer from Fon/Ewe:**
| Condition | Data | Seeds | Status |
|-----------|------|-------|--------|
| ADJA-ONLY | 4K structured only | 3 | TODO |
| ADJA + FON | 4K structured + 53K Fr-Fon | 3 | TODO |
| ADJA + EWE | 4K structured + 23K Fr-Ewe | 3 | TODO |
| ADJA + FON + EWE | 4K structured + all Gbe data | 3 | TODO |
| FON-PRETRAIN -> ADJA | Two-stage: pretrain on Fon, fine-tune on Adja | 3 | TODO |

---

## Baselines (7)

| # | Baseline | Size | BLEU | chrF | COMET | Status |
|---|----------|------|------|------|-------|--------|
| 1 | RANDOM-FULL | 10K | — | — | — | TODO |
| 2 | LENGTH-STRATIFIED | 2K | — | — | — | TODO |
| 3 | VOCAB-MAXIMIZED | 2K | — | — | — | TODO |
| 4 | TF-IDF DIVERSE | 2K | — | — | — | TODO |
| 5 | ZERO-SHOT NLLB | 0 | — | — | — | TODO |
| 6 | NLLB French->Fon proxy | 0 | — | — | — | TODO |
| 7 | Commercial LLM (GPT-4/Claude) | 0 | — | — | — | TODO |

See [BASELINES.md](BASELINES.md) for construction recipes.

## Ablations

See [ABLATIONS.md](ABLATIONS.md) for full conditions.

---

## Shared Configuration

```yaml
model: facebook/nllb-200-distilled-600M
optimizer: AdamW
learning_rate: 2e-5
lr_schedule: linear warmup 10%, linear decay
batch_size: 16
max_epochs: 50
early_stopping: patience=10 on validation chrF
label_smoothing: 0.1
beam_search: beam=5, length_penalty=1.0
max_sequence_length: 128
seeds: [42, 123, 456, 789, 1024]
```

## Evaluation Protocol

**Automatic metrics:** BLEU (SacreBLEU), chrF/chrF++, TER, COMET (wmt22-comet-da), BERTScore
- Always report SacreBLEU signature
- chrF as primary automatic metric (better for morphologically rich languages)

**Human evaluation:**
- 3 native Adja-French bilingual evaluators, blind to system identity
- Adequacy (1-5), Fluency (1-5), pairwise preference
- Inter-annotator agreement: Krippendorff's alpha (target >0.6)

**Statistical significance:**
- Paired bootstrap resampling (1,000 samples)
- Wilcoxon signed-rank across 5 seeds
- Bonferroni correction for multiple comparisons
- Report Cohen's d effect sizes and 95% CIs

**Error analysis:** 100 outputs per system, 2 annotators, Cohen's kappa, chi-squared

---

## Timeline

| Phase | Weeks | Tasks | Training Runs |
|-------|-------|-------|---------------|
| 0: Setup | 1-2 | Data audit, 80-10-10 splits, NLLB pipeline, baseline scripts | 0 |
| 1: Core | 3-5 | Experiments 1-2 + all baselines | ~135 |
| 2: Ablations | 6-7 | Module, pronoun, verb, minimal-pair ablations + curriculum | ~85 |
| 3: Transfer | 8-10 | Cross-lingual, back-translation, architecture comparison, cross-verb | ~56 |
| 4: Eval | 11-13 | Human evaluation (~45 annotator-hours), error analysis, stats | 0 |
| 5: Paper | 14-17 | Draft, internal review, submit | 0 |

**Total compute:** ~300 runs x 1-4 hrs = ~400-800 GPU-hours

## Target Venues
- AfricaNLP Workshop 2025/2026
- LoResMT Workshop
- EMNLP/ACL Findings

## Verification Checklist

- [ ] `prepare_splits.py` runs without leakage errors (group-aware 80-10-10)
- [ ] Test set is shared across all conditions (splits/shared/test.tsv)
- [ ] No `base_sentence_id` group appears in both train and test
- [ ] Vocabulary overlap statistics computed between test and each training condition
- [ ] One seed of Experiment 1 runs end-to-end with BLEU/chrF matching SacreBLEU reference
- [ ] Each baseline has expected properties (vocab coverage, length distribution)
- [ ] Each ablation removes exactly the intended component
- [ ] All library versions pinned, random seeds saved, hyperparameters logged
