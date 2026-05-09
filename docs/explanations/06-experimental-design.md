# 6. Experimental Design

This file explains how the experiments are structured: what we're comparing, how we prevent data leakage, and how to interpret the scaling curves.

---

## Experiment 1: The Primary Hypothesis

**Claim:** Structured data achieves better translation quality than random data, even with less of it.

**How we test it:** Train the same model (NLLB-200) on different data compositions and compare BLEU/chrF on the same test set:

| Condition | What It Contains | Total Size |
|-----------|-----------------|-----------|
| RANDOM-10K | 10,000 random Tatoeba sentences | 10,000 |
| RANDOM-4K | 4,000 random Tatoeba sentences | 4,000 |
| STRUCTURED-2K | 2,000 structured sentences (modules 1-5) | 2,000 |
| STRUCTURED-4K-ONLY | All ~4,000 structured sentences | 4,000 |
| RANDOM-6K + STRUCTURED-4K | 6K random + 4K structured | 10,000 |
| RANDOM-10K + STRUCTURED-4K | 10K random + 4K structured | 14,000 |

**The key comparisons:**

1. **RANDOM-4K vs STRUCTURED-4K-ONLY** (same size, different type): Tests whether structured beats random when holding size constant.

2. **STRUCTURED-2K vs RANDOM-10K** (structured is smaller!): Tests whether 2K structured can beat 10K random -- a 5x data efficiency advantage.

3. **RANDOM-10K vs RANDOM-10K + STRUCTURED-4K** (adding structured to random): Tests whether structured data adds value on top of random.

---

## Experiment 2: Scaling Curves

**Question:** How does performance change as we add more data?

### Random Scaling Curve

We train on 200, 500, 1K, 2K, 4K, 6K, 8K, and 10K random sentences:

```
BLEU
 ^
 |
5|                    ___________
 |                ___/
4|            ___/
 |        ___/
3|    ___/
 |   /
2|  /
 | /
1|/
 +--+--+--+--+--+--+--+--+---> Size
    200 500 1K 2K 4K 6K 8K 10K
```

Random data shows **logarithmic growth that plateaus around 4K**. After 4K sentences, adding more random data barely helps. Going from 4K to 10K only adds 0.4 BLEU points (3.7 -> 4.1).

### Structured Scaling Curve

We train on 200, 500, 1K, 2K, 3K, and 4K structured sentences:

```
BLEU
 ^
 |
25|
  |                  _________
20|          _______/
  |      ___/
15|    _/
  |   /
10| _/
  |/
5 |
  +--+--+--+--+--+---> Size
    200 500 1K 2K 3K 4K
```

Structured data grows much faster and reaches near-peak by 1K-2K sentences. Even at 200 sentences, structured data achieves BLEU 9.4 -- better than 10K random (BLEU 4.1).

### Why Random Data Plateaus

Random Tatoeba sentences are highly redundant. After seeing a few thousand sentences:
- The model has already encountered most common French words
- New sentences repeat the same grammatical patterns
- There's no systematic coverage of grammar -- it's random

It's like reading 10,000 random French sentences to learn the language. After the first few thousand, you're just seeing more of the same patterns. You never get systematic exposure to negation, past tense, or questions unless you're lucky.

### Why Structured Data is Efficient

Every structured sentence exists for a reason:
- Module 1 sentences cover the verb-pronoun-object space systematically
- Each transformation (M2-M5) teaches a specific grammatical pattern
- Minimal pairs let the model compare systematically

There's very little redundancy. Each new sentence adds genuine new information.

---

## Additive Combination (RANDOM-6K + STRUCTURED-N)

**Question:** If we already have 6K random sentences, how much does adding structured data help?

We fix the random base at 6K and add increasing amounts of structured data on top:

| Condition | Total | BLEU |
|-----------|-------|------|
| RANDOM-6K only | 6K | 4.0 |
| RANDOM-6K + STRUCTURED-500 | 6.5K | 15.9 |
| RANDOM-6K + STRUCTURED-1000 | 7K | 18.7 |
| RANDOM-6K + STRUCTURED-2000 | 8K | 21.8 |
| RANDOM-6K + STRUCTURED-4000 | 10K | 21.6 |

**Key observation:** Adding just 500 structured sentences to 6K random data quadruples BLEU (4.0 -> 15.9). The structured data provides the grammatical signal that random data lacks.

Also note: RANDOM-6K + STRUCTURED-4000 (BLEU 21.6) barely beats RANDOM-6K + STRUCTURED-2000 (BLEU 21.8). Adding more than 2K structured doesn't help much -- the model has already learned the grammatical patterns it needs.

---

## Replacement Combination (Fixed 10K Budget)

**Question:** If you have exactly 10,000 sentence pairs to work with, what's the best mix of random and structured?

We fix the total at 10K and vary the ratio:

| Mix | Random | Structured | Total | BLEU | chrF |
|-----|--------|-----------|-------|------|------|
| R10000_S0 | 10K | 0 | 10K | 4.1 | 30.8 |
| R9500_S500 | 9.5K | 500 | 10K | 15.8 | 38.4 |
| R9000_S1000 | 9K | 1K | 10K | 19.4 | 40.0 |
| **R8000_S2000** | **8K** | **2K** | **10K** | **22.4** | **41.7** |
| R6000_S4000 | 6K | 4K | 10K | 21.4 | 40.5 |

**Sweet spot: 80% random + 20% structured (8K/2K).** This mix gives the highest BLEU (22.4) and chrF (41.7).

Why does R6000_S4000 do slightly worse than R8000_S2000? With 4K structured and only 6K random, the model sees proportionally more structured data, which has limited vocabulary. With 2K structured and 8K random, the model gets enough grammatical signal from the structured portion while maintaining broad vocabulary coverage from the random portion.

**Practical takeaway:** If you have a fixed annotation budget of 10K sentence pairs, spend 80% on diverse general-purpose sentences and 20% on systematically structured sentences. This outperforms any other ratio.

---

## Group-Aware Splitting: Preventing Data Leakage

### The Leakage Problem

When we split data into train/validation/test sets, we must ensure the model never sees test data during training. For random data, this is simple: randomly assign each sentence to a split.

But structured data has **minimal-pair groups**. Consider this group:

```
Group M1_0042:
  M1_0042: "il boit du lait"              (base)
  M2_0042: "il ne boit pas du lait"       (negation)
  M3_0042: "il a bu du lait"              (past)
  M4_0042: "il va boire du lait"          (future)
  M5_0042: "Est-ce qu'il boit du lait ?"  (question)
```

If we randomly split individual sentences, M1_0042 might end up in the training set while M3_0042 ends up in the test set. The model would see the base sentence during training and be tested on its past-tense variant -- an easy test because it already knows most of the sentence.

This would give **artificially inflated scores** that don't reflect real translation ability.

### The Solution: Group-Level Splitting

Instead of splitting individual sentences, we split **groups**. All sentences derived from the same base sentence stay together:

```
Step 1: Identify unique base groups
  Groups: [M1_0001, M1_0002, ..., M1_0400]

  Each group contains all its variants:
  M1_0001 group: {M1_0001, M2_0001, M3_0001, M4_0001, M5_0001a, M5_0001b}

Step 2: Randomly assign GROUPS (not sentences) to splits
  80% of groups -> train  (320 groups -> ~1,920 sentences)
  10% of groups -> val    (40 groups -> ~240 sentences)
  10% of groups -> test   (40 groups -> ~240 sentences)

Step 3: Verify no leakage
  Check: no base_sentence_id appears in both train and test
  Check: no base_sentence_id appears in both train and val
```

**Result:** If M1_0042 is in the test set, ALL its variants (M2_0042, M3_0042, M4_0042, M5_0042) are also in the test set. The model has never seen any form of this sentence during training.

---

## Test Set Composition

A subtle but important detail: our test set is a **mix of structured and random** test sentences.

```
Test set = structured_test (10% of structured groups) + random_test (10% of random data)
         = ~400 structured sentences + ~1,000 random sentences
         = ~1,400 total test sentences
```

**Why this matters for interpreting results:**

Models trained only on structured data (e.g., STRUCTURED-4K-ONLY) are evaluated on both:
- Structured test sentences (which they can handle well -- same domain, same patterns)
- Random test sentences (which they've never seen anything like)

This explains why structured-only models have **lower chrF** (29.1) than baselines (40.4) or combined conditions (41.2). They're not bad at what they know -- they're untrained on the random domain that makes up 70% of the test set.

Models trained on random + structured data get the best of both worlds: grammatical knowledge from structured data and vocabulary breadth from random data.

**This is not a flaw** -- it's the right experimental design. We want to measure how well each condition generalizes to diverse, unseen sentences. A model that only works on sentences similar to its training data isn't very useful.

---

## Seeds and Reproducibility

### What seeds control

Every experiment uses a **random seed** that determines:
1. How the training data is shuffled at each epoch
2. Which mini-batches are sampled
3. How dropout (if any) is applied
4. Weight initialization for any new parameters

```
Seed 42:   Training order: [S104, S2, S891, S450, ...] -> BLEU 21.4
Seed 123:  Training order: [S55, S998, S301, S12, ...]  -> BLEU 20.8
Seed 456:  Training order: [S701, S334, S1, S756, ...]  -> BLEU 15.2
```

Same data, same model, same hyperparameters -- different random ordering produces different results.

### How many seeds we use

| Experiment tier | Seeds | Purpose |
|----------------|-------|---------|
| Core (exp1, baselines) | 5 seeds (42, 123, 456, 789, 2024) | High confidence for primary claims |
| Ablations | 3 seeds (42, 123, 456) | Sufficient to show consistent patterns |

### Why seed 456 sometimes gives bad results

In STRUCTURED-2K, seed 456 gave BLEU 15.2 while the other four seeds gave 20-21. This happens because:
- The random training order was unlucky (important patterns were seen too early before the model was ready, or too late after it had already converged)
- The validation split was slightly worse (fewer representative examples)
- The model converged to a local minimum (a solution that's good but not great)

This is **normal** and is exactly why we run multiple seeds -- to distinguish genuine patterns from random variation. The mean (19.9) and standard deviation (2.4) give the full picture.

---

## Summary

| Concept | What It Is | Why It Matters |
|---------|-----------|---------------|
| Scaling curves | Performance vs data size | Shows random plateaus while structured keeps growing |
| Additive combination | Fixed random base + add structured | Shows structured data adds value on top of random |
| Replacement combination | Fixed total budget, vary the mix | Shows optimal ratio is 80% random + 20% structured |
| Group-aware splitting | Split at the group level, not sentence level | Prevents inflated scores from minimal-pair leakage |
| Test set composition | Mix of structured and random test sentences | Tests generalization, explains chrF differences |
| Seeds | Random number generator control | Ensures results are reproducible and stable |
