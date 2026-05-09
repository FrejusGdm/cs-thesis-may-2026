# 3. Evaluation Metrics

This file explains how we measure translation quality -- what BLEU, chrF, and chrF++ actually calculate, and how to interpret the numbers.

---

## Why Do We Need Metrics?

After training a model, we need to know: **how good are its translations?** We can't manually read thousands of translations, so we use automated metrics that compare the model's output (the **prediction**) against a human-written translation (the **reference**).

No single metric is perfect, so we use multiple metrics that capture different aspects of quality.

---

## BLEU (Bilingual Evaluation Understudy)

BLEU is the most widely used translation metric. It measures **how many word sequences in the prediction match the reference**.

### How BLEU works

BLEU counts matching **n-grams** (sequences of N consecutive words) between the prediction and reference:

```
Reference:  "the cat sits on the mat"
Prediction: "the cat is on the mat"
```

**1-grams (individual words):**
```
Reference words:  {the, cat, sits, on, the, mat}
Prediction words: {the, cat, is, on, the, mat}

Matches: the, cat, on, the, mat = 5 out of 6 prediction words match
1-gram precision = 5/6 = 83%
```

**2-grams (pairs of consecutive words):**
```
Reference 2-grams:  {the cat, cat sits, sits on, on the, the mat}
Prediction 2-grams: {the cat, cat is, is on, on the, the mat}

Matches: "the cat", "on the", "the mat" = 3 out of 5
2-gram precision = 3/5 = 60%
```

**3-grams:**
```
Reference 3-grams:  {the cat sits, cat sits on, sits on the, on the mat}
Prediction 3-grams: {the cat is, cat is on, is on the, on the mat}

Matches: "on the mat" = 1 out of 4
3-gram precision = 1/4 = 25%
```

**4-grams:**
```
Reference 4-grams:  {the cat sits on, cat sits on the, sits on the mat}
Prediction 4-grams: {the cat is on, cat is on the, is on the mat}

Matches: 0 out of 3
4-gram precision = 0/3 = 0%
```

**Final BLEU score** is the geometric mean of 1-gram through 4-gram precision, scaled 0-100:
```
BLEU = exp(average of log-precisions) x brevity_penalty x 100
```

In this example: the prediction replaced "sits" with "is" -- just one word wrong -- but it killed the 3-gram and 4-gram matches, dragging BLEU down significantly.

### The Brevity Penalty

BLEU also penalizes translations that are too short. If the model outputs "cat mat" for a 6-word reference, the 2-word prediction would get high precision (both words match) but it's clearly a bad translation. The brevity penalty corrects for this.

### What BLEU Scores Mean in Practice

| BLEU | Quality |
|------|---------|
| 0-5 | Garbage. Almost nothing matches. |
| 5-15 | Getting the gist but very rough. |
| 15-25 | Decent for low-resource. Captures basic structure. |
| 25-35 | Good. Readable translations with some errors. |
| 35-50 | Very good. Near-professional quality. |
| 50+ | Excellent. Hard to distinguish from human. |

**Our results:** BLEU ~20 from structured data is solidly in the "decent for low-resource" range. For a language with zero prior NMT work, this is a strong starting point.

---

## chrF (Character F-score)

chrF measures overlap at the **character level** instead of the word level. It counts matching character n-grams (sequences of characters).

### How chrF works

```
Reference:  "il mange du riz"
Prediction: "il manger du riz"
```

**Word-level (BLEU perspective):** "manger" != "mange", so this word is wrong. BLEU counts 3/4 word matches.

**Character-level (chrF perspective):** "manger" and "mange" share the characters m-a-n-g-e. Only the trailing "r" is extra. chrF gives high credit because most characters match.

chrF looks at character n-grams of length 1 through 6:
```
"mange":  {m, a, n, g, e, ma, an, ng, ge, man, ang, nge, mang, ange, mange}
"manger": {m, a, n, g, e, r, ma, an, ng, ge, er, man, ang, nge, ger, mang, ange, nger, mange, anger, manger}

Overlap is high because they share most character sequences.
```

chrF then combines precision and recall using the F-score formula.

### Why chrF Matters

For morphologically rich languages (like Adja, French, or other West African languages), chrF is often more informative than BLEU because:

1. **Partial credit for close words:** "mange" vs "manger" gets credit from chrF but zero from BLEU
2. **More stable on small test sets:** BLEU can swing wildly with small amounts of data
3. **Better for agglutinative languages:** Languages where a single word contains multiple morphemes (prefixes, suffixes) benefit from character-level matching

---

## chrF++ (chrF Plus Plus)

chrF++ is chrF with an added word-level component. It computes character n-gram F-score AND word n-gram F-score (up to bigrams), then combines them.

```
chrF  = character n-grams only
chrF++ = character n-grams + word unigrams + word bigrams
```

chrF++ typically scores 1-3 points lower than chrF because the word-level component is stricter.

---

## Understanding the BLEU vs chrF Gap

One of the most revealing patterns in our results is the gap between BLEU and chrF:

| Condition | BLEU | chrF | Gap |
|-----------|------|------|-----|
| RANDOM-10K | 4.1 | 30.8 | 26.7 |
| STRUCTURED-4K | 19.5 | 29.1 | 9.6 |
| VOCAB-MAXIMIZED (2K) | 24.0 | 40.4 | 16.4 |

**Why does RANDOM-10K have chrF 30.8 but BLEU only 4.1?**

The model trained on random data is getting **characters** roughly right but **words** wrong. It's producing output that looks like Adja (correct character patterns, right kinds of syllables) but the actual word choices and order are wrong.

Think of it like someone who can pronounce French words but strings them together nonsensically: the sounds are right but the meaning is garbled.

**Why does STRUCTURED-4K have a smaller gap (9.6)?**

The structured model gets both characters AND words right -- it's producing coherent translations with correct word choices. BLEU and chrF agree more closely because the output is genuinely good at multiple levels.

---

## Seeds and What "BLEU 19.9 +/- 2.4" Means

### What is a seed?

A **seed** is a number that controls randomness. Machine learning involves many random steps:
- Which order the training data is shuffled in
- How model weights are initialized
- Which samples are in the validation set

If you set `seed=42`, every random choice is made the same way every time. Run the same experiment twice with seed=42 and you get identical results.

**Different seeds give different results** because the random choices differ:

```
seed=42:  BLEU = 21.4   (lucky shuffle: model sees a good training order)
seed=123: BLEU = 20.8   (different shuffle: slightly different order)
seed=456: BLEU = 15.2   (unlucky shuffle: something was suboptimal)
seed=789: BLEU = 20.6   (back to normal range)
seed=2024: BLEU = 21.6  (another good run)
```

### Why run multiple seeds?

If we only ran seed=42 and got BLEU 21.4, someone could say "maybe you just got lucky." By running 5 seeds, we prove the result is **stable and reproducible**, not a fluke.

### Mean +/- Standard Deviation

We report the average (mean) and spread (standard deviation) across seeds:

```
Seeds: [21.4, 20.8, 15.2, 20.6, 21.6]
Mean: (21.4 + 20.8 + 15.2 + 20.6 + 21.6) / 5 = 19.9
Std:  measure of how spread out the values are = 2.4

Reported as: BLEU 19.9 +/- 2.4
```

**What the +/- means:**
- Small std (e.g., +/- 0.2): Results are very consistent across seeds. You can trust the mean.
- Large std (e.g., +/- 2.4): Results vary more. The mean is still the best estimate, but any single run could be higher or lower.

In our STRUCTURED-2K results (BLEU 19.9 +/- 2.4), the higher std is mostly due to one unlucky seed (456 = 15.2). The other four are all 20-21. This happens sometimes -- one training run converges to a weaker solution.

---

## Summary

| Metric | What It Measures | Level | Score Range | Our Target |
|--------|-----------------|-------|-------------|------------|
| BLEU | Word n-gram overlap | Word | 0-100 | ~20 |
| chrF | Character n-gram overlap | Character | 0-100 | ~30 |
| chrF++ | Character + word overlap | Both | 0-100 | ~28 |
| mean +/- std | Average and consistency across seeds | -- | -- | Low std is good |
