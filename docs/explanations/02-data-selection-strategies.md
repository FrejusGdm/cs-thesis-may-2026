# 2. Data Selection Strategies

This file explains the different ways we selected training data and why "which sentences you pick" matters more than "how many sentences you have."

---

## The Problem

Imagine you have a pool of 10,000 translated sentence pairs (French-Adja) from Tatoeba (a crowdsourced translation database). You want to train a translation model, but you can only afford to use 2,000 sentences (because of compute, time, or annotation budget).

**Which 2,000 do you pick?**

This question is the entire point of our baselines. We tested three smart selection strategies against plain random selection. The results are dramatic.

---

## Strategy 1: Random Selection (the naive approach)

**How it works:** Pick N sentences uniformly at random from the pool.

```
Pool of 10,000 sentences:
  "Bonjour, comment allez-vous ?"
  "Le chat dort sur le canape."
  "Je vais au marche demain matin."
  "Il pleut depuis trois jours."
  ... (9,996 more)

Random-2K: Just pick 2,000 at random.
Random-4K: Just pick 4,000 at random.
Random-10K: Use all 10,000.
```

**The problem:** You might get 50 sentences about weather, 3 about cooking, and 0 about school. The selection is unbalanced and wastes your budget on redundant sentences.

**Our results:**

| Size | BLEU |
|------|------|
| 200 random | 0.5 |
| 500 random | 1.7 |
| 1,000 random | 2.6 |
| 2,000 random | 3.0 |
| 4,000 random | 3.7 |
| 6,000 random | 4.0 |
| 8,000 random | 4.1 |
| 10,000 random | 4.1 |

Notice: after 4,000 sentences, adding more random data barely helps. The model plateaus at BLEU ~4. You could have 10,000 or 100,000 random sentences and it wouldn't matter much -- the data is redundant and doesn't cover the patterns the model needs.

---

## Strategy 2: Length-Stratified Selection

**How it works:** Bin all sentences by word count, then sample equally from each bin.

```
Bin 1 (short, 3-5 words):    "Bonjour.", "Merci beaucoup.", "Il pleut."
Bin 2 (medium, 6-9 words):   "Je vais au marche demain.", "Le chat dort sur le canape."
Bin 3 (long, 10+ words):     "Ma grand-mere prepare le meilleur riz au poisson du village."

Target: 2,000 sentences total
  -> Sample 666 from short bin
  -> Sample 666 from medium bin
  -> Sample 666 from long bin
  -> Top up remaining 2 from any bin
  -> Total: 2,000
```

**Why it works:** A model that only sees short sentences (3-5 words) struggles with long ones. By guaranteeing variety in sentence length, the model learns to handle both simple and complex structures.

**Result:** BLEU 23.7 (vs 4.1 for random-10K). Six times better, with 5x less data.

---

## Strategy 3: Vocab-Maximized (Greedy Set Cover)

This is the most interesting strategy. It selects sentences to maximize the number of **unique words** the model sees.

**How it works:** A greedy algorithm that, at each step, picks the sentence that adds the most new words to the training set.

Let's walk through a small example with 6 sentences and a target of 3:

```
Sentence pool:
  S1: "le chat mange"          -> tokens: {le, chat, mange}
  S2: "le chien dort"          -> tokens: {le, chien, dort}
  S3: "la fille chante bien"   -> tokens: {la, fille, chante, bien}
  S4: "le chat dort"           -> tokens: {le, chat, dort}
  S5: "un garcon mange bien"   -> tokens: {un, garcon, mange, bien}
  S6: "elle parle fort"        -> tokens: {elle, parle, fort}
```

**Iteration 1:** Start with empty vocabulary. Count new tokens each sentence would add:
```
  S1: 3 new tokens (le, chat, mange)
  S2: 3 new tokens (le, chien, dort)
  S3: 4 new tokens (la, fille, chante, bien)     <- BEST
  S4: 3 new tokens (le, chat, dort)
  S5: 4 new tokens (un, garcon, mange, bien)      <- tied
  S6: 3 new tokens (elle, parle, fort)
```
Select S3 (or S5, ties broken randomly). Say we pick S3.
Covered vocabulary: {la, fille, chante, bien}

**Iteration 2:** Count new tokens each remaining sentence would add:
```
  S1: 3 new (le, chat, mange) -- none overlap with {la, fille, chante, bien}
  S2: 3 new (le, chien, dort)
  S4: 3 new (le, chat, dort)
  S5: 3 new (un, garcon, mange) -- "bien" already covered!
  S6: 3 new (elle, parle, fort)
```
All tied at 3. Pick one randomly, say S6.
Covered vocabulary: {la, fille, chante, bien, elle, parle, fort}

**Iteration 3:** Count new tokens:
```
  S1: 3 new (le, chat, mange)
  S2: 3 new (le, chien, dort)
  S4: 3 new (le, chat, dort)
  S5: 2 new (un, garcon) -- "mange" and "bien" already covered!
```
Pick S1 (or S2/S4, all have 3 new).
Covered vocabulary: {la, fille, chante, bien, elle, parle, fort, le, chat, mange}

**Final selection:** S3, S6, S1 -- covering 10 unique words from just 3 sentences.

Compare to randomly picking S1, S2, S4 -- which would cover only {le, chat, mange, chien, dort} = 5 unique words, because `le` and `chat` appear in multiple sentences.

**Result:** BLEU 24.0 (the best of all baselines). By maximizing vocabulary coverage, we ensure the model sees the widest possible range of French words and patterns.

---

## Strategy 4: TF-IDF Diverse (Clustering)

This strategy is more sophisticated. Instead of just counting unique words, it tries to select sentences that are **semantically diverse** -- covering different topics and sentence patterns.

### What is TF-IDF?

TF-IDF stands for **Term Frequency - Inverse Document Frequency**. It's a way to measure how important a word is in a specific sentence relative to the entire corpus.

**Term Frequency (TF):** How often a word appears in this sentence.
```
Sentence: "le chat mange le poisson"
  TF("le") = 2/5 = 0.40    (appears twice in 5 words)
  TF("chat") = 1/5 = 0.20
  TF("mange") = 1/5 = 0.20
  TF("poisson") = 1/5 = 0.20
```

**Inverse Document Frequency (IDF):** How rare a word is across all sentences. Common words like "le" get low IDF; rare words like "poisson" get high IDF.
```
Corpus: 10,000 sentences
  "le" appears in 8,000 sentences  -> IDF = log(10000/8000) = 0.10 (very common, low weight)
  "chat" appears in 50 sentences   -> IDF = log(10000/50) = 2.30 (rare, high weight)
  "poisson" appears in 30 sentences -> IDF = log(10000/30) = 2.52 (rarer, even higher weight)
```

**TF-IDF = TF x IDF:**
```
  TF-IDF("le") = 0.40 x 0.10 = 0.04       (common word, low score)
  TF-IDF("chat") = 0.20 x 2.30 = 0.46     (distinctive word, high score)
  TF-IDF("poisson") = 0.20 x 2.52 = 0.50  (most distinctive, highest score)
```

So TF-IDF tells you: "poisson" is the most distinctive word in this sentence -- it's what makes this sentence different from most others.

### How k-means clustering works

Once every sentence has a TF-IDF vector (a list of scores for every word), we can group similar sentences together using k-means clustering:

1. **Initialize:** Place 20 random "centroids" (cluster centers) in the TF-IDF space
2. **Assign:** Each sentence goes to its nearest centroid
3. **Update:** Move each centroid to the average of its assigned sentences
4. **Repeat** steps 2-3 until clusters stabilize

The result: 20 clusters of semantically similar sentences:
```
Cluster 1 (food/cooking):    "je mange du riz", "elle prepare le repas", ...
Cluster 2 (school):          "il va a l'ecole", "le professeur enseigne", ...
Cluster 3 (family):          "ma mere est gentille", "mon frere joue", ...
Cluster 4 (weather):         "il pleut aujourd'hui", "le soleil brille", ...
...
Cluster 20 (greetings):      "bonjour monsieur", "comment allez-vous", ...
```

### The selection step

Sample 100 sentences from each of the 20 clusters:
```
100 from food cluster + 100 from school cluster + ... + 100 from greetings cluster = 2,000 total
```

This guarantees topical diversity. Even if 4,000 of the 10,000 sentences are about food, we'll only take 100 food sentences.

**Result:** BLEU 19.1. Lower than vocab-maximized, but still far better than random.

---

## Comparison Table: All Strategies at 2K Sentences

| Strategy | Size | BLEU | chrF | How It Selects |
|----------|------|------|------|----------------|
| Random | 2K | 3.0 | 25.7 | Uniformly at random |
| Length-stratified | 2K | 23.7 | 39.1 | Equal from short/medium/long bins |
| TF-IDF diverse | 2K | 19.1 | 36.6 | Equal from 20 semantic clusters |
| Vocab-maximized | 2K | 24.0 | 40.4 | Greedy: pick sentence adding most new words |
| Structured (ours) | 2K | 19.9 | 29.2 | Grammar-based curriculum (modules 1-5) |
| Random | 10K | 4.1 | 30.8 | All available random data |

Key observations:
- **2K vocab-maximized beats 10K random by 6x** (BLEU 24.0 vs 4.1)
- **Smart selection at 2K beats brute-force at 10K** for every strategy
- Structured data (ours) has lower chrF than baselines because the test set includes random-domain sentences that structured-only models haven't seen (more on this in [06-experimental-design.md](06-experimental-design.md))

---

## Why Does Random Data Fail So Badly?

Think about what 10,000 random Tatoeba sentences look like. Many are:
- **Chaotic:** No systematic coverage of grammar features the model needs
- Really when I was starting the project a year and half ago, I just used it because it gave me a lot of french data to translate (without thinking much about anything really)

The model sees thousands of sentences but never gets a clear signal about how French grammar maps to Adja grammar. It's like trying to learn algebra from 10,000 random math problems without anyone explaining the rules.

Structured data solves this by being **intentional**: every sentence exists for a reason, and the minimal-pair structure helps the model isolate exactly what each grammatical feature does.

---

## The Big Takeaway

The central finding of this research: **how you select your data matters more than how much data you have.** This is true whether you use grammar-based structure (our approach) or smart algorithmic selection (vocab-maximized). Both crush random selection by 5-6x.
