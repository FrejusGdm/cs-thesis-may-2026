# Baseline Construction Recipes

All baselines use the same model (NLLB-200-distilled-600M), hyperparameters, and test set as the main experiments. Only the training data differs.

---

## Baseline 1: RANDOM-FULL
- **Data:** All ~10K Tatoeba random sentences (French-Adja parallel)
- **Construction:** Use as-is, no filtering
- **Controls for:** "Just use all your data" — the most generous baseline

```python
# No sampling needed — use the full Tatoeba dataset
df_random_full = pd.read_csv("tatoeba_fr_adja_10k.csv")
```

## Baseline 2: LENGTH-STRATIFIED
- **Data:** 2,000 sentences from 10K Tatoeba, stratified by sentence length
- **Construction:**
  1. Compute word count for each sentence
  2. Bin into: short (3-5 words), medium (6-9 words), long (10+ words)
  3. Sample equally from each bin (~667 per bin)
- **Controls for:** Whether structured data's advantage comes from uniform sentence lengths

```python
import pandas as pd

df = pd.read_csv("tatoeba_fr_adja_10k.csv")
df["word_count"] = df["french"].str.split().str.len()

bins = pd.cut(df["word_count"], bins=[0, 5, 9, float("inf")], labels=["short", "medium", "long"])
df["length_bin"] = bins

per_bin = 2000 // 3  # ~667
df_stratified = df.groupby("length_bin", group_keys=False).apply(
    lambda x: x.sample(n=min(len(x), per_bin), random_state=42)
)
# Top up if any bin had fewer than 667
remaining = 2000 - len(df_stratified)
if remaining > 0:
    pool = df.drop(df_stratified.index)
    df_stratified = pd.concat([df_stratified, pool.sample(n=remaining, random_state=42)])

df_stratified = df_stratified.head(2000)
```

## Baseline 3: VOCAB-MAXIMIZED
- **Data:** 2,000 sentences from 10K Tatoeba, maximizing vocabulary coverage
- **Construction:**
  1. Initialize selected set as empty
  2. Greedy loop: select the sentence that adds the most new word types
  3. Repeat until 2,000 sentences selected
- **Controls for:** Whether structured data's advantage comes from better lexical coverage

```python
import pandas as pd

df = pd.read_csv("tatoeba_fr_adja_10k.csv")
df["tokens"] = df["french"].str.lower().str.split()

selected_indices = []
covered_vocab = set()

for _ in range(2000):
    best_idx = -1
    best_new = 0
    for idx in df.index:
        if idx in selected_indices:
            continue
        new_words = len(set(df.loc[idx, "tokens"]) - covered_vocab)
        if new_words > best_new:
            best_new = new_words
            best_idx = idx
    if best_idx == -1:
        break
    selected_indices.append(best_idx)
    covered_vocab.update(df.loc[best_idx, "tokens"])

df_vocab = df.loc[selected_indices]
```

**Note:** The greedy approach is O(n*k) where n=10K and k=2K. For efficiency, consider the lazy greedy variant or submodular optimization library (`apricot`).

## Baseline 4: TF-IDF DIVERSE
- **Data:** 2,000 sentences from 10K Tatoeba, maximizing semantic diversity
- **Construction:**
  1. Compute TF-IDF vectors for all 10K sentences
  2. Run k-means clustering (k=20 clusters)
  3. Sample 100 sentences from each cluster
- **Controls for:** Whether ANY intelligent selection strategy beats random

```python
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

df = pd.read_csv("tatoeba_fr_adja_10k.csv")

vectorizer = TfidfVectorizer(max_features=5000)
tfidf_matrix = vectorizer.fit_transform(df["french"])

kmeans = KMeans(n_clusters=20, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(tfidf_matrix)

per_cluster = 2000 // 20  # 100
df_diverse = df.groupby("cluster", group_keys=False).apply(
    lambda x: x.sample(n=min(len(x), per_cluster), random_state=42)
)
# Top up if any cluster had fewer than 100
remaining = 2000 - len(df_diverse)
if remaining > 0:
    pool = df.drop(df_diverse.index)
    df_diverse = pd.concat([df_diverse, pool.sample(n=remaining, random_state=42)])

df_diverse = df_diverse.head(2000)
```

## Baseline 5: ZERO-SHOT NLLB-200
- **Data:** None (zero-shot inference)
- **Construction:**
  1. Load `facebook/nllb-200-distilled-600M`
  2. Translate test set French -> Adja using Fon language code (`fon_Latn`) as closest proxy
  3. No fine-tuning
- **Controls for:** Absolute floor — how badly NLLB handles an unseen Gbe language

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Use Fon as proxy for Adja (closest available language code)
tokenizer.src_lang = "fra_Latn"

def translate(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=128, truncation=True)
    translated = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids("fon_Latn"),
        max_length=128,
        num_beams=5,
    )
    return tokenizer.decode(translated[0], skip_special_tokens=True)
```

## Baseline 6: NLLB-200 French -> Fon Proxy
- **Data:** None (zero-shot inference)
- **Construction:**
  1. Translate test set French -> Fon using NLLB-200
  2. Have Adja speakers rate comprehensibility of Fon output (1-5 scale)
  3. Human evaluation only (no automatic metrics — different target language)
- **Controls for:** Cross-Gbe intelligibility as a baseline

## Baseline 7: Commercial LLM Zero-Shot
- **Data:** None (zero-shot prompting)
- **Construction:**
  1. Prompt GPT-4 / Claude with the prompt below
  2. Translate full test set
  3. Evaluate with same metrics as other systems
- **Controls for:** "Throw money at it" baseline — can a frontier LLM do this without training?

**Prompt:**
```
Translate the following French sentence into Adja (Aja-Gbe), a Gbe language spoken
in southern Benin and Togo, closely related to Fon and Ewe. Adja uses Latin script
with tonal diacritics. Provide only the Adja translation, nothing else.

French: {sentence}
Adja:
```

---

## Baseline Results Tracking

| Baseline | BLEU | chrF | chrF++ | TER | COMET | BERTScore | Human Adequacy | Status |
|----------|------|------|--------|-----|-------|-----------|----------------|--------|
| RANDOM-FULL | — | — | — | — | — | — | — | TODO |
| LENGTH-STRATIFIED | — | — | — | — | — | — | — | TODO |
| VOCAB-MAXIMIZED | — | — | — | — | — | — | — | TODO |
| TF-IDF DIVERSE | — | — | — | — | — | — | — | TODO |
| ZERO-SHOT NLLB | — | — | — | — | — | — | — | TODO |
| NLLB Fr->Fon proxy | N/A | N/A | N/A | N/A | N/A | N/A | — | TODO |
| Commercial LLM | — | — | — | — | — | — | — | TODO |

## Data Properties to Report

For each baseline, compute and report:
- Vocabulary size (French side, Adja side)
- Mean sentence length (words)
- Type-token ratio
- Vocabulary overlap with test set (%)
- Vocabulary overlap with structured data (%)
