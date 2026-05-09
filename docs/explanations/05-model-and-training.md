# 5. Model and Training

This file explains the machine learning side: what model we use, how fine-tuning works, and what happens during training.

---

## NLLB-200: The Base Model

**NLLB** stands for **No Language Left Behind**. It's a translation model created by Meta (Facebook) in 2022, designed to translate between 200+ languages.

We use the **distilled 600M parameter** version: `facebook/nllb-200-distilled-600M`.

- **600M parameters** = 600 million learnable numbers that define the model's knowledge. For reference, GPT-4 has trillions of parameters. 600M is small enough to run on a single GPU.
- **Distilled** = a smaller version of the full model (3.3B parameters), trained to mimic the larger model's behavior. Faster and cheaper but nearly as good.
- **200 languages** = NLLB supports 200 language directions out of the box, including many African languages (Fon, Ewe, Yoruba, Igbo, etc.). But **not Adja** -- that's why we need to fine-tune it.

### Encoder-Decoder Architecture

NLLB is a **sequence-to-sequence** (seq2seq) model with two parts:

```
French input: "je mange du riz"
       |
       v
  +-----------+
  |  ENCODER  |  <- reads the French sentence
  |           |     produces a rich internal representation
  +-----------+     (captures meaning, grammar, context)
       |
       v
  [internal representation: a matrix of numbers]
       |
       v
  +-----------+
  |  DECODER  |  <- generates the Adja translation
  |           |     one token at a time, left to right
  +-----------+     uses the encoder's representation as context
       |
       v
Adja output: "un du nu"
```

The encoder processes the entire French sentence at once and encodes its meaning. The decoder then generates the Adja translation word by word, referring back to the encoder's output at each step to decide what to produce next.

---

## Why Fine-Tune Instead of Training from Scratch?

**Training from scratch** means starting with random weights and learning everything from our ~4,000 sentence pairs. This would fail catastrophically because:
- 4,000 sentences is far too few to learn a language from zero
- The model would need millions of examples to learn basic grammar, vocabulary, word order, etc.

**Fine-tuning** means starting with NLLB's pretrained weights and adjusting them slightly for our specific task. NLLB already knows:
- How languages work in general (grammar, morphology, syntax)
- How translation works (aligning meaning across languages)
- French specifically (one of its 200 supported languages)
- Gbe language patterns (it supports Fon and Ewe, which are related to Adja)

We're not teaching the model to translate from zero. We're saying: "You already know French and you know related languages. Now adjust what you know to handle Adja specifically." This is why 4,000 sentences can be enough.

---

## The aj_Latn Token: Adding a New Language

NLLB uses language tags to tell the model which language to expect and produce. When translating French to Fon, you'd set:

```
Source language: fra_Latn   (French, Latin script)
Target language: fon_Latn   (Fon, Latin script)
```

The model sees `fon_Latn` at the start of decoding and knows: "I should produce Fon output."

**Problem:** Adja (`aj_Latn`) isn't in NLLB's 200 languages. The model has never seen this tag.

**Solution:** We add `aj_Latn` as a new special token in three steps:

### Step 1: Register the token

```python
tokenizer.add_special_tokens({"additional_special_tokens": [..., "aj_Latn"]})
```

This tells the tokenizer: "`aj_Latn` is a valid token with a unique ID." Before this, typing `aj_Latn` would be treated as unknown characters.

### Step 2: Resize the model's embedding tables

```python
model.resize_token_embeddings(len(tokenizer))
```

The model has an **embedding matrix** -- a table where each row corresponds to a token and contains a vector of numbers (the token's "meaning" in the model's internal language). Adding a new token means adding a new row, initialized with random numbers.

### Step 3: Initialize from Ewe

Random initialization is bad -- the model would start with a meaningless representation for `aj_Latn`. Instead, we copy Ewe's embedding:

```python
emb_in[aj_Latn_id].copy_(emb_in[ewe_Latn_id])
```

**Why Ewe?** Adja is a Gbe language, closely related to Ewe. They share:
- Similar grammatical structures (SVO word order, tense marking)
- Related vocabulary (many cognates)
- Similar phonological patterns

By starting with Ewe's learned representation, the model begins with a useful approximation of Adja's linguistic properties. Fine-tuning then adjusts this to capture Adja's specific differences from Ewe.

Think of it like: if you've never studied Portuguese but you know Spanish, you'd do much better starting from your Spanish knowledge than starting from scratch. Ewe is the "Spanish" to Adja's "Portuguese."

---

## Text Preprocessing

Before feeding text to the model, we normalize it to match NLLB's pretraining format. This is critical -- if we feed text that looks different from what NLLB was trained on, the model won't recognize it.

The `preproc()` function applies three transformations:

### 1. Punctuation Normalization (MosesPunctNormalizer)

```
Input:  "J\u2019ai mange \u2014 c\u2019est bon\u2026"    (fancy quotes, em dash, ellipsis)
Output: "J'ai mange -- c'est bon..."           (ASCII equivalents)
```

Converts typographic characters to their plain ASCII equivalents. This matters because text from different sources uses different Unicode characters for the same punctuation.

### 2. Non-Printing Character Removal

```
Input:  "bonjour\u200b monde"     (invisible zero-width space between words)
Output: "bonjour monde"           (clean text)
```

Removes invisible Unicode characters (zero-width spaces, soft hyphens, control characters) that can confuse the tokenizer.

### 3. Unicode NFKC Normalization

```
Input:  "cafe\u0301"              (e + combining accent = two Unicode characters)
Output: "cafe"                    (single composed character)
```

NFKC (Normalization Form KC) ensures that characters with multiple possible Unicode representations are standardized to one canonical form. This prevents the model from treating the same visible character as two different tokens.

---

## The Adafactor Optimizer

An **optimizer** is the algorithm that updates the model's weights during training. It takes the gradient (the direction the loss is decreasing) and decides how to adjust each weight.

### Why Adafactor instead of Adam?

**Adam** (the most common optimizer) stores two extra values per parameter:
- First moment (running average of gradients)
- Second moment (running average of squared gradients)

For a 600M parameter model, that's 1.2 billion extra values, which requires ~4.8 GB of GPU memory just for the optimizer.

**Adafactor** approximates Adam's second moment using a factorized representation that requires much less memory. For a matrix of weights (m x n), Adam stores m x n values for the second moment, while Adafactor stores only m + n values.

This is crucial when fine-tuning on a T4 GPU (16 GB VRAM) -- every gigabyte counts.

### Our Adafactor Configuration

```python
optimizer = Adafactor(
    model.parameters(),
    scale_parameter=False,    # Don't auto-scale the learning rate
    relative_step=False,      # Use our fixed learning rate
    lr=1e-4,                  # Learning rate: 0.0001
    clip_threshold=1.0,       # Clip gradients to prevent explosions
    weight_decay=1e-3,        # L2 regularization to prevent overfitting
)
```

**Learning rate (lr=1e-4):** Controls how big each weight update is. Too high = model overshoots and diverges. Too low = model learns too slowly. 1e-4 (0.0001) is a common sweet spot for fine-tuning.

**Gradient clipping (clip_threshold=1.0):** If the gradient at any step is larger than 1.0, it gets scaled down. This prevents "gradient explosions" -- rare batches where the loss surface is very steep and the update would be enormous and destructive.

**Weight decay (weight_decay=1e-3):** Gently pushes all weights toward zero. This prevents overfitting -- the model can't memorize the training data by making weights arbitrarily large. Especially important with only ~4,000 training sentences.

---

## Warmup

The first 500 training steps use **learning rate warmup**: the learning rate starts at 0 and linearly increases to the target (1e-4).

```
Step 1:    lr = 0.0000002  (nearly zero)
Step 100:  lr = 0.00002    (20% of target)
Step 250:  lr = 0.00005    (50% of target)
Step 500:  lr = 0.0001     (full target)
Step 501+: lr = 0.0001     (constant from here on)
```

**Why warmup?**

At the start of training, the model's gradients are computed based on weights that aren't yet adapted to the new task. These gradients can be noisy and misleading. If we immediately apply full-sized updates, we might corrupt the pretrained knowledge.

Warmup lets the model "get its bearings" with tiny updates first, then gradually increases the learning rate once the gradients become more reliable.

Think of it like driving a car: you don't floor the accelerator from a stop. You gradually press the pedal to build up speed smoothly.

---

## Early Stopping

**The problem:** If you train for too long, the model starts **overfitting** -- it memorizes the training data instead of learning general patterns. Performance on the training set keeps improving, but performance on unseen data gets worse.

**The solution:** Monitor performance on a **validation set** (a small held-out portion of the data, typically 10%) and stop training when it stops improving.

### How our early stopping works

```
Every 200 training steps:
  1. Pause training
  2. Translate all validation sentences
  3. Compute chrF score on validation translations
  4. If chrF improved: save the model weights (keep the best version)
  5. If chrF didn't improve: increment a counter
  6. If counter reaches 10: stop training entirely

After training: restore the best model weights (not the final ones)
```

**Example timeline:**
```
Step 200:   val chrF = 15.2   -> New best! Save model.  Counter = 0
Step 400:   val chrF = 22.1   -> New best! Save model.  Counter = 0
Step 600:   val chrF = 25.3   -> New best! Save model.  Counter = 0
Step 800:   val chrF = 27.8   -> New best! Save model.  Counter = 0
Step 1000:  val chrF = 28.5   -> New best! Save model.  Counter = 0
Step 1200:  val chrF = 28.2   -> No improvement.        Counter = 1
Step 1400:  val chrF = 28.6   -> New best! Save model.  Counter = 0
Step 1600:  val chrF = 28.4   -> No improvement.        Counter = 1
Step 1800:  val chrF = 28.3   -> No improvement.        Counter = 2
Step 2000:  val chrF = 28.1   -> No improvement.        Counter = 3
...
Step 3400:  val chrF = 27.5   -> No improvement.        Counter = 10
                                 STOP! Restore model from step 1400 (chrF 28.6)
```

**Why chrF for early stopping?** chrF is more stable than BLEU on small validation sets. BLEU can jump around a lot because it requires exact word matches, while chrF gives partial credit for partial matches.

**Patience = 10:** We wait 10 evaluation cycles (2,000 training steps) without improvement before stopping. This prevents premature stopping -- sometimes the model plateaus briefly before finding another improvement.

---

## The Training Loop: Putting It All Together

Here's what happens during a complete training run:

```
1. Load model (NLLB-200 600M)
2. Add aj_Latn token, initialize from Ewe
3. Preprocess all training data (normalize text)
4. Split into batches of 8 sentence pairs

For each step (up to max_steps):
  a. Sample a random batch of 8 pairs
  b. Tokenize French (encoder input)
  c. Tokenize Adja (decoder target)
  d. Forward pass: encoder processes French, decoder predicts Adja
  e. Compute loss (how wrong were the predictions?)
  f. Backward pass: compute gradients (which direction to adjust weights)
  g. Optimizer step: adjust weights using Adafactor
  h. Scheduler step: adjust learning rate (warmup then constant)

  Every 200 steps:
  i. Evaluate on validation set
  j. Check early stopping condition
  k. Print progress: step, loss, BLEU, chrF

5. Restore best model weights
6. Evaluate on test set (final metrics)
7. Upload results to HuggingFace Hub
```

A typical run trains for 3,000-5,000 steps and takes about 45-90 minutes on a T4 GPU.

---

## Summary

| Concept | What It Is | Why It Matters |
|---------|-----------|---------------|
| NLLB-200 | Meta's 200-language translation model | Provides pretrained knowledge of languages |
| Fine-tuning | Adjusting pretrained weights for a new task | Lets us use 4K sentences instead of millions |
| aj_Latn token | Custom language tag for Adja | Tells the model to generate Adja output |
| Ewe initialization | Copy Ewe's embedding to Adja | Related language gives better starting point |
| Preprocessing | Normalize text to match NLLB's format | Model expects standardized input |
| Adafactor | Memory-efficient optimizer | Fits in GPU memory on T4 (16GB) |
| Warmup | Gradually increase learning rate | Prevents corrupting pretrained weights |
| Early stopping | Stop when validation performance plateaus | Prevents overfitting on small datasets |
