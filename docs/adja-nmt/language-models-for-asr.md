# Language Models for ASR

A progressive tutorial for someone who already understands NMT (NLLB, mBART, encoder-decoder
transformers) but has never worked with n-gram language models or LM-augmented CTC decoding.
Written for Josue, building on concepts from the [ASR Learning Guide](asr-learning-guide.md).

**Prerequisites**: You should have read at least Sections 1 and 3 of the ASR Learning Guide
(CTC basics). This guide explains what comes *after* CTC gives you its raw output.

**How to use this guide**: Sections 1-3 are the core -- read them in order. Section 4 and 5
are independent techniques you can read in any order. Section 6 is a hands-on walkthrough
you should follow when you are ready to train your first Adja LM.

---

## 1. What Is a Language Model (for ASR)?

### You already know one

Here is the thing: you have been using language models for over a year. In your NLLB and
mBART experiments, the decoder *is* a language model. When it generates an Adja translation
token by token, it is implicitly answering the question "given everything generated so far,
what is the most likely next token in valid Adja text?" That is exactly what a language
model does -- it assigns probabilities to sequences of text.

The decoder in NMT gets to cheat a little: it also has cross-attention to the source
sentence. But strip away the cross-attention, and what remains is a pure language model --
something that knows what valid target-language text looks like.

### CTC has no language model

Now recall how CTC works. The encoder looks at audio frames and outputs a probability
distribution over characters at *each frame independently*. There is no decoder. There
is no autoregressive generation. CTC just picks the most likely character at each time
step, collapses repeated characters, and removes blanks.

This means CTC has no concept of what valid Adja text looks like. It does not know that
`ŋɖuɖu` is a real Adja word but `ŋŋŋɖɖɖuuu` is not. It does not know that `n` followed
by `y` is common in Adja but `ɖ` followed by `ɖ` almost never happens. It just picks
whatever the acoustic model thinks is most likely at each frame, with no linguistic context.

This is like running your NMT model with the decoder replaced by a simple linear layer
that picks one token per source position. You would get garbage translations, because there
is nothing enforcing fluency in the target language.

### The fix: add an external language model

An external language model fixes this by providing exactly the missing piece: knowledge of
what valid Adja text looks like. During decoding, you combine two scores:

1. **Acoustic score** (from CTC): "The audio at this frame sounds like ɖ"
2. **Language model score** (from the LM): "After seeing ŋ, the character ɖ is very likely"

The combined system can override CTC's frame-level mistakes. If CTC thinks frame 47 is
probably `ɖ` with 40% confidence and `d` with 35% confidence, but the LM strongly predicts
`ɖ` should follow the previous context, the combined score tips in favor of `ɖ`.

**Analogy**: Think of the LM as spell-check for ASR output. CTC gives you a rough draft
with lots of character-level noise. The LM cleans it up by asking "does this look like
real Adja text?"

---

## 2. n-gram Language Models (The Simplest LM)

You might expect the language model to be a transformer or an LSTM. For ASR decoding,
it almost never is. The standard tool is something much simpler: an **n-gram model**.

### What is an n-gram?

An n-gram is just a sequence of n consecutive items (characters or words). For a
character-level model over Adja text:

- **Unigram (n=1)**: single characters: `ŋ`, `ɖ`, `u`, `e`
- **Bigram (n=2)**: pairs: `ŋɖ`, `ɖu`, `uɖ`, `ɖu`
- **Trigram (n=3)**: triples: `ŋɖu`, `ɖuɖ`, `uɖu`
- **5-gram (n=5)**: sequences of 5 characters

### How it works

An n-gram language model answers one question: **given the previous (n-1) characters,
what is the probability of the next character?**

For a bigram model: P(ɖ | ŋ) = "how often does ɖ appear after ŋ in Adja text?"

For a 5-gram model: P(u | ŋɖuɖ) = "given the previous 4 characters are ŋɖuɖ, how likely
is u next?"

### How to train it

Training an n-gram model is almost comically simple compared to what you are used to with
transformers. There is no backpropagation. No GPU. No learning rate. You literally just:

1. Take all your Adja text
2. Count every sequence of n characters
3. Normalize the counts to get probabilities
4. Apply smoothing (so unseen sequences get a small probability instead of zero)

That is it. The "model" is a lookup table of probabilities. If `ŋɖ` appeared 47 times
in your training text and `ŋ` appeared 200 times total as a bigram prefix, then
P(ɖ | ŋ) = 47/200 = 0.235 (before smoothing).

### Why character-level for Adja?

In English ASR, you typically use a **word-level** n-gram model. But Adja has a problem:
with only ~1,600 utterances of transcribed text, the word vocabulary is small and sparse.
Many valid words will appear only once or twice, making word-level statistics unreliable.

Character-level n-grams sidestep this entirely. Your character vocabulary is around 50-100
characters (Adja alphabet + tone marks + punctuation). Even with limited text, you get
dense statistics for character sequences. A 5-gram character model captures enough local
context to distinguish real character patterns from CTC noise.

This is analogous to why you use character-level CTC instead of BPE for low-resource ASR:
the smaller vocabulary gives you better coverage with less data.

### The tool: KenLM

[KenLM](https://kheafield.com/code/kenlm/) is the standard tool for building n-gram
language models. It is:

- **Fast**: trains in seconds on your data (literally seconds, not hours)
- **Memory-efficient**: uses trie data structures for compact storage
- **Battle-tested**: used in almost every CTC-based ASR system, Moses MT, and more
- **Not a neural network**: it is pure counting and lookup. No GPU needed.

You may recognize the name from MT -- KenLM was originally built for phrase-based
statistical MT, which used n-gram LMs for fluency scoring. Same idea here, different
application.

KenLM outputs an `.arpa` file -- a plain-text file of n-gram probabilities. This file
is what you feed to the decoder at inference time.

**Paper**: Kenneth Heafield, "KenLM: Faster and Smaller Language Model Queries"
(WMT 2011) https://aclanthology.org/W11-2123/

---

## 3. How LM + CTC Decoding Works

Now the key question: how do you actually *combine* the CTC acoustic model with the
n-gram language model at decoding time?

### Greedy decoding (no LM)

This is what you get by default. At each audio frame, CTC picks the most probable
character, then collapses repeats and removes blanks. No language model is involved.

```
Frame outputs:  ŋ ŋ _ ɖ ɖ ɖ _ u u _ ɖ _ u u u
After collapse: ŋ       ɖ       u     ɖ   u
Result:         ŋɖuɖu
```

This works okay when the acoustic model is confident. But when it is not -- and with only
~1,600 training utterances, it often is not -- you get garbled output because each frame
is decided independently.

### Beam search with LM (the real decoder)

Beam search keeps the top-K hypotheses alive at each step and scores them with a combined
objective:

```
combined_score = acoustic_score + alpha * lm_score + beta * num_words
```

Where:

- **acoustic_score**: log probability from CTC (how well the audio matches this hypothesis)
- **lm_score**: log probability from the n-gram LM (how likely this character sequence is
  in Adja)
- **alpha**: LM weight -- how much to trust the language model (tuned on your dev set)
- **beta**: word insertion bonus -- a small reward for producing more words

#### Why alpha matters

If alpha is too low, the LM barely helps -- you are back to near-greedy decoding. If alpha
is too high, the LM dominates and the system ignores the audio, hallucinating fluent Adja
text that has nothing to do with what was said.

Typical values: alpha is usually between 0.5 and 2.0. You tune it on the dev set by trying
a grid of values and picking the one that gives the best WER/CER.

#### Why beta matters

Without the word insertion bonus, the LM has a subtle bias toward shorter outputs. Why?
Because every additional character is another opportunity to incur a penalty (no character
sequence has probability 1.0). The beta bonus counteracts this by giving a small reward
each time a word boundary (space) is produced.

Typical values: beta is usually between 0.0 and 3.0. Tuned alongside alpha on the dev set.

### Concrete example with Adja

Imagine the audio is someone saying "ŋɖuɖu na" (an Adja phrase). CTC outputs frame-level
probabilities and beam search explores hypotheses:

```
Hypothesis A: "ŋɖuɖu na"
  acoustic_score = -12.3   (CTC says: plausible)
  lm_score       = -8.1    (LM says: this is valid Adja)
  combined       = -12.3 + 1.5*(-8.1) + 0.5*2 = -23.45

Hypothesis B: "ŋdudu na"
  acoustic_score = -11.8   (CTC says: slightly more plausible acoustically)
  lm_score       = -14.2   (LM says: "ŋdudu" is not an Adja pattern)
  combined       = -11.8 + 1.5*(-14.2) + 0.5*2 = -32.1

Winner: Hypothesis A (less negative = better)
```

The acoustic model slightly preferred "ŋdudu" (maybe `d` and `ɖ` sound similar), but the
LM knows that `ŋɖuɖu` is an Adja word and `ŋdudu` is not. The LM tips the balance toward
the correct transcription. This is exactly the kind of correction that matters most for
Adja's special characters.

### The tool: pyctcdecode

[pyctcdecode](https://github.com/kensho-technologies/pyctcdecode) is a Python library
that implements beam search + LM decoding for CTC models. It is a drop-in replacement
for greedy decoding:

```python
# Before (greedy, no LM):
predicted_ids = torch.argmax(logits, dim=-1)
transcription = processor.decode(predicted_ids[0])

# After (beam search + LM):
from pyctcdecode import build_ctcdecoder

decoder = build_ctcdecoder(
    labels=vocab_list,            # your character vocabulary
    kenlm_model_path="adja.arpa", # the n-gram LM file
    alpha=1.5,                    # LM weight (tune on dev)
    beta=0.5,                     # word insertion bonus (tune on dev)
)
logits_np = logits.cpu().numpy()[0]
transcription = decoder.decode(logits_np)
```

That is it. Same CTC model, same weights, just a smarter decoding strategy. The LM adds
no training cost -- only a small inference cost (beam search is slightly slower than greedy,
but still fast).

**GitHub**: https://github.com/kensho-technologies/pyctcdecode

---

## 4. What Is Focal CTC Loss?

This section is about training, not decoding. It addresses a different problem: the model
learning frequent characters well but ignoring rare ones.

### The problem

Standard CTC loss computes the negative log probability of the correct transcript. Every
character contributes equally to the gradient. But in Adja text:

- `e` and `a` appear hundreds of times
- `ɖ` and `ŋ` appear rarely

The model quickly learns to predict `e` and `a` correctly (easy, lots of examples). It
gets a small gradient signal from rare characters like `ɖ` because they appear infrequently.
Over time, the model becomes very confident on common characters and barely improves on
rare ones.

This is why your CTC outputs are mostly common vowels with almost no Adja-specific
characters.

### The fix: focal loss

Focal loss was originally proposed for object detection (where most candidate boxes are
"easy negatives" and the detector ignores rare objects). The idea transfers directly to
CTC and rare characters.

Standard CTC loss for a character prediction:

```
loss = -log(p)
```

where p is the model's predicted probability for the correct character.

Focal CTC loss:

```
loss = -(1 - p)^gamma * log(p)
```

The `(1 - p)^gamma` factor is the key. Let us see what it does:

**For a character the model already gets right** (say, `e` with p = 0.9):
```
(1 - 0.9)^2 = 0.01
```
The loss is multiplied by 0.01. Gradient is tiny. The model does not waste effort
improving on something it already knows.

**For a character the model struggles with** (say, `ɖ` with p = 0.1):
```
(1 - 0.1)^2 = 0.81
```
The loss is multiplied by 0.81. Gradient is large. The model focuses its learning on
the character it is bad at.

### The gamma parameter

Gamma controls how aggressively the loss re-weights:

| gamma | Effect |
|-------|--------|
| 0     | Standard CTC loss (no re-weighting) |
| 1     | Mild re-weighting |
| 2     | Typical choice -- strong focus on hard examples |
| 5     | Very aggressive -- almost ignores easy examples |

Start with gamma = 2 and tune from there. Higher gamma is especially useful for Adja
because the character frequency imbalance is extreme.

### Why this matters for Adja specifically

Adja's writing system has exactly the kind of long-tail distribution that focal loss was
designed for. The characters that *matter most* for meaning -- `ɛ` vs `e`, `ɔ` vs `o`,
`ɖ` vs `d`, and the tone marks -- are the rarest. Without focal loss, the model learns
"Adja sounds like it could be French" (all common Latin characters). With focal loss, it
learns "Adja has specific characters that distinguish it from French."

### Papers

- Original focal loss: Tsung-Yi Lin et al., "Focal Loss for Dense Object Detection"
  (ICCV 2017) https://arxiv.org/abs/1708.02002
- Applied to CTC for ASR: Feng et al. (2019)
  https://www.hindawi.com/journals/complexity/2019/9345861/

---

## 5. What Is Hybrid Vocabulary?

This section addresses a vocabulary design choice that interacts with both the CTC output
head and the language model.

### Current approach: NFC characters

Right now, each NFC-normalized Adja character is one token in your CTC vocabulary:

```
Vocabulary: [a, b, d, e, ɛ, ɖ, ŋ, ɔ, é, è, ɛ́, ɛ̀, ɔ́, ɔ̀, ...]
```

The problem: `ɔ` appears maybe 200 times in your training data. But `ɔ̀` (ɔ with low
tone) appears maybe 15 times. And `ɔ́` (ɔ with high tone) appears maybe 20 times. The
model treats these as three completely unrelated tokens. It cannot share what it learns
about the *sound* of `ɔ` across its toned variants.

### The analogy to BPE

You know this problem from NMT. In early word-level MT, "running" and "runner" were
unrelated tokens. BPE fixed this by splitting into `run` + `ning` and `run` + `ner` --
now the model shares the `run` component. Same idea here, but for characters and tone.

### Hybrid vocabulary

Split each toned character into its base character plus a separate tone token:

```
Before (NFC):  [ɔ̀, ɔ́, ɔ, ɛ̀, ɛ́, ɛ, ...]
After (hybrid): [ɔ, ɛ, ŋ, ɖ, ..., TONE_LOW, TONE_HIGH, TONE_NASAL, ...]
```

Now when the model hears the sound of `ɔ`, it can learn from ALL occurrences of `ɔ` --
whether toned or untoned. The tone mark becomes a separate prediction: "this vowel also
has a low tone."

```
NFC:    ɔ̀      -> 1 CTC output token (rare, hard to learn)
Hybrid: ɔ + ̀   -> 2 CTC output tokens (ɔ is common, tone mark is a modifier)
```

### Trade-offs

| Aspect | NFC | Hybrid |
|--------|-----|--------|
| Vocab size | Larger (every toned variant is separate) | Smaller (bases + tone marks) |
| Data efficiency | Worse (rare combined tokens) | Better (shared base characters) |
| CTC alignment | Simpler (1 token per character) | Harder (2 tokens for toned characters) |
| LM modeling | Tone patterns captured in character n-grams | Tone mark patterns modeled separately |

The trade-off on CTC alignment is real: CTC now needs to align two output tokens to one
sound, which is slightly harder. But with our extremely limited data (~1,600 utterances),
the data efficiency gain usually wins.

### Implementation

The implementation is straightforward -- just change the character vocabulary and the
text preprocessing:

1. Define the new vocabulary: base characters + tone mark tokens
2. Write a function that decomposes NFC characters into base + tone
3. Update the CTC output head size
4. Retrain (or fine-tune from the NFC checkpoint)
5. At inference time, recombine base + tone back to NFC for evaluation

No architectural changes. No new model components. Just a different tokenization of the
output space.

---

## 6. Practical: How to Train an Adja LM (Step by Step)

This section gives you the exact commands. No GPU needed -- this runs on your laptop
in seconds.

### Step 1: Prepare the text data

Extract just the Adja text from your training data. One sentence per line, already
NFC-normalized:

```bash
# From your train.tsv (assumes text is in the "sentence" column)
cut -f2 data/train.tsv > data/adja_text.txt

# Or from Python if your data is in a different format:
python -c "
import csv, unicodedata
with open('data/train.tsv') as f, open('data/adja_text.txt', 'w') as out:
    reader = csv.DictReader(f, delimiter='\t')
    for row in reader:
        text = unicodedata.normalize('NFC', row['sentence'])
        out.write(text + '\n')
"
```

For a character-level LM, add spaces between characters so KenLM treats each character
as a "word":

```bash
python -c "
import unicodedata
with open('data/adja_text.txt') as f, open('data/adja_chars.txt', 'w') as out:
    for line in f:
        line = unicodedata.normalize('NFC', line.strip())
        # Space-separate each character (KenLM treats whitespace-separated tokens as words)
        chars = ' '.join(list(line))
        out.write(chars + '\n')
"
```

### Step 2: Install KenLM

```bash
# On your Mac (for local testing):
pip install https://github.com/kpu/kenlm/archive/master.zip

# On HPC (in your container):
# KenLM is often already available in speech processing containers.
# If not, add to your requirements or build from source.
```

### Step 3: Train the n-gram model

```bash
# Train a 5-gram character-level LM
# -o 5 = 5-gram
# --discount_fallback = use simple smoothing (safer for small data)
lmplz -o 5 --discount_fallback < data/adja_chars.txt > models/adja_5gram.arpa
```

That is it. On ~1,600 sentences this takes under a second. The output file `adja_5gram.arpa`
is a plain-text file listing all n-gram probabilities.

### Step 4: (Optional) Convert to binary for faster loading

```bash
build_binary models/adja_5gram.arpa models/adja_5gram.bin
```

The binary format loads faster at inference time. Not strictly necessary for small models.

### Step 5: Use the LM during CTC decoding

```python
from pyctcdecode import build_ctcdecoder

# Your CTC vocabulary (must match the model's output tokens exactly)
vocab_list = ["<pad>", "<s>", "</s>", "<unk>", "|",
              "a", "b", "d", "e", "ɛ", "ɖ", "ŋ", "ɔ", ...]

decoder = build_ctcdecoder(
    labels=vocab_list,
    kenlm_model_path="models/adja_5gram.arpa",
    alpha=1.5,   # LM weight -- tune on dev set
    beta=0.5,    # word insertion bonus -- tune on dev set
)

# During inference:
import torch
# logits shape: (batch, time, vocab_size) -- raw CTC output before softmax
logits = model(input_values).logits

# Decode one utterance at a time
logits_np = logits.cpu().detach().numpy()
for i in range(logits_np.shape[0]):
    text = decoder.decode(logits_np[i])
    print(text)
```

### Step 6: Tune alpha and beta on dev set

```python
import numpy as np
from itertools import product

alphas = [0.5, 1.0, 1.5, 2.0, 2.5]
betas = [0.0, 0.5, 1.0, 1.5, 2.0]

best_cer = float("inf")
best_params = None

for alpha, beta in product(alphas, betas):
    decoder = build_ctcdecoder(
        labels=vocab_list,
        kenlm_model_path="models/adja_5gram.arpa",
        alpha=alpha,
        beta=beta,
    )
    predictions = [decoder.decode(logits_np[i]) for i in range(len(dev_logits))]
    cer = compute_cer(predictions, dev_references)  # your CER function
    if cer < best_cer:
        best_cer = cer
        best_params = (alpha, beta)
        print(f"alpha={alpha}, beta={beta}, CER={cer:.4f} (new best)")
```

This grid search is fast because there is no model retraining -- you are just re-decoding
the same logits with different LM weights.

---

## 7. Recommended Reading

**Core concepts**:
- KenLM paper: Kenneth Heafield, "KenLM: Faster and Smaller Language Model Queries"
  https://aclanthology.org/W11-2123/
- pyctcdecode: https://github.com/kensho-technologies/pyctcdecode
  (the README is an excellent practical guide)
- Distill.pub CTC article: https://distill.pub/2017/ctc/
  (read the "Decoding" section specifically for LM integration)

**Focal loss**:
- Original: Lin et al. (2017), "Focal Loss for Dense Object Detection"
  https://arxiv.org/abs/1708.02002
- Applied to CTC: Feng et al. (2019)
  https://www.hindawi.com/journals/complexity/2019/9345861/

**LMs for ASR (deeper)**:
- Whisper-LM: Jiang et al. (2025), combining external LMs with Whisper
  https://arxiv.org/abs/2503.23542
- Shallow fusion survey: Gulcehre et al. (2015), "On Using Monolingual Corpora in
  Neural Machine Translation" https://arxiv.org/abs/1503.06204
  (the "shallow fusion" technique described here is the same idea as LM + CTC beam search)

**For Adja specifically**:
- EUSIPCO 2024 tokenization study for low-resource ASR:
  https://eurasip.org/Proceedings/Eusipco/Eusipco2024/pdfs/0000141.pdf
  (character-level CTC preferred over BPE in low-resource)
