# 01 — What is a tokenizer (and why it killed CSM on Adja)

Last updated: **2026-04-21**
Prereq: basic familiarity with transformers.

## The one-sentence version

A tokenizer is a dictionary that turns characters into integer IDs, and when
that dictionary doesn't know a character, it fragments it into useless bytes.

## A slightly longer version

Language models operate on sequences of integers, not characters. The
**tokenizer** is the lookup table that maps text → int IDs. Modern LLMs use
**subword tokenizers** that break rare words into smaller pieces:

```
"hello world" → ["hello", " world"]                    # whole-word tokens
"unprecedented" → ["un", "prec", "edent", "ed"]        # subword fragments
```

The tokenizer is trained on a big text corpus once, then frozen. Its vocab is
fixed — typically 32k to 256k subword pieces.

### What happens when the tokenizer sees a character it was never trained on?

It falls back to **bytes**. Every character encodes to 1-4 UTF-8 bytes, and the
tokenizer has a reserved set of 256 "byte tokens" to represent the raw bytes.
Example with Llama 3.2 BPE (used by CSM and Orpheus):

```python
>>> tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
>>> tok.tokenize("hello")
['hello']                                   # one clean token
>>> tok.tokenize("ɛnyi wɛ dze è")
['Ġ', '<0xC9>', '<0x9B>', 'nyi', 'Ġ', ...]   # ɛ → two byte tokens
```

Now the LM has to learn that `<0xC9><0x9B>` taken together means "the Adja
phoneme /ɛ/". It has to do this from scratch, with 1.7 hours of training data.
It never gets there.

## Why this blew up our CSM experiments

Sesame CSM uses the Llama 3.2 tokenizer. Adja's written alphabet contains
characters that never appear in Llama's English-heavy training data:

- `ɛ` (open e), `ɔ` (open o), `ŋ` (eng), `ɖ` (d with tail)
- Tone marks: `è`, `é`, `ɔ̀`, `ɛ́`, combining accents U+0300, U+0301

Every one of these becomes a 2-3 byte sequence. A typical Adja sentence that
would be 12 text tokens in a well-fitted tokenizer becomes 60+ byte fragments
in Llama. We had **6 hours of labeled audio** for Adja and were asking the
model to learn byte→phoneme→audio. Impossible.

### The control that proved this

Spark TTS uses **Qwen2 BPE**, trained on a far more multilingual corpus
including African scripts. Qwen2 tokenizes our Adja test sentence as:

```python
['ɛnyi', 'wɛ', 'dze', 'è']           # clean, native tokens
```

Spark fine-tuned on the same 1.6k Adja utterances and produced intelligible
speech. Same codec family as CSM (waveform quantizer), same low data regime,
different tokenizer. The tokenizer was the main difference.

## How to fix it (two strategies)

1. **Expand the existing tokenizer**: call `tokenizer.add_tokens([ɛ, ɔ, ŋ, ...])`
   and `model.resize_token_embeddings()`. New IDs get randomly initialized
   embedding vectors. This is what `scripts/hf_jobs/T1_csm_tokfix.py` does.
2. **Swap the whole tokenizer** (e.g. use the NLLB SentencePiece or Qwen2 BPE).
   Much more surgery — you'd have to re-align the LM's embedding layer.

Option 1 is our current test. If it works, we recover CSM for Adja. If it
doesn't, either the LoRA rank is too low to learn the new tokens' embeddings,
or some OTHER layer (the codec, the audio head) is also broken.

## What to check in your own pipelines

Before fine-tuning any LM-based model on a new language, run:

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("<your-model>")
print(tok.tokenize("<sample-sentence-in-target-language>"))
```

If you see `<0x??>` or short `ĥ`-style fragments, your tokenizer is not ready.
See `scripts/diagnostics/tokenizer_check.py` for a full comparison across
Llama / Qwen2 / NLLB tokenizers on Adja text.

## Papers to read

- Sennrich et al., "Neural Machine Translation of Rare Words with Subword
  Units" — https://arxiv.org/abs/1508.07909 — the original BPE paper, only ~8
  pages.
- Kudo, "SentencePiece" — https://arxiv.org/abs/1808.06226 — the other
  widely-used family (used by NLLB). Compare the tokenizer philosophy.

## Related files in this project

- `../experiments-explained/T1-csm-why-it-failed.md` — the story told chronologically.
- `../../learnings-from-the-past/training-gotchas.md` — the correction to the
  original "it was the codec" hypothesis.
- `../../scripts/hf_jobs/T1_csm_tokfix.py` — where we actually test the fix.
