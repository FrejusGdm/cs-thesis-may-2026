# CER and WER: A Deep Dive for Adja ASR

> **Audience**: Josue, coming from NMT background (BLEU/chrF) into ASR.
> **Goal**: Understand *why* we report the metrics we do, what's a "good" number for low-resource Adja, and how to standardize spacing so the numbers are fair.
> **TL;DR**: Lead with CER, report WER as secondary, normalize whitespace + punctuation before either, never touch tone marks or ɛ/ɔ/ŋ/ɖ.

---

## Table of Contents

1. [The Fundamentals: What Do WER and CER Actually Measure?](#1-the-fundamentals)
2. [Why the Choice of Metric Matters for Adja Specifically](#2-why-the-choice-of-metric-matters-for-adja)
3. [Published Benchmarks: What's a "Good" CER?](#3-published-benchmarks)
4. [Rules of Thumb: Targets by Data Size](#4-rules-of-thumb)
5. [Standardizing Word Boundaries (The Spacing Fix)](#5-standardizing-word-boundaries)
6. [How to Report Metrics in a Paper](#6-how-to-report-metrics-in-a-paper)
7. [Connection to BLEU/chrF (What You Already Know)](#7-connection-to-bleu-chrf)
8. [Practical Recommendations for Your Paper](#8-practical-recommendations)
9. [Recommended Reading Order](#9-recommended-reading-order)

---

## 1. The Fundamentals

### 1.1 Edit distance — the common root

You already know Levenshtein edit distance from NMT work. Both WER and CER are just edit distance dressed up with a denominator. The difference is what counts as a "token":

- **WER** operates over **words** (anything separated by whitespace).
- **CER** operates over **characters** (Unicode codepoints, usually after NFC normalization).

The formula is identical in both cases:

```
         S + I + D
metric = ---------
             N
```

Where:
- `S` = number of substitutions (wrong token where a correct one was expected)
- `I` = number of insertions (extra token in hypothesis)
- `D` = number of deletions (missing token in hypothesis)
- `N` = number of tokens in the **reference** (not hypothesis)

The division by `N` (reference length) is crucial — it's why both metrics can exceed 100%. If the hypothesis is much longer than the reference, insertions pile up without a ceiling.

### 1.2 Worked example: "Enu maku enyi" vs "Enu ma ku enyi"

Let's walk this through for real. Say the reference is what a human transcriber wrote, and the hypothesis is what our Whisper-Ewe→Adja model produced:

```
Reference:   Enu maku enyi           (3 words, 13 characters including spaces)
Hypothesis:  Enu ma ku enyi          (4 words, 14 characters including spaces)
```

#### WER computation

Align word-by-word:

| Ref     | Hyp   | Operation       |
|---------|-------|-----------------|
| Enu     | Enu   | match           |
| maku    | ma    | substitution    |
| —       | ku    | insertion       |
| enyi    | enyi  | match           |

- `S = 1`, `I = 1`, `D = 0`, `N = 3`
- **WER = (1 + 1 + 0) / 3 = 0.667 = 66.7%**

Ouch. Two thirds of the words are "wrong" even though the content is basically identical — just spaced differently.

#### CER computation

Now align character-by-character (let `_` denote a space):

```
Ref:  E n u _ m a k u _ e n y i
Hyp:  E n u _ m a _ k u _ e n y i
```

Align them:

| Ref | Hyp | Op    |
|-----|-----|-------|
| E   | E   | match |
| n   | n   | match |
| u   | u   | match |
| _   | _   | match |
| m   | m   | match |
| a   | a   | match |
| —   | _   | ins   |
| k   | k   | match |
| u   | u   | match |
| _   | _   | match |
| e   | e   | match |
| n   | n   | match |
| y   | y   | match |
| i   | i   | match |

- `S = 0`, `I = 1`, `D = 0`, `N = 13`
- **CER = 1 / 13 = 0.077 = 7.7%**

**Same output, 66.7% WER vs 7.7% CER.** The model got the phonemes right; it just hallucinated a space. WER makes it look like a disaster; CER reveals the actual quality.

### 1.3 Why WER can exceed 100%

Because `N` is the reference length, not the max of ref/hyp. If the model outputs 5 words for a 2-word reference and all 5 are wrong, you get:

- `S = 2` (2 mismatched positions) + `I = 3` (extras) + `D = 0` = 5
- `WER = 5 / 2 = 250%`

This is very real in ASR: a model that inserts garbage tokens (common with beam search misbehaving, or when the AM is uncertain) will balloon past 100%. Seeing WER > 100% in your logs is not a bug; it's a signal of pathological decoding.

### 1.4 Connection to metrics you already know (BLEU, chrF)

You know this intuition from NMT:

- **BLEU** is a **precision-oriented** metric: "of the n-grams in my hypothesis, what fraction matched the reference?" It penalizes garbage tokens through the denominator, but it also rewards short outputs (hence the brevity penalty).
- **WER/CER** are **error-oriented** metrics: "how many edits do I need to fix my hypothesis?" They penalize garbage tokens through insertions, and they penalize short outputs through deletions. They're symmetric in spirit but not in formula.

Think of it this way:
- BLEU/chrF: higher is better, bounded in [0, 1] or [0, 100].
- WER/CER: lower is better, bounded in [0, ∞).

The **deep parallel** (which will matter in §7):
- BLEU is to WER as chrF is to CER.
- Word-level metrics are fragile to tokenization; character-level metrics are robust.

---

## 2. Why the Choice of Metric Matters for Adja

This is where it gets real. Three reasons CER is the better primary metric for Adja.

### 2.1 The word boundary problem

Adja orthography isn't fully standardized. Three writers can transcribe the same utterance with three different spacings:

```
Writer A:  Enu maku enyi
Writer B:  Enu ma ku enyi
Writer C:  Enumaku enyi
```

All three mean the same thing. None is objectively "correct" — standardization is an ongoing project for most Gbe languages. The variation you see in your training data is a real phenomenon, not a labeling error.

**What this does to WER**: punishes the model for producing "the wrong word" when the word is a legitimate alternative spacing. Look at your own decode logs — you'll almost certainly see samples where the transcript is essentially perfect but WER is 50%+ because of a single space disagreement.

**What this does to CER**: barely moves. A wrong space is 1 character out of 13-50; it's a rounding error. CER lets the model be right about *what was said* even when it's wrong about *where the word boundary falls*.

This is not a metric hack to make numbers look good. It's a metric that aligns with the research question: "did the model recognize the speech?" not "did the model match the annotator's spacing convention?"

### 2.2 The tonal language problem

Adja is tonal. The character inventory is larger than, say, French because tone marks carry meaning:

- `má` vs `mà` — different tones, different meaning
- `kɔ́` vs `kɔ̀` — ditto
- `ẽ` vs `e` — nasalization matters
- `ŋ`, `ɖ`, `ɛ`, `ɔ` — distinct phonemes, not "weird French"

Consider a model output: reference `má` (high tone), hypothesis `mà` (low tone).

- **WER**: this is one wrong word out of N. Binary. You lose the signal that the model got the consonant and vowel right and missed only the diacritic.
- **CER**: after NFC normalization, `má` is 2 codepoints (`m` + `á`) or 3 (`m` + `a` + combining acute), depending on normalization form. Either way, the wrong tone is 1 error out of 2-3 characters. You *see* in the metric that the tone was the failure mode, not the phoneme.

Tone recognition is a known ASR failure mode for low-resource tonal languages (see the Ewe paper in §3). CER gives you partial credit — and more importantly, diagnostic signal — that WER obliterates.

### 2.3 The low-resource problem

This one is the most important and the least obvious.

With ~2 hours of training data and ~1.6k utterances, a huge fraction of words in your test set **never appeared in training**. This is not a bug; it's the fundamental statistics of small corpora (Zipf's law: type count grows ~log-linearly with token count, so your 2h corpus has seen a small subset of the language's vocabulary).

What happens to those unseen words at test time?

- A from-scratch ASR model will almost certainly get them wrong character-by-character.
- A transfer model (Whisper-Ewe→Adja in your case) has seen similar-looking words in Ewe and can often reconstruct them. It might miss a character or two.

**WER**: any unseen word that's not reconstructed *perfectly* is 100% wrong. The metric is binary per word. You lose all signal about how close the model got.

**CER**: partial credit. If the true word is `sɛgbasú` and the model writes `segbasú`, that's 1/7 = 14% error on that word, not 100%. You see that the model is *close* and that the specific failure is ɛ→e (a known low-resource failure mode — unusual characters get backed off to more-common ones).

For low-resource ASR, the research question is usually "how close can we get?" rather than "did we get it exactly right?" CER answers the first question. WER only answers the second.

---

## 3. Published Benchmarks: What's a "Good" CER?

The only honest way to answer "is 24.9% CER good?" is to compare to published numbers for similar data regimes. Here's a curated table.

| Data regime          | Example language            | Hours  | Model                  | CER       | WER      | Source |
|----------------------|-----------------------------|--------|------------------------|-----------|----------|--------|
| High-resource        | English (LibriSpeech clean) | 960h   | wav2vec2-large         | 2-3%      | 5-6%     | [Baevski et al. 2020](https://arxiv.org/abs/2006.11477) |
| Medium               | Vietnamese (CommonVoice)    | ~300h  | XLS-R-300M             | 8-12%     | 20-25%   | [FLEURS 2022](https://arxiv.org/abs/2205.12446) |
| Low                  | Yoruba                      | 20h    | wav2vec2-Yoruba        | ~17%      | ~40%     | [ACM TALLIP 2024](https://dl.acm.org/doi/10.1145/3690384) |
| Very low             | Ewe                         | 100h   | XLS-R-300M             | ~15%      | ~31%     | [ACL ICNLSP 2025](https://aclanthology.org/2025.icnlsp-1.32.pdf) |
| Ultra-low (FLEURS)   | Multiple African langs      | 3-10h  | MMS-1B                 | 18-35%    | 45-75%   | [Pratap 2023](https://jmlr.org/papers/v25/23-1318.html) |
| Ultra-low (PazaBench)| 39 African langs avg        | ~2-5h  | Fine-tuned baselines   | 25-40%    | 60-85%   | [PazaBench 2025](https://arxiv.org/html/2512.10968v1) |
| **Adja (this work)** | **Adja**                    | **2h** | **Whisper-Ewe→Adja**   | **24.9%** | **73.1%**| **This repo** |

A few things to notice:

1. **Your 24.9% CER at 2h is squarely in the PazaBench range** (25-40% for similar data regimes), and actually on the better side of it.
2. **Your WER/CER ratio (73.1/24.9 ≈ 2.9) is high**, which is consistent with the word-boundary problem from §2.1. For better-standardized languages you'd expect a ratio closer to 2.0-2.5. That "extra" WER is almost certainly spacing.
3. **Per-hour efficiency is excellent** — most papers reporting single-digit CERs have 10-100× more data. The Ewe→Adja transfer is doing real work.

A quick heuristic for judging CER:
- If it beats the MMS-1B number for the same language, you're doing well.
- If it's within 2× of MMS-1B, you're in the game.
- If it's 5×+ worse, something's wrong (data quality, tokenizer, normalization).

### 3.1 Key papers (clickable)

- **Whisper** (reference architecture, robust multilingual): https://cdn.openai.com/papers/whisper.pdf
- **MMS** (Meta's 1000-language ASR, main comparison point): https://jmlr.org/papers/v25/23-1318.html
- **FLEURS** (low-resource multilingual benchmark, methodology): https://arxiv.org/abs/2205.12446
- **PazaBench** (39 African languages, the gold standard for your exact problem): https://arxiv.org/html/2512.10968v1 / https://arxiv.org/abs/2408.04773
- **Ewe ASR** (closest-relative benchmark, same Gbe family as Adja): https://aclanthology.org/2025.icnlsp-1.32.pdf
- **Yoruba ASR** (tonal language ASR, same challenges): https://dl.acm.org/doi/10.1145/3690384
- **wav2vec 2.0** (self-supervised pretraining, the foundation of XLS-R/MMS): https://arxiv.org/abs/2006.11477
- **CTC** (the training objective most of these systems use): https://www.cs.toronto.edu/~graves/icml_2006.pdf

---

## 4. Rules of Thumb: Targets by Data Size

Compiled from the above papers and general ASR folklore. These are **CER** ranges for "reasonable fine-tuned models" on unseen test data:

| Training hours | Expected CER range | Example        |
|----------------|--------------------|----------------|
| < 1h           | 40-70%             | FLEURS zero-shot / few-shot |
| **1-5h**       | **20-40%**         | **← you are here (24.9% at 2h)** |
| 5-20h          | 12-25%             | Yoruba with smaller fine-tune sets |
| 20-100h        | 8-15%              | Ewe-sized corpora |
| 100-1000h      | 4-10%              | CommonVoice mid-tier languages |
| > 1000h        | 2-5%               | LibriSpeech, Gigaspeech |

A useful interpretation: **you need roughly 10× the data to halve the CER** in low-resource settings. This is the core reason transfer learning is so valuable — you're effectively "buying" data from a related language (Ewe) that has more of it.

Your 24.9% at 2h corresponds roughly to what you'd expect from 10-20h of from-scratch training on a typical African language. That's the transfer bonus quantified.

---

## 5. Standardizing Word Boundaries (The Spacing Fix)

This is the direct answer to your question #3. There are three techniques, ordered from least to most invasive.

### 5.1 Text normalization before WER (the 80/20 fix)

Apply a normalization pipeline to **both** reference and hypothesis before computing WER. This is what Whisper does (see [`whisper/normalizers/basic.py`](https://github.com/openai/whisper/blob/main/whisper/normalizers/basic.py) and [`whisper/normalizers/english.py`](https://github.com/openai/whisper/blob/main/whisper/normalizers/english.py) for a full production example).

Here's an Adja-specific normalizer you can drop into your eval pipeline:

```python
# experiments/asr/shared/normalize.py
"""
Text normalization for Adja ASR evaluation.

Applied to BOTH reference and hypothesis before computing WER/CER.
The goal is to neutralize cosmetic differences (punctuation, case,
whitespace) while preserving semantic content (tones, special chars).

References:
- Whisper basic normalizer: https://github.com/openai/whisper/blob/main/whisper/normalizers/basic.py
- Whisper English normalizer (full pipeline): https://github.com/openai/whisper/blob/main/whisper/normalizers/english.py
"""
import re
import unicodedata


# Punctuation that's always safe to strip from Adja transcripts.
# Sentence-ending marks have no phonetic content.
SAFE_PUNCT_TO_STRIP = set(".,!?;:\"'`()[]{}—–-…")


def normalize_for_asr(text: str) -> str:
    """
    Canonicalize text for ASR evaluation.

    Operations (in order):
      1. NFC unicode normalization (canonical composition).
         Without this, 'á' as NFC (1 codepoint) and 'a' + combining
         acute as NFD (2 codepoints) would score as different characters.
      2. Lowercase (Adja is case-insensitive in practice).
      3. Strip sentence-ending punctuation. This is SAFE — punctuation
         has no acoustic realization.
      4. Collapse whitespace runs to a single space.
      5. Strip leading/trailing whitespace.

    Operations we DO NOT do (these would break Adja):
      - Strip tone marks (á, à, ẽ, etc.) — changes meaning.
      - Map ɛ→e, ɔ→o, ŋ→n, ɖ→d — these are distinct phonemes.
      - Remove apostrophes in the middle of words — sometimes load-bearing.
    """
    # 1. Unicode canonicalization
    text = unicodedata.normalize("NFC", text)

    # 2. Lowercase
    text = text.lower()

    # 3. Strip safe punctuation
    text = "".join(ch for ch in text if ch not in SAFE_PUNCT_TO_STRIP)

    # 4+5. Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


# Optional: a stricter normalizer that also drops all whitespace.
# This gives you "CER with normalized text" — useful for sanity checks.
def normalize_no_spaces(text: str) -> str:
    """Like normalize_for_asr but also removes all whitespace.
    Used to compute 'space-blind' CER as a reference lower bound."""
    return normalize_for_asr(text).replace(" ", "")
```

And your evaluation loop should look like:

```python
import jiwer
from normalize import normalize_for_asr

refs_raw = [...]  # from your test set
hyps_raw = [...]  # from your model

refs = [normalize_for_asr(r) for r in refs_raw]
hyps = [normalize_for_asr(h) for h in hyps_raw]

wer_raw  = jiwer.wer(refs_raw, hyps_raw)
wer_norm = jiwer.wer(refs, hyps)
cer_raw  = jiwer.cer(refs_raw, hyps_raw)
cer_norm = jiwer.cer(refs, hyps)

print(f"WER raw:        {wer_raw:.3f}")
print(f"WER normalized: {wer_norm:.3f}")
print(f"CER raw:        {cer_raw:.3f}")
print(f"CER normalized: {cer_norm:.3f}")
```

Expected effect on your numbers: normalized WER typically drops 3-10 absolute points from raw WER in low-resource settings. The gap is a direct measure of how much of your "error" is cosmetic. **Report both**; the delta is interpretable information.

#### What's safe and what's not — quick reference

| Operation                           | Adja | Why |
|-------------------------------------|------|-----|
| NFC unicode normalization           | Yes  | Canonical representation, zero-cost |
| Lowercase                           | Yes  | Adja doesn't mark meaning with case |
| Strip `. , ! ? ; :`                 | Yes  | No acoustic content |
| Strip parens/brackets `()[]{}`      | Yes  | Annotator artifacts |
| Collapse whitespace                 | Yes  | Different annotators, different conventions |
| Strip tone marks (`á`→`a`, `è`→`e`) | **No**  | Changes word meaning |
| Map ɛ→e, ɔ→o, ŋ→n, ɖ→d              | **No**  | Distinct phonemes |
| Strip apostrophes                   | Maybe | Depends — audit your corpus first |
| Strip digits                        | Depends | If annotators write "3" for "etɔn", normalize to letters |
| Transliterate to ASCII              | **No**  | Destroys the language |

### 5.2 SentencePiece word tokenizer (consistent segmentation)

The next step up: learn a consistent subword segmentation from your training data and apply it at eval time. This is the technique used in NLLB, SeamlessM4T, and most modern multilingual systems.

The idea: instead of trusting human spacing conventions (which vary), train a BPE/Unigram model on your training text. Then at eval, tokenize both reference and hypothesis with the same model. Compute "token error rate" instead of WER.

```python
import sentencepiece as spm

# One-time: train on your training text (NOT test text — data leakage!)
spm.SentencePieceTrainer.train(
    input="data/adja_train.txt",
    model_prefix="adja_spm",
    vocab_size=2000,              # small for low-resource
    character_coverage=1.0,       # critical: don't drop ɛ, ɔ, ŋ, ɖ
    model_type="unigram",         # or "bpe"
    normalization_rule_name="nmt_nfkc_cf",  # careful — check this preserves tones
)

# At eval time:
sp = spm.SentencePieceProcessor(model_file="adja_spm.model")

def tokenize(text: str) -> str:
    return " ".join(sp.encode(text, out_type=str))

ref_tok = [tokenize(normalize_for_asr(r)) for r in refs_raw]
hyp_tok = [tokenize(normalize_for_asr(h)) for h in hyps_raw]

ter = jiwer.wer(ref_tok, hyp_tok)  # "token error rate"
```

Pros: consistent segmentation, robust to annotator spacing.
Cons: TER is not directly comparable to published WER numbers. Use as a secondary diagnostic, not as your headline metric.

### 5.3 Character-level with word boundaries as markers

Some papers treat the word boundary as just another character — conventionally `|` or `▁` (the SentencePiece underscore). All evaluation then reduces to CER, but you keep the signal about where the model thought words ended.

```python
def encode_with_word_boundary(text: str, sep: str = "|") -> str:
    return sep.join(normalize_for_asr(text).split())

# "Enu maku enyi" -> "Enu|maku|enyi"
# "Enu ma ku enyi" -> "Enu|ma|ku|enyi"
#
# Now CER on these strings:
#   S=0, I=1 (the extra "|"), D=0, N=13
#   CER ≈ 7.7%  <- same as space-level CER, which is the point
```

This is primarily a training technique (wav2vec2 and many CTC systems already use it internally), but it's worth knowing about because you'll see `|` appear in MMS and XLS-R tokenizer vocabularies — that's where it comes from.

### 5.4 Which approach should you use?

For your paper, I'd recommend:

- **Headline metric**: CER with normalization (§5.1)
- **Secondary metric**: WER with normalization (§5.1)
- **Diagnostic**: raw WER (unnormalized), to show what the spacing problem costs
- **Don't bother (yet)**: §5.2 and §5.3. They're useful for model training, not for reporting.

This is also what PazaBench does — read their §3 for the exact normalization pipeline.

---

## 6. How to Report Metrics in a Paper

Standard practice for low-resource ASR papers. These are not rules; they're conventions that reviewers expect.

### 6.1 Report both CER and WER, lead with CER

Reviewers who know the low-resource space expect CER to be primary. Reviewers who only know English ASR will default to expecting WER and will be suspicious if you only report CER. Give them both, lead with CER, and explain *why* CER is the primary signal in one paragraph. Something like:

> "We report CER as our primary metric. Adja orthography is not fully standardized [cite your data section], and tonal distinctions are realized at the character level. WER, which treats every misaligned word as a total failure regardless of character overlap, systematically overstates error in both cases. CER provides a more faithful measure of phonetic recognition quality. We report WER as a secondary metric for comparability with prior work."

### 6.2 Report normalized alongside raw

Showing both lets reviewers see what the normalizer is doing:

```
                    CER      WER
Raw                 24.9%    73.1%
Normalized          22.1%    64.2%  (-2.8 / -8.9)
```

The WER gap is what the spacing and punctuation problem costs you. Being transparent about this builds reviewer trust.

### 6.3 Include confidence intervals for small test sets

Your test set is 160 utterances (10% of 1.6k). That's small enough that a few samples can move the number meaningfully. Report 95% CIs computed via bootstrap:

```python
import numpy as np

def bootstrap_ci(refs, hyps, metric_fn, n=1000, alpha=0.05):
    """Paired bootstrap over utterances."""
    scores = []
    N = len(refs)
    for _ in range(n):
        idx = np.random.choice(N, size=N, replace=True)
        sampled_refs = [refs[i] for i in idx]
        sampled_hyps = [hyps[i] for i in idx]
        scores.append(metric_fn(sampled_refs, sampled_hyps))
    lo = np.percentile(scores, 100 * alpha / 2)
    hi = np.percentile(scores, 100 * (1 - alpha / 2))
    return np.mean(scores), lo, hi
```

Report as `CER = 24.9% (95% CI: 22.7-27.1)`. This is now expected in ASR papers, especially for small test sets.

### 6.4 Include a qualitative analysis table

Pick 5-10 representative samples from your decode logs. Show:

| Reference          | Hypothesis         | CER  | WER  | Notes |
|--------------------|--------------------|------|------|-------|
| Enu maku enyi      | Enu ma ku enyi     | 7.7% | 66.7%| Spacing only |
| sɛgbasú nyí nɔví   | segbasú nyí nɔví   | 6.7% | 33.3%| ɛ→e, a common failure |
| má wó yì           | mà wó yì           | 11.1%| 33.3%| Wrong tone on má    |
| etɔn alafa ewò     | 3 100              | 89%  | 100% | Digit vs word form |
| kò nɔ nu ɖé        | kpɔ nɔ nu ɖé       | 27%  | 20%  | ASR confused k/kp |

This is gold for the discussion section. It turns numbers into stories and shows reviewers you understand your model's failure modes.

### 6.5 Reference papers for exact formatting

- **PazaBench** ([arxiv](https://arxiv.org/html/2512.10968v1)): the current gold standard for African ASR papers. Tables 2-4 show exactly the format reviewers expect. Pay attention to how they handle normalization reporting.
- **Ewe ASR paper** ([ACL ICNLSP 2025](https://aclanthology.org/2025.icnlsp-1.32.pdf)): the closest comparable setup to yours — Gbe family, similar data regime. Their metric section is short and clear.
- **MMS paper** ([JMLR 2024](https://jmlr.org/papers/v25/23-1318.html)): Appendix has per-language CER/WER for hundreds of languages. You can pull the Adja and Ewe numbers from here as baselines.

---

## 7. Connection to BLEU/chrF

This section is a bridge from what you know (NMT metrics) to what you're learning (ASR metrics).

### 7.1 The core parallel

| NMT                          | ASR                         |
|------------------------------|-----------------------------|
| BLEU (word-level, precision) | WER (word-level, error)     |
| chrF (char-level, F-score)   | CER (char-level, error)     |

The *direction* of the metric is different (BLEU higher-is-better, WER lower-is-better), but the *granularity choice* is the same.

### 7.2 The shared lesson

You already know why chrF is preferred over BLEU for low-resource NMT (from your ACL paper work):

1. **Tokenization fragility**: BLEU depends on tokenization. A different tokenizer → different BLEU. chrF sidesteps this by working at the character level.
2. **Morphological richness**: languages with productive morphology (Adja has some, Turkic languages have a lot) have huge vocabularies. Word-level metrics punish rare morphological variants; character-level metrics give partial credit.
3. **Partial credit**: chrF rewards getting *most* of a word right. BLEU doesn't.

**Exactly the same three reasons apply to CER vs WER for ASR in low-resource languages.** It's not a coincidence — it's the same underlying problem (word-level granularity is wrong when words are not the right unit).

Once you see this parallel, a lot of the low-resource ASR literature reads the same way as low-resource NMT literature. The vocabulary is different (phonemes vs morphemes, WER vs BLEU) but the arguments are isomorphic.

### 7.3 Going deeper: chrF and CER both come from the same statistical intuition

chrF is essentially character n-gram F-score. In the limit of unigrams (n=1), chrF precision + recall tracks character-level agreement closely. CER is one minus normalized edit distance over characters.

They're not the same metric — chrF uses F1 of n-gram sets, CER uses edit distance over sequences. But they correlate very highly (r > 0.9 in practice on the same outputs). If you've written about chrF being robust, you've already written the argument for CER being robust. Just swap the nouns.

### 7.4 The canonical chrF reference

- **Popović (2015) — chrF paper**: https://aclanthology.org/W16-2301/
  - The "why character-level" arguments in §2 of that paper transfer almost verbatim to the CER-over-WER argument.

---

## 8. Practical Recommendations for Your Paper

Concrete, actionable. Do these next.

### 8.1 Report three numbers, not one

```
CER:                   24.9%    (primary)
WER (normalized):       X.X%    (secondary, after §5.1 normalization)
WER (raw):             73.1%    (diagnostic — the gap to normalized shows the spacing cost)
```

Add 95% bootstrap CIs for each. Your N=160 test set genuinely needs them.

### 8.2 Write the "why CER" paragraph once, reuse it

Draft something like (condensed version of §2):

> "We report CER as our primary metric for three reasons. First, Adja orthography lacks full standardization [cite data section], so annotators vary in word-boundary placement. WER penalizes spacing variants as full word errors, which overstates recognition failure. Second, tonal distinctions in Adja are realized as diacritical marks at the character level; CER localizes errors in tone recognition, while WER collapses them into full-word errors. Third, in the low-resource regime (~2h training), many test words are out-of-vocabulary; CER rewards near-miss reconstruction of unseen words, while WER treats all near-misses as complete failures. CER is also the primary metric in recent African-language ASR benchmarks (PazaBench [cite], MMS [cite])."

That's your justification paragraph. Drop it in and move on.

### 8.3 Include a qualitative error table

Six to ten rows like the table in §6.4. Pick samples that illustrate:
- Pure spacing failures (high WER, low CER)
- Tone failures (low CER, still misleading)
- Rare-phoneme backoff (ɛ→e, ɔ→o)
- Digits vs written number forms
- Genuine acoustic confusion (k/kp, p/b)

This turns the numbers into a story and is the part reviewers actually read.

### 8.4 Mention future work explicitly

Low-effort, high-payoff statements for §6 of the paper:

- "Shallow fusion with an Adja n-gram language model" — proven to reduce WER substantially; see Whisper's LM decoder design.
- "Standardized orthography curation" — frame as a community contribution, not just an engineering task.
- "Joint character-word vocabulary" (hybrid CTC-attention decoding) — mitigates OOV without losing speed.
- "Tone-aware decoding" — explicit modeling of tone as a parallel output stream.

### 8.5 For comparisons, pull CER numbers (not just WER)

When comparing to PazaBench, MMS, or the Ewe paper, find their CER numbers. Many multilingual benchmarks report only WER in the abstract; CER is often in the appendix. Use the appendix numbers — they're the ones that compare apples to apples with yours.

Key CER numbers to find and cite:
- **MMS-1B on Adja** (if reported) or on Ewe — from the MMS paper appendix.
- **PazaBench Ewe/Fon** — closest linguistic neighbors to Adja.
- **FLEURS Yoruba/Igbo** — other West African languages, different families but similar data regime.

If MMS-1B on Adja has CER X% at 0 hours of fine-tuning and yours has 24.9% at 2h, that's a clean comparison showing the value of targeted fine-tuning.

---

## 9. Recommended Reading Order

If you have 4-6 hours total to spend on this topic before writing your paper's metrics section:

1. **Whisper normalizer source** (30 min, practical) — https://github.com/openai/whisper/blob/main/whisper/normalizers/basic.py
   - Read `basic.py` first. Then skim `english.py` to see what a "full" normalizer looks like. Model your Adja normalizer on `basic.py` + tone-safety rules from §5.1 of this doc.

2. **PazaBench §3-4** (45 min, methodology) — https://arxiv.org/html/2512.10968v1
   - The closest published template for your exact paper. Read §3 (methodology) and §4 (results). Note how they report normalization.

3. **chrF paper** (30 min, conceptual) — https://aclanthology.org/W16-2301/
   - You may have read this before. Re-read §2 with the "CER vs WER" parallel in mind. The arguments are isomorphic.

4. **Ewe ASR paper** (45 min, closest-peer) — https://aclanthology.org/2025.icnlsp-1.32.pdf
   - Same language family, similar data regime. Their metric section and qualitative analysis are directly transferable. Also a good citation target.

5. **FLEURS paper** (60 min, broader context) — https://arxiv.org/abs/2205.12446
   - Read §4 (methodology) and §5 (results). Helps you calibrate where Adja sits in the broader multilingual landscape.

6. **MMS paper low-resource section** (60 min, main comparison baseline) — https://jmlr.org/papers/v25/23-1318.html
   - Read §4 (ASR results), especially the low-resource language table. Pull baseline numbers for your comparison table.

Skip for now (come back if you go deeper):
- Baevski wav2vec 2.0 paper — relevant only if you're explaining the architecture.
- CTC original paper — only if your model uses CTC and a reviewer asks.

---

## Appendix A: Reference implementation of the evaluation pipeline

Putting it all together, here's what your `experiments/asr/shared/eval.py` should look like at minimum:

```python
"""
Evaluation pipeline for Adja ASR.

Usage:
    python eval.py --refs refs.txt --hyps hyps.txt --output results.json

Reports:
    CER (raw + normalized) with 95% CI
    WER (raw + normalized) with 95% CI
    Qualitative sample table for the 10 worst utterances
"""
import argparse
import json
from pathlib import Path

import jiwer
import numpy as np

from normalize import normalize_for_asr  # from §5.1


def bootstrap_ci(refs, hyps, metric_fn, n=1000, seed=42):
    """Paired bootstrap over utterances. Returns (mean, lo, hi)."""
    rng = np.random.default_rng(seed)
    N = len(refs)
    scores = []
    for _ in range(n):
        idx = rng.choice(N, size=N, replace=True)
        scores.append(metric_fn([refs[i] for i in idx],
                                [hyps[i] for i in idx]))
    return float(np.mean(scores)), \
           float(np.percentile(scores, 2.5)), \
           float(np.percentile(scores, 97.5))


def per_utt(refs, hyps, metric_fn):
    """Per-utterance metric for error analysis."""
    return [metric_fn([r], [h]) for r, h in zip(refs, hyps)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs", required=True)
    ap.add_argument("--hyps", required=True)
    ap.add_argument("--output", default="results.json")
    args = ap.parse_args()

    refs_raw = Path(args.refs).read_text().splitlines()
    hyps_raw = Path(args.hyps).read_text().splitlines()
    assert len(refs_raw) == len(hyps_raw), "ref/hyp length mismatch"

    refs_norm = [normalize_for_asr(r) for r in refs_raw]
    hyps_norm = [normalize_for_asr(h) for h in hyps_raw]

    results = {}
    for suffix, refs, hyps in [("raw", refs_raw, hyps_raw),
                               ("norm", refs_norm, hyps_norm)]:
        for metric_name, metric_fn in [("cer", jiwer.cer),
                                       ("wer", jiwer.wer)]:
            mean, lo, hi = bootstrap_ci(refs, hyps, metric_fn)
            results[f"{metric_name}_{suffix}"] = {
                "mean": mean, "ci_lo": lo, "ci_hi": hi,
            }

    # Per-utterance CER for error analysis — top 10 worst
    cer_per_utt = per_utt(refs_norm, hyps_norm, jiwer.cer)
    worst_idx = sorted(range(len(cer_per_utt)),
                       key=lambda i: -cer_per_utt[i])[:10]
    results["worst_samples"] = [
        {"ref": refs_raw[i], "hyp": hyps_raw[i],
         "cer": cer_per_utt[i]}
        for i in worst_idx
    ]

    Path(args.output).write_text(json.dumps(results, indent=2,
                                            ensure_ascii=False))
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
```

This gives you every number your paper's metrics section needs, plus the qualitative table content, in one run.

---

## Appendix B: Quick-reference cheat sheet

Stick this on a post-it:

- **Primary metric**: CER, normalized.
- **Secondary metric**: WER, normalized.
- **Always also report**: raw WER (to quantify the spacing cost).
- **Normalization**: NFC, lowercase, strip `.,!?;:`, collapse whitespace. Never strip tones or ɛ/ɔ/ŋ/ɖ.
- **For your 2h / 24.9% CER result**: you're in the PazaBench "good" range. Mention Ewe transfer as the secret sauce.
- **Comparison targets**: MMS-1B Ewe/Adja CER, PazaBench Ewe/Fon CER, FLEURS West African CER. Pull from appendices, not abstracts.
- **Confidence intervals**: bootstrap with n=1000, report 95% CI. Your N=160 needs this.
- **Qualitative table**: 10 rows, showing 4-5 distinct failure modes.
- **The "why CER" paragraph**: write once in §3 of the paper, reference from §6 of the paper.

---

*Last updated: 2026-04-16. Owner: Josue. If CER numbers change, update §3 of this doc and `results/comparison.md`.*
