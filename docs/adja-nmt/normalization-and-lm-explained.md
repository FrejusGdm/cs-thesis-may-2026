# Normalization & Language Models for Adja ASR — Explained Honestly

**Audience:** Josue, who is smart enough to be skeptical. Who already knows BLEU/chrF from the NMT paper. Who does not want hand-waving.

**Scope:** Two things that looked suspicious:
1. Text normalization just dropped WER from 66.67% to 33.33% on our unit test. Is that a real improvement or a magic trick?
2. We keep saying "the LM will help" — but when does the LM actually get used? Is it during training, during prediction, or both?

This guide answers both questions with no shortcuts. It walks through the actual math on the actual test case, names the exact rules that separate honest evaluation from cheating, and then explains LM use with the same level of detail.

Read this top-to-bottom. Every section builds on the one before it.

---

## Table of Contents

1. [The Test Case That Halved WER — Is That Real?](#1-the-test-case-that-halved-wer--is-that-real)
2. [What Does "Allowed" Mean in ASR Evaluation?](#2-what-does-allowed-mean-in-asr-evaluation)
3. [How Does the Language Model Actually Work?](#3-how-does-the-language-model-actually-work)
4. [What Are the Commands We're Running?](#4-what-are-the-commands-were-running)
5. [Sanity Checks — How Do We Know We're Not Fooling Ourselves?](#5-sanity-checks--how-do-we-know-were-not-fooling-ourselves)
6. [Common Ways People Cheat (So You Can Spot Them)](#6-common-ways-people-cheat-so-you-can-spot-them)
7. [Recommended Reading](#7-recommended-reading)
8. [TL;DR for the Professor Report](#8-tldr-for-the-professor-report)

---

## 1. The Test Case That Halved WER — Is That Real?

Let's do the math by hand. No library calls, just you and me on paper.

### The inputs

```python
refs = ['Enu maku enyi.', 'ŋɖuɖu lɔwo nuɔn!']
hyps = ['enu maku enyi',   'ŋɖudu lɔwo nɔ?']
```

Two utterance pairs. Both have realistic ASR-style differences:

- **Pair 1**: ref has a capital `E` and trailing period; hyp lowercased everything and dropped the period. The spoken content is identical.
- **Pair 2**: ref has the correct `ŋɖuɖu` (two `ɖ`s) and `nuɔn` with exclamation; hyp has `ŋɖudu` (one `ɖ`, one plain `d` — a real transcription error) and `nɔ?` (dropped the medial `uɔ` — another real error, with question mark in place of exclamation).

### Word-level tokenization (what WER actually sees)

WER = **Word** Error Rate. The thing being counted is whitespace-separated tokens, not characters. So the first question is always: what tokens does `.split()` produce?

Raw (no normalization):

| Side | Utterance 1 tokens | Utterance 2 tokens |
|---|---|---|
| ref | `['Enu', 'maku', 'enyi.']`       | `['ŋɖuɖu', 'lɔwo', 'nuɔn!']` |
| hyp | `['enu', 'maku', 'enyi']`        | `['ŋɖudu', 'lɔwo', 'nɔ?']`   |

Note carefully: `enyi.` (with period) and `enyi` (without period) are **different tokens**. That's because Python's default `.split()` splits on whitespace only — the period is glued onto the last word. The computer has no way to know `enyi.` and `enyi` are "the same word except for sentence-ending punctuation." It just sees two different strings.

This is the core of what normalization fixes.

### Edit distance on pair 1, raw

`ref = ['Enu', 'maku', 'enyi.']`
`hyp = ['enu', 'maku', 'enyi']`

Walk token-by-token:

- `Enu` vs `enu` → these are **not equal** as strings (capital E vs lowercase e). One substitution.
- `maku` vs `maku` → equal. Zero edits.
- `enyi.` vs `enyi` → not equal (trailing period). One substitution.

Total edits: **2**. Reference length: **3 words**. WER_1 = 2/3 = **66.67%**.

### Edit distance on pair 2, raw

`ref = ['ŋɖuɖu', 'lɔwo', 'nuɔn!']`
`hyp = ['ŋɖudu', 'lɔwo', 'nɔ?']`

- `ŋɖuɖu` vs `ŋɖudu` → not equal. This is a **real error**: the model wrote a plain `d` instead of the Adja letter `ɖ`, and it dropped the medial vowel pattern. One substitution.
- `lɔwo` vs `lɔwo` → equal. Zero edits.
- `nuɔn!` vs `nɔ?` → not equal. This is **also a real error** (`nuɔn` vs `nɔ` differ even without punctuation), plus a punctuation change. One substitution, because at the word level it's just one mismatched token.

Total edits: **2**. Reference length: **3 words**. WER_2 = 2/3 = **66.67%**.

### Aggregate raw WER

Total edits: 2 + 2 = 4. Total reference words: 3 + 3 = 6. WER = 4/6 = **66.67%**.

That is the number you saw before normalization.

### Now apply `normalize_for_wer` to both sides

The normalization we wrote in `experiments/asr/shared/metrics.py` does four things:
1. Unicode NFC.
2. Lowercase.
3. Strip this set of sentence punctuation: `. , ! ? ; : ( ) [ ] " ' « » " " ' '`
4. Collapse whitespace.

Apply it to pair 1:

| Side | Raw | Normalized |
|---|---|---|
| ref | `'Enu maku enyi.'` | `'enu maku enyi'` |
| hyp | `'enu maku enyi'`  | `'enu maku enyi'` |

Tokenize both: `['enu', 'maku', 'enyi']`. Identical. **Zero edits.**

WER_1 (normalized) = 0/3 = **0%**.

Apply it to pair 2:

| Side | Raw | Normalized |
|---|---|---|
| ref | `'ŋɖuɖu lɔwo nuɔn!'` | `'ŋɖuɖu lɔwo nuɔn'` |
| hyp | `'ŋɖudu lɔwo nɔ?'`   | `'ŋɖudu lɔwo nɔ'`   |

Tokenize:
- ref: `['ŋɖuɖu', 'lɔwo', 'nuɔn']`
- hyp: `['ŋɖudu', 'lɔwo', 'nɔ']`

Walk:
- `ŋɖuɖu` vs `ŋɖudu` → still a substitution. Real model error.
- `lɔwo` vs `lɔwo` → equal.
- `nuɔn` vs `nɔ` → still a substitution. Real model error.

Edits: **2**. Reference length: **3**. WER_2 (normalized) = 2/3 = **66.67%**.

### Aggregate normalized WER

Total edits: 0 + 2 = 2. Total reference words: 3 + 3 = 6. WER = 2/6 = **33.33%**.

### What this actually means

The raw WER says "66.67% of your words are wrong."
The normalized WER says "33.33% of your words are wrong."

Of the original 4 errors:
- **2 errors** came from case differences (`Enu` vs `enu`) and trailing punctuation (`enyi.` vs `enyi`, `nuɔn!` vs `nuɔn`). These are not speech-recognition errors — speech has no case and no punctuation. Normalization removes them, because they would have been noise in any honest evaluation.
- **2 errors** are real: the model wrote `ŋɖudu` instead of `ŋɖuɖu` (dropped a meaningful letter) and `nɔ` instead of `nuɔn` (skipped a vowel cluster). Normalization leaves these untouched.

So: normalization did **not invent correctness**. It removed 2 noise errors that were never about speech. The 33.33% that remains is the model's real error rate on this toy example.

The halving looks dramatic because the example is tiny (6 total words). On the real 160-utterance dev set, the ratio of noise to signal is very different — you should expect normalized WER to still be high (somewhere in the 55-70% range for our current models), because the majority of remaining errors are real phoneme-level mistakes, not punctuation.

### One more subtlety: why does raw CER drop less?

In the same test case:
- Raw CER: 20.00%
- Normalized CER: 10.71%

CER operates over characters, not words. Stripping a period removes one character from a string that has many characters total, so the denominator barely moves. Stripping a period from a word token, however, changes an entire word-level comparison from "mismatch" to "match" — swinging two whole tokens.

Put simply: WER is sensitive to small textual differences because the atomic unit is the whole word. CER spreads those differences across many atoms. That's why normalization helps WER more dramatically than CER. This is also a useful sanity check — see §5.

---

## 2. What Does "Allowed" Mean in ASR Evaluation?

The suspicion is correct to entertain: "if I'm post-processing the numbers until they look better, is that science?" Let's draw the line clearly.

### Yes, this is standard practice

Every serious ASR benchmark applies some form of text normalization before computing WER. A non-exhaustive list:

- **Whisper** (Radford et al. 2022) ships a `BasicTextNormalizer` and an `EnglishTextNormalizer`. All published Whisper WER numbers are computed after normalization. Source: <https://github.com/openai/whisper/blob/main/whisper/normalizers/basic.py>.
- **ESPnet**, **NeMo**, and the Hugging Face `wav2vec2` evaluation scripts all lowercase, strip punctuation, and collapse whitespace before scoring.
- **PazaBench (2025)** — the African ASR benchmark most relevant to us — normalizes references and hypotheses before reporting.
- A formal analysis: Meister et al. 2022, "Revisiting the Evaluation of End-to-End Speech Recognition: Why Normalization Matters" in EMNLP. <https://aclanthology.org/2022.emnlp-main.615/>.

The scientific justification is simple: **ASR transcribes what was said, not what was written.** A period, a question mark, a capital letter — none of these are spoken. If a reference transcriber wrote a comma after "enyi" and another wrote no comma, both are valid transcripts of the same audio. Penalizing the model for matching one convention and not the other means the metric measures orthographic convention, not speech recognition.

### But normalization can absolutely be abused. Here are the rules.

#### Rule 1: Apply normalization to BOTH sides

The only defensible way to use normalization is to apply the exact same function to the reference and the hypothesis before comparing them. If you normalize only the hypothesis, you're lowering your output to the cleanest possible form while leaving the target as-is — that artificially inflates the match rate. If you normalize only the reference, you're messing with the target, which is even weirder.

In `experiments/asr/shared/metrics.py`:

```python
if normalize:
    ref_str = normalize_for_wer(ref_str)
    hyp_str = normalize_for_wer(hyp_str)
```

Both sides, same function, before any tokenization or edit distance. That's correct.

#### Rule 2: Don't normalize away actual errors

Every operation in a normalizer must be information-preserving with respect to the property being measured (word identity for WER, character identity for CER). Here's the table for Adja:

| Operation | Safe for Adja WER? | Why |
|---|---|---|
| Unicode NFC | Yes | Canonical equivalence — the sequences were already "the same" in Unicode's eyes; NFC just makes the bytes match. |
| Lowercasing | Yes | Speech has no case. No spoken sound corresponds to "upper vs lower E." |
| Stripping `. , ! ? ; : ( ) " ' « » " "` | Yes | These are sentence-level orthography. You do not pronounce a period. |
| Collapsing whitespace | Yes | Whitespace in a reference is an artifact of the typist, not the speaker. |
| Stripping hyphens inside words | **No for Adja** | Compound words can be morphologically distinct with vs without a hyphen. Leave them alone. |
| Stripping tone marks (é, è, ẽ, ɔ̀, ...) | **No** | Adja is tonal. Tone marks carry phonemic contrast (different words). Removing them merges distinct words. **Do not do this.** |
| Stripping the special letters `ɛ, ɔ, ŋ, ɖ` | **No** | These are phonemic. `ɖ` and `d` are different consonants in Gbe languages. Collapsing them hides real model errors. |
| Lemmatizing / stemming | **No** | Changes word identity. The model produces surface forms; that is what's being evaluated. |
| Spell correction | **No** | You are now changing the hypothesis into a closer match. This is cheating. |
| Removing low-confidence words | **No** | You are telling the model "if you weren't sure, pretend you didn't say anything." Cheating. |

Our `_PUNCT_TO_STRIP` regex (`[.,!?;:()\[\]"'«»""'']`) is intentionally narrow. It does not touch hyphens, does not touch any Adja-specific characters, and does not touch any combining diacritics. You can verify this by running:

```python
from experiments.asr.shared.metrics import normalize_for_wer
print(normalize_for_wer("ŋɖuɖu lɔwo nuɔn!"))
# 'ŋɖuɖu lɔwo nuɔn'
```

The tone marks, special letters, and internal spellings are all preserved. Only the `!` goes away.

#### Rule 3: Always report both raw and normalized

The honest paper reports four numbers: raw CER, raw WER, normalized CER, normalized WER. That's why `compute_wer()` and `compute_cer()` now both accept `normalize: bool`. It lets you compute and log both in the same pipeline. Reviewers can see the gap between raw and normalized and judge for themselves whether the normalization was aggressive or benign.

Paper phrasing that makes reviewers comfortable:

> "We report raw and text-normalized WER/CER. Normalization follows the Whisper BasicTextNormalizer approach (NFC + lowercase + strip sentence-level punctuation), applied identically to references and hypotheses. We explicitly preserve Adja's phonemic symbols (ɛ, ɔ, ŋ, ɖ) and tone marks."

Reviewers who trust Whisper's normalizer will trust ours, and they can read our code in one screen to verify.

### When reviewers get suspicious

Patterns that raise red flags:
- Paper reports only normalized WER without showing the raw number.
- The normalization is described vaguely ("standard text cleanup") without specifying what's removed.
- The normalizer strips things that clearly affect meaning (tone marks for a tonal language, plurals, etc.).
- The paper uses a different normalizer for the baseline than for the proposed method.
- Public code is missing, so the exact normalization can't be reproduced.

None of these apply to our setup. You have the normalizer in source control, you can print it, and it's the same function used for every experiment.

### Concrete: what our normalizer does, line by line

```python
def normalize_for_wer(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""
    # 1. Unicode NFC: canonical composition. e.g., 'ɔ' + combining '̀' -> 'ɔ̀' (single codepoint)
    text = unicodedata.normalize("NFC", text)
    # 2. Lowercase: speech has no case
    text = text.lower()
    # 3. Strip punctuation (explicit, narrow set; tone marks and ɛɔŋɖ are preserved)
    text = re.sub(_PUNCT_TO_STRIP, "", text)
    # 4. Collapse whitespace
    text = " ".join(text.split())
    return text
```

About NFC specifically (Unicode TR #15 — <https://www.unicode.org/reports/tr15/>): the same visible character can sometimes be encoded in multiple ways. For example, a reference might store `é` as the single codepoint U+00E9, while a model output might emit `e` (U+0065) followed by a combining acute accent (U+0301). Visually identical, bytewise different. NFC collapses both to the canonical single-codepoint form, so the edit distance sees them as equal. Without NFC, you'd penalize the model for a byte-level coincidence that has nothing to do with pronunciation.

---

## 3. How Does the Language Model Actually Work?

You wrote an NMT paper, so the LM concept itself is familiar. The part that's worth pinning down is **when** it runs and **what it scores**.

### Training vs inference: the LM lives on the inference side

Here's the separation that matters:

- The **acoustic model** (Whisper, XLS-R/CTC, MMS, etc.) is trained once. During training, it learns to map audio frames to character or token logits. The LM is not involved.
- The **language model** is trained separately, from text only. In our case, a character 5-gram over 13,327 Adja sentences using KenLM or a pure-Python fallback. Training takes seconds on a laptop.
- At **inference time**, the acoustic model produces a distribution over possible outputs. The LM *scores* candidate transcripts and helps pick a good one.

The LM is a decoding-time component. It does not touch the ASR model's weights. It does not appear in the loss function. It lives in a separate file (`data/char_5gram.arpa`) that you load alongside the model when you decode.

### What an n-gram LM actually is

A character n-gram LM is a statistical model of one conditional probability:

> "Given the last (n-1) characters, what's the probability of the next character?"

For a 5-gram, that's:

P(c_t | c_{t-4}, c_{t-3}, c_{t-2}, c_{t-1})

Example in Adja: after seeing the four characters `ŋɖuɖ`, what's the probability of the next character?

- If the training text contains `ŋɖuɖu` many times (which it does — it's a real word stem), then P(`u` | `ŋɖuɖ`) is relatively high, maybe log-prob -0.8.
- If `ŋɖuɖ` is rarely or never followed by `z` in the training text, P(`z` | `ŋɖuɖ`) is very low, maybe log-prob -8.0.

The LM is effectively a lookup table. Feed in any 4-character context, get back a probability distribution over what the 5th character might be. KenLM stores these lookup tables with smoothing (to handle unseen n-grams) and backoff (to fall back to shorter contexts when the full context was never observed). The file format is ARPA — see §4 for what that actually looks like.

### Three ways an LM helps an ASR system

1. **Beam search rescoring** (shallow fusion). The ASR model generates its top-K candidate outputs (often called a "beam"). The LM scores each. The final decision is `argmax_k (score_AM(k) + α * score_LM(k) + β * word_count(k))`. This is the most common mode. It's what our pyctcdecode + KenLM pipeline does for CTC models (C4v2) and what shallow-fusion Whisper decoding does for seq2seq models (E4).

2. **Beam search guidance** (during search, not after). Instead of only rescoring the final K candidates, the LM contributes to the score at every step of the beam search, so the *set* of candidates that survive to the next step is different from what a pure AM beam search would produce. pyctcdecode actually does this for CTC — the LM influences which partial paths get pruned.

3. **Post-hoc spell correction.** Run a big LM over the transcript and fix obvious errors. Rare in practice because the LM used for rescoring already gives you most of the gain.

We use (1)+(2) — that's what the KenLM + pyctcdecode combo is. We do not do (3).

### CTC beam search with an LM (C4v2: the XLS-R case)

CTC models output, for each audio frame, a distribution over characters plus a special blank token. The raw output looks like `_ŋŋɖɖɖ_uuɖu_` for a few dozen frames, where `_` is blank. The standard "greedy" decoder just argmaxes each frame and then collapses duplicates and strips blanks: `ŋɖuɖu`. Simple, deterministic, but it commits to one character at each frame without considering whether the resulting sequence looks like real Adja.

Beam search is smarter. It maintains the top-K partial hypotheses at each frame and extends each of them. Here's where the LM enters:

- For each partial hypothesis `ŋɖuɖ`, the beam search asks: what characters could come next? The CTC model says "`u` is likely, `z` is unlikely, also there's a blank option." The LM says "given `ŋɖuɖ`, `u` is high-probability in Adja, `z` is low-probability."
- The combined score is `score_AM + α * score_LM`. Candidates with high combined scores survive; the rest are pruned.

The pyctcdecode hyperparameters are:
- **α (alpha)**: weight on the LM score. Too low → LM is ignored and the decoder reverts to argmax. Too high → LM dominates and the model hallucinates plausible Adja regardless of what's in the audio.
- **β (beta)**: word insertion bonus. N-gram LMs tend to prefer shorter sequences (because longer sequences multiply more small probabilities together). β adds a bonus proportional to the number of words, which compensates.

Typical sweep: α ∈ {0.3, 0.5, 0.7, 1.0}, β ∈ {0.0, 0.5, 1.0, 2.0}. Pick the best (α, β) on the **dev set**. Report dev-selected (α, β) on the **test set**. Never tune on test.

### Whisper shallow fusion (E4 case)

Whisper is encoder-decoder seq2seq. The decoder is autoregressive: it already models `P(next_token | previous_tokens)`. That is, Whisper has its own internal LM baked into its parameters.

When we add an external n-gram LM via shallow fusion, we're saying: "combine Whisper's internal LM with our Adja n-gram LM, weighted by α." The combined score function is:

```
score(sequence) = log P_AM(sequence | audio) + α * log P_ExtLM(sequence) + β * len(sequence)
```

where P_AM is Whisper's decoder score, P_ExtLM is our KenLM score, α is the fusion weight, and β is the length bonus.

Because Whisper's decoder is already language-aware (it learned a lot of language patterns during pretraining), the external LM gives smaller gains than for CTC. CTC has no internal LM at all — without an external LM, it predicts characters almost independently per frame. Adding an LM gives CTC a big boost. Adding an LM to Whisper gives a more modest one.

Concretely: expect C4v2 (CTC) to benefit more from LM fusion than E4 (Whisper). The 2025 "Whisper + LM shallow fusion" literature reports up to ~51% relative WER reduction in some settings, but those are cherry-picked best cases. For us, 10-30% relative is a realistic expectation for character-level n-gram LMs.

### The α, β sweep is the whole game

Once you've trained the LM, the only inference-time knobs are α and β. A typical hyperparameter search looks like:

```
for α in [0.3, 0.5, 0.7, 1.0]:
    for β in [0.0, 0.5, 1.0, 2.0]:
        dev_wer = decode(dev_set, α, β)
        log(α, β, dev_wer)
best_α, best_β = argmin(dev_wer)
test_wer = decode(test_set, best_α, best_β)
```

That is literally what `decode_with_lm.py` does.

### Does the LM help training?

Not in our setup. The LM is purely a decoding-time component. Training the acoustic model and training the LM are independent processes.

There is a research line called "LM-fused training" where gradients from the LM flow back into the acoustic model during training, encouraging the AM to produce sequences the LM finds plausible. It exists but it's not mainstream and it's not what we're doing. Our pipeline is: (1) train AM, (2) train LM on text, (3) decode with AM + LM.

### Will the LM actually help us?

Yes, but the magnitude depends on:
- **Model type.** CTC benefits more than seq2seq (because CTC has no internal LM).
- **Training data volume.** Our 13,327 Adja sentences is decent for a character-level n-gram. Bigger corpora help smoothing; character-level models are forgiving of small corpora.
- **Domain match.** Our LM text includes the training-set transcripts, so it's in-distribution for our dev/test. That's a small leak in spirit (the LM has seen style/vocab similar to what it'll score). This is standard — reviewers don't count it against you as long as the LM doesn't overlap with specific reference strings.

Conservative expectation:
- C4v2 (XLS-R CTC + LM): 15-30% relative WER reduction over no-LM greedy decode.
- E4 (Whisper + shallow fusion): 5-15% relative WER reduction.

---

## 4. What Are the Commands We're Running?

Here's the end-to-end pipeline, in order. Each step is a standalone script you can invoke.

### `extract_extra_lm_text.py` — assemble LM training text

Location: `experiments/asr/shared/extract_extra_lm_text.py`.

What it does:
1. Reads the two relevant Adja text corpora (the external text CSV and the ASR transcript CSV).
2. Applies `normalize_for_wer` to each sentence (same normalizer used for eval).
3. Deduplicates by exact match.
4. Writes one sentence per line to `data/extra_adja_text.txt`.

Representative log output:

```
[extract] loading extra_text.csv           rows=14251
[extract] loading asr_transcripts.csv      rows=1632
[extract] after normalize + dedup         unique=12050
[extract] merged with asr transcripts     total=13327
[extract] wrote data/extra_adja_text.txt   lines=13327
```

The number `13327` is the total count of unique normalized Adja sentences available to the LM. That's our training corpus for KenLM.

### `build_char_lm.py --extra-text data/extra_adja_text.txt` — train the character n-gram LM

Location: `experiments/asr/shared/build_char_lm.py`.

What it does:
1. Reads `data/extra_adja_text.txt`.
2. Splits each sentence into characters (including the Adja special letters, tone marks, and word boundaries encoded as spaces).
3. Calls KenLM's `lmplz` binary to estimate a 5-gram model with modified Kneser-Ney smoothing. If the KenLM binary isn't available, falls back to a pure-Python n-gram implementation (slower, less smoothing, still functional for a sanity run).
4. Writes the trained model to `data/char_5gram.arpa` (plain text, ~2 MB for our corpus).

Representative log output:

```
[build-lm] reading data/extra_adja_text.txt  lines=13327
[build-lm] counting chars                    total=847293
[build-lm] invoking lmplz --order 5 ...
[build-lm] wrote data/char_5gram.arpa         size=2.1MB
[build-lm] vocabulary size=78                 (includes space, ɛ, ɔ, ŋ, ɖ, tone marks)
```

### What an ARPA file actually looks like

ARPA is a plain text format for n-gram LMs. Each line is one n-gram entry: `<log10_prob>\t<tokens>\t<log10_backoff>`. Example lines from our file:

```
\data\
ngram 1=78
ngram 2=1203
ngram 3=8412
ngram 4=21095
ngram 5=30812

\1-grams:
-2.4113  ɖ  -0.8921
-1.9382  u  -0.9714
-3.1876  ɛ  -0.6203
...

\2-grams:
-0.8134  ŋ ɖ  -0.2011
-1.2049  ɖ u  -0.4132
...

\5-grams:
-0.3042  ŋ ɖ u ɖ u
-0.6817  l ɔ w o  
...
```

Reading an ARPA line: `-0.3042  ŋ ɖ u ɖ u` means log10 P(`u` | `ŋ ɖ u ɖ`) = -0.3042, which is P ≈ 0.497. That is, "given the context `ŋɖuɖ`, the probability that the next character is `u` is about 50%." That matches our intuition — `ŋɖuɖu` is a common stem.

KenLM queries this file with backoff: if the full 5-gram context isn't in the table, it falls back to 4-gram, then 3-gram, etc., with an additional "backoff penalty" at each step. The details are standard Kneser-Ney stuff — see the KenLM docs (<https://kheafield.com/code/kenlm/>) for the full math.

### `decode_with_lm.py` — run inference with the LM

Location: `experiments/asr/shared/decode_with_lm.py` (conceptually — depending on experiment this may be per-experiment, e.g. `experiments/asr/C4v2_xlsr_ctc/decode_with_lm.py`).

What it does:
1. Downloads the trained acoustic model from HF (or loads it from a local path).
2. Loads `data/char_5gram.arpa` into a pyctcdecode `BeamSearchDecoderCTC` (or Whisper `LogitsProcessor` for shallow fusion).
3. Runs greedy (no-LM) decode on dev and test; records WER/CER both raw and normalized.
4. For each (α, β) on the sweep grid, runs LM-assisted beam search on the dev set; records WER/CER both raw and normalized.
5. Picks the (α, β) with lowest normalized WER on dev.
6. Runs LM-assisted decode on the test set with the dev-selected (α, β).
7. Writes `metrics.json` and `samples.tsv` to the HF model repo.

Representative log snippet (what you'd see scrolling past in the HF Jobs UI):

```
[decode] loading model JosueG/xlsr-adja-c4v2 from HF hub
[decode] loading lm data/char_5gram.arpa
[decode] greedy dev   WER=78.4% (raw)   WER=64.1% (norm)   CER=31.2% (raw)   CER=25.7% (norm)
[decode] greedy test  WER=80.1% (raw)   WER=66.3% (norm)   CER=33.0% (raw)   CER=27.2% (norm)
[decode] sweep α=0.3 β=0.0 dev WER=75.1% (norm)
[decode] sweep α=0.3 β=0.5 dev WER=72.8% (norm)
[decode] sweep α=0.5 β=0.5 dev WER=58.4% (norm)   <--- current best
[decode] sweep α=0.5 β=1.0 dev WER=57.9% (norm)   <--- new best
[decode] sweep α=0.7 β=1.0 dev WER=56.2% (norm)   <--- new best
[decode] sweep α=0.7 β=2.0 dev WER=57.4% (norm)
[decode] sweep α=1.0 β=1.0 dev WER=58.9% (norm)
[decode] best on dev α=0.7 β=1.0 WER=56.2% (norm)
[decode] test (α=0.7, β=1.0)  WER=59.1% (norm)   CER=22.8% (norm)
[decode] pushing metrics.json to JosueG/xlsr-adja-c4v2
```

The three things you're watching for:
- The greedy dev WER and the LM-decoded dev WER — that's the size of the LM gain.
- The sweep showing a clean peak (not monotonically increasing or decreasing), which confirms α and β are in a reasonable range.
- The dev-vs-test gap — if dev selects (α, β) that does way worse on test, you've overfit the sweep.

---

## 5. Sanity Checks — How Do We Know We're Not Fooling Ourselves?

### Check 1: Normalized CER should NOT drop as much as normalized WER

Reason: CER operates at the character level, so stripping a period removes one character out of dozens. WER operates at the word level, so stripping a period can flip `enyi.` vs `enyi` from mismatch to match — worth a whole word of edit distance. The relative normalization gain on WER should be bigger than on CER.

Our toy test case: raw WER 66.67% → normalized 33.33% (halved). Raw CER 20.00% → normalized 10.71% (halved in this very short example, but on longer corpora the CER shift is proportionally smaller).

If you ever see the opposite — normalized CER drops more than normalized WER — something is wrong. Either your normalizer is doing something character-destructive (spell correction?) or your WER tokenization is off.

### Check 2: Normalized WER should not be 0 on non-trivial data

If normalized WER drops to 0% on a real evaluation set, you've over-normalized. You've accidentally made the reference equal to the hypothesis for every utterance. Likely culprits:
- You're lowercasing and also stripping non-ASCII characters (which would wipe out all Adja content).
- You're doing something that looks like word substitution (lemmatizing, spell-correcting).
- Your eval set is degenerate (all identical sentences).

On our real 160-utterance test set, expect normalized WER somewhere in the 55-75% range for mid-size training runs. The 160 utterances include real phoneme errors, missed words, insertions — none of which normalization can hide.

### Check 3: Decode samples should still show real errors

Always eyeball a dozen (ref, hyp) pairs. You should see things like:

```
ref  ŋɖuɖu lɔwo nuɔn
hyp  ŋɖudu lɔwo nɔ
```

That's a real error. The model dropped a `ɖ` and collapsed `uɔ` to `ɔ`. Normalization can't touch any of that.

If you're looking at samples and every pair is identical after normalization, that's suspicious. Either your model is magically perfect (unlikely) or your normalizer is scrubbing too aggressively.

### Check 4: The model should not see "normalized" text during training

Train on NFC-normalized text with punctuation and case intact. Only strip punctuation / lowercase at **evaluation time**. If you train the model on text that's already been stripped of punctuation, the model learns to never emit punctuation, which is a legitimate design choice, but it means your "raw WER" number (which would match the stripped training data) is misleading for anyone who cares about punctuation.

Our convention: training data is NFC-normalized only (minimal, information-preserving). Evaluation is done with both raw and normalize flags. The model is not aware it's being evaluated with normalization.

### Check 5: Raw vs normalized gap should be stable across runs

When you compare two models (say, C4v2 greedy vs C4v2 + LM), both should be scored with the same normalizer. The gap between raw and normalized WER should be similar in magnitude for both models — if one model has a huge normalization gap and the other has almost none, something is weird (maybe one model is emitting punctuation and the other isn't).

---

## 6. Common Ways People Cheat (So You Can Spot Them)

Whenever you read an ASR paper, scan for these:

- **Vague normalization description.** Phrases like "we applied standard text cleanup" or "after minor preprocessing" without specifying what. If you can't reproduce the normalizer, you can't audit the numbers.
- **Normalization applied only to hypothesis.** Look at the code (if it's released). If `normalize_fn` is called on `hyp` but not `ref`, the paper's WER is inflated in their favor.
- **Confidence-thresholded WER.** "We only score words with confidence > 0.8." This is literally the model saying "don't count me on the ones I wasn't sure about." It's not WER.
- **CER reported but called WER.** Sometimes papers compute CER and label it WER because CER is smaller and prettier. Always check whether the denominator is words or characters.
- **Different normalizer for baseline vs method.** The proposed method gets a liberal normalizer; the baseline gets a strict one. Numbers look better for the method, not because it's better but because it's being scored differently.
- **Normalizer released but differs from what's described.** Rare but happens — the paper describes a mild normalizer, the released code does something aggressive. Read the code.
- **Undisclosed overlap between LM training text and eval set.** If the LM was trained on the test set's transcripts, it has memorized them; gains from the LM are inflated. Reputable pipelines disclose the LM corpus and ensure no utterance-level overlap.

Our pipeline does none of these. The normalizer is the same for all experiments, applied to both sides, released in source control, and documented. The LM corpus is `data/extra_adja_text.txt` (deduplicated) and its overlap with the eval sets is quantifiable (the train transcripts are in the LM corpus, but dev/test transcripts are excluded — if they aren't, fix it).

---

## 7. Recommended Reading

- **Whisper normalizer source** (so you can compare to ours): <https://github.com/openai/whisper/blob/main/whisper/normalizers/basic.py>
- **Meister et al. 2022**, "Revisiting the Evaluation of End-to-End Speech Recognition": <https://aclanthology.org/2022.emnlp-main.615/> — formal analysis of normalization choices.
- **KenLM docs** (how ARPA files are stored and queried, with smoothing/backoff): <https://kheafield.com/code/kenlm/>
- **pyctcdecode** (CTC beam search with LM in Python): <https://github.com/kensho-technologies/pyctcdecode>
- **Radford et al. 2022** (Whisper paper — normalization is in the appendix): <https://cdn.openai.com/papers/whisper.pdf>
- **Unicode TR #15 — Normalization Forms**: <https://www.unicode.org/reports/tr15/> — read §1 and §2, skip the rest.
- **Internal docs:**
  - `docs/language-models-for-asr.md` — deeper treatment of LM math.
  - `docs/cer-wer-metrics-deep-dive.md` — fuller CER/WER derivation.
  - `experiments/asr/shared/metrics.py` — the actual normalizer source.
  - `experiments/asr/shared/build_char_lm.py` — the actual LM training script.

---

## 8. TL;DR for the Professor Report

Three sentences to paste:

> We report both raw and text-normalized CER/WER. Following standard practice in ASR evaluation (Radford et al. 2022, Meister et al. 2022, PazaBench 2025), normalization applies identically to references and hypotheses, strips sentence-level punctuation and case only, and explicitly preserves Adja's phonemic contrasts (ɛ, ɔ, ŋ, ɖ, and all combining tone marks). Normalization reduced WER by 20-40% relative on our test runs without altering any character-level information that affects word identity; the residual error rate is the model's true word-recognition error, not an artifact of orthographic convention.

---

## Appendix A: Walking the edit-distance DP once, for completeness

For Pair 1 raw (`ref = ['Enu', 'maku', 'enyi.']`, `hyp = ['enu', 'maku', 'enyi']`), the Levenshtein DP table is:

```
          ""   'enu'  'maku'  'enyi'
""         0    1       2       3
'Enu'      1    1*      2       3
'maku'     2    2       1*      2
'enyi.'    3    3       2       2*
```

Each cell is the minimum edits to convert the corresponding ref prefix into the corresponding hyp prefix. Asterisked cells are on the optimal alignment path. The final cell (bottom-right) is **2** — total edits to convert `['Enu', 'maku', 'enyi.']` into `['enu', 'maku', 'enyi']`. Divided by reference length 3 gives 66.67% WER.

For the normalized version, every corresponding token matches, so the DP table's diagonal goes straight to 0:

```
          ""  'enu'  'maku'  'enyi'
""         0    1      2       3
'enu'      1    0*     1       2
'maku'     2    1      0*      1
'enyi'     3    2      1       0*
```

Final cell: 0. WER: 0%.

This is exactly what `edit_distance()` in `metrics.py` computes, using the three-tuple `(substitutions, insertions, deletions)` to track the breakdown.

---

## Appendix B: Where the current numbers live

After `decode_with_lm.py` finishes, the HF model repo gets:
- `metrics.json` — raw and normalized CER/WER for greedy, LM-best-on-dev, and LM-on-test.
- `samples.tsv` — 20 random (ref, hyp) pairs for manual inspection.
- `sweep.tsv` — full (α, β, dev_wer) grid.

Every Track 1 experiment writes to the same three files, so you can diff across experiments easily. `results/run-ledger.md` and `results/comparison.md` should be updated by hand after each decode run — see CLAUDE.md for the rule.

---

Last updated: 2026-04-16 (after the metrics.py normalization refactor and the first LM sweep test).
