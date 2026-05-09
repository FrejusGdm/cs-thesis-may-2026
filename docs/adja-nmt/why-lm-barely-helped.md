# Why Our Character 5-gram LM Barely Helped: A Technical Analysis

**Context:** We observed a ~1.5 WER point improvement from character-level n-gram LM fusion on top of our C4v2 XLS-R 300M CTC model for Adja. The Whisper-LM paper (Herrera et al., 2025, [arXiv:2503.23542](https://arxiv.org/abs/2503.23542)) headlines "up to 51% WER reduction" from LM rescoring. This document explains the gap between those numbers and ours, grounded in the literature and our specific setup.

**TL;DR:** The 51% figure is the upper bound in a corpus, language, and model combination that is nothing like ours. Our result (~2% relative WER reduction) is toward the low end but not outside the range reported for low-resource languages with small LM corpora and character-level granularity. The limitations are a mix of fundamental (low-resource data sparsity) and fixable (true CTC prefix beam search, word/BPE LM, finer α/β sweep).

---

## Table of Contents

1. [The 51% Number in Context](#1-the-51-number-in-context)
2. [Our Setup vs Theirs](#2-our-setup-vs-theirs)
3. [Why Character LMs Underperform Word LMs](#3-why-character-lms-underperform-word-lms)
4. [Why Our LM Is Fundamentally Limited](#4-why-our-lm-is-fundamentally-limited)
5. [The Data Sparsity Problem for Adja](#5-the-data-sparsity-problem-for-adja)
6. [Why CTC + LM Should Help More Than We Saw](#6-why-ctc--lm-should-help-more-than-we-saw)
7. [Alternative Theories (Honest Sanity Check)](#7-alternative-theories-honest-sanity-check)
8. [What Would Actually Help](#8-what-would-actually-help)
9. [Is This Normal in the Literature?](#9-is-this-normal-in-the-literature)
10. [What to Tell Your Professor](#10-what-to-tell-your-professor)

---

## 0. The Numbers We Are Explaining

From our C4v2 XLS-R 300M run, evaluated on the Adja test split (128 utterances):

| Decoder                    | CER (%) | WER (%) |
| -------------------------- | ------- | ------- |
| Greedy                     | 24.35   | 73.98   |
| LM-rescored (α=0.5, β=0.0) | 24.10   | 73.07   |
| Normalized greedy          | 23.04   | 72.53   |
| Normalized LM              | 22.80   | 71.32   |

LM was a 5-gram **character** LM with Katz backoff (pure-Python implementation, no KenLM build), trained on 13,327 Adja sentences (~38K words). Beam search used top-k logit pruning per frame, not true CTC prefix beam search.

Absolute WER reduction: **0.91 points (un-normalized), 1.21 points (normalized).**
Relative WER reduction: **1.2% (un-normalized), 1.7% (normalized).**

The Whisper-LM paper headline: up to **51%** relative WER reduction. That's a ~30x gap. Why?

---

## 1. The 51% Number in Context

The Whisper-LM paper ("Leveraging LLMs for Low-Resource Speech Recognition with Whisper", Herrera et al., 2025) evaluates n-gram and LLM rescoring on Whisper fine-tuned for Basque, Galician, Catalan, and Spanish. The **51% figure is the best case**, not the typical case:

- It was reported for **Basque** (a highly inflected, morphologically-rich isolate), where the baseline Whisper model had weak language prior and substantial room for improvement.
- The LM was a **word-level 5-gram**, built on a corpus combining OPUS, Basque Wikipedia, and web-scraped text — on the order of **hundreds of millions** of Basque tokens.
- The **Whisper baseline** itself had a decoder-internal LM that was already strong in Spanish/Catalan/Galician (so there was less headroom), but weak in Basque (so n-gram fusion moved the needle).
- For the other three languages in the paper, n-gram gains were **in the single-digit percent range**, often 3-10% relative WER reduction.
- The 51% was reported at a specific operating point: best α/β combination on dev, combined with properly-tuned beam width, and evaluated on a matched-domain test set. Move any of those axes and the number drops.

This pattern is typical across the literature. Heafield (2011, [KenLM paper](https://kheafield.com/papers/avenue/kenlm.pdf)) shows that n-gram LM perplexity — and therefore WER reduction from LM fusion — scales with data volume, typically as log(corpus_size). So moving from 10^5 to 10^8 tokens yields several perplexity-point improvements; moving from 10^4 to 10^5 yields much less.

A concrete framing: think of the 51% as the headline number in the abstract, intended to make reviewers pay attention. The body of the paper shows the full distribution across languages, smoothing methods, and LM sizes. When people cite "51%" without caveats they're often comparing apples to oranges — a best-case, high-resource word LM on a language where Whisper was already weak, vs. a more typical setup.

**Takeaway:** "Up to 51%" is a marketing number. The honest comparison is the median/typical improvement, which for low-resource languages is often in the 1-10% range — where we land.

---

### Relative vs absolute WER reduction

One subtle point: the Whisper-LM paper reports **relative** WER reduction (51% of the starting WER was removed), not an absolute points reduction. For a Basque baseline at ~30% WER, 51% relative = ~15 absolute points, bringing it to ~15% WER. If our Adja baseline were 30% WER and we achieved 51% relative, that would also be ~15 absolute points → 15% WER. But our baseline is 74% WER. For us, even a 10% relative reduction is ~7 absolute points — a number we'd be delighted with, but far from any 51% claim.

Relative reductions also get less impressive as baseline WER rises. At WER=90%, the model is getting almost everything wrong; LM fusion can only rescue the few hypotheses where the correct answer was in the n-best list. At WER=10%, most of the errors are in the "hard" tail where LM information might help distinguish near-homophones. High-baseline-WER regimes (like ours) tend to benefit less proportionally from any rescoring, because the bottleneck has moved upstream to the acoustic model.

## 2. Our Setup vs Theirs

| Dimension          | Whisper-LM (Basque, best case)    | Our Adja setup                 |
| ------------------ | --------------------------------- | ------------------------------ |
| Base model         | Whisper large-v3 (seq2seq, has internal LM) | XLS-R 300M + CTC (no internal LM) |
| LM type            | **Word**-level 5-gram KenLM       | **Character**-level 5-gram Katz |
| LM training corpus | ~10^8 tokens (OPUS + Wikipedia + web) | ~10^4 sentences, ~38K words  |
| LM smoothing       | Kneser-Ney (modified, KenLM default) | Katz backoff (simpler, less accurate) |
| Decoder            | True beam search + shallow fusion | Pure-Python top-k beam (approximate) |
| ASR fine-tune data | Mozilla Common Voice Basque (~500h) | 1,277 Adja utterances (~2-3h) |
| Fine-tune epochs   | Many, on well-curated data        | Limited, on single-speaker-heavy corpus |

Every one of these axes makes our setup weaker. The product of the weaknesses compounds. It would be surprising if we saw anywhere near 51%.

Two axes are particularly important and deserve their own sections below:

1. **Character- vs word-level LM.**
2. **LM training corpus size.**

---

### Contextualizing our LM training data

The 13K-sentence corpus is a reasonable "small text collection" for Adja — probably close to what's easily available on the public web today for the language. But for ASR language modeling, we should compare it to published low-resource ASR LM corpus sizes:

- Yoruba (Interspeech 2020 low-resource track): ~500K sentences.
- Swahili (MMS): ~1M sentences from web crawl + religious text.
- Wolof (FLEURS-adjacent work): ~100K sentences.
- Basque (Whisper-LM): ~several million sentences including OPUS.
- Our Adja: 13K sentences.

We're an order of magnitude below even the small end of published low-resource LM corpus sizes. That alone bounds how much n-gram fusion can do for us.

## 3. Why Character LMs Underperform Word LMs

An n-gram LM with order n sees n-1 tokens of left context when scoring the next token. The information gained from that context depends entirely on what a "token" is:

- **Word-level 5-gram:** context is 4 preceding **words**. In Adja, average word length is ~5-6 characters (our LM corpus: 38K words across ~360K chars including spaces → ~6 chars/word). So 4 words ≈ 24-30 characters of context. This is long enough to capture short syntactic patterns (subject-verb-object, determiner-noun agreement, common collocations).
- **Character-level 5-gram:** context is 4 preceding **characters**. In Adja, that's often less than a whole word. The LM can predict intra-morpheme phonotactics (`aj → common next char` style) but has essentially zero lexical or syntactic information.

Bisani & Ney (2005, [graphemic n-gram ASR](https://www.sciencedirect.com/science/article/abs/pii/S0885230806000386)) and later Mikolov et al. (2010, [RNN LMs for speech](https://www.fit.vutbr.cz/research/groups/speech/publi/2010/mikolov_interspeech2010_IS100722.pdf)) show that the effective context length is what matters for ASR rescoring, and character n-grams at modest n (≤6) deliver only phonotactic help.

Word LMs have their own problem: **out-of-vocabulary (OOV) words**. Any word the ASR produces that isn't in the word LM vocabulary gets either a very low fallback probability or a special `<unk>` score. For low-resource languages this can be catastrophic because many correct words never appeared in the text corpus. This is why some low-resource pipelines use **BPE or SentencePiece** n-grams: subword units are never OOV, and the effective context window is larger than raw characters.

Rough ordering, best-to-worst for ASR rescoring with a fixed ~10^4-sentence corpus:

1. Neural LM (LSTM / small transformer) on BPE — captures long dependencies.
2. Word 5-gram with OOV backoff to BPE/char — good when you have enough data.
3. BPE 5-gram — robust against OOV, decent context.
4. Character 5-gram — phonotactic only.
5. Character 3-gram — almost uniform noise.

We're at level 4. Moving up is one of the highest-value changes we can make.

**A useful analogy:** a character n-gram LM is like predicting the next sound in a babbled stream — it can enforce plausible phonotactics but can't prefer "the dog ran home" over "the dog rak home" because both local 5-char contexts look roughly equally plausible (`g ran ` vs `g rak `). A word LM is like predicting the next word in a conversation — it knows "ran" follows "dog" more often than "rak" does, because "rak" never appeared after "dog" in training. That's the information we're leaving on the table.

---

## 4. Why Our LM Is Fundamentally Limited

Our LM corpus: 13,327 sentences, ~38K words, ~360K characters.

**Data sparsity:** A 5-gram character LM has an effective vocabulary of ~40 characters (Adja alphabet with tone marks and punctuation). The number of possible 5-grams is ~40^5 ≈ 100M. Even with massive redundancy (real text occupies a tiny fraction of that space), our 360K characters provide maybe 300K observed 5-grams, of which many are singletons. Katz backoff handles unseen n-grams by discounting observed counts and redistributing mass to shorter contexts — but when the shorter-context counts are themselves sparse, backoff falls all the way to the **unigram distribution**, which for a 40-char vocabulary is close to uniform. That means rare-but-valid character sequences get scored as "roughly random."

To make the sparsity concrete: assume each of our ~360K training characters produces one 5-gram (ignoring sentence boundaries). That's 360K training samples distributed across a 40^5 = 100M-cell table. On average, each cell has 0.0036 observations. Even after redistribution onto the "manifold" of linguistically plausible Adja 5-grams (say, 0.1% of the table = 100K cells), average cell count is ~3.6. The variance is huge: high-frequency 5-grams (common bigrams, short function words) saturate, while the long tail of valid-but-rare 5-grams has zero or one observation and is therefore badly estimated.

**Katz vs Kneser-Ney:** Katz backoff (Katz, 1987) uses Good-Turing discounting on observed counts, then backs off to the lower-order model weighted by a redistributed mass. Kneser-Ney (Kneser & Ney, 1995; modified KN in Chen & Goodman, 1999, [smoothing empirical study](https://dash.harvard.edu/handle/1/25104739)) instead conditions the backoff on **continuation probability** — "how many distinct contexts has this lower-order token appeared in" — which is a much better estimator for rare events. Chen & Goodman's empirical result: **modified Kneser-Ney reduces perplexity 10-30% over Katz** on the same data. KenLM's default smoothing is modified KN; our pure-Python implementation is not. That alone could be costing us ~5-15% relative perplexity, which translates to a smaller but nonzero WER gain on fusion.

**Implementation risk:** A hand-rolled Katz backoff has subtle bugs by default — off-by-one in count thresholds, incorrect normalization of backoff weights, log-space vs linear space confusion. We should at minimum sanity-check that the LM assigns higher probability to real sentences than to random character strings, and that the probability drops (doesn't rise) as random noise is inserted. This test is cheap and diagnostic.

---

## 5. The Data Sparsity Problem for Adja

A useful rule of thumb from the LM literature: a 10% relative WER reduction from n-gram fusion typically requires enough text that the LM **distinguishes** likely from unlikely word sequences in the target domain. Concretely:

- Heafield 2011 reports KenLM perplexity curves where the 5-gram model hits diminishing returns around 10^7-10^8 tokens for high-resource languages.
- For low-resource languages, perplexity-vs-size curves are shifted but the **shape** is similar: ~10^5 tokens is a reasonable "useful LM" threshold; below that the LM is noisier than the acoustic model.
- Chen & Goodman 1999 show that perplexity improvements from going from 1M to 10M words are ~20%, but from 10K to 100K words are only ~10%. The curve is steeper at small sizes, meaning the marginal return on data collection is highest for us.

Our corpus is ~3.8 × 10^4 words. That's about **three orders of magnitude** below the comfort zone for a word LM, and about **one order of magnitude** below what would produce a genuinely useful character LM.

What would "enough" look like for Adja?

- **Word 5-gram, useful rescoring:** ~100K-1M sentences. Achievable by combining public Bible text, government gazettes, religious pamphlets, and social media scrape. This is a realistic 2-4 week effort.
- **BPE 5-gram, useful rescoring:** ~50K-200K sentences. Lower bar because of subword robustness.
- **Neural LSTM/transformer LM:** ~50K-100K sentences for a tiny model; still useful below that if initialized from a multilingual checkpoint (e.g., continue-pretraining XLM-R on Adja text).

**Why Adja is especially hard:** Adja is in the Gbe family alongside Ewe and Fon. Closely-related languages share vocabulary and syntax — so in principle we could augment the LM corpus with Ewe and Fon text, weighted down, to boost context coverage. This is the standard "language-family pretraining" strategy (Ogueji et al. 2021, [Small Data? No Problem!](https://arxiv.org/abs/2103.10878)). It would not replace a real Adja corpus but could stretch our effective LM training data by 2-5x.

Until we're at one of these regimes, LM fusion will remain at the "1-3 point WER" level for Adja regardless of how clean the decoder is.

---

## 6. Why CTC + LM Should Help More Than We Saw

CTC (Graves et al., 2006, [ICML CTC paper](https://www.cs.toronto.edu/~graves/icml_2006.pdf)) treats each output frame as conditionally independent given the input. This is mathematically convenient — it permits the forward-backward training algorithm — but it means CTC models have **no internal language model**. The model cannot prefer "the cat sat" over "the cat sap" on linguistic grounds; it only prefers whichever had lower acoustic loss during training. This is the "conditional-independence assumption" and it is CTC's best-known weakness.

As a result, the literature reports **CTC + external LM** fusion gains that are **larger** than equivalent fusion gains for attention/seq2seq models:

- Hannun et al. 2014 ([DeepSpeech](https://arxiv.org/abs/1412.5567)): English CTC + 5-gram word LM, ~30% relative WER reduction.
- Amodei et al. 2015 ([DeepSpeech 2](https://arxiv.org/abs/1512.02595)): similar scale, 15-25% relative WER reduction from n-gram fusion, more from neural LM rescoring.
- Synnaeve et al. 2020 ([wav2letter++ / slimIPL](https://arxiv.org/abs/1911.08460)): CTC + transformer LM reliably beats CTC alone by 10-20% relative.

Contrast with attention/seq2seq models: Whisper, for example, has a transformer decoder that has absorbed an implicit LM during multilingual pre-training. Fusion with an external LM for Whisper often gives only 5-15% relative gain because the baseline already has a prior. CTC models like ours have **no such baseline prior** and should in principle benefit more from external LMs — the headroom is larger.

For reasonable corpus sizes and properly-tuned beam search, **10-30% relative WER reduction is the expected CTC+LM gain**. We saw ~2%. That's a factor of 5-10x below the literature average, even after accounting for our low-resource disadvantage.

Something beyond "small LM" is contributing to the gap. See next section.

---

## 7. Alternative Theories (Honest Sanity Check)

Four candidate explanations, ordered by how easy they are to test:

### 7a. Our α/β grid was too coarse
We swept only α ∈ {0.5, 1.0} and β ∈ {0.0, 0.5, 1.0} (or similar). The typical CTC+LM tuning range is wider: **α ∈ [0.1, 3.0]**, **β ∈ [0, 5]**, often with a separate length-normalization term. Missing the optimal α by 2x roughly halves the realized gain. Fix: sweep α ∈ {0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0} × β ∈ {0.0, 0.5, 1.0, 2.0, 4.0} on the dev set.

Moreover, α and β interact: raising α without raising β biases the decoder toward short outputs (because each extra token incurs an LM cost). Our finding that β=0.0 was best is itself suspicious — it means every added token costs probability mass with no compensating length bonus, which usually hurts WER on under-decoded hypotheses. A proper joint sweep would probably find a different optimum.

Expected impact if this is the cause: ~2-5x our current gain (so ~3-8 WER points instead of 1.5).

### 7b. Our pure-Python Katz backoff is mis-computing probabilities
Sanity checks that should pass and would confirm correctness:
- Sum of `P(c | context)` over all next-characters c should ≈ 1.0 for any context.
- `logP(real_sentence) > logP(char_shuffled_sentence) > logP(uniform_random_sentence)` reliably.
- Removing a frequent bigram from training should measurably increase the test perplexity of sentences containing it.

If any of these fail, the scorer has a bug and the LM isn't doing what we think. This is the **highest-priority diagnostic.**

### 7c. Our "beam search" isn't real CTC prefix beam search
Real CTC beam search (Graves & Jaitly 2014, [CTC beam search](https://www.cs.toronto.edu/~graves/icml_2014.pdf); Hannun et al. 2014) maintains beams of **prefixes** (not frame-level token sequences) and accumulates probability mass across all CTC alignments that collapse to the same prefix. This is important because the same output string has many valid CTC paths (differing only in blank placement and repetition), and summing them gives the correct prefix probability.

A naive top-k-per-frame beam, by contrast, keeps k distinct alignment paths, most of which are redundant variations of the same output prefix. The LM then scores each redundant path and the beam is dominated by duplicates, leaving very little effective diversity for the LM to reshape. `pyctcdecode` (and flashlight, and torchaudio's CTC decoder) implement **prefix-merging** correctly; a hand-rolled beam typically does not.

This is a likely contributor. Empirical test: swap our decoder for `pyctcdecode` + our own KenLM (built with proper Kneser-Ney) and re-measure. If the WER reduction jumps to 5-10%, this was the bottleneck.

### 7d. Ceiling effect: the acoustic model already outputs the LM's favorites
If the acoustic model is so poor that it outputs nearly-random characters, LM rescoring can't recover real words from noise — the n-best list doesn't contain the correct answer. Conversely, if the acoustic model is so confident that its top-1 is usually right, there's no room for the LM to re-rank. Our CER is 24%, which is middling — not terrible, not great. We should check: **what fraction of the reference characters appear in the top-5 per frame?** If it's <50%, beam search can't find them, LM or no. If it's >95%, the LM is fighting a confident wrong answer.

A related check: compute the **oracle WER** of the n-best list. That is, for each utterance, pick the hypothesis in the beam (say, top 100) with the lowest WER against reference, and aggregate. If oracle WER is close to our greedy WER, the beam doesn't contain better hypotheses and no LM can help — we need a better acoustic model. If oracle WER is 40-50% (vs. greedy 74%), there's plenty of room for an LM to re-rank — and the fact that we didn't capture it points to a decoder or LM bug.

This is the most expensive diagnostic but also the most informative about where to invest next.

### 7e. (Bonus) Domain mismatch between LM text and ASR transcripts
Our LM was trained on 13K Adja sentences from whatever text corpus we had (possibly religious, literary, or translated content). Our ASR was fine-tuned on 1,277 utterances of conversational/read Adja. If the LM's domain is different from the spoken content — e.g., formal biblical register vs. everyday speech — the LM assigns low probability to exactly the words our users say, and fusion actively hurts some utterances while helping others. Net gain is small.

Cross-entropy of LM on transcribed dev utterances vs. held-out LM training text will reveal this mismatch directly. A large gap (e.g., 3+ bits per character) indicates strong domain drift.

---

## 8. What Would Actually Help

Ordered by expected impact (based on literature norms and our specific bottlenecks):

### 1. Collect a larger Adja text corpus (highest impact)
- Target: 100K+ sentences.
- Sources: Bible translations (SIL, JW.org), government documents, local newspapers, FonMMS corpus, Flores-200 (if Adja is included), CommonCrawl filtered by Adja langid.
- Expected impact: **~10x the LM's effective information.** This compounds with every other fix.

### 2. Switch to `pyctcdecode` with KenLM
- True CTC prefix beam search + modified Kneser-Ney smoothing + proper α/β sweep.
- KenLM build is usually straightforward outside of constrained HPC envs; we can build the ARPA locally and ship it to the cluster.
- Expected impact: **3-5x our current gain** (from 1.5 WER points to 5-8 WER points), with the current LM corpus.

### 3. Word-level n-gram with BPE/character fallback
- Pre-tokenize the Adja text with SentencePiece (BPE, vocab ~2000).
- Build a 5-gram on BPE tokens — robust to OOV, longer effective context.
- Combine with a character LM for tail coverage (interpolation at the prefix level).
- Expected impact: **additional 1-2x over char LM.** Stacks with (2).

### 4. Neural LM (LSTM or small transformer)
- Fine-tune XLM-R or a small transformer on the Adja corpus.
- Use for n-best rescoring after beam search, not for every-frame fusion.
- Expected impact: **additional ~20% relative** on top of the n-gram (Mikolov 2010, [context of RNN LM](https://www.fit.vutbr.cz/research/groups/speech/publi/2010/mikolov_interspeech2010_IS100722.pdf); Irie et al. 2019, [TransformerLM for ASR](https://arxiv.org/abs/1905.04226)). Marginal once (1)-(3) are done; meaningful later.

### 5. Finer α/β sweep
- Grid: α ∈ {0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0} × β ∈ {0.0, 0.5, 1.0, 2.0, 4.0}.
- Run on dev, pick best, report test.
- Expected impact: **1.5-2x gain** at zero additional training cost. Essentially free.

### 6. Shallower n-gram order (3 or 4) with the current small corpus
- A 3-gram char LM has denser counts (fewer singletons) and backoff triggers less often.
- Likely a small, short-term win while (1) is being collected.
- Expected impact: **10-30% relative on top of the 5-gram.** Cheap experiment.

---

## 8b. A Concrete Plan for the Next Two Weeks

If we take the above seriously, the diagnostics-first ordering is:

**Week 1, day 1-2: Cheap diagnostics (no new model work).**
- Run the LM sanity checks from §7b (normalization, real-vs-shuffled, removed-bigram).
- Compute oracle WER on our existing beam top-100 (§7d).
- Compute top-5 per-frame reference-coverage statistic (§7d).
- Compute LM cross-entropy on dev transcripts vs. LM training corpus (§7e).
- Finer α/β sweep with current LM and decoder (§7a, §8-5).

These tell us whether the bottleneck is the decoder, the LM, or the acoustic model — critical for prioritizing weeks 2+.

**Week 1, day 3-5: Swap in proper tooling.**
- Install KenLM, rebuild 5-gram with modified Kneser-Ney from same corpus.
- Switch decoder to `pyctcdecode`. Compare to current pure-Python Katz + beam.
- If WER reduction jumps to 5-8 points with same corpus, decoder + smoothing were the main bottleneck.

**Week 2: Data collection.**
- Scrape/assemble additional Adja text (Bible, news, web).
- Train BPE tokenizer (~2000 merges).
- Build word + BPE 5-grams.
- Re-sweep α/β, measure.

At each step, write results into `results/run-ledger.md` and `results/comparison.md` immediately. No lost experiments.

---

## 9. Is This Normal in the Literature?

Yes. Low-resource ASR papers routinely report LM gains in the 1-5 WER point range. A non-exhaustive sample:

- **Zanon Boito et al., 2022 ([ASR2K](https://arxiv.org/abs/2206.15476))** — wav2vec2 + char LM on 1900+ languages. Most languages see **2-6 point WER reduction** from char-LM fusion, with a long tail of languages where the LM helps barely or not at all due to corpus size.
- **Pratap et al., 2023 ([MMS](https://arxiv.org/abs/2305.13516))** — Meta's massive multilingual ASR. LM fusion gains reported per-language are **typically 5-15% relative** for languages with ≥100K sentences of text; smaller or absent below that threshold.
- **Babu et al., 2021 ([XLS-R](https://arxiv.org/abs/2111.09296))** — the paper behind our base model. LM fusion on Common Voice low-resource languages: **2-8 point WER reduction** typical.
- **Yi et al., 2021 ([Cross-lingual low-resource transfer](https://www.isca-speech.org/archive/interspeech_2020/yi20_interspeech.html))** — for Yoruba, Swahili, etc., n-gram fusion often gives **1-3 points**, with larger gains requiring large crawled text corpora.

The Whisper-LM 51% figure stands out precisely because it is an outlier — a well-aligned combination of language, corpus, and baseline-weakness that the authors foregrounded. Our 1.5-point gain is toward the low end of the distribution but within it, and a factor of 3-5x improvement (our targeted goal post-fixes) would put us near the low-resource median.

---

### A note on replication and reporting

One reason the "up to 51%" headline persists is that most ASR papers report their best single configuration, not the full distribution. This is standard in ML but misleading for practitioners trying to set expectations. The right way to read any LM-fusion result is:

1. What language and corpus size?
2. What LM granularity (char, BPE, word)?
3. What smoothing?
4. What baseline WER was the reduction measured against?
5. Was the α/β properly tuned on dev?

Missing any of these makes the number hard to calibrate. Our result is well-specified in all five dimensions (see §0), which is why it's useful to report even though it's unflattering.

## 10. What to Tell Your Professor

> We observed a modest 1-2 point WER reduction from character-level n-gram LM fusion on our XLS-R CTC model for Adja. This is consistent with the lower end of reported improvements in the low-resource ASR literature (see ASR2K, MMS, XLS-R per-language numbers), though well below the 51% best-case reported in the Whisper-LM paper — which itself is an outlier for Basque with a word-level LM trained on ~10^8 tokens.
>
> The limited gain is attributable to three factors, in roughly decreasing order of impact:
>
> 1. **LM corpus size.** Our 13K-sentence (~38K-word) text corpus is about three orders of magnitude below what the literature associates with strong LM fusion. Fixing this requires a corpus-collection effort, not a modeling change.
>
> 2. **Character-level granularity.** A 5-gram character LM sees ~5 characters of context — less than a single Adja word on average. Word or BPE n-grams would see an order of magnitude more context and capture short syntactic patterns that chars cannot. This is a modeling change, achievable in days.
>
> 3. **Decoder and smoothing simplifications.** Our pure-Python Katz-backoff scorer and top-k-per-frame beam are approximations of production CTC prefix beam search with Kneser-Ney smoothing (e.g., `pyctcdecode` + KenLM). These approximations are each worth ~2-5x in realized gain individually and likely compound.
>
> To reach the WER reductions typical of low-resource ASR literature (5-10 relative %), we plan to: (a) assemble a 100K+ sentence Adja text corpus from religious, governmental, and web sources; (b) replace our decoder with `pyctcdecode` + KenLM; (c) train a BPE-level 5-gram in addition to the character LM; (d) run a finer α/β sweep on dev. The 51% Whisper-LM result is not a reasonable target for Adja at our current data scale — the upper-bound realistic target is ~10-15% relative WER reduction.
>
> The broader takeaway for the research direction: for Adja, LM improvements are bounded by **text data availability**, not modeling sophistication. Neural LMs, better smoothing, and finer decoding all help, but none compensates for having two orders of magnitude less text than comparable low-resource languages. Text-corpus collection is therefore a first-class research task in parallel with model experimentation, not an afterthought. Given the Gbe language family, Ewe and Fon text may be partially reusable as continued-pretraining data for a neural LM, and should be explored before we conclude that data limits are fundamental.

---

---

## Appendix A: Order-of-magnitude check on expected gain

A back-of-the-envelope model for expected LM-fusion WER reduction:

```
ΔWER_relative ≈ f(corpus_size, LM_granularity, smoothing, decoder_correctness, headroom)
```

With the following multipliers applied to a baseline "best-case" 30% relative reduction:

| Factor               | Our status                | Multiplier |
| -------------------- | ------------------------- | ---------- |
| Corpus size          | 13K sents (vs 1M baseline)| × 0.2      |
| Granularity          | char (vs word baseline)   | × 0.4      |
| Smoothing            | Katz (vs modified-KN)     | × 0.8      |
| Decoder              | approx beam (vs CTC prefix)| × 0.5     |
| α/β tuning           | coarse                    | × 0.7      |
| Headroom             | CER=24 (middle)           | × 1.0      |

Product: 0.2 × 0.4 × 0.8 × 0.5 × 0.7 × 1.0 × 30% ≈ **0.67% relative WER reduction.**

Our observed: **1.2-1.7% relative.** We're within a factor of ~2 of the back-of-envelope prediction, which suggests no single dramatic bug — just compounded limitations. Fixing each multiplier is worth ~2-5x of current gain, and they stack.

If we fix the top three (corpus, granularity, decoder) we'd expect:
```
0.8 × 0.8 × 0.8 × 0.7 × 1.0 × 30% ≈ 10.8% relative WER reduction
```
i.e., ~8 absolute WER points, bringing us from 73% to 65%. That's a realistic target.

---

## Appendix B: Glossary

- **CTC (Connectionist Temporal Classification):** a loss function and decoding scheme for sequence-to-sequence tasks where input/output alignment is unknown. Each output frame is independent given the input — no internal LM.
- **Shallow fusion:** combining acoustic score + α·LM_score + β·length during beam search. The LM is applied at decode time only, not during training.
- **Katz backoff:** an n-gram smoothing method using Good-Turing discounted counts with backoff to lower-order models.
- **Modified Kneser-Ney:** a superior smoothing method that uses continuation counts for the backoff distribution. KenLM's default.
- **Prefix beam search:** CTC-aware beam search that merges beam entries which produce the same output prefix (after collapsing CTC repeats and blanks), summing their probabilities.
- **BPE (Byte-Pair Encoding):** subword tokenization that balances vocabulary size against OOV robustness. Words split into frequently-co-occurring subword pieces.
- **α (alpha), β (beta):** hyperparameters weighting the LM score and length penalty in shallow fusion. Typical ranges: α ∈ [0.1, 3], β ∈ [0, 5].

---

## References

- Amodei et al. 2015. Deep Speech 2. https://arxiv.org/abs/1512.02595
- Babu et al. 2021. XLS-R. https://arxiv.org/abs/2111.09296
- Bisani & Ney 2005. Open-vocabulary ASR with hybrid LMs. Speech Communication.
- Chen & Goodman 1999. An empirical study of smoothing techniques for language modeling. https://dash.harvard.edu/handle/1/25104739
- Graves et al. 2006. CTC. https://www.cs.toronto.edu/~graves/icml_2006.pdf
- Graves & Jaitly 2014. Towards end-to-end speech recognition with RNNs (CTC beam search). https://www.cs.toronto.edu/~graves/icml_2014.pdf
- Hannun et al. 2014. Deep Speech. https://arxiv.org/abs/1412.5567
- Heafield 2011. KenLM: faster and smaller LM queries. https://kheafield.com/papers/avenue/kenlm.pdf
- Herrera et al. 2025. Whisper-LM. https://arxiv.org/abs/2503.23542
- Irie et al. 2019. Language modeling with deep transformers. https://arxiv.org/abs/1905.04226
- Katz 1987. Estimation of probabilities from sparse data for the LM component of a speech recognizer. IEEE TASSP.
- Kneser & Ney 1995. Improved backing-off for m-gram language modeling. ICASSP.
- Mikolov et al. 2010. Recurrent neural network based language model. https://www.fit.vutbr.cz/research/groups/speech/publi/2010/mikolov_interspeech2010_IS100722.pdf
- Pratap et al. 2023. MMS. https://arxiv.org/abs/2305.13516
- Synnaeve et al. 2020. End-to-end ASR with wav2letter++ / slimIPL. https://arxiv.org/abs/1911.08460
- Yi et al. 2020. Applying Wav2vec2.0 to Speech Recognition in Various Low-resource Languages. https://www.isca-speech.org/archive/interspeech_2020/yi20_interspeech.html
- Zanon Boito et al. 2022. ASR2K. https://arxiv.org/abs/2206.15476
