# Can I Train a Direct Speech-to-Speech Model for Adja?

Written 2026-05-04 to answer the question: *"Can I train a speech-to-speech model
for Adja with what I have?"* The short answer is no. The full answer is below,
with the data-budget math and citations for every claim, so the verdict can be
checked by clicking through.

This document also handles a creative variant of the question: *"What if I split
Adja audio in half and have the model predict the second half from the first?"*
That has a name in the literature (generative spoken language modeling) and lands
in the same place: not feasible at this data scale, for closely related reasons.

The cascade pipeline in [Chapter 6 of the thesis](../thesis-writing/cs-thesis-josue-2026/chapters/tex/06%20Pipeline.tex)
remains the right architecture. This doc is the source-of-truth justification
that thesis chapters 6 and 7 cite back to.

## TL;DR

| Question | Verdict | Why |
|---|---|---|
| Direct S2ST: train Adja audio → French audio end-to-end? | **Not feasible.** | 0 hours of paired Adja↔target speech vs. **~127 h** minimum paired source speech in the survey row for Translatotron 1 on Fisher (below), and much larger budgets for every other row. |
| Speech continuation / GSLM: predict the next chunk of Adja audio from the previous chunk? | **Not feasible at intelligible-output quality.** | 1.6 h monolingual Adja vs. a literature floor of 6,000 h ([GSLM, Lakhotia et al. 2021](https://arxiv.org/abs/2102.01192)) up to 60,000 h ([AudioLM, Borsos et al. 2022](https://arxiv.org/abs/2209.03143)). The current Adja TTS results in this project ([Sesame CSM, Orpheus, Qwen3-TTS in `experiments/tts/`](../experiments/tts/)) are the same paradigm with stronger conditioning, and they already produce unintelligible output. |
| Cascade S2S: ASR → MT → TTS, what the thesis is doing? | **Right call.** | It is the only architecture that matches the data shape Adja actually has: monolingual paired audio↔text and monolingual text. |

The rest of this doc is the math behind these three rows.

## What "speech-to-speech" can mean

The phrase collapses four distinct problems with different data needs. Pinning
down which one I am answering is the first step.

1. **Speech-to-speech translation (S2ST)**. Adja audio in, French audio out.
   This is what most people mean by "speech-to-speech model." It needs paired
   translation speech, which Adja does not have.
2. **Speech continuation, also called generative spoken language modeling
   (GSLM)**. Adja audio in, more Adja audio out, in the same language. This is
   the textless-NLP paradigm. It needs only monolingual audio, which makes it
   sound like the cheap option, but it does not produce intelligible speech
   without large-scale pretraining.
3. **Voice conversion**. Speaker A's Adja, in speaker B's voice. Different
   problem. Out of scope here.
4. **Speech editing or inpainting**. Mask part of an utterance, predict the
   missing audio. Different problem. Out of scope here.

This doc treats (1) and (2) seriously, names (3) and (4) for completeness,
and concludes that both (1) and (2) are out of reach at Adja's data scale.

## Architecture taxonomy

For (1), direct S2ST, the canonical line of work as of 2026:

- **Translatotron 1** ([Jia et al. 2019, arXiv:1904.06037](https://arxiv.org/abs/1904.06037)): direct attention-based spectrogram-to-spectrogram.
- **Translatotron 2** ([Jia et al. 2022, arXiv:2107.08661](https://arxiv.org/abs/2107.08661)): adds phoneme auxiliary supervision.
- **Translatotron 3** ([Nachmani et al. 2023, arXiv:2305.17547](https://arxiv.org/abs/2305.17547)): unsupervised via back-translation.
- **UnitY** ([Inaguma et al. 2023, ACL](https://aclanthology.org/2023.acl-long.872/)): two-pass discrete-unit S2ST.
- **UnitY2 / SeamlessM4T v2** ([Barrault et al. 2023, arXiv:2312.05187](https://arxiv.org/abs/2312.05187)): non-autoregressive unit decoding, multilingual.
- **AudioPaLM** ([Rubenstein et al. 2023, arXiv:2306.12925](https://arxiv.org/abs/2306.12925)): audio tokens fed into a decoder LM.
- **Direct discrete-unit S2ST** ([Lee et al. 2022, arXiv:2107.05604](https://arxiv.org/abs/2107.05604)): fairseq HuBERT units.

For (2), generative spoken LMs:

- **GSLM** ([Lakhotia et al. 2021, arXiv:2102.01192](https://arxiv.org/abs/2102.01192)): discretize speech with HuBERT or CPC, train a unit LM.
- **AudioLM** ([Borsos et al. 2022, arXiv:2209.03143](https://arxiv.org/abs/2209.03143)): hierarchical semantic + acoustic tokens.
- **Slam** ([2025, arXiv:2502.15814](https://arxiv.org/abs/2502.15814)): efficient speech LM, "efficient" in compute, not data.

A 2025 review ([Direct S2ST review, arXiv:2503.04799](https://arxiv.org/html/2503.04799v1))
covers the direct S2ST landscape end-to-end and is the cleanest single citation
for "the field has tried this and it is hard."

## Data each architecture was trained on

This is the table that decides the question. Every number traces to a citation.

| System | Paired speech | Unlabeled pretraining | Notes |
|---|---|---|---|
| Translatotron 1 ([arXiv:1904.06037](https://arxiv.org/abs/1904.06037)) | Fisher Es-En **127 h source / 96 h synthesized target** | none | The "1,400 h" number sometimes cited is an internal Google "Conversational" dataset, not Fisher. |
| Translatotron 2 ([arXiv:2107.08661](https://arxiv.org/abs/2107.08661)) | CoVoST 2 **476 h source / 296 h synthesized target** | none | Synthetic target via TTS. |
| Translatotron 3 ([arXiv:2305.17547](https://arxiv.org/abs/2305.17547)) | Common Voice 11 (Es, En), monolingual only, exact hours not reported in paper | strong pretrained substrate | Unsupervised back-translation. |
| UnitY ([ACL 2023](https://aclanthology.org/2023.acl-long.872/)) | **CVSS-C ~1,900 h** (21 source languages → English) | none | [CVSS, arXiv:2201.03713](https://arxiv.org/abs/2201.03713). |
| SeamlessM4T v2 ([arXiv:2312.05187](https://arxiv.org/abs/2312.05187)) | **351 k h S2TT, 145 k h S2ST**, plus **+114.8 k h** SeamlessAlign v2 | **4.5 M h** w2v-BERT 2.0 unlabeled | The "470 k" number floating around is from [v1 mined paired, arXiv:2308.11596](https://arxiv.org/abs/2308.11596). |
| AudioPaLM ([arXiv:2306.12925](https://arxiv.org/abs/2306.12925)) | **~38.4 k h** total (AST + S2ST + WMT/TED-TTS + PaLM-MT-TTS) | PaLM-2 8B substrate | |
| GSLM ([arXiv:2102.01192](https://arxiv.org/abs/2102.01192)) | none (textless) | **6 k h Libri-Light** (CPC) up to **60 k h** with wav2vec 2.0 | |
| AudioLM ([arXiv:2209.03143](https://arxiv.org/abs/2209.03143)) | none | **60 k h Libri-Light** | |
| Slam ([arXiv:2502.15814](https://arxiv.org/abs/2502.15814)) | none | **~81 k h** (LibriSpeech 960h + Libri-Light 50k + sTinyStories 30k) | "Efficient" means compute-efficient, not data-efficient. |
| Hokkien S2ST ([Meta 2022, arXiv:2211.06474](https://arxiv.org/abs/2211.06474)) | **8,000+ h** mined Hokkien + 2 h annotated TAT + Mandarin pivot | none | Closest published "low-resource" S2ST. Three orders of magnitude above what Adja has. |

One footnote that matters for the verdict: every direct S2ST benchmark
(CVSS, Translatotron evals) trains on **TTS-synthesized target speech**, not
natural target audio ([CVSS, arXiv:2201.03713](https://arxiv.org/abs/2201.03713)).
The field works around the natural-natural data shortage by synthesizing one
side. That option is closed to Adja right now because the synthesizer for the
target side (Adja TTS) does not yet produce intelligible speech.

## What Adja has

Verified from this repo, not assumed.

- **1.57 hours** of paired Adja audio↔Adja text. From the
  [`JosueG/adja-tts-orpheus`](https://huggingface.co/datasets/JosueG/adja-tts-orpheus)
  dataset. 1,277 utterances train, 160 dev, 160 test, 48 kHz, ~3.4 s average
  duration. Splits via [`experiments/asr/shared/data_prep.py`](../experiments/asr/shared/data_prep.py).
- **0 hours** of paired Adja↔French speech, or any other language pair.
- **12 k monolingual Adja text lines** in [`data/extra_adja_text.txt`](../data/extra_adja_text.txt).
  No parallel text in this repo.
- **One usable ASR checkpoint**: XLS-R 300M C4v2 at 70.76% WER with a language
  model. Useful as a cascade encoder. Not enough to bootstrap direct S2ST.
- **TTS scaffolding**: Sesame CSM, Orpheus, Qwen3-TTS, MMS-Ewe in
  [`experiments/tts/`](../experiments/tts/). None currently produce intelligible
  Adja audio. Loss converges, audio does not.

The binding constraint for any system that *outputs* Adja audio is the same
constraint Track 4 is fighting: generating intelligible Adja from ~1.6 h of
data. Adding a translation step in front of it does not relax that constraint.

## Gap analysis

Five honest scenarios, with the verified denominators above.

1. **From-scratch direct S2ST.** Adja has 0 h paired Adja↔target speech. The
   smallest paired set in the table above is Translatotron 1's 127 h. The gap
   is unbounded in the strict sense (you cannot take the log of zero).

2. **Adapter fine-tune of SeamlessM4T v2.** This still requires paired
   Adja↔target speech, which the project does not have. SeamlessM4T v2's nearest
   Niger-Congo language is Yoruba, and Yoruba in v2 is **speech-input only,
   no Yoruba speech output** ([model card](https://huggingface.co/facebook/seamless-m4t-v2-large)).
   The closest neighbor of Adja in the foundation model already forces a
   cascade for Yoruba audio output. Adapter fine-tuning will not change that.

3. **Translatotron-3-style unsupervised back-translation.** Needs a strong
   pretrained substrate that already covers the source and target language
   phonotactics. None covers Adja. The recipe is also brittle in the
   resource-rich Es↔En setting it was demonstrated on, and there is no
   evidence it scales down to a language pair where neither side is in the
   pretraining data.

4. **Translatotron‑1‑scale paired speech.** The smallest paired-speech budget
   in the citation-backed table above is still **~127 h** of source speech on
   Fisher (plus **~96 h** of synthesized target speech), with no path to
   reproduce that recipe for Adja↔French today. Adja has **0 h** paired
   speech—below that floor and below benchmarks such as IWSLT's low-resource
   track, which assumes **tens to hundreds of hours** of paired data.

5. **Speech continuation, the user's original idea.** GSLM and AudioLM define
   the floor at 6,000 to 60,000 hours of monolingual speech. Adja has 1.6 h.
   That is **3.5 to 4.5 orders of magnitude short** of the regime where the
   model produces intelligible output. Pretrained-model fine-tuning lowers the
   bar for speech *understanding* tasks ([Yang et al. 2025, arXiv:2508.05149](https://arxiv.org/html/2508.05149)
   shows tens of hours can move ASR WER), but not for speech *generation*. No
   published audio-LM result on under 5 h of monolingual data produces
   intelligible output.

## Could anything close the gap?

I considered four ways to narrow the gap and ruled each out.

- **Multilingual Gbe transfer** through SeamlessM4T's Ewe and Fon adapters or
  through MMS Gbe-family pretraining. Narrows the gap, does not close it. The
  speech-output side still has no Adja and no Adja phonotactics in any
  pretrained model.
- **Synthetic parallel data via TTS-generated bitext.** Train MT, then use
  Adja TTS to synthesize the target side, then train direct S2ST on the
  synthetic pairs. This is exactly what CVSS does for Es↔En. For Adja the
  approach is chicken-and-egg: it needs a working Adja TTS, which is the open
  problem in Track 4 of this project.
- **Cross-lingual unit transfer.** Train HuBERT or XLS-R on related Gbe
  languages (Ewe, Fon), apply the discrete units to Adja audio. Promising in
  principle, untested in this project, and still does not address the
  generation side.
- **Cascade with a neural vocoder backend** (the `S2` track in
  [`adja-speech-architecture-exploration-plan.md`](../adja-speech-architecture-exploration-plan.md)).
  This is just cascade with TTS at the end. It is what the thesis already does.

## Where the field draws the low-resource line

A few load-bearing reference points, all clickable.

- The smallest **paired-speech** budget in the table above is Translatotron 1
  on Fisher: **~127 h** source / **~96 h** synthesized target. Every other
  direct system in that table is larger; several assume mined or TTS target
  speech at scale.
- **No Gbe-family direct S2ST** has been published, ever. The 2025 review
  ([arXiv:2503.04799](https://arxiv.org/html/2503.04799v1)) catalogs the
  landscape. Fongbe shows up as a speech-to-text task in survey work, never
  as a direct S2ST language.
- **SeamlessM4T v2** covers Yoruba as speech input, but the model does not
  produce Yoruba speech output ([model card](https://huggingface.co/facebook/seamless-m4t-v2-large)).
  The set of African languages supported as speech-out is Afrikaans, Amharic,
  Arabic variants, Swahili, Igbo, Fulfulde, Oromo, Luo, Nyanja, Shona, Somali,
  Xhosa, Zulu, Kamba, and Ganda. No Gbe family.
- **MMS** scaled to 1,107 languages with **44.7 k h paired audio-text**
  ([JMLR 2024](https://jmlr.org/papers/v25/23-1318/23-1318.pdf)). Adja is not
  in MMS.
- **IWSLT 2025 low-resource track** ([page](https://iwslt.org/2025/low-resource))
  defines low-resource as tens to hundreds of hours of paired data. Adja is
  below the field's working definition of low-resource.
- **Speech LM scaling laws** ([Cuervo and Marxer 2024, arXiv:2404.00685](https://arxiv.org/abs/2404.00685))
  estimate that speech-LM linguistic ability scales **roughly three orders of
  magnitude slower** than text LLM ability with respect to data. Data-starved
  speech generation is a quantifiably harder problem than data-starved text
  generation.

## Verdict

Direct S2ST for Adja: **not feasible** with current data. Even the
smallest paired-speech row in the survey table (Translatotron 1) is two
orders of magnitude above what exists for Adja↔French paired speech (0 h).

Speech continuation / GSLM-style models on monolingual Adja:
**not feasible at intelligible-output quality** with current data. The
empirical evidence is already in this repo: the TTS experiments in
[`experiments/tts/`](../experiments/tts/) are the same paradigm with stronger
conditioning, and they fail to produce intelligible audio on the same 1.6 h
budget. Removing the text conditioning makes the task harder, not easier.

Cascade S2S (ASR → MT → TTS): **the right architecture.** It is the only one
that matches the data shape Adja actually has. This is what
[Chapter 6 of the thesis](../thesis-writing/cs-thesis-josue-2026/chapters/tex/06%20Pipeline.tex)
already builds toward. The forward-looking subsection of
[Chapter 7](../thesis-writing/cs-thesis-josue-2026/chapters/tex/07%20Discussion.tex)
("What I'd do with another year") describes the data collection that would
unlock the direct alternatives in a future project.

## Optional appendix experiment (out of scope here, flagged for the thesis)

A zero-shot SeamlessM4T v2 inference run on a small sample of Adja test audio
would produce a citable empirical anchor for the "direct does not work out of
the box" claim. Inference-only, no training. The model has no Adja and no
Adja-related Gbe outputs in its supported set, so it should fail in
characteristic ways (route to a phonologically nearby language, hallucinate,
or output silence). Documenting that failure with three to five example
transcriptions would be a tight appendix item. Not done in this doc.

## References

Grouped by topic. Every entry is clickable.

### Direct speech-to-speech translation

- Jia et al. 2019, *Direct Speech-to-Speech Translation with a Sequence-to-Sequence Model* (Translatotron 1): [arXiv:1904.06037](https://arxiv.org/abs/1904.06037)
- Jia et al. 2022, *Translatotron 2: High-quality direct speech-to-speech translation with voice preservation*. [arXiv:2107.08661](https://arxiv.org/abs/2107.08661)
- Nachmani et al. 2023, *Translatotron 3: Speech to Speech Translation with Monolingual Data*. [arXiv:2305.17547](https://arxiv.org/abs/2305.17547)
- Inaguma et al. 2023, *UnitY: Two-pass Direct Speech-to-speech Translation with Discrete Units*, ACL. [aclanthology.org/2023.acl-long.872](https://aclanthology.org/2023.acl-long.872/)
- Barrault et al. 2023, *Seamless: Multilingual Expressive and Streaming Speech Translation* (SeamlessM4T v2): [arXiv:2312.05187](https://arxiv.org/abs/2312.05187)
- SeamlessCommunication et al. 2023, *SeamlessM4T: Massively Multilingual & Multimodal Machine Translation* (v1): [arXiv:2308.11596](https://arxiv.org/abs/2308.11596)
- SeamlessCommunication et al. 2025, joint speech and text translation. [Nature 2025](https://www.nature.com/articles/s41586-024-08359-z)
- Rubenstein et al. 2023, *AudioPaLM*. [arXiv:2306.12925](https://arxiv.org/abs/2306.12925)
- Lee et al. 2022, *Direct speech-to-speech translation with discrete units*. [arXiv:2107.05604](https://arxiv.org/abs/2107.05604)
- Jia et al. 2022, *CVSS Corpus and Massively Multilingual Speech-to-Speech Translation*. [arXiv:2201.03713](https://arxiv.org/abs/2201.03713)
- 2025 review of direct S2ST. [arXiv:2503.04799](https://arxiv.org/html/2503.04799v1)
- Translatotron-style review 2025. [arXiv:2502.05980](https://arxiv.org/abs/2502.05980)
- SeamlessM4T v2 model card. [huggingface.co/facebook/seamless-m4t-v2-large](https://huggingface.co/facebook/seamless-m4t-v2-large)

### Generative spoken language modeling

- Lakhotia et al. 2021, *Generative Spoken Language Modeling from Raw Audio* (GSLM): [arXiv:2102.01192](https://arxiv.org/abs/2102.01192)
- Borsos et al. 2022, *AudioLM: a Language Modeling Approach to Audio Generation*. [arXiv:2209.03143](https://arxiv.org/abs/2209.03143)
- *Slam* 2025, efficient speech language models. [arXiv:2502.15814](https://arxiv.org/abs/2502.15814)
- Cuervo and Marxer 2024, *Scaling Properties of Speech Language Models*. [arXiv:2404.00685](https://arxiv.org/abs/2404.00685)

### Low-resource S2ST and benchmarks

- Hokkien S2ST (Meta 2022): [arXiv:2211.06474](https://arxiv.org/abs/2211.06474)
- Yang et al. 2025, low-resource speech-LM fine-tuning. [arXiv:2508.05149](https://arxiv.org/html/2508.05149)
- IWSLT 2025 low-resource track. [iwslt.org/2025/low-resource](https://iwslt.org/2025/low-resource)
- FLEURS. [arXiv:2205.12446](https://arxiv.org/abs/2205.12446) and [HF page](https://huggingface.co/datasets/google/fleurs)
- CoVoST 2. [arXiv:2007.10310](https://arxiv.org/abs/2007.10310)
- AfriSpeech-200. [HF dataset page](https://huggingface.co/datasets/intronhealth/afrispeech-200)
- MMS. [JMLR 2024](https://jmlr.org/papers/v25/23-1318/23-1318.pdf)

### Adja-side data

- `JosueG/adja-tts-orpheus`. [HF dataset page](https://huggingface.co/datasets/JosueG/adja-tts-orpheus)
- Repo references: [`experiments/registry.md`](../experiments/registry.md), [`results/comparison.md`](../results/comparison.md), [`adja-speech-architecture-exploration-plan.md`](../adja-speech-architecture-exploration-plan.md), [`docs/multilingual-speech-strategy-2026-04-18.md`](multilingual-speech-strategy-2026-04-18.md).
