# Papers — annotated bibliography

Last updated: **2026-04-21**

Papers that directly inform what we are running, organized by area. When a
paper has a concept file in `../concepts/`, read the concept first.

## ASR (`asr/`)

- **Whisper** — Radford et al. 2022 — https://cdn.openai.com/papers/whisper.pdf
  Core of our C2, E4, E7 baselines. Read §3 (model) and §6 (multilingual).
- **wav2vec 2.0** — Baevski et al. 2020 — https://arxiv.org/abs/2006.11477
  SSL pretraining objective we run on HPC. Read §2 (model) and §3 (SSL pretraining).
- **XLS-R** — Babu et al. 2021 — https://arxiv.org/abs/2111.09296
  Massively multilingual wav2vec 2.0. 128 languages, 436k hours.
- **MMS** — Pratap et al. 2023 — https://arxiv.org/abs/2305.13516
  1107 languages. Ewe is included. Language adapters + single encoder.
- **CTC** — Graves et al. 2006 — https://www.cs.toronto.edu/~graves/icml_2006.pdf
  The loss function for all our non-Whisper ASR. Dense but short (8 pages).
- **HuBERT** — Hsu et al. 2021 — https://arxiv.org/abs/2106.07447
  Alternative SSL objective, often wins on low-resource. Worth trying.
- **SeamlessM4T v2** — Communication et al. 2023 — https://arxiv.org/abs/2312.05187
  Multilingual speech+text, 101 languages. C6 baseline.

## TTS (`tts/`)

- **Sesame CSM blog** — https://www.sesame.com/research/crossing_the_uncanny_valley_of_voice
  Our primary TTS target. No formal paper; blog + code is the reference.
- **Mimi codec** — Défossez et al. 2024 — https://arxiv.org/abs/2410.00037
  The codec CSM uses. Read §2-3 for the tokenization.
- **SNAC** — Siuzdak 2024 — https://github.com/hubertsiuzdak/snac
  Orpheus's codec. Repo README has the details.
- **Orpheus** — no paper, model cards at https://huggingface.co/canopylabs
  We test all 3 multilingual variants (EN/zh/fr).
- **Spark-TTS** — https://github.com/SparkAudio/Spark-TTS
  Our only working TTS result for Adja so far.
- **AudioLM** — Borsos et al. 2022 — https://arxiv.org/abs/2209.03143
  The paradigm behind LM-based TTS.
- **VALL-E** — Wang et al. 2023 — https://arxiv.org/abs/2301.02111
  First practical neural-codec TTS. Read §3 (the tokenization approach).
- **F5-TTS** — Chen et al. 2024 — https://arxiv.org/abs/2410.06885
  Flow-matching TTS. Our T10 experiment.

## Tokenization (`tokenization/`)

- **BPE** — Sennrich et al. 2015 — https://arxiv.org/abs/1508.07909
  Subword tokenization, used by Llama. Short and easy.
- **SentencePiece** — Kudo & Richardson 2018 — https://arxiv.org/abs/1808.06226
  The other widely-used family (NLLB, T5).
- **Byte-level BPE** — Radford et al. 2019 (GPT-2 paper) —
  https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf
  Why tokenizers fall back to bytes for OOV characters.

## Low-resource (`low-resource/`)

- **NLLB** — Team et al. 2022 — https://arxiv.org/abs/2207.04672
  No Language Left Behind. 200 languages.
- **African NLP survey** — Adelani et al. 2022 — https://arxiv.org/abs/2211.04529
  AfroLM. Good Gbe-adjacent context.
- **Small Data? No Problem!** — Ogueji et al. 2021 — https://aclanthology.org/2021.mrl-1.11/
  Practical recipe for low-resource LM.
- **LoRA** — Hu et al. 2021 — https://arxiv.org/abs/2106.09685
  Parameter-efficient fine-tuning. Short (13 pages), must-read.

## Surveys to skim when you have time

- Radhakrishnan, "Recent advances in end-to-end automatic speech recognition" (2023)
- Tan, Qin, et al., "A survey on neural speech synthesis" —
  https://arxiv.org/abs/2106.15561 — dated but comprehensive TTS survey.
