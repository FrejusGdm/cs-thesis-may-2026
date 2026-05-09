# Reading List: Cross-lingual Tokenizer Transfer for Low-Resource TTS

Companion to `nllb-tokenizer-for-tts.md`. Papers organized by topic, 2022-2025.

## 1. Tokenizer Choice / Effects on LLM-based TTS

- **Speak, Read and Prompt** — Kharitonov et al., 2023
  https://arxiv.org/abs/2302.03540
  Text tokenization interacts with speech generation quality in prompting-based TTS.

- **UniAudio** — Yang et al., 2023
  https://arxiv.org/abs/2310.00704
  Unified tokenization of text+audio; tokenizer design impacts cross-modal generation quality.

- **BASE TTS** — Lajszczak et al., 2024
  https://arxiv.org/abs/2402.08093
  At billion-parameter scale: BPE tokenizer choice and vocab size affect TTS naturalness.

- **CosyVoice** — Du et al., 2024
  https://arxiv.org/abs/2407.05407
  Multilingual BPE tokenizer effects on cross-lingual TTS synthesis.

## 2. Cross-lingual Transfer for Low-resource TTS

- **YourTTS** — Casanova et al., 2022
  https://arxiv.org/abs/2112.02418
  Zero-shot speaker transfer on VITS; fine-tuning on ~1h of new language works.

- **Voicebox** — Le et al., 2023
  https://arxiv.org/abs/2306.15687
  Flow-matching TTS trained multilingually; zero-shot cross-lingual synthesis.

- **XTTS** — Casanova et al., 2024
  https://arxiv.org/abs/2406.04904
  Extends cross-lingual TTS to 16 languages via shared acoustic space.

- **Meta-TTS** — Huang et al., 2022
  https://arxiv.org/abs/2111.04535
  MAML-based meta-learning for few-shot TTS adaptation to new languages.

## 3. NLLB / Multilingual Tokenizers in Speech Models

- **SeamlessM4T** — Barrault et al., 2023
  https://arxiv.org/abs/2308.11596
  Uses NLLB-style multilingual tokenizer inside a speech-text model; 100+ languages.

- **SeamlessM4T v2** — Seamless Communication et al., 2023
  https://arxiv.org/abs/2312.05187
  How shared multilingual text tokenization affects S2TT and TTS quality.

- **AudioPaLM** — Rubenstein et al., 2023
  https://arxiv.org/abs/2306.12925
  PaLM text tokenizer + audio tokens; multilingual vocab strongly affects speech generation.

## 4. LLM-based TTS Architectures

- **VALL-E** — Wang et al., 2023
  https://arxiv.org/abs/2301.02111
  Foundational LLM-based TTS using discrete audio codec tokens. AR+NAR paradigm.

- **SoundStorm** — Borsos et al., 2023
  https://arxiv.org/abs/2305.09636
  Non-autoregressive codec-based generation. Fast and high quality.

- **Llasa** — Ye et al., 2025
  https://arxiv.org/abs/2501.09166
  Single-codebook LLM TTS that scales simply; SOTA with minimal complexity.

- **Orpheus-TTS** — Canopylabs, 2025
  https://huggingface.co/canopylabs/orpheus-3b-0.1-pretrain
  LLaMA-3 finetuned for TTS with emotion tags. No arXiv paper yet.

- **CSM (Sesame)** — Sesame AI, 2025
  https://github.com/SesameAILabs/csm
  LLaMA-based speech generation with backbone+decoder. No arXiv paper yet.

## 5. Neural Audio Codecs

- **SoundStream** — Zeghidour et al., 2022
  https://arxiv.org/abs/2107.03312
  Foundational RVQ-based neural codec. Enables discrete speech tokens.

- **EnCodec** — Défossez et al., 2022
  https://arxiv.org/abs/2210.13438
  Meta's codec. Most widely used backend for VALL-E and LLM-TTS.

- **DAC** — Kumar et al., 2023
  https://arxiv.org/abs/2306.06546
  Improved RVQ codec with better codebook utilization.

- **Mimi** — Défossez et al. (Kyutai), 2024
  https://arxiv.org/abs/2410.00037
  Streaming codec combining semantic+acoustic tokens. Used by CSM.

## 6. Low-resource TTS

- **IMS-Toucan** — Lux et al., 2022
  https://arxiv.org/abs/2206.12229
  Purpose-built for low-resource tonal languages. Uses Glottolog language embeddings. **Highly relevant for Adja.**

- **TTS for Low-Resource via Massively Multilingual Pre-Training** — Xu et al., 2024
  https://arxiv.org/abs/2406.07515
  Pretrain on 1000+ languages, fine-tune on <10 min of target language. **Directly applicable.**

- **Matcha-TTS** — Mehta et al., 2024
  https://arxiv.org/abs/2309.03199
  Lightweight flow-matching TTS that trains well in low-data regimes.

- **UTTS** — Chen et al., 2023
  https://arxiv.org/abs/2306.09893
  Reduces supervised data requirements via self-supervised audio representations.

- **Low-Resource Expressive TTS via Data Augmentation** — Upadhyay & Rao, 2023
  https://arxiv.org/abs/2305.09191
  Augmentation strategies for <1h TTS training data.

## Priority Reading Order

1. **VALL-E** — understand the LLM-TTS paradigm
2. **Mimi** — understand what CSM's decoder actually does
3. **BASE TTS** — closest work on tokenizer effects for TTS
4. **SeamlessM4T** — how NLLB tokenizer is used in speech models
5. **IMS-Toucan** — most relevant low-resource tonal TTS work
6. **CosyVoice** — multilingual tokenizer for TTS
7. **Xu et al. 2024** — multilingual pretraining for low-resource TTS

## Key Gap (Our Opportunity)

Nobody has isolated the effect of text tokenizer choice on LLM-based TTS quality for low-resource languages.
BASE TTS touches this at scale for high-resource, but the low-resource angle is completely open.
