# 08 — Low-resource cross-lingual transfer

Last updated: **2026-04-21**

The thesis of this project, compressed to one page.

## The setting

Adja has:
- ~6k French↔Adja translation pairs (for NMT track)
- ~1.6h of Adja speech with transcripts (for ASR/TTS)
- No unlabeled Adja audio of meaningful size

English has effectively unlimited data in every modality. Most modern
speech/NMT models are trained primarily on English + 20-200 high-resource
languages. The gap between English and Adja data is ~6 orders of magnitude.

## The three transfer strategies we use

### 1. Same-family bridge (our headline hypothesis)
Pretrain on Ewe (Adja's nearest relative), then fine-tune on Adja. Ewe has
15k labeled + 183k unlabeled utterances via WaxalNLP — 100× our Adja data.
Shared phonology means the Ewe→Adja fine-tune should converge fast.

This is Track 1B (TTS Ewe bridge) and Track 2B (ASR Ewe bridge) in the
master plan. Evidence it works: our E4 result (Whisper → Ewe → Adja) hit
CER 24.9%, best for any fine-tune-only ASR result at the time.

### 2. Massively multilingual SSL
Pretrain on 128-1107 languages with contrastive SSL (XLS-R, MMS). Adja is
usually not in the pretrain set, but neighboring languages are, so the
encoder learns a shared phonetic representation space. Then CTC fine-tune on
Adja. This is Track 2D on HPC.

### 3. High-resource tonal donor
If the target language is tonal, start from an LM that's already seen tonal
text. Hence our T2-zh experiments (Mandarin Orpheus → Adja) — Mandarin gives
the LM tonal priors even though Mandarin and Adja are otherwise unrelated.

## What tends to work (empirically)

From our own experiments + literature:
- **SSL pretraining on related-language unlabeled audio**: consistent 20-40%
  relative WER reduction in low-resource settings.
- **Cross-lingual supervised fine-tune from a related language**: 10-30%
  relative reduction if the language is phonologically close.
- **Tonal-prior transfer**: less established in the literature; this project
  is partly trying to characterize it.
- **Tokenizer expansion** (add new chars + resize embeddings): a prerequisite,
  not an accelerator. If the tokenizer is broken, nothing else matters.

## What tends NOT to work

- From-scratch baselines (B2/B3/B4): almost always lose to fine-tuned
  pretrained models for languages in our data regime. We deprioritized them.
- Few-shot in-context learning with general-purpose LLMs (Omni ICL k=1/3/10
  experiments): got CER 284% / 311% / 511% — much worse than fine-tuning.
- Massive-model zero-shot (Whisper-large, Omni 7B zero-shot): hallucination-prone
  on truly unseen languages. Need at least some supervised signal.

## Where the literature is thin

- **TTS for low-resource tonal languages**: very few published results. Our
  T3-Spark was arguably the first intelligible Adja TTS.
- **Audio-LM pretraining for cross-lingual transfer**: AudioLM-style pretraining
  on codec tokens is well-studied for high-resource data, barely studied for
  low-resource.
- **Gbe-family specifically**: only a handful of ML papers exist (Google AfriTTS,
  a few MMS sub-papers). Lots of room for contribution.

## Papers to read

- NLLB team, "No Language Left Behind" — https://arxiv.org/abs/2207.04672 —
  the giant survey of low-resource MT techniques.
- Pratap et al., "MMS" — https://arxiv.org/abs/2305.13516 — see §4 (data
  collection for 1000+ languages) for the blueprint.
- Hsu et al., "HuBERT" — https://arxiv.org/abs/2106.07447 — alternative SSL
  approach, often better than wav2vec 2.0 for low-resource.
- Ogueji, Zhu, Lin, "Small Data? No Problem! Exploring the Viability of
  Pretrained Multilingual LMs for Low-Resourced Languages" — https://aclanthology.org/2021.mrl-1.11/

## Related files

- Master plan: `../../experiments/asr-tts-getting-right-2026-04-21.md`
- Experiments sorted by track: `../../experiments/registry.md`
