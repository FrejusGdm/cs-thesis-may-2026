# Research Idea: Cross-lingual Tokenizer Transfer for Low-Resource TTS

## Status: Promising — needs literature review + prototype

## Core Hypothesis

Low-resource TTS models benefit from tokenizers trained on typologically related languages,
because better subword segmentation produces more consistent phoneme-to-audio mappings.

## The Problem

LLM-based TTS models (CSM, Orpheus, Llasa) use the backbone's tokenizer (usually Llama BPE).
Llama's tokenizer was trained on English-heavy web data. For low-resource languages like Adja:
- Special characters (ɛ, ɔ, ŋ, ɖ) get split into meaningless byte fragments
- Tone marks (é, è) don't get sensible subword boundaries
- The model must learn "random byte sequences → these specific sounds" — wasteful

## The Proposed Fix

Replace the Llama tokenizer with NLLB's SentencePiece tokenizer:
- Trained on 200+ languages including Gbe-family languages (Ewe, Fon)
- Much more linguistically meaningful subword units for Adja
- Tone marks and special characters get sensible tokenizations

## Why This Is Novel

1. Nobody has studied tokenizer choice for LLM-based TTS quality
2. The TTS community inherits whatever tokenizer the backbone LLM has
3. For high-resource languages (English, Chinese) this doesn't matter — the tokenizer works fine
4. For low-resource languages, tokenizer quality could be the bottleneck
5. This bridges NMT research (where tokenization is well-studied) with TTS research (where it's ignored)

## Implementation Plan

### Phase 1: Measure the problem
- Tokenize Adja text with Llama tokenizer vs NLLB tokenizer
- Compare: tokens per sentence, byte-to-token ratio, subword alignment with phonemes
- Show quantitatively that Llama fragments Adja text badly

### Phase 2: Architecture surgery
- Take CSM-1B (Llama 3.2-1B backbone + Mimi decoder)
- Replace the embedding layer to accept NLLB vocab (~256k tokens vs ~128k)
- Keep the Mimi audio decoder unchanged
- Re-initialize the new embeddings (random or from NLLB encoder)

### Phase 3: Ablation experiments

| Experiment | Tokenizer | Embedding init | Backbone | What it tests |
|-----------|-----------|---------------|----------|---------------|
| T1 (baseline) | Llama BPE | Original | CSM-1B | Baseline |
| T1-nllb-tok | NLLB SPM | Random | CSM-1B | Tokenizer effect alone |
| T1-nllb-emb | NLLB SPM | NLLB encoder | CSM-1B | Tokenizer + pretrained embeddings |
| T1-nllb-full | NLLB SPM | NLLB encoder | NLLB encoder → CSM decoder | Full cross-model transfer |

### Phase 4: Evaluation
- MOS (Mean Opinion Score) human eval on generated Adja speech
- Token-level analysis: which subwords produce better audio?
- Compare on multiple low-resource languages (Adja, Ewe, Fon) for generalizability

## Related Work to Investigate

- Tokenizer effects on multilingual NLP (lots of NMT work here)
- Cross-lingual transfer in speech (wav2vec2-XLSR, MMS)
- LLM-based TTS architectures (VALL-E, SoundStorm, CSM, Orpheus)
- Neural audio codecs (EnCodec, Mimi, DAC) — how they discretize audio
- Low-resource TTS approaches (few-shot voice cloning, transfer learning)

## Why It Builds on Our Strengths

- We already have NLLB fine-tuning experience (prior ACL paper)
- We know Adja tokenization issues intimately (aj_Latn custom token work)
- We have the Adja TTS dataset ready
- CSM baseline (T1) will be done by the time we start this

## Potential Paper Angles

1. **Workshop paper**: "Does Tokenization Matter for Low-Resource TTS? Evidence from Adja"
2. **Full paper**: "Cross-lingual Tokenizer Transfer for Neural Speech Synthesis in Under-Resourced Languages"
3. **System description**: If it works well, release a model + recipe for the community

## Open Questions

- Does the Mimi decoder care about the distribution of hidden states from the backbone?
  If NLLB encoder produces very different hidden state distributions, the decoder might need retraining.
- Would a simpler character-level tokenizer work just as well?
  (Adja text is short — character-level might be fine for TTS)
- Can we do this with LoRA only (freeze backbone, adapt embeddings + LoRA)?
  That would make it accessible on consumer GPUs.
