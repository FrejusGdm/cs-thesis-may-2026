# TTS Models Learning Guide

Reading list and key concepts for text-to-speech fine-tuning, especially for low-resource languages like Adja.

## Models to Learn About

### Sesame CSM (1B) — Primary Target
- **What**: Conversational Speech Model, generates natural speech with speaker conditioning
- **Architecture**: Llama 3.2-1B backbone + Mimi RVQ audio decoder
- **Key insight**: Base model (not fine-tuned on specific voices), needs reference audio at inference for speaker consistency
- **Audio codec**: Mimi at 24kHz, residual vector quantization
- **Paper/Blog**: https://www.sesame.com/research/crossing_the_uncanny_valley_of_voice
- **Code**: https://github.com/SesameAILabs/csm
- **HuggingFace**: https://huggingface.co/sesame/csm-1b
- **Unsloth notebook**: `references/unsloth-tts-notebooks/Sesame_CSM_1B_TTS.ipynb`

### Orpheus TTS (3B)
- **What**: Llama-based TTS model, fine-tuned on 8 professional voice actors
- **Architecture**: Llama-3.2-3B backbone + SNAC 24kHz codec (7 tokens/frame, 3 hierarchical codebooks)
- **Key insight**: Already fine-tuned = better voice consistency out of box, but 3x larger than CSM
- **Supports**: Emotion tags like `<laugh>`, `<sigh>` as special tokens
- **Multilingual variants** (April 2025 research release, all use same SNAC codec + same Llama backbone):
  - `3b-fr` — French
  - `3b-de` — German
  - `3b-zh` — **Mandarin Chinese (tonal! proves SNAC handles tones)**
  - `3b-hi` — Hindi
  - `3b-ko` — Korean
  - `3b-es_it` — Spanish + Italian (bilingual)
  - All checkpoints at `canopylabs/orpheus-3b-[lang]-pretrain-research_release`
- **HuggingFace**: https://huggingface.co/canopylabs/orpheus-tts-0.1-finetune-prod
- **Unsloth notebook**: `references/unsloth-tts-notebooks/Orpheus_3B_TTS.ipynb`

### Spark TTS (0.5B)
- **What**: Lightweight TTS model, only 500M parameters
- **Key insight**: Smallest viable TTS model — interesting for deployment on resource-constrained hardware
- **HuggingFace**: https://huggingface.co/SparkAudio/Spark-TTS-0.5B
- **Unsloth notebook**: `references/unsloth-tts-notebooks/Spark_TTS_0.5B.ipynb`

### Llasa TTS (1B / 3B)
- **What**: LLM-based speech synthesis
- **Architecture**: LLM backbone that directly generates audio codec tokens
- **Variants**: 1B (lighter) and 3B (heavier)
- **HuggingFace**: https://huggingface.co/HKUSTAudio/Llasa-1B
- **Unsloth notebooks**: `references/unsloth-tts-notebooks/Llasa_TTS_1B.ipynb`, `Llasa_TTS_3B.ipynb`

### OuteTTS (1B)
- **What**: Open-source TTS with voice cloning capabilities
- **Key insight**: Focus on voice cloning from short reference clips
- **HuggingFace**: https://huggingface.co/OuteAI/OuteTTS-1.0-1B
- **Unsloth notebook**: `references/unsloth-tts-notebooks/OuteTTS_1B.ipynb`

### Inworld TTS
- **What**: Open-source SpeechLM-based TTS with emotional control and zero-shot voice cloning
- **Architecture**: Dual-encoder — audio encoder tokenizes reference audio into discrete tokens, concatenated with text tokens for an autoregressive SpeechLM decoder
- **Key insight**: 48kHz output (matches our raw data!), supports 11 languages, fine-grained emotion control + non-verbal vocalizations. Training code included (pre-train, fine-tune, RL alignment).
- **Zero-shot cloning**: Only needs seconds of reference audio via in-context learning
- **Code**: https://github.com/inworld-ai/tts
- **Blog**: https://inworld.ai/blog/introducing-inworld-tts
- **Demo**: https://inworld-ai.github.io/tts/
- **Docs**: https://docs.inworld.ai/tts/tts

### Whisper Large V3 (STT — related)
- **What**: OpenAI's multilingual speech recognition model
- **Why relevant**: Our ASR experiments (C2) already use Whisper; the Unsloth notebook shows their optimized fine-tuning approach
- **Paper**: https://cdn.openai.com/papers/whisper.pdf
- **Unsloth notebook**: `references/unsloth-tts-notebooks/Whisper_STT.ipynb`

## Key Concepts to Understand

---

### The Three-Layer Architecture of Modern LLM-Based TTS

Every modern LLM-based TTS model (CSM, Orpheus, Spark, Llasa, etc.) is built from three distinct layers. Understanding which layer does what is critical for debugging failures and planning fine-tuning.

```
TEXT INPUT  →  [1. Text Tokenizer]  →  [2. LM Backbone]  →  [3. Audio Codec]  →  AUDIO
                 "how do I chop       LLM predicts next     Codec decodes
                  this text into       audio token given     token IDs back
                  token IDs?"          text token IDs"       to waveform"
```

#### Layer 1: Text Tokenizer
Converts input text (e.g. "Ɛnyi wɛ dze") into integer token IDs fed to the LM.

- **What it must handle**: every character in your target language's orthography
- **Failure mode**: characters not in the vocabulary get split into **byte fallback tokens** (e.g., `ɛ` → `[0xc9, 0x9b]`). The LM then has to learn that these two bytes together represent the phoneme /ɛ/. With 1.7h of data this never converges.
- **Llama 3.2 BPE** (used by CSM and Orpheus): trained on English-dominant text. Adja diacritics (ɛ, ɔ, ŋ, ɖ, tone marks) fall back to bytes. This is likely the primary reason CSM/Orpheus failed on Adja.
- **Qwen2 BPE** (used by Spark): trained on massively multilingual text including African and Asian scripts. Adja diacritics are native tokens — no byte fragmentation.
- **Character-level** (used by MMS-TTS, IMS-Toucan): no tokenizer at all — each character maps directly to a phoneme. Completely immune to OOV issues.
- **SentencePiece with NLLB vocab** (under investigation): trained on 200 languages including many African ones. Likely handles Adja characters natively.

**Practical implication**: before fine-tuning any model on Adja, tokenize a sample sentence and inspect the output. If you see byte tokens (like `▁Ã`, `©`, or raw hex), the tokenizer will bottleneck training.

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
print(tok.tokenize("ɛnyi wɛ dze è"))
# Bad:  ['â', 'ĩ', 'ŀ', 'ĩ', 'Ĵ', ...]  ← byte fragments
# Good: ['ɛnyi', 'wɛ', 'dze', 'è']       ← native tokens
```

---

#### Layer 2: LM Backbone
A standard autoregressive transformer (usually Llama or Qwen) that takes interleaved text+audio token IDs and predicts the next audio token at each step.

- **What it learns**: the mapping from text token sequences → audio token sequences, including prosody, rhythm, and tone
- **Fine-tuning this layer** is what LoRA targets — you're teaching the LM a new language's text→audio mapping
- **Transfer**: If a related language was in pretraining, the LM already has some phonological priors. English-pretrained LMs have no Gbe-family prior.
- **Capacity is NOT the bottleneck** (confirmed by our experiments): CSM at full fine-tune (1.6B params) got dev loss ≈ LoRA r=32 dev loss. The architecture was the problem, not the number of trainable parameters.

---

#### Layer 3: Audio Codec
Converts between raw waveform audio and discrete integer token sequences. This is the "vocabulary" of audio.

- **What it must handle**: faithfully encode and reconstruct the acoustic features of your target language — including tones, tonal sandhi, ATR vowel distinctions, nasal vowels, etc.
- **Failure mode**: if the codec was trained only on English, it may **quantize away** phonemic distinctions that don't appear in English (e.g., high/low tone on the same vowel). The LM then generates valid-looking tokens that decode to phonologically ambiguous audio.
- **How to test**: run `codec.encode(wav)` then `codec.decode(tokens)` on Adja audio and **listen to the reconstruction**. If the reconstruction is clear, the codec is fine. If it sounds smeared or loses tonal quality, the codec is the bottleneck.

**Codec comparison across our models:**

| Codec | Used by | Codebooks | Frame rate | Language-agnostic? | Tonal language support |
|---|---|---|---|---|---|
| **Mimi** (Kyutai) | CSM | 32 (1 semantic + 31 acoustic) | 12.5 Hz | Likely yes — waveform-level | Needs reconstruction test |
| **SNAC** (Siuzdak) | Orpheus (all variants) | 3 hierarchical | ~86 Hz | **Yes — confirmed by `3b-zh` (Mandarin)** | **Proven: Mandarin Orpheus works** |
| **BiCodec/XLSR-53** | Spark | Semantic (XLSR) + acoustic | — | Yes — XLSR-53 covers 53 langs | Strong — XLSR-53 includes African langs |
| **EnCodec** (Meta) | Llasa, others | 8 RVQ | 75 Hz | Largely yes — waveform-level | Moderate |

**Critical revision**: both Mimi and SNAC are **waveform-level acoustic codecs** — they quantize raw audio without language-specific priors. Tone (F0 contour), nasal vowels, ATR distinctions are all acoustic features encoded in the waveform and faithfully captured. SNAC is confirmed language-agnostic by the existence of a working **Mandarin Orpheus** variant — Mandarin has 4 tones, and `3b-zh` works.

**What this means**: the previous experiment log entry blaming the codec for CSM/Orpheus failures was likely incorrect. The real bottlenecks were:
1. **Layer 1 — text tokenizer**: Llama 3.2 BPE byte-fragments Adja diacritics (ɛ, ɔ, ŋ, ɖ, tone marks). The LM cannot learn phoneme→audio mapping from byte sequences with 1.7h of data.
2. **Layer 2 — LM prior**: English/French-pretrained LM has no Gbe-family acoustic prior. The backbone needs to see enough target-language audio to build that mapping.

---

#### How They Interact: Why Spark Won (Revised Analysis)

Spark succeeded where CSM and Orpheus failed because it got **all three layers right**:
- Qwen2 tokenizer → native Adja character handling (Layer 1 ✓)
- Qwen2-based LM with multilingual pretraining (Layer 2 ✓)
- BiCodec with XLSR-53 semantic tokens covering African languages (Layer 3 ✓)

CSM and Orpheus **did NOT fail because of the codec** (Layer 3). They failed on Layer 1: Llama 3.2 BPE fragmentation of Adja characters into byte tokens. The codec (Mimi, SNAC) is language-agnostic and handles waveforms fine — this is confirmed by Orpheus's working Mandarin (`3b-zh`) variant. The experiment log's "Mimi is English-heavy" claim should be read as "the CSM *stack* is English-heavy" — the tokenizer is the culprit.

**Implication for future CSM/Orpheus attempts**: fix Layer 1 first (expand tokenizer vocab for Adja characters or switch to a tokenizer that handles them natively), then fine-tune. Starting from `canopylabs/3b-zh-ft-research_release` (tonal Mandarin base) before adapting to Adja Ewe is also worth exploring — the LM already knows how to map tonal text to audio tokens.

**Path forward for Orpheus → Ewe → Adja:**
1. Start from `3b-zh` pretrain checkpoint (tonal LM prior, same SNAC codec)
2. Expand Llama tokenizer with Adja/Ewe-specific characters (ɛ, ɔ, ŋ, ɖ, è, é, ɛ̀, ɔ́, etc.)
3. Fine-tune on 15k labeled Ewe speech (WaxalNLP)
4. Adapt to Adja (1.6k utterances)

---

#### What Fine-tuning Actually Changes

| What you're fine-tuning | Which layer | What it teaches |
|---|---|---|
| Full model fine-tune | All layers (usually codec frozen) | Everything — most data-hungry |
| LoRA on LM backbone | Layer 2 only | Text→audio mapping for new language/voice |
| Codec continue-training | Layer 3 only | Better acoustic representation of target phonology |
| Audio LM pretraining (unlabeled) | Layer 2 only (codec frozen) | Acoustic patterns of target language without text |
| Tokenizer vocab expansion | Layer 1 | New characters mapped to native tokens instead of bytes |

For Adja with ~1.6k utterances: LoRA on Layer 2 is the most efficient. If Layer 1 is broken (byte fragmentation), fix that first — no amount of Layer 2 training recovers from it.

---

### Audio Codecs (Reference)
- **Mimi** (used by CSM): 24kHz, 32-codebook split-RVQ. Paper: https://arxiv.org/abs/2410.00037
- **SNAC** (used by Orpheus): 24kHz, 3-codebook hierarchical RVQ. Repo: https://github.com/hubertsiuzdak/snac
- **EnCodec** (Meta): 24kHz, 8-codebook RVQ. Paper: https://arxiv.org/abs/2210.13438
- **DAC** (Descript): 44.1kHz, high-quality. Paper: https://arxiv.org/abs/2306.06546
- **BiCodec** (Spark): XLSR-53 semantic + acoustic RVQ. Part of Spark-TTS repo.

### LoRA for TTS
Same LoRA technique as LLM fine-tuning (https://arxiv.org/abs/2106.09685) applied to TTS:
- Train low-rank adapters on attention + MLP layers
- Keeps base model frozen, learns voice/language-specific patterns
- Especially effective for low-resource scenarios like Adja (~1.6k utterances)

### Unsloth Optimizations
- Flash Attention 2 for 1.5x faster training
- 50% less VRAM through custom Triton kernels
- Gradient checkpointing optimized for their pipeline
- Runs on free T4 Colab GPUs (huge for accessibility)
- Docs: https://unsloth.ai/docs/basics/text-to-speech-tts-fine-tuning

### Voice Cloning vs Voice Fine-tuning
- **Voice cloning** (inference-time): Provide a reference audio clip, model mimics the voice. CSM supports this.
- **Voice fine-tuning** (training-time): Train the model on many examples of a target voice. More consistent but needs data.
- For Adja: we're doing language fine-tuning more than voice cloning — teaching the model Adja phonology.

## Low-Resource TTS Considerations for Adja

1. **Phonology matters**: Adja has tones (é, è), special characters (ɛ, ɔ, ŋ, ɖ) — the model needs to learn these phoneme-to-audio mappings
2. **Data size**: ~1.6k utterances is small even for TTS fine-tuning. LoRA helps a lot here.
3. **Resampling**: Our data is 48kHz, CSM expects 24kHz — downsampling is handled in `data_prep.py`
4. **NFC normalization**: Must be consistent between training and inference (same as ASR experiments)
5. **Evaluation**: Listen to generated samples! Automated metrics (MOS prediction) exist but human eval is king for TTS.

## Reading Order

1. Start with the **Sesame CSM notebook** — our primary experiment
2. Read the **Unsloth TTS docs** for the fine-tuning pipeline overview
3. Skim **Orpheus notebook** as a comparison approach
4. Read the **Mimi codec paper** to understand what CSM actually generates
5. **Spark TTS** is interesting for future deployment considerations
