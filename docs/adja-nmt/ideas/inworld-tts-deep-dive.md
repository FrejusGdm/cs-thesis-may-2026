# Inworld TTS — Research Deep Dive

My personal reading notes on the Inworld TTS-1 technical report and framework. Kept separate from the T4 README (which is the project-facing doc) because this is where I think out loud about the research claims.

**Paper**: Inworld TTS-1 Technical Report, arxiv `2507.21138`
**Code**: https://github.com/inworld-ai/tts (MIT license)

---

## What the paper claims (from abstract)

> "We introduce Inworld TTS-1, a set of two Transformer-based autoregressive text-to-speech (TTS) models. Our largest model, TTS-1-Max, has 8.8B parameters and is designed for utmost quality and expressiveness in demanding applications. TTS-1 is our most efficient model, with 1.6B parameters, built for real-time speech synthesis and on-device use cases. By scaling train-time compute and applying a sequential process of pre-training, fine-tuning, and RL-alignment of the speech-language model (SpeechLM) component, both models achieve state-of-the-art performance on a variety of benchmarks, demonstrating exceptional quality relying purely on in-context learning of the speaker's voice."

Parsed:
- **Two models**: TTS-1 (1.6B) and TTS-1-Max (8.8B)
- **Three-stage training**: pre-training → fine-tuning → RL-alignment
- **48 kHz output** (high-resolution)
- **11 languages** (the paper abstract says 11; the code says 12 — probably a training-vs-deployed discrepancy)
- **Emotional control** and **non-verbal vocalizations** via "audio markups" (presumably special tokens like `<laugh>`, `<sigh>` — similar to Orpheus)
- **In-context voice cloning** — no per-voice fine-tuning needed

---

## What's novel vs prior work

I don't have the full paper yet — just the abstract + README. But based on my understanding of the landscape:

### Probably novel
- **The RL alignment stage for TTS quality**. Most open-source TTS (CSM, Orpheus, Llasa) stop at SFT. Inworld explicitly frames RLHF as a third stage. The reward model / algorithm isn't clear from the abstract — worth digging into.
- **8.8B parameter scale**. Most released TTS checkpoints are ≤3B (Orpheus). If TTS-1-Max actually beats smaller models on benchmarks, that's a useful data point for TTS scaling laws.
- **MIT-licensed training code**. Unusual — most companies release either weights (Apache) or nothing.

### Probably not novel
- **Dual-encoder with audio codec tokens**: same recipe as VALL-E (2023), CSM, Orpheus, Llasa.
- **Llama-3.2-1B backbone**: same choice as CSM and Llasa.
- **Zero-shot voice cloning via reference audio**: VALL-E did this in 2023.
- **xcodec2 codec**: not Inworld's — it's from HKUST (`HKUSTAudio/xcodec2`).

---

## Comparison table: SpeechLM-based TTS landscape

| Model | Params | Backbone | Audio Codec | Output SR | Pretrained weights? | Voice cloning | License | Year |
|-------|--------|----------|-------------|-----------|---------------------|---------------|---------|------|
| VALL-E | 400M | Custom Transformer | EnCodec | 24kHz | ❌ No (Microsoft-internal) | ✅ | — | 2023 |
| VALL-E X | — | — | EnCodec | 24kHz | ❌ No | ✅ multilingual | — | 2023 |
| Sesame CSM | 1B | Llama 3.2-1B | Mimi (Kyutai) | 24kHz | ✅ `sesame/csm-1b` | ✅ | Apache 2.0 | 2024 |
| Orpheus | 3B | Llama | Custom | 24kHz | ✅ `canopylabs/...` | Partial | Apache 2.0 | 2024 |
| Spark TTS | 0.5B | Custom small LM | Custom | 24kHz | ✅ `SparkAudio/Spark-TTS-0.5B` | ✅ | Apache 2.0 | 2024 |
| Llasa | 1B / 3B | Llama | xcodec2 | 16kHz | ✅ `HKUSTAudio/Llasa-*` | ✅ | Apache 2.0 | 2025 |
| OuteTTS | 1B | Llama | Custom | 24kHz | ✅ `OuteAI/OuteTTS-1.0-1B` | ✅ | Apache 2.0 | 2025 |
| **Inworld TTS-1** | 1.6B | Llama 3.2-1B | xcodec2 | **48kHz** | ❌ No (code only) | ✅ | MIT (code) | 2026 |
| **Inworld TTS-1-Max** | 8.8B | Llama-ish | xcodec2 | **48kHz** | ❌ No (code only) | ✅ | MIT (code) | 2026 |

**My reading**:
- Inworld is architecturally conservative — same recipe everyone else uses.
- What's interesting is the **48 kHz output** (no one else does this — they all stop at 24 kHz) and **RL alignment** (explicit third stage).
- The lack of released weights is the dealbreaker for us. Sesame CSM is basically the same architecture, smaller by 0.6B params, and lets you LoRA-fine-tune in Colab today.

---

## Questions I'd want the full paper to answer

1. **RL alignment specifics**: What reward signal? PPO, DPO, or GRPO? Is the reward a WER model, a MOS predictor, a speaker-similarity score, or a learned preference model? This is the most novel claim and the one I understand least.

2. **48 kHz training vs output**: Their data is 24 kHz (LibriTTS example, JSONL format requires `sample_rate` field), but output is 48 kHz. How? I assume xcodec2's decoder upsamples, but:
   - Does the codec actually model 48 kHz content, or just interpolate?
   - Is high-frequency content (sibilants, brightness) real or synthesized?
   - Would it even matter for Adja speech, where the meaningful phonetic content is mostly ≤8 kHz?

3. **Multilingual training strategy**: 11-12 languages in one model. Did they mix them in every batch? Train one-at-a-time with curriculum? How is code-switching handled at inference?

4. **Emotional control**: The abstract mentions "fine-grained emotional control." How? Special tokens like Orpheus's `<laugh>`? A continuous conditioning vector? Prompt tokens? This matters because tonal languages like Adja might benefit from explicit prosodic control.

5. **Training compute**: Total GPU-hours for TTS-1 vs TTS-1-Max? What was the pre-training corpus (paper abstract says "pre-training" but only LibriTTS is referenced in the README's example)?

6. **Non-verbal vocalizations**: Laughs, sighs, breaths — how many classes? Are they learned from data (Spotify-style noisy labels) or manually annotated?

---

## What I'd do if I were inspired by this

Order of plausible extensions, if we ever come back to this:

### Tier 1 — leverage what they released
- **Use their `data_vectorizer.py` for our Adja audio** → gives us pre-cached xcodec2 tokens for experimentation. Cheap, works today.
- **Benchmark xcodec2 reconstruction on Adja audio** — does it preserve ɛ, ɔ, ŋ, ɖ, and tones after round-trip? If not, no SpeechLM downstream will fix it.

### Tier 2 — hybrid approaches
- **Fine-tune Llasa 1B on Adja**. Same codec (xcodec2), same Llama backbone, already pretrained on English/Chinese TTS. Closest we can get to "Inworld-style but with released weights."
- **Try Inworld's RL alignment stage on top of a pretrained model from somewhere else** (if we had an Adja SFT model first).

### Tier 3 — actually train from scratch
- Only after getting 50+ hours of Adja audio, or if T1/T2/T3 all clearly bottleneck on codec quality.
- Even then, I'd want a warm-up ablation: Llama-3.2-1B + Adja-only training for 1k steps, just to verify the codec tokens learn before burning compute on a full run.

---

## Why this matters for our research narrative

If we ever write a paper on "TTS for low-resource Gbe languages," Inworld TTS is a useful **reference point** to cite:
- As the state-of-the-art in high-resolution open-source TTS architecture
- As evidence that 48 kHz output is achievable in the SpeechLM paradigm
- As a comparison for our own fine-tune results on smaller pretrained backbones

We don't need to run Inworld's code to cite it. That's the main value of this deep-dive.

---

## Links for future-me

- Paper: https://arxiv.org/abs/2507.21138 (get the full PDF when serious)
- Code: https://github.com/inworld-ai/tts
- Related codec: https://huggingface.co/HKUSTAudio/xcodec2
- Related fine-tuneable model on same codec: https://huggingface.co/HKUSTAudio/Llasa-1B
- Companion doc: `experiments/tts/T4_inworld_tts_finetune/README.md` (project-facing)
- Broader TTS guide: `ideas/tts-models-guide.md`
