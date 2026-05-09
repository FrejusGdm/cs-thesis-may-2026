# T4: Inworld TTS — A Student's Guide

**Status**: `planned` (docs only — no code or runs yet)
**Why it's here**: Inworld open-sourced a fascinating SpeechLM-based TTS training framework. This doc is our deep-dive to understand *how it works* and *why it's interesting for Adja*, so we can decide whether to invest real engineering time in it later.

---

## Part 1: What is Inworld TTS?

Inworld AI is a company building conversational AI characters for games. In 2026 they open-sourced the **training code** for their TTS-1 and TTS-1-Max models (MIT license). The paper is on arxiv: `2507.21138`.

**Important distinction**: They released the **training code**, not the **trained model weights**. This is different from Sesame CSM (where you can `huggingface-cli download sesame/csm-1b` and fine-tune). With Inworld, you're getting a recipe, not a cake.

### The two official models (reference only — weights not released)
| Model | Parameters | Intended use |
|-------|-----------|--------------|
| TTS-1 | 1.6B | Real-time, on-device |
| TTS-1-Max | 8.8B | Maximum quality |

Both are autoregressive Transformers. TTS-1 is roughly the same size as Sesame CSM; TTS-1-Max is ~5× larger than the biggest model we've considered elsewhere.

---

## Part 2: SpeechLM-Based TTS (The Core Idea)

This is the conceptual piece that took me (Claude) the longest to fully grasp, and it's the key to understanding Inworld, Sesame CSM, Orpheus, Llasa, and VALL-E all at once. Worth reading slowly.

### The old way: Text → spectrogram → waveform
Classic TTS systems (Tacotron 2, FastSpeech 2) did:
1. Text → mel-spectrogram (a 2D image of frequency vs time) — done by a seq2seq model
2. Mel-spectrogram → waveform — done by a vocoder (e.g., HiFi-GAN, WaveNet)

Two models, chained. Works, but spectrograms are a **continuous** representation — hard to mix with text tokens in a unified architecture.

### The SpeechLM way: Text tokens + audio tokens, one model
What if audio could be represented as **discrete tokens**, just like text? Then you could train a single LLM that predicts the next token — and sometimes that token is text, sometimes it's "a chunk of audio."

This is exactly what happens:
1. An **audio codec** (xcodec2, Mimi, EnCodec, DAC — these are all neural networks) compresses a 1-second clip of audio into, say, ~75 discrete tokens drawn from a vocabulary of 65,536 possible values.
2. You interleave these audio tokens with text tokens in a single sequence.
3. You train a standard autoregressive LLM to predict the next token.
4. At generation time, the LLM sees `<text> "Hello world" </text> <audio> [codec tokens...]`, and generates audio tokens which the codec **decoder** then reconstructs back into a waveform.

The big insight: **because a pretrained text LLM already "knows" sequential patterns, grammar, and attention over long contexts, it's a surprisingly good starting point for a TTS model.** You're just teaching it a new "language" (audio) alongside the text it already knows.

### Why this matters for Adja

The LLM backbone has already seen massive amounts of text. Even if Adja isn't in its training data, its **transformer machinery** (attention, positional encoding, token mixing) is battle-tested. We're teaching it an Adja→audio mapping, not rebuilding a transformer from scratch.

---

## Part 3: Inworld's Specific Architecture

### The dual-encoder setup
Inworld uses what they call a "dual-encoder" approach. There are really three components at play:

```
                    ┌─────────────────────────────────────┐
                    │       Reference audio (optional)    │
                    │       — "sound like this voice"     │
                    └──────────────┬──────────────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────┐
                    │   xcodec2 AUDIO ENCODER  │  (frozen, pretrained)
                    │   waveform → discrete    │
                    │   codec tokens           │
                    └──────────────┬───────────┘
                                   │
                                   │  [audio tokens from reference]
                                   ▼
┌──────────────┐   tokens   ┌──────────────────────────────────┐
│ Text:        │ ─────────► │   SpeechLM (Llama-3.2-1B base)  │
│ "wézé kan..."│            │   Autoregressive Transformer    │
└──────────────┘            │   Predicts next audio token     │
                            └──────────────┬───────────────────┘
                                           │
                                           │  [generated audio tokens]
                                           ▼
                            ┌──────────────────────────┐
                            │   xcodec2 AUDIO DECODER  │  (frozen, pretrained)
                            │   discrete codec tokens  │
                            │   → 48kHz waveform       │
                            └──────────────┬───────────┘
                                           │
                                           ▼
                                    🔊 Output audio
```

**Three components**:
1. **xcodec2 encoder/decoder** — a separately trained neural audio codec. Used as a frozen "tokenizer" for audio. You don't train this; you use their public checkpoint.
2. **SpeechLM** — a Llama-3.2-1B-Instruct that gets further trained to predict audio tokens. This is what you train on your Adja data.
3. **Reference audio** (optional) — at inference, you give it a few seconds of a target voice; the model picks up the voice characteristics via in-context learning. No voice-specific fine-tuning needed.

### What "zero-shot voice cloning" actually means
Normal ML intuition would say: "to clone a voice, you need to fine-tune on that voice's recordings." Inworld (and CSM, VALL-E, XTTS) do something smarter: they train the model such that **the first N seconds of the audio token stream condition the rest**.

At inference:
- Prepend `[codec tokens of reference audio]` + `[text tokens of what you want to say]` as the prompt.
- The LLM generates audio tokens that continue the prosody, timbre, and pacing of the reference.

Think of it like few-shot prompting for GPT-4: you show a voice example, and the model generalizes.

**For Adja**: this is great — it means if we ever train a strong Adja SpeechLM, we could clone different speakers from short reference clips without per-speaker fine-tuning.

---

## Part 4: The Training Pipeline (Their Framework)

Inworld's code supports **three stages** (can pick which to run):

### Stage 1: Data vectorization
**What**: Convert every audio clip in your dataset into discrete codec tokens, saving them as `.pt` files.
**Why separate**: The codec forward pass is expensive and deterministic — cache the output so you don't re-vectorize every training epoch.
**Tool**: `tools/data/data_vectorizer.py` (their README shows 8-GPU example via `torchrun`).
**xcodec2 details**: Codebook size 65,536, accepts 24kHz input. The output is a sequence of ~75 tokens per second of audio.

### Stage 2: Supervised Fine-Tuning (SFT)
**What**: Train the SpeechLM (starting from Llama-3.2-1B-Instruct) to predict audio tokens from text.
**Loss**: Cross-entropy over next-token prediction (same as any LLM training).
**Data format**: JSONL with `transcript`, `language`, `wav_path`, `duration`, `sample_rate`.
**Command pattern**: `fabric run --devices=$NUM_GPU tts/training/main.py --config_path=./example/configs/sft.json`

**Trap to notice**: In their vocabulary, "SFT" means "initial speech training starting from the text-only LLM." This is not "fine-tune a pretrained TTS model"— there is no Inworld TTS checkpoint to fine-tune. From our perspective, this stage is **training a TTS model from scratch** (with the text LLM as a head start).

### Stage 3: RL alignment (RLHF)
**What**: Further align the SFT model using reward signals.
**Requires**: Multi-node compute (they show `--nodes=2 --gpus-per-node=8` in their example). One node runs inference (vLLM), one runs training.
**Skip for us**: Way too expensive for a research exploration with 1.6k utterances.

---

## Part 5: Supported Languages

Inworld's framework supports these language codes (in `allowed_languages` config):

`en, es, fr, de, it, pt, ru, zh, ko, ja, nl, pl`

That's 12 codes — though the paper abstract says "11 languages." The discrepancy probably reflects that one code is for an ancillary dataset and not an officially released language in the shipped model.

**Adja (`aj_Latn`) is NOT in this list.** If we were to train, we'd need to:
- Option A: Use a close proxy code like `"fr"` (French — shares the Latin script, and our MMS-French-adapter ASR experiment (C3) showed some phonetic transfer)
- Option B: Add a custom `"aj"` code and see if their config accepts it (they filter by string matching in `allowed_languages`, so extending the whitelist should work)

---

## Part 6: Compute and Data Requirements (Honest)

### What their examples assume
- **CUDA 12.4+**
- **Python 3.10**
- **PyTorch 2.6 (CUDA 12.4) or 2.7 (CUDA 12.8)**
- **Package manager**: `uv` (not pip)
- **Data vectorization**: 8 GPUs (`torchrun --nproc_per_node 8`)
- **SFT training**: Multi-GPU (they use `fabric` from Lightning)
- **RLHF**: 2+ nodes, 8 GPUs each
- **Reference dataset**: LibriTTS ≈ 585 hours, 2,456 speakers, 24 kHz sampling

### What we have
- ~2 hours of Adja audio from ~1.6k utterances
- One A100 40GB MIG slice per SLURM job (when HPC is up)
- Colab T4/A100 for experiments right now

The data gap is **~300×** between our dataset and their reference. This is the single biggest risk for T4: training a SpeechLM from scratch (even from a text LLM base) on 1.6k utterances is a known-poor regime. VALL-E, the original SpeechLM paper, used 60,000 hours of audio.

---

## Part 7: Comparison with Our Other TTS Options

| Feature | **T1 Sesame CSM** | **T2 Orpheus** | **T3 Spark** | **T4 Inworld** |
|---------|-------------------|----------------|--------------|----------------|
| Pretrained weights released? | ✅ Yes (`unsloth/csm-1b`) | ✅ Yes (Canopy Labs HF) | ✅ Yes (SparkAudio HF) | ❌ No |
| Base architecture | Llama 3.2-1B + Mimi | Llama 3B + audio head | 0.5B custom | Llama 3.2-1B + xcodec2 |
| Audio codec | Mimi (24 kHz) | Custom | Custom | xcodec2 (24 kHz in, 48 kHz out) |
| Output sample rate | 24 kHz | 24 kHz | 24 kHz | **48 kHz** ← matches our raw audio |
| Realistic fine-tune on 1.6k utterances? | ✅ Yes (LoRA) | ✅ Yes (LoRA) | ✅ Yes | ⚠️ Risky (from scratch) |
| Voice cloning | ✅ Reference audio | Partial | ✅ | ✅ Reference audio |
| Unsloth support? | ✅ | ✅ | ✅ | ❌ (custom stack) |
| License | Apache 2.0 | Apache 2.0 | Apache 2.0 | MIT |

**The key takeaway**: For a first TTS result on Adja, T1/T2/T3 are all "fine-tune a pretrained model with LoRA on our 1.6k utterances" — well-trodden, achievable. T4 is "train a new SpeechLM nearly from scratch" — exploratory, risky, compute-heavy.

---

## Part 8: When Would We Actually Build This?

Conditions under which T4 becomes worth the engineering:

1. **T1/T2/T3 all fail at audio quality** and we suspect the 24 kHz codec is the bottleneck (Inworld's 48 kHz native would be an upgrade).
2. **We get access to significantly more Adja data** — say, 50+ hours — making from-scratch training less futile.
3. **Inworld (or a community) releases pretrained SpeechLM weights** — then we fine-tune instead of training, and the whole picture changes.
4. **We're writing a paper** and need a "trained a TTS from scratch on a low-resource language" angle. (Arguably novel but risky.)

Until then: T4 stays in docs-only mode.

---

## Part 9: Open Questions (Worth Investigating Later)

- What's in xcodec2's public checkpoint? Does it generalize to Adja phonology without retraining? (The codec is trained on Western-language audio; tonal / non-Latin sounds may be poorly represented.)
- Can we "warm-start" the SpeechLM from someone's released Llama-based TTS (e.g., Llasa) instead of raw Llama-3.2-1B? This would be a hybrid between "from scratch" and "fine-tune."
- Inworld's RL alignment uses a reward model — do they release it? What does it reward (prosody? intelligibility? speaker similarity?)?
- How does their 48 kHz output actually work given 24 kHz training input? (Presumably the codec's decoder upsamples — worth confirming in the paper.)

---

## Part 10: Links and Further Reading

- **Paper (TTS-1 Technical Report)**: https://arxiv.org/abs/2507.21138
- **GitHub**: https://github.com/inworld-ai/tts
- **Blog**: https://inworld.ai/blog/introducing-inworld-tts
- **Demo**: https://inworld-ai.github.io/tts/
- **Docs**: https://docs.inworld.ai/tts/tts
- **xcodec2 (codec dependency)**: https://huggingface.co/HKUSTAudio/xcodec2
- **Deep-dive reading notes**: see `ideas/inworld-tts-deep-dive.md`

---

## Companion Learning

For the bigger picture of how SpeechLM-based TTS relates to other approaches:
- VALL-E (Microsoft, 2023) — the original "LLM-style TTS" paper: https://arxiv.org/abs/2301.02111
- EnCodec (Meta, 2022) — founded the neural audio codec trend: https://arxiv.org/abs/2210.13438
- Mimi codec (Kyutai, 2024) — what Sesame CSM uses: https://arxiv.org/abs/2410.00037
- Llama 3.2 (Meta, 2024) — base LLM for Inworld, CSM, Llasa: https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct

Read VALL-E first if you only read one. It set the template that CSM, Orpheus, Llasa, and Inworld all follow.
