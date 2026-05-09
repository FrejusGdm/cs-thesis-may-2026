# Neural TTS from First Principles

> **Estimated reading time: 18 minutes**
>
> This guide takes you from human speech physiology to the cutting edge of neural TTS,
> ending with exactly where Qwen3-TTS sits in this evolution. No prerequisites beyond
> curiosity and basic linear algebra intuition.

---

## Table of Contents

1. [How Humans Produce Speech](#1-how-humans-produce-speech)
2. [The Digital Representation Problem](#2-the-digital-representation-problem)
3. [Classical TTS: Before Deep Learning](#3-classical-tts-before-deep-learning)
4. [The Neural Revolution: Tacotron to FastSpeech](#4-the-neural-revolution-tacotron-to-fastspeech)
5. [Modern Approaches: Diffusion, Flow-Matching, and Codebook LMs](#5-modern-approaches-diffusion-flow-matching-and-codebook-lms)
6. [Where Qwen3-TTS Fits](#6-where-qwen3-tts-fits)
7. [Why Pre-training + Fine-tuning Works for TTS](#7-why-pre-training--fine-tuning-works-for-tts)
8. [Mathematical Intuition](#8-mathematical-intuition)
9. [What This Means for Your Fine-tuning Project](#9-what-this-means-for-your-fine-tuning-project)

---

## 1. How Humans Produce Speech

Before building a machine that speaks, let's understand what speech actually *is*.

### The Human Speech Pipeline

```
Brain (intent) → Lungs (airflow) → Larynx (vibration) → Vocal tract (shaping) → Lips/tongue → Sound waves
```

**Lungs**: Push air upward. The amount of air determines loudness (amplitude).

**Larynx (voice box)**: Contains the **vocal folds** (commonly called vocal cords). When air
passes through:
- **Voiced sounds** (vowels, 'b', 'd', 'z'): Vocal folds vibrate, creating a periodic buzz.
  The vibration frequency = **fundamental frequency (F0)** = perceived pitch.
  - Adult male: ~85-180 Hz
  - Adult female: ~165-255 Hz
- **Unvoiced sounds** ('s', 'f', 'sh'): Vocal folds open, air flows through freely,
  creating noise/turbulence.

**Vocal tract** (throat, mouth, nasal cavity): Acts as a **resonant filter**. Different shapes
amplify different frequencies. The peaks of amplified frequencies are called **formants** (F1, F2, F3...).
- Changing tongue position moves formants → different vowels
- Closing/opening lips → different consonants
- Nasal cavity coupling → nasal sounds ('m', 'n')

**Key insight**: Speech is a *source-filter system*. The source (vocal folds) generates raw sound,
the filter (vocal tract) shapes it into recognizable speech. This insight from the 1960s still
influences modern TTS design.

### What Makes Speech Hard to Synthesize

1. **Prosody**: The melody of speech — pitch contours, stress patterns, rhythm, pacing.
   "I didn't say he stole the money" has 7 different meanings depending on which word you stress.
2. **Coarticulation**: Each sound is influenced by surrounding sounds. The 'k' in "key" is
   physically different from the 'k' in "cool" because your mouth is already shaping the
   following vowel.
3. **Speaker identity**: Same words sound different from different people — not just pitch,
   but timbre, breathiness, speaking rate, accent.
4. **Expressiveness**: Emotion, emphasis, hesitation, laughter — all conveyed through subtle
   acoustic variations.

---

## 2. The Digital Representation Problem

### Waveforms

Audio is stored as a sequence of amplitude samples over time. A waveform at 24,000 Hz
(Qwen3-TTS's sample rate) means 24,000 numbers per second of audio.

- 10 seconds of audio = 240,000 numbers
- That's a LOT of data for a neural network to generate from scratch

### Spectrograms and Mel-Spectrograms

Instead of raw waveforms, we often work with **spectrograms** — a time-frequency representation.

A **spectrogram** is computed using the Short-Time Fourier Transform (STFT):
1. Slide a window (e.g., 1024 samples) across the waveform
2. For each window position, compute the frequency content (via FFT)
3. Stack these frequency snapshots side-by-side → 2D image (time × frequency)

A **mel-spectrogram** warps the frequency axis to the **mel scale**, which approximates
human hearing perception. We hear the difference between 100Hz and 200Hz much more clearly
than between 8000Hz and 8100Hz. The mel scale compresses high frequencies accordingly.

**Qwen3-TTS mel parameters** (from `dataset.py`):
- Sample rate: 24,000 Hz
- FFT size: 1024
- Mel bins: 128
- Hop size: 256 (~93.75 frames/second)

### Why This Matters

Early neural TTS systems generate mel-spectrograms (a manageable ~93 frames/second)
rather than raw waveforms (24,000 samples/second). A separate **vocoder** then converts
mel-spectrograms back to waveforms. This decomposition made neural TTS tractable.

---

## 3. Classical TTS: Before Deep Learning

### Concatenative TTS (1990s-2000s)

**Idea**: Record a person saying thousands of short speech segments. At synthesis time,
find the best segments and splice them together.

**How it works:**
1. Record a speaker reading many hours of scripted text
2. Segment the recordings into small units (phones, diphones, syllables)
3. Store in a large database
4. At synthesis: select the best-matching units and concatenate them

**Pros**: Very natural-sounding for the recorded speaker
**Cons**: Sounds robotic at unit boundaries, can't change speaker, requires huge databases,
poor for new languages (need massive recordings per speaker)

### Parametric TTS (2000s-2015)

**Idea**: Instead of storing actual audio, model speech as a set of parameters
(pitch, duration, spectral envelope) and generate audio from those parameters.

**How it works:**
1. Train statistical models (Hidden Markov Models → later deep neural networks)
   to predict speech parameters from text
2. Use a signal processing vocoder (like STRAIGHT or WORLD) to generate audio
   from the predicted parameters

**Pros**: Flexible, compact models, can interpolate between speakers
**Cons**: Muffled, buzzy quality — the vocoder is the bottleneck.
Sounded noticeably synthetic compared to concatenative approaches.

### The quality ceiling problem

Both approaches hit a wall: concatenative TTS was natural but inflexible,
parametric TTS was flexible but unnatural. Something fundamentally different was needed.

---

## 4. The Neural Revolution: Tacotron to FastSpeech

### WaveNet (2016) — DeepMind

The first breakthrough. WaveNet generates audio **sample by sample** using a deep
autoregressive neural network.

**Architecture**: Dilated causal convolutions — each layer looks at exponentially
more past samples, efficiently capturing both short-term (phonetics) and long-term
(prosody) dependencies.

**Impact**: First time a synthetic voice was genuinely hard to distinguish from human speech.
**Problem**: Excruciatingly slow — generating 1 second of audio took ~1 minute on a GPU.

### Tacotron (2017) — Google

**Key insight**: Use a sequence-to-sequence model (like machine translation) but instead
of translating English→French, translate text→mel-spectrogram.

```
Text → Encoder → Attention → Decoder → Mel-spectrogram → Vocoder → Audio
```

**Architecture**:
- **Encoder**: Processes text characters/phonemes into a sequence of hidden states
- **Attention**: Learns to align text positions with mel-spectrogram frames
  (which character is being spoken at which time)
- **Decoder**: Autoregressively generates mel-spectrogram frames, one at a time

**Tacotron 2** (2018) combined this with a WaveNet vocoder for state-of-the-art quality.

**Problems**:
- Autoregressive = slow (generates one mel frame at a time)
- Attention alignment can fail → skipped/repeated words
- Hard to control speaking speed

### FastSpeech (2019) — Microsoft

**Key insight**: Remove the autoregressive bottleneck. Predict all mel frames in parallel.

```
Text → Encoder → Duration Predictor → Length Regulator → Decoder → Mel-spectrogram
```

The **Duration Predictor** explicitly estimates how many mel frames each phoneme should occupy.
The **Length Regulator** expands the text sequence to match the target mel length.
Then the decoder generates ALL mel frames simultaneously.

**FastSpeech 2** added pitch and energy predictors for better prosody control.

**Pros**: 10-100x faster than autoregressive models, controllable duration/pitch
**Cons**: Quality slightly below autoregressive models, still needs a separate vocoder

### Neural Vocoders

The mel→waveform step also went neural:
- **WaveRNN** (2018): Efficient RNN-based vocoder
- **HiFi-GAN** (2020): GAN-based, real-time, excellent quality
- **Vocos** (2023): Even more efficient

---

## 5. Modern Approaches: Diffusion, Flow-Matching, and Codebook LMs

The field has exploded since 2022. Three major paradigms have emerged:

### Approach A: Diffusion-Based TTS

**Examples**: Grad-TTS, DiffSinger, NaturalSpeech 2/3

**Idea**: Start with random noise, iteratively denoise it into a mel-spectrogram
(or directly into a waveform), conditioned on the input text.

**How it works**:
1. **Forward process**: Gradually add Gaussian noise to a real mel-spectrogram until
   it's pure noise (over T timesteps)
2. **Reverse process**: Train a neural network to predict and remove the noise at each step
3. **At inference**: Start from random noise, run the reverse process to get clean audio

**Intuition**: Think of it like a sculptor. Forward = covering a statue with clay until
it's a featureless blob. Reverse = carefully removing clay to reveal the statue.
The text tells the model what statue to carve.

**Pros**: Excellent quality, diverse outputs (same text → different valid pronunciations)
**Cons**: Multiple denoising steps = slow, typically 50-1000 steps at inference

### Approach B: Flow-Matching TTS

**Examples**: Voicebox (Meta), E2 TTS, Matcha-TTS

**Idea**: Similar to diffusion, but learns a direct *flow* (continuous transformation)
from noise to data. Mathematically cleaner and often needs fewer steps.

**Flow matching** defines a path from noise distribution to data distribution and trains
a neural network to follow this path. Think of it as learning the optimal trajectory
to morph noise into speech, rather than learning incremental denoising.

### Approach C: Discrete Codebook Language Models ← **Qwen3-TTS is here**

**Examples**: VALL-E (Microsoft), SoundStorm (Google), Qwen3-TTS (Alibaba)

**The big idea**: Treat speech generation as a **language modeling** problem.

**Step 1**: Train a **speech tokenizer** (codec) that converts audio → discrete tokens,
like how text is broken into word pieces. This is typically done using **Vector Quantization (VQ)**
or **Residual Vector Quantization (RVQ)**.

**Step 2**: Train a **language model** (transformer) to generate these speech tokens,
conditioned on text tokens. The LM generates speech the same way GPT generates text —
predict the next token, autoregressively.

**Why this is revolutionary**: It unifies text and speech into the same framework.
A single transformer can do TTS, ASR, voice conversion, speech translation — all as
different "translation" tasks between token sequences.

### Residual Vector Quantization (RVQ) — The Key Technique

RVQ is how Qwen3-TTS compresses audio into discrete tokens:

```
Audio frame → VQ Layer 1 (coarse) → Residual → VQ Layer 2 (medium) → Residual → ... → VQ Layer N (fine)
```

1. **Layer 1**: Quantize the audio feature to the nearest codebook entry.
   This captures the coarsest information (roughly: what phoneme is being said).
2. **Compute residual**: What's left = original - quantized approximation
3. **Layer 2**: Quantize the residual. This captures finer details (timbre, pitch nuance).
4. **Repeat** for N layers. Each layer captures progressively finer acoustic details.

**Qwen3-TTS-Tokenizer-12Hz** uses **16 layers** at 12.5 Hz:
- Layer 0: Semantic codebook (guided by WavLM teacher — captures *meaning*)
- Layers 1-15: Acoustic RVQ (captures *sound quality*)

This means 1 second of audio = 12.5 frames × 16 codebook indices = **200 discrete tokens**.
Compare with raw audio: 24,000 samples/second. That's a 120x compression!

---

## 6. Where Qwen3-TTS Fits

Qwen3-TTS represents the **state of the art in codebook LM TTS** as of January 2026.
Here's what makes it architecturally distinctive:

### The Architecture

```
Text → Text Tokenizer → [Token Embeddings] ─┐
                                             ├→ Backbone Transformer → Codebook 0 prediction
Reference Audio → Mel Extraction → Speaker Encoder ─┘                          │
                                                                    MTP Module → Codebooks 1-15 prediction
                                                                                │
                                                              All 16 codebooks → Tokenizer Decoder → Waveform
```

### Key Innovation: Hierarchical Prediction

Instead of generating all 16 codebook layers autoregressively (which would be 16x slower),
Qwen3-TTS uses a two-stage approach:

1. **Backbone** (the main transformer): Predicts codebook 0 (semantic) autoregressively.
   This is the "what is being said" layer.

2. **MTP (Multi-Token Prediction) module**: Given codebook 0, generates codebooks 1-15
   in a single forward pass. This is the "how it sounds" layer.

**Why this is smart**: Semantic content (codebook 0) is fundamentally sequential — word order
matters. But acoustic details (codebooks 1-15) are largely determined by codebook 0 plus
speaker identity, so they can be predicted in parallel.

### The Tokenizer: Semantic + Acoustic Disentanglement

Qwen3-TTS-Tokenizer-12Hz separates speech into two streams:

**Semantic stream** (codebook 0):
- Trained with **WavLM distillation**: WavLM is a pre-trained speech understanding model.
  The tokenizer's first codebook is trained to reproduce WavLM's representations.
- Captures: phonetic content, word boundaries, linguistic structure
- Analogy: The "script" of the speech

**Acoustic stream** (codebooks 1-15):
- Trained with **RVQ** on the residual after semantic quantization
- Captures: pitch, timbre, breathiness, room acoustics, emotion
- Analogy: The "performance" of the speech

### Training with GAN + Mel Loss

The tokenizer is trained with a GAN framework:
- **Generator**: Encodes audio → codes → decodes audio (should reconstruct faithfully)
- **Discriminator**: Tries to distinguish real audio from reconstructed audio
- **Multi-scale mel-spectrogram loss**: Ensures frequency-domain accuracy at multiple time scales

### How It Compares

| Model          | Approach         | Speed      | Quality | Voice Cloning |
|----------------|------------------|------------|---------|---------------|
| Tacotron 2     | Autoregressive   | Slow       | Good    | No            |
| FastSpeech 2   | Non-autoregressive| Fast      | Good    | Limited       |
| VALL-E         | Codebook LM      | Medium     | Great   | Yes (3-10s)   |
| NaturalSpeech3 | Diffusion+LM     | Medium     | Excellent| Yes          |
| Qwen3-TTS      | Multi-codebook LM| Fast (97ms)| Excellent| Yes (3s)     |

---

## 7. Why Pre-training + Fine-tuning Works for TTS

### The Transfer Learning Principle

Pre-training on massive data (5M+ hours for Qwen3-TTS) teaches the model:

1. **Universal phonetics**: How any human language's sounds map to acoustic features
2. **Prosodic patterns**: Intonation, rhythm, stress patterns across languages
3. **Speaker modeling**: How to disentangle "what" from "who" and "how"
4. **Robustness**: Handling noise, varied recording conditions, spontaneous speech

When you fine-tune on your language's data:
- The model already knows most speech sounds (many phonemes are shared across languages)
- It has learned general prosodic patterns (rising pitch for questions is nearly universal)
- It knows how to handle the source-filter decomposition
- It just needs to learn: your language's specific phoneme inventory, prosodic rules,
  and phonotactics (valid sound combinations)

### Why Small Datasets Can Work

A pre-trained TTS model has built a rich internal representation of speech. Fine-tuning
adjusts this representation — it doesn't build from scratch. This is why:

- **Similar languages** (your language shares many sounds with the pre-training languages):
  1-5 hours may suffice. The model mostly needs to learn new sound combinations and prosody.

- **Distant languages** (has sounds not in pre-training data):
  10-50+ hours. The model needs to learn new acoustic patterns, but it still benefits from
  pre-trained knowledge of general speech structure.

### What Fine-tuning Actually Changes

When you run `sft_12hz.py`, you're updating the **backbone transformer** weights.
The tokenizer is frozen. Conceptually:

- **Before fine-tuning**: The backbone generates codebook sequences that sound like the
  10 pre-training languages
- **After fine-tuning**: The backbone generates codebook sequences that sound like
  your language + speaker

The codebook vocabulary doesn't change — the model learns to *compose* existing codes
in new ways that represent your language's sounds.

---

## 8. Mathematical Intuition

### Cross-Entropy Loss (How the Model Learns)

The TTS model predicts the probability distribution over codebook entries for each position.
The loss measures how wrong these predictions are:

```
L = -Σ log P(correct_token | context)
```

**Intuition**: If the correct next codebook index is 42, and the model assigns:
- P(42) = 0.9 → Loss = -log(0.9) = 0.105 (low — model is confident and correct)
- P(42) = 0.01 → Loss = -log(0.01) = 4.605 (high — model is very wrong)

The model's weights are adjusted to increase the probability of correct tokens.

### Why AdamW and Learning Rate Matter

**AdamW optimizer**: Adapts the learning rate per-parameter based on gradient history.
Parameters that rarely get large gradients get boosted (useful for rare phonemes in your data);
parameters that always get large gradients get dampened (prevents runaway updates).

**Learning rate = 2e-5**: This is intentionally small for fine-tuning. The model already
has good weights — we want to nudge them, not overhaul them.

- **Too high** (e.g., 1e-3): The model "forgets" its pre-training knowledge (catastrophic forgetting).
  You'd hear random noise or garbled speech.
- **Too low** (e.g., 1e-7): The model barely changes. You'd hear the pre-training languages'
  accent bleeding through, or the model ignoring your language's patterns.
- **Just right** (1e-5 to 5e-5): The model retains general speech knowledge while adapting
  to your language.

### Gradient Clipping (max_norm=1.0)

Occasionally, a batch of data produces unusually large gradients. Without clipping,
this one batch could undo hours of training. Gradient clipping scales down the gradient
vector if its norm exceeds 1.0, preventing catastrophic updates.

### Warmup Schedule

For the first few percent of training, the learning rate ramps up from near-zero to the
target value. Why? At the start, the model's running averages in AdamW aren't calibrated
yet. Large LR + uncalibrated optimizer = chaotic initial updates. Warmup lets the optimizer
"settle in" before applying full learning rate.

---

## 9. What This Means for Your Fine-tuning Project

### Your Pipeline, Explained

```
Your audio files (WAV)
    ↓
[process_audio.py] — Resample to 24kHz, trim silence, normalize volume
    ↓
[process_text.py] — Normalize text for your language
    ↓
[prepare_data.py → Qwen3TTSTokenizer] — Convert audio to 16 codebook sequences
    ↓
Training JSONL: {text, codes[16 layers], ref_audio}
    ↓
[sft_12hz.py] — Fine-tune backbone LM
    ↓
Fine-tuned model checkpoint
```

### Expected Outcomes by Dataset Size

| Dataset Size    | Expected Quality                                   |
|-----------------|---------------------------------------------------|
| < 30 minutes    | Basic phonemes, likely unstable prosody             |
| 1-3 hours       | Recognizable language, decent prosody               |
| 5-10 hours      | Good quality, natural prosody for common patterns   |
| 20+ hours       | Excellent quality, handles edge cases               |

### How to Know If Training Is Working

**Good signs** (check TensorBoard or log output):
- Loss decreases steadily over the first epoch
- Loss stabilizes (doesn't oscillate wildly) in later epochs
- Generated audio at epoch 1: recognizable speech with artifacts
- Generated audio at epoch 3: natural-sounding, correct phonemes

**Bad signs:**
- Loss increases or oscillates wildly → LR too high, reduce to 1e-5
- Loss doesn't decrease at all → LR too low, data format wrong, or model not loading correctly
- Loss decreases but audio sounds like the pre-training language → Need more data or more epochs
- Loss decreases but audio is garbled → Data quality issue (mismatched text/audio, corrupt files)

### Language-Specific Considerations

For non-English languages, pay special attention to:

1. **Tonal languages** (Chinese, Vietnamese, Thai, Yoruba): Pitch contours carry meaning.
   The model needs enough data to learn your language's tone system. Budget for more data.

2. **Agglutinative languages** (Turkish, Finnish, Korean, Swahili): Very long words.
   Check that the text tokenizer handles long words without excessive fragmentation.

3. **RTL scripts** (Arabic, Hebrew): Ensure your text preprocessing handles directionality.
   The model processes text as a token sequence, so script direction shouldn't matter,
   but text normalization utilities may have bugs with RTL.

4. **Click consonants** (Zulu, Xhosa): These sounds may not exist in the tokenizer's
   training data. Test codec reconstruction quality on your audio first.

5. **Diacritical marks** (Vietnamese, Arabic, many African languages): NFC Unicode
   normalization is critical. Without it, the same character with a diacritic may be
   encoded differently across your dataset.

---

## Further Reading

- **Speech and Language Processing** (Jurafsky & Martin) — Chapters 16-17 on speech synthesis
- **Neural Network Methods for Natural Language Processing** (Goldberg) — Foundations
- **Qwen3-TTS Technical Report**: https://arxiv.org/abs/2601.15621
- **VALL-E paper**: https://arxiv.org/abs/2301.02111 — The foundational codebook LM for TTS
- **SoundStorm**: https://arxiv.org/abs/2305.09636 — Parallel codebook generation (related to MTP)
- **WavLM**: https://arxiv.org/abs/2110.13900 — The teacher model for semantic codebook

---

*This document was written to give you the conceptual foundation to make informed decisions
during fine-tuning. When you adjust a hyperparameter, you should understand what it does
and why. When training goes wrong, you should have mental models to diagnose the problem.
Copy-paste scripts get you started; understanding gets you to a working model.*
