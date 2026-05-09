# 02 — What is an audio codec (in ML-for-speech terms)

Last updated: **2026-04-21**
Prereq: `01-what-is-a-tokenizer.md`.

## One sentence

An audio codec is to a waveform what a tokenizer is to text: it converts a
continuous signal into a compact sequence of integer IDs that a language model
can consume, and it can decode the IDs back to audio.

## Why we care

Modern TTS models (CSM, Orpheus, Spark, Llasa, F5-TTS) don't predict raw audio
samples. They predict **codec tokens**, which are discrete IDs produced by a
separately-trained neural audio codec. The LM is standard autoregressive
transformer architecture; the "magic" is the codec that maps speech → IDs.

A TTS model is therefore a stack of three things (see `03-three-layer-tts.md`
for the full mental model):

1. Text tokenizer.
2. LM backbone predicting the next code.
3. Audio codec: encoder (audio→codes) + decoder (codes→audio).

## What a neural audio codec actually is

The core primitive is **residual vector quantization (RVQ)**:

1. An encoder compresses the waveform to a sequence of latent vectors at some
   frame rate (e.g. 12.5 or 75 Hz — much slower than raw 24kHz audio).
2. Each latent vector is quantized into a sequence of codebook indices via RVQ:
   first codebook captures the coarsest structure, each subsequent codebook
   captures the residual error. Typical: 4-32 codebooks.
3. A decoder reconstructs the waveform from the codebook indices.

Encoder and decoder are trained together end-to-end to minimize reconstruction
error (usually a mix of spectral, adversarial, and perceptual losses).

## Codecs used in this project

| Codec | Repo | Codebooks | Frame rate | Audio SR | Used by |
|-------|------|-----------|-----------|----------|---------|
| **Mimi** | kyutai/mimi | 32 (1 semantic + 31 acoustic) | 12.5 Hz | 24 kHz | Sesame CSM |
| **SNAC** | hubertsiuzdak/snac | 3 hierarchical | ~86 Hz | 24 kHz | Orpheus |
| **BiCodec** | SparkAudio/Spark-TTS | XLSR-53 semantic + acoustic | — | 16 kHz | Spark TTS |
| **EnCodec** | facebook/encodec | 8 | 75 Hz | 24 kHz | Llasa, others |
| **DAC** | descriptinc/descript-audio-codec | variable | — | 44.1 kHz | high-quality TTS |

Fewer codebooks × higher frame rate × fewer IDs = shorter sequence for the LM,
but also lower reconstruction quality. Design trade-off.

## The myth we had to correct

Early in this project, we assumed CSM and Orpheus failed on Adja because their
codecs (Mimi, SNAC) were "English-centric" — trained on English speech and
therefore unable to represent African tones, ATR vowels, etc.

**This was wrong.** Codecs are trained on the waveform, not on phonemes or
text. They learn to reproduce any acoustic signal, regardless of what language
is being spoken. Tones are just F0 contours in the waveform — a codec doesn't
know tones from whispers. Evidence:

- **Orpheus has a Mandarin variant** (`canopylabs/3b-zh-ft-research_release`)
  that produces intelligible Mandarin — a 4-tone language — using the *same
  SNAC codec* as the English Orpheus. If SNAC could not represent tones, the
  Mandarin model would not work. It does.
- **You can directly test it** — run `experiments/tts/T1_sesame_csm_finetune/mimi_reconstruction_test.py`.
  It encodes an Adja clip with Mimi, decodes, and writes the reconstruction to
  disk. Listen. The quality tells you whether Mimi preserves Adja phonetics.

So the codec is not the bottleneck. See `01-what-is-a-tokenizer.md` for what
the real bottleneck turned out to be.

## When would the codec actually be broken for a language?

Theoretically, if a language had acoustic features completely outside the
distribution the codec was trained on — clicks (Khoisan), unusual pitch
patterns (laryngeal consonants), breathy voice. For Gbe-family languages
(Adja, Ewe, Fon), the phonetic inventory is well within the space any decent
waveform codec covers.

## Papers to read

- Défossez et al., "Moshi / Mimi" — https://arxiv.org/abs/2410.00037 — the
  detailed paper on Mimi. §2 (audio tokenization) and §3 (split RVQ) are the
  relevant bits.
- Zeghidour et al., "SoundStream" — https://arxiv.org/abs/2107.03312 — earlier
  foundational codec, easier entry point.
- Kumar et al., "High-Fidelity Audio Compression with Improved RVQGAN (DAC)" —
  https://arxiv.org/abs/2306.06546 — a more recent and readable codec paper.

## Related files in this project

- `../../ideas/tts-models-guide.md` — longer-form overview including
  architectural comparisons.
- `../../experiments/tts/T1_sesame_csm_finetune/mimi_reconstruction_test.py` —
  run this to verify codec behavior on your own audio.
