# 07 — Tonal languages for ML people

Last updated: **2026-04-21**

Adja (like its Gbe-family cousins Ewe, Fon, Ge) is a **tonal language**. Tone
affects meaning: the syllable `ma` with a high tone means something different
from `ma` with a low tone. For TTS and ASR, this changes how we think about
what the model has to learn.

## Tones = F0 contours

Physically, "tone" is a pattern in the fundamental frequency (F0) over a
syllable. A high tone rises in pitch, a low tone falls, a mid tone stays flat.
On a spectrogram, tones are visible as slopes in the lowest (yellow) band.

For Adja specifically, the written orthography uses:
- `á`, `é`, `í`, `ó`, `ú` — acute accent = high tone
- `à`, `è`, `ì`, `ò`, `ù` — grave accent = low tone
- Unmarked vowels for mid tone

Plus combining accents in decomposed form (U+0300, U+0301) and caron (`ǎ`)
for contour tones.

## What this means for each layer

### For Layer 1 (text tokenizer)
Every tone-marked vowel is a separate character. If the tokenizer
byte-fragments them, the LM loses tone information in the text stream. Fix:
expand the vocab (see concept 01).

### For Layer 2 (LM backbone)
Ideally we want the LM to have seen tonal text before. Hence the
`orpheus-3b-zh-pretrain` variant for Orpheus — Mandarin has 4 tones and is
tonal, so the LM already has "tonal text → tonal audio token" priors. Our
comparison of EN / zh / fr Orpheus bases on the Ewe→Adja bridge will tell us
whether this tonal prior actually transfers across language families.

### For Layer 3 (audio codec)
Tones are ordinary acoustic features. A codec trained on any speech corpus
should preserve them, because it doesn't know tones from speech rhythm — both
are just F0 contours in the waveform. This is the test we run with
`mimi_reconstruction_test.py`.

## Why Gbe specifically

- **Gbe-family cross-lingual transfer**: Adja (ajg), Ewe (ewe), Fon (fon), Ge
  (gej) all share phonological structure. A model trained on Ewe should
  generalize to Adja with minimal extra data. This is the core hypothesis
  behind our T1/T3 Ewe-bridge experiments.
- **Shared orthography**: most Gbe languages use the same special characters
  (ɛ, ɔ, ŋ, ɖ), so a tokenizer expansion on one helps the others for free.
- **WaxalNLP coverage**: Google's WaxalNLP dataset gives us 15k labeled +
  183k unlabeled Ewe utterances. That's 10-100× our Adja data.

## What tonal sandhi does (a worry)

In some tonal languages, tones interact contextually — a low tone followed by
a low tone becomes a rising tone, etc. This is called "tonal sandhi" and it
complicates the text→audio mapping because the *written* tone doesn't always
match the *spoken* tone. Mandarin has it (3rd-tone sandhi). Gbe-family has
some contextual tone shifts but less than Mandarin. Not a front-burner worry
for us, but useful to know if the TTS outputs sound "wrong pitch" in ways
that could be a sandhi issue rather than a training issue.

## Further reading

- Ladefoged & Maddieson, "The Sounds of the World's Languages" — book, ~$80,
  worth it if you do speech research on multiple languages.
- Kisler, Reichel, Schiel, "Multilingual processing of speech via web services"
  (particularly the WebMAUS G2P tool) — has African language phoneme inventories.
- There's basically no ML paper that focuses on tonal languages specifically —
  the SOTA approach is still "use whatever state-of-the-art English model and
  fine-tune". Our experiments are meant to move that a little.
