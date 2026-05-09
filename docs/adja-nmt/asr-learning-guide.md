# ASR Learning Guide for NMT Practitioners

A progressive learning path for someone who already understands NLP, transformers, and NMT
but is new to speech recognition. Written specifically for Josue, who has experience
fine-tuning NLLB, mBART, and Gemini for Adja machine translation.

**How to use this guide**: Read sections 1-3 first (they build on each other). Sections 4-7
can be read in any order. Sections 8-9 are short and should be read before running experiments.
Section 10 maps everything to our specific experiments.

---

## 1. From Text to Speech: What Changes

You know how to go from a French sentence to an Adja sentence (NMT). Now imagine the input
is not a French sentence but a recording of someone *saying* the French sentence. That is
what ASR does -- but in reverse: it maps audio to text in the *same* language.

### Three things that make speech fundamentally different from text

**1. The input is continuous, not discrete.**
In NMT, your input is a sequence of tokens -- discrete symbols from a vocabulary. In ASR,
your input is a waveform: a 1D array of floating-point amplitude values sampled 16,000
times per second. A 5-second utterance is 80,000 numbers. There is no natural "vocabulary"
in the raw signal.

**2. There are no word boundaries.**
In text, spaces separate words. In speech, words flow into each other. The phrase "je suis"
sounds like one blob, not two separate units. The model must learn where one word ends and
another begins.

**3. Speed varies.**
The same sentence spoken slowly might produce 4 seconds of audio; spoken quickly, 2 seconds.
The same word might occupy 200ms in one utterance and 500ms in another. The model must be
invariant to speaking rate.

### What an ASR system does

```
Audio waveform (1D array, 16kHz)
    |
Feature extraction (convert to spectrogram)
    |
Encoder (processes acoustic features)
    |
Decoder / Output head (produces text)
```

This should look familiar: it is an encoder-decoder architecture, just like your NMT models.
The difference is that the "source language" is audio features instead of text tokens.

### The two paradigms

**CTC-based (Connectionist Temporal Classification)**
- The encoder reads the spectrogram and outputs one prediction per audio frame.
- A special "CTC loss" handles the alignment problem (more on this in Section 3).
- There is no autoregressive decoder. Output is produced in one pass.
- Analogy: imagine if your NMT model could only do non-autoregressive decoding --
  predicting all target tokens simultaneously, one per source position.
- Used by: wav2vec 2.0, MMS, XLS-R (our experiments C3, C4, C5).

**Attention-based (encoder-decoder)**
- Works almost identically to your NMT models.
- The encoder reads the spectrogram. The decoder autoregressively generates text tokens,
  attending to the encoder output via cross-attention.
- Used by: Whisper (our experiments C1, C2), LAS (our experiment B4).

**Hybrid (CTC + attention)**
- Many modern systems use both: CTC loss on the encoder output AND cross-entropy on the
  decoder output. The CTC helps the encoder learn good alignments; the decoder refines the
  output.
- Used by: ESPnet models, Conformer-based systems.

---

## 2. Audio Features 101

In NMT, your preprocessing is tokenization (text -> token IDs). In ASR, your preprocessing
is feature extraction (audio waveform -> spectrogram). This section explains that pipeline.

### The feature extraction pipeline

```
Raw waveform (16kHz, 1D array of amplitude values)
    |  [Short-Time Fourier Transform (STFT)]
    v
Spectrogram (2D: time x frequency)
    |  [Apply mel filterbank]
    v
Mel spectrogram (2D: time x 80 mel bins)
    |  [Take log]
    v
Log-mel spectrogram (2D: time x 80)  <-- this is what models actually eat
```

### What each step does

**Raw waveform -> Spectrogram (STFT)**

A waveform tells you the amplitude at each moment in time, but it does not directly tell you
which *frequencies* are present. The Short-Time Fourier Transform slides a small window
(typically 25ms) across the waveform, hopping 10ms at a time, and computes the frequency
content within each window.

The result is a 2D matrix:
- Horizontal axis: time (one column per 10ms hop)
- Vertical axis: frequency (from 0 Hz to 8000 Hz for 16kHz audio)
- Cell value: energy at that frequency at that time

A 5-second utterance at 10ms hop = 500 time frames. With 16kHz sampling and a 25ms window,
you get ~201 frequency bins. So the spectrogram is 500 x 201.

**Spectrogram -> Mel spectrogram**

Human hearing does not perceive frequencies linearly. We are much better at distinguishing
low frequencies (200 Hz vs 300 Hz sounds very different) than high frequencies (5000 Hz vs
5100 Hz sounds almost the same). The mel scale compresses high frequencies and expands low
frequencies to match human perception.

A mel filterbank applies 80 triangular filters to the frequency axis, converting the ~201
frequency bins into 80 mel bins. The spectrogram becomes 500 x 80.

Why 80? It is a convention that works well empirically. Whisper, wav2vec 2.0, and most
modern ASR systems use 80 mel bins.

**Mel spectrogram -> Log-mel spectrogram**

Energy values span a huge range (quiet sounds might be 0.001, loud sounds might be 1000).
Taking the logarithm compresses this range, making the features easier for neural networks
to process. This is similar to why you might use log-scaled learning rates.

### The analogy to NMT

| NMT | ASR |
|-----|-----|
| Raw text | Raw waveform |
| Tokenizer (BPE/SentencePiece) | Feature extractor (STFT + mel + log) |
| Token embeddings (seq_len x d_model) | Log-mel spectrogram (time_frames x 80) |
| Encoder input | Encoder input |

The log-mel spectrogram is to ASR what token embeddings are to NMT: the first real
representation the model sees.

### Practical note

In our experiments, feature extraction is handled by the framework:
- `torchaudio.transforms.MelSpectrogram` for from-scratch models (B1)
- Built into the model processor for HuggingFace models (C2-C5)

You do not need to implement this yourself, but understanding it helps you debug when
things go wrong (wrong sampling rate, wrong number of mel bins, etc.).

### Further reading

- PyTorch Audio feature extraction tutorial:
  <https://pytorch.org/audio/stable/tutorials/audio_feature_extractions_tutorial.html>

---

## 3. CTC (Connectionist Temporal Classification) -- The Key Concept

CTC is the single most important concept to understand for our experiments. It shows up in
B1 (BiLSTM-CTC), B2 (Conformer-CTC), B3 (Transformer-CTC), and is the output head for
all wav2vec 2.0 / MMS / XLS-R fine-tuning (C3, C4, C5).

### The alignment problem

Consider the word "bonne" (5 characters). An utterance of this word might produce 50 audio
frames (500ms at 10ms per frame). The encoder processes these 50 frames and outputs 50
predictions -- one per frame.

But we only want 5 output characters. Which of the 50 frames corresponds to "b"? Which to
"o"? We do not know, and manually aligning audio frames to characters for every training
example would be impossibly expensive.

**NMT analogy**: Imagine you had to train a translation model but you did not know which
source words corresponded to which target words. In NMT, attention learns soft alignments.
CTC takes a different approach.

### How CTC solves this

CTC introduces a special **blank token** (written as `-` or `<blank>`). At each audio frame,
the model predicts either a real character or the blank token. The blank means "nothing here
yet" or "still on the same character."

For the word "bonne", valid CTC output sequences (over 50 frames) include:

```
---bbbb-oooo-nnn-nnn-eee---    (valid: collapses to "bonne")
b---ooo---nn---nn---eee-----    (valid: collapses to "bonne")
bbbbbbooonnnnnnnnnneeeeee---    (valid: collapses to "bone" -- WAIT, wrong!)
```

The collapsing rule: (1) remove all blanks, (2) collapse consecutive repeated characters.

So `bbbb-oooo-nnn-nnn-eee` -> remove blanks -> `bbbboooonnnnneee` -> collapse repeats -> `bone`.

That is wrong! To get the double "n" in "bonne", there must be at least one blank *between*
the two n's:

```
---bbbb-oooo-nnn-n-nnn-eee---
                ^-- this blank separates the two n's
```

After removing blanks: `bbbboooonnnnnnneee`... wait, that still collapses to `bone`.
Let me be more precise:

```
---bbbb-oooo-nn-nn-eee---
```

Remove blanks: `bbbboooonnnnneee` -> collapse repeats: `bone`. Still wrong!

The correct way to emit "bonne" with CTC:

```
---bbbb-oooo-nnn--nnn-eee---
```

Remove blanks: `bbbboooonnnnnnneee` -> collapse repeats: `bone`. Hmm.

Actually, let me clarify the rule precisely. CTC operates in two steps:
1. First collapse consecutive identical characters: `bbbboooonnnnnnneee` -> `bone`
2. Then remove blank symbols

No -- the standard formulation is:
1. Remove all blank tokens
2. Collapse consecutive identical characters

So for `nnn-nnn`: remove blanks -> `nnnnnn` -> collapse -> `n`. That gives a single "n".

To get "nn", you need the blank to *separate* two groups during collapsing, but blanks are
removed first. The actual mechanism: CTC collapses runs of the same character, and a blank
between two identical characters prevents them from being collapsed into one.

The precise algorithm:
1. Collapse consecutive identical tokens (including blanks): `aaa--bbb` -> `a-b`
2. Remove blanks: `a-b` -> `ab`

So `nnn-nnn` -> collapse identical -> `n-n` -> remove blanks -> `nn`. That gives "nn".
And `nnnnnn` (no blank) -> collapse -> `n`. That gives a single "n".

**This is why the blank token matters**: it is the only way to produce repeated characters.

### The CTC blank vs NMT padding

You already know the `<pad>` token in NMT -- it fills unused positions in batched sequences
and is ignored during loss computation. The CTC blank is similar in spirit (it occupies frames
where "nothing meaningful" happens) but different in mechanics (it actively participates in
the collapsing algorithm and is part of the loss).

A closer analogy: the blank is like the "hold" state in a state machine. The model stays on
blank while it is "between" characters, and emits a real character when it is confident about
what comes next.

### Training: marginalizing over all alignments

During training, CTC does not pick one alignment. Instead, it sums the probabilities of
*all* possible alignments that produce the correct output text. This is computed efficiently
using dynamic programming (the forward-backward algorithm -- similar to the Viterbi algorithm
if you have seen HMMs).

You do not need to implement this. `torch.nn.CTCLoss` handles it. But knowing this is why
CTC training can feel slow: it considers exponentially many alignment paths.

### Decoding: getting text from CTC output

**Greedy decoding**: At each frame, take the argmax character. Then apply the collapse +
blank-removal rule. This is fast but can make errors.

```python
# Pseudocode for greedy CTC decoding
predictions = model(spectrogram)          # shape: (time_frames, vocab_size)
best_chars = predictions.argmax(dim=-1)   # shape: (time_frames,)
# Collapse consecutive duplicates, then remove blanks
output_text = ctc_collapse(best_chars)
```

**Beam search with language model**: Keep the top-k partial hypotheses at each frame.
Optionally re-score with an external language model (like the n-gram LM in experiment D4).
This is more expensive but significantly improves accuracy, especially for low-resource
languages where the acoustic model alone is noisy.

### Key references

- **Original CTC paper**: Graves et al. (2006). Connectionist Temporal Classification:
  Labelling Unsegmented Sequence Data with Recurrent Neural Networks.
  <https://www.cs.toronto.edu/~graves/icml_2006.pdf>

- **Best visual explanation** (highly recommended): Sequence Modeling With CTC.
  <https://distill.pub/2017/ctc/>

- **Video**: Awni Hannun's CTC tutorial explains the forward-backward algorithm step by step.

---

## 4. SpecAugment -- Data Augmentation for Speech

You know about dropout (randomly zero out activations during training). SpecAugment is the
speech equivalent, but applied to the *input features* rather than hidden layers.

### What it does

SpecAugment applies two types of masks directly to the log-mel spectrogram:

1. **Frequency masking**: Zero out a random contiguous band of frequency bins.
   Example: mask mel bins 20-35 (out of 80). This forces the model to recognize speech
   even when certain frequency ranges are missing.

2. **Time masking**: Zero out a random contiguous span of time frames.
   Example: mask frames 100-120 (out of 500). This forces the model to handle missing
   audio segments, simulating brief noise or dropouts.

```
Original spectrogram:         After SpecAugment:
+------------------+          +------------------+
|  ####   ###  ##  |          |  ####   ###  ##  |
|  ## ## ## ## ##   |          |  ## ## ## ## ##   |
| ###  ###  ## ### |   --->   | ████████████████ |  <-- frequency mask
|  ##  # ## ##  ## |          |  ##  # ## ##  ## |
|  ##  #  ###   ## |          |  ##  #  ###   ## |
+------------------+          +----████----------+
                                    ^-- time mask
```

### Why it works

This is essentially input-level regularization. With only ~1,280 training utterances
(80% of our 1.6k data), overfitting is a serious risk. SpecAugment:

- Prevents the model from memorizing specific frequency patterns
- Simulates real-world audio degradation (noise, bandwidth limitations)
- Forces the model to use contextual information from surrounding frames

**NMT analogy**: SpecAugment is like randomly replacing some input tokens with `<unk>`
during training. It forces the model to rely on context rather than memorizing specific
input patterns.

### In our experiments

SpecAugment is used in B1 (BiLSTM-CTC) and applied automatically by HuggingFace's
feature extractors in C2-C5. The typical configuration:

- 2 frequency masks of width up to 27
- 2 time masks of width up to 100 frames (or 5% of the utterance, whichever is smaller)

### Reference

- Park, D. S., et al. (2019). SpecAugment: A Simple Data Augmentation Method for ASR.
  <https://arxiv.org/abs/1904.08779>

---

## 5. Speed Perturbation

An even simpler data augmentation technique: change the playback speed of the audio.

### What it does

For each training utterance, create up to three versions:
- **0.9x speed**: slower, lower pitch -- simulates a slow speaker
- **1.0x speed**: original
- **1.1x speed**: faster, higher pitch -- simulates a fast speaker

This effectively **triples your training data** with zero annotation cost. The transcription
stays the same; only the audio changes.

### Why it works

Different speakers naturally vary in speaking rate. Speed perturbation teaches the model
to handle this variation. For Adja specifically, this is valuable because:

- Tonal distinctions may become clearer or more ambiguous at different speeds
- We have very little data (~1.6k utterances), so any augmentation helps

**NMT analogy**: This is like paraphrasing your source sentences to create additional
training examples, except here we "paraphrase" the acoustic signal.

### In our experiments

Speed perturbation is applied in B1 at data loading time (see `train.py`). For HuggingFace
experiments (C2-C5), it can be applied via `torchaudio.functional.speed` in the data
collator.

### Reference

- Ko, T., et al. (2015). Audio Augmentation for Speech Recognition.
  <https://www.danielpovey.com/files/2015_interspeech_augmentation.pdf>

---

## 6. Self-Supervised Learning for Speech (wav2vec, XLS-R, MMS)

This is the section that matters most for understanding why fine-tuning pretrained models
(experiments C3-C5) is expected to massively outperform training from scratch (B1-B4).

### The core idea

**wav2vec 2.0 is to speech what BERT is to text.**

You already understand BERT-style pretraining for NMT: train on massive amounts of
unlabeled text with a masked language model objective, then fine-tune on your downstream
task. wav2vec 2.0 does exactly the same thing, but for audio:

| | Text (BERT) | Speech (wav2vec 2.0) |
|---|---|---|
| Pretraining data | Unlabeled text (Wikipedia, etc.) | Unlabeled audio (LibriSpeech, etc.) |
| Pretraining task | Masked token prediction | Masked audio frame prediction |
| What it learns | Language structure, semantics | Acoustic patterns, phonemes, prosody |
| Fine-tuning | Add task head, train on labeled data | Add CTC head, train on transcribed audio |

### The pretraining process

1. **Input**: Raw waveform (no spectrogram needed -- the model learns its own features).
2. **Feature encoder**: A CNN converts the waveform into a sequence of frame-level features.
3. **Masking**: Random spans of frames are masked (like BERT's `[MASK]` tokens).
4. **Transformer encoder**: Processes the masked sequence.
5. **Contrastive loss**: The model must identify the correct masked frame from a set of
   distractors (quantized codebook entries).

This is trained on hundreds of thousands of hours of unlabeled audio. The model learns
rich representations of speech without ever seeing a single transcription.

### Fine-tuning for ASR

After pretraining, you add a **CTC head** (a simple linear layer from the Transformer
output to the character vocabulary) and fine-tune on your small labeled dataset.

```
[Pretrained wav2vec 2.0 encoder]
    |
    v
Linear projection (hidden_dim -> vocab_size)
    |
    v
CTC loss against character targets
```

This is almost embarrassingly simple -- and it works phenomenally well, especially for
low-resource languages.

### The model family

| Model | Pretraining data | Languages | Params | Our experiment |
|-------|-----------------|-----------|--------|----------------|
| wav2vec 2.0 | 960h LibriSpeech (English) | 1 | 300M | C5 |
| XLS-R | 436K hours, 128 languages | 128 | 300M-2B | C4 |
| MMS | 491K hours, 1,100+ languages | 1,100+ | 1B | C3 |

**Why MMS is our strongest candidate**: It was pretrained on audio from 1,100+ languages,
including many tonal African languages. Even if Adja is not directly in the pretraining
data, the model has learned acoustic patterns from closely related Gbe languages, tonal
systems, and West African phonology. This gives it a huge head start.

**Why wav2vec 2.0 (English-only) is our control**: By comparing C5 (English-only pretraining)
against C4 (128-language) and C3 (1,100+ languages), we can measure how much multilingual
pretraining helps. If English-only wav2vec 2.0 performs nearly as well as MMS on Adja, that
tells us the general speech representations are more important than language-specific ones.
If MMS is much better, it means cross-lingual transfer is crucial.

### Why this matters for low-resource Adja

Without pretraining (experiments B1-B4), your model must learn everything from ~1,280
training utterances:
- What speech sounds like in general
- What Adja sounds like specifically
- How to map sounds to characters

With pretraining (experiments C3-C5), the model already knows what speech sounds like in
general. It only needs to learn the Adja-specific mapping. This is dramatically easier.

**NMT analogy**: This is exactly like your experience fine-tuning NLLB for Adja. NLLB was
pretrained on 200 languages. Even though Adja was not a primary training language, the model
had learned general translation patterns that transferred. The same principle applies here.

### Key references

- **wav2vec 2.0**: Baevski, A., et al. (2020). wav2vec 2.0: A Framework for Self-Supervised
  Learning of Speech Representations.
  <https://arxiv.org/abs/2006.11477>

- **XLS-R**: Babu, A., et al. (2021). XLS-R: Self-supervised Cross-lingual Speech
  Representation Learning at Scale.
  <https://arxiv.org/abs/2111.09296>

- **MMS**: Pratap, V., et al. (2023). Scaling Speech Technology to 1,000+ Languages.
  <https://jmlr.org/papers/v25/23-1318.html>

- **HuggingFace blog** (hands-on fine-tuning tutorial, highly recommended):
  <https://huggingface.co/blog/fine-tune-wav2vec2-english>

---

## 7. Whisper -- The Generalist

Whisper is the other major model family in our experiments (C1, C2). It takes a completely
different approach from wav2vec 2.0 / MMS.

### Training approach

While wav2vec 2.0 uses self-supervised pretraining on unlabeled audio, Whisper uses
**weakly supervised training** on 680,000 hours of audio paired with approximate
transcriptions scraped from the web (subtitles, captions, etc.).

**NMT analogy**: wav2vec 2.0's approach is like pretraining mBART on monolingual text.
Whisper's approach is like training NLLB directly on noisy parallel data scraped from
the web (like CCMatrix/CCAligned). Both work, but the philosophies differ.

### Architecture

Whisper's architecture should feel immediately familiar to you:

```
Log-mel spectrogram (80 dims)
    |
Convolutional feature encoder (2 conv layers)
    |
Transformer encoder (positional encoding + self-attention)
    |
    +----- cross-attention -----+
    |                           |
    v                           v
Transformer decoder (causal self-attention + cross-attention)
    |
Token predictions (BPE vocabulary)
```

This is almost identical to mBART or NLLB, except:
- The input is a spectrogram instead of token embeddings
- The first layers are convolutions instead of an embedding lookup
- The decoder is autoregressive (just like your NMT models)

Whisper does NOT use CTC. It uses standard cross-entropy loss with teacher forcing, exactly
like NMT training.

### Multitask design

Whisper uses special tokens to control behavior, similar to how NLLB uses language tokens:

```
<|startoftranscript|> <|en|> <|transcribe|> <|notimestamps|>
```

This tells the model: "transcribe this audio into English." By changing the tokens:

```
<|startoftranscript|> <|fr|> <|translate|> <|notimestamps|>
```

You get: "translate this audio into French." Whisper supports transcription (ASR) and
translation (speech-to-text translation) in one model.

### Why zero-shot will fail on Adja (experiment C1)

Whisper was trained on 99 languages, heavily skewed toward English and European languages.
Adja (ajg) is almost certainly not in the training data. When Whisper encounters Adja audio,
it will likely:

- Detect the language as French (geographically plausible) or Ewe/Fon (phonetically similar)
- Produce transcriptions in the detected language, not in Adja
- Achieve near-100% WER

Experiment C1 quantifies this failure. The WER/CER numbers become the ceiling that
fine-tuning (C2) must beat.

### Fine-tuning Whisper for Adja (experiment C2)

Fine-tuning Whisper for Adja follows the same approach as fine-tuning NLLB:

1. Load the pretrained model
2. Optionally freeze the encoder (it already understands speech well)
3. Train the decoder to produce Adja text instead of English/French
4. Use the Adja character/BPE vocabulary

The decoder's cross-attention mechanism learns to map the encoder's speech representations
to Adja characters. This is exactly how attention works in your NMT models, just with a
different source modality.

### Key references

- **Whisper paper**: Radford, A., et al. (2022). Robust Speech Recognition via Large-Scale
  Weak Supervision.
  <https://cdn.openai.com/papers/whisper.pdf>

- **HuggingFace fine-tuning guide** (practical, step-by-step):
  <https://huggingface.co/blog/fine-tune-whisper>

---

## 8. Evaluation Metrics

### WER (Word Error Rate)

WER measures the edit distance between the predicted word sequence and the reference word
sequence, normalized by the reference length:

```
WER = (Substitutions + Insertions + Deletions) / Reference_words
```

Example:
```
REF: "moi je suis content"       (4 words)
HYP: "moi suis je contentu"      (4 words)
     "moi" correct, "suis" and "je" swapped (2 subs), "contentu" wrong (1 sub)
WER = 3/4 = 75%
```

WER can exceed 100% if the model inserts many extra words.

**NMT analogy**: WER is like computing word-level edit distance on your NMT output.
You are familiar with BLEU (precision-based) and chrF (character n-gram). WER is
edit-distance-based, which penalizes differently.

### CER (Character Error Rate)

Same as WER but at the character level:

```
CER = (Char_substitutions + Char_insertions + Char_deletions) / Reference_chars
```

### Why CER matters more for Adja

For Adja specifically, CER is the more informative metric for three reasons:

1. **Tone marks**: "ma" vs "ma" (with a tone diacritice) are one character apart. WER would
   count the entire word as wrong; CER captures that the model was *almost* right.

2. **Special characters**: Adja uses characters like open-mid vowels and a velar nasal that
   standard models often confuse. CER shows whether the model gets these specific characters
   right.

3. **Small vocabulary**: With limited training data, the model may learn character patterns
   before it learns word patterns. CER tracks this intermediate progress.

In our experiment registry, we report both WER and CER, but CER is the primary metric for
model selection (the B1 training script selects the best checkpoint by dev CER).

### Computing metrics

Our shared metrics module (`experiments/asr/shared/metrics.py`) handles WER and CER
computation using the `jiwer` library and applies NFC normalization before scoring.

---

## 9. ASR for Tonal Languages -- Special Considerations

Adja is a tonal language. This is both a challenge and what makes our research interesting.

### Why tone matters

In Adja and other Gbe languages, tone marks change meaning. The same consonant-vowel
sequence with different tones can mean completely different things. This means that
getting the tone wrong is not just a cosmetic error -- it is a semantic error.

### Why standard ASR struggles with tone

Tone is primarily carried by the fundamental frequency (F0) of the voice -- the pitch
contour. Standard ASR features (mel spectrograms) do capture some F0 information, but
it is mixed in with other spectral information and is not explicitly represented.

Most pretrained ASR models were primarily trained on non-tonal languages (English, Spanish,
French, etc.) where pitch carries prosodic information (question intonation, emphasis) but
not lexical meaning. These models have learned to be somewhat *invariant* to pitch -- which
is exactly the wrong thing for a tonal language.

### Techniques for tonal ASR

Several approaches have been explored in the literature:

1. **F0 features**: Extract pitch contour explicitly and concatenate with mel spectrogram
   features. This gives the model direct access to tone information.

2. **Tone-aware tokenization**: Separate tone marks from base characters in the output
   vocabulary, making it easier for the model to learn tone patterns independently.

3. **Multi-task learning**: Train the model to simultaneously predict characters AND tone
   categories. The auxiliary tone prediction task forces the encoder to preserve tonal
   information.

4. **Data augmentation**: Be careful with speed perturbation -- changing speed also changes
   pitch, which can distort tonal information. Consider pitch-preserving speed changes.

### What this means for our experiments

- Our from-scratch baselines (B1-B4) must handle Adja tone from scratch with very little
  data. We expect high CER, especially on tone-marked characters.
- Pretrained models (C3-C5) may have an advantage if they were pretrained on tonal languages.
  MMS (1,100+ languages) likely saw many tonal languages; wav2vec 2.0 (English only) did not.
- Comparing tone-mark accuracy across models will be one of our most interesting analyses.
- Experiment D4 (Whisper + n-gram LM) may help: the language model can learn tone patterns
  from text even if the acoustic model struggles.

---

## 10. Our Experiment Roadmap

Here is how every experiment in our registry maps to the concepts above.

### Group B: From-Scratch Baselines

These experiments use NO pretrained components. They establish the floor performance --
what can a model learn from just our ~1,280 training utterances?

| Exp | Model | Key concepts | What we learn |
|-----|-------|-------------|---------------|
| B1 | BiLSTM-CTC | Sections 2, 3, 4, 5 | Simplest possible neural ASR. How much can a small recurrent model learn from scratch? Sets the absolute floor. |
| B2 | Conformer-CTC | Sections 2, 3, 4 | Adds convolution + self-attention to the encoder. Tests whether a better encoder architecture helps when data is limited. |
| B3 | Transformer-CTC | Sections 2, 3 | Pure self-attention encoder, no convolution. Isolates the effect of attention vs convolution+attention. |
| B4 | LAS (Listen-Attend-Spell) | Section 7 (architecture), Section 3 (comparison) | Attention-based decoding instead of CTC. Tests the CTC-vs-attention paradigm on low-resource Adja. |

**The question Group B answers**: How bad is from-scratch ASR on ultra-low-resource tonal
Adja? (Expected: quite bad, CER 40-70%+.)

### Group C: Pretrained Model Fine-Tuning

These experiments leverage massive pretraining and test how much transfer learning helps.

| Exp | Model | Key concepts | What we learn |
|-----|-------|-------------|---------------|
| C1 | Whisper (zero-shot) | Section 7 | What does a large model produce with NO adaptation to Adja? Establishes the error ceiling. |
| C2 | Whisper (fine-tuned) | Section 7 | How much does fine-tuning an attention-based model help? Direct comparison with C1. |
| C3 | MMS (fine-tuned) | Section 6 | The strongest candidate per PazaBench. 1,100+ language pretraining + CTC head. Our best bet. |
| C4 | XLS-R (fine-tuned) | Section 6 | 128-language pretraining. Tests whether massive multilingual SSL helps Adja. |
| C5 | wav2vec 2.0 (fine-tuned) | Section 6 | English-only pretraining. Control experiment: does the language of pretraining matter? |
| C6 | SeamlessM4T ASR | Sections 6, 7 | Multimodal model from Meta. Tests a different model family. |

**The question Group C answers**: How much does pretraining help, and which pretrained model
transfers best to Adja? (Expected: MMS > XLS-R > wav2vec 2.0 >> from-scratch.)

**Key comparisons**:
- C3 vs C5: multilingual pretraining vs English-only pretraining
- C3 vs C4: 1,100 languages vs 128 languages
- C2 vs C3: attention-based (Whisper) vs CTC-based (MMS) fine-tuning
- C1 vs C2: zero-shot vs fine-tuned (measures the value of our labeled data)

### Group D: Advanced Techniques and Efficiency

These experiments test specific hypotheses and practical concerns.

| Exp | Model | Key concepts | What we learn |
|-----|-------|-------------|---------------|
| D1 | W2v-BERT 2.0 | Section 6 | Newer SSL model designed for data efficiency. Tests whether newer pretraining approaches help. |
| D2 | Moonshine | Efficiency | Tiny model for edge deployment. Can a small model work for Adja ASR on a phone? |
| D3 | Distil-Whisper | Section 7, efficiency | Distilled Whisper. Tests quality-vs-speed tradeoff. |
| D4 | Whisper + n-gram LM | Section 3 (beam search + LM) | Adds a language model on top of the best Whisper checkpoint. Tests whether an Adja text LM can fix acoustic errors. |

**The question Group D answers**: Can we make ASR faster/smaller/better with post-hoc
techniques? (D4 is especially interesting for tonal accuracy.)

### How they build on each other

```
B1-B4 (from scratch)
  |
  |  "Here is the floor performance"
  v
C1 (zero-shot Whisper)
  |
  |  "Here is what pretrained models give for free"
  v
C2-C6 (fine-tuned pretrained)
  |
  |  "Here is what fine-tuning on our labeled data adds"
  v
D1-D4 (advanced techniques)
  |
  |  "Here is what targeted improvements can do on top"
  v
Final comparison: which approach + model gives the best Adja ASR?
```

---

## 11. Recommended Reading Order

### Start here (1-2 days)

These are beginner-friendly and build the core intuitions:

1. **This guide** -- Read sections 1-3 carefully. Sections 4-9 can be skimmed for now.

2. **Distill.pub CTC explainer** -- The best visual explanation of CTC. Read this until the
   collapsing mechanism clicks.
   <https://distill.pub/2017/ctc/>

3. **HuggingFace Audio Course, Chapter 1-3** -- Interactive, hands-on introduction to audio
   processing and ASR. Covers spectrograms, feature extraction, and ASR architectures.
   <https://huggingface.co/learn/audio-course>

4. **HuggingFace blog: Fine-tune wav2vec2 for English ASR** -- Walk through a complete
   fine-tuning pipeline. The code patterns are nearly identical to what our C3-C5 scripts do.
   <https://huggingface.co/blog/fine-tune-wav2vec2-english>

### Deeper understanding (1 week)

Once the basics click, these give you the full picture:

5. **Whisper paper** -- Read sections 1-3 (approach) and section 4 (experiments). Skip the
   appendix on first read. Connect the architecture to what you know about mBART.
   <https://cdn.openai.com/papers/whisper.pdf>

6. **MMS paper** -- Focus on section 2 (approach) and section 4 (experiments). Pay attention
   to the low-resource results -- they are directly relevant to Adja.
   <https://jmlr.org/papers/v25/23-1318.html>

7. **wav2vec 2.0 paper** -- The foundational self-supervised speech paper. Read section 2
   (model) and section 4 (experiments). The contrastive loss is analogous to BERT's MLM.
   <https://arxiv.org/abs/2006.11477>

8. **HuggingFace blog: Fine-tune Whisper** -- Practical guide for Whisper fine-tuning.
   Compare the approach with the wav2vec2 blog post above.
   <https://huggingface.co/blog/fine-tune-whisper>

9. **SpecAugment paper** -- Short and easy to read. The results section shows dramatic
   improvements from a simple technique.
   <https://arxiv.org/abs/1904.08779>

### Reference papers (when you need details)

10. **Original CTC paper** (Graves et al., 2006) -- Dense mathematical treatment of CTC.
    Read if you want to understand the forward-backward algorithm.
    <https://www.cs.toronto.edu/~graves/icml_2006.pdf>

11. **XLS-R paper** -- Cross-lingual self-supervised speech representations. Relevant for
    understanding experiment C4.
    <https://arxiv.org/abs/2111.09296>

12. **Speed perturbation paper** (Ko et al., 2015) -- Short paper, establishes the 0.9/1.0/1.1
    speed factors as standard practice.
    <https://www.danielpovey.com/files/2015_interspeech_augmentation.pdf>

### Courses and documentation (ongoing reference)

- **Stanford CS224S** -- Full speech and language processing course. Good for filling gaps
  in understanding of classical ASR concepts (HMMs, GMMs, language models).
  <http://web.stanford.edu/class/cs224s/>

- **HuggingFace Audio Course** -- Complete course with code notebooks. Chapters 5-7 cover
  fine-tuning and evaluation in depth.
  <https://huggingface.co/learn/audio-course>

- **ESPnet tutorial notebooks** -- Useful if we move to ESPnet-based experiments. The Colab
  notebooks walk through training recipes.
  <https://github.com/espnet/espnet#notebooks>

- **Kaldi documentation** -- The classical ASR toolkit. Useful for understanding concepts
  like WFST decoding, language model integration, and n-gram LMs (relevant to experiment D4).
  <https://kaldi-asr.org/doc/>

---

## Quick Reference: Concept Map

When you encounter a concept in the code or papers, here is where to look:

| Concept | Guide section | Appears in experiments |
|---------|--------------|----------------------|
| Log-mel spectrogram | Section 2 | All (B1-D4) |
| CTC loss / decoding | Section 3 | B1, B2, B3, C3, C4, C5 |
| SpecAugment | Section 4 | B1, C2-C5 |
| Speed perturbation | Section 5 | B1 |
| Self-supervised pretraining | Section 6 | C3, C4, C5, D1 |
| Encoder-decoder attention | Section 7 | B4, C1, C2, C6 |
| WER / CER | Section 8 | All (evaluation) |
| Tonal considerations | Section 9 | All (analysis) |
| Beam search + LM | Section 3 | D4 |

---

## 12. Lessons From Actually Running Experiments (April 2026)

This section documents what we learned the hard way from submitting jobs to
HuggingFace Jobs. These are not in any textbook.

### 12.1. Zero-Shot Whisper on Unseen Languages = Spectacular Hallucinations

When we ran Whisper (tiny, small, large-v3) zero-shot on Adja:
- **whisper-tiny** output "banana", "Let!", "win win win..." (WER: 1208%)
- **whisper-small** output Georgian script (ლლლ...), "www.www.www..." (WER: 101%)

**Why this happens**: Whisper was trained on 680K hours of web audio, but Adja was
almost certainly not in the training data. When Whisper encounters audio it cannot
recognize, it hallucinates tokens from its training distribution — English words,
URLs, repeated characters, or even characters from other scripts. This is similar
to how an NMT model generates fluent-but-wrong translations for out-of-domain input.

**What this means for research**: Zero-shot WER > 100% is a clear sign the language
is truly out-of-distribution. Any improvement from fine-tuning is real signal, not
noise. This makes our fine-tuning experiments scientifically interesting.

**Paper to read**: [Hallucination in Whisper](https://arxiv.org/abs/2402.08846) — 
Koenecke et al. 2024, documents systematic hallucination patterns in Whisper,
especially for underrepresented languages.

### 12.2. CTC Collapse: When the Model Outputs All Blanks

Our first C4 (XLS-R) and C5 (wav2vec2) runs showed:
- Loss dropped to 0.0000 after epoch 1
- CER stayed at 100%
- Model output was all garbage characters or empty strings

This is **CTC collapse** — the model learns that the easiest way to minimize CTC loss
is to predict the blank token for every frame. Once it collapses, the gradient signal
disappears (loss = 0) and it never recovers.

**Root cause in our case**: We were pre-padding audio arrays before passing them to
the `Wav2Vec2FeatureExtractor`. This corrupted the attention masks — the model saw
silence as real audio and couldn't learn meaningful alignments.

**Fix**: Pass variable-length audio arrays to the feature extractor and let it
handle padding internally. The extractor computes correct attention masks.

**Debugging CTC collapse**: If loss goes to 0 early and output is blank:
1. Check attention masks — are they correct?
2. Check that `feat_length >= target_length` for all samples
3. Lower the learning rate (CTC is sensitive to LR)
4. Check that blank token index matches what CTC expects (usually 0)
5. Try gradient accumulation instead of large batch sizes

**Papers/resources on CTC collapse**:
- [Practical CTC Training Tips](https://distill.pub/2017/ctc/) — Distill article,
  has a section on training stability
- [wav2vec 2.0 fine-tuning gotchas](https://huggingface.co/blog/fine-tune-wav2vec2-english) —
  HuggingFace blog, mentions freezing the feature encoder to prevent collapse

### 12.3. CUDA Driver Compatibility on Cloud GPUs

HuggingFace Jobs A100 machines had CUDA driver 12.0.9, but PyTorch 2.8 (latest)
requires newer drivers. The symptom: `torch.cuda.is_available()` returns False, and
the 1B-param model silently falls back to CPU (training would take days).

**Fix**: Pin PyTorch to a version compatible with the cloud's CUDA driver:
```python
# /// script
# dependencies = ["torch==2.5.1", "torchaudio==2.5.1"]
# ///
```

**Lesson**: Always check `Device: cuda` in the first few log lines. If it says `cpu`,
something is wrong — don't let a 1B model train on CPU.

### 12.4. CTC Loss Doesn't Support float16 on CUDA

MMS (1B params) doesn't fit in float32 on an A10G (24GB). We tried `model.half()` to
use float16, but PyTorch's `CTCLoss` throws:
```
NotImplementedError: "ctc_loss_cuda" not implemented for 'Half'
```

**Fix**: Keep the model in fp16 for the forward pass (saves VRAM), but cast the
log-probs to float32 for the loss computation:
```python
loss = ctc_loss_fn(log_probs.float(), targets, output_lens, target_lens)
```

This is a mixed-precision pattern: compute in fp16, loss in fp32. The backward pass
auto-casts gradients back to fp16.

**Alternative**: Use PyTorch AMP (`torch.cuda.amp.autocast`) which handles this
automatically, but manual casting is simpler for a custom training loop.

### 12.5. MMS 1B Needs A100 (80GB), Not A10G (24GB)

Even with fp16, MMS 1B (965M params) OOM'd on A10G (24GB VRAM). The model weights
fit (~2GB in fp16) but the activations during the forward pass through 48 transformer
layers blow up memory.

**Rule of thumb for GPU memory**:
- 300M params (XLS-R, wav2vec2): A10G (24GB) is fine
- 1B params (MMS): needs A100 (80GB) or multi-GPU
- Whisper-small (244M): A10G is fine
- Whisper-large-v3 (1.5B): needs A100 or fp16 on A10G with small batch

**Cost reference** (HuggingFace Jobs, April 2026):
- A10G-large: $1.50/hr
- A100-large: $2.50/hr
- For a 3-hour training run, going from A10G to A100 costs $3 extra

---

## 13. Additional Reading: Low-Resource ASR

### Benchmarks and Surveys (read these for context)

- **PazaBench: Benchmarking ASR for African Languages** (2025)
  First comprehensive ASR benchmark for 39 African languages, 52 models.
  Key finding: MMS is best with <1 hour of data.
  <https://arxiv.org/html/2512.10968v1>

- **WAXAL: A Large-Scale Dataset for African Language ASR** (Google, 2026)
  21 African languages including Ewe, 1,250 hours, CC-BY-4.0.
  <https://huggingface.co/datasets/google/WaxalNLP>

- **AfricaNLP Workshop at ACL 2025**
  Primary venue for African language NLP/speech papers.
  <https://africanlp.masakhane.io/>

### Tonal Language ASR (critical for Adja)

- **How ASR Models Fail on Tone** (SIGTYP 2025)
  Contour tones collapse toward level tones in standard models.
  <https://aclanthology.org/2025.sigtyp-1.11/>

- **wav2vec 2.0 for Yoruba ASR** (ACM TALLIP 2025)
  Fine-tuning reduced WER from 28% to 17% with improved tone recognition.
  Directly applicable approach for Adja.
  <https://dl.acm.org/doi/10.1145/3690384>

- **Building an Ewe Language Dataset** (ICNLSP 2025)
  1,130 hours of Ewe speech data. Ewe is Adja's closest well-resourced relative.
  <https://aclanthology.org/2025.icnlsp-1.32.pdf>

### Data Augmentation for Low-Resource

- **Frustratingly Easy Data Augmentation for Low-Resource ASR** (2025)
  LLM text generation + TTS synthesis = 14.3% WER reduction.
  <https://arxiv.org/abs/2509.15373>

- **ASR from a Spoken Dictionary** (2025)
  Bootstrap ASR from word-level recordings alone. Practical for
  truly under-resourced languages.
  <https://arxiv.org/abs/2510.04832>

### Cross-Lingual Transfer

- **Meta-Adaptable Adapters for Low-Resource ASR** (2024)
  Meta-learning adapters across language families, tested on 31 languages.
  <https://www.sciencedirect.com/science/article/abs/pii/S0925231024012645>

- **Whisper + n-gram Language Model** (2025)
  Adding even a simple n-gram LM to fine-tuned Whisper gives up to 51% WER reduction.
  <https://arxiv.org/abs/2503.23542>

### Efficient / Small Models

- **Moonshine: Tiny ASR** (2024)
  27M params, outperforms Whisper Tiny by 48%. Relevant for edge deployment.
  <https://arxiv.org/abs/2410.15608>

- **Distil-Whisper** (Gandhi et al., 2023)
  Distilled Whisper — 6x faster, 49% fewer params, within 1% WER.
  <https://arxiv.org/abs/2311.00430>

### Practical Guides and Tutorials

- **HuggingFace Audio Course** — Complete course with code notebooks.
  Start with Chapter 5 (ASR fine-tuning).
  <https://huggingface.co/learn/audio-course>

- **HuggingFace: Fine-Tune wav2vec2 for English ASR** — Step-by-step blog
  post. The same approach works for any language with a CTC head.
  <https://huggingface.co/blog/fine-tune-wav2vec2-english>

- **HuggingFace: Fine-Tune Whisper** — Step-by-step for encoder-decoder ASR.
  <https://huggingface.co/blog/fine-tune-whisper>

- **Stanford CS224S: Spoken Language Processing** — Full university course.
  Covers classical ASR (HMMs, GMMs) through modern neural approaches.
  <http://web.stanford.edu/class/cs224s/>

- **ESPnet Tutorial Notebooks** — Interactive notebooks for training ASR
  models with ESPnet recipes.
  <https://github.com/espnet/espnet#notebooks>

### Our Specific Models

| Model | HuggingFace Link | Paper |
|-------|-----------------|-------|
| Whisper (all sizes) | [openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3) | [Paper](https://cdn.openai.com/papers/whisper.pdf) |
| MMS 1B | [facebook/mms-1b-all](https://huggingface.co/facebook/mms-1b-all) | [Paper](https://jmlr.org/papers/v25/23-1318.html) |
| XLS-R 300M | [facebook/wav2vec2-xls-r-300m](https://huggingface.co/facebook/wav2vec2-xls-r-300m) | [Paper](https://arxiv.org/abs/2111.09296) |
| wav2vec 2.0 | [facebook/wav2vec2-large-960h](https://huggingface.co/facebook/wav2vec2-large-960h) | [Paper](https://arxiv.org/abs/2006.11477) |
| Whisper-Ewe | [dodziraynard/whisper-small-ee](https://huggingface.co/dodziraynard/whisper-small-ee) | — |
| MMS TTS Ewe | [facebook/mms-tts-ewe](https://huggingface.co/facebook/mms-tts-ewe) | — |
