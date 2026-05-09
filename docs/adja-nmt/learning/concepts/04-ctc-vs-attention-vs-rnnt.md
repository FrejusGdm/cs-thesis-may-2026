# 04 — CTC vs attention vs RNN-T: the three ASR decoding families

Last updated: **2026-04-21**

ASR models turn audio → text. The big architectural question is *how the
alignment between audio frames and text tokens is handled*. Three answers:

## CTC (Connectionist Temporal Classification)

- Paper: https://www.cs.toronto.edu/~graves/icml_2006.pdf
- Model outputs a distribution over vocab+blank at every frame.
- A differentiable loss marginalizes over all valid alignments (where
  repeated chars and blanks collapse to the true label).
- **Pros**: simple, fast, non-autoregressive at inference (greedy argmax).
- **Cons**: assumes conditional independence of output tokens — no "language
  model" inside. Works great when paired with an external LM at decode time.
- **Who uses it**: MMS, wav2vec 2.0 for ASR, XLS-R, our B1/C3/C4/C5 baselines.

## Attention (encoder-decoder)

- A transformer encoder sees the audio, a transformer decoder predicts text
  autoregressively with cross-attention to the encoder.
- **Pros**: strong implicit LM (the decoder learns one), handles long context.
- **Cons**: slower at inference, prone to hallucination (can emit text that
  doesn't correspond to any audio). Needs forced-alignment tricks.
- **Who uses it**: Whisper (our C2, E4, E7 ASR baselines), SeamlessM4T.

## RNN-T (RNN Transducer)

- Joint model: encoder outputs frame embeddings, prediction network outputs
  text embeddings, joiner predicts next token or blank.
- **Pros**: streaming-friendly, monotonic alignment (no hallucination), good
  for production.
- **Cons**: harder to train, more memory, smaller community.
- **Who uses it**: NVIDIA Parakeet (our D5 baseline), Google's production ASR.

## How to choose for a new language

- **Low resources + strong LM**: CTC + shallow-fusion LM rescoring (our D4
  experiment combined C4v2 XLS-R with a 13k-sentence Adja char LM — that was
  our best ASR at **CER 22.67%**).
- **Need highest absolute quality**: Whisper-style attention, especially with
  cross-lingual transfer (our E4 Ewe→Adja Whisper hit CER 24.90% in the
  original training log — but that checkpoint was never uploaded; the
  deployable reproduction at `JosueG/whisper-ewe-adja-e4v4` lands at
  CER 37.18%, see `results/comparison.md` 2026-04-29 correction).
- **Streaming / edge / latency**: RNN-T (Parakeet).

## For this project

- Fine-tune tracks: Whisper (attention), MMS/XLS-R (CTC).
- SSL pretraining on HPC: wav2vec 2.0 / XLS-R / MMS, all CTC at the supervised
  stage.
- Best practice we learned: always attach a character LM to the CTC output at
  decode time. 2-3 CER points for almost free (see D4 experiments).

## Papers to read

- Graves, "CTC" (2006) — https://www.cs.toronto.edu/~graves/icml_2006.pdf
- Graves et al., "Sequence transduction with RNNs" — https://arxiv.org/abs/1211.3711
- Chan et al., "Listen, Attend and Spell" — https://arxiv.org/abs/1508.01211
- Radford et al., "Whisper" — https://cdn.openai.com/papers/whisper.pdf
- Kürzinger et al., "CTC-Segmentation" — helpful for forced alignment
