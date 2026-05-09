# 05 — SSL pretraining: wav2vec 2.0, XLS-R, MMS

Last updated: **2026-04-21**

## Why SSL matters for low-resource ASR

If you have 1.6k hours of audio but only 1.6 hours of transcripts, the
transcripts are the bottleneck — a fully-supervised ASR model fits 1.6 hours
of audio in a few minutes on an A100 and then overfits.

**SSL pretraining** uses the *untranscribed* audio to teach the encoder
acoustic structure (phonemes, their transitions, prosody) without needing
labels. Then you fine-tune on your tiny labeled set; the encoder already
knows what speech sounds like, so it just needs to learn the mapping to
orthography.

For Adja, WaxalNLP gives us:
- 15k labeled Ewe utterances (useful for supervised cross-lingual transfer)
- **183k unlabeled Ewe utterances** (gold for SSL pretraining)
- And separately our ~1.6k labeled Adja utterances (the target task)

## wav2vec 2.0

- Paper: https://arxiv.org/abs/2006.11477
- The SSL objective: mask ~50% of the encoder output frames; predict the
  correct quantized latent among a set of distractors (contrastive loss).
- Two forms: `ForPreTraining` (the SSL objective) and `ForCTC` (supervised
  fine-tune).
- Our HPC script: `hpc/scripts/pretraining/wav2vec2_ssl_ewe.py` — Stage A runs
  `ForPreTraining` on unlabeled Ewe, Stage B runs `ForCTC` on Adja.

## XLS-R

- Paper: https://arxiv.org/abs/2111.09296
- Same wav2vec 2.0 architecture, but pretrained on 128 languages (436k hours).
- Already has exposure to a lot of African languages — so "continue
  pretraining" on Ewe should push it further rather than teach it the basics.
- Sizes: 300M, 1B, 2B. We run both 300M and 1B on HPC.

## MMS

- Paper: https://arxiv.org/abs/2305.13516
- Extension of wav2vec 2.0 to 1107 languages — they literally cover Ewe.
- Model sizes: 300M and 1B.
- Comes with language-specific CTC heads; if you want pure phoneme ASR in
  one of the covered languages, you can use MMS zero-shot.
- For Adja, we continue-pretrain on unlabeled Ewe then fine-tune CTC on Adja.

## Why XLS-R is arguably SOTA for low-resource ASR right now (2026)

- Already multilingual, so it has priors for Adja-like phonology.
- Has a well-studied SSL objective that is known to transfer.
- Pretraining data is diverse enough that continued pretraining doesn't
  catastrophically forget.
- MMS covers more languages but its CTC head is locked to its 1107-language
  vocabulary; XLS-R lets you attach any vocabulary.

## Practical notes

- SSL needs *lots* of compute. 183k utterances × 3 epochs of contrastive loss
  on XLS-R-1B takes ~24h on an A100. Hence HPC (not HF Jobs).
- Gradient checkpointing is mandatory to fit the bigger models.
- After SSL, supervised fine-tuning converges in 4-8 hours.

## Related files

- `../../hpc/scripts/pretraining/wav2vec2_ssl_ewe.py` — full two-stage pipeline.
- `../../hpc/scripts/pretraining/xlsr_ssl_ewe.py` — XLS-R wrapper.
- `../../hpc/scripts/pretraining/mms_ssl_ewe.py` — MMS wrapper.
