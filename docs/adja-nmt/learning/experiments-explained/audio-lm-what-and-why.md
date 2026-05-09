# Audio-LM pretraining — what it is and why we're trying it

Last updated: **2026-04-21**

## The idea

Standard TTS fine-tuning needs (text, audio) pairs. We only have ~6 hours
of (text, audio) pairs across Adja + Ewe — not much.

But we have **183k utterances of unlabeled Ewe audio**. Can we pretrain a
TTS model on just the audio, with no text? Yes, using a technique borrowed
from AudioLM / VALL-E:

1. Pass the audio through a frozen codec (Mimi for CSM, SNAC for Orpheus).
2. The codec spits out an integer sequence of codec tokens.
3. Train the LM backbone on pure next-token prediction over that sequence
   — standard causal LM loss. No text involved.
4. The LM learns the distribution over codec tokens — i.e., what "natural
   Ewe speech" sounds like in codec-token-space.
5. Then supervised fine-tune on the small labeled dataset, attaching text
   → codec-token mapping.

If the audio-LM pretraining is useful, the supervised fine-tune should
converge faster and produce more natural-sounding output than training
from the base checkpoint.

## Why this is speculative

- **Published evidence is thin** for low-resource TTS. AudioLM and VALL-E
  both used thousands of hours of pretraining audio. 183k utterances is
  maybe 100-200 hours — enough to help, maybe not enough to be transformative.
- **Depends on codec fidelity**. If Mimi or SNAC subtly lose tone
  information during encode/decode, the LM learns a corrupted distribution
  and the supervised stage fights a losing battle. Mimi reconstruction test
  (local script) must pass before this track is greenlit.
- **Catastrophic forgetting**. If we pretrain CSM's LM on pure Ewe codec
  tokens, it might forget how to condition on text. Our Stage B re-
  introduces text, but the model may never recover fully.

## Two scripts on HPC

### `hpc/scripts/pretraining/audio_lm_csm_ewe.py`
- Encodes 183k unlabeled Ewe utts with Mimi → CSM next-token loss on the
  audio token stream.
- Stage B: supervised fine-tune on Ewe TTS + Adja TTS combined.

### `hpc/scripts/pretraining/audio_lm_orpheus_ewe.py`
- Same pipeline but SNAC + Orpheus.
- Default base: `canopylabs/3b-zh-ft-research_release` (tonal
  prior gives us a better starting point).

## Expected runtime (A100 80GB)

- Stage A Mimi (CSM 1B): ~24h for 183k × 3 epochs.
- Stage A SNAC (Orpheus 3B): ~18h for 183k × 2 epochs (smaller codec
  sequence despite bigger model).
- Stage B: 6-8h each.

Total ~60h compute. Cluster-only.

## Gate

Both scripts should be held until:
1. Mimi reconstruction test passes locally (decoded Adja audio sounds fine).
2. SNAC reconstruction test passes (for Orpheus variant).
3. We have Stage 1 checkpoints from the simpler Ewe-bridge experiments
   (T1_csm_ewe_stage1, T2_orpheus_ewe_stage1). If those already give us
   good results, audio-LM pretraining is a lower-priority marginal win.

## Success criteria

- Audio-LM Stage A converges (train loss decreases, no NaNs).
- Stage B matches or beats the Ewe-bridge-only result (T1 Stage 2 /
  T2 Stage 2).
- Generated Adja audio is intelligible to a native speaker.

## Papers to read

- Borsos et al., "AudioLM" — https://arxiv.org/abs/2209.03143 — the original.
- Wang et al., "VALL-E" — https://arxiv.org/abs/2301.02111 — neural-codec TTS
  with in-context learning.
- SoundStorm, NaturalSpeech 3, and other audio-LM papers have similar
  architectures if you want broader context.

## Files

- `../../hpc/scripts/pretraining/audio_lm_csm_ewe.py`
- `../../hpc/scripts/pretraining/audio_lm_orpheus_ewe.py`
- `../../experiments/tts/T1_sesame_csm_finetune/mimi_reconstruction_test.py` — run first.
- `../concepts/02-what-is-an-audio-codec.md`
