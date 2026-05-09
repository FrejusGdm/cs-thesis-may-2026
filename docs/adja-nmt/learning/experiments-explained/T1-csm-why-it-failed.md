# T1 (Sesame CSM) — why it failed on Adja

Last updated: **2026-04-21**

## What we tried

Fine-tune Sesame CSM 1B on the 1.6k-utterance Adja TTS dataset. Multiple
variants:
- LoRA r=32 on the LM backbone
- LoRA r=64, r=128
- Full fine-tune (all 1.6B params trainable, no LoRA)
- Both English base and the Unsloth redistribution `unsloth/csm-1b`

## What we expected

Intelligible Adja speech after 20 epochs. CSM was advertised as a
conversational TTS model that adapts to new voices with small data.

## What we got

Noise. Every variant produced output that was either silence, static, or
garbled non-speech. Native-speaker listening confirmation (Josue) on all
variants. Dev loss plateaued around 6.5, far above the intelligibility
threshold that Spark reached (~5.5 when it started producing coherent Adja).

## Our first (wrong) explanation

"The Mimi codec is English-heavy and can't represent Adja phonetics." We
wrote this down as the root cause and moved on.

## What made us change our minds

Orpheus was released with multilingual variants (French, German, Mandarin,
Hindi, Korean, Spanish/Italian) — *all using the same SNAC codec as the
English Orpheus*. The Mandarin one (3b-zh) produces intelligible Mandarin,
a 4-tone language. If the codec were the bottleneck, this would not work.

This forced a re-analysis. The real problem is almost certainly Layer 1 of
the stack (see `../concepts/03-three-layer-tts.md`): the Llama 3.2 BPE
tokenizer byte-fragments Adja's special characters (ɛ, ɔ, ŋ, ɖ, and tone
marks). With only 1.6 hours of training data, the LM cannot learn to
reconstruct the missing phonemes from byte-level fragments.

Spark TTS uses the Qwen2 tokenizer, which has native tokens for these
characters — and Spark produced intelligible Adja on the same dataset.

## What we learned

1. **Diagnose the layer before diagnosing the fix.** A codec reconstruction
   test on the same data would have shown within an hour that Mimi handles
   Adja fine.
2. **Mark causal claims as hypotheses until tested.** Our early logs said
   "the codec is English-centric" as fact. It was an assumption. See the
   Epistemology Policy we added to the master plan.
3. **Re-analysis is cheap.** Orpheus multilingual variants are a natural
   experiment that exposed the original hypothesis for free.

## What we're doing about it

Two experiments currently in flight to test the revised hypothesis:
- `scripts/hf_jobs/T1_csm_tokfix.py` — expand the CSM tokenizer with 70+
  Adja characters (empirically derived from our corpus), resize embeddings,
  re-fine-tune on Adja.
- `scripts/hf_jobs/T2_orpheus_tokfix.py` — same fix for Orpheus EN base.

If T1-tokfix produces intelligible Adja, the tokenizer hypothesis is
confirmed and we have a recoverable CSM path.

If it still fails, we need to look at Layer 2 (LM backbone) more carefully —
perhaps LoRA r=32 is not enough to learn good embeddings for the new tokens.

## Files

- Registry: `../../experiments/registry.md` rows T1-* (LoRA r=32/64/128, fullft)
- Run ledger: `../../results/run-ledger.md` — 2026-04-16 through 2026-04-18 entries
- Script: `../../scripts/hf_jobs/T1_sesame_csm_finetune.py` (vanilla)
- Fix script: `../../scripts/hf_jobs/T1_csm_tokfix.py`
- Related concept: `../concepts/01-what-is-a-tokenizer.md`
