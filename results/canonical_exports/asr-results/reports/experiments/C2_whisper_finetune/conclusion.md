# C2: Conclusion

## Verdict
**converged**

## What happened
Whisper-small fine-tuned from catastrophic zero-shot failure (see C1: CER=366% → hallucinations) to CER=27.02% on the Adja dev split over 50 epochs. The training curve shows steady improvement through epoch ~19, after which dev CER plateaued in the 27-28% band with mild oscillation. Best model saved at epoch 47 (CER=27.02%, WER=74.15%); ran the full 50 epochs without triggering early stopping (patience=20). Decode samples at the best epoch show the model reliably producing real Adja word shapes — e.g. `Enu maku enyi` → `Enu ma ku enyi` — rather than hallucinated English or Georgian.

## Why it worked
Whisper's multilingual acoustic encoder transfers well once the decoder is exposed to Adja output. The custom training loop + per-epoch CER evaluation + chrF-aware best-model selection match the pattern from `learnings-from-the-past/` and avoided the fragility of `Seq2SeqTrainer`. The small Adja vocabulary (115 tokens after `fix_tokenizer()`) and the 1277-utterance train split were enough to move a 244M-parameter Whisper-small from pure hallucination to near-intelligible output in under 6 GPU-hours.

## What to do next
Done. Keep as the Whisper-small reference point. Immediate higher-capacity follow-ups worth considering:

- **Whisper-medium / large-v3 fine-tune** on HPC (needs A100 80GB, multi-hour job). Expect a modest CER drop from the added capacity given the small dataset.
- **Add LM rescoring** — pair with `D4-lm` (5-gram char LM already built under `data/char_5gram.arpa`) to close spacing/vowel-length errors seen in decode samples.
- **Extend decoding constraints** — suppress non-Adja unicode ranges to eliminate residual `!` / `?` punctuation artifacts.

## Known follow-ups already queued
None explicitly for C2. The best-currently-available model is tracked in `E4_whisper_ewe_finetune/` (Whisper-Ewe → Adja, CER=24.9%), and `E4v2-s42` is queued to push that direction further (100 epochs, patience=25). If E4v2 beats this run by a meaningful margin, C2 stays as the "from raw Whisper-small" baseline.

## Caveat
The log header of job `69e04dbfac288e522d8eee2f` reads `C3v2: facebook/mms-1b-all`, suggesting this job may have actually dispatched the MMS-1B+fra rerun. The dev CER curve, however, matches the registry's C2 numbers (best 27.02% at ep 47). Before citing these numbers in a paper, confirm which script was actually submitted under this job ID.
