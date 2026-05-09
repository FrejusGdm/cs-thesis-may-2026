# E4: Conclusion

## Verdict
**still improving** — best-in-project model, but training was cut short.

## What happened
Starting from a Whisper checkpoint already fine-tuned on Ewe, we fine-tuned onto Adja for 50 epochs. The training curve has three distinct phases:

1. **Adaptation (ep 1-14)**: Dev CER stays pinned at 100% while training loss crashes from 17.54 → 3.14. The Ewe tokenizer needs ~14 epochs to remap its character distribution onto Adja — during this phase the model emits Ewe-style output that fully misses the Adja references.
2. **Rapid unlock (ep 15-21)**: Dev CER falls from 90.9% to 35.4% in six epochs. The Ewe-initialized encoder starts outputting Adja characters once the decoder catches up.
3. **Long tail (ep 22-50)**: Continued improvement every epoch or two, ending at CER=24.9%, WER=73.09% at the final (50th) epoch.

**Best CER is at the final epoch.** The model had not converged — loss was still dropping, CER was still posting new bests. If we had kept training, it would almost certainly have gotten better.

Decode samples at ep 50 show substantial real-Adja recovery: `ŋɖuɖu lɔwo nuɔn` → `ŋ ɖudu lɔwo nɔ?` (5 of ~6 content units correct); `Eshilɔ ɖote yi mi jaja a ?` → `Eɖote yi mi jaja` (all main verbs and nouns preserved, just a prefix drop).

This is the best model on the project so far. It beats C2 (Whisper-small fine-tune, CER=27.02%) by about 2 CER points and is the only run with clear further headroom.

## Why it worked
The Ewe → Adja transfer hypothesis held: Ewe and Adja are closely related Gbe languages, so the Whisper-Ewe checkpoint's encoder already had the right acoustic biases for Adja (the velar nasal ŋ, implosive ɖ, front-mid vowels ɛ/ɔ, tone prosody). The long ep 1-14 plateau at 100% CER is deceptive — the encoder was ready; only the tokenizer/decoder alignment needed rewiring, which is fast once it starts.

## What to do next
**E4v2 is already queued** — continue the Ewe transfer story:

- `MAX_EPOCHS=100` (double the training budget)
- `PATIENCE=25` (tolerate larger oscillation in the long-tail regime)
- Warm-start from E4's best checkpoint if the script supports it; otherwise restart from Whisper-Ewe and re-pay the adaptation tax with more headroom.

If E4v2 plateaus, consider:
- Add an LM rescorer (data/char_5gram.arpa is ready) — likely to close spacing and "eye"/"enyi" substitution errors.
- Character-level dropout / augmentation to smooth the long-tail improvement.
- A larger base model (Whisper-Ewe-medium if it exists, or fine-tune Whisper-medium on Ewe first).

## Known follow-ups already queued
- **E4v2-s42**: Whisper-Ewe → Adja continuation, 100 epochs, patience=25. In the registry as `queued`. Top-priority run once HPC frees up — most likely path to the project's next best number.
