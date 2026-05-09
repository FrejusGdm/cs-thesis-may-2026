# E4: Decode Samples (best epoch = 50 — the final epoch)

Best-available samples from the log at the final (and best) epoch.

## Epoch 50 (best CER=24.9%)

**Reference**: ŋɖuɖu lɔwo nuɔn
**Hypothesis**: ŋ ɖudu lɔwo nɔ?
**Notes**: Very close — ŋ, ɖudu, lɔwo all correct. Missed the second ɖ in ɖuɖu and substituted u→ɔ in nuɔn. Stray question mark.

**Reference**: Eshilɔ ɖote yi mi jaja a ?
**Hypothesis**: Eɖote yi mi jaja
**Notes**: Gets the main verb ɖote, yi, mi, and jaja exactly right. Drops the "Eshilɔ" prefix and the final "a ?". Most tokens recovered.

**Reference**: Enu maku enyi
**Hypothesis**: Enu maku eye
**Notes**: "Enu maku" is perfect; "enyi" → "eye" — lost the nasal stop.

## Epoch 46 (runner-up, lowest WER=72.57%)

**Reference**: ŋɖuɖu lɔwo nuɔn
**Hypothesis**: ŋ dudu lɔ wo nɔ?
**Notes**: Slightly worse on the ɖ → d substitution.

**Reference**: Eshilɔ ɖote yi mi jaja a ?
**Hypothesis**: Eɖote yi mi jaja
**Notes**: Same as ep 50.

**Reference**: Enu maku enyi
**Hypothesis**: Enu ma ku eŋuɖe
**Notes**: Odd over-extension "eŋuɖe" — the model is still hunting at ep 46.

## Epoch 45 (first time CER dipped below 26%)

**Reference**: ŋɖuɖu lɔwo nuɔn
**Hypothesis**: ŋɖudu lɔwo nɔ ?
**Notes**: Better than ep 50 on this sample — preserves ɖ. Shows CER isn't monotone per-utterance even as the mean improves.

**Reference**: Eshilɔ ɖote yi mi jaja a ?
**Hypothesis**: Eɖote yi mi jaja
**Notes**: Stable — same as ep 46 and ep 50.

**Reference**: Enu maku enyi
**Hypothesis**: Enu maku eŋuɖe
**Notes**: Same quirk as ep 46.

## Earlier, for comparison — Epoch 25 (CER=29.78%)

Recovery is already underway but less refined:

**Reference**: ŋɖuɖu lɔwo nuɔn
**Hypothesis (ep 25 era)**: missing — only the best-epoch samples are logged verbatim, but the quality is visibly worse based on the CER trajectory.

## Error patterns at the top of the curve
- **Prefix drop**: "Eshilɔ" at the start of utterances is often elided.
- **ɖ → d substitution**: The implosive ɖ is sometimes rendered as plain d.
- **Long-vowel flattening**: ɖuɖu → ɖudu (one repetition instead of two).
- **Terminal material dropped**: "a ?" sometimes lost.
- **Spacing noise**: occasional space inside a word (e.g. "ma ku" for "maku").

Every error above is a refinable type. This is why the conclusion argues strongly for more epochs.
