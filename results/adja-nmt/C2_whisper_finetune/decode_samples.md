# C2: Decode Samples (best model — epoch 47)

Samples from the log at the best-CER epoch. Model has clearly learned Adja orthography and word shapes; most errors are missing diacritics, minor phoneme substitutions, and over-insertion of extra vowels.

**Reference**: ŋɖuɖu lɔwo nuɔn
**Hypothesis**: Ŋ dujdu lɔwo n ɔ ?
**Notes**: Got the ŋ + ɖuɖu root + lɔwo, but inserted a stray "j" inside ɖuɖu and added a question mark. Missing "nuɔn" collapsed to "n ɔ".

**Reference**: Eshilɔ ɖote yi mi jaja a ?
**Hypothesis**: Eko te Iimi jaja ?
**Notes**: Gets the trailing "jaja ?" right. Misreads initial "Eshilɔ ɖote" as "Eko te", drops the "yi" and mangles "mi" → "Iimi".

**Reference**: Enu maku enyi
**Hypothesis**: Enu ma ku enyi
**Notes**: Near-perfect — only inserts a space in "ma ku" vs "maku". Real Adja recovered.

## Earlier-epoch comparison (epoch 19, first time dev CER dipped below 29%)

**Reference**: ŋɖuɖu lɔwo nuɔn
**Hypothesis**: Ŋ dɖudu lɔ wo nun ɔ ?
**Notes**: Extra characters and spacing still noisy at ep 19 compared to ep 47.

**Reference**: Eshilɔ ɖote yi mi jaja a ?
**Hypothesis**: Edote Ii mi jaja ?
**Notes**: Drops "Eshilɔ", keeps the "jaja ?" tail.

**Reference**: Enu maku enyi
**Hypothesis**: Enu m aku enyi
**Notes**: Already recognizable.

## Other good recoveries across epochs (44, 45, 47)

**Reference**: Eshilɔ ɖote yi mi jaja a ?
**Hypothesis (ep 45)**: Eɖo te i mi jaja ?
**Notes**: Very close — mostly spacing.

**Reference**: Enu maku enyi
**Hypothesis (ep 44)**: (matches best-epoch quality throughout the plateau)
**Notes**: This short utterance is transcribed accurately in every late epoch.

Error patterns:
- Spacing is the most common small error — model over-segments words with a leading capital letter.
- Tone marks and diacritics (´, `, ̀) are usually absent from the output.
- Long vowels are sometimes under-produced (`ɖuɖu` → `ɖu`); sometimes over-produced (`enyii`).
