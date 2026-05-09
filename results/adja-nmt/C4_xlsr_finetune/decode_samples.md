# C4: Decode Samples (all epochs)

Every decoded hypothesis across all 6 epochs is the empty string. These samples are representative — not cherry-picked.

**Reference**: ŋɖuɖu lɔwo nuɔn
**Hypothesis**: (empty)
**Notes**: No output — CTC argmax is always blank.

**Reference**: Eshilɔ ɖote yi mi jaja a ?
**Hypothesis**: (empty)
**Notes**: Same.

**Reference**: Enu maku enyi
**Hypothesis**: (empty)
**Notes**: Same.

Because WER/CER compare to the reference, every sample scores 100%.

No useful error analysis is possible here. The model never emitted a non-blank character during evaluation on any of the 160 dev utterances, across any of the 6 epochs.
