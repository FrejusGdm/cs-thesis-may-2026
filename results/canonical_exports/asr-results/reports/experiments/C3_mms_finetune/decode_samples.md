# C3: Decode Samples (best epoch = epoch 1)

Best CER was observed at the very first epoch (87.19%). By epoch 6 the model has stopped producing varied output at all — everything collapses to a single leading character. This is the start of CTC blank collapse but caught just before it becomes total.

## Epoch 1 (best)

**Reference**: ŋɖuɖu lɔwo nuɔn
**Hypothesis**: e on
**Notes**: Two tokens surviving — `e` and `on`. Enough to register as < 100% CER.

**Reference**: Eshilɔ ɖote yi mi jaja a ?
**Hypothesis**: eo on
**Notes**: Minimal output but not completely blank.

**Reference**: Enu maku enyi
**Hypothesis**: eo o n
**Notes**: Still emitting varied characters.

## Epoch 2

**Reference**: ŋɖuɖu lɔwo nuɔn
**Hypothesis**: Ei io
**Notes**: A tiny bit more output than epoch 1; dev CER slightly worse because lengths increased without being correct.

**Reference**: Eshilɔ ɖote yi mi jaja a ?
**Hypothesis**: E io
**Notes**: Capital E suggests the model is picking up initial capitals in the target.

**Reference**: Enu maku enyi
**Hypothesis**: E o
**Notes**: Two-character output.

## Epochs 3-6 (collapse phase)

**Reference**: ŋɖuɖu lɔwo nuɔn
**Hypothesis (ep 3-5)**: `E` / `M` / `E`
**Notes**: Output collapses to a single character regardless of input — CTC starting to favor the blank token.

**Reference**: Eshilɔ ɖote yi mi jaja a ?
**Hypothesis (ep 6)**: `E  ?`
**Notes**: Question mark sometimes survives — these appear in ~40% of training references and are the easiest token to learn.

**Reference**: Enu maku enyi
**Hypothesis (ep 6)**: `E`
**Notes**: Full collapse. By this point only the initial capital is coming through.
