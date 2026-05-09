# C4: Training Curve (collapsed)

Pulled from `hf jobs logs 69e03d0ecd8c002f31dfc3f4`.

| Epoch | Loss | Dev CER | Dev WER |
|-------|------|---------|---------|
| 1 | 22.7231 | 100.00% | 100.00% |
| 2 | 11.7448 | 100.00% | 100.00% |
| 3 | 8.4848 | 100.00% | 100.00% |
| 4 | 6.7917 | 100.00% | 100.00% |
| 5 | 5.5631 | 100.00% | 100.00% |
| 6 | 4.7352 | 100.00% | 100.00% |

`Done! Best CER=100.0% at epoch 1, time=27.6min`.

The training loss **does drop** (22.72 → 4.74 — almost 5× reduction in 6 epochs), but dev CER/WER stay pinned at 100%. The model is minimizing CTC loss by emitting blank tokens everywhere; decoder output is empty for every utterance on every epoch. Classic CTC collapse.
