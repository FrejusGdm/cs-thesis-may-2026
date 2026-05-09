# C5: Training Curve (collapsed)

Pulled from `hf jobs logs 69e03d15cd8c002f31dfc3f6`.

| Epoch | Loss | Dev CER | Dev WER |
|-------|------|---------|---------|
| 1 | 16.4372 | 100.00% | 100.00% |
| 2 | 3.9285 | 100.00% | 100.00% |
| 3 | 3.6376 | 100.00% | 100.00% |
| 4 | 3.4861 | 100.00% | 100.00% |
| 5 | 3.4143 | 100.00% | 100.00% |
| 6 | 3.5107 | 100.00% | 100.00% |

`Done! Best CER=100.0% at epoch 1, time=28.1min`.

Loss drops sharply from 16.44 to 3.93 in one epoch and then flatlines around 3.5. Dev CER/WER stay at 100% throughout. Same CTC blank-collapse pattern as C4 — model minimizes CTC loss by emitting blanks everywhere.
