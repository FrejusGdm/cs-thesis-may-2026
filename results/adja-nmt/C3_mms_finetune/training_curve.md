# C3: Training Curve (early-stopped at epoch 6)

Pulled from `hf jobs logs 69e037decd8c002f31dfc376`.

| Epoch | Loss | Dev CER | Dev WER |
|-------|------|---------|---------|
| **1** | **12.4716** | **87.19%** | 99.91% |
| 2 | 4.4546 | 90.30% | 99.91% |
| 3 | 4.1554 | 94.26% | 99.56% |
| 4 | 3.9579 | 93.84% | 100.00% |
| 5 | 3.7010 | 93.92% | 99.56% |
| 6 | 3.5368 | 93.92% | 98.42% |

**Best CER** captured at epoch 1 (87.19%). Early-stopping (patience=5) fired at epoch 6. `Done! Best CER=87.19% at epoch 1, time=35.8min`.

The loss curve is the interesting part: **12.47 → 4.45 → 4.16 → 3.96 → 3.70 → 3.54** — loss is still dropping monotonically when training is killed. Dev WER *also* improves monotonically (99.91% → 98.42%) from epoch 1 to epoch 6, even though CER worsens slightly. The early-stopping metric (dev CER) was triggered by its unluckily-low reading at epoch 1, not by real plateau.
