# C1: Zero-Shot Metrics (per model size)

No training — this was a one-shot inference pass. Instead of an epoch-by-epoch curve, we report per-model-size metrics on the dev and test splits.

| Model | Split | WER | CER | Inference time | N samples |
|-------|-------|-----|-----|----------------|-----------|
| whisper-tiny | test | 1208.25% | 2047.84% | 161.7s | 160 |
| whisper-tiny | dev | 1161.26% | 2005.57% | 138.7s | 160 |
| whisper-small | test | 100.91% | 366.74% | 524.0s | 160 |
| whisper-small | dev | 100.09% | 384.31% | 558.7s | 160 |
| whisper-large-v3 | test | 129.65% | 131.81% | 189.0s | 160 |
| whisper-large-v3 | dev | 121.65% | 139.45% | 181.3s | 160 |

CER values above 100% indicate that hypotheses were *longer* than references (hallucinated repetition such as `www.www.www...` or extended Georgian-script strings). WER > 100% has the same cause plus wrong-language output.

Source: `hf jobs logs 69e02cb9ac288e522d8eed44` — bottom of log contains the JSON `RESULTS` block.
