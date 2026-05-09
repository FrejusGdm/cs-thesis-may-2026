# E4: Whisper-Small-Ewe → Fine-Tuned on Adja

## Hypothesis

An Ewe-tuned Whisper should outperform a vanilla Whisper (C2) on Adja,
because the model has already learned Ewe acoustic patterns which are
closely related to Adja (both Gbe family, tonal, similar phoneme inventory).

## Approach

1. Load `dodziraynard/whisper-small-ee` (Whisper-small fine-tuned on Ewe, WER=37.5%)
2. Continue fine-tuning on Adja ASR data
3. Compare against C2 (Whisper-small from OpenAI, no Ewe knowledge)

## Key Comparison

| Experiment | Starting Checkpoint | Ewe Knowledge |
|-----------|-------------------|---------------|
| C2 | `openai/whisper-small` | None |
| **E4** | **`dodziraynard/whisper-small-ee`** | **WER=37.5% on Ewe** |

This directly measures whether Ewe-to-Adja transfer helps in the
encoder-decoder (Whisper) paradigm, complementing E1 which tests
transfer in the CTC (MMS) paradigm.

## Source Model

- **HuggingFace**: `dodziraynard/whisper-small-ee`
- **Architecture**: Whisper-small (244M params)
- **Trained on**: University of Ghana Ewe Speech Data (`ugspeechdata-ewe`)
- **Performance**: WER=37.48%, CER=12.97%
- **License**: Apache 2.0

## References

- Radford et al. 2022. *Robust Speech Recognition via Large-Scale Weak Supervision*.
  https://cdn.openai.com/papers/whisper.pdf
- HuggingFace Whisper fine-tuning guide:
  https://huggingface.co/blog/fine-tune-whisper

## Usage

```bash
# Same train.py as C2, just different config
python experiments/asr/E4_whisper_ewe_finetune/train.py \
    --config experiments/asr/E4_whisper_ewe_finetune/config.yaml \
    --data-dir data --output-dir results/E4/seed42

# Dry run
python experiments/asr/E4_whisper_ewe_finetune/train.py \
    --config experiments/asr/E4_whisper_ewe_finetune/config.yaml \
    --data-dir data --output-dir results/E4/seed42 --dry-run
```
