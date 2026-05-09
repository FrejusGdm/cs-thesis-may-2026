# E1: MMS with Ewe Adapter → Fine-Tuned on Adja

## Hypothesis

Starting from an Ewe adapter should outperform starting from French (C3),
because Ewe is the closest well-resourced Gbe language to Adja. The acoustic
and phonological similarities (both tonal, similar vowel inventory including
ɛ/ɔ, similar consonant clusters) mean the Ewe adapter's learned representations
should transfer better than French.

## Approach

1. Load `facebook/mms-1b-all` with `target_lang="ewe"` (Ewe adapter)
2. Replace CTC head with Adja character vocabulary
3. Fine-tune on Adja ASR data
4. Compare against C3 (MMS + French adapter) — same everything except adapter init

## Key Comparison

| Experiment | Adapter Init | Expected Result |
|-----------|-------------|-----------------|
| C3 | French (`fra`) | Baseline — colonial lingua franca |
| **E1** | **Ewe (`ewe`)** | **Should be better — closest Gbe relative** |

If E1 >> C3: cross-lingual Gbe transfer works, worth exploring more Gbe adapters.
If E1 ≈ C3: adapter init matters less than the fine-tuning data itself.
If E1 << C3: surprising — would need investigation.

## MMS Ewe Support

MMS has full Ewe support including:
- ASR adapter (what we use here)
- TTS: `facebook/mms-tts-ewe`
- Also has adapters for related Gbe languages: Ci Gbe (`cib`), Ayizo Gbe (`ayb`)

## References

- Pratap et al. 2023. *Scaling Speech Technology to 1000+ Languages*.
  https://jmlr.org/papers/v25/23-1318.html
- Building an Ewe Language Dataset (ACL 2025):
  https://aclanthology.org/2025.icnlsp-1.32.pdf
- Benchmarking ASR for African Languages (2025):
  https://arxiv.org/html/2512.10968v1

## Usage

```bash
# Same train.py as C3, just different config
python experiments/asr/E1_mms_ewe_finetune/train.py \
    --config experiments/asr/E1_mms_ewe_finetune/config.yaml \
    --data-dir data --output-dir results/E1/seed42

# Dry run
python experiments/asr/E1_mms_ewe_finetune/train.py \
    --config experiments/asr/E1_mms_ewe_finetune/config.yaml \
    --data-dir data --output-dir results/E1/seed42 --dry-run
```
