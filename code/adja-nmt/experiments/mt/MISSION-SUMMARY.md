# MISSION SUMMARY — NLLB chrF mission

Status: in progress

What I learned so far:
- The pipeline is working end-to-end on HF Jobs.
- The dataset is valid and correctly shaped for French→Adja MT.
- The stable training script is ready to use from the HF URL.
- Research points toward three promising levers:
  1. related-language / target-token initialization for the new `aj_Latn` token
  2. low-resource fine-tuning with strong data upsampling and careful LR control
  3. decode-time beam / length-penalty tuning on a fixed checkpoint

Best result so far:
- Smoke only; no real chrF yet.

Next likely direction:
- Run the highest-confidence hypothesis first: related-language target-token initialization with Adja-heavy training.
- If that underperforms, the cheapest fallback is decode-time tuning on the best checkpoint.
