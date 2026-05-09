# NFC Normalization — Technical Note

Unicode normalization is a correctness requirement, not an optimization, for every script in this
ablation suite that touches Adja or Ewe text. Getting it wrong silently breaks tokenization by
creating different byte sequences for visually identical characters.

---

## The Bug

`JosueG/adja-tts-orpheus` (Adja TTS dataset) was created with NFC normalization applied.
`google/WaxalNLP` `ewe_tts` text was uploaded without a documented normalization pass — it may
not be NFC.

Any script that **concatenates or jointly trains on both datasets** without normalizing both to NFC
first risks inconsistent tokenization: the same character (e.g., `è` = U+00E8) could appear as its
precomposed form in one dataset and as a base character + combining grave (`e` + U+0300) in the
other. This produces different token IDs in every tokenizer, inflating the apparent vocabulary and
breaking cross-dataset consistency.

---

## Why This Matters for Adja and Ewe

Adja and Ewe both use:
- **Special Latin characters**: ɛ (U+025B), ɔ (U+0254), ŋ (U+014B), ɖ (U+0256)
- **Tone marks**: composed forms like `é` (U+00E9), `è` (U+00E8), `ê` (U+00EA) as well as
  decomposed equivalents (`e` + combining accent, e.g., U+0301, U+0300, U+0302)
- **Rare stacked diacritics**: some Gbe orthographies use combinations that have multiple valid
  Unicode representations

NFC (Canonical Decomposition followed by Canonical Composition) is the correct form for these
because it produces the maximally-composed precomposed characters that are most likely to appear
as single tokens in Unicode-aware tokenizers.

NFKC (Compatibility Decomposition followed by Canonical Composition) is used in the T1-tokfix
variant specifically because the `ADJA_CHARS` character set was derived from a corpus under NFKC
normalization. **Do not use NFKC as a default** — NFKC applies additional compatibility
decompositions (e.g., ligatures, width variants) that are inappropriate for general TTS training
text.

---

## The Fix

Apply `unicodedata.normalize("NFC", text)` to **every text string** before tokenization, in
every script that handles Adja or Ewe text. This includes:

- Dataset loading functions
- `formatting_func` / `formatting_audio_func` inside training scripts
- Evaluation / decoding scripts
- Any script that reads text from Hub datasets and writes processed metadata

```python
import unicodedata

def normalize_text(text: str) -> str:
    """Normalize to NFC. Apply to all Adja/Ewe text before tokenization."""
    return unicodedata.normalize("NFC", text)
```

---

## Verification Snippet

Use this to check whether a dataset's text column is already NFC-normalized before relying on it:

```python
import unicodedata
from datasets import load_dataset

def is_nfc(text: str) -> bool:
    return unicodedata.is_normalized("NFC", text)

# Load and check
ds = load_dataset("google/WaxalNLP", "ewe_tts", split="train")
samples = ds["text"][:10]
results = [is_nfc(s) for s in samples]
print(results)          # [True, True, ...] means already NFC
print(samples[:3])      # Visual inspection

# Check Adja dataset
ds_adja = load_dataset("JosueG/adja-tts-orpheus", split="train")
adja_samples = ds_adja["text"][:10]
print([is_nfc(s) for s in adja_samples])
```

If the WaxalNLP output contains any `False` values, the text is not NFC and normalization is
required before use.

---

## Affected Scripts

Every multi-source script is affected. This includes all planned experiments in this suite:

| Experiment | Datasets used | NFC required? |
|------------|--------------|--------------|
| CF1 | Adja (`JosueG/adja-tts-orpheus`) + Ewe (`google/WaxalNLP ewe_tts`) | YES — both |
| CF2 | Same | YES |
| CF3 | Same | YES |
| TK1 | Same | YES |
| AT1 | Ewe ASR unlabeled + Ewe TTS + Adja | YES — all three |
| M1 | Adja audio only (no text loss) | N/A for codec training |
| M2 | Adja audio only | N/A for codec training |
| M0 | Adja audio only | N/A (reconstruction gate, no text) |

For M1/M2 (codec fine-tuning), text normalization is irrelevant because those experiments train
on audio reconstruction loss, not token prediction. For all others, normalization is required.

---

## Known Deviation

`hpc/scripts/pretraining/audio_lm_csm_ewe.py` Stage B uses **NFKC** normalization. This was
intentional for the tokfix variant (ADJA_CHARS was derived under NFKC), but is incorrect for the
general Gbe cascade training pipeline.

**SageMaker/HF Jobs versions of the same scripts should use NFC, not NFKC.**

When porting any script from the HPC pretraining path to a CF or AT experiment script, verify the
normalization call and switch `"NFKC"` to `"NFC"` if present.

---

## Quick Audit

To find all normalization calls across the repo:

```bash
grep -r "normalize" scripts/hf_jobs/ hpc/scripts/ --include="*.py" -n
```

Every hit should read `"NFC"` except for the explicitly-documented `T1_csm_tokfix.py` and
`audio_lm_csm_ewe.py` Stage B which use `"NFKC"` for the tokfix experiment only.
