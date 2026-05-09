# Upstream Orpheus 3B Notebook Runbook (Colab path)

When you'd rather run interactively than via HF Jobs — e.g. for rapid prompt tuning — use the Unsloth Colab notebook instead of `scripts/hf_jobs/T2_orpheus_finetune.py`.

**This path is secondary.** HF Jobs is canonical. Colab is kept because:
- Interactive audio playback is useful for quick A/B
- Free T4 tier is cheap for smoke tests
- The Reddit Kazakh/Finnish success cases both used the Unsloth Colab path

## Source notebooks

- Vendor: [references/unsloth-tts-notebooks/Orpheus_3B_TTS.ipynb](../../../references/unsloth-tts-notebooks/Orpheus_3B_TTS.ipynb) (clean blueprint)
- Adapted (reference only): [T2_adja_orpheus_finetune_reference.py](T2_adja_orpheus_finetune_reference.py) (the Kinyarwanda notebook you downloaded, with tokens stripped)

## Required edits when adapting the vendor notebook for Adja

The Kinyarwanda-adapted reference shows what these changes look like but hardcodes its own gotchas. Start from the clean vendor notebook and apply these:

### 1. Base model
```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "canopylabs/orpheus-3b-0.1-ft",  # English. NO leading slash.
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = False,
)
```
French variant: `"canopylabs/3b-fr-ft-research_release"`.

### 2. Dataset via HF Hub (not local file)
```python
from datasets import load_dataset
import os
dataset = load_dataset(
    "JosueG/adja-tts-orpheus",
    split="train",
    token=os.environ["HF_TOKEN"],  # Colab: set as a Secret, then os.environ pickup
)
```

### 3. NFC normalise Adja text
Insert before `create_input_ids`:
```python
import unicodedata
def normalize_text(text):
    return " ".join(unicodedata.normalize("NFC", text.strip()).split())
dataset = dataset.map(lambda ex: {**ex, "text": normalize_text(ex["text"])})
```

### 4. LoRA rank
The upstream notebook uses r=512 / alpha=512. That's unusually high and the Reddit Kazakh thread reports r=64–128 works. Use `r=64, lora_alpha=64` as the starting point for Adja — matches the HF Jobs canonical script default.

### 5. Training length
Replace `max_steps = 300` with:
```python
num_train_epochs = 20,
max_steps = -1,  # use epochs
```
Add early stopping + eval by setting `evaluation_strategy="steps"`, `eval_steps=50`, plus a held-out dev `Dataset` and `EarlyStoppingCallback(patience=5)`.

### 6. Inference playback
The upstream notebook uses `IPython.display.Audio` which only works in Jupyter. For Colab that's fine; for anywhere else (e.g. a Jupyter-less container), swap to:
```python
import soundfile as sf
sf.write("/tmp/out.wav", samples.detach().squeeze().cpu().numpy(), 24000)
```

### 7. Saving
Replace the two `hf_YopVMaE...` hardcoded tokens with `os.environ["HF_TOKEN"]`. **Never commit a real token.**

## Package pins (known-good as of 2026-04-18)

From the Kinyarwanda notebook's install cell + upstream Unsloth TTS docs:
```
unsloth                 (latest — follows Unsloth's rolling release)
transformers==4.56.2
trl==0.22.2 --no-deps
snac                    (latest)
bitsandbytes            (latest)
peft                    (latest)
datasets>=3.4.1,<4.0.0
huggingface_hub>=0.34.0
hf_transfer
```

If you hit `ConstantLengthDataset` import errors from `unsloth_zoo`, see the T1 recovery note: downgrade `trl` to the version `unsloth_zoo` expects (T1 found `trl==0.19.1` worked in that regime). Orpheus on Unsloth does *not* currently show that bug, but it's the same failure mode if it reappears.

## When to abandon the Colab path

If the dry-run passes on HF Jobs (scripts/hf_jobs/T2_orpheus_finetune.py --dry-run), there is no reason to use Colab for a long run. HF Jobs is cheaper, reproducible, and avoids Colab idle-kicks.

The Colab path stays in the repo only for interactive audio QA — load a trained adapter from `JosueG/adja-tts-results/T2_<prefix>/adapter/` and generate with different prompts.
