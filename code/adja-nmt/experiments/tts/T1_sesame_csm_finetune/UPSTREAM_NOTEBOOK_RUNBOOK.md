# Upstream Sesame CSM Notebook Runbook

This is the practical "make it work" path for Adja.

## Source of truth

Start from the vendor notebook:

- `references/unsloth-tts-notebooks/Sesame_CSM_1B_TTS.ipynb`

Do **not** start from the local adapted notebook if you are trying to re-establish a clean baseline.

## Runtime

- Primary runtime: Google Colab T4
- Only move to Colab A100 if T4 fails for VRAM/runtime reasons

## Fix the install cell first

If you are seeing:

- `unsloth_zoo` installed
- but `from unsloth import FastModel` fails

then do **not** trust the upstream install cell as-is on the first attempt.

Two practical problems show up in notebooks:

1. the upstream cell uses shell-style installs, so a package failure can be easy to miss
2. `%%capture` hides the exact pip failure, which makes it look like the install succeeded
3. if a bundled dependency install aborts midway, `unsloth_zoo` can appear present while `unsloth` never finished installing

### First-pass Colab install cell

Use this **without** `%%capture` on the first attempt:

```python
import re
import subprocess
import sys
import torch

def pip_install(*packages):
    cmd = [sys.executable, "-m", "pip", "install", *packages]
    print("+", " ".join(cmd))
    subprocess.check_call(cmd)

pip_install("-U", "pip", "setuptools", "wheel")
pip_install("--no-deps", "unsloth==2025.5.4", "unsloth_zoo==2025.5.6")
pip_install("sentencepiece", "protobuf", "datasets==4.3.0", "huggingface_hub>=0.34.0", "hf_transfer")

torch_version = re.match(r"[\d]+\.[\d]+", str(torch.__version__)).group(0)
xformers_version = {"2.10": "0.0.34", "2.9": "0.0.33.post1", "2.8": "0.0.32.post2"}.get(torch_version)
cuda_stack = ["--no-deps", "bitsandbytes==0.45.5", "accelerate==1.6.0", "peft==0.14.0", "triton==3.2.0"]
if xformers_version is not None:
    cuda_stack.append(f"xformers=={xformers_version}")
else:
    print(f"No pinned xformers version for torch {torch_version}; skipping explicit xformers install.")

pip_install(*cuda_stack)
pip_install("transformers==4.52.3")
pip_install("--no-deps", "trl==0.19.1")
pip_install("torchcodec", "datasets>=3.4.1,<4.0.0")
```

### Immediate verification cell

Run this right after the install cell:

```python
import sys
print(sys.executable)
!{sys.executable} -m pip show unsloth unsloth_zoo

import unsloth
import unsloth_zoo
print("unsloth:", unsloth.__file__)
print("unsloth_zoo:", unsloth_zoo.__file__)
```

If this cell fails, stop there. Do not continue to model loading.

The current local recovery notebook already includes this install gate:

- `experiments/tts/T1_sesame_csm_finetune/T1_adja_csm_finetune.ipynb`

### Why this repins `trl`

If you see:

```python
ImportError: cannot import name 'ConstantLengthDataset' from 'trl.trainer.utils'
```

then `unsloth` is installed, but `unsloth_zoo` is importing against an older
`trl` API. For the current recovery path, pin:

```python
%pip install -U --force-reinstall --no-deps "trl==0.19.1"
```

before retrying `import unsloth`.

## Fix the compiled-autograd patch crash

If model loading fails inside:

```python
unsloth_zoo.patching_utils.patch_compiled_autograd
```

with:

```python
AttributeError: 'NoneType' object has no attribute 'group'
```

then the pinned `unsloth_zoo` build is trying to rewrite a torch source snippet
that does not exactly match your Colab runtime. This is separate from the install
problem and happens later, during `FastModel.from_pretrained(...)`.

Run this hotfix cell once, before the model-load cell:

```python
import importlib
import inspect
import re
import torch
import torch._dynamo.compiled_autograd

def apply_compiled_autograd_hotfix():
    fx = torch._dynamo.compiled_autograd.AutogradCompilerInstance.end_capture
    if fx.__name__ == "unsloth_end_capture":
        return

    source = inspect.getsource(fx)
    if "with disable()" in source:
        return

    spaces = source.find("def")
    source = source.split("\n")
    source = "\n".join(x[spaces:] for x in source)
    old = "return compiled_fn(inputs, sizes, scalars, hooks)"
    match = re.search(r"\n([ ]{1,})return compiled_fn", source)
    n = len(match.group(1)) if match else 0
    source = source.replace(old, f"with disable():\n{' ' * (n + 4)}{old}")
    source = source.replace("def end_capture", "def unsloth_end_capture", 1)

    all_items = dir(torch._dynamo.compiled_autograd)
    good_items = [x for x in all_items if x in source]
    exec("from torch._dynamo.compiled_autograd import (" + ", ".join(x for x in good_items) + ")", globals())
    exec(source, globals())
    torch._dynamo.compiled_autograd.AutogradCompilerInstance.end_capture = unsloth_end_capture

    try:
        dynamo_misc = importlib.import_module("torch._dynamo.variables.misc")
        fx = dynamo_misc.AutogradEngineVariable.call_method
    except Exception:
        return

    if fx.__name__ == "unsloth_call_method":
        return

    source = inspect.getsource(fx)
    if "in_compiled_autograd_region" in source:
        return

    spaces = source.find("def")
    source = source.split("\n")
    source = "\n".join(x[spaces:] for x in source)
    source = source.replace(
        "torch._dynamo.compiled_autograd.compiled_autograd_enabled",
        "torch._dynamo.compiled_autograd.in_compiled_autograd_region",
        1,
    )
    source = source.replace("def call_method", "def unsloth_call_method", 1)

    all_items = dir(dynamo_misc)
    good_items = [x for x in all_items if x in source]
    exec("from torch._dynamo.variables.misc import (" + ", ".join(x for x in good_items) + ")", globals())
    exec(source, globals())
    dynamo_misc.AutogradEngineVariable.call_method = unsloth_call_method

apply_compiled_autograd_hotfix()
print("compiled-autograd hotfix applied")
```

This mirrors the guard that exists in newer upstream `unsloth_zoo` code, while
keeping the older CSM-compatible pin set for the rest of the notebook.

### Official force-reinstall fallback

If `unsloth` still does not import, use the official Unsloth reinstall fallback from their pip install docs:

```python
%pip uninstall -y unsloth unsloth_zoo
%pip install --upgrade --force-reinstall --no-cache-dir --no-deps git+https://github.com/unslothai/unsloth-zoo.git
%pip install --upgrade --force-reinstall --no-cache-dir --no-deps git+https://github.com/unslothai/unsloth.git
```

Then rerun the verification cell above.

## Minimal Adja edits to the upstream notebook

### 1. Dataset cell

Replace the upstream dataset load:

```python
raw_ds = load_dataset("MrDragonFox/Elise", split = "train")
```

with:

```python
from google.colab import userdata
token = userdata.get("HF_TOKEN")
raw_ds = load_dataset("JosueG/adja-tts-orpheus", token=token, split="train")
```

### 2. Speaker handling

Keep a single-speaker default for now:

```python
speaker_key = "source"
if "source" not in raw_ds.column_names:
    raw_ds = raw_ds.add_column("source", ["0"] * len(raw_ds))
```

### 3. Text normalization

Before passing text to `apply_chat_template(...)`, NFC-normalize it:

```python
import unicodedata

def normalize_text(text):
    return " ".join(unicodedata.normalize("NFC", text.strip()).split())
```

and use:

```python
{"type": "text", "text": normalize_text(example["text"])}
```

### 4. Filter over-long audio before preprocessing

`processor.apply_chat_template(...)` pads shorter audio up to
`audio_kwargs["max_length"]`, but it does not reliably truncate longer clips for
this CSM path. If clips longer than `240001` samples reach the default Trainer
collator, training fails with a tensor stacking error like:

```python
ValueError: expected sequence of length 280800 at dim 2 (got 240001)
```

So before `map(...)`, filter the dataset:

```python
MAX_AUDIO_SAMPLES = 240001
raw_ds = raw_ds.filter(
    lambda ex: len(ex["audio"]["array"]) <= MAX_AUDIO_SAMPLES,
    desc="Filtering over-long audio",
    load_from_cache_file=False,
)
```

and force a fresh preprocessing run:

```python
processed_ds = raw_ds.map(
    preprocess_example,
    remove_columns=raw_ds.column_names,
    desc="Preprocessing Adja dataset",
    load_from_cache_file=False,
)
```

If you changed the preprocessing logic after an earlier failed run, restart the
runtime or rerun the dataset + preprocessing cells so you do not keep stale
cached rows.

### 5. Keep the CSM preprocessing shape

Do not simplify this into text-only tokenization.

The example must still go through:

- `processor.apply_chat_template(...)`
- `output_labels=True`
- required keys:
  - `input_ids`
  - `attention_mask`
  - `labels`
  - `input_values`
  - `input_values_cutoffs`

### 6. First success gate

Before a full run:

- run the compiled-autograd hotfix cell
- subset to 5 training examples
- train for 2 steps
- generate one plain waveform
- generate one speaker-conditioned waveform

## When to apply the repo workaround

If the upstream notebook fails with the known CSM patch issue, switch to the repo workaround:

- pin:
  - `unsloth==2025.5.4`
  - `unsloth_zoo==2025.5.6`
  - `transformers==4.52.3`
  - `trl==0.19.1`
- change:
  - `use_gradient_checkpointing="unsloth"`
  - to
  - `use_gradient_checkpointing=True`

This is the exact workaround encoded in:

- `scripts/hf_jobs/T1_sesame_csm_finetune.py`

## Hard stop

If the upstream notebook plus the workaround still fails:

1. run `scripts/hf_jobs/T1_csm_vanilla.py`
2. if that does not clearly isolate the failure, pivot to:
   - `scripts/hf_jobs/T3_spark_tts_finetune.py`
