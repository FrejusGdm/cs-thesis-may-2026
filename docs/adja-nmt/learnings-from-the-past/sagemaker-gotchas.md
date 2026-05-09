# SageMaker Training Gotchas — Adja NMT/TTS/ASR Project

Learned 2026-04-26 during the first SageMaker submission wave (S1–S7, CF1–CF3, TK1, AT1, S2, S3, S6).
Written for AI agents picking up this work. Every item here caused a real job failure.

---

## 1. PyTorch Container Version — Use 2.5.1, NOT 2.1.0

**What happened:** All CF1/CF2/S7 jobs crashed within 4 minutes with:
```
AttributeError: module 'torch.utils._pytree' has no attribute 'register_pytree_node'
```

**Root cause:** `transformers>=4.50.0` (needed for `CsmForConditionalGeneration`) installs
transformers 4.52.x. That version requires `torch.utils._pytree.register_pytree_node`
which was **added in PyTorch 2.3.0**. The default SageMaker container (2.1.0) doesn't have it.

**Fix in `launch.py`:**
```python
# WRONG — too old for transformers 4.50+
framework_version="2.1.0",
py_version="py310",

# CORRECT
framework_version="2.5.1",
py_version="py311",
```

**Also update requirements.txt extra-index-url:**
```
# 2.5.1 container uses CUDA 12.4
--extra-index-url https://download.pytorch.org/whl/cu124
```

**Verified against local SDK config** at:
`<LOCAL_PATH>`
- 2.3.0 → py311 ✓
- 2.5.1 → py311 ✓
- 2.6.0 → py312 (use only if you need the newer features)

---

## 2. gradient_checkpointing_enable() + manual backward outside autocast = crash

**What happened:** S7 (Whisper-tiny) crashed with:
```
RuntimeError: Trying to backward through the graph a second time (or directly access
saved tensors after they have already been freed).
```

**Root cause:** In PyTorch 2.5.1, calling `.backward()` outside a `torch.autocast` context
while gradient checkpointing is active triggers a recompute pass without the correct dtype
context. Scripts that do this:
```python
with torch.autocast("cuda", dtype=torch.bfloat16, ...):
    loss = model(...).loss
# ← autocast context EXITED here
(loss / grad_accum).backward()   # ← CRASH with gradient checkpointing
```

**Fix A (small models — Whisper-tiny, 39M params):** Just remove `gradient_checkpointing_enable()`.
A10G has 24 GB; models under ~300M params don't need it.

**Fix B (large models — Whisper large-v3, CSM 1B):** Move backward inside the autocast block:
```python
with torch.autocast("cuda", dtype=torch.bfloat16, ...):
    loss = model(...).loss
    if loss is None or torch.isnan(loss): continue
    (loss / grad_accum).backward()   # ← safe inside context
```

**Fix C (HuggingFace Trainer):** No fix needed. Trainer handles autocast + backward
internally and works correctly with gradient_checkpointing_enable(). Only manual loops break.

---

## 3. Spot Quota for g5.2xlarge is 0 — Use On-demand

**What happened:** All spot submissions to g5.2xlarge failed at submission time:
```
ResourceLimitExceeded: ml.g5.2xlarge for spot training job usage is 0 Instances
```

**Fix:** Always pass `--no-spot` flag, or in launch.py set `use_spot=False` for g5.2xlarge.
The on-demand quota IS available. Spot saving (~30%) is not worth the quota fight.

```bash
python scripts/sagemaker_jobs/launch.py --experiment CF1 --no-spot
```

---

## 4. Region Mismatch — Jobs Must Go to us-west-2

**What happened:** `aws configure get region` returned `us-east-2`. Jobs submitted to
us-east-2 had 0 quota for g5.2xlarge on-demand too.

**Fix:** Always set `AWS_DEFAULT_REGION=us-west-2` before submitting:
```bash
AWS_DEFAULT_REGION=us-west-2 python scripts/sagemaker_jobs/launch.py --experiment CF1 --no-spot
```

Or just add to your shell profile:
```bash
export AWS_DEFAULT_REGION=us-west-2
```

---

## 5. HF_TOKEN Not Loaded Automatically → Silent SystemExit

**What happened:** CF1 failed after 4 minutes with empty error message `""`. Scripts that
do `if not HF_TOKEN: raise SystemExit("HF_TOKEN env var is required")` exit with code 1
but no stack trace, so the SageMaker failure reason is blank.

**Root cause:** `HF_TOKEN` was in `.env` but not exported to shell before running launch.py.

**Fix:** Added dotenv auto-load to `launch.py`:
```python
try:
    from dotenv import load_dotenv
    _env = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
    load_dotenv(os.path.abspath(_env), override=False)
except ImportError:
    pass
```

`override=False` means shell environment wins over `.env` (safe default).
The `.env` file lives at repo root: `adja-nmt/.env` and contains `HF_TOKEN=hf_...`

---

## 6. PEP 723 Script Headers Are Ignored by SageMaker

**What happened:** CF1/CF2/CF3 scripts have inline dependency blocks at the top:
```python
# /// script
# dependencies = ["torch==2.5.1", "transformers==4.52.3", ...]
# ///
```

These are **PEP 723 format** — processed by `uv run script.py` but completely ignored
by Python and SageMaker. SageMaker ONLY reads `requirements.txt` from the `source_dir`.

**Implication:** If a script lists a package in its PEP 723 header but not in
`requirements.txt`, it will NOT be installed. The headers are just documentation.

**Action:** Treat `requirements.txt` as the single source of truth for SageMaker.
Update headers to match if you want them kept for local `uv run` usage.

---

## 7. Missing Packages in requirements.txt

Packages that turned out to be needed but weren't listed:

| Package | Used by | Why missing |
|---------|---------|-------------|
| `sentencepiece>=0.2.0` | CF1/CF2/CF3 (LlamaTokenizer for CSM) | Not an explicit import — transformers needs it at model load |
| `librosa>=0.10.0` | S2/S3/S6 (audio resampling in safe_load_audio_array) | Assumed pre-installed in container |
| `bitsandbytes>=0.45.0` | S6 (adamw_8bit optimizer in TrainingArguments) | Not an explicit import — HF Trainer needs it for adamw_8bit |

**Rule:** If a script uses `optim="adamw_8bit"`, add bitsandbytes. If it uses librosa
anywhere for resampling, add librosa. If it loads a Llama/CSM tokenizer, add sentencepiece.

---

## 8. SageMaker IAM Role — Not in Shell by Default

**What happened:** Submission failed with:
```
Couldn't call 'get_role' to get Role ARN from role name josue-cli
```
Then fell through to `SAGEMAKER_ROLE` env var which was also unset.

**Fix:** Always export before submitting (or put in `.env`):
```bash
export SAGEMAKER_ROLE="arn:aws:iam::974640818655:role/service-role/AmazonSageMaker-ExecutionRole-20260426T083875"
```

The warning `Couldn't call 'get_role'...` is harmless as long as `SAGEMAKER_ROLE` is set.
It just means the code tried `sagemaker.get_execution_role()` first (works in SageMaker
notebooks, not from local CLI) then fell back to the env var.

---

## 9. peft/accelerate Upper Bound Pins Were a 2.1.0 Workaround

**Background:** requirements.txt previously had:
```
peft>=0.10.0,<0.14.0      # peft 0.14+ required torch>=2.3.0
accelerate>=0.28.0,<0.34.0  # accelerate 0.34+ required torch>=2.3.0
```

These pins were added because on torch 2.1.0, installing a newer peft/accelerate
caused pip to pull in a CPU-only torch from PyPI, overwriting the CUDA container version.

**With PyTorch 2.5.1:** these pins are no longer needed. torch 2.5.1 satisfies any
`torch>=2.3.0` requirement, so pip won't try to install a newer torch. Current requirements:
```
peft>=0.13.0      # no upper bound
accelerate>=0.30.0  # no upper bound
```

---

## 10. Output Locations

SageMaker automatically tars and uploads:
- `SM_MODEL_DIR` → `s3://<bucket>/<job-name>/output/model.tar.gz`
- `SM_OUTPUT_DATA_DIR` → `s3://<bucket>/<job-name>/output/output.tar.gz`

Default bucket: `sagemaker-us-west-2-974640818655`

To download results locally:
```bash
AWS_DEFAULT_REGION=us-west-2 aws s3 cp \
  s3://sagemaker-us-west-2-974640818655/<job-name>/output/output.tar.gz \
  results/<EXP>/output.tar.gz

tar -xzf results/<EXP>/output.tar.gz -C results/<EXP>/
```

Generated audio + metrics land in `output.tar.gz`, not `model.tar.gz`.
The adapter/checkpoint is in `model.tar.gz`.

---

## 11. g5.12xlarge = 4× A10G, NOT A100

Common confusion: `g5.12xlarge` sounds like it might be A100-class.
It is **4× NVIDIA A10G GPUs, 24 GB each = 96 GB total VRAM**.
A10G is Ampere-generation (GA102 chip), similar era to A100 but lower-tier.
A100s are on `p4d.24xlarge` (8× A100 40GB) or `p5.48xlarge` (8× H100 80GB).

For most of our experiments, 4× A10G is plenty.

---

## 12. WaxalNLP `google/WaxalNLP` — Cannot Use Hub Builder in datasets>=2.18

**What happened:** All WaxalNLP-using jobs (S1, S2, S3, S6) failed with:
```
datasets.table.CastError: Couldn't cast
  audio: struct<bytes: binary, path: string>
to Audio(sampling_rate=None, mono=True, decode=True)
because column names don't match
```

**Root cause — two compounding issues:**
1. The Hub builder declares `Audio()` features. In datasets>=2.18 (installed in the
   PyTorch 2.5.1 DLC), the `Audio` type's Arrow representation changed, causing the
   cast from the raw parquet `struct<bytes, path>` to `Audio()` to fail during
   `download_and_prepare`.

2. WaxalNLP parquet shards are **inconsistent**: some shards include `__index_level_0__`
   (a pandas index column) and some don't. Any hardcoded `features=` schema either
   fails on shards that have the column (if it's not in the schema) or fails on shards
   that don't have it (if it IS in the schema). The `_is_schema_cast_error` fallback
   pattern does NOT solve this — both branches hit different shards and each fails.

**Wrong approach (tried, failed):**
```python
# Attempt 1: hardcode features with Audio() → fails on Audio cast
# Attempt 2: hardcode raw schema without __index_level_0__ → fails on shards that have it
# Attempt 3: try plus_index (with __index_level_0__) → fails on shards without it
# All fallback chains hit different shard variants and fail
```

**Correct fix:**
Use `load_dataset("parquet", data_files=hf://...)` to bypass the Hub builder entirely.
This avoids ALL schema-level issues: no Audio() cast, no fixed schema requirement.

**Actual WaxalNLP parquet file paths (verified 2026-04-27):**
```
data/ASR/ewe/ewe-{split}-NNNNN.parquet   # ewe_asr config
data/TTS/ewe/ewe-{split}-NNNNN.parquet   # ewe_tts config
```

**Pattern to use in every script:**
```python
_WAXAL_PARQUET_PATHS = {
    "ewe_asr": {
        "train":      "hf://datasets/google/WaxalNLP/data/ASR/ewe/ewe-train*.parquet",
        "validation": "hf://datasets/google/WaxalNLP/data/ASR/ewe/ewe-validation*.parquet",
        "test":       "hf://datasets/google/WaxalNLP/data/ASR/ewe/ewe-test*.parquet",
        "unlabeled":  "hf://datasets/google/WaxalNLP/data/ASR/ewe/ewe-unlabeled*.parquet",
    },
    "ewe_tts": {
        "train": "hf://datasets/google/WaxalNLP/data/TTS/ewe/ewe-train*.parquet",
        "test":  "hf://datasets/google/WaxalNLP/data/TTS/ewe/ewe-test*.parquet",
    },
}

def load_waxal_split(*, dataset="google/WaxalNLP", config, split, cache_dir, token):
    from datasets import load_dataset
    hf_path = _WAXAL_PARQUET_PATHS[config][split]
    ds = load_dataset("parquet", data_files={split: hf_path},
                      split=split, token=token, cache_dir=cache_dir)
    if "__index_level_0__" in (ds.column_names or []):
        ds = ds.remove_columns("__index_level_0__")
    return ds
```

**datasets version behavior differences (CRITICAL):**
- datasets 2.18.x (SageMaker 2.5.1 DLC): `cast_table_to_schema` requires EXACT bidirectional
  column matching. Both extra source columns AND missing target columns cause failure.
- datasets 4.x (local): missing target columns are filled with nulls (no error); only extra
  source columns cause failure.
- This means: even if source has `__index_level_0__` and target doesn't, 2.18.x fails.
  And even if target has `__index_level_0__` and source doesn't, 2.18.x also fails.

**Note on AT1:** AT1 uses the old Hub builder approach but runs fine because it was
submitted with the PyTorch 2.1.0 container (datasets<2.18). The cast issue only
manifests in the 2.5.1 container.

**Note on S6:** S6 loads both ewe_tts (consistent schema) and ewe_asr/unlabeled (mixed
schema). The ewe_tts loading succeeds, but ewe_asr/unlabeled fails without the two-group fix.
The "Uploading" secondary status seen briefly means SageMaker started the upload process
when it thought training was ending, not that training succeeded.

---

## 13. `HF_HUB_ENABLE_HF_TRANSFER=1` — Do NOT Set Without Installing `hf_transfer`

**What happened:** T3A/T3C crashed immediately with:
```
ModuleNotFoundError: No module named 'hf_transfer'
```

**Root cause:** Setting `os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"` causes
`huggingface_hub` to try to import `hf_transfer`, which is not installed in the
SageMaker PyTorch DLC. Any `snapshot_download` or `hf_hub_download` call then fails.

**Fix:** Remove `HF_HUB_ENABLE_HF_TRANSFER` from scripts. The 3600s timeout alone is
sufficient. Adding `hf-transfer` to requirements.txt would also work but is not needed.

---

## 14. SageMaker Default `volume_size=30 GB` Is Too Small for WaxalNLP-Unlabeled Jobs

**What happened:** AT1 (`adja-at1-full-20260426-114103`) hung silently for 24 h at the
HF datasets "Downloading data" progress bar around shard 39/106. `describe-training-job`
showed `SecondaryStatus: Training`, `LastModifiedTime` frozen at training-start,
`FailureReason: null` — no crash, no log line, just a frozen bar.

**Root cause:** When the `PyTorch` estimator is constructed without a `volume_size=`
argument, SageMaker provisions a **30 GB EBS volume** for the entire container
(root + `/opt/ml/*` + `/tmp`). `launch.py` redirects `HF_DATASETS_CACHE=/tmp/hf_cache`
to dodge the small `/root` partition, but `/tmp` lives on that same 30 GB volume.

`google/WaxalNLP` `ewe_asr` `unlabeled` is **~25–30 GB** (183 k clips, 106 parquet
shards). HF `datasets` writes **two copies** on disk:
1. Raw downloaded parquet shards under `…/downloads/`
2. Extracted/processed Arrow tables under the dataset name

So real disk usage ≈ 2× shards-downloaded. The volume fills around shard ~39, after
which `urllib3` blocks on the next `f.write(chunk)` (`ENOSPC`); HF datasets'
retry-with-resume catches the exception and silently retries forever — frozen bar,
no traceback, billing keeps ticking. Job will only be killed when SageMaker hits
`MaxRuntimeInSeconds`.

**Fix in `launch.py`:**
```python
# Per-experiment override in EXPERIMENT_CONFIGS
"AT1": { ..., "volume_size_gb": 200 },

# In main(), pass to the estimator
estimator = PyTorch(
    ...,
    volume_size=cfg.get("volume_size_gb", 30),
    ...
)
```

**Sizing decision (2026-04-27):** the launcher now defaults **every** experiment to 200 GB
(`cfg.get("volume_size_gb", 200)`). Rationale: EBS gp2 is ~$0.10/GB-month prorated to job
runtime, so an extra 170 GB on a 5 h CF* run costs ~$0.12 — not worth the cognitive load
of remembering which experiments need it. Override with `"volume_size_gb": N` per-experiment
only if you ever need more than 200 GB.

**Verify real dataset size before sizing the volume.** HF Hub exposes per-file sizes:
```bash
curl -s "https://huggingface.co/api/datasets/google/WaxalNLP/tree/main/data/ASR/ewe?recursive=true" \
  -H "Authorization: Bearer $HF_TOKEN" \
  | python3 -c "import json,sys; print(sum(it.get('size',0) for it in json.load(sys.stdin) if it['path'].endswith('.parquet'))/1e9, 'GB')"
```
WaxalNLP `ewe_asr` `unlabeled` is **56 GB raw parquet** (verified 2026-04-27); HF datasets stores both raw and extracted Arrow → ~112 GB on disk during download. Initial estimate of 25-30 GB was off by 2×, which is why the first version of this gotcha set S2/S3 to 150 GB — bumped to 200 GB after measuring.

**Verification after fix:**
```bash
aws sagemaker describe-training-job --training-job-name <name> --region us-west-2 \
  --query 'ResourceConfig.VolumeSizeInGB'   # should be 200, not 30
```

**Symptom to watch for:** if "Downloading data" stalls AND `LastModifiedTime` stops
advancing AND `FailureReason` is null, it's almost certainly disk-full — not a network
hang. The longer-term fix is staging WaxalNLP to S3 once and using a SageMaker
channel input so HF Hub never touches the container.

---

## 15. CUDA OOM on 24 GB A10G — Use g6e.xlarge (L40S 48 GB) for Big-Model Workloads

**What happened (2026-04-27):** Three separate jobs all OOM'd on 24 GB A10G:
- **S6** (Orpheus 3B + max_audio_sec=30 + grad-checkpointing) — OOM at step 0/11492
- **S1B** (Whisper-large-v3 1.5B FULL FT + adam-8bit + grad-checkpointing + bs=1) — OOM
- **T3C** (CSM 1B + Spark + curriculum mixing at max_seq_length=1024) — OOM, retried 9× before diagnosis

**Root cause:** for 1B+-parameter models with long sequences and gradient checkpointing,
activation memory dominates. 8-bit Adam saves optimizer-state RAM but doesn't help
activations. There is no further squeeze on 24 GB.

**Fix:** move to **`ml.g6e.xlarge`** — 1× L40S 48 GB Ada-gen GPU at $1.86/hr.
Same single-GPU footprint as g5.2xlarge but **2× the VRAM and ~2× faster**.
Often beats `ml.g5.12xlarge` (4× A10G 24 GB total 96 GB) for single-GPU jobs because
each individual GPU can hold the full model + activations without sharding overhead.

**Rule of thumb (us-west-2 quotas as of 2026-04-27):**

| If your job needs… | Pick |
|---|---|
| ≤ 24 GB VRAM, single GPU | `ml.g5.2xlarge` (cheapest, $1.69/hr) |
| 24–48 GB VRAM, single GPU | **`ml.g6e.xlarge`** ($1.86/hr) — sweet spot |
| Multi-GPU DDP at 24 GB/GPU | `ml.g5.12xlarge` (4× A10G) |
| Multi-GPU DDP at 48 GB/GPU | `ml.g6e.12xlarge` (4× L40S) — file quota request, often fast approval |
| 80 GB VRAM (final-paper run) | `ml.p4d.24xlarge` (8× A100) — slow approval, file early |

**Spot warning:** spot quota for `ml.g6e.xlarge` is also 0 in us-west-2 (same as g5.2xlarge per gotcha #3). Always pass `--no-spot` for g6e jobs.

---

## 16. CUDA-Fork Hazard — Load Model AFTER `.map(num_proc=N)`

**What happened (2026-04-27):** `adja-at1-full-20260427-104104` ran on `ml.g5.12xlarge` (200 GB volume — disk was fine), reached `decode+encode (num_proc=8): 0/183920`, then **silently froze for 3+ hours** with GPU 0%, CPU 0.7%, GPU memory pinned at 28.5%. No traceback, no log line, no progress. CloudWatch metrics screenshot was the smoking gun — model was loaded in VRAM but doing nothing.

**Root cause:** Python's `multiprocessing` library uses `fork()` to create worker processes (default on Linux). Forking a process with an **already-initialized CUDA context** is unsafe — child workers inherit broken CUDA state and deadlock on their first CUDA op (silently — there's no exception, just an indefinite wait).

The bad pattern in [train_AT1_audio_lm_csm_ewe.py](../scripts/sagemaker_jobs/train_AT1_audio_lm_csm_ewe.py) at submission time:

```python
model = CsmForConditionalGeneration.from_pretrained(...).to("cuda")  # ← initializes CUDA in parent
processor = AutoProcessor.from_pretrained(...)
...
encoded = unlab.map(encode_audio_only, num_proc=8, ...)              # ← fork() — workers deadlock
```

**The rule:**

> **Load any tokenizer / processor / feature_extractor BEFORE `.map(num_proc=N)`** — workers need them, and they're CPU-only so they don't initialize CUDA.
> **Defer `model.from_pretrained(...).to("cuda")` until AFTER all `.map(num_proc=N)` calls complete.** No exceptions.

The good pattern (now in train_AT1, train_S2_S3, partially in train_S6):

```python
processor = AutoProcessor.from_pretrained(...)        # CPU-only — safe before fork
encoded = unlab.map(encode_fn, num_proc=8, ...)       # fork() — clean (no CUDA in parent)

# All .map() calls done. Now safe to init CUDA.
model = CsmForConditionalGeneration.from_pretrained(...).to("cuda")
trainer = Trainer(model=model, train_dataset=encoded, ...)
trainer.train()
```

If you absolutely must use a CUDA model inside a multiprocess `.map()` (rare — typically the work can be done CPU-side), force `num_proc=1`. With one GPU, multi-process .map is useless anyway.

**Symptom to watch for:** SageMaker `LastModifiedTime` frozen at training-start, `SecondaryStatus=Training` indefinitely, `FailureReason=null`, CloudWatch metrics showing GPU 0% / CPU < 1% / GPU memory non-zero. That combination = CUDA-fork deadlock 100% of the time. Stop the job and audit your `.map(num_proc=...)` calls for upstream `.to("cuda")`.

---

## 17. Don't store padded raw audio in `.map()` output

**What happened (2026-04-27 20:00 EDT, 4th AT1 failure mode this day):** `adja-at1-full-20260427-192004` failed with `ClientError: Please use an instance type with more memory` at `decode+encode 6255/183920 (3.4%)`. **Not** CUDA OOM — host RAM. The script's `processor.apply_chat_template(..., audio_kwargs={"padding": "max_length", "max_length": 720001, ...})` produced `input_values` of shape `[1, 720001]` float32 = **2.88 MB per clip**. Across 183 k clips that's **~527 GB** of padded audio in the encoded dataset. Host RAM blew up under the combination of Arrow writer buffers + multiprocess COW pages (8 workers × 56 GB MP3-bytes WaxalNLP source).

**Rule:** when you `.map()` over an audio dataset, **don't store padded raw audio in the output rows**. Two viable patterns:

1. **Variable-length with batch-time padding.** Set `padding=False` in `apply_chat_template`. Store ragged sequences (HF datasets supports `Sequence(Sequence(float32))`). Add a custom data collator that pads to the longest in each training batch. AT1 v2 uses this — see `at1_collate` in [train_AT1_audio_lm_csm_ewe.py](../scripts/sagemaker_jobs/train_AT1_audio_lm_csm_ewe.py).

2. **Precompute codec tokens once.** Run a separate cheap job that encodes audio → codec tokens (Mimi/SNAC/etc.) → S3 parquet. Train job mounts the cache via SageMaker channel input and skips audio encoding entirely. The output dataset stores ~6 KB/clip of int16 codes vs 2.88 MB/clip of padded float32 audio (~480× smaller). Pattern: [precompute_waxal_tokens.py](../scripts/sagemaker_jobs/precompute_waxal_tokens.py) (SNAC for S6) and [precompute_waxal_mimi_tokens.py](../scripts/sagemaker_jobs/precompute_waxal_mimi_tokens.py) (Mimi for AT1, future-iteration).

Pattern 1 is faster to ship; pattern 2 is the architectural fix and lets you re-run training many times without re-encoding (worth it if you're iterating on the LM hyperparameters but the codec is fixed).

**Symptom to watch for:** `.map()` succeeds for a few thousand rows then `ClientError: Please use an instance type with more memory`. That's host RAM, not GPU. Check that the `.map()` output rows aren't gigantic (> 1 MB per row is suspicious for any non-image task).

---

## 18. SageMaker empty-string hyperparameters get mangled by argparse

**What happened (2026-04-28 EDT, M0 first submit):** `adja-m0-full-20260427-235959` failed at startup with `ExitCode 2` (argparse error) because `extra_hyperparameters: {"config": ""}` in launch.py got serialized as `--config ` (trailing space, no value) on the command line. argparse then read the next flag (`--dataset`) as `--config`'s value, leaving `JosueG/adja-tts-orpheus` as a stray positional argument → exit 2 with no Python traceback.

**Failed command (visible in CloudWatch):**
```
M0_mimi_reconstruction_test.py --config  --dataset JosueG/adja-tts-orpheus --n-samples 30 ...
```
Note the double space between `--config` and `--dataset` — the empty-string value vanished but the flag remained.

**Rule:** **Never pass an empty string as a SageMaker `extra_hyperparameters` value.** SageMaker constructs the CLI as `--key value` pairs and an empty value emits a bare flag.

**Workarounds (pick one):**
- **Best:** make the script's argparse default match the desired empty-string behavior, then OMIT the key from `extra_hyperparameters` entirely. Done for M0 (script default `--config ""`, no `config` key in launch.py).
- Use a sentinel value (`"none"`, `"NONE"`) and have the script treat it as "no value".
- For boolean flags, use `"flag-name": ""` is actually OK because argparse `action="store_true"` doesn't expect a value — the bare flag works. The bug only bites on `type=str` arguments.

---

## 19. Don't `rmtree` paths under `/opt/ml/output/data/` before script exit

**What happened (2026-04-28 EDT):** M0 v2 (`adja-m0-full-20260428-001339`) ran successfully, computed verdict `SC=0.4425 PASS`, and wrote 60 WAVs to `/opt/ml/output/data/mimi_recon_test/`. Then the script's local-Mac cleanup block ran `shutil.rmtree(out_dir, ignore_errors=True)` — wiping the WAVs. SageMaker's auto-upload of `SM_OUTPUT_DATA_DIR → s3://…/output/output.tar.gz` happens AFTER your script returns; by then the directory was empty. EXTRACT_M0 then failed with `botocore HeadObject 404` because the file didn't exist.

**Rule:** `SM_OUTPUT_DATA_DIR` (= `/opt/ml/output/data`) and `SM_MODEL_DIR` (= `/opt/ml/model`) are **owned by SageMaker** for the post-script upload. Don't delete anything in them yourself. Restrict cleanup to `/tmp/...` and similar truly-ephemeral paths.

**Symptom to watch for:** SageMaker reports `Status=Completed` but `s3://…/output/output.tar.gz` is missing or has size 0. The CloudWatch log shows the script finished cleanly. Nothing useful in S3.

**Fix pattern:**
```python
if str(out_dir).startswith("/tmp/"):
    shutil.rmtree(out_dir, ignore_errors=True)
# else: SageMaker uploads it; don't touch.
```

---

## 20. transformers 4.50+ blocks legacy `torch.load` checkpoints (CVE-2025-32434)

**What happened (2026-04-28 EDT, M2 first submit):** `adja-m2-full-20260428-003259` failed at startup with:
```
ValueError: Due to a serious vulnerability issue in `torch.load`, even with `weights_only=True`, we now require users to upgrade torch to at least v2.6 in order to use the function. ... See https://nvd.nist.gov/vuln/detail/CVE-2025-32434
```
when calling `Wav2Vec2Model.from_pretrained("facebook/mms-300m")`. M2b would have hit the same bug ~30 min later loading `facebook/mms-1b-all`.

**Why it bites SageMaker specifically:** the SageMaker PyTorch DLC pins torch 2.5.1. transformers 4.50.x (which we install via requirements.txt for `CsmForConditionalGeneration` + other features) refuses to load any checkpoint stored in the legacy `torch.load` format unless torch >= 2.6 OR the file is in safetensors. **`facebook/wav2vec2-*` and `facebook/mms-*` are stored in legacy format**, so they trigger the safety check.

**Rule:** any script that loads an older HF checkpoint via `from_pretrained` (especially `facebook/wav2vec2-*`, `facebook/mms-*`, anything that predates safetensors) needs the monkeypatch:

```python
def _noop(*a, **kw): return None
try:
    import transformers.utils.import_utils as _tf_iu
    import transformers.modeling_utils as _tf_mu
    _tf_iu.check_torch_load_is_safe = _noop
    _tf_mu.check_torch_load_is_safe = _noop  # patch call site too (local binding)
    print("[compat] check_torch_load_is_safe patched")
except Exception as _e:
    print(f"[compat] could not patch: {_e}")
```

**Important:** patch BOTH `transformers.utils.import_utils` AND `transformers.modeling_utils`. The `modeling_utils` module re-imports the function as a local binding at import time, so patching only `import_utils` doesn't reach the call site. (S2/S3 train script learned this the hard way — see [train_S2_S3_wav2vec2_ssl_ewe.py:288-298](../scripts/sagemaker_jobs/train_S2_S3_wav2vec2_ssl_ewe.py:288).)

**Already patched in:** S2/S3, M2, M2b. **Should also be in:** any future script that loads `facebook/wav2vec2-*` / `facebook/mms-*` / older checkpoints.

---

## 21. OneCycleLR `total_steps` must use `math.ceil(len/batch)`, not floor division

**What happened (2026-04-28 EDT, M1):** `adja-m1-full-20260428-003246` ran all 3510 training steps successfully, then crashed on a redundant `scheduler.step()` call with:
```
ValueError: Tried to step 3511 times. The specified number of total steps is 3510
```
at [train_M1_mimi_acoustic.py:295](../scripts/sagemaker_jobs/train_M1_mimi_acoustic.py:295). **No checkpoint was saved** — the save at [line 323-328](../scripts/sagemaker_jobs/train_M1_mimi_acoustic.py:323) runs AFTER the training loop returns, and the crash inside the loop short-circuits the post-loop save. Lost ~22 min of GPU time + the entire Mimi-FT run.

**Root cause:** [train_M1:216](../scripts/sagemaker_jobs/train_M1_mimi_acoustic.py:216) computes:
```python
total_steps = (len(combined) // args.mimi_batch) * args.mimi_epochs    # FLOOR division
```
But the inner training loop runs `math.ceil(len(combined) / args.mimi_batch)` iterations per epoch (it processes any partial final batch). When `len(combined) % args.mimi_batch != 0`, `total_steps` is undercounted by 1 per epoch, and the last `scheduler.step()` exceeds the schedule.

**Fix:**
```python
import math
total_steps = math.ceil(len(combined) / args.mimi_batch) * args.mimi_epochs
```

Alternative: wrap `scheduler.step()` in `try/except ValueError` — works but masks the real bug; prefer the `math.ceil` fix.

**Audit:** any script using `OneCycleLR(total_steps=...)` with a `for batch in DataLoader` style inner loop. M1 confirmed bitten. Worth checking AT1, S6, M2, M2b, T3* for the same pattern.

---

## 22. LoRA jobs hitting `MaxRuntimeExceeded` save the LAST epoch, not the BEST

**What happened (2026-04-28 EDT, S1-091126):** `adja-s1-full-20260427-091126` ran for 21 h and was Stopped by SageMaker with `MaxRuntimeExceeded`. SageMaker uploaded `model.tar.gz` (49 MB) to S3 — the LoRA adapter from the **final completed epoch (18)** with CER 18.33%, WER 53.19%. But the best metrics seen during the run were at **epoch 16: CER 17.80%, WER 51.86%** — about 0.5 CER points BETTER than what was saved. Without periodic checkpointing, that better adapter is gone.

**Why it happens:** Most train scripts in this repo save via `model.save_pretrained(...)` AFTER the training loop ends. Inside the training loop, only `[Ewe] Epoch N | loss=... | CER=X.XX% | WER=Y.YY%` is printed; no per-epoch checkpoint is uploaded to S3 unless the script explicitly does so. When SageMaker kills the container at MaxRuntime, only the most recent in-memory state survives the post-script upload. If that state is mid-epoch or the most recent completed epoch (which may not be the best), the saved artifact is suboptimal.

**Fixes (pick one):**

- **(A) Use HF `Trainer` with `load_best_model_at_end=True` + `save_strategy="epoch"` + `save_total_limit=1`**. Trainer saves to `output_dir` per epoch and at end loads the best by `metric_for_best_model`. Then the script does `trainer.model.save_pretrained(...)` AFTER training and you get the best-epoch state in `model.tar.gz`. This is what CF1/CF2 do. Custom training loops (S1, M1) bypass this and need fix B.
- **(B) For custom loops: track best metric in-loop and save to a *separate* path on improvement.** E.g., `if dev_cer < best_cer: model.save_pretrained(SM_MODEL_DIR + "/best")` inside the epoch loop. Then post-loop, the "best" subdir always has the best adapter.
- **(C) Use SageMaker `checkpoint_s3_uri`.** Set `checkpoint_s3_uri="s3://bucket/job-name/checkpoints"` in the `PyTorch` estimator, then `model.save_pretrained("/opt/ml/checkpoints/epoch_N")` per epoch; SageMaker syncs that prefix to S3 continuously. Survives MaxRuntime kill cleanly. Heavier ops but most resilient.

**Audit:** S1, M1, M2, M2b all use custom loops without best-checkpoint tracking. Add fix (B) to anything LoRA-flavored where MaxRuntime kill is a real risk.

---

## Quick Submission Checklist

Before submitting any new SageMaker job:

- [ ] `framework_version="2.5.1"`, `py_version="py311"` in launch.py
- [ ] `--extra-index-url https://download.pytorch.org/whl/cu124` in requirements.txt
- [ ] HF_TOKEN in `.env` (auto-loaded by launch.py dotenv block)
- [ ] SAGEMAKER_ROLE in `.env` or shell
- [ ] `AWS_DEFAULT_REGION=us-west-2` set
- [ ] `--no-spot` flag (spot quota = 0 for g5.2xlarge)
- [ ] Any new packages added to requirements.txt (check for librosa, bitsandbytes, sentencepiece)
- [ ] Manual training loops: backward is inside `torch.autocast` block
- [ ] Trainer-based scripts: gradient_checkpointing is fine as-is
- [ ] WaxalNLP loading: use `load_dataset("parquet", data_files=hf://...)` — NOT `load_dataset("google/WaxalNLP", features=...)` (see gotcha #12)
- [ ] Do NOT set `HF_HUB_ENABLE_HF_TRANSFER=1` unless `hf-transfer` is in requirements.txt (see gotcha #13)
- [ ] WaxalNLP-unlabeled jobs (AT1/M1/M2/M2b/S2/S3/S6) have `volume_size_gb` set in the registry — default 30 GB will silently hang at ~shard 39 (see gotcha #14)
- [ ] 1B+-parameter / long-sequence / grad-checkpointing job → use `ml.g6e.xlarge` (48 GB L40S), not `g5.2xlarge` (see gotcha #15)
- [ ] Any train script with `.map(num_proc=N)` does **not** call `.to("cuda")` before that map — defer model init to after preprocessing (see gotcha #16)
- [ ] Always `--no-spot` for `g5.2xlarge` AND `g6e.xlarge` (spot quota 0 for both)
- [ ] Audio `.map()` output stores variable-length sequences (NOT padded to max_length) AND has a custom data collator that pads at training time — or precomputes codec tokens to a separate cache (see gotcha #17)
- [ ] No `rmtree` / `os.remove` on paths under `/opt/ml/output/data/` or `/opt/ml/model/` (see gotcha #19)
- [ ] Scripts loading `facebook/wav2vec2-*` / `facebook/mms-*` / other legacy-format HF checkpoints have the CVE-2025-32434 monkeypatch BEFORE the first `from_pretrained` call (see gotcha #20)
- [ ] Scripts using `OneCycleLR(total_steps=...)` compute `total_steps = math.ceil(len/batch) * epochs`, not floor division (see gotcha #21)
- [ ] LoRA / custom-loop jobs at risk of `MaxRuntimeExceeded` have per-epoch best-checkpoint saving (see gotcha #22)
