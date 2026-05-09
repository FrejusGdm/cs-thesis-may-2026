# Qwen3 on Hugging Face Jobs: Retrospective and Lessons

Prepared for: Josue Godeme  
Requested by: Josue Godeme

This document explains how the Adja Qwen3 Hugging Face path was built, what had to change, what failed, what was learned, and what should happen next. It is written as an internal technical retrospective, not as a status-only note.

## If you read this a year from now

Start with:

1. [docs/qwen3-documentation-map.md](<LOCAL_PATH>
2. [results/qwen3_hf_jobs/README.md](<LOCAL_PATH>
3. this retrospective

That order gives you the navigation layer, the operational history, and then the engineering interpretation.

## Why this exists

The first Qwen3 submission path was structurally wrong for this repo:

- it tried to clone `adja-nmt` from inside the HF container
- it mixed local-repo assumptions with `hf jobs uv run`
- it treated the uploaded job like a full checkout, which HF does not do in this mode

That made the first failures mostly infrastructure failures, not model failures. The goal of this writeup is to separate:

- Hugging Face launcher mistakes
- general HF Jobs constraints
- Qwen3-ASR specific issues
- Qwen3-TTS specific issues

## Scope

This retrospective covers the Adja Qwen3 HF job work centered on:

- [scripts/hf_jobs/qwen3_asr_adja.py](<LOCAL_PATH>
- [scripts/hf_jobs/qwen3_tts_adja.py](<LOCAL_PATH>
- [scripts/hf_jobs/submit_qwen3_jobs.py](<LOCAL_PATH>
- [experiments/finetuning-qwen3/docs/adja_runbook.md](<LOCAL_PATH>
- [results/qwen3_hf_jobs/README.md](<LOCAL_PATH>

The upstream/vendor Qwen tree under [experiments/finetuning-qwen3](<LOCAL_PATH> was intentionally preserved.

## Repo rule we had to respect

The repo already has a consistent HF Jobs pattern:

- keep upstream/vendor code intact
- add Adja-specific glue outside the vendor tree
- submit a self-contained script from `scripts/hf_jobs/`
- use `hf jobs uv run`
- do not bootstrap the job by cloning this repo inside the container

That pattern is visible in:

- [scripts/hf_jobs/parakeet_finetune.py](<LOCAL_PATH>
- [experiments/asr/D5_parakeet_finetune/README.md](<LOCAL_PATH>
- [scripts/hf_jobs/T1_sesame_csm_finetune.py](<LOCAL_PATH>

Once Qwen was forced into that same shape, the failures became meaningful.

## What we changed

### 1. We removed the container-side clone pattern

The first submission path used a container command that attempted:

```bash
git clone https://github.com/FrejusGdm/adja-nmt.git ...
```

That was wrong for this repo and wrong for `hf jobs uv run`.

Why it failed:

- the HF job environment did not have the right GitHub auth context
- even if it had worked, it would still have violated the repo’s established job pattern

What replaced it:

- direct local submission of `scripts/hf_jobs/qwen3_asr_adja.py`
- direct local submission of `scripts/hf_jobs/qwen3_tts_adja.py`
- helper wrapper [submit_qwen3_jobs.py](<LOCAL_PATH>

### 2. We made the launchers standalone

HF `uv run` uploads the target script file, not an entire local Python package structure. That means:

- local helper imports are unsafe unless the helper is also part of the uploaded execution unit
- `from qwen3_job_utils import ...` was a bad assumption

So the launchers were rewritten to inline the minimum required logic:

- env parsing
- logging
- dataset materialization
- JSONL writing
- validation
- upload-to-Hub behavior

### 3. We switched dependency resolution to inline `uv` metadata

The repo’s working HF scripts already rely on:

```python
#!/usr/bin/env -S uv run --script
# /// script
# dependencies = [...]
# ///
```

This matters because runtime bootstrapping inside the job turned out to be brittle:

- `python -m pip` failed because the UV image did not expose `pip` the way we expected
- `uv pip --system` installed into the wrong environment
- `uv pip --python /usr/bin/python3.11 ...` hit the externally-managed interpreter guard

The correct pattern here was not “find the right pip invocation.” It was “stop bootstrapping inside the script.”

### 4. We matched the repo’s self-contained job style

The final launcher shape now matches the rest of the repo:

- one script per HF job entrypoint
- dependencies declared in the UV header
- line-buffered logs
- env-driven execution
- no repo clone
- no mutation of the upstream vendor tree

## Submission timeline

### Attempt 1: wrong bootstrap model

- ASR: `69e30a7bac288e522d8efb5c`
- TTS: `69e30a8eac288e522d8efb5e`
- Failure: container-side Git clone
- Signature:
  - `fatal: could not read Username for 'https://github.com': No such device or address`

Interpretation:

- not a Qwen failure
- not a dataset failure
- not an Adja data-shape failure
- purely launcher design error

### Attempt 2: repo-native submission, but launcher still assumed local module availability

- ASR: `69e382dfac288e522d8efd10`
- TTS: `69e382e1cd8c002f31dfe942`
- Failure:
  - `ModuleNotFoundError: No module named 'qwen3_job_utils'`

Interpretation:

- confirmed that `hf jobs uv run` was not providing the full local module graph
- forced the standalone-script decision

### Attempt 3: standalone launchers, but wrong package bootstrap

- ASR: `69e384d0cd8c002f31dfe952`
- TTS: `69e384d1cd8c002f31dfe954`
- Failure:
  - `/usr/bin/python3.11: No module named pip`

Interpretation:

- the script still had the wrong responsibility
- package installation had to move out of runtime logic

### Attempt 4: package bootstrap partly improved, but still wrong model

- ASR: `69e38514cd8c002f31dfe956`
- TTS: `69e38515ac288e522d8efd1c`

Failures:

- ASR package resolution:
  - only `qwen-asr==0.0.6` was available in the HF runtime
- TTS runtime import:
  - `ModuleNotFoundError: No module named 'librosa'`

Interpretation:

- ASR dependency versions in the local requirements file were too optimistic for the HF runtime
- TTS proved that “installed” did not necessarily mean “installed into the interpreter actually running the script”

### Attempt 5: interpreter-targeted `uv pip`, still wrong pattern

- ASR: `69e3856aac288e522d8efd1e`
- TTS: `69e3856bcd8c002f31dfe964`
- Failure:
  - externally-managed `/usr` interpreter rejected package installation

Interpretation:

- this was the final confirmation that runtime package bootstrapping was the wrong path entirely

### Attempt 6: correct launcher shape, first real Qwen failures

- ASR: `69e3865bac288e522d8efd24`
- TTS: `69e3865dcd8c002f31dfe96e`

This was the first useful pair.

ASR result:

- got through dependency resolution
- got through dataset access
- got through split/materialization
- got through model download/load
- then failed in training setup:
  - `TrainingArguments.__init__() got an unexpected keyword argument 'evaluation_strategy'`

TTS result:

- got through dependency resolution
- got through dataset access
- got through split/materialization
- got through tokenizer/model download/load
- failed after the flash-attention fallback:
  - `RuntimeError('The size of tensor a (2048) must match the size of tensor b (1024) at non-singleton dimension 2')`

Interpretation:

- ASR had reached a compatibility issue with the runtime Transformers API
- TTS had reached a genuine model/data-path incompatibility, not a launcher problem

### Attempt 7: ASR compatibility fix + fuller TTS diagnostics

- ASR: `69e388ccac288e522d8efd26`
- TTS: `69e388cdcd8c002f31dfe98a`
- Status when this document was written: `RUNNING`

Intent:

- ASR: tolerate `evaluation_strategy` versus `eval_strategy` differences
- TTS: print fuller traceback and dimension diagnostics so the next failure is actionable

## What we learned about HF Jobs

### 1. `hf jobs uv run` wants self-contained scripts

Practical rule:

- assume HF uploads and runs the target script, not your whole repo as an importable local package

Implication:

- avoid local helper imports unless you explicitly package or inline them

### 2. Inline UV dependencies are the correct pattern here

Practical rule:

- put dependency declarations in the script header
- do not build a mini installer inside the job script unless you have a very specific reason

### 3. “The job started” is not the same as “the launcher is correct”

For Qwen, the quality ladder was:

1. container bootstrap succeeds
2. dependencies resolve
3. dataset access works
4. materialization works
5. model loads
6. train loop starts
7. checkpoint saves
8. inference/generation works

Only at stage 6+ do failures become model-meaningful.

### 4. HF logs are good enough to do real forensics, but only if the script prints enough

Needed:

- line-buffered stdout
- explicit traceback printing
- explicit config and dimension logging

Without that, long HF logs only tell you where the crash surfaced, not why.

## What we learned about Qwen3-ASR

### 1. The Adja JSONL path is viable

By attempt 6, ASR had already proven:

- private dataset access works
- Adja data can be split/materialized inside HF
- Adja text normalization and `language None<asr_text>...` formatting are accepted far enough to reach model/trainer setup

That means the first-pass Adja ASR data contract was reasonable.

### 2. The current blocker is not data size

The dataset is small enough for HF. The active ASR blocker was:

- a compatibility mismatch in the Trainer setup layer

That is much better than finding out the whole pipeline shape was invalid.

### 3. `qwen-asr` packaging on HF currently matters

The HF runtime resolved `qwen-asr==0.0.6`, not the newer spec we first assumed.

Implication:

- version assumptions from local docs or loose requirements are not enough
- HF-resolvable versions need to be treated as part of the operational contract

## What we learned about Qwen3-TTS

### 1. The first meaningful TTS failure is inside the model path

By attempt 6, TTS had already proven:

- private dataset access works
- dataset materialization works
- reference-audio attachment works
- code/tokenizer preparation gets far enough to load the model path

So the current TTS failure is not a trivial launcher error anymore.

### 2. The official fine-tuning path appears more 1.7B-shaped than 0.6B-shaped

Evidence:

- the vendored official script defaults to `Qwen/Qwen3-TTS-12Hz-1.7B-Base`
- our failure for the 0.6B path is a representation-size mismatch:
  - `2048` vs `1024`

This strongly suggests one of the following:

- the official fine-tuning path was validated primarily against 1.7B
- the 0.6B variant exposes a different hidden-size or embedding-size contract
- our inlined wrapper path is missing an adaptation step the official code assumes indirectly

This is not yet a final conclusion, but it is the leading hypothesis.

### 3. Flash attention is optional here, but not the real blocker

The TTS job already falls back from `flash_attention_2` to `eager`.

Meaning:

- missing flash-attn is a performance/runtime convenience issue
- the real blocker is the later tensor-shape mismatch in eager mode

### 4. The container does not provide `sox`

The logs also showed:

- `sox: not found`

That may or may not be fatal depending on which code path is taken, but it is part of the real runtime contract for this stack and should be documented.

## Distinguishing general infra issues from Qwen-specific issues

### General infra issues

- container-side clone bootstrap was wrong
- local helper imports were unsafe under `hf jobs uv run`
- runtime package installation was the wrong pattern
- script logging needed stronger traceback output

These lessons are reusable across models.

### Qwen-specific issues

- `qwen-asr` version availability in the HF environment
- ASR Trainer API compatibility around `evaluation_strategy`
- TTS 0.6B dimension mismatch after model load
- likely mismatch between the official TTS finetuning path and the 0.6B variant

These are specific to the Qwen stack or to the exact versions exposed in this environment.

## Practical next steps

### ASR

1. Wait for the current compatibility-fix retry to finish.
2. If it still fails, capture the exact new traceback and patch that layer only.
3. Once ASR reaches checkpoint save + one decode sample, freeze the launcher shape and stop changing infra around it.

### TTS

1. Wait for the current diagnostic retry to finish.
2. Use the fuller traceback to identify exactly where the 2048/1024 mismatch occurs.
3. Compare the 0.6B config against the assumptions in the official TTS finetuning path.
4. If the official path is effectively 1.7B-only, decide whether:
   - to patch the 0.6B path correctly, or
   - to move TTS experimentation to the 1.7B base on larger hardware

## What is already documented elsewhere

- runbook:
  - [experiments/finetuning-qwen3/docs/adja_runbook.md](<LOCAL_PATH>
- submission log:
  - [results/qwen3_hf_jobs/README.md](<LOCAL_PATH>
- chronological ledger:
  - [results/run-ledger.md](<LOCAL_PATH>
- experiment registry:
  - [experiments/registry.md](<LOCAL_PATH>

This document is the explanatory layer that ties those operational notes together.

## Sources

Local repo sources:

- [scripts/hf_jobs/parakeet_finetune.py](<LOCAL_PATH>
- [scripts/hf_jobs/T1_sesame_csm_finetune.py](<LOCAL_PATH>
- [learnings-from-the-past/hf-jobs-gotchas.md](<LOCAL_PATH>
- [experiments/finetuning-qwen3/vendor/qwen3_tts_official/dataset.py](<LOCAL_PATH>
- [experiments/finetuning-qwen3/vendor/qwen3_tts_official/sft_12hz.py](<LOCAL_PATH>
- [experiments/finetuning-qwen3/docs/01_repository_forensics.md](<LOCAL_PATH>

External sources previously consulted for this Qwen path:

- `QwenLM/Qwen3-ASR`
- `QwenLM/Qwen3-TTS`
- Hugging Face CLI help for `hf jobs run` and `hf jobs uv run`
