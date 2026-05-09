# Qwen3 Documentation Map

Prepared for: Josue Godeme  
Requested by: Josue Godeme

This is the navigation document for the Qwen3 Adja work. If you come back a year from now and do not remember the details, start here.

## Read this first

If you only have 10 minutes:

1. Read [results/qwen3_hf_jobs/README.md](<LOCAL_PATH>
2. Read [docs/qwen3-hf-jobs-retrospective.md](<LOCAL_PATH>
3. Read [experiments/finetuning-qwen3/docs/adja_runbook.md](<LOCAL_PATH>

That gives you:

- current and past job IDs
- what failed and why
- the corrected submission pattern
- the reasoning behind the design decisions

## Which document answers which question

If you want the latest operational history:

- [results/qwen3_hf_jobs/README.md](<LOCAL_PATH>

If you want the chronological record across the repo:

- [results/run-ledger.md](<LOCAL_PATH>
- [experiments/registry.md](<LOCAL_PATH>

If you want the exact submission commands and environment contract:

- [experiments/finetuning-qwen3/docs/adja_runbook.md](<LOCAL_PATH>
- [scripts/hf_jobs/submit_qwen3_jobs.py](<LOCAL_PATH>

If you want the engineering explanation of what changed and why:

- [docs/qwen3-hf-jobs-retrospective.md](<LOCAL_PATH>

If you want a public-facing or blog-style narrative:

- [docs/qwen3-hf-jobs-blog-draft.md](<LOCAL_PATH>

If you want the model/vendor context:

- [experiments/finetuning-qwen3/README.md](<LOCAL_PATH>
- [experiments/finetuning-qwen3/docs/01_repository_forensics.md](<LOCAL_PATH>
- [experiments/finetuning-qwen3/docs/02_infrastructure_decision_matrix.md](<LOCAL_PATH>

If you want the reusable Hugging Face lessons:

- [learnings-from-the-past/hf-jobs-gotchas.md](<LOCAL_PATH>

## Code map

Primary Adja-specific HF launchers:

- [scripts/hf_jobs/qwen3_asr_adja.py](<LOCAL_PATH>
- [scripts/hf_jobs/qwen3_tts_adja.py](<LOCAL_PATH>

Submission helper:

- [scripts/hf_jobs/submit_qwen3_jobs.py](<LOCAL_PATH>

Vendor/upstream reference area that should stay intact:

- [experiments/finetuning-qwen3](<LOCAL_PATH>
- [experiments/finetuning-qwen3/vendor/qwen3_tts_official](<LOCAL_PATH>

## Long-term invariants

These are the rules that explain the structure. If you forget everything else, remember these.

- Do not clone `adja-nmt` inside the HF container for this workflow.
- Submit local self-contained launcher scripts with `hf jobs uv run`.
- Keep upstream/vendor Qwen files intact unless there is a deliberate vendor sync.
- Put Adja-specific execution glue in `scripts/hf_jobs/` and documentation around it.
- Record every meaningful retry with job IDs and failure signatures.
- Distinguish infra failures from model failures in the docs.

## Current open questions

- Does the current ASR retry get past the `TrainingArguments` compatibility issue and save a checkpoint?
- Is the Qwen3-TTS 0.6B fine-tuning path genuinely incompatible with the official 1.7B-oriented finetuning logic, or is there a missing adaptation step in our wrapper?
- Is `sox` actually required for the TTS path we are using, or only for a branch of the stack that the container touched during initialization?

## Updating rule

When the Qwen situation changes, update these in this order:

1. [results/qwen3_hf_jobs/README.md](<LOCAL_PATH>
2. [results/run-ledger.md](<LOCAL_PATH>
3. [experiments/registry.md](<LOCAL_PATH>
4. [experiments/finetuning-qwen3/docs/adja_runbook.md](<LOCAL_PATH>
5. [docs/qwen3-hf-jobs-retrospective.md](<LOCAL_PATH> if the new result changes the engineering interpretation

## Ownership and provenance

- Project owner: Josue Godeme
- The Adja-specific Qwen launcher and documentation changes in this repo were prepared for Josue Godeme and should keep that provenance note when expanded later into blog posts or reports.
