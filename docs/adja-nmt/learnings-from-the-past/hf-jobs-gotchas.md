# HF Jobs Gotchas

Learned during the neurosymbolic/ACL paper experiments (2025-2026).

## Script Execution

- **`hf jobs uv run` — script path must be LAST argument**, after all `--flag` and `-e` options. Putting it elsewhere silently breaks.
- **Stdout is fully buffered in containers.** Add `sys.stdout.reconfigure(line_buffering=True)` at the top of any training script, otherwise `print()` output won't appear in the HF Jobs log viewer until the script exits.
- **Treat `hf jobs uv run` scripts as self-contained units.** Do not assume helper modules from the local repo will be importable unless they are packaged or inlined into the submitted execution unit.
- **Do not clone the repo inside the HF container when the repo already uses repo-native `uv run` scripts.** For this workspace, the correct pattern is to submit the local launcher script directly.
- **Do not build package bootstrap logic into the job unless absolutely necessary.** Inline UV dependency metadata is much more reliable than trying `pip`, `uv pip --system`, or `uv pip --python ...` inside the running job.
- **Print full tracebacks for long-running jobs.** A short exception string at the end of a 2 GB model download log is not enough for diagnosis.

## Dependency Reality

- **HF-resolvable versions are part of the operational contract.** Local docs may say `qwen-asr>=0.1.0`, but if the runtime only resolves `qwen-asr==0.0.6`, the launcher has to respect that.
- **Missing `flash-attn` is not necessarily the real blocker.** A good launcher should fall back and keep going so the underlying model-path error is exposed.
- **System tools are not guaranteed.** Qwen TTS logs showed `sox: not found`; if a model path assumes it, that assumption must be documented.

## Rate Limits

- HF allows 1,000 API calls per 5-min window (free) or 2,500 (PRO)
- Each `hf jobs uv run` makes ~3-4 API calls (whoami, upload, create)
- With DELAY=5s between submissions, you'll hit 429 every ~15 jobs
- Solution: retry logic with 65s backoff, up to 3 retries

## Cost

- 233 jobs across 5 tiers (smoke → core → scale → ablate → arch) was the full experiment set
- Smoke tests first (6 jobs) to validate before scaling up
