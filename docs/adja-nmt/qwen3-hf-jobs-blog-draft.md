# Draft Blog: What It Took to Get Qwen3 Running on Hugging Face Jobs for Adja

Prepared for: Josue Godeme  
Requested by: Josue Godeme

## Working title

Getting Qwen3-ASR and Qwen3-TTS onto Hugging Face Jobs for a low-resource language: what broke, what we fixed, and what we learned

## Short abstract

We tried to fine-tune Qwen3-ASR and Qwen3-TTS for Adja on Hugging Face Jobs. The hard part was not dataset size. The hard part was respecting the execution model of HF Jobs, preserving upstream code without forking it into a mess, and separating infrastructure failures from real model failures. Once the launcher shape was corrected, the failures became useful: Qwen3-ASR reached trainer setup, and Qwen3-TTS reached a real model-shape mismatch in the 0.6B fine-tuning path.

## Angle

This is not a “we trained a state-of-the-art Adja model” story yet.

It is a better engineering story:

- how to keep vendor code intact
- how to make HF Jobs reproducible
- how to debug multi-stage failures without confusing infra problems with model problems
- what Qwen3 currently seems to support cleanly versus what still needs patching

## Draft

When we started trying to run Qwen3 for Adja on Hugging Face Jobs, the first failure had nothing to do with speech, data quality, or low-resource language adaptation. It failed because the container tried to clone our repo from GitHub.

That was the wrong model for the repo and the wrong model for `hf jobs uv run`.

The repo already had a better pattern. Other experiments submitted one self-contained script from `scripts/hf_jobs/`, declared dependencies in the script header, and ran directly under HF’s UV runtime. No container-side clone. No hidden assumption that the whole repo would be importable once the job started.

So the first real engineering task was not “make Qwen train.” It was “make Qwen obey the repo’s execution contract.”

That led to three major changes.

First, the Hugging Face launchers were made standalone. HF `uv run` uploads the target script, not a magical full Python environment built from the repo. That meant helper imports that worked locally were unsafe in the container. The fix was to inline the minimum needed logic into the launcher itself: environment parsing, dataset materialization, JSONL writing, validation, and upload behavior.

Second, package installation moved out of runtime shell commands and into inline UV metadata. We tried the obvious alternatives first. `python -m pip` broke. `uv pip --system` installed into the wrong context. `uv pip --python /usr/bin/python3.11` hit the externally-managed interpreter guard. The right answer was simpler: stop trying to install things at runtime and let the HF UV path resolve dependencies the way the repo’s working jobs already do.

Third, we kept the upstream Qwen tree intact. The Qwen files under `experiments/finetuning-qwen3/` stayed as the vendor/reference layer. Adja-specific logic lived outside that tree in the HF launcher scripts and supporting documentation. That separation mattered because it made every local modification auditable. Later, if upstream changes, we know exactly what is ours and what is theirs.

Once those fixes were in place, the job failures got much more interesting.

Qwen3-ASR stopped failing at container bootstrap and started failing inside trainer setup. That is progress. It means private dataset access worked, split materialization worked, the Adja JSONL shape was accepted, and the model loaded far enough to reach the training API. The actual blocker was a compatibility mismatch around `TrainingArguments` in the runtime environment.

Qwen3-TTS got even further. It downloaded the dataset, materialized job-local audio, attached reference audio, loaded the tokenizer and model path, fell back cleanly when flash attention was unavailable, and then failed with a real tensor-size mismatch: 2048 versus 1024.

That changed the interpretation completely. At that point, the problem was no longer “HF Jobs is broken” or “the launcher is broken.” The problem had moved into the Qwen3-TTS fine-tuning path itself. And because the official fine-tuning script defaults to the 1.7B model while we were targeting 0.6B, one plausible hypothesis is that the official path is more validated for 1.7B than for 0.6B.

That is the most important engineering lesson from the whole exercise: the first job of a training pipeline is to fail honestly.

Before the launcher cleanup, the failures were noisy but low-value. After the cleanup, the failures became actionable:

- ASR: runtime API compatibility issue
- TTS: model-shape mismatch in a real fine-tuning path

That is exactly where you want to be when working on a new model family for a low-resource language.

The broader lesson is that low-resource work is usually bottlenecked by operational clarity before it is bottlenecked by GPU-hours. If your launcher is ambiguous, your logs are weak, and your vendor boundary is blurry, then every failure looks like “the model didn’t work.” But if your execution path is clean, then you can tell the difference between:

- a bad container assumption
- a packaging/version mismatch
- a training API mismatch
- a genuine architecture issue

For Adja, that distinction matters. The dataset is small. Every training cycle is precious. Wasting those cycles on avoidable infrastructure mistakes is much worse than discovering quickly that a particular model path still needs patching.

So the current state is not “Qwen is done.” It is better than that. We now have a Qwen HF path that matches the rest of the repo, preserves upstream code, documents every failed attempt, and has finally reached failures that teach us something real about Qwen itself.

That is a solid place to continue from.

## Suggested pull quotes

- “The first job of a training pipeline is to fail honestly.”
- “The dataset was not too small for Hugging Face. The integration was not mature enough.”
- “Once the launcher matched the repo’s execution model, the failures became useful.”

## Suggested companion artifacts

- timeline graphic of attempts 1 through 7
- side-by-side view of the wrong launcher pattern versus the corrected `uv` script pattern
- one diagram showing vendor code versus Adja-specific glue

## Source pointers for expanding this draft

- [docs/qwen3-hf-jobs-retrospective.md](<LOCAL_PATH>
- [experiments/finetuning-qwen3/docs/adja_runbook.md](<LOCAL_PATH>
- [results/qwen3_hf_jobs/README.md](<LOCAL_PATH>
