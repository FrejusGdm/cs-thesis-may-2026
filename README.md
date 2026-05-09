# CS Thesis May 2026

Curated public release for the French-Adja machine translation and Adja speech
experiments supporting the May 2026 CS thesis.

This repository is a clean release surface generated from the private research
workspace. It contains publishable code, reproducibility documentation, result
summaries, data manifests, and model indexes. Large datasets and model weights
live on Hugging Face.

## Layout

- `code/`: preprocessing, data preparation, training, evaluation, and analysis code.
- `data/`: dataset documentation, manifests, French prompt-generation outputs, and Hugging Face helpers.
- `models/`: model registry grouped by MT, ASR/STT, and TTS.
- `results/`: thesis-facing MT, ASR, TTS, and S2TT result summaries plus result-repo consolidation notes.
- `docs/`: reproducibility, ethics, data access, and experiment-framework documentation.

## Canonical Artifact Policy

Existing Hugging Face repos are treated as immutable inputs. This release does
not rename, move, delete, make public, or edit any existing repo used by active
training runs. New canonical artifacts are created by duplication or export.

Generated from private workspace revision `63c4ce98572512d3e84ff1704950a740337ee169` at `2026-05-09T15:02:38.979124+00:00`.
