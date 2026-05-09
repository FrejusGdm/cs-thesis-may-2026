# CS Thesis May 2026

This repository is the public artifact trail for two years of work on Adja
language technology: grammar-guided French-Adja machine translation, speech
recognition, text-to-speech, and speech-to-text-to-speech experiments. It is
part research archive, part reproducibility bundle, and part map of the many
failed, partial, and useful paths that led to the May 2026 CS thesis.

The goal is not to pretend the work was linear. The goal is to make the real
artifact inspectable: the datasets, training scripts, evaluation reports,
model links, pipeline attempts, hard lessons, and cleanup decisions that turn a
private research workspace into something other people can audit and build on.
Large datasets and model weights live on Hugging Face.

## Layout

- `code/`: preprocessing, data preparation, training, evaluation, and analysis code.
- `data/`: dataset documentation, manifests, French prompt-generation outputs, and Hugging Face helpers.
- `models/`: model registry grouped by MT, ASR/STT, and TTS.
- `results/`: thesis-facing MT, ASR, TTS, and S2TT result summaries plus result-repo consolidation notes.
- `docs/`: reproducibility, ethics, data access, and experiment-framework documentation.

## Why This Exists

Adja is a low-resource Gbe language with very little public NLP or speech
infrastructure. This release collects the thesis-facing artifacts behind a
larger question: how much useful language technology can be built when the
available data is small, locally assembled, and uneven, but carefully designed?

The repository therefore includes polished pieces and rough edges. Some folders
show the final thesis path. Others show negative results, infrastructure notes,
and model attempts that explain why the final choices were made.

## Canonical Artifact Policy

Existing Hugging Face repos are treated as immutable inputs. This release does
not rename, move, delete, make public, or edit any existing repo used by active
training runs. New canonical artifacts are created by duplication or export.

Generated from private workspace revision `eb84dbbe5483d35481ece2edf553fbc59a105474` at `2026-05-09T15:44:10.898918+00:00`.
