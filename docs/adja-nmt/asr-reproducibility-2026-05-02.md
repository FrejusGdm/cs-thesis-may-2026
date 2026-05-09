# Reproducible Adja ASR qualitative decodes

Last updated: 2026-05-03.

## What this doc is

This is the paper-prep doc for **Chapter 4's qualitative-decode table**. The
review committee should be able to take the published model repos +
this script and reproduce every ref/hyp row in the paper. This doc
freezes the data slice, the inference protocol, the normalization
pipeline, and the exact command line used to generate the JSON+MD
artifacts checked into `results/qualitative/`.

If any of those settings change, update this doc — it is the single
source of truth, not the script docstring.

## Why we needed a regen utility

The original ASR training runs (E4, C2, C4v2) wrote per-utterance decode
CSVs to ephemeral SLURM scratch directories. Those CSVs were copied into
the thesis draft once but were never archived to the Hub or to git. Two
consequences:

1. The exact (ref, hyp) pairs the thesis cites can no longer be
   regenerated from local files.
2. We have **two architecturally different deployable ASRs** now —
   C4v2 (XLS-R + character CTC, the lower-CER one) and E4v4
   (Whisper-small fine-tuned via the Ewe stage, overfit-prone) — and
   the paper benefits from showing both architectures side-by-side on
   the *same* test sentences.

`scripts/qualitative/regen_asr_decodes.py` rebuilds the table from
scratch against any subset of the public endpoints, with a deterministic
sample selection that anyone reading the paper can replay.

## Scope of reproducibility

| Component | Pinned source | Why pinned |
|---|---|---|
| Dataset | `JosueG/adja-tts-orpheus` (HF, private) | The only Adja ASR dataset; private but accessible to thesis reviewers. |
| Split | `train_test_split(test_size=0.10, seed=42)` on the `train` split | Matches `experiments/asr/shared/data_prep.py` and every Whisper FT script — same test rows the trained models never saw. |
| Sample order | `test[0..n-1]` | Deterministic. No second shuffle. |
| Reference text | `unicodedata.normalize("NFC", row["text"].strip())` | Project-wide rule (CLAUDE.md). |
| Hypothesis cleaning | Strip literal `<pad>` / `<unk>` / `<blank>` / `<s>` / `</s>` markers + collapse whitespace | Mirrors `experiments/tts/eval/reverse_wer.py:_clean_hyp` so qualitative rows agree with reverse-WER aggregates on the same audio. |
| Metrics | `experiments/asr/shared/metrics.py` `compute_wer` / `compute_cer` with `normalize=True` | Same NFC-safe normalizer used everywhere else (preserves ɛ, ɔ, ŋ, ɖ, tone marks; strips ASCII punct). |
| C4v2 model | `JosueG/wav2vec2-xlsr-adja-c4v2` | Reproducible flagship ASR. Greedy CTC. |
| E4v4 model | `JosueG/whisper-ewe-adja-e4v4` | Whisper-small Ewe→Adja two-stage FT. Overfits past epoch 20. |

## Run protocol

### 1. Bring the endpoints up

```bash
python scripts/hub/manage_endpoints.py up \
    JosueG/wav2vec2-xlsr-adja-c4v2 \
    JosueG/whisper-ewe-adja-e4v4
```

Each endpoint takes 1–2 minutes to come up cold; subsequent warm
requests are fast. Default tier is CPU `intel-spr` x2 (~$0.067/hr) with
scale-to-zero, which is sufficient for a 20-sample regen.

Once the endpoints are listed in the HF UI, copy the two URLs.

### 2. Run the driver

```bash
python scripts/qualitative/regen_asr_decodes.py \
    --n 20 \
    --asrs '<C4V2_ENDPOINT_URL>:c4v2,<E4V4_ENDPOINT_URL>:e4v4' \
    --tag 2026-05-03
```

Outputs:

- `results/qualitative/asr_decodes_2026-05-03.json` — full structured
  payload (per-sample ref, hyp, raw + normalized WER/CER, latency,
  cache flag, error if any) plus a corpus-level aggregate per ASR.
- `results/qualitative/asr_decodes_2026-05-03.md` — paper-ready table
  with the aggregate row and the per-sample side-by-side comparison.
- `results/qualitative/.cache/<asr_label>/<sha256>.json` — per-audio
  endpoint response cache. Re-runs against the same audio are free.

### 3. Bring the endpoints down

```bash
python scripts/hub/manage_endpoints.py down \
    aja-xlsr-adja-c4v2 \
    aja-ewe-adja-e4v4
```

Endpoints will scale to zero on idle anyway, but explicit `down` removes
the deployment so it does not show up as billable in the HF console.

## Local fallback

If the endpoints are not available (e.g. the reviewer is offline or
does not have an HF Pro subscription to spin up dedicated endpoints),
the same driver runs through `transformers.pipeline`:

```bash
python scripts/qualitative/regen_asr_decodes.py \
    --n 5 --backend local \
    --asrs JosueG/wav2vec2-xlsr-adja-c4v2:c4v2,JosueG/whisper-ewe-adja-e4v4:e4v4
```

Slow on CPU (Whisper-small is the bottleneck — ~20 s per 4-second clip
on an M-series MacBook). Use it as a hermetic correctness check or for
ad-hoc inspection of a few samples; the canonical paper artifact comes
from the endpoint path because that is what the deployed pipeline
actually serves.

## Output schema

`asr_decodes_<tag>.json`:

```jsonc
{
  "generated_utc": "2026-05-03T15:42:01Z",
  "dataset": "JosueG/adja-tts-orpheus",
  "split": "test (10% of seed=42 train_test_split)",
  "asr_labels": ["c4v2", "e4v4"],
  "samples": [
    {
      "index": 0,
      "ref": "Ŋu nya kpɔ́kpɔ a ?",
      "decodes": {
        "c4v2": {
          "hyp": "ŋu nya kpɔkpɔ a",
          "metrics": { "wer_raw": ..., "wer_norm": ..., "cer_raw": ..., "cer_norm": ... },
          "asr_target": "https://...endpoints.huggingface.cloud",
          "latency_sec": 0.83,
          "cached": false,
          "error": null
        },
        "e4v4": { ... }
      }
    },
    ...
  ],
  "aggregate": {
    "c4v2": { "n": 20, "wer_raw": ..., "cer_raw": ..., "wer_norm": ..., "cer_norm": ..., "errors": 0 },
    "e4v4": { ... }
  }
}
```

Per-sample metrics are computed with `compute_wer([ref], [hyp])` —
single-utterance WER/CER. The aggregate is the corpus-level
sum-of-edits / sum-of-refs across every (ref, hyp) pair, which is the
right shape for paper-grade reporting.

## Floors and caveats

- **C4v2 reverse-CER floor on real audio**: 25.05 % on the test set
  (published in `results/comparison.md`). The aggregate row of any
  regen run should land within a few points of that — a large
  divergence indicates an endpoint regression or a normalization
  drift, not a real signal.
- **E4v4 reverse-CER floor on real audio**: 37.18 %. Same caveat.
- **The thesis cites a 24.90 % CER for the original Whisper-Ewe
  checkpoint** that was never uploaded to the Hub. This regen does
  **not** reproduce that number — it reproduces the deployable E4v4
  checkpoint, which is at 37.18 % CER. Any paper claim using the
  24.90 % figure must be paired with the lost-checkpoint correction
  block in `results/comparison.md` and the E4_v5 row in
  `experiments/registry.md`. See
  [docs/pipeline-handoff-2026-05-02.md §1](pipeline-handoff-2026-05-02.md)
  for the full caveat.
- **C4v2 endpoint emits literal `<pad>`/`<unk>` markers** because the
  uploaded tokenizer's `additional_special_tokens` list was empty at
  publication time. The driver strips those at the
  `_clean_hyp` step so the rendered table looks correct. Don't
  "fix" the cached responses — fix the upstream tokenizer if it ever
  matters.

## Cross-references

- ASR endpoint management: [`scripts/hub/manage_endpoints.py`](../scripts/hub/manage_endpoints.py)
- Regen driver: [`scripts/qualitative/regen_asr_decodes.py`](../scripts/qualitative/regen_asr_decodes.py)
- Reference HTTP-POST pattern: [`experiments/tts/eval/reverse_wer.py:CachingASRClient.transcribe`](../experiments/tts/eval/reverse_wer.py)
- Project-wide WER/CER: [`experiments/asr/shared/metrics.py`](../experiments/asr/shared/metrics.py)
- Pipeline handoff (which model to use, when): [`docs/pipeline-handoff-2026-05-02.md`](pipeline-handoff-2026-05-02.md)
- ASR leaderboard: [`results/comparison.md`](../results/comparison.md)
- Lost-checkpoint context: `experiments/registry.md` E4_v5 row
