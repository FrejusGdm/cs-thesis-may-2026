# Retraining the Parakeet-TDT Tokenizer for Adja

**Experiment**: D5 (Parakeet-TDT 0.6B v3 fine-tune on Adja)
**Date**: 2026-04-18
**Status**: implementation landed; pilot v3 submission pending
**Author**: Josue Godeme (Dartmouth, Adja low-resource speech track)

## 1. Motivation

NVIDIA's Parakeet-TDT 0.6B v3 (FastConformer encoder + Token-and-Duration
Transducer) ships with an **English-only SentencePiece BPE tokenizer** of
8,192 subwords. Its inventory covers ASCII + common Latin-1 accents but
does not include the Adja-specific characters:

- **ɛ** (U+025B, open-e, very frequent in Adja — ~9.5k occurrences in our 12k-sentence corpus)
- **ɔ** (U+0254, open-o, the most frequent special — ~23.7k occurrences)
- **ɖ** (U+0256, retroflex-d, ~6.5k)
- **ŋ** (U+014B, velar nasal, ~5.2k)
- tone marks on vowels (é, è, à, ô, etc.)

When we ran pilot v2 (HF Job `69e406c9ac288e522d8efe49`, 20 epochs on
`l40sx1`, 30.5 min) without retraining the tokenizer, we observed:

| Metric | Value | Interpretation |
|---|---|---|
| Best dev `val_wer` | **0.6535** (epoch 17) | Encoder adapted — dropped 34 pp from 0.996 at init |
| Per-sentence test Mean WER | 1.00 | All hyps saturated with unk |
| Per-sentence test Median WER | 1.00 | Same |
| Corpus-level test WER (jiwer) | **NaN** | Zero-division because every token is `⁇` |

Sample decode:

```
REF: Ŋu nya kpɔ́kpɔ a ?
HYP: ⁇  nya kp ⁇  kp ⁇  a ?
```

**The FastConformer encoder *did* learn Adja acoustics** — the monotonic
34-point fall in dev WER proves the gradient signal reaches the front-end.
The ceiling is the decoder's output vocabulary: it has no Adja-character
IDs to emit, so everything collapses to `<unk>` = `⁇`.

## 2. Why a fresh tokenizer rather than a bigger fine-tune

Three options were considered:

1. **Train longer / tune LR on the English tokenizer** — cannot fix the
   vocabulary ceiling; best-case output remains `⁇`-dominated.
2. **Switch to a different base model** (e.g. MMS, Whisper, SeamlessM4T) —
   drops the 36h of RNN-T acoustic pretraining Parakeet gives us "for
   free". Already covered by separate tracks in `experiments/asr/`.
3. **Retrain tokenizer + `change_vocabulary`** — keeps FastConformer
   encoder weights, rebuilds only the joint + decoder embedding. NeMo
   supports this natively via
   `ASRModel.change_vocabulary(new_tokenizer_dir, new_tokenizer_type)`.

Option 3 is the standard NeMo recipe for cross-lingual transfer and was
chosen.

## 3. Corpus

Source: `data/extra_adja_text.txt` (private — tracked in `.gitignore`,
mirrored to private Hub dataset `JosueG/adja-text-corpus`).

Provenance — see `experiments/asr/shared/extract_extra_lm_text.py`:

- **Source A**: `10_000_for_data_paper_LREC_cleaned_v2_normalized.csv`,
  Translation column (the LREC data-paper corpus, post-cleaning pass).
- **Source B**: `simple-dataset-enriched.csv`, `adja_translation` column.

Processing:

1. Concatenate both sources (translation columns only).
2. NFC-normalize every line (`unicodedata.normalize("NFC", ...)`).
3. Deduplicate line-exact.
4. Final: **12,050 lines**, ~146 unique Unicode code points.

Character frequency (full corpus):

| Char | Count | Role |
|---|---|---|
| ɔ | 23,719 | back rounded vowel |
| ɛ | 9,531 | front unrounded vowel |
| ɖ | 6,479 | retroflex stop |
| ŋ | 5,159 | velar nasal |
| (plus tone-marked vowels) | — | |

At training time, the pipeline additionally appends the training split's
transcripts (~1.3k lines from the HF dataset `JosueG/adja-tts-orpheus`
after the 80/10/10 split), NFC-normalized, so SP sees both the broader
text corpus and the exact orthography of the ASR training data. Total SP
training sentences ≈ 13.3k.

## 4. Tokenizer hyperparameters

Trained with `sentencepiece.SentencePieceTrainer.train`:

| Parameter | Value | Rationale |
|---|---|---|
| `model_type` | `bpe` | Matches Parakeet's native format; compatible with `change_vocabulary(..., "bpe")` |
| `vocab_size` | **1024** | Default for this experiment. Corpus is small (13k lines); too-large vocabs yield sparse merges and rare subwords that see few gradient steps. 1024 is the NeMo cookbook default for low-resource ASR fine-tune. Adjustable via `VOCAB_SIZE` env. |
| `character_coverage` | **1.0** | Critical. Parakeet-EN uses 0.9995, which silently drops the lowest-frequency characters. At 1.0, every char in the corpus gets a token — non-negotiable for Adja specials. |
| `normalization_rule_name` | `identity` | We've already NFC-normalized upstream; don't let SP's `nmt_nfkc` re-normalize and collapse combining marks. |
| `unk_id` | `0` | Standard. |
| `bos_id` / `eos_id` / `pad_id` | `-1` | NeMo's RNN-T / TDT doesn't need BOS/EOS/PAD tokens in the SP model (it handles those at the decoder level). |

Output artifacts written to `{WORKSPACE}/tokenizer_adja/`:

- `tokenizer.model` — SentencePiece binary (consumed by NeMo)
- `tokenizer.vocab` — human-readable SP vocab
- `vocab.txt` — one token per line (some NeMo paths check for this)
- `spm_corpus.txt` — the merged training text (kept for reproducibility)

## 5. Model surgery

After `ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v3")`, before
training wires up, we call:

```python
asr_model.change_vocabulary(
    new_tokenizer_dir=str(tokenizer_out_dir),
    new_tokenizer_type="bpe",
)
```

Effect (per NeMo 2.0.0 source, `nemo/collections/asr/models/rnnt_bpe_models.py`):

- **Encoder (FastConformer)**: **untouched**. All pretrained weights
  retained.
- **Joint network**: rebuilt. `num_classes_with_blank` changes from
  `8192 + 1 + 5 durations = 8198` to `1024 + 1 + 5 = 1030`. Projection
  matrix is re-initialized.
- **Decoder (prediction network) embedding table**: rebuilt to
  `[1030, d_model]`. Re-initialized.
- **Loss**: rebuilt to match the new `num_classes`. We then explicitly
  swap it once more via `_swap_rnnt_loss(...)` to select between
  `tdt` (fused CUDA) and `tdt_pytorch` (pure-PyTorch fallback), which
  must happen **after** `change_vocabulary` so it sees the new shapes.

Training then proceeds normally. The joint + decoder learn Adja subword
distribution from scratch; the encoder adapts its 36h-of-speech acoustic
representations to Adja phonetics (the part that was already working in
pilot v2).

## 6. Why this is the right call for thesis reporting

- **Ablation-friendly**: D5 pilot v2 is the "off-the-shelf tokenizer"
  baseline (val_wer 0.6535). D5 pilot v3 is the "retrained tokenizer"
  condition. Difference is attributable to vocabulary / decoder re-init,
  with encoder initialization identical.
- **Reproducible**: corpus is versioned (Hub repo), SP hyperparameters
  deterministic, random seed fixed at 42 for splits.
- **Defensible methodologically**: `change_vocabulary` is NeMo's own
  documented API for cross-lingual transfer (used in MLS, VoxPopuli,
  AfriVoice pipelines). No custom surgery.
- **Aligns with existing work**: pairs cleanly with the character-LM
  results (`docs/language-models-for-asr.md`, `docs/why-lm-barely-helped.md`)
  — the 12k corpus we use here was *originally* curated for that LM
  experiment, so the thesis narrative is: "same corpus, two orthogonal
  uses, each closing a different vocabulary gap."

## 7. How to run

### Smoke (UV, fallback loss, 1 epoch, 4 samples)

```bash
hf jobs uv run \
  --flavor a10g-small --timeout 2h --secrets HF_TOKEN \
  -p 3.11 \
  -e DRY_RUN=1 -e EXP_ID=D5_smoke_tok \
  -e RETRAIN_TOKENIZER=1 -e VOCAB_SIZE=1024 \
  -e RNNT_LOSS_NAME=tdt_pytorch \
  scripts/hf_jobs/parakeet_finetune.py
```

Expected log:

```
[tokenizer] Downloading corpus JosueG/adja-text-corpus/extra_adja_text.txt ...
[tokenizer] Corpus: 12050 LM lines + 4 train transcripts = 12054
[tokenizer] Training SentencePiece BPE (vocab_size=1024, character_coverage=1.0) ...
[change_vocabulary] old num_classes_with_blank=8198
[change_vocabulary] new num_classes_with_blank=1030
[loss-swap] original=tdt -> selected=tdt_pytorch
```

### Pilot (Docker CUDA-devel, fused loss, 20 epochs, full split)

```bash
SCRIPT_URL="https://huggingface.co/datasets/JosueG/hf-cli-jobs-uv-run-scripts/resolve/main/parakeet_finetune.py"

hf jobs run \
  --flavor l40sx1 --timeout 24h --secrets HF_TOKEN \
  -e EXP_ID=D5_l40_tokv1 \
  -e MAX_EPOCHS=20 -e BATCH_SIZE=8 -e EVAL_BATCH_SIZE=16 \
  -e RNNT_LOSS_NAME=tdt \
  -e RETRAIN_TOKENIZER=1 -e VOCAB_SIZE=1024 \
  -e SCRIPT_URL="$SCRIPT_URL" \
  pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel \
  bash -lc 'apt-get update -qq && apt-get install -y -qq git && \
    python -m pip install --no-cache-dir "setuptools<80" "nemo_toolkit[asr]==2.0.0" \
      datasets soundfile librosa "numpy<2" pandas huggingface-hub jiwer \
      "lightning>=2.2,<2.4" sentencepiece && \
    python -c "import os, urllib.request; from pathlib import Path; req = urllib.request.Request(os.environ[\"SCRIPT_URL\"], headers={\"Authorization\": \"Bearer \" + os.environ[\"HF_TOKEN\"]}); Path(\"/tmp/parakeet_finetune.py\").write_bytes(urllib.request.urlopen(req).read())" && \
    python /tmp/parakeet_finetune.py'
```

## 8. Expected outcome

Acceptable baselines, ordered from weakest to strongest:

- **Absolute floor**: corpus WER computes to a real number (i.e. no `⁇`
  saturation). Anything < NaN is progress.
- **Reasonable target**: corpus WER ≤ 0.50. This would mean the decoder
  is emitting Adja characters and the subword merges are helping
  compose Adja morphemes.
- **Strong target**: corpus WER ≤ 0.30. Competitive with our best
  Whisper-Ewe fine-tune on a related Gbe language.

If pilot v3 lands in the "strong" band, Parakeet becomes the primary ASR
candidate for the thesis. If it only reaches the "reasonable" band, the
comparison table still has a clean multi-condition story.

## 9. Results — pilot v3 (2026-04-18, job `69e437a7cd8c002f31dff009`)

Setup: `l40sx1`, `pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel`, 20 epochs,
BATCH_SIZE=8, LR=1e-4, fused `tdt` loss, `RETRAIN_TOKENIZER=1`,
`VOCAB_SIZE=1024`. Training wall-clock: **20.0 min**. Uploaded to
`JosueG/adja-asr-results/D5_l40_tokv1_parakeet/` (2.47 GB `.nemo`).

Tokenizer-swap pipeline verified end-to-end:

```
[tokenizer] Corpus: 12050 LM lines + 1277 train transcripts = 13327
[tokenizer] Training SentencePiece BPE (vocab_size=1024, character_coverage=1.0) ...
[change_vocabulary] old num_classes_with_blank=8198
Tokenizer SentencePieceTokenizer initialized with 1024 tokens
[change_vocabulary] new num_classes_with_blank=1030
[loss-swap] original=tdt -> selected=tdt
[loss-swap] num_classes_with_blank=1030 num_durations=5 -> num_classes=1024
```

### Metrics

| Metric | Pilot v2 (no retrain) | Pilot v3 (retrain, vocab=1024) | Δ |
|---|---|---|---|
| Best dev `val_wer` | 0.6535 (epoch 17) | **0.9544** (epoch 12) | **+0.301 worse** |
| Test per-sentence Mean WER | 1.00 | 1.00 | — |
| Test per-sentence Median WER | 1.00 | 1.00 | — |
| Test corpus WER (jiwer) | NaN | NaN | — |
| Training wall-clock | 30.5 min | 20.0 min | −10.5 min |

`val_wer` trajectory (retrain): 1.000 (ep 0) → 0.996 (ep 2) → 0.968 (ep
10) → 0.961 (ep 11) → **0.954 (ep 12)** → no further improvement through
ep 19.

### Sample decodes (test)

```
REF: Ŋu nya kpɔ́kpɔ a ?
HYP: E

REF: ŋnyan go. Kpɔ ŋuɖejikɔ a nyan wo
HYP: E yi le le wo

REF: Ŋ nyanyɔ mɔ wo anu ahán !
HYP: E
```

Compare pilot v2 (English-only tokenizer):
```
REF: Ŋu nya kpɔ́kpɔ a ?
HYP: ⁇  nya kp ⁇  kp ⁇  a?
```

### Interpretation

**The retrain mechanically succeeded — the new vocabulary is Adja-aware,
the model emits Adja characters (`E`, `yi`, `le`, `wo`) instead of `⁇`.**
However, decoded hypotheses are drastically under-length (mostly 1 token)
because the freshly re-initialised joint + decoder embedding haven't
converged: 1277 samples × 20 epochs = 25.5k gradient steps is enough for
the encoder to adapt (pilot v2 proved that) but insufficient for a
from-scratch RNN-T prediction network with a new 1024-vocab softmax.

Put differently: **v2 has a decoder that knows how to compose subwords
but no Adja characters; v3 has Adja characters but a decoder that
doesn't know how to emit them in sequence.** Neither is a shipping
model. The dev-WER regression (+30 pp) reflects the cost of resetting
the decoder.

### Recommendation (for thesis / follow-up)

The tokenizer swap is a correct-in-principle design; the fine-tune
recipe needs to be adapted to accommodate the re-initialised decoder.
Candidate next steps, in order of expected payoff:

1. **Longer training + higher LR** — 80–100 epochs at LR=5e-4 with a
   longer warmup. The encoder can tolerate this because it's starting
   from well-conditioned weights; the decoder needs it.
2. **Staged training** — freeze encoder for epochs 1–10 (let decoder
   catch up on frozen features), then unfreeze at a reduced LR.
3. **Smaller vocabulary** — 256 or 512. With 13k sentences many of the
   1024 subwords are rare; a smaller vocab yields denser gradient
   signal per token.
4. **Differential LR** — same idea as staged, but smoother: 5×LR on
   `joint` and prediction-network parameters, 1×LR on encoder.

Any of (1)–(4) requires a new pilot, not a recipe tweak. They are
candidates for D5b/D5c.

### Artifacts

- Job: [69e437a7cd8c002f31dff009](https://huggingface.co/jobs/JosueG/69e437a7cd8c002f31dff009)
- Results: `JosueG/adja-asr-results/D5_l40_tokv1_parakeet/`
  (`metrics.json`, `evaluation_summary.txt`, `test_results.csv`,
  `parakeet_tdt_D5_l40_tokv1.nemo`)
- Archived logs: `results/parakeet_hf_jobs/logs/69e437a7_pilot_v3.txt`
  (7148 lines)

## 10. Known limitations / future work

- **Vocab size was picked by rule of thumb**, not sweep. Follow-up:
  `VOCAB_SIZE ∈ {512, 1024, 2048}` at fixed training budget.
- **Duration tokens unchanged**: Parakeet-TDT has 5 duration tokens
  (configured in the base model). We inherit them as-is; tuning these
  for Adja prosody is a separate experiment.
- **No held-out text set**: SP is trained on corpus + training split
  transcripts. Dev/test are *not* in SP training (splits are
  `seed=42`). If SP ends up touching test transcripts indirectly (they
  shouldn't, but audit: the LM corpus is from the LREC data paper,
  which is a distinct source from the ASR dataset), document it.
- **Sub-1k rare subwords**: with vocab 1024 and only 13k sentences, many
  subwords will be low-frequency. Early stopping on `val_wer` should
  protect against overfit, but monitor the learning curve shape.

## 11. Files touched

- `scripts/hf_jobs/parakeet_finetune.py` — added `RETRAIN_TOKENIZER`,
  `VOCAB_SIZE`, `TOKENIZER_CORPUS_REPO`, `TOKENIZER_CORPUS_FILE` env
  vars; added `_retrain_tokenizer_and_swap` helper; runs before
  `_swap_rnnt_loss`.
- `data/extra_adja_text.txt` — moved to `.gitignore` (private), mirrored
  to Hub dataset `JosueG/adja-text-corpus`.
- `docs/parakeet-tokenizer-retrain-adja.md` — this document.

Downstream after pilot v3 completes:

- `results/parakeet_hf_jobs/README.md` — append pilot v3 row.
- `experiments/registry.md` — bump D5 row.
- `results/run-ledger.md` — detailed entry with sample decodes.
