# SSL pretraining on 183k unlabeled Ewe — why it matters

Last updated: **2026-04-21**

## The opportunity

Google's WaxalNLP dataset has an `ewe_asr` config with an **`unlabeled`
split of ~183k utterances**. Audio only, no transcripts. This is 100× the
size of all our labeled Gbe data combined.

Untranscribed audio has historically been worthless for supervised ASR
training. But with SSL objectives (wav2vec 2.0, HuBERT, data2vec), it can
teach a model's encoder rich acoustic representations — specifically, which
sounds co-occur, how phonemes transition, and how speakers vary. The
downstream supervised fine-tune then starts from an encoder that already
"understands" Gbe speech.

See `../concepts/05-ssl-pretraining-wav2vec.md` for mechanics.

## What we're running (on HPC)

`hpc/scripts/pretraining/wav2vec2_ssl_ewe.py` — parameterized for 3 bases:

1. **XLS-R 300M** (`facebook/wav2vec2-xls-r-300m`). Pretrained on 128
   languages. Continue-pretrain on unlabeled Ewe, then CTC fine-tune on Adja.
2. **XLS-R 1B** (`facebook/wav2vec2-xls-r-1b`). Same pipeline, bigger model.
3. **MMS 1B** (`facebook/mms-1b-all`). Different pretrain data (1107
   languages incl. Ewe), so "continue pretraining" is a domain-adapt push.

Each is a job-array slot in `submit_ssl_ewe.sbatch`.

## Expected budget

- Pretrain stage A: ~24-36 hours per model on A100 80GB (183k utterances
  × 3 epochs × ~1 step/sec).
- Fine-tune stage B: 4-8 hours per model on A100 (Adja 1.6k utterances
  × 30 epochs with early stopping).

Total ~3-5 cluster days per model × 3 models. Fits easily in our HPC budget,
infeasible on HF Jobs.

## Expected wins

Literature gives us ranges:
- SSL on related-language unlabeled audio: **20-40% relative WER reduction**
  in low-resource settings (XLS-R paper, MMS paper, various surveys).
- If we hit the lower end: our ASR goes from CER ~23% → ~18%.
- If we hit the upper end: CER ~23% → ~14%.

Either would be the best Adja ASR anywhere.

## What could go wrong

- **Dataset quality**. WaxalNLP unlabeled split might contain a lot of
  music, noise, or non-Ewe audio. First step is to stream through a few
  clips and confirm they're usable.
- **Distribution mismatch** between Ewe (dataset) and Adja (target task).
  Gbe-family share most phonemes but not all. The SSL encoder might
  over-specialize to Ewe-only features. Mitigation: Adja fine-tune is
  supervised, so the decoder can re-tune.
- **Catastrophic forgetting**. Continue-pretraining on Ewe might erase the
  128-language priors in XLS-R. Mitigation: short Stage A (3 epochs max),
  low LR.
- **Time**. 3 days per model × 3 models = ~9 cluster days. Queue time can
  push this out. We start SSL first in the sprint because it dominates wall
  time.

## Evaluation plan

After Stage B, compare dev CER/WER against:
- E4 (Whisper Ewe→Adja) — CER 24.90% logged baseline; 37.18% deployable
  (see `results/comparison.md` lost-checkpoint correction 2026-04-29).
- C4v2 + LM (XLS-R CTC + Optuna-tuned LM) — CER 22.67% best result to date.
- Each SSL variant vs its no-SSL counterpart (would need an explicit
  ablation).

## Papers to read

- Baevski et al., "wav2vec 2.0" — https://arxiv.org/abs/2006.11477 §3 (SSL pretraining).
- Babu et al., "XLS-R" — https://arxiv.org/abs/2111.09296 §2-3.
- Pratap et al., "MMS" — https://arxiv.org/abs/2305.13516 §3-4.
- Berrebbi et al., "CATR: Continue AUDIO-only TRaining" — the theory behind
  exactly what we're doing.

## Files

- Pretraining script: `../../hpc/scripts/pretraining/wav2vec2_ssl_ewe.py`
- Wrapper for MMS: `../../hpc/scripts/pretraining/mms_ssl_ewe.py`
- Wrapper for XLS-R: `../../hpc/scripts/pretraining/xlsr_ssl_ewe.py`
- SLURM launcher: `../../hpc/slurm/submit_ssl_ewe.sbatch`
- Related concept: `../concepts/05-ssl-pretraining-wav2vec.md`
