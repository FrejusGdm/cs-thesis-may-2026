# Retraining NLLB for weights — bidirectional, public release

Retrain `facebook/nllb-200-distilled-600M` in **both directions** (fr→aj, aj→fr)
on Dartmouth HPC, save the best-by-val-chrF checkpoint per run, and publish
to the public model repo [`JosueG/adja-mt-best`](https://huggingface.co/JosueG/adja-mt-best).

## Why this folder exists

Prior NLLB runs (LRAC paper era and the recent HF Jobs `h0`/`h1`/`h2`) didn't
persist weights, so the P1 cascade pipeline runs in MT-`stub` mode. We're also
joining the open-source family alongside `adja-tts-best` and `adja-asr-best`.

## What's here

| File | Role |
|---|---|
| `train_nllb_hpc.py` | Copied from neurosymbolic LRAC project + 3 small additions: `--src-lang` / `--tgt-lang` / `--new-lang` (defaults preserve fr→aj), and `--push-to-hub` / `--hub-repo` / `--hub-subfolder` for the public release. |
| `prep_data.py` | Login-node script: reads `JosueG/adja-fr-mt-acl-paper-private` once, writes direction-swapped TSVs into the layout the trainer expects. |
| `submit_array.sbatch` | SLURM array (1–6) that reads `jobs.tsv` and runs the trainer. |
| `jobs.tsv` | 6-row matrix: NLLB-600M × {fr_aj, aj_fr} × {seed 42, 123, 456}. |
| `select_and_publish_best.py` | Run after the array finishes: picks the best seed per direction (by test chrF) and uploads it to the canonical `{direction}/` path. |
| `deploy_to_discovery.sh` | scp this folder to `adja-nmt-hpc/` on Discovery (one password prompt via SSH ControlMaster). |
| `fetch_results.sh` | Pull metrics + predictions back to the laptop (skips checkpoint dirs — those live on HF Hub). |

## End-to-end flow

```bash
# 0. From laptop: deploy this folder to HPC
bash experiments/mt/retraining-nllb-for-weights/deploy_to_discovery.sh

# 1. SSH in
ssh f006g5b@discovery.dartmouth.edu
cd <HPC_WORKDIR>

# 2. One-time: prep data on login node (no GPU)
export HF_TOKEN=hf_xxx
python3 scripts/prep_data.py --data-root data

# 3. One-time: pre-cache the NLLB-600M model and build a container
#    (Reuse hpc/apptainer/speech-training.def or build a lighter NMT one.)
#    Layout expected by submit_array.sbatch:
#      ./nmt-training.sif
#      ./models/nllb-200-distilled-600M/

# 4. Smoke: one job, short time
sbatch --array=1 --time=00:30:00 submit_array.sbatch
squeue -u f006g5b   # watch
tail -f logs/nllb-bidir-*_1.out

# 5. Full array (6 runs)
sbatch submit_array.sbatch

# 6. Once all 6 finish: pick winners per direction and re-upload to the
#    canonical fr_aj/ and aj_fr/ paths
python3 scripts/select_and_publish_best.py \
    --results-dir results/bidir-paper-repro

# 7. From laptop: pull metrics back, log to registry/run-ledger
bash experiments/mt/retraining-nllb-for-weights/fetch_results.sh
```

## Repo layout after step 6

```
JosueG/adja-mt-best/
├── README.md                 # model card (write after step 6)
├── fr_aj/                    # OFFICIAL best fr→aj
├── aj_fr/                    # OFFICIAL best aj→fr
├── fr_aj/seed{42,123,456}/   # all seeds for transparency
└── aj_fr/seed{42,123,456}/
```

## Cascade hookup (next plan)

After `fr_aj/` exists on the Hub, edit
[experiments/s2tt/P1_cascade_pipeline/configs/p1_config.yaml](../../s2tt/P1_cascade_pipeline/configs/p1_config.yaml):

```yaml
mt:
  checkpoint: "JosueG/adja-mt-best"
  subfolder: "fr_aj"
  mode: "nllb"          # was "stub"
```

…and verify `stages/mt.py` passes `subfolder=` through to
`AutoModelForSeq2SeqLM.from_pretrained`.

## Hyperparameters

Locked to the LRAC paper config so fr→aj reproduces chrF 41.2 and aj→fr is
apples-to-apples: LR=1e-4, BS=16, max_epochs=50, patience=10, warmup=500,
beam=5. All defaults — no need to override.

## Source attribution

`train_nllb_hpc.py` is a copy of the neurosymbolic-ai-paper-experiments
LRAC HPC trainer:
`<LOCAL_PATH>`

Diff vs source = 3 small, defaults-preserving additions documented in the
file's header docstring.
