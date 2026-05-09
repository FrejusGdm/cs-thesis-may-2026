# HPC — Dartmouth Discovery speech training

Last updated: **2026-04-21**

This folder is the HPC piece of the `adja-nmt` repo. It gets `scp`'d (not rsync — macOS 15 breaks old rsync) as a unit to Dartmouth Discovery at:

```
<HPC_WORKDIR>
```

Everything outside this folder stays on laptop + GitHub. The `hpc/` folder IS the HPC deployment — do not symlink files in from elsewhere.

## What runs here (not on HF Jobs)

- **SSL pretraining** on 183k unlabeled Ewe utterances (`scripts/pretraining/{wav2vec2,mms,xlsr}_ssl_ewe.py`) — days-long runs.
- **Audio-LM pretraining** for CSM/Orpheus using Mimi/SNAC tokens on unlabeled Ewe (`scripts/pretraining/audio_lm_{csm,orpheus}_ewe.py`) — gated on local Mimi reconstruction test passing.
- **Whisper large-v3 Ewe ASR at scale** (`scripts/large/whisper_largev3_ewe_hpc.py`) — longer/bigger than the HF Jobs version.

HF Jobs (at `../scripts/hf_jobs/`) handles short fine-tunes (< 8 h). HPC handles everything that doesn't fit there.

## First-time setup (do once, on the Discovery login node)

```bash
# 1. from your laptop, deploy hpc/ to the cluster
bash hpc/deploy/deploy_to_discovery.sh

# 2. ssh in and build the container
ssh f006g5b@discovery.dartmouth.edu
cd <HPC_WORKDIR>
bash apptainer/build.sh

# 3. pre-cache all models + datasets (once)
export HF_TOKEN=hf_xxxxxxxxxxxx
bash deploy/download_models.sh
```

## Running experiments

**Mandatory pre-flight before every sbatch** (see `smoke/README.md`):

```bash
# 1. login-node smoke (3 min, no GPU)
bash smoke/login_check.sh

# 2. interactive GPU smoke (~15 min, one short srun allocation)
srun --pty --partition=gpuq --gres=gpu:nvidia_a100_80gb_pcie_3g.40gb:1 \
     --mem=64G --time=30:00 bash
bash smoke/gpu_smoke.sh
```

Only sbatch real jobs after both smokes pass. Queue time is too expensive to burn on trivial bugs.

```bash
# SSL pretraining (array: 1=xlsr-300m, 2=xlsr-1b, 3=mms-1b)
sbatch slurm/submit_ssl_ewe.sbatch

# Audio-LM (once Mimi recon test passes locally)
sbatch slurm/submit_audiolm_csm.sbatch
sbatch slurm/submit_audiolm_orpheus.sbatch

# Whisper large-v3
sbatch slurm/submit_whisper_largev3_ewe.sbatch

# Monitor
squeue -u f006g5b
tail -f logs/ssl-<JOBID>_1.out
```

## Fetching results back

```bash
# from laptop
bash hpc/deploy/fetch_results.sh    # pulls metrics.json + generated/ into results/hpc/
```

## Layout

```
hpc/
├── slurm/                 sbatch files (one per experiment family)
├── apptainer/             container recipe + build script
├── deploy/                scp-based deploy + fetch + model pre-cache
├── jobs/                  TSVs read by sbatch (row == array task)
└── scripts/
    ├── pretraining/       SSL + audio-LM training scripts
    └── large/             big-model runs (Whisper large-v3, etc.)
```

## Gotchas (read these first)

Full list: `../learnings-from-the-past/hpc-gotchas.md`. Key ones:

- **Absolute paths only** — `$(dirname "$0")` points to `/var/spool/slurmd/` inside SLURM, not your script dir.
- **No rsync from macOS 15+** — use the provided `scp` deploy script.
- **Pre-cache models before submitting** — HF rate-limits and jobs will hang on multi-GB downloads.
- **tmux** for long upload/download sessions over SSH.
- **Memory = 64–96 G** for speech (vs 32 G for NMT) — audio batches are large.

## Cross-references

- Master plan: `../experiments/asr-tts-getting-right-2026-04-21.md`
- HF Jobs scripts: `../scripts/hf_jobs/`
- Learning notes (concepts, papers, explainers): `../learning/README.md`
- Session log for the sprint that spawned this folder: `../session-logs/2026-04-21-expand-and-hpc.md`
- Main project CLAUDE.md: `../CLAUDE.md`
