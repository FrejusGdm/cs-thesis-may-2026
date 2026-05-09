# HPC Gotchas (Dartmouth Discovery)

Learned during the neurosymbolic/ACL paper experiments (2025-2026).

## SLURM / Cluster

- **rsync from cluster**: Use `-av` not `-avz` — Discovery's rsync lacks old-style compress flag
- **`$(dirname "$0")` breaks in SLURM** — `$0` points to `/var/spool/slurmd/`, not your script dir. Use absolute paths.
- **tmux is essential** for surviving SSH disconnects during long upload/download sessions
- **Login**: `f006g5b@discovery.dartmouth.edu`
- **Storage (NMT project)**: `<HPC_WORKDIR>`
- **Storage (audio exploration)**: `<HPC_WORKDIR>`

## Container (Apptainer/Singularity)

- Build with `apptainer build` from a `.def` recipe file for reproducibility
- Pre-download model weights to cluster storage — don't pull from HF Hub during job execution (rate limits + slow)

## Job Design

- Use job arrays with a TSV parameter file (one row per job: experiment, condition, seed, model, etc.)
- A100 80GB GPUs, 32GB RAM, 16h time limit was sufficient for NLLB-1.3B fine-tuning
- Result file pattern: `{model}/{experiment}/{condition}/seed{seed}/test_metrics.json`
