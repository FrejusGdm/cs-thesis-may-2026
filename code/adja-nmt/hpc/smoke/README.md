# HPC Smoke Tests

Last updated: **2026-04-22**

Run these **before every sbatch** of real training jobs. Queue time on Discovery is hours to days; we can't afford to burn a 48h allocation on a missing `token=` or a read-only bind mount.

## Two tiers

### Tier 1 — login-node smoke (no GPU, ~3 min)

```bash
cd <HPC_WORKDIR>
export HF_TOKEN=hf_xxx
bash smoke/login_check.sh
```

Validates: Python imports, HF auth, `/data` is writable, pre-cached models have config files, `load_dataset_builder()` for WaxalNLP and the Adja dataset, SNAC loads. Catches the most common pre-flight failures in one pass.

**Run this every time you pull changes from the laptop** — takes 3 minutes, no queue cost, and it's saved us multiple hours of failed queue time so far.

### Tier 2 — interactive GPU smoke (one short allocation, ~15 min)

```bash
srun --pty --partition=gpuq \
     --gres=gpu:nvidia_a100_80gb_pcie_3g.40gb:1 \
     --mem=64G --time=30:00 bash
# inside the allocation:
cd <HPC_WORKDIR>
bash smoke/gpu_smoke.sh
```

Runs each training script with `--smoke`: 4 samples, 2 steps, Stage A only, no checkpoint save. Validates CUDA visibility, model loads on GPU, data from `/data` cache works, forward + backward + optimizer step all execute, and — critically — peak VRAM per model. If `whisper-largev3` or `audiolm-orpheus` peak >35 GB on the 40GB MIG, flip those sbatch files to request a full A100 before the real run.

Total runtime ~10–15 minutes. One 30-min srun allocation covers the whole pass.

## When smokes fail

- **Tier 1 fail**: the bug is environmental (bind mount, auth, missing model). Fix on laptop, push, re-scp to cluster, re-run login_check.
- **Tier 2 fail on the first model only**: pipeline bug specific to that model (check the traceback).
- **Tier 2 fail on all four**: likely a shared issue — container missing a dep, CUDA not visible, GPU driver mismatch.

## When smokes pass

Submit real jobs with confidence:

```bash
sbatch slurm/submit_ssl_ewe.sbatch
sbatch slurm/submit_audiolm_csm.sbatch
sbatch slurm/submit_audiolm_orpheus.sbatch
sbatch slurm/submit_whisper_largev3_ewe.sbatch
```
