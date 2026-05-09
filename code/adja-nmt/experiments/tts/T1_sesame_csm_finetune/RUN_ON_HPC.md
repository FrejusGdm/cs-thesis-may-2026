# T1: Running Sesame CSM Fine-tuning on HPC

This is no longer the first-success path.

Use Google Colab T4 with the vendor notebook
`references/unsloth-tts-notebooks/Sesame_CSM_1B_TTS.ipynb` plus the Adja runbook
`experiments/tts/T1_sesame_csm_finetune/UPSTREAM_NOTEBOOK_RUNBOOK.md` first.
Only come back to this HPC path after the canonical Colab dry-run succeeds and
generates real waveforms.

## Prerequisites

Your Adja audio data should already be on the cluster at:
```
<HPC_WORKDIR>
```

## Step-by-step

### 1. SSH into Discovery
```bash
ssh f006g5b@discovery.dartmouth.edu
```

### 2. Sync this experiment to the cluster
```bash
# From your local machine:
rsync -av experiments/tts/ f006g5b@discovery.dartmouth.edu:<HPC_WORKDIR>
rsync -av scripts/containers/tts_unsloth.def f006g5b@discovery.dartmouth.edu:<HPC_WORKDIR>
rsync -av scripts/slurm/T1_sesame_csm.sbatch f006g5b@discovery.dartmouth.edu:<HPC_WORKDIR>
rsync -av scripts/download_csm_model.sh f006g5b@discovery.dartmouth.edu:<HPC_WORKDIR>
```

### 3. Build the container (on login node, ~10-15 min)
```bash
cd <HPC_WORKDIR>
apptainer build --fakeroot tts_unsloth.sif tts_unsloth.def
```

### 4. Pre-download model weights (on login node)
```bash
# Load python if needed
module load python/3.10

cd <HPC_WORKDIR>
bash scripts/download_csm_model.sh
```

### 5. Submit the job
```bash
cd <HPC_WORKDIR>
sbatch scripts/slurm/T1_sesame_csm.sbatch
```

### 6. Monitor
```bash
# Check queue status
squeue -u f006g5b

# Watch logs (once job starts)
tail -f logs/tts-T1_<JOB_ID>.out
```

### 7. Results
Once complete, results will be at:
```
results/T1/seed42/
├── prepared_data/          # 24kHz resampled audio + HF dataset
├── training/
│   ├── adapter/            # LoRA adapter weights
│   └── training_info.json  # Training stats
└── generated/
    ├── test_sentences.txt  # Input sentences
    └── (generated outputs) # Generated audio tokens
```

### 8. Pull results back to local
```bash
# From your local machine:
rsync -av f006g5b@discovery.dartmouth.edu:<HPC_WORKDIR> results/T1/
```

## Troubleshooting

- **Container build fails**: Check if `apptainer` module is loaded (`module load apptainer`)
- **Model download fails**: Make sure `HF_TOKEN` is set (`export HF_TOKEN=...`)
- **OOM during training**: Reduce `per_device_batch_size` to 1 in `conf/config.yaml`
- **Job timeout**: Increase `--time` in the sbatch file (8h should be plenty for 1.6k samples)
