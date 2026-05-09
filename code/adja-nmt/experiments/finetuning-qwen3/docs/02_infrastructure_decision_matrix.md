# Phase 2: Infrastructure Decision Matrix — Hugging Face vs. Dartmouth HPC

---

## Executive Summary

| Criteria                | Hugging Face                  | Dartmouth HPC (Discovery)      |
|-------------------------|-------------------------------|--------------------------------|
| **Setup complexity**    | Low (managed platform)        | Medium (SLURM + modules)       |
| **Cost**                | $$ (paid compute)             | Free (institutional access)    |
| **GPU quality**         | A10G/A100 (depending on tier) | Varies (check `sinfo`)         |
| **Max runtime**         | Hours-days (depending on plan)| Typically 24-72h per job        |
| **Data privacy**        | Data uploaded to HF servers   | On-premises, full control      |
| **Reproducibility**     | Excellent (containerized)     | Good (module system)           |
| **Collaboration**       | Built-in (HF Hub)             | Manual (scp/git)               |
| **Debugging**           | Limited (remote)              | Full (SSH, interactive jobs)   |

**Recommendation:** Use **Dartmouth HPC** for the main training runs (free, more control, likely better GPUs),
and **Hugging Face** for sharing the final model and running inference demos.

---

## Option A: Hugging Face

### Feasibility

Qwen3-TTS uses a custom model architecture (`Qwen3TTSModel`) that is **not** a standard HuggingFace
Transformers model. It's distributed as a pip package (`qwen-tts`), not through `AutoModel`.

This means:
- You **cannot** use `transformers.Trainer` directly for TTS fine-tuning
- You **can** use HF's compute infrastructure (Spaces, Training endpoints) but must bring your own training script
- The ASR model (`qwen-asr`) works with HF Trainer via a forward-patching trick (see forensics doc)

### Setup

#### For Training:
1. **HF Training with Notebooks**: Use a paid GPU notebook (A10G ~$1.05/hr, A100 ~$4.13/hr on HF Pro)
2. **HF Spaces for Inference**: Free tier for demos (but limited to CPU or T4)
3. **HF AutoTrain**: NOT compatible — Qwen3-TTS is not a standard transformers model

#### For the Dataset:
```python
# Upload your processed dataset to HF Hub
from datasets import Dataset, Audio
ds = Dataset.from_dict({
    "audio": [...],  # audio file paths
    "text": [...],   # transcripts
}).cast_column("audio", Audio(sampling_rate=24000))
ds.push_to_hub("your-username/your-language-tts")
```

### Data Pipeline in HF

Streaming from external audio links within HF:
```python
import requests
import io
import soundfile as sf

def download_audio(url, target_sr=24000):
    response = requests.get(url, timeout=30)
    audio, sr = sf.read(io.BytesIO(response.content))
    if sr != target_sr:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio
```

### Cost Estimation

| Scenario                          | Instance   | $/hour | Hours  | Total     |
|-----------------------------------|-----------|--------|--------|-----------|
| TTS fine-tune, 0.6B, 1K samples  | A10G      | ~$1.05 | ~3-5   | ~$3-5     |
| TTS fine-tune, 1.7B, 5K samples  | A100 40GB | ~$4.13 | ~15-25 | ~$60-100  |
| ASR fine-tune, 1.7B, 10K samples | A100 80GB | ~$6.50 | ~10-20 | ~$65-130  |

**Free tier limitations:**
- HF Spaces: T4 GPU with 16GB VRAM, 72-hour timeout, limited storage
- Enough for 0.6B inference, not enough for 1.7B fine-tuning

### Pros
- Easy model sharing after training (push to Hub)
- Built-in experiment tracking (TensorBoard integration)
- Datasets library handles audio loading/resampling
- Good for collaboration and reproducibility

### Cons
- Costs real money for training
- Limited debugging (can't SSH into instances easily)
- Session timeouts can kill long training runs
- Data must be uploaded to HF servers (privacy concern for some)
- Qwen3-TTS not natively supported by AutoModel/Trainer

---

## Option B: Dartmouth HPC (Discovery Cluster)

### Cluster Overview

Discovery is a Linux cluster managed by Dartmouth Research Computing:
- **Nodes**: 128 nodes, 6,712 CPU cores, 54.7 TB RAM, 2.8+ PB storage
- **OS**: Red Hat Enterprise Linux 8.10
- **Scheduler**: SLURM
- **Access**: SSH via campus network or VPN

> **IMPORTANT**: Run `sinfo -o "%P %G %m %c"` on the cluster to see exact GPU types and availability.
> Common GPUs on university clusters: NVIDIA V100 (32GB), A100 (40/80GB), A6000 (48GB).

### Setup Guide

#### Step 1: Connect and Configure
```bash
# SSH into the cluster
ssh your_netid@discovery7.dartmouth.edu

# Check available modules
module avail cuda
module avail python
module avail anaconda

# Check GPU partitions
sinfo -o "%20P %10G %10m %5c %10l" | head -20
# This shows: Partition, GPUs, Memory, CPUs, Time Limit
```

#### Step 2: Create Conda Environment
```bash
# Load anaconda module (exact name may vary)
module load anaconda3

# Create environment
conda create -n qwen3-speech python=3.12 -y
conda activate qwen3-speech

# Install PyTorch with CUDA (check cluster CUDA version first with `nvidia-smi`)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install Qwen packages
pip install qwen-tts qwen-asr

# Install FlashAttention (limit parallel jobs if RAM is shared)
MAX_JOBS=4 pip install flash-attn --no-build-isolation

# Additional dependencies
pip install librosa soundfile datasets accelerate tensorboard pyyaml
```

#### Step 3: Storage Strategy

| Location          | Typical Quota | Use For                          |
|-------------------|--------------|----------------------------------|
| `$HOME`           | 50 GB        | Code, configs, conda env         |
| `/scratch/$USER`  | 500 GB-1 TB  | Training data, checkpoints       |
| `/datasets`       | Shared       | Pre-downloaded common datasets   |

```bash
# Create working directories
mkdir -p /scratch/$USER/qwen3-tts/{data,checkpoints,logs}
mkdir -p /scratch/$USER/qwen3-asr/{data,checkpoints,logs}

# Symlink for convenience
ln -s /scratch/$USER/qwen3-tts ~/qwen3-tts-work
ln -s /scratch/$USER/qwen3-asr ~/qwen3-asr-work
```

> **Check actual paths**: Run `echo $SCRATCH` or `quota` to see your actual scratch directory and limits.

### SLURM Job Configuration

#### Interactive Session (for debugging)
```bash
srun --partition=gpu \
     --gres=gpu:1 \
     --mem=64G \
     --cpus-per-task=8 \
     --time=04:00:00 \
     --pty bash
```

#### Batch Job (for training)
See `training/hpc/train_tts.sbatch` and `training/hpc/train_asr.sbatch` for complete job scripts.

### Monitoring Training Remotely

#### Option 1: TensorBoard over SSH tunnel
```bash
# On the cluster (inside your job or interactive session)
tensorboard --logdir=/scratch/$USER/qwen3-tts/logs --port=6006 --bind_all

# On your local machine
ssh -L 6006:compute-node:6006 your_netid@discovery7.dartmouth.edu
# Then open http://localhost:6006 in your browser
```

#### Option 2: Check logs directly
```bash
# View training output
tail -f /scratch/$USER/qwen3-tts/logs/slurm-*.out

# Check job status
squeue -u $USER

# Check GPU utilization of your job
srun --jobid=YOUR_JOB_ID nvidia-smi
```

#### Option 3: Periodic email updates
Add to your SLURM script:
```bash
#SBATCH --mail-user=your.name@dartmouth.edu
#SBATCH --mail-type=BEGIN,END,FAIL
```

### Pros
- **Free** — no compute costs
- Full control over environment and debugging
- Can SSH into compute nodes for interactive debugging
- Likely has A100s or V100s (good for ML training)
- Data stays on-premises (good for privacy)
- Can run multi-day jobs with checkpointing

### Cons
- Queue wait times (especially for GPU nodes)
- Job time limits (typically 24-72 hours, requires checkpointing)
- Steeper learning curve (SLURM, module system)
- Shared resources (other users competing for GPUs)
- Model sharing requires manual steps (scp to local → push to HF)

---

## My Recommendation

### Primary Path: Dartmouth HPC

Use Discovery for all training because:
1. It's free — you're a student, save your money
2. University clusters typically have excellent GPUs (A100s are common)
3. You have full SSH access for debugging
4. Multi-hour training runs are standard on HPC systems
5. You get to learn SLURM, which is a valuable skill for ML research

### Secondary Path: Hugging Face (for sharing)

After training on Discovery:
1. Export your best checkpoint
2. Push to HF Hub for easy sharing and inference
3. Create a HF Space demo with Gradio

### Workflow

```
[Your Dataset] → [Data Pipeline on Discovery] → [Training on Discovery]
                                                       ↓
                                      [Best Checkpoint] → [Push to HF Hub]
                                                              ↓
                                                     [HF Space Demo]
```

---

## Quick-Start Decision Flowchart

```
Do you have Dartmouth HPC access?
├── YES → Use Discovery for training
│         ├── Need GPU? → srun --gres=gpu:1
│         ├── Long training? → sbatch with checkpointing
│         └── Done? → Push model to HF Hub
│
└── NO → Use HF Compute
          ├── Budget > $100? → A100 instance
          ├── Budget < $50? → A10G instance
          └── Free tier? → 0.6B model only, T4 GPU
```
