# Qwen3 Speech Fine-tuning System

Production-grade fine-tuning infrastructure for **Qwen3-TTS** (Text-to-Speech) and **Qwen3-ASR** (Automatic Speech Recognition), with dual deployment paths for HuggingFace and HPC (SLURM) environments.

## Quick Start

### Prerequisites
- Python 3.12+
- NVIDIA GPU with 8+ GB VRAM (0.6B models) or 16+ GB (1.7B models)
- `pip install qwen-tts qwen-asr librosa soundfile datasets`

### TTS Fine-tuning (4 steps)

```bash
# 1. Prepare your data (CSV with text + audio URL columns)
python data_pipeline/prepare_dataset.py \
    --input_file your_dataset.csv \
    --work_dir ./workspace \
    --mode tts \
    --language YOUR_LANG_CODE \
    --ref_audio reference_speaker.wav

# 2. Tokenize audio with Qwen3 codec
python -c "
from qwen_tts import Qwen3TTSTokenizer
# Use official prepare_data.py from Qwen3-TTS repo
"

# 3. Fine-tune
python training/hf/train_tts.py \
    --config configs/tts_config.yaml \
    --train_jsonl workspace/splits/train_with_codes.jsonl

# 4. Test
python inference/test_tts.py \
    --model_path tts_output/best \
    --speaker_name my_speaker \
    --text "Test sentence in your language"
```

### ASR Fine-tuning (3 steps)

```bash
# 1. Prepare data
python data_pipeline/prepare_dataset.py \
    --input_file your_dataset.csv \
    --work_dir ./workspace \
    --mode asr \
    --language YOUR_LANG_CODE

# 2. Fine-tune
python training/hf/train_asr.py \
    --config configs/asr_config.yaml \
    --train_file workspace/splits/train.jsonl \
    --eval_file workspace/splits/val.jsonl

# 3. Test
python inference/test_asr.py \
    --model_path asr_output/checkpoint-200 \
    --audio_path test_audio.wav
```

---

## Project Structure

```
finetuning-qwen3/
├── README.md                        # This file
├── docs/
│   ├── 01_repository_forensics.md   # Deep analysis of Qwen3-TTS/ASR repos
│   ├── 02_infrastructure_decision_matrix.md  # HF vs HPC comparison
│   ├── adja_runbook.md              # Adja-specific operational runbook
│   └── 03_neural_tts_from_first_principles.md  # Educational guide (~18 min read)
├── data_pipeline/
│   ├── download_audio.py            # Stage 1: Parallel audio download with retries
│   ├── process_audio.py             # Stage 2: Resample, trim, normalize, filter
│   ├── process_text.py              # Stage 3: Language-specific text normalization
│   ├── prepare_dataset.py           # Full pipeline orchestrator
│   └── validate_dataset.py          # Pre-training dataset validation
├── training/
│   ├── hf/                          # HuggingFace path
│   │   ├── train_tts.py             # TTS fine-tuning script
│   │   ├── train_asr.py             # ASR fine-tuning script
│   │   └── requirements.txt         # Pinned dependencies
│   └── hpc/                         # Dartmouth HPC path
│       ├── setup_env.sh             # One-time environment setup
│       ├── train_tts.sbatch         # SLURM job script for TTS
│       ├── train_asr.sbatch         # SLURM job script for ASR
│       └── monitor.sh               # Remote training monitoring
├── inference/
│   ├── test_tts.py                  # TTS inference testing
│   └── test_asr.py                  # ASR inference testing
└── configs/
    ├── tts_config.yaml              # TTS hyperparameters (documented)
    └── asr_config.yaml              # ASR hyperparameters (documented)
```

## Documentation

| Document | What It Covers |
|----------|----------------|
| [Repository Forensics](docs/01_repository_forensics.md) | Architecture deep dive, training pipeline analysis, data formats, compute requirements, language support |
| [Infrastructure Decision Matrix](docs/02_infrastructure_decision_matrix.md) | HuggingFace vs Dartmouth HPC comparison with setup guides for both |
| [Adja Runbook](docs/adja_runbook.md) | Adja-specific HF submission path, job IDs, and operational notes |
| [Qwen Documentation Map](../../docs/qwen3-documentation-map.md) | One-year-later navigation guide for the whole Qwen Adja effort |
| [Neural TTS from First Principles](docs/03_neural_tts_from_first_principles.md) | From human speech physiology to Qwen3-TTS architecture — conceptual understanding for informed fine-tuning |

## Data Format

### TTS Input (JSONL)
```json
{"audio": "/path/to/utterance.wav", "text": "Transcript here", "ref_audio": "/path/to/ref_speaker.wav"}
```

### ASR Input (JSONL)
```json
{"audio": "/path/to/utterance.wav", "text": "language YourLanguage<asr_text>Transcript here"}
```

## Key References

- [Qwen3-TTS GitHub](https://github.com/QwenLM/Qwen3-TTS)
- [Qwen3-ASR GitHub](https://github.com/QwenLM/Qwen3-ASR)
- [Qwen3-TTS Technical Report (arXiv)](https://arxiv.org/abs/2601.15621)
- [Qwen3-TTS HuggingFace](https://huggingface.co/Qwen)

## License

This fine-tuning infrastructure is provided under the MIT License.
The Qwen3-TTS and Qwen3-ASR models are licensed under Apache 2.0 by Alibaba Cloud.
