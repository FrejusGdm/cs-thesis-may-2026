# B1: BiLSTM-CTC Baseline ASR for Adja

## Overview

From-scratch BiLSTM encoder with CTC loss for Adja automatic speech recognition.
This is the simplest neural baseline in the ASR experiment series -- no pretrained
components, no attention mechanism, no language model. It establishes the floor
performance for character-level ASR on Adja.

## Architecture

```
Raw waveform (16kHz)
    |
Log-mel spectrogram (80 dims, 25ms window, 10ms hop)
    |
SpecAugment (freq + time masking)
    |
3-layer BiLSTM (256 hidden, dropout 0.3)
    |
Linear projection -> vocab_size
    |
CTC loss (blank = index 0)
```

- **Input**: 16kHz mono WAV files
- **Features**: 80-dimensional log-mel spectrograms
- **Encoder**: 3-layer bidirectional LSTM, 256 hidden units per direction (512 total)
- **Output**: Character-level tokens (Adja orthography including special Gbe characters)
- **Loss**: CTC (Connectionist Temporal Classification)
- **Decoding**: Greedy (argmax + collapse repeats + remove blanks)

## Key Design Decisions

- **Character-level**: Adja has limited training data (~1.6k utterances), so subword
  tokenization is not beneficial. Character-level modeling also handles the special
  Gbe orthographic characters naturally.
- **CTC blank at index 0**: The shared `char_vocab.json` places `<blank>` at index 4.
  The training script remaps the vocabulary so blank is at index 0 (CTC convention).
- **SpecAugment**: Applied during training only. Critical for regularization given the
  small dataset size.
- **Speed perturbation**: Factors [0.9, 1.0, 1.1] applied at data loading time. This
  effectively triples the training data diversity.
- **No Trainer**: Custom PyTorch training loop for full control and debuggability.

## References

1. **CTC**: Graves, A., Fernandez, S., Gomez, F., & Schmidhuber, J. (2006).
   Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with
   Recurrent Neural Networks. *ICML 2006*.
   https://www.cs.toronto.edu/~graves/icml_2006.pdf

2. **SpecAugment**: Park, D. S., Chan, W., Zhang, Y., Chiu, C., Zoph, B., Cubuk, E. D.,
   & Le, Q. V. (2019). SpecAugment: A Simple Data Augmentation Method for Automatic
   Speech Recognition. *Interspeech 2019*.
   https://arxiv.org/abs/1904.08779

3. **BiLSTM for ASR**: Graves, A., Mohamed, A., & Hinton, G. (2013). Speech Recognition
   with Deep Recurrent Neural Networks. *ICASSP 2013*.

## Usage

### Prerequisites

```bash
pip install torch torchaudio pyyaml numpy
```

### Data Preparation

Run the shared data prep script first (if not already done):

```bash
cd experiments/asr/shared
python data_prep.py --output-dir ../../../data --hf-token $HF_TOKEN
```

This creates `data/manifests/{train,dev,test}.tsv` and `data/char_vocab.json`.

### Training

```bash
cd experiments/asr/B1_bilstm_ctc

# Full training
python train.py \
    --config conf/train.yaml \
    --data-dir ../../../data \
    --output-dir ./output

# Dry run (2 steps, 5 samples -- sanity check)
python train.py \
    --config conf/train.yaml \
    --data-dir ../../../data \
    --output-dir ./output_dry \
    --dry-run
```

### Output Files

After training, `output/` contains:

| File | Description |
|------|-------------|
| `best_model.pt` | Best checkpoint (by dev CER) |
| `metrics.json` | Full training history and best scores |
| `decode_samples.txt` | Sample REF/HYP pairs from best epoch |
| `timing.json` | Wall-clock timing information |
| `vocab_remapped.json` | Vocabulary with CTC blank at index 0 |
| `train_config.yaml` | Copy of the config used |

## Expected Results

On ~1.6k Adja utterances with 80/10/10 train/dev/test split, expect:

- **Dev CER**: 40--70% (this is a from-scratch baseline on a very low-resource tonal language)
- **Training time**: ~30 minutes on A100 80GB, ~2 hours on T4

The high CER is expected. This baseline exists to measure how much pretrained models
(Whisper, MMS, wav2vec2) improve over a from-scratch system on Adja.

## Files

```
B1_bilstm_ctc/
    README.md           # This file
    conf/
        train.yaml      # All hyperparameters
    model.py            # BiLSTM-CTC model definition
    train.py            # Training script (custom loop)
```
