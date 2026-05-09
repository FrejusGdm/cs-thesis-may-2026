# Adja ASR Data

## Source
- **HuggingFace Dataset**: `JosueG/adja-tts-orpheus` (private)
- **Format**: HuggingFace `datasets` with `text` (string) and `audio` (Audio feature) columns
- **Size**: ~1.6k utterances, train split only
- **Language**: Adja (Gbe family, tonal)

## Orthography
Adja text uses special characters from the Gbe orthographic tradition:
- ɛ (open-mid front unrounded vowel)
- ɔ (open-mid back rounded vowel)
- ŋ (velar nasal)
- ɖ (retroflex stop)
- Tone marks: é, è, ẽ, etc.

These MUST be preserved in the character vocabulary. Do NOT normalize them away.

## Split Policy
Created by `experiments/asr/shared/data_prep.py` with:
- **Train**: ~80% (~1,280 utterances)
- **Dev**: ~10% (~160 utterances)
- **Test**: ~10% (~160 utterances)
- **Seed**: 42 (fixed for reproducibility)

All experiments MUST use the same splits. Never re-split.

## Generated Files
After running data_prep.py:
```
data/
  manifests/
    train.tsv          # id, audio_path, text, duration_sec, sampling_rate
    dev.tsv
    test.tsv
  wavs/
    train/             # WAV files for train split
    dev/
    test/
  char_vocab.json      # character-to-index mapping
  split_info.json      # split reproducibility info
```

## Manifest Format
TSV with columns:
```
id	audio_path	text	duration_sec	sampling_rate
train_00000	/path/to/train/train_00000.wav	Lé mí wɛ yi nyamɔn nyɛ ŋ mɔnɖuјеmɛ	4.320	16000
```
