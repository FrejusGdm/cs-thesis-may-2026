# Orpheus Audio Audit

This helper exists because the processed MMS-ready derivative sounded noisy
during manual Hugging Face playback review. The canonical speech release goes
back to the Orpheus source dataset, and this audit makes that decision
testable.

## Goal

Export a few rows from `JosueG/adja-speech-asr-tts` to WAV and record:

- sample rate
- duration
- array dtype
- shape
- min/max range
- peak amplitude
- RMS amplitude

Optionally export matching rows from `JosueG/adja-tts-mms-ready` for comparison.

## Example

```bash
python data/audio_audit/export_orpheus_audio_samples.py \
  --repo JosueG/adja-speech-asr-tts \
  --config adja_speech_orpheus_48khz \
  --split train \
  --output-dir /tmp/adja-audio-audit \
  --max-samples 5
```

Comparison against the MMS-ready derivative:

```bash
python data/audio_audit/export_orpheus_audio_samples.py \
  --repo JosueG/adja-speech-asr-tts \
  --config adja_speech_orpheus_48khz \
  --split train \
  --compare-repo JosueG/adja-tts-mms-ready \
  --compare-split train \
  --output-dir /tmp/adja-audio-audit \
  --max-samples 5
```

## Interpretation

Plain resampling from 48 kHz to 16 kHz should not turn normal speech into pure
noise by itself. If the MMS-ready examples are noisy while Orpheus exports play
normally, likely causes include array dtype/range interpretation, channel shape,
codec reconstruction, or dataset materialization issues.
