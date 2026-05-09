#!/usr/bin/env bash
# Dry run — exercises all pipeline modes with stubs.
# No GPU, no API key, no Adja speaker needed. Runs in <30s on Mac.
# 2026-05-02: added Mode B (--input-text) and OpenRouter stub coverage.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SAMPLE_WAV="samples/test_sample.wav"

# Generate a 1-second silent test WAV if samples/ is empty
if [ ! -f "$SAMPLE_WAV" ]; then
  echo "Generating silent test sample ..."
  python3 -c "
import numpy as np, soundfile as sf
sf.write('samples/test_sample.wav', np.zeros(16000, dtype='float32'), 16000)
print('Created samples/test_sample.wav')
"
fi

echo ""
echo "=== Test 1: Mode A — Adja audio, full round-trip (stubs) ==="
python3 pipeline.py \
  --dry-run \
  --mode roundtrip \
  --input "$SAMPLE_WAV" \
  --ref-transcript "mo yi adja bo" \
  --output-dir runs/dry_run \
  "$@"

echo ""
echo "=== Test 2: Mode B — French text input, qa mode (stubs) ==="
python3 pipeline.py \
  --dry-run \
  --mode qa \
  --input-text "Quel est le plat traditionnel adja ?" \
  --output-dir runs/dry_run_mode_b \
  "$@"

echo ""
echo "=== Test 3: Mode B — French text input, roundtrip (MT+TTS only, no LLM) ==="
python3 pipeline.py \
  --dry-run \
  --mode roundtrip \
  --input-text "Le dolo est une boisson fermentée traditionnelle." \
  --output-dir runs/dry_run_mode_b_rt \
  "$@"

echo ""
echo "=== Test 4: ASR stage only (Mode A) ==="
python3 pipeline.py \
  --dry-run \
  --stages asr \
  --input "$SAMPLE_WAV" \
  --ref-transcript "mo yi adja bo" \
  --output-dir runs/dry_run_asr \
  "$@"

echo ""
echo "=== Test 5: Batch mode ==="
python3 pipeline.py \
  --dry-run \
  --mode roundtrip \
  --input-dir samples/ \
  --output-dir runs/dry_run_batch \
  "$@"

echo ""
echo "All dry-run tests passed. Check runs/ for cached outputs and metrics.json."
