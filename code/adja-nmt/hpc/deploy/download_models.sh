#!/usr/bin/env bash
# download_models.sh — pre-cache all Adja speech models to the cluster.
# Last updated: 2026-04-21
#
# Run ONCE on the Discovery login node AFTER building speech-training.sif.
# Skips any model already present. Requires HF_TOKEN env var.
#
# Usage:
#   export HF_TOKEN=hf_xxxxx
#   bash download_models.sh

set -euo pipefail
HPC_DIR="<HPC_WORKDIR>"
CONTAINER="${HPC_DIR}/speech-training.sif"
MODELS_DIR="${HPC_DIR}/models"
DATA_DIR="${HPC_DIR}/data"
mkdir -p "$MODELS_DIR" "$DATA_DIR"

if [ ! -f "$CONTAINER" ]; then
    echo "ERROR: ${CONTAINER} missing. Build first: bash apptainer/build.sh"
    exit 1
fi
if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN not set. export HF_TOKEN=hf_xxxxx"
    exit 1
fi

echo "Pre-caching models to: ${MODELS_DIR}"

# ASR models
MODELS_ASR=(
    "openai/whisper-large-v3"
    "facebook/wav2vec2-xls-r-300m"
    "facebook/wav2vec2-xls-r-1b"
    "facebook/mms-1b-all"
)

# TTS models
MODELS_TTS=(
    "unsloth/csm-1b"
    "canopylabs/orpheus-3b-0.1-ft"
    "canopylabs/3b-zh-ft-research_release"
    "canopylabs/3b-fr-ft-research_release"
    "hubertsiuzdak/snac_24khz"
)

download_model() {
    local repo="$1"
    local safe_name="${repo//\//__}"
    local target="${MODELS_DIR}/${safe_name}"
    if [ -d "$target" ] && [ -n "$(ls -A "$target" 2>/dev/null)" ]; then
        echo "  [skip] ${repo} (already at ${target})"
        return
    fi
    echo "  [dl]   ${repo}"
    apptainer exec \
        --bind "${MODELS_DIR}:/models" \
        --env "HF_TOKEN=${HF_TOKEN}" \
        "$CONTAINER" \
        python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('${repo}', local_dir='/models/${safe_name}', token='${HF_TOKEN}')
"
}

echo "=== ASR models ==="
for m in "${MODELS_ASR[@]}"; do download_model "$m"; done

echo "=== TTS models ==="
for m in "${MODELS_TTS[@]}"; do download_model "$m"; done

# Datasets
echo "=== Datasets ==="
for cfg in ewe_asr ewe_tts; do
    target="${DATA_DIR}/WaxalNLP_${cfg}"
    if [ -d "$target" ] && [ -n "$(ls -A "$target" 2>/dev/null)" ]; then
        echo "  [skip] google/WaxalNLP/${cfg} (already at ${target})"
        continue
    fi
    echo "  [dl]   google/WaxalNLP/${cfg}"
    apptainer exec \
        --bind "${DATA_DIR}:/data" \
        --env "HF_TOKEN=${HF_TOKEN}" \
        "$CONTAINER" \
        python3 -c "
from datasets import load_dataset
from datasets import Features, Value, load_dataset_builder

def _is_schema_cast_error(e: Exception) -> bool:
    msg = str(e)
    return ('column names don\\'t match' in msg) or ('Couldn\\'t cast' in msg)

def _load_waxal(split: str):
    cache = '/data/WaxalNLP_${cfg}'
    builder = load_dataset_builder('google/WaxalNLP', name='${cfg}', token='${HF_TOKEN}', cache_dir=cache)
    feats = builder.info.features
    feats_plus = Features(dict(feats))
    if '__index_level_0__' not in feats_plus:
        feats_plus['__index_level_0__'] = Value('int64')
    try:
        ds = load_dataset('google/WaxalNLP', name='${cfg}', split=split, token='${HF_TOKEN}',
                          cache_dir=cache, features=feats_plus)
    except Exception as e:
        if not _is_schema_cast_error(e):
            raise
        ds = load_dataset('google/WaxalNLP', name='${cfg}', split=split, token='${HF_TOKEN}',
                          cache_dir=cache, features=feats)
    if '__index_level_0__' in ds.column_names:
        ds = ds.remove_columns('__index_level_0__')
    return ds

for split in ('train', 'validation', 'test'):
    try:
        _load_waxal(split)
    except Exception as e:
        print(f'  note: split={split} not available ({e})')
# unlabeled split only exists for ewe_asr
if '${cfg}' == 'ewe_asr':
    try:
        _load_waxal('unlabeled')
    except Exception as e:
        print(f'  note: unlabeled split ({e})')
"
done

# Adja dataset
target="${DATA_DIR}/JosueG_adja-tts-orpheus"
if [ -d "$target" ] && [ -n "$(ls -A "$target" 2>/dev/null)" ]; then
    echo "  [skip] JosueG/adja-tts-orpheus"
else
    echo "  [dl]   JosueG/adja-tts-orpheus"
    apptainer exec \
        --bind "${DATA_DIR}:/data" \
        --env "HF_TOKEN=${HF_TOKEN}" \
        "$CONTAINER" \
        python3 -c "
from datasets import load_dataset
load_dataset('JosueG/adja-tts-orpheus', token='${HF_TOKEN}',
             cache_dir='/data/JosueG_adja-tts-orpheus')
"
fi

echo "All downloads done."
du -sh "${MODELS_DIR}" "${DATA_DIR}"
