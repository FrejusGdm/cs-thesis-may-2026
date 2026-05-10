# TTS audio-range normalization reruns, 2026-05-09

**Status:** rerun wave in progress. This note documents the CSM/Orpheus TTS correction made after
auditing the Adja parquet audio arrays. It is meant to preserve provenance for the thesis and for
later listening comparisons.

## Short conclusion

The April CSM and Orpheus Adja TTS failures should be treated as **dirty/preliminary** until the
May 9 audio-range reruns finish. The old runs remain valid as observed outputs, but they may have
been trained on badly scaled Adja waveforms.

Spark TTS is not part of this caveat: its training path already performed explicit audio volume
normalization, and its intelligible Adja result still stands as the clean direct-Adja TTS result.
Whisper/ASR is tracked separately from this TTS note.

## What was wrong

The Adja dataset `JosueG/adja-tts-orpheus` can expose audio arrays as PCM-scale floats instead of
normal `[-1, 1]` waveforms. In practice, local parquet inspection found float arrays with peaks
consistent with integer PCM magnitudes.

That matters because several TTS paths passed `example["audio"]["array"]` straight into the model
frontend:

- CSM paths passed the array directly into `processor.apply_chat_template(...)`.
- Orpheus paths converted the array directly to a tensor and passed it into SNAC encoding.
- SageMaker CF paths decoded or loaded arrays and then passed them into the CSM processor without
  an explicit range guard.

If the array has peak magnitudes around thousands or tens of thousands, the model frontend sees a
clipped or invalid waveform distribution. That can plausibly turn a weak low-resource TTS run into
pure noise, so the old CSM/Orpheus conclusions need an asterisk until the fixed runs are listened to.

## Normalization rule now used

All patched paths now apply the same conservative waveform guard before model-specific processing:

1. Convert to `float32`.
2. Fold multichannel audio to mono if needed.
3. If `max(abs(wav)) <= 2.0`, treat it as already normalized and leave it alone.
4. If the peak is PCM-like, scale by `65536.0`.
5. If the peak is larger than that expected PCM range, peak-scale by the observed maximum.
6. After resampling, normalize again as a guard against numeric overshoot.

This intentionally avoids changing already-normalized audio while rescuing PCM-scale arrays.

## Code changes

### Hugging Face CSM scripts

Patched scripts:

- `scripts/hf_jobs/T1_sesame_csm_finetune.py`
- `scripts/hf_jobs/T1_csm_tokfix.py`
- `scripts/hf_jobs/T1_csm_ewe_adja_stage2.py`

Changes:

- Added `normalize_waveform_range(...)`, `prepare_audio_array(...)`, and `target_sample_count(...)`.
- Removed Adja `cast_column("audio", Audio(sampling_rate=24000))` from the patched CSM paths so the
  scripts operate on the raw dataset arrays before any implicit recoding.
- Length filtering now uses the projected 24 kHz sample count, not the raw source sample count.
- Preprocessing now passes a normalized/resampled `audio_array` into the CSM processor.
- `T1_csm_tokfix.py` and `T1_csm_ewe_adja_stage2.py` now include `torchaudio==2.5.1` in the PEP 723
  uv dependency block. The first AUDIOFIX submissions for those two jobs failed because this
  dependency was missing; the AUDIOFIX2 jobs are the replacement jobs.

### Hugging Face Orpheus scripts

Patched scripts:

- `scripts/hf_jobs/T2_orpheus_finetune.py`
- `scripts/hf_jobs/T2_orpheus_tokfix.py`
- `scripts/hf_jobs/T2_orpheus_ewe_adja_stage2.py`
- `scripts/hf_jobs/T2_orpheus_ewe_adja_stage2_mixed.py`
- `scripts/hf_jobs/T2_orpheus_zh_adja.py`

Changes:

- `build_snac_codes(...)` now normalizes PCM-scale arrays before creating the waveform tensor.
- Direct Adja, tokenizer-fix, full fine-tune, EN/FR/ZH base, pure Stage 2, and mixed Stage 2
  Orpheus variants are all covered by the patched SNAC input path.

### SageMaker CSM curriculum scripts

Patched scripts:

- `scripts/sagemaker_jobs/train_CF1_ewc_stage2.py`
- `scripts/sagemaker_jobs/train_CF2_curriculum_stage2.py`
- `scripts/sagemaker_jobs/train_CF3_frozen_backbone_stage2.py`
- `scripts/sagemaker_jobs/launch.py`

Changes:

- Added `normalize_waveform_range(...)`.
- `decode_audio(...)` now handles `ex_audio["array"]` directly and normalizes before resampling.
- Decoded bytes and file-path audio also go through the same normalization guard.
- Adja datasets are no longer recast to `Audio(decode=False)` before normalization.
- The launcher now has `CF1_AUDIOFIX`, `CF2_AUDIOFIX`, and `CF3_AUDIOFIX` entries with fresh
  result prefixes.

## Rerun matrix

All reruns use fresh prefixes so old April result folders remain untouched.

### Hugging Face Jobs

| Prefix | Scope | Job ID | Current note |
|--------|-------|--------|--------------|
| `T1_AUDIOFIX_20260509` | CSM direct Adja LoRA | `69ff7b51317220dbbd1a7251` | running |
| `T1_full_ft_AUDIOFIX_20260509` | CSM direct Adja full fine-tune | `69ff7be4317220dbbd1a7265` | scheduling/running wave |
| `T1_csm_tokfix_AUDIOFIX_20260509` | CSM tokenizer-expanded direct Adja | `69ff7b52aff1cd33e8f32156` | failed fast: missing `torchaudio`; superseded |
| `T1_csm_tokfix_AUDIOFIX2_20260509` | CSM tokenizer-expanded direct Adja | `69ff7d15aff1cd33e8f32171` | running; past dependency failure |
| `T1_csm_ewe_adja_stage2_AUDIOFIX_20260509` | CSM Ewe Stage 1 -> Adja Stage 2 | `69ff7b50317220dbbd1a724d` | failed fast: missing `torchaudio`; superseded |
| `T1_csm_ewe_adja_stage2_AUDIOFIX2_20260509` | CSM Ewe Stage 1 -> Adja Stage 2 | `69ff7d1daff1cd33e8f32173` | running; past dependency failure |
| `T2_orpheus_en_lora_r32_AUDIOFIX_20260509` | Orpheus EN direct Adja LoRA r=32 | `69ff7bc6aff1cd33e8f32162` | running |
| `T2_orpheus_en_lora_r64_AUDIOFIX_20260509` | Orpheus EN direct Adja LoRA r=64 | `69ff7b51aff1cd33e8f3214e` | running |
| `T2_orpheus_en_lora_r128_AUDIOFIX_20260509` | Orpheus EN direct Adja LoRA r=128 | `69ff7bd5aff1cd33e8f32164` | running |
| `T2_orpheus_en_fullft_AUDIOFIX_20260509` | Orpheus EN direct Adja full fine-tune | `69ff7bd5317220dbbd1a7263` | scheduling/running wave |
| `T2_orpheus_fr_lora_r64_AUDIOFIX_20260509` | Orpheus FR direct Adja LoRA r=64 | `69ff7bd5aff1cd33e8f32166` | running |
| `T2_orpheus_tokfix_AUDIOFIX_20260509` | Orpheus tokenizer-expanded direct Adja | `69ff7be6317220dbbd1a7269` | running |
| `T2_orpheus_zh_AUDIOFIX_20260509` | Orpheus ZH direct Adja | `69ff7be5317220dbbd1a7267` | running |
| `T2_orpheus_en_ewe_adja_stage2_AUDIOFIX_20260509` | Orpheus EN Ewe Stage 1 -> Adja Stage 2 | `69ff7b51317220dbbd1a724f` | running |
| `T2_orpheus_en_ewe_adja_stage2_mixed_AUDIOFIX_20260509` | Orpheus EN mixed Ewe+Adja Stage 2 | `69ff7b50317220dbbd1a724b` | running |
| `T2_orpheus_fr_ewe_adja_stage2_AUDIOFIX_20260509` | Orpheus FR Ewe Stage 1 -> Adja Stage 2 | `69ff7bf4317220dbbd1a726b` | running |
| `T2_orpheus_zh_ewe_adja_stage2_AUDIOFIX_20260509` | Orpheus ZH Ewe Stage 1 -> Adja Stage 2 | `69ff7bf4aff1cd33e8f32168` | running; depends on ZH Stage 1 checkpoint availability |

### SageMaker

| Prefix | Scope | Training job | Instance | Current note |
|--------|-------|--------------|----------|--------------|
| `CF1_ewc_stage2_AUDIOFIX_20260509` | CSM EWC Stage 2 | `adja-cf1-audiofix-full-20260509-142331` | `ml.g5.2xlarge` | `InProgress`, `Training` |
| `CF2_curriculum_stage2_AUDIOFIX_20260509` | CSM curriculum Stage 2 | `adja-cf2-audiofix-full-20260509-142331` | `ml.g5.2xlarge` | `InProgress`, `Training` |
| `CF3_frozen_backbone_stage2_AUDIOFIX_20260509` | CSM frozen-backbone Stage 2 | `adja-cf3-audiofix-full-20260509-142331` | `ml.g5.2xlarge` | `InProgress`, `Training` |

## How to interpret this in the thesis

Until the reruns finish and the generated audio is listened to, the safest thesis language is:

> A later audit found that some CSM and Orpheus TTS runs may have consumed Adja waveforms without
> explicit range normalization. Those runs are therefore treated as preliminary and are being rerun
> with explicit waveform normalization. The Spark result is unaffected because its training path
> already included explicit audio normalization.

Do not overclaim that normalization will fix CSM/Orpheus. The honest claim is narrower:

- The old CSM/Orpheus failures are potentially confounded.
- The new audio-fix wave is the correct controlled comparison.
- If the fixed outputs remain noise, the original catastrophic-forgetting / data-volume story is
  strengthened.
- If the fixed outputs improve materially, the thesis should treat the earlier CSM/Orpheus runs as
  contaminated ablations rather than model-family failures.

## Follow-up checklist

- Wait for all HF and SageMaker jobs to complete.
- Download or inspect generated audio under the new prefixes.
- Record native-speaker listening verdicts next to the old April verdicts.
- Update `results/tts-comparison.md` with a separate `AUDIOFIX_20260509` block.
- If CSM/Orpheus improve, revise the TTS chapter language away from "model failure" and toward
  "pipeline sensitivity to waveform normalization plus low-resource instability."
- If they do not improve, keep the main thesis claim but cite this rerun wave as the normalization
  control.
