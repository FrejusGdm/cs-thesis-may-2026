
## 2026-04-28 — MT-NLLB smoke script prep
- Status: script ready, smoke job submitted
- Platform: HF Jobs | GPU: cpu-upgrade | Job: `69f043d0d2c8bd8662bd1953`
- Script: `experiments/mt/train_nllb.py`
- Notes: custom PEP-723 uv script written, py_compile passed locally. First smoke job failed with exit code 2 during container execution; logs need exact traceback before retry.
# Run Ledger

Chronological log of every run attempt — including failed and crashed runs.

---

## 2026-05-09 — OmniASR audio-range fixed full fine-tune matrix [RUNNING]

- Trigger:
  - Local parquet audit found `JosueG/adja-tts-orpheus` audio arrays are stored as int-like PCM magnitudes in float arrays.
  - Prior Omni materialization wrote those arrays directly to WAV, clipping the training audio before the recipe-level `normalize_audio` flag could help.
- Code fix:
  - `scripts/hf_jobs/omni_asr_finetune.py` now scales HF audio arrays before WAV write.
  - `scripts/hf_jobs/omni_asr_finetune_recipe.py` has the same pre-WAV range fix.
  - Scaling rule: leave `[-1, 1]` audio unchanged; divide PCM-like peaks up to `65536` by `65536`; otherwise peak-scale with headroom.
- Scope:
  - Rerun all Omni fine-tuning-capable checkpoints: CTC + LLM at 300M, 3B, and 7B.
  - Do not rerun `Omni_ZS_*` or `Omni_ICL_*` here; those are inference-only / prompting baselines, not fine-tuning jobs.
  - Keep old April result folders intact by using fresh `Omni_AUDIOFIX_*_20260509_*` prefixes.

| Exp ID | Model | Job | Hardware | Key settings |
|--------|-------|-----|----------|--------------|
| `Omni_AUDIOFIX2_CTC300M_20260509_M2` | `omniASR_CTC_300M_v2` | `69ff5bfaaff1cd33e8f31fe6` | `h200` | full train/dev/test, `VALID_SPLITS=dev,test`, `MIN_AUDIO_LEN=32000`, 400 steps |
| `Omni_AUDIOFIX2_LLM300M_20260509_M0` | `omniASR_LLM_300M_v2` | `69ff5bf9aff1cd33e8f31fe0` | `h200` | full train/dev/test, `MIN_AUDIO_LEN=1600`, 8s cap, 200 steps |
| `Omni_AUDIOFIX2_CTC3B_20260509_M2` | `omniASR_CTC_3B_v2` | `69ff5bfa317220dbbd1a71b5` | `h200` | full train/dev/test, `MIN_AUDIO_LEN=32000`, `DATA_PARALLELISM=fsdp`, 390 steps |
| `Omni_AUDIOFIX2_LLM3B_20260509_M0` | `omniASR_LLM_3B_v2` | `69ff5bfaaff1cd33e8f31fe4` | `h200` | full train/dev/test, `MIN_AUDIO_LEN=1600`, 8s cap, 200 steps |
| `Omni_AUDIOFIX2_CTC7B_20260509_M2` | `omniASR_CTC_7B_v2` | `69ff5bfaaff1cd33e8f31fe2` | `h200` | full train/dev/test, `MIN_AUDIO_LEN=32000`, 400 steps |
| `Omni_AUDIOFIX2_LLM7B_20260509_M0` | `omniASR_LLM_7B_v2` | `69ff5bfaaff1cd33e8f31fe8` | `h200x4` | full train/dev/test, `MIN_AUDIO_LEN=1600`, 8s cap, `DATA_PARALLELISM=fsdp`, 200 steps |

- Immediate status:
  - First submission wave (`Omni_AUDIOFIX_*`; jobs `69ff5893`, `69ff598e`, `69ff5af5`/`69ff5af6`) was canceled after noticing the trainer could print secret env vars in logs.
  - Launcher now removes `HF_TOKEN` / `UV_SCRIPT_HF_TOKEN` from child-process environments before the fairseq trainer starts.
  - Replacement `Omni_AUDIOFIX2_*` wave above was accepted and is running.

## 2026-05-09 — TTS audio-range fixed CSM/Orpheus rerun wave [RUNNING]

- Trigger:
  - Parquet audit found `JosueG/adja-tts-orpheus` audio arrays can be stored as PCM-scale float arrays instead of normalized `[-1, 1]` waveforms.
  - Spark is excluded from this rerun wave because its training path already performs explicit volume normalization. Whisper/ASR is tracked separately in the OmniASR section above.
- Code fix:
  - CSM HF scripts now keep raw Adja audio arrays, scale PCM-like values before processing, resample explicitly to 24 kHz, and pass normalized waveforms into `processor.apply_chat_template`.
  - Orpheus scripts now normalize PCM-like values before SNAC encoding.
  - SageMaker CF1/CF2/CF3 now avoid recasting Adja to `Audio(decode=False)` before normalization and normalize both direct-array and decoded-byte paths.
- HF Jobs submitted with fresh output prefixes so April results remain untouched:

| Exp ID / prefix | Scope | Job | Status note |
|-----------------|-------|-----|-------------|
| `T1_AUDIOFIX_20260509` | CSM direct Adja LoRA | `69ff7b51317220dbbd1a7251` | running |
| `T1_full_ft_AUDIOFIX_20260509` | CSM direct Adja full fine-tune | `69ff7be4317220dbbd1a7265` | submitted |
| `T1_csm_tokfix_AUDIOFIX_20260509` | CSM tokenizer-expanded direct Adja | `69ff7b52aff1cd33e8f32156` | failed fast: missing `torchaudio`; superseded |
| `T1_csm_tokfix_AUDIOFIX2_20260509` | CSM tokenizer-expanded direct Adja | `69ff7d15aff1cd33e8f32171` | replacement running after dependency fix |
| `T1_csm_ewe_adja_stage2_AUDIOFIX_20260509` | CSM Ewe Stage 1 → Adja Stage 2 | `69ff7b50317220dbbd1a724d` | failed fast: missing `torchaudio`; superseded |
| `T1_csm_ewe_adja_stage2_AUDIOFIX2_20260509` | CSM Ewe Stage 1 → Adja Stage 2 | `69ff7d1daff1cd33e8f32173` | replacement running after dependency fix |
| `T2_orpheus_en_lora_r32_AUDIOFIX_20260509` | Orpheus EN direct Adja LoRA r=32 | `69ff7bc6aff1cd33e8f32162` | running |
| `T2_orpheus_en_lora_r64_AUDIOFIX_20260509` | Orpheus EN direct Adja LoRA r=64 | `69ff7b51aff1cd33e8f3214e` | running |
| `T2_orpheus_en_lora_r128_AUDIOFIX_20260509` | Orpheus EN direct Adja LoRA r=128 | `69ff7bd5aff1cd33e8f32164` | running |
| `T2_orpheus_en_fullft_AUDIOFIX_20260509` | Orpheus EN direct Adja full fine-tune | `69ff7bd5317220dbbd1a7263` | submitted |
| `T2_orpheus_fr_lora_r64_AUDIOFIX_20260509` | Orpheus FR direct Adja LoRA r=64 | `69ff7bd5aff1cd33e8f32166` | running |
| `T2_orpheus_tokfix_AUDIOFIX_20260509` | Orpheus tokenizer-expanded direct Adja | `69ff7be6317220dbbd1a7269` | running |
| `T2_orpheus_zh_AUDIOFIX_20260509` | Orpheus ZH direct Adja | `69ff7be5317220dbbd1a7267` | running |
| `T2_orpheus_en_ewe_adja_stage2_AUDIOFIX_20260509` | Orpheus EN Ewe Stage 1 → Adja Stage 2 | `69ff7b51317220dbbd1a724f` | running |
| `T2_orpheus_en_ewe_adja_stage2_mixed_AUDIOFIX_20260509` | Orpheus EN mixed Ewe+Adja Stage 2 | `69ff7b50317220dbbd1a724b` | running |
| `T2_orpheus_fr_ewe_adja_stage2_AUDIOFIX_20260509` | Orpheus FR Ewe Stage 1 → Adja Stage 2 | `69ff7bf4317220dbbd1a726b` | running |
| `T2_orpheus_zh_ewe_adja_stage2_AUDIOFIX_20260509` | Orpheus ZH Ewe Stage 1 → Adja Stage 2 | `69ff7bf4aff1cd33e8f32168` | running; depends on ZH Stage 1 checkpoint availability |

- SageMaker CF reruns submitted on the original SageMaker path:

| Exp ID / prefix | Job name | Instance | Status note |
|-----------------|----------|----------|-------------|
| `CF1_ewc_stage2_AUDIOFIX_20260509` | `adja-cf1-audiofix-full-20260509-142331` | `ml.g5.2xlarge` | in training |
| `CF2_curriculum_stage2_AUDIOFIX_20260509` | `adja-cf2-audiofix-full-20260509-142331` | `ml.g5.2xlarge` | in training |
| `CF3_frozen_backbone_stage2_AUDIOFIX_20260509` | `adja-cf3-audiofix-full-20260509-142331` | `ml.g5.2xlarge` | in training |

## 2026-05-07 — P1 extended testing: Mode C implementation + cultural Q&A battery

### Mode C: French audio input (new feature, --input-lang fr)
- Implemented `FrenchASR` in `stages/asr.py` using `openai/whisper-small` with `language=fr`
- Added `--input-lang fr` flag and `run_french_audio_sample()` in `pipeline.py`
- Added `asr_fr` config block in `p1_config.yaml`
- French audio generated with `gTTS` for testing (free, no API key)
- Route: French WAV → Whisper-small → LLM → NLLB reverse → Spark TTS

### Test battery — Mode B (French text) and Mode C (French audio)

All questions answered in Adja audio. Representative results:

| Q | LLM Answer (FR) | Adja output |
|---|---|---|
| Quelle est la capitale du Bénin ? | La capitale du Bénin est Porto-Novo. Cependant, Cotonou abrite le siège du gouvernement... | Benin ƒe dugã enye Porto-Novo. Ke hã, Cotonou, si nye du si me dziɖuɖu le la... |
| Combien parlent adja dans le monde ? | L'adja est parlé par environ 1 à 2 millions de personnes au Bénin et au Togo. | Amegbetɔ miliɔn ɖeka alo eve sɔŋ ye doa gbe adja, eye wo dometɔ akpa gãtɔ le Benin kple Togo. |
| Mode C: capitale (French audio) | La capitale du Bénin est Porto-Novo. | Benin ƒe dugã enye Porto-Novo. |
| Mode C: nourriture (French audio) | Le plat traditionnel des Adja... est le "djenkoumé", pâte de maïs avec sauce tomate | Adja la ƒe nunyiamee nye "jenkumé", si nye mɔ aɖe si dzi woɖuna... |

Note: Mode C French ASR correctly transcribed "Quel est la capitale du bénin?" from gTTS audio.

### Bugs fixed
- `no_repeat_ngram_size=3, repetition_penalty=1.2` added to NLLBTranslator (repetition loop on >50-char inputs)
- Spark handler.py updated to return base64 JSON instead of raw bytes (400 error from HF toolkit)
- einx>=0.3.0 added to Spark endpoint requirements (ModuleNotFoundError)
- audio_tokenizer.py patched to fall back to Hub for wav2vec2 weights when LFS stub detected

---

## 2026-05-07 — P1 full pipeline: all 4 modes end-to-end with live Spark TTS endpoint

### Spark TTS endpoint deployment
- Published `JosueG/spark-tts-adja-t3` to Hub via `publish_spark_t3.py --private` (3.94 GB upload)
- 3 deployment failures resolved in sequence:
  1. `text-to-speech` is not a valid HF API task → fixed: use `--task custom`
  2. `ModuleNotFoundError: einx` → fixed: added `einx>=0.3.0` to requirements.txt
  3. Git LFS stub for `wav2vec2-large-xlsr-53/pytorch_model.bin` → fixed: audio_tokenizer.py
     falls back to `facebook/wav2vec2-large-xlsr-53` from Hub when file size < 1 MB
  4. `Type is not JSON serializable: bytes` (400 error) → fixed: handler returns
     `{"audio": base64, "sample_rate": 16000}`; EndpointTTS decodes base64 response
- Final endpoint: `https://sn1m01ssi8x1u8mv.us-east-1.aws.endpoints.huggingface.cloud`
  (nvidia-l4 x1, us-east-1, task=custom, scale-to-zero)

### Bug found and fixed: NLLB repetition loop
- Symptom: "Doloe nye nunyiame si wonyãna le gbea gbea gbea..." on >50-char inputs
- Root cause: NLLB-600M beam search without repetition penalty enters token loops
- Fix: `no_repeat_ngram_size=3, repetition_penalty=1.2` in NLLBTranslator.generate()
- After fix: "Doloe nye nunyiame si wonyãna tso ale si wodoa gbe ɖae la me." ✅

### Mode B roundtrip — `runs/mode_b_roundtrip_ngrep_20260507/`
- Input FR: "Le dolo est une boisson fermentée traditionnelle préparée à partir de mil."
- MT Adja: "Doloe nye nunyiame si wonyãna tso ale si wodoa gbe ɖae la me."
- TTS: 6.6s WAV (211 KB) at 16 kHz
- RT path: FR text → NLLB reverse → Spark TTS endpoint

### Mode B QA — `runs/mode_b_qa_20260507/`
- Input FR: "Quel est le plat traditionnel adja ?"
- OpenRouter GPT-4o response: "Le plat traditionnel des Adja... est le 'amiwo' ou 'ékpessi'..."
- MT Adja: "Adja, si nye anyiehe Bénin kple Togo tɔ, ƒe nunyiamee nye 'amiwo' alo 'ekpessi'..."
- TTS: 30.9s WAV (989 KB)
- RT path: FR text → OpenRouter → NLLB reverse → Spark TTS endpoint
- MT Back latency: 145s (CPU; would be ~15s on GPU)

### Mode A roundtrip — `runs/mode_a_roundtrip_20260507/`
- Input: `test_00000.wav` (ref: "Ŋu nya kpɔ́kpɔ a ?", 2.7s, 48kHz)
- ASR xlsr: "M i a kpɔ w a e tu e" — CER 87.5%
- MT fwd FR: "Je t'ai vu et j'ai fini"
- MT rev Adja: "Mekpɔ wò, eye nuwuwua wu enu."
- TTS: 2.4s WAV (77 KB)
- First full end-to-end Mode A run ✅

### Mode A QA — `runs/mode_a_qa_20260507/`
- Input: `test_00001.wav` (ref: "ŋnyan go. Kpɔ ŋuɖejikɔ a nyan wo", 5.3s, 48kHz)
- ASR xlsr: "M pɔ u n a v ɔ lɔ xu lɔ k u ɖe" — CER 103.2%
- MT fwd FR: highly garbled text
- OpenRouter response: "Désolé, je ne comprends pas ce que vous avez dit..." (LLM detected bad input!)
- MT rev Adja: "Meɖe kuku, nyemese nu si gblɔm..." (Adja apology for not understanding)
- TTS: audio generated — pipeline gracefully handles bad ASR via LLM gatekeeper
- Note: pipeline degraded gracefully when ASR quality was too low for LLM to understand

### Key metrics from 2026-05-07 runs
| Run | Mode | Stages | Status | Notable |
|-----|------|--------|--------|---------|
| mode_b_roundtrip_ngrep | B roundtrip | MT rev + TTS | ✅ | Clean MT after fix |
| mode_b_qa | B qa | OpenRouter + MT rev + TTS | ✅ | 30.9s Adja response |
| mode_a_roundtrip | A roundtrip | ASR + MT fwd + MT rev + TTS | ✅ | Full chain works |
| mode_a_qa | A qa | ASR + MT fwd + OpenRouter + MT rev + TTS | ✅ | LLM handles bad ASR |

### ASR quality on real speech (endpoint vs. published test-set CER)
- Published c4v2 test-set CER: 25.05%
- Observed on today's clips: 68–103% CER
- Likely causes: (1) 48kHz clips resampled to 16kHz for endpoint (may add artifacts);
  (2) test_00000/01 are different speakers/domains than training data;
  (3) endpoint greedy CTC vs. training-time eval may differ slightly

---

## 2026-05-06 — P1 cascade pipeline smoke and reporting pass

- **Code path:** `experiments/s2tt/P1_cascade_pipeline/`
- **Main implementation changes:**
  - `stages/mt.py` now loads two directional NLLB checkpoints: Adja->FR forward and FR->Adja reverse.
  - `configs/p1_config.yaml` now points at the two live MT checkpoints and the two deployed ASR endpoint URLs.
  - `pipeline.py` now supports `--input-manifest`, per-sample manifest references, `.env` loading, batch error capture, and `aggregate_summary.json`.
  - `stages/asr.py` local XLS-R loading now falls back to standard Hub `vocab.json` when `ctc_vocab.json` is absent.
  - `eval/write_pipeline_report.py` writes thesis-facing Markdown from pipeline run artifacts.
  - `scripts/hub/publish_spark_t3.py` now defaults to the canonical Spark source run `T3` and endpoint target `JosueG/spark-tts-adja-t3`.
- **Dry-run validation:** full 160-row manifest dry run completed under `/tmp/p1_dry_manifest/`; no live models or GPU required.
- **Live ASR endpoint deployment:**
  - XLS-R endpoint: `https://nk5kx1pt7b7d0087.us-east-1.aws.endpoints.huggingface.cloud`
  - Whisper endpoint: `https://pwrgcv1ycb8hfalv.us-east-1.aws.endpoints.huggingface.cloud`
  - Single endpoint smoke on `test_00000.wav`: primary transcript `M i a kpɔ w a e tu e`, CER 87.5% against `Ŋu nya kpɔ́kpɔ a ?`.
- **Local ASR->MT smoke:** `experiments/s2tt/P1_cascade_pipeline/runs/p1_local_asr_mt3_20260506/`
  - 3 samples, 0 failures.
  - Mean primary ASR CER: 68.45%; mean ASR latency: 11.31s; mean MT forward latency: 8.55s.
  - Sample decodes:
    - `test_00000`: ref `Ŋu nya kpɔ́kpɔ a ?` -> ASR `Na kpɔ woa u?` -> FR `Viens les voir?`
    - `test_00001`: ref `ŋnyan go. Kpɔ ŋuɖejikɔ a nyan wo` -> ASR `Nɔuɛavilɔxulɔ deii lɔó` -> FR `Il y a aussi une bonne mère qui aide les enfants.`
    - `test_00002`: ref `Ŋ nyanyɔ mɔ wo anu ahán !` -> ASR `Nɖayeabwa nɛ ahantu` -> FR `Je lui donnerai de l'huile`
- **Mode B response->reverse-MT smoke:** `experiments/s2tt/P1_cascade_pipeline/runs/p1_mode_b_response_mt_20260506/`
  - Input: `Quel est le plat traditionnel adja ?`
  - French response: `Le plat traditionnel adja est le "âmegbôwé", ou "amiwo"...`
  - Adja reverse-MT output: `Amegbôwé, alo amiwo, si nye nuɖuɖu si wodoa mɔna ɖe tɔmelã alo lã ŋu la, nye nuɖuɖu xɔŋkɔ aɖe si wodona le Adja nutoa me le Benin kple Togo.`
- **Spark TTS source discovery:**
  - Canonical Spark source adapter exists at `JosueG/adja-tts-results/T3/adapter` in the authenticated Hub model repo.
  - The deployable local artifact exists at `/tmp/spark-publish-T3` (2.5 GB) with `LLM/`, `BiCodec/`, `wav2vec2-large-xlsr-53/`, `handler.py`, and copied Spark code.
  - Pipeline local-mode checkpoint now points at the real source adapter: `JosueG/adja-tts-results/T3/adapter`.
- **Thesis/report artifact:** `results/pipeline-comparison.md`
  - Includes system schema, output-boundary schema, Mermaid metric/latency charts, deployment notes, blockers, and qualitative examples.
- **Blocked items:**
  - Spark TTS endpoint was not completed because Codex blocked the external upload of `/tmp/spark-publish-T3` to Hugging Face even after user approval. The ready commands are `python scripts/hub/publish_spark_t3.py --private` and then `python scripts/hub/manage_endpoints.py up JosueG/spark-tts-adja-t3 --task text-to-speech --gpu`.
  - Full endpoint batch was not run because sending the full local manifest audio to third-party endpoints needs explicit approval; local CPU path works but is not practical for all 160 samples.

## 2026-05-03 — Qualitative ASR decodes: regen utility shipped + 20-sample paper table generated

### Summary
Added [`scripts/qualitative/regen_asr_decodes.py`](../scripts/qualitative/regen_asr_decodes.py) and ran the canonical 20-sample regen end-to-end against both deployed Adja ASR endpoints. The driver pulls a deterministic seed=42 slice of `JosueG/adja-tts-orpheus`'s test split (matches `experiments/asr/shared/data_prep.py:create_splits`), transcribes each clip via the dedicated HF Inference Endpoint URLs, scores with the project's NFC-safe `compute_wer`/`compute_cer`, and emits both a structured JSON and a paper-ready Markdown table to `results/qualitative/`.

### Run details
- **Endpoints:** spun up via `scripts/hub/manage_endpoints.py up` on intel-icl×2 CPUs.
  - C4v2: `https://lt1vri9jiv7tl5a3.us-east-1.aws.endpoints.huggingface.cloud`
  - E4v4: `https://czb0f91gbsa0pcy0.us-east-1.aws.endpoints.huggingface.cloud`
  - Torn down after run completed (~6 min total uptime; both deleted, not just scaled-to-zero).
- **Command:** `arch -arm64 python3 scripts/qualitative/regen_asr_decodes.py --n 20 --asrs '<C4V2>:c4v2,<E4V4>:e4v4' --tag 2026-05-03`
- **Outputs:**
  - `results/qualitative/asr_decodes_2026-05-03.json` (21 KB, full per-sample payload)
  - `results/qualitative/asr_decodes_2026-05-03.md` (4.8 KB, paper-ready table)
  - `results/qualitative/.cache/{c4v2,e4v4}/*.json` (40 cached endpoint responses)

### Aggregate (n=20, normalized)
| ASR | WER (corpus) | CER (corpus) | Errors |
|---|---|---|---|
| `c4v2` | 108.20 % | 67.53 % | 0 |
| `e4v4` | 154.10 % | 393.91 % | 0 |

### Caveats — DO NOT cite these aggregates as the paper headline numbers
1. **E4v4's 393.91 % CER is dominated by two Whisper hallucination loops.** Sample 10 emitted ~600 literal `A`s; sample 16 emitted a repeating `ɖɔ wɔ ɖɔ wɔ` cycle. Both inflate the corpus aggregate (sum-of-edits / sum-of-refs) by ~10×. The median per-utterance CER is a fairer summary; for paper reporting, we should either (a) report median, (b) drop hallucination outliers and label them separately, or (c) cite the published 37.18 % from the model card and note that hallucination is a known E4v4 failure mode (see `docs/whisper-training-gotchas.md`).
2. **C4v2's 67.53 % CER is ~2.7× the published 25.05 %** for the same model on what should be the same test split. Two suspects worth checking before this number ships: (i) the rebuilt tokenizer (vocab reconstructed at endpoint deploy time per the audit block in `experiments/registry.md`) may behave differently from the in-training vocab; (ii) the deployed `automatic-speech-recognition` pipeline does greedy CTC decoding without the chunk_length_s tuning the training-time eval used. Worth a 5-sample diff with the local backend (`--backend local` against the same hub repo) before claiming a regression.
3. The split logic mirrors `data_prep.py` (single 90/10 with seed=42 → take `["test"]`), but the original C4v2 training run may have used a 3-way split where the test slice is a different 10 % of the data. Verify by comparing manifest hashes if a discrepancy investigation is warranted.

### Per-sample highlights (for the paper)
Cleanly intelligible C4v2 decode, low-CER:
- `REF: Ganɛni mɛ ayi ɔ̀ ?` → `HYP: Manɛni mɛ ai ɛx` (CER 25.00 %)
- `REF: Mi gbe kpɔtɔ le xuemɛ` → `HYP: Mibekpɔtɔ loxo enɔ` (CER 42.86 %)

Cleanly intelligible E4v4 decode, low-CER:
- `REF: Mi gbe kpɔtɔ le xuemɛ` → `HYP: Ɖegbe kpɔtɔ le kpɔɖo` (CER 38.10 %)
- `REF: Eha jiji cɛ ɖonɔŋwi nyɔlunvi ɖeka wo nɔŋu` → `HYP: Fɛha jiji cɛ ɖe nu ŋu, ŋu nɔ nuji ɖeka wo` (CER 48.78 %)

These are exactly the kind of rows the thesis Chapter 4 sample table needs — Adja-script out, recognizable phonological alignment with the reference, no `<pad>` literal pollution.

### Pinned reproducibility command (paper)
```bash
python scripts/qualitative/regen_asr_decodes.py \
  --n 20 \
  --asrs '<C4V2_ENDPOINT_URL>:c4v2,<E4V4_ENDPOINT_URL>:e4v4' \
  --tag 2026-05-03
```
Also available: `--dry-run` (1 sample, 1 ASR) and `--backend local` (CPU fallback, no endpoints needed).

### Tracking docs updated
- `experiments/registry.md` — note added in the deployable-checkpoint audit block linking to the regen utility.
- `docs/asr-reproducibility-2026-05-02.md` — paper-prep doc covering scope, protocol, output schema, floors and caveats.
- `results/comparison.md` — **not** updated. The corpus aggregates from this regen disagree with the published per-model numbers (see caveats above), so the leaderboard should keep citing the model-card numbers until the C4v2 gap is investigated.

### Local infra note (one-off)
The user's macOS Python 3.12 site-packages had x86_64 numpy/pandas wheels under arm64 hardware, which broke `import datasets`. Resolution: `arch -arm64 pip3 install --force-reinstall --no-deps numpy pandas pyarrow`. Subsequent runs use `arch -arm64 python3 …`. Project-side, no requirements.txt change is needed — this is a workstation pip-cache hygiene issue, not a project-dep regression.

---

## 2026-04-28 ~13:00 EDT — M2 / M2b v4 final results: loss balance fixed, codec FT degraded reconstruction (negative result)

### Summary

After applying `--sem-weight=1e-9` (gotcha #21 family) and dual-mirror best-checkpoint (gotcha #22), M2 and M2b completed cleanly. Both pushed to HF model repos via EXTRACT_M2 / EXTRACT_M2B.

| Run | Job | Wall | best `recon` (L1+STFT) | raw `sem` | `weighted_sem` | vs M0 baseline (0.55) |
|-----|-----|------|-----|-----|-----|-----|
| M2 v4  | `adja-m2-full-20260428-081616`  | 19.1 min | **1.30304** @ ep10 | 1.05e8 | ~0.105 | **+137% WORSE** |
| M2b v4 | `adja-m2b-full-20260428-081625` | 28.4 min | **1.28283** @ ep10 | 1.06e8 | ~0.106 | **+133% WORSE** |

### What this confirms

1. **Loss balance is correct now.** Per-step logs show `recon ≈ 1.0–1.5` and `weighted_sem ≈ 0.05–0.4` — same OOM, both terms get gradients. The M2/M2b v2 catastrophic-imbalance bug (sem ≈ 10⁸ swamping recon ≈ 1) is gone. The negative result is real, not a math error.
2. **Codec FT degrades reconstruction even with correct balance.** ~3 hours of training data (1.2k Ewe + 1.6k Adja TTS clips) is not enough to budge a Mimi codec pretrained on orders-of-magnitude more multilingual audio. The semantic-distillation pull moved the encoder, but not toward better waveform fidelity.
3. **M2b > M2 by a hair (1.28 vs 1.30).** Bigger Adja-aware teacher (mms-1b-all + Adja adapter) gives marginally better supervision than mms-300m, but the effect is dwarfed by the regression from M0. A larger teacher does not rescue the data-volume problem.
4. **Strengthens paper-1's "priors > capacity" thesis.** Adding capacity (semantic teacher) and a learning signal aimed at phonemic identity did not recover Gbe-specific quality at this data scale.

### What this does NOT prove yet

- L1 is not the user-facing metric. M0 already sounded "noisy but intelligible Adja" to the native ear (2026-04-28 listening verdict). A 2.3× L1 regression may or may not cross the perceptual threshold. **Need M2/M2b reconstruction WAVs and listening verdict** before declaring the codec-FT track dead.
- Submitted **M0_FROM_M2** (`adja-m0-from-m2-full-20260428-125936`) and **M0_FROM_M2B** (`adja-m0-from-m2b-full-20260428-125938`) on g5.xlarge — these run M0_mimi_reconstruction_test.py with `--mimi-repo` pointing at the two FT codecs on HF, produce 30 reconstructed WAVs each. After they complete, EXTRACT_M0_FROM_M2 / EXTRACT_M0_FROM_M2B will push the WAVs to `JosueG/adja-tts-results/M0_mimi_reconstruction_adja_FROM_M2(B)` for A/B listening.

### Current best codec for downstream cascade
**Off-the-shelf `kyutai/mimi`** — pending listening verdict on FT variants, the M0 baseline (L1=0.55) is still the best codec we have for AT1 / CSM Stage 1 warmstart. Updated registry to reflect this.

### Files

- HF model repos:
  - https://huggingface.co/JosueG/mimi-adja-m2-mms300m-semweight-1e-9
  - https://huggingface.co/JosueG/mimi-adja-m2b-mms1b-adja-semweight-1e-9
- S3 outputs:
  - `s3://sagemaker-us-west-2-974640818655/adja-m2-full-20260428-081616/output/model.tar.gz`
  - `s3://sagemaker-us-west-2-974640818655/adja-m2b-full-20260428-081625/output/model.tar.gz`

### Open: M1 v2 still failing

`adja-m1-smoke-20260428-104045` failed on the v2 fix. Need to investigate separately — different failure mode than the OneCycleLR off-by-one (which is patched). Not blocking the M2/M2b verdict.

---

## 2026-04-27 — Wave 2: resubmit AT1/S2/S6 + new S1B/T3A/T3C/EXTRACT_T3A

### Summary

Full resubmission wave after diagnosing multiple compounding failures in the 2026-04-26 wave:

| Job | Issue | Fix | New Job ID |
|-----|-------|-----|------------|
| AT1 | 30 GB EBS (disk-full) → WaxalNLP CastError (old code deployed before fix) | 200 GB EBS + two-group schema peek | adja-at1-full-20260427-104104 |
| S2 | 30 GB EBS (would hang on WaxalNLP unlabeled) | 200 GB EBS + two-group schema fix | adja-s2-full-20260427-100855 |
| S6 | 30 GB EBS (same risk) | 200 GB EBS + pre-pad fix for DataCollator | adja-s6-full-20260427-100901 |
| S1B | OOM (FP32 AdamW on 1.5B = 12 GB optimizer states) | `--adam-8bit` (bitsandbytes AdamW8bit → ~3 GB) + `use_reentrant=False` | adja-s1b-full-20260427-104124 |
| T3C | OOM (curriculum Ewe+Adja seqs at max_seq_length=2048) | `--max-seq-length 1024` (clips >20 s dropped, ~0 in WaxalNLP TTS) | adja-t3c-full-20260427-104138 |
| T3A | Completed successfully (output.tar.gz in S3) | → launch EXTRACT_T3A to push audio to HF Hub | adja-extract-t3a-full-20260427-104049 |

All new jobs use 200 GB EBS, on-demand (no spot — quota 0 for g5.2xlarge spot).

### Root causes (multi-level, why prior sessions were insufficient)

1. **WaxalNLP `CastError`**: datasets 2.18.x requires EXACT bidirectional schema match. `__index_level_0__` is inconsistently present (6/9 train, 104/106 unlabeled). Fixed: peek schema per shard with `pq.read_schema()`, group by variant (with/without index col), load each group with its exact `features=`, concatenate.
2. **Two-location CVE monkeypatch**: `transformers.utils.import_utils.check_torch_load_is_safe` was patched but `transformers.modeling_utils` had its own local binding to the original — must patch both.
3. **S1B `--adam-8bit` not deployed**: old code (submitted before flag was added) ran OOM. Now confirmed in HP dict: `adam-8bit=""`.
4. **T3C `max-seq-length=2048`** with curriculum mixing fills GPU at batch_size=1. Reduced to 1024.
5. **WaxalNLP S3 channel safety**: check now requires sentinel `ewe-unlabeled-00105.parquet` (last shard) — partial upload would silently provide fewer shards than HF Hub.

### S3 channel status

- Spark-TTS: fully synced (`repos/spark-tts/`, 96 objects). T3A/T3C use it. ✓
- WaxalNLP: 19/161 ewe files uploaded; resuming in background. Jobs use HF Hub fallback until complete.

---

## 2026-04-27 evening — S1B completes: negative-result confirms LoRA > full FT for low-resource Whisper

**Job:** `adja-s1b-full-20260427-190549` | ml.g6e.xlarge (1× L40S 48 GB) | **Completed** in 172 min, early-stopped at epoch 17/30 (no improvement vs best at epoch 12). Peak VRAM 23.58 GB (fits comfortably in 48 GB — OOM that killed prior S1B attempts on g5.2xlarge 24 GB is solved).

**Setup:** Whisper-large-v3 **full fine-tune** (no LoRA), **direct Adja only** (no Ewe Stage A — that's S1's design, not S1B's). `--adam-8bit`, `--gradient-checkpointing` with `use_reentrant=False`, `bs=1`, `grad-accum=16`, `lr=1e-5`. Used `train_S7_whisper_tiny_adja.py` with `--base openai/whisper-large-v3` (hence `[S7]` log prefix despite running large-v3).

**Result — bad, but valuable as a negative control:**

| Metric | Value | Compare |
|---|---|---|
| Test WER | **101%** (1.01) | E4 prior best: 73.09% |
| Test CER | **54.71%** | E4 prior best: 24.90% (**2.2× worse**) |
| Best dev CER | 52.03% @ epoch 12 | — |
| Train loss | 3.486 → 0.0082 (**425×** drop, classic memorization) | — |
| Dev CER trajectory | chaotic 52–119% across 17 epochs | — |

**What went wrong (intentional ablation):**
- **Catastrophic forgetting.** Full FT on 1.5B params + ~1.7 h Adja wipes Whisper-LV3's multilingual prior. Train loss collapses to ~0 (memorization) while dev metrics oscillate.
- **Whisper hallucination.** Decode samples show classic Whisper-large-v3 hallucination signature — English/spurious tokens leak into Adja output, e.g. `REF "ŋɖuɖu lɔwo nuɔn"` → `HYP "ŋ by by lɔwo nɔ̀"` at epoch 15. Full FT on tiny data exacerbates the known Whisper hallucination quirk.

**Thesis interpretation (the actually-useful framing):**

S1B is the negative-control ablation that strengthens the LoRA hypothesis. The pattern parallels [`research-paper-exploration/paper-one/why-spark-worked.md`](../research-paper-exploration/paper-one/why-spark-worked.md): *"Capacity is not the ceiling; priors are."* — full FT didn't help CSM Stage 2, doesn't help Whisper-LV3 ASR either. **LoRA's parameter-efficient regularization (S1: 28 M trainable / 1.5B = 1.78%) is what makes 1.5B-class models trainable on low-resource speech data without catastrophic forgetting.**

Suggested paper sentence: *"Full fine-tuning of Whisper-large-v3 on 1.7 h Adja produced WER 101%, CER 54.71% (2.2× worse than the prior best Whisper-Ewe-warmstart LoRA baseline E4: WER 73.09%, CER 24.90%), confirming that LoRA's parameter-efficient regularization is necessary, not optional, for low-resource adaptation of large pretrained ASR models."*

Artifact: `s3://sagemaker-us-west-2-974640818655/adja-s1b-full-20260427-190549/output/model.tar.gz` (5.7 GB; kept as the cited negative-result artifact).

---

## 2026-04-28 ~07:30 EDT — Morning recovery batch: M0 audio on HF, M1 lost, M2/M2b done-but-broken, S1 LoRA pushing

Overnight summary across the M-series + S1-091126:

### What completed cleanly

- **`adja-m0-full-20260428-004538`** (M0 v3) — Completed. `output.tar.gz` (6.6 MB) with 60 WAVs uploaded to S3. Reconstruction baseline confirmed: **mean L1=0.55, mean SC=0.4425 (PASS, mid acceptable band)**.
- **`adja-extract-m0-full-20260428-073143`** (EXTRACT_M0 v3) — Completed. **60 M0 WAVs pushed to HF Hub** at https://huggingface.co/datasets/JosueG/adja-tts-results/tree/main/M0_mimi_reconstruction_adja_v3/generated_audio. Native-speaker A/B listening can begin from there.
- **`adja-m2-full-20260428-004548`** (M2 v2) — Completed in 18.2 min, 7010 steps × 10 epochs. CVE-2025-32434 monkeypatch worked ✓.
- **`adja-m2b-full-20260428-004558`** (M2b v2) — Completed in 28.0 min, 13280 steps × 10 epochs. CVE patch worked ✓.
- **`adja-s1-full-20260427-091126`** (Whisper-LV3 LoRA Stage A) — `Stopped` via `MaxRuntimeExceeded` at 06:14 EDT after 21 h. **49 MB `model.tar.gz` uploaded.** **Best Ewe CER=17.80%, WER=51.86% at epoch 16; final saved adapter is epoch 18** (CER=18.33%, regression — see gotcha below).
- **`adja-extract-s1-lora-full-20260428-073658`** (EXTRACT_S1_LORA) — currently `InProgress` (just submitted ~10 min ago). Pushing to **`JosueG/whisper-large-v3-ewe-lora-s1-091126`** (HF *model* repo, new pattern via [scripts/sagemaker_jobs/extract_and_push_model.py](../scripts/sagemaker_jobs/extract_and_push_model.py) — adapted from the audio extractor with `repo_type="model"` and a generated model card).

### What FAILED — and what each failure actually means

**M1 (`adja-m1-full-20260428-003246`)** — failed at the very last training step with `ValueError: Tried to step 3511 times. The specified number of total steps is 3510`. Off-by-one in `total_steps` calc: [train_M1_mimi_acoustic.py:216](../scripts/sagemaker_jobs/train_M1_mimi_acoustic.py:216) uses `len(combined) // args.mimi_batch * args.mimi_epochs` (floor division) but the inner loop runs `math.ceil(len/batch)` iterations. With `len(combined) % args.mimi_batch != 0`, `OneCycleLR` runs out of schedule one step early. **No checkpoint was saved** — the save (`mimi.save_pretrained`) at [line 323-328](../scripts/sagemaker_jobs/train_M1_mimi_acoustic.py:323) runs AFTER the training loop returns; the crash inside the loop short-circuits everything. **M1 needs the 1-line fix (`math.ceil` instead of `//`) and a resubmit (~30 h again).** Documented as gotcha #21.

**M2 / M2b — completed but loss math is broken.** Reconstruction L1 went UP, not DOWN, vs M0 baseline:

| Run | Mean L1 | vs M0 baseline (0.55) |
|-----|---------|------------------------|
| M0 (off-the-shelf Mimi) | **0.55** | — |
| M2 final epoch | 1.228 | **+123% WORSE** |
| M2b final epoch | 1.325 | **+141% WORSE** |

Cause: the semantic-distillation loss term is on a totally different scale (≈10⁸) than the reconstruction term (≈1). The optimizer effectively ignored reconstruction. **Both M2 and M2b need a sem-loss weight rescale** (current implicit weight ≈ 1e-7 isn't small enough — likely needs normalization or explicit re-weighting around 1e-9 to 1e-10). Until then, neither M2 nor M2b can be claimed as a working Mimi-FT codec. The CVE patch fix succeeded; the loss-balancing is a separate bug.

### S2 / S3 also failed overnight (carryover from wave-3 v2)

`adja-s2-full-20260427-191943` and `adja-s3-full-20260427-191954` both failed in the same place as their predecessors despite the **32000-sample minimum filter**: `RuntimeError` in `wav2vec2/modeling_wav2vec2.py:1543` `compute_contrastive_logits → torch.cosine_similarity`. Same wav2vec2 contrastive `index out of bounds`. The 32000 filter floor was supposed to fix this. **The min-length theory is wrong** — needs deeper diagnosis (possibly mask-coverage edge case, or DDP-specific batch padding mismatch, or NaN propagation). For now S2/S3 are stuck.

### Surfaced gotchas (added to learnings-from-the-past/sagemaker-gotchas.md)

- **#21** — `OneCycleLR.total_steps` must use `math.ceil(len/batch)`, not floor division. Floor undercounts by one whenever `len % batch != 0`; the crash happens AFTER all real training steps but BEFORE the post-loop save → loses the entire run. Audit affects M1, AT1, S6 — anywhere using OneCycleLR.
- **#22** — For LoRA jobs hitting `MaxRuntimeExceeded`, the saved `model.tar.gz` contains the **last epoch** state, not the best-CER epoch. S1's epoch-18 saved adapter (CER 18.33%) is *worse* than the best seen at epoch 16 (CER 17.80%). For LoRA add `save_total_limit=1 + load_best_model_at_end=True` or persist intermediates via `checkpoint_s3_uri`.

### Currently still training (longest jobs)

- `adja-at1-full-20260427-201244` (CSM Stage A, ~10 h in)
- `adja-s6-full-20260427-190527` (Orpheus 3B audio-LM, ~12 h in)
- `adja-s1-full-20260427-192235` (Whisper-LV3 LoRA fresh on g6e.xlarge, ~12 h in)
- `adja-extract-s1-lora-full-20260428-073658` (S1 LoRA → HF model repo, just submitted)

---

## 2026-04-28 ~00:45 EDT — M-series late-night: M0 PASS verdict captured; M2/EXTRACT_M0/M2b recovery wave

**M0 v2 succeeded numerically but lost its WAV artifacts.** `adja-m0-full-20260428-001339` ran clean and reported `MEAN spectral convergence = 0.4425, Verdict: PASS` (mid "acceptable" band; SC < 0.60). However the script's local-Mac cleanup block ran `shutil.rmtree(out_dir)` on `/opt/ml/output/data/mimi_recon_test/` — wiping the 60 WAVs BEFORE SageMaker's post-exit upload to S3. `output.tar.gz` was missing/empty; `EXTRACT_M0` failed `botocore HeadObject 404`. **The verdict number is preserved in CloudWatch logs but the audio artifacts for native-speaker A/B listening are lost.** Fixed in [M0_mimi_reconstruction_test.py:217](../scripts/sagemaker_jobs/M0_mimi_reconstruction_test.py:217) — cleanup now guards on `out_dir.startswith("/tmp/")`. Documented as gotcha #19. Resubmitted as `adja-m0-full-20260428-004538`.

**M2 failed with CVE-2025-32434 / `check_torch_load_is_safe`.** `adja-m2-full-20260428-003259` failed at `Wav2Vec2Model.from_pretrained("facebook/mms-300m")` because transformers 4.50+ refuses to load legacy-format checkpoints without torch≥2.6, but the SageMaker DLC has 2.5.1. Same bug we already patched in S2/S3 with a `check_torch_load_is_safe = _noop` monkeypatch. M2 was missing that patch. **Stopped M2b before it hit the same bug** at the analogous MMS-1B-all load. Both scripts now patched (M2 line 244-258, M2b line 191-205) — the patch must touch BOTH `transformers.utils.import_utils` AND `transformers.modeling_utils` (the latter has a local binding to the function at import time). Documented as gotcha #20. Resubmitted as `adja-m2-full-20260428-004548` and `adja-m2b-full-20260428-004558`.

**Net wave state:** M0 v3 + M2 v2 + M2b v2 all submitted on g6e.xlarge (M0 on g5.xlarge). Once M0 v3 finishes (~30 min) we'll resubmit EXTRACT_M0 with the new S3 URI to push WAVs to HF Hub for native-speaker listening.

---

## 2026-04-27 ~20:00 EDT — Wave-3 v2: AT1 host-RAM fix + Mimi precompute groundwork + T3C abandoned

After wave-3's first round shipped, two more failures surfaced and were fixed:

**AT1 host-RAM OOM (4th distinct AT1 failure mode this day).** `adja-at1-full-20260427-192004` failed with `ClientError: Please use an instance type with more memory` at `decode+encode 6255/183920 (3.4%)`. Root: `apply_chat_template` padding `max_length` × `max_audio_sec=30` → 2.88 MB padded audio per clip × 183 k clips ≈ 527 GB on disk + 8 workers × 56 GB COW WaxalNLP source ≈ 192 GB host RAM exhausted. **Fix:** `padding=False` (variable-length, ~1.8 MB/clip avg → 330 GB), `--preprocess-proc` default 8 → 2, AT1 `volume_size_gb` 200 → 500, custom `at1_collate` pads to longest in batch. Resubmitted as `adja-at1-full-20260427-201244`.

**Mimi precompute pipeline built** (mirroring SNAC pattern that already saved S6). New [scripts/sagemaker_jobs/precompute_waxal_mimi_tokens.py](../scripts/sagemaker_jobs/precompute_waxal_mimi_tokens.py) + 4 PRECOMPUTE_MIMI registry entries + `mimi_channel: True` channel-mount logic. Submitted as `adja-precompute-mimi-{0..3}-full-20260427-200835..200906` on `ml.g5.xlarge` × 4 (~$11 total, ~75 min wall clock). **Cache loader on the train-AT1 side is NOT yet implemented** — deferred because deriving CSM's exact `input_ids` structure from cached Mimi codes needs local introspection that won't fit the thesis week. The cache becomes useful when M-series produces a fine-tuned Mimi codec; in the meantime AT1 ships via the `padding=False` route.

**T3C abandoned for thesis.** 11 OOM crashes today across `max_seq_length` ∈ {2048, 1024, 768} on every available instance up to L40S 48 GB. Activation memory is base-model-dominated, not seq-length-dominated; dropping seq length further only loses data. Listed as future work on `g6e.12xlarge` DDP or `p4de.24xlarge`. T3A (Spark direct) stands alone for the Spark Stage 2 paper result; CF1/CF2 cover curriculum within the CSM/Mimi family.

> **Paper-prep doc:** [docs/wave-3-fairness-and-recovery-2026-04-27.md](../docs/wave-3-fairness-and-recovery-2026-04-27.md) updated with §1.4b (host-RAM OOM), §1.4c (Mimi precompute groundwork), and revised T3A↔T3C row.

---

## 2026-04-27 evening — Wave-3: g6e.xlarge migration + CUDA-fork + wav2vec2 short-clip fixes

> **Paper-prep doc:** [docs/wave-3-fairness-and-recovery-2026-04-27.md](../docs/wave-3-fairness-and-recovery-2026-04-27.md) — single source of truth for thesis/paper writing on what changed, why, and which comparisons stay fair.


After diagnosing three new failure modes on top of the morning's volume-size and schema fixes:

**Failure modes diagnosed:**
1. **CUDA-fork deadlock (silent 3 h hang).** `adja-at1-full-20260427-104104` froze at `decode+encode 0/183920` with GPU 0%, CPU 0.7%, GPU memory pinned at 28.5%. Root cause: `model.to("cuda")` was called BEFORE `unlab.map(num_proc=8)` — forking with an initialized CUDA context deadlocks workers silently. Documented as [gotcha #16](../learnings-from-the-past/sagemaker-gotchas.md).
2. **CUDA OOM on 24 GB A10G.** S6 (Orpheus 3B + max_audio_sec=30 + grad-checkpointing), S1B (Whisper-LV3 full FT + adam-8bit at bs=1), T3C (Spark + curriculum at max_seq_length=1024) all exhausted the 24 GB A10G even with 8-bit Adam. Documented as [gotcha #15](../learnings-from-the-past/sagemaker-gotchas.md).
3. **wav2vec2 contrastive `index out of bounds` on short post-conv sequences.** `adja-s2-full-20260427-100855` and `adja-s3-full-20260427-093734` crashed ~30 min into SSL training with CUDA assertion storms in `compute_contrastive_logits`. Root cause: `prep_ssl` only dropped `arr is None` — short clips (after 320× conv subsample → only ~30 features) couldn't supply 100 negative samples cleanly.
4. **Drop-vs-truncate semantic confusion.** AT1 + S6 hard-dropped clips > `max_audio_sec * sr`; with default 10 s and WaxalNLP unlabeled having median 18 s, this dropped 99.7% of training data. Bumped default `max_audio_sec` to 30 s (covers 100% of unlabeled per probe).

**Fixes applied:**
- [scripts/sagemaker_jobs/train_AT1_audio_lm_csm_ewe.py](../scripts/sagemaker_jobs/train_AT1_audio_lm_csm_ewe.py): processor loaded BEFORE `.map(num_proc=8)`; `model.from_pretrained(...).to("cuda")` deferred to AFTER `.map()` finishes. Default `max_audio_sec` 10 → 30; `audio_kwargs.max_length` now derives from `max_audio_sec`.
- [scripts/sagemaker_jobs/train_S2_S3_wav2vec2_ssl_ewe.py](../scripts/sagemaker_jobs/train_S2_S3_wav2vec2_ssl_ewe.py): `prep_ssl` now drops `len(arr) < 32000` (2 s @ 16 kHz, 100 features post-conv) per code-review at 95% confidence (initial 10000 floor was too low). Also deferred `model.to("cuda")` until after `.map()`.
- [scripts/sagemaker_jobs/train_S6_audio_lm_orpheus_ewe.py](../scripts/sagemaker_jobs/train_S6_audio_lm_orpheus_ewe.py): cold-cache SNAC encode path forced to `num_proc=1` (CUDA-fork preventive — workers can't share a CUDA-loaded snac_model anyway). Warm-cache fast path is preferred and unchanged.
- [scripts/sagemaker_jobs/launch.py](../scripts/sagemaker_jobs/launch.py): S6, S1B, T3C, S1 (next clean run) moved to **`ml.g6e.xlarge`** (1× L40S 48 GB) — same single-GPU footprint as g5.2xlarge but 2× VRAM and ~2× faster. est_hours bumped: AT1 28→40, S1 20→40, S1B 24→40, T3C 6→12, S2/S3 36→48, S6 26→36.
- [learnings-from-the-past/sagemaker-gotchas.md](../learnings-from-the-past/sagemaker-gotchas.md): added gotchas #15 (OOM-on-A10G → g6e.xlarge sweet spot) and #16 (CUDA-fork hazard rule).

**Wave-3 jobs submitted (all on-demand; spot quota = 0 for both g5.2xlarge and g6e.xlarge):**

| Job | Instance | Status | Notes |
|-----|----------|--------|-------|
| adja-s6-full-20260427-190527 | ml.g6e.xlarge | Training | Mounts SNAC cache via S3 channel; bypasses cold-encode entirely |
| adja-s1b-full-20260427-190549 | ml.g6e.xlarge | Training | Whisper-LV3 full FT, 48 GB has the headroom |
| adja-t3c-full-20260427-190605 | ml.g6e.xlarge | Training | Spark curriculum, 48 GB fixes the OOM |
| adja-s2-full-20260427-191943 | ml.g5.12xlarge | Just submitted | 32000 floor + CUDA-fork fix; first wave (190627) stopped at user direction after code review flagged 10000 as insufficient |
| adja-s3-full-20260427-191954 | ml.g5.12xlarge | Just submitted | Same as S2 |
| adja-at1-full-20260427-192004 | ml.g5.12xlarge | Just submitted | CUDA-fork fix in place |
| adja-s1-full-20260427-192235 | ml.g6e.xlarge | Just submitted | Fresh LoRA Ewe→Adja run; concurrent with adja-s1-full-20260427-091126 still on g5.2xlarge for parallel partial-Stage-A |

**Code review:** ran 4 parallel reviewer agents on the launch.py + train_AT1 + train_S2_S3 changes. Findings ≥80 confidence: 95-conf "10000 too low for wav2vec2 contrastive — need 32000" (acted on, S2/S3 stopped + resubmitted with new floor); 95-conf "S6 has same CUDA-fork hazard as AT1" (acted on, num_proc=1 forced in cold-cache path); 100-conf "tracking files need wave-3 entries" (this entry).

**Quota state (2026-04-27 evening):** g6e.xlarge=8 (just bumped from 4 to 8), g5.12xlarge=5, g5.2xlarge=6, g5.xlarge=7. p4d.24xlarge=4 still pending review. Recommended next ask: **g6e.12xlarge → 1** (4× L40S 192 GB total) for AT1/M1/M2 wave once we want to scale beyond single-GPU.

---

## 2026-04-27 — AT1 silent disk-full hang; launcher `volume_size` fix

**Failure:** `adja-at1-full-20260426-114103` (AT1, ml.g5.12xlarge) sat at "Downloading data: 39/106 [17:12<29:35, 26.50s/files]" for **~24 hours** with no progress. `aws sagemaker describe-training-job` showed `Status=InProgress`, `SecondaryStatus=Training`, `LastModifiedTime` frozen at 2026-04-26 15:45 UTC (training-start), `FailureReason=null` — silent hang, no crash.

**Root cause:** [scripts/sagemaker_jobs/launch.py](../scripts/sagemaker_jobs/launch.py) constructed the `PyTorch` estimator without a `volume_size=` argument, so SageMaker provisioned the **30 GB default EBS volume** for the entire container. `HF_DATASETS_CACHE=/tmp/hf_cache` lives on that same volume. `google/WaxalNLP` `ewe_asr` `unlabeled` is ~25-30 GB stored **twice** by HF datasets (raw shards in `…/downloads/` + extracted Arrow), so disk filled around shard ~39. `urllib3` blocked on the next `f.write(chunk)` (`ENOSPC`); HF datasets' retry-with-resume swallowed the exception and looped forever — frozen progress bar, billing kept ticking.

**Fix:** Added `volume_size_gb` to `EXPERIMENT_CONFIGS` for jobs that pull WaxalNLP unlabeled or train 1B+ models — AT1/M1/M2/M2b/S6 → 200 GB, S2/S3 → 150 GB. Estimator now passes `volume_size=cfg.get("volume_size_gb", 30)`. Cost banner prints the chosen size for visibility on every submission. Documented as gotcha #14 in [learnings-from-the-past/sagemaker-gotchas.md](../learnings-from-the-past/sagemaker-gotchas.md) with sizing guide.

**Action:** Stopped the hung job (`aws sagemaker stop-training-job --training-job-name adja-at1-full-20260426-114103 --region us-west-2`). AT1 to be resubmitted under the fix.

**Wider impact:** Same failure would have hit M1/M2/M2b/S2/S3/S6 — all use the same launcher and pull either the WaxalNLP unlabeled split or 1B+ checkpoints. They are now safe to submit.

**Cost of this incident:** ~24 h of ml.g5.12xlarge wall time at on-demand rates (~$385). Filed under "the kind of failure CLAUDE.md was warning about" — *"Pre-download model weights to cluster storage — never pull from HF Hub during job execution"*. Long-term cleanup: stage WaxalNLP to S3 once and switch all unlabeled-using jobs to a SageMaker channel input, so the container never touches HF Hub.

---

## 2026-04-26 — SageMaker CF1/CF2: First intelligible Adja Stage 2 output

**Infrastructure fix:** Upgraded SageMaker container from PyTorch 2.1.0/py310/cu121 → **2.5.1/py311/cu124**. Root cause of all CF1/CF2/S7 startup crashes: `transformers>=4.50.0` installs 4.52.x which calls `torch.utils._pytree.register_pytree_node` (added in PyTorch 2.3.0, absent in 2.1.0). Also fixed: S7 `gradient_checkpointing_enable()` incompatible with manual backward-outside-autocast in PyTorch 2.5.1; fixed S1 same way (moved backward inside autocast). Added dotenv auto-load of `.env` to launch.py. Spot quota for g5.2xlarge is 0 — using on-demand.

---

### CF2 — Curriculum Annealing Stage 2 (CSM 1B, Ewe → Adja)
- **Job:** `adja-cf2-full-20260426-161752` | ml.g5.2xlarge | 49.9 min training
- **Setup:** Start from `JosueG/adja-tts-checkpoints/T1_csm_ewe_stage1` (intelligible Ewe). LoRA r=32, lr=5e-5, 20 epochs. Ewe:Adja ratio linearly annealed **5:1 → 1:5** over epochs. Data: WaxalNLP ewe_tts (1,215 clips) + adja-tts-orpheus 80/10/10 split (~987 train). NFC normalization on all text.
- **Metrics:** best_eval_loss=6.6203 @ epoch 19 (still improving — never hit early stopping). Compare: naive Stage 2 baseline=6.5115, prior mixed 1:1=5.648 (but different architecture).
- **Audio:** 5 samples generated at 24 kHz. Natural durations (2.3–7.8s) — **not hitting 10s cap** (contrast with baseline where ALL 5 hit cap).
- **🔑 Listening verdict (Josue Godeme, native speaker, 2026-04-26):** 4/5 noise. **1/5 intelligible Adja: `CF2_adja_sent02.wav` — "Ŋ nyanyɔ mɔ wo anu ahán !"** — confirmed as a real Adja sentence. **First intelligible Adja in any Stage 2 experiment.** Previous record: all Stage 2 outputs across T1/T2 CSM/Orpheus were noise.
- **Audio on HF:** https://huggingface.co/datasets/JosueG/adja-tts-results/tree/main/CF2_curriculum_stage2/generated_audio

Sample decodes (CF2):
```
[0] 'Ŋu nya kpɔ́kpɔ a ?'                     → 2.3s  [noise]
[1] 'ŋnyan go. Kpɔ ŋuɖejikɔ a nyan wo'       → 7.8s  [noise]
[2] 'Ŋ nyanyɔ mɔ wo anu ahán !'               → 4.8s  [INTELLIGIBLE ADJA ✅]
[3] 'Tɛnigbe ciyi vayi de ŋweba'              → 4.6s  [noise]
[4] 'Ganɛni mɛ ayi ɔ̀ ?'                      → 2.3s  [noise]
```

---

### CF1 — EWC Stage 2 (CSM 1B, Ewe → Adja, Elastic Weight Consolidation)
- **Job:** `adja-cf1-full-20260426-183144` | ml.g5.2xlarge | 71.5 min
- **Setup:** Same Stage 1 checkpoint as CF2. EWC λ=1000, Fisher diagonal estimated on 500 Ewe samples. LoRA r=32, lr=5e-5, 20 epochs. Pure Adja Stage 2 data (no Ewe in training loop — EWC penalty keeps Ewe priors via regularization, not data).
- **Metrics:** best_eval_loss=**6.4941** — **best of all Stage 2 experiments** (beats baseline 6.5115, beats CF2 6.6203). Natural durations on all 5 samples (1.76–5.6s).
- **🔑 Listening verdict (Josue Godeme, native speaker, 2026-04-26):** All noise — "no intelligible sentence, crispy noise, you could maybe hear some words in the background but it's not intelligible." Slightly better texture than the old naive Stage 2 baseline (which was flat silence/noise), but no intelligible content.
- **Audio on HF:** https://huggingface.co/datasets/JosueG/adja-tts-results/tree/main/CF1_ewc_stage2/generated_audio

Sample decodes (CF1):
```
[0] 'Ŋu nya kpɔ́kpɔ a ?'                     → 1.76s
[1] 'ŋnyan go. Kpɔ ŋuɖejikɔ a nyan wo'       → 4.24s
[2] 'Ŋ nyanyɔ mɔ wo anu ahán !'               → 3.2s
[3] 'Tɛnigbe ciyi vayi de ŋweba'              → 5.6s
[4] 'Ganɛni mɛ ayi ɔ̀ ?'                      → 3.68s
```

---

### Jobs also running as of 2026-04-26 20:00
- **AT1** (`adja-at1-full-20260426-114103`): CSM audio-LM 3-stage pretrain on 183k unlabeled Ewe, ~8.9h in. 
- **CF3** (`adja-cf3-full-20260426-203521`): frozen backbone Stage 2 — trains only audio heads.
- **TK1** (`adja-tk1-full-20260426-203528`): expanded tokenizer ablation on Ewe Stage 1.
- **S1** (`adja-s1-full-20260426-203534`): Whisper-large-v3 LoRA Ewe→Adja (~20h).
- **S4** (`adja-s4-full-20260426-203540`): MMS-1B + Ewe adapter → Adja CTC (~5h).
- **S7** (`adja-s7-full-20260426-183137`): Whisper-tiny Adja FT — **completed** (14 min). CER pending.

---

## 2026-04-22 — TTS Gbe-family Stage 1 results + failure-mode debugging

### Completed (with listening test from Josue)

- **T1 CSM-Ewe Stage 1** — Job `69e830e1ac288e522d8f0782` | L40S | 17.4 min | best eval_loss 4.326 @ ep 3.86
  - **Audio verdict (native speaker Josue, 2026-04-22): intelligible Ewe.**
  - Checkpoint pushed to `JosueG/adja-tts-checkpoints/T1_csm_ewe_stage1`.
  - First CSM run to produce intelligible Gbe-family speech anywhere in this repo. Confirms architecture + Gbe data works; earlier direct-Adja failures were data-volume-bound, not architecture-bound.

- **T10 F5-TTS full fine-tune (Adja only)** — Job `69e8318fac288e522d8f0787` | L40S | 48.9 min
  - Audio verdict: pure noise. Natural durations (2.6–4.3s) but no intelligible content.
- **T11 E2-TTS full fine-tune (Adja only)** — Job `69e831e4ac288e522d8f078b` | L40S | 50.9 min
  - Audio verdict: pure noise. Same character as F5.
- **T1 CSM tokfix (Adja only + 35 expanded Llama tokens)** — Job `69e857f3cd8c002f31e016a6` | L40S
  - Audio verdict: speech-like noise with rare intelligible fragments.
  - Josue's quote: *"just noise but a tiny better noise than in the past — you could hear someone speaking sometime in the background but unintelligible (lots of crisp noise like iron being rammed through the ground)."*
  - **Retires the "Llama BPE fragmentation is the primary TTS bottleneck" hypothesis.** With the tokenizer fixed, output is still noise. Data volume / Gbe-family prior dominates.
- **T2 Orpheus EN-Ewe Stage 1** — Job `69e846e8ac288e522d8f084f` | L40S | best eval_loss ~0.16
  - Checkpoint pushed. Audio generation deferred to Stage 2.
- **T2 Orpheus FR-Ewe Stage 1** — Job `69e846eaac288e522d8f0851` | L40S | best eval_loss ~0.16
  - Checkpoint pushed. Audio generation deferred to Stage 2.
- **T2 Orpheus tokfix** — Job `69e83cf1ac288e522d8f07e9` | L40S
  - Completed but produced 0 audio files. Needs investigation; not counted as a clean result.

### Fixes landed across the 2026-04-21/22 wave (full writeup in `session-logs/2026-04-22-tts-failure-modes-debug.md`)

1. **Orpheus grad-graph crash** (`loss does not require grad`): added `model.enable_input_require_grads()` after `get_peft_model()` in `T2_orpheus_ewe_stage1.py`. Root cause: `gradient_checkpointing=True` + PEFT LoRA without input-grad hook breaks the backward pass. Working `T1_csm_ewe_stage1.py` avoided this by leaving gradient_checkpointing OFF (explicit code comment warns about PEFT).
2. **CSM tokfix vocab-size crash** (`shape '[-1, 128291]' is invalid`): in `T1_csm_tokfix.py`, save `config.vocab_size` before any resize attempt and restore it after. `CsmForConditionalGeneration.backbone_loss` uses `config.vocab_size` for the Mimi audio codebook (size 2051), *not* the text vocab. Both HF's `resize_token_embeddings` and our `_manual_resize_token_embeddings` fallback mutated that field to 128291.
3. **Spark CVE-2025-32434** (`torch<2.6` blocked by `transformers>=4.50`): bumped torch 2.5.1→2.6.0 in `T3_spark_ewe_stage1.py` uv metadata. Surfaced two follow-ons: (a) `Python.h: No such file` because triton 3.2+ JIT-compiles a Python C extension and the uv container has no `libpython3.11-dev`; (b) `-lcuda` link error because libcuda is only mounted as `libcuda.so.1` at `/usr/lib64-nvidia/`. Fixed by adding `python3-dev libpython3.11-dev` to the `apt-get install` and extending the libcuda shim to symlink + LIBRARY_PATH the driver directory. Latest retry `69e8dd0bd2fd2eb837d76959` — outcome pending.

### Stage 2 submissions (same day, after tracking docs updated)

All `l40sx1`, 8h, `--push-to-hub`, loading Stage 1 merged checkpoint from `JosueG/adja-tts-checkpoints/`. Hyperparameters identical to Stage 1 except LR 5e-5 (vs 2e-4) and gradient_checkpointing OFF.

- **T1 CSM Ewe → Adja Stage 2** — Job `69e8e3e8d2fd2eb837d769ad` | Loads `T1_csm_ewe_stage1` (intelligible-Ewe checkpoint). Headline experiment of the Gbe-cascade hypothesis.
- **T2 Orpheus EN Ewe → Adja Stage 2** — Job `69e8e3ead2fd2eb837d769af` | Loads `T2_orpheus_en_ewe_stage1`.
- **T2 Orpheus FR Ewe → Adja Stage 2** — Job `69e8e3ec2aa1660eaffa8a83` | Loads `T2_orpheus_fr_ewe_stage1`. Paired comparison with EN.

### Stage 1 follow-up resubmissions (same day)

- **T2 Orpheus ZH Ewe Stage 1** — Job `69e8e3edd2fd2eb837d769b1` | `l40sx1` | First ZH submission in this wave; uses the fixed `T2_orpheus_ewe_stage1.py` with `enable_input_require_grads()`. Will fill out the EN/FR/ZH comparison once done.
- **T3 Spark Ewe Stage 1** — Job `69e8e3efd2fd2eb837d769b3` | **`a100-large`** | Previous retry `69e8dd0b` hit CUDA OOM on `l40sx1` after the Python.h + libcuda shim finally let it past import. Moving to 80GB A100 for now; proper memory fix (bf16 load / lower LoRA rank) is TBD.

### Stage 2 resubmissions (after `No module named pip` failure)

The three Stage 2 jobs from the first wave all failed instantly with `No module named pip`. Root cause: Stage 2 scripts did `python -m pip install ...` at runtime, but the `hf jobs uv run` container doesn't ship pip — only `uv`. Stage 1 scripts avoid this by declaring dependencies in an inline PEP-723 `/// script ///` block that uv reads before launching the script. Migrated both Stage 2 scripts to the same pattern (added inline deps, removed `install_env()` / `pip_install` helpers).

- `69e8f632d2fd2eb837d76a70` — CSM Ewe → Adja Stage 2 (resubmit)
- `69e8f6352aa1660eaffa8ae4` — Orpheus EN Ewe → Adja Stage 2 (resubmit)
- `69e8f6372aa1660eaffa8ae6` — Orpheus FR Ewe → Adja Stage 2 (resubmit)

### Orpheus Stage 1 inference-only jobs (so we can listen to the Ewe output)

Orpheus Stage 1 training scripts intentionally skip audio generation to keep Stage 1 fast — the Ewe output had never been auditioned (unlike CSM, which did generate audio at the end of Stage 1). Without a listening test we cannot judge whether Orpheus-Ewe is intelligible the way CSM-Ewe is. Added new script `scripts/hf_jobs/T2_orpheus_ewe_stage1_inference.py` that loads a Stage 1 checkpoint and decodes 5 held-out Ewe test-split sentences via SNAC (same generation path as Stage 2). Parameterized by `--checkpoint-tag`.

- `69e8f639d2fd2eb837d76a72` — Orpheus EN Ewe Stage 1 inference
- `69e8f63ad2fd2eb837d76a74` — Orpheus FR Ewe Stage 1 inference

Outputs will appear under `JosueG/adja-tts-results/T2_orpheus_{en,fr}_ewe_stage1_inference/generated/`.

### Paper-prep documentation landed this session

- New: [docs/gbe-cascade-tts-settings-2026-04-22.md](../docs/gbe-cascade-tts-settings-2026-04-22.md) — frozen data/splits/hyperparameters/rationale for the Gbe-cascade wave, for direct citation.
- CLAUDE.md Track 4 updated to reflect current TTS state (Spark direct-Adja intelligible; CSM Ewe intelligible; direct-Adja noise across the other LLM-TTS).
- `experiments/registry.md` gained a TTS data-choice note explaining ewe_tts (1,215) vs ewe_asr (15k) choice.
- `experiments/tts/attempt-log.md` Cycle 16 documents this session's submissions, failure modes, and next actions.

### What this unblocks

Stage 2 (Ewe-pretrained → Adja) is now in flight for CSM, Orpheus EN, Orpheus FR. Once those return:
- If CSM produces intelligible Adja: first confirmed recipe for an English-centric LLM-TTS on Adja; paper has its headline.
- If Orpheus EN/FR/ZH differ meaningfully: confirms FR sociolinguistic and/or ZH tonal priors add signal on top of Ewe acoustic prior.
- T3 Spark Ewe Stage 1 still pending (infra). Stage 2 Spark script ready to fire once Stage 1 lands.

---

## 2026-04-22 (continued) — Stage 2 listening verdicts + catastrophic forgetting diagnosis + mixed-data experiment

### Listening verdicts (native speaker, Josue 2026-04-22)

- **Orpheus EN Ewe Stage 1 inference** (job `69e8f639d2fd2eb837d76a72`) — **Orpheus EWE intelligible** per Josue: *"Orpheus EWE is also intelligible good stuff (i sent the audios to friends who speak ewe to get their takes on how it sounds if it's super good and etc)"*. Confirms the Gbe-family bridge works for Orpheus EN as well as CSM (different text tokenizers, different audio codecs: Mimi vs SNAC).
- **Orpheus FR Ewe Stage 1 inference** (job `69e8f63ad2fd2eb837d76a74`) — intelligible Ewe, listening test ongoing with Ewe-speaking friends.
- **Orpheus ZH Ewe Stage 1 inference** (job `69e937b6d2fd2eb837d76d54`) — only 1/5 samples generated (sample 2 triggered a CUDA device-side assert that poisoned the CUDA context for samples 3-5). Sample 1 is 14.6s of Ewe-ish audio.
- **Spark Ewe Stage 1 inference on L40S** (job `69e94068d2fd2eb837d76d98`) — 5 Ewe samples with natural durations (6.8-25.3s), listening verdict pending. Queue-race: L40S finished while a100-large duplicate `69e93aba` was still SCHEDULING. L40S inference worked without OOM — confirms the Stage 1 training OOM was purely the memory footprint of BiCodec + Spark LLM + optimizer state co-resident, not the size of inference itself.

### Stage 2 outputs — all three produce NOISE

- **Orpheus EN → Adja Stage 2** (`69e937572aa1660eaffa8c5a`) — 29.5 min, best_eval_loss 5.6156 @ ep 5.63, patience=5 early-stopped at ep 7.19. Samples 1.88-2.9s natural durations. **Audio: noise**. Josue 2026-04-22: *"noise - i am super surprised because the EWE had good EWE speaking stuff! now that we finetuned it on adja - it is just noise"*.
- **Orpheus FR → Adja Stage 2** (`69e9375a2aa1660eaffa8c5c`) — 34.4 min, best_eval_loss 5.6356, same patience=5 early stop. Natural durations. **Audio: noise, same pattern as EN**.
- **CSM → Adja Stage 2** (`69e93755d2fd2eb837d76d4e`) — 37.5 min, best_eval_loss 6.5115, **ran full 20-epoch budget with no early stopping firing**. All 5 samples hit the 10s cap. **Audio: noise, silence-termination not learned** (same pathology as Stage 1 CSM Ewe).

### Catastrophic forgetting diagnosis

Initial Stage-2 training-step loss was ~17 on Adja (ln(4096)=8.3 is random-chance for the SNAC codebook). A loss of 17 at step 1 means the Stage 1 model is *confidently producing Ewe-patterned audio tokens* when prompted with Adja text — not ignorant, actively wrong-for-Adja. Adaptation has to overcome the learned Ewe behavior before producing anything Adja-shaped. On 1,276 Adja clips, the LoRA adapter drift erases the Gbe prior faster than Adja grammar is acquired, so both ends crash together and we land on noise.

**Critical evidence that this is forgetting, not a training-budget issue**: CSM Stage 2 ran the **full 20-epoch budget** (37.5 min, early stopping never triggered) and still produced noise. If training time were the bottleneck, CSM would have produced at least partially coherent output.

### Early-stopping / training-budget review

- Current config everywhere: `EarlyStoppingCallback(early_stopping_patience=5, early_stopping_threshold=0.0)` with `eval_steps=50` → 250 non-improving steps ≈ 1.5 epochs of wait time.
- HF/Whisper field norm: patience 10 + threshold 0.001-0.01. TTS paper norm (Orpheus, SNAC, Tacotron 2, VITS): no early stopping, fixed schedule.
- Orpheus EN S2 stopped after 5 evals where eval_loss drifted by only 0.035 nats — within noise, not divergence.
- Planned follow-up (parked for now — CSM ran full budget and still failed, so patience is at best a minor contributor): bump to patience=10, threshold=0.01 if we run another baseline ablation. Currently LOWER priority than the mixed-data experiment.

### Mixed-data Stage 2 submission (the real experiment)

New script: `scripts/hf_jobs/T2_orpheus_ewe_adja_stage2_mixed.py`. Based on `T2_orpheus_ewe_adja_stage2.py` with one change — Stage 2 trains on **Adja + Ewe 1:1 concat (shuffled, seed=42)**, eval remains Adja-only. Canonical anti-catastrophic-forgetting recipe. Same hyperparameters as baseline for A/B: LR 5e-5, LoRA r=64 on seven linear projections, patience=5 (keeps apples-to-apples), 20 epochs max, bf16, adamw_8bit.

Submitted: **Job `69e977f82aa1660eaffa8d21`** on L40S with 8h timeout. Expected wall time 60-90 min.

Output will push to [T2_orpheus_en_ewe_adja_stage2_mixed/](https://huggingface.co/JosueG/adja-tts-results/tree/main/T2_orpheus_en_ewe_adja_stage2_mixed) — separate prefix so the baseline `T2_orpheus_en_ewe_adja_stage2/` is preserved for the A/B.

### Thesis-ready writeup

Created `thesis-writing/gbe-cascade-results-2026-04-22.md` — thesis-ready writeup for the undergrad graduation thesis. Parallels the style of `research-paper-exploration/paper-one/why-spark-worked.md` but lives in `thesis-writing/` since the thesis and research-paper outputs are separate tracks. Captures: the intelligible-Ewe Stage 1 finding across CSM and Orpheus EN/FR (architectures + codec families), the catastrophic-forgetting failure mode with loss-curve evidence, the retired tokenizer-expansion hypothesis, and the mixed-data recipe under test. Ready to cite directly in the thesis TTS chapter.

---

## 2026-04-23 — T2-orpheus-en-ewe-adja-stage2-mixed — anti-catastrophic-forgetting experiment (COMPLETED, audio: better noise)

- **Job:** `69e977f82aa1660eaffa8d21` | L40S | 53.7 min
- **Script:** `scripts/hf_jobs/T2_orpheus_ewe_adja_stage2_mixed.py`
- **Setup:** Orpheus 3B EN, Stage 1 Ewe checkpoint (`JosueG/adja-tts-checkpoints/T2_orpheus_en_ewe_stage1`) → Stage 2 on Adja+Ewe 1:1 mix (~1,276 Adja + 1,215 Ewe train clips, shuffled seed=42). Eval Adja-only. Canonical anti-catastrophic-forgetting recipe.
- **Training:** train_loss=2.4659, best_eval_loss=**5.648** @ epoch 6.79. Early stopping fired (patience=5). Only ran ~7/20 epochs — mirror of the pure-Adja baseline behaviour.
- **Generated files:** 3/5 samples uploaded (`adja_01.wav`, `adja_02.wav`, `adja_04.wav`; samples 03 and 05 failed silently during generation). Audio: [T2_orpheus_en_ewe_adja_stage2_mixed/generated_audio](https://huggingface.co/JosueG/adja-tts-results/tree/main/T2_orpheus_en_ewe_adja_stage2_mixed/generated_audio)
- **Listening verdict (Josue 2026-04-23): "better noise than before, but still noise."**
- **Key finding:** The 1:1 Ewe reinforcement recipe marginally improves audio texture vs. the pure-Adja Stage 2 baseline (best_eval_loss 5.648 vs 5.616 — virtually identical). The acoustic character is perceptibly slightly better but falls well short of intelligibility. Catastrophic forgetting is not fully addressed by naive data mixing at 1:1; the Ewe prior is still being eroded.
- **Adapter push:** failed with `ValueError: Invalid metadata in README.md` (`base_model` path is a local `/tmp/` path — not a Hub ID). WAVs and `metrics.json` uploaded successfully.
- **Comparison to baseline:** pure-Adja Stage 2 best_eval_loss=5.616 → mixed best_eval_loss=5.648 (effectively the same). The loss plateau is at the same level. The slight audio improvement suggests some acoustic texture benefit from Ewe reinforcement, but not enough to cross the intelligibility threshold.
- **What this retires:** the hypothesis that naive 1:1 data mixing is sufficient to prevent catastrophic forgetting of the Gbe prior. Something stronger is needed (higher Ewe ratio, elastic weight consolidation, lower Stage-2 LR, or more Adja data).

---

## 2026-04-16 — C1-zs — Whisper zero-shot (tiny + small)
- Status: completed
- Platform: HF Jobs | GPU: A10G | Job: 69e02cb9
- WER: tiny=1208% small=101% | CER: tiny=2048% small=367%
- Storage: JosueG/adja-asr-results/C1/
- Notes: Whisper has zero Adja knowledge. Outputs "banana", Georgian script (ლლლ), "www.www.www". Confirms Adja is fully out-of-distribution.

## 2026-04-16 — C3 — MMS-1B + French adapter (CTC)
- Status: completed (early stopped epoch 6)
- Platform: HF Jobs | GPU: A100 80GB | Job: 69e037de | Wall time: 36min
- WER: 98.42% | CER: 87.19%
- Storage: JosueG/adja-asr-results/C3/
- Notes: Only CTC model that learned anything. Built-in CTC loss (model.forward with labels) was key fix. Outputs "E" and "E ?" — learned most common char but not more.

## 2026-04-16 — C4 — XLS-R 300M (CTC) [CRASHED]
- Status: crashed (CTC collapse)
- Platform: HF Jobs | GPU: A10G | Job: 69e03d0e
- Notes: Loss dropped to 0 after epoch 1, all-blank output. Multiple fixes attempted: collate padding fix, lower LR, built-in loss. Loss was decreasing (22→4.7) but CER stayed 100%. Early stopped at patience 5.

## 2026-04-16 — C5 — wav2vec2 English (CTC) [CRASHED]
- Status: crashed (CTC collapse / plateau)
- Platform: HF Jobs | GPU: A10G | Job: 69e03d15
- Notes: Similar to C4. Loss plateaued at 3.5, CER stayed 100%. English-only pretraining may be too distant for Adja.

## 2026-04-16 — E1 — MMS-1B + Ewe adapter (CTC) [CRASHED]
- Status: crashed (CTC collapse)
- Platform: HF Jobs | GPU: A100 | Job: 69e03bb9
- Notes: Ewe adapter vocab_size=55 < Adja vocab 115 caused label error. After fix, still collapsed (loss→0 from epoch 2). OOM issues required gradient checkpointing.

## 2026-04-16 — C2 — Whisper-small fine-tuned on Adja
- Status: completed (50 epochs)
- Platform: HF Jobs | GPU: A10G | Job: 69e04dbf | Wall time: 5.5h
- **WER: 74.15% | CER: 27.02%** (best at epoch 47)
- Storage: JosueG/adja-asr-results/C2/
- Decode samples:
  - REF: "Enu maku enyi" → HYP: "Enu ma ku enyi" (near-perfect)
  - REF: "Eshilɔ ɖote yi mi jaja a ?" → HYP: "Ekɖo te yi mi jaja" (core words correct)
  - REF: "ŋɖuɖu lɔwo nuɔn" → HYP: "Ŋɖudu lɔwo nunn ɔ ?" (got ŋɖu, lɔwo)

## 2026-04-16 — E4 — Whisper-Ewe → Adja (BEST MODEL)
- Status: completed (50 epochs, still improving)
- Platform: HF Jobs | GPU: A10G | Job: 69e04dc7 | Wall time: 3.7h
- **WER: 73.09% | CER: 24.90%** (best at epoch 50 — hasn't converged!)
- Storage: JosueG/adja-asr-results/E4/
- Decode samples:
  - REF: "ŋɖuɖu lɔwo nuɔn" → HYP: "ŋ ɖudu lɔwo nɔ?" (ŋ, ɖ, lɔwo correct)
  - REF: "Eshilɔ ɖote yi mi jaja a ?" → HYP: "Eɖote yi mi jaja" (ɖote, yi, mi, jaja perfect)
  - REF: "Enu maku enyi" → HYP: "Enu maku eye" (first two words perfect)
- Notes: Ewe transfer gives 2.1% CER improvement over C2. Model was STILL improving at epoch 50 — more training would yield better results.

## 2026-04-16 — Omni_ZS — Omnilingual-ASR zero-shot (300M)

- Status: completed
- Platform: HF Jobs | GPU: A10G | Job: `69e37788`
- Metrics:
  - `dev`: WER 129.32% | CER 145.22%
  - `test`: WER 133.87% | CER 139.49%
- Storage: `JosueG/adja-asr-results/Omni_ZS/`
- Notes:
  - Script: `scripts/hf_jobs/omni_asr_zeroshot.py`
  - Language code mapping used in the original branch: `aj_Latn -> adj_Latn`
  - Inference-only zero-shot check; decode quality remained very poor for Adja.

## 2026-04-18 — Omni ICL (7B_ZS) initial smoke + A100 fallback [COMPLETED]

- Status: mixed attempts, then completed on fallback
- Platform: HF Jobs
- Initial smoke submissions:
  - `Omni_ICL_k1` (A10G small): Job `69e37a48ac288e522d8efce6` → failed (`Job timeout`)
  - `Omni_ICL_k3` (A10G small): Job `69e37a48cd8c002f31dfe8c8` → failed (`Job timeout`)
  - `Omni_ICL_k10` (A10G small): Job `69e37a48cd8c002f31dfe8ca` → failed (`OOMKilled`, exit 137)
- First completed fallback:
  - Consolidated A100 run (`CONTEXT_K_LIST=1,3,10`): Job `69e38205cd8c002f31dfe92e`
- Storage: `JosueG/adja-asr-results/Omni_ICL_k*/metrics.json`
- A100 fallback metrics (test split):
  - `k=1`: WER 134.48% | CER 75.63%
  - `k=3`: WER 489.66% | CER 307.56%
  - `k=10`: WER 793.10% | CER 576.47%
- Notes:
  - This was the first completed Omni ICL path before the later A10Gx4 rerun.
  - On this smoke slice, ICL was still much worse than even the poor 300M zero-shot run and degraded as context size grew.

## 2026-04-18 — OmniASR ICL reruns (k=1/3/10) [COMPLETED]

- Status: completed
- Platform: HF Jobs | GPU: A10Gx4 | Job: `69e38655ac288e522d8efd20` | Wall time: ~7 min
- Script: `scripts/hf_jobs/omni_asr_icl.py` (kept zero-shot script unchanged)
- Storage:
  - `JosueG/adja-asr-results/Omni_ICL_k1/`
  - `JosueG/adja-asr-results/Omni_ICL_k3/`
  - `JosueG/adja-asr-results/Omni_ICL_k10/`
- Metrics (test split):
  - `k=1`: **WER 424.14% | CER 284.87%**
  - `k=3`: WER 493.10% | CER 311.76%
  - `k=10`: WER 696.55% | CER 511.76%
- Notes:
  - Best Omni ICL setting in this rerun was `k=1`; adding more context hurt Adja performance.
  - Earlier ICL tries included one completed run on A100 (`69e38205cd8c002f31dfe92e`) and failed startup/OOM attempts (`69e37a48cd8c002f31dfe8ca`, `69e37a48ac288e522d8efce6`, `69e37a48cd8c002f31dfe8c8`) before stabilizing on A10Gx4.

## 2026-04-18 — OmniASR CTC fine-tune (official recipe) [COMPLETED]

- Status: completed
- Platform: HF Jobs | GPU: A10G | Job: `69e3ae07cd8c002f31dfeac1`
- Script: `scripts/hf_jobs/omni_asr_finetune_recipe.py` with `FT_MODE=ctc`
- Storage: `JosueG/adja-asr-results/Omni_FT_CTC_ctc_1776528945/`
- Runtime evidence:
  - reached `Train Metrics (step 400)` and saved checkpoint at step 400
  - uploaded `run_summary.json` + `recipe_config.yaml` artifact bundle
- Notes:
  - This is the first stable official Omni recipe fine-tune completion in this repo.

## 2026-04-18 — OmniASR LLM fine-tune (official recipe) [COMPLETED]

- Status: completed
- Platform: HF Jobs | GPU: H200 | Job: `69e3b1f9cd8c002f31dfeae4`
- Script: `scripts/hf_jobs/omni_asr_finetune_recipe.py` with `FT_MODE=llm`
- Storage: `JosueG/adja-asr-results/Omni_FT_LLM_llm_1776529986/`
- Runtime evidence:
  - reached `End of training reached at step 200`
  - checkpoint saved at step 200
  - uploaded `run_summary.json` + `recipe_config.yaml` artifact bundle
- Notes:
  - Initial LLM attempts on `a10g-large` and `a10g-largex4` failed with CUDA OOM (`69e3ae07ac288e522d8efd62`, `69e3b0d9cd8c002f31dfeada`); moving to H200 resolved memory pressure.
  - One extra queued fallback submission remained in `SCHEDULING` on `a100-large` (`69e3af7fac288e522d8efd64`) at update time.

## 2026-04-18 — OmniASR CTC fine-tune + terminal eval pass [COMPLETED]

- Status: completed
- Platform: HF Jobs | GPU: A10G | Job: `69e3c637ac288e522d8efd94`
- Script: `scripts/hf_jobs/omni_asr_finetune_recipe.py` with `FT_MODE=ctc`, `VALID_SPLITS=dev,test`
- Storage: `JosueG/adja-asr-results/Omni_FT_CTC_EVAL_ctc_1776535219/`
- Runtime evidence:
  - reached `End of training reached at step 400`
  - checkpoint saved at step 400
  - terminal validation ran over both `dev` and `test` splits at step 400
- Metrics (validation pass over dev+test):
  - `dev`: WER **96.4419%**, UER 56.9519%, CTC Loss 82.7402
  - `test`: WER **98.8327%**, UER 56.5178%, CTC Loss 86.2806
- Notes:
  - fairseq2 reports one `Validation Metrics` block per split, in split order (`dev`, then `test`).
  - `run_summary.json` + `recipe_config.yaml` uploaded; eval transcriptions were written under nested fairseq2 workspace paths, so script-level `eval_metrics` stayed null for this run.

## 2026-04-18 — OmniASR LLM fine-tune + terminal eval pass [COMPLETED]

- Status: completed
- Platform: HF Jobs | GPU: H200 | Job: `69e3c63fcd8c002f31dfeb8d`
- Script: `scripts/hf_jobs/omni_asr_finetune_recipe.py` with `FT_MODE=llm`, `VALID_SPLITS=dev,test`
- Storage: `JosueG/adja-asr-results/Omni_FT_LLM_EVAL_llm_1776535142/`
- Runtime evidence:
  - reached `End of training reached at step 200`
  - checkpoint saved at step 200
  - terminal validation ran over both `dev` and `test` splits at step 200
- Metrics (validation pass over dev+test):
  - `dev`: WER **101.124%**, UER 55.2139%, CTC Loss 2.89444
  - `test`: WER **100.973%**, UER 59.4046%, CTC Loss 2.9504
- Notes:
  - This confirms Omni LLM 300M fine-tune converges in training loss but still performs very poorly on held-out Adja transcription quality.
  - `run_summary.json` + `recipe_config.yaml` uploaded; eval transcriptions were written under nested fairseq2 workspace paths, so script-level `eval_metrics` stayed null for this run.

## 2026-04-18 — OmniASR CTC fine-tune + FULL DATA terminal eval pass [COMPLETED]

- Status: completed
- Platform: HF Jobs | GPU: A10G | Job: `69e3ce60cd8c002f31dfebf9`
- Script: `scripts/hf_jobs/omni_asr_finetune_recipe.py` with `FT_MODE=ctc`, `VALID_SPLITS=dev,test`, `MAX_TRAIN_SAMPLES=0`, `MAX_DEV_SAMPLES=0`, `MAX_TEST_SAMPLES=0`
- Storage: `JosueG/adja-asr-results/Omni_FT_CTC_FULLDATA_ctc_1776537232/`
- Runtime evidence:
  - reached `End of training reached at step 400`
  - checkpoint saved at step 400
  - terminal validation ran over both `dev` and `test` splits at step 400
- Data coverage:
  - split prep showed `train=1277`, `dev=160`, `test=160`
  - manifest stats confirmed all prepared examples were kept
- Metrics (validation pass over dev+test):
  - `dev`: WER **97.0483%**, UER 57.7278%, CTC Loss 87.0044
  - `test`: WER **96.6574%**, UER 55.7161%, CTC Loss 85.7181
  - fairseq2 score line: WER **193.706%** (dev+test aggregate)
- Notes:
  - This run removed sample caps exactly as requested and still produced very poor WER, so the earlier poor Omni CTC result is not explained by capped subset size alone.
  - Script-level post-run `eval_metrics` also printed from transcription files (`dev WER 96.95`, `test WER 96.75`; 310 paired lines used), consistent with the fairseq2 terminal metrics.

## 2026-04-18 — OmniASR LLM fine-tune + FULL DATA terminal eval pass [COMPLETED]

- Status: completed
- Platform: HF Jobs | GPU: H200 | Job: `69e3d20ccd8c002f31dfec21`
- Script: `scripts/hf_jobs/omni_asr_finetune_recipe.py` with `FT_MODE=llm`, `VALID_SPLITS=dev,test`, `MAX_TRAIN_SAMPLES=0`, `MAX_DEV_SAMPLES=0`, `MAX_TEST_SAMPLES=0`
- Storage: `JosueG/adja-asr-results/Omni_FT_LLM_FULLDATA_llm_1776538162/`
- Runtime evidence:
  - reached `End of training reached at step 200`
  - checkpoint saved at step 200
  - terminal validation ran over both `dev` and `test` splits at step 200
- Data coverage:
  - split prep showed `train=1277`, `dev=160`, `test=160`
  - script-level eval used `310` paired lines from generated transcriptions
- Metrics (validation pass over dev+test):
  - fairseq2 `dev`: WER **90.4293%**, UER 48.5102%, CTC Loss 1.85602
  - fairseq2 `test`: WER **96.9359%**, UER 50.6040%, CTC Loss 1.89563
  - fairseq2 score line: WER **187.365%** (dev+test aggregate)
  - script `dev`: WER **91.04%**, CER **48.85%**
  - script `test`: WER **96.46%**, CER **50.31%**
- Notes:
  - This run removed sample caps exactly as requested and improved over the prior eval-enabled LLM run (`202.10%` aggregate WER) but is still far behind Whisper baselines.
  - Script CER/WER differs slightly from fairseq2 because only 310 transcription pairs were present (vs expected 320), as shown in `_meta`.

## 2026-04-18 — OmniASR LLM 3B fine-tune + FULL DATA terminal eval pass [COMPLETED]

- Status: completed
- Platform: HF Jobs | GPU: H200 | Job: `69e3d998ac288e522d8efdda`
- Script: `scripts/hf_jobs/omni_asr_finetune_recipe.py` with `FT_MODE=llm`, `MODEL_NAME=omniASR_LLM_3B_v2`, `VALID_SPLITS=dev,test`, `MAX_TRAIN_SAMPLES=0`, `MAX_DEV_SAMPLES=0`, `MAX_TEST_SAMPLES=0`
- Storage: `JosueG/adja-asr-results/Omni_FT_LLM_3B_FULLDATA_R1_llm_1776540095/`
- Runtime evidence:
  - reached `End of training reached at step 200`
  - checkpoint saved at step 200
  - terminal validation ran over both `dev` and `test` splits at step 200
- Data coverage:
  - split prep showed `train=1277`, `dev=160`, `test=160`
  - script-level eval used `310` paired lines from generated transcriptions
- Metrics (validation pass over dev+test):
  - fairseq2 `dev`: WER **88.4615%**, UER 45.1661%, CTC Loss 1.66532
  - fairseq2 `test`: WER **89.9721%**, UER 44.6290%, CTC Loss 1.74362
  - fairseq2 score line: WER **178.434%** (dev+test aggregate)
  - script `dev`: WER **88.60%**, CER **45.12%**
  - script `test`: WER **89.87%**, CER **44.66%**
- Notes:
  - This is clearly better than the 300M full-data LLM rerun (aggregate fairseq2 WER `187.365%`) but still far behind Whisper baselines.
  - Initial 3B submission `69e3d685cd8c002f31dfec4a` failed due to a runtime device init issue (`process device cpu` in distributed mode); immediate retry `69e3d998ac288e522d8efdda` completed.

## 2026-04-18 — OmniASR CTC 3B fine-tune + FULL DATA terminal eval pass [COMPLETED]

- Status: completed
- Platform: HF Jobs | GPU: H200 | Final successful Job: `69e3e7e6ac288e522d8efe11`
- Script: `scripts/hf_jobs/omni_asr_finetune_recipe.py` with `FT_MODE=ctc`, `MODEL_NAME=omniASR_CTC_3B_v2`, `DATA_PARALLELISM=fsdp`, `SAVE_MODEL_ONLY=true`, `CHECKPOINT_EVERY_N_STEPS=390`, `VALID_SPLITS=dev,test`, `MAX_TRAIN_SAMPLES=0`, `MAX_DEV_SAMPLES=0`, `MAX_TEST_SAMPLES=0`
- Storage: `JosueG/adja-asr-results/Omni_FT_CTC_3B_FULLDATA_R3_ctc_1776543757/`
- Runtime evidence:
  - reached `End of training reached at step 390`
  - checkpoint saved at step 390
  - terminal validation ran over both `dev` and `test` splits at step 390
- Data coverage:
  - split prep showed `train=1277`, `dev=160`, `test=160`
  - script-level eval used `310` paired lines from generated transcriptions
- Metrics (validation pass over dev+test):
  - fairseq2 `dev`: WER **91.3238%**, UER 50.0322%, CTC Loss 75.3940
  - fairseq2 `test`: WER **91.5506%**, UER 47.9940%, CTC Loss 73.8813
  - fairseq2 score line: WER **182.874%** (dev+test aggregate)
  - script `dev`: WER **91.47%**, CER **50.01%**
  - script `test`: WER **91.40%**, CER **47.95%**
- Notes:
  - First two 3B CTC attempts (`69e3d5f9ac288e522d8efdd5`, `69e3dc86cd8c002f31dfec86`) reached terminal training step but were OOM-killed at final checkpoint prep (`exit code 137`).
  - Third attempt (`69e3e7e6ac288e522d8efe11`) succeeded after switching CTC to `fsdp` and using `save_model_only=true` with checkpoint cadence aligned to final step.

## 2026-04-18 — OmniASR 3B forensics: eval-mismatch + truncation controls [COMPLETED]

- Status: completed (2 successful comparison runs + 1 failed infra run)
- Goal:
  - explain persistent `310/320` eval-pair mismatch
  - isolate effect of aggressive LLM audio truncation (`max_audio_sec=8`) under controlled settings
- Script revision:
  - updated `scripts/hf_jobs/omni_asr_finetune_recipe.py` to support `MIN_AUDIO_LEN` and emit per-split manifest diagnostics (`dropped_short_audio`, `dropped_empty_text`)
- Runs:
  - **A8_M2** (`69e40d23ac288e522d8efe5e`) — `MAX_AUDIO_SEC=8`, `MIN_AUDIO_LEN=32000`
  - **A8_M0** (`69e40d23cd8c002f31dfee7a`) — `MAX_AUDIO_SEC=8`, `MIN_AUDIO_LEN=1600`
  - **A15_M0** (`69e41393cd8c002f31dfeea8`) — `MAX_AUDIO_SEC=15`, `MIN_AUDIO_LEN=1600`
  - **A15_M2** (`69e40d23ac288e522d8efe5f`) — failed at startup with the known HF device-init issue (`process device cpu`)
- Key evidence (root cause of 310/320):
  - A8_M2 manifest diagnostics explicitly show `dev kept=155` and `test kept=155`, each with `dropped_short_audio=5` at `min_audio_len=32000`.
  - A8_M2 eval metadata reports `num_ref_lines=310`, `expected_total=310`, `used_pairs=310` (so this is expected after filtering, not silent eval loss).
  - A8_M0 and A15_M0 keep all dev/test samples (`160+160`) and both report `num_ref_lines=320`, `expected_total=320`, `used_pairs=320`.
- Metrics snapshot (script post-eval):
  - A8_M2: dev WER 89.80 / CER 45.08, test WER 90.53 / CER 44.84 (310 pairs)
  - A8_M0: dev WER 89.13 / CER 45.49, test WER 88.40 / CER 44.11 (320 pairs)
  - A15_M0: dev WER 90.71 / CER 47.74, test WER 91.57 / CER 46.86 (320 pairs)
- Takeaways:
  - the `310/320` mismatch is explained by short-audio filtering (`MIN_AUDIO_LEN=32000`), not by transcription-file discovery bugs.
  - moving to 320-pair evaluation (`MIN_AUDIO_LEN=1600`) did not close the performance gap to Whisper.
  - increasing truncation budget from 8s to 15s did not help in this 200-step setup (A15_M0 underperformed A8_M0).

## 2026-04-19 — OmniASR 3B AJG reruns (Aja Benin token fix) [COMPLETED]

- Status: completed (LLM + CTC reruns with fresh prefixes, no overwrite)
- Goal:
  - rerun Omni 3B with correct Aja (Benin) language code `ajg_Latn`
  - preserve all prior `adj_Latn` runs and compare deltas
- Token confirmation:
  - `adj` = Adioukrou (Côte d'Ivoire), `ajg` = Aja (Benin)
  - Omni table rows differ (`adj_Latn` CER 3.2 vs `ajg_Latn` CER 6.5)
- Script updates:
  - `scripts/hf_jobs/omni_asr_finetune_recipe.py` default `LANGUAGE_CODE` switched to `ajg_Latn`
  - `scripts/hf_jobs/omni_asr_zeroshot.py` default/alias switched to `ajg_Latn`
- Runs:
  - **AJG LLM 3B R1** (`69e428fccd8c002f31dfef7b`) — completed
    - Prefix: `Omni_AJG_LLM3B_R1`
    - Key env: `LANGUAGE_CODE=ajg_Latn`, `MAX_AUDIO_SEC=8`, `MIN_AUDIO_LEN=1600`
    - Manifest stats: full keep (`train=1277`, `dev=160`, `test=160`)
    - Final script eval: dev WER `89.13`, CER `45.93`; test WER `89.57`, CER `44.57`
    - Eval meta: `expected_total=320`, `used_pairs=320`
  - **AJG CTC 3B R5_M2** (`69e43365cd8c002f31dfefdd`) — completed
    - Prefix: `Omni_AJG_CTC3B_R5_M2`
    - Key env: `LANGUAGE_CODE=ajg_Latn`, `MIN_AUDIO_LEN=32000`, `DATA_PARALLELISM=fsdp`, `SAVE_MODEL_ONLY=true`
    - Manifest stats: short-audio filtered (`train=1223`, `dev=155`, `test=155`)
    - Final script eval: dev WER `93.92`, CER `52.56`; test WER `96.29`, CER `51.75`
    - Eval meta: `expected_total=310`, `used_pairs=310`
- Retry history (kept for traceability):
  - CTC startup infra failures (`process device cpu`): `R1=69e428fcac288e522d8efe83`, `R2=69e42aeecd8c002f31dfef8d`, `R3=69e42d0dcd8c002f31dfefa1`
  - CTC `MIN_AUDIO_LEN=1600` validation crash at step 390 (`R4=69e42f05cd8c002f31dfefb9`):
    - `ValueError: ... seq_lens ... length ... is 0`
  - Stable completion obtained with `MIN_AUDIO_LEN=32000` (`R5_M2`)
- AJG vs prior `adj`-coded Omni 3B snapshot:
  - LLM (8s, min=1600): prior test WER/CER `88.40/44.11` -> AJG `89.57/44.57` (slightly worse)
  - CTC (min=32000): prior test WER/CER `91.55/47.95` -> AJG `96.29/51.75` (worse)
- Interpretation:
  - The language-code fix was necessary correctness-wise.
  - In this fine-tune setup, switching from `adj` to `ajg` did not improve metrics; it modestly worsened them.
  - This strengthens the conclusion that the large gap to paper-level numbers is mainly setup/data/eval mismatch, not a single token typo.

## 2026-04-19 — OmniASR 7B recipe scale-up (AJG) [COMPLETED]

- Goal:
  - push the same working Omni recipe path from 3B to the largest v2 checkpoints
  - keep prior 300M/3B runs untouched via fresh prefixes
  - determine whether 7B materially improves over the current Omni floor
- Source path:
  - launcher: `scripts/hf_jobs/omni_asr_finetune_recipe.py`
  - dataset: `JosueG/adja-tts-orpheus`
  - results repo: `JosueG/adja-asr-results`

- **CTC 7B**
  - `R1=69e4dc5ecd8c002f31dff619`
    - failed at startup with the known CPU-init path (`cudaGetDeviceCount` warning, fairseq2 process pinned to `cpu`)
    - canceled immediately
  - `R2=69e4dd98cd8c002f31dff621`
    - repeated the same CPU-init failure
    - canceled immediately
  - `R3=69e4de80cd8c002f31dff62b`
    - finally bound to `cuda:0` on `NVIDIA H200`
    - completed end-to-end
    - run path: `JosueG/adja-asr-results/Omni_AJG_CTC7B_R3_M2_ctc_1776606884/`
    - final eval:
      - dev WER `91.86`, CER `49.15`
      - test WER `92.48`, CER `47.63`
      - eval meta: `expected_total=310`, `used_pairs=310`
  - Interpretation:
    - the 7B CTC path is real and reproducible once HF hands back a healthy GPU init
    - it improves over AJG CTC 3B (`51.75 -> 47.63` CER on test)
    - it still does not beat Omni 3B LLM (`44.57` CER on test)

- **LLM 7B**
  - `R1=69e4dc45cd8c002f31dff617`
    - healthy single-H200 startup
    - failed with CUDA OOM during optimizer step
  - `R2=69e4e4c0cd8c002f31dff665`
    - retried with stricter memory knobs:
      - `MAX_AUDIO_SEC=4`
      - `MAX_NUM_ELEMENTS=50000`
      - `GRAD_ACCUM=64`
    - still failed with CUDA OOM during optimizer step
  - `R3=69e520beac288e522d8f009f`
    - attempted the move to `h200x2`
    - failed before training from a launcher packaging miss (`scripts/hf_jobs/omni_asr_finetune_recipe.py` missing in-container)
  - `R4=69e5213cac288e522d8f00a1`
    - new retry on `h200x2`
    - returned to the more faithful original settings:
      - `MAX_AUDIO_SEC=8`
      - `MAX_NUM_ELEMENTS=100000`
      - `GRAD_ACCUM=32`
    - explicit fix vs `R3`: submit through `--repo JosueG/hf-cli-jobs-uv-run-scripts` so the recipe launcher is uploaded
    - outcome: still failed with CUDA OOM during the Adam optimizer step
    - memory snapshot at failure showed ~`136 GiB` allocated by PyTorch on GPU 0, so `h200x2` FSDP was still not enough
  - `R5=69e52820ac288e522d8f00b1`
    - retry on `h200x4`
    - keeps the same core settings as `R4`:
      - `MAX_AUDIO_SEC=8`
      - `MAX_NUM_ELEMENTS=100000`
      - `GRAD_ACCUM=32`
    - adds `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
    - outcome: still failed with CUDA OOM during the Adam optimizer step
    - critical evidence: job log reported `Running on 1 process(es)` and `Device of the process set to cuda:0`, so the multi-GPU flavor was still launched as a single-rank job
  - `R6=69e52eccac288e522d8f00b9`
    - retry on `h200x4`
    - same core settings as `R5`
    - launcher fix: `scripts/hf_jobs/omni_asr_finetune_recipe.py` now detects visible GPUs and uses `python -m torch.distributed.run --standalone --nproc_per_node=<gpu_count> --module workflows.recipes.wav2vec2.asr ...` for `ddp/fsdp`
    - outcome: confirmed real multi-process launch instead of the old fake single-rank behavior
    - evidence:
      - `Detected 4 visible GPU(s) for training.`
      - `Launching distributed recipe with torch.distributed.run across 4 processes.`
      - `LOCAL_WORLD_SIZE=4, WORLD_SIZE=4`
      - fairseq2 created the 4-process data-parallel gang and completed FSDP wrap/broadcast
    - failure: rank `1` exited with code `1`, causing a `ChildFailedError`; no OOM signature appeared and the child traceback was not surfaced by default torchrun logging
  - `R7=69e53548cd8c002f31dff936`
    - final retry on `h200x4`
    - same core settings as `R6`
    - launcher instrumentation: distributed launch now adds `--log-dir <train_output_dir>/torchrun_logs --redirects 3 --tee 3`, and the wrapper prints `error.json` / `stderr.log` files into stdout on failure
    - outcome: completed successfully
    - run path: `JosueG/adja-asr-results/Omni_AJG_LLM7B_R7_h200x4_torchrunlogs_llm_1776629102/`
    - final script eval:
      - dev WER `85.71`, CER `44.84`
      - test WER `87.49`, CER `43.22`
      - eval meta: `expected_total=320`, `used_pairs=320`
    - key point: the successful run still uses the same 7B LLM recipe; the decisive fix was the real multi-process FSDP launch path

- Answer to the immediate memory-question:
  - yes, stricter memory settings can impact performance if they shrink the training distribution
  - reducing `MAX_AUDIO_SEC` can remove useful acoustic context
  - lowering `MAX_NUM_ELEMENTS` can change which sequence mixtures fit per optimizer step
  - increasing `GRAD_ACCUM` is less harmful because it mostly preserves effective batch size, but it does not restore lost context from shorter audio
  - in this specific case, the repeated OOMs plus `Running on 1 process(es)` showed the first blocker was the missing multi-process launch path
  - after fixing launch, the blocker briefly moved to a distributed child failure, and the rank-log instrumentation on `R7` gave a stable enough launch to finish
  - final result: Omni 7B LLM improved modestly over Omni 3B LLM (`44.57 -> 43.22` test CER) and is now the best Omni run in the repo, but it is still nowhere near the repo-best non-Omni ASR line

---

## Bugs encountered and fixed (chronological)

1. Python 3.9 type syntax (`str | None`) — added `from __future__ import annotations`
2. `torchaudio.sox_effects` not available on macOS — replaced with `torchaudio.functional.resample`
3. DataLoader `collate_fn` lambda not picklable — replaced with `functools.partial`
4. HF Jobs `--secret` flag → correct flag is `--secrets`
5. Whisper-large-v3 float16/float32 dtype mismatch — load with `torch_dtype=torch.float32`
6. CTC loss not implemented for Half — cast `log_probs.float()` before loss
7. MMS 1B OOM on A10G (24GB) — moved to A100 (80GB)
8. A100 CUDA driver 12.0.9 incompatible with PyTorch 2.8 — pinned `torch==2.5.1`
9. Manual CTC loss causing collapse — switched to model's built-in loss
10. Feature extractor collate: pre-padded arrays corrupted attention masks — pass variable-length arrays
11. `transformers>=4.50` requires `torch>=2.6` for torch.load — pinned `transformers<4.50`
12. MMS Ewe adapter `vocab_size=55` vs Adja 115 — set `model.config.vocab_size`
13. Early stopping patience=5 too aggressive — increased to 20

---

## TTS Runs

No TTS runs have been recorded in this ledger yet.

When a TTS cycle finishes, add:

- runtime: Colab T4 / Colab A100 / HF Jobs / Discovery
- exact package pins
- canonical entrypoint used
- success artifact or failure signature
- whether the issue was environment, model patching, preprocessing, training, or generation

Current first-success order:

1. `T1-csm-colab` — canonical Unsloth CSM notebook/script path
2. `T1-csm-vanilla` — short diagnostic without Unsloth patching
3. `T3-spark` — immediate fallback if CSM stays blocked

---

## 2026-04-18 — Qwen3 HF smoke submissions [FAILED BEFORE TRAINING]

- Requested by: Josue Godeme
- Documentation root: `results/qwen3_hf_jobs/README.md`
- Runbook: `experiments/finetuning-qwen3/docs/adja_runbook.md`
- Git branch / commit: `codex/qwen3-hf-jobs` / `0402b6a`
- Submitted jobs:
  - `QASR-s42` — Qwen3-ASR-0.6B smoke on HF Jobs | A10G | Job `69e30a7bac288e522d8efb5c`
  - `T5-qwen3-0p6b` — Qwen3-TTS-12Hz-0.6B smoke on HF Jobs | A10G | Job `69e30a8eac288e522d8efb5e`
- Notes:
  - These submissions use the Adja-specific materialization layer and vendored upstream Qwen3-TTS finetuning files.
  - Submission metadata and source references live in `results/qwen3_hf_jobs/README.md`.
  - Both jobs failed during container bootstrap, before dataset access or training began.
  - Exact failure signature from both HF logs: `fatal: could not read Username for 'https://github.com': No such device or address`.
  - Root cause: the launch command attempted `git clone --branch codex/qwen3-hf-jobs https://github.com/FrejusGdm/adja-nmt.git /workspace/adja-nmt` from inside the HF container, but that GitHub URL required authentication in the job environment.

## 2026-04-18 — Qwen3 HF smoke relaunches [FAILED AFTER REAL EXECUTION]

- Requested by: Josue Godeme
- Documentation root: `results/qwen3_hf_jobs/README.md`
- Runbook: `experiments/finetuning-qwen3/docs/adja_runbook.md`
- Git branch / commit: `codex/qwen3-hf-jobs` / `538d415`
- Launch correction:
  - switched from container bootstrap logic to the repo's normal `hf jobs uv run` pattern
  - made each Qwen launcher self-contained because HF uploads only the target script file
  - moved dependency resolution into inline `uv` script metadata instead of runtime `pip` / `uv pip`
- Intermediate failed retries:
  - `69e382dfac288e522d8efd10` / `69e382e1cd8c002f31dfe942` — missing local helper module `qwen3_job_utils`
  - `69e384d0cd8c002f31dfe952` / `69e384d1cd8c002f31dfe954` — `No module named pip`
  - `69e38514cd8c002f31dfe956` / `69e38515ac288e522d8efd1c` — ASR package version mismatch; TTS wrong interpreter context
  - `69e3856aac288e522d8efd1e` / `69e3856bcd8c002f31dfe964` — externally managed `/usr` interpreter rejected `uv pip --python ...`
- Submitted jobs:
  - `QASR-s42` — Qwen3-ASR-0.6B smoke on HF Jobs | A10G | Job `69e388ccac288e522d8efd26`
  - `T5-qwen3-0p6b` — Qwen3-TTS-12Hz-0.6B smoke on HF Jobs | A10G | Job `69e388cdcd8c002f31dfe98a`
- Notes:
  - These corrected launchers now match the same self-contained HF script pattern used elsewhere in this repo.
  - The upstream Qwen tree under `experiments/finetuning-qwen3/` remains preserved; the retry work was limited to Adja-specific launcher code and documentation.
  - The first repo-native `uv` metadata pair (`69e3865bac288e522d8efd24` / `69e3865dcd8c002f31dfe96e`) reached real training code:
    - ASR failed on a `TrainingArguments` API mismatch (`evaluation_strategy`)
    - TTS failed after the eager-attention fallback with a 2048/1024 tensor-size mismatch
  - The retry pair added an ASR compatibility fix and fuller TTS traceback logging.
  - ASR smoke itself succeeded: dataset materialization, 2-step train, checkpoint save, and sample decode all completed.
  - ASR still ended `ERROR` because the launcher closed the tee log handle before interpreter shutdown completed, producing `Exception ignored in sys.unraisablehook`.
  - TTS still ended `ERROR` because the real training path tried to add `2048`-wide text embeddings to `1024`-wide codec embeddings.

## 2026-04-18 — Qwen3 follow-up submissions [ASR COMPLETED, TTS A100 FAILED LATE]

- Requested by: Josue Godeme
- Documentation root: `results/qwen3_hf_jobs/README.md`
- Runbook: `experiments/finetuning-qwen3/docs/adja_runbook.md`
- Git branch / commit: `codex/qwen3-hf-jobs` / `538d415` plus uncommitted local launcher fixes
- Launch correction:
  - fixed the launcher teardown bug by restoring `stdout` / `stderr` before closing the tee log file
  - added explicit model/flavor/timeout overrides to `scripts/hf_jobs/submit_qwen3_jobs.py`
  - moved the next TTS retry from `0.6B` to `1.7B` because the vendored upstream fine-tuning script defaults to `Qwen/Qwen3-TTS-12Hz-1.7B-Base`
- Current submitted jobs:
  - `QASR-s42` — Qwen3-ASR-0.6B full pilot on HF Jobs | A10G | Job `69e38fe4ac288e522d8efd32`
  - `T5-qwen3-1p7b` — Qwen3-TTS-12Hz-1.7B-Base smoke on HF Jobs | A100 | Job `69e38fe3ac288e522d8efd30`
- Final outcomes:
  - `QASR-s42` full pilot `69e38fe4ac288e522d8efd32`: `COMPLETED`
  - `T5-qwen3-1p7b` A100 smoke `69e38fe3ac288e522d8efd30`: `ERROR`
- Notes:
  - The live `1.7B` smoke log reached real training in eager mode and no longer shows the old `2048` vs `1024` width mismatch; both reported dimensions are `2048`.
  - ASR full pilot materialized the full `1277 / 160 / 160` split, completed smoke + pilot, and saved final pilot checkpoint `checkpoint-320`.
  - The A100 `1.7B` smoke no longer had an architectural mismatch, but still failed after training when writing `sample.wav` with `soundfile.LibsndfileError: Format not recognised`.

## 2026-04-18 — Qwen3 TTS duplicate full run for faster testing [A100 FAILED LATE]

- Requested by: Josue Godeme
- Documentation root: `results/qwen3_hf_jobs/README.md`
- Runbook: `experiments/finetuning-qwen3/docs/adja_runbook.md`
- Reason:
  - user explicitly preferred faster testing over conservative compute usage
  - the `1.7B` smoke had already entered real training, so launching the full run in parallel could save wall-clock time if the path stays valid
- Current submitted job:
  - `T5-qwen3-1p7b-full` — Qwen3-TTS-12Hz-1.7B-Base full pilot on HF Jobs | A100 | Job `69e3942dac288e522d8efd40`
- Final status:
  - `ERROR`
- Important tradeoff:
  - this full run shares the same output repo and results-repo prefix as the `1.7B` smoke run, so the last finisher can overwrite repo-level artifacts
- Failure:
  - same late-stage sample-write bug as the A100 smoke run: `soundfile.LibsndfileError: Format not recognised`

## 2026-04-19 — QASR-s42 held-out evaluation after full pilot [COMPLETED]

- Requested by: Josue Godeme
- Documentation root: `results/qwen3_hf_jobs/README.md`
- Runbook: `experiments/finetuning-qwen3/docs/adja_runbook.md`
- Local branch / commit at scoring patch time: `codex/omni-7b-finetune` / `df69aa1`
- Eval job: `69e4db9cac288e522d8f001f`
- Platform: HF Jobs | GPU: A10G | Model: `JosueG/qwen3-adja-asr-0p6b`
- Final metrics:
  - raw WER: `100.0%`
  - raw CER: `53.14%`
  - normalized WER: `100.65%`
  - normalized CER: `52.26%`
- Decode time: `106.32s` for `160` held-out utterances
- Decode samples:
  - `REF: Ŋu nya kpɔ́kpɔ a ?` → `HYP: ŋu a kpɔ kpɔ a ?`
  - `REF: Ganɛni mɛ ayi ɔ̀ ?` → `HYP: Ganɛn mi mɛn ayi ɔ ?`
  - `REF: Je agbla do ŋukeki alo azan nyui wo` → `HYP: Je agbe agbe nɔn koki alo azan nɔn lɔ`
  - `REF: Tom zɔnɔ afɔ iyi nɔ yidɔmɛ` → `HYP: Tom zɔnɔ afɛn yi yi yi yi yi yi yi yi yi yi yi`
- Notes:
  - Eval attempts 1-3 failed on input-interface mismatches (`dict`, `torchcodec`, `numpy.ndarray`).
  - Eval attempt 4 matched the repo-proven Qwen inference path by materializing 16 kHz WAV files and passing file paths into `transcribe()`.
  - Result quality is poor, but the model is not outputting pure garbage; it preserves some Adja-like orthography and phonetic shape.

## 2026-04-19 — Qwen3 TTS 1.7B patched L40 reruns [COMPLETED]

- Requested by: Josue Godeme
- Documentation root: `results/qwen3_hf_jobs/README.md`
- Runbook: `experiments/finetuning-qwen3/docs/adja_runbook.md`
- Local branch / commit at patch time: `codex/omni-7b-finetune` / `df69aa1`
- Why:
  - fix the late-stage `soundfile.LibsndfileError` from the A100 jobs
  - move off `a100-large` because queue time was worse than `l40sx1`
- Completed jobs:
  - smoke rerun `69e4da24ac288e522d8f001a` | `l40sx1` | `COMPLETED`
  - full rerun `69e4da24ac288e522d8f0018` | `l40sx1` | `COMPLETED`
- Smoke rerun result:
  - matching `2048` / `2048` embedding widths in eager mode
  - first observed loss: `10.4225`
  - wrote `checkpoint-epoch-0`, `sample.wav`, and `sample.json`
- Full rerun result:
  - matching `2048` / `2048` embedding widths in eager mode
  - training loss observed from `9.0878` at step 0 down to `4.7412` by step 630
  - wrote smoke artifacts, pilot sample metadata, and final pilot checkpoint `checkpoint-epoch-0`
- Notes:
  - The April 19 patch unwrapped the model output into a real 1-D float waveform and wrote it with explicit `format=\"WAV\"`.
  - This resolves the earlier late-stage generation/export failure.
  - No human listening verdict is recorded yet for the Qwen TTS outputs, so this counts as an end-to-end systems success, not yet an audio-quality success.

---

## 2026-04-17 — T1-diagnostic — Sesame CSM 1B vanilla pipeline validation

- Status: completed (first-success on vanilla path)
- Platform: HF Jobs | GPU: L40S (48GB) | Job: `69e2fe29cd8c002f31dfe3c1` | Wall time: ~2 min
- Train loss: 16.04 → 16.05 (5 samples × 2 steps, no learning expected)
- Storage: `JosueG/adja-tts-results/T1_diagnostic/`
- Notes: **No Unsloth.** Uses vanilla `CsmForConditionalGeneration` + `peft.LoraConfig`. Proved pipeline works. Generated plain + speaker-conditioned wav files. Critical fix: load model in fp32, let Trainer handle bf16 mixed-precision (loading in bf16 causes CSM internal `index_put_` dtype mismatch).

## 2026-04-17 — T1-baseline — Sesame CSM 1B full 120-step run

- Status: completed
- Platform: HF Jobs | GPU: L40S | Job: `69e3053eac288e522d8efb53` | Wall time: 2.8 min
- Train loss: 19.95 → 18.61 | LoRA r=32, lr 2e-4 linear
- Storage: `JosueG/adja-tts-results/T1/`
- Notes: Dropped 43 clips >10s via `MAX_AUDIO_SAMPLES=240001` filter (collator otherwise crashes with `expected sequence of length 240001 at dim 2 (got 339840)`). Peak VRAM 15.6 GB. Audio noise/Mimi artifacts — 120 steps is nowhere near converged.

## 2026-04-18 — T1-long-20ep-es — Sesame CSM 1B overnight with early stopping

- Status: completed (early stopped at epoch 7)
- Platform: HF Jobs | GPU: L40S | Job: `69e37748ac288e522d8efcdc` | Wall time: 24.7 min
- Train loss end: ~17.4 | Best dev loss: **6.488** at epoch 3.85 (step 300)
- Storage: `JosueG/adja-tts-results/T1_long_20ep_earlystop_2026-04-18/`
- Notes: First submission `69e309c6cd8c002f31dfe427` crashed at first eval with `KeyError: 'eval_loss'`. Root cause: PeftModel hides base forward, Trainer can't infer labels, `label_names=[]`, eval loop skips `compute_loss`. Fix: explicit `label_names=["labels"]` in TrainingArguments. Resubmitted cleanly. Early stopping patience-5 fired after 5 consecutive eval regressions (6.488 → 6.785). Load-best-model-at-end restored epoch-3.85 checkpoint for generation. Audio still noise — dataset likely insufficient for language adaptation (1.7h audio for English→Adja is 3-5x below typical).

---

## 2026-04-18 — T3-spark — Spark TTS 0.5B finetune attempt 1 (FAILED)

- Status: ERROR
- Platform: HF Jobs | GPU: L40S | Job: `69e38873cd8c002f31dfe97e` | Wall time: ~3 min to failure
- Failure: torchaudio/torch binary mismatch (`undefined symbol: _ZNK5torch8autograd4Node4nameEv`). Unsloth latest pulled torch 2.10 but container's torchaudio was 2.6.
- Next: resubmitted with torchaudio reinstall fix

## 2026-04-18 — T1-full-ft — Sesame CSM 1B FULL fine-tune (capacity hypothesis REFUTED)

- Status: COMPLETED
- Platform: HF Jobs | GPU: L40S | Job: `69e3856e` | Wall time: 30.1 min | Peak VRAM: 34.6 GB
- Training: 20 epochs max, early stopped at epoch 5.13, 1.6B trainable params (100%)
- Best dev loss: **6.496** (at epoch 1.3) — essentially identical to LoRA r=32 (6.488)
- Train loss: noisy 22–28 (full FT less stable than LoRA)
- Generation quality: **Noise** — same as all CSM variants
- Conclusion: Unlocking 100% of parameters did absolutely nothing. The ceiling for CSM on Adja is representational (Mimi codec + Llama tokenizer), not param count. This result was the key control that proved the Spark win is about architecture priors, not capacity.

## 2026-04-18 — T3-spark — Spark TTS 0.5B 120-step run (SUCCESS — intelligible Adja)

- Status: COMPLETED — **best TTS result to date**
- Platform: HF Jobs | GPU: L40S | Job: `69e3987d`
- Training: 120 steps, < 1 epoch (1.4 min wall time)
- Train loss: 7.2 → 6.43
- Generation quality: **INTELLIGIBLE — native speaker (Josue) confirmed coherent Adja words: "so, so good. I can understand words. It actually produces coherent words."**
- Why it worked: BiCodec uses wav2vec2-XLSR-53 semantic tokens (multilingual, 53 languages incl. African). Qwen2 tokenizer handles Adja diacritics natively. Both contrast with CSM's Mimi (English-heavy codec) + Llama BPE (fragments ɛ/ɔ/ŋ/ɖ to bytes). Full analysis: `research-paper-exploration/paper-one/why-spark-worked.md`.
- Note: attempt 2 job `69e39023ac288e522d8efd34` was a separate 20ep run that was the precursor; the 120-step diagnostic run on job `69e3987d` was the key result.

## 2026-04-18 — T3-spark — Spark TTS 0.5B 20-epoch eval+early-stop run (OUTCOME NOT LOGGED)

- Status: submitted; outcome not recorded
- Platform: HF Jobs | GPU: L40S | Job: `69e39bc2`
- Notes: Submitted after 120-step success. Whether longer training improved or plateaued like CSM is still an open question. Check Hub: `JosueG/adja-tts-results/T3_spark_20ep_2026-04-18/`.

## 2026-04-18 — T6-mms-ewe — MMS-TTS-Ewe → Adja finetune via ylacombe/finetune-hf-vits

- Status: running
- Platform: HF Jobs | GPU: L40S | Job: `69e390f1cd8c002f31dfe9d4` | Timeout: 4h
- Storage target: `JosueG/adja-tts-results/T6_mms_ewe_20ep_2026-04-18/`
- Storage side-effect: pushes `JosueG/adja-tts-mms-ready` (private) with NFC-normalised text + 16 kHz audio + 80/10/10 splits
- Notes: VITS + GAN training via ylacombe's `run_vits_finetuning.py` (accelerate launch). Converts Ewe discriminator on-the-fly. Hypothesis: starting from a model that already speaks Ewe (Gbe-family) should dramatically outperform English-centric TTS bases (CSM, Orpheus, Spark) on 1.7h of Adja.
- License note: MMS base is CC-BY-NC 4.0. Research use only.

## 2026-04-18 — T2-diagnostic-v2 — Orpheus 3B dry-run gate (passed)

- Status: completed
- Platform: HF Jobs | GPU: L40S | Job: `69e3c450cd8c002f31dfeb72` | Wall time: ~5 min
- Best dev loss: n/a (2 train steps, 1 eval — diagnostic only)
- Storage: `JosueG/adja-tts-results/T2_diagnostic_2026-04-18_v2/`
- Notes: Dry-run gate passed end-to-end. snac/transformers/peft/trl installed cleanly, 3.8B base downloaded, 5-sample preprocess + 2 train steps + 1 eval + 2 WAV generations completed, adapter + tokenizer + WAVs uploaded to Hub. Final log line: "T2 canonical training finished." Gate cleared; 5-variant capacity sweep fired immediately after.

## 2026-04-18 — T2-orpheus-r32 — Orpheus 3B LoRA r=32 (20 ep, L40S)

- Status: completed (early stopped)
- Platform: HF Jobs | GPU: L40S | Job: `69e3c68dcd8c002f31dfeb91` | Wall time: 26.8 min
- Best dev loss: **5.4976** at epoch 5.00
- Storage: `JosueG/adja-tts-results/T2_orpheus_en_lora_r32_20ep_2026-04-18/`
- Hyperparameters: English base `canopylabs/orpheus-3b-0.1-ft`, LoRA r=32, lr 2e-4, 20 epochs max, early stopping patience 5
- Generation quality: **unintelligible noise** — user listened 2026-04-18, no identifiable Adja words or phonology
- Notes: Mirrors T1's LoRA rank on the 3.8B backbone. Beats T1 CSM (6.488) by ~1.0 nat but audio still unintelligible. Data bottleneck confirmed.

## 2026-04-18 — T2-orpheus-r64 — Orpheus 3B LoRA r=64 (20 ep, L40S)

- Status: completed (early stopped)
- Platform: HF Jobs | GPU: L40S | Job: `69e3c68eac288e522d8efd97` | Wall time: 21.6 min
- Best dev loss: **5.4823** at epoch 3.75
- Storage: `JosueG/adja-tts-results/T2_orpheus_en_lora_r64_20ep_2026-04-18/`
- Hyperparameters: English base, LoRA r=64 (canonical mid-rank), lr 2e-4, 20 epochs max
- Generation quality: **unintelligible noise** — user listened 2026-04-18, no identifiable Adja words or phonology
- Notes: Canonical T2 variant. Clean monotonic improvement over r=32 (5.4823 vs 5.4976). Audio still unintelligible despite better fit.

## 2026-04-18 — T2-orpheus-r128 — Orpheus 3B LoRA r=128 (20 ep, L40S)

- Status: completed (early stopped)
- Platform: HF Jobs | GPU: L40S | Job: `69e3c68fac288e522d8efd99` | Wall time: 17.8 min
- Best dev loss: **5.4596** at epoch 2.82
- Storage: `JosueG/adja-tts-results/T2_orpheus_en_lora_r128_20ep_2026-04-18/`
- Hyperparameters: English base, LoRA r=128, lr 2e-4, 20 epochs max
- Generation quality: **unintelligible noise** — user listened 2026-04-18, no identifiable Adja words or phonology
- Notes: Best LoRA variant. Monotonic improvement over r=64 continues (5.4596 vs 5.4823). Confirms capacity ordering is clean but better fit does not produce audible Adja.

## 2026-04-18 — T2-orpheus-fullft — Orpheus 3B FULL fine-tune (10 ep, A100-large)

- Status: completed (catastrophic divergence caught by early stopping)
- Platform: HF Jobs | GPU: A100-large (80GB) | Job: `69e3c692ac288e522d8efd9b` | Wall time: 23.3 min
- Best dev loss: **5.4242** at epoch 1.88 (best across all TTS runs to date)
- Storage: `JosueG/adja-tts-results/T2_orpheus_en_fullft_10ep_2026-04-18/`
- Hyperparameters: English base, full fine-tune (no PEFT), lr 5e-5, 10 epochs max
- Generation quality: **unintelligible noise** — user listened 2026-04-18, no identifiable Adja words or phonology
- Notes: Peaked at epoch 1.88 then diverged catastrophically to dev loss 9.5+ by epoch 3.13 — classic small-data full-FT failure. Early stopping caught it. Best numerical result in TTS leaderboard but audio indistinguishable from LoRA variants. Signals data is the ceiling, not capacity.

## 2026-04-18 — T2-orpheus-fr-r64 — Orpheus 3B French base LoRA r=64 (20 ep, L40S)

- Status: completed (early stopped)
- Platform: HF Jobs | GPU: L40S | Job: `69e3cba3cd8c002f31dfebdc` | Wall time: 21.4 min
- Best dev loss: **5.5058** at epoch 3.75
- Storage: `JosueG/adja-tts-results/T2_orpheus_fr_lora_r64_20ep_2026-04-18/`
- Hyperparameters: French base `canopylabs/3b-fr-ft-research_release`, LoRA r=64, lr 2e-4, 20 epochs max
- Generation quality: **unintelligible noise** — user listened 2026-04-18, no identifiable Adja words or phonology
- Notes: Direct comparison to English r=64 at matched hyperparameters. French base lost by 0.02 nat (5.5058 vs 5.4823). Hypothesis falsified: French TTS prior does NOT transfer better to Adja than English prior. Adja is Gbe/Niger-Congo; French proximity is sociolinguistic, not phonological. Phonological proximity to Ewe matters more — see T6.

## 2026-04-18 — Multilingual TTS expansion tracks landed [READY]

- Status: ready (code + docs landed; smoke submissions pending from this branch)
- Platform target: HF Jobs, primarily A100 80GB for smoke validation
- New experiment roots:
  - `experiments/tts/T7_voxcpm_finetune/`
  - `experiments/tts/T8_ims_toucan_finetune/`
  - `experiments/tts/T9_xtts_v2_finetune/`
  - `experiments/tts/T10_f5_e2_tts_finetune/`
- New canonical launchers:
  - `scripts/hf_jobs/T7_voxcpm_finetune.py`
  - `scripts/hf_jobs/T8_ims_toucan_finetune.py`
  - `scripts/hf_jobs/T9_xtts_v2_finetune.py`
  - `scripts/hf_jobs/T10_f5_e2_tts_finetune.py`
- Strategy doc:
  - `docs/multilingual-speech-strategy-2026-04-18.md`
- Notes:
  - **T7 VoxCPM** materializes Adja to upstream JSONL manifests and auto-detects sample rate from the base model config.
  - **T8 IMS-Toucan** uses direct ISO code `ajg`, relying on Transphone fallback where eSpeak support is incomplete.
  - **T9 XTTS-v2** materializes Adja into Coqui's `metadata.csv + wavs/` structure for the public GPT-encoder recipe.
  - **T10/T11 F5/E2** materialize Adja into `raw.arrow + duration.json + vocab.txt` and expand text embeddings for unseen Adja characters.

## 2026-04-18 — Multilingual TTS smoke submissions [MIXED]

- Requested by: Josue Godeme
- Git branch / commits:
  - `cursor/add-multilingual-tts-asr-tracks-652c` / `f446840`
  - follow-up runtime fixes: `f581727`, `72eb27d`
- Submitted smoke jobs:
  - `T8-ims-toucan` retry 1: `69e3e3e3ac288e522d8efdfb` — failed before execution due to broken HF command shape (`bash` command missing `--` separator)
  - `T9-xtts-v2` retry 1: `69e3e3e3ac288e522d8efdfd` — same command-shape issue / no useful logs
  - `T8-ims-toucan` retry 2: `69e3e521ac288e522d8efe01` — got into real launcher execution, then failed on upstream dependency resolution conflict
  - `T9-xtts-v2` retry 2: `69e3e521ac288e522d8efe03` — got through install + dataset materialization, then failed inside Coqui XTTS import path (`BeamSearchScorer` export mismatch)
  - `T7-voxcpm`: `69e3ea21cd8c002f31dfed1e` — **completed real dry-run training**
  - `T10-f5`: `69e3e9dbcd8c002f31dfed12` — launcher completed dataset prep, then failed on upstream EMA deepcopy (`cannot pickle '_thread._local' object`)
  - `T11-e2`: `69e3e9dccd8c002f31dfed14` — same upstream EMA/deepcopy failure after dataset prep
  - `T8-ims-toucan` retry 3: `69e3ed35cd8c002f31dfed3e` — moved further after adding `matplotlib`, then later errored again
  - `T10-f5` retry 2: `69e3ed35cd8c002f31dfed40` — `SCHEDULING` at last poll
  - `T11-e2` retry 2: `69e3ed33cd8c002f31dfed3c` — `SCHEDULING` at last poll
- Concrete outcomes:
  - **T7 VoxCPM** succeeded end-to-end for the smoke definition: real train steps, validation, sample audio generation, checkpoint upload, and metrics write.
  - **T10/T11 F5/E2** validated the Adja data path, but the trainer died before updates because upstream EMA used `deepcopy(model)`.
  - **T8 IMS-Toucan** validated the Adja preprocessing path; the remaining blocker was import-time dependency compatibility.
  - **T9 XTTS-v2** validated the Coqui data materialization path; the remaining blocker was Coqui/Transformers API compatibility.

## 2026-04-19 — Multilingual TTS retry loop [MIXED, IMPROVING]

- Git branch / commits:
  - `cursor/add-multilingual-tts-asr-tracks-652c` / `8786040`
  - follow-up Toucan/Vox fixes: `957f625`
- Confirmed completions:
  - `T7-voxcpm` longer LoRA pilot: `69e4261ecd8c002f31dfef6b` — **COMPLETED**
  - `T7-voxcpm` audio-upload smoke retry: `69e40c12cd8c002f31dfee6e` — **COMPLETED**
  - `T10-f5` retry 6: `69e42b13ac288e522d8efe87` — **COMPLETED**
  - `T11-e2` retry 6: `69e42b13ac288e522d8efe8b` — **COMPLETED**
- New/recent retries:
  - `T9-xtts-v2` retry 3: `69e42b13ac288e522d8efe85` — still `SCHEDULING` at last poll
  - `T8-ims-toucan` retry 6: `69e42b13ac288e522d8efe86` — moved into real preprocessing, then exposed a new `phonepiece` asset race
- Concrete findings:
  - **T10 F5 / T11 E2** now genuinely enter training and log checkpoint saves.
  - **T8 Toucan** cleared the earlier Transphone blocker and reached dataset cache build; the new blocker became missing `phonepiece` assets.
  - **T7 VoxCPM** longer run finished cleanly but still lacked listenable Hub audio because the post-training export path failed before upload.

## 2026-04-19 — VoxCPM fair-evaluation full fine-tune submitted [RUNNING]

- Requested by: Josue Godeme
- Reason:
  - the completed Vox smoke and short L40S LoRA pilot were enough to prove startup and checkpointing,
    but not enough to judge the architecture fairly against stronger local Spark results
- Submitted job:
  - `T7-voxcpm-fullft` — VoxCPM2 **full fine-tune** on HF Jobs | A100 | Job `69e43943ac288e522d8efea4`
- Command shape:
  - `--full-finetune --learning-rate 1e-5 --num-iters 1000 --valid-interval 50 --save-interval 100`
- Goal:
  - run a materially stronger VoxCPM experiment than the earlier short LoRA pilot so quality can be judged on fairer terms
- Status at ledger update time:
  - `RUNNING`
- Related completed Vox jobs:
  - short L40S LoRA pilot `69e4261ecd8c002f31dfef6b` — completed
  - patched audio-export smoke `69e42c72ac288e522d8efe91` — completed with uploaded WAV files under
    `T7_voxcpm_smoke_audio_retry2_2026-04-19-011425/generated_audio/`

## 2026-04-18 — D5 Parakeet TDT 0.6B v3, HF Jobs pilot (l40sx1, fused tdt loss)

- **Job:** `69e406c9ac288e522d8efe49` (l40sx1, `pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel`, `RNNT_LOSS_NAME=tdt`, 20 epochs, batch 8, lr 1e-4, warmup 500)
- **Status:** COMPLETED — 30.5 min training + 4 s transcribe on 160 test utts + 5 min Hub upload.
- **Metrics (dev):** best `val_wer` = **0.6535** at epoch 17. Monotonic descent 0.996 → 0.654 with a couple of plateaus.
- **Metrics (test):** Mean/Median WER & CER = 100%. Corpus WER/CER = NaN.
- **Why the flat-100% test metric:** decoder hyps are saturated with `⁇` (SentencePiece unknown-token byte fallback). Parakeet's tokenizer was trained on English text only; Adja characters (`ɛ ɔ ŋ ɖ`, tone marks) have no direct subword and collapse to unk. The encoder adapted well (dev WER fell 34 pp), but the output alphabet is wrong.
- **Sample decodes:**
  - `REF: Ŋu nya kpɔ́kpɔ a ?` → `HYP: ⁇  nya kp ⁇  kp ⁇  a?`
  - `REF: ŋnyan go. Kpɔ ŋuɖejikɔ a nyan wo` → `HYP: T ⁇ n ⁇ u fini a  ⁇ eka kp ⁇  de jiji ⁇   ⁇`
- **Infrastructure lessons (see `results/parakeet_hf_jobs/README.md` for full context):**
  - UV Bookworm image + numba NVVM is a dead end — switched to `pytorch:2.5.1-cuda12.1-cudnn9-devel` with real NVCC/libnvvm. Fused `tdt` loss worked first try on that image.
  - First pilot (`69e3d5b4ac288e522d8efdd1`) crashed with `FileNotFoundError: 'git'` — NeMo's `exp_manager` calls `git rev-parse HEAD` for run tagging, and the PyTorch image ships without git. Fix: `apt-get update -qq && apt-get install -y -qq git &&` prepended to bootstrap.
  - Eval aggregation NaN on corpus WER is a jiwer zero-division when all hyps contain unk tokens — clean up in the eval helper alongside the tokenizer retrain.
- **Storage:** `JosueG/adja-asr-results/D5_l40_parakeet/` (metrics.json, evaluation_summary.txt, test_results.csv, 2.51 GB `parakeet_tdt_D5_l40.nemo`)
- **Next step:** retrain SentencePiece on Adja text + `model.resize_token_embeddings(...)`, analogous to the `aj_Latn` fix in `learnings-from-the-past/`. Without that, any further tuning is wasted.

## 2026-04-18 — D5 Parakeet TDT 0.6B v3 — tokenizer retrain pilot (D5_l40_tokv1)

- **Job:** `69e437a7cd8c002f31dff009` (l40sx1, `pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel`, `RNNT_LOSS_NAME=tdt`, `RETRAIN_TOKENIZER=1`, `VOCAB_SIZE=1024`, 20 epochs, batch 8, lr 1e-4, warmup 500)
- **Status:** COMPLETED — 20.0 min training + eval + Hub upload (2.47 GB `.nemo`).
- **Tokenizer swap pipeline (verified end-to-end):**
  - Downloaded corpus from `JosueG/adja-text-corpus/extra_adja_text.txt` (private, 12,050 NFC-normalized Adja sentences curated during prior char-LM work — source: LREC data-paper Translation col + simple-dataset-enriched adja_translation col, deduped).
  - Merged with 1,277 train-split transcripts → 13,327 training sentences for SentencePiece.
  - Trained SP BPE `vocab_size=1024, character_coverage=1.0, model_type=bpe, normalization=identity`.
  - `asr_model.change_vocabulary(tokenizer_dir, "bpe")` rebuilt the joint network: `num_classes_with_blank` went from 8198 (8192 SP + 1 blank + 5 durations) → 1030 (1024 SP + 1 blank + 5 durations).
  - `_swap_rnnt_loss` then instantiated a fresh fused `tdt` RNNT-Loss with `num_classes=1024` to match.
- **Metrics (dev):** best `val_wer` = **0.9544** at epoch 12. Trajectory: 1.000 (ep 0) → 0.996 (ep 2) → 0.968 (ep 10) → 0.961 (ep 11) → 0.954 (ep 12) → flat through ep 19.
- **Metrics (test):** per-sentence Median WER = 100%, Corpus WER = NaN.
- **Sample decodes (test):**
  - `REF: Ŋu nya kpɔ́kpɔ a ?` → `HYP: E`
  - `REF: ŋnyan go. Kpɔ ŋuɖejikɔ a nyan wo` → `HYP: E yi le le wo`
  - `REF: Ŋ nyanyɔ mɔ wo anu ahán !` → `HYP: E`
- **Headline finding — the tokenizer-swap works mechanically but the recipe is wrong for this condition:** the decoder now emits Adja characters (`E`, `yi`, `le`, `wo`) instead of `⁇`, confirming vocabulary coverage. But hypotheses collapse to 1–4 tokens because the freshly re-initialised joint + prediction-network embedding cannot converge in 20 epochs at LR 1e-4 on only 1,277 utterances. Pilot v2 (no retrain) had a trained decoder but no Adja chars; pilot v3 (retrain) has Adja chars but a decoder that hasn't learned to emit them in sequence. **Dev WER regressed +30 pp** (0.6535 → 0.9544). Neither baseline is a shipping model.
- **Recommended next pilots (candidates for D5b/D5c):**
  1. **Longer training + higher LR on fresh head** — 80–100 epochs at LR=5e-4, longer warmup. Most likely payoff.
  2. **Staged freeze** — freeze encoder for 10 ep to let decoder catch up on frozen features, then unfreeze at reduced LR.
  3. **Smaller vocab** — 256 or 512. With 13k sents many of the 1024 subwords are rare.
  4. **Differential LR** — 5× on joint + prediction net, 1× on encoder.
- **Storage:** `JosueG/adja-asr-results/D5_l40_tokv1_parakeet/` (metrics.json, evaluation_summary.txt, test_results.csv, `parakeet_tdt_D5_l40_tokv1.nemo`). Archived logs: `results/parakeet_hf_jobs/logs/69e437a7_pilot_v3.txt`.
- **Thesis documentation:** `docs/parakeet-tokenizer-retrain-adja.md` (full rationale, corpus provenance, hyperparameter justification, and §9 results/analysis — written for direct lift into the write-up).
- **Code changes in this run:**
  - `scripts/hf_jobs/parakeet_finetune.py`: added `RETRAIN_TOKENIZER / VOCAB_SIZE / TOKENIZER_CORPUS_REPO` env vars, `_retrain_tokenizer_and_swap` helper (SentencePiece train + `change_vocabulary`), and reordered so `_swap_rnnt_loss` runs after vocabulary change (joint gets rebuilt).
  - `data/extra_adja_text.txt` moved to `.gitignore`; mirrored to private Hub `JosueG/adja-text-corpus`.
  - `docs/parakeet-tokenizer-retrain-adja.md` created.

## 2026-04-26 — SageMaker bootstrap: HPC migration + new MMS-Ewe-adapter baseline (S1–S7)

- **Trigger:** Discovery HPC has been failing repeatedly — 40 GB MIG slice instead of full 80 GB A100, Apptainer/HF cache thrash, queue waits costing days per crash. HF Jobs has been reliable for everything we've shipped. Decision: stop fighting HPC; migrate all queued/planned ASR + audio-LM jobs to AWS SageMaker (where I have credits) on a single, unified launcher.
- **Plan doc:** `<LOCAL_PATH>` — verified against AWS docs (SDK pin, DLC tag, instance types, env vars, FastFile mode). SageMaker Python SDK v3 (released 2025-11-19) drops the `HuggingFace` estimator, so the launcher pins `sagemaker<3.0.0`.
- **Files touched:**
  - `scripts/sagemaker_jobs/launch.py` — extended with S1–S7 entries, `extra_hyperparameters` field for shared scripts, instance prices for ml.g5.xlarge / ml.g5.2xlarge / ml.g5.4xlarge / ml.g6e.xlarge / ml.p5.48xlarge, and the `sagemaker<3.0.0` pin in the docstring.
  - `scripts/sagemaker_jobs/train_S1_whisper_largev3_lora.py` — Whisper-large-v3 LoRA (Ewe → Adja two-stage). Ported from `hpc/scripts/large/whisper_largev3_ewe_hpc.py` with PEFT LoRA always on (r=32, q_proj/v_proj). HF Hub at runtime; SM_MODEL_DIR / SM_OUTPUT_DATA_DIR for outputs.
  - `scripts/sagemaker_jobs/train_S7_whisper_tiny_adja.py` — direct Adja FT (no Ewe transfer, no LoRA). Closes the size ablation low end vs C-small (CER 27.02%). Single-stage protocol matches C-small for apples-to-apples.
  - `scripts/sagemaker_jobs/train_S4_mms_ewe_adapter_adja.py` — **NEW experiment, no HPC source.** Loads `facebook/mms-1b-all` with `target_lang="ewe"` (Meta's per-language adapter), swaps `lm_head` for an Adja CTC vocab built from training data, optional `--freeze-base-encoder`. Cheap reputable baseline — no SSL pretrain needed.
  - `scripts/sagemaker_jobs/train_S2_S3_wav2vec2_ssl_ewe.py` — single script for both S2 (XLS-R 1B) and S3 (MMS-1B). Two-stage: contrastive SSL on WaxalNLP `ewe_asr` unlabeled (~183k) → CTC FT on Adja. `waxalnlp_loader` inlined (handles `__index_level_0__` schema cast + ffmpeg MP3 bytes decode).
  - `scripts/sagemaker_jobs/train_S6_audio_lm_orpheus_ewe.py` — Orpheus 3B audio-LM port from `hpc/scripts/pretraining/audio_lm_orpheus_ewe.py`. Stage A: next-token LM on SNAC codes (Ewe). Stage B: text→audio on Ewe + Adja TTS combined. LoRA r=64, `enable_input_require_grads()` after `get_peft_model()`.
- **Deliberately not (yet) created:** `train_S5_audio_lm_csm_ewe.py` (CSM/Mimi parallel of S6) — listed as `planned` in the registry; port pending.
- **Verified-against-docs decisions** (URL-cited in plan): DLC `huggingface-pytorch-training:2.5.1-transformers4.49.0-gpu-py311-cu124-ubuntu22.04`; FastFile input mode for audio (no manifest, S3 prefix only); `ml.g5.2xlarge` for Whisper LoRA per AWS Whisper-LoRA blog; `ml.p5.4xlarge` (1× H100, GA 2025-08-27) instead of `ml.p4d.24xlarge` for SSL-1B / audio-LM.
- **Existing scaffolding reused:** the launcher pre-existed (created 2026-04-26 morning) for M*/CF*/TK1/AT1 HF-Jobs-shaped experiments; this wave plugged into it via the `EXPERIMENT_CONFIGS` dict so HPC-migrated jobs use the same submission entry point.
- **Artifact preservation:** SageMaker auto-uploads `SM_MODEL_DIR` to `s3://<bucket>/adja/<exp-id>/output/model.tar.gz`. For S1/S4 (LoRA adapters <200 MB) save all epochs; for S2/S3/S5/S6 save best+last only. For S1 ASR best, also save merged (`merge_and_unload`) `transformers`-loadable directory so the cascade pipeline can `AutoModel.from_pretrained()` without PEFT installed.
- **What this unblocks:** every previously HPC-bound experiment can now ship without further SLURM debugging. SLURM scripts under `hpc/slurm/` are kept intact in case Discovery recovers a stable full-A100 allocation.
- **Next steps:**
  1. Local `--smoke` validation per script before any SageMaker submission (matches the project's "code must work before HPC submission" rule).
  2. First real submission: S4 (MMS-Ewe-adapter) — cheapest, fastest to a real result, validates the launcher end-to-end. Then S7 (Whisper-tiny) in parallel.
  3. After S4/S7 smoke pass on SageMaker: submit S1 (Whisper-large LoRA), then S2/S3, then S6.
  4. Port S5 (CSM audio-LM) and submit alongside S6 once S6 is healthy.
