# Wave-3 Recovery — Failure Modes, Fixes, and Comparison Fairness

**Date:** 2026-04-27 evening EDT.
**Audience:** future-me writing thesis chapters / paper / experiment-section tables.
**Why this doc exists:** the day was chaos — 30+ failed jobs, multiple fix waves, three new failure modes diagnosed in one evening. **Three months from now I will not remember why we changed `max_seq_length` from 1024 to 768 or why AT1's first 4 attempts don't count.** This is the single source of truth for that.

Cross-references:
- [results/run-ledger.md](../results/run-ledger.md) — chronological log
- [experiments/registry.md](../experiments/registry.md) — current state per experiment
- [learnings-from-the-past/sagemaker-gotchas.md](../learnings-from-the-past/sagemaker-gotchas.md) — generalized gotcha index (#14, #15, #16 added in wave-3)

---

## 1. Failure modes diagnosed in wave-3

Three new compounding failure modes on top of the morning's volume-size + WaxalNLP-schema fixes.

### 1.1 Disk-full silent hang (gotcha #14)

- **First seen:** `adja-at1-full-20260426-114103` — 24 hours frozen at "Downloading data 39/106" with no traceback.
- **Cause:** SageMaker default volume = 30 GB; HF datasets writes raw shards + extracted Arrow → ~112 GB on disk for WaxalNLP `ewe_asr/unlabeled` (56 GB raw). Disk filled, `urllib3` blocked on `f.write(chunk)` (`ENOSPC`), HF datasets retry-with-resume swallowed the exception forever.
- **Fix:** [scripts/sagemaker_jobs/launch.py](../scripts/sagemaker_jobs/launch.py) now defaults `volume_size=200 GB` for every job.

### 1.2 CUDA OOM on 24 GB A10G (gotcha #15)

- **Affected:** S6 (Orpheus 3B + max_audio_sec=30 + grad-checkpointing), S1B (Whisper-LV3 1.5B full FT + adam-8bit + grad-checkpointing + bs=1), T3C (Spark + curriculum + max_seq_length=1024).
- **Cause:** activation memory of 1B+ models with long sequences exceeds 24 GB on A10G even with 8-bit Adam.
- **Fix:** moved S6, S1B, T3C, S1 (next clean run) to `ml.g6e.xlarge` (1× L40S 48 GB, $1.86/hr). For T3C this still wasn't enough at `max_seq_length=1024` so we further dropped to 768 (see §3.4).

### 1.3 CUDA-fork deadlock — silent 3-hour hang (gotcha #16)

- **First seen:** `adja-at1-full-20260427-104104` — froze at `decode+encode 0/183920` with GPU 0%, CPU 0.7%, GPU memory pinned at 28.5%.
- **Cause:** `model.from_pretrained(...).to("cuda")` was called **before** `unlab.map(num_proc=8)`. Forking with an already-initialized CUDA context corrupts the workers' CUDA state; first CUDA op silently deadlocks.
- **Fix rule:** load processor / tokenizer / feature_extractor **before** `.map(num_proc=N)` (CPU-only, safe to fork). Defer `model.to("cuda")` until **after** all `.map(num_proc=N)` calls finish. Applied to AT1, S2/S3, S6 cold-cache path (which forces num_proc=1 since the SNAC model is needed inside the workers).

### 1.4 wav2vec2 contrastive `index out of bounds`

- **Affected:** S2 (XLS-R 1B), S3 (MMS 1B). Both crashed ~30 min into SSL training, after `drop bad/long` filter completed.
- **Cause:** `prep_ssl` only dropped clips where `arr is None`. Short clips (after wav2vec2's 320× conv subsample → very few features) couldn't supply `num_negatives=100` cleanly; sampling produced degenerate / OOB indices that the CUDA kernel asserted on.
- **Fix:** changed `if arr is None:` to `if arr is None or len(arr) < 32000:` — 2 s @ 16 kHz floor → 100 features post-conv, safely covers num_negatives.
- **Note on iteration:** initial fix was 10000 (0.625 s, 31 features). Code review at 95% confidence flagged this as still too low; user agreed, stopped first resubmit at ~5 min, redeployed with 32000.

### 1.4b AT1 host-RAM OOM (4th distinct AT1 failure mode this day)

- **Job:** `adja-at1-full-20260427-192004` — failed 4 min in at `decode+encode 6255/183920 (3.4%)` with `ClientError: Please use an instance type with more memory`. **Not** CUDA OOM — host RAM.
- **Root cause:** Original `apply_chat_template` call had `text_kwargs.padding="max_length"` AND `audio_kwargs.padding="max_length"`. With `max_audio_sec=30`, every encoded row stored `input_values` of shape `[1, 720001]` float32 = **2.88 MB / clip**. `.map()`'s Arrow output materializes 183 k clips × 2.88 MB ≈ **527 GB on disk** plus per-process Arrow writer buffers + WaxalNLP source dataset COW pages (8 workers × 56 GB MP3 bytes inheritance = up to 448 GB if all written) → 192 GB host RAM exhausted.
- **Fixes applied (3 layers):**
  1. **Padding:** `apply_chat_template` now uses `padding=False` (variable-length output). On-disk dataset shrinks to ~330 GB (~1.8 MB/clip avg).
  2. **`num_proc`:** default reduced 8 → 2 to cap COW RAM and Arrow writer buffer multiplication.
  3. **Volume:** AT1 now gets `volume_size_gb=500` (vs the global 200 GB default).
  4. **Custom data collator:** added `at1_collate` that pads to longest in batch at training time, replacing HF's default collator (which requires identical shapes).
- **New AT1 submission:** `adja-at1-full-20260427-201244` carries this fix.

### 1.4c Mimi precompute pipeline (groundwork for future AT1 iterations)

In parallel with the AT1 ship-fix above, we built the architectural fix the user wanted:
- **New script:** [scripts/sagemaker_jobs/precompute_waxal_mimi_tokens.py](../scripts/sagemaker_jobs/precompute_waxal_mimi_tokens.py) — direct port of [precompute_waxal_tokens.py](../scripts/sagemaker_jobs/precompute_waxal_tokens.py) (SNAC) with `kyutai/mimi` swapped in. Output schema: `{id, duration_sec, mimi_codes [T_frames, 8] int16}`. Output prefix: `s3://…/datasets/waxalnlp_precomputed_mimi/`. Manifest naming: `ewe_unlabeled_mimi_codes_manifest-part-NN-of-04.json`.
- **4 sharded jobs** (`PRECOMPUTE_MIMI_0..3`) on `ml.g5.xlarge` ($1.41/hr × 4 × ~75 min ≈ **$11**). Submitted at 20:08 EDT (`adja-precompute-mimi-{0..3}-full-20260427-200835..200906`).
- **`mimi_channel: True`** flag added to launch.py mounting logic. AT1's registry entry has it set; mount is gated on all 4 manifests being present (else falls back to in-job .map()).
- **Status of cache integration:** the channel-mount logic is wired, but the **train-side cache LOADER is not yet implemented** in train_AT1. Reason: deriving CSM's exact `input_ids` structure from cached Mimi codes (vocab offset scheme, audio-segment markers, BOS/EOS placement) requires runtime introspection of the CSM processor that's risky to commit to without testing locally on a g5 instance — and the simpler `padding=False` + 500 GB volume fix above ships AT1 *now*, this week, in time for the thesis. **The precompute cache is a future-iteration artifact:** when M1/M2/M2b ever produce a fine-tuned Mimi codec, we re-run precompute with that checkpoint, then implement the cache loader at that point.

---

### 1.5 Drop-vs-truncate semantic confusion (max_audio_sec)

- AT1 / S6 originally `max_audio_sec=10`. Probed `data/ASR/ewe/ewe-unlabeled-00000.parquet` (300 clips): median 18 s, range 8.5–30 s. **At 10 s, AT1/S6 drop 99.7% of unlabeled training data.** Scripts updated to default `max_audio_sec=30`. AT1 audio_kwargs.max_length now derives from this so the CSM processor doesn't silently re-pad.
- For T3C (Spark, TTS data), the relevant cap is `max_seq_length` measured in BiCodec tokens — see §3.4.

---

## 2. The 8-job wave-3 fleet (state as of 2026-04-27 19:30 EDT)

| Job | Instance | Why | What it tests |
|---|---|---|---|
| `adja-s1-full-20260427-091126` | g5.2xlarge | Original morning run; left to ride to max_run | Whisper-LV3 LoRA Ewe→Adja, partial Stage A reference |
| `adja-s6-full-20260427-190527` | g6e.xlarge | OOM on g5.12xlarge fixed by 48 GB L40S | Orpheus 3B audio-LM; Ewe SNAC pretrain → Ewe+Adja TTS FT |
| `adja-s1b-full-20260427-190549` | g6e.xlarge | OOM on g5.2xlarge fixed by 48 GB | Whisper-LV3 **full FT** Adja-only (vs S1's LoRA) |
| `adja-s2-full-20260427-191943` | g5.12xlarge | 32000 floor + CUDA-fork fix | XLS-R 1B SSL on Ewe → Adja CTC |
| `adja-s3-full-20260427-191954` | g5.12xlarge | Same as S2 | MMS 1B SSL on Ewe → Adja CTC |
| `adja-at1-full-20260427-192004` | g5.12xlarge | CUDA-fork fix; max_audio_sec=30 | CSM 1B audio-LM 3-stage (unlabeled Ewe → Ewe TTS → Adja TTS) |
| `adja-s1-full-20260427-192235` | g6e.xlarge | Fresh full S1 run on faster GPU | Whisper-LV3 LoRA, complete 30+20 epochs |
| `adja-t3c-full-20260427-192725` | g6e.xlarge | max_seq_length 1024→768 | Spark 0.5B + curriculum Ewe+Adja Stage 2 |

---

## 3. Comparison fairness — the table to remember

This is the section to pin to the wall when writing the experiments chapter.

| Comparison | Fairness | Caveat to write down |
|---|---|---|
| **S1 (LoRA) vs S1B (full FT)** | 🟢 **Headline result, both on g6e.xlarge L40S** | **S1B completed 2026-04-27 evening: Test CER 54.71%, WER 101%** — ~1.5× worse than the deployable Adja Whisper (E4v4, CER 37.18%); the original E4 CER 24.90% number was a logged-only result whose checkpoint was never uploaded, see `results/comparison.md` 2026-04-29 correction. Train loss collapsed 425× while dev CER oscillated chaotically 52–119% (catastrophic forgetting + Whisper hallucination). Compared against S1's still-running Stage A (Ewe CER 18.71% at epoch 10). **S1B is the negative-control ablation that confirms LoRA's parameter-efficient regularization is necessary for low-resource adaptation of 1.5B Whisper.** Important paper claim: "Capacity is not the ceiling; priors are" — same finding as the CSM Stage 2 / Spark hypothesis from `why-spark-worked.md`. |
| **S2 (XLS-R 1B SSL) vs S3 (MMS 1B SSL)** | 🟢 Preserved | Same fixes (32000 floor, deferred .to(cuda)), same g5.12xlarge instance, only `--base` differs. |
| **CF1 / CF2 / CF3 vs T3A / T3C** ("best Stage 2 strategy") | 🟡 Was always architecture-different | CSM 1B + Mimi codec vs Spark 0.5B + BiCodec — never apples-to-apples. T3C's 13% extra data drop (vs T3A) is a new wrinkle but doesn't change the headline story (curriculum vs direct comparison is still valid within the Spark family). |
| **T3A vs T3C** (curriculum hypothesis within Spark) | 🔴 **T3C abandoned for thesis** | T3C OOM'd 11 times today across `max_seq_length` ∈ {2048, 1024, 768} on every available instance up to g6e.xlarge (L40S 48 GB). At every seq length the failure point is identical: 44.11/44.40 GB used at a wav2vec2 forward-pass linear layer, ~287 MB free. Activation memory is dominated by Spark's BiCodec base + LoRA, not seq length. **For thesis: report T3A standalone (Spark direct, completed and on HF Hub).** Use CF1/CF2 within the CSM/Mimi family for the curriculum-vs-direct signal. **Future work:** retry T3C on `g6e.12xlarge` (4× L40S DDP, 192 GB total) or `p4de.24xlarge` (8× A100 80 GB) once quota lands. |
| **AT1 (CSM audio-LM) vs prior CSM/T1/T1-tokfix/CF*** | 🟡 Different data scope | Prior CSM Stage A attempts hit `max_audio_sec=10` and dropped 99.7% of WaxalNLP unlabeled. New AT1 uses `max_audio_sec=30`, keeps ~100%. **Don't compare new AT1 to anything called "AT1" before 2026-04-27 evening.** Prior AT1 attempts produced no valid Stage A. New AT1 is the first valid CSM-Stage-A run. |
| **S6 (Orpheus audio-LM) vs prior S6 attempts** | 🟢 Preserved | Now uses precomputed SNAC cache, but SNAC tokens are deterministic — same data. |
| **AT1 vs S6** (CSM/Mimi vs Orpheus/SNAC head-to-head) | 🟡 Was always architecture-different | Different codecs, different LMs, different hyperparameters. Comparison is qualitative/perceptual, not quantitative. Audio listening verdicts (per the CF2 lesson — loss does not predict intelligibility at Stage 2) are the metric. |
| **AT1 vs CF2** (with-Stage-A vs without-Stage-A on CSM) | 🟢 Meaningful for paper headline | This is the "does Ewe-unlabeled audio-LM pretraining help downstream Adja TTS?" question. CF2 doesn't have an AT1-equivalent Stage A; AT1 does. Apples-to-apples on the Stage-A question. |

### The two big asterisks for the paper

1. **T3A vs T3C now have different training-data scopes.** T3A trains on ~78% of WaxalNLP TTS (cap 20 s); T3C on ~65% (cap 15 s). Mention in the Methods section: *"T3C uses max_seq_length=768 because the curriculum-mixed Ewe+Adja sequences exceed the 48 GB L40S memory budget at 1024."*
2. **AT1 results before 2026-04-27 evening are not comparable to AT1 results after.** All prior AT1 attempts were broken (disk-full, schema, CUDA-fork, 99.7% drop rate). The first valid run is `adja-at1-full-20260427-192004`.

For the Adja paper headline ("does Stage A audio-LM pretraining on Ewe unlabeled help downstream Adja TTS?"), AT1 vs CF2 is the meaningful comparison and that's still valid since CF2 doesn't have an AT1-equivalent Stage A.

---

## 4. Data scope per experiment (from actual probes — not guesses)

Two corpora hit by these jobs: WaxalNLP `ewe_asr/unlabeled` (used by AT1, S2, S3, S6 Stage A) and `ewe_tts/train` (used by S6 Stage B, T3A, T3C, plus Stage-2-on-Adja runs).

### `ewe_asr/unlabeled` — measured 2026-04-27 (300-clip sample from shard 0)

```
min=8.54s   median=18.18s   mean=19.13s   max=29.92s
≤10s: keep   0.3%   ← old AT1/S6 default would drop 99.7% (!)
≤15s: keep   0.3%
≤20s: keep  71.0%
≤30s: keep 100.0%   ← new default: max_audio_sec=30
```

**Take-away:** the unlabeled ASR split is mostly long-form speech (storytelling). For audio-LM pretraining (AT1, S6 Stage A) we want all of it; cap ≥30 s.

### `ewe_tts/train` — measured 2026-04-27 (49-clip sample from shard 0)

```
min=3.71s   median=9.33s   mean=22.03s   max=166.66s   p95=83.06s
```

Tokens-to-seconds approximate mapping (Spark BiCodec ~50 tokens/sec):

| max_seq_length | ~audio cap | clip retention |
|---|---|---|
| 512 | 10.2 s | 53% |
| **768** | **15.4 s** | **65%** ← T3C |
| 1024 | 20.5 s | 78% (T3A "default", but OOM under curriculum) |
| **2048** | **41.0 s** | **86%** ← T3A |

**Take-away:** TTS data is more bimodal than I initially claimed. There IS a long tail of multi-minute clips. The cap matters.

---

## 5. Decision log — why we made specific calls

| Decision | Why | Alternative considered |
|---|---|---|
| Default volume = 200 GB for every job | EBS gp2 is ~$0.10/GB-month prorated; extra 170 GB on a 5 h job costs $0.12. Not worth tracking which experiments need more. | Per-experiment `volume_size_gb`. Was the first fix; later collapsed to a default. |
| g6e.xlarge for OOM jobs (S6, S1B, T3C, S1) | Single L40S 48 GB, 2× VRAM of A10G, ~2× faster, $1.86/hr (cheaper effective $/result than g5.12xlarge for single-GPU work). | g5.12xlarge (4× A10G) — works for DDP-able jobs but hits per-GPU 24 GB ceiling for big-model+long-seq combos. |
| 32000-sample floor (not 10000) for S2/S3 prep | Code review at 95% confidence: 31 features post-conv is degenerate for num_negatives=100 + mask_time_length=10 mask spans. 100 features (= 32000 samples) is safe. | 10000 (initial fix). Was deployed, code review caught it before crash, user authorized stop+resubmit. |
| max_seq_length=768 (not 1024) for T3C | g6e.xlarge OOM'd at 1024 with curriculum Ewe+Adja sequences. Probe showed 65% clip retention at 768 vs 78% at 1024 — accepted the data loss to fit memory. | g6e.12xlarge (4× L40S DDP) — quota not yet approved. |
| max_audio_sec=30 (not 10) for AT1, S6 | Probe: 99.7% of unlabeled clips are >10 s. Hard-dropping them wastes the entire dataset. | Stayed at 10 — but the original failure mode (Stage A trains on ~600 clips out of 183 k) makes the experiment meaningless. |
| Don't stop S1-091126 (the original LoRA on g5.2xlarge) | It's productively training. Even partial Stage A (~16 epochs) yields a useful Ewe LoRA adapter for HF Hub release. | Stop and resubmit on g6e.xlarge — duplicates effort; we already have a fresh g6e.xlarge S1 running in parallel. |

---

## 6. What's NOT in this doc but matters

- The morning's WaxalNLP `CastError` fix (datasets 2.18.x bidirectional schema match, two-group `__index_level_0__` peek) — see [sagemaker-gotchas.md #12](../learnings-from-the-past/sagemaker-gotchas.md).
- The Spark-TTS S3 channel + WaxalNLP S3 channel staging — see the `waxal_channel: True` and `spark_channel: True` flags in `EXPERIMENT_CONFIGS`.
- The CSM Stage 2 perceptual lesson from CF1/CF2: **loss does not predict intelligibility at Stage 2.** CF1 had the lowest eval_loss across all Stage 2 experiments and was 0/5 intelligible; CF2 was 1/5 intelligible. Always do native-speaker listening tests.

---

## 7. Bigger-VRAM chips — what to request next

User has g6e.xlarge=8 ✅, g5.12xlarge=5 ✅, g5.2xlarge=6 ✅, g5.xlarge=7 ✅, p4d.24xlarge=4 🟡 pending.

**Important:** the pending `p4d.24xlarge` (A100 40 GB) has **less per-GPU VRAM than the current L40S 48 GB**. p4d gives 8× parallelism, not bigger memory headroom per sample. So if T3C OOMs at 48 GB on one L40S, it'll also OOM on one A100-40GB.

The actually-useful upgrades for "more VRAM per GPU":

| Chip | Per-GPU VRAM | Instance | $/hr | Approval |
|---|---|---|---|---|
| **L40S** | 48 GB | g6e.xlarge / g6e.12xlarge | $1.86 / ~$15 | Have it |
| **A100 80 GB** | 80 GB | p4de.24xlarge (8×) | ~$40 | Slow — same channel as p4d |
| **H100 80 GB** | 80 GB | p5.48xlarge (8×) | $98 | Slow, justification required |
| **H200 141 GB** | 141 GB | p5en.48xlarge (8×) | ~$140 | Slowest, capacity tight |

**Recommendation:** next ask is **`g6e.12xlarge → 1`** (4× L40S 192 GB total via DDP) — cleanest next step, likely fast approval, unblocks AT1/M1/M2 multi-GPU. Optionally file `p4de.24xlarge → 1` in parallel with the existing p4d request (same support channel) — sits there until needed.

Skip g5.48xlarge (same arch as g5.12xlarge), g6.xlarge (no VRAM upgrade vs g5).

---

## 8. Open threads / future work (post-thesis)

- **M0 / M1 / M2 / M2b** — Mimi codec fine-tuning track. Not yet run. Would test: "is the codec a bottleneck?" Separate hypothesis from AT1 (which trains the LM on top of the codec). See §9 below for thesis-week timing.
- **g6e.12xlarge quota** — request `→ 1` whenever bandwidth allows.
- **p4de.24xlarge or p5.48xlarge** — only useful for "winner config" final-paper run.
- **S3 staging of WaxalNLP** — partially done; finish to remove HF Hub dependency entirely.
- **Resume-from-checkpoint** for S1 — would let us continue the partial 091126 run instead of restarting fresh.

---

## 9. Thesis week — what to actually do (next 7 days)

User's constraint: thesis presentation in ~1 week. Wave-3 jobs were just submitted; most finish 1–2 days from now. Pragmatic priorities:

1. **Days 1–2:** let the 8 active jobs run. Don't add more. Watch for the next failure mode.
2. **Day 2:** S1-091126 will hit max_run; pull the Ewe LoRA checkpoint, push to HF Hub for community (planned), use as a paper artifact.
3. **Day 3:** listen to S6 + AT1 outputs (when they generate). Native-speaker verdict is the metric per CF1/CF2 lesson.
4. **Day 4:** harvest CER from S1 (LoRA) and S1B (full FT) — the head-to-head ASR result.
5. **Day 5:** write thesis sections referencing the experiments, using THIS DOC as the single source of truth for fairness asterisks.
6. **Day 6–7:** smoke-test M0 only (1 h on g5.xlarge, ~$1.41) to validate the Mimi codec hypothesis is testable. Don't run M1/M2/M2b — they're sequential dependencies (M → re-AT1 → re-CF*) that won't fit in a week.

### Mimi (M-series) verdict for thesis: **future work**

Document the hypothesis in the thesis ("would Mimi codec fine-tuning on Gbe audio improve downstream CSM Stage 2 quality?") with M0 smoke test as evidence the experiment is set up. Full M1/M2/M2b results post-thesis.

**Pre-processing note:** for the M-series, the only useful precompute is MMS teacher embeddings for M2/M2b (~5 min one-shot, frozen model). The Mimi codec itself is what's being trained, so you can't precompute Mimi tokens — that defeats the purpose.

The bigger pre-processing wins available right now (worth doing this week regardless of M-series):

| What | Saves | Effort |
|---|---|---|
| Finish WaxalNLP S3 channel staging (you started, partially done) | ~30 min HF Hub download per future submission of AT1/M1/M2/M2b/S2/S3/S6 | ~10 min, one-shot `aws s3 sync` |
| Mimi token precompute for AT1 (mirror SNAC pattern) | hours of preprocess time per AT1 resubmit | ~3 h to write + run; only worth it if AT1 needs another iteration |

---

## 10. tl;dr for thesis writing

- **Wave-3 (2026-04-27 evening) is the canonical experimental run.** Anything before it is broken or inferior.
- **AT1 is now the first valid CSM-Stage-A run.** Compare against CF2 for the Stage-A-helps-Adja headline.
- **T3A vs T3C carry a seq-length asterisk** — note in Methods.
- **Loss doesn't predict intelligibility at Stage 2.** Use native-speaker listening as the metric (CF1/CF2 evidence).
- **M-series is future work.** Run M0 smoke this week if time permits; defer M1/M2/M2b.
- **Next quota ask: `g6e.12xlarge → 1`.** Don't bother with p4d/p5 until you have a winner config.
