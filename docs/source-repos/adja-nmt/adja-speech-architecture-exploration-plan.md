# Adja Speech Architecture Exploration Plan

## Why This Plan Exists

You want to experiment broadly before committing to a paper angle.  
This plan is built for that: maximize learning and signal on Adja by comparing many architectures, especially:

- direct speech translation (and eventually speech-to-speech)
- cascade systems
- hybrid systems
- newer omni/conversational speech stacks

The goal is to discover what actually works for Adja under low-resource constraints, then choose the best paper direction from evidence.

---

## Primary Exploration Questions

- When does direct speech translation beat cascade on Adja?
- Which architecture family is most data-efficient?
- Which systems degrade least under low-resource and noisy conditions?
- Can omni/conversational speech models help low-resource translation, or are they better for downstream interaction only?
- What are the quality vs latency vs compute tradeoffs for each approach?

---

## Architecture Landscape to Explore

## 1) Cascade Baselines (must be strong)

Build these first so direct models are compared against a fair baseline.

- `C1`: ASR -> MT
  - ASR candidates: Whisper, MMS-adapted ASR
  - MT candidates: your strongest Adja<->French checkpoints
- `C2`: ASR -> MT -> text post-edit/rerank (optional LLM layer)
- `C3`: ASR -> MT -> TTS (end-user spoken output path)

What this teaches:
- upper bound from modular pipelines
- where errors enter (ASR vs MT vs TTS)
- practical production baseline

---

## 2) Direct Speech Translation (S2TT / S2ST)

Core direct candidates:

- `D1`: encoder-decoder direct speech-to-text translation baseline
- `D2`: fairseq UnitY-style two-pass/direct unit-based systems
- `D3`: SeamlessM4T/UnitY2-style direct multilingual ST transfer

Optional speech-to-speech:

- `S1`: direct speech-to-speech through discrete units
- `S2`: direct S2TT + neural vocoder back-end for spoken output

What this teaches:
- whether direct systems can outperform cascade for Adja
- benefits of unit-based approaches for low-resource speech
- speed and latency profile against cascades

---

## 3) Hybrid Systems (usually underrated)

- `H1`: ASR encoder initialization + ST fine-tuning
- `H2`: Multi-task training (ASR + ST jointly)
- `H3`: Cascade with direct-model rescoring or minimum Bayes risk style fusion

What this teaches:
- whether transfer from ASR improves low-resource ST
- whether hybrid systems dominate strict direct or strict cascade

---

## 4) Omni / Conversational Speech Models

Treat these as a separate track, not the first baseline track.

- Qwen2-Audio / Qwen2.5-Omni style models
- Sesame CSM family
- Fish Speech family

What this track is good for:
- conversational naturalness
- multimodal integration
- spoken agent behavior and voice quality

What this track is not guaranteed to solve:
- low-resource translation quality on Adja-specific parallel tasks
- fair apples-to-apples ST benchmarking without adaptation

Use this track to test frontier ideas after core benchmark tracks are stable.

---

## 5) TTS — Text-to-Speech Synthesis for Adja

Generate spoken Adja from text. This track complements ASR (speech->text) by going text->speech.
Fine-tune modern LLM-based TTS models using Unsloth's optimized pipeline.

### Experiments:

- `T1`: Sesame CSM (1B) fine-tuning via Unsloth LoRA
  - Base model: Llama 3.2-1B backbone + Mimi RVQ audio decoder (24kHz)
  - Why: 1B is right-sized for our data (~1.6k utterances), base model so we teach it Adja phonology
  - Fine-tuning: LoRA on all attention + MLP projections
  - Compute: runs on free T4 (Colab) or A100 (HPC)
  - Code: `experiments/tts/T1_sesame_csm_finetune/`

- `T2` (future): Orpheus TTS (3B) fine-tuning
  - Already fine-tuned on 8 professional voices — starts with better voice consistency
  - 3x larger = more expensive but potentially better quality
  - Supports emotion tags (`<laugh>`, `<sigh>`)

- `T3` (future): Spark TTS (0.5B)
  - Smallest model — interesting for deployment on constrained hardware
  - Quick experiment to see how small we can go

What this teaches:
- whether modern TTS models can learn Adja phonology (tones, special characters)
- quality vs model size tradeoffs for low-resource TTS
- practical viability of end-to-end Adja speech synthesis
- feeds into cascade pipeline: ASR -> MT -> **TTS** for spoken output

### References:
- Sesame CSM: https://github.com/SesameAILabs/csm
- Unsloth TTS fine-tuning: https://unsloth.ai/docs/basics/text-to-speech-tts-fine-tuning
- Mimi audio codec: https://arxiv.org/abs/2410.00037
- LoRA: https://arxiv.org/abs/2106.09685
- All Unsloth TTS notebooks: `references/unsloth-tts-notebooks/`

---

## Tooling and Frameworks to Learn

Start from frameworks with reproducible recipes:

- fairseq + Seamless stack
  - strong for unit-based direct S2ST/S2TT and multilingual transfer
- ESPnet-ST
  - flexible research recipes for ST and multi-task configurations
- SpeechBrain
  - fast prototyping for low-resource speech and ST experiments

Use one framework as your primary benchmark engine and one as backup for validation.

---

## Suggested Learning Track (Architecture First)

### Stage A: Foundation Reading (1 week)

Focus topics:

- Whisper architecture and robustness assumptions
- MMS multilingual ASR design and adapter strategy
- UnitY and UnitY2 design (two-pass, discrete units, non-autoregressive unit decoding)
- SeamlessM4T v2 modular architecture
- Recent low-resource ST shared-task findings on direct vs cascade

Expected outcome:
- architecture map and hypotheses before any heavy training

### Stage B: Reproduce Known Baselines (2-3 weeks)

- run one clean cascade baseline end-to-end
- run one direct ST baseline end-to-end
- lock data manifests, split protocol, metric scripts

Expected outcome:
- verified harness that can compare systems fairly

### Stage C: Breadth Sweep (3-5 weeks)

- compare many systems with shallow tuning
- include at least three data regimes (tiny, medium, full)
- record quality, latency, and compute

Expected outcome:
- architecture ranking for Adja

### Stage D: Depth and Stress Tests (4-6 weeks)

- select top 2-3 systems
- deeper tuning, robustness tests, noise/accent sensitivity
- low-resource scaling curves

Expected outcome:
- evidence for final direction

---

## Data and Split Design (Critical)

Keep one canonical split definition for all tracks:

- same train/val/test for cascade/direct/hybrid
- same normalization and text policy
- clear metadata on speaker/domain/noise when available

Create extra low-resource subsets:

- tiny slice
- medium slice
- full paired data

This gives data-scaling curves and prevents false conclusions from mismatched splits.

---

## Metrics to Log for Every Run

Translation quality:

- BLEU
- chrF++

ASR quality (if ASR involved):

- WER
- CER

Efficiency:

- train wall time
- inference latency
- GPU-hours

For spoken output tracks:

- intelligibility proxy
- human preference on small evaluation slices

Do not ship comparisons without both quality and efficiency metrics.

---

## Experiment Matrix (Wide, Then Deep)

Phase 1 matrix (breadth-first):

- 3 cascade variants x 2 seeds
- 3 direct variants x 2 seeds
- 2 hybrid variants x 2 seeds
- 3 data sizes

Phase 2 matrix (depth-first):

- top 2-3 systems only
- more seeds
- robustness and decode tuning

Decision gate:

- If direct beats cascade consistently at similar compute: prioritize direct track
- If cascade stays dominant: pivot to hybrid/cascade optimization
- If split by objective: choose one target (quality-first or latency-first)

---

## What to Explore About Fairseq/Meta Stack Specifically

- UnitY two-pass decoding vs single-pass direct systems
- discrete-unit generation vs text intermediate
- SeamlessM4T v2 UnitY2 non-autoregressive unit decoding for speed
- effect of initializing from multilingual pretrained encoders in low-resource Adja

Practical hypothesis:
- low-resource direct systems may need strong pretraining and careful transfer; weak transfer makes cascade hard to beat.

---

## What to Explore About Omni Models Specifically

For Qwen2.5-Omni, Sesame, Fish:

- use as inference adapters and conversational layers first
- measure if they help translation quality or only spoken interaction quality
- test whether they can replace specialized ST, or only complement it

Practical hypothesis:
- they may shine in dialogue naturalness and speech generation, but core low-resource translation quality may still come from specialized ST/cascade components.

---

## Risks and How to Avoid Wasted Cycles

- Weak baseline risk
  - fix by building strong cascade baseline first
- Split inconsistency risk
  - fix by single shared benchmark harness
- Metric mismatch risk
  - fix by mandatory quality + efficiency reporting
- Frontier model distraction risk
  - fix by gating omni-model work until core benchmark is stable

---

## Deliverables You Should Produce During Exploration

- `architecture-map.md`
  - concise notes on each architecture family and expected failure modes
- `benchmark-protocol.md`
  - split policy, metrics, logging schema
- `run-ledger.md`
  - run-by-run summary with compute, scores, notes
- `decision-checkpoint.md`
  - every 2-3 weeks: what to continue, drop, or pivot

---

## Reference Starting Points (Primary Sources)

- NLLB (No Language Left Behind): https://arxiv.org/abs/2207.04672
- Whisper: https://cdn.openai.com/papers/whisper.pdf
- MMS (Scaling Speech to 1000+ languages): https://jmlr.org/papers/v25/23-1318.html
- UnitY (direct S2ST, ACL 2023): https://aclanthology.org/2023.acl-long.872
- SeamlessM4T v2 report: https://arxiv.org/abs/2312.05187
- Qwen2-Audio report: https://arxiv.org/abs/2407.10759
- Qwen2.5-Omni report: https://arxiv.org/abs/2503.20215
- Sesame CSM repo: https://github.com/SesameAILabs/csm
- Fish Speech repo: https://github.com/fishaudio/fish-speech

Use these references to build your own architecture notes before large runs.

---

## Final Guidance

Experiment like a benchmark team first, not like a single-model paper team.

If you do this exploration well, your eventual paper options become stronger:

- direct vs cascade benchmark on Adja
- hybrid transfer method on Adja
- efficiency-focused low-resource ST analysis
- conversational spoken QA extension after core translation evidence is stable
