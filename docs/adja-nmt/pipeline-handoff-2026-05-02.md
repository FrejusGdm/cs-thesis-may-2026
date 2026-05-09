# Pipeline handoff — what an agent building Chapter 6 needs to know

Last updated: 2026-05-02. Authoritative sources cited inline; this doc is a
**digest**, not a replacement.

The end-to-end Adja pipeline (Chapter 6) is

```
spoken Adja  ─►  ASR  ─►  MT (Adja→fr)  ─►  LLM  ─►  MT (fr→Adja)  ─►  TTS  ─►  spoken Adja
```

This document tells the next agent what is **already deployable** for the
ASR and TTS legs, what numbers we know about each leg, and how to validate
the TTS leg quantitatively without a human listener in the loop.

---

## 1. ASR leg — already deployed, two architectures

We have **two Adja ASRs** published as public Hugging Face model repos and
ready to be served via Inference Endpoints:

| Repo | Architecture | Real-Adja test CER | Real-Adja test WER | Notes |
|---|---|---|---|---|
| [`JosueG/wav2vec2-xlsr-adja-c4v2`](https://huggingface.co/JosueG/wav2vec2-xlsr-adja-c4v2) | XLS-R 300M + character CTC | **25.05 %** | 72.39 % | Reproducible flagship. Greedy CTC. |
| [`JosueG/whisper-ewe-adja-e4v4`](https://huggingface.co/JosueG/whisper-ewe-adja-e4v4) | Whisper-small fine-tuned on Ewe then Adja | 37.18 % | 83.61 % | Overfits past epoch 20. |

**Important caveat that the pipeline agent should know:** the thesis ASR
table cites a 24.90 % CER number for the Whisper-Ewe row. That checkpoint
was never uploaded to the Hub; only the metrics file survived. Two
reproduction attempts plateau at ~37 %, so the **deployable Whisper number
is 37.18 %, not 24.90 %**. Working hypothesis is that the original run
trained with the EOS-mask hallucination bug we later fixed; that bug acted
as accidental regularization. See
[`experiments/registry.md`](../experiments/registry.md) E4_v5 row and
[`results/comparison.md`](../results/comparison.md) lost-checkpoint
correction block.

**For the pipeline:** use C4v2 (XLS-R + CTC) by default. It is the
reproducible ASR, has the lower CER, and degrades gracefully on
out-of-distribution audio (TTS noise, codec artifacts) where Whisper
hallucinates. If you want a second-opinion read, transcribe in parallel
with E4v4 and average the normalized CERs.

### Spinning up the ASR endpoints

The `scripts/hub/manage_endpoints.py` helper already exists. Default tier
is CPU `intel-spr` x2 (about $0.067/hr) with scale-to-zero on idle:

```bash
python scripts/hub/manage_endpoints.py up \
    JosueG/wav2vec2-xlsr-adja-c4v2 \
    JosueG/whisper-ewe-adja-e4v4

# When done:
python scripts/hub/manage_endpoints.py down aja-xlsr-adja-c4v2 aja-ewe-adja-e4v4
```

Each endpoint takes 1–2 minutes to come up the first time; subsequent
warm requests are fast. Cold start after scale-to-zero is about
60 seconds.

### Calling an ASR endpoint

Plain HTTP POST with audio bytes and the right Content-Type:

```python
import requests
from huggingface_hub import get_token

resp = requests.post(
    ENDPOINT_URL,                 # https://<id>.us-east-1.aws.endpoints.huggingface.cloud
    data=open("clip.wav", "rb").read(),
    headers={
        "Authorization": f"Bearer {get_token()}",
        "Content-Type": "audio/wav",
        "Accept": "application/json",
    },
    timeout=180,
)
hyp = resp.json()["text"]
```

Do **not** use `huggingface_hub.InferenceClient`'s
`automatic_speech_recognition` against an endpoint URL — it omits the
`Content-Type` header and the endpoint rejects with `Content type "None"
not supported`. Verified 2026-04-28. The pattern above is what
`experiments/tts/eval/reverse_wer.py` uses; copy from there.

---

## 2. TTS leg — Spark TTS is the only working architecture

The TTS chapter (Chapter 5) tested CSM 1B, Orpheus 3B, F5-TTS, E2-TTS,
VoxCPM, Qwen3-TTS, MMS-TTS-Ewe, and Spark TTS 0.5B. **Only the Spark
direct-Adja fine-tunes produced intelligible Adja**. CSM and Orpheus
produced intelligible Ewe at Stage 1 of the Gbe cascade but collapsed
back to noise on Stage 2 adaptation to Adja.

Reverse-WER auto-eval (n = 22 runs, 2 ASRs, 2026-04-29) confirms the
listening verdict: the Spark family clusters at **33–47 % CTC CER**,
within ~10 points of C4v2's own published test CER on real Adja audio
(25.05 %). All other architectures sit at ≥100 % CER. See
[`results/reverse-wer-summary.md`](../results/reverse-wer-summary.md).

### Where the Spark checkpoints live

| Run | Hub path | Audio | Listening |
|---|---|---|---|
| Spark TTS 0.5B direct Adja (canonical) | `JosueG/adja-tts-results/T3/adapter/` | `T3/generated_audio/` | Intelligible Adja |
| Spark TTS 0.5B 120-step | `JosueG/adja-tts-results/T3_spark_120steps_2026-04-18/adapter/` | yes | Intelligible Adja |
| Spark TTS 0.5B 20-epoch early-stop | `JosueG/adja-tts-results/T3_spark_20ep_earlystop_2026-04-18/adapter/` | yes | Intelligible Adja |

Each `adapter/` folder contains the fine-tuned **LLM half** of Spark
(Qwen2 0.5B, the part that emits BiCodec semantic + global tokens). The
**audio-tokenizer half** (BiCodec encoder/decoder + the small
wav2vec2-xlsr-53 used for prompt encoding) lives at the public base
`unsloth/Spark-TTS-0.5B/{BiCodec,wav2vec2-large-xlsr-53}/`. Inference
needs **both halves**, plus the [`Spark-TTS/`](../Spark-TTS/) repo
checked out at the project root for the `BiCodecTokenizer` and the
`SparkTTS.inference()` plumbing.

### Deploying Spark for the pipeline

This is the missing piece. We have not yet pushed Spark as a
deployable model repo. The work is concretely:

1. **Combine adapter + base** into a single repo
   `JosueG/spark-tts-adja-t3-120step` (or whichever Spark variant ranks
   highest on reverse-WER for your pipeline tests). Copy the
   adapter/ files and pull in the BiCodec + wav2vec2 sub-folders from
   `unsloth/Spark-TTS-0.5B`.
2. **Add a custom `handler.py`** that loads the combined artifact and
   exposes a `text_to_speech(text)` interface. The serverless task
   `text-to-speech` will not auto-handle Spark because the architecture
   is not a single-pipeline-tag model; it is a Qwen2 LLM emitting
   BiCodec tokens that the BiCodec decoder turns into a waveform.
3. **Push and deploy** as a dedicated Inference Endpoint. Spark
   inference is realistic on CPU but slow; a small GPU
   (`nvidia-l4` x1, ~$0.80/hr) gives the pipeline acceptable latency
   for an end-to-end demo. Use scale-to-zero so cost only burns when
   the demo is live.

A starter publishing script and a starter `handler.py` are checked in
at:

- [`scripts/hub/publish_spark_t3.py`](../scripts/hub/publish_spark_t3.py)
- [`scripts/hub/spark_handler.py`](../scripts/hub/spark_handler.py)

Both are **starters, not finished**. The publishing script picks up
the adapter, downloads the base, and writes the model card. The
handler loads `SparkTTS` and invokes `inference()` with the simple
control-mode (gender/pitch/speed) so prompt audio is not required.
**Test the handler locally on CPU before pushing the repo.**

### Calling the Spark endpoint (proposed shape)

Once deployed, the same HTTP POST pattern as ASR but with text input:

```python
resp = requests.post(
    SPARK_ENDPOINT_URL,
    json={"inputs": "Tɛnigbe ciyi vayi de ŋweba"},
    headers={"Authorization": f"Bearer {get_token()}"},
    timeout=180,
)
wav_bytes = resp.content   # the handler returns audio/wav
open("synthesized.wav", "wb").write(wav_bytes)
```

The pipeline agent should treat the Spark endpoint as a black-box TTS:
text in, WAV bytes out. Sample rate is 16 kHz at the LLM side and
16 kHz at the BiCodec output (Spark resamples internally).

---

## 3. Validating the TTS leg in pipeline tests

The pipeline produces synthesized Adja from an LLM completion that
went `Adja → fr → LLM → fr → Adja`. There is no human in the loop for
end-to-end CI tests. Use **reverse-WER** to score the TTS output:

```bash
# experiments/tts/eval/reverse_wer.py already deployed
python experiments/tts/eval/reverse_wer.py \
    --runs <name-of-your-pipeline-test-run> \
    --adja-asrs '<C4V2_ENDPOINT_URL>:c4v2,<E4V4_ENDPOINT_URL>:e4v4'
```

The driver expects a directory layout matching what the existing TTS
runs use: `<run>/generated_audio/*.wav` plus a `metrics.json`
containing `{"generated": [{"file": "...", "text": "..."}, ...]}`.
Have your pipeline tests write their outputs in that shape and you
get reverse-WER scoring for free.

**Floor:** reverse-WER inherits the ASR's own real-Adja error rate
(~25 % CTC CER, ~37 % Whisper CER). Treat reverse-WER as a relative
ranking signal across pipeline configurations, not as an absolute
intelligibility percentage. A pipeline that scores worse than 100 %
CTC CER is producing noise; a pipeline that scores under 40 % is in
the same band as the standalone Spark-direct TTS.

---

## 4. Other things the pipeline agent should know

- **MT leg**: Chapter 3. The fr↔Adja MT is a separate fine-tuned NLLB
  (and a Gemini few-shot ICL fallback). See
  [`thesis-writing/cs-thesis-josue-2026/chapters/tex/03 MT.tex`](../thesis-writing/cs-thesis-josue-2026/chapters/tex/03%20MT.tex)
  and the `experiments/mt/` directory.
- **LLM leg**: not project-internal; pick whichever provider fits the
  demo. The MT legs handle the language conversion to/from a
  high-resource language.
- **Tokenizer / NFC**: every text leg of the pipeline must apply
  `unicodedata.normalize("NFC", text.strip())` before the next leg.
  Adja special characters (`ɛ`, `ɔ`, `ŋ`, `ɖ`) and tone marks survive
  NFC; they do not survive NFKD. This is a project-wide rule
  (CLAUDE.md).
- **Audio sample rates**: ASR side is 16 kHz; Spark TTS side is also
  16 kHz at output. If the pipeline records from a mic at 48 kHz,
  resample with `librosa.resample(y, orig_sr=48000, target_sr=16000)`
  before posting to the ASR endpoint.
- **Lost-checkpoint correction**: the thesis (chapter 4 results) cites
  both 24.90 % (logged) and 37.18 % (deployable) for the Whisper-Ewe
  ASR. Cite the deployable number in pipeline-end performance claims;
  cite the logged number only when discussing the original
  experiment record.

## 5. Quick-reference table

| Concern | Where to look |
|---|---|
| ASR endpoint management | `scripts/hub/manage_endpoints.py` |
| ASR HTTP call pattern | `experiments/tts/eval/reverse_wer.py:CachingASRClient.transcribe` |
| Reverse-WER eval driver | `experiments/tts/eval/reverse_wer.py` |
| Reverse-WER full results | `results/reverse-wer-summary.md` (22 runs × 2 ASRs) |
| TTS leaderboard | `results/tts-comparison.md` |
| ASR leaderboard | `results/comparison.md` |
| Experiment ledger | `experiments/registry.md` |
| Spark TTS source code | `Spark-TTS/cli/SparkTTS.py` |
| Spark deploy starter | `scripts/hub/publish_spark_t3.py`, `scripts/hub/spark_handler.py` |
| Curated best-of manifest | `scripts/hub/best_models.yaml` |
| Project conventions | `CLAUDE.md` |
| Thesis correction context | `experiments/registry.md` E4_v5 row |
