#!/usr/bin/env python3
from __future__ import annotations
"""
Stage 1: ASR — audio → transcript.

Adja ASR (Mode A):
  Runs both Whisper-Ewe (e4v4, 37.18% CER) and XLS-R 300M CTC (c4v2, 25.05%
  CER) on each audio file and returns both hypotheses. The configured 'primary'
  feeds downstream MT.

French ASR (Mode C):
  Uses standard openai/whisper-small with language=fr (no fine-tuning needed).
  Output goes directly to the response stage, skipping MT→FR.

Two inference modes (set per-model in p1_config.yaml):
  endpoint  POST raw WAV bytes to a live HF Inference Endpoint.
            Endpoint URLs come from `scripts/hub/manage_endpoints.py up`.
            CRITICAL: must set Content-Type: audio/wav — InferenceClient
            omits this header and the endpoint rejects the request.
            (Verified 2026-04-28, documented in reverse_wer.py.)
  local     Load model weights from HF Hub and run inference in-process.
            Requires GPU for reasonable speed. Use for dev/debug only
            when endpoints are not up.

Default is endpoint mode with placeholder URLs — set them from the output
of `python scripts/hub/manage_endpoints.py up ...`.

2026-05-07: FrenchASR added for Mode C (--input-lang fr).
"""
import time
import unicodedata
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

# Cold-start retry policy — mirrors CachingASRClient in reverse_wer.py
_RETRY_BACKOFFS = (5, 15, 30, 60)   # up to ~110s total
_COLD_START_WAIT = 65               # explicit wait on "model is loading" response


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------

def _load_audio(path: str, target_sr: int = 16000) -> np.ndarray:
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if sr != target_sr:
        n_out = int(len(audio) * target_sr / sr)
        audio = np.interp(np.linspace(0, len(audio) - 1, n_out), np.arange(len(audio)), audio)
    return audio.astype(np.float32)


def _parse_hf_path(path: str) -> tuple[str, Optional[str]]:
    parts = path.split("/")
    if len(parts) >= 3:
        return f"{parts[0]}/{parts[1]}", "/".join(parts[2:])
    return path, None


def _clean_ctc_markers(text: str) -> str:
    """Strip <pad>, <unk>, <blank> markers that some Wav2Vec2 endpoints emit.

    Copied from reverse_wer.py:_clean_hyp. CTC tokenizers that don't register
    these as additional_special_tokens leak them into the decoded text.
    """
    import re
    text = re.sub(r"<[a-z_]+>", " ", text)
    return " ".join(text.split())


# ---------------------------------------------------------------------------
# Endpoint ASR (primary mode — uses running HF Inference Endpoints)
# ---------------------------------------------------------------------------

class EndpointASR:
    """POST raw WAV bytes to a HF Inference Endpoint and parse the transcript.

    HTTP pattern from CachingASRClient in experiments/tts/eval/reverse_wer.py.
    Content-Type: audio/wav is mandatory — omitting it causes the endpoint
    to reject with 'Content type "None" not supported'.
    """

    def __init__(self, endpoint_url: str, label: str = ""):
        self.endpoint_url = endpoint_url.rstrip("/")
        self.label = label
        self._token: Optional[str] = None

    def _get_token(self) -> str:
        if self._token is None:
            import os
            try:
                from huggingface_hub import get_token
                self._token = os.environ.get("HF_TOKEN") or get_token() or ""
            except Exception:
                self._token = os.environ.get("HF_TOKEN", "")
        return self._token

    def transcribe(self, audio_path: str) -> str:
        import requests
        token = self._get_token()
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()

        last_err: Optional[Exception] = None
        for attempt, backoff in enumerate([0, *_RETRY_BACKOFFS]):
            if backoff:
                time.sleep(backoff)
            try:
                r = requests.post(
                    self.endpoint_url,
                    data=audio_bytes,
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Content-Type": "audio/wav",
                        "Accept": "application/json",
                    },
                    timeout=180,
                )
                if r.status_code == 503 or "loading" in r.text.lower():
                    print(f"  [{self.label}] cold start (attempt {attempt + 1}), "
                          f"waiting {_COLD_START_WAIT}s ...")
                    time.sleep(_COLD_START_WAIT)
                    continue
                r.raise_for_status()
                raw = r.json()
                if isinstance(raw, dict):
                    hyp = raw.get("text", "")
                elif isinstance(raw, list) and raw:
                    hyp = raw[0].get("text", "") if isinstance(raw[0], dict) else str(raw[0])
                else:
                    hyp = str(raw)
                hyp = _clean_ctc_markers(hyp)
                return unicodedata.normalize("NFC", hyp.strip())
            except Exception as exc:
                last_err = exc
                msg = str(exc).lower()
                if "loading" in msg or "503" in msg or "cold-start" in msg:
                    print(f"  [{self.label}] cold start (attempt {attempt + 1}), "
                          f"waiting {_COLD_START_WAIT}s ...")
                    time.sleep(_COLD_START_WAIT)
                else:
                    print(f"  [{self.label}] attempt {attempt + 1} error: {str(exc)[:120]}")

        raise RuntimeError(f"[{self.label}] ASR endpoint failed after all retries: {last_err}")


# ---------------------------------------------------------------------------
# Local ASR — Whisper (E4v4)
# ---------------------------------------------------------------------------

class WhisperASR:
    """Whisper-Ewe fine-tuned on Adja (E4v4 checkpoint). Local inference."""

    def __init__(self, model_id: str):
        self.model_id = model_id
        self._model = None
        self._processor = None
        self._device = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        repo_id, subfolder = _parse_hf_path(self.model_id)
        print(f"[ASR/Whisper] Loading local model from {self.model_id} ...")
        kw: dict = {}
        if subfolder:
            kw["subfolder"] = subfolder
        self._processor = WhisperProcessor.from_pretrained(repo_id, **kw)
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self._model = WhisperForConditionalGeneration.from_pretrained(
            repo_id, torch_dtype=dtype, **kw
        )
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device).eval()
        self._model.config.forced_decoder_ids = None
        self._model.generation_config.forced_decoder_ids = None
        print(f"[ASR/Whisper] Ready on {self._device}")

    def transcribe(self, audio_path: str) -> str:
        import torch
        self._load()
        audio = _load_audio(audio_path, target_sr=16000)
        inputs = self._processor(audio, sampling_rate=16000, return_tensors="pt")
        feats = inputs.input_features.to(self._device)
        if self._model.dtype == torch.float16:
            feats = feats.half()
        with torch.no_grad():
            ids = self._model.generate(feats, max_new_tokens=225)
        text = self._processor.batch_decode(ids, skip_special_tokens=True)[0]
        return unicodedata.normalize("NFC", text.strip())


# ---------------------------------------------------------------------------
# Local ASR — XLS-R CTC (C4v2)
# ---------------------------------------------------------------------------

class XLSRASR:
    """XLS-R 300M CTC fine-tuned on Adja, greedy decoding. Local inference.

    Checkpoint may contain the older ctc_vocab.json from the C4 training
    script or the standard Hub vocab.json emitted by the Transformers CTC
    tokenizer.
    """

    def __init__(self, model_id: str, base_model: str = "facebook/wav2vec2-xls-r-300m"):
        self.model_id = model_id
        self.base_model = base_model
        self._model = None
        self._feature_extractor = None
        self._idx2char: dict[int, str] = {}
        self._device = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import json
        import torch
        from huggingface_hub import hf_hub_download
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC

        repo_id, subfolder = _parse_hf_path(self.model_id)
        print(f"[ASR/XLSR] Loading local model from {self.model_id} ...")
        self._feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.base_model)
        vocab_candidates = ["ctc_vocab.json", "vocab.json"]
        vocab_local = None
        for vocab_name in vocab_candidates:
            vocab_fn = f"{subfolder}/{vocab_name}" if subfolder else vocab_name
            try:
                vocab_local = hf_hub_download(repo_id, filename=vocab_fn)
                break
            except Exception:
                if vocab_name == vocab_candidates[-1]:
                    raise
        assert vocab_local is not None
        with open(vocab_local, encoding="utf-8") as f:
            char2idx: dict[str, int] = json.load(f)
        self._idx2char = {v: k for k, v in char2idx.items()}
        kw: dict = {"vocab_size": len(char2idx)}
        if subfolder:
            kw["subfolder"] = subfolder
        self._model = Wav2Vec2ForCTC.from_pretrained(repo_id, **kw)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device).eval()
        print(f"[ASR/XLSR] Ready — {len(char2idx)} tokens on {self._device}")

    def _greedy_ctc(self, logits) -> str:
        import torch
        preds = torch.argmax(logits, dim=-1).squeeze(0).tolist()
        chars: list[str] = []
        prev = None
        for tok in preds:
            if tok != 0 and tok != prev:   # blank_id = 0
                ch = self._idx2char.get(tok, "")
                if ch in {"<pad>", "<unk>", "<blank>"}:
                    ch = ""
                elif ch == "|":
                    ch = " "
                chars.append(ch)
            prev = tok
        return unicodedata.normalize("NFC", "".join(chars).strip())

    def transcribe(self, audio_path: str) -> str:
        import torch
        self._load()
        audio = _load_audio(audio_path, target_sr=16000)
        inputs = self._feature_extractor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
        input_values = inputs.input_values.to(self._device)
        with torch.no_grad():
            logits = self._model(input_values).logits
        return self._greedy_ctc(logits)


# ---------------------------------------------------------------------------
# French ASR — Mode C (--input-lang fr)
# ---------------------------------------------------------------------------

class FrenchASR:
    """Standard Whisper for French speech transcription.

    Mode C (2026-05-07): When the user provides French audio instead of Adja
    audio (--input-lang fr), we use openai/whisper-small with language=fr to
    transcribe it. No fine-tuning needed — French is well-represented in
    Whisper's training data. The output goes directly to the response stage,
    bypassing MT→FR.

    Model: openai/whisper-small by default. Any Whisper checkpoint works;
    whisper-large-v3 gives better accuracy but loads slower on CPU.
    """

    def __init__(self, model_id: str = "openai/whisper-small"):
        self.model_id = model_id
        self._model = None
        self._processor = None
        self._device = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        print(f"[ASR/French] Loading {self.model_id} ...")
        self._processor = WhisperProcessor.from_pretrained(self.model_id)
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self._model = WhisperForConditionalGeneration.from_pretrained(
            self.model_id, torch_dtype=dtype
        )
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device).eval()
        print(f"[ASR/French] Ready on {self._device}")

    def transcribe(self, audio_path: str) -> str:
        import torch
        self._load()
        audio = _load_audio(audio_path, target_sr=16000)
        inputs = self._processor(
            audio, sampling_rate=16000, return_tensors="pt"
        )
        feats = inputs.input_features.to(self._device)
        if self._model.dtype == torch.float16:
            feats = feats.half()
        # Force French language + transcription task.
        # Use generate_config approach (language/task kwargs) to avoid the
        # "forced_decoder_ids is deprecated" warning in Transformers ≥ 4.40.
        with torch.no_grad():
            ids = self._model.generate(
                feats,
                language="fr",
                task="transcribe",
                max_new_tokens=225,
            )
        text = self._processor.batch_decode(ids, skip_special_tokens=True)[0]
        return unicodedata.normalize("NFC", text.strip())


# ---------------------------------------------------------------------------
# Stub (dry-run)
# ---------------------------------------------------------------------------

class StubASR:
    def __init__(self, label: str = ""):
        self.label = label

    def transcribe(self, audio_path: str) -> str:
        name = Path(audio_path).stem.replace("_", " ")
        return f"[STUB-{self.label or 'asr'}] {name[:50]}"


# ---------------------------------------------------------------------------
# Per-model builder
# ---------------------------------------------------------------------------

def _build_model(cfg: dict, label: str, dry_run: bool):
    """Build the right backend for one ASR model config entry."""
    if dry_run:
        return StubASR(label)
    mode = cfg.get("mode", "endpoint")
    if mode == "endpoint":
        url = cfg.get("endpoint_url", "")
        if not url or "PLACEHOLDER" in url:
            raise ValueError(
                f"[ASR/{label}] mode=endpoint but endpoint_url is not set. "
                "Run `python scripts/hub/manage_endpoints.py up ...` first, "
                "or set mode=local in p1_config.yaml."
            )
        return EndpointASR(url, label=label)
    else:
        model_id = cfg.get("model_id", cfg.get("checkpoint", ""))
        if label == "whisper":
            return WhisperASR(model_id)
        return XLSRASR(model_id, cfg.get("base_model", "facebook/wav2vec2-xls-r-300m"))


# ---------------------------------------------------------------------------
# Stage wrapper
# ---------------------------------------------------------------------------

class ASRStage:
    """Runs both ASR models; returns whisper, xlsr, and primary outputs."""

    def __init__(self, config: dict, dry_run: bool = False):
        self.primary = config.get("primary", "xlsr")
        self._whisper = _build_model(config.get("whisper", {}), "whisper", dry_run)
        self._xlsr = _build_model(config.get("xlsr", {}), "xlsr", dry_run)

    def run(self, audio_path: str) -> dict[str, str]:
        whisper_out = self._whisper.transcribe(audio_path)
        xlsr_out = self._xlsr.transcribe(audio_path)
        primary_out = whisper_out if self.primary == "whisper" else xlsr_out
        return {"whisper": whisper_out, "xlsr": xlsr_out, "primary": primary_out}
