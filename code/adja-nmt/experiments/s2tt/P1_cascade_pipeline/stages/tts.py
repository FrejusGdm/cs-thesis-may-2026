#!/usr/bin/env python3
from __future__ import annotations
"""
Stage 5: TTS — Adja text → Adja speech (Spark TTS 0.5B).

Spark TTS (SparkAudio/Spark-TTS) uses a Qwen2-based LLM with BiCodec audio
tokenizer. The only model family with native-speaker-confirmed intelligible
Adja output (2026-04). Architecture: https://github.com/SparkAudio/Spark-TTS

Canonical Spark variant for the pipeline:
  JosueG/spark-tts-adja-t3 (canonical T3, CTC reverse-CER 36.14% — lowest)
See scripts/hub/best_models.yaml for all three variants with reverse-WER numbers.

Two inference modes (set in p1_config.yaml):
  endpoint  POST JSON {"inputs": text} to a running HF Inference Endpoint.
            Response is raw 16 kHz WAV bytes. Deploy first with:
              python scripts/hub/publish_spark_t3.py
              python scripts/hub/manage_endpoints.py up \\
                JosueG/spark-tts-adja-t3 --task text-to-speech --gpu
  local     Load Spark + BiCodec in-process. Requires GPU. For dev/debug
            before the endpoint is up.

Default is local mode until the endpoint is deployed.
"""
import re
import sys
import unicodedata
from pathlib import Path
from typing import Optional

import time

_COLD_START_WAIT = 65
_RETRY_BACKOFFS = (5, 15, 30, 60)


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text.strip())


def _parse_hf_path(path: str) -> tuple[str, Optional[str]]:
    parts = path.split("/")
    if len(parts) >= 3:
        return f"{parts[0]}/{parts[1]}", "/".join(parts[2:])
    return path, None


# ---------------------------------------------------------------------------
# Endpoint TTS (primary once Spark endpoint is deployed)
# ---------------------------------------------------------------------------

class EndpointTTS:
    """POST text to a deployed Spark TTS Inference Endpoint, get WAV bytes back.

    The handler (scripts/hub/spark_handler.py) accepts:
      {"inputs": "<adja-text>", "parameters": {...}}  (optional gender/pitch/speed)
    and returns raw 16 kHz WAV bytes.
    """

    def __init__(self, endpoint_url: str):
        self.endpoint_url = endpoint_url.rstrip("/")
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

    def synthesize(self, text: str, output_path: str) -> str:
        import requests
        import soundfile as sf
        import numpy as np
        import io

        token = self._get_token()
        text = _nfc(text)
        payload = {"inputs": text}

        last_err: Optional[Exception] = None
        for attempt, backoff in enumerate([0, *_RETRY_BACKOFFS]):
            if backoff:
                time.sleep(backoff)
            try:
                r = requests.post(
                    self.endpoint_url,
                    json=payload,
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=180,
                )
                if r.status_code == 503 or "loading" in r.text.lower():
                    print(f"  [TTS/endpoint] cold start (attempt {attempt + 1}), "
                          f"waiting {_COLD_START_WAIT}s ...")
                    time.sleep(_COLD_START_WAIT)
                    continue
                r.raise_for_status()
                # 2026-05-07: handler now returns {"audio": "<base64>", "sample_rate": 16000}
                # because HF Inference Endpoints toolkit cannot JSON-serialize raw bytes.
                # Fall back to r.content for any future handler that returns raw WAV bytes.
                import base64 as _b64
                ct = r.headers.get("Content-Type", "")
                if "json" in ct or r.content.startswith(b"{"):
                    data = r.json()
                    if isinstance(data, dict) and "audio" in data:
                        wav_bytes = _b64.b64decode(data["audio"])
                    else:
                        # Unexpected JSON shape — warn and attempt raw
                        print(f"  [TTS/endpoint] unexpected JSON shape: {str(data)[:120]}")
                        wav_bytes = r.content
                else:
                    wav_bytes = r.content  # raw WAV bytes (legacy)
                Path(output_path).write_bytes(wav_bytes)
                return output_path
            except Exception as exc:
                last_err = exc
                msg = str(exc).lower()
                if "loading" in msg or "503" in msg:
                    print(f"  [TTS/endpoint] cold start (attempt {attempt + 1}), "
                          f"waiting {_COLD_START_WAIT}s ...")
                    time.sleep(_COLD_START_WAIT)
                else:
                    print(f"  [TTS/endpoint] attempt {attempt + 1} error: {str(exc)[:120]}")

        raise RuntimeError(f"[TTS/endpoint] failed after all retries: {last_err}")


# ---------------------------------------------------------------------------
# Local Spark TTS (fallback / dev mode)
# ---------------------------------------------------------------------------

class SparkTTS:
    """Spark TTS 0.5B inference in-process. Requires GPU + unsloth."""

    def __init__(self, checkpoint: str, base_model: str = "unsloth/Spark-TTS-0.5B",
                 max_new_tokens: int = 2048, temperature: float = 0.8, top_k: int = 50):
        self.checkpoint = checkpoint
        self.base_model = base_model
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self._model = None
        self._tokenizer = None
        self._audio_tokenizer = None
        self._sample_rate: int = 16000
        self._device: str = "cpu"

    def _load(self) -> None:
        if self._model is not None:
            return
        import subprocess
        import torch
        from huggingface_hub import snapshot_download
        from unsloth import FastModel  # type: ignore

        spark_src = Path("/tmp/Spark-TTS")
        if not spark_src.exists():
            print("[TTS/Spark] Cloning SparkAudio/Spark-TTS ...")
            subprocess.check_call(["git", "clone", "--depth", "1",
                                   "https://github.com/SparkAudio/Spark-TTS", str(spark_src)])
        if str(spark_src) not in sys.path:
            sys.path.insert(0, str(spark_src))
        from sparktts.models.audio_tokenizer import BiCodecTokenizer  # type: ignore

        print(f"[TTS/Spark] Downloading base assets from {self.base_model} ...")
        base_local = snapshot_download(self.base_model)

        repo_id, subfolder = _parse_hf_path(self.checkpoint)
        print(f"[TTS/Spark] Loading LLM from {self.checkpoint} ...")
        if subfolder:
            ckpt_local = snapshot_download(repo_id, allow_patterns=f"{subfolder}/*")
            model_path = str(Path(ckpt_local) / subfolder)
        else:
            model_path = repo_id

        self._model, self._tokenizer = FastModel.from_pretrained(
            model_name=model_path,
            max_seq_length=2048,
            dtype=torch.bfloat16,
            full_finetuning=False,
            load_in_4bit=False,
        )
        FastModel.for_inference(self._model)
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._audio_tokenizer = BiCodecTokenizer(base_local, self._device)
        self._sample_rate = self._audio_tokenizer.config.get("sample_rate", 16000)
        print(f"[TTS/Spark] Ready — sample_rate={self._sample_rate}")

    def synthesize(self, text: str, output_path: str) -> str:
        import torch
        import soundfile as sf
        self._load()
        text = _nfc(text)
        prompt = f"<|task_tts|><|start_content|>{text}<|end_content|><|start_global_token|>"
        inputs = self._tokenizer([prompt], return_tensors="pt").to(self._device)
        with torch.no_grad():
            gen_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=1.0,
                eos_token_id=self._tokenizer.eos_token_id,
                pad_token_id=self._tokenizer.pad_token_id,
            )
        gen_trimmed = gen_ids[:, inputs.input_ids.shape[1]:]
        gen_text = self._tokenizer.batch_decode(gen_trimmed, skip_special_tokens=False)[0]

        semantic_ids = [int(m) for m in re.findall(r"<\|bicodec_semantic_(\d+)\|>", gen_text)]
        global_ids = [int(m) for m in re.findall(r"<\|bicodec_global_(\d+)\|>", gen_text)]
        if not semantic_ids:
            raise RuntimeError(f"Spark TTS: no semantic tokens for: {text[:60]!r}")

        pred_sem = torch.tensor(semantic_ids, dtype=torch.long).unsqueeze(0).to(self._device)
        pred_glob = (torch.tensor(global_ids, dtype=torch.long).unsqueeze(0).unsqueeze(0).to(self._device)
                     if global_ids else torch.zeros((1, 1, 1), dtype=torch.long, device=self._device))

        wav_np = self._audio_tokenizer.detokenize(pred_glob.squeeze(0), pred_sem)
        sf.write(output_path, wav_np, self._sample_rate)
        return output_path


# ---------------------------------------------------------------------------
# Stub (dry-run)
# ---------------------------------------------------------------------------

class StubTTS:
    """Writes a 1-second silent WAV — pipeline wiring test, no GPU needed."""

    def synthesize(self, text: str, output_path: str) -> str:
        import numpy as np
        import soundfile as sf
        sf.write(output_path, np.zeros(16000, dtype=np.float32), 16000)
        return output_path


# ---------------------------------------------------------------------------
# Stage wrapper
# ---------------------------------------------------------------------------

class TTSStage:
    """TTS stage: endpoint, local Spark, or dry-run stub."""

    def __init__(self, config: dict, dry_run: bool = False):
        if dry_run:
            self._tts: EndpointTTS | SparkTTS | StubTTS = StubTTS()
            return

        mode = config.get("mode", "local")
        if mode == "endpoint":
            url = config.get("endpoint_url", "")
            if not url or "PLACEHOLDER" in url:
                raise ValueError(
                    "[TTS] mode=endpoint but endpoint_url is not set. "
                    "Deploy Spark first: `python scripts/hub/publish_spark_t3.py` "
                    "then `python scripts/hub/manage_endpoints.py up ... --task text-to-speech --gpu`, "
                    "or set mode=local in p1_config.yaml."
                )
            self._tts = EndpointTTS(url)
        else:
            # local mode
            ckpt = config.get("checkpoint", "")
            if not ckpt or "PLACEHOLDER" in ckpt:
                raise ValueError(
                    "[TTS] mode=local but no checkpoint set in p1_config.yaml. "
                    "Publish first with `python scripts/hub/publish_spark_t3.py`."
                )
            self._tts = SparkTTS(
                checkpoint=ckpt,
                base_model=config.get("base_model", "unsloth/Spark-TTS-0.5B"),
                max_new_tokens=config.get("max_new_tokens", 2048),
                temperature=config.get("temperature", 0.8),
                top_k=config.get("top_k", 50),
            )

    def run(self, text: str, output_path: str) -> str:
        return self._tts.synthesize(text, output_path)
