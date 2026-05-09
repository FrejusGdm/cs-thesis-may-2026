"""
Inference Endpoints handler for Spark TTS Adja.

Hugging Face Inference Endpoints loads `handler.py` from the model repo
root, instantiates `EndpointHandler(model_dir)` once, and calls the
instance with a dict per request. Return value is the response body.

For TTS we want bytes back (a WAV file). Inference Endpoints accepts
returning a `bytes` object directly with `Content-Type: audio/wav`,
or a base64-encoded string inside JSON. We use raw bytes here because
the pipeline agent's HTTP client is simpler that way.

Layout this handler expects (set up by `scripts/hub/publish_spark_t3.py`):

    model_dir/
        LLM/                       Qwen2 fine-tune (LLM half of Spark)
        BiCodec/                   audio tokenizer / decoder
        wav2vec2-large-xlsr-53/    semantic encoder used by BiCodec
        config.yaml                Spark base config
        cli/SparkTTS.py            class SparkTTS(model_dir, device)
        sparktts/                  helper modules

Local smoke-test:

    cd <assembled-folder>
    python -c "import handler; h = handler.EndpointHandler('.'); \\
               wav = h({'inputs': 'Tɛnigbe ciyi vayi de ŋweba'}); \\
               open('/tmp/spark.wav', 'wb').write(wav)"
"""

from __future__ import annotations

import io
import os
import sys
import unicodedata
from pathlib import Path
from typing import Any


class EndpointHandler:
    """Hugging Face Inference Endpoints entry point."""

    def __init__(self, model_dir: str = "."):
        # The Spark library lives at <model_dir>/cli and <model_dir>/sparktts.
        # Add them to sys.path so `from cli.SparkTTS import SparkTTS` resolves.
        self.model_dir = Path(model_dir).resolve()
        sys.path.insert(0, str(self.model_dir))

        import torch  # noqa: WPS433
        self._torch = torch
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        print(f"[spark-handler] device={self.device} model_dir={self.model_dir}", flush=True)

        # Lazy import — these need sys.path patched first.
        from cli.SparkTTS import SparkTTS  # type: ignore[import-not-found]
        self.spark = SparkTTS(self.model_dir, device=self.device)
        print("[spark-handler] SparkTTS loaded", flush=True)

    def __call__(self, data: dict[str, Any]) -> bytes:
        """Synthesize Adja speech from text.

        Request shape:
          {
            "inputs": "Tɛnigbe ciyi vayi de ŋweba",
            "parameters": {                         # optional
              "gender": "female",                   # female | male
              "pitch": "moderate",                  # very_low | low | moderate | high | very_high
              "speed": "moderate",
              "temperature": 0.8,
              "top_k": 50,
              "top_p": 0.95
            }
          }

        Response: raw bytes of a 16 kHz mono WAV file. Set
        `Content-Type: audio/wav` on the way out.
        """
        text = data.get("inputs") or data.get("text") or ""
        text = unicodedata.normalize("NFC", str(text).strip())
        if not text:
            return b""

        params = data.get("parameters") or {}
        gender = params.get("gender", "female")
        pitch = params.get("pitch", "moderate")
        speed = params.get("speed", "moderate")
        temperature = float(params.get("temperature", 0.8))
        top_k = int(params.get("top_k", 50))
        top_p = float(params.get("top_p", 0.95))

        wav = self.spark.inference(
            text=text,
            gender=gender,
            pitch=pitch,
            speed=speed,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        # `wav` is a torch.Tensor on the active device. SparkTTS produces
        # 16 kHz mono. Encode to WAV via soundfile.
        import soundfile as sf  # noqa: WPS433
        if hasattr(wav, "detach"):
            wav = wav.detach().cpu().numpy()
        if wav.ndim > 1:
            wav = wav.squeeze()

        buf = io.BytesIO()
        sf.write(buf, wav, samplerate=16000, format="WAV", subtype="PCM_16")
        # 2026-05-07: HF Inference Endpoints toolkit JSON-serializes handler
        # return values; raw bytes are not JSON serializable and cause 400.
        # Return base64-encoded audio in a dict instead. EndpointTTS in
        # stages/tts.py decodes {"audio": "<b64>", "sample_rate": 16000}.
        import base64 as _b64
        return {
            "audio": _b64.b64encode(buf.getvalue()).decode("utf-8"),
            "sample_rate": 16000,
        }
