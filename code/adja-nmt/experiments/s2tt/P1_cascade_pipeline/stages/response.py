#!/usr/bin/env python3
from __future__ import annotations
"""
Stage 3: Response generation — French text → French answer.

2026-05-02  Added OpenRouterResponse backend (replaces Gemini as default).
            Rationale: OpenRouter provides a single OpenAI-compatible endpoint
            for any hosted model (GPT-4o, Claude, Llama 3, Mistral, etc.).
            One API key, model slug in config → swap without code changes.
            Ref: https://openrouter.ai/docs/api-reference

Backends (set response.mode in p1_config.yaml):
  openrouter  POST to openrouter.ai/api/v1/chat/completions — default.
              Requires: OPENROUTER_API_KEY env var.
              Config:   response.model = any OpenRouter slug, e.g. "openai/gpt-4o"
  gemini      Google Gemini API (kept for backward compat).
              Requires: GEMINI_API_KEY env var.
  identity    Echo the French input unchanged — use for roundtrip eval where
              you want to measure error accumulation WITHOUT an LLM in the loop.
  stub        Dry-run stub, no network calls.

Why the LLM speaks French, not Adja:
  Adja is not in the training data of any public LLM. The MT legs (stages/mt.py)
  handle the Adja↔French bridge so the LLM only ever sees and produces French,
  where every model performs well. This is the key architectural insight: isolate
  the LLM from the low-resource language entirely.
"""
import os
import unicodedata


def _nfc(text: str) -> str:
    """Unicode NFC normalization — mandatory between every pipeline stage.

    Adja uses composed characters (ɛ, ɔ, ŋ, ɖ) and combining tone marks
    (é, è, ẽ). NFC ensures a single codepoint representation, which matters
    for the downstream MT tokenizer and for consistent display. NFKD would
    strip combining marks — never use it. See CLAUDE.md § Tokenizer / NFC.
    """
    return unicodedata.normalize("NFC", text.strip())


# ---------------------------------------------------------------------------
# OpenRouter backend (default)
# ---------------------------------------------------------------------------

class OpenRouterResponse:
    """Calls any LLM via OpenRouter's OpenAI-compatible chat completions API.

    Design choices (2026-05-02):
    - Uses plain `requests` rather than the `openai` SDK to avoid adding a
      heavyweight dependency. The OpenRouter spec is a strict superset of the
      OpenAI chat completions spec, so the request shape is identical.
    - HTTP-Referer and X-Title headers are recommended by OpenRouter for usage
      attribution and routing priority — they don't affect correctness but are
      good practice for research projects.
    - The system prompt instructs the model to answer in French. The question
      arrives in French (translated from Adja by stages/mt.py), so the LLM
      stays entirely in a language it was trained on.
    - No streaming: we need the full answer before passing it to MT→Adja.
    - 60s timeout: generous for a typical 1–2 sentence Q&A response.

    Ref: https://openrouter.ai/docs/api-reference
    """

    _OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, model: str, system_prompt: str):
        self.model = model
        self.system_prompt = system_prompt
        self._api_key: str | None = None

    def _get_key(self) -> str:
        if self._api_key is None:
            key = os.environ.get("OPENROUTER_API_KEY", "")
            if not key:
                raise SystemExit(
                    "OPENROUTER_API_KEY env var not set.\n"
                    "Get a key at https://openrouter.ai/keys and export it:\n"
                    "  export OPENROUTER_API_KEY=sk-or-..."
                )
            self._api_key = key
        return self._api_key

    def respond(self, french_text: str) -> str:
        import requests

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user",   "content": _nfc(french_text)},
            ],
        }
        r = requests.post(
            self._OPENROUTER_URL,
            json=payload,
            headers={
                "Authorization": f"Bearer {self._get_key()}",
                "Content-Type": "application/json",
                # Recommended by OpenRouter for attribution + routing priority
                "HTTP-Referer": "https://github.com/JosueG/adja-nmt",
                "X-Title": "Adja Pipeline P1",
            },
            timeout=60,
        )
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"]
        return _nfc(content.strip())


# ---------------------------------------------------------------------------
# Gemini backend (backward compat)
# ---------------------------------------------------------------------------

class GeminiResponse:
    """Generates a French response via Google Gemini API.

    Kept for backward compatibility and as a fallback when OpenRouter is
    unavailable. Prefer OpenRouter for new runs — model-agnostic config.
    Requires: GEMINI_API_KEY env var + `pip install google-generativeai`.
    """

    def __init__(self, model: str, system_prompt: str):
        self.model = model
        self.system_prompt = system_prompt
        self._client = None

    def _load(self) -> None:
        if self._client is not None:
            return
        import google.generativeai as genai  # type: ignore
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise SystemExit("GEMINI_API_KEY env var required for Gemini response mode")
        genai.configure(api_key=api_key)
        self._client = genai.GenerativeModel(
            self.model,
            system_instruction=self.system_prompt,
        )

    def respond(self, french_text: str) -> str:
        self._load()
        response = self._client.generate_content(_nfc(french_text))
        return _nfc(response.text.strip())


# ---------------------------------------------------------------------------
# Identity backend (roundtrip eval, no LLM)
# ---------------------------------------------------------------------------

class IdentityResponse:
    """Pass-through — echoes French input as the 'answer'.

    Use this when you want to measure error accumulation from ASR + MT alone,
    without introducing LLM variance. The roundtrip path becomes:
      Adja speech → ASR → MT→FR → [identity] → MT→Adja → TTS → RTT-ASR
    which isolates the speech ↔ text degradation signal.
    """

    def respond(self, french_text: str) -> str:
        return _nfc(french_text)


# ---------------------------------------------------------------------------
# Stub (dry-run)
# ---------------------------------------------------------------------------

class StubResponse:
    def respond(self, french_text: str) -> str:
        return f"[STUB RESPONSE] {french_text[:80]}"


# ---------------------------------------------------------------------------
# Stage wrapper
# ---------------------------------------------------------------------------

class ResponseStage:
    """Configurable response step.

    Selects backend from config['response']['mode']:
      openrouter  → OpenRouterResponse  (default, recommended)
      gemini      → GeminiResponse
      identity    → IdentityResponse    (roundtrip eval, no LLM)
      dry_run     → StubResponse        (pipeline wiring tests)
    """

    def __init__(self, config: dict, dry_run: bool = False):
        mode = config.get("mode", "openrouter")

        if dry_run:
            self._responder: OpenRouterResponse | GeminiResponse | IdentityResponse | StubResponse = StubResponse()
        elif mode == "openrouter":
            self._responder = OpenRouterResponse(
                model=config.get("model", "openai/gpt-4o"),
                system_prompt=config.get(
                    "system_prompt",
                    "Tu es un assistant utile. Réponds en français de manière concise et directe.",
                ),
            )
        elif mode == "gemini":
            self._responder = GeminiResponse(
                config.get("gemini_model", "gemini-2.0-flash"),
                config.get("system_prompt", "Réponds en français de manière concise."),
            )
        else:
            # identity: explicit or any unknown mode falls through safely
            self._responder = IdentityResponse()

    def run(self, french_text: str) -> str:
        return self._responder.respond(french_text)
