#!/usr/bin/env python3
from __future__ import annotations
"""
Stage 2 / Stage 4: Machine Translation — Adja ↔ French.

Uses two directional fine-tuned NLLB checkpoints with the custom aj_Latn
token (same fix_tokenizer + resize_token_embeddings pattern as in training).
The Adja->FR and FR->Adja models are separate fine-tunes; each one is loaded
independently and used only for its trained direction.

Modes (set in p1_config.yaml):
  nllb   — two fine-tuned NLLB checkpoints, one per translation direction
  stub   — identity transform, no model needed (default for dry runs)
  gemini — Google Gemini API fallback when checkpoint is not ready yet
"""
import os
import unicodedata
from typing import Optional


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text.strip())


def _parse_hf_path(path: str) -> tuple[str, Optional[str]]:
    parts = path.split("/")
    if len(parts) >= 3:
        return f"{parts[0]}/{parts[1]}", "/".join(parts[2:])
    return path, None


# ---------------------------------------------------------------------------
# NLLB translator
# ---------------------------------------------------------------------------

class NLLBTranslator:
    """Fine-tuned NLLB translator for one configured direction.

    The tokenizer language arguments are still supplied at call time, but the
    checkpoint itself is directional in the live P1 pipeline. MTStage owns the
    routing so a forward-only fine-tune is never used for reverse translation.
    """

    def __init__(self, checkpoint: str, src_lang_adja: str = "aj_Latn",
                 tgt_lang_fr: str = "fra_Latn", max_length: int = 256, num_beams: int = 4,
                 no_repeat_ngram_size: int = 3, repetition_penalty: float = 1.2):
        self.checkpoint = checkpoint
        self.src_lang_adja = src_lang_adja
        self.tgt_lang_fr = tgt_lang_fr
        self.max_length = max_length
        self.num_beams = num_beams
        # 2026-05-07: Without repetition controls NLLB-600M enters token loops
        # on inputs longer than ~50 chars (observed: "gbea gbea gbea...").
        # no_repeat_ngram_size=3 prevents any 3-gram from repeating; combined
        # with repetition_penalty=1.2 this eliminates the loop without hurting
        # quality on short sentences. Standard practice for NLLB inference.
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.repetition_penalty = repetition_penalty
        self._model = None
        self._tokenizer = None
        self._device = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForSeq2SeqLM, NllbTokenizer

        repo_id, subfolder = _parse_hf_path(self.checkpoint)
        print(f"[MT/NLLB] Loading from {self.checkpoint} ...")
        load_kwargs: dict = {}
        if subfolder:
            load_kwargs["subfolder"] = subfolder

        self._tokenizer = NllbTokenizer.from_pretrained(repo_id, **load_kwargs)

        # Add aj_Latn if not already in vocabulary (same fix as training)
        if self.src_lang_adja not in self._tokenizer.additional_special_tokens:
            self._tokenizer.add_special_tokens(
                {"additional_special_tokens": [self.src_lang_adja]}
            )

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self._model = AutoModelForSeq2SeqLM.from_pretrained(
            repo_id, torch_dtype=dtype, **load_kwargs
        )
        self._model.resize_token_embeddings(len(self._tokenizer))
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device).eval()
        print(f"[MT/NLLB] Ready on {self._device}")

    def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        import torch
        self._load()
        self._tokenizer.src_lang = src_lang
        text = _nfc(text)
        inputs = self._tokenizer(
            text, return_tensors="pt", padding=True,
            truncation=True, max_length=self.max_length,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        tgt_id = self._tokenizer.convert_tokens_to_ids(tgt_lang)
        with torch.no_grad():
            out_ids = self._model.generate(
                **inputs,
                forced_bos_token_id=tgt_id,
                max_length=self.max_length,
                num_beams=self.num_beams,
                no_repeat_ngram_size=self.no_repeat_ngram_size,
                repetition_penalty=self.repetition_penalty,
            )
        return _nfc(self._tokenizer.batch_decode(out_ids, skip_special_tokens=True)[0])


# ---------------------------------------------------------------------------
# Gemini translator (fallback)
# ---------------------------------------------------------------------------

class GeminiTranslator:
    """Gemini API fallback — use when NLLB checkpoint is not yet available."""

    _LANG_NAMES = {
        "aj_Latn": "Adja (a Gbe language from Benin, written in Latin script)",
        "fra_Latn": "French",
    }

    def __init__(self, model: str = "gemini-2.0-flash"):
        self.model = model
        self._client = None

    def _load(self) -> None:
        if self._client is not None:
            return
        import google.generativeai as genai  # type: ignore
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise SystemExit("GEMINI_API_KEY env var required for Gemini MT mode")
        genai.configure(api_key=api_key)
        self._client = genai.GenerativeModel(self.model)

    def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        self._load()
        src_name = self._LANG_NAMES.get(src_lang, src_lang)
        tgt_name = self._LANG_NAMES.get(tgt_lang, tgt_lang)
        prompt = (
            f"Translate the following text from {src_name} to {tgt_name}.\n"
            f"Output only the translation, nothing else.\n\nText: {_nfc(text)}"
        )
        response = self._client.generate_content(prompt)
        return _nfc(response.text.strip())


# ---------------------------------------------------------------------------
# Stub (dry-run / placeholder)
# ---------------------------------------------------------------------------

class StubTranslator:
    def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        return f"[STUB MT {src_lang}→{tgt_lang}] {text[:80]}"


# ---------------------------------------------------------------------------
# Stage wrapper
# ---------------------------------------------------------------------------

class MTStage:
    """Wraps configured MT backends; exposes adja_to_fr / fr_to_adja.

    In nllb mode this stage loads two translator instances. The forward
    checkpoint is trained Adja->French and feeds the LLM. The reverse
    checkpoint is trained French->Adja and feeds TTS. Treating these as a
    single bidirectional model would run one of the directions out of
    distribution and produce poor cascade text.
    """

    def __init__(self, config: dict, dry_run: bool = False):
        self.src_lang_adja = config.get("src_lang_adja", "aj_Latn")
        self.tgt_lang_fr = config.get("tgt_lang_fr", "fra_Latn")
        mode = config.get("mode", "stub")
        self._translator: NLLBTranslator | GeminiTranslator | StubTranslator | None = None
        self._fwd_translator: NLLBTranslator | None = None
        self._rev_translator: NLLBTranslator | None = None

        if dry_run:
            self._translator = StubTranslator()
        elif mode == "nllb":
            fwd_ckpt = config.get("forward_checkpoint", "")
            rev_ckpt = config.get("reverse_checkpoint", "")
            if not fwd_ckpt or "PLACEHOLDER" in fwd_ckpt:
                raise ValueError(
                    "mt.mode is 'nllb' but mt.forward_checkpoint is not set in p1_config.yaml. "
                    "Set it to the Adja->French NLLB checkpoint or switch mode to 'stub'."
                )
            if not rev_ckpt or "PLACEHOLDER" in rev_ckpt:
                raise ValueError(
                    "mt.mode is 'nllb' but mt.reverse_checkpoint is not set in p1_config.yaml. "
                    "Set it to the French->Adja NLLB checkpoint or switch mode to 'stub'."
                )
            _common = dict(
                src_lang_adja=self.src_lang_adja,
                tgt_lang_fr=self.tgt_lang_fr,
                max_length=config.get("max_length", 256),
                num_beams=config.get("num_beams", 4),
                no_repeat_ngram_size=config.get("no_repeat_ngram_size", 3),
                repetition_penalty=config.get("repetition_penalty", 1.2),
            )
            self._fwd_translator = NLLBTranslator(fwd_ckpt, **_common)
            self._rev_translator = NLLBTranslator(rev_ckpt, **_common)
        elif mode == "gemini":
            self._translator = GeminiTranslator(config.get("gemini_model", "gemini-2.0-flash"))
        else:
            self._translator = StubTranslator()

    def adja_to_fr(self, text: str) -> str:
        if self._fwd_translator is not None:
            return self._fwd_translator.translate(text, self.src_lang_adja, self.tgt_lang_fr)
        if self._translator is None:
            raise RuntimeError("MT forward translator is not configured")
        return self._translator.translate(text, self.src_lang_adja, self.tgt_lang_fr)

    def fr_to_adja(self, text: str) -> str:
        if self._rev_translator is not None:
            return self._rev_translator.translate(text, self.tgt_lang_fr, self.src_lang_adja)
        if self._translator is None:
            raise RuntimeError("MT reverse translator is not configured")
        return self._translator.translate(text, self.tgt_lang_fr, self.src_lang_adja)
