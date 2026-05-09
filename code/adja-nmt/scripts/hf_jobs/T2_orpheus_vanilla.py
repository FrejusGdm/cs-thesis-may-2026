#!/usr/bin/env python3
from __future__ import annotations

"""
T2 diagnostic: 5-sample, 2-step sanity check for Orpheus 3B Adja TTS.

Validates the pipeline end-to-end before a long run:
    - HF dataset load (JosueG/adja-tts-orpheus) + 80/10/10 split
    - Orpheus English base load (canopylabs/orpheus-3b-0.1-ft)
    - SNAC 24kHz encode of 5 clips → Orpheus token layout → tokenizer wrap
    - 2 train steps + 1 eval → eval_loss must appear in metrics.json
    - 1 generated WAV written via SNAC decode (no IPython)
    - Optional Hub push

Success criteria (mirrors T1_csm_vanilla gate + T1 README's dry-run gate):
    1. install_env() returns without pip errors
    2. Orpheus base model loads in fp32, LoRA applies, gradient checkpointing OFF
    3. processed_train has >= 1 item with input_ids/labels/attention_mask
    4. eval_loss present in trainer history
    5. at least one non-empty WAV in /tmp/orpheus_adja_diag/generated/

If this fails, pivot per T1 README's hard-stop rule:
    - to scripts/hf_jobs/T6_mms_tts_ewe_finetune.py (MMS-TTS-Ewe, Gbe-family prior)

Uses the exact same pipeline as scripts/hf_jobs/T2_orpheus_finetune.py via
--dry-run, so any drift between diagnostic and canonical is impossible.
"""

import importlib.util
import sys
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)


def _load_canonical():
    """Import the canonical T2 script as a module so this stays a thin wrapper
    (avoids a 400-line copy drifting out of sync). HF Jobs submit packages
    either file standalone; for the diagnostic to be runnable via
    `hf jobs uv run`, ship both files or use --dry-run on the canonical directly.
    """
    here = Path(__file__).resolve().parent
    canonical = here / "T2_orpheus_finetune.py"
    spec = importlib.util.spec_from_file_location("t2_orpheus_finetune", canonical)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    argv_forced = [
        sys.argv[0],
        "--dry-run",
        "--output-dir", "/tmp/orpheus_adja_diag",
        "--results-prefix", "T2_diagnostic",
    ]
    # Preserve any extra flags the caller passed (e.g. --push-to-hub).
    passthrough = [a for a in sys.argv[1:] if a not in {"--dry-run"}]
    sys.argv = argv_forced + passthrough

    canonical = _load_canonical()
    canonical.main()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        raise
