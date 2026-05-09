#!/usr/bin/env python3
# /// script
# dependencies = ["torch==2.5.1", "torchaudio==2.5.1", "huggingface-hub>=0.34.0", "datasets>=3.4.1,<4.0.0", "soundfile", "librosa", "numpy", "coqpit>=0.0.16", "pysbd>=0.3.4", "num2words", "unidecode>=1.3.2"]
# ///
from __future__ import annotations

"""
T9: Fine-tune XTTS-v2 GPT encoder on Adja via Hugging Face Jobs.

This launcher is self-contained and mirrors Coqui's public XTTS v2 recipe:
- materialize dataset in LJSpeech format
- run GPT encoder training
- save metrics and checkpoints

References:
  - https://huggingface.co/coqui/XTTS-v2
  - https://tts.readthedocs.io/en/latest/models/xtts.html
  - upstream recipe: recipes/ljspeech/xtts_v2/train_gpt_xtts.py
"""

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import time
import unicodedata
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)


UPSTREAM_REPO = "https://github.com/coqui-ai/TTS.git"
UPSTREAM_DIR = Path("/tmp/coqui-TTS")
DATA_DIR = Path("/tmp/xtts_adja")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="T9 XTTS-v2 fine-tune on Adja")
    parser.add_argument("--dataset", default="JosueG/adja-tts-orpheus")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--num-epochs", type=int, default=20)
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--results-repo", default="JosueG/adja-tts-results")
    parser.add_argument("--results-prefix", default="T9_xtts")
    parser.add_argument("--output-dir", default="/tmp/t9_xtts_output")
    return parser.parse_args()


def run(cmd: str, cwd: str | None = None) -> None:
    print(f"$ {cmd}")
    subprocess.check_call(cmd, shell=True, cwd=cwd)

def install_env() -> None:
    run("apt-get update -q && apt-get install -y -q git ffmpeg")


def clone_upstream() -> None:
    if UPSTREAM_DIR.exists():
        shutil.rmtree(UPSTREAM_DIR)
    run(f'git clone --depth 1 --branch dev "{UPSTREAM_REPO}" "{UPSTREAM_DIR}"')
    run('uv pip install -q -e ".[all]"', cwd=str(UPSTREAM_DIR))
    # XTTS currently imports several generation helpers from top-level
    # transformers symbols that moved across 4.x releases. Patch the import site
    # to a compatibility block that works across the layouts observed on HF Jobs.
    run(
        "python3 - <<'PY'\n"
        "from pathlib import Path\n"
        "path = Path('/tmp/coqui-TTS/TTS/tts/layers/xtts/stream_generator.py')\n"
        "text = path.read_text(encoding='utf-8')\n"
        "old = '''from transformers import (\\n"
        "    BeamSearchScorer,\\n"
        "    ConstrainedBeamSearchScorer,\\n"
        "    DisjunctiveConstraint,\\n"
        "    GenerationConfig,\\n"
        "    GenerationMixin,\\n"
        "    LogitsProcessorList,\\n"
        "    PhrasalConstraint,\\n"
        "    PreTrainedModel,\\n"
        "    StoppingCriteriaList,\\n"
        ")\\n"
        "from transformers.generation.utils import GenerateOutput, SampleOutput, logger\\n'''\n"
        "new = '''from transformers.modeling_utils import PreTrainedModel\\n"
        "try:\\n"
        "    from transformers import GenerationConfig, GenerationMixin\\n"
        "except ImportError:\\n"
        "    from transformers.generation.configuration_utils import GenerationConfig\\n"
        "    from transformers.generation.utils import GenerationMixin\\n"
        "try:\\n"
        "    from transformers.generation.beam_search import BeamSearchScorer, ConstrainedBeamSearchScorer\\n"
        "except ImportError:\\n"
        "    from transformers.generation_beam_search import BeamSearchScorer, ConstrainedBeamSearchScorer\\n"
        "try:\\n"
        "    from transformers.generation.beam_constraints import DisjunctiveConstraint, PhrasalConstraint\\n"
        "except ImportError:\\n"
        "    from transformers.generation_beam_constraints import DisjunctiveConstraint, PhrasalConstraint\\n"
        "try:\\n"
        "    from transformers.generation.logits_process import LogitsProcessorList\\n"
        "except ImportError:\\n"
        "    from transformers.generation_logits_process import LogitsProcessorList\\n"
        "try:\\n"
        "    from transformers.generation.stopping_criteria import StoppingCriteriaList\\n"
        "except ImportError:\\n"
        "    from transformers.generation_stopping_criteria import StoppingCriteriaList\\n"
        "from transformers.generation.utils import GenerateOutput, logger\\n"
        "try:\\n"
        "    from transformers.generation.utils import SampleOutput\\n"
        "except ImportError:\\n"
        "    SampleOutput = GenerateOutput\\n'''\n"
        "if old in text:\n"
        "    text = text.replace(old, new)\n"
        "else:\n"
        "    raise SystemExit('expected XTTS import block not found')\n"
        "path.write_text(text, encoding='utf-8')\n"
        "print('patched', path)\n"
        "PY",
        cwd=str(UPSTREAM_DIR),
    )


def nfc(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text.strip()).split())


def materialize_dataset(args: argparse.Namespace, token: str) -> tuple[Path, Path, Path]:
    import soundfile as sf
    from datasets import load_dataset

    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)
    wav_dir = DATA_DIR / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(args.dataset, token=token, split="train")
    split1 = ds.train_test_split(test_size=0.1, seed=args.seed)
    split2 = split1["train"].train_test_split(test_size=0.1 / 0.9, seed=args.seed)
    train_ds = split2["train"]
    dev_ds = split2["test"]
    test_ds = split1["test"]
    if args.dry_run:
        train_ds = train_ds.select(range(min(8, len(train_ds))))
        dev_ds = dev_ds.select(range(min(3, len(dev_ds))))
        test_ds = test_ds.select(range(min(3, len(test_ds))))

    def write_split(split, metadata_path: Path, prefix: str) -> None:
        lines = []
        for idx, example in enumerate(split):
            audio = example["audio"]["array"]
            sr = example["audio"]["sampling_rate"]
            text = nfc(example["text"])
            stem = f"{prefix}_{idx:06d}"
            wav_path = wav_dir / f"{stem}.wav"
            sf.write(str(wav_path), audio, sr)
            # LJSpeech formatter reads cols[0] as wav stem, cols[2] as text.
            lines.append(f"{stem}|{text}|{text}\n")
        metadata_path.write_text("".join(lines), encoding="utf-8")

    train_meta = DATA_DIR / "metadata.csv"
    dev_meta = DATA_DIR / "metadata_dev.csv"
    write_split(train_ds, train_meta, "train")
    write_split(dev_ds, dev_meta, "dev")
    return train_meta, dev_meta, test_ds


def build_train_script(args: argparse.Namespace, train_meta: Path, dev_meta: Path) -> Path:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_script_path = output_dir / "run_xtts_train.py"

    epochs = 1 if args.dry_run else args.num_epochs
    batch_size = 1 if args.dry_run else args.batch_size
    grad_accum = 1 if args.dry_run else args.grad_accum
    save_step = 2 if args.dry_run else 100

    script = f"""import os
from trainer import Trainer, TrainerArgs
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig, XttsAudioConfig
from TTS.utils.manage import ModelManager

RUN_NAME = "GPT_XTTS_Adja_FT"
PROJECT_NAME = "XTTS_Adja"
OUT_PATH = {json.dumps(str(output_dir))}
CHECKPOINTS_OUT_PATH = os.path.join(OUT_PATH, "XTTS_v2_original_model_files")
os.makedirs(CHECKPOINTS_OUT_PATH, exist_ok=True)

DVAE_CHECKPOINT_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/dvae.pth"
MEL_NORM_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/mel_stats.pth"
TOKENIZER_FILE_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/vocab.json"
XTTS_CHECKPOINT_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/model.pth"

for link in [MEL_NORM_LINK, DVAE_CHECKPOINT_LINK, TOKENIZER_FILE_LINK, XTTS_CHECKPOINT_LINK]:
    target = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(link))
    if not os.path.isfile(target):
        ModelManager._download_model_files([link], CHECKPOINTS_OUT_PATH, progress_bar=True)

config_dataset = BaseDatasetConfig(
    formatter="ljspeech",
    dataset_name="adja",
    path={json.dumps(str(DATA_DIR))},
    meta_file_train={json.dumps(train_meta.name)},
    meta_file_val={json.dumps(dev_meta.name)},
    language="en",
)
DATASETS_CONFIG_LIST = [config_dataset]

model_args = GPTArgs(
    max_conditioning_length=132300,
    min_conditioning_length=66150,
    debug_loading_failures=False,
    max_wav_length=255995,
    max_text_length=200,
    mel_norm_file=os.path.join(CHECKPOINTS_OUT_PATH, "mel_stats.pth"),
    dvae_checkpoint=os.path.join(CHECKPOINTS_OUT_PATH, "dvae.pth"),
    xtts_checkpoint=os.path.join(CHECKPOINTS_OUT_PATH, "model.pth"),
    tokenizer_file=os.path.join(CHECKPOINTS_OUT_PATH, "vocab.json"),
    gpt_num_audio_tokens=1026,
    gpt_start_audio_token=1024,
    gpt_stop_audio_token=1025,
    gpt_use_masking_gt_prompt_approach=True,
    gpt_use_perceiver_resampler=True,
)

audio_config = XttsAudioConfig(sample_rate=22050, dvae_sample_rate=22050, output_sample_rate=24000)
config = GPTTrainerConfig(
    epochs={epochs},
    output_path=OUT_PATH,
    model_args=model_args,
    run_name=RUN_NAME,
    project_name=PROJECT_NAME,
    run_description="XTTS Adja GPT fine-tuning",
    dashboard_logger="tensorboard",
    logger_uri=None,
    audio=audio_config,
    batch_size={batch_size},
    batch_group_size=8,
    eval_batch_size={batch_size},
    num_loader_workers=2,
    eval_split_max_size=64,
    print_step=1,
    plot_step=10,
    log_model_step=10,
    save_step={save_step},
    save_n_checkpoints=2,
    save_checkpoints=True,
    print_eval=False,
    optimizer="AdamW",
    optimizer_wd_only_on_weights=True,
    optimizer_params={{"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2}},
    lr={args.learning_rate},
    lr_scheduler="MultiStepLR",
    lr_scheduler_params={{"milestones": [1000, 2000, 4000], "gamma": 0.5, "last_epoch": -1}},
    test_sentences=[
        {{
            "text": "Nye ŋkɔ nyé Tom",
            "speaker_wav": [os.path.join({json.dumps(str(DATA_DIR))}, "wavs", os.listdir(os.path.join({json.dumps(str(DATA_DIR))}, "wavs"))[0])],
            "language": "en",
        }}
    ],
)

model = GPTTrainer.init_from_config(config)
train_samples, eval_samples = load_tts_samples(
    DATASETS_CONFIG_LIST,
    eval_split=True,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size,
)
trainer = Trainer(
    TrainerArgs(
        restore_path=None,
        skip_train_epoch=False,
        start_with_eval=True,
        grad_accum_steps={grad_accum},
    ),
    config,
    output_path=OUT_PATH,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
)
trainer.fit()
"""
    train_script_path.write_text(script, encoding="utf-8")
    return train_script_path


def collect_metrics(args: argparse.Namespace, elapsed_min: float) -> Path:
    output_dir = Path(args.output_dir)
    checkpoints = []
    if output_dir.exists():
        checkpoints = [str(p.relative_to(output_dir)) for p in output_dir.rglob("*") if p.is_file()]
    metrics = {
        "experiment": "T9",
        "model_name": "coqui/XTTS-v2",
        "dataset": args.dataset,
        "seed": args.seed,
        "dry_run": args.dry_run,
        "elapsed_min": round(elapsed_min, 1),
        "checkpoint_files": checkpoints[:200],
        "output_dir": str(output_dir),
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return metrics_path


def push_results(args: argparse.Namespace, token: str, metrics_path: Path) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(args.results_repo, private=True, exist_ok=True)
    prefix = args.results_prefix.rstrip("/")
    api.upload_file(
        path_or_fileobj=metrics_path.read_bytes(),
        path_in_repo=f"{prefix}/metrics.json",
        repo_id=args.results_repo,
        token=token,
    )
    try:
        api.upload_folder(
            folder_path=str(Path(args.output_dir)),
            path_in_repo=f"{prefix}/training_output",
            repo_id=args.results_repo,
            token=token,
            ignore_patterns=["**/__pycache__/**"],
        )
    except Exception as exc:
        print(f"Training output upload best-effort: {exc}")


def main() -> None:
    args = parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is required")

    random.seed(args.seed)

    print("=" * 60)
    print("T9 XTTS-v2 fine-tune on Adja")
    print("=" * 60)
    print(f"Dry run: {args.dry_run}")
    print()

    install_env()
    clone_upstream()
    train_meta, dev_meta, _ = materialize_dataset(args, token)
    train_script = build_train_script(args, train_meta, dev_meta)

    t0 = time.time()
    run(f'python3 "{train_script}"')
    elapsed_min = (time.time() - t0) / 60.0

    metrics_path = collect_metrics(args, elapsed_min)
    print(metrics_path.read_text(encoding="utf-8"))

    if args.push_to_hub:
        push_results(args, token, metrics_path)

    print("T9 complete.")


if __name__ == "__main__":
    main()
