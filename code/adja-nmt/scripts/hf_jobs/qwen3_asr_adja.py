#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#     "torch==2.5.1",
#     "torchaudio==2.5.1",
#     "qwen-asr==0.0.6",
#     "transformers>=4.40.0",
#     "datasets>=2.18.0",
#     "accelerate>=0.28.0",
#     "safetensors>=0.4.0",
#     "librosa>=0.10.0",
#     "soundfile>=0.12.0",
#     "numpy>=1.24.0",
#     "tensorboard>=2.15.0",
#     "huggingface-hub>=0.34.0",
# ]
# ///
from __future__ import annotations

"""
Qwen3-ASR Adja fine-tuning launcher for Hugging Face Jobs.

Prepared for: Josue Godeme
Requested by: Josue Godeme

This launcher is intentionally standalone so it can run under:
    hf jobs uv run ... scripts/hf_jobs/qwen3_asr_adja.py

It does not assume the rest of the repo is present inside the container.
The vendored/local Qwen3 code remains in the repo as source material, but this
HF launcher inlines the minimum required logic to match the existing project
pattern for self-contained job scripts.
"""

import json
import inspect
import os
import re
import shutil
import sys
import traceback
import unicodedata
from pathlib import Path
from typing import Any

sys.stdout.reconfigure(line_buffering=True)


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def require_env(name: str, default: str | None = None, required: bool = True) -> str:
    value = os.environ.get(name, default)
    if required and not value:
        raise SystemExit(f"{name} is required")
    return value or ""


def setup_tee_logging(log_path: Path) -> Any:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("a", encoding="utf-8")
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = TeeStream(original_stdout, log_handle)
    sys.stderr = TeeStream(original_stderr, log_handle)
    return log_handle, original_stdout, original_stderr


def restore_tee_logging(log_handle: Any, original_stdout: Any, original_stderr: Any) -> None:
    sys.stdout.flush()
    sys.stderr.flush()
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    log_handle.close()


def normalize_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text.strip()).split())


def asr_text(text: str) -> str:
    return f"language None<asr_text>{text}"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def subset_jsonl(source: Path, destination: Path, limit: int) -> Path:
    write_jsonl(destination, read_jsonl(source)[:limit])
    return destination


def ensure_repo(api, repo_id: str, repo_type: str = "model") -> None:
    api.create_repo(repo_id, private=True, exist_ok=True, repo_type=repo_type)


def upload_file_if_exists(api, repo_id: str, local_path: Path, path_in_repo: str, repo_type: str = "model") -> None:
    if not local_path.exists():
        return
    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type=repo_type,
    )


def upload_folder_if_exists(api, repo_id: str, local_path: Path, path_in_repo: str | None = None, repo_type: str = "model") -> None:
    if not local_path.exists():
        return
    api.upload_folder(
        folder_path=str(local_path),
        repo_id=repo_id,
        path_in_repo=path_in_repo,
        repo_type=repo_type,
    )


def create_splits(dataset, seed: int):
    split1 = dataset.train_test_split(test_size=0.1, seed=seed)
    split2 = split1["train"].train_test_split(test_size=0.1 / 0.9, seed=seed)
    return {"train": split2["train"], "val": split2["test"], "test": split1["test"]}


def materialize_asr(dataset_id: str, token: str, workspace_dir: Path, seed: int) -> dict[str, Any]:
    import librosa
    import numpy as np
    import soundfile as sf
    from datasets import load_dataset

    dataset = load_dataset(dataset_id, token=token, split="train")
    splits = create_splits(dataset, seed)
    base_dir = workspace_dir / "asr"
    summary: dict[str, Any] = {
        "dataset_id": dataset_id,
        "seed": seed,
        "mode": "asr",
        "splits": {"asr": {}},
    }

    for split_name, split_data in splits.items():
        wav_dir = base_dir / "wavs" / split_name
        wav_dir.mkdir(parents=True, exist_ok=True)
        records: list[dict[str, Any]] = []
        total_duration = 0.0

        for index in range(len(split_data)):
            sample = split_data[index]
            audio = np.asarray(sample["audio"]["array"], dtype=np.float32)
            sr = int(sample["audio"]["sampling_rate"])
            if audio.ndim > 1:
                audio = np.mean(audio, axis=-1)
            if sr != 16000:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                sr = 16000

            wav_path = wav_dir / f"{split_name}_{index:05d}.wav"
            sf.write(wav_path, audio, sr)
            duration = len(audio) / float(sr)
            total_duration += duration
            records.append({"audio": str(wav_path), "text": asr_text(normalize_text(sample["text"]))})

        jsonl_path = base_dir / f"{split_name}.jsonl"
        write_jsonl(jsonl_path, records)
        summary["splits"]["asr"][split_name] = {
            "jsonl": str(jsonl_path),
            "num_records": len(records),
            "total_duration_sec": round(total_duration, 3),
            "target_sample_rate": 16000,
        }

    write_json(workspace_dir / "materialization_summary.json", summary)
    return summary


def validate_asr_jsonl(jsonl_path: Path) -> None:
    import soundfile as sf

    records = read_jsonl(jsonl_path)
    if not records:
        raise SystemExit(f"Empty ASR JSONL: {jsonl_path}")
    for record in records:
        audio_path = Path(record["audio"])
        if not audio_path.exists():
            raise FileNotFoundError(f"Missing audio file: {audio_path}")
        if not record["text"].startswith("language None<asr_text>"):
            raise ValueError(f"Unexpected ASR target text format in {jsonl_path}")
        info = sf.info(str(audio_path))
        if info.samplerate != 16000:
            raise ValueError(f"Unexpected sample rate for {audio_path}: {info.samplerate}")


def find_latest_checkpoint(output_dir: Path) -> Path | None:
    pattern = re.compile(r"^checkpoint-(\d+)$")
    best: tuple[int, Path] | None = None
    if not output_dir.exists():
        return None
    for path in output_dir.iterdir():
        match = pattern.match(path.name)
        if not match or not path.is_dir():
            continue
        step = int(match.group(1))
        if best is None or step > best[0]:
            best = (step, path)
    return best[1] if best else None


def run_asr_train(train_jsonl: Path, val_jsonl: Path, output_dir: Path, model_id: str, seed: int, smoke: bool) -> tuple[Path, Path]:
    import librosa
    import torch
    from dataclasses import dataclass
    from datasets import load_dataset
    from huggingface_hub import snapshot_download
    from qwen_asr import Qwen3ASRModel
    from transformers import GenerationConfig, Trainer, TrainerCallback, TrainingArguments

    def patch_outer_forward(model):
        cls = model.__class__
        if getattr(cls, "_forward_patched", False):
            return

        def forward(
            self,
            input_ids=None,
            attention_mask=None,
            input_features=None,
            feature_attention_mask=None,
            labels=None,
            **kwargs,
        ):
            return self.thinker.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                input_features=input_features,
                feature_attention_mask=feature_attention_mask,
                labels=labels,
                **kwargs,
            )

        cls.forward = forward
        cls._forward_patched = True

    def load_audio(path: str, sr: int = 16000):
        wav, _ = librosa.load(path, sr=sr, mono=True)
        return wav

    def build_prefix_messages(prompt: str, audio_array):
        return [
            {"role": "system", "content": prompt or ""},
            {"role": "user", "content": [{"type": "audio", "audio": audio_array}]},
        ]

    def make_preprocess_fn(processor):
        def _preprocess(ex: dict[str, Any]) -> dict[str, Any]:
            prompt = ex.get("prompt", "")
            prefix_msgs = build_prefix_messages(prompt, None)
            prefix_text = processor.apply_chat_template([prefix_msgs], add_generation_prompt=True, tokenize=False)[0]
            return {
                "prompt": prompt,
                "audio": ex["audio"],
                "target": ex["text"],
                "prefix_text": prefix_text,
            }

        return _preprocess

    @dataclass
    class DataCollatorForQwen3ASR:
        processor: Any
        sampling_rate: int = 16000

        def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
            audio_paths = [f["audio"] for f in features]
            prefix_texts = [f["prefix_text"] for f in features]
            targets = [f["target"] for f in features]

            eos = self.processor.tokenizer.eos_token or ""
            full_texts = [prefix + target + eos for prefix, target in zip(prefix_texts, targets)]
            audios = [load_audio(path, sr=self.sampling_rate) for path in audio_paths]

            full_inputs = self.processor(text=full_texts, audio=audios, return_tensors="pt", padding=True, truncation=False)
            prefix_inputs = self.processor(text=prefix_texts, audio=audios, return_tensors="pt", padding=True, truncation=False)

            prefix_lens = prefix_inputs["attention_mask"].sum(dim=1).tolist()
            labels = full_inputs["input_ids"].clone()
            for index, prefix_len in enumerate(prefix_lens):
                labels[index, :prefix_len] = -100

            pad_id = self.processor.tokenizer.pad_token_id
            if pad_id is not None:
                labels[labels == pad_id] = -100

            full_inputs["labels"] = labels
            return full_inputs

    class CastFloatInputsTrainer(Trainer):
        def _prepare_inputs(self, inputs):
            inputs = super()._prepare_inputs(inputs)
            model_dtype = getattr(self.model, "dtype", None)
            if model_dtype is not None:
                for key, value in list(inputs.items()):
                    if torch.is_tensor(value) and value.is_floating_point():
                        inputs[key] = value.to(dtype=model_dtype)
            return inputs

    def copy_hf_files(src_dir: str, dst_dir: str):
        os.makedirs(dst_dir, exist_ok=True)
        required = [
            "config.json",
            "generation_config.json",
            "preprocessor_config.json",
            "processor_config.json",
            "tokenizer_config.json",
            "tokenizer.json",
            "special_tokens_map.json",
            "chat_template.json",
            "merges.txt",
            "vocab.json",
        ]
        for filename in required:
            src = os.path.join(src_dir, filename)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(dst_dir, filename))

    class InferableCheckpointCallback(TrainerCallback):
        def __init__(self, base_model_path: str):
            self.base_model_path = base_model_path

        def on_save(self, args: TrainingArguments, state, control, **kwargs):
            if args.process_index != 0:
                return control
            ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            if os.path.isdir(ckpt_dir):
                copy_hf_files(self.base_model_path, ckpt_dir)
            return control

    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    resolved_model_path = model_id if Path(model_id).exists() else snapshot_download(model_id)
    asr_wrapper = Qwen3ASRModel.from_pretrained(
        resolved_model_path,
        dtype=torch.bfloat16 if use_bf16 else torch.float16,
        device_map=None,
    )
    model = asr_wrapper.model
    processor = asr_wrapper.processor
    patch_outer_forward(model)
    model.generation_config = GenerationConfig.from_model_config(model.config)

    raw_ds = load_dataset("json", data_files={"train": str(train_jsonl), "validation": str(val_jsonl)})
    ds = raw_ds.map(make_preprocess_fn(processor), num_proc=1)
    keep = {"prompt", "audio", "target", "prefix_text"}
    for split_name in ds.keys():
        drop = [column for column in ds[split_name].column_names if column not in keep]
        if drop:
            ds[split_name] = ds[split_name].remove_columns(drop)

    collator = DataCollatorForQwen3ASR(processor=processor, sampling_rate=16000)
    output_dir.mkdir(parents=True, exist_ok=True)
    training_kwargs = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": 1 if smoke else 2,
        "gradient_accumulation_steps": 1 if smoke else 2,
        "learning_rate": 2e-5,
        "num_train_epochs": 1,
        "logging_steps": 1 if smoke else 10,
        "lr_scheduler_type": "linear",
        "warmup_ratio": 0.02,
        "dataloader_num_workers": 0,
        "dataloader_pin_memory": True,
        "save_strategy": "steps",
        "save_steps": 1 if smoke else 50,
        "save_total_limit": 2,
        "max_steps": 2 if smoke else -1,
        "save_safetensors": True,
        "eval_steps": 1 if smoke else 50,
        "do_eval": True,
        "bf16": use_bf16,
        "fp16": not use_bf16,
        "ddp_find_unused_parameters": False,
        "remove_unused_columns": False,
        "seed": seed,
        "data_seed": seed,
        "report_to": "tensorboard",
        "logging_dir": str(output_dir / "tensorboard"),
    }
    eval_arg_name = "evaluation_strategy"
    if "evaluation_strategy" not in inspect.signature(TrainingArguments.__init__).parameters:
        eval_arg_name = "eval_strategy"
    training_kwargs[eval_arg_name] = "steps"
    training_args = TrainingArguments(**training_kwargs)

    trainer = CastFloatInputsTrainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=collator,
        tokenizer=processor.tokenizer,
        callbacks=[InferableCheckpointCallback(base_model_path=str(resolved_model_path))],
    )
    trainer.train()

    metrics_path = output_dir / "trainer_state.json"
    write_json(metrics_path, {"log_history": trainer.state.log_history, "global_step": trainer.state.global_step})
    checkpoint = find_latest_checkpoint(output_dir)
    if checkpoint is None:
        raise FileNotFoundError(f"No ASR checkpoint found in {output_dir}")
    return checkpoint, metrics_path


def decode_sample(checkpoint_dir: Path, test_jsonl: Path, artifact_dir: Path) -> Path:
    import torch
    from qwen_asr import Qwen3ASRModel

    artifact_dir.mkdir(parents=True, exist_ok=True)
    record = read_jsonl(test_jsonl)[0]
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    device_map = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = Qwen3ASRModel.from_pretrained(str(checkpoint_dir), dtype=dtype, device_map=device_map)
    result = model.transcribe(audio=record["audio"])[0]
    output_path = artifact_dir / "sample_decode.json"
    write_json(
        output_path,
        {
            "audio": record["audio"],
            "reference": record["text"],
            "prediction": result.text if hasattr(result, "text") else str(result),
            "language": getattr(result, "language", None),
            "checkpoint": str(checkpoint_dir),
        },
    )
    return output_path


def main() -> None:
    hf_token = require_env("HF_TOKEN")
    dataset_id = require_env("HF_DATASET_ID", default="JosueG/adja-tts-orpheus", required=False)
    model_id = require_env("MODEL_ID", default="Qwen/Qwen3-ASR-0.6B", required=False)
    output_repo_id = require_env("OUTPUT_REPO_ID")
    results_repo_id = require_env("RESULTS_REPO_ID")
    workspace_dir = Path(require_env("WORKSPACE_DIR", default="/tmp/qwen3_adja_asr", required=False)).resolve()
    seed = int(require_env("SEED", default="42", required=False))
    smoke_only = require_env("SMOKE_RUN", default="0", required=False).lower() in {"1", "true", "yes"}

    logs_dir = workspace_dir / "logs" / "asr"
    artifacts_dir = workspace_dir / "artifacts" / "asr"
    smoke_dir = workspace_dir / "runs" / "asr" / "smoke"
    pilot_dir = workspace_dir / "runs" / "asr" / "pilot"
    log_handle, original_stdout, original_stderr = setup_tee_logging(logs_dir / "job.log")
    try:
        materialization_summary = materialize_asr(dataset_id, hf_token, workspace_dir, seed)
        train_jsonl = workspace_dir / "asr" / "train.jsonl"
        val_jsonl = workspace_dir / "asr" / "val.jsonl"
        test_jsonl = workspace_dir / "asr" / "test.jsonl"
        validate_asr_jsonl(train_jsonl)
        validate_asr_jsonl(val_jsonl)

        resolved_config_path = artifacts_dir / "resolved_config.json"
        write_json(
            resolved_config_path,
            {
                "dataset_id": dataset_id,
                "model_id": model_id,
                "seed": seed,
                "smoke_run_only": smoke_only,
                "workspace_dir": str(workspace_dir),
            },
        )

        smoke_train = subset_jsonl(train_jsonl, smoke_dir / "train_smoke.jsonl", 8)
        smoke_val = subset_jsonl(val_jsonl, smoke_dir / "val_smoke.jsonl", 4)
        smoke_checkpoint, smoke_metrics = run_asr_train(smoke_train, smoke_val, smoke_dir / "output", model_id, seed, smoke=True)
        sample_decode = decode_sample(smoke_checkpoint, test_jsonl, artifacts_dir / "smoke")

        final_checkpoint = smoke_checkpoint
        final_metrics = smoke_metrics
        pilot_status: dict[str, Any] = {"skipped": smoke_only}
        if not smoke_only:
            final_checkpoint, final_metrics = run_asr_train(train_jsonl, val_jsonl, pilot_dir / "output", model_id, seed, smoke=False)
            pilot_sample = decode_sample(final_checkpoint, test_jsonl, artifacts_dir / "pilot")
            pilot_status = {
                "skipped": False,
                "checkpoint": str(final_checkpoint),
                "sample_decode": str(pilot_sample),
                "trainer_state": str(final_metrics),
            }

        summary_path = artifacts_dir / "run_summary.json"
        write_json(
            summary_path,
            {
                "dataset_id": dataset_id,
                "model_id": model_id,
                "seed": seed,
                "smoke_run_only": smoke_only,
                "workspace_dir": str(workspace_dir),
                "materialization_summary": materialization_summary,
                "smoke": {
                    "checkpoint": str(smoke_checkpoint),
                    "sample_decode": str(sample_decode),
                    "trainer_state": str(smoke_metrics),
                },
                "pilot": pilot_status,
                "final_checkpoint": str(final_checkpoint),
            },
        )

        from huggingface_hub import HfApi

        api = HfApi(token=hf_token)
        ensure_repo(api, output_repo_id)
        ensure_repo(api, results_repo_id)
        upload_folder_if_exists(api, output_repo_id, final_checkpoint)
        upload_file_if_exists(api, results_repo_id, resolved_config_path, "qwen3_asr_adja/resolved_config.json")
        upload_file_if_exists(api, results_repo_id, summary_path, "qwen3_asr_adja/run_summary.json")
        upload_file_if_exists(api, results_repo_id, sample_decode, "qwen3_asr_adja/sample_decode.json")
        upload_file_if_exists(api, results_repo_id, final_metrics, "qwen3_asr_adja/trainer_state.json")
        upload_file_if_exists(api, results_repo_id, logs_dir / "job.log", "qwen3_asr_adja/logs/job.log")
        print(summary_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        raise
    finally:
        restore_tee_logging(log_handle, original_stdout, original_stderr)


if __name__ == "__main__":
    main()
