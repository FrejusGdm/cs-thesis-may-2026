#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#     "torch==2.5.1",
#     "torchaudio==2.5.1",
#     "qwen-tts>=0.1.0",
#     "transformers>=4.40.0",
#     "datasets>=2.18.0",
#     "accelerate>=0.28.0",
#     "safetensors>=0.4.0",
#     "librosa>=0.10.0",
#     "soundfile>=0.12.0",
#     "numpy>=1.24.0",
#     "huggingface-hub>=0.34.0",
# ]
# ///
from __future__ import annotations

"""
Qwen3-TTS Adja fine-tuning launcher for Hugging Face Jobs.

Prepared for: Josue Godeme
Requested by: Josue Godeme

This launcher is intentionally standalone so it can run under:
    hf jobs uv run ... scripts/hf_jobs/qwen3_tts_adja.py

It does not assume the rest of the repo is present inside the container.
The vendored/local Qwen3 code remains in the repo as source material, but this
HF launcher inlines the minimum required logic to match the existing project
pattern for self-contained job scripts.
"""

import json
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


def materialize_tts(dataset_id: str, token: str, workspace_dir: Path, seed: int) -> dict[str, Any]:
    import librosa
    import numpy as np
    import soundfile as sf
    from datasets import load_dataset

    dataset = load_dataset(dataset_id, token=token, split="train")
    splits = create_splits(dataset, seed)
    base_dir = workspace_dir / "tts"
    summary: dict[str, Any] = {
        "dataset_id": dataset_id,
        "seed": seed,
        "mode": "tts",
        "splits": {"tts": {}},
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
            if sr != 24000:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=24000)
                sr = 24000

            wav_path = wav_dir / f"{split_name}_{index:05d}.wav"
            sf.write(wav_path, audio, sr)
            duration = len(audio) / float(sr)
            total_duration += duration
            records.append({"audio": str(wav_path), "text": normalize_text(sample["text"])})

        jsonl_path = base_dir / f"{split_name}_raw.jsonl"
        write_jsonl(jsonl_path, records)
        summary["splits"]["tts"][split_name] = {
            "jsonl": str(jsonl_path),
            "num_records": len(records),
            "total_duration_sec": round(total_duration, 3),
            "target_sample_rate": 24000,
        }

    write_json(workspace_dir / "materialization_summary.json", summary)
    return summary


def choose_reference_audio(train_raw_jsonl: Path, reference_path: Path) -> Path:
    import soundfile as sf

    records = read_jsonl(train_raw_jsonl)
    selected = records[0]["audio"]
    for record in records:
        duration = sf.info(record["audio"]).duration
        if 3.0 <= duration <= 8.0:
            selected = record["audio"]
            break
    reference_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(selected, reference_path)
    return reference_path


def attach_reference_audio(source_jsonl: Path, destination_jsonl: Path, reference_audio: Path) -> Path:
    updated_records = []
    for record in read_jsonl(source_jsonl):
        updated_records.append(
            {
                "audio": record["audio"],
                "text": record["text"],
                "ref_audio": str(reference_audio),
            }
        )
    write_jsonl(destination_jsonl, updated_records)
    return destination_jsonl


def validate_tts_jsonl(jsonl_path: Path) -> None:
    import soundfile as sf

    records = read_jsonl(jsonl_path)
    if not records:
        raise SystemExit(f"Empty TTS JSONL: {jsonl_path}")
    for record in records:
        audio_path = Path(record["audio"])
        ref_audio_path = Path(record["ref_audio"])
        if not audio_path.exists():
            raise FileNotFoundError(f"Missing audio file: {audio_path}")
        if not ref_audio_path.exists():
            raise FileNotFoundError(f"Missing reference audio file: {ref_audio_path}")
        info = sf.info(str(audio_path))
        ref_info = sf.info(str(ref_audio_path))
        if info.samplerate != 24000 or ref_info.samplerate != 24000:
            raise ValueError(f"Unexpected sample rate in {jsonl_path}")


def prepare_tts_codes(input_jsonl: Path, output_jsonl: Path) -> Path:
    import torch
    from qwen_tts import Qwen3TTSTokenizer

    tokenizer = Qwen3TTSTokenizer.from_pretrained(
        "Qwen/Qwen3-TTS-Tokenizer-12Hz",
        device_map="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    records = read_jsonl(input_jsonl)
    final_records = []
    batch_lines: list[dict[str, Any]] = []
    batch_audios: list[str] = []
    batch_size = 32

    for record in records:
        batch_lines.append(dict(record))
        batch_audios.append(record["audio"])
        if len(batch_lines) >= batch_size:
            enc_res = tokenizer.encode(batch_audios)
            for code, line in zip(enc_res.audio_codes, batch_lines):
                line["audio_codes"] = code.cpu().tolist()
                final_records.append(line)
            batch_lines.clear()
            batch_audios.clear()

    if batch_lines:
        enc_res = tokenizer.encode(batch_audios)
        for code, line in zip(enc_res.audio_codes, batch_lines):
            line["audio_codes"] = code.cpu().tolist()
            final_records.append(line)

    write_jsonl(output_jsonl, final_records)
    return output_jsonl


def find_latest_checkpoint(output_dir: Path) -> Path | None:
    pattern = re.compile(r"^checkpoint-epoch-(\d+)$")
    best: tuple[int, Path] | None = None
    if not output_dir.exists():
        return None
    for path in output_dir.iterdir():
        match = pattern.match(path.name)
        if not match or not path.is_dir():
            continue
        epoch = int(match.group(1))
        if best is None or epoch > best[0]:
            best = (epoch, path)
    return best[1] if best else None


def run_tts_train(train_jsonl: Path, output_dir: Path, model_id: str, speaker_name: str, smoke: bool) -> tuple[Path, Path]:
    import librosa
    import numpy as np
    import torch
    from accelerate import Accelerator
    from huggingface_hub import snapshot_download
    from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram
    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
    from safetensors.torch import save_file
    from torch.optim import AdamW
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoConfig

    class TTSDataset(Dataset):
        def __init__(self, data_list, processor, config, lag_num: int = -1):
            self.data_list = data_list
            self.processor = processor
            self.lag_num = lag_num
            self.config = config

        def __len__(self):
            return len(self.data_list)

        def _load_audio_to_np(self, path: str):
            audio, sr = librosa.load(path, sr=None, mono=True)
            if audio.ndim > 1:
                audio = np.mean(audio, axis=-1)
            return audio.astype(np.float32), int(sr)

        def _build_assistant_text(self, text: str) -> str:
            return f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"

        def _tokenize_texts(self, text):
            inputs = self.processor(text=text, return_tensors="pt", padding=True)
            input_ids = inputs["input_ids"]
            input_ids = input_ids.unsqueeze(0) if input_ids.dim() == 1 else input_ids
            return input_ids

        @torch.inference_mode()
        def extract_mels(self, audio, sr):
            assert sr == 24000, "Only support 24kHz audio"
            mels = mel_spectrogram(
                torch.from_numpy(audio).unsqueeze(0),
                n_fft=1024,
                num_mels=128,
                sampling_rate=24000,
                hop_size=256,
                win_size=1024,
                fmin=0,
                fmax=12000,
            ).transpose(1, 2)
            return mels

        def __getitem__(self, idx):
            item = self.data_list[idx]
            text_ids = self._tokenize_texts(self._build_assistant_text(item["text"]))
            audio_codes = torch.tensor(item["audio_codes"], dtype=torch.long)
            wav, sr = self._load_audio_to_np(item["ref_audio"])
            ref_mel = self.extract_mels(audio=wav, sr=sr)
            return {"text_ids": text_ids[:, :-5], "audio_codes": audio_codes, "ref_mel": ref_mel}

        def collate_fn(self, batch):
            item_length = [b["text_ids"].shape[1] + b["audio_codes"].shape[0] for b in batch]
            max_length = max(item_length) + 8
            batch_size = len(batch)

            input_ids = torch.zeros((batch_size, max_length, 2), dtype=torch.long)
            codec_ids = torch.zeros((batch_size, max_length, 16), dtype=torch.long)
            text_embedding_mask = torch.zeros((batch_size, max_length), dtype=torch.bool)
            codec_embedding_mask = torch.zeros((batch_size, max_length), dtype=torch.bool)
            codec_mask = torch.zeros((batch_size, max_length), dtype=torch.bool)
            attention_mask = torch.zeros((batch_size, max_length), dtype=torch.long)
            codec_0_labels = torch.full((batch_size, max_length), -100, dtype=torch.long)

            for i, data in enumerate(batch):
                text_ids = data["text_ids"]
                audio_codec_0 = data["audio_codes"][:, 0]
                audio_codecs = data["audio_codes"]

                text_ids_len = text_ids.shape[1]
                codec_ids_len = audio_codec_0.shape[0]

                input_ids[i, :3, 0] = text_ids[0, :3]
                input_ids[i, 3:7, 0] = self.config.tts_pad_token_id
                input_ids[i, 7, 0] = self.config.tts_bos_token_id
                input_ids[i, 8 : 8 + text_ids_len - 3, 0] = text_ids[0, 3:]
                input_ids[i, 8 + text_ids_len - 3, 0] = self.config.tts_eos_token_id
                input_ids[i, 8 + text_ids_len - 2 : 8 + text_ids_len + codec_ids_len, 0] = self.config.tts_pad_token_id
                text_embedding_mask[i, : 8 + text_ids_len + codec_ids_len] = True

                input_ids[i, 3:8, 1] = torch.tensor(
                    [
                        self.config.talker_config.codec_nothink_id,
                        self.config.talker_config.codec_think_bos_id,
                        self.config.talker_config.codec_think_eos_id,
                        0,
                        self.config.talker_config.codec_pad_id,
                    ]
                )
                input_ids[i, 8 : 8 + text_ids_len - 3, 1] = self.config.talker_config.codec_pad_id
                input_ids[i, 8 + text_ids_len - 3, 1] = self.config.talker_config.codec_pad_id
                input_ids[i, 8 + text_ids_len - 2, 1] = self.config.talker_config.codec_bos_id
                input_ids[i, 8 + text_ids_len - 1 : 8 + text_ids_len - 1 + codec_ids_len, 1] = audio_codec_0
                input_ids[i, 8 + text_ids_len - 1 + codec_ids_len, 1] = self.config.talker_config.codec_eos_token_id

                codec_0_labels[i, 8 + text_ids_len - 1 : 8 + text_ids_len - 1 + codec_ids_len] = audio_codec_0
                codec_0_labels[i, 8 + text_ids_len - 1 + codec_ids_len] = self.config.talker_config.codec_eos_token_id
                codec_ids[i, 8 + text_ids_len - 1 : 8 + text_ids_len - 1 + codec_ids_len, :] = audio_codecs

                codec_embedding_mask[i, 3 : 8 + text_ids_len + codec_ids_len] = True
                codec_embedding_mask[i, 6] = False
                codec_mask[i, 8 + text_ids_len - 1 : 8 + text_ids_len - 1 + codec_ids_len] = True
                attention_mask[i, : 8 + text_ids_len + codec_ids_len] = True

            ref_mels = torch.cat([data["ref_mel"] for data in batch], dim=0)
            return {
                "input_ids": input_ids,
                "ref_mels": ref_mels,
                "attention_mask": attention_mask,
                "text_embedding_mask": text_embedding_mask.unsqueeze(-1),
                "codec_embedding_mask": codec_embedding_mask.unsqueeze(-1),
                "codec_0_labels": codec_0_labels,
                "codec_ids": codec_ids,
                "codec_mask": codec_mask,
            }

    attempts = ["flash_attention_2", "eager"]
    if not smoke:
        attempts = ["flash_attention_2", "eager"]
    num_epochs = 1
    batch_size = 1 if smoke else 2
    max_train_steps = 2 if smoke else None
    output_dir.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    resolved_model_path = model_id if Path(model_id).exists() else snapshot_download(model_id)

    for attn_implementation in attempts:
        try:
            mixed_precision = "bf16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "no"
            accelerator = Accelerator(gradient_accumulation_steps=4, mixed_precision=mixed_precision)
            qwen3tts = Qwen3TTSModel.from_pretrained(
                resolved_model_path,
                torch_dtype=torch.bfloat16 if mixed_precision == "bf16" else torch.float32,
                attn_implementation=attn_implementation,
            )
            config = AutoConfig.from_pretrained(resolved_model_path)
            text_embedding_dim = int(qwen3tts.model.talker.model.text_embedding.weight.shape[-1])
            codec_embedding_dim = int(qwen3tts.model.talker.model.codec_embedding.weight.shape[-1])
            print(
                "TTS dims:",
                json.dumps(
                    {
                        "attn_implementation": attn_implementation,
                        "text_embedding_dim": text_embedding_dim,
                        "codec_embedding_dim": codec_embedding_dim,
                        "speaker_encoder_dim": getattr(getattr(config, "speaker_encoder", None), "enc_dim", None),
                        "talker_hidden_size": getattr(getattr(config, "talker_config", None), "hidden_size", None),
                    },
                    ensure_ascii=False,
                ),
            )
            train_data = read_jsonl(train_jsonl)
            dataset = TTSDataset(train_data, qwen3tts.processor, config)
            train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=dataset.collate_fn)
            optimizer = AdamW(qwen3tts.model.parameters(), lr=2e-5, weight_decay=0.01)
            model, optimizer, train_dataloader = accelerator.prepare(qwen3tts.model, optimizer, train_dataloader)
            model.train()

            global_step = 0
            last_loss = None
            target_speaker_embedding = None
            stop_training = False

            for epoch in range(num_epochs):
                for step, batch in enumerate(train_dataloader):
                    with accelerator.accumulate(model):
                        input_ids = batch["input_ids"]
                        codec_ids = batch["codec_ids"]
                        ref_mels = batch["ref_mels"]
                        text_embedding_mask = batch["text_embedding_mask"]
                        codec_embedding_mask = batch["codec_embedding_mask"]
                        attention_mask = batch["attention_mask"]
                        codec_0_labels = batch["codec_0_labels"]
                        codec_mask = batch["codec_mask"]

                        speaker_embedding = model.speaker_encoder(ref_mels.to(model.device).to(model.dtype)).detach()
                        if target_speaker_embedding is None:
                            target_speaker_embedding = speaker_embedding

                        input_text_ids = input_ids[:, :, 0]
                        input_codec_ids = input_ids[:, :, 1]
                        input_text_embedding = model.talker.model.text_embedding(input_text_ids) * text_embedding_mask
                        input_codec_embedding = model.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask
                        input_codec_embedding[:, 6, :] = speaker_embedding
                        input_embeddings = input_text_embedding + input_codec_embedding

                        for index in range(1, 16):
                            codec_i_embedding = model.talker.code_predictor.get_input_embeddings()[index - 1](codec_ids[:, :, index])
                            codec_i_embedding = codec_i_embedding * codec_mask.unsqueeze(-1)
                            input_embeddings = input_embeddings + codec_i_embedding

                        outputs = model.talker(
                            inputs_embeds=input_embeddings[:, :-1, :],
                            attention_mask=attention_mask[:, :-1],
                            labels=codec_0_labels[:, 1:],
                            output_hidden_states=True,
                        )

                        hidden_states = outputs.hidden_states[0][-1]
                        talker_hidden_states = hidden_states[codec_mask[:, :-1]]
                        talker_codec_ids = codec_ids[codec_mask]
                        _, sub_talker_loss = model.talker.forward_sub_talker_finetune(talker_codec_ids, talker_hidden_states)
                        loss = outputs.loss + 0.3 * sub_talker_loss
                        last_loss = float(loss.item())

                        accelerator.backward(loss)
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        optimizer.zero_grad()

                    if accelerator.sync_gradients:
                        global_step += 1
                    if step % 10 == 0:
                        accelerator.print(f"attn={attn_implementation} epoch={epoch} step={step} loss={loss.item():.4f}")
                    if max_train_steps is not None and global_step >= max_train_steps:
                        stop_training = True
                        break

                if accelerator.is_main_process:
                    checkpoint_dir = output_dir / f"checkpoint-epoch-{epoch}"
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copytree(resolved_model_path, checkpoint_dir, dirs_exist_ok=True)

                    config_path = checkpoint_dir / "config.json"
                    config_dict = json.loads(Path(resolved_model_path).joinpath("config.json").read_text(encoding="utf-8")) if Path(resolved_model_path).is_dir() else None
                    if config_dict is None:
                        config_dict = AutoConfig.from_pretrained(resolved_model_path).to_dict()
                    config_dict["tts_model_type"] = "custom_voice"
                    talker_config = config_dict.get("talker_config", {})
                    talker_config["spk_id"] = {speaker_name: 3000}
                    talker_config["spk_is_dialect"] = {speaker_name: False}
                    config_dict["talker_config"] = talker_config
                    config_path.write_text(json.dumps(config_dict, indent=2, ensure_ascii=False), encoding="utf-8")

                    unwrapped_model = accelerator.unwrap_model(model)
                    state_dict = {key: value.detach().to("cpu") for key, value in unwrapped_model.state_dict().items()}
                    keys_to_drop = [key for key in state_dict if key.startswith("speaker_encoder")]
                    for key in keys_to_drop:
                        del state_dict[key]

                    weight = state_dict["talker.model.codec_embedding.weight"]
                    state_dict["talker.model.codec_embedding.weight"][3000] = target_speaker_embedding[0].detach().to(weight.device).to(weight.dtype)
                    save_file(state_dict, str(checkpoint_dir / "model.safetensors"))

                if stop_training:
                    break

            metrics_path = output_dir / "trainer_state.json"
            write_json(
                metrics_path,
                {
                    "attn_implementation": attn_implementation,
                    "global_step": global_step,
                    "last_loss": last_loss,
                    "num_epochs": num_epochs,
                    "batch_size": batch_size,
                    "smoke": smoke,
                },
            )
            checkpoint = find_latest_checkpoint(output_dir)
            if checkpoint is None:
                raise FileNotFoundError(f"No TTS checkpoint found in {output_dir}")
            return checkpoint, metrics_path
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            traceback.print_exc()
            print(f"TTS training failed with attn_implementation={attn_implementation}: {exc}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if last_error is not None:
        raise last_error
    raise RuntimeError("TTS training failed without a captured exception")


def generate_sample(checkpoint_dir: Path, test_jsonl: Path, output_wav: Path, speaker_name: str) -> Path:
    import numpy as np
    import torch
    import soundfile as sf
    from qwen_tts import Qwen3TTSModel

    def to_audio_array(value):
        # Qwen returns batch-oriented outputs; unwrap the first sample exactly as
        # shown in the official model cards before writing to disk.
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        if isinstance(value, (list, tuple)):
            if not value:
                raise ValueError("Generated audio list is empty")
            value = value[0]
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
        value = np.asarray(value)
        if value.ndim > 1:
            value = value[0]
        if value.ndim != 1:
            raise ValueError(f"Unexpected generated audio shape: {value.shape!r}")
        return value.astype(np.float32, copy=False)

    record = read_jsonl(test_jsonl)[0]
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    model = Qwen3TTSModel.from_pretrained(
        str(checkpoint_dir),
        device_map="cuda:0" if torch.cuda.is_available() else "cpu",
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        attn_implementation="eager",
    )
    wavs, sample_rate = model.generate_custom_voice(text=record["text"], speaker=speaker_name, language="Auto")
    wav_array = to_audio_array(wavs)
    sample_rate = int(sample_rate)
    sf.write(str(output_wav), wav_array, sample_rate, format="WAV")
    metadata_path = output_wav.with_suffix(".json")
    write_json(
        metadata_path,
        {
            "text": record["text"],
            "speaker_name": speaker_name,
            "checkpoint": str(checkpoint_dir),
            "output_wav": str(output_wav),
            "sample_rate": sample_rate,
            "num_samples": int(wav_array.shape[0]),
        },
    )
    return metadata_path


def main() -> None:
    hf_token = require_env("HF_TOKEN")
    dataset_id = require_env("HF_DATASET_ID", default="JosueG/adja-tts-orpheus", required=False)
    model_id = require_env("MODEL_ID", default="Qwen/Qwen3-TTS-12Hz-0.6B-Base", required=False)
    output_repo_id = require_env("OUTPUT_REPO_ID")
    results_repo_id = require_env("RESULTS_REPO_ID")
    workspace_dir = Path(require_env("WORKSPACE_DIR", default="/tmp/qwen3_adja_tts", required=False)).resolve()
    seed = int(require_env("SEED", default="42", required=False))
    smoke_only = require_env("SMOKE_RUN", default="0", required=False).lower() in {"1", "true", "yes"}
    speaker_name = "adja_speaker"

    logs_dir = workspace_dir / "logs" / "tts"
    artifacts_dir = workspace_dir / "artifacts" / "tts"
    smoke_dir = workspace_dir / "runs" / "tts" / "smoke"
    pilot_dir = workspace_dir / "runs" / "tts" / "pilot"
    log_handle, original_stdout, original_stderr = setup_tee_logging(logs_dir / "job.log")
    try:
        materialization_summary = materialize_tts(dataset_id, hf_token, workspace_dir, seed)
        train_raw = workspace_dir / "tts" / "train_raw.jsonl"
        val_raw = workspace_dir / "tts" / "val_raw.jsonl"
        test_raw = workspace_dir / "tts" / "test_raw.jsonl"
        reference_audio = choose_reference_audio(train_raw, workspace_dir / "tts" / "reference.wav")
        train_with_ref = attach_reference_audio(train_raw, workspace_dir / "tts" / "train_with_ref.jsonl", reference_audio)
        val_with_ref = attach_reference_audio(val_raw, workspace_dir / "tts" / "val_with_ref.jsonl", reference_audio)
        test_with_ref = attach_reference_audio(test_raw, workspace_dir / "tts" / "test_with_ref.jsonl", reference_audio)
        validate_tts_jsonl(train_with_ref)
        validate_tts_jsonl(val_with_ref)
        validate_tts_jsonl(test_with_ref)

        resolved_config_path = artifacts_dir / "resolved_config.json"
        write_json(
            resolved_config_path,
            {
                "dataset_id": dataset_id,
                "model_id": model_id,
                "seed": seed,
                "smoke_run_only": smoke_only,
                "workspace_dir": str(workspace_dir),
                "speaker_name": speaker_name,
            },
        )

        smoke_train_ref = subset_jsonl(train_with_ref, smoke_dir / "train_smoke_with_ref.jsonl", 8)
        smoke_train_codes = prepare_tts_codes(smoke_train_ref, smoke_dir / "train_smoke_with_codes.jsonl")
        smoke_checkpoint, smoke_metrics = run_tts_train(smoke_train_codes, smoke_dir / "output", model_id, speaker_name, smoke=True)
        smoke_wav = artifacts_dir / "smoke" / "sample.wav"
        smoke_metadata = generate_sample(smoke_checkpoint, test_raw, smoke_wav, speaker_name)

        final_checkpoint = smoke_checkpoint
        final_metrics = smoke_metrics
        pilot_status: dict[str, Any] = {"skipped": smoke_only}
        if not smoke_only:
            pilot_train_codes = prepare_tts_codes(train_with_ref, pilot_dir / "train_with_codes.jsonl")
            final_checkpoint, final_metrics = run_tts_train(pilot_train_codes, pilot_dir / "output", model_id, speaker_name, smoke=False)
            pilot_wav = artifacts_dir / "pilot" / "sample.wav"
            pilot_metadata = generate_sample(final_checkpoint, test_raw, pilot_wav, speaker_name)
            pilot_status = {
                "skipped": False,
                "checkpoint": str(final_checkpoint),
                "sample_metadata": str(pilot_metadata),
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
                "reference_audio": str(reference_audio),
                "materialization_summary": materialization_summary,
                "smoke": {
                    "checkpoint": str(smoke_checkpoint),
                    "sample_wav": str(smoke_wav),
                    "sample_metadata": str(smoke_metadata),
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
        upload_file_if_exists(api, results_repo_id, resolved_config_path, "qwen3_tts_adja/resolved_config.json")
        upload_file_if_exists(api, results_repo_id, summary_path, "qwen3_tts_adja/run_summary.json")
        upload_file_if_exists(api, results_repo_id, smoke_wav, "qwen3_tts_adja/smoke/sample.wav")
        upload_file_if_exists(api, results_repo_id, smoke_metadata, "qwen3_tts_adja/smoke/sample.json")
        upload_file_if_exists(api, results_repo_id, final_metrics, "qwen3_tts_adja/trainer_state.json")
        upload_file_if_exists(api, results_repo_id, logs_dir / "job.log", "qwen3_tts_adja/logs/job.log")
        print(summary_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        raise
    finally:
        restore_tee_logging(log_handle, original_stdout, original_stderr)


if __name__ == "__main__":
    main()
