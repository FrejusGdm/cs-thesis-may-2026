#!/usr/bin/env python3
"""
Spin Hugging Face Inference Endpoints up/down for the Adja ASR + TTS auto-eval.

The HF *Serverless* Inference API does not auto-deploy custom user models
(verified 2026-04-28: returned "Model not supported by provider hf-inference"
on the freshly published `JosueG/whisper-ewe-adja-e4v4`). Dedicated
Inference Endpoints work for any public model repo and bill per uptime hour.

Hardware / cost
---------------
We default to the smallest CPU tier with scale-to-zero on idle:
  vendor=aws, region=us-east-1, accelerator=cpu, instance_type=intel-icl,
  instance_size=x2 (~ $0.06 / hr), scale_to_zero_timeout=300s.

For Whisper-small / Wav2Vec2-XLS-R 300M and ~40 audio files of <10s each,
that is plenty. After 5 minutes of idleness the replica spins down to zero
so you stop paying — wakeup is ~30-60 s on next request.

Usage
-----
    HF_TOKEN=$HF_TOKEN python scripts/hub/manage_endpoints.py up \\
        JosueG/whisper-ewe-adja-e4v4 \\
        JosueG/wav2vec2-xlsr-adja-c4v2

    python scripts/hub/manage_endpoints.py list
    python scripts/hub/manage_endpoints.py down adja-asr-e4v4
    python scripts/hub/manage_endpoints.py wait adja-asr-e4v4

The `up` command prints a YAML-style block of `<label>=<url>` pairs ready
to paste into `experiments/tts/eval/reverse_wer.py --adja-asrs ...`.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Iterable

sys.stdout.reconfigure(line_buffering=True)


DEFAULTS = {
    "vendor": "aws",
    "region": "us-east-1",
    "accelerator": "cpu",
    "instance_type": "intel-spr",   # Intel Sapphire Rapids (cheapest CPU tier)
    "instance_size": "x2",          # 2 vCPU / ~4 GB — $0.067/hr
    "framework": "pytorch",
    "scale_to_zero_timeout": 300,
}


def repo_to_endpoint_name(repo: str) -> str:
    """Map a model repo id to a short, valid endpoint name.

    HF endpoint names: lowercase, alphanumeric and hyphen, ≤32 chars. We pick
    the trailing path component, strip 'adja-' / 'whisper-' boilerplate, and
    prefix with 'aja-' to keep the namespace tidy.
    """
    base = repo.split("/")[-1].lower()
    for prefix in ("adja-", "whisper-", "wav2vec2-"):
        if base.startswith(prefix):
            base = base[len(prefix):]
    name = "aja-" + base
    return name[:32]


GPU_DEFAULTS = {
    "vendor": "aws",
    "region": "us-east-1",
    "accelerator": "gpu",
    "instance_type": "nvidia-l4",   # smallest current GPU tier on AWS
    "instance_size": "x1",          # 1× L4, ~24 GB VRAM, around $0.80/hr
    "framework": "pytorch",
    "scale_to_zero_timeout": 300,
}


def cmd_up(repos: list[str], task: str, gpu: bool) -> None:
    from huggingface_hub import create_inference_endpoint, get_inference_endpoint

    cfg = GPU_DEFAULTS if gpu else DEFAULTS

    out_pairs: list[str] = []
    for repo in repos:
        name = repo_to_endpoint_name(repo)
        print(f"\n[create] {name}  <-  {repo}  (task={task}, accelerator={cfg['accelerator']})")
        try:
            ep = create_inference_endpoint(
                name=name,
                repository=repo,
                framework=cfg["framework"],
                accelerator=cfg["accelerator"],
                instance_size=cfg["instance_size"],
                instance_type=cfg["instance_type"],
                region=cfg["region"],
                vendor=cfg["vendor"],
                task=task,
                type="public",  # no token needed on the endpoint side
                # scale-to-zero requires min_replica=0; max_replica=1 keeps cost low
                min_replica=0,
                max_replica=1,
                scale_to_zero_timeout=cfg["scale_to_zero_timeout"],
            )
            print(f"  created: {ep.name} | {ep.url}")
        except Exception as e:  # already exists, etc.
            print(f"  create failed ({e}); fetching existing endpoint")
            ep = get_inference_endpoint(name)

        # Block until it's running
        print(f"  waiting for status=running …")
        ep = ep.wait(timeout=900)
        print(f"  status: {ep.status} | url: {ep.url}")
        # label is a slug derived from the model name's last token
        label = repo.split("/")[-1].rsplit("-", 1)[-1]
        out_pairs.append(f"{ep.url}:{label}")

    if task == "automatic-speech-recognition":
        print("\n--- Paste into reverse_wer.py ---")
        print(f"--adja-asrs '{','.join(out_pairs)}'")
        print("---------------------------------")
    else:
        print("\n--- Endpoint URLs ---")
        for p in out_pairs:
            print(f"  {p}")
        print("---------------------")


def cmd_down(names: list[str]) -> None:
    from huggingface_hub import get_inference_endpoint

    for name in names:
        ep = get_inference_endpoint(name)
        print(f"[down] {name}")
        ep.delete()
        print(f"  deleted")


def cmd_list() -> None:
    from huggingface_hub import list_inference_endpoints

    eps = list_inference_endpoints()
    if not eps:
        print("(no endpoints)")
        return
    for ep in eps:
        url = getattr(ep, "url", "") or ""
        print(f"  {ep.name:30}  {ep.status:12}  {ep.repository:50}  {url}")


def cmd_wait(name: str) -> None:
    from huggingface_hub import get_inference_endpoint
    ep = get_inference_endpoint(name)
    print(f"Waiting for {name} (current: {ep.status})…")
    ep = ep.wait(timeout=600)
    print(f"  -> {ep.status} | {ep.url}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    up = sub.add_parser("up", help="Create + wait for endpoints")
    up.add_argument("repos", nargs="+", help="Model repo ids (e.g. JosueG/whisper-ewe-adja-e4v4)")
    up.add_argument("--task", default="automatic-speech-recognition",
                    help="Endpoint task tag. Use 'text-to-speech' for Spark "
                         "or 'custom' for a handler.py-driven model. "
                         "Default: automatic-speech-recognition.")
    up.add_argument("--gpu", action="store_true",
                    help="Use the GPU defaults (nvidia-l4 x1, ~$0.80/hr). "
                         "Required for Spark TTS; the CPU default is "
                         "too slow to be usable for synthesis.")

    down = sub.add_parser("down", help="Delete endpoints")
    down.add_argument("names", nargs="+", help="Endpoint names (use `list` to find them)")

    sub.add_parser("list", help="Show all endpoints in the account")

    wait = sub.add_parser("wait", help="Block until endpoint is running")
    wait.add_argument("name")

    args = p.parse_args()

    if not os.environ.get("HF_TOKEN"):
        # huggingface_hub also reads cached token from ~/.cache/huggingface/token
        print("(HF_TOKEN not in env; using cached login)")

    if args.cmd == "up":
        cmd_up(args.repos, task=args.task, gpu=args.gpu)
    elif args.cmd == "down":
        cmd_down(args.names)
    elif args.cmd == "list":
        cmd_list()
    elif args.cmd == "wait":
        cmd_wait(args.name)


if __name__ == "__main__":
    main()
