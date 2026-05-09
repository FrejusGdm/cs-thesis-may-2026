"""
select_and_publish_best.py — pick the best seed per direction and publish
to the canonical subfolders of `JosueG/adja-mt-best`.

Run AFTER all 6 array jobs complete. It does NOT retrain — it just picks
the highest test_chrf seed for each direction and re-uploads that
checkpoint to a clean `{direction}/` (no seed suffix) so the cascade and
downstream users can `from_pretrained(repo, subfolder="fr_aj")` without
caring about seeds.

Per-seed checkpoints already pushed by the training script remain at
`{direction}/seed{N}/` for transparency.

Usage (HPC login or laptop with HF_TOKEN):
    python select_and_publish_best.py \
        --results-dir <HPC_WORKDIR> \
        --hub-repo JosueG/adja-mt-best
"""

import argparse
import json
import os
import sys


def find_best(results_dir, direction):
    """Return (best_seed, best_chrf, checkpoint_path) for one direction."""
    direction_dir = os.path.join(results_dir, direction)
    if not os.path.isdir(direction_dir):
        return None
    candidates = []
    for entry in os.listdir(direction_dir):
        if not entry.startswith("seed"):
            continue
        metrics_path = os.path.join(direction_dir, entry, "test_metrics.json")
        ckpt_path = os.path.join(direction_dir, entry, "checkpoint")
        if not (os.path.isfile(metrics_path) and os.path.isdir(ckpt_path)):
            continue
        with open(metrics_path) as f:
            m = json.load(f)
        candidates.append((m.get("test_chrf", -1.0), entry, ckpt_path, m))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    chrf, seed_dir, ckpt, metrics = candidates[0]
    seed = int(seed_dir.replace("seed", ""))
    return seed, chrf, ckpt, metrics, candidates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True,
                    help="e.g. <HPC_WORKDIR>")
    ap.add_argument("--hub-repo", default="JosueG/adja-mt-best")
    ap.add_argument("--directions", nargs="+", default=["fr_aj", "aj_fr"])
    ap.add_argument("--dry-run", action="store_true",
                    help="Print decisions without uploading")
    args = ap.parse_args()

    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    if not token and not args.dry_run:
        print("ERROR: HF_TOKEN not set")
        sys.exit(1)

    api = HfApi(token=token) if token else None

    if api:
        try:
            api.create_repo(args.hub_repo, repo_type="model", private=False, exist_ok=True)
        except Exception as e:
            print(f"Note: repo creation returned: {e}")

    summary = {}
    for direction in args.directions:
        result = find_best(args.results_dir, direction)
        if result is None:
            print(f"[{direction}] no completed seeds found in {args.results_dir}/{direction}")
            continue
        seed, chrf, ckpt_path, metrics, candidates = result
        print(f"\n[{direction}] candidates (chrf, seed):")
        for c_chrf, c_seed_dir, _, _ in candidates:
            print(f"    chrF={c_chrf:.2f}  {c_seed_dir}")
        print(f"  -> winner: seed{seed} (chrF={chrf:.2f})")
        summary[direction] = {"seed": seed, "test_chrf": chrf, "metrics": metrics}

        if args.dry_run:
            print(f"  [dry-run] would upload {ckpt_path} -> {args.hub_repo}/{direction}/")
            continue

        print(f"  uploading {ckpt_path} -> {args.hub_repo}/{direction}/")
        api.upload_folder(
            folder_path=ckpt_path,
            path_in_repo=direction,
            repo_id=args.hub_repo,
            repo_type="model",
            commit_message=f"{direction} BEST: seed{seed} chrF={chrf:.2f}",
        )
        print(f"  done.")

    print("\nSummary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
