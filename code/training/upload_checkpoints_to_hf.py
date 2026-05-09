"""
upload_checkpoints_to_hf.py — push trained NLLB-600M checkpoints to Hugging Face.

Designed to run ON THE HPC where the checkpoints already live, so you don't have
to scp a 5+ GB tree to your laptop and back up. Can also run locally if you've
fetched the pipeline results.

For each checkpoint dir found under <root>/<direction>/<model>/<exp>/<condition>/seedN/checkpoint/,
creates / updates a Hugging Face model repo at:
    <HF_USERNAME>/adja-nmt-<model>-<direction>-<condition_slug>-seed<seed>

Adds:
  - All files in checkpoint/ (model weights, tokenizer, config)
  - The matching test_metrics.json from the parent dir
  - The matching test_predictions.jsonl from the parent dir (if present)
  - A README.md with model card metadata + benchmark numbers

Auth: needs an HF token. Either:
  - Set HF_TOKEN env var, OR
  - Run `huggingface-cli login` once before running this script

Usage (on HPC):
    cd <HPC_WORKDIR>
    python upload_checkpoints_to_hf.py \\
        --root results/pipeline \\
        --hf-username JosueG \\
        --visibility public

Usage (local, after fetch):
    python experiments/training/upload_checkpoints_to_hf.py \\
        --root experiments/results/pipeline \\
        --hf-username JosueG

Dry-run to see what WOULD be uploaded:
    python experiments/training/upload_checkpoints_to_hf.py \\
        --root experiments/results/pipeline --hf-username JosueG --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def slugify_condition(c: str) -> str:
    """Turn 'RANDOM-10K_STRUCTURED-4K' -> 'r10ks4k' for repo naming."""
    return (c.lower()
              .replace("_", "")
              .replace("random", "r")
              .replace("structured", "s")
              .replace("-", ""))


def find_checkpoints(root: Path) -> list[dict]:
    """Walk root for checkpoint/ dirs. Each result row contains the metadata we
    need to construct a repo id and a README. Layout expected:

        root/<direction>/<model>/<experiment>/<condition>/seed<N>/checkpoint/
                                                                  /test_metrics.json
                                                                  /test_predictions.jsonl
    """
    rows = []
    for ckpt_dir in root.rglob("checkpoint"):
        if not ckpt_dir.is_dir():
            continue
        seed_dir = ckpt_dir.parent
        if not seed_dir.name.startswith("seed"):
            continue
        try:
            condition_dir = seed_dir.parent
            experiment_dir = condition_dir.parent
            model_dir = experiment_dir.parent
            direction_dir = model_dir.parent
        except IndexError:
            continue
        rows.append({
            "ckpt_dir":       ckpt_dir,
            "seed":           int(seed_dir.name.replace("seed", "")),
            "condition":      condition_dir.name,
            "experiment":     experiment_dir.name,
            "model":          model_dir.name,
            "direction":      direction_dir.name,
            "test_metrics":   seed_dir / "test_metrics.json",
            "test_predictions": seed_dir / "test_predictions.jsonl",
        })
    return rows


def build_readme(row: dict, metrics: dict) -> str:
    direction = row["direction"]
    src_lang, tgt_lang = ("French", "Adja") if direction == "forward" else ("Adja", "French")
    src_code, tgt_code = ("fra_Latn", "aj_Latn") if direction == "forward" else ("aj_Latn", "fra_Latn")
    bleu = metrics.get("test_bleu", "?")
    chrf = metrics.get("test_chrf", "?")
    chrfpp = metrics.get("test_chrfpp", "?")

    md = f"""---
license: cc-by-nc-4.0
language:
  - {"fr" if direction == "forward" else "aj"}
  - {"aj" if direction == "forward" else "fr"}
tags:
  - translation
  - low-resource
  - adja
  - gbe
  - nllb
  - french
pipeline_tag: translation
base_model: facebook/nllb-200-distilled-600M
---

# NLLB-200 fine-tuned for {src_lang} → {tgt_lang}

Fine-tuned from [facebook/nllb-200-distilled-600M](https://huggingface.co/facebook/nllb-200-distilled-600M)
on the Adja parallel corpus described in our ACL submission *"Composition over
Quantity: Structured Corpus Design for Extremely Low-Resource Neural Machine
Translation"*.

- **Direction:** {src_lang} ({src_code}) → {tgt_lang} ({tgt_code})
- **Training data:** {row['condition']} ({row['experiment']}, seed {row['seed']})
- **Custom vocabulary:** `aj_Latn` token added and initialized from `ewe_Latn` embeddings.

## Test-set scores (1455 segments: 455 structured + 1000 Tatoeba)

| Metric | Score |
|---|---|
| BLEU | {bleu:.2f} |
| chrF | {chrf:.2f} |
| chrF++ | {chrfpp:.2f} |

## Usage

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_id = "<this repo>"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

tok.src_lang = "{src_code}"
inputs = tok("{'Bonjour' if direction == 'forward' else 'eŋushe'}", return_tensors="pt")
out = model.generate(**inputs,
                     forced_bos_token_id=tok.convert_tokens_to_ids("{tgt_code}"),
                     max_length=128, num_beams=5)
print(tok.batch_decode(out, skip_special_tokens=True)[0])
```

## License & ethics

Adja translations were produced by native-speaker translators with informed
consent. The community governs data sharing — please cite the paper and respect
the privacy/use restrictions noted in the repository.
"""
    return md


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path,
                    help="Root dir containing pipeline results (forward/, reverse/, ...)")
    ap.add_argument("--hf-username", required=True,
                    help="Hugging Face username or org name (e.g. 'JosueG')")
    ap.add_argument("--visibility", choices=["public", "private"], default="private",
                    help="Repo visibility (default: private — flip to public when ready)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Show what would be uploaded without doing it")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip repos that already exist on HF")
    args = ap.parse_args()

    if not args.root.exists():
        print(f"ERROR: --root not found: {args.root}", file=sys.stderr)
        sys.exit(1)

    rows = find_checkpoints(args.root)
    if not rows:
        print(f"No checkpoint/ dirs found under {args.root}", file=sys.stderr)
        print("Hint: training jobs must be run with --save-checkpoint (default ON for "
              "submit_pipeline.sbatch).", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(rows)} checkpoint(s) to upload:\n")
    for r in rows:
        slug = slugify_condition(r["condition"])
        repo_id = f"{args.hf_username}/adja-nmt-{r['model']}-{r['direction']}-{slug}-seed{r['seed']}"
        print(f"  {repo_id}")
        print(f"    direction={r['direction']:<8} cond={r['condition']:<28} model={r['model']}")
        print(f"    ckpt={r['ckpt_dir']}")

    if args.dry_run:
        print("\n--dry-run: not uploading.")
        return

    try:
        from huggingface_hub import HfApi, create_repo, upload_folder
    except ImportError:
        print("ERROR: pip install huggingface_hub", file=sys.stderr)
        sys.exit(1)

    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)

    for r in rows:
        slug = slugify_condition(r["condition"])
        repo_id = f"{args.hf_username}/adja-nmt-{r['model']}-{r['direction']}-{slug}-seed{r['seed']}"
        print(f"\n→ {repo_id}")

        try:
            create_repo(
                repo_id=repo_id,
                repo_type="model",
                private=(args.visibility == "private"),
                exist_ok=True,
                token=token,
            )
        except Exception as e:  # noqa: BLE001
            if args.skip_existing and "already exists" in str(e):
                print(f"   [skip-existing] {e}")
                continue
            print(f"   ERROR creating repo: {e}", file=sys.stderr)
            continue

        # Build & write README into the checkpoint dir
        metrics = {}
        if r["test_metrics"].exists():
            metrics = json.loads(r["test_metrics"].read_text())
        readme_path = r["ckpt_dir"] / "README.md"
        readme_path.write_text(build_readme(r, metrics))
        print(f"   wrote {readme_path.name}")

        # Copy metrics + predictions next to the checkpoint so they're in the upload
        if r["test_metrics"].exists():
            (r["ckpt_dir"] / "test_metrics.json").write_text(r["test_metrics"].read_text())
        if r["test_predictions"].exists():
            (r["ckpt_dir"] / "test_predictions.jsonl").write_text(r["test_predictions"].read_text())

        try:
            upload_folder(
                folder_path=str(r["ckpt_dir"]),
                repo_id=repo_id,
                repo_type="model",
                token=token,
                commit_message=f"Upload {r['direction']} checkpoint ({r['condition']}, seed {r['seed']})",
            )
            print(f"   uploaded → https://huggingface.co/{repo_id}")
        except Exception as e:  # noqa: BLE001
            print(f"   ERROR upload: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
