"""
prep_data.py — write direction-swapped TSVs into the layout train_nllb_hpc.py expects.

Run on the HPC LOGIN NODE (no GPU needed) before sbatch. Reads
`JosueG/adja-fr-mt-acl-paper-private` once, writes:

  {data_root}/bidir-paper-repro/{direction}/train.tsv
  {data_root}/bidir-paper-repro/{direction}/val.tsv
  {data_root}/shared/test_{direction}.tsv

Column order in the TSV is (src, tgt) for the chosen direction. The
training script's loader treats parts[0] as src and parts[1] as tgt, so
direction is fully encoded in the data — no logic change in the trainer.

Note: train_nllb_hpc.py expects test at {data_dir}/shared/test.tsv by
default but accepts --test-path. We write per-direction test TSVs and
pass --test-path explicitly from the sbatch script.

Usage:
    python prep_data.py --data-root <HPC_WORKDIR> \
                        --dataset-repo JosueG/adja-fr-mt-acl-paper-private
"""

import argparse
import os
import sys


def write_split(ds_split, src_col, tgt_col, out_path):
    rows = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in ds_split:
            src = (ex[src_col] or "").replace("\t", " ").replace("\n", " ").strip()
            tgt = (ex[tgt_col] or "").replace("\t", " ").replace("\n", " ").strip()
            if not src or not tgt:
                continue
            f.write(f"{src}\t{tgt}\n")
            rows += 1
    print(f"  {out_path}: {rows} rows")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True,
                    help="HPC data root, e.g. <HPC_WORKDIR>")
    ap.add_argument("--dataset-repo", default="JosueG/adja-fr-mt-acl-paper-private")
    ap.add_argument("--experiment", default="bidir-paper-repro")
    ap.add_argument("--smoke", action="store_true",
                    help="Cap each split to 50 rows for smoke tests")
    args = ap.parse_args()

    from datasets import load_dataset

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("WARNING: HF_TOKEN not set; private dataset will fail.")

    print(f"Loading {args.dataset_repo}...")
    ds = load_dataset(args.dataset_repo, token=token)

    expected = {"train", "validation", "test"}
    missing = expected - set(ds.keys())
    if missing:
        print(f"ERROR: dataset missing splits: {missing}")
        sys.exit(1)

    for s in expected:
        cols = set(ds[s].column_names)
        if not {"fra_Latn", "aj_Latn"}.issubset(cols):
            print(f"ERROR: split {s} missing fra_Latn/aj_Latn cols (got {cols})")
            sys.exit(1)

    if args.smoke:
        ds = {s: ds[s].select(range(min(50, len(ds[s])))) for s in expected}

    shared_dir = os.path.join(args.data_root, "shared")
    os.makedirs(shared_dir, exist_ok=True)

    for direction in ("fr_aj", "aj_fr"):
        src_col, tgt_col = ("fra_Latn", "aj_Latn") if direction == "fr_aj" else ("aj_Latn", "fra_Latn")
        out_dir = os.path.join(args.data_root, args.experiment, direction)
        os.makedirs(out_dir, exist_ok=True)
        print(f"\nDirection {direction}  (col0={src_col}, col1={tgt_col})")
        write_split(ds["train"], src_col, tgt_col, os.path.join(out_dir, "train.tsv"))
        write_split(ds["validation"], src_col, tgt_col, os.path.join(out_dir, "val.tsv"))
        write_split(ds["test"], src_col, tgt_col, os.path.join(shared_dir, f"test_{direction}.tsv"))

    print("\nDone. Layout:")
    print(f"  {args.data_root}/{args.experiment}/{{fr_aj,aj_fr}}/{{train,val}}.tsv")
    print(f"  {shared_dir}/test_{{fr_aj,aj_fr}}.tsv")


if __name__ == "__main__":
    main()
