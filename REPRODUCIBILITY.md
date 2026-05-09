# Reproducibility

This release is designed to make thesis claims traceable without publishing the
private working repo.

Recommended workflow:

1. Install dependencies from `requirements.txt`.
2. Read `data/huggingface-datasets.md` for dataset locations and access policy.
3. Read `models/README.md` for model IDs and task grouping.
4. Use `code/preprocessing`, `code/training`, and `code/evaluation` for the
   main pipeline.
5. Use `results/` for thesis-facing tables and consolidated result manifests.

The release build manifest is in `data/manifests/release_file_manifest.json`.
