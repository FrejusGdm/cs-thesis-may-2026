# Release Safety Checklist

Before publishing:

- Run the safety scan against the generated release repo.
- Confirm no `.env`, tokens, private local paths, or raw private dumps are present.
- Confirm existing Hugging Face repos were only read, never mutated.
- Confirm dataset and model cards state access policy, limitations, and citation.
- Confirm result exports are separated from datasets and models.
