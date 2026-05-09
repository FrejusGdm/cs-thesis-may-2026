# SageMaker setup guide for adja-nmt

Last updated: 2026-04-26

This guide covers everything needed to submit training jobs from your local Mac
to AWS SageMaker. Once set up, submission is a single command.

---

## 1. IAM role

You need a SageMaker execution role with permission to train and write to S3.

1. Open the [IAM console](https://console.aws.amazon.com/iam/home).
2. Create a new role: Trusted entity = **AWS service**, Use case = **SageMaker**.
3. Attach these managed policies:
   - `AmazonSageMakerFullAccess`
   - `AmazonS3FullAccess` (or a scoped S3 read/write policy limited to your bucket)
4. Name it something like `SageMakerExecutionRole-adja`.
5. Copy the ARN — you will need it in step 4.

---

## 2. Local install

```bash
pip install "sagemaker>=2.200" boto3
```

The SageMaker SDK is only needed on your laptop for submission. All training
dependencies (transformers, datasets, etc.) are installed inside the container
via `scripts/sagemaker_jobs/requirements.txt`.

---

## 3. AWS credentials

```bash
aws configure
```

Enter:
- **AWS Access Key ID** — from IAM > Users > your user > Security credentials
- **AWS Secret Access Key** — same place
- **Default region** — `us-east-1` (change if you prefer a different region)
- **Default output format** — `json`

Credentials are stored in `~/.aws/credentials` (never commit this file).

Verify with:
```bash
aws sts get-caller-identity
```

---

## 4. Environment variables

```bash
# Paste your HuggingFace token (needed for JosueG/adja-tts-orpheus and CSM)
export HF_TOKEN=hf_...

# Paste the IAM role ARN from step 1 (only needed outside SageMaker notebooks)
export SAGEMAKER_ROLE=arn:aws:iam::<account-id>:role/SageMakerExecutionRole-adja
```

Add these to your `~/.zshrc` so they persist across terminal sessions.

---

## 5. First test — dry run (no AWS calls)

```bash
python scripts/sagemaker_jobs/launch.py --experiment CF2 --smoke --dry-run
```

Expected output: prints instance type, estimated cost, job name, then exits.
No AWS API calls are made. If this errors, fix the install before proceeding.

---

## 6. Instance types and costs

| Instance | GPUs | VRAM | On-demand (USD/hr) | Spot (est.) | Use for |
|---|---|---|---|---|---|
| ml.p3.2xlarge | 1x V100 | 16 GB | $3.83 | ~$2.68 | CF1, CF2, CF3, TK1 |
| ml.p3.8xlarge | 4x V100 | 64 GB | $14.69 | ~$10.28 | M1, M2, M2b, AT1 |
| ml.g5.2xlarge | 1x A10G | 24 GB | $1.21 | ~$0.85 | Cheap smoke tests |
| ml.p4d.24xlarge | 8x A100 | 320 GB | $32.77 | ~$22.94 | Large-scale only |

Spot instances are enabled by default in `launch.py`. Use `--no-spot` to
force on-demand if spot availability is low.

---

## 7. Submitting a job

```bash
# Full AT1 run on spot
python scripts/sagemaker_jobs/launch.py --experiment AT1

# Smoke test (4 samples, 2 steps) on spot
python scripts/sagemaker_jobs/launch.py --experiment AT1 --smoke

# On-demand (disable spot)
python scripts/sagemaker_jobs/launch.py --experiment AT1 --no-spot
```

The launcher prints a job name and exits immediately. Training continues
asynchronously in AWS.

---

## 8. Monitoring in CloudWatch

```bash
# Stream logs live (replace JOB_NAME with the printed job name)
aws logs tail /aws/sagemaker/TrainingJobs \
    --follow \
    --log-stream-name-prefix JOB_NAME
```

Or open the SageMaker console:
```
https://console.aws.amazon.com/sagemaker/home?#/jobs/JOB_NAME
```

---

## 9. Retrieving results

When the job completes, SageMaker uploads `SM_OUTPUT_DATA_DIR` and `SM_MODEL_DIR`
to S3 automatically. To download locally:

```python
import sagemaker

session = sagemaker.Session()
session.download_data(
    path="./results/AT1",
    bucket=session.default_bucket(),
    key_prefix="JOB_NAME/output",
)
```

Or with the AWS CLI:
```bash
aws s3 cp s3://sagemaker-<region>-<account>/JOB_NAME/output/ ./results/AT1/ --recursive
```

The final model checkpoint is in `model.tar.gz` inside the output prefix.

---

## 10. Cost control

**Spot instances** — enabled by default, ~30% cheaper than on-demand. Risk:
spot interruption mid-job. For AT1 (~34h), consider setting
`checkpoint_s3_uri` in `launch.py` so training can resume after interruption.

**Billing alerts** — set a CloudWatch billing alarm:
1. Open [Billing Alerts](https://console.aws.amazon.com/billing/home#/preferences)
2. Enable billing alerts, then go to CloudWatch > Alarms > Create alarm
3. Metric: Billing > Total Estimated Charge
4. Threshold: e.g. $100 — sends email before a surprise bill

**Smoke before full run** — always run `--smoke` first to confirm the job
starts and the script is importable. A smoke job on ml.p3.2xlarge costs < $0.10.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `NoCredentialsError` | `aws configure` not done | Run `aws configure` |
| `ClientError: AccessDenied` | Role lacks S3 or SageMaker permission | Check IAM policies |
| `SAGEMAKER_ROLE not set` | Running outside notebook, no env var | `export SAGEMAKER_ROLE=arn:...` |
| Job fails immediately | Missing import in script | Run `--smoke --dry-run` locally first |
| Spot job interrupted | AWS reclaimed capacity | Re-submit; consider `--no-spot` for AT1 |
| `No space left on device` | HF cache on small root disk | Already mitigated: cache redirected to `/tmp/hf_cache` via env vars |
