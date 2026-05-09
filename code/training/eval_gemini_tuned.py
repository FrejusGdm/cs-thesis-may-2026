"""Quick script to evaluate already-completed Gemini tuning jobs."""
import json, os, sys, time, re, unicodedata
from datetime import datetime, timezone

sys.stdout.reconfigure(line_buffering=True)

from sacremoses import MosesPunctNormalizer
import sacrebleu

_mpn = MosesPunctNormalizer(lang="en")
_mpn.substitutions = [(re.compile(r), sub) for r, sub in _mpn.substitutions]

def _get_npc_replacer(replace_by=" "):
    m = {ord(c): replace_by for c in (chr(i) for i in range(sys.maxunicode + 1))
         if unicodedata.category(c) in {"C","Cc","Cf","Cs","Co","Cn"}}
    return lambda line: line.translate(m)

_rnp = _get_npc_replacer(" ")

def preproc(text):
    return unicodedata.normalize("NFKC", _rnp(_mpn.normalize(text)))

SYSTEM_PROMPT = "You are a French to Adja translator. Translate the French text to Adja. Output only the Adja translation, nothing else."

# Load test data
test_tsv = "experiments/data/splits/shared/test.tsv"
sources, references = [], []
with open(test_tsv) as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) >= 2:
            sources.append(preproc(parts[0]))
            references.append(preproc(parts[1]))
print(f"Loaded {len(sources)} test sentences")

from google import genai
client = genai.Client(vertexai=True, project="testingout-423013", location="us-central1")

models = {
    "RANDOM-10K": "projects/86019127115/locations/us-central1/endpoints/8369658982247170048",
    "RANDOM-6K_STRUCTURED-4K": "projects/86019127115/locations/us-central1/endpoints/8654511658678353920",
}

for condition, endpoint in models.items():
    print(f"\n{'='*60}")
    print(f"Evaluating: {condition}")
    print(f"{'='*60}")

    predictions = []
    start = time.time()
    for i, src in enumerate(sources):
        try:
            response = client.models.generate_content(
                model=endpoint, contents=src,
                config={"system_instruction": SYSTEM_PROMPT, "temperature": 0, "max_output_tokens": 256},
            )
            pred = response.text.strip() if response.text else ""
            predictions.append(pred)
        except Exception as e:
            print(f"  Warning [{i}]: {e}")
            predictions.append("")
        if (i+1) % 100 == 0:
            print(f"  Progress: {i+1}/{len(sources)}")

    elapsed = time.time() - start
    print(f"  Done: {len(predictions)} predictions in {elapsed:.0f}s")

    bleu = sacrebleu.corpus_bleu(predictions, [references])
    chrf = sacrebleu.corpus_chrf(predictions, [references])
    chrfpp = sacrebleu.corpus_chrf(predictions, [references], word_order=2)
    ter = sacrebleu.corpus_ter(predictions, [references])

    metrics = {
        "test_bleu": bleu.score, "test_chrf": chrf.score,
        "test_chrfpp": chrfpp.score, "test_ter": ter.score,
        "test_bleu_signature": str(bleu), "test_n_samples": len(predictions),
        "inference_time_seconds": elapsed,
        "experiment": "exp1", "condition": condition, "seed": 42,
        "model": endpoint, "base_model": "gemini-2.5-flash",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "platform": "gemini-vertex",
    }

    output_dir = f"results/gemini/exp1/{condition}/seed42"
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(f"{output_dir}/predictions.tsv", "w") as f:
        for s, r, p in zip(sources, references, predictions):
            f.write(f"{s}\t{r}\t{p}\n")

    print(f"  BLEU:   {metrics['test_bleu']:.1f}")
    print(f"  chrF:   {metrics['test_chrf']:.1f}")
    print(f"  chrF++: {metrics['test_chrfpp']:.1f}")
    print(f"  TER:    {metrics['test_ter']:.1f}")
    print(f"  Saved:  {output_dir}/")

print("\nAll done!")
