# Registry — NLLB chrF mission

Target: beat **chrF 41.2** (paper baseline, NLLB-200-distilled-600M, fr→aj on 1455-row test set).

| run_id | hf_job_id | experiment | condition | seed | status | test_chrF | test_spBLEU | notes |
|---|---:|---|---|---:|---|---:|---:|---|
| smoke | 69f0a25ed70108f37ace0f19 | smoke | t4-validation | 42 | completed | 13.0 | 0.2 | 3 train steps only — infra validation, not a real result |
| h0 | 69f0e977d2c8bd8662bd22e9 | h0-baseline | ewe-lr1e4 | 42 | **completed** | **42.1** | 24.0 BLEU | **Beats paper 41.2 by +0.9.** Best val chrF 47.1 at eval 2, then declined → model overfits early. chrF++=40.0, TER=72.7. |
| h1 | 69f0e978d70108f37ace11d1 | h1-fon-donor | fon-lr1e4 | 42 | **completed** | 41.5 | 23.5 BLEU | Hypothesis: Fon donor. Result: -0.6 vs h0 (Ewe still wins as Gbe donor). Best val chrF 47.0 at eval 2. |
| h2 | 69f0e97ad70108f37ace11d3 | h2-lowlr-long | ewe-lr5e5-ep80 | 42 | **completed** | 41.4 | 22.5 BLEU | Hypothesis: LR=5e-5, 80 epochs. Result: -0.7 vs h0. Lower LR didn't find a better basin. Best val chrF 46.1. |

## Failed submissions (intern's first attempt, missing RESULTS_REPO env)
- 69f0e8c9d2c8bd8662bd22e2 — KeyError: 'RESULTS_REPO'
- 69f0e8e1d70108f37ace11cd — KeyError: 'RESULTS_REPO'
- 69f0e8f8d2c8bd8662bd22e4 — KeyError: 'RESULTS_REPO'
