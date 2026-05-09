# Thesis Results Map

This is the public map for the first lightweight cleanup pass. It tells readers
which artifacts are thesis-final, which mixed repos are provenance, and where
to look before the ASR/TTS result archives are consolidated.

## Best Public Models

| Task | Public artifact | Role | Headline |
| --- | --- | --- | --- |
| MT French -> Adja | `JosueG/adja-nmt-nllb-600m-forward-r10ks4k-seed42` | thesis-final NLLB forward checkpoint | full/module NLLB runs around 21-22 BLEU and 29 chrF in MT summaries |
| MT Adja -> French | `JosueG/adja-nmt-nllb-600m-reverse-r10ks4k-seed42` | thesis-final NLLB reverse checkpoint | used for Adja text interpretation and speech-pipeline MT |
| ASR primary | `JosueG/wav2vec2-xlsr-adja-c4v2` | deployable C4v2 XLS-R CTC ASR | 25.05% CER greedy; 22.67% normalized CER with character LM |
| ASR complementary | `JosueG/whisper-ewe-adja-e4v4` | deployable Whisper-Ewe -> Adja ASR | 37.18% dev CER; useful second judge with hallucination caveats |
| TTS primary | `JosueG/spark-tts-adja-t3` | thesis-best Spark T3 Adja TTS | intelligible native-listening result; 36.14% C4v2 reverse-CER for canonical T3 |

## Experiment Archive

The release keeps negative and partial results because they explain the final
choices:

- CTC blank collapse: early XLS-R and wav2vec2 runs learned to emit blanks,
  which led to the built-in CTC-loss fix and C4v2.
- MMS and Omni experiments: useful for architecture exploration but not the
  final public ASR weights.
- Whisper redo runs: some checkpoints hallucinated or overfit; E4v4 is the
  surviving deployable Whisper-Ewe artifact, while the stronger original E4
  metric is treated as logged-only.
- TTS trials: Orpheus, CSM, F5, and several direct LoRA attempts produced
  noise or non-Adja output; Spark T3 is the best surviving family.
- Qwen/Omni pilots: operational but behind the XLS-R and Whisper-Ewe systems.

## Where Things Live

| Artifact class | Current public location | Provenance / private source |
| --- | --- | --- |
| Thesis GitHub release | `FrejusGdm/cs-thesis-may-2026` | generated from curated private workspaces |
| MT dataset | `JosueG/french-adja-parallel-corpus` | unchanged canonical public MT corpus |
| Speech dataset | `JosueG/adja-speech-asr-tts` | duplicated from `JosueG/adja-tts-orpheus` |
| MMS-ready derivative | not canonical public source | `JosueG/adja-tts-mms-ready`, kept as provenance only |
| MT models | two NLLB model repos listed above | existing best model repos remain weight locations |
| ASR models | C4v2 and E4v4 model repos listed above | result dumps remain in `JosueG/adja-asr-results` |
| TTS model | `JosueG/spark-tts-adja-t3` | source/checkpoint provenance in `JosueG/adja-tts-results` and `JosueG/adja-tts-checkpoints` |
| ASR/TTS result dumps | this GitHub release summary for now | mixed private result/checkpoint repos |
| Generated audio | selected reports only in this pass | future canonical TTS result repo should export reviewed examples |

## Next Cleanup Pass

The next pass should create consistent public result archives without changing
old experiment repos:

- `JosueG/cs-thesis-may-2026-asr-results`
- `JosueG/cs-thesis-may-2026-tts-results`
- optional `JosueG/cs-thesis-may-2026-mt-results`
