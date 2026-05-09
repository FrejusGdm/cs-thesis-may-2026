# Reading order — a curriculum for this project

Last updated: **2026-04-21**

Roughly 2-3 weeks of evening reading if you do one paper + its linked concept
file per day. Ordered by prerequisite, not by importance.

## Week 1 — foundations

| Order | File / Paper | Why it matters | Time |
|-------|-------------|----------------|------|
| 1 | `concepts/01-what-is-a-tokenizer.md` | Byte-fragmentation broke CSM/Orpheus | 30 min |
| 2 | Sennrich et al., "Neural Machine Translation of Rare Words with Subword Units" (https://arxiv.org/abs/1508.07909) | BPE, the basis of Llama's tokenizer | 60 min |
| 3 | `concepts/02-what-is-an-audio-codec.md` | How raw audio → tokens and back | 30 min |
| 4 | Défossez et al., "Moshi / Mimi codec" (https://arxiv.org/abs/2410.00037) | The codec CSM uses (see §3 for design) | 90 min |
| 5 | `concepts/03-three-layer-tts.md` | Mental model for debugging any TTS model | 30 min |

## Week 2 — ASR

| Order | File / Paper | Why it matters | Time |
|-------|-------------|----------------|------|
| 6 | `concepts/04-ctc-vs-attention-vs-rnnt.md` | The three decoding families, when to use each | 40 min |
| 7 | Graves et al., "Connectionist Temporal Classification" (https://www.cs.toronto.edu/~graves/icml_2006.pdf) | The loss our XLS-R and MMS runs use | 90 min |
| 8 | Radford et al., "Whisper" (https://cdn.openai.com/papers/whisper.pdf) | The architecture of our best-performing ASR | 120 min |
| 9 | `concepts/05-ssl-pretraining-wav2vec.md` | What 'pretraining' actually changes | 30 min |
| 10 | Baevski et al., "wav2vec 2.0" (https://arxiv.org/abs/2006.11477) | The SSL objective we run on unlabeled Ewe | 120 min |
| 11 | Babu et al., "XLS-R" (https://arxiv.org/abs/2111.09296) | Why multilingual SSL helps low-resource | 60 min |
| 12 | Pratap et al., "MMS" (https://arxiv.org/abs/2305.13516) | 1100+ language coverage, includes Ewe | 90 min |

## Week 3 — TTS + synthesis

| Order | File / Paper | Why it matters | Time |
|-------|-------------|----------------|------|
| 13 | `concepts/07-tonal-languages-for-ML.md` | Why Gbe is interesting — tones as F0 contours | 30 min |
| 14 | Borsos et al., "AudioLM" (https://arxiv.org/abs/2209.09143) | The paradigm behind CSM/Orpheus audio tokens | 90 min |
| 15 | Sesame CSM blog + code (https://www.sesame.com/research/crossing_the_uncanny_valley_of_voice) | Our primary TTS target | 60 min |
| 16 | Siuzdak, "SNAC" (https://github.com/hubertsiuzdak/snac, README + repo) | The codec Orpheus uses (we also confirmed it's language-agnostic) | 30 min |

## Week 4 — LoRA, transfer, writing

| Order | File / Paper | Why it matters | Time |
|-------|-------------|----------------|------|
| 17 | `concepts/06-lora-and-peft.md` | Why we can train a 3B model on a laptop GPU | 30 min |
| 18 | Hu et al., "LoRA" (https://arxiv.org/abs/2106.09685) | The original paper, easy read | 60 min |
| 19 | `concepts/08-low-resource-transfer.md` | Positions our work in the literature | 30 min |
| 20 | NLLB team, "No Language Left Behind" (https://arxiv.org/abs/2207.04672) | Baseline MT the NMT project built on | 120 min |
| 21 | Experiments-explained deep dive: read every file in that folder in chronological order | The story of the whole sprint | 2-3 hrs |

## When you get stuck

- Drop into the paper's GitHub repo and run the minimal example — it's faster
  than re-reading the math.
- Re-read `concepts/NN-*.md` for the layer that's confusing. Those files are
  written deliberately as "refresh when lost" material, not as a formal textbook.
- Skim `../learnings-from-the-past/training-gotchas.md` — most confusing bugs
  are already documented there.

## Don't worry about

- Understanding every line of math in papers. Skim proofs on first read.
- Reading papers outside this list early — depth > breadth while you build
  the foundation.
- Perfectly memorizing hyperparameters. You'll see them a dozen times in the
  scripts and internalize them.
