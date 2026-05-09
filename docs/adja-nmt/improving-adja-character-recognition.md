# Improving Adja Character Recognition in ASR

Research-backed techniques for improving recognition of Adja's special characters
(ɛ, ɔ, ŋ, ɖ) and tone marks (é, è, ẽ, ɔ̀) in CTC-based ASR systems.

## The Problem

Adja uses characters that most ASR models have never seen:
- **IPA extensions**: ɛ (open-mid front), ɔ (open-mid back), ŋ (velar nasal), ɖ (retroflex)
- **Tone marks**: é, è, ẽ, ɔ̀ — these change word meaning (tonal minimal pairs)
- **Combined**: characters like ɔ̀ = base character + combining accent

Our current CTC models output mostly "E" — they've learned the most frequent character
but not the full alphabet. Rare characters like ɖ and ŋ barely appear in the output.

---

## 1. Vocabulary Design (Quick Win)

### NFC vs NFD: Composed vs Decomposed

| Approach | Example | Vocab Size | Pros | Cons |
|----------|---------|-----------|------|------|
| **NFC** (composed) | `ɔ̀` = 1 token | Smaller | Simpler, fewer alignment steps | Rare combined chars hard to learn |
| **NFD** (decomposed) | `ɔ` + `̀` = 2 tokens | Larger | Base char shared across toned/untoned | CTC must align 2 tokens to 1 sound |
| **Hybrid** | Keep ɛ,ɔ,ŋ,ɖ composed, separate tone marks | Medium | Best of both worlds | Slightly complex |

**Recommendation**: Try the hybrid approach — keep IPA characters composed but split
tone diacritics as separate tokens. This lets the model share base character knowledge
across toned and untoned variants.

**Paper**: [EUSIPCO 2024 tokenization study](https://eurasip.org/Proceedings/Eusipco/Eusipco2024/pdfs/0000141.pdf) —
character-level CTC is strongly preferred over BPE in low-resource settings.

### Character vs BPE/Subword

With ~1,600 utterances: **stick with character-level CTC**. BPE needs much more data
to learn meaningful subword statistics. Character-level saturates better on small data.

---

## 2. Focal / Weighted CTC Loss (High Impact)

Standard CTC loss treats all characters equally. But in Adja, `e` and `a` appear
hundreds of times while `ɖ` and `ŋ` appear rarely. The model learns the frequent
characters and ignores the rare ones.

**Focal CTC Loss**: Down-weight easy/frequent character predictions so gradient
signal flows toward hard/rare ones. Adds a `(1-p)^γ` modulating factor.

```python
# Concept: weight each character inversely proportional to frequency
char_freq = count_chars(all_transcripts)
char_weights = 1.0 / (char_freq + 1e-6)
char_weights = char_weights / char_weights.sum() * len(char_weights)
```

**Papers**:
- [Focal CTC Loss (Feng et al., 2019)](https://www.hindawi.com/journals/complexity/2019/9345861/) —
  3-9% accuracy improvement on rare characters in Chinese OCR
- [Re-weighted CTC for speech (2023)](https://www.researchgate.net/publication/372310797) —
  confirms weighted CTC helps with imbalanced data in speech

---

## 3. Tone Recognition Techniques

### Pitch (F0) Features

Standard mel spectrograms capture frequency info but F0 (fundamental frequency) carries
the primary tone signal. Options:

1. **Concatenate F0 to mel features**: Extract F0 with CREPE or pYIN, append as extra
   dimension to the 80-dim mel spectrogram → 81 dims
2. **Multi-resolution**: Compute F0 at different time scales, concatenate all
3. **Let the model learn it**: wav2vec2 implicitly captures pitch info. Fine-tuning
   may be enough if the model has seen tonal languages during pretraining.

### Multi-Task Learning (Recommended)

Add an auxiliary tone classification head alongside CTC:
```
wav2vec2 encoder → CTC head (character prediction)
                 → Tone head (tone classification per frame)
```

Loss = CTC_loss + λ × tone_classification_loss

This forces the encoder to explicitly represent tonal information.

**Paper**: [wav2vec2 for Yoruba Tone Recognition (ACM TALLIP 2024)](https://dl.acm.org/doi/10.1145/3690384) —
Tone Error Rate 17.72%, 50% error reduction over prior work. Directly applicable to Adja.

### AfriHuBERT (Alternative Base Model)

[AfriHuBERT (Alabi et al., Interspeech 2025)](https://arxiv.org/abs/2409.20201) extends
mHuBERT with continued pretraining on 10K+ hours from 1,226 African languages.
Since it's trained on African tonal languages, it may have better tonal representations
than generic wav2vec2/MMS.

- Model: search HuggingFace for "AfriHuBERT"
- Expected benefit: better tone features out of the box
- +3.6% F1 for language ID, -2.1% WER over mHuBERT-147

---

## 4. Language Model Integration (High Impact, Easy)

Even a simple n-gram character LM can dramatically improve CTC decoding by
constraining outputs to valid Adja character sequences.

### How to Build

```bash
# 1. Extract all transcriptions
cut -f3 data/manifests/train.tsv | tail -n +2 > adja_text.txt

# 2. Train character-level 5-gram LM with KenLM
# (need to space-separate characters first)
sed 's/./& /g' adja_text.txt > adja_chars.txt
lmplz -o 5 < adja_chars.txt > adja_char_5gram.arpa

# 3. Use with pyctcdecode during inference
from pyctcdecode import build_ctcdecoder
decoder = build_ctcdecoder(
    labels=vocab_list,
    kenlm_model_path="adja_char_5gram.arpa",
    alpha=0.5,  # LM weight (tune on dev set)
    beta=1.0,   # word insertion bonus
)
```

**Paper**: [Whisper-LM (2025)](https://arxiv.org/abs/2503.23542) — up to 51% WER
reduction with LM integration. Even for CTC models, LM rescoring helps significantly.

**Tool**: [pyctcdecode](https://github.com/kensho-technologies/pyctcdecode)

---

## 5. Data Augmentation

### Audio-Level (Apply All of These)

| Technique | Effect | Tool |
|-----------|--------|------|
| **SpecAugment** | Mask frequency/time bands | torchaudio |
| **Speed perturbation** (0.9, 1.0, 1.1) | Simulate speaking rate variation | torchaudio |
| **Noise injection** | Add background noise from MUSAN | torchaudio |
| **Pitch shifting** | Simulate different speakers | torchaudio |
| **Room impulse response** | Simulate different rooms | torchaudio |

### Text-Level (for Rare Characters)

Generate synthetic text with more rare characters, then use TTS to create audio:

1. Find words containing ɖ, ŋ, ɔ, ɛ in your transcriptions
2. Create new sentences combining these words
3. Use a TTS model (MMS-TTS-Ewe exists!) to synthesize audio
4. Add to training data

**Paper**: [TTS Augmentation for African ASR (2025)](https://arxiv.org/html/2507.17578v1) —
text diversity matters more than speaker diversity.

### TTS-Based Data Multiplication

The [Facebook MMS TTS for Ewe](https://huggingface.co/facebook/mms-tts-ewe) exists.
While it's Ewe not Adja, the phoneme systems are similar enough that it could generate
plausible Adja-like audio for augmentation. Worth experimenting.

---

## 6. Transfer Learning Tricks

### MMS Adapter Fine-Tuning (Current Approach)

MMS with adapter fine-tuning is the most data-efficient approach per the
[2024 African ASR benchmark](https://arxiv.org/html/2512.10968v1). Only a small
fraction of parameters are trained, preserving multilingual acoustic knowledge.

### Cross-Lingual Character Mapping

[Arxiv 2412.16474](https://arxiv.org/abs/2412.16474) proposes computing weighted sums
of language embeddings for unseen languages. For Adja:
- Map ɛ → Ewe/Fon ɛ (MMS has seen these)
- Map ɔ → Ewe/Fon ɔ
- Map ŋ → any language with velar nasal
- Map ɖ → retroflex in Hindi/Marathi (MMS has seen these)

### Character-Level Models Transfer Better

[2025 study](https://arxiv.org/html/2505.24561) confirmed character-based approaches
achieve better cross-lingual transfer than subword models in low-resource settings.
This validates our CTC character-level approach.

---

## 7. Practical Priority List

What to try first (ordered by effort vs impact):

### Immediate (No Code Changes)
1. ✅ **NFC normalization** — already done
2. ✅ **SpecAugment + speed perturbation** — already in B1
3. 🔄 **Train longer** (patience=20, epochs=50) — C3v2 running now

### Next Round (Small Code Changes)
4. **Build character n-gram LM** + use pyctcdecode for beam search decoding
5. **Add noise augmentation** (MUSAN dataset) to training pipeline
6. **Try hybrid vocab** — separate tone marks from base characters

### Research Experiments (More Involved)
7. **Focal CTC loss** — weight rare characters higher
8. **Multi-task tone classification** — auxiliary head
9. **AfriHuBERT** as alternative base model
10. **TTS augmentation** — use MMS-TTS-Ewe to generate synthetic training data

---

## Key Papers (Full List)

| Topic | Paper | Link |
|-------|-------|------|
| CTC vocab design | EUSIPCO 2024 Tokenization | https://eurasip.org/Proceedings/Eusipco/Eusipco2024/pdfs/0000141.pdf |
| Focal CTC loss | Feng et al. 2019 | https://www.hindawi.com/journals/complexity/2019/9345861/ |
| Yoruba tone ASR | ACM TALLIP 2024 | https://dl.acm.org/doi/10.1145/3690384 |
| AfriHuBERT | Alabi et al. 2025 | https://arxiv.org/abs/2409.20201 |
| Cross-lingual char mapping | arXiv 2024 | https://arxiv.org/abs/2412.16474 |
| Whisper + LM | de Zuazo et al. 2025 | https://arxiv.org/abs/2503.23542 |
| TTS augmentation for African ASR | 2025 | https://arxiv.org/html/2507.17578v1 |
| Data augmentation | Ibaraki et al. 2025 | https://arxiv.org/abs/2509.15373 |
| Practitioner's guide | Klejch et al. 2025 | https://arxiv.org/abs/2506.04915 |
| African ASR benchmark | PazaBench 2024 | https://arxiv.org/html/2512.10968v1 |
| Character-level transfer | 2025 study | https://arxiv.org/html/2505.24561 |
| Inter-layer CTC | Hojo et al. 2024 | https://www.isca-archive.org/interspeech_2024/hojo24_interspeech.pdf |
| pyctcdecode | Tool | https://github.com/kensho-technologies/pyctcdecode |
| KenLM | Tool | https://github.com/kpu/kenlm |
