# Voice AI Research

Research notes comparing audEERING's commercial Voice AI stack, open-source tooling, and alternative approaches for **age**, **gender**, and **emotion** recognition from speech.

**Source workbook:** `audEERING Voice AI Research.xlsx`

---

## Table of Contents

1. [audEERING Products & devAIce®](#1-audeering-products--devaice)
2. [openSMILE & Wav2Vec2 Models](#2-opensmile--wav2vec2-models)
3. [MAM-CNN Age & Gender Classification](#3-mam-cnn-age--gender-classification)
4. [Model Comparison (Side-by-Side Results)](#4-model-comparison-side-by-side-results)
5. [Emotion Recognition — Open Source Options](#5-emotion-recognition--open-source-options)
6. [Gender Recognition — GitHub Repositories](#6-gender-recognition--github-repositories)
7. [Overall Recommendations](#7-overall-recommendations)
8. [References](#8-references)

---

## 1. audEERING Products & devAIce®

**Summary:** [devAIce®](https://www.audeering.com/products/devaice/) is audEERING's core commercial Voice & Audio AI platform, available as an **SDK** and **Web API**. It bundles multiple specialized audio-analysis modules for configurable, real-time or on-device deployment.

**Platforms:** Windows, Linux, macOS, Android, iOS (x86-64 and ARMv8)

### Product / Module Overview

| Product / Module | Category | What it does | Key outputs | Deployment notes |
|---|---|---|---|---|
| **devAIce®** | Core Voice & Audio AI | Commercial platform combining audEERING audio/voice models | SDK, Web API, real-time/on-device options | Multi-platform |
| **Voice Activity Detection (VAD)** | devAIce® module | Detects presence/absence of human voice; robust to noise | Voice vs. background/non-voice; low-latency detection | Front-end before expensive voice analysis |
| **Expression** | devAIce® module | Analyzes emotional expression in voice | Dimensional: arousal, valence, dominance; categorical: happy, angry, sad, neutral | Large, Small, Tiny variants |
| **Acoustic Scene Detection** | devAIce® module | Classifies acoustic environment | Indoor (small/medium/large); Outdoor (traffic/no traffic); Transport (bus/car/railroad) | 3 top-level classes; 8 sub-scenes |
| **Acoustic Event Detection (AED)** | devAIce® module | Detects overlapping acoustic events | Speech, music (documented categories) | Audio-context/event tagging |
| **Speaker Attributes** | devAIce® module | Estimates speaker-related attributes | Age; perceived gender; ComParE-2016 (6,373 features); GeMAPS+ (276 features) | Model estimates; gender is perceived/self-reported label |
| **Speaker Verification** | devAIce® module | Verifies voice against enrolled reference speaker | Speaker identity matching | Authentication workflows |
| **Automatic Speech Recognition (ASR)** | devAIce® module | Speech-to-text | Whisper.cpp-based transcription | Reduces need for separate ASR |
| **Audio Quality** | devAIce® module | Measures recording/acoustic quality | SNR; RT60 (reverberation) | Any duration; one value per parameter |
| **Prosody** | devAIce® module | Captures how speech is delivered | Pitch/F0, loudness, speaking rate, intonation | Part of Expression package |
| **Features** | devAIce® module | Engineered acoustic feature sets | ComParE-2016 (6,373); GeMAPS+ (276) | Numeric descriptors for classification |
| **devAIce® XR** | XR plugin | Voice analysis for Unity/Unreal | Expression dimensions, VAD, speaker attributes | VR/AR/MR; lightweight/on-device |
| **AI SoundLab** | R&D platform | Web platform for audio data collection and analysis | Surveys, recording library, devAIce integration | Research/health/data collection |

### Official Sources

- [audEERING Products](https://www.audeering.com/products/)
- [devAIce®](https://www.audeering.com/products/devaice/)
- [devAIce® XR](https://www.audeering.com/products/devaice-xr/)
- [AI SoundLab](https://www.audeering.com/products/ai-soundlab/)

---

## 2. openSMILE & Wav2Vec2 Models

**Summary:** [openSMILE](https://audeering.github.io/opensmile/about.html) is a modular **signal-processing and feature-extraction toolkit** (not a single ML model). It computes low-level acoustic descriptors, functionals, normalization, VAD/turn detection, and can interface with classifiers. The **Wav2Vec2** models listed below are separate pretrained deep-learning models.

**License note:** openSMILE is research-only per official docs. Wav2Vec2 model licenses must be checked before commercial use.

### openSMILE Capabilities

| Area / Feature | Type | What it provides | Examples | Notes |
|---|---|---|---|---|
| **openSMILE** | Toolkit | Modular C++ feature extractor; real-time and batch | Feature extraction, preprocessing, classifiers, I/O | Linux, Windows, macOS, Android, iOS |
| Signal processing | Pre-processing | Windowing, resampling, FFT/IFFT, ACF, AMDF | Hann/Hamming windows; FFT | Foundation for feature extraction |
| Data processing | Normalization | Mean-variance/range normalization; deltas; smoothing | Online/offline normalization | Stabilizes features for ML |
| Frame Energy | Low-level feature | Short-time signal energy | Energy contours; mean/max energy | Speech activity, loudness |
| Frame Intensity / Loudness | Low-level feature | Loudness/intensity over time | Loudness contour | Speaking intensity, prosody |
| MFCC / Mel-Bark Cepstral | Low-level feature | Cepstral speech spectrum | MFCCs; delta MFCCs | Common in speech ML |
| Spectral features | Low-level feature | Spectral distribution and shape | Centroid, entropy, skewness, kurtosis | Timbre, classification |
| Fundamental Frequency (F0) | Low-level feature | Vocal pitch via ACF/Cepstrum/SHS | F0 contour; F0 statistics | Core prosody descriptor |
| Probability of Voicing | Low-level feature | Voiced/unvoiced likelihood | Voiced confidence | Pitch and VAD processing |
| Jitter & Shimmer | Voice-quality | Cycle-to-cycle frequency/amplitude variation | Jitter; shimmer | Vocal stability |
| Formants | Low-level feature | Vocal tract resonances | F1/F2/F3; bandwidths | Phonetic characterization |
| Zero / Mean Crossing Rate | Low-level feature | Waveform sign changes | ZCR; MCR | Simple temporal descriptor |
| PLP / PLP-CC | Low-level feature | Perceptual Linear Predictive coefficients | PLP; PLP-CC | Speech-oriented features |
| LPC / LSP (LSF) | Low-level feature | Linear predictive representations | LPC; LSP/LSF | Vocal-tract modeling |
| Auditory / Critical-band spectra | Low-level feature | Mel/Bark/Octave filterbanks | Auditory spectrum | Perceptual analysis |
| Psychoacoustic features | Low-level feature | Perceptual sound descriptors | Sharpness; harmonicity | Timbre analysis |
| CHROMA / CENS | Music/audio | Pitch-class spectral representations | CHROMA; CENS | Music information retrieval |
| F0 Harmonics Ratios | Voice/spectral | Harmonic amplitude ratios | Harmonic ratios | Voice quality |
| Functionals | Aggregation | Time-varying contours → fixed vectors | Means, variance, percentiles, regression, DCT | Creates fixed-size feature vectors |
| VAD / Turn detection | Segmentation | Voice activity and turn segmentation | Fuzzy-logic VAD; LSTM-RNN VAD | Live stream segmentation |
| Classifiers | ML components | Incremental classification | LIBSVM/SVM; GMM; LSTM-RNN; openEAR | Toolkit hosts classifiers |
| Video features | Multimodal | OpenCV video descriptors | HSV histograms; LBP; optical flow | Optional; not audio-only |
| Data output | I/O | Export to common formats | WAV, CSV, HTK, ARFF, LIBSVM, binary matrix | PortAudio playback |
| Feature-set scale | Scale | Large engineered feature spaces | ComParE-2016: 6,373; GeMAPS+: 276 | Depends on LLD + functionals config |

### audEERING Wav2Vec2 Pretrained Models

| Model | Type | Outputs | Notes |
|---|---|---|---|
| **Wav2Vec2 Age/Gender** | Pretrained DL model | Age ~0–100; gender probs (child/female/male) | [24-layer](https://huggingface.co/audeering/wav2vec2-large-robust-24-ft-age-gender) and [6-layer](https://huggingface.co/audeering/wav2vec2-large-robust-6-ft-age-gender) variants on Hugging Face |
| **Wav2Vec2 Emotion** | Pretrained DL model | Emotion-related outputs | [wav2vec2-large-robust-12-ft-emotion-msp-dim](https://huggingface.co/audeering/models) |

### openSMILE Default Feature Sets

Sample configurations for common tasks (usable directly or as custom starting points):

| Feature Set / Configuration | Domain / Task | Max Features | Typical Use |
|---|---|---|---|
| Chroma features | Music Information Retrieval | Config-dependent | Key/chord recognition |
| MFCC | Speech processing | Config-dependent | Speech recognition/classification |
| PLP | Speech processing | Config-dependent | Speech analysis |
| Prosody | Paralinguistics | Config-dependent | Speaking style, affect |
| INTERSPEECH 2009 Emotion Challenge | Speech emotion | 384 | Emotion research baseline |
| INTERSPEECH 2010 Paralinguistic Challenge | Paralinguistics | 1,582 | Speaker/paralinguistic analysis |
| INTERSPEECH 2011 Speaker State Challenge | Speaker state | Config-dependent | Speaker-state research |
| INTERSPEECH 2012 Speaker Trait Challenge | Speaker traits | 6,125 | Speaker-trait research |
| INTERSPEECH 2013 ComParE | Paralinguistics | Config-dependent | Broad paralinguistic tasks |
| MediaEval 2012 TUM | Audio-visual / scene | Config-dependent | Violent-scenes detection |
| Older emotion reference sets 1–3 | Speech emotion | — | Obsoleted reference sets |
| Audio-visual INTERSPEECH 2010 | Audio-visual paralinguistics | — | Audio + visual features |

### Official Sources

- [openSMILE documentation](https://audeering.github.io/opensmile/about.html)
- [audEERING Hugging Face models](https://huggingface.co/audeering/models)

---

## 3. MAM-CNN Age & Gender Classification

**Repository:** [Anvarjon/Age-Gender-Classification](https://github.com/Anvarjon/Age-Gender-Classification)

**Paper:** [Age and Gender Recognition Using a CNN with a Multi-Attention Module through Speech Spectrograms](https://www.mdpi.com/1424-8220/21/17/5892) ([PubMed](https://pubmed.ncbi.nlm.nih.gov/34502785/))

### Research Objective

Classify speaker attributes from speech using a **Convolutional Neural Network with a Multi-Attention Module (MAM)**:

- Gender classification
- Age classification
- Combined age + gender classification

The repo provides an end-to-end R&D workflow: preprocessing → spectrogram → TFRecord → training/testing → inference with pretrained checkpoints.

> **Scope:** Supervised classification. "Age" refers to dataset age classes, not exact-age regression. Gender is limited by training labels and population.

### Approach: Input → Model → Output

```
Audio → preprocessing → spectrogram → TFRecord → CNN + MAM → class prediction
```

| Stage | Details |
|---|---|
| **Input** | Speech/audio recordings |
| **Features** | Spectrogram (time-frequency representation for CNN) |
| **Model** | CNN + Multi-Attention Module |
| **MAM** | Time-attention (temporal cues) + frequency-attention (frequency cues), combined |
| **Output** | Age class, gender class, or combined age-gender class |

**Hypothesis:** MAM focuses the network on the most discriminative spectrogram regions rather than treating all time-frequency areas equally.

**Environment:** Python 3.8, TensorFlow <2.11, CUDA 11.2, cuDNN 8.1 (documented Windows GPU setup).

### Dataset & Reported Results

| Dataset | Gender | Age | Age + Gender | Observation |
|---|---|---|---|---|
| Mozilla Common Voice | 0.96 | 0.73 | 0.76 | Gender strong; age substantially lower |
| Korean dataset (local) | 0.97 | 0.97 | 0.90 | Very strong; cross-domain validation needed |

> These are benchmark results for reported setups — **not universal accuracy guarantees**. The large gap between datasets suggests domain sensitivity.

**Proper evaluation should include:** precision, recall, F1, confusion matrix, per-class performance, class distribution, and MAE/age-distance where applicable.

### Merits

- Complete end-to-end research pipeline with pretrained checkpoints
- Clear architectural contribution (CNN + temporal/frequency attention)
- Supports age, gender, and combined classification
- Adaptable to custom datasets
- Useful reproducible baseline for ablation studies

### Demerits / Risks

- 2021-era architecture; may be outperformed by newer pretrained speech models
- Old TensorFlow/CUDA stack increases maintenance cost
- Dataset-specific accuracy; Common Voice age (73%) << gender (96%)
- No built-in robustness across languages, accents, codecs, noise, or reverberation
- Speaker leakage must be ruled out during reproduction
- Research code ≠ production-ready API or streaming system

### R&D Recommendation

**Verdict: KEEP AS RESEARCH BASELINE — DO NOT SELECT AS FINAL PRODUCTION MODEL YET**

**Recommended path:**

1. Reproduce published benchmarks
2. Verify speaker-disjoint train/val/test splits
3. Test on internal target-domain data
4. Compare: plain CNN → CNN + MAM → OpenSMILE + ML → Wav2Vec2/HuBERT/WavLM
5. Evaluate accuracy + robustness + latency + cost
6. Select model based on evidence, not headline benchmarks

**Key ablation:** plain CNN vs. CNN + MAM vs. modern pretrained speech encoder.

---

## 4. Model Comparison (Side-by-Side Results)

Empirical comparison of **audEERING Wav2Vec2** vs. **MAM-CNN** on 8 test audio clips (English and Telugu, various emotions).

### Models Used

| Model | Path / Identifier |
|---|---|
| audEERING Age/Gender | `audeering/wav2vec2-large-robust-24-ft-age-gender` |
| MAM Age | `/data/Age-Gender-Classification/models/age/best_model_age.h5` |
| MAM Gender | `/data/Age-Gender-Classification/models/gender/best_model_gender.h5` |
| MAM Age+Gender | `/data/Age-Gender-Classification/models/gender_age/best_model_gender_age.h5` |

### Summary Results

| Audio | audEERING Age (years) | audEERING Gender | MAM Age | MAM Gender | Age Match | Gender Match | MAM Gender-Age |
|---|---|---|---|---|---|---|---|
| eng-happy | 13.9 | male | twenties | male | no | yes | female-fifties |
| eng-neutral | 16.9 | male | twenties | male | no | yes | male-fourties |
| eng-sad | 19.8 | male | twenties | male | — | yes | male-fourties |
| eng-surprised | 25.6 | male | twenties | male | yes | yes | male-twenties |
| tel-happy | 15.1 | male | twenties | male | no | yes | female-thirties |
| tel-neutral | 15.7 | male | twenties | male | no | yes | male-fourties |
| tel-sad | 14.8 | male | twenties | male | no | yes | male-fourties |
| tel-surprised | 15.6 | male | twenties | male | no | yes | male-twenties |

### Key Observations

| Metric | audEERING Wav2Vec2 | MAM-CNN |
|---|---|---|
| **Gender** | Consistently **male** with high confidence (~99%) | Consistently **male** with high confidence (~65–99%) |
| **Gender agreement** | — | **8/8 match** with audEERING |
| **Age** | Continuous estimate (~14–26 years); maps to teens/twenties buckets | Always predicts **twenties** (very high confidence ~90–99%) |
| **Age agreement** | — | **1/8 bucket match** (eng-surprised only) |
| **Combined gender-age** | N/A | Often inconsistent with separate age/gender predictions (e.g., female-fifties when gender=male, age=twenties) |

### Detailed Raw Results

| Audio | Duration (s) | audEERING Age | audEERING Gender (F/M/Child) | MAM Age (conf) | MAM Gender (conf) | MAM Gender-Age (conf) | Consistent? |
|---|---|---|---|---|---|---|---|
| audio-eng-happy.wav | 6.17 | 13.95 | 0.002 / **0.995** / 0.004 | twenties (0.99) | male (0.999) | female-fifties (0.45) | No |
| audio-eng-neutral.wav | 7.93 | 16.89 | 0.001 / **0.995** / 0.004 | twenties (0.90) | male (0.999) | male-fourties (0.83) | No |
| audio-eng-sad.wav | 8.45 | 19.82 | 0.002 / **0.995** / 0.003 | twenties (0.94) | male (0.998) | male-fourties (0.67) | No |
| audio-eng-surprised.wav | 6.45 | 25.56 | 0.012 / **0.987** / 0.001 | twenties (0.999) | male (0.647) | male-twenties (0.53) | Yes |
| audio-tel-happy.wav | 4.13 | 15.09 | 0.003 / **0.993** / 0.004 | twenties (0.996) | male (0.748) | female-thirties (0.58) | No |
| audio-tel-neutral.wav | 3.91 | 15.69 | 0.001 / **0.993** / 0.007 | twenties (0.967) | male (1.000) | male-fourties (0.88) | No |
| audio-tel-sad.wav | 5.83 | 14.76 | 0.001 / **0.979** / 0.020 | twenties (0.948) | male (1.000) | male-fourties (0.83) | No |
| audio-tel-surprised.wav | 5.47 | 15.59 | 0.001 / **0.990** / 0.010 | twenties (0.944) | male (0.999) | male-twenties (0.64) | Yes |

**Takeaway:** Both models agree on gender for all test clips. audEERING provides more granular continuous age estimates; MAM-CNN collapses to "twenties" for every clip. The MAM combined gender-age model frequently contradicts its own separate predictions.

---

## 5. Emotion Recognition — Open Source Options

### 5.1 SenseVoice ([QwenAudio/SenseVoice](https://github.com/QwenAudio/SenseVoice))

**SenseVoiceSmall** is the released encoder-only, non-autoregressive speech foundation model supporting ASR, LID, SER, and AED in a single checkpoint.

#### Models and Roles

| Model / Component | Role | Main outputs | Released? |
|---|---|---|---|
| **SenseVoiceSmall** | Encoder-only foundation model | ASR text, language tags, emotion tags, event tags | **Yes** — main released checkpoint |
| SenseVoice-Large | Research benchmark variant | Emotion predictions (benchmark) | No — benchmark only |
| FSMN-VAD | Voice Activity Detection | Speech/non-speech segments | Separate FunASR model |
| CAM++ | Speaker recognition (diarization) | Speaker IDs | Separate FunASR model |
| CT-Punc | Punctuation restoration | Punctuated transcript | Separate FunASR model |

#### Feature & Label Mapping (SenseVoiceSmall)

| Feature | Abbreviation | Output vocabulary |
|---|---|---|
| Automatic Speech Recognition | ASR | Transcribed text (Mandarin, Cantonese, English, Japanese, Korean) |
| Spoken Language Identification | LID | `<|zh|>`, `<|en|>`, `<|yue|>`, `<|ja|>`, `<|ko|>` |
| Speech Emotion Recognition | SER | `<|HAPPY|>`, `<|SAD|>`, `<|ANGRY|>`, `<|NEUTRAL|>`, `<|FEARFUL|>`, `<|DISGUSTED|>`, `<|SURPRISED|>` |
| Audio Event Detection | AED | `<|BGM|>`, `<|Speech|>`, `<|Applause|>`, `<|Laughter|>`, `<|Cry|>`, `<|Sneeze|>`, `<|Breath|>`, `<|Cough|>` |
| Rich Transcription | — | Text + emotion/event tags |
| Timestamp Alignment | CTC | Token/sentence timing |

> Speaker diarization requires composing SenseVoiceSmall + FSMN-VAD + CAM++ + CT-Punc — not a native SenseVoice output.

### 5.2 emotion2vec ([ddlBoJack/emotion2vec](https://github.com/ddlBoJack/emotion2vec))

Self-supervised speech-emotion representation models for research and downstream SER.

| Model | Purpose | Output classes | Training data |
|---|---|---|---|
| **emotion2vec** | Self-supervised emotion representation | Embedding (downstream classifier needed) | Pretrained self-supervised |
| **emotion2vec+ seed** | Direct SER inference + features | 9 classes (see below) | 201 hours |
| **emotion2vec+ base** | Direct SER inference + features | 9 classes | 4,788 hours |
| **emotion2vec+ large** | Direct SER inference + features | 9 classes | 42,526 hours |

**9 emotion classes:** Angry, Disgusted, Fearful, Happy, Neutral, Other, Sad, Surprised, Unknown

**Use cases:** Feature extraction, downstream SER systems, comparing accuracy/latency/memory across model sizes.

---

## 6. Gender Recognition — GitHub Repositories

Comparison of open-source gender recognition repos for R&D baseline selection.

| Repository | Approach | Input / Features | Dataset | Reported Accuracy | Cost | R&D Priority |
|---|---|---|---|---|---|---|
| [ReeraAI/gender-recognition](https://github.com/ReeraAI/gender-recognition) | DNN (TensorFlow/Keras) | Mel Spectrogram → feature vector | Mozilla Common Voice (balanced M/F) | Not stated | Low–Medium | **#1 — Best lightweight DL baseline** |
| [grep-rohan/GenderRecognitionByVoice](https://github.com/grep-rohan/GenderRecognitionByVoice) | MLP + classical ML | Acoustic features (R/warbleR) | voice.csv | Not stated | Low | **#2 — Multi-classifier comparison baseline** |
| [SuperKogito/Voice-based-gender-recognition](https://github.com/SuperKogito/Voice-based-gender-recognition) | MFCC + GMM | MFCC features | SLR45 (10 speakers) | 95% | Very Low | **#3 — Classical ML reference** |

### Assessment

| Repo | Reliability view | Main insight |
|---|---|---|
| ReeraAI | Good candidate; verify speaker-independent evaluation | Best overall modern lightweight DL baseline |
| grep-rohan | Useful baseline; verify dataset/speaker split | Compare MLP, SVM, k-NN, Decision Tree, Random Forest |
| SuperKogito | Very small speaker population (5M/5F) limits generalization | Good classical reference, not preferred final approach |

> Reported accuracy values are **not directly comparable** unless datasets, speaker-independent splits, and metrics are aligned.

---

## 7. Overall Recommendations

### Age & Gender from Speech

| Use case | Recommended approach |
|---|---|
| **Production / commercial** | audEERING devAIce® or Wav2Vec2 Age/Gender (check license) |
| **Interpretable baseline** | openSMILE (ComParE/GeMAPS+) + classical ML |
| **Research baseline** | MAM-CNN (reproduce + ablate attention module) |
| **Modern comparison** | Wav2Vec2, HuBERT, WavLM fine-tuned on target domain |

### Emotion from Speech

| Use case | Recommended approach |
|---|---|
| **Multi-task (ASR + emotion + events)** | SenseVoiceSmall |
| **Dedicated SER research** | emotion2vec+ (seed/base/large by data/latency budget) |
| **Commercial / dimensional** | audEERING devAIce® Expression module |

### Gender-Only Baselines

1. **ReeraAI/gender-recognition** — primary lightweight DL baseline
2. **grep-rohan/GenderRecognitionByVoice** — classical ML comparison
3. **SuperKogito** — MFCC+GMM reference only

### Validation Checklist (All Models)

- [ ] Speaker-disjoint train/validation/test splits
- [ ] Test unseen speakers and target audio domain
- [ ] Test languages, accents, microphones, codecs, noise, reverberation
- [ ] Report per-class F1, precision, recall, confusion matrices
- [ ] Check confidence calibration
- [ ] Measure latency, throughput, memory, GPU/CPU requirements
- [ ] Keep a final held-out set untouched until model selection is complete

---

## 8. References

### audEERING / devAIce®

- https://www.audeering.com/products/
- https://www.audeering.com/products/devaice/
- https://www.audeering.com/products/devaice-xr/
- https://www.audeering.com/products/ai-soundlab/

### openSMILE & Wav2Vec2

- https://audeering.github.io/opensmile/about.html
- https://huggingface.co/audeering/models
- https://huggingface.co/audeering/wav2vec2-large-robust-24-ft-age-gender
- https://huggingface.co/audeering/wav2vec2-large-robust-6-ft-age-gender

### MAM-CNN Age/Gender

- https://github.com/Anvarjon/Age-Gender-Classification
- https://www.mdpi.com/1424-8220/21/17/5892
- https://pubmed.ncbi.nlm.nih.gov/34502785/

### Emotion Recognition

- https://github.com/QwenAudio/SenseVoice
- https://github.com/ddlBoJack/emotion2vec

### Gender Recognition

- https://github.com/ReeraAI/gender-recognition
- https://github.com/grep-rohan/GenderRecognitionByVoice
- https://github.com/SuperKogito/Voice-based-gender-recognition

---

*Generated from `audEERING Voice AI Research.xlsx` — September 2026*
