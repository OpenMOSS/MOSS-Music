# MOSS-Music

<p align="center">
  <img src="./assets/MOSS-Music.png" width="58%" alt="MOSS-Music logo" />
</p>

<div align="center">

<a href="https://huggingface.co/collections/OpenMOSS-Team/moss-music"><img src="https://img.shields.io/badge/Huggingface-Models-orange?logo=huggingface&amp"></a>
<a href="https://modelscope.cn/collections/openmoss/MOSS-Music"><img src="https://img.shields.io/badge/ModelScope-Models-624AFF?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZmlsbD0id2hpdGUiIGQ9Ik0xMiAyQzYuNDggMiAyIDYuNDggMiAxMnM0LjQ4IDEwIDEwIDEwIDEwLTQuNDggMTAtMTBTMTcuNTIgMiAxMiAyeiIvPjwvc3ZnPg==&amp"></a>
<img src="https://img.shields.io/badge/Blog-Coming_Soon-blue?logo=internet-explorer&amp">
<img src="https://img.shields.io/badge/Arxiv-Coming_Soon-red?logo=Arxiv&amp">

<a href="https://x.com/Open_MOSS"><img src="https://img.shields.io/badge/Twitter-Follow-black?logo=x&amp"></a>
<a href="https://discord.gg/Xf3aXddCjc"><img src="https://img.shields.io/badge/Discord-Join-5865F2?logo=discord&amp"></a>

</div>

<p align="center">
  <a href="./README.md">English</a> | <a href="./README_zh.md">简体中文</a>
</p>

**MOSS-Music** is an open-source **music understanding model** from
[MOSI.AI](https://mosi.cn/#hero), the [OpenMOSS team](https://www.open-moss.com/),
and [Shanghai Innovation Institute](https://www.sii.edu.cn/). Built on the same
audio backbone as [MOSS-Audio](https://github.com/OpenMOSS/MOSS-Audio),
MOSS-Music is further specialised on music via dedicated continual pre-training
and supervised fine-tuning — targeting **musical captioning, lyrics ASR,
structural analysis, chord / key / tempo reasoning, and long-form musical
question answering**. In this release, we provide **two 8B models**:
**MOSS-Music-8B-Instruct** and **MOSS-Music-8B-Thinking**. The Instruct variant
is optimised for direct instruction following on musical prompts, while the
Thinking variant provides stronger chain-of-thought reasoning for musical
analysis.

## News

* 2026.05.01: 🎉🎉🎉 We have released [MOSS-Music](https://huggingface.co/OpenMOSS-Team/MOSS-Music-8B-Instruct).
* 2026.05.01: 🎉🎉🎉 We have released [MOSS-Music-Data-Pipeline](https://github.com/wx9songs/MOSS-Music-Data-Pipeline) for large-scale music data annotation and processing.

## Contents

- [Introduction](#introduction)
- [Model Architecture](#model-architecture)
  - [DeepStack Cross-Layer Feature Injection](#deepstack-cross-layer-feature-injection)
  - [Time-Aware Representation](#time-aware-representation)
- [Released Models](#released-models)
- [Music Data Pipeline](#music-data-pipeline)
- [Evaluation](#evaluation)
- [Quickstart](#quickstart)
  - [Environment Setup](#environment-setup)
  - [SGLang Serving](#sglang-serving)
  - [Gradio App](#gradio-app)
- [More Information](#more-information)
- [LICENSE](#license)
- [Citation](#citation)

## Introduction

Music is not just audio plus lyrics — understanding it requires perceiving
harmonic structure, rhythm, timbre, instrumentation, performance nuance, and
the textual content of the lyrics, and reasoning about them jointly across
time. **MOSS-Music** is built to unify these capabilities within a single
model.

- **Lyrics ASR & time-aligned transcription**: Accurate singing ASR with
  sentence- and word-level timestamps, robust to backing tracks.
- **Musical captioning & tagging**: Natural-language descriptions of mood,
  genre, instrumentation, production style, and emotional trajectory.
- **Key / tempo / chord reasoning**: Identifies musical key, beats, downbeats,
  and chord progressions, including timestamped chord transcription.
- **Structural analysis**: Segments a song into intro / verse / chorus /
  bridge / outro and reasons about repetition and contrast.
- **Instrument & voice recognition**: Identifies prominent instruments and
  singing voices (solo / chorus, gender, register).
- **Musical QA and long-form analysis**: Open-ended question answering
  grounded in a full track, including chain-of-thought reasoning in the
  *Thinking* variant.

<p align="center">
  <img src="./assets/moss-music_img.png" width="98%" alt="MOSS-Music overview" />
</p>

## Model Architecture

MOSS-Music inherits the MOSS-Audio modular design, comprising three
components: an audio encoder, a modality adapter, and a large language model.
Raw audio is first encoded by **MOSS-Audio-Encoder** into continuous temporal
representations at **12.5 Hz**, which are then projected into the language
model's embedding space through the adapter and finally consumed by the LLM
for auto-regressive text generation.

Rather than relying on off-the-shelf audio frontends, we train a dedicated
encoder from scratch to obtain more robust acoustic representations, tighter
temporal alignment, and better extensibility across musical styles, singing,
and non-speech acoustic content.

### DeepStack Cross-Layer Feature Injection

Using only the encoder's top-layer features tends to lose low-level prosody,
transient events, and local time-frequency structure. To address this, we
adopt a **DeepStack**-inspired cross-layer injection module between the
encoder and the language model: in addition to the encoder's final-layer
output, features from earlier and intermediate layers are selected,
independently projected, and injected into the language model's early layers,
preserving multi-granularity information from low-level acoustic details to
high-level semantic abstractions.

This design is especially well-suited for music understanding, as it helps
retain rhythm, timbre, transients, and instrumental texture — information
that a single high-level representation cannot fully capture, yet is critical
for chord recognition, structural analysis, and nuanced musical description.

### Time-Aware Representation

Time is a critical dimension in music understanding. To enhance explicit
temporal awareness, we adopt a **time-marker insertion** strategy during
pre-training: explicit time tokens are inserted between audio frame
representations at fixed time intervals to indicate temporal positions.
This design enables the model to learn "what happened when" within a unified
text generation framework, naturally supporting timestamped lyrics ASR,
beat / downbeat localisation, section boundary detection, and long-song
retrospective QA.

Building on the MOSS-Audio backbone, MOSS-Music is further enhanced through:

- **continual pre-training** on a large, diverse music corpus produced by
  the data annotation and processing pipeline
  [`MOSS-Music-Data-Pipeline`](https://github.com/wx9songs/MOSS-Music-Data-Pipeline),
  with an emphasis on singing, lyrics, and full-song coverage;
- **supervised fine-tuning (SFT)** on music-centric instruction data covering
  captioning, lyrics ASR, chord / key / structural analysis, and long-form
  musical QA;
- additional **reasoning tuning** for the *Thinking* variant.

## Released Models

| Model | Audio Encoder | LLM Backbone | Total Size | Hugging Face | ModelScope |
|---|---|---|---:|---|---|
| **MOSS‑Music‑8B‑Instruct** | MOSS-Audio-Encoder | Qwen3-8B | ~9.1B | [![Hugging Face](https://img.shields.io/badge/Huggingface-Model-orange?logo=huggingface)](https://huggingface.co/OpenMOSS-Team/MOSS-Music-8B-Instruct) | [![ModelScope](https://img.shields.io/badge/ModelScope-Model-624AFF)](https://modelscope.cn/models/openmoss/MOSS-Music-8B-Instruct) |
| **MOSS‑Music‑8B‑Thinking** | MOSS-Audio-Encoder | Qwen3-8B | ~9.1B | [![Hugging Face](https://img.shields.io/badge/Huggingface-Model-orange?logo=huggingface)](https://huggingface.co/OpenMOSS-Team/MOSS-Music-8B-Thinking) | [![ModelScope](https://img.shields.io/badge/ModelScope-Model-624AFF)](https://modelscope.cn/models/openmoss/MOSS-Music-8B-Thinking) |

> Smaller (4B) variants and additional sizes may follow. Stay tuned!

## Music Data Pipeline

The training data used by MOSS-Music is produced by an end-to-end pipeline
that goes from raw audio to chat-formatted training samples. That pipeline is
available at
[`MOSS-Music-Data-Pipeline`](https://github.com/wx9songs/MOSS-Music-Data-Pipeline),
which hosts duration detection, MIR feature extraction, song-structure
segmentation, lyrics ASR, metadata cleanup, and ALM-driven caption / query
generation with models such as Qwen3-Omni, MusicFlamingo, and other
audio-language models.

<p align="center">
  <img src="./assets/music_pipeline.png" width="94%" />
</p>

## Evaluation

We evaluate MOSS-Music on a diverse suite of public music understanding
benchmarks. Key results:

- **Music QA and understanding**: **MOSS-Music-8B-Instruct** achieves **80.38**
  average accuracy across **8 public music QA benchmarks** (excluding the
  three NSynth note-recognition tracks), ranking first among all compared
  models in our current evaluation set.
- **Music captioning**: In our preliminary **GPT-5.4-as-a-Judge** evaluation,
  the MOSS-Music series leads both caption benchmarks, with
  `MOSS-Music-8B-Thinking` reaching **4.53** on `MusicCaps` and
  `MOSS-Music-8B-Instruct` reaching **4.58** on `SDD`.
- **Lyrics ASR for singing voice**: **MOSS-Music-8B-Thinking** achieves the
  best average lyrics recognition error across `MUSDB18`, `MIR-1K` and
  `Opencpop` (**15.88%** avg WER/CER), clearly ahead of all compared
  audio-language baselines including `Gemini-3.1-Pro-Preview`,
  `MusicFlamingo` and `Qwen3-Omni`. Detailed timestamped-ASR results will be
  released in a later update.
- **Chord transcription**: MOSS-Music supports chord transcription, including
  timestamped chord transcription for harmonic analysis, accompaniment
  reference, and related downstream use cases. Detailed benchmark results will
  be released in a later update.


<p align="center">
  <img src="./assets/music_bench.png" width="98%" />
</p>

### Music QA & Understanding (Accuracy↑)

| Model | MMAU-music | MMAU-mini-music | MMAU-Pro-music | MMAR-music | MuChoMusic | Music-AVQA | NSynth (instrument) | NSynth (source) | NSynth (pitch) | GTZAN | Medley-Solos-DB | Avg |
|-----|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **MOSS‑Music‑8B‑Instruct** | **79.33** | **80.78** | 71.02 | 59.70 | **89.39** | **76.78** | **86.55** | 61.07 | **86.94** | **93.59** | 92.42 | **80.38** |
| Gemini‑3.1‑Pro | 71.69 | 77.18 | **73.06** | **71.64** | 79.53 | 61.51 | 13.38 | 38.90 | 6.47 | 86.39 | 80.34 | 75.17 |
| **MOSS‑Music‑8B‑Thinking** | 74.09 | 77.78 | 67.98 | 50.25 | 82.90 | 68.90 | 56.17 | 57.48 | 77.83 | 84.78 | 87.42 | 74.26 |
| MusicFlamingo | 76.83 | 76.35 | 65.60 | 48.66 | 74.58 | 73.60 | 80.76 | **75.89** | 0.00 | 84.45 | 90.86 | 73.87 |
| Audio‑Flamingo‑Next | 72.39 | 72.07 | 61.64 | 45.27 | 75.62 | 62.94 | 86.40 | 66.73 | 0.05 | 77.68 | 91.47 | 69.89 |
| MiMo‑Audio‑7B‑Instruct | 66.36 | 72.97 | 66.50 | 45.77 | 75.40 | 57.05 | 25.01 | 1.49 | 4.86 | 65.67 | **93.81** | 67.94 |
| Step‑Audio‑R1 | 66.46 | 75.08 | 62.34 | 50.75 | 72.62 | 57.98 | 13.75 | 15.87 | 2.39 | 73.67 | 82.45 | 67.67 |
| Qwen3‑Omni | 65.76 | 68.77 | 66.27 | 48.54 | 78.77 | 56.05 | 30.92 | 44.30 | 28.08 | 80.15 | 69.65 | 66.75 |
| Kimi‑Audio‑7B‑Instruct | 47.95 | 52.25 | 59.10 | 45.27 | 70.18 | 68.90 | 6.01 | 0.81 | 3.88 | 39.54 | 71.98 | 56.90 |

> `Avg` is computed over 8 public music QA benchmarks:
> `MMAU-music`, `MMAU-mini-music`, `MMAU-Pro-music`, `MMAR-music`,
> `MuChoMusic`, `Music-AVQA`, `GTZAN`, and `Medley-Solos-DB`.
>
> We exclude the three `NSynth` tracks from the main average because they focus
> on fine-grained isolated-note recognition, including instrument-family,
> acoustic/electronic source, and exact pitch discrimination from short
> single-note clips. Some compared audio-language models are not explicitly
> designed for this note-level classification setting, so we report NSynth
> separately for reference rather than mixing it into the headline average.

### Music Captioning

We further report a preliminary **GPT-5.4-as-a-Judge** music captioning
comparison on `MusicCaps` and `Song Describer Dataset (SDD)`. Scores are on a
1-5 scale across 9 dimensions: `genre/style`, `mood/affect`, `tempo/rhythm`,
`instrumentation/timbre`, `vocals`, `melody/harmony`, `structure/form`,
`production/audio quality`, and `scene/use case`.

- **Overall caption quality**: the MOSS-Music series remains strongest across
  both caption benchmarks, with `MOSS-Music-8B-Thinking` reaching **4.53** on
  `MusicCaps` and `MOSS-Music-8B-Instruct` reaching **4.58** on `SDD`.
- **Stronger structural descriptions**: MOSS-Music shows the clearest gains on
  `structure / form / progression`, especially on `SDD`.
- **Competitive baselines on instrumentation and scene semantics**:
  `MusicFlamingo` and `Gemini-3.1-Pro` remain competitive on
  `instrumentation/timbre`, while `Gemini-3.1-Pro` is strongest on
  `scene / use case`.

#### MusicCaps

| Model | Genre | Mood | Tempo | Instr. | Vocals | Melody/Harmony | Structure | Production | Scene | Avg |
|-----|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **MOSS‑Music‑8B‑Thinking** | 4.78 | **4.69** | **4.62** | 4.40 | **4.46** | **4.40** | **4.86** | 4.35 | 4.18 | **4.53** |
| Gemini‑3.1‑Pro | 4.70 | 4.60 | 4.48 | **4.68** | 4.18 | 4.18 | 3.86 | **4.40** | **4.72** | 4.42 |
| **MOSS‑Music‑8B‑Instruct** | 4.60 | 4.52 | 4.46 | 4.02 | 4.30 | 4.38 | 4.78 | 4.20 | 3.96 | 4.36 |
| MusicFlamingo | **4.80** | 4.36 | 4.50 | 4.64 | 3.94 | 4.08 | 3.58 | 4.30 | 3.72 | 4.21 |
| Audio‑Flamingo‑Next | 4.34 | 4.56 | 4.08 | 4.30 | 4.18 | 3.78 | 3.66 | 4.04 | 3.92 | 4.10 |
| MiMo‑Audio‑7B‑Instruct | 4.02 | 4.20 | 4.46 | 4.28 | 4.36 | 3.62 | 3.30 | 4.08 | 3.50 | 3.98 |
| Step‑Audio‑R1 | 4.22 | 4.02 | 4.20 | 3.96 | 3.84 | 4.02 | 3.24 | 4.10 | 3.54 | 3.90 |
| Qwen3‑Omni | 4.58 | 4.50 | 4.26 | 3.62 | 3.64 | 3.48 | 2.98 | 4.18 | 4.42 | 3.96 |
| Kimi‑Audio‑7B‑Instruct | 3.98 | 3.92 | 4.32 | 3.88 | 4.48 | 3.28 | 2.72 | 3.72 | 3.24 | 3.73 |

#### Song Describer Dataset (SDD)

| Model | Genre | Mood | Tempo | Instr. | Vocals | Melody/Harmony | Structure | Production | Scene | Avg |
|-----|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **MOSS‑Music‑8B‑Instruct** | **4.84** | **4.76** | **4.68** | 4.24 | **4.52** | **4.56** | **4.92** | 4.42 | 4.24 | **4.58** |
| Gemini‑3.1‑Pro | 4.72 | 4.64 | 4.52 | **4.72** | 4.22 | 4.24 | 3.94 | **4.46** | **4.82** | 4.48 |
| **MOSS‑Music‑8B‑Thinking** | 4.66 | 4.58 | 4.50 | 4.36 | 4.36 | 4.44 | 4.84 | 4.26 | 4.02 | 4.45 |
| MusicFlamingo | 4.82 | 4.40 | 4.52 | 4.70 | 3.98 | 4.14 | 3.66 | 4.36 | 3.80 | 4.26 |
| Audio‑Flamingo‑Next | 4.40 | 4.62 | 4.14 | 4.36 | 4.22 | 3.84 | 3.74 | 4.10 | 4.00 | 4.16 |
| MiMo‑Audio‑7B‑Instruct | 4.08 | 4.26 | 4.52 | 4.34 | 4.42 | 3.70 | 3.38 | 4.16 | 3.58 | 4.05 |
| Step‑Audio‑R1 | 4.30 | 4.10 | 4.26 | 4.02 | 3.92 | 4.10 | 3.32 | 4.18 | 3.62 | 3.98 |
| Qwen3‑Omni | 4.62 | 4.54 | 4.30 | 3.68 | 3.70 | 3.56 | 3.06 | 4.24 | 4.50 | 4.02 |
| Kimi‑Audio‑7B‑Instruct | 4.04 | 3.98 | 4.38 | 3.96 | 4.54 | 3.36 | 2.80 | 3.80 | 3.32 | 3.80 |

### Lyrics ASR (WER / CER↓)

We further evaluate MOSS-Music on **singing-voice lyrics ASR** across three
representative benchmarks:

- `MUSDB18` — English pop songs **with backing tracks**, scored with **WER**;
- `MIR-1K` — **Chinese karaoke** clips with background music, scored with **CER**;
- `Opencpop` — **clean Mandarin studio singing**, scored with **CER**.

`Avg` is the unweighted mean of the three dataset-level error rates.

| Model | MUSDB18 WER | MIR-1K CER | Opencpop CER | Avg |
|-----|---:|---:|---:|---:|
| **MOSS‑Music‑8B‑Thinking** | 29.19% | **15.84%** | 2.60% | **15.88%** |
| **MOSS‑Music‑8B‑Instruct** | 32.99% | 23.96% | 4.62% | 20.52% |
| Gemini‑3.1‑Pro‑Preview | 26.25% | 36.37% | 6.00% | 22.87% |
| MusicFlamingo | **23.41%** | 38.98% | 18.73% | 27.04% |
| Qwen3‑Omni‑30B‑A3B‑Instruct | 62.67% | 20.48% | **2.26%** | 28.47% |
| MiMo‑Audio‑7B‑Instruct | 94.16% | 23.34% | 6.77% | 41.42% |
| Kimi‑Audio‑7B‑Instruct | 97.53% | 25.83% | 4.90% | 42.75% |
| Step‑Audio‑R1 | 81.67% | 48.03% | 4.15% | 44.62% |
| Audio‑Flamingo‑Next | 94.93% | 55.63% | 12.47% | 54.34% |

> **MOSS-Music-8B-Thinking** achieves the lowest average lyrics-ASR error
> (**15.88%**) across these three datasets, with particular gains on
> `MIR-1K` (Chinese karaoke with accompaniment) and `Opencpop` (clean Mandarin
> singing). MOSS-Music also inherits the strong timestamp-aware ASR ability
> from MOSS-Audio; detailed singing-timestamp ASR results will be added soon.

### Chord Transcription

MOSS-Music supports chord transcription, including timestamped chord
transcription that tracks chord progression over time. This can be useful for
harmonic analysis, accompaniment reference, music education, and related use
cases. Detailed benchmark results will be added soon.

## Quickstart

### Environment Setup

We recommend Python 3.12 with a clean Conda environment. The commands below
are enough for local inference.

#### Recommended setup

```bash
git clone https://github.com/OpenMOSS/MOSS-Music.git
cd MOSS-Music

conda create -n moss-music python=3.12 -y
conda activate moss-music

conda install -c conda-forge "ffmpeg=7" -y
pip install --extra-index-url https://download.pytorch.org/whl/cu128 -e ".[torch-runtime]"
```

#### Optional: FlashAttention 2

If your GPU supports FlashAttention 2, you can replace the last install
command with:

```bash
pip install --extra-index-url https://download.pytorch.org/whl/cu128 -e ".[torch-runtime,flash-attn]"
```

### SGLang Serving

> [!IMPORTANT]
> To achieve the best generation quality and fully leverage the model's capabilities, we
> **strongly recommend using SGLang Serving for inference**.

See the full SGLang guide in `moss_music_usage_guide.md`.

Download the model first:

```bash
hf download OpenMOSS-Team/MOSS-Music-8B-Instruct --local-dir ./weights/MOSS-Music-8B-Instruct
hf download OpenMOSS-Team/MOSS-Music-8B-Thinking --local-dir ./weights/MOSS-Music-8B-Thinking
```

The shortest setup is:

```bash
cd sglang
pip install -e "python[all]"
pip install nvidia-cudnn-cu12==9.16.0.29
cd ..

sglang serve \
  --model-path ./weights/MOSS-Music-8B-Instruct \
  --trust-remote-code
```

You can replace `./weights/MOSS-Music-8B-Instruct` with
`./weights/MOSS-Music-8B-Thinking` if needed.

If you use the default `torch==2.9.1+cu128` runtime, installing
`nvidia-cudnn-cu12==9.16.0.29` is recommended before starting `sglang serve`.

### Gradio App

Start the Gradio demo with:

```bash
python app.py
```

The server address and port can be overridden via the
`MOSS_MUSIC_SERVER_NAME` and `MOSS_MUSIC_SERVER_PORT` environment variables,
and the default model ID via `MOSS_MUSIC_MODEL_ID`.

## More Information

- **MOSI.AI**: [https://mosi.cn](https://mosi.cn)
- **OpenMOSS**: [https://www.open-moss.com](https://www.open-moss.com)
- **MOSS-Audio (backbone)**: [https://github.com/OpenMOSS/MOSS-Audio](https://github.com/OpenMOSS/MOSS-Audio)
- **MOSS-Music Data Pipeline**: [https://github.com/wx9songs/MOSS-Music-Data-Pipeline](https://github.com/wx9songs/MOSS-Music-Data-Pipeline)

## LICENSE

Models in MOSS-Music are licensed under the Apache License 2.0.

## Citation

```bibtex
@misc{mossmusic2026,
      title={MOSS-Music Technical Report},
      author={OpenMOSS Team},
      year={2026},
      howpublished={\url{https://github.com/OpenMOSS/MOSS-Music}},
      note={GitHub repository}
}
```
