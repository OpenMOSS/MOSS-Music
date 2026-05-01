[English](README.md) | [简体中文](README_zh.md)

This repository provides SGLang support for the **MOSS-TTS Family**, covering the following models:

- **MOSS-TTS (Delay)**
- **MOSS-SoundEffect**
- **MOSS-TTSD v1.0**
- **MOSS-TTSD v0.7**

> Note: This repository does **not** include some `fuse` / `request` / `inference` scripts.
> You can use the external script links in this document directly, or download those scripts separately before running them.

## Contents

- [MOSS-TTS (Delay) / MOSS-SoundEffect](#moss-tts-delay-soundeffect)
- [MOSS-TTSD v1.0](#moss-ttsd-v10)
- [MOSS-TTSD v0.7](#moss-ttsd-v07)

<a id="moss-tts-delay-soundeffect"></a>

## MOSS-TTS (Delay) / MOSS-SoundEffect

Source: [MOSS-TTS README](https://github.com/OpenMOSS/MOSS-TTS/blob/main/README.md)

MOSS-TTS (Delay) supports running the fused MOSS-TTS and MOSS-Audio-Tokenizer model with the deeply extended [SGLang](https://github.com/OpenMOSS/sglang) from OpenMOSS, enabling efficient inference for audio generation.

**Single-concurrency end-to-end throughput (measured on RTX 4090):** 45 token/s

### 1) Install SGLang

```bash
# 1. Clone the SGLang repository
git clone https://github.com/OpenMOSS/sglang.git

# 2. Install SGLang
pip install -e ./sglang/python[all]

# 3. (Optional) Fix the SGLang CuDNN compatibility error
#    RuntimeError: CRITICAL WARNING: PyTorch 2.9.1 & CuDNN Compatibility Issue Detected
pip install nvidia-cudnn-cu12==9.16.0.29
```

### 2) Download the model and tokenizer

```bash
huggingface-cli download OpenMOSS-Team/MOSS-TTS --local-dir weights/MOSS-TTS
huggingface-cli download OpenMOSS-Team/MOSS-Audio-Tokenizer --local-dir weights/MOSS-Audio-Tokenizer
```

### 3) Fuse the model

Script: [`scripts/fuse_moss_tts_delay_with_codec.py`](https://github.com/OpenMOSS/MOSS-TTS/blob/main/scripts/fuse_moss_tts_delay_with_codec.py)

```bash
python scripts/fuse_moss_tts_delay_with_codec.py \
  --model-path weights/MOSS-TTS \
  --codec-model-path weights/MOSS-Audio-Tokenizer \
  --save-path weights/MOSS-TTS-Delay-With-Codec
```

> If the fused output directory already exists, you can append `--overwrite` to replace it directly, or confirm the overwrite interactively when prompted.

### 4) Start the service

```bash
sglang serve \
  --model-path weights/MOSS-TTS-Delay-With-Codec \
  --delay-pattern \
  --trust-remote-code
```

> **Note:** The first request after starting the service for the first time may trigger a lengthy compilation step. This is expected, not a bug, so please wait patiently.

### 5) MOSS-TTS (Delay) request

```bash
curl -X POST http://localhost:30000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Added SGLang backend support for efficient inference.",
    "audio_data": "https://cdn.jsdelivr.net/gh/OpenMOSS/MOSS-TTSD@main/legacy/v0.7/examples/zh_spk1_moon.wav",
    "sampling_params": {
      "max_new_tokens": 512,
      "temperature": 1.7,
      "top_p": 0.8,
      "top_k": 25
    }
  }'
```

- `text` denotes the text content to be synthesized; you can prepend `${token:25}` for token control, for example `${token:25}Hello World`
- `audio_data` denotes the optional reference audio; if omitted, the model generates audio with a random timbre, and it can be either `<path-to-audio-file>` or `data:audio/wav;base64,{b64_audio}`, where `b64_audio` is the base64 string of a wav file.

### 6) MOSS-SoundEffect request

```bash
curl -X POST http://localhost:30000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "${token:125}${ambient_sound:a sports car roaring past on the highway.}",
    "sampling_params": {
      "max_new_tokens": 512,
      "temperature": 1.5,
      "top_p": 0.6,
      "top_k": 50
    }
  }'
```

- `text` should contain only two tagged fields: `${token:125}` and `${ambient_sound:...}`, where the content after `${ambient_sound:...}` is a natural-language description of the target sound effect.
- `${token:125}` is recommended for more stable generation.
- Do not pass `audio_data`, or the model may go OOD.

### 7) Response format

```json
{"text": "<wav-base64>", "...": "..."}
```

The HTTP response is a JSON object and may contain multiple fields. The `.text` field stores the WAV base64 string for the generated audio. In most cases, you only need to extract that field and base64-decode it; for example, after saving the response as `response.json`, you can run:

```bash
jq -r '.text' response.json | base64 -d -i > output.wav
```

---

<a id="moss-ttsd-v10"></a>

## MOSS-TTSD v1.0

Source: [MOSS-TTSD README](https://github.com/OpenMOSS/MOSS-TTSD/blob/main/README.md)

MOSS-TTSD v1.0 supports running the fused MOSS-TTSD and MOSS-Audio-Tokenizer model with the deeply extended [SGLang](https://github.com/OpenMOSS/sglang) from OpenMOSS, enabling efficient inference for audio generation.

**Single-concurrency end-to-end throughput (measured on RTX 4090):** 43.5 token/s

### 1) Get the corresponding SGLang branch

```bash
git clone https://github.com/OpenMOSS/sglang -b moss-ttsd-v1.0-with-cat
```

### 2) Create the environment and install dependencies

#### Using venv

```bash
python -m venv moss_ttsd_sglang
source moss_ttsd_sglang/bin/activate
pip install ./sglang/python[all]
```

#### Using conda

```bash
conda create -n moss_ttsd_sglang python=3.12
conda activate moss_ttsd_sglang
pip install ./sglang/python[all]
```

### 3) Download the model and audio tokenizer

```bash
git clone https://huggingface.co/OpenMOSS-Team/MOSS-TTSD-v1.0
git clone https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer
```

Or:

```bash
hf download OpenMOSS-Team/MOSS-TTSD-v1.0 --local-dir ./MOSS-TTSD-v1.0
hf download OpenMOSS-Team/MOSS-Audio-Tokenizer --local-dir ./MOSS-Audio-Tokenizer
```

### 4) Fuse the model

After the download is complete, run the following command using [`scripts/fuse_moss_tts_delay_with_codec.py`](https://github.com/OpenMOSS/MOSS-TTSD/blob/main/scripts/fuse_moss_tts_delay_with_codec.py) to fuse MOSS-TTSD v1.0 and MOSS-Audio-Tokenizer into a single-directory model that can be loaded by SGLang. After fusion, the model uses `voice_clone_and_continuation` inference mode by default:

```bash
python scripts/fuse_moss_tts_delay_with_codec.py \
  --model-path <path-to-moss-ttsd-v1.0> \
  --codec-model-path <path-to-moss-audio-tokenizer> \
  --save-path <path-to-fused-model>
```

### 5) Start the service

```bash
sglang serve \
  --model-path <path-to-fused-model> \
  --delay-pattern \
  --trust-remote-code \
  --port 30000 --host 0.0.0.0
```

> The first service startup may take longer due to compilation. Once you see `The server is fired up and ready to roll!`, the service is ready. The first request after startup may still trigger a lengthy compilation, which is expected behavior, so please be patient.

> **Tip:** The end-to-end inference service may cause some VRAM fragmentation during runtime. If GPU memory is tight, we recommend using `--mem-fraction-static` when starting SGLang to reserve enough space for intermediate tensors.

### 6) Send a generation request

The repository currently provides a minimal request example script: [`scripts/request_sglang_generation.py`](https://github.com/OpenMOSS/MOSS-TTSD/blob/main/scripts/request_sglang_generation.py)

```bash
python scripts/request_sglang_generation.py
```

This script will:

- send requests to `http://localhost:30000/generate` by default
- use `asset/reference_02_s1.wav` and `asset/reference_02_s2.wav` in the repository as reference audio
- save the returned audio to `outputs/output.wav`

If you need to change the reference audio, input text, sampling parameters, or server URL, you can directly edit the corresponding constants in `scripts/request_sglang_generation.py`.

---

<a id="moss-ttsd-v07"></a>

## MOSS-TTSD v0.7

Source: [MOSS-TTSD v0.7 README](https://github.com/OpenMOSS/MOSS-TTSD/blob/main/legacy/v0.7/README.md)

**Single-concurrency end-to-end throughput (measured on RTX 4090):** 140 token/s

### 1) Get the corresponding SGLang branch

```bash
git clone https://github.com/OpenMOSS/sglang -b moss-ttsd-v0.7-with-xy
```

### 2) Create the environment and install dependencies

#### Using venv

```bash
python -m venv moss_ttsd_sglang
source moss_ttsd_sglang/bin/activate
pip install ./sglang/python[all]
```

#### Using conda

```bash
conda create -n moss_ttsd_sglang python=3.12
conda activate moss_ttsd_sglang
pip install ./sglang/python[all]
```

### 3) Download the model and XY-Tokenizer

```bash
git clone https://huggingface.co/OpenMOSS-Team/MOSS-TTSD-v0.7
git clone https://huggingface.co/OpenMOSS-Team/MOSS_TTSD_Tokenizer_hf
```

Or:

```bash
hf download OpenMOSS-Team/MOSS-TTSD-v0.7 --local-dir ./MOSS-TTSD-v0.7
hf download OpenMOSS-Team/MOSS_TTSD_Tokenizer_hf --local-dir ./MOSS_TTSD_Tokenizer_hf
```

### 4) Fuse the model

After the download is complete, fuse the MOSS-TTSD and XY-Tokenizer weights using [`legacy/v0.7/fuse_model_with_codec.py`](https://github.com/OpenMOSS/MOSS-TTSD/blob/main/legacy/v0.7/fuse_model_with_codec.py):

```bash
python fuse_model_with_codec.py \
  --model-path <path-to-moss-ttsd> \
  --codec-path <path-to-xy-tokenizer> \
  --output-dir <path-to-save-model>
```

### 5) Start the service

```bash
SGLANG_VLM_CACHE_SIZE_MB=0 \
sglang serve \
  --model-path <path-to-save-model> \
  --delay-pattern \
  --trust-remote-code \
  --disable-radix-cache \
  --port 30000 --host 0.0.0.0
```

The first startup may take longer due to compilation. Once you see `The server is fired up and ready to roll!` the server is ready.

Tips: Our end-to-end inference server may have some fragmented VRAM usage. If your GPU has limited VRAM, set SGLang's VRAM allocation ratio with the `--mem-fraction-static` flag when starting the server to reserve enough memory for intermediate tensors.

### 6) Run inference

The service API is a standard multimodal text-generation API; the returned text field is a base64-encoded audio file (WAV).

We provide an example script that sends generation requests to the server: [`legacy/v0.7/inference_sglang_server.py`](https://github.com/OpenMOSS/MOSS-TTSD/blob/main/legacy/v0.7/inference_sglang_server.py)

```bash
python inference_sglang_server.py --host localhost --port 30000 --jsonl examples/examples.jsonl --output_dir outputs --use_normalize
```

Or:

```bash
python inference_sglang_server.py --url http://localhost:30000 --jsonl examples/examples.jsonl --output_dir outputs --use_normalize
```

Parameters:

- `--url`: Base server URL (e.g., `http://localhost:30000`). When set, `--host` and `--port` are ignored.
- `--host`: Server host.
- `--port`: Server port.
- `--jsonl`: Path to the input JSONL file containing dialogue scripts and speaker prompts.
- `--output_dir`: Directory where the generated audio files will be saved. The script saves files as `output_<idx>.wav`.
- `--use_normalize`: Whether to normalize the text input (**recommended to enable**).
- `--max_new_tokens`: The maximum number of tokens the model will generate.

Additionally, you can modify and set specific sampling parameters in the `inference_sglang_server.py` file.
