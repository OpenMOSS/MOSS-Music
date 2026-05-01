[English](README.md) | [简体中文](README_zh.md)

本仓库提供 **MOSS-TTS Family** 的 SGLang 支持，当前覆盖以下模型：

- **MOSS-TTS（Delay）**
- **MOSS-SoundEffect**
- **MOSS-TTSD v1.0**
- **MOSS-TTSD v0.7**

> 说明：当前仓库**不包含**部分 `fuse` / `request` / `inference` 脚本。
> 可以直接使用文中的外部脚本链接运行，或单独下载后使用。

## 目录

- [MOSS-TTS（Delay）/ MOSS-SoundEffect](#moss-tts-delay-soundeffect)
- [MOSS-TTSD v1.0](#moss-ttsd-v10)
- [MOSS-TTSD v0.7](#moss-ttsd-v07)

<a id="moss-tts-delay-soundeffect"></a>

## MOSS-TTS（Delay）/ MOSS-SoundEffect

来源：[MOSS-TTS README_zh](https://github.com/OpenMOSS/MOSS-TTS/blob/main/README_zh.md)

MOSS-TTS（Delay）支持使用 OpenMOSS 深度扩展的 [SGLang](https://github.com/OpenMOSS/sglang) 运行融合后的 MOSS-TTS 与 MOSS-Audio-Tokenizer 模型，实现面向音频生成的 **高效推理**。

**单并发端到端吞吐（在 RTX 4090 上测得）：** 45 token/s

### 1) 安装 SGLang

```bash
# 1. 克隆 SGLang 仓库
git clone https://github.com/OpenMOSS/sglang.git

# 2. 安装 SGLang
pip install -e ./sglang/python[all]

# 3. (可选) 解决 SGLang 的 CuDNN 兼容性报错
#    RuntimeError: CRITICAL WARNING: PyTorch 2.9.1 & CuDNN Compatibility Issue Detected
pip install nvidia-cudnn-cu12==9.16.0.29
```

### 2) 下载模型与 tokenizer

```bash
huggingface-cli download OpenMOSS-Team/MOSS-TTS --local-dir weights/MOSS-TTS
huggingface-cli download OpenMOSS-Team/MOSS-Audio-Tokenizer --local-dir weights/MOSS-Audio-Tokenizer
```

### 3) 融合模型

脚本：[`scripts/fuse_moss_tts_delay_with_codec.py`](https://github.com/OpenMOSS/MOSS-TTS/blob/main/scripts/fuse_moss_tts_delay_with_codec.py)

```bash
python scripts/fuse_moss_tts_delay_with_codec.py \
  --model-path weights/MOSS-TTS \
  --codec-model-path weights/MOSS-Audio-Tokenizer \
  --save-path weights/MOSS-TTS-Delay-With-Codec
```

> 如果融合输出目录已存在，可以在命令中追加 `--overwrite` 直接覆盖，或在脚本提示后输入字符确认覆盖。

### 4) 启动服务

```bash
sglang serve \
  --model-path weights/MOSS-TTS-Delay-With-Codec \
  --delay-pattern \
  --trust-remote-code
```

> **注意：** 首次启动服务后的第一次请求会触发较长时间的编译，这不是故障，请耐心等待。

### 5) MOSS-TTS（Delay）请求

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

- `text` 表示待合成的文本内容；可在前缀加入 `${token:25}` 进行 token 控制，例如 `${token:25}你好 世界`
- `audio_data` 表示可选的参考音频；不传入时会生成随机音色的音频，也可以是 `<path-to-audio-file>` 或 `data:audio/wav;base64,{b64_audio}`，其中 `b64_audio` 为 wav 文件的 base64 字符串。

### 6) MOSS-SoundEffect 请求

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

- `text` 中只能包含 `${token:125}` 和 `${ambient_sound:...}` 这两个字段，其中 `${ambient_sound:...}` 后填写音效的文字描述。
- 对于 MOSS-SoundEffect，建议使用 `${token:125}`，生成会更稳定。
- 不要传 `audio_data`，否则模型可能会 OOD。

### 7) 返回格式

```json
{"text": "<wav-base64>", "...": "..."}
```

HTTP 响应为 JSON 对象，可能包含多个字段；其中 `.text` 字段存放生成音频的 wav base64 字符串。通常只需提取该字段并做 base64 解码；例如将响应保存为 `response.json` 后，可执行：

```bash
jq -r '.text' response.json | base64 -d -i > output.wav
```

---

<a id="moss-ttsd-v10"></a>

## MOSS-TTSD v1.0

来源：[MOSS-TTSD README_zh](https://github.com/OpenMOSS/MOSS-TTSD/blob/main/README_zh.md)

MOSS-TTSD v1.0 支持使用 OpenMOSS 深度扩展的 [SGLang](https://github.com/OpenMOSS/sglang) 运行融合后的 MOSS-TTSD 与 MOSS-Audio-Tokenizer 模型，实现面向音频生成的**高效推理**。

**单并发端到端吞吐（在 RTX 4090 上测得）：** 43.5 token/s

### 1) 获取对应 SGLang 分支

```bash
git clone https://github.com/OpenMOSS/sglang -b moss-ttsd-v1.0-with-cat
```

### 2) 创建环境并安装依赖

#### 使用 venv

```bash
python -m venv moss_ttsd_sglang
source moss_ttsd_sglang/bin/activate
pip install ./sglang/python[all]
```

#### 使用 conda

```bash
conda create -n moss_ttsd_sglang python=3.12
conda activate moss_ttsd_sglang
pip install ./sglang/python[all]
```

### 3) 下载模型与 audio tokenizer

```bash
git clone https://huggingface.co/OpenMOSS-Team/MOSS-TTSD-v1.0
git clone https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer
```

或者：

```bash
hf download OpenMOSS-Team/MOSS-TTSD-v1.0 --local-dir ./MOSS-TTSD-v1.0
hf download OpenMOSS-Team/MOSS-Audio-Tokenizer --local-dir ./MOSS-Audio-Tokenizer
```

### 4) 融合模型

下载完成后，执行以下命令，使用 [`scripts/fuse_moss_tts_delay_with_codec.py`](https://github.com/OpenMOSS/MOSS-TTSD/blob/main/scripts/fuse_moss_tts_delay_with_codec.py) 将 MOSS-TTSD v1.0 与 MOSS-Audio-Tokenizer 融合为可供 SGLang 服务加载的单目录模型，融合后默认使用 `voice_clone_and_continuation` 推理模式：

```bash
python scripts/fuse_moss_tts_delay_with_codec.py \
  --model-path <path-to-moss-ttsd-v1.0> \
  --codec-model-path <path-to-moss-audio-tokenizer> \
  --save-path <path-to-fused-model>
```

### 5) 启动服务

```bash
sglang serve \
  --model-path <path-to-fused-model> \
  --delay-pattern \
  --trust-remote-code \
  --port 30000 --host 0.0.0.0
```

> 首次启动服务时可能因编译而耗时较长；当看到 `The server is fired up and ready to roll!` 时，即表示服务已经就绪。服务启动后的第一次请求仍可能触发较长时间的编译，这属于正常现象，请耐心等待。

> **提示：** 端到端推理服务在运行时可能会有一定的显存碎片占用。如果显存较紧张，建议在启动 SGLang 时通过 `--mem-fraction-static` 控制静态显存分配比例，为中间张量预留空间。

### 6) 发送生成请求

当前仓库提供了一个最小请求示例脚本：[`scripts/request_sglang_generation.py`](https://github.com/OpenMOSS/MOSS-TTSD/blob/main/scripts/request_sglang_generation.py)

```bash
python scripts/request_sglang_generation.py
```

这个脚本会：

- 默认向 `http://localhost:30000/generate` 发送请求
- 使用仓库内 `asset/reference_02_s1.wav` 和 `asset/reference_02_s2.wav` 作为参考音频
- 将返回的音频保存到 `outputs/output.wav`

如果你需要替换参考音频、输入文本、采样参数或服务地址，可以直接修改 `scripts/request_sglang_generation.py` 中的对应常量。

---

<a id="moss-ttsd-v07"></a>

## MOSS-TTSD v0.7

来源：[MOSS-TTSD v0.7 README_zh](https://github.com/OpenMOSS/MOSS-TTSD/blob/main/legacy/v0.7/README_zh.md)

**单并发端到端吞吐（在 RTX 4090 上测得）：** 140 token/s

### 1) 获取对应 SGLang 分支

```bash
git clone https://github.com/OpenMOSS/sglang -b moss-ttsd-v0.7-with-xy
```

### 2) 创建环境并安装依赖

#### 使用 venv

```bash
python -m venv moss_ttsd_sglang
source moss_ttsd_sglang/bin/activate
pip install ./sglang/python[all]
```

#### 使用 conda

```bash
conda create -n moss_ttsd_sglang python=3.12
conda activate moss_ttsd_sglang
pip install ./sglang/python[all]
```

### 3) 下载模型与 XY-Tokenizer

```bash
git clone https://huggingface.co/OpenMOSS-Team/MOSS-TTSD-v0.7
git clone https://huggingface.co/OpenMOSS-Team/MOSS_TTSD_Tokenizer_hf
```

或者：

```bash
hf download OpenMOSS-Team/MOSS-TTSD-v0.7 --local-dir ./MOSS-TTSD-v0.7
hf download OpenMOSS-Team/MOSS_TTSD_Tokenizer_hf --local-dir ./MOSS_TTSD_Tokenizer_hf
```

### 4) 融合模型

下载完成后，执行以下命令，使用 [`legacy/v0.7/fuse_model_with_codec.py`](https://github.com/OpenMOSS/MOSS-TTSD/blob/main/legacy/v0.7/fuse_model_with_codec.py) 整合 MOSS-TTSD 和 XY-Tokenizer 的权重：

```bash
python fuse_model_with_codec.py \
  --model-path <path-to-moss-ttsd> \
  --codec-path <path-to-xy-tokenizer> \
  --output-dir <path-to-save-model>
```

### 5) 启动服务

```bash
SGLANG_VLM_CACHE_SIZE_MB=0 \
sglang serve \
  --model-path <path-to-save-model> \
  --delay-pattern \
  --trust-remote-code \
  --disable-radix-cache \
  --port 30000 --host 0.0.0.0
```

首次启动可能因编译耗时较长。看到 `The server is fired up and ready to roll!` 即表示服务器已就绪。

提示：我们的端到端推理服务器会存在一些碎片化的显存（VRAM）占用。如果您使用的 GPU 显存有限，在启动服务器时，请通过 `--mem-fraction-static` 参数设置 SGLang 的显存分配比例，以确保为中间变量预留足够的显存。

### 6) 运行推理

推理服务的接口是标准的多模态模型文本生成接口，返回的文本字段是音频文件（wav格式）的 base64 编码。

我们提供了一个示例脚本，用于向服务器发送生成请求；你可以使用它进行推理：[`legacy/v0.7/inference_sglang_server.py`](https://github.com/OpenMOSS/MOSS-TTSD/blob/main/legacy/v0.7/inference_sglang_server.py)

```bash
python inference_sglang_server.py --host localhost --port 30000 --jsonl examples/examples.jsonl --output_dir outputs --use_normalize
```

或者：

```bash
python inference_sglang_server.py --url http://localhost:30000 --jsonl examples/examples.jsonl --output_dir outputs --use_normalize
```

参数说明：

- `--url`：服务器基础 URL（例如 `http://localhost:30000`）。设置该项后将忽略 `--host` 和 `--port`。
- `--host`：服务器主机名。
- `--port`：服务器端口。
- `--jsonl`：输入 JSONL 文件路径，包含对话脚本和参考音频。
- `--output_dir`：生成音频的保存目录。脚本会将文件保存为 `output_<idx>.wav`。
- `--use_normalize`：是否启用文本归一化（**建议开启**）。
- `--max_new_tokens`：模型将生成的 token 数量上限。

此外，还可以在 `inference_sglang_server.py` 文件中修改和设置具体的采样参数。
