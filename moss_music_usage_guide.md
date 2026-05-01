# MOSS-Music SGLang Usage Guide

Quick Jump: [English](#english-version) | [中文](#chinese-version)

---

<a id="english-version"></a>

## English Version

### Installation

If your current repository already includes a compatible `sglang` checkout
(for example, the `moss-audio` branch is already present under `./sglang`),
you do not need to clone it again. You only need to install it in the current
environment if it has not been installed yet.

Otherwise:

```bash
git clone -b moss-audio https://github.com/OpenMOSS/sglang.git
cd sglang
pip install -e "python[all]"
pip install nvidia-cudnn-cu12==9.16.0.29
```

If `./sglang` already exists in this repository, you can instead run:

```bash
cd sglang
pip install -e "python[all]"
pip install nvidia-cudnn-cu12==9.16.0.29
```

If you have already downloaded the model locally, such as
`./weights/MOSS-Music-8B-Instruct` or `./weights/MOSS-Music-8B-Thinking`, you
can directly pass that local path to `--model-path`.

### Notes

All MOSS-Music model weights already include a multimodal chat template
(`chat_template.jinja`), so you do not need to provide an extra template file.
Both `/generate` and `/v1/chat/completions` can be used directly.

All commands below assume you are already running inside an environment where
`sglang` has been installed.

If you are using `torch==2.9.1+cu128`, it is recommended to install
`nvidia-cudnn-cu12==9.16.0.29` first. Otherwise, `sglang` may refuse to start
because of a known CuDNN compatibility check.

MOSS-Music checkpoints expose `MossMusicModel` / `MossMusicProcessor`, and the
remote code maps them onto SGLang's built-in MOSS-Audio multimodal path at
load time. So you can directly use the `moss-audio` branch without patching
your local SGLang checkout.

### Launch Modes

#### Mode 1: Basic Service

Use this mode for music understanding, lyrics transcription, and text chat via
`/generate` and `/v1/chat/completions`.

```bash
sglang serve \
  --model-path ./weights/MOSS-Music-8B-Instruct \
  --trust-remote-code
```

#### Mode 2: Separate Reasoning

Based on Mode 1, this mode automatically splits `<think>...</think>` from the
main response into the `reasoning_content` field.

```bash
sglang serve \
  --model-path ./weights/MOSS-Music-8B-Thinking \
  --trust-remote-code \
  --reasoning-parser qwen3
```

#### Mode 3: Separate Reasoning + Thinking Budget Control (Recommended)

Based on Mode 2, this mode adds thinking budget control using the instruction
injection approach described in the Qwen3 technical report.

```bash
sglang serve \
  --model-path ./weights/MOSS-Music-8B-Thinking \
  --trust-remote-code \
  --reasoning-parser qwen3-instruction-injection \
  --enable-custom-logit-processor
```

### Launch Arguments

| Argument | Description |
|---|---|
| `--reasoning-parser qwen3` | Split `<think>...</think>` using the Qwen3 format |
| `--reasoning-parser qwen3-instruction-injection` | Same as above, but also strips the transition sentence injected by thinking budget control |
| `--enable-custom-logit-processor` | Allows requests to pass a custom logit processor, required for thinking budget control |

### Request Patterns

#### 1. Native `/generate` (Available in all modes)

##### Basic music description

```bash
curl -X POST http://localhost:30000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Please give a detailed musical description of this clip.",
    "audio_data": "/path/to/music.wav",
    "sampling_params": {
      "max_new_tokens": 1024,
      "temperature": 0.0
    }
  }'
```

##### Lyrics transcription

```bash
curl -X POST http://localhost:30000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Transcribe the lyrics of this song.",
    "audio_data": "/path/to/song.wav",
    "sampling_params": {
      "max_new_tokens": 1024,
      "temperature": 0.0
    }
  }'
```

##### `/generate` + post-processing reasoning split

Generate first, then split with `/separate_reasoning`:

```bash
curl -X POST http://localhost:30000/separate_reasoning \
  -H "Content-Type: application/json" \
  -d '{
    "text": "<think>\nreasoning content\n</think>\n\nfinal answer content",
    "reasoning_parser": "qwen3"
  }'
```

#### 2. OpenAI Chat `/v1/chat/completions` (Available in all modes)

##### Music description + separated reasoning

```bash
curl -X POST http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "audio_url",
            "audio_url": {
              "url": "/path/to/music.wav"
            }
          },
          {
            "type": "text",
            "text": "Please describe this music clip in detail."
          }
        ]
      }
    ],
    "max_tokens": 1024,
    "temperature": 0.0,
    "separate_reasoning": true
  }'
```

##### Pure text reasoning

```bash
curl -X POST http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [
      {
        "role": "user",
        "content": "There are 3 singers on stage. 2 leave, and then 5 join. How many singers are on stage now? Please reason step by step."
      }
    ],
    "max_tokens": 2048,
    "temperature": 0.0,
    "separate_reasoning": true
  }'
```

### Thinking Control

#### Method 1: Template-level switch (`enable_thinking`)

Use the chat template to control whether the model enters thinking mode. This
only applies to pure text chat requests. Audio requests take the shortcut
branch in the template, so this switch does not affect them.

```json
{
  "model": "default",
  "messages": [{"role": "user", "content": "Hello"}],
  "max_tokens": 1024,
  "chat_template_kwargs": {
    "enable_thinking": false
  }
}
```

#### Method 2: Thinking Budget (sampling-level control, requires Mode 3)

Use a custom logit processor to limit the number of tokens spent in thinking.
Based on the Qwen3 technical report, once the budget is reached, a
natural-language transition sentence is injected so the model can smoothly
switch to answer mode.

##### Get the serialized processor string

```python
from sglang.srt.sampling.custom_logit_processor import Qwen3InstructionInjectionThinkingBudgetLogitProcessor
processor_str = Qwen3InstructionInjectionThinkingBudgetLogitProcessor.to_str()
print(processor_str)
```

##### Use it in OpenAI Chat

```bash
curl -X POST http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [
      {
        "role": "user",
        "content": "Please analyze the harmonic structure and emotional progression of this piece."
      }
    ],
    "max_tokens": 2048,
    "temperature": 0.0,
    "separate_reasoning": true,
    "custom_logit_processor": "<processor_str>",
    "custom_params": {
      "thinking_budget": 50
    }
  }'
```

Replace `<processor_str>` with the string produced in the previous step.

##### Use it in `/generate`

```bash
curl -X POST http://localhost:30000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Please analyze the harmonic structure and emotional progression of this piece.",
    "sampling_params": {
      "max_new_tokens": 2048,
      "temperature": 0.0,
      "custom_params": {
        "thinking_budget": 50
      }
    },
    "custom_logit_processor": "<processor_str>"
  }'
```

##### Meaning of `thinking_budget`

| Value | Effect |
|---|---|
| `0` | No thinking allowed; inject the transition sentence immediately after `<think>` and close it |
| `50` | Allow up to 50 thinking tokens |
| `200` | Allow a longer chain of thought |
| not provided | No limit; the model can think freely |

#### Method 3: Streaming + hidden reasoning

```bash
curl -N http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 1024,
    "stream": true,
    "separate_reasoning": true,
    "stream_reasoning": false
  }'
```

In SSE, reasoning content is emitted through `delta.reasoning_content`, while
the final answer is emitted through `delta.content`. When
`stream_reasoning=false`, reasoning tokens are not streamed out token by token.

### Thinking Budget Processor Comparison

| | `Qwen3ThinkingBudgetLogitProcessor` | `Qwen3InstructionInjectionThinkingBudgetLogitProcessor` |
|---|---|---|
| Truncation style | Force `\n` -> `</think>` | Inject the official Qwen3 transition sentence + `</think>` |
| Number of injected tokens | 2 | 24 |
| Whether the model "understands" the cutoff | No | Yes |
| Whether duplicated `</think>` may appear | Yes | No |
| Matching parser | `--reasoning-parser qwen3` | `--reasoning-parser qwen3-instruction-injection` |

Recommended combination:
`Qwen3InstructionInjectionThinkingBudgetLogitProcessor` +
`qwen3-instruction-injection`.

### Reasoning Parser Comparison

| | `qwen3` | `qwen3-instruction-injection` |
|---|---|---|
| Basic split behavior | Split by `<think>...</think>` | Same as left |
| Transition sentence cleanup | No | Strip the injected transition sentence from `reasoning_content` |
| Recommended scenario | When not using thinking budget | When using instruction injection budget |

### Quick Reference

#### Music description (minimal)

```bash
sglang serve --model-path /path/to/moss-music-model --trust-remote-code

curl -X POST http://localhost:30000/generate \
  -H "Content-Type: application/json" \
  -d '{"text":"Please give a detailed musical description of this clip.","audio_data":"/path/to/music.wav","sampling_params":{"max_new_tokens":1024,"temperature":0.0}}'
```

#### Music understanding + separated thinking + budget control (full example)

```bash
sglang serve \
  --model-path /path/to/moss-music-model \
  --trust-remote-code \
  --reasoning-parser qwen3-instruction-injection \
  --enable-custom-logit-processor
```

```python
from sglang.srt.sampling.custom_logit_processor import Qwen3InstructionInjectionThinkingBudgetLogitProcessor
import requests

processor_str = Qwen3InstructionInjectionThinkingBudgetLogitProcessor.to_str()

resp = requests.post("http://localhost:30000/v1/chat/completions", json={
    "model": "default",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "audio_url", "audio_url": {"url": "/path/to/music.wav"}},
                {"type": "text", "text": "Please describe this music clip in detail."}
            ]
        }
    ],
    "max_tokens": 1024,
    "temperature": 0.0,
    "separate_reasoning": True,
    "custom_logit_processor": processor_str,
    "custom_params": {"thinking_budget": 50}
})

data = resp.json()
print("content:", data["choices"][0]["message"]["content"])
print("reasoning:", data["choices"][0]["message"]["reasoning_content"])
```

---

<a id="chinese-version"></a>

## 中文版

### 安装

如果你当前仓库里已经自带了兼容的 `sglang` 代码，比如仓库根目录下已经有
`./sglang`，并且它本身就是 `moss-audio` 对应那套实现，那么就不用再重新
`git clone` 一份。你只需要在当前环境里完成安装即可。

否则执行：

```bash
git clone -b moss-audio https://github.com/OpenMOSS/sglang.git
cd sglang
pip install -e "python[all]"
pip install nvidia-cudnn-cu12==9.16.0.29
```

如果仓库里已经有 `./sglang`，可以直接执行：

```bash
cd sglang
pip install -e "python[all]"
pip install nvidia-cudnn-cu12==9.16.0.29
```

如果你已经把模型下载到本地，例如 `./weights/MOSS-Music-8B-Instruct` 或
`./weights/MOSS-Music-8B-Thinking`，后面的 `--model-path` 可以直接写这些
本地路径。

### 说明

所有 MOSS-Music 模型权重均自带多模态 chat 模板（`chat_template.jinja`），
无需额外指定模板文件。`/generate` 和 `/v1/chat/completions` 两种接口均可
直接使用。

下面所有命令默认假设你已经在安装好 `sglang` 的环境中执行。

如果你使用的是 `torch==2.9.1+cu128`，建议先安装
`nvidia-cudnn-cu12==9.16.0.29`，否则 `sglang` 可能会因为已知的 CuDNN
兼容性检查而拒绝启动。

MOSS-Music 权重对外暴露的是 `MossMusicModel` /
`MossMusicProcessor`，但 remote code 会在加载时把它映射到 SGLang
内置的 MOSS-Audio 多模态实现上，因此你可以直接使用 `moss-audio`
分支，而不用额外改本地 SGLang 代码。

### 启动模式

#### 模式 1：基础服务

适用于 `/generate` 和 `/v1/chat/completions` 的音乐理解、歌词转录与文本对话。

```bash
sglang serve \
  --model-path ./weights/MOSS-Music-8B-Instruct \
  --trust-remote-code
```

#### 模式 2：Reasoning 分离

在模式 1 基础上，自动将 `<think>...</think>` 从正文中拆分到
`reasoning_content` 字段。

```bash
sglang serve \
  --model-path ./weights/MOSS-Music-8B-Thinking \
  --trust-remote-code \
  --reasoning-parser qwen3
```

#### 模式 3：Reasoning 分离 + Thinking Budget 控制（推荐）

在模式 2 基础上增加 thinking budget 控制能力，使用基于 Qwen3 技术报告的
指令注入方案。

```bash
sglang serve \
  --model-path ./weights/MOSS-Music-8B-Thinking \
  --trust-remote-code \
  --reasoning-parser qwen3-instruction-injection \
  --enable-custom-logit-processor
```

### 启动参数说明

| 参数 | 作用 |
|---|---|
| `--reasoning-parser qwen3` | 按 Qwen3 格式拆分 `<think>...</think>` |
| `--reasoning-parser qwen3-instruction-injection` | 同上，额外清理 thinking budget 注入的过渡句 |
| `--enable-custom-logit-processor` | 允许请求传入自定义 logit processor（thinking budget 需要） |

### 请求方式

#### 1. 原生 `/generate`（所有模式可用）

##### 基础音乐描述

```bash
curl -X POST http://localhost:30000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Please give a detailed musical description of this clip.",
    "audio_data": "/path/to/music.wav",
    "sampling_params": {
      "max_new_tokens": 1024,
      "temperature": 0.0
    }
  }'
```

##### 歌词转录

```bash
curl -X POST http://localhost:30000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Transcribe the lyrics of this song.",
    "audio_data": "/path/to/song.wav",
    "sampling_params": {
      "max_new_tokens": 1024,
      "temperature": 0.0
    }
  }'
```

##### `/generate` + 后置 reasoning 拆分

先生成，再用 `/separate_reasoning` 拆分：

```bash
curl -X POST http://localhost:30000/separate_reasoning \
  -H "Content-Type: application/json" \
  -d '{
    "text": "<think>\n思考内容\n</think>\n\n正文内容",
    "reasoning_parser": "qwen3"
  }'
```

#### 2. OpenAI Chat `/v1/chat/completions`（所有模式可用）

##### 音乐描述 + reasoning 分离

```bash
curl -X POST http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "audio_url",
            "audio_url": {
              "url": "/path/to/music.wav"
            }
          },
          {
            "type": "text",
            "text": "Please describe this music clip in detail."
          }
        ]
      }
    ],
    "max_tokens": 1024,
    "temperature": 0.0,
    "separate_reasoning": true
  }'
```

##### 纯文本推理

```bash
curl -X POST http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [
      {
        "role": "user",
        "content": "舞台上有3位歌手，离开了2位，又上来了5位。现在台上有多少位歌手？请一步一步推理。"
      }
    ],
    "max_tokens": 2048,
    "temperature": 0.0,
    "separate_reasoning": true
  }'
```

### Thinking 控制

#### 方式 1：模板级开关（`enable_thinking`）

通过 chat template 控制模型是否进入 thinking 模式。仅对纯文本 chat 请求生效；
音频请求走模板的短路分支，此开关不生效。

```json
{
  "model": "default",
  "messages": [{"role": "user", "content": "你好"}],
  "max_tokens": 1024,
  "chat_template_kwargs": {
    "enable_thinking": false
  }
}
```

#### 方式 2：Thinking Budget（采样层控制，需要模式 3）

通过 custom logit processor 在采样时限制 thinking 的 token 数量。基于
Qwen3 技术报告，当 budget 到达时注入一段自然语言过渡句，让模型自然切换到
answer 模式。

##### 获取 processor 序列化字符串

```python
from sglang.srt.sampling.custom_logit_processor import Qwen3InstructionInjectionThinkingBudgetLogitProcessor
processor_str = Qwen3InstructionInjectionThinkingBudgetLogitProcessor.to_str()
print(processor_str)
```

##### 在 OpenAI Chat 中使用

```bash
curl -X POST http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [
      {
        "role": "user",
        "content": "请分析这段音乐的和声结构与情绪推进。"
      }
    ],
    "max_tokens": 2048,
    "temperature": 0.0,
    "separate_reasoning": true,
    "custom_logit_processor": "<processor_str>",
    "custom_params": {
      "thinking_budget": 50
    }
  }'
```

`<processor_str>` 替换为上一步生成的字符串。

##### 在 `/generate` 中使用

```bash
curl -X POST http://localhost:30000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "请分析这段音乐的和声结构与情绪推进。",
    "sampling_params": {
      "max_new_tokens": 2048,
      "temperature": 0.0,
      "custom_params": {
        "thinking_budget": 50
      }
    },
    "custom_logit_processor": "<processor_str>"
  }'
```

##### `thinking_budget` 值的含义

| 值 | 效果 |
|---|---|
| `0` | 不允许 thinking，`<think>` 后立刻注入过渡句并闭合 |
| `50` | 允许最多 50 个 token 的思考 |
| `200` | 允许较长的思考链 |
| 不传 | 不限制，模型自由思考 |

#### 方式 3：流式 + 隐藏 reasoning

```bash
curl -N http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "你好"}],
    "max_tokens": 1024,
    "stream": true,
    "separate_reasoning": true,
    "stream_reasoning": false
  }'
```

SSE 中 reasoning 内容走 `delta.reasoning_content`，正文走 `delta.content`。
`stream_reasoning=false` 时不会逐 token 流出 reasoning。

### Thinking Budget Processor 对比

| | `Qwen3ThinkingBudgetLogitProcessor` | `Qwen3InstructionInjectionThinkingBudgetLogitProcessor` |
|---|---|---|
| 截断方式 | 强制 `\n` -> `</think>` | 注入 Qwen3 官方过渡句 + `</think>` |
| 注入 token 数 | 2 | 24 |
| 模型是否“理解”截断 | 否 | 是 |
| 是否产生重复 `</think>` | 是 | 否 |
| 搭配 parser | `--reasoning-parser qwen3` | `--reasoning-parser qwen3-instruction-injection` |

推荐使用 `Qwen3InstructionInjectionThinkingBudgetLogitProcessor` +
`qwen3-instruction-injection`。

### Reasoning Parser 对比

| | `qwen3` | `qwen3-instruction-injection` |
|---|---|---|
| 基础拆分 | 按 `<think>...</think>` 拆 | 同左 |
| 过渡句清理 | 不清理 | 从 `reasoning_content` 中 strip 注入的过渡句 |
| 适用场景 | 不使用 thinking budget 时 | 使用 instruction injection budget 时 |

### 快速参考

#### 音乐理解（最简）

```bash
sglang serve --model-path /path/to/moss-music-model --trust-remote-code

curl -X POST http://localhost:30000/generate \
  -H "Content-Type: application/json" \
  -d '{"text":"Please give a detailed musical description of this clip.","audio_data":"/path/to/music.wav","sampling_params":{"max_new_tokens":1024,"temperature":0.0}}'
```

#### 音乐理解 + thinking 分离 + budget 控制（完整）

```bash
sglang serve \
  --model-path /path/to/moss-music-model \
  --trust-remote-code \
  --reasoning-parser qwen3-instruction-injection \
  --enable-custom-logit-processor
```

```python
from sglang.srt.sampling.custom_logit_processor import Qwen3InstructionInjectionThinkingBudgetLogitProcessor
import requests

processor_str = Qwen3InstructionInjectionThinkingBudgetLogitProcessor.to_str()

resp = requests.post("http://localhost:30000/v1/chat/completions", json={
    "model": "default",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "audio_url", "audio_url": {"url": "/path/to/music.wav"}},
                {"type": "text", "text": "Please describe this music clip in detail."}
            ]
        }
    ],
    "max_tokens": 1024,
    "temperature": 0.0,
    "separate_reasoning": True,
    "custom_logit_processor": processor_str,
    "custom_params": {"thinking_budget": 50}
})

data = resp.json()
print("content:", data["choices"][0]["message"]["content"])
print("reasoning:", data["choices"][0]["message"]["reasoning_content"])
```
