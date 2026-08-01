# CosyVoice.cpp

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-blue.svg)]()
[![GitHub Release](https://img.shields.io/github/v/release/Lourdle/cosyvoice.cpp)](https://github.com/Lourdle/cosyvoice.cpp/releases)
[![CI](https://github.com/Lourdle/cosyvoice.cpp/actions/workflows/build-release.yml/badge.svg)](https://github.com/Lourdle/cosyvoice.cpp/actions/workflows/build-release.yml)

语言： [English](README.md) | 简体中文

> 非官方说明：本仓库**与 CosyVoice 官方团队无隶属关系**，也未获得官方背书或维护。本项目是社区开发者发起和维护的 C++/GGML 移植实现。

> **当前状态提示：** 当前 CPU、CUDA、Metal、Vulkan 和 SYCL 后端均可正常运行。用于生产前请先阅读[后端测试情况](#后端测试情况)。

本项目将原始 CosyVoice 项目发布的 Python 推理流程迁移到 C++/GGML，目前主要支持 **CosyVoice3**。

支持 **zero-shot**、**instruct** 和 **cross-lingual** 三种 TTS 模式，同时提供同步输出与**流式输出**（streaming）。前端流水线（speech tokenizer + speaker embedding）可处理参考音频，也可复用预编码的 `prompt_speech` 跳过 ONNX 前端以提升性能。

本项目提供：
- 核心 C/C++ 推理库（`cosyvoice`）
- 命令行合成工具（`cosyvoice-cli`）
- OpenAI Speech 兼容 API 服务，含嵌入式 WebUI（`cosyvoice-server`）
- GGUF 量化工具（`quantize`）

## 目录
- [功能特性](#功能特性)
- [快速开始](#快速开始)
- [预转换模型](#预转换模型)
- [推理流程](#推理流程)
- [工具使用说明](#工具使用说明)
- [构建](#构建)
- [流式 TTS 与 DiT KV 缓存](#流式-tts-与-dit-kv-缓存)
- [模型转 GGUF](#模型转-gguf)
- [后端测试情况](#后端测试情况)
- [故障排查](#故障排查)
- [文档](#文档)
- [AI 使用说明](#ai-使用说明)
- [第三方许可说明](#第三方许可说明)
- [许可证说明](#许可证说明)
- [欢迎贡献](#欢迎贡献)

## 功能特性

| 特性 | 说明 |
|---|---|
| **OpenAI Speech API 服务** | 即插即用的 `POST /v1/audio/speech` 端点，支持多音色、鉴权和 CORS——内置 **WebUI** 用于模型/音色管理和 TTS 生成 |
| **WebUI 仪表盘** | 现代化浏览器界面，支持运行时加载/卸载模型、注册音色（GGUF 导入/音频提取/麦克风录音）、TTS 生成（实时播放、历史记录、完整采样控制） |
| **交互式 REPL** | CLI 交互模式，支持 /play、/save、/list、/query、/seed 等斜杠命令 |
| **并发服务** | Server 的 `--concurrency` 参数，支持并行请求处理 |
| **推理中断** | CLI 中按 `Ctrl+C` 或断开 server 连接时尽快停止生成——截止点之前的输出仍然有效 |
| **模型量化** | 内置 `quantize` 工具，支持 Q2_K 到 F16 多种量化格式 |
| **流式 TTS** | 实时语音生成，通过回调逐段交付音频——在完整语句尚未合成完毕前即可开始播放 |
| **DiT KV 缓存** | 在流式推理过程中，跨扩散步复用注意力 KV 缓存，避免冗余计算——可配置固定（设备内存）、可卸载（CPU）、不缓存三种槽位类型 |
| **Flash Attention** | LLM 与 Flow 模块均支持 flash attention（`--llm-flash-attn`、`--flow-flash-attn`），后端支持时可降低显存并加速推理 |
| **分块 Token 控制** | 通过 `--chunk-tokens` 调节流式推理的延迟与开销平衡——较小分块降低首块延迟，较大分块降低 RTF |
| **KV Cache 量化** | 通过 `--llm-kv-cache-type` 降低 LLM 内存占用（f32 / f16 / q8_0 / q5_1 / q4_0 / ...）。支持非对称量化，K 和 V 可独立指定类型（如 `k=q8_0,v=q4_0`）。 |
| **Prompt Speech 复用** | 一次编码参考音色，后续合成直接复用，无需再跑 ONNX |
| **音频后端可切换** | 可选 MINIAUDIO（默认）或 FFMPEG，支持 WAV、MP3、AAC、FLAC、OPUS、M4A |
| **UMA 自动检测** | 自动检测统一内存架构并调整 buffer policy，优化吞吐 |
| **推理 Buffer 策略** | `shared` / `balanced` / `dedicated` 三种模式，权衡内存与吞吐 |
| **文本拆分与淡入** | 长文本智能拆分与可配置的输出淡入后处理 |
| **多后端支持** | CPU、CUDA、Metal、Vulkan、SYCL（见[后端测试情况](#后端测试情况)） |
| **跨平台** | Windows (x64)、Linux (x86_64)、macOS (arm64) — 均在 CI 中测试 |

## 快速开始

### 预编译发布版 (Releases)

本仓库提供的 Releases 不包含 GGML 后端库：
1. 从本仓库的 [Releases 页面](https://github.com/Lourdle/cosyvoice.cpp/releases) 下载 `cosyvoice-cli` 或 `cosyvoice-server`。
2. 下载与硬件和操作系统匹配的 `llama.cpp` release。
3. 将 `cosyvoice` 可执行文件放到包含 GGML 后端共享库（`ggml.dll`、`ggml-cuda.dll` 等）的同一目录。
4. 在该目录下运行。

> **预编译 GGML CUDA 后端已知问题（Issue [#15](https://github.com/Lourdle/cosyvoice.cpp/issues/15)）：** 有用户反馈使用 `llama.cpp` 预编译发布版的 GGML CUDA 后端时，生成的音频存在噪音。测试确认了预编译 GGML CUDA 版本存在此问题，而自行从源码编译的 GGML 则未出现该问题。如果您在使用 CUDA 后端配合预编译 GGML 时遇到噪音，建议参考本文[构建](#构建)章节，将本项目与 GGML 一同从源码编译。

### 从源码构建

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

构建产物在 `build/bin`（可执行文件）和 `build/lib`（库文件）。详细构建选项、服务端特殊要求和后端配置见[构建](#构建)章节。

## 预转换模型

下载即用的 GGUF 模型（无需自行转换）：

- **ModelScope**：<https://modelscope.cn/models/Lourdle/Fun-CosyVoice3-0.5B-2512-GGUF>
- **Hugging Face**：<https://huggingface.co/Lourdle/Fun-CosyVoice3-0.5B-2512-GGUF>

上述链接包含 Q2_K 到 F16 的多种量化变体。

## 推理流程

本项目支持两条等价推理路径：

```mermaid
flowchart TD
    subgraph E2E ["端到端（前端 + TTS）"]
        A["参考音频<br/>+ 转录文本"] --> B["前端 (ONNX)<br/>SpeechTokenizer + Campplus"]
        B --> C["prompt_speech<br/>（音色嵌入）"]
        C --> D["TTS<br/>(LLM + Flow + HiFT)"]
        E["目标文本"] --> D
        D --> F["输出音频"]
    end

    subgraph REUSE["复用已保存的 prompt_speech"]
        G["已保存的 prompt_speech.gguf<br/>（通过 --frontend-only<br/>或 --prompt-speech-output）"] --> H["TTS<br/>(LLM + Flow + HiFT)"]
        I["目标文本"] --> H
        H --> J["输出音频"]
    end
```

- **路径 1（端到端）**：前端从参考音频 + 转录文本提取 `prompt_speech`，然后 TTS 与目标文本合成语音。
  - `zero-shot` 模式需要 `--prompt-text`；`instruct` / `cross-lingual` 模式忽略它。
- **路径 2（复用）**：通过 `--frontend-only` / `--prompt-speech-output` 运行一次前端，后续合成跳过 ONNX 模型。适合批量/重复合成。

## 工具使用说明

本仓库包含 3 个面向使用者的工具：
- `cosyvoice-cli`：本地文件式 TTS 合成（支持复用 prompt_speech，以及前端 + TTS 一体流程）。
- `cosyvoice-server`：OpenAI Speech 兼容 HTTP API 服务，适合服务化接入。
- `quantize`：GGUF 量化工具，用于将模型转换为更小/更快的量化格式。支持通过 PCRE2 正则逐 tensor 指定量化类型（`-M/--tensor-map`）。预置的 CosyVoice3-2512 profile 见 `tools/quantize/profiles/`。

完整命令、参数和示例见 [docs/TOOLS_zh.md](docs/TOOLS_zh.md)。

## 构建

### 环境要求
- CMake >= 3.24
- 支持 C++20 的 C/C++ 编译器
- Git（当本地缺少 GGML 源码时用于自动拉取）
- 目前 CPU 路径中的部分数据处理要求 x86 CPU 支持 AVX2
- 对 CPU 侧数学运算较重的路径（如 `log`、三角函数），当前仅 MSVC 构建可启用 SIMD 加速；其他工具链目前回退为标量实现

后端/运行时依赖会随构建选项变化（CUDA/Vulkan/CPU、ONNX Runtime、ICU 等）。

### 基本构建

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

构建产物默认输出到：
- `build/bin`（可执行文件与运行时 DLL）
- `build/lib`（库文件）

### Server 构建须知（非 Windows 平台）

在 Linux/macOS 上编译 `cosyvoice-server` 需在以下两方面额外留意：

**C23 `#embed` 嵌入 WebUI 资源**
WebUI 资源嵌入通过 C23 `#embed` 指令将 HTML/CSS/JS 打包到可执行文件中，因此需要 **C 编译器**支持 C23——即 GCC 15+ 或 Clang 19+。Windows 通过原生 RC 工具嵌入资源，无需特殊编译器。

示例——指定支持 C23 的 C 编译器：
```bash
# Ubuntu/Debian — 用 clang-20 作为 C 编译器
sudo apt install clang-20
cmake -B build -DCMAKE_C_COMPILER=clang-20 -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

**C++20 模块**
服务端使用了 C++20 模块接口（封装 nlohmann/json 和 cpp-httplib）。**建议**使用 **Ninja** 生成器（1.11+）配合较新的 C++ 编译器（GCC 14+ / Clang 16+ / MSVC 14.34+）以获得完整的模块扫描支持：
- 在 cmake 配置时添加 `-G Ninja`。
- Windows：Visual Studio 生成器完整支持模块扫描——无需额外参数。
- 不支持的生成器或旧版编译器：CMake 会自动检测并**回退到预编译头（PCH）**。

Linux/macOS 推荐配置：
```bash
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

### CMake 选项

**项目选项**

| 选项 | 取值 | 默认值 | 说明 |
|---|---|---|---|
| `BUILD_SHARED_LIBS` | ON / OFF | ON | 构建为共享库 |
| `COSYVOICE_NO_AUDIO` | ON / OFF | OFF | 关闭音频辅助 API |
| `COSYVOICE_NO_FRONTEND` | ON / OFF | OFF | 关闭 ONNX 前端 |
| `COSYVOICE_NO_ICU` | ON / OFF | OFF | 关闭 ICU 文本规范化 |
| `COSYVOICE_AUDIO_BACKEND` | MINIAUDIO / FFMPEG | MINIAUDIO | 音频编解码后端 |
| `COSYVOICE_CLI_NO_PLAYBACK` | ON / OFF | 未设置（跟随 `COSYVOICE_NO_AUDIO`） | 关闭 CLI 播放功能 |
| `COSYVOICE_SERVER_NO_WEBUI` | ON / OFF | OFF | 关闭嵌入式 WebUI，仅以 API 模式运行；非 Windows 平台可移除 C23 `#embed` 依赖 |
| `COSYVOICE_SERVER_DEFAULT_MODE` | WEBUI / API | WEBUI | 命令行未指定 `--api` 或 `--webui` 时的默认模式。设为 `API` 可用于纯后端部署 |

**后端选项**

GGML 后端选项可直接从根工程透传。常见示例：

```bash
# CUDA 后端
cmake -B build -DGGML_CUDA=ON

# Vulkan 后端
cmake -B build -DGGML_VULKAN=ON
```

完整后端选项列表及推荐配置请参考 [GGML 文档](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md)。

**Metal 后端（`GGML_METAL`）特殊处理**

`cmake/patches/ggml-metal-pad-beg.patch`（Metal PAD beg-padding 补丁）是针对特定 ggml 快照编写的。若 Metal 开启而 ggml 漂移到最新 master，`git apply` 会因行偏移/内核重写而失败，从而静默禁用 Metal PAD 支持。为保证补丁始终有效，构建系统会对 ggml 固定提交号——但仅在 Metal 构建时生效，其他后端仍像以前一样使用最新 ggml。

- `GGML_METAL` 在 Apple Silicon 上**默认为 ON**（见 ggml 自身 CMakeLists），也可用 `-DGGML_METAL=ON/OFF` 强制指定。
- **Metal 构建**（Apple Silicon 默认）：GGML 被固定到提交 `af97976c7810cdabb1863172f31c432dab767de7`（可通过 `cmake/Dependencies.cmake` 中的 `GGML_PINNED_COMMIT` 配置）。CMake 会在克隆后自动 checkout 该提交；若已有 `vendor/ggml` 检出偏离固定提交，仅警告而不中断；并幂等应用 `cmake/patches/ggml-metal-pad-beg.patch`（已应用则跳过）。
- **非 Metal 构建**：行为不变——浅克隆（`--depth=1`）最新 master，不应用任何补丁。

```bash
# 强制开启/关闭 Metal（Apple Silicon 默认开启）
cmake -B build -DGGML_METAL=ON
```

如需升级 Metal 构建使用的 ggml：先更新 `cmake/Dependencies.cmake` 中的 `GGML_PINNED_COMMIT`，并按照该文件中的说明针对新代码树重新生成补丁，再端到端验证 Metal 合成效果。

**依赖路径选项**

| 选项 | 说明 |
|---|---|
| `GGML_SOURCE_DIR=<path>` | GGML 源码路径（默认 `vendor/ggml`），目录不存在时自动克隆 |
| `ICU_PREBUILT_DIR=<path>` | ICU 预编译二进制路径（默认 `<build_dir>/_deps/icu`） |
| `ORT_PREBUILT_DIR=<path>` | ONNX Runtime 预编译二进制路径（默认 `<build_dir>/_deps/onnxruntime`） |
| `FFMPEG_PREBUILT_DIR=<path>` | FFmpeg 预编译二进制路径 |
| `SIMDE_INCLUDE_DIR=<path>` | ARM64/aarch64（含 Android 交叉编译）所需的 SIMDe 头文件目录——见 [SIMDe（SIMD Everywhere）](#simdesimd-everywhere) |

### 常见构建矩阵

| 场景 | 推荐 CMake 参数 |
|---|---|
| CUDA 后端 | `-DGGML_CUDA=ON` |
| Vulkan 后端 | `-DGGML_VULKAN=ON` |
| 仅 CPU | 通常不需要额外后端参数 |
| 仅核心能力（无 frontend / ICU） | `-DCOSYVOICE_NO_FRONTEND=ON -DCOSYVOICE_NO_ICU=ON` |
| 关闭音频辅助 API | `-DCOSYVOICE_NO_AUDIO=ON` |
| 关闭 CLI 播放功能 | `-DCOSYVOICE_CLI_NO_PLAYBACK=ON` |

常见组合示例：

```bash
# 仅核心功能构建（关闭 ONNX 前端与 ICU 文本规范化）
cmake -B build-core -DCMAKE_BUILD_TYPE=Release -DCOSYVOICE_NO_FRONTEND=ON -DCOSYVOICE_NO_ICU=ON

# 无音频辅助 API 构建（CLI 走 WAV 输出回退路径）
cmake -B build-noaudio -DCMAKE_BUILD_TYPE=Release -DCOSYVOICE_NO_AUDIO=ON

# 关闭 CLI 播放功能（音频辅助 API 仍可用）
cmake -B build-noplay -DCMAKE_BUILD_TYPE=Release -DCOSYVOICE_CLI_NO_PLAYBACK=ON
```

### 依赖解析方式

顶层 CMake 按以下顺序解析依赖：

- **PCRE2**：从 `vendor/pcre2` 构建静态库（`pcre2-8`、`pcre2-16`）。
- **GGML**：使用 `GGML_SOURCE_DIR`（默认 `vendor/ggml`）。若目录不存在，会自动克隆 `https://github.com/ggml-org/ggml.git`。
- **ICU**（用于文本规范化，除非通过 `COSYVOICE_NO_ICU` 关闭）：解析顺序：`ICU_PREBUILT_DIR` → `find_package(ICU)` → Windows 自动下载 → Linux/macOS 使用系统 ICU。
- **ONNX Runtime**（用于前端，除非通过 `COSYVOICE_NO_FRONTEND` 关闭）：解析顺序：`ORT_PREBUILT_DIR` → `find_package(onnxruntime)` → 自动下载。

Windows 下预编译依赖 DLL 会复制到可执行文件旁。

### 使用自定义依赖

可以通过缓存变量指定自定义依赖路径：

```bash
cmake -B build \
  -DGGML_SOURCE_DIR=/path/to/ggml \
  -DICU_PREBUILT_DIR=/path/to/icu \
  -DORT_PREBUILT_DIR=/path/to/onnxruntime \
  -DSIMDE_INCLUDE_DIR=/path/to/simde
```

也可以直接使用构建目录下的默认预编译依赖位置：
- `<build_dir>/_deps/icu`
- `<build_dir>/_deps/onnxruntime`

只要按期望目录结构把文件放进去，CMake 会自动识别（不需要额外 `-D`）。

期望的关键目录/文件：
- ICU：`include/unicode/utypes.h`（以及 `lib*` / `bin*` 下的库和 DLL）
- ONNX Runtime：`include/onnxruntime_c_api.h`（以及 `lib` 下的运行库文件）

### SIMDe（SIMD Everywhere）

CPU 热路径直接使用 x86 AVX2/FMA 内联函数。[SIMDe](https://github.com/simd-everywhere/simde) 是一个纯头文件库，可将这些 x86 内联函数翻译到其他指令集——在 ARM64/aarch64 上，同一份代码无需修改即可编译为 NEON。

CMake 的处理方式（`CMakeLists.txt`）：

- **x86_64（GCC/Clang）**：SIMD 路径以 `-mavx -mavx2 -mfma` 编译，原生执行。
- **ARM64/aarch64（含 Android 交叉编译）**：**必须**提供 SIMDe。CMake 会在 `/opt/homebrew/include`、`/usr/local/include`、`/usr/include` 与 `vendor/simde/` 中查找 `simde/x86/avx2.h`，找不到时配置直接报错。

获取 SIMDe：

```bash
# macOS
brew install simde

# Debian/Ubuntu（部分发行版可能没有该包）
apt install libsimde-dev

# 通用方式——克隆并用 SIMDE_INCLUDE_DIR 指向
git clone --depth=1 https://github.com/simd-everywhere/simde.git
cmake -B build -DSIMDE_INCLUDE_DIR=/path/to/simde
```

x86_64 构建不需要 SIMDe。Android 相关细节见 [docs/build-android.md](docs/build-android.md)。

### 音频后端与 FFmpeg

本项目的音频辅助 API 支持两种后端：

- `MINIAUDIO`（默认）：提供 WAV I/O 与基本 PCM 帮助函数。
- `FFMPEG`（可选）：在链接的 FFmpeg 运行时提供所需编码器时，启用更多编码/解码格式。

通过 CMake 配置音频后端：将 `COSYVOICE_AUDIO_BACKEND` 设为 `MINIAUDIO` 或 `FFMPEG`。默认值为 `MINIAUDIO`。

示例：
```bash
cmake -B build -DCOSYVOICE_AUDIO_BACKEND=MINIAUDIO
cmake -B build -DCOSYVOICE_AUDIO_BACKEND=FFMPEG
cmake -B build -DCOSYVOICE_AUDIO_BACKEND=FFMPEG -DFFMPEG_PREBUILT_DIR=/path/to/ffmpeg
```

如果启用 FFmpeg 支持，公开音频 API 的函数名保持不变。可使用 `cosyvoice_audio_supported_encoding_formats()` 查询当前链接的 FFmpeg 运行时真正支持哪些格式。

FFmpeg 使用要点：

- 在 Windows 上，构建脚本默认在未提供 `FFMPEG_PREBUILT_DIR` 时下载 BtbN 的预编译 FFmpeg。
- 在 Linux/macOS 上，若系统提供 FFmpeg（apt/homebrew），项目会优先使用系统库；否则可通过 `FFMPEG_PREBUILT_DIR` 指定预编译位置。
- API 层支持 `wav`、`mp3`、`aac`、`flac`、`m4a`、`opus`，但具体可用格式取决于当前链接的 FFmpeg 构建。库会在运行时探测可用编码器，并通过 API / CLI / server 帮助文本暴露支持集合。
- `m4a` 是这里提供的非标准便捷扩展。OpenAI Speech 标准并没有定义 `m4a`，只在你的客户端/服务端理解这个扩展时使用。
- 如果客户端请求了运行时不可用的格式，服务/CLI 会建议回退到 `wav` 或 `pcm`。
- 在 Windows 上，构建脚本会把找到的 FFmpeg 运行时 DLL 复制到可执行文件目录。若你使用自定义预编译 FFmpeg，请确认其 `bin` / `lib` 目录结构符合 `cmake/Dependencies.cmake` 的预期。

许可证提醒：

- 本仓库代码采用 MIT 许可。FFmpeg 预编译包可能是 LGPL 或 GPL，取决于编译选项。使用包含 GPL 编码器的 FFmpeg 构建并重新分发时，可能会对你的发行物带来 GPL 约束。详见 [FFmpeg-NOTICE.md](FFmpeg-NOTICE.md)。

## 流式 TTS 与 DiT KV 缓存

流式 TTS 在合成过程中通过回调函数逐段交付音频，无需等待完整语句生成完毕即可开始播放，从而实现实时播放与更低的主观延迟。

流式流水线引入了 **DiT KV 缓存** 以避免冗余计算。非流式推理时，DiT 模块运行 10 步扩散，每步对完整音频序列计算自注意力——总共会执行 10 次注意力重算。KV 缓存跨扩散步存储中间 key/value 张量，使每个位置只需计算一次。

### 槽位组织

DiT KV 缓存按 **槽位（slot）** 组织，每个槽位对应一个扩散步的 KV 缓存。默认 10 个扩散步意味着最多 10 个槽位。

槽位分为三类：

| 类别 | 内存位置 | 行为 |
|----------|--------|------|
| **固定** | 常驻设备（GPU） | 最快，从不卸载 |
| **可卸载** | 不使用时卸载到 CPU | 节省设备显存，但增加传输开销 |
| **不缓存** | 不存储 | 每步全量重算注意力，无额外内存开销 |

总槽位数 = `固定 + 可卸载`。剩余步（10 − 总槽位）使用全量重算。

KV 缓存占用较大，因此默认 0 个槽位（全部 10 步全量重算）。启用缓存后，若序列长度超过配置的缓存长度，会丢弃部分位置——推理可正常继续，但输出质量可能下降。可卸载槽位需要和设备与 CPU 间传输数据，可能无法带来速度提升，甚至比全量重算更慢。

> DiT KV 缓存**仅在流式 TTS 时使用**；非流式调用忽略此缓存。

### 配置

DiT KV 缓存参数通过 CLI/server 的 `--dit-kv-*` 参数配置：

- `--dit-kv-type`：DiT KV 缓存存储格式（f32/f16/q8_0/...）。
- `--dit-kv-fixed-slots`：常驻设备内存的槽位数。
- `--dit-kv-offloadable-slots`：可卸载到 CPU 的槽位数。
- `--dit-kv-cache-length`：缓存保留的最大序列位置数。

流式输出通过 `--stream` 标志启用。分块粒度由 `--chunk-tokens` 控制。

## 推理 Buffer 策略

推理引擎使用 buffer 策略控制中间张量的分配方式：

- `shared`：LLM KV 缓存与 DiT 部分中间 buffer 共享内存。每次推理 LLM 模块都会完整运行一次。节省内存，但在未启用 Flash Attention 时 CUDA 上运行不稳定。
- `balanced`：与 `shared` 类似，但在 LLM 推理完成后将下次可复用的 LLM KV 缓存卸载到 CPU。
- `dedicated`：完全独立分配。LLM KV 缓存驻留设备可快速复用——流式推理建议使用此模式。

## 模型转 GGUF

可使用本仓库的转换脚本 `convert_model_to_gguf.py`，将上游 CosyVoice 模型权重转换为 `cosyvoice.cpp` 可用的 GGUF。

先安装 Python 依赖：
```bash
pip install -r requirements.txt
```

最小用法：
```bash
python convert_model_to_gguf.py \
  --yaml_config /path/to/cosyvoice.yaml \
  --ftype f16 \
  --gguf_model /path/to/CosyVoice3-2512_F16.gguf
```

完整参数示例：
```bash
python convert_model_to_gguf.py \
  --yaml_config /path/to/cosyvoice.yaml \
  --llm_model /path/to/llm.pt \
  --blank_llm /path/to/CosyVoice-BlankEN \
  --flow_model /path/to/flow.pt \
  --hift_model /path/to/hift.pt \
  --gguf_model /path/to/CosyVoice3-2512_Q8_0.gguf \
  --ftype q8_0 \
  --tag 2512
```

`--ftype` 可选值：
- `default`, `f32`, `f16`, `q8_0`, `q5_0`, `q5_1`, `q4_0`, `q4_1`

未显式传入时的默认路径规则：
- `--llm_model` -> `<yaml_dir>/llm.pt`
- `--blank_llm` -> `<yaml_dir>/CosyVoice-BlankEN`
- `--flow_model` -> `<yaml_dir>/flow.pt`
- `--hift_model` -> `<yaml_dir>/hift.pt`

转换后建议：
1. 先确认生成的 `.gguf` 文件可用。
2. （可选）再使用本仓库 `quantize` 工具量化。

## 后端测试情况

当前各后端测试结果如下：

| 后端 | 状态 | 备注 |
|---|---:|---|
| CPU | 可运行 | 已在 Windows、Linux 和 Mac 上测试。 |
| CUDA | 可运行 | 已在 Ada Lovelace GPU (Windows & Linux) 上测试。 |
| Metal | 可运行 | 感谢 @[jasagiri](https://github.com/jasagiri) 的支持与代码贡献。 |
| SYCL | 可运行 | 已在 Windows 11 x64 上的 Intel Raptor Lake 集成显卡上验证。 |
| Vulkan | 可运行 | 在 NVIDIA Ada Lovelace GPU 与 Intel Raptor Lake 核显上验证通过。 |
| OpenCL | 可运行 | 在 Android 16、Qualcomm Snapdragon 8 Elite 上验证通过。OpenCL 后端缺失大量算子，需卸载到 CPU 运行，频繁切换计算后端导致上下文开支较大，相比 CPU 并未带来显著提速。 |
| 其它 | 未测试 | |

## 故障排查
- CMake 找不到 GGML：设置 `-DGGML_SOURCE_DIR=...`，或使用默认 `vendor/ggml` 并确保本机可用 Git（用于自动克隆）。
- ICU/ONNX Runtime 检测失败：可安装系统包（适用平台），或将预编译文件放到 `<build_dir>/_deps/icu` 与 `<build_dir>/_deps/onnxruntime`。
- Windows 运行时缺库：检查 `build/bin` 下是否存在构建后复制的依赖 DLL。
- 后端相关情况见[后端测试情况](#后端测试情况)。

## 文档
- API 索引：[docs/API_zh.md](docs/API_zh.md)
- 工具说明：[docs/TOOLS_zh.md](docs/TOOLS_zh.md)
- Android 构建指南：[docs/build-android_zh.md](docs/build-android_zh.md)

## AI 使用说明
- 核心库代码主要由作者手工实现。
- 工具（cli、quantize、server）和文档内容大多由 AI 协助撰写与整理。
- 仍可能存在少量错误或与实现不同步的情况；如有疑问请以源码与头文件为准，也欢迎提交 Issue/PR 纠正。

## 第三方许可说明
- 已打包依赖的许可证信息见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
- FFT 实现参考/改造自 KissFFT（BSD-3-Clause），并加入了项目内 SIMD 优化；详见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
- tokenizer 实现基于 llama.cpp（MIT）改造。

## 许可证说明
- **本仓库代码**：MIT（见 `LICENSE`）。
- **上游参考**：原始 CosyVoice 项目代码与模型为 Apache-2.0。
- **实现说明**：本仓库是基于模型架构与推理行为的独立 C++/GGML 重实现，并非官方 fork 或官方发布。
- **GGUF 模型产物**：发布的模型文件继续保持 Apache-2.0。下载链接见[预转换模型](#预转换模型)。
- **模型许可证文件**：[MODEL_LICENSE.md](MODEL_LICENSE.md)

## 欢迎贡献
欢迎提交 Issue 和 Pull Request，尤其是：
- 后端稳定性修复
- 跨平台正确性改进
- 性能与内存优化
- 文档与工具改进

如果根因在 GGML，请优先向上游 GGML 提交修复补丁。
