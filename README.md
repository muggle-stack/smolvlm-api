# Smolvlm Ollama API 调用

一个基于 SmolVLM 的多模态对话工具，支持图像理解和文本交互。

## 功能特点

- 支持本地图片分析
- 支持摄像头实时拍照分析
- 支持流式输出
- 交互式对话模式

## 下载模型

在这下：[https://huggingface.co/ggml-org/SmolVLM-256M-Instruct-GGUF/tree/main]

```bash
wget https://huggingface.co/ggml-org/SmolVLM-256M-Instruct-GGUF/resolve/main/SmolVLM-256M-Instruct-f16.gguf
wget https://huggingface.co/ggml-org/SmolVLM-256M-Instruct-GGUF/resolve/main/mmproj-SmolVLM-256M-Instruct-f16.gguf
```

## 下载ollama[https://ollama.com/download]

## 启动ollama
```bash
ollama serve
```

## 制作smolvlm模型，创建文件命名为 smolvlm.modelfile，复制以下代码粘贴到文件里，并保存：
```modelfile
FROM ./SmolVLM-256M-Instruct-f16.gguf
ADAPTER ./mmproj-SmolVLM-256M-Instruct-Q8_0.gguf

TEMPLATE """
<|im_start|>system
{{ .System }}
<end_of_utterance>
{{- range .Messages }}
<|im_start|>{{ .Role }}:
{{ .Content }}
<end_of_utterance>
{{- end }}
<|im_start|>assistant
"""

SYSTEM "You are a visual assistant. Describe images clearly and answer questions based on visual content."

PARAMETER num_ctx 4096
PARAMETER stop "<end_of_utterance>"
PARAMETER stop "<|im_start|>"
PARAMETER temperature 0.01
```

## 制作 ollama 模型
```bash
ollama create smolvlm:256m -f smolvlm.modelfile
```

## 安装依赖

```bash
pip install opencv-python
```

## 克隆仓库
```bash
git clone https://github.com/muggle-stack/smolvlm-api.git
cd smolvlm-api
```

## 使用方法

### 使用本地图片

```bash
python smolvlm-api.py --image ~/Downloads/muggle-stack.png --stream
```

### 使用摄像头拍照

```bash
python smolvlm-api.py --camera --prompt "Descripe this image"
```

使用摄像头时：
- 按空格键拍照
- 按ESC退出

### 交互模式

```bash
# 使用本地图片进入交互模式
python smolvlm-api.py --image path/to/your/image.jpg

# 使用摄像头进入交互模式
python smolvlm-api.py --camera
```

### 流式输出

添加 `--stream` 参数启用流式输出：

```bash
python smolvlm-api.py --image path/to/your/image.jpg --stream
```

## 参数说明

- `--model`: 指定使用的模型（默认：smolvlm:256m）
- `--image`: 指定要分析的图片路径
- `--camera`: 启用摄像头模式
- `--prompt`: 指定初始提示词
- `--stream`: 启用流式输出

## 注意事项

1. 使用摄像头时，按空格键拍照，按ESC退出
2. 确保系统已正确安装并配置 Ollama
3. 确保有足够的系统资源运行模型