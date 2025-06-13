# Smolvlm Ollama API 调用

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

## 克隆仓库
```bash
git clone 
cd 
```

## 运行代码
```bash
python smolvlm-api.py --image ~/Downloads/muggle-stack.png --stream
```