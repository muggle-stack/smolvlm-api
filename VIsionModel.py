from ollama import chat
import json

class VisionModel:
    def __init__(self, vision_model_path='smolvlm:256m', stream=True): # 你可以修改 llm_model_path 为自己的 ollama 通用大模型
        self._model_path = vision_model_path
        self._stream = stream
        pass

    # 获取聊天流的函数
    def get_chat_stream(self, text, messages, b64):
        messages.append({"role": "user", "content": text, "images": [b64]})
        stream = chat(
            model=self._model_path,
            messages=messages,
            stream=self._stream
        )
        return stream

    # 处理函数调用的主逻辑
    def generate(self, text, b64):
        self.b64 = b64
        self.messages = [
            {
                "role": "system",
                "content": (
                    "You are a visual assistant. Describe images clearly and answer questions based on visual content.\n"
                )
            }
        ] 
        
        # 获取聊天流
        stream = self.get_chat_stream(text, self.messages, self.b64)

        # 处理聊天流中的每一部分
        if self._stream:
            # 流式模式
            for chunk in stream:
                try:
                    if hasattr(chunk, 'message') and hasattr(chunk.message, 'content'):
                        content = chunk.message.content
                        if content:
                            yield content
                    elif isinstance(chunk, dict) and 'message' in chunk:
                        content = chunk['message'].get('content', '')
                        if content:
                            yield content
                except Exception as e:
                    print(f"处理chunk时出错: {e}")
                    continue
        else:
            # 非流式模式
            try:
                if hasattr(stream, 'message') and hasattr(stream.message, 'content'):
                    yield stream.message.content
                elif isinstance(stream, dict) and 'message' in stream:
                    yield stream['message'].get('content', '')
                else:
                    yield str(stream)
            except Exception as e:
                print(f"处理响应时出错: {e}")
                yield ""



