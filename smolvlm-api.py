#!/usr/bin/env python3
# vision_chat.py
import argparse
import base64
from pathlib import Path
import cv2
import tempfile
import os

from VIsionModel import VisionModel

# ----------------------------------------------------------------------
def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="文本 + 图像 多模态推理小工具",
    )
    p.add_argument("--model",  default="smolvlm:256m",
                   help="Vision 模型标签(ollama list)")
    p.add_argument("--image",  
                   help="要推理的图像文件")
    p.add_argument("--camera", action="store_true",
                   help="使用摄像头拍照")
    p.add_argument("--prompt", default="",
                   help="首次提问内容；留空则进入交互循环")
    p.add_argument("--stream", action="store_true",
                   help="是否流式输出")
    return p.parse_args()

# ----------------------------------------------------------------------
def capture_from_camera() -> str:
    """从摄像头捕获图像并返回base64编码"""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("无法打开摄像头")
    
    print("按空格键拍照，按ESC退出...")
    while True:
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("无法获取摄像头画面")
        
        cv2.imshow('Camera', frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC
            cap.release()
            cv2.destroyAllWindows()
            raise KeyboardInterrupt("用户取消拍照")
        elif key == 32:  # 空格
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # 保存临时文件
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        cv2.imwrite(tmp.name, frame)
        img_b64 = load_b64(Path(tmp.name))
        os.unlink(tmp.name)
        return img_b64

# ----------------------------------------------------------------------
def load_b64(img_path: Path) -> str:
    if not img_path.is_file():
        raise FileNotFoundError(img_path)
    return base64.b64encode(img_path.read_bytes()).decode()

# ----------------------------------------------------------------------
def main() -> None:
    args = build_args()

    # 1) 载入模型
    llm = VisionModel(vision_model_path=args.model, stream=args.stream)

    # 2) 获取图像
    try:
        if args.camera:
            img_b64 = capture_from_camera()
        elif args.image:
            img_b64 = load_b64(Path(args.image))
        else:
            raise ValueError("请指定 --image 或 --camera 参数")
    except Exception as e:
        print(f"错误: {e}")
        return

    # 3) 如果命令行已经给 prompt → 直接跑一次
    if args.prompt:
        run_once(llm, args.prompt, img_b64)
    else:
        # 否则进入 REPL
        try:
            while True:
                prompt = input("请输入内容（Ctrl-C 退出）：")
                run_once(llm, prompt, img_b64)
        except KeyboardInterrupt:
            print("\n已退出")


# ----------------------------------------------------------------------
def run_once(llm, prompt: str, img_b64: str) -> None:
    """调用 VisionModel 并打印结果（支持流式或非流式）"""
    for chunk in llm.generate(prompt, img_b64):
        print(chunk, end="", flush=True)
    print()


# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()