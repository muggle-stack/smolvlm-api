#!/usr/bin/env python3
# 指定脚本使用Python3解释器运行

# vision_chat.py
# 文件描述：文本+图像多模态推理工具

import argparse
# 导入argparse库，用于解析命令行参数
import base64
# 导入base64库，用于图像的base64编码转换
from pathlib import Path
# 导入Path类，用于便捷的文件路径处理
import cv2
# 导入OpenCV库，用于图像处理和摄像头捕获功能
import tempfile
# 导入tempfile库，用于创建临时文件
import os
# 导入os库，用于操作系统相关操作（如删除文件）

from VIsionModel import VisionModel
# 从VIsionModel模块导入VisionModel类，用于处理视觉模型推理

# ----------------------------------------------------------------------


def build_args() -> argparse.Namespace:
    # 创建命令行参数解析器
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        # 设置帮助信息格式，显示默认值
        description="文本 + 图像 多模态推理小工具",
        # 程序描述信息
    )
    p.add_argument("--model",  default="smolvlm:256m",
                   help="Vision 模型标签(ollama list)")
    # 添加--model参数：指定使用的视觉模型，默认值为"smolvlm:256m"
    p.add_argument("--image",
                   help="要推理的图像文件")
    # 添加--image参数：指定图像文件路径
    p.add_argument("--camera", action="store_true",
                   help="使用摄像头拍照")
    # 添加--camera参数：布尔值，指定是否使用摄像头捕获图像
    p.add_argument("--prompt", default="",
                   help="首次提问内容；留空则进入交互循环")
    # 添加--prompt参数：指定初始提问内容，默认空值
    p.add_argument("--stream", action="store_true",
                   help="是否流式输出")
    # 添加--stream参数：布尔值，指定是否使用流式输出结果
    return p.parse_args()
    # 解析命令行参数并返回结果

# ----------------------------------------------------------------------


def capture_from_camera() -> str:
    """从摄像头捕获图像并返回base64编码"""
    # 创建摄像头捕获对象，参数0表示默认摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        # 检查摄像头是否成功打开
        raise RuntimeError("无法打开摄像头")

    print("按空格键拍照，按ESC退出...")
    # 提示用户操作方法
    while True:
        # 循环读取摄像头画面
        ret, frame = cap.read()
        # 读取一帧图像，ret表示读取是否成功，frame是图像数据
        if not ret:
            # 如果读取失败，抛出异常
            raise RuntimeError("无法获取摄像头画面")

        cv2.imshow('Camera', frame)
        # 显示摄像头画面窗口
        key = cv2.waitKey(1) & 0xFF
        # 等待键盘输入，1毫秒超时

        if key == 27:  # ESC
            # 如果按下ESC键，释放资源并退出
            cap.release()
            cv2.destroyAllWindows()
            raise KeyboardInterrupt("用户取消拍照")
        elif key == 32:  # 空格
            # 如果按下空格键，跳出循环，准备保存图像
            break

    cap.release()
    # 释放摄像头资源
    cv2.destroyAllWindows()
    # 关闭所有OpenCV窗口

    # 压缩图片
    processed_frame = process_image(frame)

    # 保存临时文件
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        # 创建临时JPG文件，delete=False表示不自动删除
        cv2.imwrite(tmp.name, processed_frame)
        # 将捕获的帧写入临时文件
        img_b64 = load_b64(Path(tmp.name))
        # 读取临时文件并转换为base64编码
        os.unlink(tmp.name)
        # 删除临时文件
        return img_b64
        # 返回base64编码字符串

# ----------------------------------------------------------------------


def load_b64(img_path: Path) -> str:
    if not img_path.is_file():
        raise FileNotFoundError(img_path)
    
    # 新增：读取并处理图像
    image = cv2.imread(str(img_path))
    processed_image = process_image(image)
    
    # 保存处理后的图像到临时文件
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        cv2.imwrite(tmp.name, processed_image)
        img_b64 = base64.b64encode(Path(tmp.name).read_bytes()).decode()
        os.unlink(tmp.name)
        return img_b64

# ----------------------------------------------------------------------

def process_image(image, target_size=512):
    """处理图像: 截取中心正方形区域并调整至目标尺寸
    Args:
        image: OpenCV图像数组 (BGR格式)
        target_size: 目标尺寸(正方形边长)
    Returns:
        处理后的图像数组
    """
    # 获取图像尺寸
    height, width = image.shape[:2]

    # 计算中心正方形区域
    min_dim = min(height, width)
    start_x = (width - min_dim) // 2
    start_y = (height - min_dim) // 2

    # 裁剪中心区域
    cropped = image[start_y:start_y+min_dim, start_x:start_x+min_dim]

    # 调整大小至目标尺寸
    return cv2.resize(cropped, (target_size, target_size), interpolation=cv2.INTER_AREA)


# ----------------------------------------------------------------------
def main() -> None:
    args = build_args()
    # 解析命令行参数

    # 1) 载入模型
    llm = VisionModel(vision_model_path=args.model, stream=args.stream)
    # 创建VisionModel实例，传入模型路径和流式输出参数

    # 2) 获取图像
    try:
        if args.camera:
            # 如果指定了--camera参数
            img_b64 = capture_from_camera()
            # 调用摄像头捕获函数
        elif args.image:
            # 如果指定了--image参数
            img_b64 = load_b64(Path(args.image))
            # 调用图像加载函数
        else:
            # 如果既没有指定--camera也没有--image
            raise ValueError("请指定 --image 或 --camera 参数")
            # 抛出错误，要求用户指定图像来源
    except Exception as e:
        # 捕获所有异常
        print(f"错误: {e}")
        # 打印错误信息
        return
        # 退出程序

    # 3) 如果命令行已经给 prompt → 直接跑一次
    if args.prompt:
        # 如果指定了--prompt参数
        run_once(llm, args.prompt, img_b64)
        # 调用单次推理函数
    else:
        # 否则进入 REPL (交互式循环)
        try:
            while True:
                # 无限循环
                prompt = input("请输入内容（Ctrl-C 退出）：")
                # 获取用户输入
                run_once(llm, prompt, img_b64)
                # 调用单次推理函数
        except KeyboardInterrupt:
            # 捕获Ctrl-C中断
            print("\n已退出")
            # 打印退出信息


# ----------------------------------------------------------------------
def run_once(llm, prompt: str, img_b64: str) -> None:
    """调用 VisionModel 并打印结果（支持流式或非流式）"""
    for chunk in llm.generate(prompt, img_b64):
        # 遍历模型生成的结果块
        print(chunk, end="", flush=True)
        # 打印结果块，不换行，立即刷新缓冲区
    print()
    # 打印换行


# ----------------------------------------------------------------------
if __name__ == "__main__":
    # 当脚本直接运行时执行
    main()
    # 调用主函数
