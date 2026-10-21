#!/usr/bin/env python3
"""
WD14 Tagger CLI - 独立的命令行提示词反推工具

用法:
    python wd14_tagger_cli.py --dir /path/to/images --model wd-v1-4-moat-tagger-v2
    python wd14_tagger_cli.py --dir /path/to/images --trigger_word "1girl, "

支持递归扫描子目录，为每张图片生成同名的 .txt 标签文件。
"""

import argparse
import csv
import os
import sys
import numpy as np
from PIL import Image

# 尝试导入 onnxruntime
try:
    import onnxruntime as ort
    from onnxruntime import InferenceSession
except ImportError:
    print("Error: onnxruntime is required. Install it with: pip install onnxruntime")
    sys.exit(1)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff", ".tif"}

# 按环境自动检测可用的 ORT provider，避免指定不存在 provider 导致回退到 CPU
_AVAILABLE_PROVIDERS = set(ort.get_available_providers())
_DEFAULT_PROVIDERS = (
    ["CANNExecutionProvider"] if "CANNExecutionProvider" in _AVAILABLE_PROVIDERS else []
) + (
    ["CUDAExecutionProvider"] if "CUDAExecutionProvider" in _AVAILABLE_PROVIDERS else []
) + (
    ["DmlExecutionProvider"] if "DmlExecutionProvider" in _AVAILABLE_PROVIDERS else []
) + ["CPUExecutionProvider"]  # CPU 永远兜底

# 默认配置
DEFAULT_MODEL = "wd-v1-4-moat-tagger-v2"
DEFAULT_THRESHOLD = 0.35
DEFAULT_CHARACTER_THRESHOLD = 0.85


def find_models_dir():
    """查找模型目录"""
    # 1. 命令行参数指定的路径
    # 2. 当前脚本目录下的 models
    # 3. ComfyUI custom_nodes/ComfyUI-WD14-Tagger/models (同级插件目录)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(script_dir, "models"),
    ]
    # 尝试从 custom_nodes 同级找 ComfyUI-WD14-Tagger (大小写不敏感)
    # 脚本可能在 custom_nodes/ComfyUI-bsk_UI/wd14/ 或 custom_nodes/ComfyUI-WD14-Tagger/ 下
    custom_nodes_dir = os.path.dirname(script_dir)
    if os.path.basename(custom_nodes_dir).lower().startswith("comfyui"):
        custom_nodes_dir = os.path.dirname(custom_nodes_dir)
    if os.path.isdir(custom_nodes_dir):
        for d in os.listdir(custom_nodes_dir):
            if d.lower() == "comfyui-wd14-tagger":
                candidates.append(os.path.join(custom_nodes_dir, d, "models"))
                break

    for d in candidates:
        if os.path.isdir(d) and any(f.endswith(".onnx") for f in os.listdir(d)):
            return d

    return candidates[0]  # 返回默认路径


def get_installed_models(models_dir):
    """获取已安装的模型列表"""
    if not os.path.isdir(models_dir):
        return []
    models = [m for m in os.listdir(models_dir)
              if m.endswith(".onnx") and os.path.exists(os.path.join(models_dir, os.path.splitext(m)[0] + ".csv"))]
    return models


def tag_image(image, model_path, csv_path, threshold=0.35, character_threshold=0.85,
              exclude_tags="", replace_underscore=True, trailing_comma=False,
              max_resolution=1280, enable_resize=True):
    """对单张图片进行标签反推"""
    # 推理前缩放：长边不超过 max_resolution，保持宽高比
    if enable_resize and max_resolution > 0:
        max_side = max(image.size)
        if max_side > max_resolution:
            ratio = max_resolution / max_side
            new_size = tuple(int(x * ratio) for x in image.size)
            image = image.resize(new_size, Image.LANCZOS)
    # 使用模块级预检测的 provider 列表
    providers = _DEFAULT_PROVIDERS
    model = InferenceSession(model_path, providers=providers)
    print(f"[WD14] Using provider: {model.get_providers()}")

    input_info = model.get_inputs()[0]
    height = input_info.shape[1]

    # 缩放并填充白色
    ratio = float(height) / max(image.size)
    new_size = tuple(int(x * ratio) for x in image.size)
    image = image.resize(new_size, Image.LANCZOS)
    square = Image.new("RGB", (height, height), (255, 255, 255))
    square.paste(image, ((height - new_size[0]) // 2, (height - new_size[1]) // 2))

    img_array = np.array(square).astype(np.float32)
    img_array = img_array[:, :, ::-1]  # RGB -> BGR
    img_array = np.expand_dims(img_array, 0)

    # 读取标签
    tags = []
    general_index = None
    character_index = None
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if general_index is None and row[2] == "0":
                general_index = reader.line_num - 2
            elif character_index is None and row[2] == "4":
                character_index = reader.line_num - 2
            if replace_underscore:
                tags.append(row[1].replace("_", " "))
            else:
                tags.append(row[1])

    label_name = model.get_outputs()[0].name
    probs = model.run([label_name], {input_info.name: img_array})[0]

    result = list(zip(tags, probs[0]))
    general = [item for item in result[general_index:character_index] if item[1] > threshold]
    character = [item for item in result[character_index:] if item[1] > character_threshold]

    all_tags = character + general
    remove = [s.strip() for s in exclude_tags.lower().split(",")]
    all_tags = [t for t in all_tags if t[0] not in remove]

    res = ("" if trailing_comma else ", ").join(
        item[0].replace("(", "\\(").replace(")", "\\)") + (", " if trailing_comma else "")
        for item in all_tags
    )
    return res


def process_directory(directory, models_dir, model_name, threshold, character_threshold,
                      exclude_tags, replace_underscore, trailing_comma, trigger_word,
                      recursive=False, overwrite=False, max_resolution=1280, enable_resize=True):
    """批量处理目录中的所有图片"""
    if not os.path.isdir(directory):
        print(f"Error: Directory does not exist: {directory}")
        return

    # 确定模型文件
    if not model_name.endswith(".onnx"):
        model_name += ".onnx"
    model_path = os.path.join(models_dir, model_name)
    csv_path = os.path.join(models_dir, os.path.splitext(model_name)[0] + ".csv")

    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        installed = get_installed_models(models_dir)
        if installed:
            print(f"Available models: {', '.join(installed)}")
        return

    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found: {csv_path}")
        return

    # 处理触发词
    if trigger_word:
        trigger_word = trigger_word.strip()
        if trigger_word and not trigger_word.endswith(","):
            trigger_word = trigger_word + ","

    # 收集图片文件
    image_files = []
    if recursive:
        for root, dirs, files in os.walk(directory):
            for f in sorted(files):
                if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS:
                    image_files.append(os.path.join(root, f))
    else:
        for f in sorted(os.listdir(directory)):
            if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS:
                image_files.append(os.path.join(directory, f))

    if not image_files:
        print(f"No image files found in: {directory}")
        return

    print(f"Found {len(image_files)} images")
    print(f"Model: {model_name}")
    print(f"Threshold: {threshold}, Character threshold: {character_threshold}")
    if trigger_word:
        print(f"Trigger word: {trigger_word}")
    print("-" * 60)

    success = 0
    skipped = 0
    failed = 0

    for i, filepath in enumerate(image_files, 1):
        txt_path = os.path.splitext(filepath)[0] + ".txt"

        # 跳过已存在的 txt 文件
        if os.path.exists(txt_path) and not overwrite:
            print(f"[{i}/{len(image_files)}] SKIP (txt exists): {os.path.basename(filepath)}")
            skipped += 1
            continue

        try:
            img = Image.open(filepath).convert("RGB")
            tags = tag_image(img, model_path, csv_path, threshold, character_threshold,
                             exclude_tags, replace_underscore, trailing_comma,
                             max_resolution, enable_resize)

            if trigger_word:
                final_tags = trigger_word + " " + tags
            else:
                final_tags = tags

            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(final_tags)

            success += 1
            rel_path = os.path.relpath(filepath, directory)
            print(f"[{i}/{len(image_files)}] OK: {rel_path}")
        except Exception as e:
            failed += 1
            print(f"[{i}/{len(image_files)}] FAIL: {os.path.basename(filepath)} - {e}")

    print("-" * 60)
    print(f"Done! Success: {success}, Skipped: {skipped}, Failed: {failed}")


def main():
    parser = argparse.ArgumentParser(
        description="WD14 Tagger CLI - 批量图片标签反推工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s --dir /path/to/images
  %(prog)s --dir /path/to/images --model wd-v1-4-moat-tagger-v2 --threshold 0.35
  %(prog)s --dir /path/to/images --trigger_word "1girl" --recursive
  %(prog)s --dir /path/to/images --overwrite

可用的 ORT providers: """ + ", ".join(ort.get_available_providers())
    )

    parser.add_argument("--dir", "-d", help="目标图片目录路径")
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL, help=f"模型名称 (默认: {DEFAULT_MODEL})")
    parser.add_argument("--models_dir", help="模型文件目录路径 (默认: 自动查找)")
    parser.add_argument("--threshold", "-t", type=float, default=DEFAULT_THRESHOLD,
                        help=f"通用标签阈值 (默认: {DEFAULT_THRESHOLD})")
    parser.add_argument("--character_threshold", "-ct", type=float, default=DEFAULT_CHARACTER_THRESHOLD,
                        help=f"角色标签阈值 (默认: {DEFAULT_CHARACTER_THRESHOLD})")
    parser.add_argument("--exclude_tags", "-e", default="", help="排除的标签，逗号分隔")
    parser.add_argument("--trigger_word", "-tw", default="", help="触发词，会添加到txt最前面")
    parser.add_argument("--replace_underscore", action="store_true", default=True,
                        help="替换下划线为空格 (默认开启)")
    parser.add_argument("--no_replace_underscore", action="store_true",
                        help="不替换下划线")
    parser.add_argument("--trailing_comma", action="store_true", default=False,
                        help="每个标签后添加逗号")
    parser.add_argument("--recursive", "-r", action="store_true",
                        help="递归扫描子目录")
    parser.add_argument("--overwrite", "-o", action="store_true",
                        help="覆盖已存在的txt文件")
    parser.add_argument("--max_resolution", type=int, default=1280,
                        help="推理前缩放图片长边最大分辨率 (默认: 1280)")
    parser.add_argument("--no_resize", action="store_true",
                        help="禁用推理前图片缩放")
    parser.add_argument("--list_models", action="store_true",
                        help="列出可用模型并退出")

    args = parser.parse_args()

    if not args.list_models and not args.dir:
        parser.error("the following arguments are required: --dir/-d")

    # 确定模型目录
    models_dir = args.models_dir or find_models_dir()

    # 列出模型
    if args.list_models:
        installed = get_installed_models(models_dir)
        if installed:
            print(f"Available models in {models_dir}:")
            for m in installed:
                print(f"  - {os.path.splitext(m)[0]}")
        else:
            print(f"No models found in {models_dir}")
            print("Please download models first.")
        return

    # 处理下划线替换参数
    replace_underscore = not args.no_replace_underscore

    process_directory(
        directory=args.dir,
        models_dir=models_dir,
        model_name=args.model,
        threshold=args.threshold,
        character_threshold=args.character_threshold,
        exclude_tags=args.exclude_tags,
        replace_underscore=replace_underscore,
        trailing_comma=args.trailing_comma,
        trigger_word=args.trigger_word,
        recursive=args.recursive,
        overwrite=args.overwrite,
        max_resolution=args.max_resolution,
        enable_resize=not args.no_resize,
    )


if __name__ == "__main__":
    main()