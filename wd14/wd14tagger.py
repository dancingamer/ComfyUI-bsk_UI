# https://huggingface.co/spaces/SmilingWolf/wd-v1-4-tags

import comfy.utils
import asyncio
import aiohttp
import numpy as np
import csv
import os
import sys
import onnxruntime as ort
from onnxruntime import InferenceSession
from PIL import Image
from server import PromptServer
from aiohttp import web
import folder_paths
from .pysssss import get_ext_dir, get_comfy_dir, download_to_file, update_node_status, wait_for_async, get_extension_config, log
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))

config = get_extension_config()

# ===========================================================================
# torch_npu 初始化：昇腾 NPU 环境下必须先导入 torch_npu，CANNExecutionProvider
# 才能被 onnxruntime 识别。如果 ComfyUI 主程序已经导入了，这里跳过即可。
# ===========================================================================
# _TORCH_NPU_INITIALIZED = False


# def _ensure_torch_npu():
#     """
#     检测并初始化 torch_npu。

#     在昇腾 NPU 环境中，onnxruntime 的 CANNExecutionProvider 依赖 torch_npu
#     注册其 execution provider。如果 torch_npu 没有被导入，ort 就看不到 CANN。

#     由于 ComfyUI 的其他插件可能已经初始化了 torch_npu，这里采用安全重试策略：
#     先尝试导入，如果已经导入过则跳过，避免重复初始化导致冲突。
#     """
#     global _TORCH_NPU_INITIALIZED
#     if _TORCH_NPU_INITIALIZED:
#         return True

#     try:
#         import torch_npu  # noqa: F401
#         # 应用与 ComfyUI 插件相同的编译配置，避免 JIT 编译干扰
#         torch = __import__("torch")
#         torch.npu.config.allow_internal_format = False
#         torch.npu.set_compile_mode(jit_compile=False)
#         _TORCH_NPU_INITIALIZED = True
#         print("[WD14Tagger] torch_npu imported and initialized successfully")
#         return True
#     except ImportError:
#         # torch_npu 未安装，不是昇腾环境，正常跳过
#         print("[WD14Tagger] torch_npu not found, skipping NPU initialization (non-Ascend environment)")
#         return False
#     except Exception as e:
#         print(f"[WD14Tagger] Failed to initialize torch_npu: {e}")
#         return False


# # 在检测 provider 之前，先确保 torch_npu 已加载
# _ensure_torch_npu()


def _detect_ort_providers():
    """按环境自动检测可用的 ORT provider，避免指定不存在 provider 导致回退到 CPU。"""
    available = set(ort.get_available_providers())
    order = []
    if "CANNExecutionProvider" in available:
        order.append("CANNExecutionProvider")
    if "CUDAExecutionProvider" in available:
        order.append("CUDAExecutionProvider")
    if "DmlExecutionProvider" in available:
        order.append("DmlExecutionProvider")
    order.append("CPUExecutionProvider")  # CPU 永远兜底
    return order


defaults = {
    "model": "wd-v1-4-moat-tagger-v2",
    "threshold": 0.35,
    "character_threshold": 0.85,
    "replace_underscore": False,
    "trailing_comma": False,
    "exclude_tags": "",
    "ortProviders": _detect_ort_providers(),
    "HF_ENDPOINT": "https://huggingface.co"
}
defaults.update(config.get("settings", {}))

# 配置可能覆盖了 ortProviders（例如写死了 CUDAExecutionProvider），
# 必须在合并配置后重新检测，确保 provider 列表与实际环境一致。
if "ortProviders" in defaults:
    defaults["ortProviders"] = _detect_ort_providers()

if "wd14_tagger" in folder_paths.folder_names_and_paths:
    models_dir = folder_paths.get_folder_paths("wd14_tagger")[0]
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
else:
    models_dir = get_ext_dir("models", mkdir=True)
known_models = list(config["models"].keys())

log("Available ORT providers: " + ", ".join(ort.get_available_providers()), "DEBUG", True)
log("Using ORT providers: " + ", ".join(defaults["ortProviders"]), "DEBUG", True)

# 强制打印到 stdout，确保无论如何都能看到
print("[WD14Tagger] Available ORT providers:", ", ".join(ort.get_available_providers()))
print("[WD14Tagger] Effective ortProviders:", ", ".join(defaults["ortProviders"]))

def get_installed_models():
    models = filter(lambda x: x.endswith(".onnx"), os.listdir(models_dir))
    models = [m for m in models if os.path.exists(os.path.join(models_dir, os.path.splitext(m)[0] + ".csv"))]
    return models


async def tag(image, model_name, threshold=0.35, character_threshold=0.85, exclude_tags="", replace_underscore=True, trailing_comma=False, client_id=None, node=None, max_resolution=1280, enable_resize=True):
    if model_name.endswith(".onnx"):
        model_name = model_name[0:-5]
    installed = list(get_installed_models())
    if not any(model_name + ".onnx" in s for s in installed):
        await download_model(model_name, client_id, node)

    # 推理前缩放：长边不超过 max_resolution，保持宽高比
    if enable_resize and max_resolution > 0:
        max_side = max(image.size)
        if max_side > max_resolution:
            ratio = max_resolution / max_side
            new_size = tuple(int(x * ratio) for x in image.size)
            image = image.resize(new_size, Image.LANCZOS)

    name = os.path.join(models_dir, model_name + ".onnx")
    model = InferenceSession(name, providers=defaults["ortProviders"])

    # 打印实际使用的 provider，确认 NPU 是否生效
    actual_providers = model.get_providers()
    print(f"[WD14Tagger] Model loaded. Actual providers used: {actual_providers}")
    if "CANNExecutionProvider" in actual_providers:
        print("[WD14Tagger] ✓ NPU (CANN) inference is ACTIVE")
    elif "CPUExecutionProvider" in actual_providers:
        print("[WD14Tagger] ⚠ Falling back to CPU inference (NPU not available)")

    input = model.get_inputs()[0]
    height = input.shape[1]

    # Reduce to max size and pad with white
    ratio = float(height)/max(image.size)
    new_size = tuple([int(x*ratio) for x in image.size])
    image = image.resize(new_size, Image.LANCZOS)
    square = Image.new("RGB", (height, height), (255, 255, 255))
    square.paste(image, ((height-new_size[0])//2, (height-new_size[1])//2))

    image = np.array(square).astype(np.float32)
    image = image[:, :, ::-1]  # RGB -> BGR
    image = np.expand_dims(image, 0)

    # Read all tags from csv and locate start of each category
    tags = []
    general_index = None
    character_index = None
    with open(os.path.join(models_dir, model_name + ".csv")) as f:
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
    probs = model.run([label_name], {input.name: image})[0]

    result = list(zip(tags, probs[0]))

    # rating = max(result[:general_index], key=lambda x: x[1])
    general = [item for item in result[general_index:character_index] if item[1] > threshold]
    character = [item for item in result[character_index:] if item[1] > character_threshold]

    all = character + general
    remove = [s.strip() for s in exclude_tags.lower().split(",")]
    all = [tag for tag in all if tag[0] not in remove]

    res = ("" if trailing_comma else ", ").join((item[0].replace("(", "\\(").replace(")", "\\)") + (", " if trailing_comma else "") for item in all))

    print(res)
    return res


async def download_model(model, client_id, node):
    hf_endpoint = os.getenv("HF_ENDPOINT", defaults["HF_ENDPOINT"])
    if not hf_endpoint.startswith("https://"):
        hf_endpoint = f"https://{hf_endpoint}"
    if hf_endpoint.endswith("/"):
        hf_endpoint = hf_endpoint.rstrip("/")

    url = config["models"][model]
    url = url.replace("{HF_ENDPOINT}", hf_endpoint)
    url = f"{url}/resolve/main/"
    async with aiohttp.ClientSession(loop=asyncio.get_event_loop()) as session:
        async def update_callback(perc):
            nonlocal client_id
            message = ""
            if perc < 100:
                message = f"Downloading {model}"
            update_node_status(client_id, node, message, perc)

        try:
            await download_to_file(
                f"{url}model.onnx", os.path.join(models_dir,f"{model}.onnx"), update_callback, session=session)
            await download_to_file(
                f"{url}selected_tags.csv", os.path.join(models_dir,f"{model}.csv"), update_callback, session=session)
        except aiohttp.client_exceptions.ClientConnectorError as err:
            log("Unable to download model. Download files manually or try using a HF mirror/proxy website by setting the environment variable HF_ENDPOINT=https://.....", "ERROR", True)
            raise

        update_node_status(client_id, node, None)

    return web.Response(status=200)


@PromptServer.instance.routes.get("/pysssss/wd14tagger/tag")
async def get_tags(request):
    if "filename" not in request.rel_url.query:
        return web.Response(status=404)

    type = request.query.get("type", "output")
    if type not in ["output", "input", "temp"]:
        return web.Response(status=400)

    target_dir = get_comfy_dir(type)
    image_path = os.path.abspath(os.path.join(
        target_dir, request.query.get("subfolder", ""), request.query["filename"]))
    c = os.path.commonpath((image_path, target_dir))
    if os.path.commonpath((image_path, target_dir)) != target_dir:
        return web.Response(status=403)

    if not os.path.isfile(image_path):
        return web.Response(status=404)

    image = Image.open(image_path)

    models = get_installed_models()
    default = defaults["model"] + ".onnx"
    model = default if default in models else models[0]

    return web.json_response(await tag(image, model, client_id=request.rel_url.query.get("clientId", ""), node=request.rel_url.query.get("node", "")))

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff", ".tif"}


def batch_tag_directory(directory, model, threshold, character_threshold, exclude_tags="", replace_underscore=False, trailing_comma=False, trigger_word="", max_resolution=1280, enable_resize=True):
    """批量反推目标目录内所有图片文件并生成同名txt文件"""
    if not os.path.isdir(directory):
        print(f"[WD14Tagger] Batch directory does not exist: {directory}")
        return 0

    # 处理触发词：如果没有英文逗号则自动添加
    if trigger_word:
        trigger_word = trigger_word.strip()
        if trigger_word and not trigger_word.endswith(","):
            trigger_word = trigger_word + ","

    image_files = sorted([
        f for f in os.listdir(directory)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
    ])

    if not image_files:
        print(f"[WD14Tagger] No image files found in: {directory}")
        return 0

    print(f"[WD14Tagger] Found {len(image_files)} images in {directory}")
    count = 0
    for filename in image_files:
        filepath = os.path.join(directory, filename)
        txt_path = os.path.join(directory, os.path.splitext(filename)[0] + ".txt")
        try:
            img = Image.open(filepath).convert("RGB")
            tags_result = wait_for_async(lambda: tag(img, model, threshold, character_threshold, exclude_tags, replace_underscore, trailing_comma, max_resolution=max_resolution, enable_resize=enable_resize))
            # 拼接触发词
            if trigger_word:
                final_tags = trigger_word + " " + tags_result
            else:
                final_tags = tags_result
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(final_tags)
            count += 1
            print(f"[WD14Tagger] [{count}/{len(image_files)}] {filename} -> {os.path.basename(txt_path)}")
        except Exception as e:
            print(f"[WD14Tagger] Error processing {filename}: {e}")

    print(f"[WD14Tagger] Batch tagging complete: {count}/{len(image_files)} images processed")
    return count


class WD14Tagger:
    @classmethod
    def INPUT_TYPES(s):
        extra = [name for name, _ in (os.path.splitext(m) for m in get_installed_models()) if name not in known_models]
        models = known_models + extra
        return {"required": {
            "image": ("IMAGE", ),
            "model": (models, { "default": defaults["model"] }),
            "threshold": ("FLOAT", {"default": defaults["threshold"], "min": 0.0, "max": 1, "step": 0.05}),
            "character_threshold": ("FLOAT", {"default": defaults["character_threshold"], "min": 0.0, "max": 1, "step": 0.05}),
            "replace_underscore": ("BOOLEAN", {"default": defaults["replace_underscore"]}),
            "trailing_comma": ("BOOLEAN", {"default": defaults["trailing_comma"]}),
            "exclude_tags": ("STRING", {"default": defaults["exclude_tags"]}),
            "enabled": ("BOOLEAN", {"default": True}),
            "enable_resize": ("BOOLEAN", {"default": True}),
            "max_resolution": ("INT", {"default": 1280, "min": 64, "max": 4096, "step": 64}),
            "batch_dir": ("STRING", {"default": "", "multiline": False}),
            "trigger_word": ("STRING", {"default": "", "multiline": False}),
        }}

    RETURN_TYPES = ("STRING",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "tag"
    OUTPUT_NODE = True

    CATEGORY = "image"

    def tag(self, image, model, threshold, character_threshold, exclude_tags="", replace_underscore=False, trailing_comma=False, enabled=True, enable_resize=True, max_resolution=1280, batch_dir="", trigger_word=""):
        # 批量模式：如果提供了有效的目标目录，则执行批量反推
        if batch_dir and os.path.isdir(batch_dir):
            batch_tag_directory(batch_dir, model, threshold, character_threshold, exclude_tags, replace_underscore, trailing_comma, trigger_word, max_resolution, enable_resize)
            return {"ui": {"tags": ["[Batch mode] See console output for details."]}, "result": ([""],)}

        # 开关关闭时，返回空字符串
        if not enabled:
            return {"ui": {"tags": [""]}, "result": ([""],)}
        
        # 检查输入图像是否为空或无效
        if image is None:
            return {"ui": {"tags": [""]}, "result": ([""],)}
        
        try:
            tensor = image * 255
            tensor = np.array(tensor, dtype=np.uint8)
            
            # 检查 tensor 是否为空
            if tensor.size == 0 or tensor.shape[0] == 0:
                return {"ui": {"tags": [""]}, "result": ([""],)}
            
            pbar = comfy.utils.ProgressBar(tensor.shape[0])
            tags = []
            for i in range(tensor.shape[0]):
                image = Image.fromarray(tensor[i])
                tags.append(wait_for_async(lambda: tag(image, model, threshold, character_threshold, exclude_tags, replace_underscore, trailing_comma, max_resolution=max_resolution, enable_resize=enable_resize)))
                pbar.update(1)
            return {"ui": {"tags": tags}, "result": (tags,)}
            
        except Exception as e:
            # 捕获任何异常，返回空字符串而不是报错
            print(f"[WD14Tagger] Error during tagging: {e}")
            return {"ui": {"tags": [""]}, "result": ([""],)}


NODE_CLASS_MAPPINGS = {
    "WD14Tagger|pysssss": WD14Tagger,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "WD14Tagger|pysssss": "WD14 Tagger 🐍",
}