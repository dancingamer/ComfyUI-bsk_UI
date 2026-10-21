# SPDX-FileCopyrightText: 2026 灰暗x
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
ComfyUI bsk UI
# ==========================================================
# 훈련 완료 시 훈련 상태 자동 저장, 후속 훈련에 용이。
# 무료로 사용할 수 있으며, 당신의 AI도 재구축할 수 있지만, 원저자의 이름과 출처를 유지해주세요.。
# 내 B 즈음 공간: https://space.bilibili.com/26779709
# 당신이 할 수 있는 것x
# ==========================================================


ComfyUI를 위한 깨끗하고 사용자 정의 가능한 GUI 컨트롤 패널 확장。

하드웨어 모니터링 기능은 Crystools 플러그인의 구현을 참조했습니다:
- Crystools (https://github.com/crystian/ComfyUI-Crystools)
- Copyright (c) 2023 Crystian
- MIT License
일부 기능은 ComfyUI 구현 참조:
- ComfyUI (https://github.com/Comfy-Org/ComfyUI)
- Copyright (c) 2022 ComfyAI
- GPL-3.0 license

"""

import random
import json
import shutil
from pathlib import Path
import os, gc
import torch
import numpy as np

from comfy import model_management as mm

import sys
import ast
import math

try:
    import comfy.lora as _comfy_lora

    _original_model_lora_keys_clip = _comfy_lora.model_lora_keys_clip

    def _patched_model_lora_keys_clip(model, key_map={}):
        # 먼저 원본 함수를 호출하여 표준 키 매핑을 가져옴
        key_map = _original_model_lora_keys_clip(model, key_map)

        # 검사 Anima Qwen3 text encoder:
        # state_dict 키 형식: qwen3_06b.transformer.model.layers.X.self_attn.q_proj.weight
        sdk = model.state_dict().keys()

        # Qwen3의 TE 레이어 키 검색
        qwen3_te_keys = [k for k in sdk if k.endswith(".weight") and ".transformer.model.layers." in k]

        if qwen3_te_keys:
            # TE 모델의 전체 접두사 확인, 예시 "qwen3_06b.transformer"
            # 첫 번째 키에서 추출: qwen3_06b.transformer.model.layers.0.self_attn.q_proj.weight
            sample_key = qwen3_te_keys[0]
            # 추출 "qwen3_06b.transformer" 부분
            te_prefix_end = sample_key.find(".model.layers.")
            if te_prefix_end > 0:
                te_prefix = sample_key[:te_prefix_end]  # e.g. "qwen3_06b.transformer"

                for k in qwen3_te_keys:
                    # k = "qwen3_06b.transformer.model.layers.0.self_attn.q_proj.weight"
                    # 추출 "model.layers.0.self_attn.q_proj" 부분 (접두사 및 제거) .weight）
                    inner_path = k[len(te_prefix) + 1:-len(".weight")]  # "model.layers.0.self_attn.q_proj"

                    # sd-scripts 생성된 lora 키 형식: lora_te + inner_path 중간의 . 를 대체합니다 _
                    # e.g. lora_te_layers_0_self_attn_q_proj
                    # 참고: sd-scripts의 접두사는 입니다. "lora_te"，그리고 "model.layers.X.Y" 변화 "layers_X_Y"
                    # 즉 제거되었 "model." 접두사, .를 .로 대체 _
                    if inner_path.startswith("model.layers."):
                        lora_path = inner_path[len("model."):]  # "layers.0.self_attn.q_proj"
                    else:
                        lora_path = inner_path

                    lora_key_prefix = "lora_te_{}".format(lora_path.replace(".", "_"))

                    # 매핑 1: sd-scripts 형식 lora_te_layers_X_self_attn_q_proj -> 전체 모델 키
                    # key_map 의 key는 단순히 접두사(예: .alpha/.lokr_w1과 같은 접미사 제외)여야 합니다.），
                    # ComfyUI 의 load_lora()는 자동으로 찾습니다 {x}.alpha, {x}.lokr_w1 등
                    if lora_key_prefix not in key_map:
                        key_map[lora_key_prefix] = k

                    # Mapping 2: text_encoders.model.layers.X.Y Generic Format
                    # 기본 model_lora_keys_clip 매핑이 덮어씌워짐 text_encoders.qwen3_06b.transformer.model.layers.X.Y
                    # text_encoders.model.layers.X.Y와 유사하지만, 그렇지 않으면 보충합니다.
                    generic_key = "text_encoders.{}".format(inner_path)
                    if generic_key not in key_map:
                        key_map[generic_key] = k

        return key_map

    _comfy_lora.model_lora_keys_clip = _patched_model_lora_keys_clip

    print("[ComfyUI bsk UI] LoRA key 매핑이 패치됨: 지원 Anima Qwen3 text encoder (LoRA + LoKr)")
except Exception as e:
    print(f"[ComfyUI bsk UI] 파일 콘텐츠 해시 계산 (더 정확한 변경 감지를 위해): {e}")
# ===== End monkey patch =====

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))

# 릴리스 패키지에 배포되는 Rust(PyO3) 확장 바이너리를 실행 가능하도록 import。
# ComfyUI spec_from_file_location을 사용하여 플러그인을 로드하고 플러그인 디렉터리를 추가하지 않습니다. sys.path，
# 따라서 패키지 내에 컴파일된 확장 기능은 rust_ext/에 일괄적으로 배치되며（build_release에 의해 자동 패키징됨), 여기서는 명시적으로 추가 sys.path。
# 개발 환경에 이 디렉터리가 없을 경우 (확장 키 pip install로 site-packages에 추가), 존재하지 않으면 건너뛰고, 아무 영향도 미치지 않습니다.。
_bsk_rust_ext_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "rust_ext")
if os.path.isdir(_bsk_rust_ext_dir) and _bsk_rust_ext_dir not in sys.path:
    sys.path.insert(0, _bsk_rust_ext_dir)

# ===== 로컬 서브모듈 가져오기 도구 =====
# ComfyUI spec_from_file_location을 사용하여 이 플러그인을 로드하고 플러그인 디렉토리를 추가하지 않습니다. sys.path，
# 직접 사용 `from .xxx import ...` 일부 환경 (ComfyUI 버전 / 패키징 방식이 다름)）
# ImportError가 발생합니다. 그리고 import 실패 시, 아래에는 INPUT_TYPES가 없는 대체 클래스가 생성되고 등록됩니다
# NODE_CLASS_MAPPINGS，프론트엔드에서 오류가 발생합니다：
#   AttributeError: type object 'bsk_xxx' has no attribute 'INPUT_TYPES'
# 따라서 여기에서 세 번째 레벨 후퇴: 상대 가져오기 -> 절대 임포트 -> 파일 경로로 로드；
# 실제 INPUT_TYPES 클래스를 받기 전에는 성공으로 간주되지 않습니다(_WAN_NODES_OK 판단 참조).）。
_BSK_PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))

def _bsk_import_local(module_file, attr_names):
    """플러그인 디렉터리에서 하위 모듈을 가져오고 필요한 클래스를 추출합니다.。

    어느 단계든 실패하면 ImportError 발생 (호출자는 다운그레이드 처리 담당)），
    빈 껍데기 노드를 노드 클래스로 반환하지 않습니다。
    """
    import importlib
    import importlib.util

    target = os.path.join(_BSK_PLUGIN_DIR, "%s.py" % module_file)
    if not os.path.isfile(target):
        raise ImportError("찾을 수 없음 %s.py（%s）" % (module_file, target))

    errors = []
    mod = None

    # 1) 상대 가져오기: 플러그인이 정상 패키지로 로드될 때
    if __package__:
        try:
            mod = importlib.import_module("." + module_file, __package__)
        except Exception as e:
            errors.append("상대 가져오기: %r" % (e,))

    # 2) 절대 가져오기: 플러그인 디렉토리를 sys.path에 추가하여 모듈 이름으로 가져오기
    if mod is None:
        try:
            if _BSK_PLUGIN_DIR not in sys.path:
                sys.path.insert(0, _BSK_PLUGIN_DIR)
            mod = importlib.import_module(module_file)
        except Exception as e:
            errors.append("절대 임포트: %r" % (e,))

    # 3) 파일 경로를 직접 로드합니다: sys.path / 패키지 컨텍스트에 의존하지 않음
    if mod is None:
        unique_name = "bsk_local_%s" % module_file
        spec = importlib.util.spec_from_file_location(unique_name, target)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[unique_name] = mod
        spec.loader.exec_module(mod)

    values = {}
    missing = []
    for name in attr_names:
        cls = getattr(mod, name, None)
        # INPUT_TYPES가 정의된 클래스만 유효한 노드 클래스, INPUT_TYPES가 정의되지 않은 것은 임포트 실패로 간주
        if cls is None or not hasattr(cls, "INPUT_TYPES"):
            missing.append(name)
        else:
            values[name] = cls
    if missing:
        raise ImportError("%s.py 중요한 노드 클래스가 누락되었습니다: %s (임포트 과정): %s）"
                          % (module_file, ", ".join(missing), " | ".join(errors) or "없다"))
    return values

# Import Multi-Layer Image Loading Node
BSK_MultiLayerImageLoader = None
try:
    BSK_MultiLayerImageLoader = _bsk_import_local(
        "bsk_multi_layer_image", ["BSK_MultiLayerImageLoader"])["BSK_MultiLayerImageLoader"]
    print("[ComfyUI bsk UI] 여러 레이어 이미지 로드 노드가 로드됨")
except Exception as e:
    print(f"[ComfyUI bsk UI] 경고: 다중 레이어 이미지 로딩 노드 임포트 실패: {e}")

# Import WanVideo node (optional, do not report error if file does not exist）
_WAN_NODES_OK = False
try:
    _wan_nodes = _bsk_import_local("wan_nodes", [
        "bsk_WanVideoImageToVideoEncode",
        "bsk_WanVideoImageToVideoEncodeSelective",
        "bsk_WanVideoFrameMask",
    ])
    bsk_WanVideoImageToVideoEncode = _wan_nodes["bsk_WanVideoImageToVideoEncode"]
    bsk_WanVideoImageToVideoEncodeSelective = _wan_nodes["bsk_WanVideoImageToVideoEncodeSelective"]
    bsk_WanVideoFrameMask = _wan_nodes["bsk_WanVideoFrameMask"]
    _WAN_NODES_OK = True
    print("[ComfyUI bsk UI] WanVideo노드 로드됨")
except Exception as e:
    import traceback as _bsk_traceback
    print(f"[ComfyUI bsk UI] 경고: WanVideo 노드 임포트 실패 (관련 노드 등록 안 됨)）: {e}")
    _bsk_traceback.print_exc()
    bsk_WanVideoImageToVideoEncode = None
    bsk_WanVideoImageToVideoEncodeSelective = None
    bsk_WanVideoFrameMask = None
    _WAN_NODES_OK = False

# 임포트 마스크 디버깅 도구
DEBUG_MASK_ALPHA = False  # False로 설정하면 디버그 모드 끄기
if DEBUG_MASK_ALPHA:
    try:
        sys.path.insert(0, "/workspace")
        from debug_mask_alpha import debug_save_backup, debug_verify_png, debug_create_report
        print("[ComfyUI bsk UI] 오래된 버전의 설정을 로드하지 마십시오")
    except ImportError as e:
        print(f"[ComfyUI bsk UI] 경고: 래스터 디버그 도구 가져오기 실패: {e}")
        DEBUG_MASK_ALPHA = False

import comfy.utils

from comfy.utils import common_upscale
try:
    from server import PromptServer
except:
    PromptServer = None
import folder_paths

import comfy
import time
from PIL import Image

import torchvision.transforms as T
to_pil_image = T.ToPILImage()
# ComfyUI_StringCacheNode.py
# -*- coding: utf-8 -*-
import re


script_directory = os.path.dirname(os.path.abspath(__file__))

device = mm.get_torch_device()
offload_device = mm.unet_offload_device()

VAE_STRIDE = (4, 8, 8)
PATCH_SIZE = (1, 2, 2)
MAX_RESOLUTION=16384


EXTENSION_DIR = Path(__file__).parent
# 서버 설정 저장 디렉토리: ComfyUI 사용자 디렉토리로 이동하여 플러그인 재설치 시 설정 유실 방지
USER_PANEL_DIR = Path(folder_paths.base_path) / "user" / "default" / "bsk_panel"
# danbooru 증명 파일 구성 경로 (ComfyUI 사용자 디렉토리에 텍스트로 구성, 단일 사용자 시나리오에만 적용)）
DANBOORU_CFG_PATH = USER_PANEL_DIR / "danbooru_config.json"

# ===========================================================================
# bsk_panel/ 패널 설정 외에도 다른 기능 데이터가 저장되어 있음 (프롬프트 이미지 라이브러리, 패널 배경 이미지)、
# AnimaDex 캐시/데이터, danbooru 자격 증명…). 이러한 이름은 모든 '목록/설정 검사'에서
# 인터페이스에서 제외하고, 그렇지 않으면 구성 드롭다운에 혼입되고, 계산되어 하나의 범주로 처리되며, mtime가 변경되면 더 나빠집니다.
# autoLoadLastConfig를 "최신 설정"으로 자동 로드 (프론트엔드 에러: "데이터 타입 인식 불가")」）。
#
# 주의: 이러한 이름들은 대부분 sync_panel_configs_to_user_dir() 함수가 플러그인 디렉토리 panel/ 전체를 참조합니다.
# 사용자 디렉토리에 복사할 때 함께 가져오며, 사용자 디렉토리에 반드시 존재합니다. '사용자가 아직 생성하지 않음'을 통해 우회할 수 없습니다.。
# ===========================================================================
PANEL_RESERVED_DIRS = {
    "prompt_gallery",           # 프롬프트 라이브러리：index.json + images/ + thumbs/
    "backgrounds",              # 패널 배경 이미지（PANEL_BG_DIRNAME）
    "animadex-cache",           # AnimaDex 온라인 캐시 (포함) thumbs/）
    "animadex-data",            # AnimaDex 데이터 디렉토리/웨어하우스
    "AnimaDex",                 # AnimaDex 데이터 디렉토리 후보 경로
}
PANEL_RESERVED_FILES = {
    "danbooru_config.json",     # danbooru 증빙（DANBOORU_CFG_PATH）
    "animadex.json",            # AnimaDex 데이터 소스 설정（AD_CFG_PATH）
    "animadex_cn.json",         # AnimaDex 중국어 별칭 표 (사용자가 덮어쓸 수 있음)）
    "model-download-config.json",  # 모델 다운로드 설정 공장 기본 설정(프론트엔드는 내보내기용으로만 사용)）
    "H3.png",                   # 공장 기본 배경 이미지（PANEL_BG_BUILTIN_NAME）
    "ill_base.png",             # 공장 배경 이미지
}

def _load_danbooru_config():
    """danbooru 사용자 이름/API 키를 읽고, 파일이 없거나 손상된 경우 반환합니다. {}。"""
    try:
        if not DANBOORU_CFG_PATH.is_file():
            return {}
        import json as _json
        with open(DANBOORU_CFG_PATH, "r", encoding="utf-8") as f:
            data = _json.load(f)
        if not isinstance(data, dict):
            return {}
        return {"login": data.get("login", "") or "", "api_key": data.get("api_key", "") or ""}
    except Exception:
        return {}

def _save_danbooru_config(login, api_key):
    import json as _json
    USER_PANEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(DANBOORU_CFG_PATH, "w", encoding="utf-8") as f:
        _json.dump({"login": login or "", "api_key": api_key or ""}, f, ensure_ascii=False, indent=2)

# 플러그인 디렉토리 동기화 표시：{"copied": 0} 플러그인이 새로 업데이트되었음을 나타내며, 시작 시 panel/ 출공장 설정을 사용자 디렉터리로 복사하고 기본 배경 이미지를 초기화합니다.
PANEL_SYNC_FLAG = EXTENSION_DIR / "panel_sync_flag.json"

WEB_DIRECTORY = "js"
VERSION = "1.3.0"  # 버전 업그레이드


def sync_panel_configs_to_user_dir():
    """
    마이그레이션 시작 (유일한 로직): 플러그인 디렉토리 panel_sync_flag.json 파일 내 copied == 0 시，
    1) 플러그인 디렉토리 panel/ 아래의 공장 설정이 ComfyUI 사용자 디렉토리로 재귀적으로 복사됩니다.
       (USER_PANEL_DIR = <ComfyUI>/user/default/bsk_panel/)；
    2) 검사 사용자 폴더 배경 이미지 폴더（<ComfyUI>/user/default/bsk_video_studio/）배경 이미지 존재 여부，
       플러그인 디렉토리를 panel/H3.png로 복사하고 이름을 panel/H3.png로 변경합니다. bsk_bg.png；
    3) Write panel_sync_flag.json back {"copied": 1}，다운로드 / 내보내기: 파란 회색。
    copied != 0（파일이 없거나 손상된 경우 복사하지 않고, 기본 로직을 적용하지 않습니다.。

    덮어쓰기 세부사항：
    - PNG를 덮어쓰는 경우, 패키지에 동일한 이름의 _temp.json이 없으면 사용자 디렉터리에 동일한 이름을 삭제합니다. <stem>_temp.json，설정이 비어 있거나 유효하지 않습니다；
    - 시작 실행 파일 (예 .last_used_config.json)은 사용자 디렉터리가 없는 경우에만 복사하고, 덮어쓰기 시 건너뜁니다。
    """
    # 유일한 트리거 조건: 플러그인 디렉토리 마커 copied == 0
    try:
        with open(PANEL_SYNC_FLAG, "r", encoding="utf-8") as f:
            flag_val = json.load(f).get("copied")
    except Exception:
        flag_val = None
    if flag_val != 0:
        return

    src_dir = EXTENSION_DIR / "panel"
    try:
        USER_PANEL_DIR.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"[bsk UI] 프레임 파일 이름을 분석할 수 없음 (필요 {USER_PANEL_DIR}: {e}")
        return

    # 플러그인 디렉토리 공장 설정이 없음: 사용자 디렉토리 존재 확인만
    if not src_dir.exists():
        return

    # 재귀 덮어쓰기 복사
    for root, _dirs, files in os.walk(src_dir):
        rel = os.path.relpath(root, src_dir)
        dst_root = USER_PANEL_DIR if rel == "." else USER_PANEL_DIR / rel
        dst_root.mkdir(parents=True, exist_ok=True)
        for fn in files:
            src_path = os.path.join(root, fn)
            dst_path = dst_root / fn
            # 시작 실행 파일은 사용자가 없는 디렉터리가 없을 때만 복사하고 덮어쓰기 시 건너띕니다
            if fn.startswith(".") and dst_path.exists():
                continue
            try:
                shutil.copy2(src_path, dst_path)
            except Exception as e:
                print(f"[bsk UI] 설정 복사 실패 {src_path}: {e}")
                continue
            # PNG를 덮어쓰는 경우, 패키지에 동일한 이름의 _temp.json이 없으면 사용자 디렉터리에 동일한 이름을 삭제합니다. _temp.json
            if fn.lower().endswith(".png"):
                temp_name = f"{Path(fn).stem}_temp.json"
                if not os.path.exists(os.path.join(root, temp_name)):
                    user_temp = dst_root / temp_name
                    if user_temp.exists():
                        try:
                            user_temp.unlink()
                        except Exception:
                            pass

    # 기본 배경 이미지: 사용자 폴더 배경 이미지 디렉토리에 배경 이미지 파일이 없는 경우 panel/H3.png를 복사하고 이름을 변경합니다. bsk_bg.png
    ensure_default_bg()

    # 마크 표시, 이번 동기화 완료
    try:
        with open(PANEL_SYNC_FLAG, "w", encoding="utf-8") as f:
            json.dump({"copied": 1}, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[bsk UI] 플러그인 디렉토리 동기화 마커 실패: {e}")


def ensure_default_bg():
    """비디오 워크스테이션 기본 배경 이미지 초기화: 사용자 디렉토리 bsk_video_studio/ 아래에 bsk_bg.* 고정 파일명 이미지，
    plugin 디렉토리 panel/H3.png를 복사하고 이름을 panel/H3.png로 변경 bsk_bg.png。

    sync_panel_configs_to_user_dir에 의해 copied == 0 동기화 트리거 호출 시 호출; 이미 존재하면 건너듸, 절대 사용자 정의 배경 이미지 덮어쓰지 않음。
    """
    try:
        bg_dir = Path(folder_paths.base_path) / "user" / "default" / "bsk_video_studio"
        # 고정 파일 이름 배경 이미지 있음（bsk_bg.png/jpg/webp/...）→ 건너뛰기
        if bg_dir.exists() and list(bg_dir.glob("bsk_bg.*")):
            return
        src_bg = EXTENSION_DIR / "panel" / "H3.png"
        if not src_bg.exists():
            print("[bsk UI] 기본 배경 이미지 panel/H3.png를 찾을 수 없습니다. 배경 이미지 초기화 건너듸")
            return
        bg_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_bg, bg_dir / "bsk_bg.png")
        print("[bsk UI] 복사됨 기본 배경 이미지 panel/H3.png -> user/default/bsk_video_studio/bsk_bg.png")
    except Exception as e:
        print(f"[bsk UI] 기본 배경 이미지 로딩 실패: {e}")


# 모듈 로드 시 마이그레이션 실행 (라우팅 등록 전)）
sync_panel_configs_to_user_dir()

print(f"[ComfyUI bsk UI] Extension loading (v{VERSION})...")

from PIL.PngImagePlugin import PngInfo
import comfy.cli_args as args


class bsk_Math:
    """
    안전한 수학식 계산기
    변수 a, b, c, d 및 일반 연산자, 비교, 부울 연산 및 함수 지원 (min, max, round, sum, len, abs)
    입력은 숫자, 문자열일 수 있으며, 객체에 'shape' 속성이 있는 경우 (예: 텐서) 자동으로 shape 목록으로 변환됨
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("STRING", {"multiline": False, "default": ""}),
            },
            "optional": {
                # 사용 * 와일드카드 유형, INT/FLOAT/텐서 등 모든 입력 연결을 허용하여 유형 검사 오류를 방지합니다.
                # 워크플로우 편집기에서 연결되지 않았을 때 기본값으로 숫자 입력 필드를 생성 ( _inferWidgetType 참조) * 처리）
                "a": ("*", {"default": 0.0, "tooltip": "변수 a의 값 (표현식에서 사용 가능)）"}),
                "b": ("*", {"default": 0.0, "tooltip": "변수 b의 값 (표현식에서 사용 가능)）"}),
                "c": ("*", {"default": 0.0, "tooltip": "변수 c의 값 (표현식에서 사용 가능)）"}),
                "d": ("*", {"default": 0.0, "tooltip": "변수 d의 값 (표현식에서 사용 가능)）"}),
            },
        }

    RETURN_TYPES = ("INT", "FLOAT",)
    FUNCTION = "execute"
    CATEGORY = "bsk/utilities"

    def execute(self, value: str, a=0.0, b=0.0, c=0.0, d=0.0):
        # ---------- 사전 처리 입력 ----------
        def _prepare(var):
            """입력 표준화：shape→목록, 문자열→부동 소수점으로 변환 시도, 나머지는 변경되지 않음"""
            if hasattr(var, 'shape'):
                return list(var.shape)
            if isinstance(var, str):
                try:
                    return float(var)
                except ValueError:
                    return var   # 원래 문자열 유지, 후속 평가 오류 발생 자동 오류
            return var

        a = _prepare(a)
        b = _prepare(b)
        c = _prepare(c)
        d = _prepare(d)

        # ---------- 안전한 화이트리스트 함수 ----------
        SAFE_FUNCS = {
            'min': min,
            'max': max,
            'round': round,
            'sum': sum,
            'len': len,
            'abs': abs,
        }

        # ---------- 재귀 AST 인터프리터 ----------
        def _eval(node):
            """AST 노드 평가 순회, 백 목록 내 작업만 허용"""
            # --- 상수 ---
            if isinstance(node, ast.Constant):   # Python 3.8+ 일관성 있게 사용 Constant
                return node.value

            # --- 변수 ---
            if isinstance(node, ast.Name):
                name = node.id
                if name == 'a': return a
                if name == 'b': return b
                if name == 'c': return c
                if name == 'd': return d
                raise NameError(f"알 수 없는 변수: {name}")

            # --- 이원 연산 ---
            if isinstance(node, ast.BinOp):
                left = _eval(node.left)
                right = _eval(node.right)
                op_type = type(node.op)
                if op_type is ast.Add:      return left + right
                if op_type is ast.Sub:      return left - right
                if op_type is ast.Mult:     return left * right
                if op_type is ast.Div:      return left / right
                if op_type is ast.FloorDiv: return left // right
                if op_type is ast.Mod:      return left % right
                if op_type is ast.Pow:      return left ** right
                raise TypeError(f"지원하지 않는 이항 연산자: {op_type.__name__}")

            # --- 단순 연산 ---
            if isinstance(node, ast.UnaryOp):
                operand = _eval(node.operand)
                op_type = type(node.op)
                if op_type is ast.USub: return -operand
                if op_type is ast.UAdd: return +operand
                if op_type is ast.Not:  return not operand
                raise TypeError(f"지원하지 않는 단항 연산자: {op_type.__name__}")

            # --- 비교 연산 (체인 비교 지원) a < b < c）---
            if isinstance(node, ast.Compare):
                left = _eval(node.left)
                for op_node, comp in zip(node.ops, node.comparators):
                    right = _eval(comp)
                    op_type = type(op_node)
                    if op_type is ast.Eq:    ok = (left == right)
                    elif op_type is ast.NotEq: ok = (left != right)
                    elif op_type is ast.Lt:  ok = (left < right)
                    elif op_type is ast.LtE: ok = (left <= right)
                    elif op_type is ast.Gt:  ok = (left > right)
                    elif op_type is ast.GtE: ok = (left >= right)
                    else:
                        raise TypeError(f"지원하지 않는 비교 연산자: {op_type.__name__}")
                    if not ok:
                        return 0
                    left = right   # 오른쪽 값 전달, 체인 비교 정확하게 구현
                return 1

            # --- 부울 논리 (and, or) ---
            if isinstance(node, ast.BoolOp):
                values = [_eval(v) for v in node.values]
                if isinstance(node.op, ast.And): return all(values)
                if isinstance(node.op, ast.Or):  return any(values)
                raise TypeError(f"지원하지 않는 부울 연산자: {type(node.op).__name__}")

            # --- 함수 호출 (백 목록) ---
            if isinstance(node, ast.Call):
                func_name = node.func.id
                if func_name not in SAFE_FUNCS:
                    raise ValueError(f"금지된 함수: {func_name}")
                args = [_eval(arg) for arg in node.args]
                return SAFE_FUNCS[func_name](*args)

            # --- 하드웨어 액세스 (상수 인덱스만 지원)) ---
            if isinstance(node, ast.Subscript):
                container = _eval(node.value)
                if not isinstance(node.slice, ast.Constant):
                    raise ValueError("인덱스는 상수만 지원")
                idx = node.slice.value
                return container[idx]

            # --- 지원되지 않는 구조 ---
            raise ValueError(f"지원하지 않는 표현식 유형: {type(node).__name__}")

        # ---------- 파싱 및 평가 ----------
        try:
            parsed = ast.parse(value, mode='eval')
            result = _eval(parsed.body)
        except Exception as e:
            # 오류 발생 시 명확하게 알리고 안전하게 다운그레이드 0
            print(f"[bsk_Math] 표현 오류：'{value}' → {e}")
            return (0, 0.0)

        # ---------- 결과 후처리 ----------
        if isinstance(result, (int, float)):
            if math.isnan(result):
                result = 0.0
        else:
            # 비숫자 결과 시도 변환 실패, 값 설정0
            try:
                result = float(result)
            except (TypeError, ValueError):
                result = 0.0

        return (round(result), float(result),)

# 워크플로우 정보 (workflowInfo) / 모델 다운로드 설정 (modelDownloads)을 JPG에 embed 할 때의 볼륨 제한
# （UTF-8 Byte count, discard field if exceeding limit, ensure not to overflow EXIF segment）
_BSK_JPG_WORKFLOW_INFO_LIMIT = 4000
_BSK_JPG_MODEL_DOWNLOADS_LIMIT = 6000


def _bsk_slim_panel_config_for_jpg(workflow_json):
    """
    워크플로우의 bsk_PanelConfigSaver(패널 구성 정보) 노드의 panel_config 瘦신：
    보존 탭(레이블 구조) + 노드ID 순서 목록), currentTab, workflowInfo (워크플로우 정보，
    (제목/저자/설명/사용법/링크/배경/스타일 등) 및 modelDownloads (모델 다운로드 설정)，
    다운로드 소스/모델 유형/디렉토리/파일명/저자 정보를 포함하고 실행 중 상태 및 대용량 base64 헤어 스타일을 버립니다.），
    모두 볼륨 보호막 적용。
    Discard cardValues and all node parameter data (the workflow itself includes all modified parameters)，
    프론트엔드 로딩 시 setWorkflow에서 워크플로우 카드 초기화, 패널 설정에서 중복 저장 없이）。
    노드는 자체적으로 유지되고, frontend _extractPanelConfigFromWorkflow가 슬림화된 설정을 여전히 추출할 수 있습니다。
    """
    try:
        wf = json.loads(workflow_json)
        if not isinstance(wf, dict):
            return workflow_json
        changed = False
        for node in wf.values():
            if not isinstance(node, dict) or node.get("class_type") != "bsk_PanelConfigSaver":
                continue
            inputs = node.get("inputs") or {}
            raw = inputs.get("panel_config")
            if not raw:
                continue
            try:
                cfg = json.loads(raw)
            except Exception:
                continue
            if not isinstance(cfg, dict):
                continue
            # 슬림화: 탭 유지(노드 ID 순서 포함)）+ currentTab + workflowInfo + modelDownloads
            slim = {}
            if isinstance(cfg.get("tabs"), list):
                slim_tabs = []
                for t in cfg["tabs"]:
                    if not isinstance(t, dict):
                        continue
                    slim_tabs.append({
                        "id": t.get("id"),
                        "name": t.get("name", ""),
                        "nodeIds": t.get("nodeIds") or [],
                    })
                slim["tabs"] = slim_tabs
            if cfg.get("currentTab") is not None:
                slim["currentTab"] = cfg["currentTab"]
            # 워크플로우 정보 (제목/저자/설명/링크/배경/스타일 등), 볼륨 보호 방지 및 JPG 초과 방지.
            if isinstance(cfg.get("workflowInfo"), dict) and cfg["workflowInfo"]:
                try:
                    info_json = json.dumps(cfg["workflowInfo"], ensure_ascii=False)
                    if len(info_json.encode("utf-8")) <= _BSK_JPG_WORKFLOW_INFO_LIMIT:
                        slim["workflowInfo"] = cfg["workflowInfo"]
                except Exception:
                    pass
            # 모델 다운로드 설정: 복원 다운 행에 필요한 모든 필드 유지，
            # Discard status (runtime state, fixed reset to idle during loading) authorAvatarData
            # （base64 프로필 데이터, 크기가 클 수 있으며 복구에 필수적이지 않음; authorAvatar가 원격 URL인 경우 유지합니다.）。
            if isinstance(cfg.get("modelDownloads"), list) and cfg["modelDownloads"]:
                slim_dl = []
                for it in cfg["modelDownloads"]:
                    if not isinstance(it, dict):
                        continue
                    row = {k: it.get(k, "") for k in (
                        "model_type", "subdir", "url", "releasePageUrl", "filename",
                        "actual_filename", "actual_model_type", "actual_subdir",
                    )}
                    row["authorName"] = it.get("authorName", "")
                    if isinstance(it.get("authorLinks"), list):
                        row["authorLinks"] = it["authorLinks"]
                    else:
                        row["authorLinks"] = []
                    row["authorDesc"] = it.get("authorDesc", "")
                    avatar = it.get("authorAvatar", "")
                    # 원격 URL 프로필만 유지; base64 데이터 URL은 대용량 데이터로 간주하고 버립니다
                    if isinstance(avatar, str) and avatar and not avatar.startswith("data:"):
                        row["authorAvatar"] = avatar
                    if row.get("model_type") or row.get("subdir") or row.get("filename"):
                        slim_dl.append(row)
                if slim_dl:
                    try:
                        dl_json = json.dumps(slim_dl, ensure_ascii=False)
                        if len(dl_json.encode("utf-8")) <= _BSK_JPG_MODEL_DOWNLOADS_LIMIT:
                            slim["modelDownloads"] = slim_dl
                    except Exception:
                        pass
            if not slim:
                continue
            inputs["panel_config"] = json.dumps(slim, ensure_ascii=False)
            changed = True
        if changed:
            return json.dumps(wf)
        return workflow_json
    except Exception:
        return workflow_json


def _bsk_build_jpg_workflow_exif(workflow_json):
    """
    워크플로우 JSON을 압축하여 JPG에 씁니다. EXIF UserComment（tag 0x9286）。
    형식: "ASCII\\x00\\x00\\x00" + "BSK_WORKFLOW:" + base64(zlib(workflow_json))
    프론트엔드 js/workflow_editor.js 및 js/panel.js의 _getJpgTextChunks를 이 형식으로 역으로 압축 해제합니다.。
    EXIF APP1 64KB 제한 (최대 콘텐츠 데이터 65533 바이트, EXIF 헤더/IFD 구조 여유)），
    초과 시간 동안 None 반환 (임베딩 건너뛰기, 오류 발생 안 함)）。
    """
    import zlib, base64
    try:
        raw = workflow_json.encode("utf-8")
        payload = b"BSK_WORKFLOW:" + base64.b64encode(zlib.compress(raw, 9))
        user_comment = b"ASCII\x00\x00\x00" + payload
        if len(user_comment) > 60000:  # EXIF 헤더/IFD 구조 여유 공간 확보, APP1 구간 초과 방지 64KB
            return None
        exif = Image.Exif()
        exif[0x9286] = user_comment
        return exif
    except Exception:
        return None


def bilateral_filter_pil(img, diameter: int = 9,
                         sigmaColor: float = 75,
                         sigmaSpace: float = 75):
    """PIL 이미지에 대해 이중 보간 필터링 (경계 보존 노이즈 제거)）。

    cv2 호출 지점에서 느슨하게 로드하여 opencv가 설치되지 않은 경우 플러그인 전체가 실패로 import되지 않도록 합니다.。
    그레이스케일 이미지는 L 모드를 유지; 알파 채널이 있는 이미지는 필터링 후 원본으로 다시 덮어씁니다。
    """
    import cv2 as cv

    diameter = int(round(float(diameter)))
    if diameter < 1:
        raise ValueError("diameter 필수 >= 1 정수")
    sigmaColor = float(sigmaColor)
    sigmaSpace = float(sigmaSpace)
    if sigmaColor <= 0 or sigmaSpace <= 0:
        raise ValueError("sigmaColor / sigmaSpace 음수일 필요 없음")

    src_mode = img.mode
    alpha = img.getchannel("A") if src_mode in ("RGBA", "LA", "PA") else None

    rgb = img.convert("RGB")
    bgr = cv.cvtColor(np.ascontiguousarray(np.array(rgb)), cv.COLOR_RGB2BGR)
    bgr = cv.bilateralFilter(bgr, diameter, sigmaColor, sigmaSpace)
    out = Image.fromarray(cv.cvtColor(bgr, cv.COLOR_BGR2RGB))

    if alpha is not None:
        out = out.convert("RGBA")
        out.putalpha(alpha)
    elif src_mode == "L":
        out = out.convert("L")
    return out


# ===== 파일 이름 접두사 시간/타임스탬프 지원 =====
# 공식 folder_paths.get_save_image_path는 지원 %year%/%month%/%day%/%hour%/%minute%/%second%
# 이러한 고정 플레이스홀더는 Unix 타임스탬프도 아니고 사용자 지정 날짜 형식을 지원하지 않습니다. 여기는 공식 로직으로 들어가기 전에 먼저 대체합니다.：
#   %timestamp% / %timestamp_s%  -> 10 초당 1초 Unix 타임스탬프
#   %timestamp_ms%               -> 13 비트 밀리초 단위 unix 타임스탬프
#   중국어 "타임스탬프"」                -> 10 초당 1초 Unix 타임스탬프
#   %date:<형식>%                -> 사용자 정의 형식으로 날짜 및 시간을 출력하며, 형식 문자는 자유롭게 조합할 수 있습니다.
#   중국어 "시간」                  -> 기본 2자리 연-월-일 (예: 26-10-04）
# 날짜 형식 문자(NET / Java 스타일, 대소문자 구분)）：
#   yyyy=4위 연  yy=2위 연 | MM=2비트 월  M=무 선행 0 월 | dd=2위치  d=무 선행 0 일
#   HH=24시간 2자리  H=무 선행 0 시 | hh=12시간 2자리  h=전위 0시 없음  tt=AM/PM
#   mm=2비트 분할  m=무 선행 0 분 | ss=2초/초  s=무 선행 0 초 | fff/ff/f=밀리초(3/2/1비트)
# 나머지 문자 (-, _ . 공백, 중국어 등)는 그대로 유지합니다 (예:) %date:yyyy-MM-dd_HH-mm-ss%。
# 시간과 타임스탬프는 한 번만 가져와 모든 이미지 프리픽스를 일치시켜 카운터 로직이 혼란하지 않도록 보장 (배치 포함)。
_BSK_DATE_TOKEN_RE = re.compile(r"%date:([^%]*)%")

_BSK_DATE_TOKEN_KEYS = (
    "yyyy", "fff", "yy", "MM", "dd", "HH", "hh", "tt", "mm", "ss", "ff",
    "M", "d", "H", "h", "m", "s", "f",
)


def _bsk_format_datetime(fmt, dt):
    """.NET/Java 스타일 날짜 문자열을 dt로 렌더링하여 텍스트로 변환합니다。"""
    ms = dt.microsecond // 1000
    values = {
        "yyyy": f"{dt.year:04d}",
        "yy": f"{dt.year % 100:02d}",
        "MM": f"{dt.month:02d}",
        "M": str(dt.month),
        "dd": f"{dt.day:02d}",
        "d": str(dt.day),
        "HH": f"{dt.hour:02d}",
        "H": str(dt.hour),
        "hh": f"{dt.hour % 12 or 12:02d}",
        "h": str(dt.hour % 12 or 12),
        "tt": "AM" if dt.hour < 12 else "PM",
        "mm": f"{dt.minute:02d}",
        "m": str(dt.minute),
        "ss": f"{dt.second:02d}",
        "s": str(dt.second),
        "fff": f"{ms:03d}",
        "ff": f"{ms // 10:02d}",
        "f": str(ms // 100),
    }
    keys = sorted(_BSK_DATE_TOKEN_KEYS, key=len, reverse=True)
    out = []
    i = 0
    total = len(fmt)
    while i < total:
        for key in keys:
            if fmt.startswith(key, i):
                out.append(values[key])
                i += len(key)
                break
        else:
            out.append(fmt[i])
            i += 1
    return "".join(out)


def _bsk_resolve_filename_prefix_tokens(filename_prefix):
    if not isinstance(filename_prefix, str) or not filename_prefix:
        return filename_prefix
    has_date = "%date:" in filename_prefix
    has_stamp = ("%timestamp" in filename_prefix) or ("타임스탬프" in filename_prefix)
    has_time = "시간" in filename_prefix
    if not (has_date or has_stamp or has_time):
        return filename_prefix

    now_ts = time.time()
    import datetime as _dt
    now_dt = _dt.datetime.now()

    if has_date:
        filename_prefix = _BSK_DATE_TOKEN_RE.sub(
            lambda m: _bsk_format_datetime(m.group(1), now_dt), filename_prefix
        )
    if has_stamp:
        ts_s = str(int(now_ts))
        ts_ms = str(int(now_ts * 1000))
        # 먼저 더 긴 토큰을 대체하여 %timestamp%가 오작동하는 것을 방지합니다. %timestamp_ms% / %timestamp_s%
        filename_prefix = filename_prefix.replace("%timestamp_ms%", ts_ms)
        filename_prefix = filename_prefix.replace("%timestamp_s%", ts_s)
        filename_prefix = filename_prefix.replace("%timestamp%", ts_s)
        filename_prefix = filename_prefix.replace("타임스탬프", ts_s)
    # 「"시간"은 "타임스탬프" 뒤에 대체해야 하며, 그렇지 않으면 "타임스탬프"가 "26-10-04 스탬프"로 잘못 해석됩니다.」
    if has_time:
        filename_prefix = filename_prefix.replace("시간", _bsk_format_datetime("yy-MM-dd", now_dt))
    return filename_prefix


class BSK_SaveImage:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4

    def _image_pad(self, image, pad_left, pad_right, pad_top, pad_bottom, padding_mode, pad_color, keep_proportion):
        """Fallback image padding when ImagePadKJ is not available."""
        import torch
        b, h, w, c = image.shape
        new_h = h + pad_top + pad_bottom
        new_w = w + pad_left + pad_right
        padded = torch.zeros((b, new_h, new_w, c), dtype=image.dtype, device=image.device)
        if isinstance(pad_color, (list, tuple)):
            color_tensor = torch.tensor(pad_color, dtype=image.dtype, device=image.device).view(1, 1, 1, -1)
        else:
            color_tensor = torch.tensor(pad_color, dtype=image.dtype, device=image.device).view(1, 1, 1, -1)
        if padding_mode == "edge":
            # Edge padding
            if pad_top > 0:
                edge_top = image[:, :1, :, :]
                padded[:, :pad_top, :, :] = edge_top.repeat(1, pad_top, 1, 1)
            if pad_bottom > 0:
                edge_bottom = image[:, -1:, :, :]
                padded[:, -pad_bottom:, :, :] = edge_bottom.repeat(1, pad_bottom, 1, 1)
            if pad_left > 0:
                edge_left = image[:, :, :1, :]
                padded[:, :, :pad_left, :] = edge_left.repeat(1, 1, pad_left, 1)
            if pad_right > 0:
                edge_right = image[:, :, -1:, :]
                padded[:, :, -pad_right:, :] = edge_right.repeat(1, 1, pad_right, 1)
            # Corners
            if pad_top > 0 and pad_left > 0:
                corner = image[:, :1, :1, :]
                padded[:, :pad_top, :pad_left, :] = corner.repeat(1, pad_top, pad_left, 1)
            if pad_top > 0 and pad_right > 0:
                corner = image[:, :1, -1:, :]
                padded[:, :pad_top, -pad_right:, :] = corner.repeat(1, pad_top, pad_right, 1)
            if pad_bottom > 0 and pad_left > 0:
                corner = image[:, -1:, :1, :]
                padded[:, -pad_bottom:, :pad_left, :] = corner.repeat(1, pad_bottom, pad_left, 1)
            if pad_bottom > 0 and pad_right > 0:
                corner = image[:, -1:, -1:, :]
                padded[:, -pad_bottom:, -pad_right:, :] = corner.repeat(1, pad_bottom, pad_right, 1)
        else:
            if pad_top > 0:
                padded[:, :pad_top, :, :] = color_tensor.repeat(b, pad_top, 1, 1)
            if pad_bottom > 0:
                padded[:, -pad_bottom:, :, :] = color_tensor.repeat(b, pad_bottom, 1, 1)
            if pad_left > 0:
                padded[:, :, :pad_left, :] = color_tensor.repeat(b, 1, pad_left, 1)
            if pad_right > 0:
                padded[:, :, -pad_right:, :] = color_tensor.repeat(b, 1, pad_right, 1)
        # Center the original image
        padded[:, pad_top:pad_top+h, pad_left:pad_left+w, :] = image
        return padded, None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "【입력 이미지】저장할 이미지 텐서（B×H×W×C，값 범위 0-1). 배치 입력을 지원합니다: 각 배치는 개별 파일로 생성되며, 파일 이름에 %batch_num%는 배차 번호(0부터 시작)로 대체됩니다）。"
                }),
                "filename_prefix": ("STRING", {
                    "default": "BSK",
                    "tooltip": "파일 이름 접두사. 예: sub/BSK 는 하위 폴더를 만들고 Prefix_00001_.png 처럼 번호가 붙습니다. %date:yyyy-MM-dd_HH-mm-ss%, %timestamp%, %year%, %month%, %day%, %hour%, %minute%, %second%, %width%, %height%, %batch_num% 를 쓸 수 있습니다."
                }),
                "save_as_jpg": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "【포맷 스위치] 닫기 = PNG（손실 없음, 크기가 큼, 투명 채널 지원); 켜짐 = JPG（Lossy, small volume). JPG does not support transparency, alpha images will be automatically compressed into RGB; workflow data changes to EXIF UserComment write, limit of about 64KB, exceeds automatically skips embedding。"
                }),
                "jpg_quality": ("INT", {
                    "default": 95, "min": 1, "max": 100, "step": 1,
                    "tooltip": "【JPG 품질] 1-100, 숫자가 높을수록 선명하고, 볼륨이 커집니다. 95는 거의 눈에 보이지 않는 수준입니다. 상단의 save_as_jpg를 켜는 경우에만 적용되며, PNG를 저장할 때는 이 파라미터가 무시됩니다.。"
                }),
                "embed_workflow_in_png": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "【내부 워크플로우 삽입] 전체 워크플로우를 이미지 메타데이터에 작성하여 나중에 ComfyUI로 그래프를 다시 불러오기 용이하게 합니다: PNG는 tEXt 블록에 저장, JPG는 EXIF UserComment(zlib 압축)에 저장. 종료 시 둘 다 저장하지 않아 이미지 더 깨끗하고 크기 작게 유지됩니다. 주의: ComfyUI 시작 시 --disable-metadata 옵션이 함께 사용되면 이 스위치는 무효합니다(어떤 메타데이터도 저장되지 않음)）。"
                }),
                "save_workflow": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "【워크플로우 txt]을 저장하고 이미지 외에 다시 '동명'으로 내보냅니다. + _w 파일 이름이 .txt이고, 내용이 전체 워크플로우 JSON이며, 이미지를 직접 확인하거나 백업하는 데 유용합니다. 이미지 자체에 영향을 주지 않으며, 상단 스위치와 독립적입니다.。"
                }),
                "save_txt": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "【프롬프트를 txt로 저장합니다[저장]. 동일한 이름의 .txt 기록을 프롬프트로 내보냅니다. 콘텐츠는 other_text를 우선적으로 사용하고, other_text가 비어 있으면 전체 워크플로우 프롬프트 JSON을 저장합니다. 두 소스가 모두 비어 있으면 파일을 생성하지 않습니다.。"
                }),
            },
            "optional": {
                "other_text": ("STRING", {
                    "default": "",
                    "forceInput": True,  # 주요 파라미터: 연결 입력 허용
                    "tooltip": "【사용자 지정 프롬프트[저장]. 선택 사항이며, 연결 입력(forceInput)을 지원합니다. 비어 있지 않으면 save_txt에 쓸 내용이며, 비어 있으면 save_txt가 워크플로우 프롬프트 JSON을 저장합니다. 이미지 처리에는 참여하지 않으며 워크플로우 내포에 영향을 미치지 않습니다.。"
                }),
                # ===== 양방향 필터링 노이즈 제거 (경계 보존 노이즈 제거）=====
                "denoise_bilateral": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "【이중 필터링 노이즈 제거 스위치: 활성화하면 메타데이터 작성 및 디스크에 쓰기 전에 각 이미지에 대해 이중 필터링을 수행합니다. 평탄한 영역(입자 노이즈, 색상 블록)을 부드럽게 하고 동시에 가장자리가 흐릿해지지 않도록 보호합니다. 작은 노이즈가 있는 이미지에 적합합니다. 비활성화하면 완전히 처리하지 않고, 0 비용입니다. 활성화되면 아래 세 개의 파라미터가 표시됩니다.。"
                }),
                "denoise_diameter": ("INT", {
                    "default": 9, "min": 1, "max": 25, "step": 1,
                    "tooltip": "【필터 커널 직경] 계산에 참여하는 이웃 직경, 1-25. 클수록 참조 범위가 넓고 노이즈 제거가 강하지만 시간이 크게 증가합니다 (약간으로) O(d²)）。5-9 권장; 15 초과하면 대형 이미지에서 뚜렷하게 느려짐. denoise_bilateral만 활성화 시에만 적용。"
                }),
                "denoise_sigma_color": ("FLOAT", {
                    "default": 75.0, "min": 1.0, "max": 255.0, "step": 1.0,
                    "tooltip": "【색상 유사도 σ】1-255，‘색상 차이가 얼마나 되는지’를 결정합니다. 값이 클수록 더 큰 차이를 보이는 픽셀도 함께 평균됩니다. → 노이즈 제거 강하지만, 뭉개짐과 그라데이션 디테일이 손상될 수 있습니다. 값이 작을수록 보수적이며, 매우 가까운 색상만 부드럽게 합니다. 50-100 권장, sigma_space와 유사한 값。"
                }),
                "denoise_sigma_space": ("FLOAT", {
                    "default": 75.0, "min": 1.0, "max": 255.0, "step": 1.0,
                    "tooltip": "【공간 거리 σ】1-255，결정 '얼만큼의 픽셀이 가중치에 참여할 수 있는지'. 값이 클수록 평균 범위가 넓어지고 노이즈가 더 부드러워지지만 동시에 더 느립니다. 값이 작을수록 더 국소화되고 가장자리도 더 날카로워집니다. denoise_bilateral만 활성화될 때만 적용。"
                }),
            },
            "hidden": {
                "prompt": "PROMPT", 
                "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("filename_without_ext", "workflow_json")
    OUTPUT_TOOLTIPS = (
        "【확장자가 없는 파일 이름: 이 저장 파일 이름(경로 및 .png/.jpg 제외)은 output 디렉토리에 직접 읽을 수 있습니다. 일괄 저장 시 마지막 파일 이름을 반환합니다.。",
        "【워크플로우 JSON】이 실행 워크플로우의 JSON 문자열 (압축되지 않은 원본), 다른 노드에 연결하여 사용할 수 있습니다. 가져오지 못한 경우 빈 문자열입니다.。"
    )
    
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "image"
    DESCRIPTION = "입력 이미지 저장, 워크플로우 삽입/프롬프트 저장 선택 가능, JPG 압축 및 바이너리 필터 노이즈 감소 지원; 파일 이름 접두사 지원 %date:<형식>%（사용자 지정 날짜 시간, 형식 기호는 임의로 조합 가능 (%timestamp%/%timestamp_ms% (unix 타임스탬프) 및 한국어 ‘시간’, ‘타임스탬프’). 출력 파일 이름(확장자 제외)은 워크플로우 JSON 문자열과 일치합니다.。"

    def save_images(self, images, filename_prefix="BSK", other_text="", save_txt=False, 
                   save_workflow=False, embed_workflow_in_png=True, save_as_jpg=False, jpg_quality=95,
                   denoise_bilateral=False, denoise_diameter=9,
                   denoise_sigma_color=75.0, denoise_sigma_space=75.0,
                   prompt=None, extra_pnginfo=None):
        filename_prefix += self.prefix_append
        # 먼저 시간/타임스탬프 플레이스홀더 확장 (공식 folder_paths는 지원 안 함)）
        filename_prefix = _bsk_resolve_filename_prefix_tokens(filename_prefix)
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0]
        )
        
        results = list()
        final_filename_without_ext = ""
        final_workflow_text = ""
        
        # 호환 모드: PromptServer에서 가져옴
        try:
            # 큰 카드 오른쪽 상단 부동 닫기 버튼）
            import comfy.options
            is_metadata_disabled = getattr(comfy.options.args, "disable_server_info", 
                                          getattr(comfy.options.args, "disable_metadata", False))
        except:
            is_metadata_disabled = False

        # 사전 작업 흐름 데이터 추출
        if extra_pnginfo is not None:
            for x in extra_pnginfo:
                if x == "workflow":
                    final_workflow_text = json.dumps(extra_pnginfo[x])
                    break

        for (batch_number, image) in enumerate(images):
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

            # 양방향 필터 노이즈 감소 (보이드 데노이즈), 쓰기 전에 메타데이터와 디스크에 저장하기 전에 처리합니다.
            if denoise_bilateral:
                try:
                    img = bilateral_filter_pil(img, denoise_diameter, denoise_sigma_color, denoise_sigma_space)
                except Exception as e:
                    print(f"[BSK_SaveImage] 양방향 필터링 노이즈 감소 실패, 노이즈 감소를 건너뛰고 직접 저장: {e}")

            metadata = None

            # PNG tEXt 블록으로 메타데이터 삽입; JPG는 tEXt를 사용하지 않고 EXIF UserComment를 사용합니다 (참조) _bsk_build_jpg_workflow_exif）。
            # 사용 가능한 변수를 사용하여 판단
            if (not is_metadata_disabled) and (not save_as_jpg):
                metadata = PngInfo()
                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        if x == "workflow":
                            # 부울 값에 따라 워크플로우 삽입 여부 결정 (PNG만 적용)）
                            if embed_workflow_in_png:
                                metadata.add_text(x, final_workflow_text)
                        else:
                            metadata.add_text(x, json.dumps(extra_pnginfo[x]))

            # 처리 파일 이름
            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            ext = ".jpg" if save_as_jpg else ".png"
            file = f"{filename_with_batch_num}_{counter:05}_.{ext[1:]}"  # 원 보존 "_." 프리피 약속
            base_filename = os.path.splitext(file)[0]
            final_filename_without_ext = base_filename
            full_filepath = os.path.join(full_output_folder, file)

            # 이미지 저장
            if save_as_jpg:
                # JPEG 알파 채널을 지원하지 않음, 변환 RGB
                save_img = img.convert("RGB") if img.mode in ("RGBA", "LA", "P") else img
                jpg_exif = None
                # 작업 흐름 (zlib 압축 + base64）저장된 설정/배치' 탭: 독립적인 스크롤 레이아웃, 내부 배치 목록 스크롤 아닌 전체 페이지 스크롤）
                if (not is_metadata_disabled) and embed_workflow_in_png and final_workflow_text:
                    # JPG EXIF 크기 제한이 있습니다: bsk_PanelConfigSaver(패널 구성 정보) 노드를 먼저 줄입니다.
                    jpg_workflow_text = _bsk_slim_panel_config_for_jpg(final_workflow_text)
                    jpg_exif = _bsk_build_jpg_workflow_exif(jpg_workflow_text)
                if jpg_exif is not None:
                    save_img.save(full_filepath, format="JPEG", quality=int(jpg_quality), optimize=True, exif=jpg_exif)
                else:
                    save_img.save(full_filepath, format="JPEG", quality=int(jpg_quality), optimize=True)
            else:
                img.save(full_filepath, pnginfo=metadata, compress_level=self.compress_level)
            
            # 처리 프롬프트 저장 로직
            if save_txt:
                # other_text가 비어 있지 않은 경우에만 txt 파일을 생성합니다. 비어 있으면 생성하지 않습니다.
                actual_prompt = other_text
                if actual_prompt.strip():
                    prompt_txt_path = os.path.join(full_output_folder, f"{base_filename}.txt")
                    with open(prompt_txt_path, "w", encoding="utf-8") as f:
                        f.write(actual_prompt)

            # 워크플로우 저장 로직 수정 (변수명）
            if save_workflow and final_workflow_text:
                workflow_txt_path = os.path.join(full_output_folder, f"{base_filename}_w.txt")
                with open(workflow_txt_path, "w", encoding="utf-8") as f:
                    f.write(final_workflow_text)

            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
            counter += 1

        return {
            "ui": { "images": results }, 
            "result": (final_filename_without_ext, final_workflow_text)
        }

# 텍스트 편집기 노드

class bsk_StringMergeWithSwitch:
    """
    무제한 수의 문자열 입력 및 스위치 제어를 지원하는 병합 노드.。
    프론트엔드는 연결에 따라 자동으로 입력 추가/삭제（inputN + switchN）。
    백엔드에서 최대 MAX_INPUTS 개의 optional 입력 미리 정의, 연결되지 않은 입력은 기본값으로 설정。
    merge_order: 병합 순서 기록, 예 "1,2,3" 또는 "3,2,1"，데이터 상태/탐지/링크，
                 이 순서대로 문자열을 병합합니다. 프론트엔드에서 드래그 앤 드롭 정렬 시 이 필드에 쓰기。
    preserve_newlines: 줄 바꿈 유지 여부, 기본값 False (줄 바꿈 제거), True 시 줄 바꿈 유지。
    """
    MAX_INPUTS = 10  # 최대 입력 수량 지원

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        optional = {}
        for i in range(1, s.MAX_INPUTS + 1):
            optional[f"input{i}"] = ("STRING", {"default": "", "forceInput": False})
            optional[f"switch{i}"] = ("BOOLEAN", {"default": True})
        return {
            "required": {},
            "optional": {
                **optional,
                "merge_order": ("STRING", {"default": ""}),
                "preserve_newlines": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("merged_string",)
    FUNCTION = "merge_strings"
    CATEGORY = "utils/text"

    def merge_strings(self, **kwargs):
        merge_order_str = kwargs.get("merge_order", "")
        preserve_newlines = kwargs.get("preserve_newlines", False)

        # 파싱 병합 순서
        if merge_order_str and merge_order_str.strip():
            try:
                order = [int(x.strip()) for x in merge_order_str.split(",") if x.strip()]
            except (ValueError, TypeError):
                order = list(range(1, self.MAX_INPUTS + 1))
        else:
            # 순서 정보가 없으며, 기본 1, 2, 3... 순서대로 진행.
            order = list(range(1, self.MAX_INPUTS + 1))

        parts = []
        for i in order:
            if i < 1 or i > self.MAX_INPUTS:
                continue
            if f"input{i}" not in kwargs:
                continue
            text = kwargs.get(f"input{i}", "")
            switch = kwargs.get(f"switch{i}", True)
            if switch and text:
                clean_text = text.strip()
                if clean_text:
                    if not preserve_newlines:
                        clean_text = clean_text.replace("\n", " ").replace("\r", "")
                        # 여러 공백을 합침
                        clean_text = re.sub(r' +', ' ', clean_text).strip()
                    clean_text = clean_text.rstrip(", ")
                    parts.append(clean_text)
        result = ", ".join(parts)
        return (result,)

class MultiLineStringEditor:
    """
    ComfyUI노드: 다중 줄 문자열 편집기
    기능：
    1. 다중 줄 문자열 입력 지원
    2. 지원Ctrl+/빠른키 주석/해제 현재 행
    3. 주석 처리된 줄 및 빈 줄 자동 필터링
    4. 인라인 주석 필터링 지원 (쉼표로 구분된 프롬프트)）
    5. strip_newlines를 통해 줄바꿈 제거 여부 제어 (기본값: 줄바꿈 유지)）
    
    주석 필터링 전략：
    - 전체 주석: 줄 시작에#（공백 무시 (전체 줄 필터링)
    - 행 내 주석: 행 시작 부분에 없음#，그러나 중간 행에#，단순 필터링#프롬프트 텍스트
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True, 
                    "default": "",
                    "placeholder": "여러 줄 텍스트 입력Ctrl+/주석/해제 현재 행\n인라인 주석 지원：aa, bb, # cc, dd → aa, bb, dd",
                    "dynamicPrompts": False
                }),
                "separator": ("STRING", {
                    "default": " ",
                    "placeholder": "컴팩트 모드로 전환“삭제 줄바꿈”실시간 적용, 기본값은 공백）"
                }),
                "strip_newlines": ("BOOLEAN", {
                    "default": False,
                    "label_on": "삭제 줄바꿈",
                    "label_off": "줄 바꿈 유지",
                    "tooltip": "닫힘(기본): 줄 바꿈을 유지하고, 필터링된 각 행을 독립적인 행으로 만듭니다. 활성화하면 줄 바꿈을 제거하고 모든 행을 구분 기호로 연결하여 한 줄로 만듭니다."
                })
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "process_text"
    CATEGORY = "텍스트 처리"

    def process_text(self, text, separator=" ", strip_newlines=False):
        """
        텍스트 처리: 주석 및 빈 줄 제거

        매개변수：
            strip_newlines: 줄 바꿈 제거 여부
                - False(기본값): 줄 바꿈 제거 X, 필터링된 각 줄을 독립적인 줄로 만들기 (PIL 사용) \\n 연결）
                - True：줄 바꿈을 제거하고, 필터링된 모든 행을 구분자를 사용하여 한 줄로 연결합니다
                기존 워크플로우에 해당 입력이 없을 경우 기본값으로 처리하며, 에러가 발생하지 않습니다.

        필터 전략：
        1. 전체 행 주석: 줄이 시작되면#（공백을 무시하고, 줄 전체를 건너뜁니다
        2. 인라인 주석: 줄이 시작될 때#，하지만 업계에 있다#，필터 포함#프롬프트 텍스트
        3. 빈 줄: 완전히 빈 줄 건너뛰기
        """
        # 오래된 워크플로우와 호환: 누락되거나 값 오류가 있는 경우 기본값으로 처리(줄 바꿈 제거 안 함).
        if strip_newlines is None:
            strip_newlines = False
        elif isinstance(strip_newlines, str):
            strip_newlines = strip_newlines.strip().lower() in ("true", "1", "yes", "on")
        else:
            strip_newlines = bool(strip_newlines)
        # 호환 예외 구분 기호
        if not isinstance(separator, str):
            separator = " "

        lines = text.split('\n')
        processed_lines = []
        
        for line in lines:
            stripped = line.lstrip()
            
            # 전략 1: 전체 행 주석 - 행 시작 부분에#，행 건너뛰기
            if stripped.startswith('#'):
                continue
            
            # 전략 3: 완전 빈 줄 건너뛰기
            if not stripped:
                continue
            
            # 전략 2: 인라인 주석 - 인라인에 있습니다#，필터 포함#프롬프트 텍스트
            if '#' in line:
                processed_line = self._filter_inline_comments(line)
                if processed_line.strip():  # 필터링 후 비어 있지 않은지 확인
                    processed_lines.append(processed_line)
            else:
                # 주석 없음, 직접 추가
                processed_lines.append(line)
        
        # 줄 바꿈 제거 시 구분 문자열로 한 줄로 만듭니다, 그렇지 않으면 줄 바꿈을 유지하여 각 줄을 독립적으로 만듭니다.
        joiner = separator if strip_newlines else "\n"
        result = joiner.join(processed_lines)
        
        return (result,)
    
    def _filter_inline_comments(self, line):
        """
        필터 내부 포함된#프롬프트(콤마로 구분)）
        원래 쉼표 구조 유지
        
        예시：
        "aa, bb, cc, # dd, ff, gg, #hhh" → "aa, bb, cc, ff, gg,"
        "cat, dog, #bird, fish" → "cat, dog, fish"
        """
        # 쉼표로 분할, 각 부분 뒤에 쉼표가 있는지 기록
        parts = []
        current = ""
        for char in line:
            if char == ',':
                parts.append((current, True))  # (내용, 쉼표가 뒤따름)
                current = ""
            else:
                current += char
        parts.append((current, False))  # 마지막 부분이 쉼표로 끝나지 않았습니다
        
        # 필터 포함#의 프롬프트 텍스트, 하지만 쉼표 정보 유지
        filtered_parts = []
        for content, has_comma in parts:
            stripped = content.lstrip()
            # 검사 중인 프롬프트가 주석 처리되었는지 확인#시작）
            if stripped.startswith('#'):
                # 주석 처리된 프롬프트, 내용 건너뛰기, 하지만 이전 요소 뒤에 쉼표가 있는지 유지
                if filtered_parts and has_comma:
                    # 쉼표를 이전 요소로 이동합니다
                    filtered_parts[-1] = (filtered_parts[-1][0], True)
                continue
            else:
                filtered_parts.append((content, has_comma))
        
        # 다시 연결
        result = ""
        for content, has_comma in filtered_parts:
            result += content
            if has_comma:
                result += ","
        
        return result


"""
ComfyUI 문자열 처리 노드 - Prompt String Processor
입력 프롬프트 문자열 조건 제거 및 대체 작업 수행

규칙 형식：
- if: abc: 삭제           # 완전히 abc와 일치하는 프롬프트 삭제
- if: abc: 포함 삭제       # 전체 단어 abc를 포함하는 프롬프트를 삭제합니다
- if: abc: 대체로:xxx     # 완전 일치하는 'abc' 프롬프트를 대체합니다 xxx
- if: abc: 포함하기 대체:xxx # 전체 단어 abc를 포함하는 프롬프트를 대체합니다 xxx

주의: 일치하는 것은 완전한 단어여야 하며, 단어의 일부를 일치시키지 않습니다 (예: abc는 abcccc 또는) xabc）
"""

class PromptStringProcessor:
    """
    프롬프트 카드 문자열 처리 노드
    지원 조건 제거 및 대체 작업
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "쉼표로 구분된 프롬프트 텍스트 문자열"
                }),
                "rules": ("STRING", {
                    "default": "# 사용법 예시（# 시작 줄이 적용되지 않음）\n# if:핵심: 여기 사용\n# if:키워드: 삭제 포함\n# if:키워드: 대체됨: 새 단어\n# if:키워드: 대체 포함 시: 신어",
                    "multiline": True,
                    "tooltip": "처리 규칙, 각 줄에 한 규칙，# 시작 동작 주석이 적용되지 않음"
                }),
            },
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("processed_prompt",)
    FUNCTION = "process"
    CATEGORY = "string"
    DESCRIPTION = "프롬프트 문자열에 조건 제거 및 대체 작업을 수행합니다"
    
    def process(self, prompt: str, rules: str) -> tuple:
        # 규칙이 없으면 원본 문자열 반환
        if not rules.strip():
            return (prompt,)
        
        # 파싱 프롬프트 (쉼표로 분할, 공백 제거)）
        prompt_parts = [p.strip() for p in prompt.split(',') if p.strip()]
        
        # 각 규칙 파싱 및 적용
        for rule_line in rules.strip().split('\n'):
            rule_line = rule_line.strip()
            # 건너뛰기 # 시작 부분의 주석 줄과 빈 줄
            if not rule_line or rule_line.startswith('#'):
                continue
            
            prompt_parts = self._apply_rule(prompt_parts, rule_line)
        
        # 다시 조합 문자열
        result = ', '.join(prompt_parts)
        return (result,)
    
    def _apply_rule(self, prompt_parts: list, rule_line: str) -> list:
        parsed = self._parse_rule(rule_line)
        if parsed is None:
            return prompt_parts
        
        action, keyword, replacement = parsed
        
        if action == "remove_exact":
            return [p for p in prompt_parts if p != keyword]
        elif action == "remove_contains":
            return [p for p in prompt_parts if not self._contains_word(p, keyword)]
        elif action == "replace_exact":
            return [replacement if p == keyword else p for p in prompt_parts]
        elif action == "replace_contains":
            return [replacement if self._contains_word(p, keyword) else p for p in prompt_parts]
        
        return prompt_parts
    
    def _parse_rule(self, rule_line: str):
        """
        파싱 규칙 행
        if: abc: 및 if:abc: 두 가지 형식 지원; 키워드 공백 포함 가능 (예 "aaa abc asd"）
        """
        # 정규 표현식: if: [키워드] : [행동 내용]]
        # 키워드는 콜론으로만 구분해야 하며, 다른 문자는 사용할 수 없음 "\s+" 분리，
        # 그렇지 않으면 "if: aaa abc asd: 대체로: xxx" 첫 번째 공백에서 잘릴 수 있으며 규칙이 무효화됨
        pattern = r'^if:\s*(.+?)\s*:\s*(.+)$'
        match = re.match(pattern, rule_line)
        
        if not match:
            return None
        
        keyword = match.group(1).strip()
        rest = match.group(2).strip()
        
        # 운영 유형을 판단합니다 (길이 우선 매칭, 방지)"포함하기 대체"에"대체로"잘라내기）
        if rest.startswith("포함하기 대체:"):
            replacement = rest[len("포함하기 대체:"):].strip()
            return ("replace_contains", keyword, replacement)
        elif rest.startswith("대체로:"):
            replacement = rest[len("대체로:"):].strip()
            return ("replace_exact", keyword, replacement)
        elif rest == "포함 삭제":
            return ("remove_contains", keyword, None)
        elif rest == "삭제":
            return ("remove_exact", keyword, None)
            
        return None
    
    def _contains_word(self, text: str, word: str) -> bool:
        """
        텍스트에 지정된 단어가 포함되어 있는지 확인 (완전 단어 일치)）
        """
        pattern = r'\b' + re.escape(word) + r'\b'
        return bool(re.search(pattern, text, re.IGNORECASE))


# ---------------------------
# 도구 함수: 쉼표를 표준화 추가
# ---------------------------
def append_with_comma(original, addition):
    """addition을 쉼표로 구분하여 original 뒤에 추가"""
    if not original:
        return addition
    if original.endswith(', ') or original.endswith(','):
        return original + addition
    else:
        return original + ', ' + addition
class LoraLoaderWithPath:
    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The diffusion model the LoRA will be applied to."}),
                "clip": ("CLIP", {"tooltip": "The CLIP model the LoRA will be applied to."}),
                "lora_name": (folder_paths.get_filename_list("loras"), {"tooltip": "The name of the LoRA."}),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01, "tooltip": "How strongly to modify the diffusion model. This value can be negative."}),
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01, "tooltip": "How strongly to modify the CLIP model. This value can be negative."}),
            },
            "optional": {
                "lora_path": ("STRING", {"default": "", "tooltip": "Absolute path to LoRA file. If provided and file exists, this takes priority over lora_name."}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    OUTPUT_TOOLTIPS = ("The modified diffusion model.", "The modified CLIP model.")
    FUNCTION = "load_lora"

    CATEGORY = "loaders"
    DESCRIPTION = "LoRAs are used to modify diffusion and CLIP models, altering the way in which latents are denoised such as applying styles. Multiple LoRA nodes can be linked together."
    SEARCH_ALIASES = ["lora", "load lora", "apply lora", "lora loader", "lora model"]

    def load_lora(self, model, clip, lora_name, strength_model, strength_clip, lora_path=""):
        if strength_model == 0 and strength_clip == 0:
            return (model, clip)

        # 확인할 LoRA 경로
        import os
        use_path = None
        
        # lora_path가 제공되고 파일이 존재하면 우선 사용합니다
        if lora_path and lora_path.strip():
            if os.path.exists(lora_path):
                use_path = lora_path
            else:
                print(f"Warning: Provided LoRA path does not exist: {lora_path}")
                print(f"Falling back to lora_name: {lora_name}")
        
        # lora_path가 유효하지 않으면 사용lora_name
        if use_path is None:
            use_path = folder_paths.get_full_path_or_raise("loras", lora_name)

        # 검사 캐시된LoRA
        lora = None
        if self.loaded_lora is not None:
            if self.loaded_lora[0] == use_path:
                lora = self.loaded_lora[1]
            else:
                self.loaded_lora = None

        # ロードLoRA
        if lora is None:
            lora = comfy.utils.load_torch_file(use_path, safe_load=True)
            self.loaded_lora = (use_path, lora)

        model_lora, clip_lora = comfy.sd.load_lora_for_models(model, clip, lora, strength_model, strength_clip)
        return (model_lora, clip_lora)


class bsk_MultiLoraLoaderWithPath:
    def __init__(self):
        self.loaded_loras = {}

    @classmethod
    def INPUT_TYPES(s):
        # 최소 입력: 모델 / clip 연결과 여러 줄의 LoRA 목록 텍스트만 유지。
        # 모든 스위치, 가중치, 순서는 텍스트에 작성 (프런트엔드 카드 또는 원본 프런트엔드 편집)）。
        return {
            "required": {},
            "optional": {
                "model": ("MODEL", {"tooltip": "The diffusion model the LoRAs will be applied to."}),
                "clip": ("CLIP", {"tooltip": "The CLIP model the LoRAs will be applied to."}),
                # 선택 사항에 넣습니다: 이전 워크플로우가 이 입력이 누락된 경우 빈 문자열로 처리하고 필수 유효성 검사 실패를 트리거하지 않습니다
                "lora_text": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": (
                        "LoRA 텍스트 목록 (여러 줄), 두 형식 지원：\n"
                        "1) 한 줄에 하나: 이름: 모델 가중치, CLIP 가중치, on/off (가중치/스위치 생략 가능, 하나의 가중치만 제공 시) clip=model，스위치 쓰지 않았으면 기본 활성화）\n"
                        "2) A1111 스타일：<lora:이름: 가중치> 또는 <이름: 가중치>（단 가중치 시 clip=model，기본적으로 활성화; 또한 지원 <lora:이름: 가중치 1: 가중치2>）\n"
                        "첫 번째 쓰기 줄 master=on/off 전체 스위치 제어; 쓰지 않으면 기본적으로 켜짐。"
                    ),
                }),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    OUTPUT_TOOLTIPS = ("The modified diffusion model.", "The modified CLIP model.")
    FUNCTION = "load_multi_lora"
    CATEGORY = "loaders"
    DESCRIPTION = "LoRA를 여러 개 한 번에 로드할 때, '이름:모델 가중치,clip 가중치,on/off' 또는 A1111 스타일로 각 줄에 입력하십시오.「<lora:이름: 가중치>」，개별적으로 스위치/가중치 설정 가능, 첫 번째 줄에 작성 가능 master=on/off 제어 총 스위치。"

    def _load_single_lora(self, use_path):
        """개별 LoRA 파일 로드 (캐시 포함)）"""
        if use_path in self.loaded_loras:
            return self.loaded_loras[use_path]

        print(f"[bsk_MultiLora] Loading: {use_path}")
        lora = comfy.utils.load_torch_file(use_path, safe_load=True)
        self.loaded_loras[use_path] = lora
        return lora

    @staticmethod
    def _to_float_or_none(s):
        """유연한 부동 소수점 파싱: 실패 시 반환 None"""
        if s is None:
            return None
        try:
            return float(str(s).strip())
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _lora_item_from_token(name, weight_parts):
        """A1111 스타일 <...> 가중치 파싱: 단 가중치일 때 clip=model，기본 활성화"""
        model = 1.0
        clip = 1.0
        if weight_parts:
            w = bsk_MultiLoraLoaderWithPath._to_float_or_none(weight_parts[0])
            if w is not None:
                model = w
                clip = w
        if len(weight_parts) >= 2:
            w2 = bsk_MultiLoraLoaderWithPath._to_float_or_none(weight_parts[1])
            if w2 is not None:
                clip = w2
        return {"name": name, "model": model, "clip": clip, "switch": True}

    @staticmethod
    def _parse_lora_text(text):
        """LoRA 목록 텍스트 파싱, 두 형식 지원 (프론트엔드 mlParseLoraText와 일치)）：

        1) 행 형식: 이름: 모델 가중치, 클립 가중치,on/off
           - 가중치/스위치는 생략 가능: 이름 / 이름:0.8 / 이름:0.8,0.9 / 이름:0.8,0.9,off
           - 가중치 하나만 제공할 때 clip=model；가중치를 기본값으로 1로 설정하지 않았습니다. 스위치를 기본값으로 활성화하지 않았습니다.
        2) A1111 스타일：<lora:이름: 가중치> / <이름: 가중치> / <lora:이름: 가중치 1: 가중치2>
           - 단 가중치 시 clip=model；기본적으로 활성화됩니다. 여러 줄, 쉼표로 구분된 혼합을 사용할 수 있습니다.
        첫 번째 쓰기 줄 master=on/off 전체 스위치를 제어합니다. 값이 없으면 None(호출자에서 되돌아가 전체 스위치 컨트롤로 이동) 반환합니다.）。

        반환 (master_override, [{"name","model","clip","switch"}])
        """
        master_override = None
        loras = []
        if not text or not isinstance(text, str):
            return (master_override, loras)
        text = text.strip()
        if not text:
            return (master_override, loras)

        lines = text.split("\n")

        # 줄 단위 분석 (원래 유지)）：<...> 마크는 일반 행 형식과 혼합 시 나타나는 순서대로 출력
        for i, line in enumerate(lines):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            mm = re.match(r"^master\s*[=:]\s*(on|off|true|false|1|0)\b", line, re.IGNORECASE)
            if mm:
                master_override = mm.group(1).lower() in ("on", "true", "1")
                continue

            # 포함 <...> 행: 이 행의 모든 태그를 추출 (A1111 스타일）
            if "<" in line:
                for m in re.finditer(r"<([^<>]+)>", line):
                    token = m.group(1).strip()
                    if not token:
                        continue
                    parts = [p.strip() for p in token.split(":")]
                    item = None
                    if len(parts) >= 2 and parts[0].lower() == "lora":
                        item = bsk_MultiLoraLoaderWithPath._lora_item_from_token(parts[1], parts[2:])
                    elif len(parts) >= 2:
                        item = bsk_MultiLoraLoaderWithPath._lora_item_from_token(parts[0], parts[1:])
                    if item and item["name"] and item["name"] != "None":
                        loras.append(item)
                continue

            # 일반 행 형식: 이름:모델 가중치,clip 가중치,on/off(가중치/스위치 생략 가능)）
            colon_idx = line.find(":")
            if colon_idx < 0:
                if line != "None":
                    loras.append({"name": line, "model": 1.0, "clip": 1.0, "switch": True})
                continue
            name = line[:colon_idx].strip()
            rest = line[colon_idx + 1:].strip()
            parts = rest.split(",")
            model = 1.0
            clip = 1.0
            sw = True
            if parts and parts[0].strip() != "":
                pm = bsk_MultiLoraLoaderWithPath._to_float_or_none(parts[0])
                if pm is not None:
                    model = pm
                    clip = pm
            if len(parts) >= 2 and parts[1].strip() != "":
                pc = bsk_MultiLoraLoaderWithPath._to_float_or_none(parts[1])
                if pc is not None:
                    clip = pc
            if len(parts) >= 3:
                sw = parts[2].strip().lower() in ("on", "true", "1")
            if not name:
                name = "None"
            if name != "None":
                loras.append({"name": name, "model": model, "clip": clip, "switch": sw})

        return (master_override, loras)

    def _resolve_lora_path(self, lora_name):
        """이름으로 LoRA 파일 경로를 파싱합니다; 확장자/서브디렉토리 생략 형식을 지원"""
        try:
            return folder_paths.get_full_path_or_raise("loras", lora_name)
        except Exception:
            pass
        for ext in (".safetensors", ".pt", ".ckpt", ".bin", ".pth"):
            try:
                return folder_paths.get_full_path_or_raise("loras", lora_name + ext)
            except Exception:
                pass
        # 마지막으로 파일 이름(경로 제외)로 하위 디렉토리에서 동일 이름 파일 찾기
        try:
            base = lora_name.replace("\\", "/").rsplit("/", 1)[-1]
            for c in folder_paths.get_filename_list("loras"):
                if c.replace("\\", "/").rsplit("/", 1)[-1] == base:
                    return folder_paths.get_full_path_or_raise("loras", c)
        except Exception:
            pass
        return None

    def _collect_legacy_slots(self, kwargs, missing_out=None):
        """호환 초기 워크플로우의 채널별 입력（switch_i / lora_name_i / strength_model_i / strength_clip_i）。

        새 버전 노드는 더 이상 이러한 입력을 선언하지 않습니다. 여기는 이전 워크플로우의 일회성 임시 마이그레이션입니다；
        정상적인 경우 텍스트가 비어 있지 않으며, 절대 이 지점에 도달하지 않습니다.。
        missing_out：None가 아닐 경우, 디스크에 없지만 참조되는 LoRA 이름 추가, 호출자에게 일관된 오류 표시。
        """
        result = []
        for i in range(1, 8):
            if not kwargs.get(f"switch_{i}", False):
                continue
            lora_name = kwargs.get(f"lora_name_{i}", "None")
            if not lora_name or lora_name == "None":
                continue
            strength_model = self._to_float_or_none(kwargs.get(f"strength_model_{i}"))
            strength_clip = self._to_float_or_none(kwargs.get(f"strength_clip_{i}"))
            if strength_model is None:
                strength_model = 1.0
            if strength_clip is None:
                strength_clip = 1.0
            if strength_model == 0 and strength_clip == 0:
                continue
            lora_path = self._resolve_lora_path(lora_name)
            if lora_path is None:
                print(f"[bsk_MultiLora] 찾을 수 없음LoRA: {lora_name}")
                if missing_out is not None:
                    missing_out.append(lora_name)
                continue
            result.append((lora_path, strength_model, strength_clip))
        return result

    # 오류 텍스트의 누락된 모델 태그 줄: 프론트엔드 js/panel.js의 _parseMissingModelsFromError
    # 누르다「[BSK_MISSING_MODEL] <모델 디렉토리>|<자 디렉토리>|<파일명>」파싱, 모델 다운로드」
    # 패널에서 '내부 구조'를 누르십시오 + "파일명" 강조 위치: 동일 디렉토리 동일 이름만 일치）。
    _MISSING_MODEL_MARK = "[BSK_MISSING_MODEL]"

    @classmethod
    def _build_missing_loras_error(cls, missing_loras):
        """손실된 LoRA 오류 텍스트 생성: 첫 번째 사람이 읽을 수 있는 요약 (파일명 포함하지 않음, 프론트엔드 정규 표현식 중복 수집 방지)），
        그 뒤 각 누락된 항목은 기계가 읽을 수 있는 마커 줄에 한 줄씩。"""
        names = [str(n).strip() for n in missing_loras if str(n).strip()]
        rel_names = []
        seen = set()
        for name in names:
            rel = name.replace("\\", "/").strip().lstrip("/")
            if rel and rel not in seen:
                seen.add(rel)
                rel_names.append(rel)
        lines = [
            "LoRA 로드 실패: 다음 %d개의 모델 파일이 존재하지 않습니다. '모델 다운로드' 패널에서 다운로드한 후 다시 시도하십시오.。" % len(rel_names)
        ]
        for rel in rel_names:
            subdir, _, filename = rel.rpartition("/")
            lines.append("%s loras|%s|%s" % (cls._MISSING_MODEL_MARK, subdir, filename))
        return "\n".join(lines)

    def load_multi_lora(self, lora_text="", model=None, clip=None, **kwargs):
        # If model and clip are not connected, return directly
        if model is None and clip is None:
            return (None, None)

        # ---- 패널 구성 JSON 파싱 실패 (프론트엔드 형식)）----
        text_master, text_loras = self._parse_lora_text(lora_text)

        # 전체 스위치: 텍스트에서 master= 우선순위; 쓰지 않을 경우 기본 설정
        if text_master is False:
            return (model, clip)

        # ---- 로드해야 하는 LoRA 수집(텍스트 우선, 순서대로, 개수 무제한)）----
        lora_list = []
        missing_loras = []  # LoRA를 참조했지만 디스크에 찾을 수 없는 경우(이름 원본으로 기록, 통일 오류 진단에 사용).）
        for item in text_loras:
            if not item.get("switch"):
                continue
            lora_name = (item.get("name") or "None").strip()
            if not lora_name or lora_name == "None":
                continue
            strength_model = self._to_float_or_none(item.get("model"))
            strength_clip = self._to_float_or_none(item.get("clip"))
            if strength_model is None:
                strength_model = 1.0
            if strength_clip is None:
                strength_clip = strength_model
            if strength_model == 0 and strength_clip == 0:
                continue
            lora_path = self._resolve_lora_path(lora_name)
            if lora_path is None:
                print(f"[bsk_MultiLora] 찾을 수 없음LoRA: {lora_name}")
                missing_loras.append(lora_name)
                continue
            lora_list.append((lora_path, strength_model, strength_clip))

        # 이전 워크플로우 호환성: 텍스트가 비어 있을 때 이전 슬롯별 입력으로 되돌립니다.
        if not lora_list and not text_loras:
            lora_list = self._collect_legacy_slots(kwargs, missing_loras)

        # LoRA가 누락되면 오류가 발생합니다 (정적 무시 건너뛰기 없음): 프론트엔드 파싱 후 오류 텍스트，
        # 실행 실패 팝업에서 누락된 파일을 나열하고, ‘모델 다운로드’ 패널에서 폴더별로 진행합니다.+파일명 강조 위치 지정。
        if missing_loras:
            raise ValueError(self._build_missing_loras_error(missing_loras))

        # LoRA가 없으면 입력 그대로 반환
        if not lora_list:
            return (model, clip)

        # 순차적으로 각 개별 로드LoRA
        model_lora = model
        clip_lora = clip

        for lora_path, strength_model, strength_clip in lora_list:
            lora = self._load_single_lora(lora_path)
            model_lora, clip_lora = comfy.sd.load_lora_for_models(
                model_lora, clip_lora, lora, strength_model, strength_clip
            )

        return (model_lora, clip_lora)


# ---------------------------
# 핵심 노드 클래스
# ---------------------------

class StringRuleProcessor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_string": ("STRING", {
                    "multiline": False,      
                    "default": ""
                }),
                "rules": ("STRING", {
                    "multiline": True,       
                    "default": "# 예시 규칙：\nif perlica_(Arknights): 1girl, long_hair\nif abc: abc ccc else: default_abc"
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "process"
    CATEGORY = "utils/text"

    def append_with_comma(self, text, add_text):
        """보조 함수: 스마트 추가 텍스트"""
        if not add_text:
            return text
        text = text.strip()
        add_text = add_text.strip()
        if not text:
            return add_text
        if text.endswith(','):
            return f"{text} {add_text}"
        else:
            return f"{text}, {add_text}"

    def _parse_rule_line(self, line, current_tags):
        """if 규칙 행을 파싱하여 (keyword, add_if_exists, add_if_not_exists)를 반환하거나 None。

        오래된 버전 정규표현식 사용 `(.+?)\s*:` 키워드 비동기 일치, 첫 번째 콜론에 도달하면 중단。
        키워드 내부 포함 콜론이 있는 경우 (예로) `if ram \(re:zero\):`，`re:zero` 리 잇 다 모지），
        키워드가 잘릴 수 있습니다 `ram \(re:zero`，항상 레이블 집합의 완전한 레이블과 일치하지 않음。

        수정 방법: 내부 콜론 위치를 모두 찾아 순차적으로 테스트하여 구분자 확인，
        우선적으로 '잘린 키워드가 현재 태그 집합을 맞추는' 콜론；
        모든 규칙이 일치하지 못할 경우 (규칙 키워드가 레이블에 없는 경우), 첫 번째 콜론으로 되돌림，
        이때 키워드 파싱 오류도 결과에 영향을 미치지 않음 (미타격은 반드시 else 분기 또는 직접 건너뜦)）。
        """
        m = re.match(r'^if\s+(.*)$', line, re.IGNORECASE)
        if not m:
            return None
        rest = m.group(1)

        # else 분기를 먼저 처리합니다 (else 내용에 콜론 포함 허용)）
        else_m = re.search(r'\s+else:\s*', rest, re.IGNORECASE)
        if else_m:
            head = rest[:else_m.start()]
            add_if_not_exists = rest[else_m.end():].strip() or None
        else:
            head = rest
            add_if_not_exists = None

        # 콜론 위치 모두 수집
        colon_positions = [i for i, ch in enumerate(head) if ch == ':']
        if not colon_positions:
            return None

        # ‘타겟 태그 집합 키워드’를 포함하는 콜론을 우선 선택
        split_pos = colon_positions[0]
        for pos in colon_positions:
            if head[:pos].strip().lower() in current_tags:
                split_pos = pos
                break

        word = head[:split_pos].strip()
        add_if_exists = head[split_pos + 1:].strip()
        return word, add_if_exists, add_if_not_exists

    def process(self, input_string, rules):
        output = input_string

        # 핵심 개선 1: 정확한 판단을 위해“존재합니까?”，출력을 쉼표로 분할하여 태그 세트로 구성합니다
        # 이렇게 하면 괄호와 같은 특수 문자가 정규식 검색에서 모호성을 피하고 공백을 처리할 수 있습니다
        current_tags = set(tag.strip().lower() for tag in output.split(',') if tag.strip())

        lines = rules.split('\n')

        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            # 규칙 행 파싱: 콜론 포함 내부 키워드 호환 (예:) ram \(re:zero\)）
            parsed = self._parse_rule_line(line, current_tags)
            if parsed is None:
                continue

            word, add_if_exists, add_if_not_exists = parsed
            # 크기와 공백을 무시하고 비교
            word_check = word.lower().strip()

            # 집합 존재성 검사를 사용하고 정규식 검색은 사용하지 마십시오
            # 이렇게 word 가 "perlica_(Arknights)" 괄호가 있는 경우도 정확하게 일치하고 오류가 발생하지 않습니다.
            is_present = word_check in current_tags

            if is_present:
                if add_if_exists:
                    output = self.append_with_comma(output, add_if_exists)
                    # 업데이트 컬렉션, 후속 규칙에 추가된 태그 사용
                    new_tags = [t.strip().lower() for t in add_if_exists.split(',') if t.strip()]
                    current_tags.update(new_tags)
            else:
                if add_if_not_exists:
                    output = self.append_with_comma(output, add_if_not_exists)
                    new_tags = [t.strip().lower() for t in add_if_not_exists.split(',') if t.strip()]
                    current_tags.update(new_tags)

        return (output,)


class StringProcessor:
    """
    ComfyUI문자열 처리 플러그인
    주요 기능：
    1. 프롬프트에서 완전히 동일한 프롬프트 조각을 제거합니다.
    2. 여러 줄 문자열 입력 상자를 처리하고 쉼표로 분할하고 쉼표 뒤의 공백을 제거합니다
    3. 무시하다 "# " 시작 주석 줄
    4. 출력을 형식화합니다: 마침표 뒤에 공백을 추가하고, 줄 바꿈을 삭제하고, 불필요한 구두점과 공백을 제거합니다.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "strings_to_remove": ("STRING", {
                    "multiline": True, 
                    "default": "",
                    "placeholder": "제거할 문자열을 입력하고, 한 줄에 하나 또는 쉼표로 구분"
                }),
                "match_whole_token": ("BOOLEAN", {
                    "default": True,
                    "label": "전 단어 일치"
                }),
            },
            "optional": {
                "input_string": ("STRING", {
                    "multiline": True, 
                    "default": "",
                    "placeholder": "입력 처리할 문자열",
                    "forceInput": True  # 이 설정을 선택 사항으로 설정하고 강제로 입력
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("processed_text",)
    FUNCTION = "process_string"
    CATEGORY = "텍스트 처리"
    
    def __init__(self):
        # 프롬프트 분리 집합 확장, 모든 일반적인 프롬프트 분리자를 포함합니다.
        # 영어：, ; . ! ? 
        # 중국어：，；。！？
        self.token_separators = r""",，;；.。!！?？:："""
        self.separator_pattern = f"[{re.escape(self.token_separators)}]"
        
        # 모든 구두점은 정리용
        self.all_punctuation = r""",，;；.。!！?？:：'"'"（）【】《》〈〉『』「」[]{}()!@#$%^&*+=\-|\\~`<>"""
        self.all_punctuation_pattern = f"[{re.escape(self.all_punctuation)}]"

    def parse_removal_strings(self, strings_to_remove):
        """제거할 문자열 목록을 파싱합니다"""
        removal_strings = []
        
        # 문자열 입력 상자에 줄 단위로 문자열을 분할합니다
        lines = strings_to_remove.split('\n')
        
        for line in lines:
            line = line.strip()
            # 무시하다 "# " 시작 주석 줄
            if line.startswith("# "):
                continue
                
            # 쉼표로 분리하고 쉼표 뒤의 공백 제거
            parts = line.split(',')
            for part in parts:
                cleaned_part = part.strip()
                if cleaned_part:
                    removal_strings.append(cleaned_part)
        
        return removal_strings
    
    def split_prompt_tokens(self, text):
        """프롬프트 텍스트를 독립적인 프롬프트로 분할"""
        if not text:
            return []
            
        # 모든 구분자를 쉼표로 일괄 대체, 처리 용이
        # 정규 표현식을 사용하여 모든 구분 기호를 쉼표로 바꿉니다.
        normalized_text = re.sub(self.separator_pattern, ',', text)
        
        # 쉼표로 프롬프트 분할
        tokens = normalized_text.split(',')
        
        # 정리 각 프롬프트
        cleaned_tokens = []
        for token in tokens:
            token = token.strip()
            if token:  # 비어 있지 않은 프롬프트만 유지
                # 제거 텍스트 프롬프트의 불필요한 공백
                token = re.sub(r'\s+', ' ', token)
                cleaned_tokens.append(token)
        
        return cleaned_tokens
    
    def remove_whole_tokens(self, text, removal_strings):
        """완전 일치 패턴: 완전 동일한 프롬프트만 제거합니다."""
        if not text or not text.strip():
            return self.format_output("")
        
        if not removal_strings:
            return self.format_output(text)
        
        # 분할 프롬프트
        tokens = self.split_prompt_tokens(text)
        
        # 집합으로 변환하여 빠르게 검색합니다 (완전 일치하는 프롬프트 제거)）
        removal_set = set(removal_strings)
        
        # 필터링 완전 일치하는 프롬프트
        filtered_tokens = [token for token in tokens if token not in removal_set]
        
        return self.format_output(", ".join(filtered_tokens))
    
    def remove_containing_substrings(self, text, removal_strings):
        """부분 문자열 패턴: 지정된 부분 문자열을 포함하는 프롬프트 텍스트 제거"""
        if not text or not text.strip():
            return self.format_output("")
        
        if not removal_strings:
            return self.format_output(text)
        
        # 분할 프롬프트
        tokens = self.split_prompt_tokens(text)
        
        # 제거할 하위 문자열을 포함하는 프롬프트 필터링
        filtered_tokens = []
        for token in tokens:
            should_keep = True
            for removal_str in removal_strings:
                if removal_str and removal_str in token:
                    should_keep = False
                    break
            if should_keep:
                filtered_tokens.append(token)
        
        return self.format_output(", ".join(filtered_tokens))
    
    def format_output(self, text):
        """포맷 출력: 쉼표와 공백을 유지하고 형식을 정리"""
        if not text or text.strip() == "":
            return ", "
        
        # 텍스트 삭제
        result = text
        
        # 삭제 줄바꿈 문자
        result = result.replace('\n', ' ').replace('\r', ' ')
        
        # 연속된 구두점 정리
        result = re.sub(f'({self.all_punctuation_pattern})\\1+', r'\1', result)
        
        # 제거 불필요한 공백
        result = re.sub(r'\s+', ' ', result).strip()
        
        # 제거 시작 및 종료 쉼표
        result = re.sub(r'^[,\s]+', '', result)
        result = re.sub(r'[,\s]+$', '', result)
        
        # 확인하세요: 쉼표와 공백이 있는지
        if result and not result.endswith(", "):
            result = result + ", "
        
        return result
    
    def process_string(self, input_string="", strings_to_remove="", match_whole_token=True):
        # 제거할 문자열 파싱
        removal_strings = self.parse_removal_strings(strings_to_remove)
        
        # 스위치 선택에 따라 제거 모드 선택
        if match_whole_token:
            # 전체 단어 일치 모드 (기본): 완전히 동일한 힌트만 제거
            result = self.remove_whole_tokens(input_string, removal_strings)
        else:
            # 부분 문자열 패턴: 지정된 부분 문자열을 포함하는 프롬프트 텍스트 제거
            result = self.remove_containing_substrings(input_string, removal_strings)
        
        return (result,)



class bsk_PanelConfigSaver:
    """
    패널 설정 저장 노드 - 패널 설정 JSON을 워크플로우/이미지 메타데이터에 삽입
    프론트엔드 실행 시 패널 설정이 해당 노드 입력에 자동 기록됩니다.，
    이 노드를 포함하는 이미지/워크플로우를 드래그할 때, 프론트엔드는 설정 복원 패널을 추출。
    이 노드는 다른 노드와 연결되지 않습니다。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "panel_config": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "패널 설정 JSON (프론트엔드 자동 채우기, 수동 편집 불필요)）",
                    "dynamicPrompts": False
                })
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("config_string",)
    FUNCTION = "execute"
    CATEGORY = "utils"
    OUTPUT_NODE = True

    def execute(self, panel_config=""):
        return {"ui": {"text": [panel_config]}, "result": (panel_config,)}


class ComfyUIPanelNode:
    """플레이스홀더 노드 - 실제 기능은 프런트엔드 JavaScript를 통해 구현됩니다."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {},
        }

    RETURN_TYPES = ()
    FUNCTION = "execute"
    CATEGORY = "utils"
    OUTPUT_NODE = True

    def execute(self):
        return {}

class RandomSeedNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed_str": ("STRING", {"default": "0", "multiline": False}),
                "trigger": ("INT", {"default": 0, "min": 0, "max": 9})
            },
            "optional": {
                "auto_generate": ("BOOLEAN", {"default": True, "label": "매번 실행 시 무작위 숫자 자동 생성"})
            }
        }

    RETURN_TYPES = ("INT",)
    FUNCTION = "generate"
    CATEGORY = "utils"

    def generate(self, seed_str, trigger=0, auto_generate=True):
        if not isinstance(seed_str, str):
            seed_str = str(seed_str)
        cleaned = seed_str.strip()
        if cleaned == "-1":
            # 0에서 생성 2^64-1 중간 랜덤 정수를 시드로 사용, ComfyUI 공식 범위와 일치
            seed = random.randint(0, 2**64 - 1)
        else:
            seed = int(cleaned)
        # [BSK SEED DEBUG] 백엔드에서 실제로 소비되는 랜덤 시드 (프론트엔드 제출 값과 비교, 일치하는지 확인)）
        print(f"[BSK SEED DEBUG] RandomSeedNode.generate | 수신 seed_str={seed_str!r} trigger={trigger} | 파싱 후 seed={seed} (type={type(seed).__name__})")
        return (seed,)

#region WanVideo노드 설명
# bsk_WanVideoImageToVideoEncodeSelective 클래스가 wan_nodes.py에서 가져왔습니다
# bsk_WanVideoFrameMask 클래스가 wan_nodes.py에서 가져왔습니다  
# bsk_WanVideoImageToVideoEncode 클래스가 wan_nodes.py에서 가져왔습니다
# wan_nodes.py가 없으면 이러한 노드는 사용할 수 없음
#endregion

# Placeholder: 오류 참조를 피하기 위해만 사용 (실제 클래스는 wan_nodes.py에서 가져옴)）
# 주의: 이러한 자리 표시자 없음 INPUT_TYPES，**절대 불가**레지스트리 진입 NODE_CLASS_MAPPINGS，
# 그렇지 않으면 프론트엔드에서 오류가 발생 "AttributeError: type object 'xxx' has no attribute 'INPUT_TYPES'"。
if bsk_WanVideoImageToVideoEncodeSelective is None:
    class bsk_WanVideoImageToVideoEncodeSelective:
        pass

if bsk_WanVideoFrameMask is None:
    class bsk_WanVideoFrameMask:
        pass

if bsk_WanVideoImageToVideoEncode is None:
    class bsk_WanVideoImageToVideoEncode:
        pass

# 다음 원시 클래스 정의는 wan_nodes.py로 이동했으며, 여기서는 중복 정의하지 않습니다.

class ImageResizeAAA:
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "width": ("INT", { "default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 1, }),
                "height": ("INT", { "default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 1, }),
                "upscale_method": (s.upscale_methods,),
                "keep_proportion": (["stretch", "resize", "pad", "pad_edge", "crop"], { "default": False }),
                "pad_color": ("STRING", { "default": "0, 0, 0", "tooltip": "Color to use for padding."}),
                "crop_position": (["center", "top", "bottom", "left", "right"], { "default": "center" }),
                "divisible_by": ("INT", { "default": 2, "min": 0, "max": 512, "step": 1, }),
                "swap_dimensions": ("BOOLEAN", {"default": False, "label": "Swap Width/Height"}),
                "auto_match_aspect": ("BOOLEAN", {"default": False, "label": "Auto Match Aspect Ratio"}),
                "filename_prefix": ("STRING", {"default": "image", "tooltip": "Prefix for the output filename"}),
            },
            "optional" : {
                "mask": ("MASK",),
                "device": (["cpu", "gpu"],),
            },
             "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "MASK", "STRING")
    RETURN_NAMES = ("IMAGE", "width", "height", "mask", "filename")
    FUNCTION = "resize"
    CATEGORY = "KJNodes/image"
    DESCRIPTION = """
Resizes the image to the specified width and height.  
Size can be retrieved from the input.

Keep proportions keeps the aspect ratio of the image, by  
highest dimension.  
"""

    def resize(self, image, width, height, keep_proportion, upscale_method, divisible_by, pad_color, crop_position, swap_dimensions, auto_match_aspect, filename_prefix, unique_id, device="cpu", mask=None):
        B, H, W, C = image.shape
        
        # 자동으로 가로 세비 일치
        if auto_match_aspect:
            input_aspect = W / H
            target_aspect = width / height
            
            # 입력과 대상의 가로 세비가 일치하지 않음 (가로/세로 방향이 다름)）
            if (input_aspect > 1 and target_aspect < 1) or (input_aspect < 1 and target_aspect > 1):
                width, height = height, width
                print(f"Auto-matched aspect ratio: swapped dimensions to {width}x{height}")
        
        # 수동으로 너비와 높이 교환
        if swap_dimensions:
            width, height = height, width
            
        if device == "gpu":
            if upscale_method == "lanczos":
                raise Exception("Lanczos is not supported on the GPU")
            device = model_management.get_torch_device()
        else:
            device = torch.device("cpu")

        if width == 0:
            width = W
        if height == 0:
            height = H
        
        if keep_proportion == "resize" or keep_proportion.startswith("pad"):
            # If one of the dimensions is zero, calculate it to maintain the aspect ratio
            if width == 0 and height != 0:
                ratio = height / H
                new_width = round(W * ratio)
            elif height == 0 and width != 0:
                ratio = width / W
                new_height = round(H * ratio)
            elif width != 0 and height != 0:
                # Scale based on which dimension is smaller in proportion to the desired dimensions
                ratio = min(width / W, height / H)
                new_width = round(W * ratio)
                new_height = round(H * ratio)

            if keep_proportion.startswith("pad"):
                # Calculate padding based on position
                if crop_position == "center":
                    pad_left = (width - new_width) // 2
                    pad_right = width - new_width - pad_left
                    pad_top = (height - new_height) // 2
                    pad_bottom = height - new_height - pad_top
                elif crop_position == "top":
                    pad_left = (width - new_width) // 2
                    pad_right = width - new_width - pad_left
                    pad_top = 0
                    pad_bottom = height - new_height
                elif crop_position == "bottom":
                    pad_left = (width - new_width) // 2
                    pad_right = width - new_width - pad_left
                    pad_top = height - new_height
                    pad_bottom = 0
                elif crop_position == "left":
                    pad_left = 0
                    pad_right = width - new_width
                    pad_top = (height - new_height) // 2
                    pad_bottom = height - new_height - pad_top
                elif crop_position == "right":
                    pad_left = width - new_width
                    pad_right = 0
                    pad_top = (height - new_height) // 2
                    pad_bottom = height - new_height - pad_top

            width = new_width
            height = new_height

        if divisible_by > 1:
            width = width - (width % divisible_by)
            height = height - (height % divisible_by)

        out_image = image.clone().to(device)

        if mask is not None:
            out_mask = mask.clone().to(device)
        
        if keep_proportion == "crop":
            old_width = W
            old_height = H
            old_aspect = old_width / old_height
            new_aspect = width / height
            
            # Calculate dimensions to keep
            if old_aspect > new_aspect:  # Image is wider than target
                crop_w = round(old_height * new_aspect)
                crop_h = old_height
            else:  # Image is taller than target
                crop_w = old_width
                crop_h = round(old_width / new_aspect)
            
            # Calculate crop position
            if crop_position == "center":
                x = (old_width - crop_w) // 2
                y = (old_height - crop_h) // 2
            elif crop_position == "top":
                x = (old_width - crop_w) // 2
                y = 0
            elif crop_position == "bottom":
                x = (old_width - crop_w) // 2
                y = old_height - crop_h
            elif crop_position == "left":
                x = 0
                y = (old_height - crop_h) // 2
            elif crop_position == "right":
                x = old_width - crop_w
                y = (old_height - crop_h) // 2
            
            # Apply crop
            out_image = out_image.narrow(-2, x, crop_w).narrow(-3, y, crop_h)
            if mask is not None:
                out_mask = out_mask.narrow(-1, x, crop_w).narrow(-2, y, crop_h)
        
        out_image = common_upscale(out_image.movedim(-1,1), width, height, upscale_method, crop="disabled").movedim(1,-1)

        if mask is not None:
            if upscale_method == "lanczos":
                out_mask = common_upscale(out_mask.unsqueeze(1).repeat(1, 3, 1, 1), width, height, upscale_method, crop="disabled").movedim(1,-1)[:, :, :, 0]
            else:
                out_mask = common_upscale(out_mask.unsqueeze(1), width, height, upscale_method, crop="disabled").squeeze(1)
            
        if keep_proportion.startswith("pad"):
            if pad_left > 0 or pad_right > 0 or pad_top > 0 or pad_bottom > 0:
                padded_width = width + pad_left + pad_right
                padded_height = height + pad_top + pad_bottom
                if divisible_by > 1:
                    width_remainder = padded_width % divisible_by
                    height_remainder = padded_height % divisible_by
                    if width_remainder > 0:
                        extra_width = divisible_by - width_remainder
                        pad_right += extra_width
                    if height_remainder > 0:
                        extra_height = divisible_by - height_remainder
                        pad_bottom += extra_height
                out_image, _ = self._image_pad(out_image, pad_left, pad_right, pad_top, pad_bottom, 0, pad_color, "edge" if keep_proportion == "pad_edge" else "color")
                if mask is not None:
                    out_mask = out_mask.unsqueeze(1).repeat(1, 3, 1, 1).movedim(1,-1)
                    out_mask, _ = self._image_pad(out_mask, pad_left, pad_right, pad_top, pad_bottom, 0, pad_color, "edge" if keep_proportion == "pad_edge" else "color")
                    out_mask = out_mask[:, :, :, 0]

        # 파일 이름 생성
        # 이미지 방향 확인
        if width > height:
            orientation = "H"  # 대시보드
        elif height > width:
            orientation = "V"  # 세로 그림
        else:
            orientation = "S"  # 정사각형
        
        # 파일 이름 생성
        filename = f"{filename_prefix}_{orientation}"

        if unique_id and PromptServer is not None:
            try:
                num_elements = out_image.numel()
                element_size = out_image.element_size()
                memory_size_mb = (num_elements * element_size) / (1024 * 1024)
                
                PromptServer.instance.send_progress_text(
                    f"<tr><td>Output: </td><td><b>{out_image.shape[0]}</b> x <b>{out_image.shape[2]}</b> x <b>{out_image.shape[1]} | {memory_size_mb:.2f}MB</b></td></tr>",
                    unique_id
                )
            except:
                pass

        return(out_image.cpu(), out_image.shape[2], out_image.shape[1], out_mask.cpu() if mask is not None else torch.zeros(64,64, device=torch.device("cpu"), dtype=torch.float32), filename)

class MaskSeparatorTool:
    """
    Mask분리 도구 노드 - 투명 픽셀의 RGB 값을 올바르게 처리
    """
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            },
        }
    
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "separate_mask"
    CATEGORY = "image/mask"
    
    def separate_mask(self, image):
        """
        이미지 및 마스크 분리, 투명 픽셀 RGB 값 정확히 처리
        
        입력:
            image: PyTorch tensor (B, H, W, C) 중간C=4대표RGBA
            
        출력:
            image_tensor: RGB이미지 (B, H, W, 3)
            mask_tensor: Alpha채널로서mask (B, H, W, 1)
        """
        from PIL import Image
        import torch
        
        # 변환PIL Image
        batch_size = image.shape[0]
        images_out = []
        masks_out = []
        
        for i in range(batch_size):
            # Convert single image
            img_tensor = image[i]
            
            # 변환하여 PIL RGBA 이미지
            img_pil = to_pil_image(img_tensor.permute(2, 0, 1))
            if img_pil.mode != 'RGBA':
                img_pil = img_pil.convert('RGBA')
            
            # 채널 분리
            r, g, b, a = img_pil.split()
            
            # RGB 이미지 생성 (원래 RGB 값 유지, 심지어alpha=0)
            rgb_img = Image.merge('RGB', (r, g, b))
            
            # Alpha채널로서mask
            mask_img = a
            
            # 변환 되돌리기tensor
            # RGB이미지: [0, 255] -> [0, 1]
            rgb_tensor = torch.from_numpy(np.array(rgb_img).astype(np.float32) / 255.0).permute(2, 0, 1)
            
            # Mask: [0, 255] -> [0, 1]
            mask_tensor = torch.from_numpy(np.array(mask_img).astype(np.float32) / 255.0).unsqueeze(0)
            
            images_out.append(rgb_tensor)
            masks_out.append(mask_tensor)
        
        # Stack성batch
        result_images = torch.stack(images_out)  # (B, 3, H, W)
        result_masks = torch.stack(masks_out)    # (B, 1, H, W)
        
        return (result_images, result_masks)


# ===== 간단한 테스트 노드 =====
class BSK_TestNode:
    """간단 테스트 노드, 노드 등록 정상 확인"""
    
    @classmethod
    def INPUT_TYPES(cls):
        print("[BSK_TestNode] INPUT_TYPES called")
        return {
            "required": {
                "text": ("STRING", {"default": "hello"}),
            },
            "hidden": {}
        }
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "test"
    CATEGORY = "utils"
    
    def test(self, text):
        print(f"[BSK_TestNode] test called with: {text}")
        return (text,)

print("[BSK_TestNode] Class defined successfully")

# ===== BSK 이미지 로딩 노드(자르기 지원)）=====
class BSK_LoadImage:
    """
    BSK 이미지 로딩 노드
    지원합니다, 매개변수 형식: "x,y,w,h,version,targetW,targetH"
    latent 출력을 지원합니다: 빈 latent 또는 VAE 인코딩된 출력을 선택할 수 있습니다.latent
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        try:
            input_dir = folder_paths.get_input_directory()
            files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
            files = folder_paths.filter_files_content_types(files, ["image"])
            file_list = sorted(files)
        except Exception as e:
            print(f"[BSK_LoadImage] ERROR: {e}")
            file_list = []
        return {
            "required": {
                "image": (file_list, {"image_upload": True}),
            },
            "optional": {
                "crop_params": ("STRING", {"default": "0,0,0,0,0,0,0", "tooltip": "x,y,w,h,version,targetW,targetH"}),
                "vae": ("VAE", {"tooltip": "VAE모델, 이미지 인코딩에 사용latent"}),
                "output_empty_latent": ("BOOLEAN", {"default": True, "tooltip": "True=출력 빈latent，False=출력 VAE 인코딩된 이미지latent"}),
                "redraw_mode": ("BOOLEAN", {"default": False, "tooltip": "재그리기 모드: 잘라내기 영역 자유 선택, 출력 즉 잘라내기 영역 원본 픽셀 (확대되지 않음), 8의 배수 정렬만"}),
                "divisible_by_32": ("BOOLEAN", {"default": False, "tooltip": "해상도 단계 진입: 끄기=VAE 최소 제약 조건으로 8 키를 누르면 활성화=32로 정렬 (더 엄격한 정수 나눗셈, 특정 모델/분할에 사용)）"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK", "LATENT", "IMAGE", "BSK_CROP_DATA", "BOOLEAN")
    RETURN_NAMES = ("IMAGE", "MASK", "LATENT", "원래 이미지", "자르기 데이터", "재그리기 모드")
    FUNCTION = "load_image"
    CATEGORY = "image"
    
    def load_image(self, image, crop_params="0,0,0,0,0,0,0", vae=None, output_empty_latent=True, redraw_mode=False, divisible_by_32=False):
        from PIL import ImageOps
        
        # 파싱 잘라내기 파라미터
        try:
            parts = [int(float(p)) for p in crop_params.split(',')]
            crop_x, crop_y, crop_w, crop_h, version, target_w, target_h = parts[:7]
        except:
            crop_x, crop_y, crop_w, crop_h, version, target_w, target_h = 0, 0, 0, 0, 0, 0, 0
        
        # 이미지 로드
        image_path = folder_paths.get_annotated_filepath(image)
        img = Image.open(image_path)
        img = ImageOps.exif_transpose(img)
        
        orig_w, orig_h = img.size

        # 원래 전체 이미지 (자르기/확대하기 전), 후속 불연속 보정용
        _orig_pil = img.convert('RGB') if img.mode != 'RGB' else img.copy()
        original_tensor = torch.from_numpy(np.array(_orig_pil).astype(np.float32) / 255.0).unsqueeze(0)

        # 실행 자르기
        if crop_w > 0 and crop_h > 0:
            crop_x = max(0, min(crop_x, orig_w - 1))
            crop_y = max(0, min(crop_y, orig_h - 1))
            crop_w = min(crop_w, orig_w - crop_x)
            crop_h = min(crop_h, orig_h - crop_y)
            img = img.crop((crop_x, crop_y, crop_x + crop_w, crop_y + crop_h))

        # 해상도 단계 진입: 끄기=이름으로 정렬, 활성화=32에 맞춰。
        # 32 8의 정수 배수이므로 VAE의 /8 제약 조건을 두 step 모두 만족하며, 엄격도만 다름。
        step = 32 if divisible_by_32 else 8
        _snap = lambda v: max(step, int(round((int(v) or 0) / step)) * step)

        # 재그리기 모드: 출력은 선택 영역 1:1(확대/축소 안 함), 크기를 step의 배수만큼 맞춰 VAE 제약 준수。
        # 백엔드가 target을 강제로 덮어쓰므로, 프론트엔드 동기화 여부와 관계없이 보장됩니다. 1:1 + 정렬。
        if redraw_mode:
            _cw = crop_w if crop_w > 0 else orig_w
            _ch = crop_h if crop_h > 0 else orig_h
            target_w = _snap(_cw)
            target_h = _snap(_ch)

        # Normal Mode (Non-Redrawn): Frontend aligned and written to target by step, backend does fallback alignment，
        # 오래된 워크플로우 / 수동으로 초과 전달된 비정렬 값, 출력값을 항상 step 그리드 내에 유지 (프론트엔드 표시에서 절대 벗어나지 않음)）。
        if (not redraw_mode) and target_w > 0 and target_h > 0:
            target_w = _snap(target_w)
            target_h = _snap(target_h)

        # 실행 스케일링 (RGBA 분리 RGB 및 Alpha 개별 리사이즈, PIL 프리멀티 알파 오염 방지)RGB）
        if target_w > 0 and target_h > 0:
            if img.mode == 'RGBA':
                r, g, b, a = img.split()
                rgb_resized = Image.merge('RGB', (r, g, b)).resize((target_w, target_h), Image.Resampling.LANCZOS)
                a_resized = a.resize((target_w, target_h), Image.Resampling.LANCZOS)
                r2, g2, b2 = rgb_resized.split()
                img = Image.merge('RGBA', (r2, g2, b2, a_resized))
            else:
                img = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
        
        # 변환 tensor
        if img.mode == 'RGBA':
            r, g, b, a = img.split()
            rgb = Image.merge('RGB', (r, g, b))
            image_tensor = torch.from_numpy(np.array(rgb).astype(np.float32) / 255.0).unsqueeze(0)
            mask_tensor = torch.from_numpy(np.array(a).astype(np.float32) / 255.0).unsqueeze(0)
        else:
            # 변환 RGB
            if img.mode != 'RGB':
                img = img.convert('RGB')
            image_tensor = torch.from_numpy(np.array(img).astype(np.float32) / 255.0).unsqueeze(0)
            mask_tensor = torch.zeros((1, img.height, img.width), dtype=torch.float32)
        
        # 생성latent
        final_w = image_tensor.shape[2]  # W
        final_h = image_tensor.shape[1]  # H
        
        if output_empty_latent:
            # 출력 빈latent
            latent = torch.zeros([1, 4, final_h // 8, final_w // 8],
                                 device=comfy.model_management.intermediate_device(),
                                 dtype=comfy.model_management.intermediate_dtype())
            latent_out = {"samples": latent}
        else:
            # 출력 VAE 인코딩된 이미지latent
            if vae is not None:
                latent_out = {"samples": vae.encode(image_tensor)}
            else:
                # VAE가 없을 경우 기본값으로 설정latent
                latent = torch.zeros([1, 4, final_h // 8, final_w // 8],
                                     device=comfy.model_management.intermediate_device(),
                                     dtype=comfy.model_management.intermediate_dtype())
                latent_out = {"samples": latent}
        
        # 자르기 데이터 (BSK_CropMerge 원활한 복원 사용용)）。
        # crop_w/crop_h 0은 래퍼를 비활성화하고, 노드를 병합하면 전체 이미지 사각형으로 퇴화됩니다.。
        crop_data = {
            "crop_x": int(crop_x),
            "crop_y": int(crop_y),
            "crop_w": int(crop_w),
            "crop_h": int(crop_h),
            "orig_w": int(orig_w),
            "orig_h": int(orig_h),
            "target_w": int(target_w),
            "target_h": int(target_h),
        }

        return (image_tensor, mask_tensor, latent_out, original_tensor, crop_data, bool(redraw_mode))
    
    @classmethod
    def IS_CHANGED(cls, image, crop_params="0,0,0,0,0,0,0", vae=None, output_empty_latent=True, redraw_mode=False, divisible_by_32=False):
        image_path = folder_paths.get_annotated_filepath(image)
        mtime = os.path.getmtime(image_path) if os.path.exists(image_path) else 0
        return (mtime, crop_params, bool(redraw_mode), bool(divisible_by_32))

NODE_CLASS_MAPPINGS = {
    "bsk_StringMergeWithSwitch": bsk_StringMergeWithSwitch,
    "PromptStringProcessor": PromptStringProcessor,
    "LoraLoaderWithPath": LoraLoaderWithPath,
    "bsk_MultiLoraLoaderWithPath": bsk_MultiLoraLoaderWithPath,
    "StringRuleProcessor": StringRuleProcessor,
    "StringProcessor": StringProcessor,
    "MultiLineStringEditor": MultiLineStringEditor,
    
    "ComfyUIPanel": ComfyUIPanelNode,
    "bsk_PanelConfigSaver": bsk_PanelConfigSaver,
    "RandomSeedNode": RandomSeedNode,
    "이미지 확대aa": ImageResizeAAA,
    "MaskSeparatorTool": MaskSeparatorTool,
    "BSK_TestNode": BSK_TestNode,
    "BSK_LoadImage": BSK_LoadImage,
    "BSK_SaveImage": BSK_SaveImage,
    "bsk_Math": bsk_Math,
    
    
}

print(f"[BSK] NODE_CLASS_MAPPINGS contains BSK_TestNode: {'BSK_TestNode' in NODE_CLASS_MAPPINGS}")
print(f"[BSK] NODE_CLASS_MAPPINGS contains BSK_LoadImage: {'BSK_LoadImage' in NODE_CLASS_MAPPINGS}")

# 다중 레이어 이미지 로드 노드 추가
if BSK_MultiLayerImageLoader is not None:
    NODE_CLASS_MAPPINGS["BSK_MultiLayerImageLoader"] = BSK_MultiLayerImageLoader

# WanVideo 노드 추가 (실제 클래스 가져오기가 성공할 때만 등록, 자리 표시 클래스는 등록하지 않음)）
if _WAN_NODES_OK:
    NODE_CLASS_MAPPINGS["bsk_WanVideoImageToVideoEncode"] = bsk_WanVideoImageToVideoEncode
    NODE_CLASS_MAPPINGS["bsk_WanVideoImageToVideoEncodeSelective"] = bsk_WanVideoImageToVideoEncodeSelective
    NODE_CLASS_MAPPINGS["bsk_WanVideoFrameMask"] = bsk_WanVideoFrameMask

NODE_DISPLAY_NAME_MAPPINGS = {
    "bsk_StringMergeWithSwitch": "bsk_String Merge (Switch)",
    "PromptStringProcessor": "bsk_문자열 대체 및 삭제",
    "LoraLoaderWithPath": "bsk_LoraLoaderWithPath",
    "bsk_MultiLoraLoaderWithPath": "bsk_MultiLoraLoaderWithPath",
    "StringRuleProcessor": "bsk_StringRule",
    "StringProcessor": "bsk_문자열 제외, 주석 # , 쉼표로 구분",
    "MultiLineStringEditor": "bsk_프롬프트 편집기,ctrl+/ 주석/해제,",
    
    "ComfyUIPanel": "ComfyUI Panel",
    "bsk_PanelConfigSaver": "패널 설정 정보",
    "RandomSeedNode": "bsk_Random Seed",
    "ImageResizeAAA": "이미지 확대aa",
    "MaskSeparatorTool": "Mask도구 분리 (보존RGB)",
    "BSK_TestNode": "BSK 테스트 노드",
    "BSK_LoadImage": "BSK 이미지 불러오기",
    "BSK_SaveImage": "BSK 이미지 저장",
    "bsk_Math": "bsk_Math",
}

# 여러 레이어 이미지 로더 노드 표시 이름 추가
if BSK_MultiLayerImageLoader is not None:
    NODE_DISPLAY_NAME_MAPPINGS["BSK_MultiLayerImageLoader"] = "BSK 다중 레이어 이미지 로더"

# 추가 WanVideo 노드 표시 이름 (실제 클래스 가져오기 성공 시에만 등록)）
if _WAN_NODES_OK:
    NODE_DISPLAY_NAME_MAPPINGS["bsk_WanVideoImageToVideoEncode"] = "bsk_wan3중간 프레임"
    NODE_DISPLAY_NAME_MAPPINGS["bsk_WanVideoImageToVideoEncodeSelective"] = "bsk_WanVideoImageToVideoEncode프레임 선택"
    NODE_DISPLAY_NAME_MAPPINGS["bsk_WanVideoFrameMask"] = "bsk_WanVideoFrameMask"


# ===== MiniMax H3 Director —— 자체 작성 단일 파일 배치 노드 (타사 플러그인 복사 안 함)）=====
# 전체 패키지 복사 방식（nodes/minimax_h3_director/ + 아래 로딩 블록), 2026-08-05 폐기：
# 이름 충돌 / 규격 준수 실패 / 자체 웹/js 로드되지 않음 / 클라우드 원시 노드와 이름 충돌。
# 자신이 작성한 bsk_MiniMaxDirector(MD/MiniMaxH3_Director_기술원리 및 BSK 적응 분석 보고서 참조)로 변경.md §4）。
try:
    import importlib.util as _spec
    import os as _os
    import logging as _logging
    _mm_file = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "nodes", "bsk_minimax_director.py")
    if _os.path.isfile(_mm_file):
        _mm_name = "bsk_minimax_director"
        _mm_spec = _spec.spec_from_file_location(_mm_name, _mm_file)
        _mm_mod = _spec.module_from_spec(_mm_spec)
        _mm_spec.loader.exec_module(_mm_mod)
        for _k, _v in (_mm_mod.NODE_CLASS_MAPPINGS or {}).items():
            NODE_CLASS_MAPPINGS[_k] = _v
        for _k, _v in (_mm_mod.NODE_DISPLAY_NAME_MAPPINGS or {}).items():
            NODE_DISPLAY_NAME_MAPPINGS[_k] = _v
        _logging.getLogger("BSK").info("bsk_MiniMaxDirector loaded: %s", list(_mm_mod.NODE_CLASS_MAPPINGS.keys()))
except Exception as _mm_exc:
    _logging.getLogger("BSK").warning("bsk_MiniMaxDirector failed to load: %s", _mm_exc)


_PYNVML_ENSURE_STATE = {"attempted": False}


def _ensure_pynvml(auto_install=True):
    """확인합니다: 먼저 가져오고, 누락된 경우 현재 인터프리터 pip를 사용하여 nvidia-ml-py3를 설치한 다음 다시 시도합니다.。

    NVIDIA GPU 모니터링은 pynvml에 의존하며, 휴대용 패키지/새 환경에 기본적으로 설치되지 않아 직접 포기하면 문제가 발생
    「GPU 사용률 모니터링이 사용할 수 없습니다. 여기에서 기본 설치를 수행한 다음 로드하여 실패를 방지합니다.。
    돌아가다 (pynvml 모듈) | None, 설치 후 첫 번째인지)
    """
    import importlib

    try:
        return importlib.import_module("pynvml"), False
    except ImportError:
        pass
    except Exception:
        # 모듈이 존재하지만 손상/의존성 오류가 발생했습니다. 재설치 시도하지 않습니다.
        return None, False

    if not auto_install or _PYNVML_ENSURE_STATE["attempted"]:
        return None, False
    _PYNVML_ENSURE_STATE["attempted"] = True

    import subprocess
    import sys as _sys

    print("[ComfyUI Panel] GPU모니터링: pynvml 설치되지 않았습니다, 자동으로 설치 nvidia-ml-py3 ...")
    cmd_base = [_sys.executable, "-m", "pip", "install"]
    last_err = "알 수 없는 오류"
    for pkg in ("nvidia-ml-py3", "pynvml"):
        installed = False
        for args in ([pkg], [pkg, "--break-system-packages"]):
            try:
                proc = subprocess.run(cmd_base + args, stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, timeout=300)
                out = (proc.stdout or b"").decode("utf-8", "replace").strip()
                if proc.returncode != 0:
                    last_err = out[-300:] or "returncode=%s" % proc.returncode
                    # 최근 스레드 상세 정보 (10초마다 수집, 하지만 상주하지 않고, 수집 턴 재사용하지 않음)
                    if args[-1] == "--break-system-packages" and (
                            "no such option" in out.lower() or "unrecognized" in out.lower()):
                        continue
                    continue
                installed = True
                break
            except Exception as e:
                last_err = repr(e)
        if not installed:
            continue
        importlib.invalidate_caches()
        try:
            mod = importlib.import_module("pynvml")
        except Exception as e:
            last_err = f"설치 후에도 가져오기 실패 pynvml: {e}"
            continue
        print(f"[ComfyUI Panel] GPU모니터링: {pkg} 설치 성공, pynvml 로드됨")
        return mod, True

    print(f"[ComfyUI Panel] GPU모니터링: pynvml 자동 설치 실패({last_err})，"
          f"수동으로 실행하세요 'pip install nvidia-ml-py3'")
    return None, False


# ===== 하드웨어 통계 모니터링 클래스 (Crystools 구현 참조)）=====
# Crystools - Copyright (c) 2023 Crystian - MIT License
# https://github.com/crystian/ComfyUI-Crystools
class BSK_HardwareStats:
    """
    하드웨어 모니터링
    NVIDIA GPU 지원 (pynvml), Jetson 장치 (pynvml) jtop)
    CPU RAM 사용량은 psutil을 통해 가져옴
    """
    _instance = None
    pynvmlLoaded = False
    jtopLoaded = False
    pynvml = None
    jtopInstance = None
    deviceCount = 0
    psutilLoaded = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init()
        return cls._instance
    
    def _init(self):
        """하드웨어 모니터링 초기화"""
        # 昇腾 NPU 采集기기（및 training_api 함께용；getStatus 중에획득되지 않음 GPU 시회퇴）
        self._collect_npu_stats = None
        try:
            from .training_api import _collect_npu_stats
            self._collect_npu_stats = _collect_npu_stats
        except Exception:
            try:
                from training_api import _collect_npu_stats
                self._collect_npu_stats = _collect_npu_stats
            except Exception:
                pass
        # psutil을 초기화합니다 (CPU 및 RAM）
        try:
            import psutil
            self.psutil = psutil
            self.psutilLoaded = True
            # 预열: psutil.cpu_percent() 최초 호출 결과는 0.0 (무 baseline），
            # 먼저 한 번 기본값을 구축하고, 이후 호출에서만 실제 값을 반환합니다.
            self.psutil.cpu_percent(interval=None)
            print("[ComfyUI Panel] 하드웨어 모니터링: psutil 초기화됨")
        except ImportError:
            print("[ComfyUI Panel] 하드웨어 모니터링: psutil 설치되지 않았을 경우 CPU/RAM 모니터링 불가")
        
        # Jetson 장치 감지 여부 확인
        is_jetson = self._is_jetson()
        
        if is_jetson:
            # Jetson 장비 사용 jtop
            try:
                from jtop import jtop
                self.jtopInstance = jtop()
                self.jtopInstance.start()
                self.jtopLoaded = True
                self.deviceCount = 1
                print("[ComfyUI Panel] GPU모니터링: jtop 초기화 (Jetson 장치))")
            except ImportError:
                print("[ComfyUI Panel] GPU모니터링: jtop 설치되지 않음, 사용 시도 pynvml")
            except Exception as e:
                print(f"[ComfyUI Panel] GPU모니터: jtop 초기화 실패: {e}")
        
        if not self.jtopLoaded:
            # Jetson 장치에서는 pynvml 사용(설치되지 않은 경우 자동으로 설치 후 로드)）
            try:
                import time as _t
                pynvml, _installed = _ensure_pynvml()
                if pynvml is None:
                    print("[ComfyUI Panel] GPU모니터: pynvml 사용 불가, GPU 사용률 모니터링 불가")
                    return
                self.pynvml = pynvml
                try:
                    self.pynvml.nvmlInit()
                except Exception as _e:
                    # 설치 실패 후 발생, 드라이버 준비 시 재시도
                    print(f"[ComfyUI Panel] GPU모니터링: nvmlInit 실패({_e})，한 번 재시도")
                    _t.sleep(1)
                    self.pynvml.nvmlInit()
                self.pynvmlLoaded = True
                self.deviceCount = pynvml.nvmlDeviceGetCount()
                print(f"[ComfyUI Panel] GPU모니터링: pynvml이 초기화되었으며 감지했습니다 {self.deviceCount} 개 GPU")
            except Exception as e:
                print(f"[ComfyUI Panel] GPU모니터링: pynvml 초기화 실패: {e}")
    
    def _is_jetson(self):
        """Jetson 장치 감지 여부 확인"""
        import platform
        try:
            with open('/proc/device-tree/model', 'r') as f:
                model = f.read().strip()
                if "NVIDIA" in model:
                    return True
        except:
            pass
        if 'tegra' in platform.release().lower():
            return True
        return False
    
    def getStatus(self):
        """전체 하드웨어 상태 가져오기"""
        result = {
            'cpu_utilization': -1,
            'ram_total': 0,
            'ram_used': 0,
            'ram_percent': 0,
            'gpu_available': self.pynvmlLoaded or self.jtopLoaded,
            'gpus': [],
            'device_type': None,
        }
        
        # CPU 사용률
        if self.psutilLoaded:
            try:
                result['cpu_utilization'] = self.psutil.cpu_percent(interval=None)
            except:
                pass
        
        # RAM 사용률
        if self.psutilLoaded:
            try:
                ram = self.psutil.virtual_memory()
                result['ram_total'] = ram.total
                result['ram_used'] = ram.used
                result['ram_percent'] = ram.percent
            except:
                pass
        
        # GPU 정보
        if not result['gpu_available'] and self._collect_npu_stats is None:
            return result
        
        try:
            if self.pynvmlLoaded:
                for i in range(self.deviceCount):
                    handle = self.pynvml.nvmlDeviceGetHandleByIndex(i)
                    
                    # GPU 사용률
                    try:
                        util = self.pynvml.nvmlDeviceGetUtilizationRates(handle)
                        gpu_util = util.gpu
                        mem_util = util.memory
                    except:
                        gpu_util = -1
                        mem_util = -1
                    
                    # GPU 메모리 정보
                    try:
                        mem_info = self.pynvml.nvmlDeviceGetMemoryInfo(handle)
                        vram_total = mem_info.total
                        vram_used = mem_info.used
                        vram_percent = (vram_used / vram_total * 100) if vram_total > 0 else 0
                    except:
                        vram_total = 0
                        vram_used = 0
                        vram_percent = 0
                    
                    # GPU 온도
                    try:
                        temp = self.pynvml.nvmlDeviceGetTemperature(handle, self.pynvml.NVML_TEMPERATURE_GPU)
                    except:
                        temp = -1
                    
                    # GPU 이름
                    try:
                        name = self.pynvml.nvmlDeviceGetName(handle)
                        if isinstance(name, bytes):
                            name = name.decode('utf-8', errors='ignore')
                    except:
                        name = f"GPU {i}"
                    
                    result['gpus'].append({
                        'index': i,
                        'name': name,
                        'gpu_utilization': gpu_util,
                        'memory_utilization': mem_util,
                        'vram_total': vram_total,
                        'vram_used': vram_used,
                        'vram_percent': vram_percent,
                        'temperature': temp
                    })
            
            elif self.jtopLoaded and self.jtopInstance:
                # Jetson 장치
                try:
                    gpu_util = self.jtopInstance.stats.get('GPU', -1)
                    temp = self.jtopInstance.stats.get('Temp gpu', -1)
                    mem_data = self.jtopInstance.memory.get('RAM', {})
                    vram_total = mem_data.get('tot', 0)
                    vram_used = mem_data.get('used', 0)
                    vram_percent = (vram_used / vram_total * 100) if vram_total > 0 else 0
                    
                    gpu_name = 'Unknown'
                    try:
                        gpu_info = self.jtopInstance.gpu
                        gpu_name = next(iter(gpu_info.keys()), 'Jetson GPU')
                    except:
                        pass
                    
                    result['gpus'].append({
                        'index': 0,
                        'name': gpu_name,
                        'gpu_utilization': gpu_util,
                        'memory_utilization': -1,
                        'vram_total': vram_total,
                        'vram_used': vram_used,
                        'vram_percent': vram_percent,
                        'temperature': temp
                    })
                except Exception as e:
                    print(f"[ComfyUI Panel] Hardware monitoring: Failed to get Jetson GPU status: {e}")
        
        except Exception as e:
            print(f"[ComfyUI Panel] 하드웨어 모니터링: GPU 상태 가져오기 실패: {e}")
            result['gpu_available'] = False

        # 昇腾 NPU 회퇴（획득되지 않음 NVIDIA GPU / Jetson 시시도해 봅니다）
        if not result['gpus'] and self._collect_npu_stats is not None:
            try:
                npus = self._collect_npu_stats()
                if npus:
                    result['gpus'] = npus
                    result['gpu_available'] = True
                    result['device_type'] = 'npu'
            except Exception as e:
                print(f"[ComfyUI Panel] 하드웨어 모니터링: NPU 상태 가져오기 실패: {e}")

        if result['gpus'] and not result.get('device_type'):
            result['device_type'] = 'gpu'

        return result
    
    def close(self):
        """리소스 삭제"""
        if self.jtopLoaded and self.jtopInstance:
            try:
                self.jtopInstance.close()
            except:
                pass

# 전역 하드웨어 통계 인스턴스
_hardware_stats_instance = None

def get_hardware_stats_instance():
    """획득취하다하드웨어 통계예시（지연 초기화）"""
    global _hardware_stats_instance
    if _hardware_stats_instance is None:
        _hardware_stats_instance = BSK_HardwareStats()
    return _hardware_stats_instance


# 기존 GPU 통계 클래스 이름 호환 유지
class BSK_GPUStats:
    """오래된 코드 호환용 래퍼"""
    def __new__(cls):
        return get_hardware_stats_instance()


# 전역 GPU 통계 인스턴스 (호환성 유지)）
_gpu_stats_instance = None

def get_gpu_stats_instance():
    """GPU 통계 인스턴스 가져오기 (지연 초기화) - 호환성 유지"""
    return get_hardware_stats_instance()
    """
    GPU 사용률 (통해
    NVIDIA GPU (pynvml) 및 Jetson 장치 (통해) 지원 jtop)
    """
    _instance = None
    pynvmlLoaded = False
    jtopLoaded = False
    pynvml = None
    jtopInstance = None
    deviceCount = 0
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init()
        return cls._instance
    
    def _init(self):
        """GPU 모니터 초기화"""
        # Jetson 장치 감지 여부 확인
        is_jetson = self._is_jetson()
        
        if is_jetson:
            # Jetson 장비 사용 jtop
            try:
                from jtop import jtop
                self.jtopInstance = jtop()
                self.jtopInstance.start()
                self.jtopLoaded = True
                self.deviceCount = 1
                print("[ComfyUI Panel] GPU모니터링: jtop 초기화 (Jetson 장치))")
            except ImportError:
                print("[ComfyUI Panel] GPU모니터링: jtop 설치되지 않음, 사용 시도 pynvml")
            except Exception as e:
                print(f"[ComfyUI Panel] GPU모니터: jtop 초기화 실패: {e}")
        
        if not self.jtopLoaded:
            # Jetson 장치에서는 pynvml 사용(설치되지 않은 경우 자동으로 설치 후 로드)）
            try:
                import time as _t
                pynvml, _installed = _ensure_pynvml()
                if pynvml is None:
                    print("[ComfyUI Panel] GPU모니터: pynvml 사용 불가, GPU 사용률 모니터링 불가")
                    return
                self.pynvml = pynvml
                try:
                    self.pynvml.nvmlInit()
                except Exception as _e:
                    # 설치 실패 후 발생, 드라이버 준비 시 재시도
                    print(f"[ComfyUI Panel] GPU모니터링: nvmlInit 실패({_e})，한 번 재시도")
                    _t.sleep(1)
                    self.pynvml.nvmlInit()
                self.pynvmlLoaded = True
                self.deviceCount = pynvml.nvmlDeviceGetCount()
                print(f"[ComfyUI Panel] GPU모니터링: pynvml이 초기화되었으며 감지했습니다 {self.deviceCount} 개 GPU")
            except Exception as e:
                print(f"[ComfyUI Panel] GPU모니터링: pynvml 초기화 실패: {e}")
    
    def _is_jetson(self):
        """Jetson 장치 감지 여부 확인"""
        import platform
        try:
            with open('/proc/device-tree/model', 'r') as f:
                model = f.read().strip()
                if "NVIDIA" in model:
                    return True
        except:
            pass
        # 플랫폼 확인
        if 'tegra' in platform.release().lower():
            return True
        return False
    
    def getStatus(self):
        """GPU 상태 가져오기"""
        result = {
            'available': self.pynvmlLoaded or self.jtopLoaded,
            'gpus': []
        }
        
        if not result['available']:
            return result
        
        try:
            if self.pynvmlLoaded:
                for i in range(self.deviceCount):
                    handle = self.pynvml.nvmlDeviceGetHandleByIndex(i)
                    
                    # GPU 사용률
                    try:
                        util = self.pynvml.nvmlDeviceGetUtilizationRates(handle)
                        gpu_util = util.gpu
                        mem_util = util.memory
                    except:
                        gpu_util = -1
                        mem_util = -1
                    
                    # GPU 메모리 정보
                    try:
                        mem_info = self.pynvml.nvmlDeviceGetMemoryInfo(handle)
                        vram_total = mem_info.total
                        vram_used = mem_info.used
                        vram_percent = (vram_used / vram_total * 100) if vram_total > 0 else 0
                    except:
                        vram_total = 0
                        vram_used = 0
                        vram_percent = 0
                    
                    # GPU 온도
                    try:
                        temp = self.pynvml.nvmlDeviceGetTemperature(handle, self.pynvml.NVML_TEMPERATURE_GPU)
                    except:
                        temp = -1
                    
                    # GPU 이름
                    try:
                        name = self.pynvml.nvmlDeviceGetName(handle)
                        if isinstance(name, bytes):
                            name = name.decode('utf-8', errors='ignore')
                    except:
                        name = f"GPU {i}"
                    
                    result['gpus'].append({
                        'index': i,
                        'name': name,
                        'gpu_utilization': gpu_util,
                        'memory_utilization': mem_util,
                        'vram_total': vram_total,
                        'vram_used': vram_used,
                        'vram_percent': vram_percent,
                        'temperature': temp
                    })
            
            elif self.jtopLoaded and self.jtopInstance:
                # Jetson 장치
                try:
                    gpu_util = self.jtopInstance.stats.get('GPU', -1)
                    temp = self.jtopInstance.stats.get('Temp gpu', -1)
                    mem_data = self.jtopInstance.memory.get('RAM', {})
                    vram_total = mem_data.get('tot', 0)
                    vram_used = mem_data.get('used', 0)
                    vram_percent = (vram_used / vram_total * 100) if vram_total > 0 else 0
                    
                    gpu_name = 'Unknown'
                    try:
                        gpu_info = self.jtopInstance.gpu
                        gpu_name = next(iter(gpu_info.keys()), 'Jetson GPU')
                    except:
                        pass
                    
                    result['gpus'].append({
                        'index': 0,
                        'name': gpu_name,
                        'gpu_utilization': gpu_util,
                        'memory_utilization': -1,
                        'vram_total': vram_total,
                        'vram_used': vram_used,
                        'vram_percent': vram_percent,
                        'temperature': temp
                    })
                except Exception as e:
                    print(f"[ComfyUI Panel] GPU모니터링: Jetson GPU 상태 획득 실패: {e}")
        
        except Exception as e:
            print(f"[ComfyUI Panel] GPU모니터링: GPU 상태 획득 실패: {e}")
            result['available'] = False
        
        return result
    
    def close(self):
        """리소스 삭제"""
        if self.jtopLoaded and self.jtopInstance:
            try:
                self.jtopInstance.close()
            except:
                pass


# ===== 실시간 데이터 수집 스레드 =====
# 하드웨어 캡처(psutil/pynvml/jtop)는 동기 차단 호출이므로 aiohttp 이벤트 루프에 넣으면
# 모든 요청(추론 바쁠 때) 지연 NVML 호출 → 이벤트 루프 멈춤 → /queue 요청 대기
# → 게이트웨이 타임아웃 (반환 504). 독립적인 daemon 스레드가 주기적으로 하드웨어 데이터를 수집하여 캐시에 저장，
# API 캐시 직접 반환 (마이크로초 단위), 이벤트 루프 차단 완전 방지。
import threading as _threading

_runtime_cache_lock = _threading.Lock()
_runtime_cache = {
    "timestamp": 0,
    "hardware": {
        "cpu_utilization": -1,
        "ram_total": 0,
        "ram_used": 0,
        "ram_percent": 0,
        "gpu_available": False,
        "gpus": [],
    },
}
_runtime_collector_started = False

def _runtime_collector_loop():
    """백그라운드 수집 스레드 메인 루프: 주기적으로 하드웨어 데이터를 캐시에 기록"""
    global _runtime_cache
    last_hw = None
    while True:
        try:
            hw_stats = get_hardware_stats_instance()
            data = hw_stats.getStatus()
            with _runtime_cache_lock:
                _runtime_cache = {
                    "timestamp": time.time(),
                    "hardware": data,
                }
        except Exception as e:
            print(f"[ComfyUI Panel] 실시간 수집 스레드 예외: {e}")
        # 수집 간격 1초, 프론트엔드 기본 폴링 주기와 동기화
        time.sleep(1.0)

def start_runtime_collector():
    """실행 시간 데이터 수집 스레드를 시작합니다 (멱등, 한 번만 시작)）"""
    global _runtime_collector_started
    if _runtime_collector_started:
        return
    try:
        t = _threading.Thread(target=_runtime_collector_loop, daemon=True, name="bsk-runtime-collector")
        t.start()
        _runtime_collector_started = True
        print("[ComfyUI Panel] 실행 중인 데이터 수집 스레드 시작 (1초 간격))")
    except Exception as e:
        print(f"[ComfyUI Panel] 실행 시간 수집 스레드 시작 실패: {e}")

def get_runtime_cache():
    """캐시된 런타임 데이터를 읽습니다 (스레드 안전)）"""
    with _runtime_cache_lock:
        import copy
        return copy.deepcopy(_runtime_cache)


# ===== WebSocket 심장 방송（2026-09-01）=====
# 배경 (비디오 워크스테이션 "샘플링 시작 후 프로세스 미리보기 수신 불가, 페이지만 새로고침"의 근본적인 원인 중 하나).）：
#   - 모델 로드/대기 중에 ComfyUI는 몇 분 동안 프런트에 메시지를 푸시하지 않을 수 있음；
#   - nginx / 클라우드 게이트웨이의 유휴 시간 초과는 이 기간 동안 조용히 연결 끊김 WebSocket；
#   - 반 오픈 TCP 상태에서 브라우저 readyState가 여전히 OPEN이며 onclose가 트리거되지 않고, 프론트엔드는 완전히 무감각합니다.，
#     연결 실패 및 HTTP 프레임 복구 불가, 미리보기 프레임이 공중에 사라짐」。
# 대처: 백엔드에서 모든 클라이언트에 브로드캐스트 bsk_heartbeat（sid=None 즉 전체 브로드캐스트）。
#   1) 연결되고 지속적으로 트래픽, 프록시 연결 끊김 방지；
#   2) 프론트엔드에서 이 타이머를 실행하고, 장시간 응답이 없으면 링크가 끊어진 것으로 판단하고 자동으로 재연결합니다.。
# 간격은 환경 변수 BSK_WS_HEARTBEAT_SEC로 조정할 수 있습니다(0 또는 음수) = 닫기）。
_ws_heartbeat_started = False


def _ws_heartbeat_loop():
    """백그라운드 하트비트 스레드 메인 루프: 모든 WebSocket 클라이언트에게 주기적으로 브로드캐스트합니다. bsk_heartbeat。"""
    import asyncio as _asyncio
    try:
        interval = float(os.environ.get("BSK_WS_HEARTBEAT_SEC", "10") or 10)
    except Exception:
        interval = 10.0
    if interval <= 0:
        print("[ComfyUI Panel] WebSocket 심장 박동이 닫힘 (BSK_WS_HEARTBEAT_SEC<=0)")
        return
    while True:
        time.sleep(interval)
        try:
            ps = PromptServer.instance if PromptServer is not None else None
            if ps is None:
                continue
            loop = getattr(ps, "loop", None)
            if loop is None or loop.is_closed():
                continue
            payload = {"ts": time.time(), "interval": interval}
            try:
                # 전체 브로드캐스트 비동기 우선（sid=None = 모든 연결된 클라이언트）
                _asyncio.run_coroutine_threadsafe(
                    ps.send("bsk_heartbeat", payload, None), loop).result(timeout=5)
            except Exception:
                # 롤백 호환성: 일부 버전의 send는 코루틴/루프가 사용 불가능하므로 사용 send_sync
                try:
                    ps.send_sync("bsk_heartbeat", payload, None)
                except Exception:
                    pass
        except Exception as e:
            print(f"[ComfyUI Panel] WebSocket 심박 스레드 예외: {e}")


def start_ws_heartbeat():
    """WebSocket 하트비닝 스레드 시작 (멱등, 한 번만 시작)）。"""
    global _ws_heartbeat_started
    if _ws_heartbeat_started:
        return
    try:
        t = _threading.Thread(target=_ws_heartbeat_loop, daemon=True,
                              name="bsk-ws-heartbeat")
        t.start()
        _ws_heartbeat_started = True
        print("[ComfyUI Panel] WebSocket 심장 박동 스레드가 시작되었습니다 (10초 간격))")
    except Exception as e:
        print(f"[ComfyUI Panel] WebSocket 하트비트 스레드 시작 실패: {e}")


# ===== 모델 누락 감지: 노드 INPUT_TYPES의 combo 옵션에 따라 ↔ folder_paths 메뉴 파일 목록 역방향 매칭 =====
# 배경: 입력 필드 이름만으로 모델 디렉토리를 추측 (ckpt_name/lora_name…백 목록) + 필드 이름→디렉토리 매핑) 신뢰할 수 없음——
# ModelPatchLoader.name、StyleModelLoader.style_model_name、GLIGENLoader.gligen_name、
# DiffusionModelLoader.unet_name、CLIPVisionLoader.clip_name(clip_vision 아니오 text_encoders)
# 대량의 노드가 맞지 않거나 오류로 매핑될 수 있습니다。
# 직접 노드 정의의 combo 필드 옵션 목록과 각 models 디렉토리의 실제 파일 목록을 비교합니다: 특정 디렉토리의
# ComfyUI의 새롭고 오래된 버전과 호환되는 메타데이터 비활성화 스위치 "None" 사전 접두사 또는 다중 디렉토리 병합), 즉 해당 필드는
# 그 디렉토리의 모델 선택기. ComfyUI 자체 /object_info와 동일 출처이며, 0 필드 이름 추측, 새로운 노드 0 유지。
_MODEL_COMBO_DIR_CACHE = {}  # (class_type, input_key) -> 타겟 모델 디렉토리 목록


def _get_node_combo_options(class_type, input_key):
    """노드 combo 필드의 옵션 목록 가져오기; combo/가져오지 못하면 반환 None。
    오래된 INPUT_TYPES 및 새로운 API 노드(io.ComfyNode)와 호환 GET_NODE_INFO_V1）。"""
    try:
        import nodes as _comfy_nodes
        obj = _comfy_nodes.NODE_CLASS_MAPPINGS.get(class_type)
        if obj is None:
            return None
        types = None
        if hasattr(obj, "INPUT_TYPES"):
            try:
                types = obj.INPUT_TYPES()
            except Exception:
                types = None
        if types is None and hasattr(obj, "GET_NODE_INFO_V1"):
            try:
                info = obj.GET_NODE_INFO_V1()
                if isinstance(info, dict):
                    types = info.get("input")
            except Exception:
                types = None
        if not isinstance(types, dict):
            return None
        for section in ("required", "optional"):
            group = types.get(section)
            if not isinstance(group, dict):
                continue
            cfg = group.get(input_key)
            if isinstance(cfg, (tuple, list)) and len(cfg) > 0 and isinstance(cfg[0], (tuple, list)):
                return list(cfg[0])
    except Exception:
        return None
    return None


# ---- 빈 모델 디렉토리의 누락된 모델 감지 백업 경로 ----
# 평소에는 '내부 파일 전체'로 ⊆ combo "권위 반추 필드에 해당하는 모델 디렉토리 (영역 이름 추측)）。
# 모델 디렉토리가 비어 있을 경우 combo 옵션도 빈 리스트로 반환, 완전 작동 불가 —— 워크플로우 참조 모델이
# 모두 감지되지 않음 (증상: 모델이 누락되었지만, 요약은) 0）。
# 취소: 콤보 옵션이 비어 있습니다 ⇒ 필드가 실제로 바인딩된 모델 디렉터리는 반드시 빈 디렉터리여야 합니다. 그렇지 않으면 필드 이름 ↔ 디렉토리 이름
# 보수적인 일치 여부를 수행하고 '현재 비어 있음' 디렉토리만 유지하며, 부족하면 사용하고 과도하게 사용하지 않도록 하여 모델 조합이 아닌 경우를 모델 누락으로 오판하지 않습니다.。
_MODEL_FIELD_NAME_SUFFIXES = frozenset({
    "name", "file", "filename", "model", "path", "dir", "folder",
    "list", "id", "key", "value", "prefix", "option",
})
# 어근과 디렉토리 이름 접두사가 일치하지 않아 명시적으로 매핑해야 하는 약어/역사적 이름 (다른 디렉토리는 접두사 일치로 덮어쓸 수 있습니다)）
_MODEL_FIELD_NAME_ALIASES = {
    "ckpt": ("checkpoints",),
    "checkpoint": ("checkpoints",),
    "unet": ("diffusion_models",),
    "diffusion": ("diffusion_models",),
    "diffusionmodel": ("diffusion_models",),
    "clip": ("clip_vision", "text_encoders"),
}


def _norm_field_token(s):
    """필드명/디렉터리명은 소문자 숫자열(언더스코어, 하이픈, 공백 등)으로 통일）。"""
    import re as _re
    return _re.sub(r"[^a-z0-9]", "", (s or "").lower())


def _is_model_name_tail(tok):
    """토큰이 필드 이름의 접미사인지 판단하는 일반적인 '선택기' 꼬리 (번호 포함 가능, 예:) name2）。"""
    import re as _re
    return bool(_re.match(r"^(name|file|filename|model|path|dir|folder|list|id|key|value|prefix|option)\d*$", tok))


def _models_subdir_folders():
    """등록 디렉토리의 models 하위 폴더 키 목록을 반환 (확장자 등록)。"""
    folders = []
    try:
        models_dir = getattr(folder_paths, "models_dir", None)
        if not models_dir:
            return folders
        norm_models = os.path.normpath(models_dir)

        def _under(p):
            try:
                return os.path.commonpath([os.path.normpath(p), norm_models]) == norm_models
            except ValueError:  # 크로스 파티션
                return False

        for folder_name, (paths, exts) in folder_paths.folder_names_and_paths.items():
            if not paths or not exts:
                continue
            try:
                if any(_under(p) for p in paths):
                    folders.append(folder_name)
            except Exception:
                continue
    except Exception:
        pass
    return folders


def _match_model_dirs_by_field_name(class_type, input_key):
    """【오류 경로] combo 옵션이 비어있거나(모델 디렉토리에 파일이 없는 경우) 필드 이름으로 ↔ 디렉토리명 보수 일치。
    현재 확실히 비어 있는 디렉토리만 반환 (비어 있는 옵션 ⇒ 바인딩 디렉터리는 빈 디렉터리로 이어야 합니다）；
    일치하지 않거나 보존할 수 없는 경우 반환합니다 []。"""
    dirs = []
    try:
        # 필드 어근: _로 토큰 분할, 이름/파일/모델 등의 꼬리 단어 제거
        tokens = [t for t in (input_key or "").lower().split("_") if t]
        while tokens and _is_model_name_tail(tokens[-1]):
            tokens.pop()
        field_norm = _norm_field_token("_".join(tokens)) or _norm_field_token(input_key)
        if not field_norm or len(field_norm) < 3:
            return dirs

        alias_targets = {_norm_field_token(t) for t in _MODEL_FIELD_NAME_ALIASES.get(field_norm, ())}
        for folder_name in _models_subdir_folders():
            f_norm = _norm_field_token(folder_name)
            if not f_norm or len(f_norm) < 3:
                continue
            if f_norm in alias_targets:
                dirs.append(folder_name)
            elif f_norm == field_norm or f_norm.startswith(field_norm) or field_norm.startswith(f_norm):
                dirs.append(folder_name)
        # 현재 빈 디렉토리 히트만 유지
        dirs = [d for d in dirs if not folder_paths.get_filename_list(d)]
    except Exception:
        dirs = []
    return dirs


def _get_combo_model_dirs(class_type, input_key):
    """combo 필드에 해당하는 모델 디렉토리 목록을 파싱 (folder_paths 등록 순서대로)）。
    모델 조합이 아니거나 파싱에 실패하면 []를 반환합니다. 디렉토리 컬렉션은 (class_type, input_key)로 캐싱됩니다.；
    옵션 목록은 각 번(folder_paths 내부에는 mtime 캐시)을 먼저 가져옵니다. 세션 내에서 새 파일이 오작동으로 누락된 것으로 보고되지 않도록 합니다.。
    models 디렉토리가 비어있을 경우 필드 이름 보수 일치 경로로 진행, 누락된 경우에도 검출 가능。"""
    cache_key = (class_type, input_key)
    cached = _MODEL_COMBO_DIR_CACHE.get(cache_key)
    if cached is not None:
        return cached
    dirs = []
    try:
        options = _get_node_combo_options(class_type, input_key)
        if options:
            option_set = set(options)
            models_dir = getattr(folder_paths, "models_dir", None)
            for folder_name, (paths, exts) in folder_paths.folder_names_and_paths.items():
                if not paths or not exts:
                    continue
                # 단순히 모델 디렉토리 아래의 모델 디렉토리만 인식 (input/output/user/custom_nodes 등 모델이 아닌 디렉토리 제외)）
                if models_dir:
                    try:
                        norm_models = os.path.normpath(models_dir)
                        def _under_models(p):
                            try:
                                return os.path.commonpath([os.path.normpath(p), norm_models]) == norm_models
                            except ValueError:  # 크로스 파티션
                                return False
                        if not any(_under_models(p) for p in paths):
                            continue
                    except Exception:
                        continue
                try:
                    files = set(folder_paths.get_filename_list(folder_name))
                except Exception:
                    files = set()
                # All files in the directory must appear in the options (allow options to include additional items: None prefix / multi-directory merge）
                if files and files.issubset(option_set):
                    dirs.append(folder_name)
        # 일반적인 역추적 실패 시: combo 필드의 바인딩 모델 디렉터리가 비어 있어야 합니다(옵션이 비어 있거나 단 하나만 존재).
        # pixel_space 이러한 내장 자리표(placeholder), 파일⊆옵션」역방향.
        # 필드 이름 ↔ 폴더 이름 보수 매칭 성공 및 성공한 폴더의 모든 내용이 비어 있을 때만 신뢰 (필요하면 무시)，
        # 모델과 관련 없는 combo 필드가 누락된 모델로 오인되도록 방지）。
        if not dirs and options is not None:
            dirs = _match_model_dirs_by_field_name(class_type, input_key)
    except Exception:
        dirs = []
    _MODEL_COMBO_DIR_CACHE[cache_key] = dirs
    return dirs


# ===================== ComfyUI 설치기 (BSK): 프로세스 관리 =====================
# 목적: 패널 설정에서 bsk_comfyui_installer.py를 명령줄에 입력할 필요 없이 한 번 클릭하여 실행할 수 있습니다.。
# 설치기는 표준 라이브러리만 사용하므로 ComfyUI 자체 인터프리터를 직접 실행하고 추가 의존성이 없습니다.。
#
# 기본 포트: 8199, 설치기 스크립트의 --port 기본값 및 프런트엔드 버튼으로 페이지 주소 변경
# 대상 포트 세 가지 모두 일치 —— 사용자가 수동으로 켜는 것, 패널을 당기는 것, 버튼으로 접근하는 것
# 모두 동일하므로, 실행 중이면 재사용(사용자 수동으로 켰던 경우도 포함), 실행 중이 아니면 두 번째 인스턴스 생성 안 함.。
_BSK_INSTALLER_LOCK = None  # 지연 생성하여 모듈 임포트 시점에 락을 미리 선점하지 않도록
_BSK_INSTALLER_STATE = {"proc": None, "port": 0, "host": "127.0.0.1", "log": "", "via": ""}
_BSK_INSTALLER_SESSION = {"sess": None}  # aiohttp ClientSession (역할 기반, 필요에 따라 생성)）
_BSK_INSTALLER_DEFAULT_PORT = int(os.environ.get("BSK_INSTALLER_PORT", "8199"))
_BSK_INSTALLER_HOST = os.environ.get("BSK_INSTALLER_HOST", "127.0.0.1")

# 탐색 시 뒤쪽 포트가 더 많이 점유될 수 있습니다. 패널을 드래그하여 설치기를 설치할 때 기본 포트는 다른 프로그램에 의해 점유될 수 있습니다.，
# ComfyUI 메모리 포트가 재시작될 때 사라집니다 —— 이 후보들을 사용하여 '기경 사례'를 다시 식별합니다.，
# 전역 프롬프트, 전체 비디오에 고정(지속적인 캐릭터/객체/장면)。
_BSK_INSTALLER_PORT_SPAN = 6

# 탐지 심박 초과. 심박 인터페이스는 순 메모리 응답(실측 1~3ms），1.5s 이미 매우 높다；
# /api/env 구형 설치기를 위한 호환성 백업 경로입니다. pip/git 자식 프로세스를 실행합니다. 3s。
_BSK_INSTALLER_PING_TIMEOUT = 1.5
_BSK_INSTALLER_ENV_TIMEOUT = 3.0

# 설치기가 준비될 때의 최대 시간 (실시간으로 계산, 재시도 횟수로 계산하지 않음) —— 자세한 내용 참조 _bsk_installer_start）
_BSK_INSTALLER_START_TIMEOUT = 20.0

# Windows 데스크톱 실행기 (launcher/src/main.rs 컴파일 결과물, 플러그인 루트 디렉토리에 위치)）。
# 이를 사용하여 관리자를 시작하는 이점: GUI 서브시스템에 검은 상자 없음, 시스템 트레이 아이콘이 영구적으로 존재함(열고/재시작/중지/종료)）。
# It only recognizes --port parameter, default port is consistent here（8199）。
_BSK_INSTALLER_LAUNCHER_EXE = "작은 어두운 x_시작기.exe"

# 실행기 트레이 숨김 창 클래스 이름 (main.rs의) HIDDEN_CLASS）。
# 두 가지 작업을 수행: 우아하게 시작 앱 종료 (연관된 Python 함께 제거), 아이콘이 있는지 확인。
_BSK_INSTALLER_LAUNCHER_CLASS = "BSKInstallerLauncherHiddenWindow"

# 시작기를 실행할 때 더 긴 대기 제한 설정: 파이썬 직접 실행보다 두 단계 더 많음 ——
# 먼저 로컬 Python을 탐지합니다 (최악의 경우 venv/레지스트리/PATH/py 실행기를 시도합니다). 그런 다음 서비스가 준비될 때까지 기다립니다.
# （main.rs READY_TIMEOUT는 30초입니다). 20초 부족하면 40。
_BSK_INSTALLER_LAUNCHER_TIMEOUT = 40.0


def _bsk_detached_flags(breakaway_job=True):
    """Windows 상단으로 'ComfyUI는 목숨이 아니다' 프로세스를 시작하는 데 필요한 creationflags。

    부모 프로세스가 종료되면 자 프로세스가 자동으로 종료되지 않습니다 (Windows/Linux 모두 '부모 죽고 자식 따라가다'는 개념이 없습니다)），
    아래 세 가지가 함께 가져가도록 합니다. 각 항목을 차단합니다.：

      콘솔이 닫힘   —— 검은 상자를 닫을 때, 시스템은 컨트롤러에 추가된 모든 프로세스에 대해
                      CTRL_CLOSE_EVENT를 브로드캐스트하고, 누가 안에 있는지 죽게 합니다.。
                      → DETACHED_PROCESS：새 프로세스는 상속하거나 추가하지 않습니다.。
      Ctrl+C       —— 전면 프로세스 그룹에만 전송, 동일 그룹의 모든 프로세스에 전달됨。
                      → CREATE_NEW_PROCESS_GROUP：별도 프로세스 그룹 생성。
      Job 닫힘   —— 부모 프로세스가 Job Object 안에 있으면, 자 프로세스는 기본적으로 이를 상속받습니다. Job；
                      job KILL_ON_JOB_CLOSE가 설정된 경우, 작업이 닫히면 모든 구성원이 소멸합니다。
                      → CREATE_BREAKAWAY_FROM_JOB：요청이 그에 진입하지 않음 Job。

    또한 검은 상자 문제도 해결했습니다: 콘솔을 추가하지 않으면 python.exe 창이 팝업되지 않습니다.。

    두 가지 상황 중 어느 것도 막을 수 없음, 솔직하게 설명：
      · 작업 관리자 - 프로세스 트리 종료」/ taskkill /T —— 명시적 트리 순회 중 죽음, 해결 불가
      · Job breakaway 사용 불가 (breakaway_job 파라미터 참조)）——CreateProcess 직접 실패할 것
    """
    if os.name != "nt":
        return 0
    import subprocess as _sp
    return (getattr(_sp, "DETACHED_PROCESS", 0x00000008)
            | getattr(_sp, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
            | getattr(_sp, "CREATE_NO_WINDOW", 0x08000000)
            | (getattr(_sp, "CREATE_BREAKAWAY_FROM_JOB", 0x01000000) if breakaway_job else 0))


def _bsk_installer_lock():
    """느슨한 빌드 잠금: 모듈 수준의 직접 threading.Lock()은 가져오기 시점에 부작용을 발생시킵니다.。"""
    global _BSK_INSTALLER_LOCK
    if _BSK_INSTALLER_LOCK is None:
        import threading as _th
        _BSK_INSTALLER_LOCK = _th.Lock()
    return _BSK_INSTALLER_LOCK


def _bsk_installer_source_dir():
    return os.path.dirname(os.path.abspath(__file__))


def _bsk_installer_bootstrap_dir():
    """ComfyUI 내에 있지 않은 설치기 복사본 저장 위치）。

    설치기의 config_dir()와 동일한 원본 유지, 사용자 찾기 용이：
      Windows: %APPDATA%/ComfyUI-bsk_UI/installer/bootstrap/
      기타:     $XDG_CONFIG_HOME 또는 ~/.config/ComfyUI-bsk_UI/installer/bootstrap/
    """
    home = os.path.expanduser("~")
    if os.name == "nt":
        base = os.environ.get("APPDATA") or home
    else:
        base = os.environ.get("XDG_CONFIG_HOME") or os.path.join(home, ".config")
    return os.path.join(base, "ComfyUI-bsk_UI", "installer", "bootstrap")


def _bsk_installer_stamp(src_dir):
    """소스 콘텐츠 지문: 메인 스크립트 + installer/ 모든 파일 다운로드 + 실행기 exe (mtime, size)。

    플러그인 업데이트 후 지문이 변함 → 다시 로드; 변하지 않으면 기존 복사본 재사용。
    이 비교를 하지 않으면 매번 시작 시 다시 복사해야 함 installer/（268K）。

    exe 또한 지문 안에 있어야 합니다: 스크립트와 함께 제공되며, 스크립트 업데이트 시 실행 파일도 함께 업데이트됩니다.，
    bootstrap 디렉토리에 오래된 exe가 남게 됩니다, 플러그인 업그레이드 후에도 이전 버전이 시작됩니다.。
    """
    import hashlib
    items = []
    main = os.path.join(src_dir, "bsk_comfyui_installer.py")
    if os.path.isfile(main):
        try:
            st = os.stat(main)
            items.append("main:%d:%d" % (int(st.st_mtime), st.st_size))
        except Exception:
            pass
    exe = os.path.join(src_dir, _BSK_INSTALLER_LAUNCHER_EXE)
    if os.path.isfile(exe):
        try:
            st = os.stat(exe)
            items.append("exe:%d:%d" % (int(st.st_mtime), st.st_size))
        except Exception:
            pass
    web_dir = os.path.join(src_dir, "installer")
    if os.path.isdir(web_dir):
        for root, dirs, files in os.walk(web_dir):
            dirs.sort()
            for fn in sorted(files):
                p = os.path.join(root, fn)
                try:
                    st = os.stat(p)
                except Exception:
                    continue
                rel = os.path.relpath(p, src_dir).replace("\\", "/")
                items.append("%s:%d:%d" % (rel, int(st.st_mtime), st.st_size))
    return hashlib.md5("\n".join(items).encode("utf-8", "replace")).hexdigest()


def _bsk_installer_bootstrap():
    """설치기를 독립적인 디렉토리로 복사하고, (스크립트 경로, 설명))。

    【왜 자가 시작해야 하는가】
    설치기의 핵심은 ‘ComfyUI가 작동하지 않을 때도 사용할 수 있다’.
    custom_nodes/ComfyUI-bsk_UI/ 내부에서 실행, 사용자 재설치 또는 업데이트 ComfyUI，
    그 디렉터리가 완전히 재작성됩니다 —— 프로세스가 자신을 읽고 있지만, 파일은 다른 사람에게 변경되었습니다.，
    프론트엔드 리소스 404 오류가 발생하거나, 심각한 경우 직접 멈춥니다. 사용자 폴더에 넣으면 문제가 없습니다.。

    【실패 시 어떻게 해야 하나요?】
    다중 루프 (concat) 시나리오에서, 다음 구간이 필요할 때 자동으로 해당 모델 재로딩：
    자동 클론 저장소 및 설치 스크립트 실행, 설치 후 자동으로 연결。
    """
    src_dir = _bsk_installer_source_dir()
    main_src = os.path.join(src_dir, "bsk_comfyui_installer.py")
    if not os.path.isfile(main_src):
        return "", "bsk_comfyui_installer.py를 찾을 수 없습니다(플러그인 파일이 불완전함)）"

    dst_dir = _bsk_installer_bootstrap_dir()
    dst_main = os.path.join(dst_dir, "bsk_comfyui_installer.py")
    stamp_file = os.path.join(dst_dir, ".stamp")
    want = _bsk_installer_stamp(src_dir)

    # 복사본이 최신임 → 직접 사용, 매번 시작할 때마다 복사하지 마세요
    if os.path.isfile(dst_main) and os.path.isfile(stamp_file):
        try:
            with open(stamp_file, "r", encoding="utf-8") as f:
                if f.read().strip() == want:
                    return dst_main, "bootstrap"
        except Exception:
            pass

    try:
        import shutil
        # 먼저 임시 디렉터리로 복사한 다음 원자적 교체를 수행합니다. 중간에 실패하면 반쪽의 불완전한 복사본이 남지 않습니다.，
        # 그렇지 않으면 다음 시작 시 손상된 복사본을 들고 가장 최신 버전이라고 생각합니다.。
        tmp_dir = dst_dir + ".tmp"
        if os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)
        os.makedirs(tmp_dir, exist_ok=True)

        shutil.copy2(main_src, os.path.join(tmp_dir, "bsk_comfyui_installer.py"))
        web_src = os.path.join(src_dir, "installer")
        if os.path.isdir(web_src):
            shutil.copytree(web_src, os.path.join(tmp_dir, "installer"),
                            dirs_exist_ok=True)
        # 실행기 exe를 함께 가져가도록 합니다. 자체 시작 디렉터리가 관리자를 독립적으로 실행할 수 있도록 해야 합니다(트레이 메뉴 포함).），
        # 스크립트만 복사해서 실행 파일을 복사하지 않으면, 플러그인 디렉토리가 덮어지면 작동하지 않습니다.。
        exe_src = os.path.join(src_dir, _BSK_INSTALLER_LAUNCHER_EXE)
        if os.path.isfile(exe_src):
            shutil.copy2(exe_src, os.path.join(tmp_dir, _BSK_INSTALLER_LAUNCHER_EXE))
        with open(os.path.join(tmp_dir, ".stamp"), "w", encoding="utf-8") as f:
            f.write(want)

        if os.path.isdir(dst_dir):
            shutil.rmtree(dst_dir, ignore_errors=True)
        os.replace(tmp_dir, dst_dir)

        if os.path.isfile(dst_main):
            return dst_main, "bootstrap"
    except Exception as e:
        print("[ComfyUI bsk UI] 설치기 부팅 실패, 플러그인 디렉토리 실행: %s" % e)

    return main_src, "plugin-dir"


def _bsk_installer_script():
    """실행할 설치 스크립트 경로 (우선 부팅 복사본); 누락된 경우 반환 ''。"""
    p, _why = _bsk_installer_bootstrap()
    return p if os.path.isfile(p) else ""


def _bsk_installer_launcher():
    """Windows 데스크톱 시작기 exe 경로; 존재하지 않거나 Windows가 아니면 반환 ''。

    자동 복사본 우선 —— 스크립트와 같은 디렉토리에 있고 exe는 '자신의 디렉토리에서 올라가 찾음'
    bsk_comfyui_installer.py」스크립트의 (main.rs의) find_script），
    모든 플러그인 검사가 실패했습니다(서버가 준비되지 않았을 수 있음), 설치 패널이 팝업되지 않습니다.。
    """
    if os.name != "nt":
        return ""
    for d in (_bsk_installer_bootstrap_dir(), _bsk_installer_source_dir()):
        p = os.path.join(d, _BSK_INSTALLER_LAUNCHER_EXE)
        if os.path.isfile(p):
            return p
    return ""


def _bsk_installer_launcher_window():
    """실행기 트레이 숨김 창 핸들; 실행 중일 때 0 반환 (단) Windows）。

    이것은 [일반 윈도우]여야 메시지만 있는 윈도우가 아닙니다. —— 그것은 main.rs
    의도적으로 디자인되어 FindWindowW가 찾을 수 없게 되어 단일 인스턴스 활성화가 작동하지 않도록 합니다.。
    """
    if os.name != "nt":
        return 0
    try:
        import ctypes
        from ctypes import wintypes
        FindWindowW = ctypes.windll.user32.FindWindowW
        FindWindowW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR]
        FindWindowW.restype = wintypes.HWND
        return FindWindowW(_BSK_INSTALLER_LAUNCHER_CLASS, None) or 0
    except Exception:
        return 0


def _bsk_installer_launcher_running():
    """실행기(시스템 트레이)가 실행 중인지 확인。"""
    return _bsk_installer_launcher_window() != 0


def _bsk_installer_launcher_quit(hwnd, wait=6.0):
    """유연하게 종료하세요: 알려주세요 WM_CLOSE。

    【이 길을 따라야 합니다, 직접적으로는 안 됩니다 TerminateProcess】
    시작기는 WM_DESTROY에서 stop_service()를 호출합니다 —— WM_CLOSE를 수신할 때까지 완료됨
    창 닫기 프로세스, 그러면 python을 함께 정리합니다. 강제 시작 앱의 경우，
    python 서비스는 관리받지 않는 유기견이 되어, 다음 재시작 시에도 포트를 놓고 경쟁해야 합니다.。
    """
    if not hwnd:
        return False
    try:
        import ctypes
        from ctypes import wintypes
        WM_CLOSE = 0x0010
        PostMessageW = ctypes.windll.user32.PostMessageW
        PostMessageW.argtypes = [wintypes.HWND, wintypes.UINT,
                                 wintypes.WPARAM, wintypes.LPARAM]
        PostMessageW.restype = wintypes.BOOL
        if not PostMessageW(hwnd, WM_CLOSE, 0, 0):
            return False
        import time as _t
        deadline = _t.time() + wait
        while _t.time() < deadline:
            if _bsk_installer_launcher_window() == 0:
                return True
            _t.sleep(0.2)
        return _bsk_installer_launcher_window() == 0
    except Exception:
        return False


def _bsk_installer_port_open(port, host="127.0.0.1", timeout=0.3):
    """이 포트에서 이미 프로세스가 실행 중인지 확인 (TCP 연결 가능 여부)）。"""
    import socket as _sk
    if not port:
        return False
    s = _sk.socket(_sk.AF_INET, _sk.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect((host, port))
        return True
    except Exception:
        return False
    finally:
        try:
            s.close()
        except Exception:
            pass


def _bsk_installer_get_json(port, path, timeout):
    """로컬 설치기로 JSON을 요청했지만 받지 못했습니다 (연결 실패/타임아웃/JSON이 아님)을 반환합니다. None。"""
    import urllib.request as _u
    try:
        with _u.urlopen("http://127.0.0.1:%d%s" % (port, path), timeout=timeout) as r:
            if r.status != 200:
                return None
            raw = r.read().decode("utf-8", "replace")
        return json.loads(raw or "{}")
    except Exception:
        return None


def _bsk_installer_probe(port, timeout=_BSK_INSTALLER_PING_TIMEOUT):
    """이 포트에서 실행 확인**사실**설치 프로그램이며, 우연히 포트를 차지한 다른 프로그램이 아닙니다。

    조건은 그것이 자신의 문을 알 수 있다는 것입니다. 포트가 통하지만 답이 그렇지 않음 → 본인 서비스가 아닌 것을 절대 사용해서는 안 됩니다.
    저희를 이용해주세요。
    """
    return _bsk_installer_probe_info(port, timeout=timeout) is not None


def _bsk_installer_probe_info(port, timeout=_BSK_INSTALLER_PING_TIMEOUT):
    """동 _bsk_installer_probe_와 유사하지만, 설치기의 리스너 주소와 PID를 함께 가져옴。

    외부 인스턴스 (사용자 수동 실행/데스크톱 실행기 시작)는 프로세스 목록에 없으며, 모니터링되는 위치는 어디입니다?
    자신만 알고 있으므로 직접 물어봐야 함。

    【먼저 /api/ping】는 생명 검사용으로만 사용되며, 순 메모리 응답입니다(실측... 1~3ms）。
    이전에 질문한 것은 /api/env —— 이 인터페이스는 서브 프로세스를 시작해야 합니다 pip --version、git --version
    다시 스캔하여 일반 폴더를 확인하고, Windows에서는 쉽게 1초를 넘지 않으며 0.8초 초과로 멈추는 것을 방지합니다.
    설치 프로그램도 '실행되지 않았음'으로 판별되었습니다. 재사용 인터페이스를 심박수로 사용하고, 이것이 유일한 근본적인 문제입니다。
    구 버전 설치 프로그램은 ping을 사용하지 않아 환경으로 돌아갑니다(느리지만 인식은 합니다)）。
    """
    if not port:
        return None

    d = _bsk_installer_get_json(port, "/api/ping", timeout)
    if d is None:
        d = _bsk_installer_get_json(port, "/api/env", _BSK_INSTALLER_ENV_TIMEOUT)
    if not d:
        return None

    # 자신을 알리기: 새로운 버전은 자가 탐지 프로브 플래그, 이전 버전은 installer_port 필드에 의존
    if d.get("probe") != "bsk-installer" and "installer_port" not in d:
        return None

    return {
        # 리스닝 주소. ping 안에서 host를, 이전 환경에서 installer_host를 인식합니다.；
        # 받지 못하면 빈 문자열을 반환하고, 호출자는 루프 처리 —— 유지, 열리지 않는 페이지가 열리지 않음。
        "host": str(d.get("host") or d.get("installer_host") or "").strip(),
        "pid": int(d.get("pid") or 0),
        "port": int(d.get("port") or d.get("installer_port") or port),
    }


def _bsk_installer_port_candidates():
    """설치기를 찾을 포트: 기억된 포트 우선, 기본 포트 기반 구간。"""
    out = []
    remembered = int(_BSK_INSTALLER_STATE.get("port") or 0)
    if remembered:
        out.append(remembered)
    for cand in range(_BSK_INSTALLER_DEFAULT_PORT,
                     _BSK_INSTALLER_DEFAULT_PORT + _BSK_INSTALLER_PORT_SPAN):
        if cand not in out:
            out.append(cand)
    return out


def _bsk_installer_log_tail(n=25):
    """시작하지 못하면 로그의 끝 부분을 전송하여 사용자가 빙빙 돌지 않도록 해주세요.。"""
    p = _BSK_INSTALLER_STATE.get("log") or ""
    if not p or not os.path.isfile(p):
        return ""
    try:
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            return "".join(f.readlines()[-n:]).strip()
    except Exception:
        return ""


def _bsk_installer_status():
    """반환 {running, ours, external, port, bind, pid, via, launcher}。

    ours      = The process pulled up by this panel is still alive (we can control its on/off switch)）
    external  = 첫 번째 패스: 사용자 지정 sigma 샘플링 (스케줄러/단계는 무시）
    둘 다 계산 running —— 모두 연결 가능; 차이점은 '중지'만 적용 ours。
    via       = 이것은 어떻게 시작되었나요：'launcher'（exe）/ 'script'（python）/ ''
    launcher  = 데스크톱 실행기 툴바가 있는지 여부에 따라 버튼에 '툴바 준비 완료' 팁을 표시합니다」）
    """
    proc = _BSK_INSTALLER_STATE.get("proc")
    ours = False
    if proc is not None:
        if proc.poll() is None:
            ours = True
        else:
            # 프로세스가 종료되었으며, 핸들을 삭제하고, 다음에는 시체로 판단하지 마십시오.。
            # 참고: 시작기가 이미 종료되었을 수 있습니다 (기존 인스턴스를 깨운 후 종료됩니다).），
            # 종료는 서비스가 사라지는 것이 아니며, 아래에서 정상적으로 포트 접속을 시도합니다.。
            _BSK_INSTALLER_STATE["proc"] = None
            proc = None

    port = int(_BSK_INSTALLER_STATE.get("port") or 0)
    running = ours
    external = False
    # bind 프런트엔드의 '페이지 주소의 포트를 설치기 포트로 변경'하여 직접 연결할 수 있는지 결정합니다.，
    # 자체적으로 일치하는 디렉터리 내의 이 쌍은 자체적으로 일치하며, 플러그인 디렉터리가 다시 쓰여져도 여전히 작동합니다. /api/ping。
    bind = (_BSK_INSTALLER_STATE.get("host") or "127.0.0.1") if ours else ""

    if not running:
        # 우리가 시작한 것이 아니면 (또는 이미 종료된 경우): 개별 후보 포트 탐색。
        # 사용자가 수동으로 켜는 것, 데스크톱 실행기에서 시작하는 것, 마지막 ComfyUI에서 시작하여 고아 상태가 된 것
        # 그것은, 이 포트 구간 안에 전부 —— 찾으면 재사용, ‘내가 낳은 것이 아니다’는 이유로 새 프로젝트 생성하지 않음。
        for cand in _bsk_installer_port_candidates():
            info = _bsk_installer_probe_info(cand)
            if info is None:
                continue
            port, running, external = cand, True, True
            bind = info["host"] or bind or "127.0.0.1"
            # 인식되면 기록하고, 다음에는 그것부터 시작하여 불필요한 과정을 줄여보세요.
            _BSK_INSTALLER_STATE["port"] = cand
            break

    via = _BSK_INSTALLER_STATE.get("via") or ""
    # 저희가 직접 처리하지 않은 경우(ComfyUI 재시작 또는 사용자가 수동으로 시작한 경우), 포트가 기본값인 경우에만 exe를 사용하고, 그렇지 않으면 스크립트를 솔직하게 실행합니다.）→ 판단하기 위해 토판이 있는지 확인：
    # 시스템 트레이에 표시됨, 즉 시작기에서 끌어온 것이므로 via는 launcher。
    if not via and running and _bsk_installer_launcher_running():
        via = "launcher"

    return {
        "running": bool(running),
        "ours": ours,
        "external": bool(external),
        "port": port if running else 0,
        "bind": bind if running else "",
        "pid": (proc.pid if (ours and proc is not None) else 0),
        "via": via,
        "launcher": _bsk_installer_launcher_running(),
    }


def _bsk_installer_bind_host():
    """설치기가 어떤 주소를 리스닝해야 하는지 —— ComfyUI 자체 리스너 주소 따라가기。

    Open the manager in the browser is "replace the current page address ComfyUI port with installer port」，
    이 규칙이 적용되려면 설치기가 브라우저 한쪽에 접근 가능하고 ComfyUI와 일치해야 함：
    ComfyUI --listen 0.0.0.0 으로 로컬 네트워크 / 클라우드에 노출될 때, 브라우저는 다른 기계에서
    연결된 상태에서, 설치기가 아직 127.0.0.1만 리슨하면 포트 변경 후 브라우저에서 해당 주소를 확인
    그 기계에는 존재하지 않으므로, 되돌아가기 지점에 대리 실행 /bsk/installer/ —— 반대 대행
    ComfyUI ComfyUI 관리자가 중단되면 따라 중단되므로, 바로 그 때 사용해야 합니다.。

    ComfyUI 자신의 노출 방식은 설치기 방식과 동일하며, 노출 면적은 현재보다 더 큽니다.。
    BSK_INSTALLER_HOST 환경 변수 우선, 수동으로 지정된 포맷 유지。
    """
    env = (os.environ.get("BSK_INSTALLER_HOST") or "").strip()
    if env:
        return env
    addr = ""
    try:
        from server import PromptServer as _PS
        inst = getattr(_PS, "instance", None)
        addr = str(getattr(inst, "address", "") or "").strip()
    except Exception:
        addr = ""
    # 반향/주소가 없으면 (주소는 start()에서만 할당됨) 반향을 사용하여 처리하고 새로운 통로를 열지 않습니다
    if addr and addr.lower() not in ("127.0.0.1", "localhost", "::1", "[::1]"):
        return addr
    return _BSK_INSTALLER_HOST


def _bsk_installer_start():
    """설치기를 시작합니다(실행되기 전에 시작하고 실행되면 직접 재사용). 차단되어 있으므로 실행자 호출에 포함하십시오。

    Windows 상위 우선적으로 데스크톱 실행기 exe (검은 상자 없음) + 토끼통 아이콘, 사용자가 직접 종료），
    조건이 충족되지 않으면 파이썬 스크립트 직접 실행으로 되돌립니다. —— 토플릿이 조금이라도 적어도 괜찮지만, 시작할 수 없으면 안됨。
    """
    import subprocess as _sp
    import time as _time

    with _bsk_installer_lock():
        st = _bsk_installer_status()
        if st["running"]:
            return dict(st, success=True, msg="설치기가 실행 중입니다")

        script = _bsk_installer_script()
        if not script:
            return {"success": False, "error": "bsk_comfyui_installer.py를 찾을 수 없습니다(플러그인 파일이 불완전함)）"}

        # 선택하지 않은 포트 하나를 선택하세요
        port = _BSK_INSTALLER_DEFAULT_PORT
        for _ in range(20):
            if not _bsk_installer_port_open(port):
                break
            port += 1
        else:
            return {"success": False,
                    "error": "포트 %d~%d 모두 사용 중이며 설치 프로그램을 시작할 수 없습니다" % (_BSK_INSTALLER_DEFAULT_PORT, port)}

        # ---- 어떤 방식으로 시작할지 결정하세요 ----
        launcher = _bsk_installer_launcher()
        # exe 단순히 기본 포트를 인식합니다: 포트가 순차적으로 증가하면 따라가지 못합니다 (우리 측에서 전달한 --port 때문이지만, 순차적으로 증가하면 따라가지 못합니다)
        # 설명: 기본 포트가 이미 사용 중입니다. 실행 파일에서 "포트가 다른 프로그램에 의해 사용 중입니다" 메시지가 표시됩니다.，
        # 그리고 여기에서 응답 대기——사용자 확인 클릭을 기다리고, 저희는 타임아웃을 기다려 두자, 둘 다 멈춰버림）。
        # 자체적으로 프로세스 테이블을 가져와야 하고, 외부 인스턴스가 스스로에게 질문해야 합니다.。
        use_launcher = bool(launcher) and port == _BSK_INSTALLER_DEFAULT_PORT

        if use_launcher:
            return _bsk_installer_start_via_launcher(launcher, port)

        # ComfyUI 리스너 주소 따르기: 해당 주소에 접근 가능해야 설치기 또한 접근 가능，
        # 프론트엔드 '포트 직접 연결' 규칙 유효 (자세한 내용은 참조) _bsk_installer_bind_host）
        host = _bsk_installer_bind_host()
        # --no-open 다음 사항을 포함해야 합니다: 스크립트가 webbrowser.open을 가지고, 양쪽 모두 브라우저를 열면 두 개의 탭이 열립니다
        cmd = [sys.executable, script, "--host", host, "--port", str(port), "--no-open"]

        # 로그를 디스크에 저장해야 합니다: ComfyUI는 일반적으로 콘솔 없이 실행됩니다(GUI 서브시스템/서비스)），
        # 자식 프로세스의 print가 직접 실패, 리디렉션되지 않으면 모든 오류가 손실됨。
        import tempfile as _tf
        log_path = os.path.join(_tf.gettempdir(), "bsk_installer.log")
        _BSK_INSTALLER_STATE["log"] = log_path

        try:
            lf = open(log_path, "a", encoding="utf-8", errors="replace")
        except Exception:
            lf = _sp.DEVNULL

        kw = {
            "cwd": os.path.dirname(script),
            "stdout": lf,
            "stderr": _sp.STDOUT,
            "stdin": _sp.DEVNULL,
        }
        if os.name == "nt":
            kw["creationflags"] = _bsk_detached_flags()
        else:
            # 세션 종료 및 터미널 제어: ComfyUI가 Ctrl+C / 닫고 다시 시작，
            # 설치 프로그램은 따라가지 않아야 합니다 ( _bsk_detached_flags 설명 참조)）
            kw["start_new_session"] = True

        try:
            proc = _sp.Popen(cmd, **kw)
        except OSError:
            # Windows CREATE_BREAKAWAY_FROM_JOB을 사용할 때, 부 프로세스에 있는 Job이 허용하지 않으면
            # 분리, CreateProcess가 직접 실패합니다. 시작이 안 되는 것이 아니라 분리될 뿐입니다. ——
            # 그 플래그를 제거하고 한 번 더 시도하십시오, 자리가 있어도 실행되도록 하십시오。
            if os.name == "nt":
                kw["creationflags"] = _bsk_detached_flags(breakaway_job=False)
                proc = _sp.Popen(cmd, **kw)
            else:
                raise
        except Exception as e:
            if lf is not _sp.DEVNULL:
                try:
                    lf.close()
                except Exception:
                    pass
            return {"success": False, "error": "시작 설치기 실패: %s" % e}

        _BSK_INSTALLER_STATE["proc"] = proc
        _BSK_INSTALLER_STATE["port"] = port
        _BSK_INSTALLER_STATE["host"] = host
        _BSK_INSTALLER_STATE["via"] = "script"

        # 자식 프로세스가 Popen 시 자신의 fd를 상속받았으므로, 부모 프로세스의 fd는 닫을 수 있습니다 ——
        # 전원이 꺼져 있으면 각 시작 시 핸들 하나 누락。
        if lf is not _sp.DEVNULL:
            try:
                lf.close()
            except Exception:
                pass

        return _bsk_installer_wait_ready(proc, port, log_path,
                                         _BSK_INSTALLER_START_TIMEOUT,
                                         exit_is_fatal=True,
                                         ok_msg="설치기 시작됨")


def _bsk_installer_start_via_launcher(launcher, port):
    """Windows：‘작은 어두운 시작기.exe’를 사용하여 관리자를 드래그합니다。

    사용자가 원하는 이점: GUI 서브시스템에 검은 상자 없음, 시스템 트레이에 자리 잡아 언제든지 종료 가능。
    최적화 단계, 2 권장 GPU 최적화, 3 권장 대규모 모델.：

    1. 【브라우저가 자동으로 열립니다(main.rs에서 서비스 준비 완료 후 무조건) open_browser()。
       그래서 프론트엔드에 가져옴 via='launcher' 때로는 자신의 자리 표시 페이지를 닫아야 하며, 그렇지 않으면 두 개의 탭이。
    2. 【단일 인스턴스] 이미 실행 중이면, 새로 시작한 것은 PostMessage를 통해 기존 인스턴스 호출
       종료 —— 그러므로 'Popen 나온 프로세스가 종료되었다'는 것은 실패가 아니므로, 이를 근거로 사망을 판결할 수 없다。
    3. 【자기 자신 찾기 Python】venv → 레지스트리 → PATH → py 시작기, 반드시 ~는 아닙니다 ComfyUI
       해석기. 설치기는 표준 라이브러리만 사용하고 패키지 설치 시 '사용자 지정 대상 해석기' 사용」，
       그러므로 상관없습니다. 반대로 훨씬 더 안정적입니다: ComfyUI의 venv가 삭제되어도 실행됩니다.。
    4. 【it doesn't support --host] --port만 인식합니다. 다행히 기본적으로 로컬에만 수신하므로 충분합니다。
    """
    import subprocess as _sp

    cmd = [launcher, "--port", str(port)]
    try:
        # 실행기는 GUI 프로그램이며 콘솔이 필요 없습니다; DETACHED_PROCESS 세트는 그대로 적용됩니다.，
        # ComfyUI 콘솔/프로세스 그룹/Job이 함께 삭제되지 않도록。
        proc = _sp.Popen(cmd,
                         cwd=os.path.dirname(launcher),
                         stdin=_sp.DEVNULL, stdout=_sp.DEVNULL, stderr=_sp.DEVNULL,
                         creationflags=_bsk_detached_flags())
    except OSError:
        # Job breakaway를 허용하면 CreateProcess가 실패할 수 있습니다. 플래그를 제거하고 다시 시도해 보세요.
        try:
            proc = _sp.Popen(cmd,
                             cwd=os.path.dirname(launcher),
                             stdin=_sp.DEVNULL, stdout=_sp.DEVNULL, stderr=_sp.DEVNULL,
                             creationflags=_bsk_detached_flags(breakaway_job=False))
        except Exception as e:
            return {"success": False, "error": "시작기 시작 실패: %s" % e}
    except Exception as e:
        return {"success": False, "error": "시작기 시작 실패: %s" % e}

    _BSK_INSTALLER_STATE["proc"] = proc
    _BSK_INSTALLER_STATE["port"] = port
    _BSK_INSTALLER_STATE["host"] = "127.0.0.1"
    _BSK_INSTALLER_STATE["via"] = "launcher"

    # 시작기의 자체 로그（main.rs: temp_dir/bsk_installer_launcher.log），
    # 스크립트의 bsk_installer.log 파일이 아님, 오류 발생 시 읽어야 원인 파악 가능。
    import tempfile as _tf
    log_path = os.path.join(_tf.gettempdir(), "bsk_installer_launcher.log")
    _BSK_INSTALLER_STATE["log"] = log_path

    # exit_is_fatal=False：위에서 언급한 2번 항목, 프로세스 종료는 실패가 아님
    r = _bsk_installer_wait_ready(proc, port, log_path,
                                  _BSK_INSTALLER_LAUNCHER_TIMEOUT,
                                  exit_is_fatal=False,
                                  ok_msg="설치기 시작됨 (데스크톱 설치기)）")
    if r.get("success"):
        r["via"] = "launcher"
    return r


def _bsk_installer_wait_ready(proc, port, log_path, timeout,
                              exit_is_fatal, ok_msg):
    """서비스 준비 완료。

    【벽시계로 시간을 측정하며, 횟수로 측정하지 않습니다['루프 60회'], 핀을 사용하여 9초로 되돌아 계산합니다；
    탐지기가 타임아웃되면 (오래된 버전은 /api/env에서 3초) 한 번에 3.15초 소요됩니다.，
    60 그 다음에는 세 분 정도 —— 프론트엔드는 이 기간 동안 아무런 피드백이 없으며, '반응 없음'이 됩니다」。
    이제 실제 시간에 따라 제한하고, 펜치가 느리더라도 따라잡지 못합니다。

    exit_is_fatal: 자식 프로세스가 종료되었는지 즉시 실패로 판단。
      스크립트 방식 = True（스크립트 종료되면 완전히 사라짐）
      실행기 방식 = False（이는 '기존 인스턴스 깨우기'를 의미하며, 서비스는 이전 인스턴스에 의해 관리됩니다.）
    """
    import time as _time

    deadline = _time.time() + timeout
    exited_at = 0.0
    while True:
        if _bsk_installer_probe(port):
            st = dict(_bsk_installer_status(), success=True, msg=ok_msg)
            st["via"] = _BSK_INSTALLER_STATE.get("via") or "script"
            return st
        if proc is not None and proc.poll() is not None and exit_is_fatal:
            return {"success": False,
                    "error": "설치기 프로세스가 종료됨 (로그 참조) %s）" % log_path,
                    "log": _bsk_installer_log_tail()}
        # 프로세스가 종료되더라도 사망 판결이 내려지지 않으면, 너무 오래 지속되지 않도록 5초의 유예 기간을 주어 이전 인스턴스가 서비스를 시작하도록 합니다.
        if proc is not None and proc.poll() is not None and not exit_is_fatal:
            if not exited_at:
                exited_at = _time.time()
            elif _time.time() - exited_at > 5.0:
                return {"success": False,
                        "error": "실행기 종료 및 포트 %d 응답 없음 (자세한 내용은 로그 참조) %s）" % (port, log_path),
                        "log": _bsk_installer_log_tail()}
        if _time.time() >= deadline:
            break
        _time.sleep(0.15)

    return {"success": False, "error": "설치기 시작 시간 초과 (포트 %d 응답 없음）" % port,
            "log": _bsk_installer_log_tail()}


def _bsk_installer_stop():
    """우리만 중지하고 외부 인스턴스는 움직이지 않음 (다른 사람이 창을 열고 사용 중일 수 있음)）。

    시작기를 먼저 "알려주세요"라고 요청해서 종료하세요 —— 그것은 WM_DESTROY에서 시작됩니다.
    python 모두 제거하고, 강제 종료하면 Python이 유일한 자식 프로세스로 남게 되므로, 다음 시작 시 포트 충돌을 해결해야 합니다.。
    """
    with _bsk_installer_lock():
        # 실행기 실행 중 (우리에게 시작했는지 여부와 관계없이, 토큰바가 있으면 관리)）
        hwnd = _bsk_installer_launcher_window()
        if hwnd:
            if _bsk_installer_launcher_quit(hwnd):
                _BSK_INSTALLER_STATE["proc"] = None
                _BSK_INSTALLER_STATE["via"] = ""
                return {"success": True, "msg": "시스템 트레이에서 실행기를 종료했습니다 (서비스도 함께 중지)）"}
            # 아래로 가지 마시고 강제 종료하세요

        proc = _BSK_INSTALLER_STATE.get("proc")
        if proc is None or proc.poll() is not None:
            _BSK_INSTALLER_STATE["proc"] = None
            _BSK_INSTALLER_STATE["via"] = ""
            return {"success": True, "noop": True, "msg": "패널에서 시작한 설치 프로그램 없음"}
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        _BSK_INSTALLER_STATE["proc"] = None
        _BSK_INSTALLER_STATE["via"] = ""
        return {"success": True, "msg": "설치기 중지됨"}


# ===== 자체 재시작 기능 =====
# 패널의 '재시작' 버튼은 이전에는 ComfyUI-Manager(/manager/reboot) 또는
# ComfyUI-Easy-Use(/easyuse/reboot) 인터페이스, 이 플러그인을 설치하지 않으면 무기력。
# 자체 구현하여 명령 조립 로직을 Manager와 일치시키고, 어떤 환경에서도 외부 플러그인에 의존하지 않습니다.。

def _bsk_build_restart_cmd():
    """재시작 명령어 생성。
    1. frozen(PyInstaller) 구축 → [exe] + 매개변수 (sys.argv[0] 은 exe 자신이며, 제거)）
    2. __main__.py 실행 (python -m comfyui 등)）→ [python, -m, 모듈명] + 매개변수
    3. 일반 스크립트 시작 → [python, main.py] + 매개변수
    """
    sys_argv = list(sys.argv)
    # 독립적인 잔류 파라미터 빌드, 재시작 시 제거해야 하며, 그렇지 않으면 새 프로세스가 압축/설치 프로세스를 다시 실행합니다.
    sys_argv = [a for a in sys_argv if a != '--windows-standalone-build']

    if getattr(sys, 'frozen', False):
        return [sys.executable] + sys_argv[1:]

    if sys_argv and sys_argv[0].endswith('__main__.py'):
        module_name = os.path.basename(os.path.dirname(sys_argv[0]))
        return [sys.executable, '-m', module_name] + sys_argv[1:]

    return [sys.executable] + sys_argv


def _bsk_do_reboot(delay=0.8):
    """실제 재시작 작업: 백그라운드 스레드에서 실행: 먼저 HTTP 응답을 보내고 프로세스를 교체。"""
    import time

    try:
        time.sleep(delay)
    except Exception:
        pass

    # ComfyUI stdout을 로그 스트림으로 리디렉션하고, 재시작 전에 줄 바꿈을 처리하여 새 프로세스가 유효한 핸들러를 받지 않도록 방지
    try:
        sys.stdout.flush()
    except Exception:
        pass
    try:
        sys.stdout.close_log()
    except Exception:
        pass

    # comfy-cli 세션: .reboot 마커 작성 후 종료, CLI가 재시작
    if '__COMFY_CLI_SESSION__' in os.environ:
        try:
            with open(os.environ['__COMFY_CLI_SESSION__'] + '.reboot', 'w'):
                pass
        except Exception:
            pass
        print("\n[bsk] Restarting... (comfy-cli session)\n", flush=True)
        try:
            sys.stdout.flush()
        except Exception:
            pass
        os._exit(0)

    cmds = _bsk_build_restart_cmd()
    print(f"\n[bsk] Restarting... Command: {cmds}\n", flush=True)
    try:
        # execv 현장 대체 프로세스: PID는 변경되지 않으며, 시작기/역배/포트 점유 관계 유지
        os.execv(sys.executable, cmds)
    except Exception as e:
        print(f"[bsk] execv 실패({e})，새 프로세스로 되돌립니다 + 종료 이전 프로세스」", flush=True)
        try:
            import subprocess
            subprocess.Popen(cmds, cwd=os.getcwd())
            os._exit(0)
        except Exception as e2:
            print(f"[bsk] 재시작 실패: {e2}", flush=True)


def setup_api():
    """API 라우트 등록"""
    try:
        from aiohttp import web
        from server import PromptServer
        import folder_paths

        server = PromptServer.instance

        # ===== 디버그 핀 (독립 스레드, 이벤트 루프 차단 안 함)) =====
        # 기본적으로 시작되지 않음, API를 통해 필요에 따라 시작하여 리소스 절약
        try:
            from . import debug_probe
            debug_probe_available = True
        except Exception:
            debug_probe_available = False

        # ===== 디버깅 핀치 API (필요에 따라 시작)) =====
        if debug_probe_available:
            @server.routes.post("/comfyui_panel/debug_probe")
            async def debug_probe_control(request):
                """시작/중지 디버그 핀 꽂기
                POST body: {"action": "start"|"stop"}
                시작 시 포트 8229를 감시하고, 브라우저에서 http://localhost:8229/로 접속하여 실시간 상태 확인
                (기본 포트는 debug_probe._PROBE_PORT이며, BSK_DEBUG_PROBE_PORT로 덮어쓸 수 있습니다.)
                """
                try:
                    data = await request.json()
                    action = (data.get("action") or "").strip()
                    _probe_port = getattr(debug_probe, "_PROBE_PORT", 8229)
                    if action == "start":
                        if debug_probe._started:
                            return web.json_response({"success": True, "msg": "디버그 핀이 실행 중입니다"})
                        debug_probe.start()
                        return web.json_response({"success": True, "msg": f"디버그 핀이 시작되었습니다 → http://localhost:{_probe_port}/"})
                    elif action == "stop":
                        if not debug_probe._started:
                            return web.json_response({"success": True, "msg": "디버그 핀이 실행되지 않았습니다"})
                        # 탐침 스레드는 데몬이며, 정상 종료가 불가능합니다. 하지만 미시작으로 표시합니다.
                        debug_probe._started = False
                        return web.json_response({"success": True, "msg": "디버깅 핀이 중지됨 (daemon 스레드는 메인 프로세스 종료 시 자동으로 종료됨))"})
                    else:
                        return web.json_response({"success": False, "error": f"알 수 없는 동작: {action}"}, status=400)
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    return web.json_response({"success": False, "error": str(e)})

        # ===== 자체 재시작 인터페이스 =====
        # 먼저 200 응답을 반환하여 프론트엔드에 확인을 받고, 백엔드 스레드에서 execv를 사용하여 현지 재시작(PID는 변경되지 않음)）。
        # 프론트엔드 우선순위：/bsk/reboot → /easyuse/reboot → /manager/reboot。
        @server.routes.post("/bsk/reboot")
        async def bsk_reboot(request):
            """ComfyUI 재시작, 의존하지 않음 ComfyUI-Manager / EasyUse。"""
            try:
                import threading as _bsk_threading
                _bsk_threading.Thread(
                    target=_bsk_do_reboot,
                    kwargs={"delay": 0.8},
                    daemon=True,
                    name="bsk-reboot",
                ).start()
                return web.json_response({"success": True, "msg": "ComfyUI 재시작 중"})
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)}, status=500)

        # ===== ComfyUI 설치기（BSK）=====
        # 패널 설정의 'ComfyUI 관리자' 버튼의 백엔드: 필요에 따라 드래그 업 bsk_comfyui_installer.py。
        # 이미 실행 중인 경우 재사용(사용자가 수동으로 시작한 경우도 포함), 실행되지 않은 경우 시작합니다.。
        # 세 프로세서 모두 블로킹 작업 (자식 프로세스 시작) + 포트 탐지 + 대기 준비됨) executor，
        # aiohttp의 이벤트 루프를 절대 막지 마세요 —— ComfyUI 설치기 시작보다 200ms 더 늦게 멈추는 것은 훨씬 심각함。
        import asyncio as _bsk_aio

        async def _bsk_installer_call(fn):
            loop = _bsk_aio.get_event_loop()
            return await loop.run_in_executor(None, fn)

        @server.routes.get("/bsk/installer/status")
        async def bsk_installer_status(request):
            """설치기가 실행 중인지, 어떤 포트에서 실행 중인지.」。"""
            try:
                st = await _bsk_installer_call(_bsk_installer_status)
                return web.json_response({"success": True, **st})
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        @server.routes.post("/bsk/installer/start")
        async def bsk_installer_start(request):
            """설치기를 시작하고 준비된 포트를 반환합니다. 실패 시 로그 꼬리를 가져옵니다.。"""
            try:
                st = await _bsk_installer_call(_bsk_installer_start)
                return web.json_response(st)
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        @server.routes.post("/bsk/installer/stop")
        async def bsk_installer_stop(request):
            """이 패널에서 시작한 설치기를 중지합니다. 외부 인스턴스는 건드리지 마십시오。"""
            try:
                st = await _bsk_installer_call(_bsk_installer_stop)
                return web.json_response(st)
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        # ===== ComfyUI 설치기: 동일 원본 역방향 프록시 =====
        # 저장 /bsk/installer/** 설치 프로세스로 전달. 동일 원천 후 브라우저는 더 이상 직접 연결하지 않음
        # 127.0.0.1:<port>，그러면：
        #   · 클라우드/원격 배포(ComfyUI가 다른 기기 또는 전체 시스템에서 https를 실행)
        #   · iframe 내장되지 않음, 교차 출처 정책 차단 안됨
        # 설치기가 이미 프록시를 위한 적응형을 자체적으로 설정했습니다: X-Script-Name을 주입합니다 <base href> 및
        # window.__APP_BASE__，프론트엔드의 모든 /api 요청은 이 접두사로 조합합니다 —— 따라서 전달할 때만 '근'
        # 전체 이름은 그대로 전달, installer/ 아래 프론트 코드 수정 불필요。
        _BSK_INSTALLER_PREFIX = "/bsk/installer/"
        # 요청 본문의 메모리 제한, 초과 시 임시 파일(모델 업로드 시 며칠 MB로 상당한 양, 메모리에 모두 채울 수 없음)에 저장합니다.）
        _BSK_INSTALLER_BODY_MEM_LIMIT = 8 * 1024 * 1024

        async def _bsk_installer_session():
            """재사용 ClientSession: 요청 핸드셰이크가 있는 경우 설치 페이지가 멈추는 현상이 발생。"""
            import aiohttp as _paio
            s = _BSK_INSTALLER_SESSION.get("sess")
            if s is None or s.closed:
                s = _paio.ClientSession(
                    timeout=_paio.ClientTimeout(total=None, sock_connect=10, sock_read=300)
                )
                _BSK_INSTALLER_SESSION["sess"] = s
            return s

        async def _bsk_installer_read_body(request):
            """요청 본문을 조정하고 반환합니다 (bytes | None, 임시 파일 경로)。

            설치기는 http.server, Content-Length만 인식, 지원 안 함 chunked —— 그래서 포워딩하기 전에
            전체 길이를 알아야 하고, request.content를 직접 스트리밍하여 아래로 푸드할 수 없습니다.。
            작은 요청은 메모리에 유지하고, 제한 초과 시 임시 파일에 저장하여 큰 파일 업로드로 메모리 과부하 방지。
            """
            if not getattr(request, "can_read_body", True):
                return None, ""
            body = b""
            tmp_path = ""
            fh = None
            try:
                while True:
                    chunk = await request.content.read(1 << 16)
                    if not chunk:
                        break
                    if not tmp_path and len(body) + len(chunk) <= _BSK_INSTALLER_BODY_MEM_LIMIT:
                        body += chunk
                        continue
                    if not tmp_path:
                        import tempfile as _ptf
                        fd, tmp_path = _ptf.mkstemp(prefix="bsk_installer_body_")
                        fh = os.fdopen(fd, "wb")
                        fh.write(body)
                        body = b""
                    fh.write(chunk)
            finally:
                if fh is not None:
                    fh.close()
            if tmp_path:
                return None, tmp_path
            return body, ""

        @server.routes.get("/bsk/installer")
        async def bsk_installer_proxy_root(request):
            """꼬리 추가 슬래시: 프런트엔드 리소스는 상대 경로이며, /static/이 누락되면 루트 사이트로 해석됩니다.。"""
            return web.HTTPFound(_BSK_INSTALLER_PREFIX)

        async def _bsk_installer_proxy(request):
            tail = request.match_info.get("tail", "") or ""

            # 실행 중이 아니면 편리하게 띄우세요: 사용자가 열면 사용하고, 보이지 않아야 합니다. 503
            st = await _bsk_installer_call(_bsk_installer_status)
            if not (st.get("running") and st.get("port")):
                st = await _bsk_installer_call(_bsk_installer_start)
            if not (st.get("success") or (st.get("running") and st.get("port"))):
                return web.Response(
                    text="ComfyUI 관리자 시작 실패：%s" % (st.get("error") or "알 수 없는 이유"),
                    status=503, content_type="text/plain; charset=utf-8")

            target = "http://127.0.0.1:%d/%s" % (st["port"], tail)
            if request.query_string:
                target += "?" + request.query_string

            # 요청 헤더 전달。Host / Content-Length / Content-Encoding / Transfer-Encoding
            # 무엇보다 제거해야 할 것은 ‘브라우저’를 설명하는 것들입니다↔ComfyUI」이 부분, 그대로 아래로 전달 aiohttp
            # 데스크톱 실행기 (launcher/src/main.rs 컴파일 결과물, 플러그인 루트 디렉토리)。
            fwd_headers = {}
            for k, v in request.headers.items():
                if k.lower() in ("host", "content-length", "content-encoding",
                                 "transfer-encoding", "connection", "x-script-name"):
                    continue
                fwd_headers[k] = v
            fwd_headers["X-Script-Name"] = _BSK_INSTALLER_PREFIX

            body, tmp_path = await _bsk_installer_read_body(request)
            data = body
            fh = None
            if tmp_path:
                fh = open(tmp_path, "rb")
                data = fh
            if data is not None:
                n = os.path.getsize(tmp_path) if tmp_path else len(body)
                fwd_headers["Content-Length"] = str(n)

            sess = await _bsk_installer_session()
            try:
                resp = await sess.request(request.method, target,
                                          headers=fwd_headers, data=data,
                                          allow_redirects=False)
            except Exception as e:
                return web.Response(text="ComfyUI 관리자 연결 실패: %s" % e,
                                    status=502, content_type="text/plain; charset=utf-8")
            finally:
                if fh is not None:
                    try:
                        fh.close()
                    except Exception:
                        pass
                if tmp_path:
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass

            async with resp:
                out_headers = {}
                for k, v in resp.headers.items():
                    if k.lower() in ("transfer-encoding", "connection",
                                     "content-length", "content-encoding"):
                        continue
                    out_headers[k] = v
                for k in list(out_headers):
                    if k.lower() == "location":
                        loc = out_headers[k] or ""
                        if loc.startswith("/"):
                            out_headers[k] = _BSK_INSTALLER_PREFIX + loc.lstrip("/")

                out = web.StreamResponse(status=resp.status, headers=out_headers)
                await out.prepare(request)
                try:
                    async for chunk in resp.content.iter_any():
                        await out.write(chunk)
                finally:
                    await out.write_eof()
                return out

        # server.routes(RouteTableDef)에 등록 server.app.router ——
        # 이것은 UrlDispatcher이며, add_routes가 없으며, ComfyUI는 앱 시작 전에
        # RouteTableDef 일시적으로 연결하면 라우터에서 경로를 추가할 때 빈 상태로 뜰 수 있음。
        server.routes.get(_BSK_INSTALLER_PREFIX + "{tail:.*}")(_bsk_installer_proxy)
        server.routes.post(_BSK_INSTALLER_PREFIX + "{tail:.*}")(_bsk_installer_proxy)

        async def _bsk_installer_cleanup(app):
            """ComfyUI 종료 시 연결 풀만 수집。

            설치기 멈추는 것을 의도적으로 '아니오'로 설정 —— 이는 독립 프로세스로 살아있는 것이 정당한 의도：
            ComfyUI 멈췄습니다 / 재시작 중 / 사용자가 수동으로 중단했습니다, 모두 이를 통해 의존성을 설치해야 합니다、
            문제 플러그인 제거, 환경 재구성. 함께 종료하면 생명줄 끊는 것
            와 환자에게 배를 싣다。
            사용자가 중단하려면 자체적으로 레이블 탭 닫거나 사용 /bsk/installer/stop。
            """
            s = _BSK_INSTALLER_SESSION.get("sess")
            if s is not None and not s.closed:
                try:
                    await s.close()
                except Exception:
                    pass
            _BSK_INSTALLER_SESSION["sess"] = None

        server.app.on_cleanup.append(_bsk_installer_cleanup)

        # 훈련 도구 API
        try:
            from .training_api import setup_training_api
            setup_training_api(app=server.app)
        except Exception as e:
            print(f"[ComfyUI bsk UI] 훈련 도구 API 로드 실패: {e}")

        # 하드웨어 통계 API (포함 CPU、RAM、GPU）
        @server.routes.get("/comfyui_panel/hardware_stats")
        async def get_hardware_stats(request):
            """하드웨어 사용률 통계 가져오기（CPU、RAM、GPU、VRAM）"""
            try:
                hw_stats = get_hardware_stats_instance()
                data = hw_stats.getStatus()
                return web.json_response({
                    "success": True,
                    **data
                })
            except Exception as e:
                return web.json_response({
                    "success": False,
                    "error": str(e),
                    "cpu_utilization": -1,
                    "ram_total": 0,
                    "ram_used": 0,
                    "ram_percent": 0,
                    "gpu_available": False,
                    "gpus": []
                })

        # 보존 이전 API 호환성
        @server.routes.get("/comfyui_panel/gpu_stats")
        async def get_gpu_stats_compat(request):
            """GPU 사용률 통계 가져오기 (구 버전 호환)）"""
            try:
                hw_stats = get_hardware_stats_instance()
                data = hw_stats.getStatus()
                return web.json_response({
                    "success": True,
                    "available": data.get('gpu_available', False),
                    "gpus": data.get('gpus', [])
                })
            except Exception as e:
                return web.json_response({
                    "success": False,
                    "error": str(e),
                    "available": False,
                    "gpus": []
                })

        # ===== 통합 런타임 인터페이스 (하드웨어 모니터링 병합) + 큐 검색 + 연결 상태）=====
        # 프론트엔드는 원래 3개의 독립적인 타이머를 사용하여 각각 순회함 /queue、/hardware_stats、/system_stats，
        # 잦은 요청 + 동기 차단이 게이트웨이 504를 유발할 수 있습니다. 단일 인터페이스로 병합하면 프런트엔드에는 타이머 하나만 필요합니다.。
        # 하드웨어 데이터는 백엔드 수집 스레드 캐시에서 가져옴 (마이크로초 단위, 이벤트 루프 차단하지 않음)）；
        # 큐 데이터는 PromptServer.prompt_queue(순 메모리 작업, 매우 빠름)）。
        start_runtime_collector()
        # WebSocket 하트비트: 연결을 유지하고 트래픽이 지속적으로 있는지 확인하여 프록시가 비활성으로 끊어지는 것을 방지하고 과정 미리보기 '공허하게 사라지는' 것을 방지합니다.」，
        # 그리고 프런트엔드는 이에 따라 링크 비활성화 여부를 판단하고, 적극적으로 재연결 (자세한 내용은 start_ws_heartbeat 주석 참조)）。
        start_ws_heartbeat()

        @server.routes.get("/comfyui_panel/runtime_stats")
        async def get_runtime_stats(request):
            """통합 런타임 통계: 하드웨어 + 큐 + 연결 상태 (한 번 요청으로 모든 결과 반환)）"""
            try:
                # ===== 디버그 핀: 계산수 + 이벤트 루프 탐지 =====
                try:
                    from . import debug_probe
                    debug_probe.incr("runtime_stats")
                    debug_probe.mark_loop_alive()
                except Exception:
                    pass
                # 1. 하드웨어 데이터: 백그라운드 스레드 캐시를 읽고 이벤트 루프를 차단하지 않습니다.
                cached = get_runtime_cache()
                hardware = cached.get("hardware", {})

                # 2. 큐 데이터: 메모리 큐 동기 읽기(ComfyUI /queue 구현과 일치)）
                queue_info = {"queue_running": [], "queue_pending": []}
                try:
                    current_queue = server.prompt_queue.get_current_queue_volatile()
                    # 민감한 데이터를 필터링합니다(server.py _remove_sensitive_from_queue와 일치).）
                    queue_info["queue_running"] = [item[:5] for item in current_queue[0]]
                    queue_info["queue_pending"] = [item[:5] for item in current_queue[1]]
                except Exception as qe:
                    queue_info["error"] = str(qe)

                response = web.json_response({
                    "success": True,
                    "timestamp": cached.get("timestamp", 0),
                    "hardware": hardware,
                    "queue": queue_info,
                    "queue_remaining": len(queue_info["queue_running"]) + len(queue_info["queue_pending"]),
                })
                response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
                response.headers['Pragma'] = 'no-cache'
                return response
            except Exception as e:
                return web.json_response({
                    "success": False,
                    "error": str(e),
                    "timestamp": 0,
                    "hardware": {},
                    "queue": {"queue_running": [], "queue_pending": []},
                    "queue_remaining": 0,
                })

        @server.routes.get("/comfyui_panel/random_seed")
        async def get_random_seed(request):
            seed = random.randint(0, 0xFFFFFFFFFFFFFFFF)
            response = web.json_response({
                "seed_str": str(seed),
                "success": True
            })
            response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
            response.headers['Pragma'] = 'no-cache'
            response.headers['Expires'] = '0'
            return response

        @server.routes.post("/comfyui_panel/random_seed")
        async def post_random_seed(request):
            try:
                data = await request.json()
                min_val = int(data.get("min", 0))
                max_val = int(data.get("max", 0xFFFFFFFFFFFFFFFF))
            except:
                min_val = 0
                max_val = 0xFFFFFFFFFFFFFFFF

            if min_val > max_val:
                min_val, max_val = max_val, min_val

            seed = random.randint(min_val, max_val)
            return web.json_response({
                "seed": seed,
                "seed_str": str(seed),
                "min": min_val,
                "max": max_val,
                "success": True
            })

        @server.routes.get("/comfyui_panel/artist_files")
        async def get_artist_files(request):
            """artists 디렉토리의 텍스트 파일 목록을 가져옵니다."""
            try:
                from .nodes.batch_random_prompt_node import BatchRandomPromptNode
                files = BatchRandomPromptNode.get_artist_file_list()
                return web.json_response({
                    "success": True,
                    "files": files
                })
            except Exception as e:
                return web.json_response({
                    "success": False,
                    "error": str(e),
                    "files": []
                })

        # TXT프롬프트 배치 생성 노드 API
        @server.routes.get("/bsk_ui/txt_files")
        async def get_txt_files(request):
            """지정 디렉토리의 txt 파일 목록 가져오기"""
            try:
                directory = request.rel_url.query.get('dir', 'input/prompts')

                # 지원 절대 경로 및 상대 경로
                if os.path.isabs(directory):
                    # 절대 경로: 직접 사용
                    full_dir = os.path.abspath(directory)
                else:
                    # 상대 경로: ComfyUI 디렉토리를 기반으로
                    base = os.path.abspath('./')
                    full_dir = os.path.abspath(os.path.join(base, directory))

                # 보안 검사: 경로가 존재하고 디렉토리인지 확인
                if not os.path.exists(full_dir):
                    return web.json_response({
                        "success": False,
                        "error": f"폴더가 없습니다: {full_dir}",
                        "files": []
                    })

                if not os.path.isdir(full_dir):
                    return web.json_response({
                        "success": False,
                        "error": f"Path is not a directory: {full_dir}",
                        "files": []
                    })

                # 보안 검사: 경로 탐색 방지
                base_full = os.path.abspath(full_dir)
                files = []
                for f in os.listdir(full_dir):
                    fp = os.path.abspath(os.path.join(full_dir, f))
                    if fp.startswith(base_full + os.sep) and f.endswith('.txt') and os.path.isfile(fp):
                        files.append(f)
                return web.json_response({
                    "success": True,
                    "files": sorted(files),
                    "path": full_dir  # 실제 경로를 반환하여 디버깅
                })
            except Exception as e:
                return web.json_response({
                    "success": False,
                    "error": str(e),
                    "files": []
                })

        @server.routes.get("/bsk_ui/read_txt")
        async def read_txt_file(request):
            """지정된 txt 파일 내용 읽기"""
            try:
                directory = request.rel_url.query.get('dir', 'input/prompts')
                file_name = request.rel_url.query.get('file', '')

                if not file_name:
                    return web.json_response({
                        "success": False,
                        "error": "파일 이름 지정되지 않음",
                        "content": ""
                    })

                # 지원 절대 경로 및 상대 경로
                if os.path.isabs(directory):
                    # 절대 경로: 직접 사용
                    full_dir = os.path.abspath(directory)
                else:
                    # 상대 경로: ComfyUI 디렉토리를 기반으로
                    base = os.path.abspath('./')
                    full_dir = os.path.abspath(os.path.join(base, directory))

                # 보안 검사: 경로 탐색 방지
                base_full = os.path.abspath(full_dir)
                full_path = os.path.abspath(os.path.join(full_dir, file_name))
                if not full_path.startswith(base_full + os.sep) and full_path != base_full:
                    return web.json_response({
                        "success": False,
                        "error": f"안전 거부: 경로 침투 감지",
                        "content": ""
                    })

                if not os.path.isfile(full_path):
                    return web.json_response({
                        "success": False,
                        "error": f"파일이 존재하지 않습니다: {full_path}",
                        "content": ""
                    })

                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                return web.json_response({
                    "success": True,
                    "content": content
                })
            except Exception as e:
                return web.json_response({
                    "success": False,
                    "error": str(e),
                    "content": ""
                })

        # ============ 중단하려면 훈련을 계속 진행하시겠습니까? 저장된 체크포인트는 손실되지 않습니다.） ============
        CAMERA_CFG_DIR = os.path.join(folder_paths.base_path, "user", "default", "camera_configs")

        @server.routes.get("/comfyui_panel/camera_configs")
        async def list_camera_configs(request):
            """user/default/camera_configs 아래의 목록 표시 *.json 설정 파일。"""
            try:
                os.makedirs(CAMERA_CFG_DIR, exist_ok=True)
                base_full = os.path.abspath(CAMERA_CFG_DIR)
                files = [f for f in os.listdir(base_full)
                         if f.endswith(".json") and os.path.isfile(os.path.join(base_full, f))]
                return web.json_response({"success": True, "files": sorted(files)})
            except Exception as e:
                return web.json_response({"success": False, "error": str(e), "files": []})

        @server.routes.post("/comfyui_panel/camera_configs/save")
        async def save_camera_config(request):
            """카메라 설정 저장：body {name, content}。"""
            try:
                data = await request.json()
                name = (data.get("name") or "").strip()
                content = data.get("content", "")
                if not name:
                    return web.json_response({"success": False, "error": "파일 이름이 없습니다"})
                if not name.endswith(".json"):
                    name += ".json"
                os.makedirs(CAMERA_CFG_DIR, exist_ok=True)
                base_full = os.path.abspath(CAMERA_CFG_DIR)
                full = os.path.abspath(os.path.join(base_full, name))
                if full != base_full and not full.startswith(base_full + os.sep):
                    return web.json_response({"success": False, "error": "유효하지 않은 파일 이름 (경로 침투)）"})
                with open(full, "w", encoding="utf-8") as f:
                    f.write(content)
                return web.json_response({"success": True, "name": name})
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        @server.routes.post("/comfyui_panel/camera_configs/load")
        async def load_camera_config(request):
            """카메라 설정 읽기：body {name}，반환 {content}。"""
            try:
                data = await request.json()
                name = (data.get("name") or "").strip()
                if not name:
                    return web.json_response({"success": False, "error": "파일 이름이 없습니다"})
                if not name.endswith(".json"):
                    name += ".json"
                base_full = os.path.abspath(CAMERA_CFG_DIR)
                full = os.path.abspath(os.path.join(base_full, name))
                if full != base_full and not full.startswith(base_full + os.sep):
                    return web.json_response({"success": False, "error": "유효하지 않은 파일 이름 (경로 침투)）"})
                if not os.path.isfile(full):
                    return web.json_response({"success": False, "error": "파일이 존재하지 않습니다"})
                with open(full, "r", encoding="utf-8") as f:
                    content = f.read()
                return web.json_response({"success": True, "content": content})
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        # ============ BSK Video Studio 설정 저장 (ComfyUI 사용자 디렉토리에 저장)） ============
        VS_CFG_DIR = os.path.join(folder_paths.base_path, "user", "default", "bsk_video_studio")

        @server.routes.get("/comfyui_panel/vs_configs")
        async def list_vs_configs(request):
            """user/default/bsk_video_studio 아래의 목록을 표시합니다. *.json 설정 파일。

            오래된 프론트엔드 호환: files는 문자열 배열로 유지; 동시에 previews 매핑 반환
            {파일명: 설정 중 첫 번째 이미지의 serverPath 또는 null}，배치 목록에 썸네일 표시용。
            """
            try:
                os.makedirs(VS_CFG_DIR, exist_ok=True)
                base_full = os.path.abspath(VS_CFG_DIR)
                files = [f for f in os.listdir(base_full)
                         if f.endswith(".json") and os.path.isfile(os.path.join(base_full, f))]
                files.sort()
                previews = {}
                for fn in files:
                    previews[fn] = None
                    try:
                        with open(os.path.join(base_full, fn), "r", encoding="utf-8") as f:
                            obj = json.load(f)
                        assets = obj.get("assets") or []
                        for a in assets:
                            if isinstance(a, dict) and a.get("kind") == "image" and a.get("serverPath"):
                                previews[fn] = a["serverPath"]
                                break
                        if not previews[fn]:
                            for s in obj.get("segments") or []:
                                if isinstance(s, dict) and s.get("type") == "image" and s.get("imageFile"):
                                    previews[fn] = s["imageFile"]
                                    break
                    except Exception:
                        pass
                return web.json_response({"success": True, "files": files, "previews": previews})
            except Exception as e:
                return web.json_response({"success": False, "error": str(e), "files": []})

        @server.routes.delete("/comfyui_panel/vs_configs")
        async def delete_vs_config(request):
            """user/default/bsk_video_studio 아래의 특정 *.json 설정 파일。"""
            try:
                name = request.query.get("name", "")
                if not name:
                    return web.json_response({"success": False, "error": "누락 name"})
                # 파일 이름만 가져오고 경로 침투 금지
                safe = os.path.basename(name)
                if not safe.endswith(".json"):
                    safe += ".json"
                if safe == "autosave.json" or not safe.endswith(".json"):
                    return web.json_response({"success": False, "error": "삭제 금지 자동 저장 파일"})
                base_full = os.path.abspath(VS_CFG_DIR)
                target = os.path.abspath(os.path.join(base_full, safe))
                if os.path.dirname(target) != base_full or not os.path.isfile(target):
                    return web.json_response({"success": False, "error": "설정 파일이 존재하지 않음"})
                os.remove(target)
                return web.json_response({"success": True})
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        @server.routes.post("/comfyui_panel/vs_configs/save")
        async def save_vs_config(request):
            """Video Studio 구성 저장。

            패널 카드 호환：
            - 구형 (직연) ComfyUI）：body = {name, content}，한 번에 쓰기。
            - 블록 모드(게이트웨이 프록시, body 제한 환경): query 포함 upload=1&name=&index=&total=，
              각 body는 텍스트 조각으로 구성되며, 백엔드에서 인덱스별로 임시 파일에 추가되고 마지막 조각을 받으면 공식 파일로 이름을 변경합니다。
              구 방식에 완전히 영향을 미치지 않음。
            """
            try:
                qs = request.query
                is_chunked = qs.get("upload") == "1"
                if is_chunked:
                    name = (qs.get("name") or "").strip()
                    if not name:
                        return web.json_response({"success": False, "error": "파일 이름이 없습니다"})
                    if not name.endswith(".json"):
                        name += ".json"
                    os.makedirs(VS_CFG_DIR, exist_ok=True)
                    base_full = os.path.abspath(VS_CFG_DIR)
                    full = os.path.abspath(os.path.join(base_full, name))
                    if full != base_full and not full.startswith(base_full + os.sep):
                        return web.json_response({"success": False, "error": "유효하지 않은 파일 이름 (경로 침투)）"})
                    try:
                        index = int(qs.get("index", "0"))
                        total = int(qs.get("total", "1"))
                    except (TypeError, ValueError):
                        return web.json_response({"success": False, "error": "분리 기호 (켜짐"})
                    raw = await request.read()
                    # 임시 분할 파일: 실제 파일 이름 + ".chunk_<index>"
                    part = full + ".chunk_" + str(index)
                    with open(part, "wb") as f:
                        f.write(raw)
                    if index >= total - 1:
                        # 마지막 블록: 모든 블록을 순서대로 합쳐 디스크에 저장하고 임시 블록 삭제。
                        try:
                            with open(full, "w", encoding="utf-8") as out:
                                for i in range(total):
                                    pf = full + ".chunk_" + str(i)
                                    with open(pf, "rb") as pfh:
                                        out.write(pfh.read().decode("utf-8"))
                                    os.remove(pf)
                        except Exception as e:
                            return web.json_response({"success": False, "error": "병합 분할 실패: " + str(e)})
                    return web.json_response({"success": True, "name": name, "chunk": index, "total": total})
                # —— 구 방식: 한 번에 쓰기 ——
                data = await request.json()
                name = (data.get("name") or "").strip()
                content = data.get("content", "")
                if not name:
                    return web.json_response({"success": False, "error": "파일 이름이 없습니다"})
                if not name.endswith(".json"):
                    name += ".json"
                os.makedirs(VS_CFG_DIR, exist_ok=True)
                base_full = os.path.abspath(VS_CFG_DIR)
                full = os.path.abspath(os.path.join(base_full, name))
                if full != base_full and not full.startswith(base_full + os.sep):
                    return web.json_response({"success": False, "error": "유효하지 않은 파일 이름 (경로 침투)）"})
                with open(full, "w", encoding="utf-8") as f:
                    f.write(content)
                return web.json_response({"success": True, "name": name})
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        @server.routes.post("/comfyui_panel/vs_configs/load")
        async def load_vs_config(request):
            """Video Studio 설정 읽기：body {name}，반환 {content}。"""
            try:
                data = await request.json()
                name = (data.get("name") or "").strip()
                if not name:
                    return web.json_response({"success": False, "error": "파일 이름이 없습니다"})
                if not name.endswith(".json"):
                    name += ".json"
                base_full = os.path.abspath(VS_CFG_DIR)
                full = os.path.abspath(os.path.join(base_full, name))
                if full != base_full and not full.startswith(base_full + os.sep):
                    return web.json_response({"success": False, "error": "유효하지 않은 파일 이름 (경로 침투)）"})
                if not os.path.isfile(full):
                    return web.json_response({"success": False, "error": "파일이 존재하지 않습니다"})
                with open(full, "r", encoding="utf-8") as f:
                    content = f.read()
                return web.json_response({"success": True, "content": content})
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        # ============ BSK Video Studio：중간 생성물（user/default/bsk_video_studio/intermediates/） ============
        # 중간 결과물 = 두 개의 동일 이름 파일：
        #   <name>.json    —— 파라미터/설정 (프론트엔드 제출 시 작성, 목록 항목 클릭 시 일괄 복원)）
        #   <name>.bskdata —— CLIP 인코딩된 positive/latent (bsk_MiniMaxDirector 노드에 쓰기）
        # 의미와 vs_configs 일치: GET 목록 / POST save_config / POST load_config / DELETE 삭제 파일。
        VS_INTERMEDIATE_DIR = os.path.join(VS_CFG_DIR, "intermediates")

        @server.routes.get("/comfyui_panel/vs_intermediates")
        async def list_vs_intermediates(request):
            """중간 결과물 목록을 나열합니다 (주 이름으로 중복 제거, 표시) hasConfig/hasData）。"""
            try:
                os.makedirs(VS_INTERMEDIATE_DIR, exist_ok=True)
                items = {}
                for fn in os.listdir(VS_INTERMEDIATE_DIR):
                    if fn.startswith("."):
                        continue
                    full = os.path.join(VS_INTERMEDIATE_DIR, fn)
                    if not os.path.isfile(full):
                        continue
                    stem, ext = os.path.splitext(fn)
                    if ext not in (".json", ".bskdata"):
                        continue
                    rec = items.setdefault(stem, {"name": stem, "hasConfig": False, "hasData": False, "mtime": 0})
                    if ext == ".json":
                        rec["hasConfig"] = True
                    elif ext == ".bskdata":
                        rec["hasData"] = True
                    rec["mtime"] = max(rec["mtime"], int(os.path.getmtime(full)))
                result = sorted(items.values(), key=lambda x: x["mtime"], reverse=True)
                return web.json_response({"success": True, "files": result})
            except Exception as e:
                return web.json_response({"success": False, "error": str(e), "files": []})

        @server.routes.post("/comfyui_panel/vs_intermediates/save_config")
        async def save_vs_intermediate_config(request):
            """중간 결과 파일 템플릿 작성 <name>.json（프론트엔드 제출 시 전체 설정 저장）。"""
            try:
                data = await request.json()
                name = (data.get("name") or "").strip()
                content = data.get("content", "")
                if not name:
                    return web.json_response({"success": False, "error": "파일 이름이 없습니다"})
                safe = os.path.basename(name)
                if safe.endswith(".bskdata"):
                    safe = safe[:-len(".bskdata")]
                if not safe.endswith(".json"):
                    safe += ".json"
                os.makedirs(VS_INTERMEDIATE_DIR, exist_ok=True)
                base_full = os.path.abspath(VS_INTERMEDIATE_DIR)
                full = os.path.abspath(os.path.join(base_full, safe))
                if full != base_full and not full.startswith(base_full + os.sep):
                    return web.json_response({"success": False, "error": "유효하지 않은 파일 이름 (경로 침투)）"})
                with open(full, "w", encoding="utf-8") as f:
                    f.write(content if isinstance(content, str) else json.dumps(content))
                return web.json_response({"success": True, "name": safe})
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        @server.routes.post("/comfyui_panel/vs_intermediates/load_config")
        async def load_vs_intermediate_config(request):
            """읽기 중간 결과 구성 파일 <name>.json，반환 {content}。"""
            try:
                data = await request.json()
                name = (data.get("name") or "").strip()
                if not name:
                    return web.json_response({"success": False, "error": "파일 이름이 없습니다"})
                if name.endswith(".bskdata"):
                    name = name[:-len(".bskdata")]
                if not name.endswith(".json"):
                    name += ".json"
                safe = os.path.basename(name)
                base_full = os.path.abspath(VS_INTERMEDIATE_DIR)
                full = os.path.abspath(os.path.join(base_full, safe))
                if full != base_full and not full.startswith(base_full + os.sep):
                    return web.json_response({"success": False, "error": "유효하지 않은 파일 이름 (경로 침투)）"})
                if not os.path.isfile(full):
                    return web.json_response({"success": False, "error": "파일이 존재하지 않습니다"})
                with open(full, "r", encoding="utf-8") as f:
                    content = f.read()
                return web.json_response({"success": True, "content": content})
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        @server.routes.delete("/comfyui_panel/vs_intermediates")
        async def delete_vs_intermediate(request):
            """중간 결과물 삭제 (동명의 .json 및 .bskdata 함께 삭제）。"""
            try:
                name = request.query.get("name", "").strip()
                if not name:
                    return web.json_response({"success": False, "error": "누락 name"})
                safe = os.path.basename(name)
                if safe.endswith(".json") or safe.endswith(".bskdata"):
                    safe = os.path.splitext(safe)[0]
                base_full = os.path.abspath(VS_INTERMEDIATE_DIR)
                removed = 0
                for ext in (".json", ".bskdata"):
                    target = os.path.abspath(os.path.join(base_full, safe + ext))
                    if (target == base_full or target.startswith(base_full + os.sep)) and os.path.isfile(target):
                        os.remove(target)
                        removed += 1
                if removed == 0:
                    return web.json_response({"success": False, "error": "중간 결과물 없음"})
                return web.json_response({"success": True, "removed": removed})
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        @server.routes.post("/comfyui_panel/vs_seq_to_video")
        async def vs_seq_to_video(request):
            """BSK 이미지 시퀀스 모드 생성 PNG 프레임을 지정 프레임률로 합쳐 MP4로 만듦 (백엔드 합성, 실행 큐/메모리 점유 안 함)）。

            선택 HTTP 엔드포인트 대신 워크플로우 노드: 프레임 PNG는 노드 실행 시점 동적 낙반 파일 (타임스탬프 프리픽스)），
            워크플로우가 제출될 때 파일 목록을 미리 알 수 없습니다. 또한 ffmpeg를 사용하여 엔드포인트를 제공합니다.，
            완료 후 프론트엔드에서 /view 를 사용하여 직접 프리뷰, 다시 워크플로우를 통과할 필요 없음。

            요청 본문：{firstFile, lastFile, fps}
            - firstFile / lastFile：시퀀스 첫 번째 프레임과 마지막 프레임 파일명 (output 디렉토리 루트 아래)，
              형상과 같음 bsk_frames_20260809_120000_0000.png）。
            - fps：합성 프레임률 (프론트엔드에서 프로젝트 파라미터 프레임률 가져옴), 기본 24。
            백엔드에서 두 파일 이름에서 공통 접두사 및 숫자 범위를 추출하여 output 디렉토리 내의 해당 접두사가 있는 모든 프레임을 스캔，
            순번 오름차순 닫힌 구간 프레임 가져오기, ffmpeg(imageio_ffmpeg 또는 시스템 ffmpeg)로 합성 MP4。
            반환 {success, filename, subfolder, type, frames, fps, width, height, duration}。
            """
            try:
                data = await request.json()
            except Exception:
                data = {}
            try:
                first = (data.get("firstFile") or "").strip()
                last = (data.get("lastFile") or "").strip()
                try:
                    fps = int(float(data.get("fps") or 24) or 24)
                except (TypeError, ValueError):
                    fps = 24
                fps = max(1, min(120, fps))
                if not first or not last:
                    return web.json_response({"success": False, "error": "시작 프레임 또는 종료 프레임 파일 이름 누락"})
                first = os.path.basename(first)
                last = os.path.basename(last)
                out_dir = folder_paths.get_output_directory()
                seq_re = re.compile(r"^(.*)_(\d+)\.(?:png|jpg|jpeg|webp)$", re.I)
                m1 = seq_re.match(first)
                m2 = seq_re.match(last)
                if not m1 or not m2:
                    return web.json_response({"success": False,
                                              "error": "분석 불가, 원본 유지 *_<숫자>.png 명명）: %s / %s" % (first, last)})
                prefix1, idx1 = m1.group(1), int(m1.group(2))
                prefix2, idx2 = m2.group(1), int(m2.group(2))
                if prefix1 != prefix2:
                    return web.json_response({"success": False,
                                              "error": "시작 프레임과 종료 프레임은 동일한 시퀀스에 속하지 않습니다 (접두사 불일치)）: %s / %s" % (first, last)})
                prefix = prefix1
                start_idx, end_idx = min(idx1, idx2), max(idx1, idx2)
                # output 디렉토리 내 동일 프리픽스 프레임 스캔, 순서대로 정렬, 닫힌 구간 프레임 추출
                pat = re.compile(r"^" + re.escape(prefix) + r"_(\d+)\.(?:png|jpg|jpeg|webp)$", re.I)
                found = []
                for fn in os.listdir(out_dir):
                    m = pat.match(fn)
                    if m:
                        found.append((int(m.group(1)), fn))
                found.sort(key=lambda x: x[0])
                frames = [fn for i, fn in found if start_idx <= i <= end_idx]
                if len(frames) < 2:
                    return web.json_response({"success": False,
                                              "error": "시퀀스 범위 내 사용 가능한 프레임 부족 (찾음 %d 장)）" % len(frames)})
                # ffmpeg 위치 지정: ComfyUI 내장 imageio_ffmpeg 우선, 시스템으로 롤백 ffmpeg
                import subprocess
                import shutil as _shutil
                ffmpeg = None
                try:
                    import imageio_ffmpeg
                    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
                except Exception:
                    ffmpeg = None
                if not ffmpeg:
                    ffmpeg = _shutil.which("ffmpeg")
                if not ffmpeg:
                    return web.json_response({"success": False,
                                              "error": "ffmpeg를 찾을 수 없습니다 (imageio_ffmpeg / 시스템 ffmpeg), 비디오 합성이 불가능합니다"})
                out_mp4 = os.path.join(out_dir, prefix + ".mp4")
                frame_paths = [os.path.join(out_dir, f) for f in frames]
                # 프레임 수 <= 200：concat filter 프레임별 정확한 연결 (glob 정렬/순서 숫자 의존 X)）
                if len(frame_paths) <= 200:
                    cmd = [ffmpeg, "-y"]
                    for p in frame_paths:
                        cmd += ["-i", p]
                    labels = "".join("[%d:v]" % i for i in range(len(frame_paths)))
                    cmd += ["-filter_complex",
                            "%sconcat=n=%d:v=1:a=0[v]" % (labels, len(frame_paths)),
                            "-map", "[v]", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                            "-r", str(fps), "-movflags", "+faststart", out_mp4]
                else:
                    # 많은 프레임: ffmpeg glob 입력 (프레임 이름은 0으로 채워진 숫자, 알파벳 순서) = 숫자 순서）
                    import glob as _glob
                    cmd = [ffmpeg, "-y", "-framerate", str(fps),
                           "-pattern_type", "glob",
                           "-i", os.path.join(out_dir, prefix + "_*.png"),
                           "-c:v", "libx264", "-pix_fmt", "yuv420p",
                           "-r", str(fps), "-movflags", "+faststart", out_mp4]
                proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300)
                if proc.returncode != 0 or not os.path.isfile(out_mp4):
                    tail = proc.stderr.decode("utf-8", "replace")[-500:]
                    return web.json_response({"success": False,
                                              "error": "ffmpeg 합성 실패 (프레임 크기 불일치)）: %s" % tail})
                # 첫 번째 프레임 크기 (PIL 읽기)와 합성 시간
                width = height = 0
                try:
                    from PIL import Image as _PILImage
                    with _PILImage.open(frame_paths[0]) as _im:
                        width, height = _im.size
                except Exception:
                    pass
                return web.json_response({
                    "success": True,
                    "filename": os.path.basename(out_mp4),
                    "subfolder": "",
                    "type": "output",
                    "frames": len(frames),
                    "fps": fps,
                    "width": width, "height": height,
                    "duration": round(len(frames) / float(fps), 3),
                })
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        # ============ BSK Video Studio：리소스 라이브러리 폴더 / 링크 보조 ============
        # 목차 구조：
        #   input/whatdreamscost/                —— 전체 자료 저장소 (업로드된 모든 파일의 유일한 실체）
        #   input/whatdreamscost/projects/<명>/  —— 프로젝트 폴더(전체 자재 라이브러리 링크만 저장)）
        # 링크는 우선 하드 링크(Windows NTFS / Linux 모두 지원, 관리자 권한이 필요 없음)를 사용하며, 실패 시 심볼릭 링크를 시도합니다.。
        _os = os  # 리소스 라이브러리 지원 함수 사용 (오래된 로컬 import os as _os 방식 호환)）
        def _vs_lib_root():
            return _os.path.join(folder_paths.get_input_directory(), "whatdreamscost")

        def _vs_projects_dir():
            return _os.path.join(_vs_lib_root(), "projects")

        def _vs_safe_name(name):
            s = _os.path.basename((name or "").strip())
            # Windows/Linux를 지원합니다: 경로 구분자를 제거하고 Windows 문자를 유지하여 폴더 생성 또는 링크 실패를 방지합니다.
            s = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", s).strip(" .")
            return s

        def _vs_ensure_project_dir(folder):
            folder = _vs_safe_name(folder)
            if not folder or folder in (".", ".."):
                raise ValueError("유효하지 않은 프로젝트 폴더 이름")
            d = _os.path.join(_vs_projects_dir(), folder)
            _os.makedirs(d, exist_ok=True)
            return d

        def _vs_make_link(src, dst):
            """파일 링크를 생성합니다: 우선 하드 링크(동일 드라이브, 권한 불필요)를 사용하고, 실패하면 심볼릭 링크로 되돌립니다.。"""
            if _os.path.lexists(dst):
                return False  # 존재함
            try:
                _os.link(src, dst)
                return True
            except OSError:
                try:
                    _os.symlink(src, dst)
                    return True
                except OSError as e:
                    raise OSError(f"파일 링크 생성 실패 (하드 링크/심볼릭 링크 모두 실패)）: {e}")

        def _vs_sha256(path):
            h = hashlib.sha256()
            with open(path, "rb") as f:
                while True:
                    chunk = f.read(1024 * 1024)
                    if not chunk:
                        break
                    h.update(chunk)
            return h.hexdigest()

        def _vs_find_library_duplicate(filename, size, sha256=None):
            """전체 자재 라이브러리 루트 디렉토리에서 파일 이름으로+크기 (선택 사항 sha256) 완전히 동일한 파일 찾기。
            일치하는 파일 이름 또는 None을 반환합니다. 동일 이름 파일만 비교하여 전체 데이터베이스 해시 계산으로 인한 업로드 속도 저하를 방지합니다.。"""
            base = _vs_lib_root()
            safe = _vs_safe_name(filename)
            if not safe:
                return None
            candidate = _os.path.join(base, safe)
            if _os.path.isfile(candidate) and _os.path.getsize(candidate) == int(size or 0):
                if not sha256:
                    return safe
                try:
                    if _vs_sha256(candidate) == sha256:
                        return safe
                except OSError:
                    pass
            return None

        def _vs_project_server_path(folder, filename):
            return f"whatdreamscost/projects/{_vs_safe_name(folder)}/{_vs_safe_name(filename)}"

        def _vs_resolve_library_rel(server_path):
            """저장 whatdreamscost/... 경로안전파싱위하여루트 디렉토리의상대경로（횡단 금지）。"""
            rel = (server_path or "").strip().replace("\\", "/")
            prefix = "whatdreamscost/"
            if rel.startswith(prefix):
                rel = rel[len(prefix):]
            rel = rel.lstrip("/")
            parts = [p for p in rel.split("/") if p not in ("", ".", "..")]
            return "/".join(parts)

        @server.routes.post("/bsk/vs/library/check")
        async def vs_library_check(request):
            """업로드 전 중복 검사：body {folder?, filename, size, sha256?}
            반환 {linkExists, duplicate, libraryPath, projectPath}。"""
            try:
                data = await request.json()
                filename = _vs_safe_name(data.get("filename"))
                size = int(data.get("size") or 0)
                sha256 = (data.get("sha256") or "").strip() or None
                folder = _vs_safe_name(data.get("folder"))
                if not filename:
                    return web.json_response({"success": False, "error": "파일 이름이 없습니다"})
                base = _vs_lib_root()
                _os.makedirs(base, exist_ok=True)
                # 프로젝트 폴더에 동일 이름 링크가 이미 있는지 확인
                link_exists = False
                if folder:
                    proj_dir = _os.path.join(_vs_projects_dir(), folder)
                    if _os.path.isdir(proj_dir) and _os.path.lexists(_os.path.join(proj_dir, filename)):
                        link_exists = True
                dup = _vs_find_library_duplicate(filename, size, sha256)
                resp = {
                    "success": True,
                    "linkExists": link_exists,
                    "duplicate": bool(dup),
                    "libraryPath": f"whatdreamscost/{dup}" if dup else None,
                    "projectPath": _vs_project_server_path(folder, filename) if folder else None,
                }
                return web.json_response(resp)
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        @server.routes.post("/bsk/vs/library/link")
        async def vs_library_link(request):
            """드래그 앤 드롭 슬라이더 패널을 사용할 때 전역적으로 텍스트 선택을 비활성화：body {folder, libraryPath}。"""
            try:
                data = await request.json()
                folder = _vs_safe_name(data.get("folder"))
                lib_rel = _vs_resolve_library_rel(data.get("libraryPath"))
                if not folder or not lib_rel or "/" in lib_rel:
                    return web.json_response({"success": False, "error": "잘못된 파라미터"})
                src = _os.path.join(_vs_lib_root(), lib_rel)
                if not _os.path.isfile(src):
                    return web.json_response({"success": False, "error": "총 자료보관 파일이 존재하지 않습니다"})
                proj_dir = _vs_ensure_project_dir(folder)
                dst = _os.path.join(proj_dir, _vs_safe_name(lib_rel))
                created = _vs_make_link(src, dst)
                return web.json_response({
                    "success": True,
                    "created": created,
                    "already": not created,
                    "projectPath": _vs_project_server_path(folder, lib_rel),
                })
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        @server.routes.get("/bsk/vs/library/folders")
        async def vs_library_folders(request):
            """모든 프로젝트 폴더 목록 표시。"""
            try:
                proj = _vs_projects_dir()
                _os.makedirs(proj, exist_ok=True)
                folders = [d for d in _os.listdir(proj)
                           if _os.path.isdir(_os.path.join(proj, d)) and not d.startswith(".")]
                return web.json_response({"success": True, "folders": sorted(folders)})
            except Exception as e:
                return web.json_response({"success": False, "error": str(e), "folders": []})

        @server.routes.post("/bsk/vs/library/folders/create")
        async def vs_library_folder_create(request):
            """프로젝트 폴더 생성 (멱등)）。"""
            try:
                data = await request.json()
                folder = _vs_safe_name(data.get("name"))
                if not folder or folder in (".", ".."):
                    return web.json_response({"success": False, "error": "유효하지 않은 폴더 이름"})
                _vs_ensure_project_dir(folder)
                return web.json_response({"success": True, "folder": folder})
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        @server.routes.post("/bsk/vs/library/folders/delete")
        async def vs_library_folder_delete(request):
            """프로젝트 폴더 삭제: 삭제 projects/<명>/ 디렉토리와 그 안의 모든 링크/파일。
            총 자료 라이브러리 엔티티 파일은 영향을 받지 않습니다(프로젝트 폴더 내 링크만 저장; 엔티티 파일이 있는 경우도 사용자
            지침 디렉터리 함께 삭제）。body {name}。"""
            try:
                import shutil as _shutil
                data = await request.json()
                folder = _vs_safe_name(data.get("name"))
                if not folder or folder in (".", ".."):
                    return web.json_response({"success": False, "error": "유효하지 않은 폴더 이름"})
                d = _os.path.join(_vs_projects_dir(), folder)
                if not _os.path.isdir(d):
                    return web.json_response({"success": False, "error": "폴더가 존재하지 않음"})
                _shutil.rmtree(d, ignore_errors=True)
                return web.json_response({"success": True, "folder": folder})
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        @server.routes.post("/bsk/vs/library/folders/move_assets")
        async def vs_library_folder_move_assets(request):
            """from 프로젝트 폴더의 모든 자료를 to 프로젝트 폴더로 이동합니다(이름 저장 시)
            "기본 폴더"에서 임시 자료를 프로젝트 폴더로 마이그레이션합니다. shutil.move: 동일 드라이브 rename
            （단계 4: 오디오 트랙(주 트랙의 스케일을 재사용하고 독립적이지 않습니다).。body {from, to}。"""
            try:
                import shutil as _shutil
                data = await request.json()
                src_name = _vs_safe_name(data.get("from"))
                dst_name = _vs_safe_name(data.get("to"))
                if not dst_name or dst_name in (".", ".."):
                    return web.json_response({"success": False, "error": "유효하지 않은 대상 폴더"})
                if not src_name or src_name in (".", "..") or src_name == dst_name:
                    return web.json_response({"success": True, "moved": 0})
                src_dir = _os.path.join(_vs_projects_dir(), src_name)
                if not _os.path.isdir(src_dir):
                    return web.json_response({"success": True, "moved": 0})
                dst_dir = _vs_ensure_project_dir(dst_name)
                moved = 0
                for fn in _os.listdir(src_dir):
                    sp = _os.path.join(src_dir, fn)
                    if not _os.path.isfile(sp) and not _os.path.islink(sp):
                        continue
                    dp = _os.path.join(dst_dir, fn)
                    if _os.path.lexists(dp):
                        # 목표 동일 이름: 자동 접미사 추가, 덮어쓰기 방지
                        stem, ext = _os.path.splitext(fn)
                        n = 1
                        while _os.path.lexists(dp):
                            dp = _os.path.join(dst_dir, f"{stem}_{n}{ext}")
                            n += 1
                    _shutil.move(sp, dp)
                    moved += 1
                # 이동 후 제거 빈 폴더 디렉토리
                try:
                    _os.rmdir(src_dir)
                except OSError:
                    pass
                return web.json_response({"success": True, "moved": moved})
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        @server.routes.get("/bsk/vs/library/files")
        async def vs_library_files(request):
            """전체 자재 라이브러리 (폴더 미포함) 또는 프로젝트 폴더 (폴더 포함) 파일 목록 표시。"""
            try:
                folder = _vs_safe_name(request.query.get("folder") or "")
                if folder:
                    base = _vs_ensure_project_dir(folder)
                    prefix = f"whatdreamscost/projects/{folder}/"
                else:
                    base = _vs_lib_root()
                    prefix = "whatdreamscost/"
                _os.makedirs(base, exist_ok=True)
                result = []
                for fn in _os.listdir(base):
                    if fn.endswith(".upload"):
                        continue
                    fp = _os.path.join(base, fn)
                    if not _os.path.isfile(fp) and not _os.path.islink(fp):
                        continue
                    if not folder and fn == "projects":
                        continue
                    try:
                        st = _os.stat(fp)
                    except OSError:
                        continue
                    result.append({
                        "name": fn,
                        "serverPath": prefix + fn,
                        "libraryPath": f"whatdreamscost/{fn}",
                        "size": st.st_size,
                        "mtime": int(st.st_mtime),
                        "isLink": bool(folder) or _os.path.islink(fp),
                    })
                result.sort(key=lambda x: x["mtime"], reverse=True)
                return web.json_response({"success": True, "files": result})
            except Exception as e:
                return web.json_response({"success": False, "error": str(e), "files": []})

        @server.routes.post("/bsk/vs/library/delete")
        async def vs_library_delete(request):
            """리소스 삭제：
            - mode=link：프로젝트 폴더 내 링크만 삭제, 전체 자료 라이브러리는 유지；
            - mode=library：삭제 총 자료 라이브러리 파일 및 모든 프로젝트의 동일 이름 링크。
            body {path, mode}。"""
            try:
                data = await request.json()
                path = (data.get("path") or "").strip().replace("\\", "/")
                mode = (data.get("mode") or "link").strip()
                rel = _vs_resolve_library_rel(path)
                if not rel:
                    return web.json_response({"success": False, "error": "유효하지 않은 경로"})
                root = _vs_lib_root()
                full = _os.path.abspath(_os.path.join(root, rel.replace("/", os.sep)))
                root_abs = _os.path.abspath(root)
                if full != root_abs and not full.startswith(root_abs + os.sep):
                    return web.json_response({"success": False, "error": "유효하지 않은 경로 (경로 침투）"})
                if mode == "library":
                    # 모든 프로젝트의 동일 링크를 먼저 삭제하고, 실체 파일을 삭제하십시오.
                    proj = _vs_projects_dir()
                    if _os.path.isdir(proj):
                        for d in _os.listdir(proj):
                            dp = _os.path.join(proj, d)
                            if not _os.path.isdir(dp):
                                continue
                            link = _os.path.join(dp, os.path.basename(rel))
                            if _os.path.lexists(link):
                                try:
                                    _os.remove(link)
                                except OSError:
                                    pass
                    if not _os.path.isfile(full) and not _os.path.islink(full):
                        return web.json_response({"success": False, "error": "파일이 존재하지 않습니다"})
                    _os.remove(full)
                    return web.json_response({"success": True, "mode": "library"})
                # 기본적으로 프로젝트 링크만 삭제
                if not rel.startswith("projects/"):
                    return web.json_response({"success": False, "error": "프로젝트 링크가 아님, link 모드로 삭제 불가"})
                if not _os.path.lexists(full):
                    return web.json_response({"success": False, "error": "링크가 존재하지 않습니다"})
                _os.remove(full)
                return web.json_response({"success": True, "mode": "link"})
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        # ============ BSK Video Studio：리소스 분할 업로드 (저장) input/whatdreamscost/） ============
        # 대용량 파일 분할을 지원합니다: 프런트엔드는 4MB로 분할하고 각 블록에 대해 POST합니다. 첫 번째 블록을 덮어 쓰고 후속 블록을 추가적으로 씁니다。
        # 마지막 조각 반환 {name:"whatdreamscost/<filename>", size}，name 필드 의미와
        # delete_asset / assetServer / 131.timeline_data 완전히 일치。
        # 파일명 충돌 시 자동으로 _1/_2...를 추가하여 기존 자료를 덮어쓰지 않도록 합니다.。
        # 폴더 인수를 가져오면, 업로드 후 총 자료 라이브러리 파일 링크를 프로젝트 폴더에 자동으로 추가합니다.。
        @server.routes.post("/bsk/vs/upload_chunk")
        async def vs_upload_chunk(request):
            """블록 단위 업로드하여 총 자료 라이브러리에 저장, 프로젝트 폴더 자동 링크 옵션。

            추가 필드：
            - folder：프로젝트 폴더 이름(선택 사항). 마지막 파일 업로드 완료 시 링크 자동 생성。
            - sha256 / total_size：중복 확인 (선택 사항; 동일 파일 이름)+크기가 중복으로 판단될 경우 생략 가능 sha256）。
            - final_name：첫 번째 반환된 최종 파일 이름(슬라이스 전송, 충돌 방지를 위해 이름 변경 후 오류 방지)）。
            총 자료 라이브러리에 완전히 동일한 파일이 있으면 새 파일을 쓰지 않고 기존 파일을 링크하고 반환합니다. duplicate=true。
            """
            try:
                import os as _os
                reader = await request.multipart()
                filename = None
                chunk_index = 0
                total_chunks = 1
                file_bytes = None
                folder = None
                sha256 = None
                total_size = None
                final_name = None
                direct = False
                # multipart 필드 파싱
                async for field in reader:
                    if field.name == 'filename':
                        filename = (await field.text()).strip()
                    elif field.name == 'chunk_index':
                        chunk_index = int((await field.text()) or '0')
                    elif field.name == 'total_chunks':
                        total_chunks = int((await field.text()) or '1')
                    elif field.name == 'folder':
                        folder = (await field.text()).strip()
                    elif field.name == 'sha256':
                        sha256 = (await field.text()).strip() or None
                    elif field.name == 'total_size':
                        total_size = int((await field.text()) or '0')
                    elif field.name == 'final_name':
                        final_name = (await field.text()).strip() or None
                    elif field.name == 'direct':
                        # direct=1：파일을 프로젝트 폴더에 직접 쓰기 (총 자료 라이브러리에 저장하지 않으며 링크 생성하지 않음)）
                        direct = (await field.text()).strip() in ('1', 'true')
                    elif field.name == 'file':
                        file_bytes = await field.read()
                if not filename:
                    return web.json_response({"success": False, "error": "누락 filename"})
                if file_bytes is None:
                    return web.json_response({"success": False, "error": "파일 데이터 블록 누락"})

                # direct 모드: 디스크에 저장 = 프로젝트 폴더; 그렇지 않으면 = 총 자료 라이브러리 루트
                if direct and folder:
                    base_dir = _vs_ensure_project_dir(folder)
                else:
                    base_dir = _vs_lib_root()
                _os.makedirs(base_dir, exist_ok=True)
                base_abs = _os.path.abspath(base_dir)
                safe = _vs_safe_name(filename)
                if not safe:
                    return web.json_response({"success": False, "error": "유효하지 않은 파일 이름"})

                # 중복 감지: 동일 파일 이름 + 동일한 크기(sha256 선택 사항)는 완전히 동일한 것으로 간주되며 다시 업로드되지 않습니다.
                dup = _vs_find_library_duplicate(safe, total_size or 0, sha256)
                is_last = (chunk_index + 1) >= total_chunks
                if dup:
                    resp = {
                        "success": True,
                        "name": f"whatdreamscost/{dup}",
                        "final_name": dup,
                        "chunk_index": chunk_index,
                        "total_chunks": total_chunks,
                        "is_last": is_last,
                        "duplicate": True,
                        "size": _os.path.getsize(_os.path.join(base_abs, dup)),
                    }
                    if folder:
                        proj_dir = _vs_ensure_project_dir(folder)
                        dst = _os.path.join(proj_dir, dup)
                        created = _vs_make_link(_os.path.join(base_abs, dup), dst)
                        resp["projectPath"] = _vs_project_server_path(folder, dup)
                        resp["alreadyLinked"] = not created
                    return web.json_response(resp)

                # 최종 파일 이름: 첫 번째 블록으로 결정, 후속 블록은 첫 번째 블록에서 반환된 것을 사용 final_name
                if chunk_index == 0:
                    candidate = safe
                    n = 1
                    stem, ext = _os.path.splitext(safe)
                    while _os.path.exists(_os.path.join(base_abs, candidate)):
                        candidate = f"{stem}_{n}{ext}"
                        n += 1
                    final_name = candidate
                else:
                    final_name = final_name or safe

                temp_full = _os.path.abspath(_os.path.join(base_abs, final_name + ".upload"))
                if temp_full != base_abs and not temp_full.startswith(base_abs + _os.sep):
                    return web.json_response({"success": False, "error": "유효하지 않은 경로"})

                mode = 'wb' if chunk_index == 0 else 'ab'
                with open(temp_full, mode) as f:
                    f.write(file_bytes)

                if is_last:
                    final_full = _os.path.abspath(_os.path.join(base_abs, final_name))
                    if final_full != base_abs and not final_full.startswith(base_abs + _os.sep):
                        return web.json_response({"success": False, "error": "유효하지 않은 경로"})
                    # 충돌 파일이 이 기간 동안 생성될 경우 (극소수의 동시성), 접미사를 추가하여 덮어쓰지 않도록 방지
                    if _os.path.exists(final_full):
                        n = 1
                        stem, ext = _os.path.splitext(final_name)
                        while _os.path.exists(_os.path.join(base_abs, final_name)):
                            final_name = f"{stem}_{n}{ext}"
                            n += 1
                        final_full = _os.path.abspath(_os.path.join(base_abs, final_name))
                    _os.replace(temp_full, final_full)

                resp = {
                    "success": True,
                    "name": f"whatdreamscost/{final_name}",
                    "final_name": final_name,
                    "chunk_index": chunk_index,
                    "total_chunks": total_chunks,
                    "is_last": is_last,
                    "size": _os.path.getsize(_os.path.join(base_abs, final_name)) if is_last else 0,
                }
                if direct and folder:
                    # direct 모드: 파일 엔터티가 프로젝트 폴더 내에 직접 위치하며, serverPath는 프로젝트 파일로 가리킵니다.
                    project_path = _vs_project_server_path(folder, final_name)
                    resp["name"] = project_path
                    resp["projectPath"] = project_path
                    resp["directProject"] = True
                elif is_last and folder:
                    proj_dir = _vs_ensure_project_dir(folder)
                    dst = _os.path.join(proj_dir, final_name)
                    created = _vs_make_link(_os.path.join(base_abs, final_name), dst)
                    resp["projectPath"] = _vs_project_server_path(folder, final_name)
                    resp["alreadyLinked"] = not created
                return web.json_response(resp)
            except Exception as e:
                import traceback
                traceback.print_exc()
                return web.json_response({"success": False, "error": str(e)})

        # ============ BSK Video Studio：창문 배경 이미지 (사용자 폴더 저장) user/default/bsk_video_studio/） ============
        # 비디오 워크스테이션 툴바 '의상 아이콘' 버튼으로 로컬 이미지 배경으로 업로드。
        # 고정 파일 이름 bsk_bg.<ext>（덮어쓰기 방식, _1/_2 복사본이 생성되지 않음); 프런트엔드 로딩은 진행됩니다 GET /bsk/vs/get_bg。
        @server.routes.post("/bsk/vs/upload_bg")
        async def vs_upload_bg(request):
            """aria2 대상 파일 및 .aria2 제어 파일 삭제 (취소/청소 사용 bsk_bg.<ext>,이미지 확장명만 허용)。"""
            try:
                import os as _os
                reader = await request.multipart()
                filename = None
                chunk_index = 0
                total_chunks = 1
                file_bytes = None
                async for field in reader:
                    if field.name == 'filename':
                        filename = (await field.text()).strip()
                    elif field.name == 'chunk_index':
                        chunk_index = int((await field.text()) or '0')
                    elif field.name == 'total_chunks':
                        total_chunks = int((await field.text()) or '1')
                    elif field.name == 'file':
                        file_bytes = await field.read()
                if not filename:
                    return web.json_response({"success": False, "error": "누락 filename"})
                if file_bytes is None:
                    return web.json_response({"success": False, "error": "파일 데이터 블록 누락"})

                ext = _os.path.splitext(_os.path.basename(filename))[1].lower()
                if ext == ".jpeg":
                    ext = ".jpg"
                if ext not in (".png", ".jpg", ".webp", ".gif", ".bmp"):
                    return web.json_response({"success": False, "error": "단 이미지 파일 지원(png/jpg/webp/gif/bmp)"})

                _os.makedirs(VS_CFG_DIR, exist_ok=True)
                base_abs = _os.path.abspath(VS_CFG_DIR)
                final_name = "bsk_bg" + ext  # 고정 파일 이름, 덮어쓰기 저장

                if chunk_index == 0:
                    # 첫 번째: 먼저 이전 삭제 bsk_bg.*（형식 변환 후 이전 파일 잔류 방지, 새로운 파일 작성
                    for old in _os.listdir(base_abs):
                        if old.startswith("bsk_bg."):
                            try:
                                _os.remove(_os.path.join(base_abs, old))
                            except OSError:
                                pass
                    mode = 'wb'
                else:
                    mode = 'ab'

                full = _os.path.abspath(_os.path.join(base_abs, final_name))
                if full != base_abs and not full.startswith(base_abs + _os.sep):
                    return web.json_response({"success": False, "error": "유효하지 않은 경로"})

                with open(full, mode) as f:
                    f.write(file_bytes)

                is_last = (chunk_index + 1) >= total_chunks
                resp = {
                    "success": True,
                    "name": final_name,
                    "chunk_index": chunk_index,
                    "total_chunks": total_chunks,
                    "is_last": is_last,
                }
                if is_last:
                    resp["size"] = _os.path.getsize(full)
                return web.json_response(resp)
            except Exception as e:
                import traceback
                traceback.print_exc()
                return web.json_response({"success": False, "error": str(e)})

        # 창문 배경 이미지를 읽습니다（user/default/bsk_video_studio/bsk_bg.<ext>，경로 침투 방지）
        @server.routes.get("/bsk/vs/get_bg")
        async def vs_get_bg(request):
            """반환 창 배경 그림 이미지 내용。

            filename 선택：
            - filename (다음과 같아야 함 bsk_bg.<ext>）：사용자 폴더 bsk_video_studio/에 해당 파일을 반환합니다.；
            - filename이 없는(워크벤치 기본 로드), 사용자 디렉터리의 하나를 우선적으로 반환합니다. bsk_bg.<ext>，
              사용자 디렉토리가 없으면 패널/H3.png(공장 기본 배경 이미지)로 되돌아갑니다.）。
            """
            try:
                import os as _os
                name = (request.query.get("filename") or "").strip()
                if name:
                    safe = _os.path.basename(name)
                    if not safe.startswith("bsk_bg."):
                        return web.json_response({"success": False, "error": "유효하지 않은 파일 이름"}, status=404)
                    base_abs = _os.path.abspath(VS_CFG_DIR)
                    full = _os.path.abspath(_os.path.join(base_abs, safe))
                    if full != base_abs and not full.startswith(base_abs + _os.sep):
                        return web.json_response({"success": False, "error": "유효하지 않은 경로"}, status=404)
                    if not _os.path.isfile(full):
                        return web.json_response({"success": False, "error": "파일이 존재하지 않습니다"}, status=404)
                    return web.FileResponse(full)
                # 기본 로드: 사용자 디렉토리 우선 bsk_bg.<ext>，구체적으로 유지할 내용 panel/H3.png
                try:
                    bg_files = sorted(f for f in _os.listdir(VS_CFG_DIR) if f.startswith("bsk_bg."))
                except OSError:
                    bg_files = []
                if bg_files:
                    return web.FileResponse(_os.path.join(VS_CFG_DIR, bg_files[0]))
                fallback = _os.path.join(EXTENSION_DIR, "panel", "H3.png")
                if _os.path.isfile(fallback):
                    return web.FileResponse(fallback)
                return web.json_response({"success": False, "error": "파일이 존재하지 않습니다"}, status=404)
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)}, status=404)

        # ============ BSK Video Studio：서버에 있는 자료 목록 표시 ============
        # 목록 input/whatdreamscost/ 모든 파일, 프론트엔드 "서버 자료" 태그 표시。
        # 반환 [{name, serverPath, size, mtime}]，serverPath 형상과 같음 "whatdreamscost/xxx"。
        @server.routes.get("/bsk/vs/list_assets")
        async def vs_list_assets(request):
            try:
                import os as _os
                base_dir = _os.path.join(folder_paths.get_input_directory(), "whatdreamscost")
                _os.makedirs(base_dir, exist_ok=True)
                base_abs = _os.path.abspath(base_dir)
                result = []
                for fn in _os.listdir(base_abs):
                    if fn.endswith(".upload"):
                        continue
                    fp = _os.path.join(base_abs, fn)
                    if not _os.path.isfile(fp):
                        continue
                    try:
                        st = _os.stat(fp)
                    except OSError:
                        continue
                    result.append({
                        "name": fn,
                        "serverPath": f"whatdreamscost/{fn}",
                        "size": st.st_size,
                        "mtime": int(st.st_mtime),
                    })
                result.sort(key=lambda x: x["mtime"], reverse=True)
                return web.json_response({"success": True, "files": result})
            except Exception as e:
                return web.json_response({"success": False, "error": str(e), "files": []})

        # ============ BSK Video Studio：삭제 업로드된 자료 파일 ============
        @server.routes.post("/bsk/vs/delete_asset")
        async def vs_delete_asset(request):
            """input/whatdreamscost/ 아래의 자료 파일 삭제。
            body {path}：path 형식은 'whatdreamscost/xxx.mp4' 또는 단순 파일 이름과 같습니다 'xxx.mp4'。
            whatdreamscost 하위 디렉토리 내 파일만 삭제 가능하도록 제한, 경로 침투 방지。"""
            try:
                import os as _os
                data = await request.json()
                rel = (data.get("path") or "").strip()
                if not rel:
                    return web.json_response({"success": False, "error": "누락 path"})
                base_dir = _os.path.join(folder_paths.get_input_directory(), "whatdreamscost")
                full = _os.path.abspath(_os.path.join(base_dir, _os.path.basename(rel)))
                base_abs = _os.path.abspath(base_dir)
                if full != base_abs and not full.startswith(base_abs + _os.sep):
                    return web.json_response({"success": False, "error": "잘못된 경로(whatdreamscost 하위 디렉토리만 허용)）"})
                if not _os.path.isfile(full):
                    return web.json_response({"success": False, "error": "파일이 존재하지 않습니다"})
                _os.remove(full)
                return web.json_response({"success": True})
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        # ============ BSK Video Studio：백엔드 자르기 monkey-patch 덜렁 로드 ============
        @server.routes.get("/bsk/vs/ensure_crop_patch")
        async def vs_ensure_crop_patch(request):
            """로드된 ltx_director 모듈에 크롭 패치(멱등) 설치）。
            프론트엔드 제출 전 호출하여 _load_image_tensor/_load_video_tensor_가 bskCrop 자르기를 감싸고 있는지 확인하십시오。"""
            try:
                import importlib.util as _ilu
                _patch_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nodes", "vs_ltx_patch.py")
                _mod_name = "_bsk_vs_ltx_patch"
                _spec = _ilu.spec_from_file_location(_mod_name, _patch_path)
                _pmod = _ilu.module_from_spec(_spec)
                _spec.loader.exec_module(_pmod)
                result = _pmod.ensure_installed()
                return web.json_response({"success": result.get("ok", False), **result})
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        # ============ danbooru 증명서 저장 (ComfyUI 사용자 디렉토리 저장)） ============
        @server.routes.get("/comfyui_panel/danbooru_config")
        async def get_danbooru_config(request):
            """반환 danbooru 사용자 이름 및 키 설정 여부 (암호화되지 않음) key）。"""
            try:
                cfg = _load_danbooru_config()
                return web.json_response({
                    "success": True,
                    "login": cfg.get("login", ""),
                    "has_key": bool(cfg.get("api_key")),
                })
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        @server.routes.post("/comfyui_panel/danbooru_config")
        async def post_danbooru_config(request):
            """danbooru 사용자 이름 및 저장 API Key。body {login, api_key}。"""
            try:
                data = await request.json()
                login = (data.get("login") or "").strip()
                api_key = (data.get("api_key") or "").strip()
                if not login or not api_key:
                    return web.json_response({"success": False, "error": "사용자 이름과 API 키는 모두 비워 둘 수 없습니다."})
                _save_danbooru_config(login, api_key)
                return web.json_response({"success": True})
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        # ============ AI 텍스트 노드: 규칙 저장 (ComfyUI 사용자 디렉토리에 저장)） ============
        AI_RULE_DIR = os.path.join(folder_paths.base_path, "user", "default", "ai_rules")
        AI_TEXT_CFG_PATH = os.path.join(folder_paths.base_path, "user", "default", "ai_text_config.json")

        @server.routes.get("/comfyui_panel/ai_rules")
        async def list_ai_rules(request):
            """user/default/ai_rules 아래의 목록을 나열 *.json 규칙 파일, 수정 시간 역순 (최신순)）。"""
            try:
                os.makedirs(AI_RULE_DIR, exist_ok=True)
                base_full = os.path.abspath(AI_RULE_DIR)
                rules = []
                for f in os.listdir(base_full):
                    if f.endswith(".json") and os.path.isfile(os.path.join(base_full, f)):
                        full = os.path.join(base_full, f)
                        try:
                            mtime = os.path.getmtime(full)
                        except Exception:
                            mtime = 0
                        rules.append({"name": f, "mtime": mtime})
                rules.sort(key=lambda x: x["mtime"], reverse=True)
                return web.json_response({
                    "success": True,
                    "rules": rules,
                    "files": [r["name"] for r in rules],
                })
            except Exception as e:
                return web.json_response({"success": False, "error": str(e), "rules": [], "files": []})

        @server.routes.post("/comfyui_panel/ai_rules/save")
        async def save_ai_rule(request):
            """저장 규칙：body {name, content}。"""
            try:
                data = await request.json()
                name = (data.get("name") or "").strip()
                content = data.get("content", "")
                if not name:
                    return web.json_response({"success": False, "error": "규칙 이름이 없습니다"})
                if not name.endswith(".json"):
                    name += ".json"
                os.makedirs(AI_RULE_DIR, exist_ok=True)
                base_full = os.path.abspath(AI_RULE_DIR)
                full = os.path.abspath(os.path.join(base_full, name))
                if full != base_full and not full.startswith(base_full + os.sep):
                    return web.json_response({"success": False, "error": "유효하지 않은 파일 이름 (경로 침투)）"})
                with open(full, "w", encoding="utf-8") as f:
                    f.write(content)
                return web.json_response({"success": True, "name": name})
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        @server.routes.post("/comfyui_panel/ai_rules/load")
        async def load_ai_rule(request):
            """규칙 읽기：body {name}，반환 {content}。"""
            try:
                data = await request.json()
                name = (data.get("name") or "").strip()
                if not name:
                    return web.json_response({"success": False, "error": "규칙 이름이 없습니다"})
                if not name.endswith(".json"):
                    name += ".json"
                base_full = os.path.abspath(AI_RULE_DIR)
                full = os.path.abspath(os.path.join(base_full, name))
                if full != base_full and not full.startswith(base_full + os.sep):
                    return web.json_response({"success": False, "error": "유효하지 않은 파일 이름 (경로 침투)）"})
                if not os.path.isfile(full):
                    return web.json_response({"success": False, "error": "파일이 존재하지 않습니다"})
                with open(full, "r", encoding="utf-8") as f:
                    content = f.read()
                return web.json_response({"success": True, "content": content})
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        @server.routes.post("/comfyui_panel/ai_rules/delete")
        async def delete_ai_rule(request):
            """규칙 삭제：body {name}。"""
            try:
                data = await request.json()
                name = (data.get("name") or "").strip()
                if not name:
                    return web.json_response({"success": False, "error": "규칙 이름이 없습니다"})
                if not name.endswith(".json"):
                    name += ".json"
                base_full = os.path.abspath(AI_RULE_DIR)
                full = os.path.abspath(os.path.join(base_full, name))
                if full != base_full and not full.startswith(base_full + os.sep):
                    return web.json_response({"success": False, "error": "유효하지 않은 파일 이름 (경로 침투)）"})
                if not os.path.isfile(full):
                    return web.json_response({"success": False, "error": "파일이 존재하지 않습니다"})
                os.remove(full)
                return web.json_response({"success": True, "name": name})
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        # ============ AI 텍스트 노드: API 설정 (전역 단일, ComfyUI 사용자 디렉토리에 저장)） ============
        DEFAULT_AI_TEXT_CONFIG = {
            "provider": "deepseek",
            "api_key": "",
            "base_url": "https://api.deepseek.com",
            "model": "deepseek-chat",
            "temperature": 0.7,
            "max_tokens": 1024,
            "vision": "auto",
        }

        @server.routes.get("/comfyui_panel/ai_text_settings")
        async def get_ai_text_settings(request):
            """AI 텍스트 노드 전역 설정 읽기, 기본값과 병합하여 반환。"""
            try:
                cfg = dict(DEFAULT_AI_TEXT_CONFIG)
                if os.path.isfile(AI_TEXT_CFG_PATH):
                    with open(AI_TEXT_CFG_PATH, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if isinstance(data, dict):
                        for k in DEFAULT_AI_TEXT_CONFIG:
                            v = data.get(k)
                            if v is not None and v != "":
                                cfg[k] = v
                return web.json_response({"success": True, "config": cfg})
            except Exception as e:
                return web.json_response({"success": False, "error": str(e), "config": dict(DEFAULT_AI_TEXT_CONFIG)})

        @server.routes.post("/comfyui_panel/ai_text_settings/save")
        async def save_ai_text_settings(request):
            """저장 AI 텍스트 노드 전역 설정：body {config}。"""
            try:
                data = await request.json()
                cfg = data.get("config") or {}
                merged = dict(DEFAULT_AI_TEXT_CONFIG)
                for k in DEFAULT_AI_TEXT_CONFIG:
                    if k in cfg and cfg[k] is not None and cfg[k] != "":
                        merged[k] = cfg[k]
                os.makedirs(os.path.dirname(AI_TEXT_CFG_PATH), exist_ok=True)
                with open(AI_TEXT_CFG_PATH, "w", encoding="utf-8") as f:
                    json.dump(merged, f, ensure_ascii=False, indent=2)
                return web.json_response({"success": True, "config": merged})
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        @server.routes.get("/comfyui_panel/input_files")
        async def get_input_files(request):
            try:
                # ===== 디버그 핀: 계산수 =====
                try:
                    from . import debug_probe
                    debug_probe.incr("input_files")
                except Exception:
                    pass
                input_dir = folder_paths.get_input_directory()
                image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp'}
                files = []
                for f in os.listdir(input_dir):
                    file_path = os.path.join(input_dir, f)
                    if os.path.isfile(file_path):
                        ext = os.path.splitext(f)[1].lower()
                        if ext in image_extensions:
                            # === 수정: 잘라낸 이미지 파일 필터링 (bsk_crop_*.png) ===
                            if f.startswith('bsk_crop_'):
                                continue
                            stat = os.stat(file_path)
                            files.append({
                                "name": f,
                                "size": stat.st_size,
                                "mtime": stat.st_mtime
                            })
                files.sort(key=lambda x: x['mtime'], reverse=True)
                return web.json_response({
                    "success": True,
                    "files": files,
                    "count": len(files)
                })
            except Exception as e:
                return web.json_response({
                    "success": False,
                    "error": str(e),
                    "files": [],
                    "count": 0
                })

        @server.routes.get("/comfyui_panel/latest_image")
        async def get_latest_image(request):
            """경고: 이 플러그인은 많은 시간과 노력을 들여 만든 것입니다）"""
            try:
                # ===== 디버그 핀: 계산수 =====
                try:
                    from . import debug_probe
                    debug_probe.incr("latest_image")
                except Exception:
                    pass
                input_dir = folder_paths.get_input_directory()
                image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.tiff', '.tif', '.avif'}
                files = []
                for f in os.listdir(input_dir):
                    file_path = os.path.join(input_dir, f)
                    if os.path.isfile(file_path):
                        ext = os.path.splitext(f)[1].lower()
                        if ext in output_extensions:
                            stat = os.stat(file_path)
                            files.append((f, stat.st_mtime))
                
                # W 단축키로 워크플로우 편집기 열기）
                files.sort(key=lambda x: x[1], reverse=True)
                
                if files:
                    return web.json_response({
                        "success": True,
                        "filename": files[0][0],
                        "mtime": files[0][1]
                    })
                else:
                    return web.json_response({
                        "success": False,
                        "error": "No image files found"
                    })
            except Exception as e:
                return web.json_response({
                    "success": False,
                    "error": str(e)
                })

        # =========================================
        # 데이터베이스 생성 중 (라이브러리 미리보기 시간 표시)）
        # 사용자 디렉터리에 bsk_output_times.json으로 지속성을 유지합니다. 키는 출력 파일 이름입니다。
        # 프론트엔드는 생성 완료 시 record_output_time을 POST하고, output_files를 반환할 때 함께 보냅니다. elapsed_ms。
        # =========================================
        OUTPUT_TIME_DB_PATH = Path(folder_paths.get_user_directory()) / "bsk_output_times.json"

        def _out_relkey(name, base_dir):
            """출력/입력 파일 식별자(순수 파일 이름 / 상대 경로 / 절대 경로)를 base_dir의 posix 상대 경로로 정규화합니다. key。
            '..'과 같은 범위를 포함하는 경우 None을 반환합니다. 라이브러리 하위 디렉터리는 이 함수를 통일하여 사용합니다.。"""
            if not name:
                return None
            n = str(name).strip().replace('\\', '/')
            if not n:
                return None
            if os.path.isabs(n):
                try:
                    rel = os.path.relpath(n, base_dir)
                except ValueError:
                    return None
                if rel.startswith('..'):
                    return None
                return rel.replace(os.sep, '/')
            parts = []
            for seg in n.split('/'):
                if seg in ('', '.'):
                    continue
                if seg == '..':
                    return None
                parts.append(seg)
            return '/'.join(parts) if parts else None

        def _safe_output_path(base_dir, rel):
            """posix 상대 경로를 통해 base_dir 내의 절대 경로를 파싱하고, 범위를 벗어나면 None 반환 (Path traversal 방지)）。"""
            if not rel:
                return None
            full = os.path.realpath(os.path.join(base_dir, rel.replace('/', os.sep)))
            base = os.path.realpath(base_dir)
            if full == base or not full.startswith(base + os.sep):
                return None
            return full

        def _load_output_times():
            try:
                if OUTPUT_TIME_DB_PATH.exists():
                    with open(OUTPUT_TIME_DB_PATH, 'r', encoding='utf-8') as f:
                        return json.load(f)
            except Exception as e:
                print(f"[BSK OutputTimes] 읽기 실패: {e}")
            return {}

        def _save_output_times(data):
            try:
                OUTPUT_TIME_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
                with open(OUTPUT_TIME_DB_PATH, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"[BSK OutputTimes] 쓰기 실패: {e}")

        @server.routes.post("/comfyui_panel/record_output_time")
        async def record_output_time(request):
            """출력 파일(이미지/비디오) 생성 시간 기록。
            body: {"files": [{"filename": "x.png", "elapsed_ms": 1234}]}
            이미지 라이브러리 미리보기 상단 왼쪽 모서리에 생성 시간 표시。"""
            try:
                body = await request.json()
                files = body.get("files") or []
                if not files:
                    return web.json_response({"success": False, "error": "No files"})
                db = _load_output_times()
                now = time.time()
                changed = False
                output_dir = folder_paths.get_output_directory()
                for item in files:
                    fn = item.get("filename") if isinstance(item, dict) else None
                    ms = item.get("elapsed_ms") if isinstance(item, dict) else None
                    if not fn or ms is None:
                        continue
                    key = _out_relkey(fn, output_dir)
                    if not key:
                        continue
                    db[key] = {"elapsed_ms": int(ms), "updated_at": now}
                    changed = True
                if changed:
                    _save_output_times(db)
                return web.json_response({"success": True, "count": len(files)})
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        @server.routes.get("/comfyui_panel/output_files")
        async def get_output_files(request):
            """출력 디렉토리의 이미지 및 비디오 파일 목록을 가져오고 페이지 매김을 지원합니다.。
            name 출력 디렉토리의 posix 경로 (루트 디렉토리 파일=순 파일 이름, 하위 디렉토리 파일=subdir/xxx.png）。"""
            try:
                output_dir = folder_paths.get_output_directory()
                output_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.mp4', '.webm', '.mov', '.avi', '.mkv'}
                time_db = _load_output_times()
                files = []
                for root, _dirs, names in os.walk(output_dir):
                    for f in names:
                        ext = os.path.splitext(f)[1].lower()
                        if ext not in output_extensions:
                            continue
                        file_path = os.path.join(root, f)
                        if not os.path.isfile(file_path):
                            continue
                        rel = os.path.relpath(file_path, output_dir).replace(os.sep, '/')
                        stat = os.stat(file_path)
                        files.append({
                            "name": rel,
                            "size": stat.st_size,
                            "mtime": stat.st_mtime,
                            "elapsed_ms": (time_db.get(rel) or {}).get("elapsed_ms")
                        })
                files.sort(key=lambda x: x['mtime'], reverse=True)

                # 페이지 지원
                total = len(files)
                try:
                    offset = int(request.query.get("offset", "0"))
                    limit = int(request.query.get("limit", "0"))
                except (ValueError, TypeError):
                    offset = 0
                    limit = 0

                if limit > 0:
                    files = files[offset:offset + limit]

                return web.json_response({
                    "success": True,
                    "files": files,
                    "count": total,
                    "offset": offset,
                    "hasMore": (offset + len(files)) < total if limit > 0 else False
                })
            except Exception as e:
                return web.json_response({
                    "success": False,
                    "error": str(e),
                    "files": [],
                    "count": 0
                })

        # ========== 비디오 태그 콘텐츠 컨테이너 - 부모 컨테이너 채우기 API ==========
        def _bsk_vs_runs_dir():
            """bsk_minimax_director._runs_dir 일치하는 캐시 디렉토리 위치。"""
            try:
                base = os.path.join(folder_paths.get_user_directory(), "default", "bsk_video_studio")
            except Exception:
                base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bsk_video_studio")
            return os.path.join(base, "runs")

        @server.routes.get("/comfyui_panel/vs_runs/last")
        async def vs_runs_last(request):
            """반환 최근 생성된 포인터（unique_id / prompt_id / status）。"""
            try:
                lp = os.path.join(_bsk_vs_runs_dir(), "last.json")
                if not os.path.isfile(lp):
                    return web.json_response({"success": True, "exists": False})
                with open(lp, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return web.json_response({"success": True, "exists": True, "data": data})
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        @server.routes.get("/comfyui_panel/vs_runs/{uid}")
        async def vs_runs_get(request):
            """지정된 unique_id의 과정 프레임 캐시 (상태 포함)）。"""
            try:
                uid = request.match_info.get("uid", "")
                if not uid or "/" in uid or "\\" in uid:
                    return web.json_response({"success": False, "error": "invalid uid"})
                rp = os.path.join(_bsk_vs_runs_dir(), f"{uid}.json")
                if not os.path.isfile(rp):
                    return web.json_response({"success": True, "exists": False})
                with open(rp, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return web.json_response({"success": True, "exists": True, "data": data})
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        @server.routes.get("/comfyui_panel/vs_runs/{uid}/frames")
        async def vs_runs_frames(request):
            """점진적 프레임 추출：GET ?from=N 단위 인덱스만 반환 >= N 프레임의。

            프레임 인덱스 i는 bsk_minimax_director에서 쓰레기를 작성할 때마다 타임스탬프가 지정되며, frames 배열 인덱스와 항상 일치합니다 (단조, 후퇴 없음).）。
            프론트엔드에서 단일 커서 유지: WebSocket 실시간 프레임과 HTTP 보간을 공유합니다. 자연스러운 멱등성을 제공합니다.——
            동일 프레임 어떤 경로로 도착해도 한 번만 렌더링; WS 연결 끊김 시 HTTP로 정확하게 누락된 부분 채우기。
            Only transmit new frames per iteration, avoid full base64 repeated transmission (especially when multiple concat groups)）。
            """
            try:
                uid = request.match_info.get("uid", "")
                if not uid or "/" in uid or "\\" in uid:
                    return web.json_response({"success": False, "error": "invalid uid"})
                try:
                    frm = int(request.query.get("from", "0") or 0)
                except Exception:
                    frm = 0
                if frm < 0:
                    frm = 0
                rp = os.path.join(_bsk_vs_runs_dir(), f"{uid}.json")
                if not os.path.isfile(rp):
                    return web.json_response({"success": True, "exists": False,
                                              "frames": [], "total": 0})
                with open(rp, "r", encoding="utf-8") as f:
                    data = json.load(f)
                frames = data.get("frames") or []
                return web.json_response({
                    "success": True,
                    "exists": True,
                    "unique_id": uid,
                    "prompt_id": data.get("prompt_id") or "",
                    "run_id": data.get("run_id"),
                    "status": data.get("status") or "running",
                    "total": len(frames),
                    "frames": frames[frm:],
                })
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        @server.routes.post("/comfyui_panel/vs_runs/track")
        async def vs_runs_track(request):
            """프론트 성공 제출 후 prompt_id 보완 (process 내에서 가져올 수 없음) prompt_id）。
            자체 포함 구현 (재사용 _bsk_vs_runs_dir), 모듈 간 import 의존성 제거, 노드 로딩 시퀀스 문제 방지。"""
            try:
                data = await request.json()
                uid = (data.get("unique_id") or "").strip()
                pid = (data.get("prompt_id") or "").strip()
                if not uid:
                    return web.json_response({"success": False, "error": "unique_id required"})
                runs_dir = _bsk_vs_runs_dir()
                # last.json 업데이트 (이 실행의 포인터)）
                lp = os.path.join(runs_dir, "last.json")
                last = {"unique_id": uid, "prompt_id": pid, "status": "running",
                        "updated_at": time.time()}
                if os.path.isfile(lp):
                    try:
                        with open(lp, "r", encoding="utf-8") as f:
                            cur = json.load(f)
                        if cur.get("unique_id") == uid:
                            cur["prompt_id"] = pid
                            last = cur
                    except Exception:
                        pass
                with open(lp, "w", encoding="utf-8") as f:
                    json.dump(last, f, ensure_ascii=False)
                # 실행 파일 업데이트 (존재 시 덮어쓰기) prompt_id）
                rp = os.path.join(runs_dir, f"{uid}.json")
                if os.path.isfile(rp):
                    try:
                        with open(rp, "r", encoding="utf-8") as f:
                            rd = json.load(f)
                        rd["prompt_id"] = pid
                        with open(rp, "w", encoding="utf-8") as f:
                            json.dump(rd, f, ensure_ascii=False)
                    except Exception as e:
                        return web.json_response({"success": False, "error": f"track run 실패: {e}"})
                return web.json_response({"success": True})
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        @server.routes.post("/comfyui_panel/vs_runs/result")
        async def vs_runs_result(request):
            """프론트엔드가 최종 결과 비디오(executed 이벤트)를 가져온 후, 비디오 파일 정보를 run 캐시에 기록합니다. last.json。
            자체 포함 구현 ( _bsk_vs_runs_dir 재사용), 모듈 간 의존성 없음 import。
            목적: 최종 결과는 더 이상 의존하지 않습니다. /history/<prompt_id>（prompt_id 손실될 수 있음 / history 구조가 변하기 쉬움），
            대신 프로세스 프레임과 함께 캐시 파일에 영구 저장하고, 패널을 새로 고치거나 다시 열면 최종 비디오를 안정적으로 복원합니다。"""
            try:
                data = await request.json()
                uid = (data.get("unique_id") or "").strip()
                result = data.get("result") or {}
                if not uid:
                    return web.json_response({"success": False, "error": "unique_id required"})
                if not result.get("filename"):
                    return web.json_response({"success": False, "error": "result.filename required"})
                runs_dir = _bsk_vs_runs_dir()
                try:
                    os.makedirs(runs_dir, exist_ok=True)
                except Exception:
                    pass
                rec = {"filename": result.get("filename"),
                       "subfolder": result.get("subfolder") or "",
                       "type": result.get("type") or "output",
                       "updated_at": time.time()}
                # run 파일 작성(존재하면 병합, 존재하지 않으면 결과만 포함된 기록 생성, 복구용)）
                rp = os.path.join(runs_dir, f"{uid}.json")
                rd = {}
                if os.path.isfile(rp):
                    try:
                        with open(rp, "r", encoding="utf-8") as f:
                            rd = json.load(f)
                    except Exception:
                        rd = {}
                rd["unique_id"] = uid
                rd["result"] = rec
                rd.setdefault("status", "sampled")
                rd.setdefault("frames", [])
                tmp = rp + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(rd, f, ensure_ascii=False)
                os.replace(tmp, rp)
                # last.json 포인터 작성 (현재 uid가 last일 때만)）
                lp = os.path.join(runs_dir, "last.json")
                last = {"unique_id": uid, "prompt_id": (rd.get("prompt_id") or ""),
                        "status": rd.get("status", "sampled"),
                        "result": rec, "updated_at": time.time()}
                if os.path.isfile(lp):
                    try:
                        with open(lp, "r", encoding="utf-8") as f:
                            cur = json.load(f)
                        if cur.get("unique_id") == uid:
                            cur["result"] = rec
                            cur["status"] = rd.get("status", "sampled")
                            cur["updated_at"] = time.time()
                            last = cur
                    except Exception:
                        pass
                ltmp = lp + ".tmp"
                with open(ltmp, "w", encoding="utf-8") as f:
                    json.dump(last, f, ensure_ascii=False)
                os.replace(ltmp, lp)
                return web.json_response({"success": True})
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        @server.routes.post("/comfyui_panel/delete_output_file")
        async def delete_output_file(request):
            """단일 출력 파일을 삭제합니다(자식 디렉터리의 상대 경로를 지원)하고, 동시에 소요 시간 기록을 정리합니다.。"""
            try:
                data = await request.json()
                filename = (data.get("filename") or "").strip()
                if not filename:
                    return web.json_response({"success": False, "error": "Missing filename"})
                output_dir = folder_paths.get_output_directory()
                rel = _out_relkey(filename, output_dir)
                target = _safe_output_path(output_dir, rel)
                if not target or not os.path.isfile(target):
                    return web.json_response({"success": False, "error": "파일이 존재하지 않습니다"})
                os.remove(target)
                try:
                    db = _load_output_times()
                    if db.pop(rel, None) is not None:
                        _save_output_times(db)
                except Exception as _e:
                    print(f"[BSK OutputTimes] 시간 소요 기록 삭제 실패: {_e}")
                return web.json_response({"success": True})
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        @server.routes.post("/comfyui_panel/upload_crop")
        async def upload_cropped_image(request):
            try:
                reader = await request.multipart()
                filename = None
                image_data = None
                overwrite = False
                resize_width = None
                resize_height = None

                async for field in reader:
                    if field.name == 'image':
                        image_data = await field.read()
                        if field.filename:
                            filename = field.filename
                    elif field.name == 'filename':
                        filename = (await field.read()).decode('utf-8')
                    elif field.name == 'overwrite':
                        overwrite = (await field.read()).decode('utf-8').lower() == 'true'
                    elif field.name == 'resize_width':
                        resize_width = int((await field.read()).decode('utf-8'))
                    elif field.name == 'resize_height':
                        resize_height = int((await field.read()).decode('utf-8'))

                if not filename or not image_data:
                    return web.json_response({
                        "success": False,
                        "error": "Missing filename or image data"
                    })

                # 백엔드 줌 (작은 이미지 모드)）
                if resize_width and resize_height:
                    try:
                        from PIL import Image
                        import io
                        img = Image.open(io.BytesIO(image_data))
                        original_size = img.size
                        # 이미지 크기가 대상 크기보다 작을 때만 확대합니다
                        if img.width < resize_width or img.height < resize_height:
                            # 고품질 업샘플링 사용
                            img_resized = img.resize((resize_width, resize_height), Image.Resampling.LANCZOS)
                            output_buffer = io.BytesIO()
                            # 보존 원본 모드 (RGBA 또는)RGB）
                            img_resized.save(output_buffer, format='PNG')
                            image_data = output_buffer.getvalue()
                            print(f"[ComfyUI Panel] 백엔드 확대: {original_size} -> {resize_width}x{resize_height}")
                    except Exception as e:
                        print(f"[ComfyUI Panel] 백엔드 크기 조정 실패: {e}")

                # === 디버그: 잘라낸 이미지 저장 ===
                if DEBUG_MASK_ALPHA:
                    try:
                        debug_save_backup("05_crop_upload", image_data, f"업로드된 이미지 자르기: {filename}")
                        # RGBA 이미지인지 확인
                        from PIL import Image
                        import io
                        try:
                            img = Image.open(io.BytesIO(image_data))
                            if img.mode == 'RGBA':
                                import tempfile
                                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                                    tmp.write(image_data)
                                    tmp.flush()
                                    result = debug_verify_png(tmp.name)
                                    print(f"[MaskAlphaDebug] 자르기 업로드 검증: {result}")
                                    os.unlink(tmp.name)
                        except:
                            pass
                    except Exception as e:
                        print(f"[MaskAlphaDebug] 잘라낸 이미지 저장 실패: {e}")

                input_dir = folder_paths.get_input_directory()
                file_path = os.path.join(input_dir, filename)

                # 이미지 형식과 확장명이 일치하는지 확인: 프론트엔드는 .png로 일괄적으로 지정，
                # 드래그 앤 드롭으로 업로드한 원본 데이터는 JPEG/WebP/AVIF 등일 수 있습니다.，
                # PIL로 다시 인코딩하여 실제 PNG로 변환, 피하기"가짜PNG"파일
                try:
                    from PIL import Image
                    import io
                    img = Image.open(io.BytesIO(image_data))
                    output_buffer = io.BytesIO()
                    # RGBA 채널 유지 (투명도 있는 경우)）
                    save_img = img
                    if img.mode == 'RGBA':
                        save_img = img
                    elif img.mode in ('LA', 'PA'):
                        save_img = img.convert('RGBA')
                    elif img.mode == 'P':
                        # 컬러 팔레트 모드: 투명도가 있는 경우 RGBA로 변환하고, 그렇지 않으면 변환합니다. RGB
                        if 'transparency' in img.info:
                            save_img = img.convert('RGBA')
                        else:
                            save_img = img.convert('RGB')
                    elif img.mode not in ('RGB', 'RGBA'):
                        save_img = img.convert('RGB')
                    save_img.save(output_buffer, format='PNG')
                    image_data = output_buffer.getvalue()
                except Exception as e:
                    # PIL 처리할 수 없는 형식(극히 드물), 원본 데이터 유지
                    print(f"[ComfyUI Panel] Warning: image format conversion failed, saving raw data: {e}")

                with open(file_path, 'wb') as f:
                    f.write(image_data)

                return web.json_response({
                    "success": True,
                    "name": filename,
                    "size": len(image_data),
                    "path": file_path
                })

            except Exception as e:
                return web.json_response({
                    "success": False,
                    "error": str(e)
                })

        @server.routes.post("/comfyui_panel/crop_image")
        async def crop_image_backend(request):
            """백엔드 이미지 자르기API
            원래 이미지 파일 이름과 자르기 매개변수를 받아 자르기 및 저장
            """
            try:
                from PIL import Image
                import io
                
                data = await request.json()
                
                source_filename = data.get('source_filename')
                crop_params = data.get('crop_params', {})
                target_width = data.get('target_width')
                target_height = data.get('target_height')
                output_filename = data.get('output_filename')
                
                if not source_filename or not output_filename:
                    return web.json_response({
                        "success": False,
                        "error": "Missing source_filename or output_filename"
                    })
                
                # 원본 이미지 읽기
                input_dir = folder_paths.get_input_directory()
                source_path = os.path.join(input_dir, source_filename)
                
                if not os.path.exists(source_path):
                    return web.json_response({
                        "success": False,
                        "error": f"Source file not found: {source_filename}"
                    })
                
                # 이미지 로드
                img = Image.open(source_path)
                original_mode = img.mode
                original_size = img.size
                
                # 프롬프트 추출 파라미터
                crop_x = int(crop_params.get('x', 0))
                crop_y = int(crop_params.get('y', 0))
                crop_w = int(crop_params.get('width', original_size[0]))
                crop_h = int(crop_params.get('height', original_size[1]))
                
                # 확인하여 크롭 영역이 이미지 내에 있는지 확인
                crop_x = max(0, min(crop_x, original_size[0] - 1))
                crop_y = max(0, min(crop_y, original_size[1] - 1))
                crop_w = min(crop_w, original_size[0] - crop_x)
                crop_h = min(crop_h, original_size[1] - crop_y)
                
                # 실행 자르기
                cropped = img.crop((crop_x, crop_y, crop_x + crop_w, crop_y + crop_h))
                
                # 크기 조정이 필요할 경우
                if target_width and target_height:
                    target_width = int(target_width)
                    target_height = int(target_height)
                    
                    # === 수정: RGB 및 Alpha 채널 분리, RGB가 Alpha에 오염되지 않도록 방지 ===
                    if cropped.mode == 'RGBA':
                        # 채널 분리
                        r, g, b, a = cropped.split()
                        
                        # RGB 병합resize
                        rgb = Image.merge('RGB', (r, g, b))
                        rgb_resized = rgb.resize((target_width, target_height), Image.Resampling.LANCZOS)
                        
                        # Alpha채널 단독resize
                        a_resized = a.resize((target_width, target_height), Image.Resampling.LANCZOS)
                        
                        # 병합 되돌리기RGBA
                        r_resized, g_resized, b_resized = rgb_resized.split()
                        cropped = Image.merge('RGBA', (r_resized, g_resized, b_resized, a_resized))
                    else:
                        # 非RGBA 이미지, 원래대로 사용resize
                        cropped = cropped.resize((target_width, target_height), Image.Resampling.LANCZOS)
                
                # 잘라낸 이미지 저장
                output_path = os.path.join(input_dir, output_filename)
                
                # 원래 모드로 저장, RGBA 채널 유지
                if original_mode == 'RGBA':
                    cropped.save(output_path, format='PNG')
                else:
                    # 비-RGBA 이미지의 경우 RGB로 변환하여 저장 (필요한 경우)）
                    if cropped.mode in ('RGBA', 'LA', 'P'):
                        # 투명 채널이 있으면 PNG 형식 유지
                        cropped.save(output_path, format='PNG')
                    else:
                        cropped.save(output_path, format='PNG')
                
                return web.json_response({
                    "success": True,
                    "filename": output_filename,
                    "source": source_filename,
                    "crop_region": {
                        "x": crop_x,
                        "y": crop_y,
                        "width": crop_w,
                        "height": crop_h
                    },
                    "output_size": {
                        "width": cropped.width,
                        "height": cropped.height
                    }
                })
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                return web.json_response({
                    "success": False,
                    "error": str(e)
                })

        @server.routes.post("/comfyui_panel/delete_file")
        async def delete_input_file(request):
            try:
                data = await request.json()
                filename = data.get('filename')

                if not filename:
                    return web.json_response({
                        "success": False,
                        "error": "Missing filename"
                    })

                input_dir = folder_paths.get_input_directory()
                file_path = os.path.join(input_dir, filename)

                if not os.path.abspath(file_path).startswith(os.path.abspath(input_dir)):
                    return web.json_response({
                        "success": False,
                        "error": "Invalid file path"
                    })

                if os.path.exists(file_path):
                    os.remove(file_path)
                    return web.json_response({
                        "success": True,
                        "message": f"Deleted {filename}"
                    })
                else:
                    return web.json_response({
                        "success": False,
                        "error": "File not found"
                    })

            except Exception as e:
                return web.json_response({
                    "success": False,
                    "error": str(e)
                })

        @server.routes.post("/comfyui_panel/copy_image")
        async def copy_image(request):
            """입력 디렉토리의 이미지 파일을 새 파일 이름으로 복사
            
            파일명 충돌 처리: 사용자가 다른 노드의 bsk_source 파일을 선택할 때，
            올바른 파일 이름으로 자동 복사。
            """
            try:
                data = await request.json()
                source_filename = data.get('source_filename')
                target_filename = data.get('target_filename')

                if not source_filename or not target_filename:
                    return web.json_response({
                        "success": False,
                        "error": "Missing source_filename or target_filename"
                    })

                input_dir = folder_paths.get_input_directory()
                source_path = os.path.join(input_dir, source_filename)
                target_path = os.path.join(input_dir, target_filename)

                # 보안 검사: 파일 경로가 입력 디렉토리 내에 있어야 함
                if not os.path.abspath(source_path).startswith(os.path.abspath(input_dir)) or \
                   not os.path.abspath(target_path).startswith(os.path.abspath(input_dir)):
                    return web.json_response({
                        "success": False,
                        "error": "Invalid file path"
                    })

                if not os.path.exists(source_path):
                    return web.json_response({
                        "success": False,
                        "error": "Source file not found"
                    })

                # 파일 복사
                import shutil
                shutil.copy2(source_path, target_path)

                print(f"[ComfyUI bsk UI] Copied image: {source_filename} -> {target_filename}")

                return web.json_response({
                    "success": True,
                    "filename": target_filename,
                    "message": f"Copied {source_filename} to {target_filename}"
                })

            except Exception as e:
                import traceback
                traceback.print_exc()
                return web.json_response({
                    "success": False,
                    "error": str(e)
                })

        @server.routes.post("/comfyui_panel/get_file_counts")
        async def get_file_counts(request):
            """파일 및 출력 디렉토리의 개수 가져오기"""
            try:
                input_dir = folder_paths.get_input_directory()
                output_dir = folder_paths.get_output_directory()
                image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp'}
                output_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.mp4', '.webm', '.mov', '.avi', '.mkv'}

                input_count = 0
                try:
                    for f in os.listdir(input_dir):
                        file_path = os.path.join(input_dir, f)
                        if os.path.isfile(file_path) and os.path.splitext(f)[1].lower() in image_extensions:
                            input_count += 1
                except Exception:
                    pass

                output_count = 0
                try:
                    for f in os.listdir(output_dir):
                        file_path = os.path.join(output_dir, f)
                        if os.path.isfile(file_path) and os.path.splitext(f)[1].lower() in output_extensions:
                            output_count += 1
                except Exception:
                    pass

                return web.json_response({
                    "success": True,
                    "input_count": input_count,
                    "output_count": output_count
                })
            except Exception as e:
                return web.json_response({
                    "success": False,
                    "error": str(e)
                })

        @server.routes.post("/comfyui_panel/clear_input_files")
        async def clear_input_files(request):
            """입력 디렉토리의 모든 이미지 파일 삭제"""
            try:
                input_dir = folder_paths.get_input_directory()
                image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp'}
                deleted_count = 0
                deleted_files = []
                errors = []

                for f in os.listdir(input_dir):
                    file_path = os.path.join(input_dir, f)
                    if os.path.isfile(file_path):
                        ext = os.path.splitext(f)[1].lower()
                        if ext in image_extensions:
                            try:
                                os.remove(file_path)
                                deleted_count += 1
                                deleted_files.append(f)
                            except Exception as e:
                                errors.append({"file": f, "error": str(e)})

                return web.json_response({
                    "success": True,
                    "deleted_count": deleted_count,
                    "deleted_files": deleted_files,
                    "errors": errors,
                    "message": f"Successfully deleted {deleted_count} file(s) from input directory"
                })

            except Exception as e:
                return web.json_response({
                    "success": False,
                    "error": str(e),
                    "deleted_count": 0
                })

        @server.routes.post("/comfyui_panel/clear_output_files")
        async def clear_output_files(request):
            """출력 디렉토리의 모든 이미지 및 비디오 파일을 삭제 (재귀적으로 하위 디렉토리 포함)）"""
            try:
                output_dir = folder_paths.get_output_directory()
                output_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.mp4', '.webm', '.mov', '.avi', '.mkv'}
                deleted_count = 0
                deleted_files = []
                errors = []

                for root, _dirs, names in os.walk(output_dir):
                    for f in names:
                        ext = os.path.splitext(f)[1].lower()
                        if ext not in output_extensions:
                            continue
                        file_path = os.path.join(root, f)
                        if not os.path.isfile(file_path):
                            continue
                        rel = os.path.relpath(file_path, output_dir).replace(os.sep, '/')
                        try:
                            os.remove(file_path)
                            deleted_count += 1
                            deleted_files.append(rel)
                        except Exception as e:
                            errors.append({"file": rel, "error": str(e)})

                # 데이터베이스에 있는 해당 레코드를 지우고 이미지 라이브러리 시간 표시기를 파일과 동기화합니다.
                try:
                    time_db = _load_output_times()
                    for f in deleted_files:
                        time_db.pop(f, None)
                    _save_output_times(time_db)
                except Exception as _e:
                    print(f"[BSK OutputTimes] 출력 지우기 시 동기화 실패: {_e}")

                return web.json_response({
                    "success": True,
                    "deleted_count": deleted_count,
                    "deleted_files": deleted_files,
                    "errors": errors,
                    "message": f"Successfully deleted {deleted_count} file(s) from output directory"
                })

            except Exception as e:
                return web.json_response({
                    "success": False,
                    "error": str(e),
                    "deleted_count": 0
                })

        @server.routes.post("/comfyui_panel/copy_output_to_input")
        async def copy_output_to_input(request):
            """output 디렉토리의 파일을 input 디렉토리에 복사하고 지정된 파일 이름으로 지정합니다."""
            try:
                import shutil
                data = await request.json()
                source_filename = data.get('source_filename')
                target_filename = data.get('target_filename')
                if not source_filename or not target_filename:
                    return web.json_response({"success": False, "error": "Missing parameters"})

                output_dir = folder_paths.get_output_directory()
                input_dir = folder_paths.get_input_directory()

                src_path = _safe_output_path(output_dir, _out_relkey(source_filename, output_dir))
                target_filename = os.path.basename(target_filename)
                if not src_path or not os.path.exists(src_path):
                    return web.json_response({"success": False, "error": f"File not found in output: {source_filename}"})

                dst_path = os.path.join(input_dir, target_filename)
                shutil.copy2(src_path, dst_path)
                return web.json_response({"success": True, "filename": target_filename})
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        @server.routes.post("/comfyui_panel/add_output_to_asset")
        async def add_output_to_asset(request):
            """output 디렉토리의 이미지/비디오를 자산 라이브러리에 추가（input/whatdreamscost/）。

            파일 처리 전략 (시스템 호환성)）：
              1. os.link 하드 링크를 우선 사용합니다: 동일 볼륨 시 0-copy, 즉각적인 완료, 추가 공간 점유하지 않음(NTFS/ext4/APFS 모두 지원, 관리자 권한 불필요).）；
              2. 하드 링크 실패 (크로스 볼륨 / FAT 등 지원되지 않는 파일 시스템 / 권한 제한)。

            중복 제거: 대상 파일이 존재 → 중복 생성 안 함, 반환 already_exists=true（프론트엔드에 따라 '재료 라이브러리에 있음' 알림」）。
            body: {"filename": "xxx.png"}；반환 {"success": true, "serverPath": "whatdreamscost/xxx.png", "method": "link"|"copy", "already_exists": bool}
            """
            try:
                import shutil as _shutil
                data = await request.json()
                filename = (data.get('filename') or '').strip()
                if not filename:
                    return web.json_response({"success": False, "error": "Missing filename"})

                output_dir = folder_paths.get_output_directory()
                asset_dir = Path(folder_paths.get_input_directory()) / "whatdreamscost"
                asset_dir.mkdir(parents=True, exist_ok=True)

                # 소스 파일은 output 하위 디렉토리 상대 경로 지원
                src_path = _safe_output_path(output_dir, _out_relkey(filename, output_dir))
                if not src_path or not os.path.isfile(src_path):
                    return web.json_response({"success": False, "error": f"File not found in output: {filename}"})

                # 리소스 라이브러리 평면 유지 (list_assets/리소스와 일치), 목표는 파일 이름만 사용
                dst_name = os.path.basename(src_path)
                dst_path = os.path.join(str(asset_dir), dst_name)
                server_path = f"whatdreamscost/{dst_name}"

                if os.path.exists(dst_path):
                    return web.json_response({"success": True, "serverPath": server_path, "method": "exists", "already_exists": True})

                method = "link"
                try:
                    os.link(src_path, dst_path)
                except OSError:
                    # 볼륨별 / 하드 링크 파일 시스템 미지원: 복사 백업, 기능 중단 X
                    _shutil.copy2(src_path, dst_path)
                    method = "copy"
                except Exception:
                    _shutil.copy2(src_path, dst_path)
                    method = "copy"

                if not os.path.exists(dst_path):
                    return web.json_response({"success": False, "error": f"Failed to add {filename} to assets"})

                return web.json_response({"success": True, "serverPath": server_path, "method": method, "already_exists": False})
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        # ============ BSK Video Studio：프롬프트 설정 연관 비디오（user/default/bsk_video_studio/prompt_videos/） ============
        # 「사용 영상 복사하여 prompt_videos/에 저장 = 설정 이름 sanitize + 소스 확장자，
        # 프롬프트 텍스트 설정과 비디오 이름을 동일하게 연결; "비디오 보기" 설정 이름으로 동일 디렉토리의 동일 이름 비디오를 바로 재생。
        @server.routes.post("/comfyui_panel/prompt_videos/save")
        async def save_prompt_video(request):
            """output 폴더의 비디오를 "프롬프트 튜닝 관련 비디오"로 복사」。
            body: {"name": "설정 이름", "filename": "출력 비디오 파일 이름/상대 경로", "overwrite": bool}
            목표 디렉토리：user/default/bsk_video_studio/prompt_videos/<sanitized_name><소스 확장자>。
            동일 이름이 이미 존재하고 추가되지 않음 overwrite → 반환 exists=true（프론트엔드는 이에 따라 덮어쓰기 확인 대화 상자를 띄우고, 확인 후 overwrite를 다시 호출）。
            반환 {"success": true, "filename": "xxx.mp4", "exists": false}。
            """
            try:
                import re as _re
                import shutil as _shutil
                data = await request.json()
                name = (data.get('name') or '').strip()
                filename = (data.get('filename') or '').strip()
                overwrite = bool(data.get('overwrite'))
                if not name or not filename:
                    return web.json_response({"success": False, "error": "Missing name or filename"})
                # sanitize 이름 구성: 경로 구분자와 불법 파일 이름 문자를 제거하여 경로 침투를 방지합니다.
                safe_name = _re.sub(r'[\\/:*?"<>|\x00-\x1f]', '_', name).strip()
                safe_name = _re.sub(r'_+', '_', safe_name).strip(' .')[:120]
                if not safe_name:
                    return web.json_response({"success": False, "error": "Invalid name"})

                output_dir = folder_paths.get_output_directory()
                src_path = _safe_output_path(output_dir, _out_relkey(filename, output_dir))
                if not src_path or not os.path.isfile(src_path):
                    return web.json_response({"success": False, "error": f"File not found in output: {filename}"})
                ext = os.path.splitext(src_path)[1].lower()
                video_exts = {'.mp4', '.webm', '.mov', '.avi', '.mkv', '.gif'}
                if ext not in video_exts:
                    return web.json_response({"success": False, "error": f"Not a video file: {filename}"})

                target_dir = os.path.join(VS_CFG_DIR, "prompt_videos")
                os.makedirs(target_dir, exist_ok=True)
                dst_name = safe_name + ext
                dst_path = os.path.join(target_dir, dst_name)
                if os.path.exists(dst_path) and not overwrite:
                    return web.json_response({"success": True, "filename": dst_name, "exists": True})

                _shutil.copy2(src_path, dst_path)
                return web.json_response({"success": True, "filename": dst_name, "exists": False})
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        @server.routes.get("/comfyui_panel/prompt_videos/view")
        async def view_prompt_video(request):
            """프롬프트 튜닝 설정과 함께 비디오 파일 반환 (프론트엔드에) <video> 직접 링크 재생, HEAD 탐지 지원）。
            query: {"name": "설정 이름"}；prompt_videos/ 폴더 자동 매칭 <sanitized_name>.<비디오 확장자>。
            """
            try:
                import re as _re
                name = (request.query.get('name') or '').strip()
                if not name:
                    return web.json_response({"success": False, "error": "Missing name"}, status=404)
                safe_name = _re.sub(r'[\\/:*?"<>|\x00-\x1f]', '_', name).strip()
                safe_name = _re.sub(r'_+', '_', safe_name).strip(' .')[:120]
                target_dir = os.path.join(VS_CFG_DIR, "prompt_videos")
                video_exts = {'.mp4', '.webm', '.mov', '.avi', '.mkv', '.gif'}
                if os.path.isdir(target_dir):
                    for fn in os.listdir(target_dir):
                        stem, ext = os.path.splitext(fn)
                        if stem == safe_name and ext.lower() in video_exts:
                            full = os.path.join(target_dir, fn)
                            if os.path.isfile(full):
                                return web.FileResponse(full)
                return web.json_response({"success": False, "error": "Video not found"}, status=404)
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)}, status=404)

        @server.routes.post("/comfyui_panel/upload_output")
        async def upload_output_file(request):
            """출력 디렉토리로 파일 업로드"""
            try:
                reader = await request.multipart()
                field = await reader.next()
                
                if field.name != 'image':
                    return web.json_response({"success": False, "error": "Expected 'image' field"})
                
                filename = field.filename
                if not filename:
                    return web.json_response({"success": False, "error": "No filename provided"})
                
                # 보안 검사: 경로 탐색 방지
                filename = os.path.basename(filename)
                
                # 출력 디렉토리 가져오기
                output_dir = folder_paths.get_output_directory()
                
                # 파일 데이터 읽기
                image_data = await field.read()
                
                # 출력 디렉토리로 파일 저장
                output_path = os.path.join(output_dir, filename)
                with open(output_path, 'wb') as f:
                    f.write(image_data)
                
                return web.json_response({
                    "success": True,
                    "name": filename,
                    "path": output_path
                })
                
            except Exception as e:
                print(f"[ComfyUI Panel] Upload to output failed: {e}")
                import traceback
                traceback.print_exc()
                return web.json_response({"success": False, "error": str(e)})

        @server.routes.post("/comfyui_panel/crop_audio")
        async def crop_audio(request):
            """오디오 자르기API
            오디오 파일 이름, 시작 시간, 종료 시간을 수신하여 자르기 및 저장
            """
            try:
                data = await request.json()
                filename = data.get('filename')
                start_time = float(data.get('startTime', 0))
                end_time = float(data.get('endTime', 0))
                output_filename = data.get('outputFilename', 'cropped.mp3')
                
                if not filename:
                    return web.json_response({
                        "success": False,
                        "error": "No filename provided"
                    })
                
                # 입력 디렉토리 가져오기
                input_dir = folder_paths.get_input_directory()
                input_path = os.path.join(input_dir, filename)
                
                # 파일이 input 디렉토리에 없으면 output 디렉토리에서 찾습니다
                if not os.path.exists(input_path):
                    output_dir = folder_paths.get_output_directory()
                    input_path = os.path.join(output_dir, filename)
                
                if not os.path.exists(input_path):
                    return web.json_response({
                        "success": False,
                        "error": f"Audio file not found: {filename}"
                    })
                
                # 건축 출력 경로
                output_path = os.path.join(input_dir, output_filename)
                
                # pydub을 사용하여 오디오 자르기 시도
                try:
                    from pydub import AudioSegment
                    
                    # 오디오 로드
                    audio = AudioSegment.from_file(input_path)
                    
                    # 변환됨 밀리초
                    start_ms = int(start_time * 1000)
                    end_ms = int(end_time * 1000)
                    
                    # 자르기
                    cropped = audio[start_ms:end_ms]
                    
                    # 내보내기mp3
                    cropped.export(output_path, format='mp3')
                    
                    return web.json_response({
                        "success": True,
                        "filename": output_filename,
                        "startTime": start_time,
                        "endTime": end_time,
                        "duration": end_time - start_time
                    })
                    
                except ImportError:
                    # pydub가 없으면 ffmpeg를 사용하여 직접 처리합니다
                    import subprocess
                    
                    duration = end_time - start_time
                    # 보안 검사: 유효한 경로 검증
                    for unsafe_path in (input_path, output_path):
                        if not os.path.isfile(unsafe_path):
                            return web.json_response({
                                "success": False,
                                "error": f"경로가 존재하지 않음: {unsafe_path}"
                            })
                    cmd = [
                        'ffmpeg', '-y', '-i', input_path,
                        '-ss', str(start_time), '-t', str(duration),
                        '-c:a', 'libmp3lame', '-q:a', '2',
                        output_path
                    ]
                    
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    
                    if result.returncode != 0:
                        return web.json_response({
                            "success": False,
                            "error": f"FFmpeg failed: {result.stderr}"
                        })
                    
                    return web.json_response({
                        "success": True,
                        "filename": output_filename,
                        "startTime": start_time,
                        "endTime": end_time,
                        "duration": duration
                    })
                    
            except Exception as e:
                import traceback
                traceback.print_exc()
                return web.json_response({
                    "success": False,
                    "error": str(e)
                })
        
        @server.routes.get("/comfyui_panel/list_input_files")
        async def list_input_files(request):
            """input 디렉토리의 파일 목록 가져오기"""
            try:
                import folder_paths
                import os
                
                input_dir = folder_paths.get_input_directory()
                files = []
                
                if os.path.exists(input_dir):
                    for f in os.listdir(input_dir):
                        file_path = os.path.join(input_dir, f)
                        if os.path.isfile(file_path):
                            files.append(f)
                
                return web.json_response({
                    "success": True,
                    "files": files,
                    "folder": input_dir
                })
            except Exception as e:
                import traceback
                traceback.print_exc()
                return web.json_response({
                    "success": False,
                    "error": str(e)
                })
        
        @server.routes.get("/comfyui_panel/input_files_by_date")
        async def input_files_by_date(request):
            """입력 디렉토리의 이미지 파일 목록을 수정 날짜별로 정렬합니다(최신 파일 우선)）"""
            try:
                import folder_paths
                import os
                
                input_dir = folder_paths.get_input_directory()
                files = []
                
                if os.path.exists(input_dir):
                    for f in os.listdir(input_dir):
                        file_path = os.path.join(input_dir, f)
                        if os.path.isfile(file_path):
                            # 이미지 파일만 포함
                            ext = os.path.splitext(f)[1].lower()
                            if ext in ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif', '.tiff', '.tif'):
                                mtime = os.path.getmtime(file_path)
                                files.append({"name": f, "mtime": mtime})
                
                # 수정 날짜 순으로 내림차순 정렬 (최신 우선)）
                files.sort(key=lambda x: x["mtime"], reverse=True)
                
                return web.json_response({
                    "success": True,
                    "files": files,
                    "folder": input_dir
                })
            except Exception as e:
                import traceback
                traceback.print_exc()
                return web.json_response({
                    "success": False,
                    "error": str(e)
                })
        
        @server.routes.get("/comfyui_panel/test")
        async def test_endpoint(request):
            return web.json_response({
                "status": "ok",
                "message": "ComfyUI Panel API is working",
                "version": VERSION
            })

        @server.routes.get("/comfyui_panel/default_avatar")
        async def get_default_avatar(request):
            """반환 기본 아이콘 이미지"""
            try:
                avatar_path = EXTENSION_DIR / "assets" / "default-avatar.png"
                # print(f"[ComfyUI Panel] Looking for default avatar at: {avatar_path}")
                # print(f"[ComfyUI Panel] File exists: {avatar_path.exists()}")
                if avatar_path.exists():
                    with open(avatar_path, 'rb') as f:
                        image_data = f.read()
                    response = web.Response(body=image_data, content_type='image/png')
                    response.headers['Cache-Control'] = 'public, max-age=86400'
                    return response
                else:
                    print(f"[ComfyUI Panel] Default avatar file not found: {avatar_path}")
                    return web.json_response({"success": False, "error": "Default avatar not found"}, status=404)
            except Exception as e:
                print(f"[ComfyUI Panel] Error serving default avatar: {e}")
                import traceback
                traceback.print_exc()
                return web.json_response({"success": False, "error": str(e)}, status=500)

        @server.routes.get("/comfyui_panel/list_models")
        async def list_models(request):
            """지정된 유형의 모델 파일 목록 가져오기 (서브 디렉토리 지원)）"""
            try:
                # 쿼리 파라미터에서 모델 유형 가져옴, 기본값은 checkpoints
                model_type = request.query.get("type", "checkpoints")
                # 모델 유형이 folder_paths의 폴더 이름에 매핑되는지 확인 (미치지 못하면 되돌아감) checkpoints）
                type_map = {
                    "checkpoints": "checkpoints",
                    "loras": "loras",
                    "vae": "vae",
                    "embeddings": "embeddings",
                    "hypernetworks": "hypernetworks",
                    "controlnet": "controlnet",
                    "upscale_models": "upscale_models",
                    "text_encoders": "text_encoders",
                    "diffusion_models": "diffusion_models",
                    "clip_vision": "clip_vision",
                    "style_models": "style_models",
                    "gligen": "gligen",
                    "photomaker": "photomaker",
                    "model_patches": "model_patches",
                    "diffusers": "diffusers",
                }
                folder = type_map.get(model_type, "checkpoints")

                # ComfyUI 원본의 get_filename_list를 사용하여 모델 목록 가져오기
                # 이것은 /object_info API 사용과 동일한 데이터 소스를 사용하며, symlink 등의 상황을 올바르게 처리합니다.
                files = []
                filenames = folder_paths.get_filename_list(folder)
                for rel_path in filenames:
                    files.append({
                        "name": os.path.basename(rel_path),
                        "path": rel_path,  # 상대 경로 (포함된 하위 디렉토리)）
                    })

                files.sort(key=lambda x: x['path'].lower())
                return web.json_response({
                    "success": True,
                    "files": files,
                    "count": len(files),
                    "type": model_type
                })
            except Exception as e:
                import traceback
                traceback.print_exc()
                return web.json_response({
                    "success": False,
                    "error": str(e),
                    "files": []
                })

        # ===== 로컬 이미지 파일 자동 동기화 (가드 프로세스 ↔ 프론트엔드）=====
        # 매핑 테이블：node_id(str) -> 로컬 절대 경로 (문자열), 플러그인 디렉터리 저장
        _watch_map_path = os.path.join(script_directory, "bsk_image_watch.json")

        def _read_watch_map():
            try:
                if os.path.exists(_watch_map_path):
                    with open(_watch_map_path, 'r', encoding='utf-8') as f:
                        return json.load(f)
            except Exception:
                pass
            return {}

        def _write_watch_map(m):
            try:
                with open(_watch_map_path, 'w', encoding='utf-8') as f:
                    json.dump(m, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"[ComfyUI bsk UI] bsk_image_watch.json 쓰기 실패: {e}")

        @server.routes.post("/comfyui_panel/set_watch_map")
        async def set_watch_map(request):
            """프론트엔드 노드 동기화 표시/해제：{node_id, unbind?}
            라벨 동기화: 저장 {node_id: {remote_name}}，remote_name 기본 bsk_source_<id>.png
            취소 동기화：unbind=true 시 항목_id를 삭제합니다
            （로컬 경로를 보호 프로그램이 로컬에서 관리하며, 브라우저 측에서는 더 이상 제공되지 않음）
            """
            try:
                data = await request.json()
                node_id = str(data.get('node_id', '')).strip()
                if not node_id:
                    return web.json_response({"success": False, "error": "node_id 비어 있음"})
                m = _read_watch_map()
                if data.get('unbind'):
                    m.pop(node_id, None)
                else:
                    remote_name = (data.get('remote_name') or '').strip() or ("bsk_source_%s.png" % node_id)
                    m[node_id] = {"remote_name": remote_name}
                _write_watch_map(m)
                return web.json_response({"success": True, "map": m})
            except Exception as e:
                import traceback
                traceback.print_exc()
                return web.json_response({"success": False, "error": str(e)})

        @server.routes.get("/comfyui_panel/get_watch_map")
        async def get_watch_map(request):
            """가드 프로세스가 매핑을 가져옴 (로컬/클라우드 모두 HTTP를 통해 실행, 서버 파일 읽지 않음)）"""
            try:
                return web.json_response({"success": True, "map": _read_watch_map()})
            except Exception as e:
                return web.json_response({"success": False, "error": str(e), "map": {}})

        @server.routes.post("/comfyui_panel/notify_image_update")
        async def notify_image_update(request):
            """프로그램이 이미지를 업로드한 후 호출: 해당 이미지 로드 카드에 프런트엔드를 갱신합니다
            요청 본문：{node_id, filename}
            """
            try:
                data = await request.json()
                node_id = str(data.get('node_id', '')).strip()
                filename = (data.get('filename') or '').strip()
                if not node_id or not filename:
                    return web.json_response({"success": False, "error": "node_id 또는 파일 이름이 비어 있음"})
                if PromptServer is not None:
                    PromptServer.instance.send_sync('bsk/image_updated', {
                        'nodeId': node_id,
                        'filename': filename,
                    })
                return web.json_response({"success": True})
            except Exception as e:
                import traceback
                traceback.print_exc()
                return web.json_response({"success": False, "error": str(e)})

        @server.routes.get("/comfyui_panel/get_node_image")
        async def get_node_image(request):
            """프로세스 다운로드 노드 현재 템플릿 다운로드：?node_id=X
            input 디렉토리에서 읽음 bsk_source_<X>.png PNG 바이너리 반환; 존재하지 않으면 반환 404
            """
            try:
                node_id = (request.query.get('node_id') or '').strip()
                if not node_id:
                    return web.json_response({"success": False, "error": "node_id 비어 있음"})
                input_dir = folder_paths.get_input_directory()
                filename = "bsk_source_%s.png" % node_id
                path = os.path.join(input_dir, filename)
                if not os.path.exists(path):
                    return web.json_response(
                        {"success": False, "error": "노드 %s는 현재 이미지 없음 (먼저 카드에 이미지를 업로드하세요)）" % node_id},
                        status=404,
                    )
                return web.FileResponse(path, headers={"Content-Type": "image/png"})
            except Exception as e:
                import traceback
                traceback.print_exc()
                return web.json_response({"success": False, "error": str(e)})

        # ===== 플러그인 업데이트 API =====

        def _normalize_plugin_name(name):
            """플러그인 폴더 이름 표준화: GitHub 다운로드 압축 파일에서 흔히 볼 수 있는 -main / -master 접미사를 제거합니다 (대소문자를 구분하지 않음)）。

            예：ComfyUI-Foo-main -> ComfyUI-Foo；ComfyUI-Foo-MAIN -> ComfyUI-Foo
            """
            name = (name or "").strip()
            lowered = name.lower()
            for suffix in ('-main', '-master'):
                if lowered.endswith(suffix) and len(name) > len(suffix):
                    return name[: -len(suffix)]
            return name

        @server.routes.post("/bsk_ui/upload_plugin")
        async def upload_plugin_zip(request):
            """플러그인 zip 파일을 업로드하고, 백엔드에서 압축 해제 및 분석。

            두 가지 호출 호환：
            1) 본 플러그인 자체 업데이트 패키지 (루트 이름) ComfyUI-bsk_UI）-> 패널/다른 두 개의 칼럼 구조 반환（type=self_update）
            2) 일반 ComfyUI 플러그인 설치 패키지(다음 내용 포함) __init__.py）-> 플러그인 이름, 파일 목록, 설치 여부 반환（type=plugin）

            POST multipart: file=zip파일
            """
            try:
                reader = await request.multipart()
                zip_data = None
                zip_filename = None

                async for field in reader:
                    if field.name == 'file':
                        zip_data = await field.read()
                        zip_filename = getattr(field, 'filename', None)
                        break

                if not zip_data:
                    return web.json_response({"success": False, "error": "zip 파일이 수신되지 않음"})

                # 보안 검사: 압축 경로가 플러그인 디렉토리 내에 있어야 함
                plugin_dir = script_directory  # ComfyUI-bsk_UI/
                temp_dir = os.path.join(plugin_dir, ".update_temp")

                # 삭제 오래된 임시 폴더
                if os.path.exists(temp_dir):
                    import shutil
                    shutil.rmtree(temp_dir)
                os.makedirs(temp_dir, exist_ok=True)

                # 압축 해제zip
                import zipfile
                import io

                def _fix_zip_name(info):
                    """zip 파일 이름 디코딩 수정 (중국어 등 비 ASCII 파일 이름 인코딩 문제 해결)）。

                    우선순위：
                    1) UTF-8 플래그 설정(비트 11): Python이 올바르게 디코딩되었으며, 즉시 반환.。
                    2) 로컬 ONNX 이미지 태깅 노드, CLTagger 및 WD14 두 가지 태깅기를 지원.
                       filename.encode('cp437') 원래 바이트를 복원한 후 utf-8 / gb18030로 재해석。
                       파일명에 cp437으로 표현할 수 없는 문자(Python 실제 정확한 텍스트 해독된 경우)，
                       필드 이름/디렉토리 이름 소문자 숫자 문자열로 변환 (밑줄, 하이픈, 공백 등 제거) filename。
                    """
                    if info.flag_bits & 0x800:
                        return info.filename
                    try:
                        raw = info.filename.encode('cp437')
                    except UnicodeEncodeError:
                        return info.filename
                    for enc in ('utf-8', 'gb18030'):
                        try:
                            return raw.decode(enc)
                        except UnicodeDecodeError:
                            continue
                    return info.filename

                with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
                    # 파일명 인코딩 수정 (중국어 등 비 ASCII 파일명 깨짐 문제 해결)）
                    for _info in zf.infolist():
                        _info.filename = _fix_zip_name(_info)
                    # 안전 검사: 경로 탐색 공격 방지
                    for info in zf.infolist():
                        member_path = os.path.normpath(info.filename)
                        # 확인 중인 디렉터리에 플러그인이 있는지 확인
                        target_path = os.path.join(plugin_dir, member_path)
                        real_target = os.path.realpath(target_path)
                        real_plugin_dir = os.path.realpath(plugin_dir)
                        if not real_target.startswith(real_plugin_dir + os.sep) and real_target != real_plugin_dir:
                            print(f"[PluginUpdater] 안전하지 않은 경로를 건너뜁니다: {info.filename}")
                            continue
                        zf.extract(info, temp_dir)

                # 압축된 디렉토리 구조 분석, 실제 내용 루트 디렉토리 찾기
                # zip내부에 ComfyUI-bsk_UI/와 같은 루트 디렉토리가 있을 수 있습니다 ComfyUI-bsk_UI-main/
                temp_contents = os.listdir(temp_dir)
                root_dir = None
                for entry in temp_contents:
                    full = os.path.join(temp_dir, entry)
                    if os.path.isdir(full):
                        # 이 디렉토리에 플러그인 특징 파일이 포함되어 있는지 확인
                        sub_entries = os.listdir(full)
                        if any(e.lower().endswith('.py') for e in sub_entries):
                            root_dir = entry
                            break

                if root_dir:
                    # zip서버 자재 목록 가져오기 실패
                    actual_dir = os.path.join(temp_dir, root_dir)
                else:
                    # zip직접 파일, 사용temp_dir
                    actual_dir = temp_dir

                # ===== 상황 1: 플러그인 자체 업데이트 패키지 (루트 이름은 반드시 ~이어야 함) ComfyUI-bsk_UI）=====
                root_dir_name = ""
                if root_dir:
                    root_dir_name = root_dir
                elif not any(os.path.isfile(os.path.join(temp_dir, f)) for f in os.listdir(temp_dir)):
                    # 모두 디렉토리, 첫 번째 선택
                    for entry in os.listdir(temp_dir):
                        full = os.path.join(temp_dir, entry)
                        if os.path.isdir(full):
                            root_dir_name = entry
                            break

                if root_dir_name == "ComfyUI-bsk_UI":
                    # 선택할 파일 목록을 스캔합니다
                    preview_files = []
                    for dirpath, dirnames, filenames in os.walk(actual_dir):
                        for fname in filenames:
                            abs_path = os.path.join(dirpath, fname)
                            rel_path = os.path.relpath(abs_path, actual_dir)
                            norm_rel = rel_path.replace('\\', '/')
                            preview_files.append({
                                "relPath": norm_rel,
                                "size": os.path.getsize(abs_path),
                                "dir": os.path.dirname(norm_rel) or ".",
                                "_absPath": abs_path
                            })

                    # 디렉토리별 그룹화: 패널 디렉토리 및 하위 디렉토리를 묶음
                    panel_files = []
                    other_files = []
                    for f in preview_files:
                        d = f["dir"]
                        if d == "panel" or d.startswith("panel/"):
                            panel_files.append(f)
                        else:
                            other_files.append(f)

                    # 각 파일이 이미 존재하는지 판단
                    for f in panel_files:
                        f["exists"] = os.path.exists(os.path.join(plugin_dir, f["relPath"]))

                    return web.json_response({
                        "success": True,
                        "type": "self_update",
                        "previewFiles": {
                            "panel": panel_files,
                            "other": other_files
                        },
                        "hasPanelFiles": len(panel_files) > 0,
                        "totalFiles": len(preview_files),
                        "tempDir": temp_dir
                    })

                # ===== 상황 2: 일반 ComfyUI 플러그인 설치 패키지 =====
                # __init__.py가 포함된 디렉터리(즉, 플러그인 루트 디렉터리, actual_dir보다 한 층 깊을 수 있음)를 찾습니다.）
                plugin_root = None
                for dirpath, dirnames, filenames in os.walk(actual_dir):
                    if '__init__.py' in filenames:
                        plugin_root = dirpath
                        break

                if not plugin_root:
                    import shutil
                    shutil.rmtree(temp_dir)
                    return web.json_response({
                        "success": False,
                        "error": "ComfyUI 플러그인 또는 본 플러그인 업데이트 패키지로 인식되지 않음: 압축 파일 내에 __init__.py가 없습니다. 유효한 플러그인 압축 파일인지 확인하십시오.。"
                    })

                # 계산 플러그인 이름을 계산합니다(직접 __init__.py를 포함하는 폴더 이름; __init__.py가 루트에 있는 경우 압축 파일 이름으로 되돌아갑니다).）
                if plugin_root == temp_dir:
                    raw_name = os.path.splitext(zip_filename or 'plugin')[0]
                else:
                    raw_name = os.path.basename(plugin_root)
                raw_plugin_name = raw_name
                normalized_name = _normalize_plugin_name(raw_name)

                # 대소문자 구별 없이 중복 제거: custom_nodes 동일 이름(단순히 대소문자만 다름) 폴더가 이미 존재하면 덮어쓰고, 중복 설치 방지
                custom_nodes_dir = folder_paths.get_folder_paths("custom_nodes")
                if not custom_nodes_dir:
                    custom_nodes_dir = os.path.join(os.path.dirname(folder_paths.__file__), "custom_nodes")
                else:
                    custom_nodes_dir = custom_nodes_dir[0]

                matched_existing = None
                if os.path.isdir(custom_nodes_dir):
                    for entry in os.listdir(custom_nodes_dir):
                        full = os.path.join(custom_nodes_dir, entry)
                        if os.path.isdir(full) and entry.lower() == normalized_name.lower():
                            matched_existing = entry
                            break

                final_plugin_name = matched_existing if matched_existing else normalized_name
                target_plugin_dir = os.path.join(custom_nodes_dir, final_plugin_name)
                already_installed = os.path.isdir(target_plugin_dir)

                # 스캔 플러그인 파일 목록 (상대) plugin_root）
                file_list = []
                for dirpath, dirnames, filenames in os.walk(plugin_root):
                    for fname in filenames:
                        abs_path = os.path.join(dirpath, fname)
                        rel_path = os.path.relpath(abs_path, plugin_root)
                        norm_rel = rel_path.replace('\\', '/')
                        file_list.append({
                            "relPath": norm_rel,
                            "size": os.path.getsize(abs_path),
                            "exists": os.path.exists(os.path.join(target_plugin_dir, norm_rel)),
                            "_absPath": abs_path
                        })

                return web.json_response({
                    "success": True,
                    "type": "plugin",
                    "pluginName": final_plugin_name,
                    "rawPluginName": raw_plugin_name,
                    "matchedExisting": matched_existing,
                    "alreadyInstalled": already_installed,
                    "tempDir": temp_dir,
                    "fileList": file_list,
                    "totalFiles": len(file_list)
                })

            except zipfile.BadZipFile:
                return web.json_response({"success": False, "error": "유효하지 않은 zip 파일"})
            except Exception as e:
                import traceback
                traceback.print_exc()
                return web.json_response({"success": False, "error": str(e)})

        @server.routes.post("/bsk_ui/cleanup_temp")
        async def cleanup_temp(request):
            """임시 업데이트 폴더 정리"""
            try:
                data = await request.json()
                temp_dir = data.get("tempDir", "")
                if temp_dir and os.path.exists(temp_dir):
                    import shutil
                    shutil.rmtree(temp_dir)
                return web.json_response({"success": True})
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        @server.routes.post("/bsk_ui/confirm_update")
        async def confirm_update(request):
            """덮어쓰기 작업 확인, 파일 복사 실행
            POST json: {selectedFiles: [{relPath, _absPath?}], tempDir}
            selectedFiles 파일 덮어쓰기 경로 목록 (사용자 선택)）
            """
            try:
                data = await request.json()
                selected_files = data.get("selectedFiles", [])
                temp_dir = data.get("tempDir", "")

                if not selected_files or not temp_dir:
                    return web.json_response({"success": False, "error": "필요한 매개변수가 없습니다"})

                plugin_dir = script_directory
                copied = 0
                skipped = 0
                errors = []
                deleted_temp = 0

                # === 잔여 _temp.json 임시 구성물 정리 ===
                # 규칙: 이번에 덮어쓰는 패널 PNG, 압축 파일 내에 해당되는 것이 없는 경우 <name>_temp.json，
                # 대상 디렉터리【에】_temp.json이 있으면 먼저 panel/temp/으로 백업한 다음 삭제합니다(중복 시 자동 번호 추가).，
                # 남아있는 _temp.json 압축 PNG 내장 설정으로 인한 업데이트된 패널 설정 읽기 오류 방지。
                cleanup_temp_configs = bool(data.get("cleanupTempConfigs", True))
                backed_up_temp = 0

                for item in selected_files:
                    rel_path = item["relPath"] if isinstance(item, dict) else item
                    # _absPath (압축된 원본 파일에 직접 가리키는)를 우선 사용합니다）
                    abs_src = item.get("_absPath") if isinstance(item, dict) else None
                    if not abs_src:
                        # 되돌리기: ~부터 temp_dir + rel_path 계산 (오래된 프런트엔드 호환)）
                        # 먼저 zip 내용의 루트 디렉터리 찾기 필요（root_dir）
                        actual_dir = None
                        temp_contents = os.listdir(temp_dir)
                        for entry in temp_contents:
                            full = os.path.join(temp_dir, entry)
                            if os.path.isdir(full):
                                sub_entries = os.listdir(full)
                                if any(e.lower().endswith('.py') for e in sub_entries):
                                    actual_dir = os.path.join(temp_dir, entry)
                                    break
                        if actual_dir:
                            src = os.path.join(actual_dir, rel_path.replace('/', os.sep))
                        else:
                            src = os.path.join(temp_dir, rel_path.replace('/', os.sep))
                    else:
                        src = abs_src
                    dst = os.path.join(plugin_dir, rel_path.replace('/', os.sep))

                    try:
                        # 대상 디렉토리가 존재하는지 확인
                        dst_dir = os.path.dirname(dst)
                        if dst_dir and not os.path.exists(dst_dir):
                            os.makedirs(dst_dir, exist_ok=True)

                        # 파일 복사
                        import shutil
                        shutil.copy2(src, dst)
                        copied += 1
                    except Exception as e:
                        errors.append(f"{rel_path}: {str(e)}")
                        skipped += 1

                # === _temp.json 백업 및 정리: zip에는 _temp.json이 없지만 대상 디렉토리에 있습니다. ===
                if cleanup_temp_configs:
                    import shutil
                    # zip 내용의 루트 디렉터리 위치 지정 (위의 롤백 로직과 일치)）
                    actual_dir = None
                    try:
                        for entry in os.listdir(temp_dir):
                            full = os.path.join(temp_dir, entry)
                            if os.path.isdir(full) and any(e.lower().endswith('.py') for e in os.listdir(full)):
                                actual_dir = os.path.join(temp_dir, entry)
                                break
                    except Exception:
                        actual_dir = None
                    if not actual_dir:
                        actual_dir = temp_dir
                    zip_panel_root = os.path.join(actual_dir, "panel")
                    backup_dir = os.path.join(plugin_dir, "panel", "temp")

                    for item in selected_files:
                        rel = item["relPath"] if isinstance(item, dict) else item
                        if not (rel.startswith("panel/") and rel.lower().endswith(".png")):
                            continue
                        rel_under = rel[len("panel/"):]
                        # 압축 파일 내에 동일한 _temp.json 파일 포함 여부: 포함 시 압축 파일 우선, 대상 무시）
                        zip_temp_json = os.path.join(
                            zip_panel_root,
                            (os.path.splitext(rel_under)[0] + '_temp.json').replace('/', os.sep)
                        )
                        if os.path.isfile(zip_temp_json):
                            continue
                        # 대상 디렉토리의 동일 이름 _temp.json
                        dst_png = os.path.join(plugin_dir, rel.replace('/', os.sep))
                        temp_json = os.path.splitext(dst_png)[0] + '_temp.json'
                        if not os.path.isfile(temp_json):
                            continue
                        try:
                            os.makedirs(backup_dir, exist_ok=True)
                            base_name = os.path.basename(temp_json)  # 예: Anima_전체 기능_temp.json
                            # 순번을 _temp.json 앞 추가 (예: Anima_전체 기능)_1_temp.json），
                            # _temp.json 파일 확장명 변경 없이 유지, 패널 설정 스캔 필터링에서 백업 파일 제외
                            name_prefix = base_name[:-len('_temp.json')]
                            backup_path = os.path.join(backup_dir, base_name)
                            seq = 1
                            while os.path.exists(backup_path):
                                backup_path = os.path.join(backup_dir, f"{name_prefix}_{seq}_temp.json")
                                seq += 1
                            shutil.copy2(temp_json, backup_path)
                            backed_up_temp += 1
                            os.remove(temp_json)
                            deleted_temp += 1
                            print(f"[PluginUpdater] 백업 및 임시 구성 잔류 정리: {temp_json} -> {backup_path}")
                        except Exception as _e:
                            print(f"[PluginUpdater] 백업/임시 설정 삭제 실패 {temp_json}: {_e}")

                # 업데이트 완료: 플러그인 디렉토리 동기화 마커를 0으로 설정하고 다음 재부팅 시 공장 설정 복사 재시도
                try:
                    with open(os.path.join(plugin_dir, "panel_sync_flag.json"), "w", encoding="utf-8") as _f:
                        json.dump({"copied": 0}, _f, ensure_ascii=False, indent=2)
                except Exception as _e:
                    print(f"[PluginUpdater] 재설정 동기화 표시 오류: {_e}")

                # 정리 임시 폴더
                try:
                    import shutil
                    if os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir)
                except:
                    pass

                return web.json_response({
                    "success": True,
                    "copied": copied,
                    "skipped": skipped,
                    "deletedTemp": deleted_temp,
                    "backedUpTemp": backed_up_temp,
                    "errors": errors[:10]  # 최대 반환 오류 개수 10
                })

            except Exception as e:
                import traceback
                traceback.print_exc()
                return web.json_response({"success": False, "error": str(e)})

        @server.routes.post("/bsk_ui/confirm_install_plugin")
        async def confirm_install_plugin(request):
            """일반 플러그인 설치 확인: 임시 디렉토리의 파일을 복사합니다. custom_nodes/<pluginName>/。

            POST json: {selectedFiles:[{relPath,_absPath}], tempDir, pluginName, overwrite}
            - 이미 동일 이름의 플러그인이 존재합니다 overwrite=false -> 오류
            - overwrite=true -> 먼저 이전 폴더 삭제 후 전체 복사(깨끗한 설치, git clone 실패한 빈/손상된 폴더 정리)）
            """
            try:
                import shutil
                data = await request.json()
                selected_files = data.get("selectedFiles", [])
                temp_dir = data.get("tempDir", "")
                plugin_name = (data.get("pluginName") or "").strip()
                overwrite = bool(data.get("overwrite", False))

                if not selected_files or not temp_dir or not plugin_name:
                    return web.json_response({"success": False, "error": "필요한 매개변수가 없습니다"})

                # 안전 검사 플러그인 이름 (set_plugin_enabled과 일치)）
                if "/" in plugin_name or "\\" in plugin_name or ".." in plugin_name or plugin_name in ('.disabled', '__pycache__'):
                    return web.json_response({"success": False, "error": "유효하지 않은 플러그인 이름"})

                custom_nodes_dir = folder_paths.get_folder_paths("custom_nodes")
                if not custom_nodes_dir:
                    custom_nodes_dir = os.path.join(os.path.dirname(folder_paths.__file__), "custom_nodes")
                else:
                    custom_nodes_dir = custom_nodes_dir[0]

                if not os.path.isdir(custom_nodes_dir):
                    return web.json_response({"success": False, "error": "custom_nodes 폴더가 없습니다"})

                target_plugin_dir = os.path.join(custom_nodes_dir, plugin_name)

                if os.path.isdir(target_plugin_dir):
                    if not overwrite:
                        return web.json_response({
                            "success": False,
                            "error": "custom_nodes 중복된 플러그인 이미 존재 '{}'，덮어쓰기 설치를 선택하십시오。".format(plugin_name)
                        })
                    # 덮어 쓰기: 기존 디렉토리 삭제 후 깨끗하게 설치 (빈/손상된 잔여 파일 정리)）
                    shutil.rmtree(target_plugin_dir)

                os.makedirs(target_plugin_dir, exist_ok=True)

                copied = 0
                skipped = 0
                errors = []
                real_temp = os.path.realpath(temp_dir)

                for item in selected_files:
                    rel_path = item["relPath"] if isinstance(item, dict) else item
                    abs_src = item.get("_absPath") if isinstance(item, dict) else None
                    if not abs_src or not os.path.isfile(abs_src):
                        # 롤백: temp_dir 에서 rel_path 를 사용하여 찾음
                        abs_src = os.path.join(temp_dir, rel_path.replace('/', os.sep))
                    if not os.path.isfile(abs_src):
                        skipped += 1
                        errors.append("소스 파일 누락: " + rel_path)
                        continue
                    # 보안 검사: 원 경로가 임시 디렉토리 내에 있어야 하며, 경로 침투 방지
                    real_src = os.path.realpath(abs_src)
                    if not real_src.startswith(real_temp + os.sep):
                        skipped += 1
                        errors.append("유효하지 않은 소스 경로: " + rel_path)
                        continue
                    dst = os.path.join(target_plugin_dir, rel_path.replace('/', os.sep))
                    try:
                        dst_dir = os.path.dirname(dst)
                        if dst_dir and not os.path.exists(dst_dir):
                            os.makedirs(dst_dir, exist_ok=True)
                        shutil.copy2(abs_src, dst)
                        copied += 1
                    except Exception as e:
                        errors.append(f"{rel_path}: {str(e)}")
                        skipped += 1

                # 정리 임시 폴더
                try:
                    if os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir)
                except Exception:
                    pass

                return web.json_response({
                    "success": True,
                    "copied": copied,
                    "skipped": skipped,
                    "errors": errors[:10]
                })

            except Exception as e:
                import traceback
                traceback.print_exc()
                return web.json_response({"success": False, "error": str(e)})

        @server.routes.post("/comfyui_panel/check_workflow_resources")
        async def check_workflow_resources(request):
            """워크플로우에서 참조하는 모델 및 이미지가 존재하는지 확인"""
            try:
                data = await request.json()
                nodes = data.get("nodes", {})
                
                # 이미지 확장자
                image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.tiff', '.tif', '.avif'}
                # 미디어 확장자: 값에 이러한 접미사가 포함된 경우 이미지/오디오 비디오 채널(missingImages)로 분류되며, 누락된 모델로 간주되지 않습니다.
                media_extensions = image_extensions | {'.mp4', '.webm', '.mov', '.mkv', '.avi', '.wav', '.mp3', '.flac', '.ogg', '.m4a'}
                
                # 모델이 없습니다: INPUT_TYPES 노드 기반 combo 옵션 ↔ folder_paths 메뉴 목록 정보 확인
                # （노드 드롭다운 목록과 동일 출처, 빈 필드 이름 추론; 인식 가능. ModelPatchLoader.name→model_patches 등 일반명）
                missing_models = []  # [{nodeId, classType, inputKey, modelName, modelType, modelDirs}]
                missing_images = []  # [{nodeId, classType, inputKey, imageName}]
                auto_load_images = []  # [{nodeId, classType, inputKey, imageName, suggestedImage}]

                # 크로스 플랫폼 경로 정규화 키: Linux 저장 워크플로우 서브 디렉토리 사용 "/"，Windows 의 folder_paths
                # 목록 사용 "\"。단순히 구분 기호만 다른 동일 파일은 이미 존재된 것으로 간주하며, 그렇지 않으면 '모델 누락'으로 오탐하고 다운로드를 유도합니다.。
                import re as _re_path

                def _canon_path_key(p):
                    return _re_path.sub(r"/+", "/", str(p or "").replace("\\", "/")).strip("/")
                
                # 입력 디렉토리의 이미지 목록(이미지만, mtime 기준으로 역순 정렬, 최신 파일이 앞단)）
                input_images = None
                def get_input_images():
                    nonlocal input_images
                    if input_images is not None:
                        return input_images
                    input_images = []
                    try:
                        input_dir = folder_paths.get_input_directory()
                        if os.path.isdir(input_dir):
                            entries = []
                            for f in os.listdir(input_dir):
                                full = os.path.join(input_dir, f)
                                if not os.path.isfile(full):
                                    continue
                                ext = os.path.splitext(f)[1].lower()
                                if ext not in image_extensions:
                                    continue
                                # 절단 자리 표시 제외
                                if f.lower().startswith('bsk_crop_'):
                                    continue
                                try:
                                    mtime = os.path.getmtime(full)
                                except OSError:
                                    mtime = 0
                                entries.append((mtime, f))
                            entries.sort(key=lambda x: x[0], reverse=True)
                            input_images = [f for _, f in entries]
                    except Exception:
                        pass
                    return input_images
                
                for node_id, node_info in nodes.items():
                    if not isinstance(node_info, dict):
                        continue
                    class_type = node_info.get("class_type", "")
                    inputs = node_info.get("inputs", {})
                    
                    for input_key, input_value in inputs.items():
                        if not isinstance(input_value, str) or not input_value or input_value == 'None':
                            continue
                        # Skip connection (array format)）
                        if isinstance(input_value, list):
                            continue
                        
                        # ---- 모델 누락: combo 옵션 ↔ 메뉴 목록 정보 확인 ----
                        model_dirs = _get_combo_model_dirs(class_type, input_key)
                        if model_dirs:
                            combo_options = _get_node_combo_options(class_type, input_key)
                            if combo_options:
                                # 일반: 옵션이 비어 있지 않은 경우, 참조 값이 옵션 목록에 없습니다 → 미스
                                missing = (input_value not in combo_options and
                                           _canon_path_key(input_value) not in {_canon_path_key(o) for o in combo_options})
                            else:
                                # models 디렉토리가 비어 있습니다 (옵션이 비어 있음, 디렉토리는 필드 이름 보수 일치 매칭 위치)）：
                                # 참조 값이 어떤 히트 디렉토리 파일 목록에도 없습니다 → 미스
                                canon_files = set()
                                for _d in model_dirs:
                                    try:
                                        canon_files.update(_canon_path_key(f) for f in folder_paths.get_filename_list(_d))
                                    except Exception:
                                        pass
                                missing = _canon_path_key(input_value) not in canon_files
                            if missing:
                                # 값에 미디어 확장자 포함 → 이미지/오디오/비디오 채널 소속, 누락 모델로 간주하지 않음
                                ext = os.path.splitext(input_value)[1].lower()
                                if ext not in media_extensions:
                                    missing_models.append({
                                        "nodeId": node_id,
                                        "classType": class_type,
                                        "inputKey": input_key,
                                        "modelName": input_value,
                                        "modelType": model_dirs[0],
                                        "modelDirs": model_dirs
                                    })

                        # 이미지 파일 확인 (image 유형의 입력）
                        kl = input_key.lower()
                        if kl == 'image' or (kl.startswith('layer_') and kl.endswith('_image')) or \
                           (kl.startswith('layer_') and kl[6:7].isdigit() and '_image' not in kl and 'name' not in kl and 'strength' not in kl and 'blend' not in kl):
                            # 이미지 파일 이름인지 확인(확장자 포함)）
                            ext = os.path.splitext(input_value)[1].lower()
                            if ext in image_extensions:
                                available_images = get_input_images()
                                if input_value in available_images:
                                    # 예상 이미지는 입력 디렉터리에 있으며 처리할 필요가 없습니다
                                    pass
                                elif available_images:
                                    # 입력 디렉토리에 이미지가 있지만 예상치 못한 파일 이름일 경우: 최신 이미지를 상단에 자동 캡처합니다.
                                    auto_load_images.append({
                                        "nodeId": node_id,
                                        "classType": class_type,
                                        "inputKey": input_key,
                                        "imageName": input_value,
                                        "suggestedImage": available_images[0]
                                    })
                                else:
                                    # 폴더에 이미지가 없으면 누락 목록으로 이동하고, 팝업 메시지를 표시합니다.
                                    missing_images.append({
                                        "nodeId": node_id,
                                        "classType": class_type,
                                        "inputKey": input_key,
                                        "imageName": input_value
                                    })
                
                return web.json_response({
                    "success": True,
                    "missingModels": missing_models,
                    "missingImages": missing_images,
                    "autoLoadImages": auto_load_images
                })
            except Exception as e:
                import traceback
                traceback.print_exc()
                return web.json_response({
                    "success": False,
                    "error": str(e),
                    "missingModels": [],
                    "missingImages": [],
                    "autoLoadImages": []
                })

        # 모델 다운로드 작업 관리자
        _download_tasks = {}  # task_id -> task_info
        _download_sessions = {}  # task_id -> aiohttp ClientSession
        _download_procs = {}  # task_id -> asyncio subprocess (aria2 다중 스레드 모드)
        _aria2 = {"state": None, "exe": None, "installing": False, "mode": "aiohttp"}  # state: None=검출되지 않음/True=사용 가능/False=사용할 수 없음; 모드: 자동/aria2/aiohttp(기본 aiohttp 단일 스레드)）
        # 공식 aria2 Windows 64 비트 빌드 다운로드 주소
        _ARIA2_URL_WIN = ("https://github.com/aria2/aria2/releases/download/"
                          "release-1.37.0/aria2-1.37.0-win-64bit-build1.zip")
        import re as _re
        # aria2 진행률 요약 행 예시: [#6c6a50 15MiB/121MiB(12%) CN:5 DL:2.1MiB ETA:52s]；총 크기가 알 수 없을 때 형태는 (?) / 0B
        _ARIA2_SUMMARY_RE = _re.compile(
            r"\[\#\S+\s+([\d.]+)([KMGTP]?)iB/([\d.]+)([KMGTP]?)iB"
            r"(?:\((\d+)%\)|\(\?\))\s+CN:\d+\s+DL:([\d.]+)([KMGTP]?)iB"
        )

        @server.routes.post("/comfyui_panel/download_model")
        async def download_model(request):
            """모델 다운로드 작업 생성"""
            try:
                import asyncio
                import uuid
                data = await request.json()
                url = data.get("url", "").strip()
                model_type = data.get("model_type", "").strip()
                subdir = data.get("subdir", "").strip()
                filename = data.get("filename", "").strip()

                if not url:
                    return web.json_response({"success": False, "error": "다운로드 주소 누락"})
                # 안전 검사: HTTP/HTTPS 프로토콜만 허용
                if not url.lower().startswith(("http://", "https://")):
                    return web.json_response({"success": False, "error": "http/https 프로토콜을 지원하는 다운로드 주소만"})
                if not model_type:
                    return web.json_response({"success": False, "error": "모델 디렉토리 유형 부족"})

                # 대상 디렉터리 가져오기: 먼저 folder_paths에서 검색하고 찾을 수 없으면 상대 경로로 처리합니다.
                model_dir = None
                created_dir = False
                try:
                    model_dir = folder_paths.get_folder_paths(model_type)[0]
                except (IndexError, KeyError):
                    # model_type 알려진 디렉토리 이름이 아니면 ComfyUI 루트 디렉토리의 상대 경로로 시도
                    base_path = getattr(folder_paths, 'base_path', None)
                    if base_path and model_type:
                        # 보안 검사: 경로 교차 방지
                        norm_model_type = os.path.normpath(model_type)
                        if norm_model_type.startswith('..') or os.path.isabs(norm_model_type):
                            return web.json_response({"success": False, "error": f"금지된 경로: {model_type}"})
                        candidate = os.path.normpath(os.path.join(base_path, model_type))
                        # 정규화 후에도 ComfyUI 루트 디렉토리 내에 있어야 합니다.
                        base_norm = os.path.normpath(base_path)
                        if not (candidate == base_norm or candidate.startswith(base_norm + os.sep)):
                            return web.json_response({"success": False, "error": f"금지된 경로: {model_type}"})
                        if os.path.isdir(candidate):
                            model_dir = candidate
                        elif os.path.exists(candidate):
                            return web.json_response({"success": False, "error": f"대상 경로가 이미 존재하고 디렉터리가 아닙니다: {model_type}"})
                        else:
                            # 디렉토리가 존재하지 않음 (여러 레벨의 부모 디렉토리 포함 누락)）→ 재귀적으로 자동으로 생성하여 수동 디렉토리 생성 없이
                            try:
                                os.makedirs(candidate, exist_ok=True)
                                model_dir = candidate
                                created_dir = True
                            except Exception as _mk_err:
                                return web.json_response({"success": False, "error": f"경로가 존재하지 않으며 생성할 수 없습니다: {model_type}（{_mk_err}）"})
                    else:
                        return web.json_response({"success": False, "error": f"알 수 없는 모델 디렉토리 유형: {model_type}"})

                target_dir = os.path.join(model_dir, subdir) if subdir else model_dir
                if not os.path.exists(target_dir):
                    os.makedirs(target_dir, exist_ok=True)

                # URL에서 파일 이름 추출
                if not filename:
                    from urllib.parse import urlparse, unquote
                    parsed = urlparse(url)
                    filename = unquote(os.path.basename(parsed.path)) or "model_file"
                    # 제거 쿼리 파라미터
                    if "?" in filename:
                        filename = filename.split("?")[0]

                target_path = os.path.join(target_dir, filename)

                # 진정한 완료 조건: 최종 파일이 존재하고 임시/제어 파일이 다운로드되지 않음 (aria2의 .aria2 또는 aiohttp의) .download）
                _aria2_ctrl = target_path + ".aria2"
                _aiohttp_tmp = target_path + ".download"
                _fully_done = (os.path.exists(target_path)
                               and not os.path.exists(_aria2_ctrl)
                               and not os.path.exists(_aiohttp_tmp))
                if _fully_done:
                    return web.json_response({"success": False, "error": f"파일이 이미 존재합니다: {filename}", "exists": True})

                # 임시/제어 파일 다운로드 존재 -> 다운로드 완료되지 않은 것으로 간주하고 자동적으로 중단 전송 작업을 생성 (오류 보고 대신"존재함"차단）
                # 연결 전송 로직은 _do_download에서 통일적으로 처리합니다: aria2의 .aria2 컨트롤 파일은 자동으로 --continue 옵션을 사용하여 연결 전송을 수행합니다.，
                # aiohttp 의 .download 임시 파일도 다운로드 바이트 연속 전송。
                if os.path.exists(_aria2_ctrl) or os.path.exists(_aiohttp_tmp):
                    task_id = str(uuid.uuid4())[:8]
                    task_info = {
                        "task_id": task_id,
                        "url": url,
                        "model_type": model_type,
                        "subdir": subdir,
                        "filename": filename,
                        "target_path": target_path,
                        "status": "pending",
                        "progress": 0,
                        "downloaded_bytes": os.path.getsize(_aria2_ctrl) if os.path.exists(_aria2_ctrl) else (os.path.getsize(_aiohttp_tmp) if os.path.exists(_aiohttp_tmp) else 0),
                        "total_bytes": 0,
                        "speed": 0,
                        "error": None,
                        "start_time": None,
                    }
                    _download_tasks[task_id] = task_info
                    _engine_state = _kick_aria2_install()
                    asyncio.ensure_future(_do_download(task_id))
                    return web.json_response({
                        "success": True,
                        "task_id": task_id,
                        "resumed": True,
                        "aria2": {
                            "state": _engine_state,
                            "available": _engine_state == "available",
                            "installing": _engine_state == "installing",
                            "exe": _aria2["exe"],
                            "mode": _aria2["mode"],
                        },
                    })

                task_id = str(uuid.uuid4())[:8]
                task_info = {
                    "task_id": task_id,
                    "url": url,
                    "model_type": model_type,
                    "subdir": subdir,
                    "filename": filename,
                    "target_path": target_path,
                    "status": "pending",  # pending/downloading/paused/completed/failed/cancelled
                    "progress": 0,
                    "downloaded_bytes": 0,
                    "total_bytes": 0,
                    "speed": 0,
                    "error": None,
                    "start_time": None,
                }
                _download_tasks[task_id] = task_info

                # aria2 검사/자동 설치 동기화 및 현재 엔진 상태 응답으로 반환（available/installing/unavailable）
                _engine_state = _kick_aria2_install()

                # 다운로드 코루틴 시작
                asyncio.ensure_future(_do_download(task_id))

                return web.json_response({
                    "success": True,
                    "task_id": task_id,
                    "dir_created": created_dir,
                    "aria2": {
                        "state": _engine_state,
                        "available": _engine_state == "available",
                        "installing": _engine_state == "installing",
                        "exe": _aria2["exe"],
                        "mode": _aria2["mode"],
                    },
                })
            except Exception as e:
                import traceback
                traceback.print_exc()
                return web.json_response({"success": False, "error": str(e)})


        @server.routes.post("/comfyui_panel/model_meta")
        async def model_meta(request):
            """모델 스토어 다운로드 링크 파싱, 사이트/작가 이름/작가 프로필 이미지 반환(base64)

            모델 다운로드 패널이 사용자에게 다운로드 URL을 붙여넣은 후 작성자 정보를 자동으로 인식하도록 사용。
            브라우저와 모델 사이트 API는 CORS 제한을 받으므로 백엔드에서 프록시를 전달；
            프로필 이미지는 백엔드에서 가져와 base64 data URL로 반환하여 프론트엔드 교차 사이트 요청을 방지。
            지원: ModelScope / HuggingFace / CivitAI / GitHub
            """
            try:
                import aiohttp as _aio
                import base64 as _b64
                data = await request.json() or {}
                url = str(data.get("url") or "").strip()
                if not url:
                    return web.json_response({"success": False, "error": "누락URL"})
                from urllib.parse import urlparse
                parsed = urlparse(url)
                host = (parsed.hostname or "").lower()
                path = parsed.path or ""
                result = {"success": True, "platform": "", "author": "", "repo": "", "avatarData": "", "homepage": ""}
                _headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"}

                async def _fetch_json(api_url, timeout=8):
                    try:
                        async with _aio.ClientSession() as _sess:
                            async with _sess.get(api_url, headers=_headers, timeout=_aio.ClientTimeout(total=timeout)) as _resp:
                                if _resp.status == 200:
                                    return await _resp.json(content_type=None)
                    except Exception:
                        pass
                    return None

                async def _fetch_image_b64(img_url, timeout=10):
                    """이미지를 다운로드하고 base64 data URL로 변환(최대 2MB), 실패 시 빈 문자열 반환"""
                    try:
                        async with _aio.ClientSession() as _sess:
                            async with _sess.get(img_url, headers=_headers, timeout=_aio.ClientTimeout(total=timeout)) as _resp:
                                if _resp.status != 200:
                                    return ""
                                ct = (_resp.headers.get("Content-Type") or "").lower()
                                body = await _resp.read()
                                if not body or len(body) > 2 * 1024 * 1024:
                                    return ""
                                if "image" not in ct and not img_url.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".gif")):
                                    return ""
                                ext = "png"
                                if "jpeg" in ct or "jpg" in ct:
                                    ext = "jpeg"
                                elif "webp" in ct:
                                    ext = "webp"
                                elif "gif" in ct:
                                    ext = "gif"
                                return "data:image/" + ext + ";base64," + _b64.b64encode(body).decode("ascii")
                    except Exception:
                        pass
                    return ""

                # ----- ModelScope -----
                if "modelscope" in host:
                    m = re.search(r"/models/([^/]+)/([^/?#]+)", path)
                    if m:
                        author, repo = m.group(1), m.group(2)
                        result.update(platform="ModelScope", author=author, repo=repo,
                                      homepage="https://modelscope.cn/models/%s/%s" % (author, repo))
                        info = await _fetch_json("https://modelscope.cn/api/v1/models/%s/%s" % (author, repo))
                        if isinstance(info, dict):
                            d = info.get("Data")
                            if isinstance(d, dict):
                                owner = d.get("Owner")
                                if isinstance(owner, dict):
                                    nm = owner.get("Name") or owner.get("Username") or owner.get("Nickname") or ""
                                    av = owner.get("AvatarUrl") or ""
                                    if nm:
                                        result["author"] = nm
                                    if av:
                                        result["avatarData"] = await _fetch_image_b64(av)
                # ----- HuggingFace -----
                elif "huggingface" in host:
                    parts = [p for p in path.split("/") if p]
                    if len(parts) >= 2:
                        author, repo = parts[0], parts[1]
                        result.update(platform="HuggingFace", author=author, repo=repo,
                                      homepage="https://huggingface.co/%s/%s" % (author, repo))
                        info = await _fetch_json("https://huggingface.co/api/models/%s/%s" % (author, repo))
                        if isinstance(info, dict) and info.get("author"):
                            result["author"] = info["author"]
                        # 프로필 후보 주소 (순차적으로 시도하여 이미지를 반환하는 첫 번째를 가져옴)）
                        for cand in ("https://huggingface.co/%s.png" % author, "https://huggingface.co/%s/avatar" % author):
                            av = await _fetch_image_b64(cand)
                            if av:
                                result["avatarData"] = av
                                break
                # ----- CivitAI -----
                elif "civitai" in host:
                    m = re.search(r"/models/(\d+)", path)
                    if m:
                        mid = m.group(1)
                        result.update(platform="CivitAI", homepage="https://civitai.com/models/%s" % mid)
                        info = await _fetch_json("https://civitai.com/api/v1/models/%s" % mid)
                        if isinstance(info, dict):
                            cr = info.get("creator")
                            if isinstance(cr, dict):
                                if cr.get("username"):
                                    result["author"] = cr["username"]
                                if cr.get("image"):
                                    result["avatarData"] = await _fetch_image_b64(cr["image"])
                # ----- GitHub -----
                elif "github" in host:
                    parts = [p for p in path.split("/") if p]
                    if parts:
                        author = parts[0]
                        result.update(platform="GitHub", author=author, homepage="https://github.com/%s" % author)
                        info = await _fetch_json("https://api.github.com/users/%s" % author)
                        if isinstance(info, dict):
                            if info.get("name"):
                                result["author"] = info["name"]
                            if info.get("avatar_url"):
                                result["avatarData"] = await _fetch_image_b64(info["avatar_url"])
                return web.json_response(result)
            except Exception as e:
                import traceback
                traceback.print_exc()
                return web.json_response({"success": False, "error": str(e)})
        @server.routes.post("/comfyui_panel/download_image")
        async def download_image(request):
            """네트워크 이미지 다운로드 ComfyUI input 디렉토리에 저장, 누락된 이미지 로드 노드 사용"""
            try:
                import asyncio
                import uuid
                data = await request.json()
                url = (data.get("url") or "").strip()
                image_name = (data.get("imageName") or "").strip()

                if not url:
                    return web.json_response({"success": False, "error": "다운로드 주소 누락"})
                if not url.lower().startswith(("http://", "https://")):
                    return web.json_response({"success": False, "error": "http/https 프로토콜을 지원하는 다운로드 주소만"})
                if not image_name:
                    return web.json_response({"success": False, "error": "파일 이름 누락"})

                # 안전 검사: 경로 침투 금지, 순수 파일명만 허용
                safe_name = os.path.basename(image_name)
                if safe_name != image_name or not safe_name:
                    return web.json_response({"success": False, "error": f"유효하지 않은 이미지 파일 이름: {image_name}"})

                input_dir = folder_paths.get_input_directory()
                target_path = os.path.join(input_dir, safe_name)

                # 파일이 존재하고 비어 있지 않으면 로드된 것으로 간주합니다
                if os.path.exists(target_path) and os.path.getsize(target_path) > 0:
                    return web.json_response({"success": True, "task_id": None, "exists": True, "path": target_path})

                task_id = str(uuid.uuid4())[:8]
                _download_tasks[task_id] = {
                    "task_id": task_id,
                    "url": url,
                    "filename": safe_name,
                    "target_path": target_path,
                    "status": "pending",
                    "progress": 0,
                    "downloaded_bytes": 0,
                    "total_bytes": 0,
                    "speed": 0,
                    "error": None,
                    "start_time": None,
                }
                asyncio.ensure_future(_do_download(task_id))
                return web.json_response({"success": True, "task_id": task_id})
            except Exception as e:
                import traceback
                traceback.print_exc()
                return web.json_response({"success": False, "error": str(e)})

        async def _do_download(task_id):
            """다운로드 작업 통합 진입점: 사용자 선택 엔진 모드(auto/aria2/aiohttp)에 따라 분기"""
            import asyncio
            task = _download_tasks.get(task_id)
            if not task:
                return
            mode = _aria2["mode"]
            # 사용자가 aria2 또는 자동 모드에서 모델 다운로드를 수동으로 지정합니다. → aria2 다중 스레드 시도
            if mode == "aria2" or (mode == "auto" and task.get("model_type")):
                # 검출/설치되지 않을 때 설치를 트리거하고 결과를 기다림(최대 180초)）
                _kick_aria2_install()
                if _aria2["installing"]:
                    for _ in range(180):
                        if not _aria2["installing"]:
                            break
                        await asyncio.sleep(1)
                if _get_aria2():
                    await _do_download_aria2(task_id)
                    return
                if mode == "aria2" and task.get("model_type"):
                    task["status"] = "failed"
                    task["error"] = "사용자가 aria2를 지정했지만 자동 설치가 실패했습니다. aria2를 수동으로 설치한 후 다시 시도하십시오."
                    return
            # 사용자가 aiohttp / aria2를 지정할 수 없음 (설치 실패 / 자동 모드에서 모델 다운로드 불가) → aiohttp 회퇴
            await _do_download_aiohttp(task_id)

        def _find_aria2():
            """시스템에 aria2c(PATH 또는 사용자 지정 설치 디렉터리)가 있는지 확인하고 실행 파일 경로 또는
모델 저장 형식 None"""
            p = shutil.which("aria2c")
            if p:
                return p
            base = getattr(folder_paths, 'base_path', None)
            if base:
                exe_name = "aria2c.exe" if os.name == "nt" else "aria2c"
                cand_dir = os.path.join(base, "user", "aria2")
                if os.path.isdir(cand_dir):
                    for root, _, files in os.walk(cand_dir):
                        if exe_name in files:
                            cand = os.path.join(root, exe_name)
                            if os.path.isfile(cand):
                                return cand
            return None

        def _get_aria2():
            """aria2 실행 파일 경로를 가져옵니다. 감지되지 않으면 먼저 감지합니다 (단순 감지, 설치를 트리거하지 않음)）"""
            if _aria2["state"] is True:
                return _aria2["exe"]
            if _aria2["state"] is None:
                exe = _find_aria2()
                if exe:
                    _aria2["state"] = True
                    _aria2["exe"] = exe
                return exe
            return None

        def _aria2_bytes(num, unit):
            """aria2 진행 단위 (예: MiB/GiB)를 바이트로 변환"""
            try:
                v = float(num)
            except (TypeError, ValueError):
                return 0
            mult = {"K": 1 << 10, "M": 1 << 20, "G": 1 << 30, "T": 1 << 40, "P": 1 << 50}
            return int(v * mult.get(unit or "", 1))

        def _cleanup_aria2(task):
            """지정된 카테고리 하의 MultiLora 구성 프리셋 삭제 (지원 .json 및）"""
            for p in (task["target_path"], task["target_path"] + ".aria2"):
                if os.path.exists(p):
                    try:
                        os.remove(p)
                    except Exception:
                        pass

        def _install_aria2_windows():
            """Windows：공식 64비트 릴리스 다운로드 <ComfyUI>/user/aria2 또한 압축 해제"""
            import urllib.request
            import zipfile
            base = getattr(folder_paths, 'base_path', None) or os.getcwd()
            dest_dir = os.path.join(base, "user", "aria2")
            exe = os.path.join(dest_dir, "aria2c.exe")
            if os.path.isfile(exe):
                return exe
            arch = os.environ.get("PROCESSOR_ARCHITEW6432") or os.environ.get("PROCESSOR_ARCHITECTURE") or ""
            if arch and "64" not in arch:
                print("[bsk_MiniMaxDirector][aria2] 현재 32비트 Windows이며 자동 설치 불가 aria2")
                return None
            try:
                os.makedirs(dest_dir, exist_ok=True)
                zip_path = os.path.join(dest_dir, "aria2-dl.zip")
                req = urllib.request.Request(_ARIA2_URL_WIN, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=180) as resp, open(zip_path, "wb") as f:
                    shutil.copyfileobj(resp, f)
                with zipfile.ZipFile(zip_path) as zf:
                    zf.extractall(dest_dir)
                os.remove(zip_path)
                found = _find_aria2()
                if found:
                    print("[bsk_MiniMaxDirector][aria2] Windows 자동 설치 성공:", found)
                    return found
                return None
            except Exception as e:
                print("[bsk_MiniMaxDirector][aria2] Windows 자동 설치 실패:", e)
                return None

        def _install_aria2_linux():
            """Linux：패키지 관리자를 통해 aria2(root 직접 설치, root 아닌 경우 sudo -n 암호 없이)）"""
            import subprocess as _sp
            is_root = hasattr(os, "geteuid") and os.geteuid() == 0
            managers = [
                ["apt-get", "install", "-y", "aria2"],
                ["apt", "install", "-y", "aria2"],
                ["dnf", "install", "-y", "aria2"],
                ["yum", "install", "-y", "aria2"],
                ["zypper", "--non-interactive", "install", "aria2"],
                ["apk", "add", "--no-cache", "aria2"],
                ["pacman", "-S", "--noconfirm", "aria2"],
            ]
            for mgr_args in managers:
                mgr = mgr_args[0]
                if not shutil.which(mgr):
                    continue
                if not is_root and not shutil.which("sudo"):
                    continue
                cmd = mgr_args if is_root else ["sudo", "-n"] + mgr_args
                try:
                    r = _sp.run(cmd, capture_output=True, text=True, timeout=600)
                    if r.returncode == 0:
                        p = shutil.which("aria2c")
                        if p:
                            print("[bsk_MiniMaxDirector][aria2] Linux 설치 성공:", p)
                            return p
                except Exception:
                    continue
            print("[bsk_MiniMaxDirector][aria2] Linux 패키지 매니저 설치 실패, 수동 설치 aria2")
            return None

        def _install_aria2():
            """현재 시스템에 자동으로 aria2를 설치하고 실행 파일 경로를 반환하거나 None"""
            try:
                if os.name == "nt":
                    return _install_aria2_windows()
                return _install_aria2_linux()
            except Exception as e:
                print("[bsk_MiniMaxDirector][aria2] 자동 설치 예외:", e)
                return None

        def _kick_aria2_install():
            """aria2 자동 설치를 트리거합니다 (멱등, installing 플래그를 동기화한 후 비동기적으로 실행)）。
            ‘available’ / ‘installing’ / ‘unavailable’을 반환하여 인터페이스가 즉시 프런트엔드에 엔진 상태를 알려주십시오.。"""
            import asyncio
            if _aria2["state"] is True:
                return "available"
            if _aria2["installing"]:
                return "installing"
            if _aria2["state"] is False:
                return "unavailable"
            _aria2["installing"] = True
            asyncio.ensure_future(_ensure_aria2())
            return "installing"

        async def _ensure_aria2():
            """aria2 사용 가능성 확인: 검사 -> 자동 전환 (원래 탭 버튼 이동)）。
            호출자는 먼저 _kick_aria2_install로 installing 플래그를 동기화해야 합니다; 기본값으로 처리합니다.。"""
            import asyncio
            if _aria2["state"] is not None:
                return _aria2["exe"]
            if not _aria2["installing"]:
                _aria2["installing"] = True
            try:
                exe = await asyncio.to_thread(_install_aria2)
                if exe:
                    _aria2["state"] = True
                    _aria2["exe"] = exe
                else:
                    _aria2["state"] = False
                    _aria2["exe"] = None
            finally:
                _aria2["installing"] = False
            return _aria2["exe"]

        async def _do_download_aria2(task_id):
            """aria2 다중 스레드 다운로드 사용（-x8/--split=8），중단점 재개 지원: 기존 임시 파일 확인"""
            import asyncio
            import time as _time
            task = _download_tasks.get(task_id)
            if not task:
                return
            aria2 = _get_aria2()
            if not aria2:
                task["status"] = "failed"
                task["error"] = "aria2 사용 불가"
                return
            task["status"] = "downloading"
            if not task["start_time"]:
                task["start_time"] = _time.time()
            target = task["target_path"]
            target_dir = os.path.dirname(target) or "."
            filename = os.path.basename(target)
            control = target + ".aria2"

            # 跨引擎전환보호：정리 aiohttp 잔류 임시파일（.download），피하기및 aria2 동일시쓰다동일일목표파일
            _aiohttp_tmp = target + ".download"
            if os.path.exists(_aiohttp_tmp):
                try:
                    os.remove(_aiohttp_tmp)
                except Exception:
                    pass

            # 먼저 HEAD를 사용하여 총 크기를 가져옵니다 (서버에 Content-Length가 없는 경우 진행률을 계산할 수 없습니다. 다운로드에 영향을 미치지 않습니다).）
            if not task["total_bytes"]:
                try:
                    import aiohttp as _aio
                    timeout = _aio.ClientTimeout(total=30, connect=15)
                    async with _aio.ClientSession(timeout=timeout) as _s:
                        async with _s.head(task["url"], allow_redirects=True) as _r:
                            _cl = _r.headers.get("Content-Length")
                            if _cl:
                                task["total_bytes"] = int(_cl)
                except Exception:
                    pass

            base_cmd = [
                aria2, "--no-conf", "--quiet=false",
                "--console-log-level=notice", "--summary-interval=1",
                "--file-allocation=none",
                "--auto-file-renaming=false", "--allow-overwrite=false",
                "--max-tries=5", "--retry-wait=2",
                "--split=8", "-x8", "--timeout=60",
                "-d", target_dir, "-o", filename,
            ]

            while True:
                if task["status"] == "cancelled":
                    _cleanup_aria2(task)
                    return
                # 일시 중지 상태: 프론트엔드 복원 대기(resume_download는 상태를 되돌린다) downloading）。
                # 내부 루프를 사용하여 기다리기, 맨 위로 돌아간 후 'paused' 분기 다시 돌지 않도록。
                while task["status"] == "paused":
                    await asyncio.sleep(0.3)
                    if task["status"] == "cancelled":
                        _cleanup_aria2(task)
                        return
                if not task["start_time"]:
                    task["start_time"] = _time.time()
                # 매번(재) 시작 시 현재 .aria2 컨트롤 파일 존재 여부에 따라 결정 --continue=true，
                # 일시 정지 후 재개할 때 breakpoint를 올바르게 이어받도록 합니다。
                cmd = list(base_cmd)
                if os.path.exists(control):
                    cmd.insert(1, "--continue=true")
                # 불완전한 타겟 파일이 존재하지만 컨트롤 파일이 없으면(예외 중단) 삭제하여 aria2가 오류 보고하지 않도록"존재함"
                if os.path.exists(target) and not os.path.exists(control):
                    try:
                        os.remove(target)
                    except Exception:
                        pass
                cmd = cmd + [task["url"]]
                kwargs = {}
                if os.name == "nt":
                    import subprocess as _sp
                    kwargs["creationflags"] = getattr(_sp, "CREATE_NO_WINDOW", 0x08000000)
                try:
                    proc = await asyncio.create_subprocess_exec(
                        *cmd,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.STDOUT,
                        **kwargs
                    )
                except Exception as e:
                    task["status"] = "failed"
                    task["error"] = f"aria2 시작 실패: {e}"
                    return
                _download_procs[task_id] = proc
                last_time = _time.time()
                last_dl = task["downloaded_bytes"]
                try:
                    async for raw in proc.stdout:
                        line = raw.decode("utf-8", "ignore").strip()
                        if task["status"] == "cancelled":
                            if proc.returncode is None:
                                proc.kill()
                            _download_procs.pop(task_id, None)
                            _cleanup_aria2(task)
                            return
                        if task["status"] == "paused":
                            if proc.returncode is None:
                                proc.kill()
                            _download_procs.pop(task_id, None)
                            break
                        m = _ARIA2_SUMMARY_RE.match(line)
                        if m:
                            dl = _aria2_bytes(m.group(1), m.group(2))
                            tl = _aria2_bytes(m.group(3), m.group(4))
                            pct = int(m.group(5)) if m.group(5) else 0
                            speed = _aria2_bytes(m.group(6), m.group(7))
                            if tl:
                                task["total_bytes"] = tl
                            task["downloaded_bytes"] = dl
                            if pct and task["total_bytes"]:
                                task["progress"] = min(99, pct)
                            now = _time.time()
                            if speed:
                                task["speed"] = speed
                            else:
                                el = now - last_time
                                if el >= 0.5:
                                    task["speed"] = max(0, (dl - last_dl)) / el
                                    last_time = now
                                    last_dl = dl
                except asyncio.CancelledError:
                    if proc.returncode is None:
                        proc.kill()
                    raise
                finally:
                    _download_procs.pop(task_id, None)
                try:
                    await proc.wait()
                except Exception:
                    pass
                if task["status"] == "cancelled":
                    _cleanup_aria2(task)
                    return
                if task["status"] == "paused":
                    continue  # 대기 루프로 돌아가, resume 후 브레이크포인트를 포함한 재시작
                if proc.returncode == 0 and os.path.isfile(target) and os.path.getsize(target) > 0:
                    task["progress"] = 100
                    task["status"] = "completed"
                    task["speed"] = 0
                    task["downloaded_bytes"] = task["total_bytes"] or os.path.getsize(target)
                    return
                task["status"] = "failed"
                task["error"] = f"aria2 다운로드 실패 (exit {proc.returncode})"
                return

        async def _do_download_aiohttp(task_id):
            """다운로드 작업 실행 (원래 aiohttp 단일 연결 구현, aria2 사용 불가능 시의 대체).）"""
            import asyncio
            import time
            try:
                import aiohttp as aiohttp_client
            except ImportError:
                _download_tasks[task_id]["status"] = "failed"
                _download_tasks[task_id]["error"] = "aiohttp not available"
                return

            task = _download_tasks.get(task_id)
            if not task:
                return

            task["status"] = "downloading"
            task["start_time"] = time.time()
            target_path = task["target_path"]
            url = task["url"]

            # 엔진 간 전환 보호: .aria2 컨트롤 파일이 남아있으면 이전에는 aria2 모드로 다운로드한 것을 의미합니다，
            # 이 컨트롤 파일은 aria2 연속 전송만 지원하며, aiohttp는 인식하지 못함. aria2 연속 전송으로 우선 전달。
            aria2_ctrl = target_path + ".aria2"
            if os.path.exists(aria2_ctrl):
                aria2 = _get_aria2()
                if aria2:
                    await _do_download_aria2(task_id)
                    return
                # aria2 사용할 수 없음: .aria2 죽은 파일 정리, aiohttp로 처음부터 다운로드(재시작 불가, 차단 안 함)）
                try:
                    os.remove(aria2_ctrl)
                except Exception:
                    pass

            # 간단한 HTML 지원, 빈 줄 자동 단락 분리
            temp_path = target_path + ".download"
            resume_from = 0
            if os.path.exists(temp_path):
                resume_from = os.path.getsize(temp_path)
                task["downloaded_bytes"] = resume_from

            try:
                headers = {}
                if resume_from > 0:
                    headers["Range"] = f"bytes={resume_from}-"

                timeout = aiohttp_client.ClientTimeout(total=None, connect=30, sock_read=60)
                async with aiohttp_client.ClientSession(timeout=timeout) as session:
                    _download_sessions[task_id] = session
                    async with session.get(url, headers=headers) as resp:
                        if resp.status not in (200, 206):
                            task["status"] = "failed"
                            task["error"] = f"HTTP {resp.status}"
                            return

                        # 총 크기 가져오기
                        content_length = resp.headers.get("Content-Length")
                        if content_length:
                            total = int(content_length)
                            if resume_from > 0 and resp.status == 206:
                                task["total_bytes"] = resume_from + total
                            else:
                                # 서버는 중단점 전송을 지원하지 않으므로 200 대신 206을 반환합니다. 파일을 처음부터 다시 다운로드해야 합니다.
                                task["total_bytes"] = total
                                resume_from = 0
                        else:
                            task["total_bytes"] = 0

                        mode = "ab" if resume_from > 0 else "wb"
                        last_time = time.time()
                        last_bytes = resume_from

                        with open(temp_path, mode) as f:
                            async for chunk in resp.content.iter_chunked(1048576):  # 1MB chunks
                                if task["status"] == "cancelled":
                                    break
                                if task["status"] == "paused":
                                    # 일시 중지 시 복구 대기
                                    while task["status"] == "paused":
                                        await asyncio.sleep(0.5)
                                    if task["status"] == "cancelled":
                                        break
                                    if task["status"] != "downloading":
                                        break

                                f.write(chunk)
                                task["downloaded_bytes"] += len(chunk)

                                # 속도 및 진행률 계산
                                now = time.time()
                                elapsed = now - last_time
                                if elapsed >= 0.5:
                                    task["speed"] = (task["downloaded_bytes"] - last_bytes) / elapsed
                                    last_time = now
                                    last_bytes = task["downloaded_bytes"]

                                if task["total_bytes"] > 0:
                                    task["progress"] = min(99, int(task["downloaded_bytes"] / task["total_bytes"] * 100))

                if task["status"] == "cancelled":
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    return

                # 다운로드 완료, 임시 파일 이름 변경
                if os.path.exists(temp_path):
                    os.rename(temp_path, target_path)
                task["progress"] = 100
                task["status"] = "completed"
                task["speed"] = 0

            except asyncio.CancelledError:
                task["status"] = "cancelled"
            except Exception as e:
                task["status"] = "failed"
                task["error"] = str(e)
                if os.path.exists(temp_path):
                    pass  # 중단 재개 지원을 위해 임시 파일 유지
            finally:
                _download_sessions.pop(task_id, None)

        @server.routes.get("/comfyui_panel/fetch_remote_gallery")
        async def fetch_remote_gallery(request):
            """원격 보드 추출: safebooru dapi JSON API; 다른 보드용 표준 라이브러리 HTML 파싱 추출 이미지+페이지。
            반환 {success, source, items:[{thumb, full, page_url}], pages:{prev,next,current}}"""
            try:
                import json as _json
                from urllib.parse import urlparse, urljoin
                try:
                    import aiohttp as aiohttp_client
                except ImportError:
                    return web.json_response({"success": False, "error": "aiohttp not available"})

                q = request.query
                url = (q.get("url") or "").strip()
                source = (q.get("source") or "auto").strip().lower()
                tags = (q.get("tags") or "").strip()
                try:
                    page = int(q.get("page") or "0")
                except ValueError:
                    page = 0
                if not url:
                    return web.json_response({"success": False, "error": "URL 누락"})
                if not url.lower().startswith(("http://", "https://")):
                    return web.json_response({"success": False, "error": "http/https URL만 지원합니다."})

                host = urlparse(url).netloc.lower()
                is_booru = source == "safebooru" or "booru" in host
                is_danbooru = host in ("danbooru.donmai.us",)
                timeout = aiohttp_client.ClientTimeout(total=None, connect=15, sock_read=30)
                headers = {"User-Agent": "Mozilla/5.0 (compatible; BskUI/1.0)"}

                # 스크래핑 방지 페이지 감지: Cloudflare / Anubis 등 서버 측 HTTP 클라이언트가 우회할 수 없음
                def _is_challenge_page(text):
                    t = (text or "").lower()
                    if not t:
                        return False
                    markers = ("just a moment", "checking your browser", "cf-chl", "cf-challenge",
                               "verify you are human", "verify that you are human",
                               "enable javascript and cookies", "ray id:", "anubis", "attention required")
                    return any(m in t for m in markers)

                async with aiohttp_client.ClientSession(timeout=timeout) as session:
                    if is_booru:
                        if is_danbooru:
                            # === danbooru 공식 JSON API (페이지는 1부터 시작)）===
                            # danbooru API 키(로그인 자격 증명)가 필요하며, 그렇지 않으면 Cloudflare에 의해 차단됩니다.
                            cfg = _load_danbooru_config()
                            danbooru_auth = None
                            if cfg.get("api_key"):
                                danbooru_auth = aiohttp_client.BasicAuth(cfg.get("login", ""), cfg["api_key"])
                            if not danbooru_auth:
                                return web.json_response({"success": False, "error": "웹 페이지 소스 설정에서 danbooru 사용자 이름과 API 키를 입력하십시오 (danbooru는 승인된 자격 증명으로만 캡처)）"})
                            api_url = (f"https://{host}/posts.json"
                                       f"?tags={tags}&page={page + 1}&limit=20")
                            async with session.get(api_url, headers=headers, auth=danbooru_auth) as resp:
                                text = await resp.text()
                            if resp.status != 200:
                                if _is_challenge_page(text):
                                    return web.json_response({"success": False, "error": "이 사이트는 Cloudflare/Anubis 방지 스크립트를 사용하며, 서버 측에서 이미지를 가져올 수 없습니다. 이미지를 수동으로 다운로드하여 카드에 드래그 앤 드롭하십시오."})
                                return web.json_response({"success": False, "error": f"이미지 스테이션 API 반환 HTTP {resp.status}"})
                            if _is_challenge_page(text):
                                return web.json_response({"success": False, "error": "이 사이트는 Cloudflare/Anubis 방지 스크립트를 사용하며, 서버 측에서 이미지를 가져올 수 없습니다. 이미지를 수동으로 다운로드하여 카드에 드래그 앤 드롭하십시오."})
                            try:
                                posts = _json.loads(text)
                            except Exception:
                                return web.json_response({"success": False, "error": "이미지 스테이션 API가 응답을 하지 않음 JSON"})
                            if isinstance(posts, dict):
                                if posts.get("error"):
                                    return web.json_response({"success": False, "error": f"이미지 서버에서 오류 발생: {posts.get('error')}"})
                                posts = posts.get("posts") or ([posts] if posts.get("id") else [])
                            items = []
                            for p in posts:
                                preview = p.get("preview_file_url") or ""
                                full = p.get("file_url") or p.get("large_file_url") or ""
                                preview = urljoin(f"https://{host}/", preview) if preview else ""
                                full = urljoin(f"https://{host}/", full) if full else ""
                                if full:
                                    items.append({"thumb": preview or full, "full": full, "page_url": "", "prompt": p.get("tag_string") or ""})
                            return web.json_response({
                                "success": True,
                                "source": "danbooru",
                                "items": items,
                                "pages": {"prev": max(0, page - 1), "next": page + 1, "current": page},
                            })
                        else:
                            # === gelbooru / safebooru dapi ===
                            api_url = (f"https://{host}/index.php?page=dapi&s=post&q=index"
                                       f"&json=1&tags={tags}&pid={page}&limit=20")
                            async with session.get(api_url, headers=headers) as resp:
                                if resp.status != 200:
                                    return web.json_response({"success": False, "error": f"이미지 스테이션 API 반환 HTTP {resp.status}"})
                                text = await resp.text()
                            try:
                                posts = _json.loads(text)
                            except Exception:
                                return web.json_response({"success": False, "error": "이미지 스테이션 API가 응답을 하지 않음 JSON"})
                            if isinstance(posts, dict):
                                posts = posts.get("post") or posts.get("posts") or []
                            items = []
                            for p in posts:
                                preview = p.get("preview_url") or ""
                                full = p.get("file_url") or ""
                                # safebooru 오래된 필드 바닥내림：directory + image
                                if not full and p.get("directory") and p.get("image"):
                                    full = f"https://{host}/images/{p['directory']}/{p['image']}"
                                    if not preview:
                                        preview = f"https://{host}/thumbnails/{p['directory']}/{p['image']}"
                                preview = urljoin(f"https://{host}/", preview) if preview else ""
                                full = urljoin(f"https://{host}/", full) if full else ""
                                if full:
                                    items.append({"thumb": preview or full, "full": full, "page_url": "", "prompt": p.get("tags") or p.get("tag") or ""})
                            return web.json_response({
                                "success": True,
                                "source": "safebooru",
                                "items": items,
                                "pages": {"prev": max(0, page - 1), "next": page + 1, "current": page},
                            })
                    else:
                        # === 일반 스테이션: 표준 라이브러리 HTML 파싱, 추출 <img> + 페이지 <a> ===
                        async with session.get(url, headers=headers) as resp:
                            html = await resp.text()
                        if resp.status != 200:
                            if _is_challenge_page(html):
                                return web.json_response({"success": False, "error": "이 사이트는 Cloudflare/Anubis 방지 스크립트를 사용하며, 서버 측에서 이미지를 가져올 수 없습니다. 이미지를 수동으로 다운로드하여 카드에 드래그 앤 드롭하십시오."})
                            return web.json_response({"success": False, "error": f"웹 페이지 반환 HTTP {resp.status}"})
                        if _is_challenge_page(html):
                            return web.json_response({"success": False, "error": "이 사이트는 Cloudflare/Anubis 방지 스크립트를 사용하며, 서버 측에서 이미지를 가져올 수 없습니다. 이미지를 수동으로 다운로드하여 카드에 드래그 앤 드롭하십시오."})
                        from html.parser import HTMLParser

                        class _ImgPager(HTMLParser):
                            def __init__(self, base):
                                super().__init__()
                                self.base = base
                                self.imgs = []
                                self.pages = []
                            def handle_starttag(self, tag, attrs):
                                d = dict(attrs)
                                if tag == "img":
                                    src = d.get("src") or d.get("data-src") or d.get("data-original") or ""
                                    if src:
                                        abs = urljoin(self.base, src)
                                        if abs not in self.imgs:
                                            self.imgs.append(abs)
                                elif tag == "a":
                                    href = d.get("href") or ""
                                    if href and any(k in href.lower() for k in ("page", "p=", "next", "?p", "offset")):
                                        abs = urljoin(self.base, href)
                                        if abs not in self.pages:
                                            self.pages.append(abs)

                        parser = _ImgPager(url)
                        parser.feed(html)
                        items = [{"thumb": s, "full": s, "page_url": ""} for s in parser.imgs[:60]]
                        pages = parser.pages[:20]
                        return web.json_response({
                            "success": True,
                            "source": "generic",
                            "items": items,
                            "pages": {"prev": "", "next": pages[-1] if pages else "", "current": url, "all": pages},
                        })
            except Exception as e:
                import traceback
                traceback.print_exc()
                return web.json_response({"success": False, "error": str(e)})

        @server.routes.post("/comfyui_panel/pause_download")
        async def pause_download(request):
            """다운로드 작업 일시 중지"""
            try:
                data = await request.json()
                task_id = data.get("task_id", "")
                task = _download_tasks.get(task_id)
                if not task:
                    return web.json_response({"success": False, "error": "작업이 존재하지 않음"})
                if task["status"] == "downloading":
                    task["status"] = "paused"
                    # aria2 모드: 서브 프로세스 즉시 종료 (aria2 컨트롤 파일 유지, 복원 시 재시작)）
                    proc = _download_procs.get(task_id)
                    if proc and proc.returncode is None:
                        try:
                            proc.kill()
                        except Exception:
                            pass
                    # aiohttp 모드 종료 세션을 막지 않고, 다운로드 코루틴이 루프에서 상태를 감지하고 쓰기를 일시 중지합니다.
                return web.json_response({"success": True})
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        @server.routes.post("/comfyui_panel/resume_download")
        async def resume_download(request):
            """다운로드 작업 복원"""
            try:
                data = await request.json()
                task_id = data.get("task_id", "")
                task = _download_tasks.get(task_id)
                if not task:
                    return web.json_response({"success": False, "error": "작업이 존재하지 않음"})
                if task["status"] == "paused":
                    task["status"] = "downloading"
                    # 다운로드 코루틴은 루프에서 상태가 downloading으로 변경되면 계속 진행
                return web.json_response({"success": True})
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        @server.routes.post("/comfyui_panel/cancel_download")
        async def cancel_download(request):
            """다운로드 작업 취소"""
            try:
                data = await request.json()
                task_id = data.get("task_id", "")
                task = _download_tasks.get(task_id)
                if not task:
                    return web.json_response({"success": False, "error": "작업이 존재하지 않음"})
                task["status"] = "cancelled"
                # aria2 패턴: 자식 프로세스 종료 및 대상 파일 및 .aria2 제어 파일 정리
                proc = _download_procs.get(task_id)
                if proc and proc.returncode is None:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                _cleanup_aria2(task)
                session = _download_sessions.get(task_id)
                if session:
                    await session.close()
                # aiohttp 임시 파일 삭제
                temp_path = task["target_path"] + ".download"
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                return web.json_response({"success": True})
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        @server.routes.post("/comfyui_panel/rename_model")
        async def rename_model(request):
            """이름 바꾸기 다운로드된 모델 파일"""
            try:
                data = await request.json()
                model_type = data.get("model_type", "").strip()
                subdir = data.get("subdir", "").strip()
                old_filename = data.get("old_filename", "").strip()
                new_filename = data.get("new_filename", "").strip()

                if not model_type or not old_filename or not new_filename:
                    return web.json_response({"success": False, "error": "필요한 매개변수가 없습니다"})

                if old_filename == new_filename:
                    return web.json_response({"success": False, "error": "새 파일 이름과 기존 파일 이름이 동일"})

                # 안전 검사: 새 파일 이름에 경로 구분자가 포함될 수 없습니다
                if "/" in new_filename or "\\" in new_filename or ".." in new_filename:
                    return web.json_response({"success": False, "error": "파일명에 불법 문자 포함"})

                # 대상 디렉터리 가져오기: 먼저 folder_paths에서 검색하고 찾을 수 없으면 상대 경로로 처리합니다.
                model_dir = None
                try:
                    model_dir = folder_paths.get_folder_paths(model_type)[0]
                except (IndexError, KeyError):
                    base_path = getattr(folder_paths, 'base_path', None)
                    if base_path and model_type:
                        norm_model_type = os.path.normpath(model_type)
                        if norm_model_type.startswith('..') or os.path.isabs(norm_model_type):
                            return web.json_response({"success": False, "error": f"금지된 경로: {model_type}"})
                        candidate = os.path.join(base_path, model_type)
                        if os.path.exists(candidate) and os.path.isdir(candidate):
                            model_dir = candidate
                        else:
                            return web.json_response({"success": False, "error": f"경로가 존재하지 않음: {model_type}"})
                    else:
                        return web.json_response({"success": False, "error": f"알 수 없는 모델 디렉토리 유형: {model_type}"})

                target_dir = os.path.join(model_dir, subdir) if subdir else model_dir
                old_path = os.path.join(target_dir, old_filename)
                new_path = os.path.join(target_dir, new_filename)

                if not os.path.exists(old_path):
                    return web.json_response({"success": False, "error": f"파일이 존재하지 않습니다: {old_filename}"})

                if os.path.exists(new_path):
                    return web.json_response({"success": False, "error": f"파일이 이미 존재합니다: {new_filename}"})

                os.rename(old_path, new_path)
                return web.json_response({"success": True, "new_filename": new_filename})
            except Exception as e:
                import traceback
                traceback.print_exc()
                return web.json_response({"success": False, "error": str(e)})

        @server.routes.get("/comfyui_panel/download_status")
        async def download_status(request):
            """모든 다운로드 작업 상태 및 현재 다운로드 엔진 확인"""
            try:
                tasks = []
                for tid, task in _download_tasks.items():
                    tasks.append({
                        "task_id": task["task_id"],
                        "url": task["url"],
                        "model_type": task.get("model_type", ""),
                        "subdir": task.get("subdir", ""),
                        "filename": task["filename"],
                        "status": task["status"],
                        "progress": task["progress"],
                        "downloaded_bytes": task["downloaded_bytes"],
                        "total_bytes": task["total_bytes"],
                        "speed": task["speed"],
                        "error": task["error"],
                    })
                # 다운로드 엔진 상태 부가(단순 감지, 설치 유발 X)）
                if _aria2["state"] is None and not _aria2["installing"]:
                    _get_aria2()
                return web.json_response({
                    "success": True,
                    "tasks": tasks,
                    "aria2": {
                        "available": _aria2["state"] is True,
                        "installing": _aria2["installing"],
                        "exe": _aria2["exe"],
                        "mode": _aria2["mode"],
                    },
                })
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        @server.routes.post("/comfyui_panel/download_engine")
        async def download_engine(request):
            """다운로드 엔진 모드 설정: auto (자동) / aria2 (강제 멀티 스레드) / aiohttp (강제 싱글 스레드)）"""
            try:
                data = await request.json() or {}
                mode = str(data.get("mode", "")).strip().lower()
                if mode not in ("auto", "aria2", "aiohttp"):
                    return web.json_response({"success": False, "error": f"유효하지 않은 엔진 모드: {mode}"})
                _aria2["mode"] = mode
                return web.json_response({
                    "success": True,
                    "mode": mode,
                    "aria2": {
                        "available": _aria2["state"] is True,
                        "installing": _aria2["installing"],
                        "exe": _aria2["exe"],
                        "mode": mode,
                    },
                })
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        def _aria2_install_command():
            """현재 시스템의 aria2 수동 설치 명령 생성 (프론트엔드 팝업 표시/복사)）"""
            base = getattr(folder_paths, 'base_path', None) or os.getcwd()
            if os.name == "nt":
                dest = os.path.join(base, "user", "aria2")
                cmd = (
                    "$dest = '" + dest.replace("'", "''") + "'; "
                    "New-Item -ItemType Directory -Path $dest -Force | Out-Null; "
                    "Invoke-WebRequest -Uri '" + _ARIA2_URL_WIN + "' -OutFile \"$env:TEMP\\aria2-dl.zip\"; "
                    "Expand-Archive -Path \"$env:TEMP\\aria2-dl.zip\" -DestinationPath $dest -Force; "
                    "Remove-Item \"$env:TEMP\\aria2-dl.zip\""
                )
                return "PowerShell에서 다음 명령을 실행하십시오：\n" + cmd
            # Linux：현재 사용 가능한 패키지 관리자를 사용하여 생성합니다
            cmds = []
            for mgr, cmd in (
                ("apt-get", "sudo apt-get install -y aria2"),
                ("apt", "sudo apt install -y aria2"),
                ("dnf", "sudo dnf install -y aria2"),
                ("yum", "sudo yum install -y aria2"),
                ("zypper", "sudo zypper --non-interactive install aria2"),
                ("apk", "sudo apk add --no-cache aria2"),
                ("pacman", "sudo pacman -S --noconfirm aria2"),
            ):
                if shutil.which(mgr):
                    cmds.append(cmd)
            if not cmds:
                cmds.append("sudo apt-get install -y aria2")
            return "터미널에서 다음 명령을 실행하십시오 (하나 선택, root 권한 필요)）：\n" + "\n".join(cmds)

        @server.routes.get("/comfyui_panel/aria2_install_info")
        async def aria2_install_info(request):
            """aria2 설치 정보 반환: 현재 상태 + 수동 설치 명령"""
            try:
                if _aria2["state"] is None and not _aria2["installing"]:
                    _get_aria2()
                installed = _aria2["state"] is True
                return web.json_response({
                    "success": True,
                    "installed": installed,
                    "installing": _aria2["installing"],
                    "command": _aria2_install_command() if not installed else "",
                    "system": "Windows" if os.name == "nt" else "Linux",
                    "aria2": {
                        "available": _aria2["state"] is True,
                        "installing": _aria2["installing"],
                        "exe": _aria2["exe"],
                        "mode": _aria2["mode"],
                    },
                })
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        @server.routes.post("/comfyui_panel/install_aria2")
        async def install_aria2(request):
            """aria2 자동 설치 트리거 (비동기 실행, 프런트엔드 폴링으로 진행 상황 확인)）"""
            try:
                if _aria2["state"] is False:
                    # 이전 자동 설치 실패: 수동 재시도 전 상태 초기화
                    _aria2["state"] = None
                _kick_aria2_install()
                return web.json_response({
                    "success": True,
                    "aria2": {
                        "available": _aria2["state"] is True,
                        "installing": _aria2["installing"],
                        "exe": _aria2["exe"],
                        "mode": _aria2["mode"],
                    },
                })
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        @server.routes.get("/comfyui_panel/model_dirs")
        async def model_dirs(request):
            """모든 모델 디렉토리 유형 및 하위 디렉토리 가져오기"""
            try:
                result = {}
                for folder_name, folder_paths_list in folder_paths.folder_names_and_paths.items():
                    if not folder_paths_list:
                        continue
                    # folder_names_and_paths 구조: {name: ([paths], {extensions})}
                    try:
                        paths_list = folder_paths_list[0] if isinstance(folder_paths_list, (list, tuple)) else [folder_paths_list]
                    except (IndexError, TypeError):
                        continue
                    if not paths_list:
                        continue
                    base_dir = paths_list[0] if isinstance(paths_list, list) else paths_list
                    if not isinstance(base_dir, str) or not os.path.isdir(base_dir):
                        continue
                    subdirs = [""]
                    for root, dirs, files in os.walk(base_dir):
                        for d in dirs:
                            rel = os.path.relpath(os.path.join(root, d), base_dir).replace("\\", "/")
                            subdirs.append(rel)
                    result[folder_name] = {
                        "path": base_dir,
                        "subdirs": sorted(subdirs)
                    }
                return web.json_response({"success": True, "dirs": result})
            except Exception as e:
                import traceback
                traceback.print_exc()
                return web.json_response({"success": False, "error": str(e)})

        def _scan_workflow_dir(dir_path, base_path, depth=0, max_depth=5):
            """재귀적으로 워크플로우 디렉토리를 스캔하고 트리 구조를 반환합니다."""
            items = []
            try:
                entries = sorted(os.listdir(dir_path))
            except:
                return items
            for entry in entries:
                if entry.startswith('.'):
                    continue
                full_path = os.path.join(dir_path, entry)
                rel_path = os.path.relpath(full_path, base_path)
                if os.path.isdir(full_path):
                    if depth < max_depth:
                        children = _scan_workflow_dir(full_path, base_path, depth + 1, max_depth)
                        if children:  # 콘텐츠가 있는 폴더만 포함
                            items.append({
                                "type": "dir",
                                "name": entry,
                                "rel_path": rel_path,
                                "children": children
                            })
                elif os.path.isfile(full_path) and entry.endswith('.json'):
                    stat = os.stat(full_path)
                    items.append({
                        "type": "file",
                        "name": entry,
                        "rel_path": rel_path,
                        "size": stat.st_size,
                        "mtime": stat.st_mtime
                    })
            return items

        @server.routes.get("/comfyui_panel/workflow_files")
        async def get_workflow_files(request):
            """워크플로우 파일 목록 가져오기
            
            사용자 워크플로우 디렉토리에서 .json 워크플로우 파일 목록 가져오기。
            여러 레벨 디렉터리를 재귀적으로 스캔하고, 트리 구조를 반환。
            """
            try:
                # 워크플로우 디렉토리 설정
                workflow_dir = None
                
                # 1. folder_paths에서 가져오려고 시도합니다
                try:
                    workflow_paths = folder_paths.get_folder_paths("workflows")
                    if workflow_paths:
                        workflow_dir = workflow_paths[0]
                except:
                    pass
                
                # 2. 사용자 디렉터리 아래의 workflows 폴더 시도
                if not workflow_dir or not os.path.exists(workflow_dir):
                    try:
                        base_path = folder_paths.base_path
                        user_workflow_dir = os.path.join(base_path, "user", "default", "workflows")
                        if os.path.exists(user_workflow_dir):
                            workflow_dir = user_workflow_dir
                    except:
                        pass
                
                # 3. ComfyUI 루트 디렉토리 아래 workflows 폴더 시도
                if not workflow_dir or not os.path.exists(workflow_dir):
                    try:
                        base_path = folder_paths.base_path
                        root_workflow_dir = os.path.join(base_path, "workflows")
                        if os.path.exists(root_workflow_dir):
                            workflow_dir = root_workflow_dir
                    except:
                        pass
                
                if not workflow_dir or not os.path.exists(workflow_dir):
                    return web.json_response({
                        "success": False,
                        "error": "Workflow directory not found",
                        "tree": [],
                        "path": workflow_dir or ""
                    })
                
                # 디렉토리 재귀 스캔
                tree = _scan_workflow_dir(workflow_dir, workflow_dir)
                
                return web.json_response({
                    "success": True,
                    "tree": tree,
                    "path": workflow_dir
                })
                
            except Exception as e:
                return web.json_response({
                    "success": False,
                    "error": str(e),
                    "tree": [],
                    "path": ""
                })

        @server.routes.get("/comfyui_panel/read_workflow")
        async def read_workflow_file(request):
            """읽기 워크플로우 파일 내용"""
            try:
                rel_path = request.query.get("path", "")
                if not rel_path:
                    return web.json_response({"success": False, "error": "Missing path parameter"})
                
                # 보안 검사: 경로 탐색 방지
                safe_path = os.path.normpath(rel_path)
                if safe_path.startswith('..') or os.path.isabs(safe_path):
                    return web.json_response({"success": False, "error": "Invalid path"})
                
                # 워크플로우 루트 디렉토리 결정
                workflow_dir = None
                try:
                    workflow_paths = folder_paths.get_folder_paths("workflows")
                    if workflow_paths:
                        workflow_dir = workflow_paths[0]
                except:
                    pass
                if not workflow_dir or not os.path.exists(workflow_dir):
                    try:
                        base_path = folder_paths.base_path
                        for candidate in [
                            os.path.join(base_path, "user", "default", "workflows"),
                            os.path.join(base_path, "workflows"),
                        ]:
                            if os.path.exists(candidate):
                                workflow_dir = candidate
                                break
                    except:
                        pass
                
                if not workflow_dir:
                    return web.json_response({"success": False, "error": "Workflow directory not found"})
                
                full_path = os.path.join(workflow_dir, safe_path)
                if not os.path.abspath(full_path).startswith(os.path.abspath(workflow_dir)):
                    return web.json_response({"success": False, "error": "Path traversal denied"})
                
                if not os.path.isfile(full_path):
                    return web.json_response({"success": False, "error": "File not found"})
                
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                import json as _json
                data = _json.loads(content)
                
                return web.json_response({
                    "success": True,
                    "data": data,
                    "path": rel_path
                })
                
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})
                
        # ---------- 새로 추가: 패널 설정 관리 API ----------
        @server.routes.post("/comfyui_panel/save_config")
        async def save_config(request):
            """분석 '설명' 유지: 다중 텍스트 상자(내용에 따라 높이 자동 조절)"""
            try:
                data = await request.json()
                filename = data.get("filename")
                config = data.get("config")

                if not filename or not config:
                    return web.json_response({"success": False, "error": "Missing filename or config"})

                # 파일 이름이 안전하고 경로 탐색을 방지합니다
                safe_filename = os.path.basename(filename)
                if not safe_filename.endswith(".json"):
                    safe_filename += ".json"

                # 패널 디렉토리 존재 확인
                panel_dir = USER_PANEL_DIR
                panel_dir.mkdir(exist_ok=True)

                file_path = panel_dir / safe_filename

                # 파일 쓰기
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(config, f, indent=2, ensure_ascii=False)

                return web.json_response({
                    "success": True,
                    "filename": safe_filename,
                    "message": f"Config saved to {safe_filename}"
                })

            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        # ---------- 워크플로우 템플릿 라이브러리 ----------
        def _get_templates_file():
            """템플릿 라이브러리 JSON 파일 경로 가져오기 (ComfyUI/user/default/bsk_workflow_templates/templates.json)"""
            try:
                base_path = folder_paths.base_path
            except Exception:
                base_path = os.path.abspath('./')
            tmpl_dir = os.path.join(base_path, "user", "default", "bsk_workflow_templates")
            os.makedirs(tmpl_dir, exist_ok=True)
            return os.path.join(tmpl_dir, "templates.json")

        def _read_templates():
            """모든 템플릿 읽기, 반환 list"""
            fpath = _get_templates_file()
            if not os.path.exists(fpath):
                return []
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict) and isinstance(data.get("templates"), list):
                    return data["templates"]
                if isinstance(data, list):
                    return data
            except Exception as e:
                print(f"[ComfyUI Panel] 템플릿 라이브러리를 읽는 데 실패했습니다: {e}")
            return []

        def _write_templates(templates):
            """이미지 시퀀스 준비"""
            fpath = _get_templates_file()
            with open(fpath, "w", encoding="utf-8") as f:
                json.dump({"templates": templates}, f, indent=2, ensure_ascii=False)

        @server.routes.get("/comfyui_panel/templates/list")
        async def templates_list(request):
            """모든 템플릿 나열 분류별 그룹으로 반환"""
            try:
                templates = _read_templates()
                # 구조 분류 -> 템플릿 목록
                categories = {}
                for t in templates:
                    cat = t.get("category", "") or "분류되지 않음"
                    categories.setdefault(cat, []).append({
                        "id": t.get("id"),
                        "name": t.get("name", ""),
                        "category": cat,
                        "nodeCount": len(t.get("nodes", {})),
                        "createdAt": t.get("createdAt", 0),
                    })
                return web.json_response({"success": True, "categories": categories})
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        @server.routes.post("/comfyui_panel/templates/save")
        async def templates_save(request):
            """템플릿 저장 (새로 추가 또는 업데이트))"""
            try:
                data = await request.json()
                tmpl_id = data.get("id")
                name = data.get("name", "").strip()
                category = data.get("category", "").strip()
                nodes = data.get("nodes")

                if not name or not nodes:
                    return web.json_response({"success": False, "error": "이름 또는 노드 데이터 부족"})

                templates = _read_templates()
                import time as _time
                import uuid as _uuid

                if tmpl_id:
                    # 템플릿 업데이트
                    found = False
                    for t in templates:
                        if t.get("id") == tmpl_id:
                            t["name"] = name
                            t["category"] = category
                            t["nodes"] = nodes
                            found = True
                            break
                    if not found:
                        templates.append({
                            "id": tmpl_id, "name": name, "category": category,
                            "nodes": nodes, "createdAt": _time.time(),
                        })
                else:
                    tmpl_id = str(_uuid.uuid4())
                    templates.append({
                        "id": tmpl_id, "name": name, "category": category,
                        "nodes": nodes, "createdAt": _time.time(),
                    })

                _write_templates(templates)
                return web.json_response({"success": True, "id": tmpl_id})
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        @server.routes.post("/comfyui_panel/templates/delete")
        async def templates_delete(request):
            """템플릿 삭제"""
            try:
                data = await request.json()
                tmpl_id = data.get("id")
                if not tmpl_id:
                    return web.json_response({"success": False, "error": "템플릿 누락 ID"})
                templates = _read_templates()
                templates = [t for t in templates if t.get("id") != tmpl_id]
                _write_templates(templates)
                return web.json_response({"success": True})
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        @server.routes.get("/comfyui_panel/templates/load")
        async def templates_load(request):
            """로드 단일 템플릿의 완전한 노드 데이터"""
            try:
                tmpl_id = request.query.get("id", "")
                if not tmpl_id:
                    return web.json_response({"success": False, "error": "템플릿 누락 ID"})
                templates = _read_templates()
                for t in templates:
                    if t.get("id") == tmpl_id:
                        return web.json_response({"success": True, "template": t})
                return web.json_response({"success": False, "error": "템플릿이 없음"})
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        # ---------- 전역 프롬프트 텍스트 설정 라이브러리 (사용자 디렉토리, 업그레이드 시 덮어쓰기 방지)) ----------
        def _get_prompt_configs_dir():
            """사용자 전역 프롬프트 텍스트 설정 디렉토리 가져오기 (ComfyUI/user/default/bsk_workflow_templates)"""
            try:
                base_path = folder_paths.base_path
            except Exception:
                base_path = os.path.abspath('./')
            cfg_dir = os.path.join(base_path, "user", "default", "bsk_workflow_templates")
            os.makedirs(cfg_dir, exist_ok=True)
            return cfg_dir

        # 제외해야 할 파일 이름(워크플로우 템플릿 라이브러리 메인 파일, 프롬프트 설정에 해당하지 않음)）
        _PROMPT_CONFIG_EXCLUDE = {"templates.json"}

        @server.routes.get("/comfyui_panel/prompt_configs/list")
        async def prompt_configs_list(request):
            """모든 글로벌 프롬프트 텍스트 설정 파일 나열 (.json)"""
            try:
                cfg_dir = _get_prompt_configs_dir()
                files = []
                for f in os.listdir(cfg_dir):
                    if not f.lower().endswith(".json"):
                        continue
                    if f in _PROMPT_CONFIG_EXCLUDE:
                        continue
                    full = os.path.join(cfg_dir, f)
                    if not os.path.isfile(full):
                        continue
                    files.append({
                        "name": f,
                        "display": os.path.splitext(f)[0],
                        "mtime": os.path.getmtime(full)
                    })
                files.sort(key=lambda x: x["mtime"], reverse=True)
                return web.json_response({"success": True, "files": files})
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        @server.routes.get("/comfyui_panel/prompt_configs/load")
        async def prompt_configs_load(request):
            """지정된 전역 프롬프트 튜닝 설정 로드"""
            try:
                filename = request.query.get("name", "")
                safe_filename = os.path.basename(filename)
                if not safe_filename:
                    return web.json_response({"success": False, "error": "파일 이름이 없습니다"})
                if not safe_filename.lower().endswith(".json"):
                    safe_filename += ".json"
                if safe_filename in _PROMPT_CONFIG_EXCLUDE:
                    return web.json_response({"success": False, "error": "이 파일에 접근할 수 없습니다"})
                cfg_dir = _get_prompt_configs_dir()
                file_path = os.path.join(cfg_dir, safe_filename)
                if not os.path.exists(file_path):
                    return web.json_response({"success": False, "error": "파일이 존재하지 않습니다"})
                with open(file_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                return web.json_response({"success": True, "name": safe_filename, "config": config})
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        @server.routes.post("/comfyui_panel/prompt_configs/save")
        async def prompt_configs_save(request):
            """전역 프롬프트 설정 저장 사용자 디렉토리"""
            try:
                data = await request.json()
                filename = data.get("filename", "")
                config = data.get("config")
                if config is None:
                    return web.json_response({"success": False, "error": "설정 데이터를 누락했습니다"})
                safe_filename = os.path.basename(filename) if filename else "전역 프롬프트"
                if not safe_filename:
                    safe_filename = "전역 프롬프트"
                if not safe_filename.lower().endswith(".json"):
                    safe_filename += ".json"
                if safe_filename in _PROMPT_CONFIG_EXCLUDE:
                    return web.json_response({"success": False, "error": "이 파일 이름은 허용되지 않습니다"})
                cfg_dir = _get_prompt_configs_dir()
                file_path = os.path.join(cfg_dir, safe_filename)
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(config, f, indent=2, ensure_ascii=False)
                return web.json_response({"success": True, "name": safe_filename})
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        @server.routes.post("/comfyui_panel/prompt_configs/delete")
        async def prompt_configs_delete(request):
            """전역 프롬프트 설정 삭제"""
            try:
                data = await request.json()
                filename = data.get("name", "")
                safe_filename = os.path.basename(filename)
                if not safe_filename.lower().endswith(".json"):
                    safe_filename += ".json"
                if safe_filename in _PROMPT_CONFIG_EXCLUDE:
                    return web.json_response({"success": False, "error": "이 파일은 삭제할 수 없습니다"})
                cfg_dir = _get_prompt_configs_dir()
                file_path = os.path.join(cfg_dir, safe_filename)
                if not os.path.exists(file_path):
                    return web.json_response({"success": False, "error": "파일이 존재하지 않습니다"})
                os.remove(file_path)
                return web.json_response({"success": True})
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        # ---------- MultiLora 프리셋 라이브러리 구성 (사용자 디렉토리, 업그레이드 시 덮어쓰기 방지)) ----------
        _MLORA_DEFAULT_CATEGORY = "기본"

        def _get_multilora_configs_dir():
            """MultiLora 설정 프리셋 디렉토리를 가져옵니다. (ComfyUI/user/default/bsk_multilora_configs)"""
            try:
                base_path = folder_paths.base_path
            except Exception:
                base_path = os.path.abspath('./')
            cfg_dir = os.path.join(base_path, "user", "default", "bsk_multilora_configs")
            os.makedirs(cfg_dir, exist_ok=True)
            return cfg_dir

        def _mlora_safe_name(name):
            """분류/파일명 정리, 경로 침투 방지; 중영문 유지"""
            if not name:
                return ""
            name = str(name).strip()
            # 경로 구분자 및 위험 문자 제거
            for ch in ("/", "\\", ":", "*", "?", '"', "<", ">", "|"):
                name = name.replace(ch, "_")
            # 마지막 구간 (방어적)）
            name = os.path.basename(name)
            return name

        def _mlora_category_dir(cfg_dir, category):
            """분류명으로 해당 디렉토리 반환: 기본 분류=루트 디렉토리, 기타 카테고리=자 디렉토리"""
            if not category or category == _MLORA_DEFAULT_CATEGORY:
                return cfg_dir
            safe = _mlora_safe_name(category)
            d = os.path.join(cfg_dir, safe)
            return d

        def _mlora_read_config_file(file_path):
            """.json 또는“내장된 설정 .png”MultiLora 설정 객체를 읽어옵니다; 실패 시 반환합니다. None"""
            try:
                if file_path.lower().endswith(".png"):
                    from PIL import Image as _Image
                    import zlib as _zlib
                    import base64 as _b64
                    with _Image.open(file_path) as img:
                        info = img.info or {}
                    if "config_json" not in info:
                        return None
                    json_str = info["config_json"]
                    if info.get("config_compressed") == "1":
                        json_str = _zlib.decompress(_b64.b64decode(json_str)).decode("utf-8")
                    return json.loads(json_str)
                with open(file_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"[MultiLora] 설정 읽기 실패 {file_path}: {e}")
                return None

        @server.routes.get("/comfyui_panel/multilora_configs/list_categories")
        async def multilora_configs_list_categories(request):
            """모든 MultiLora 설정 카테고리 나열 (기본 카테고리) + 자 디렉토리）"""
            try:
                cfg_dir = _get_multilora_configs_dir()
                cats = [_MLORA_DEFAULT_CATEGORY]
                for f in os.listdir(cfg_dir):
                    full = os.path.join(cfg_dir, f)
                    if os.path.isdir(full) and not f.startswith("."):
                        cats.append(f)
                return web.json_response({"success": True, "categories": cats})
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        @server.routes.post("/comfyui_panel/multilora_configs/create_category")
        async def multilora_configs_create_category(request):
            """새로운 MultiLora 구성 분류 (자식 디렉토리)를 만듭니다）"""
            try:
                data = await request.json()
                name = _mlora_safe_name(data.get("name", ""))
                if not name:
                    return web.json_response({"success": False, "error": "분류 이름 누락"})
                if name == _MLORA_DEFAULT_CATEGORY:
                    return web.json_response({"success": False, "error": "이 이름은 보존 분류"})
                cfg_dir = _get_multilora_configs_dir()
                target = os.path.join(cfg_dir, name)
                if os.path.exists(target):
                    return web.json_response({"success": False, "error": "분류가 있음"})
                os.makedirs(target, exist_ok=True)
                return web.json_response({"success": True, "name": name})
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        @server.routes.post("/comfyui_panel/multilora_configs/delete_category")
        async def multilora_configs_delete_category(request):
            """MultiLora 구성 범주를 삭제합니다(그 아래의 모든 프리셋 포함)）"""
            try:
                data = await request.json()
                name = _mlora_safe_name(data.get("name", ""))
                if not name:
                    return web.json_response({"success": False, "error": "분류 이름 누락"})
                if name == _MLORA_DEFAULT_CATEGORY:
                    return web.json_response({"success": False, "error": "기본적으로 카테고리 삭제 불가"})
                cfg_dir = _get_multilora_configs_dir()
                target = os.path.join(cfg_dir, name)
                # 중복 검사: 구성 디렉터리 내에 있어야 하며 디렉터리여야 함
                if not os.path.abspath(target).startswith(os.path.abspath(cfg_dir)):
                    return web.json_response({"success": False, "error": "유효하지 않은 경로"})
                if not os.path.isdir(target):
                    return web.json_response({"success": False, "error": "분류가 없음"})
                import shutil
                shutil.rmtree(target)
                return web.json_response({"success": True})
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        @server.routes.get("/comfyui_panel/multilora_configs/list")
        async def multilora_configs_list(request):
            """지정된 분류의 모든 MultiLora 구성 프리셋 파일(.json 및 임베딩 구성 포함) .png)"""
            try:
                category = request.query.get("category", _MLORA_DEFAULT_CATEGORY)
                cfg_dir = _get_multilora_configs_dir()
                cat_dir = _mlora_category_dir(cfg_dir, category)
                files = []
                if os.path.isdir(cat_dir):
                    for f in os.listdir(cat_dir):
                        low = f.lower()
                        if not (low.endswith(".json") or low.endswith(".png")):
                            continue
                        full = os.path.join(cat_dir, f)
                        if not os.path.isfile(full):
                            continue
                        files.append({
                            "name": f,
                            "display": os.path.splitext(f)[0],
                            "type": "png" if low.endswith(".png") else "json",
                            "mtime": os.path.getmtime(full)
                        })
                files.sort(key=lambda x: x["mtime"], reverse=True)
                return web.json_response({"success": True, "category": category, "files": files})
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        @server.routes.get("/comfyui_panel/multilora_configs/load")
        async def multilora_configs_load(request):
            """로드 지정 분류의 MultiLora 구성 프리셋 (지원 .json 및 임베디드 구성) .png）"""
            try:
                filename = request.query.get("name", "")
                category = request.query.get("category", _MLORA_DEFAULT_CATEGORY)
                safe_filename = _mlora_safe_name(filename)
                if not safe_filename:
                    return web.json_response({"success": False, "error": "파일 이름이 없습니다"})
                low = safe_filename.lower()
                if not (low.endswith(".json") or low.endswith(".png")):
                    safe_filename += ".json"
                cfg_dir = _get_multilora_configs_dir()
                cat_dir = _mlora_category_dir(cfg_dir, category)
                file_path = os.path.join(cat_dir, safe_filename)
                if not os.path.exists(file_path):
                    return web.json_response({"success": False, "error": "파일이 존재하지 않습니다"})
                config = _mlora_read_config_file(file_path)
                if config is None:
                    return web.json_response({"success": False, "error": "설정 읽기 실패"})
                return web.json_response({"success": True, "name": safe_filename, "category": category, "config": config})
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        @server.routes.post("/comfyui_panel/multilora_configs/save")
        async def multilora_configs_save(request):
            """MultiLora 설정 프리셋을 지정된 카테고리로 저장"""
            try:
                data = await request.json()
                filename = _mlora_safe_name(data.get("filename", ""))
                category = data.get("category", _MLORA_DEFAULT_CATEGORY) or _MLORA_DEFAULT_CATEGORY
                config = data.get("config")
                if config is None:
                    return web.json_response({"success": False, "error": "설정 데이터를 누락했습니다"})
                if not filename:
                    filename = "MultiLora설정"
                if not filename.lower().endswith(".json"):
                    filename += ".json"
                cfg_dir = _get_multilora_configs_dir()
                cat_dir = _mlora_category_dir(cfg_dir, category)
                if not os.path.isdir(cat_dir):
                    os.makedirs(cat_dir, exist_ok=True)
                file_path = os.path.join(cat_dir, filename)
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(config, f, indent=2, ensure_ascii=False)
                return web.json_response({"success": True, "name": filename, "category": category})
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        @server.routes.post("/comfyui_panel/multilora_configs/delete")
        async def multilora_configs_delete(request):
            """삭제 지정된 카테고리 하의 MultiLora 구성 프리셋 (지원 .json 및 .png）"""
            try:
                data = await request.json()
                filename = _mlora_safe_name(data.get("name", ""))
                category = data.get("category", _MLORA_DEFAULT_CATEGORY) or _MLORA_DEFAULT_CATEGORY
                low = filename.lower()
                if not (low.endswith(".json") or low.endswith(".png")):
                    filename += ".json"
                cfg_dir = _get_multilora_configs_dir()
                cat_dir = _mlora_category_dir(cfg_dir, category)
                file_path = os.path.join(cat_dir, filename)
                if not os.path.exists(file_path):
                    return web.json_response({"success": False, "error": "파일이 존재하지 않습니다"})
                os.remove(file_path)
                return web.json_response({"success": True})
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        @server.routes.get("/comfyui_panel/multilora_configs/thumbnail")
        async def multilora_configs_thumbnail(request):
            """임베딩 설정의 PNG 프리셋 이미지 데이터 반환(설정 목록 마우스를 올려놓기 미리보기)）"""
            try:
                filename = request.query.get("name", "")
                category = request.query.get("category", _MLORA_DEFAULT_CATEGORY)
                safe_filename = _mlora_safe_name(filename)
                if not safe_filename or not safe_filename.lower().endswith(".png"):
                    return web.Response(status=400, text="Not a PNG file")
                cfg_dir = _get_multilora_configs_dir()
                cat_dir = _mlora_category_dir(cfg_dir, category)
                file_path = os.path.join(cat_dir, safe_filename)
                if not os.path.exists(file_path):
                    return web.Response(status=404, text="File not found")
                with open(file_path, "rb") as f:
                    image_data = f.read()
                return web.Response(
                    body=image_data,
                    content_type="image/png",
                    headers={"Cache-Control": "no-cache"}
                )
            except Exception as e:
                return web.Response(status=500, text=str(e))

        @server.routes.post("/comfyui_panel/multilora_configs/embed_image")
        async def multilora_configs_embed_image(request):
            """사용자 제공 이미지에 특정 프리셋(.json 또는 .png)을 삽입하고 .png 프리셋으로 저장"""
            try:
                from PIL import Image as _Image, PngImagePlugin
                import io as _io
                import base64 as _b64
                import zlib as _zlib

                data = await request.json()
                filename = _mlora_safe_name(data.get("name", ""))
                category = data.get("category", _MLORA_DEFAULT_CATEGORY) or _MLORA_DEFAULT_CATEGORY
                image_data = data.get("image")
                if not filename or not image_data:
                    return web.json_response({"success": False, "error": "파일 이름 또는 이미지 부족"})

                cfg_dir = _get_multilora_configs_dir()
                cat_dir = _mlora_category_dir(cfg_dir, category)
                src_path = os.path.join(cat_dir, filename)
                if not os.path.exists(src_path):
                    return web.json_response({"success": False, "error": "파일이 존재하지 않습니다"})

                # 기존 설정 읽기 (.json 또는 내장된 설정) .png）
                config = _mlora_read_config_file(src_path)
                if config is None:
                    return web.json_response({"success": False, "error": "설정 읽기 실패"})

                # base64 이미지 파싱
                if image_data.startswith("data:image"):
                    image_data = image_data.split(",", 1)[1]
                img = _Image.open(_io.BytesIO(_b64.b64decode(image_data)))

                # 두 배 보장: 최대 512로 등비율로 스케일링 (이미 프론트엔드에서 스케일링됨)）
                max_size = 512
                if img.width > max_size or img.height > max_size:
                    ratio = min(max_size / img.width, max_size / img.height)
                    img = img.resize((int(img.width * ratio), int(img.height * ratio)), _Image.LANCZOS)
                if img.mode != "RGB":
                    img = img.convert("RGB")

                base_name = filename.rsplit(".", 1)[0]
                png_filename = base_name + ".png"
                png_path = os.path.join(cat_dir, png_filename)

                # 압축된 설정을 PNG에 씁니다. tEXt chunk
                metadata = PngImagePlugin.PngInfo()
                json_content = json.dumps(config, ensure_ascii=False, indent=2)
                compressed = _zlib.compress(json_content.encode("utf-8"))
                metadata.add_text("config_json", _b64.b64encode(compressed).decode("ascii"))
                metadata.add_text("config_compressed", "1")
                img.save(png_path, "PNG", pnginfo=metadata)

                # 원래 파일이 동일한 .png가 아니면 삭제（.json -> .png 변환）
                if os.path.abspath(src_path) != os.path.abspath(png_path) and os.path.exists(src_path):
                    os.remove(src_path)

                return web.json_response({"success": True, "old_name": filename, "new_name": png_filename})
            except Exception as e:
                import traceback
                traceback.print_exc()
                return web.json_response({"success": False, "error": str(e)})

        # ===== 병합 여러 LoRA / LoKr를 하나의 표준 LoRA로 (bsk_MultiLoraLoaderWithPath 동반)）=====
        # 표준 LoRA(comfy: lora_A/lora_B) 및 LoKr(kron 분해) 형식 지원，
        # 각 모듈의 강도(strength_model / strength_clip)를 가중 합산 ΔW，
        # SVD 재 추출을 통해 지정된 순위의 표준 LoRA를 생성하거나, 동일한 표준 LoRA를 무손실 concat 모드로 처리。
        # 수학 합치는 엄격하게 상위 구현과 일치 (다음 프로젝트에서 파생, 알고리즘만 재작성, 소스 코드 복사하지 않음)）：
        #   - ComfyUI (GPL-3.0, https://github.com/Comfy-Org/ComfyUI)
        #        comfy/weight_adapter/lora.py、comfy/weight_adapter/lokr.py 의 ΔW 공식 및 키 이름 약속
        #   - sd-scripts (Apache-2.0, https://github.com/kohya-ss/sd-scripts)
        #        networks/svd_merge_lora.py（SVD 병합 아이디어）、networks/lokr.py、
        #        networks/convert_anima_lora_to_comfy.py（lora_te 키 이름 매핑）
        # 결과물 출력 ComfyUI 원본 표준 LoRA 형식（xxx.lora_B.weight + xxx.lora_A.weight + xxx.alpha），
        # comfy.sd.load_lora_for_models로 직접 로드 가능。
        # ===== 병합 작업: 백그라운드 스레드 실행 + SSE 진행률(블로킹 이벤트 루프/끊김/진행 없음)）=====
        _MERGE_TASKS = {}                       # task_id -> 작업 유형 접두사
        _MERGE_TASKS_LOCK = _threading.Lock()
        _MERGE_TASKS_MAX = 64

        def _merge_emit(task, stage, percent, current=""):
            task["stage"] = stage
            task["percent"] = percent
            task["current"] = current

        def _bsk_merge_core(data, task):
            import re
            from safetensors.torch import save_file
            import comfy.utils as _cu

            loras_req = data.get("loras") or []
            mode = (data.get("mode") or "svd").lower()
            new_rank = int(data.get("new_rank") or 64)
            device = (data.get("device") or "cpu")
            # device GPU로 실제로 실행; CUDA가 사용되지 않으면 되돌아갑니다 cpu
            if device != "cpu" and (not torch.cuda.is_available()):
                device = "cpu"
            out_name = (data.get("output_name") or "").strip()

            # 각 입력 LoRA (무겁고, 스레드에 실행하며 진행 상황 표시)）
            files = []
            total_files = len(loras_req)
            for fi, item in enumerate(loras_req):
                name = (item.get("name") or "").strip()
                if not name or name == "None":
                    continue
                path = folder_paths.get_full_path_or_raise("loras", name)
                sd = _cu.load_torch_file(path, safe_load=True)
                strength_model = float(item.get("strength_model", 1.0) or 0.0)
                strength_clip = float(item.get("strength_clip", 1.0) or 0.0)
                if strength_model == 0.0 and strength_clip == 0.0:
                    continue
                files.append((name, sd, strength_model, strength_clip))
                _merge_emit(task, "ロード LoRA ({}/{})".format(fi + 1, total_files),
                            0.05 + 0.10 * (fi + 1) / max(total_files, 1), name)

            if not files:
                raise RuntimeError("유효하지 않은 LoRA (가중치는 모두 0이거나 파일이 누락되었습니다)）")

            # ---- 키 해석 ----
            _KIND_SUFFIXES = [
                (".lora_B.weight", "up"),
                (".lora_A.weight", "down"),
                (".lora_up.weight", "up"),
                (".lora_down.weight", "down"),
                # LyCORIS LoKr / LoHa：가중치 키가 많음**없이 .weight** 확장자
                # （…lokr_w1 / …lokr_w2 / …lokr_w1_a)와 같이 필요하면 개별적으로 나열하세요
                (".lokr_w1.weight", "lokr_w1"),
                (".lokr_w1", "lokr_w1"),
                (".lokr_w1_a.weight", "lokr_w1_a"),
                (".lokr_w1_a", "lokr_w1_a"),
                (".lokr_w1_b.weight", "lokr_w1_b"),
                (".lokr_w1_b", "lokr_w1_b"),
                (".lokr_w2.weight", "lokr_w2"),
                (".lokr_w2", "lokr_w2"),
                (".lokr_w2_a.weight", "lokr_w2_a"),
                (".lokr_w2_a", "lokr_w2_a"),
                (".lokr_w2_b.weight", "lokr_w2_b"),
                (".lokr_w2_b", "lokr_w2_b"),
                (".lokr_t2.weight", "lokr_t2"),
                (".lokr_t2", "lokr_t2"),
                (".hada_w1_a", "hada_w1_a"),
                (".hada_w1_b", "hada_w1_b"),
                (".hada_w2_a", "hada_w2_a"),
                (".hada_w2_b", "hada_w2_b"),
                (".hada_t1", "hada_t1"),
                (".hada_t2", "hada_t2"),
                (".alpha", "alpha"),
            ]

            def _parse_key(k):
                for suf, kind in _KIND_SUFFIXES:
                    if k.endswith(suf):
                        return k[: -len(suf)], kind
                return None

            def _domain(mod):
                # 텍스트 인코더(CLIP/TE) 모듈은 strength_clip를 사용하고, 나머지는 (예: DiT)를 사용합니다. strength_model
                # sd-scripts/Kohya TE 모듈 형식은 lora_te / lora_te1 / lora_te2로 시작합니다.
                if (mod.startswith("text_encoders.") or mod.startswith("text_encoder.")
                        or mod.startswith("lora_te")):
                    return "clip"
                return "model"

            def _matmul_ud(up, down):
                # up: (out, rank[, k,k]); down: (rank, in[, k,k]) -> (out, in[, k,k])
                if up.dim() == 2:
                    # 선형 LoRA(DiT/CLIP의 투영 레이어와 같이): 직접 행렬 곱 ΔW = up @ down
                    return up @ down
                # 컨벌루션 LoRA: 공간 차원을 평면화한 후 채널별 matmul
                u = up.flatten(start_dim=2)      # (out, rank, kk)
                d = down.flatten(start_dim=2)    # (rank, in, kk)
                out = torch.einsum('a b c, b d c -> a d c', u, d)
                k = list(up.shape[2:])
                return out.reshape(up.shape[0], down.shape[1], *k)

            def _reconstruct(md, s):
                # 표준 LoRA：ΔW = (alpha/rank) * (up @ down) * strength
                if "up" in md and "down" in md:
                    up = md["up"].to(torch.float32)
                    down = md["down"].to(torch.float32)
                    alpha = md.get("alpha")
                    rank = down.shape[0]
                    scale = (float(alpha.item()) if alpha is not None else float(rank)) / float(rank)
                    return _matmul_ud(up, down) * (s * scale)
                # LoKr / LoHa：ΔW = kron(w1, w2) * (alpha/dim) * strength
                # 어떤 요소가 누락된 경우(예: LyCORIS 키가 / 지원되지 않는 LoHa로 파싱되지 않음) 해당 모듈을 건너뜁니다.
                w1 = md.get("lokr_w1")
                w2 = md.get("lokr_w2")
                w1_a = md.get("lokr_w1_a"); w1_b = md.get("lokr_w1_b")
                w2_a = md.get("lokr_w2_a"); w2_b = md.get("lokr_w2_b")
                t2 = md.get("lokr_t2")
                alpha = md.get("alpha")
                # w1
                if w1 is None:
                    if w1_a is not None and w1_b is not None:
                        w1 = w1_a.to(torch.float32) @ w1_b.to(torch.float32)
                    else:
                        return None
                else:
                    w1 = w1.to(torch.float32)
                # w2
                if w2 is not None:
                    w2 = w2.to(torch.float32)
                elif t2 is not None and w2_a is not None and w2_b is not None:
                    w2 = torch.einsum("i j k l, j r, i p -> p r k l",
                                     t2.to(torch.float32),
                                     w2_b.to(torch.float32),
                                     w2_a.to(torch.float32))
                elif w2_a is not None and w2_b is not None:
                    w2 = w2_a.to(torch.float32) @ w2_b.to(torch.float32)
                else:
                    return None
                if w2.dim() == 4:
                    w1 = w1.unsqueeze(2).unsqueeze(2)
                # dim alpha 정규화에 사용 (정렬 comfy weight_adapter/lokr.py）
                if w1_a is not None:
                    dim = w1_b.shape[0]
                elif w2_a is not None:
                    dim = w2_b.shape[0]
                else:
                    dim = None
                if alpha is not None and dim is not None:
                    scale = float(alpha.item()) / float(dim)
                else:
                    scale = 1.0
                return torch.kron(w1, w2) * (s * scale)

            # ---- 각 파일의 모듈 파싱 (원래 소형 가중치만 참조, 여기에서 거대한 것을 확장하지 않음) ΔW）----
            per_file = []
            types_per_file = []
            module_sets = []
            for name, sd, sm, sc in files:
                mods = {}
                for k, v in sd.items():
                    if not isinstance(v, torch.Tensor):
                        continue
                    p = _parse_key(k)
                    if p is None:
                        continue
                    mod, kind = p
                    mods.setdefault(mod, {})[kind] = v
                per_file.append((name, sm, sc, mods))
                types_per_file.append(mods)
                module_sets.append(frozenset(mods.keys()))

            any_lokr = any(
                any(k.startswith("lokr") for k in md.keys())
                for m in types_per_file for md in m.values()
            )
            uniform = len(set(module_sets)) == 1 and len(module_sets) > 0
            use_concat = (mode == "concat") and (not any_lokr) and uniform
            used_mode = "concat" if use_concat else "svd"

            def _ds(mod, sm, sc):
                return sc if _domain(mod) == "clip" else sm

            out_sd = {}
            if use_concat:
                # 일반적인 표준 LoRA: 랭크 차원에 오류 없이 연결（concat）
                raw = {}
                for name, sm, sc, mods in per_file:
                    for mod, md in mods.items():
                        if "up" in md and "down" in md:
                            s = _ds(mod, sm, sc)
                            if s == 0.0:
                                continue
                            raw.setdefault(mod, []).append(
                                (md["up"].to(torch.float32), md["down"].to(torch.float32),
                                 s, md.get("alpha"))
                            )
                total_mods = max(len(raw), 1)
                mi = 0
                for mod, items in raw.items():
                    mi += 1
                    ups = []; downs = []; total_rank = 0
                    conv = False; conv_shape = None  # [in, k, k]
                    for up, down, s, alpha in items:
                        if down.dim() == 4 and conv_shape is None:
                            conv = True
                            conv_shape = list(down.shape[1:])
                        if down.dim() == 4:
                            up = up.flatten(start_dim=2)
                            down = down.flatten(start_dim=2)
                        rank = down.shape[0]
                        a = float(alpha.item()) if alpha is not None else float(rank)
                        scale = a / float(rank)
                        ups.append((up * (s * scale)).to(torch.float32))
                        downs.append(down.to(torch.float32))
                        total_rank += rank
                    new_up = torch.cat(ups, dim=1)    # (out, Σr)
                    new_down = torch.cat(downs, dim=0)  # (Σr, in)
                    out_sd[mod + ".lora_B.weight"] = new_up.cpu().contiguous().half()
                    out_sd[mod + ".lora_A.weight"] = new_down.cpu().contiguous().half()
                    out_sd[mod + ".alpha"] = torch.tensor(float(total_rank), dtype=torch.float32)
                    if conv and conv_shape is not None:
                        out_sd[mod + ".reshape_weight"] = torch.tensor(
                            [new_up.shape[0]] + conv_shape, dtype=torch.int64
                        )
                    _merge_emit(task, "병합 모듈 ({}/{})".format(mi, total_mods),
                                0.15 + 0.80 * mi / total_mods, mod)
            else:
                # SVD 재추출하여 지정된 rank의 표준 LoRA (LoKr / 혼합 / 컨벌루션 지원）
                # 모듈별 처리하여 동시에 모든 모듈의 거대한 ΔW 거주 메모리(LoKr 대규모 모델 레이어는) OOM）
                all_mods = set()
                for _, _, _, mods in per_file:
                    all_mods.update(mods.keys())
                total_mods = max(len(all_mods), 1)
                mi = 0
                for mod in all_mods:
                    mi += 1
                    acc = None
                    for name, sm, sc, mods in per_file:
                        if mod not in mods:
                            continue
                        md = mods[mod]
                        s = _ds(mod, sm, sc)
                        if s == 0.0:
                            continue
                        delta = _reconstruct(md, s)
                        if delta is None:
                            continue
                        if acc is None:
                            acc = delta
                        else:
                            if acc.shape != delta.shape:
                                raise RuntimeError(
                                    "형태블록 {} 다양한 LoRA에서 셰이프가 일치하지 않아 병합이 불가능합니다 (표준 LoRA와 LoKr/컨벌루션 혼용 가능성 있음)？）".format(mod))
                            acc = acc + delta
                    if acc is None:
                        continue
                    mat = acc.to(device).float(); del acc
                    is_conv = mat.dim() == 4
                    if is_conv:
                        out_dim, in_dim = mat.shape[0], mat.shape[1]
                        k = mat.shape[2:]
                        flat = mat.flatten(start_dim=1)
                    else:
                        out_dim, in_dim = mat.shape[0], mat.shape[1]
                        flat = mat
                    r = min(new_rank, in_dim, out_dim)
                    if r < 1:
                        r = 1
                    U, S, Vh = torch.linalg.svd(flat)
                    U = U[:, :r]
                    S = S[:r]
                    U = U @ torch.diag(S)
                    Vh = Vh[:r, :]
                    up = U  # (out, r)
                    down = Vh  # (r, in)
                    if is_conv:
                        up = U.reshape(out_dim, r, *([1] * len(k)))
                        down = Vh.reshape(r, in_dim, *k)
                    out_sd[mod + ".lora_B.weight"] = up.cpu().contiguous().half()
                    out_sd[mod + ".lora_A.weight"] = down.cpu().contiguous().half()
                    out_sd[mod + ".alpha"] = torch.tensor(float(r), dtype=torch.float32)
                    if is_conv:
                        out_sd[mod + ".reshape_weight"] = torch.tensor(
                            [out_dim, in_dim] + list(k), dtype=torch.int64
                        )
                    del mat, flat, U, S, Vh, up, down
                    _merge_emit(task, "SVD 재 추출 모듈 ({}/{}) [{}]".format(mi, total_mods, device),
                                0.15 + 0.80 * mi / total_mods, mod)

            # ---- 파일 이름 & 디스크에 저장 ----
            if not out_name:
                out_name = f"merged_{len(files)}lora.safetensors"
            if not out_name.lower().endswith(".safetensors"):
                out_name += ".safetensors"
            out_name = re.sub(r'[\\/:*?"<>|]', '_', out_name)
            lora_dirs = folder_paths.get_folder_paths("loras")
            if not lora_dirs:
                raise RuntimeError("loras 디렉토리를 찾을 수 없습니다")
            out_path = os.path.join(lora_dirs[0], out_name)
            metadata = {
                "format": "pt",
                "bsk_merged": "true",
                "bsk_merge_mode": used_mode,
                "bsk_merge_inputs": json.dumps([f[0] for f in files], ensure_ascii=False),
                # 각 입력 LoRA 병합 시 실제 사용된 가중치 (로더/팝업 설정과 일치)）
                "bsk_merge_weights": json.dumps(
                    [{"name": f[0], "strength_model": f[2], "strength_clip": f[3]}
                     for f in files],
                    ensure_ascii=False,
                ),
            }
            _merge_emit(task, "파일 저장", 0.98, out_name)
            save_file(out_sd, out_path, metadata=metadata)

            return {
                "success": True,
                "output_name": out_name,
                "mode": used_mode,
                "module_count": sum(1 for k in out_sd if k.endswith(".lora_B.weight")),
                "msg": f"병합됨 {len(files)} 개 LoRA -> {out_name}（모드：{used_mode}）",
            }

        def _bsk_run_merge(task_id, data):
            task = _MERGE_TASKS.get(task_id)
            if task is None:
                return
            try:
                result = _bsk_merge_core(data, task)
                task["status"] = "done"
                task["result"] = result
                task["percent"] = 1.0
                task["stage"] = "완료"
            except Exception as e:
                import traceback
                traceback.print_exc()
                task["status"] = "error"
                task["error"] = str(e)

        @server.routes.post("/comfyui_panel/merge_loras")
        async def merge_loras(request):
            try:
                import uuid as _uuid
                data = await request.json()
                loras_req = data.get("loras") or []
                if not isinstance(loras_req, list) or len(loras_req) == 0:
                    return web.json_response({"success": False, "error": "어떤 것도 제공되지 않음 LoRA"})
                # 각 LoRA 이름 파싱을 사전 검증 (내부 경로 문제로 인한 스레드 실패 방지)）
                for item in loras_req:
                    name = (item.get("name") or "").strip()
                    if not name or name == "None":
                        continue
                    try:
                        folder_paths.get_full_path_or_raise("loras", name)
                    except Exception:
                        return web.json_response({"success": False, "error": f"찾을 수 없음 LoRA: {name}"})

                # 작업 생성 및 백그라운드 스레드에서 실행 (이벤트 루프 차단 안 함)）
                task_id = _uuid.uuid4().hex
                with _MERGE_TASKS_LOCK:
                    if len(_MERGE_TASKS) >= _MERGE_TASKS_MAX:
                        _MERGE_TASKS.pop(next(iter(_MERGE_TASKS)), None)
                    _MERGE_TASKS[task_id] = {
                        "status": "pending", "percent": 0.0,
                        "stage": "대기 중", "current": "",
                        "result": None, "error": None,
                    }
                _threading.Thread(
                    target=_bsk_run_merge, args=(task_id, data), daemon=True
                ).start()
                return web.json_response({
                    "success": True,
                    "task_id": task_id,
                    "msg": "병합 작업이 시작되었으며, 진행 상황은 SSE를 통해 전송",
                })
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        @server.routes.get("/comfyui_panel/merge_loras_progress")
        async def merge_loras_progress(request):
            import asyncio as _asyncio
            import time as _time
            task_id = request.query.get("task_id")
            resp = web.StreamResponse()
            resp.headers["Content-Type"] = "text/event-stream"
            resp.headers["Cache-Control"] = "no-cache"
            resp.headers["Connection"] = "keep-alive"
            resp.headers["X-Accel-Buffering"] = "no"
            await resp.prepare(request)

            last_key = None
            last_ping = 0.0
            try:
                while True:
                    task = _MERGE_TASKS.get(task_id)
                    if task is None:
                        payload = {"type": "error", "error": "작업이 존재하지 않거나 만료되었습니다"}
                        await resp.write(b"data: " + json.dumps(payload, ensure_ascii=False).encode("utf-8") + b"\n\n")
                        break
                    status = task["status"]
                    if status == "done":
                        payload = {"type": "done"}
                        payload.update(task.get("result") or {})
                        await resp.write(b"data: " + json.dumps(payload, ensure_ascii=False).encode("utf-8") + b"\n\n")
                        break
                    if status == "error":
                        payload = {"type": "error", "error": task.get("error") or "알 수 없는 오류"}
                        await resp.write(b"data: " + json.dumps(payload, ensure_ascii=False).encode("utf-8") + b"\n\n")
                        break
                    key = (status, round(task["percent"], 4), task["stage"], task["current"])
                    if key != last_key:
                        last_key = key
                        evt = {"type": "progress", "status": status,
                               "percent": task["percent"],
                               "stage": task["stage"], "current": task["current"]}
                        await resp.write(b"data: " + json.dumps(evt, ensure_ascii=False).encode("utf-8") + b"\n\n")
                    now = _time.time()
                    if now - last_ping > 3.0:
                        last_ping = now
                        await resp.write(b": ping\n\n")
                    await _asyncio.sleep(0.25)
            except (ConnectionResetError, ConnectionAbortedError, _asyncio.CancelledError):
                pass
            return resp

        @server.routes.get("/comfyui_panel/list_configs")
        async def list_configs(request):
            """panel 디렉토리의 모든 .json 구성 파일을 나열합니다"""
            try:
                panel_dir = USER_PANEL_DIR
                if not panel_dir.exists():
                    return web.json_response({"success": True, "files": []})

                files = []
                for f in panel_dir.iterdir():
                    if f.is_file() and f.suffix.lower() == ".json":
                        # BskPanel의 비구성 JSON(인증/데이터 소스/이미지 라이브러리 인덱스/출공장 설정)을 건너뜁니다.）
                        if f.name in PANEL_RESERVED_FILES:
                            continue
                        files.append({
                            "name": f.name,
                            "display": f.stem,  # 제거 .json 표시용
                            "mtime": f.stat().st_mtime
                        })

                # 수정 날짜 기준으로 내림차순 정렬
                files.sort(key=lambda x: x["mtime"], reverse=True)

                return web.json_response({
                    "success": True,
                    "files": files
                })

            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        @server.routes.get("/comfyui_panel/load_config")
        async def load_config(request):
            """읽기 지정 구성 파일 내용"""
            try:
                filename = request.query.get("name")
                if not filename:
                    return web.json_response({"success": False, "error": "Missing filename"})

                safe_filename = os.path.basename(filename)
                panel_dir = USER_PANEL_DIR
                file_path = panel_dir / safe_filename

                if not file_path.exists():
                    return web.json_response({"success": False, "error": "File not found"})

                with open(file_path, "r", encoding="utf-8") as f:
                    config = json.load(f)

                return web.json_response({
                    "success": True,
                    "config": config
                })

            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        @server.routes.post("/comfyui_panel/delete_config")
        async def delete_config(request):
            """지정된 설정 파일 삭제"""
            try:
                data = await request.json()
                filename = data.get("filename")
                category = data.get("category", "")  # 분류 이름
                
                if not filename:
                    return web.json_response({"success": False, "error": "Missing filename"})

                safe_filename = os.path.basename(filename)
                panel_dir = USER_PANEL_DIR
                
                # 분류가 있으면 분류 폴더에서 찾음
                if category:
                    safe_category = os.path.basename(category)
                    panel_dir = panel_dir / safe_category
                
                file_path = panel_dir / safe_filename

                if not file_path.exists():
                    return web.json_response({"success": False, "error": "File not found"})

                os.remove(file_path)
                
                # PNG 파일인 경우 해당 임시 JSON 파일도 함께 삭제
                if safe_filename.lower().endswith('.png'):
                    temp_json_name = safe_filename.rsplit('.', 1)[0] + '_temp.json'
                    temp_json_path = panel_dir / temp_json_name
                    if temp_json_path.exists():
                        os.remove(temp_json_path)

                return web.json_response({"success": True, "message": f"Deleted {safe_filename}"})

            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        # ---------- 분류 관리 API ----------
        @server.routes.get("/comfyui_panel/list_categories")
        async def list_categories(request):
            """모든 카테고리 나열 (폴더）"""
            try:
                panel_dir = USER_PANEL_DIR
                if not panel_dir.exists():
                    return web.json_response({"success": True, "categories": []})

                categories = []
                for item in panel_dir.iterdir():
                    # 디렉터리(이미지 라이브러리/배경 이미지/AnimaDex 데이터 등)를 보존하는 것은 구성 요소가 아니므로 목록에 포함되지 않습니다.
                    if item.is_dir() and not item.name.startswith('.') and item.name not in PANEL_RESERVED_DIRS:
                        # 폴더 내 설정 파일 수 통계
                        config_count = len([f for f in item.iterdir() 
                                          if f.is_file() and f.suffix.lower() in ['.json', '.png']])
                        categories.append({
                            "name": item.name,
                            "count": config_count
                        })

                # 이름으로 정렬
                categories.sort(key=lambda x: x["name"])
                return web.json_response({"success": True, "categories": categories})

            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        @server.routes.post("/comfyui_panel/create_category")
        async def create_category(request):
            """새 분류 (폴더) 생성）"""
            try:
                data = await request.json()
                name = data.get("name", "").strip()
                
                if not name:
                    return web.json_response({"success": False, "error": "Category name is required"})
                
                # 이름 안전하게 처리, 경로 탐색 방지
                safe_name = os.path.basename(name)
                if not safe_name or safe_name.startswith('.'):
                    return web.json_response({"success": False, "error": "Invalid category name"})

                panel_dir = USER_PANEL_DIR
                panel_dir.mkdir(exist_ok=True)
                
                category_path = panel_dir / safe_name
                if category_path.exists():
                    return web.json_response({"success": False, "error": "Category already exists"})

                category_path.mkdir()
                return web.json_response({"success": True, "name": safe_name})

            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        @server.routes.post("/comfyui_panel/delete_category")
        async def delete_category(request):
            """카테고리 (폴더) 및 모든 콘텐츠 삭제"""
            try:
                data = await request.json()
                name = data.get("name", "").strip()
                
                if not name:
                    return web.json_response({"success": False, "error": "Category name is required"})

                safe_name = os.path.basename(name)
                panel_dir = USER_PANEL_DIR
                category_path = panel_dir / safe_name

                if not category_path.exists() or not category_path.is_dir():
                    return web.json_response({"success": False, "error": "Category not found"})

                # 폴더 및 모든 내용 삭제
                import shutil
                shutil.rmtree(category_path)
                return web.json_response({"success": True, "message": f"Deleted category: {safe_name}"})

            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        # ---------- 구성 목록 보존 API ----------
        @server.routes.get("/comfyui_panel/list_configs_v2")
        async def list_configs_v2(request):
            """지정된 범주 아래의 모든 구성 파일을 나열합니다 (JSON 및 지원)PNG）"""
            try:
                category = request.query.get("category", "")  # 분류 이름, 빈 문자열은 루트 디렉토리 의미
                # 디렉토리 보존 (이미지 라이브러리/배경 이미지/AnimaDex 데이터…)는 설정 분류가 아니므로 항상 비워짐 반환
                if os.path.basename(category) in PANEL_RESERVED_DIRS:
                    return web.json_response({"success": True, "files": [], "category": category})

                panel_dir = USER_PANEL_DIR
                if category:
                    safe_category = os.path.basename(category)
                    panel_dir = panel_dir / safe_category
                
                if not panel_dir.exists():
                    return web.json_response({"success": True, "files": [], "category": category})

                # 먼저 모든 임시 파일 수집
                temp_files = set()
                for f in panel_dir.iterdir():
                    if f.is_file() and f.name.endswith('_temp.json'):
                        # 추출 해당 PNG 파일 이름
                        png_name = f.name.replace('_temp.json', '.png')
                        temp_files.add(png_name)

                files = []
                for f in panel_dir.iterdir():
                    if f.is_file() and f.suffix.lower() in ['.json', '.png']:
                        # 임시 파일 및 숨김 파일 필터링
                        if f.name.startswith('.'):
                            continue  # 숨겨진 파일 (예로) .last_used_config.json）
                        if f.name.endswith('_temp.json'):
                            continue  # PNG설정의 임시 JSON 파일
                        
                        # 임베디드 JSON인지 확인PNG
                        is_png_config = f.suffix.lower() == '.png'
                        has_thumbnail = is_png_config
                        
                        # 확인 중인 임시 편집 파일이 있습니까?
                        has_temp = f.name in temp_files
                        
                        # 설정 유형 검사：p=패널 설정, w=워크플로우
                        config_type = "p"  # 기본 패널 설정
                        
                        file_info = {
                            "name": f.name,
                            "display": f.stem,  # 확장자 제거하여 표시
                            "type": "png" if is_png_config else "json",
                            "config_type": config_type,
                            "has_thumbnail": has_thumbnail,
                            "has_temp": has_temp,  # 임시 편집 파일이 있습니까?
                            "mtime": f.stat().st_mtime,
                            "size": f.stat().st_size
                        }
                        
                        # PNG 파일에 대한 가로 세로 비율 및 설정 유형 가져오기
                        if is_png_config:
                            try:
                                import zlib as _zlib
                                import base64 as _b64
                                with Image.open(f) as img:
                                    file_info["width"] = img.width
                                    file_info["height"] = img.height
                                    file_info["aspect_ratio"] = img.height / img.width if img.width > 0 else 1.0
                                    # 설정 유형 검사
                                    info = img.info or {}
                                    if "config_json" in info:
                                        try:
                                            json_str = info["config_json"]
                                            if info.get("config_compressed") == "1":
                                                json_str = _zlib.decompress(_b64.b64decode(json_str)).decode('utf-8')
                                            config_data = json.loads(json_str)
                                            # 구성에서 workflow 필드가 있는 경우, 순수 워크플로우인지 확인합니다
                                            if isinstance(config_data, dict):
                                                wf = config_data.get("workflow")
                                                if wf and isinstance(wf, dict):
                                                    # 워크플로우 객체가 워크플로우 형식인지 확인
                                                    if wf.get("nodes") and isinstance(wf["nodes"], list):
                                                        # 탭 등 패널 필드 있음=패널 설정이 없으면=순수 워크플로우
                                                        if config_data.get("tabs") is not None or config_data.get("cardValues") is not None:
                                                            config_type = "p"
                                                        else:
                                                            config_type = "w"
                                                    else:
                                                        # API형식 워크플로우
                                                        for key in wf:
                                                            val = wf[key]
                                                            if isinstance(val, dict) and 'class_type' in val and 'inputs' in val:
                                                                if config_data.get("tabs") is not None or config_data.get("cardValues") is not None:
                                                                    config_type = "p"
                                                                else:
                                                                    config_type = "w"
                                                                break
                                        except Exception:
                                            pass
                                    elif "workflow" in info or "prompt" in info:
                                        config_type = "w"
                                    file_info["config_type"] = config_type
                            except Exception as e:
                                print(f"[Config List] Failed to get image size for {f.name}: {e}")
                                file_info["width"] = 0
                                file_info["height"] = 0
                                file_info["aspect_ratio"] = 1.0
                        else:
                            # JSON파일 검사 구성 유형
                            try:
                                with open(f, 'r', encoding='utf-8') as jf:
                                    config_data = json.load(jf)
                                if isinstance(config_data, dict):
                                    # 워크플로우 형식 검사
                                    if config_data.get("nodes") and isinstance(config_data["nodes"], list):
                                        if config_data.get("tabs") is not None or config_data.get("cardValues") is not None:
                                            config_type = "p"
                                        else:
                                            config_type = "w"
                                    elif config_data.get("tabs") is not None or config_data.get("cardValues") is not None or config_data.get("tabNodes") is not None:
                                        config_type = "p"
                                    else:
                                        # API형식 워크플로우
                                        for key in config_data:
                                            val = config_data[key]
                                            if isinstance(val, dict) and 'class_type' in val and 'inputs' in val:
                                                config_type = "w"
                                                break
                                file_info["config_type"] = config_type
                            except Exception:
                                pass
                        
                        files.append(file_info)

                # 수정 날짜 기준으로 내림차순 정렬
                files.sort(key=lambda x: x["mtime"], reverse=True)
                return web.json_response({
                    "success": True, 
                    "files": files,
                    "category": category
                })

            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        @server.routes.get("/comfyui_panel/list_all_configs")
        async def list_all_configs(request):
            """모든 분류별 설정 파일 목록 (자동 로딩 최신 설정)）

            샘플러 프로세스 미리보기 방식 <bsk_panel>/ 루트 디렉토리 또는 그**1단계**서브 디렉토리 (카테고리) 내：
            create_category / save_config_v2 / delete_config / list_configs_v2 의 category
            항등적으로 os.path.basename으로 처리하며, 다중 레벨 디렉터리가 발생하지 않습니다.。
            따라서 여기서는 '근' + "카테고리 레이어"를 사용하고 os.scandir을 DirEntry：
            Windows 상위 디렉터리에서 자체 size/mtime를 나열하며, 각 파일에 대해 개별적으로 요청할 필요 없음 stat syscall。

            오래된 구현은 전체 bsk_panel에 대해 재귀적으로 os.walk + Path.stat()，저장합니다
            prompt_gallery/{images,thumbs}、backgrounds/、animadex-cache/thumbs 등
            수천 장의 비 구성 이미지도 stat 한 번 실행 (현실 테스트 결과 9.5초). 하지만 동기 프로세서입니다.，
            aiohttp 이벤트 루프가 실행 중인 동안 채워짐 → 동일 페이지의 다른 요청이 모두 죽음
            （4ms 의 인터페이스가 변경되었습니다 8s+、심장 박동이 3초 초과로 중단되어(패널 첫 화면이 그만큼 비어 있음)。

            또한, PANEL_RESERVED_DIRS / PANEL_RESERVED_FILES에 이미지 라이브러리 및 배경 이미지를 기록、
            AnimaDex 데이터, 증명서 등 'bsk_panel 내의 기능 데이터'는 설정으로 간주하지 않는다——
            그들이 목록에 섞여서 "최신 설정"으로 자동 로드되고 "데이터 유형을 인식할 수 없음" 오류를 발생시킨다」。
            """
            try:
                panel_dir = USER_PANEL_DIR
                if not panel_dir.exists():
                    return web.json_response({"success": True, "files": []})

                all_files = []

                def _scan(entries, category):
                    """디렉토리 스캔 (재귀적이지 않음), 수집 .json / .png"""
                    # PNG 설정의 부가적인 임시 JSON：xxx.png ↔ xxx_temp.json
                    temp_pngs = set()
                    for e in entries:
                        if e.name.endswith('_temp.json'):
                            temp_pngs.add(e.name.replace('_temp.json', '.png'))

                    for e in entries:
                        if e.name.startswith('.') or e.name.endswith('_temp.json'):
                            continue
                        # 파일(증명서/데이터 소스/이미지 라이브러리 인덱스/출고 설정 및 배경 이미지) 보관은 설정이 아닙니다.
                        if e.name in PANEL_RESERVED_FILES:
                            continue
                        ext = os.path.splitext(e.name)[1].lower()
                        if ext not in ('.json', '.png'):
                            continue
                        try:
                            if not e.is_file():   # 동일한 디렉토리에서 가져온 정보 재사용
                                continue
                            st = e.stat()
                        except OSError:
                            continue
                        all_files.append({
                            "name": e.name,
                            "display": os.path.splitext(e.name)[0],
                            "type": "png" if ext == '.png' else "json",
                            "category": category,
                            "has_temp": e.name in temp_pngs,
                            "mtime": st.st_mtime,
                            "size": st.st_size
                        })

                try:
                    with os.scandir(panel_dir) as _it:
                        root_entries = list(_it)
                except OSError:
                    root_entries = []

                _scan(root_entries, "")
                # 분류 디렉토리 (한 층만 하위로 드래그); 디렉토리가 분류가 아니면 전체를 건너뜁니다
                for e in root_entries:
                    if e.name.startswith('.') or e.name in PANEL_RESERVED_DIRS:
                        continue
                    try:
                        if not e.is_dir():
                            continue
                        with os.scandir(e.path) as _it2:
                            _scan(list(_it2), e.name)
                    except OSError:
                        continue

                # 수정 날짜 기준으로 내림차순 정렬
                all_files.sort(key=lambda x: x["mtime"], reverse=True)
                return web.json_response({
                    "success": True,
                    "files": all_files
                })

            except Exception as e:
                import traceback
                traceback.print_exc()
                return web.json_response({"success": False, "error": str(e)})

        # ---------- PNGJSON 관련 내장 API ----------
        @server.routes.get("/comfyui_panel/config_thumbnail")
        async def get_config_thumbnail(request):
            """PNG 설정 파일 썸네일 가져오기"""
            try:
                filename = request.query.get("name")
                category = request.query.get("category", "")
                
                if not filename:
                    return web.Response(status=400, text="Missing filename")

                safe_filename = os.path.basename(filename)
                panel_dir = USER_PANEL_DIR
                
                if category:
                    safe_category = os.path.basename(category)
                    panel_dir = panel_dir / safe_category
                
                file_path = panel_dir / safe_filename

                if not file_path.exists():
                    return web.Response(status=404, text="File not found")

                if not safe_filename.lower().endswith('.png'):
                    return web.Response(status=400, text="Not a PNG file")

                # PNG 이미지 데이터 직접 반환
                with open(file_path, 'rb') as f:
                    image_data = f.read()

                return web.Response(
                    body=image_data,
                    content_type='image/png',
                    headers={
                        'Cache-Control': 'no-cache',
                        'Content-Disposition': f'inline; filename="{safe_filename}"'
                    }
                )

            except Exception as e:
                import traceback
                traceback.print_exc()
                return web.Response(status=500, text=str(e))

        @server.routes.post("/comfyui_panel/json_to_png")
        async def json_to_png(request):
            """JSON 설정 파일을 임베디드 JSON의 PNG 이미지로 변환"""
            try:
                from PIL import Image
                import io
                import base64
                import zlib
                
                data = await request.json()
                filename = data.get("filename")  # 원래 JSON 파일 이름
                category = data.get("category", "")
                image_data = data.get("image")  # base64인코딩된 이미지 데이터
                
                if not filename or not image_data:
                    return web.json_response({"success": False, "error": "Missing filename or image"})

                safe_filename = os.path.basename(filename)
                panel_dir = USER_PANEL_DIR
                
                if category:
                    safe_category = os.path.basename(category)
                    panel_dir = panel_dir / safe_category
                
                # 읽기 원본 JSON 파일
                json_path = panel_dir / safe_filename
                if not json_path.exists() or not safe_filename.lower().endswith('.json'):
                    return web.json_response({"success": False, "error": "JSON file not found"})

                with open(json_path, 'r', encoding='utf-8') as f:
                    json_content = f.read()

                # base64 이미지 데이터 파싱
                if image_data.startswith('data:image'):
                    image_data = image_data.split(',', 1)[1]
                
                image_bytes = base64.b64decode(image_data)
                img = Image.open(io.BytesIO(image_bytes))
                
                # 두 배 보장: 최대 비율로 스케일링512x512
                # 참고: 프론트엔드는 이미 크기를 조정했으며, 이 기능은 다른 호출 소스에서 큰 이미지를 전달하는 안전 조치입니다.
                max_size = 512
                if img.width > max_size or img.height > max_size:
                    ratio = min(max_size / img.width, max_size / img.height)
                    new_width = int(img.width * ratio)
                    new_height = int(img.height * ratio)
                    img = img.resize((new_width, new_height), Image.LANCZOS)
                
                # RGB 모드인지 확인
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # JSON을 PNG에 삽입합니다tEXt chunk
                png_filename = safe_filename.rsplit('.', 1)[0] + '.png'
                png_path = panel_dir / png_filename
                
                # PNG 정보 생성
                from PIL import PngImagePlugin
                metadata = PngImagePlugin.PngInfo()
                # 압축 JSON 데이터
                compressed_json = zlib.compress(json_content.encode('utf-8'))
                metadata.add_text("config_json", base64.b64encode(compressed_json).decode('ascii'))
                metadata.add_text("config_compressed", "1")
                
                # 저장PNG
                img.save(png_path, "PNG", pnginfo=metadata)
                
                # 삭제 원본 JSON 파일
                os.remove(json_path)
                
                return web.json_response({
                    "success": True,
                    "old_name": safe_filename,
                    "new_name": png_filename,
                    "message": f"Converted {safe_filename} to {png_filename}"
                })

            except Exception as e:
                import traceback
                traceback.print_exc()
                return web.json_response({"success": False, "error": str(e)})

        @server.routes.post("/comfyui_panel/separate_mask")
        async def separate_mask(request):
            """PNG 이미지의 마스크와 이미지를 분리하고, 투명 픽셀의 RGB 값을 올바르게 처리합니다."""
            try:
                from PIL import Image
                import io
                import base64
                
                # 읽기 업로드된 파일 데이터
                reader = await request.multipart()
                
                image_file = None
                while True:
                    field = await reader.next()
                    if not field:
                        break
                    if field.name == 'image':
                        image_file = field
                        break
                
                if not image_file:
                    return web.json_response({"success": False, "error": "No image file provided"})
                
                # 이미지 데이터 읽기
                image_bytes = await image_file.read()
                
                # PIL로 이미지 열기
                img = Image.open(io.BytesIO(image_bytes))
                
                # RGBA 모드로 확인
                if img.mode != 'RGBA':
                    img = img.convert('RGBA')
                
                # RGBA 채널 분리
                r, g, b, a = img.split()
                
                # RGB 이미지를 생성합니다 (원래 RGB 값을 유지하며, 심지어alpha=0）
                rgb_img = Image.merge('RGB', (r, g, b))
                
                # Alpha채널로서mask
                mask_img = a
                
                # RGB 및 마스크를base64
                rgb_buffer = io.BytesIO()
                rgb_img.save(rgb_buffer, format='PNG')
                rgb_base64 = base64.b64encode(rgb_buffer.getvalue()).decode('utf-8')
                rgb_data_url = f"data:image/png;base64,{rgb_base64}"
                
                mask_buffer = io.BytesIO()
                mask_img.save(mask_buffer, format='PNG')
                mask_base64 = base64.b64encode(mask_buffer.getvalue()).decode('utf-8')
                mask_data_url = f"data:image/png;base64,{mask_base64}"
                
                # 투명 영역의 RGB 값 확인
                rgb_pixels = list(rgb_img.getdata())
                alpha_pixels = list(a.getdata())
                
                # 첫 번째 투명 픽셀의 RGB 값 확인
                preserved_rgb = None
                for i, alpha_val in enumerate(alpha_pixels):
                    if alpha_val == 0:
                        preserved_rgb = rgb_pixels[i]
                        break
                
                return web.json_response({
                    "success": True,
                    "image": rgb_data_url,
                    "mask": mask_data_url,
                    "dimensions": {
                        "width": img.width,
                        "height": img.height
                    },
                    "preserved_rgb": preserved_rgb,
                    "message": "Mask separated successfully with RGB values preserved"
                })
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                return web.json_response({"success": False, "error": str(e)})

        @server.routes.post("/comfyui_panel/save_masked_image")
        async def save_masked_image(request):
            """마스크와 이미지를 PNG로 합성하고 입력 디렉토리에 저장"""
            try:
                from PIL import Image
                import io
                import base64
                
                # 읽기 업로드된 파일 데이터
                reader = await request.multipart()
                
                image_data = None
                mask_data = None
                filename = None
                overwrite = False
                
                while True:
                    field = await reader.next()
                    if not field:
                        break
                    if field.name == 'image':
                        image_data = await field.read()
                    elif field.name == 'mask':
                        mask_data = await field.read()
                    elif field.name == 'filename':
                        filename = (await field.read()).decode('utf-8')
                    elif field.name == 'overwrite':
                        overwrite = (await field.read()).decode('utf-8').lower() == 'true'
                
                if not image_data or not mask_data or not filename:
                    return web.json_response({"success": False, "error": "Missing image, mask or filename"})
                
                # === 디버그: 프론트에서 받은 원시 이미지를 저장합니다 ===
                if DEBUG_MASK_ALPHA:
                    try:
                        debug_save_backup("01_frontend_image", image_data, "프론트엔드에서 전송된 RGB 이미지")
                        debug_save_backup("02_frontend_mask", mask_data, "프론트엔드에서 전송됨mask")
                    except Exception as e:
                        print(f"[MaskAlphaDebug] 프론트엔드 데이터 저장 실패: {e}")
                
                # PIL로 이미지 열기
                img = Image.open(io.BytesIO(image_data))
                mask = Image.open(io.BytesIO(mask_data))
                
                # 확인하여 이미지가 RGB 모드로 되어 있는지 확인합니다(원래 색상 값 유지, 투명 영역은 검정으로 채워지지 않음)）
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                if mask.mode != 'L':
                    mask = mask.convert('L')
                
                # 조정 마스크 크기를 이미지에 맞게
                if mask.size != img.size:
                    mask = mask.resize(img.size, Image.LANCZOS)
                
                # 분리 RGB 채널
                r, g, b = img.split()
                
                # 사용 마스크를 알파 채널로 합성하여 RGBA 이미지를 만듭니다
                rgba_img = Image.merge('RGBA', (r, g, b, mask))
                
                # === 디버깅: 합성된 RGBA 이미지 저장 ===
                if DEBUG_MASK_ALPHA:
                    try:
                        import tempfile
                        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                            rgba_img.save(tmp.name, format='PNG')
                            debug_save_backup("03_backend_composed", tmp.name, "백엔드 합성된 RGBA 이미지")
                            result = debug_verify_png(tmp.name)
                            print(f"[MaskAlphaDebug] 백엔드 합성 유효성 검사: {result}")
                            os.unlink(tmp.name)
                    except Exception as e:
                        print(f"[MaskAlphaDebug] 합성 이미지 저장 실패: {e}")
                
                # 입력 디렉터리에 저장
                input_dir = folder_paths.get_input_directory()
                safe_filename = os.path.basename(filename)
                file_path = os.path.join(input_dir, safe_filename)
                
                # 덮어쓰기 모드이고 파일이 이미 존재하면 먼저 기존 파일 삭제
                if overwrite and os.path.exists(file_path):
                    os.remove(file_path)
                
                # PNG 파일 저장
                rgba_img.save(file_path, format='PNG')
                
                # === 디버그: 최종 저장 파일 유효성 검증 ===
                if DEBUG_MASK_ALPHA:
                    try:
                        debug_save_backup("04_final_saved", file_path, "최종 저장 파일")
                        result = debug_verify_png(file_path)
                        print(f"[MaskAlphaDebug] 최종 파일 검증: {result}")
                        debug_create_report()
                    except Exception as e:
                        print(f"[MaskAlphaDebug] 최종 파일 유효성 검사 실패: {e}")
                
                return web.json_response({
                    "success": True,
                    "filename": safe_filename,
                    "path": file_path,
                    "size": os.path.getsize(file_path),
                    "message": "Masked image saved successfully"
                })
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                return web.json_response({"success": False, "error": str(e)})

        @server.routes.post("/comfyui_panel/embed_config_to_image")
        async def embed_config_to_image(request):
            """설정을 클립보드 이미지에 내장하여 원본 설정 파일 대체"""
            try:
                from PIL import Image
                import io
                import base64
                import zlib
                
                data = await request.json()
                filename = data.get("filename")  # 원래 설정 파일 이름
                category = data.get("category", "")
                image_data = data.get("image")  # base64인코딩된 이미지 데이터
                config = data.get("config")  # 설정 데이터
                
                if not filename or not image_data or not config:
                    return web.json_response({"success": False, "error": "Missing filename, image or config"})

                safe_filename = os.path.basename(filename)
                panel_dir = USER_PANEL_DIR
                
                if category:
                    safe_category = os.path.basename(category)
                    panel_dir = panel_dir / safe_category
                
                # 폴더가 존재하도록 보장
                panel_dir.mkdir(parents=True, exist_ok=True)
                
                # base64 이미지 데이터 파싱
                if image_data.startswith('data:image'):
                    image_data = image_data.split(',', 1)[1]
                
                image_bytes = base64.b64decode(image_data)
                img = Image.open(io.BytesIO(image_bytes))
                
                # 두 배 보장: 최대 비율로 스케일링512x512
                # 참고: 프론트엔드는 이미 크기를 조정했으며, 이 기능은 다른 호출 소스에서 큰 이미지를 전달하는 안전 조치입니다.
                max_size = 512
                if img.width > max_size or img.height > max_size:
                    ratio = min(max_size / img.width, max_size / img.height)
                    new_width = int(img.width * ratio)
                    new_height = int(img.height * ratio)
                    img = img.resize((new_width, new_height), Image.LANCZOS)
                
                # RGB 모드인지 확인
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # 확인 새로운 PNG 파일 이름
                base_name = safe_filename.rsplit('.', 1)[0]
                png_filename = base_name + '.png'
                png_path = panel_dir / png_filename
                
                # PNG 정보 생성
                from PIL import PngImagePlugin
                metadata = PngImagePlugin.PngInfo()
                # 압축 JSON 데이터
                json_content = json.dumps(config, ensure_ascii=False, indent=2)
                compressed_json = zlib.compress(json_content.encode('utf-8'))
                metadata.add_text("config_json", base64.b64encode(compressed_json).decode('ascii'))
                metadata.add_text("config_compressed", "1")
                
                # 저장PNG
                img.save(png_path, "PNG", pnginfo=metadata)
                
                # 원본 파일이 PNG가 아니면 원본 파일 삭제
                old_path = panel_dir / safe_filename
                if old_path.exists() and safe_filename.lower() != png_filename.lower():
                    os.remove(old_path)
                
                return web.json_response({
                    "success": True,
                    "old_name": safe_filename,
                    "new_name": png_filename,
                    "message": f"Embedded config into {png_filename}"
                })

            except Exception as e:
                import traceback
                traceback.print_exc()
                return web.json_response({"success": False, "error": str(e)})

        @server.routes.post("/comfyui_panel/update_png_config")
        async def update_png_config(request):
            """PNG 파일의 구성 데이터를 업데이트합니다(원본 이미지를 유지)）"""
            try:
                from PIL import Image
                from PIL import PngImagePlugin
                import base64
                import zlib
                
                data = await request.json()
                filename = data.get("filename")  # PNG파일명
                category = data.get("category", "")
                config = data.get("config")  # 새 구성 데이터
                
                if not filename or not config:
                    return web.json_response({"success": False, "error": "Missing filename or config"})

                safe_filename = os.path.basename(filename)
                
                # PNG 파일인지 확인
                if not safe_filename.lower().endswith('.png'):
                    return web.json_response({"success": False, "error": "Not a PNG file"})
                
                panel_dir = USER_PANEL_DIR
                
                if category:
                    safe_category = os.path.basename(category)
                    panel_dir = panel_dir / safe_category
                
                file_path = panel_dir / safe_filename

                if not file_path.exists():
                    return web.json_response({"success": False, "error": "File not found"})

                # 원래 PNG 이미지 열기
                img = Image.open(file_path)
                
                # RGB 모드인지 확인
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # 새 PNG 메타데이터 생성
                metadata = PngImagePlugin.PngInfo()
                # 압축 JSON 데이터
                json_content = json.dumps(config, ensure_ascii=False, indent=2)
                compressed_json = zlib.compress(json_content.encode('utf-8'))
                metadata.add_text("config_json", base64.b64encode(compressed_json).decode('ascii'))
                metadata.add_text("config_compressed", "1")
                
                # PNG 저장 (원본 덮어쓰기）
                img.save(file_path, "PNG", pnginfo=metadata)
                
                return web.json_response({
                    "success": True,
                    "filename": safe_filename,
                    "message": f"Updated config in {safe_filename}"
                })

            except Exception as e:
                import traceback
                traceback.print_exc()
                return web.json_response({"success": False, "error": str(e)})

        # ---------- 보존된 구성 로드 API ----------
        @server.routes.get("/comfyui_panel/load_config_v2")
        async def load_config_v2(request):
            """구성 파일 내용 읽기 (JSON 및 내장 JSON 지원)PNG）"""
            try:
                import zlib
                import base64
                
                filename = request.query.get("name")
                category = request.query.get("category", "")
                
                if not filename:
                    return web.json_response({"success": False, "error": "Missing filename"})

                safe_filename = os.path.basename(filename)
                panel_dir = USER_PANEL_DIR
                
                if category:
                    safe_category = os.path.basename(category)
                    panel_dir = panel_dir / safe_category
                
                file_path = panel_dir / safe_filename

                if not file_path.exists():
                    return web.json_response({"success": False, "error": "File not found"})

                # 파일 유형별 처리
                if safe_filename.lower().endswith('.png'):
                    # PNG에서 데이터 추출
                    from PIL import Image
                    from PIL import PngImagePlugin
                    
                    img = Image.open(file_path)
                    info = img.info or {}
                    
                    result = {
                        "success": True,
                        "filename": safe_filename,
                        "category": category
                    }
                    
                    # 패널 설정 확인 우선 (config_json)
                    if "config_json" in info:
                        json_str = info["config_json"]
                        # 압축 확인
                        if info.get("config_compressed") == "1":
                            json_bytes = base64.b64decode(json_str)
                            json_str = zlib.decompress(json_bytes).decode('utf-8')
                        result["config"] = json.loads(json_str)
                        result["config_type"] = "p"
                    
                    # 워크플로우 데이터 확인 (workflow/prompt)
                    if "workflow" in info:
                        try:
                            result["workflow"] = json.loads(info["workflow"])
                            if "config_type" not in result:
                                result["config_type"] = "w"
                        except Exception:
                            pass
                    elif "prompt" in info:
                        try:
                            result["workflow"] = json.loads(info["prompt"])
                            if "config_type" not in result:
                                result["config_type"] = "w"
                        except Exception:
                            pass
                    
                    if "config" not in result and "workflow" not in result:
                        return web.json_response({
                            "success": False, 
                            "error": "No config or workflow data found in PNG"
                        })
                    
                    return web.json_response(result)
                else:
                    # 일반 JSON 파일
                    with open(file_path, "r", encoding="utf-8") as f:
                        config = json.load(f)

                    return web.json_response({
                        "success": True,
                        "config": config,
                        "filename": safe_filename,
                        "category": category
                    })

            except Exception as e:
                import traceback
                traceback.print_exc()
                return web.json_response({"success": False, "error": str(e)})

        # ---------- 보존된 구성 저장 API ----------
        @server.routes.post("/comfyui_panel/save_config_v2")
        async def save_config_v2(request):
            """설정을 지정된 카테고리로 저장"""
            try:
                data = await request.json()
                filename = data.get("filename")
                config = data.get("config")
                category = data.get("category", "")  # 분류 이름

                if not filename or not config:
                    return web.json_response({"success": False, "error": "Missing filename or config"})

                # 파일 이름이 안전한지 확인하세요
                safe_filename = os.path.basename(filename)
                if not safe_filename.endswith(".json"):
                    safe_filename += ".json"

                # 패널 디렉토리 존재 확인
                panel_dir = USER_PANEL_DIR
                panel_dir.mkdir(exist_ok=True)
                
                # 카테고리가 있으면 카테고리 폴더 생성
                if category:
                    safe_category = os.path.basename(category)
                    category_dir = panel_dir / safe_category
                    category_dir.mkdir(exist_ok=True)
                    file_path = category_dir / safe_filename
                else:
                    file_path = panel_dir / safe_filename

                # 파일 쓰기
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(config, f, indent=2, ensure_ascii=False)

                return web.json_response({
                    "success": True,
                    "filename": safe_filename,
                    "category": category,
                    "message": f"Config saved to {category}/{safe_filename}" if category else f"Config saved to {safe_filename}"
                })

            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        # ---------- 구성 파일 업로드(JSON 및PNG） ----------
        @server.routes.post("/comfyui_panel/upload_config_file")
        async def upload_config_file(request):
            """설정 파일 업로드 (JSON 또는 내장된 PNG), 지정된 카테고리로 저장"""
            try:
                reader = await request.multipart()
                file_data = None
                filename = None
                category = ""
                overwrite = "false"

                while True:
                    field = await reader.next()
                    if not field:
                        break
                    if field.name == 'file':
                        file_data = await field.read()
                        filename = field.filename
                    elif field.name == 'category':
                        category = (await field.read()).decode('utf-8')
                    elif field.name == 'overwrite':
                        overwrite = (await field.read()).decode('utf-8')

                if not file_data or not filename:
                    return web.json_response({"success": False, "error": "Missing file"})

                safe_filename = os.path.basename(filename)
                panel_dir = USER_PANEL_DIR
                
                if category:
                    safe_category = os.path.basename(category)
                    panel_dir = panel_dir / safe_category
                
                panel_dir.mkdir(parents=True, exist_ok=True)

                # 덮어쓰지 않으면 동일 이름 파일 확인 및 자동 이름 변경
                if overwrite != "true":
                    file_path = panel_dir / safe_filename
                    if file_path.exists():
                        base, ext = os.path.splitext(safe_filename)
                        counter = 1
                        while (panel_dir / f"{base}_{counter}{ext}").exists():
                            counter += 1
                        safe_filename = f"{base}_{counter}{ext}"

                file_path = panel_dir / safe_filename

                # 파일 유형 확인
                is_json = safe_filename.lower().endswith('.json')
                is_png = safe_filename.lower().endswith('.png')

                if not is_json and not is_png:
                    return web.json_response({"success": False, "error": "Only .json and .png files are supported"})

                # JSON 파일에 대한 내용 검증
                if is_json:
                    try:
                        config_data = json.loads(file_data.decode('utf-8'))
                        is_workflow = False
                        # 워크플로우 여부 확인
                        if isinstance(config_data, dict):
                            if config_data.get('nodes') and isinstance(config_data['nodes'], list):
                                is_workflow = True
                            else:
                                for key in config_data:
                                    val = config_data[key]
                                    if isinstance(val, dict) and 'class_type' in val and 'inputs' in val:
                                        is_workflow = True
                                        break
                        
                        if not is_workflow and not (config_data.get('tabs') is not None or config_data.get('cardValues') is not None or config_data.get('tabNodes') is not None or config_data.get('workflow') is not None):
                            return web.json_response({"success": False, "error": "File does not contain recognized workflow or panel config data"})
                    except (json.JSONDecodeError, UnicodeDecodeError) as e:
                        return web.json_response({"success": False, "error": f"Invalid JSON: {str(e)}"})

                # PNG 파일의 경우 구성 데이터를 확인합니다
                if is_png:
                    try:
                        from PIL import Image
                        import io as _io
                        import base64 as _b64
                        import zlib as _zlib
                        
                        img = Image.open(_io.BytesIO(file_data))
                        info = img.info or {}
                        
                        has_config = "config_json" in info or "workflow" in info or "prompt" in info
                        
                        if not has_config:
                            return web.json_response({"success": False, "error": "PNG file does not contain embedded config or workflow data"})
                    except Exception as e:
                        return web.json_response({"success": False, "error": f"Invalid PNG: {str(e)}"})

                # 파일 저장
                with open(file_path, 'wb') as f:
                    f.write(file_data)

                return web.json_response({
                    "success": True,
                    "filename": safe_filename,
                    "category": category,
                    "message": f"File uploaded to {category}/{safe_filename}" if category else f"File uploaded: {safe_filename}"
                })

            except Exception as e:
                import traceback
                traceback.print_exc()
                return web.json_response({"success": False, "error": str(e)})

        # -----------------------------------------
        # RGBA채널 편집기API
        # -----------------------------------------
        
        @server.routes.post("/rgba_editor/save_channel")
        async def rgba_editor_save_channel(request):
            """RGBA 채널 저장"""
            try:
                data = await request.json()
                channel_type = data.get("type")  # "r", "g", "b", "a", "rgba"
                channels = data.get("channels", {})
                output_dir = data.get("output_dir", "rgba_editor_output")
                
                # 채널 데이터 가져오기
                r_data = channels.get("r", [])
                g_data = channels.get("g", [])
                b_data = channels.get("b", [])
                a_data = channels.get("a", [])
                
                # numpy 배열로 변환하고 크기를 추론
                r_array = np.array(r_data, dtype=np.float32) if r_data else None
                g_array = np.array(g_data, dtype=np.float32) if g_data else None
                b_array = np.array(b_data, dtype=np.float32) if b_data else None
                a_array = np.array(a_data, dtype=np.float32) if a_data else None
                
                # 이미지 크기 확인
                if r_array is not None:
                    total_pixels = r_array.size
                    # 가정: 정사각형 이미지, 계산 너비 및 높이
                    side = int(np.sqrt(total_pixels))
                    if side * side == total_pixels:
                        width = height = side
                    else:
                        # 일반 크기 시도
                        if total_pixels == 256 * 256:
                            width = height = 256
                        elif total_pixels == 512 * 512:
                            width = height = 512
                        elif total_pixels == 1024 * 1024:
                            width = height = 1024
                        else:
                            # 기본256x256
                            width = height = 256
                            r_array = r_array.reshape(height, width)
                            g_array = g_array.reshape(height, width) if g_array is not None else None
                            b_array = b_array.reshape(height, width) if b_array is not None else None
                            a_array = a_array.reshape(height, width) if a_array is not None else None
                else:
                    width = height = 256
                
                # Reshape배열은2D
                if r_array is not None and r_array.ndim == 1:
                    r_array = r_array.reshape(height, width)
                if g_array is not None and g_array.ndim == 1:
                    g_array = g_array.reshape(height, width)
                if b_array is not None and b_array.ndim == 1:
                    b_array = b_array.reshape(height, width)
                if a_array is not None and a_array.ndim == 1:
                    a_array = a_array.reshape(height, width)
                
                # 출력 디렉토리 생성
                output_path = os.path.join(folder_paths.get_output_directory(), output_dir)
                os.makedirs(output_path, exist_ok=True)
                
                # 유형별로 저장
                if channel_type == "rgba":
                    if all([r_array is not None, g_array is not None, b_array is not None, a_array is not None]):
                        rgba_array = np.stack([
                            (r_array * 255).astype(np.uint8),
                            (g_array * 255).astype(np.uint8),
                            (b_array * 255).astype(np.uint8),
                            (a_array * 255).astype(np.uint8)
                        ], axis=-1)
                        
                        img = Image.fromarray(rgba_array)
                        filename = f"rgba_output_{int(time.time() * 1000)}.png"
                        filepath = os.path.join(output_path, filename)
                        img.save(filepath)
                        
                        return web.json_response({
                            "success": True,
                            "filename": filename,
                            "path": filepath,
                            "type": "rgba"
                        })
                
                elif channel_type in ["r", "g", "b", "a"]:
                    channel_map = {"r": r_array, "g": g_array, "b": b_array, "a": a_array}
                    channel_array = channel_map[channel_type]
                    
                    if channel_array is not None:
                        uint8_array = (channel_array * 255).astype(np.uint8)
                        img = Image.fromarray(uint8_array)
                        filename = f"channel_{channel_type}_{int(time.time() * 1000)}.png"
                        filepath = os.path.join(output_path, filename)
                        img.save(filepath)
                        
                        return web.json_response({
                            "success": True,
                            "filename": filename,
                            "path": filepath,
                            "type": channel_type
                        })
                
                return web.json_response({"success": False, "error": "Invalid channel type"})
                
            except Exception as e:
                print(f"[RGBA Editor] Save error: {e}")
                import traceback
                traceback.print_exc()
                return web.json_response({"success": False, "error": str(e)})

        # =========================================
        # 썸네일 캐시 기능
        # =========================================
        import hashlib
        from io import BytesIO

        # 썸네일 캐시 디렉토리
        THUMBNAIL_CACHE_DIR = Path(folder_paths.get_input_directory()) / ".thumbnail_cache"
        THUMBNAIL_MAX_WIDTH = 400   # 썸네일 최대 너비
        THUMBNAIL_MAX_HEIGHT = 650  # 썸네일 최대 높이
        THUMBNAIL_QUALITY = 80      # WebP품질
        THUMBNAIL_FORMAT = 'WEBP'   # 슬라이드 형식
        
        # 용량 제한: 20MB, 평균 40KB/장 계산 시 약 500장
        THUMBNAIL_MAX_COUNT = 500   # 최대 썸네일 수

        def get_thumbnail_path(original_filename, dir_type='input'):
            """스냅샷 캐시 경로 가져오기"""
            # 사용 파일 이름 해시를 캐시 파일 이름으로 사용
            name_hash = hashlib.md5(f"{dir_type}:{original_filename}".encode('utf-8')).hexdigest()[:16]
            return THUMBNAIL_CACHE_DIR / dir_type / f"{name_hash}.webp"

        def generate_thumbnail(original_path, thumbnail_path):
            """썸네일 생성 및 캐시에 저장。
            이미지 처리 Pillow; 비디오(mp4/webm/mov/avi/mkv/m4v)는 ffmpeg으로 첫 프레임을 추출하고, ffmpeg이 없을 경우 False 반환 (프론트엔드 백업 대체 이미지)）。"""
            ext = os.path.splitext(str(original_path))[1].lower()
            if ext in ('.mp4', '.webm', '.mov', '.avi', '.mkv', '.m4v'):
                try:
                    import shutil as _sh
                    import subprocess as _sp
                    ffmpeg = _sh.which("ffmpeg")
                    if not ffmpeg:
                        return False
                    thumbnail_path.parent.mkdir(parents=True, exist_ok=True)
                    cmd = [
                        ffmpeg, "-y", "-v", "error",
                        "-ss", "0", "-i", str(original_path),
                        "-frames:v", "1",
                        "-vf", f"scale='min({THUMBNAIL_MAX_WIDTH},iw)':-2",
                        "-f", "webp", "-quality", str(THUMBNAIL_QUALITY),
                        str(thumbnail_path),
                    ]
                    r = _sp.run(cmd, stdout=_sp.DEVNULL, stderr=_sp.DEVNULL, timeout=30)
                    return r.returncode == 0 and thumbnail_path.exists()
                except Exception as e:
                    print(f"[Thumbnail Cache] 비디오 프레임 추출 실패 {original_path}: {e}")
                    return False
            try:
                with Image.open(original_path) as img:
                    # RGB 모드로 변환 (RGBA 등 처리)）
                    if img.mode in ('RGBA', 'LA', 'P'):
                        # 흰색 배경 생성
                        background = Image.new('RGB', img.size, (255, 255, 255))
                        if img.mode == 'P':
                            img = img.convert('RGBA')
                        if img.mode in ('RGBA', 'LA'):
                            background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                            img = background
                        else:
                            img = img.convert('RGB')
                    elif img.mode != 'RGB':
                        img = img.convert('RGB')

                    # 새 크기를 계산하고 가로 세로 비율을 유지합니다
                    width, height = img.size
                    ratio = min(THUMBNAIL_MAX_WIDTH / width, THUMBNAIL_MAX_HEIGHT / height, 1.0)
                    new_width = int(width * ratio)
                    new_height = int(height * ratio)

                    # 사용 LANCZOS 재샘플링
                    img = img.resize((new_width, new_height), Image.LANCZOS)

                    # 저장WebP
                    thumbnail_path.parent.mkdir(parents=True, exist_ok=True)
                    img.save(thumbnail_path, THUMBNAIL_FORMAT, quality=THUMBNAIL_QUALITY, method=6)
                    return True
            except Exception as e:
                print(f"[Thumbnail Cache] Failed to generate thumbnail for {original_path}: {e}")
                return False

        def get_or_create_thumbnail(original_filename, dir_type='input'):
            """Get thumbnail, if not exists, generate
            
            Args:
                original_filename: 원래 파일 이름
                dir_type: 'input' 또는 'output'，기본값으로 'input'
            """
            if dir_type == 'output':
                source_dir = folder_paths.get_output_directory()
            else:
                source_dir = folder_paths.get_input_directory()
            
            original_path = os.path.join(source_dir, original_filename)

            # 검사 원본 파일이 존재하는가
            if not os.path.exists(original_path):
                return None

            # 스냅샷 경로 가져오기
            thumbnail_path = get_thumbnail_path(original_filename, dir_type)

            # 재생성 여부를 확인(캐시가 없거나 만료됨)）
            need_regenerate = True
            if thumbnail_path.exists():
                try:
                    original_mtime = os.path.getmtime(original_path)
                    thumbnail_mtime = os.path.getmtime(thumbnail_path)
                    # 캐시가 원본 파일보다 새면 캐시 사용
                    if thumbnail_mtime >= original_mtime:
                        need_regenerate = False
                except:
                    pass

            if need_regenerate:
                success = generate_thumbnail(original_path, thumbnail_path)
                if not success:
                    return None

            return thumbnail_path

        @server.routes.get("/comfyui_panel/thumbnail")
        async def get_thumbnail(request):
            """이미지 미리보기 가져오기, 브라우저 캐시 포함"""
            try:
                filename = request.query.get("filename", "")
                dir_type = request.query.get("type", "input")  # 'input' 또는 'output'
                
                if not filename:
                    return web.json_response({"success": False, "error": "No filename provided"})

                # 검증 dir_type 파라미터
                if dir_type not in ('input', 'output'):
                    dir_type = 'input'

                # 입력을 안전한 상대 경로로 지정합니다(input/output 아래의 하위 디렉터리 지원, 경로 침투 방지)）
                source_dir = folder_paths.get_output_directory() if dir_type == 'output' else folder_paths.get_input_directory()
                rel = _out_relkey(filename, source_dir)
                if not rel:
                    return web.json_response({"success": False, "error": "Invalid filename"})

                # === 수정: 잘라낸 이미지 파일 필터링 (bsk_crop_*.png) ===
                if os.path.basename(rel).startswith('bsk_crop_'):
                    return web.json_response({"success": False, "error": " Cropped files are hidden"})

                # 썸네일 생성 또는 가져오기 (인수 전달 유형, get_or_create_thumbnail 내부 join 자식 경로 지원）
                thumbnail_path = get_or_create_thumbnail(rel.replace('/', os.sep), dir_type)

                if not thumbnail_path or not thumbnail_path.exists():
                    return web.json_response({"success": False, "error": "Thumbnail not found"})

                # 스냅샷 데이터를 읽습니다
                with open(thumbnail_path, 'rb') as f:
                    image_data = f.read()

                # 캐시 헤더 (1주)）
                headers = {
                    'Cache-Control': 'public, max-age=604800, immutable',
                    'ETag': f'"{os.path.getmtime(thumbnail_path)}"',
                    'Content-Type': 'image/webp'
                }

                return web.Response(body=image_data, headers=headers)

            except Exception as e:
                print(f"[Thumbnail Cache] Error serving thumbnail: {e}")
                return web.json_response({"success": False, "error": str(e)})

        # =========================================
        # BSK Video Studio 재료 썸네일 (지원 whatdreamscost 하위 폴더) + 비디오 프레임）
        # 재사용 위의 WebP 캐시 전략: MD5 명명, mtime 기준 만료, 1주 브라우저 캐시。
        # 프론트엔드 자산 라이브러리는 이 URL 문자열만 저장하고 구성에 추가하지 않습니다. base64。
        # =========================================
        VS_ASSET_SUBFOLDER = "whatdreamscost"
        # gif 동적 콘텐츠(비디오)로 분류: 썸네일은 ffmpeg로 프레임 추출하며 정적 생성 방식이 아님
        VS_VIDEO_EXTS = {'.mp4', '.webm', '.mov', '.mkv', '.avi', '.m4v', '.gif'}
        VS_IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif', '.tiff'}

        def _vs_thumb_cache_path(filename):
            name_hash = hashlib.md5(f"vs:{filename}".encode('utf-8')).hexdigest()[:16]
            return THUMBNAIL_CACHE_DIR / "vs" / f"{name_hash}.webp"

        def _vs_extract_video_frame(video_path, out_path):
            """ffmpeg를 사용하여 첫 번째 프레임을 추출하여 WebP 미리보를 생성합니다. ffmpeg가 없는 경우 False를 반환하고, 프런트엔드는 대체 래스터를 사용합니다.。"""
            import subprocess as _sp
            import shutil as _sh
            ffmpeg = _sh.which("ffmpeg")
            if not ffmpeg:
                return False
            try:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                # -ss 0 첫 번째 프레임 가져오기; scale 비율 유지 및 확대 안 함
                cmd = [
                    ffmpeg, "-y", "-v", "error",
                    "-ss", "0", "-i", str(video_path),
                    "-frames:v", "1",
                    "-vf", f"scale='min({THUMBNAIL_MAX_WIDTH},iw)':-2",
                    "-f", "webp", "-quality", str(THUMBNAIL_QUALITY),
                    str(out_path),
                ]
                r = _sp.run(cmd, stdout=_sp.DEVNULL, stderr=_sp.DEVNULL, timeout=30)
                return r.returncode == 0 and out_path.exists()
            except Exception as e:
                print(f"[VS Thumb] ffmpeg 프레임 추출 실패 {video_path}: {e}")
                return False

        @server.routes.get("/bsk/vs/thumbnail")
        async def vs_thumbnail(request):
            """리소스 라이브러리 미리보기：/bsk/vs/thumbnail?filename=xxx.mp4&subfolder=projects/foo
            지원하는 입력/whatdreamscost/ 파일 유형.。"""
            try:
                import os as _os
                filename = request.query.get("filename", "")
                subfolder = request.query.get("subfolder", "").strip().replace("\\", "/")
                if not filename:
                    return web.json_response({"success": False, "error": "No filename provided"})
                filename = _os.path.basename(filename)
                # 전체 ‘whatdreamscost/xxx.png’를 호환 (자동 접두사 제거）
                if filename.lower().startswith("whatdreamscost/"):
                    filename = _os.path.basename(filename)
                subfolder = subfolder.replace("whatdreamscost/", "", 1).lstrip("/")
                subfolder = "/".join([p for p in subfolder.split("/") if p not in ("", ".", "..")])

                base_dir = _os.path.join(folder_paths.get_input_directory(), VS_ASSET_SUBFOLDER)
                base_abs = _os.path.abspath(base_dir)
                rel = os.path.join(subfolder.replace("/", os.sep), filename) if subfolder else filename
                original_path = _os.path.abspath(_os.path.join(base_abs, rel))
                # 경로 침투 방지
                if not original_path.startswith(base_abs + os.sep):
                    return web.json_response({"success": False, "error": "유효하지 않은 경로"})
                if not _os.path.isfile(original_path) and not _os.path.islink(original_path):
                    return web.json_response({"success": False, "error": "파일이 존재하지 않습니다"})

                ext = _os.path.splitext(filename)[1].lower()
                if ext not in VS_IMAGE_EXTS and ext not in VS_VIDEO_EXTS:
                    # 오디오 등 시각적 콘텐츠가 없으므로, 프런트엔드에서 아이콘으로 대체
                    return web.json_response({"success": False, "error": "이 유형에는 썸네일이 없습니다"})

                thumb_path = _vs_thumb_cache_path(rel)

                # 캐시 유효성: 썸네일이 원본 파일보다 오래되지 않으면 재사용
                need_regen = True
                if thumb_path.exists():
                    try:
                        if _os.path.getmtime(thumb_path) >= _os.path.getmtime(original_path):
                            need_regen = False
                    except OSError:
                        pass

                if need_regen:
                    if ext in VS_VIDEO_EXTS:
                        ok = _vs_extract_video_frame(original_path, thumb_path)
                    else:
                        ok = generate_thumbnail(original_path, thumb_path)
                    if not ok or not thumb_path.exists():
                        return web.json_response({"success": False, "error": "썸네일 생성 실패"})

                with open(thumb_path, 'rb') as f:
                    image_data = f.read()

                headers = {
                    'Cache-Control': 'public, max-age=604800',
                    'ETag': f'"{_os.path.getmtime(thumb_path)}"',
                    'Content-Type': 'image/webp',
                }
                return web.Response(body=image_data, headers=headers)
            except Exception as e:
                print(f"[VS Thumb] Error: {e}")
                return web.json_response({"success": False, "error": str(e)})

        # ========== WebP 전체 크기 미리보기 ==========
        WEBP_PREVIEW_CACHE_DIR = Path(folder_paths.get_input_directory()) / ".webp_preview_cache"
        WEBP_PREVIEW_QUALITY = 85
        WEBP_PREVIEW_MAX_COUNT = 200

        def _webp_preview_path(original_filename, dir_type='output'):
            name_hash = hashlib.md5(f"webp:{dir_type}:{original_filename}".encode('utf-8')).hexdigest()[:16]
            return WEBP_PREVIEW_CACHE_DIR / dir_type / f"{name_hash}.webp"

        def _generate_webp_preview(original_path, preview_path):
            try:
                with Image.open(original_path) as img:
                    if img.mode in ('RGBA', 'LA', 'P'):
                        background = Image.new('RGB', img.size, (255, 255, 255))
                        if img.mode == 'P':
                            img = img.convert('RGBA')
                        if img.mode in ('RGBA', 'LA'):
                            background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                            img = background
                        else:
                            img = img.convert('RGB')
                    elif img.mode != 'RGB':
                        img = img.convert('RGB')
                    preview_path.parent.mkdir(parents=True, exist_ok=True)
                    img.save(preview_path, 'WEBP', quality=WEBP_PREVIEW_QUALITY, method=6)
                    return True
            except Exception as e:
                print(f"[WebP Preview] Failed to generate for {original_path}: {e}")
                return False

        def _cleanup_webp_cache(dir_path):
            if not dir_path.exists():
                return
            files = sorted(dir_path.glob('*.webp'), key=lambda f: f.stat().st_mtime)
            if len(files) > WEBP_PREVIEW_MAX_COUNT:
                for f in files[:len(files) - WEBP_PREVIEW_MAX_COUNT]:
                    try:
                        f.unlink()
                    except Exception:
                        pass

        @server.routes.get("/comfyui_panel/webp_preview")
        async def get_webp_preview(request):
            """전체 크기 WebP 미리보기 이미지 가져오기, 독립 캐싱"""
            try:
                filename = request.query.get("filename", "")
                dir_type = request.query.get("type", "output")

                if not filename:
                    return web.json_response({"success": False, "error": "No filename provided"})

                if dir_type not in ('input', 'output'):
                    dir_type = 'output'

                source_dir = folder_paths.get_output_directory() if dir_type == 'output' else folder_paths.get_input_directory()
                rel = _out_relkey(filename, source_dir)
                if not rel:
                    return web.json_response({"success": False, "error": "Invalid filename"})

                if os.path.basename(rel).startswith('bsk_crop_'):
                    return web.json_response({"success": False, "error": "Cropped files are hidden"})

                original_path = _safe_output_path(source_dir, rel)
                if not original_path or not os.path.exists(original_path):
                    return web.json_response({"success": False, "error": "File not found"})

                preview_path = _webp_preview_path(rel.replace('/', os.sep), dir_type)

                need_regenerate = True
                if preview_path.exists():
                    try:
                        if os.path.getmtime(preview_path) >= os.path.getmtime(original_path):
                            need_regenerate = False
                    except:
                        pass

                if need_regenerate:
                    success = _generate_webp_preview(original_path, preview_path)
                    if not success:
                        return web.json_response({"success": False, "error": "WebP generation failed"})
                    _cleanup_webp_cache(preview_path.parent)

                with open(preview_path, 'rb') as f:
                    image_data = f.read()

                headers = {
                    'Cache-Control': 'public, max-age=604800, immutable',
                    'ETag': f'"{os.path.getmtime(preview_path)}"',
                    'Content-Type': 'image/webp'
                }
                return web.Response(body=image_data, headers=headers)

            except Exception as e:
                print(f"[WebP Preview] Error: {e}")
                return web.json_response({"success": False, "error": str(e)})

        @server.routes.get("/comfyui_panel/output_caption")
        async def get_output_caption(request):
            """출력 디렉터리에서 이미지 이름과 동일한 프롬프트 파일 읽기(.txt)"""
            try:
                filename = request.query.get("filename", "").strip()
                if not filename:
                    return web.json_response({"success": False, "error": "Missing filename"})
                output_dir = folder_paths.get_output_directory()
                # 안전한 상대 경로로 정규화 (서브 디렉토리 지원, 경로 침투 방지)）
                rel = _out_relkey(filename, output_dir)
                if not rel:
                    return web.json_response({"success": False, "error": "Invalid filename"})
                base = os.path.splitext(rel)[0]
                if not base:
                    return web.json_response({"success": False, "error": "Invalid filename"})
                txt_path = os.path.join(output_dir, base.replace('/', os.sep) + ".txt")
                # 확인 후 출력 디렉터리에 남아 있는지 확인
                if not os.path.realpath(txt_path).startswith(os.path.realpath(output_dir) + os.sep):
                    return web.json_response({"success": False, "error": "Invalid path"})
                if os.path.isfile(txt_path):
                    try:
                        with open(txt_path, "r", encoding="utf-8") as f:
                            caption = f.read()
                        return web.json_response({"success": True, "caption": caption, "exists": True})
                    except Exception as e:
                        return web.json_response({"success": False, "error": str(e)})
                return web.json_response({"success": True, "caption": "", "exists": False})
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        @server.routes.post("/comfyui_panel/output_caption")
        async def save_output_caption(request):
            """출력 디렉터리에 동일한 이름의 프롬프트 파일 저장(.txt)"""
            try:
                data = await request.json()
                filename = data.get("filename", "").strip()
                caption = data.get("caption", "")
                if not filename:
                    return web.json_response({"success": False, "error": "Missing filename"})
                output_dir = folder_paths.get_output_directory()
                rel = _out_relkey(filename, output_dir)
                if not rel:
                    return web.json_response({"success": False, "error": "Invalid filename"})
                base = os.path.splitext(rel)[0]
                if not base:
                    return web.json_response({"success": False, "error": "Invalid filename"})
                txt_path = os.path.join(output_dir, base.replace('/', os.sep) + ".txt")
                if not os.path.realpath(txt_path).startswith(os.path.realpath(output_dir) + os.sep):
                    return web.json_response({"success": False, "error": "Invalid path"})
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(caption)
                return web.json_response({"success": True, "file": txt_path})
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        @server.routes.post("/comfyui_panel/clear_thumbnails")
        async def clear_thumbnails(request):
            """썸네임 캐시 지우기"""
            try:
                if not THUMBNAIL_CACHE_DIR.exists():
                    return web.json_response({
                        "success": True,
                        "message": "Thumbnail cache directory does not exist",
                        "deleted_count": 0
                    })

                deleted_count = 0
                for f in THUMBNAIL_CACHE_DIR.iterdir():
                    if f.is_file() and f.suffix == '.jpg':
                        try:
                            f.unlink()
                            deleted_count += 1
                        except Exception as e:
                            print(f"[Thumbnail Cache] Failed to delete {f}: {e}")

                return web.json_response({
                    "success": True,
                    "message": f"Deleted {deleted_count} thumbnail(s)",
                    "deleted_count": deleted_count
                })

            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        # ========== 워크플로우 변환 API ==========
        # 일반 워크플로우 형식을 API 형식으로 변환하는 데 사용

        def _get_node_input_order(node_class):
            """노드 입력 순서 정보 가져오기 (프론트엔드 모방) input_order）"""
            try:
                import nodes
                if node_class not in nodes.NODE_CLASS_MAPPINGS:
                    return None
                obj_class = nodes.NODE_CLASS_MAPPINGS[node_class]
                input_types = obj_class.INPUT_TYPES()
                return {
                    'required': list(input_types.get('required', {}).keys()),
                    'optional': list(input_types.get('optional', {}).keys())
                }
            except Exception as e:
                return None

        def _is_widget_type(type_def):
            """widget 유형인지 판단합니다 (배열 형식, 첫 번째 요소는 옵션 목록 또는 기본값)）"""
            if not isinstance(type_def, list) or len(type_def) == 0:
                return False
            first = type_def[0]
            # widget 유형: ["옵션1", "옵션2", ...] 또는 [기본값, ...설정]...]
            # 연결 유형: ("Type name", ...) 如 ("IMAGE",)
            if isinstance(first, tuple):
                return False  # 이것은 연결 유형
            return True

        def _is_seed_widget(name, config):
            """seed 유형인지 판단 widget"""
            if not isinstance(config, list) or len(config) < 2:
                return False
            # seed 일반적으로 control_after_generate 옵션이 있습니다.
            if len(config) >= 2 and isinstance(config[1], dict):
                return 'control_after_generate' in config[1] or name.lower() == 'seed' or name.lower() == 'noise_seed'
            return name.lower() == 'seed' or name.lower() == 'noise_seed'

        def _get_hardcoded_widget_names(class_type, values_count):
            """내장된 위젯 이름 목록 가져오기"""
            # 미니 버전 다중 LoRA 로더：
            #   새 버전은 1개만 widget（lora_text）；
            #   구 버전은 65개의 슬롯별 위젯(widget)으로 구성되어 있었으며, 이전 레이아웃에 따라 필드 이름을 반환，
            #   프론트엔드에 _migrateMultiLoraLegacyInputs 합성 후 lora_text를 추출하여 정리합니다.。
            if class_type == 'bsk_MultiLoraLoaderWithPath':
                if values_count == 1:
                    return ['lora_text']
                legacy = ['master_switch']
                for i in range(1, 8):
                    legacy += [f'switch_{i}', f'lora_name_{i}', f'strength_model_{i}', f'strength_clip_{i}']
                for i in range(1, 8):
                    legacy += [f'batch_test_{i}', f'batch_test_min_{i}', f'batch_test_max_{i}',
                               f'batch_test_mode_{i}', f'batch_test_steps_{i}']
                legacy.append('lora_text')
                if len(legacy) == values_count:
                    return legacy
                return None

            widget_map = {
                'KSampler': ['seed', 'control_after_generate', 'steps', 'cfg', 'sampler_name', 'scheduler', 'denoise'],
                'KSampler (Efficient)': ['seed', 'control_after_generate', 'steps', 'cfg', 'sampler_name', 'scheduler', 'denoise', 'preview_method', 'vae_decode'],
                'KSamplerAdvanced': ['noise_seed', 'control_after_generate', 'steps', 'cfg', 'start_at_step', 'end_at_step', 'add_noise'],
                'CheckpointLoaderSimple': ['ckpt_name'],
                'VAELoader': ['vae_name'],
                'LoraLoader': ['lora_name', 'strength_model', 'strength_clip'],
                'LoraLoaderModelOnly': ['lora_name', 'strength_model'],
                'LoraLoaderWithPath': ['lora_name', 'strength_model', 'strength_clip', 'lora_path'],
                'ControlNetLoader': ['control_net_name'],
                'EmptyLatentImage': ['width', 'height', 'batch_size'],
                'LatentUpscale': ['upscale_method', 'width', 'height', 'crop'],
                'LatentUpscaleBy': ['upscale_method', 'scale_by'],
                'LoadImage': ['image'],
                'ImageScale': ['upscale_method', 'width', 'height', 'crop'],
                'ImageScaleBy': ['upscale_method', 'scale_by'],
                'SaveImage': ['filename_prefix'],
                'PreviewImage': [],
                'CLIPTextEncode': ['text', 'speak_and_recognation'],
                'CLIPTextEncodeSDXL': ['text_g', 'text_l', 'width', 'height', 'crop_w', 'crop_h', 'target_width', 'target_height'],
                'VAEDecode': [],
                'VAEEncode': [],
                'PrimitiveInt': ['value'],
                'PrimitiveFloat': ['value'],
                'PrimitiveString': ['value'],
                'LoadAudio': ['audio'],
                'LoadVideo': ['video'],
                'VHS_LoadVideo': ['video', 'force_rate', 'force_size', 'custom_width', 'custom_height', 'frame_load_cap', 'skip_first_frames', 'select_every_nth'],
                'VideoLoaderPlus': ['video', 'frame_load_cap', 'start_index', 'force_rate', 'force_size', 'custom_width', 'custom_height'],
                'PreviewAny': [],
                'ControlNetApplyAdvanced': ['strength', 'start_percent', 'end_percent'],
            }
            
            names = widget_map.get(class_type)
            if names and len(names) == values_count:
                return names
            return None

        def _get_extra_widgets(class_type, current_count, target_count):
            """프론트엔드에 동적으로 추가된 추가 widget"""
            extra_map = {
                'CLIPTextEncode': ['speak_and_recognation'],
                'KSampler (Efficient)': ['preview_method', 'vae_decode'],
            }
            
            extra = extra_map.get(class_type, [])
            if extra and current_count + len(extra) == target_count:
                return extra
            
            # 일반 처리: 차이가 1이면, 아마도 부울 스위치일 수 있음
            if target_count - current_count == 1:
                return ['speak_and_recognation']
            
            return []

        def _resolve_widget_names(class_type, widgets_values, node_def=None):
            """widget 이름 목록 파싱 (핵심 변환 로직)）"""
            if not widgets_values:
                return []
            
            values_count = len(widgets_values)
            
            # 1. 먼저 하드코딩 매핑 시도
            hardcoded = _get_hardcoded_widget_names(class_type, values_count)
            if hardcoded:
                return hardcoded
            
            names = []
            
            # 2. 노드 정의 사용
            if node_def:
                required = node_def.get('input', {}).get('required', {})
                optional = node_def.get('input', {}).get('optional', {})
                
                # required에서 widget 유형을 처리합니다
                for name, config in required.items():
                    if isinstance(config, list) and _is_widget_type(config):
                        names.append(name)
                        # seed 종류 후 추가 control_after_generate
                        if _is_seed_widget(name, config):
                            names.append('control_after_generate')
                
                # optional에서 widget 유형을 처리합니다
                for name, config in optional.items():
                    if isinstance(config, list) and _is_widget_type(config):
                        names.append(name)
            
            # 3. 프론트엔드 동적으로 추가된 widget
            extra = _get_extra_widgets(class_type, len(names), values_count)
            if extra:
                names.extend(extra)
            
            # 4. 수량이 일치하지 않으면 param_i로 보충
            while len(names) < values_count:
                names.append(f'param_{len(names)}')
            
            return names[:values_count]

        def _get_node_def_cached(class_type, node_defs_cache):
            """노드 정의 가져오기（포함캐시）"""
            if class_type in node_defs_cache:
                return node_defs_cache[class_type]
            
            try:
                import nodes
                if class_type not in nodes.NODE_CLASS_MAPPINGS:
                    node_defs_cache[class_type] = None
                    return None
                
                obj_class = nodes.NODE_CLASS_MAPPINGS[class_type]
                node_def = {
                    'input': obj_class.INPUT_TYPES(),
                    'name': class_type
                }
                node_defs_cache[class_type] = node_def
                return node_def
            except Exception as e:
                node_defs_cache[class_type] = None
                return None

        @server.routes.post("/comfyui_panel/convert_workflow")
        async def convert_workflow(request):
            """일반 워크플로우 형식을 API 형식으로 변환합니다
            
            입력 형식 (일반 워크플로우)）:
            {
                "nodes": [
                    {
                        "id": 1,
                        "type": "KSampler",
                        "widgets_values": [123456, "fixed", 20, 7, "euler", "normal", 1.0],
                        "inputs": [{"name": "model", "link": 1}]
                    }
                ],
                "links": [[1, 2, 0, 1, "MODEL"]]
            }
            
            출력 형식 (API 워크플로우）:
            {
                "1": {
                    "class_type": "KSampler",
                    "inputs": {
                        "seed": 123456,
                        "control_after_generate": "fixed",
                        "steps": 20,
                        ...
                    }
                }
            }
            """
            try:
                data = await request.json()
                frontend_workflow = data.get('workflow')
                
                if not frontend_workflow:
                    return web.json_response({
                        "success": False,
                        "error": "Missing 'workflow' in request body"
                    })
                
                # API 형식인지 확인
                if 'class_type' in str(frontend_workflow) and 'nodes' not in frontend_workflow:
                    return web.json_response({
                        "success": True,
                        "workflow": frontend_workflow,
                        "message": "Already in API format"
                    })
                
                nodes_data = frontend_workflow.get('nodes', [])
                links = frontend_workflow.get('links', [])
                
                # 건축 연결 매핑: link_id -> [source_node_id, source_slot, target_node_id, target_slot, type]
                link_map = {}
                for link in links:
                    if len(link) >= 5:
                        link_id = link[0]
                        link_map[link_id] = link
                
                # 노드 정의 캐시
                node_defs_cache = {}
                
                # 변환 결과
                api_workflow = {}
                
                for node in nodes_data:
                    node_id = str(node.get('id'))
                    class_type = node.get('type')
                    
                    if not class_type:
                        continue
                    
                    # 노드 정의 가져오기
                    node_def = _get_node_def_cached(class_type, node_defs_cache)
                    
                    # API 형식의 노드 생성
                    api_node = {
                        "class_type": class_type,
                        "inputs": {}
                    }
                    
                    # 처리 widgets_values
                    widgets_values = node.get('widgets_values', [])
                    if widgets_values:
                        widget_names = _resolve_widget_names(class_type, widgets_values, node_def)
                        for i, value in enumerate(widgets_values):
                            widget_name = widget_names[i] if i < len(widget_names) else f'param_{i}'
                            api_node['inputs'][widget_name] = value
                    
                    # 입력 연결 처리
                    node_inputs = node.get('inputs', [])
                    for inp in node_inputs:
                        inp_name = inp.get('name')
                        link_id = inp.get('link')
                        
                        if inp_name and link_id is not None and link_id in link_map:
                            link_info = link_map[link_id]
                            source_node_id = str(link_info[1])
                            source_slot = link_info[2]
                            # API 포맷 연결 참조
                            api_node['inputs'][inp_name] = [source_node_id, source_slot]
                    
                    api_workflow[node_id] = api_node
                
                return web.json_response({
                    "success": True,
                    "workflow": api_workflow
                })
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                return web.json_response({
                    "success": False,
                    "error": str(e)
                })

        @server.routes.get("/comfyui_panel/convert_workflow")
        async def convert_workflow_info(request):
            """반환 API 사용 설명"""
            return web.json_response({
                "success": True,
                "description": "Convert frontend workflow format to API format",
                "usage": {
                    "method": "POST",
                    "content_type": "application/json",
                    "body_example": '{"workflow": {"nodes": [...], "links": [...]}}'
                },
                "response_example": '{"success": true, "workflow": {"1": {"class_type": "KSampler", "inputs": {...}}}}'
            })

        # ---------- BSK 패널 전역 설정 (사용자 디렉토리, 업그레이드 시 덮어쓰기 방지)) ----------
        def _get_bsk_settings_path():
            """bsk_settings.json 경로를 가져옵니다: ComfyUI/user/default/bsk_settings.json"""
            try:
                base_path = folder_paths.base_path
            except Exception:
                base_path = os.path.abspath('./')
            cfg_dir = os.path.join(base_path, "user", "default")
            os.makedirs(cfg_dir, exist_ok=True)
            return os.path.join(cfg_dir, "bsk_settings.json")

        def _read_bsk_settings():
            """bsk_settings.json 읽기, 존재하지 않거나 파싱 실패 시 기본값 반환"""
            try:
                path = _get_bsk_settings_path()
                if os.path.isfile(path):
                    with open(path, 'r', encoding='utf-8') as f:
                        return json.load(f)
            except Exception as e:
                print(f"[ComfyUI Panel] Failed to read bsk_settings.json: {e}")
            # 기본값: 두 스위치 모두 끄기
            return {"nodeTrackingEnabled": False, "autoSwitchTab": False}

        def _write_bsk_settings(data):
            """쓰기 bsk_settings.json"""
            try:
                path = _get_bsk_settings_path()
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                return True
            except Exception as e:
                print(f"[ComfyUI Panel] Failed to write bsk_settings.json: {e}")
                return False

        @server.routes.get("/comfyui_panel/bsk_settings")
        async def get_bsk_settings(request):
            """BSK 패널 전역 설정 가져오기"""
            return web.json_response({"success": True, "settings": _read_bsk_settings()})

        @server.routes.post("/comfyui_panel/bsk_settings")
        async def save_bsk_settings(request):
            """BSK 패널 전역 설정 저장。

            ‘병합 쓰기’ 사용: 기존 내용 먼저 읽어내고 업데이트, 서로 다른 기능 모듈 (노드 실행 추적 /
            패널 배경 이미지 …) 각각 저장 시 다른 필드 전체 덮어쓰기。
            """
            try:
                data = await request.json()
                settings = data.get("settings", {}) or {}
                merged = _read_bsk_settings()
                if not isinstance(merged, dict):
                    merged = {}
                merged.update(settings)
                if _write_bsk_settings(merged):
                    return web.json_response({"success": True, "settings": merged})
                return web.json_response({"success": False, "error": "Write failed"})
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        # =====================================================================
        # 패널 배경 이미지 (생성 결과 / 고정 배경) + 마스크 강도）
        # ---------------------------------------------------------------------
        # 배경 이미지 폴더：<ComfyUI>/user/default/bsk_panel/backgrounds/
        # 공장 기본 이미지：<ComfyUI>/user/default/bsk_panel/H3.png，누락된 경우 롤백 플러그인 디렉토리 panel/H3.png
        # 지속성 설정: 재사용 <ComfyUI>/user/default/bsk_settings.json panelBackground 필드，
        #            새로운 설정 파일 추가 안 함 (노드 실행 추적 등 스위치와 공유)）
        # =====================================================================
        PANEL_BG_DIRNAME = "backgrounds"
        PANEL_BG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
        PANEL_BG_MAX_BYTES = 25 * 1024 * 1024
        PANEL_BG_BUILTIN_NAME = "H3.png"
        PANEL_BG_THUMB_MAX_WIDTH = 320
        PANEL_BG_DEFAULT_MASK = 0.34
        PANEL_BG_DEFAULT_BLUR = 0
        # 배경 이미지 로드용 WebP: 좁은 비율 압축 후 원본 이미지(PNG/JPG)보다 크기가 훨씬 작아 클라우드 환경에서 빠르게 로드됨
        PANEL_BG_WEBP_MAX_WIDTH = 1920
        PANEL_BG_WEBP_QUALITY = 80

        def _panel_bg_dir():
            """배경 이미지 저장 폴더 (ComfyUI 사용자 폴더, 플러그인 재설치 시 이미지 손실 X)）"""
            d = USER_PANEL_DIR / PANEL_BG_DIRNAME
            try:
                d.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                print(f"[ComfyUI Panel] 배경 이미지 디렉터리 생성 불가 {d}: {e}")
            return d

        def _panel_bg_builtin_path():
            """공장 기본 배경: 사용자 디렉토리 bsk_panel/H3.png; 존재하지 않으면 플러그인 디렉토리로 되돌림 panel/H3.png。"""
            p = USER_PANEL_DIR / PANEL_BG_BUILTIN_NAME
            if p.is_file():
                return p
            p2 = EXTENSION_DIR / "panel" / PANEL_BG_BUILTIN_NAME
            if p2.is_file():
                return p2
            return None

        def _panel_bg_safe_name(name):
            """안전한 파일 이름을 유지하고 경로 침투를 방지합니다. 빈/불법적인 반환 값은 '' (''는 공장 기본 그림)）。"""
            n = os.path.basename(str(name or "").replace("\\", "/")).strip()
            if n in ("", ".", ".."):
                return ""
            return n

        def _panel_bg_resolve(name):
            """파일 이름으로 절대 경로를 파싱하고, 존재하지 않으면 None 반환; 빈 파일 이름은 공장 기본 이미지로 파싱。"""
            n = _panel_bg_safe_name(name)
            if not n:
                return _panel_bg_builtin_path()
            p = _panel_bg_dir() / n
            return p if p.is_file() else None

        def _panel_bg_thumb_path(name):
            """배경 이미지 WebP 썸네일 캐시 경로 (파일 이름 해시, mtime 변경 시 재생성)）"""
            key = _panel_bg_safe_name(name) or "__builtin__"
            h = hashlib.md5(("panel_bg:" + key).encode("utf-8")).hexdigest()[:16]
            return THUMBNAIL_CACHE_DIR / "panel_bg" / f"{h}.webp"

        def _panel_bg_webp_path(name):
            """배경 이미지 "로드용" WebP 캐시 경로 (파일명 해시, mtime 변경 시 재생성）"""
            key = _panel_bg_safe_name(name) or "__builtin__"
            h = hashlib.md5(("panel_bg_webp:" + key).encode("utf-8")).hexdigest()[:16]
            return THUMBNAIL_CACHE_DIR / "panel_bg_webp" / f"{h}.webp"

        def _panel_bg_make_webp(src, out, max_width, quality):
            """최대 너비로 이미지를 WebP로 변환합니다. 실패하면 False 반환(호출자 원본 이미지로 되돌림)）"""
            try:
                out.parent.mkdir(parents=True, exist_ok=True)
                with Image.open(src) as im:
                    im = im.convert("RGB")
                    w, h = im.size
                    if max_width and w > max_width and w > 0:
                        h = max(1, int(h * max_width / w))
                        im = im.resize((max_width, h), Image.Resampling.LANCZOS)
                    im.save(out, "WEBP", quality=quality)
                return True
            except Exception as e:
                print(f"[ComfyUI Panel] 배경 이미지 WebP 생성 실패 {src}: {e}")
                return False

        def _panel_bg_make_thumb(src, out):
            """WebP 썸네일 생성; 실패하면 False 반환 (호출자는 원래 이미지로 되돌림)）"""
            return _panel_bg_make_webp(src, out, PANEL_BG_THUMB_MAX_WIDTH, 75)

        def _panel_bg_list_items():
            """배경 이미지 디렉토리의 이미지를 나열합니다(새로 업로드된 이미지는 맨 앞에 표시), 원본 이미지/소개 그림을 포함합니다 URL"""
            from urllib.parse import quote as _quote
            items = []
            d = _panel_bg_dir()
            try:
                names = sorted(os.listdir(d))
            except OSError:
                return items
            for n in names:
                full = d / n
                if not full.is_file() or os.path.splitext(n)[1].lower() not in PANEL_BG_EXTS:
                    continue
                try:
                    st = full.stat()
                except OSError:
                    continue
                q = _quote(n)
                items.append({
                    "file": n,
                    "size": st.st_size,
                    "mtime": int(st.st_mtime),
                    "url": f"/comfyui_panel/panel_bg/file?file={q}",
                    "thumb": f"/comfyui_panel/panel_bg/thumb?file={q}",
                })
            items.sort(key=lambda x: x["mtime"], reverse=True)
            return items

        def _panel_bg_current():
            """현재 배경 설정을 읽고 유효성을 검사/기본값으로 되돌립니다(설정 파일이 삭제된 경우 공장 초기화 기본 이미지로 돌아갑니다).）

            신규 사용자(패널 배경을 저장하지 않음)는 기본적으로 "고정 배경"을 사용하고, 기존 사용자(저장된 파라미터 유지)는 이를 유지합니다.。
            """
            raw = _read_bsk_settings().get("panelBackground")
            has_saved = isinstance(raw, dict)
            s = raw if has_saved else {}
            mode = "fixed" if (not has_saved or s.get("mode") == "fixed") else "auto"
            try:
                mask = float(s.get("mask", PANEL_BG_DEFAULT_MASK))
            except (TypeError, ValueError):
                mask = PANEL_BG_DEFAULT_MASK
            mask = max(0.0, min(1.0, mask))
            try:
                blur = float(s.get("blur", PANEL_BG_DEFAULT_BLUR))
            except (TypeError, ValueError):
                blur = PANEL_BG_DEFAULT_BLUR
            blur = max(0.0, min(60.0, blur))
            file = _panel_bg_safe_name(s.get("file", ""))
            if file and _panel_bg_resolve(file) is None:
                file = ""
            return {"mode": mode, "file": file, "mask": round(mask, 3), "blur": round(blur, 1)}

        def _panel_bg_write_current(mode=None, file=None, mask=None, blur=None):
            """배경 설정을 씁니다 (단지 전달된 필드만 변경하고 나머지는 원래 값을 유지합니다)）"""
            s = _read_bsk_settings()
            cur = s.get("panelBackground") or {}
            if mode is not None:
                cur["mode"] = "fixed" if mode == "fixed" else "auto"
            if file is not None:
                cur["file"] = _panel_bg_safe_name(file)
            if mask is not None:
                try:
                    cur["mask"] = max(0.0, min(1.0, round(float(mask), 3)))
                except (TypeError, ValueError):
                    pass
            if blur is not None:
                try:
                    cur["blur"] = max(0.0, min(60.0, round(float(blur), 1)))
                except (TypeError, ValueError):
                    pass
            cur.setdefault("mask", PANEL_BG_DEFAULT_MASK)
            cur.setdefault("blur", PANEL_BG_DEFAULT_BLUR)
            s["panelBackground"] = cur
            _write_bsk_settings(s)
            return cur

        @server.routes.get("/comfyui_panel/panel_bg/list")
        async def panel_bg_list(request):
            """배경 이미지 목록 + 현재 설정 + 공장 기본 이미지 정보"""
            try:
                builtin = _panel_bg_builtin_path()
                return web.json_response({
                    "success": True,
                    "current": _panel_bg_current(),
                    "builtin": {
                        "available": bool(builtin),
                        "name": PANEL_BG_BUILTIN_NAME,
                        "url": "/comfyui_panel/panel_bg/webp",
                        "thumb": "/comfyui_panel/panel_bg/thumb",
                    },
                    "items": _panel_bg_list_items(),
                })
            except Exception as e:
                return web.json_response({"success": False, "error": str(e), "items": []})

        @server.routes.get("/comfyui_panel/panel_bg/file")
        async def panel_bg_file(request):
            """취소 배경 원본 이미지; 파일 파라미터 없음 = 공장 기본 H3.png"""
            p = _panel_bg_resolve(request.query.get("file", ""))
            if not p:
                return web.json_response({"success": False, "error": "배경 파일이 없음"}, status=404)
            try:
                with open(p, "rb") as f:
                    data = f.read()
            except OSError as e:
                return web.json_response({"success": False, "error": str(e)}, status=404)
            ext = os.path.splitext(str(p))[1].lower()
            ctype = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                     ".webp": "image/webp", ".gif": "image/gif", ".bmp": "image/bmp"}.get(ext, "application/octet-stream")
            return web.Response(body=data, headers={
                "Content-Type": ctype,
                "Cache-Control": "public, max-age=604800, immutable",
            })

        @server.routes.get("/comfyui_panel/panel_bg/thumb")
        async def panel_bg_thumb(request):
            """배경 이미지 WebP 미리보기 (캐시 포함, 목록은 원본 이미지 반복 로딩 X)）"""
            name = request.query.get("file", "")
            src = _panel_bg_resolve(name)
            if not src:
                return web.json_response({"success": False, "error": "배경 파일이 없음"}, status=404)
            headers = {"Cache-Control": "public, max-age=604800, immutable"}
            try:
                out = _panel_bg_thumb_path(name)
                need = True
                if out.exists():
                    need = out.stat().st_mtime < src.stat().st_mtime
                if need:
                    _panel_bg_make_thumb(src, out)
                if out.exists():
                    with open(out, "rb") as f:
                        return web.Response(body=f.read(),
                                            headers={**headers, "Content-Type": "image/webp"})
            except Exception as e:
                print(f"[ComfyUI Panel] 배경 이미지 미리보기 로드 실패: {e}")
            # 썸네일 생성 실패: 원본 이미지로 되돌리고 목록이 계속 표시되도록 합니다.
            try:
                with open(src, "rb") as f:
                    return web.Response(body=f.read(), headers=headers)
            except OSError as e:
                return web.json_response({"success": False, "error": str(e)}, status=404)

        # ---------- 배경 이미지 WebP (로드 시 사용）----------
        async def _serve_panel_bg_webp(request):
            """WebP 버전의 배경 이미지를 출력 (가로 압축만 허용), 실패 시 원본 이미지로 복귀。

            배경 이미지 전체 화면 바닥에 사용, 원본 (PNG/JPG)은 보통 수 MB, 클라우드 환경에서 로드 시 딜레이 발생；
            WebP로 변환 후 크기가 보통 몇백 KB로 줄어들고, 디스크 캐시를 사용하여 첫 번째 요청 시에만 생성합니다.。
            """
            name = request.query.get("file", "")
            src = _panel_bg_resolve(name)
            if not src:
                return web.json_response({"success": False, "error": "배경 파일이 없음"}, status=404)
            headers = {"Cache-Control": "public, max-age=604800, immutable"}
            try:
                out = _panel_bg_webp_path(name)
                need = True
                if out.exists():
                    need = out.stat().st_mtime < src.stat().st_mtime
                if need:
                    _panel_bg_make_webp(src, out, PANEL_BG_WEBP_MAX_WIDTH, PANEL_BG_WEBP_QUALITY)
                if out.exists():
                    with open(out, "rb") as f:
                        return web.Response(body=f.read(),
                                            headers={**headers, "Content-Type": "image/webp"})
            except Exception as e:
                print(f"[ComfyUI Panel] 배경 이미지 WebP 읽기 실패: {e}")
            # 변환 실패: 원래 이미지로 되돌아가
            try:
                with open(src, "rb") as f:
                    return web.Response(body=f.read(), headers=headers)
            except OSError as e:
                return web.json_response({"success": False, "error": str(e)}, status=404)

        @server.routes.get("/comfyui_panel/panel_bg/webp")
        async def panel_bg_webp(request):
            return await _serve_panel_bg_webp(request)

        def _panel_bg_save_bytes(filename, data):
            """업로드된 이미지 바이트를 배경 이미지 폴더에 쓰고, 중복 제거된 안전한 파일 이름을 반환"""
            filename = filename or "background.png"
            ext = os.path.splitext(filename)[1].lower()
            if ext == ".jpeg":
                ext = ".jpg"
            if ext not in PANEL_BG_EXTS:
                ext = ".png"
            base = re.sub(r"[^\w.\-]+", "_", os.path.splitext(os.path.basename(filename))[0],
                          flags=re.UNICODE)[:60] or "background"
            d = _panel_bg_dir()
            name = base + ext
            i = 1
            while (d / name).exists():
                name = f"{base}_{i}{ext}"
                i += 1
            with open(d / name, "wb") as f:
                f.write(data)
            return name

        @server.routes.post("/comfyui_panel/panel_bg/upload")
        async def panel_bg_upload(request):
            """배경 이미지 업로드(multipart), 저장 후 자동으로 해당 이미지로 전환"""
            try:
                reader = await request.multipart()
                filename = None
                data = None
                async for field in reader:
                    if field.name in ("file", "image"):
                        if field.filename:
                            filename = field.filename
                        data = await field.read()
                    elif field.name == "filename":
                        filename = (await field.read()).decode("utf-8", "ignore")
                if not data:
                    return web.json_response({"success": False, "error": "파일 내용이 수신되지 않음"})
                if len(data) > PANEL_BG_MAX_BYTES:
                    return web.json_response({"success": False, "error": "이미지가 너무 큼 (최대값) 25MB）"})
                name = _panel_bg_save_bytes(filename, data)
                cur = _panel_bg_write_current(mode="fixed", file=name)
                return web.json_response({"success": True, "file": name,
                                          "current": cur, "items": _panel_bg_list_items()})
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        @server.routes.post("/comfyui_panel/panel_bg/delete")
        async def panel_bg_delete(request):
            """특정 배경 이미지를 삭제합니다. 사용 중인 경우 공장 기본 배경 이미지로 돌아갑니다."""
            try:
                data = await request.json()
                name = _panel_bg_safe_name(data.get("file", ""))
                if not name:
                    return web.json_response({"success": False, "error": "파일 지정되지 않음"})
                p = _panel_bg_dir() / name
                if p.is_file():
                    try:
                        p.unlink()
                    except OSError as e:
                        return web.json_response({"success": False, "error": f"삭제 실패: {e}"})
                # 동기 썸네일/WebP 캐시 삭제
                try:
                    for t in (_panel_bg_thumb_path(name), _panel_bg_webp_path(name)):
                        if t.exists():
                            t.unlink()
                except OSError:
                    pass
                s = _read_bsk_settings()
                cur = s.get("panelBackground") or {}
                if _panel_bg_safe_name(cur.get("file", "")) == name:
                    cur["file"] = ""
                    s["panelBackground"] = cur
                    _write_bsk_settings(s)
                return web.json_response({"success": True,
                                          "current": s.get("panelBackground") or {},
                                          "items": _panel_bg_list_items()})
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        # =====================================================================
        # 워크플로우 편집기 배경 이미지 (고정 배경만)）
        # ---------------------------------------------------------------------
        # 패널 배경 이미지와 동일한 이미지 디렉터리 및 file/thumb 인터페이스를 사용하며, "현재 선택"만 독립적으로 저장，
        # 저장 bsk_settings.json의 workflowEditorBackground 필드 (새로운 설정 파일 추가 없음)）。
        # =====================================================================
        WE_BG_SETTINGS_KEY = "workflowEditorBackground"

        def _we_bg_current():
            """에디터 배경 설정 읽고 검증/기본값 설정"""
            s = _read_bsk_settings().get(WE_BG_SETTINGS_KEY) or {}
            try:
                mask = float(s.get("mask", PANEL_BG_DEFAULT_MASK))
            except (TypeError, ValueError):
                mask = PANEL_BG_DEFAULT_MASK
            mask = max(0.0, min(1.0, mask))
            file = _panel_bg_safe_name(s.get("file", ""))
            if file and _panel_bg_resolve(file) is None:
                file = ""
            return {"file": file, "mask": round(mask, 3)}

        def _we_bg_write_current(file=None, mask=None):
            """편집기 배경 설정으로 다시 쓰기 (단순히 전달 필드 변경)）"""
            s = _read_bsk_settings()
            cur = s.get(WE_BG_SETTINGS_KEY) or {}
            if file is not None:
                cur["file"] = _panel_bg_safe_name(file)
            if mask is not None:
                try:
                    cur["mask"] = max(0.0, min(1.0, round(float(mask), 3)))
                except (TypeError, ValueError):
                    pass
            cur.setdefault("mask", PANEL_BG_DEFAULT_MASK)
            s[WE_BG_SETTINGS_KEY] = cur
            _write_bsk_settings(s)
            return cur

        @server.routes.get("/comfyui_panel/we_bg/list")
        async def we_bg_list(request):
            """워크플로우 편집기 배경 이미지 목록 + 현재 설정 + 공장 기본 이미지 정보"""
            try:
                builtin = _panel_bg_builtin_path()
                return web.json_response({
                    "success": True,
                    "current": _we_bg_current(),
                    "builtin": {
                        "available": bool(builtin),
                        "name": PANEL_BG_BUILTIN_NAME,
                        "url": "/comfyui_panel/panel_bg/webp",
                        "thumb": "/comfyui_panel/panel_bg/thumb",
                    },
                    "items": _panel_bg_list_items(),
                })
            except Exception as e:
                return web.json_response({"success": False, "error": str(e), "items": []})

        @server.routes.get("/comfyui_panel/we_bg/file")
        async def we_bg_file(request):
            """취소 배경 원본 이미지; 파일 파라미터 없음 = 공장 기본 H3.png (패널과 공유)）"""
            return await panel_bg_file(request)

        @server.routes.get("/comfyui_panel/we_bg/thumb")
        async def we_bg_thumb(request):
            """배경 이미지 미리보기 (패널과 공유）"""
            return await panel_bg_thumb(request)

        @server.routes.get("/comfyui_panel/we_bg/webp")
        async def we_bg_webp(request):
            """배경 이미지 WebP 버전 (패널과 공유)）"""
            return await _serve_panel_bg_webp(request)

        @server.routes.post("/comfyui_panel/we_bg/upload")
        async def we_bg_upload(request):
            """배경 이미지(multipart) 업로드 후 편집기에 자동 전환 (단, 설정만 업데이트)）"""
            try:
                reader = await request.multipart()
                filename = None
                data = None
                async for field in reader:
                    if field.name in ("file", "image"):
                        if field.filename:
                            filename = field.filename
                        data = await field.read()
                    elif field.name == "filename":
                        filename = (await field.read()).decode("utf-8", "ignore")
                if not data:
                    return web.json_response({"success": False, "error": "파일 내용이 수신되지 않음"})
                if len(data) > PANEL_BG_MAX_BYTES:
                    return web.json_response({"success": False, "error": "이미지가 너무 큼 (최대값) 25MB）"})
                name = _panel_bg_save_bytes(filename, data)
                cur = _we_bg_write_current(file=name)
                return web.json_response({"success": True, "file": name,
                                          "current": cur, "items": _panel_bg_list_items()})
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        @server.routes.post("/comfyui_panel/we_bg/delete")
        async def we_bg_delete(request):
            """삭제 특정 배경 이미지; 편집기가 사용 중이면 공장 기본 이미지로 되돌아감"""
            try:
                data = await request.json()
                name = _panel_bg_safe_name(data.get("file", ""))
                if not name:
                    return web.json_response({"success": False, "error": "파일 지정되지 않음"})
                p = _panel_bg_dir() / name
                if p.is_file():
                    try:
                        p.unlink()
                    except OSError as e:
                        return web.json_response({"success": False, "error": f"삭제 실패: {e}"})
                # 동기 썸네일/WebP 캐시 삭제
                try:
                    for t in (_panel_bg_thumb_path(name), _panel_bg_webp_path(name)):
                        if t.exists():
                            t.unlink()
                except OSError:
                    pass
                s = _read_bsk_settings()
                cur = s.get(WE_BG_SETTINGS_KEY) or {}
                if _panel_bg_safe_name(cur.get("file", "")) == name:
                    cur["file"] = ""
                    s[WE_BG_SETTINGS_KEY] = cur
                    _write_bsk_settings(s)
                return web.json_response({"success": True,
                                          "current": s.get(WE_BG_SETTINGS_KEY) or {},
                                          "items": _panel_bg_list_items()})
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        # =====================================================================
        # 프롬프트 라이브러리（Prompt Gallery）—— 유발어 ↔ 이미지 ↔ 프롬프트
        # ---------------------------------------------------------------------
        # 디렉토리 구조 (ComfyUI 사용자 디렉토리에 위치, 플러그인 재설치 시 설정 유실 방지)）:
        #   <ComfyUI>/user/default/bsk_panel/prompt_gallery/
        #       index.json            분류 / 하위 분류 / 항목 메타데이터
        #       images/<id>.webp      이미지 (일괄 변환) webp）
        #       images/<id>.txt       1행 = 유발어, 나머지 줄 = 프롬프트
        #       thumbs/<id>.webp      썸네일 목록 (자동 생성, 삭제 가능)）
        # 디자인 포인트：
        #   · 「이미지와 동일 이름의 txt 연결」—— 경고: LoRA 키 매핑 패치 실패，
        #     사용자가 images/에 그림을 직접 던집니다+txt 스캔 가능하며, 의존하지 않음 index.json；
        #   · index.json 단순히 "카테고리/하위 카테고리"와 같은 추가 메타데이터만 저장하고, 잃어도 재건립 가능；
        #   · 프롬프트 이미지 라이브러리.js(기본) ~~유발어~~）。
        # =====================================================================

        PGALLERY_DIR = USER_PANEL_DIR / "prompt_gallery"
        PGALLERY_IMAGES = PGALLERY_DIR / "images"
        PGALLERY_THUMBS = PGALLERY_DIR / "thumbs"
        PGALLERY_INDEX = PGALLERY_DIR / "index.json"
        PGALLERY_IMG_EXT = (".webp", ".png", ".jpg", ".jpeg", ".bmp", ".gif")
        PGALLERY_MAX_EDGE = 1024     # 최대 경계선 길이 (가로 세로 비율 유지)）
        PGALLERY_THUMB_EDGE = 320    # 썸네일 최대 변

        def _pg_ensure_dirs():
            PGALLERY_IMAGES.mkdir(parents=True, exist_ok=True)
            PGALLERY_THUMBS.mkdir(parents=True, exist_ok=True)

        def _pg_load_index():
            """인덱스 읽기; 파일이 없거나 손상된 경우 빈 인덱스 반환(예외를 던지지 않으며, 라이브러리 갤러리 사용 가능)）。"""
            default = {"version": 1, "categories": [], "subcategories": {}, "meta": {}}
            try:
                if not PGALLERY_INDEX.exists():
                    return default
                with open(PGALLERY_INDEX, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if not isinstance(data, dict):
                    return default
                # 항목별 보정 유형: 인덱스 파일은 수동 편집 또는 중단 상태일 수 있음，
                # 비 목록의 categories는 프런트를 통해 문자열로 개별적으로 순회합니다。
                if not isinstance(data.get("categories"), list):
                    data["categories"] = []
                if not isinstance(data.get("subcategories"), dict):
                    data["subcategories"] = {}
                if not isinstance(data.get("meta"), dict):
                    data["meta"] = {}
                return data
            except Exception:
                return default

        def _pg_save_index(idx):
            _pg_ensure_dirs()
            tmp = str(PGALLERY_INDEX) + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(idx, f, ensure_ascii=False, indent=2)
            os.replace(tmp, str(PGALLERY_INDEX))

        def _pg_read_txt(txt_path):
            """읽기 동일 이름 txt: 1행 = 유발어, 나머지 줄 = 프롬프트。"""
            if not txt_path.exists():
                return "", ""
            try:
                with open(txt_path, "r", encoding="utf-8", errors="replace") as f:
                    raw = f.read()
            except Exception:
                return "", ""
            raw = raw.replace("\r\n", "\n").replace("\r", "\n")
            if not raw.strip():
                return "", ""
            nl = raw.find("\n")
            if nl < 0:
                return raw.strip(), ""
            return raw[:nl].strip(), raw[nl + 1:].strip("\n")

        def _pg_write_txt(txt_path, trigger, prompt):
            """동일 명칭 txt: 1행은 트리거 단어, 이후는 프롬프트。"""
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write((trigger or "").strip() + "\n")
                body = (prompt or "").strip("\n")
                if body:
                    f.write(body + "\n")

        def _pg_slug(text, fallback="item"):
            """유발어 → 안전 파일 이름(중국어 등 유니코드 유지, 파일 시스템 불법 문자 제거).）。"""
            s = (text or "").strip()
            s = re.sub(r'[\\/:*?"<>|\r\n\t]+', "", s)
            s = s.strip().strip(".")
            if len(s) > 60:
                s = s[:60]
            return s or fallback

        def _pg_unique_id(base):
            """images/ 아래 사용하지 않은 파일 이름 찾기 (확장자 제외)）。"""
            _pg_ensure_dirs()
            cand = base
            i = 2
            while any((PGALLERY_IMAGES / (cand + ext)).exists() for ext in PGALLERY_IMG_EXT):
                cand = "%s_%d" % (base, i)
                i += 1
            return cand

        def _pg_find_image(item_id):
            """수정 시간 기준으로 내림차순 정렬 (최신 항목이 먼저) webp/png/jpg…）。"""
            for ext in PGALLERY_IMG_EXT:
                p = PGALLERY_IMAGES / (item_id + ext)
                if p.exists():
                    return p
            return None

        def _pg_make_thumb(src_path, item_id):
            """Generate thumbnail (webp, limit the longest side, maintain aspect ratio). Return None if failed, does not affect the main process。"""
            if not src_path or not src_path.exists():
                return None
            try:
                _pg_ensure_dirs()
                out = PGALLERY_THUMBS / (item_id + ".webp")
                with Image.open(src_path) as im:
                    im.load()
                    w, h = im.size
                    if w <= 0 or h <= 0:
                        return None
                    edge = max(w, h)
                    if edge > PGALLERY_THUMB_EDGE:
                        scale = PGALLERY_THUMB_EDGE / float(edge)
                        im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
                    if im.mode not in ("RGB", "RGBA"):
                        im = im.convert("RGBA" if "A" in im.getbands() else "RGB")
                    im.save(out, "WEBP", quality=82)
                return out
            except Exception:
                return None

        def _pg_save_image(item_id, image_data):
            """base64 이미지를 webp로 저장합니다(가장 긴 변 제한, 가로 세비유지). 반환 (path, w, h)。"""
            import io as _io
            import base64 as _b64
            if not image_data:
                raise ValueError("Missing image data")
            if isinstance(image_data, str) and image_data.startswith("data:image"):
                image_data = image_data.split(",", 1)[1]
            raw = _b64.b64decode(image_data)
            im = Image.open(_io.BytesIO(raw))
            im.load()
            try:
                im.seek(0)          # 애니메이션은 첫 번째 프레임만 추출
            except Exception:
                pass
            w, h = im.size
            edge = max(w, h)
            if edge > PGALLERY_MAX_EDGE:
                scale = PGALLERY_MAX_EDGE / float(edge)
                im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
                w, h = im.size
            if im.mode not in ("RGB", "RGBA"):
                im = im.convert("RGBA" if "A" in im.getbands() else "RGB")
            _pg_ensure_dirs()
            # 덮어쓰기: 먼저 동일한 이름의 다른 확장명을 삭제하여 webp / png가 공존하지 않도록
            for ext in PGALLERY_IMG_EXT:
                p = PGALLERY_IMAGES / (item_id + ext)
                if p.exists():
                    try:
                        os.remove(p)
                    except Exception:
                        pass
            out = PGALLERY_IMAGES / (item_id + ".webp")
            im.save(out, "WEBP", quality=90, method=4)
            return out, w, h

        def _pg_collect():
            """images/를 스캔하고 전체 항목 목록을 생성합니다. 파일 시스템이 유일한 진실이며, 인덱스는 분류만 보충합니다。"""
            _pg_ensure_dirs()
            idx = _pg_load_index()
            meta = idx.get("meta") or {}
            items = []
            for entry in os.listdir(PGALLERY_IMAGES):
                p = PGALLERY_IMAGES / entry
                if not p.is_file():
                    continue
                base, ext = os.path.splitext(entry)
                if ext.lower() not in PGALLERY_IMG_EXT:
                    continue
                trigger, prompt = _pg_read_txt(PGALLERY_IMAGES / (base + ".txt"))
                m = meta.get(base) or {}
                w = h = 0
                try:
                    with Image.open(p) as im:
                        w, h = im.size
                except Exception:
                    pass
                try:
                    mtime = int(p.stat().st_mtime)
                except Exception:
                    mtime = 0
                items.append({
                    "id": base,
                    "trigger": trigger,
                    "prompt": prompt,
                    "category": m.get("category", "") or "",
                    "subcategory": m.get("subcategory", "") or "",
                    "w": w, "h": h, "mtime": mtime,
                    "ext": ext.lower().lstrip("."),
                })
            items.sort(key=lambda x: (x["trigger"] or x["id"]).lower())
            return idx, items

        @server.routes.get("/bsk/pgallery/list")
        async def pgallery_list(request):
            """다운로드 항목 생성 + 분류. images/에 넣은 파일도 스캔됩니다。"""
            try:
                idx, items = _pg_collect()
                return web.json_response({
                    "success": True,
                    "root": str(PGALLERY_DIR),
                    "categories": idx.get("categories") or [],
                    "subcategories": idx.get("subcategories") or {},
                    "items": items,
                })
            except Exception as e:
                import traceback
                traceback.print_exc()
                return web.json_response({"success": False, "error": str(e)})

        @server.routes.get("/bsk/pgallery/triggers")
        async def pgallery_triggers(request):
            """유발어 → 프롬프트 맵핑. 워크플로우 실행 시 프론트엔드에서 이를 사용하여 ~~유발어~~ 프롬프트로 확장。"""
            try:
                _idx, items = _pg_collect()
                mapping = {}
                conflicts = []
                for it in items:
                    t = (it.get("trigger") or "").strip()
                    if not t or not (it.get("prompt") or "").strip():
                        continue          # txt 없음 / 프롬프트가 비어 있음 → 트리거 단어로 사용 불가
                    if t in mapping:
                        if mapping[t] != it["prompt"]:
                            conflicts.append(t)
                        continue          # 반복 트리거 단어: 먼저 스캔한 것을 유지합니다(목록은 이름으로 정렬되어 있으며 안정적입니다)）
                    mapping[t] = it["prompt"]
                return web.json_response({"success": True, "map": mapping, "conflicts": conflicts})
            except Exception as e:
                return web.json_response({"success": False, "error": str(e), "map": {}, "conflicts": []})

        @server.routes.get("/bsk/pgallery/image")
        async def pgallery_image(request):
            """이미지 가져오기。?id=항목ID&size=thumb|full"""
            try:
                item_id = os.path.basename(request.query.get("id", "") or "")
                size = request.query.get("size", "full")
                if not item_id or item_id in (".", ".."):
                    return web.Response(status=400, text="Missing id")
                if size == "thumb":
                    tp = PGALLERY_THUMBS / (item_id + ".webp")
                    if not tp.exists():
                        tp = _pg_make_thumb(_pg_find_image(item_id), item_id)
                    if tp and tp.exists():
                        with open(tp, "rb") as f:
                            data = f.read()
                        return web.Response(body=data, content_type="image/webp",
                                            headers={"Cache-Control": "no-cache"})
                    # 썸네일 생성 실패 → 원래 이미지로 되돌리기
                src = _pg_find_image(item_id)
                if not src:
                    return web.Response(status=404, text="Not found")
                ctype = {
                    ".webp": "image/webp", ".png": "image/png",
                    ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                    ".gif": "image/gif", ".bmp": "image/bmp",
                }.get(src.suffix.lower(), "application/octet-stream")
                with open(src, "rb") as f:
                    data = f.read()
                return web.Response(body=data, content_type=ctype,
                                    headers={"Cache-Control": "no-cache"})
            except Exception as e:
                return web.Response(status=500, text=str(e))

        @server.routes.post("/bsk/pgallery/save")
        async def pgallery_save(request):
            """새 항목/항목 업데이트: 이미지 (선택 사항)）+ txt（유발어 + 프롬프트）+ 분류 메타데이터。"""
            try:
                data = await request.json()
                trigger = (data.get("trigger") or "").strip()
                prompt = (data.get("prompt") or "").rstrip()
                category = (data.get("category") or "").strip()
                subcategory = (data.get("subcategory") or "").strip()
                image_data = data.get("image") or ""

                if not trigger:
                    return web.json_response({"success": False, "error": "유발어는 비어 있을 수 없습니다"})
                if not prompt.strip():
                    return web.json_response({"success": False, "error": "프롬프트 텍스트를 입력한 후에만 저장할 수 있습니다"})

                _pg_ensure_dirs()
                idx = _pg_load_index()

                item_id = os.path.basename((data.get("id") or "").strip())
                if item_id and not _pg_find_image(item_id) and not image_data:
                    item_id = ""                       # id 지정한 파일이 존재하지 않고 이미지를 전송하지 않았습니다 → 새로 만들기
                if not item_id:
                    item_id = _pg_unique_id(_pg_slug(trigger))

                if image_data:
                    _pg_save_image(item_id, image_data)
                    tp = PGALLERY_THUMBS / (item_id + ".webp")
                    if tp.exists():
                        try:
                            os.remove(tp)              # 이미지 변경 시 이전 썸네일 무효화
                        except Exception:
                            pass

                _pg_write_txt(PGALLERY_IMAGES / (item_id + ".txt"), trigger, prompt)

                meta = idx.setdefault("meta", {})
                m = meta.setdefault(item_id, {})
                m["category"] = category
                m["subcategory"] = subcategory
                m["trigger"] = trigger
                m["mtime"] = int(time.time())

                # 사용된 분류/자 분류를 자동으로 기록하여 '선택한 분류가 목록에 없음' 문제를 방지」
                cats = idx.setdefault("categories", [])
                if category and category not in cats:
                    cats.append(category)
                if category:
                    subs = idx.setdefault("subcategories", {}).setdefault(category, [])
                    if subcategory and subcategory not in subs:
                        subs.append(subcategory)
                _pg_save_index(idx)

                src = _pg_find_image(item_id)
                if src:
                    _pg_make_thumb(src, item_id)

                return web.json_response({"success": True, "id": item_id})
            except Exception as e:
                import traceback
                traceback.print_exc()
                return web.json_response({"success": False, "error": str(e)})

        @server.routes.post("/bsk/pgallery/delete")
        async def pgallery_delete(request):
            """항목 삭제: 이미지 + 동명 txt + Thumbnail + 메타데이터 인덱스。"""
            try:
                data = await request.json()
                item_id = os.path.basename((data.get("id") or "").strip())
                if not item_id:
                    return web.json_response({"success": False, "error": "Missing id"})

                removed = []
                for ext in PGALLERY_IMG_EXT:
                    p = PGALLERY_IMAGES / (item_id + ext)
                    if p.exists():
                        os.remove(p)
                        removed.append(p.name)
                for p in (PGALLERY_IMAGES / (item_id + ".txt"),
                          PGALLERY_THUMBS / (item_id + ".webp")):
                    if p.exists():
                        os.remove(p)
                        removed.append(p.name)

                idx = _pg_load_index()
                if item_id in (idx.get("meta") or {}):
                    idx["meta"].pop(item_id, None)
                    _pg_save_index(idx)

                return web.json_response({"success": True, "removed": removed})
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        @server.routes.post("/bsk/pgallery/categories")
        async def pgallery_categories(request):
            """카테고리 유지 관리: create / delete / rename를 지원하며, category 및 subcategory 두 계층을 지원합니다.。"""
            try:
                data = await request.json()
                action = (data.get("action") or "").strip()
                level = (data.get("level") or "category").strip()
                name = (data.get("name") or "").strip()
                new_name = (data.get("new_name") or "").strip()
                parent = (data.get("parent") or "").strip()

                idx = _pg_load_index()
                idx.setdefault("categories", [])
                idx.setdefault("subcategories", {})
                idx.setdefault("meta", {})

                def _pool():
                    if level == "subcategory":
                        return idx["subcategories"].setdefault(parent or "_", [])
                    return idx["categories"]

                def _field():
                    return "subcategory" if level == "subcategory" else "category"

                pool = _pool()
                field = _field()

                if action == "create":
                    if not name:
                        return web.json_response({"success": False, "error": "이름이 비어 있을 수 없음"})
                    if name not in pool:
                        pool.append(name)
                elif action == "delete":
                    if name in pool:
                        pool.remove(name)
                    # 항목에서 해당 범주 메타데이터 삭제
                    for _id, m in idx["meta"].items():
                        if level == "subcategory":
                            if (m.get("category") or "") == parent and (m.get("subcategory") or "") == name:
                                m["subcategory"] = ""
                        else:
                            if (m.get("category") or "") == name:
                                m["category"] = ""
                                m["subcategory"] = ""
                elif action == "rename":
                    if not name or not new_name:
                        return web.json_response({"success": False, "error": "이름이 비어 있을 수 없음"})
                    if name in pool:
                        pool[pool.index(name)] = new_name
                    for _id, m in idx["meta"].items():
                        if level == "subcategory":
                            if (m.get("category") or "") == parent and (m.get("subcategory") or "") == name:
                                m["subcategory"] = new_name
                        else:
                            if (m.get("category") or "") == name:
                                m["category"] = new_name
                else:
                    return web.json_response({"success": False, "error": "Unsupported action: %s" % action})

                _pg_save_index(idx)
                return web.json_response({
                    "success": True,
                    "categories": idx["categories"],
                    "subcategories": idx["subcategories"],
                })
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        # =====================================================================
        # AnimaDex 데이터셋 연결 (읽기 전용 연결, 데이터 복사 안 함)）
        # ---------------------------------------------------------------------
        # 【타사 저작권 선언 / Third-party attribution】
        #   프로젝트：AnimaDex
        #   주페이지：https://github.com/zetaneko/AnimaDex
        #   허가：MIT License, Copyright (c) 2026 AnimaDex contributors
        #   전체: 보기 LICENSES/AnimaDex.txt
        #
        #   이 구간의 코드**포함 안 함** AnimaDex 소스 코드, 읽기 전용 방식으로 읽기
        #   animadex.db（SQLite）및 import/*.csv，또한 다음 규칙에 따라 다시 구현되었습니다
        #   호환 가능한 쿼리(다각/페이지네이션/파일명 규칙). 실행 시에도 호출할 수 있습니다.
        #   자신의 install.sh / install.bat를 사용하여 데이터 설치 및 가져오기 —— 그 스크립트는
        #   AnimaDex 프로젝트의 일부, 원 저자 소유, 호출만 수행하고 수정하지 않음。
        #
        #   데이터 자체(역할 이름/아티스트 이름/태그/이미지)는 AnimaDex의 공개에서 제공됩니다
        #   카메라 제어 노드: ComfyUI 사용자 디렉토리에 저장/불러오기
        #   Laxhar/noob-wiki（https://huggingface.co/datasets/Laxhar/noob-wiki），
        #   저작권은 각 권리자에게 있음. 이 플러그인은 이러한 데이터를 패키지하거나 재배포하지 않고, 단지
        #   로컬 단독 읽기 참조, 사용자는 AnimaDex를 통해 직접 가져와야 합니다。
        # ---------------------------------------------------------------------
        # AnimaDex 자체 호스팅된 애니메이션 캐릭터/아티스트 참조 이미지 라이브러리（Flask + SQLite + WebP）。
        # 데이터 양은 10,000개 레벨이므로 여기에서**아니**복사 / 폴더 탐색 경로를 따르세요：
        #
        #   1. 읽기 전용 연결: 설정 저장 <bsk_panel>/animadex.json，이미지 절대 경로로 직접 읽기，
        #      복사하지 않고, AnimaDex 측에서 몇 개를 보내면 즉시 여기에서 캔버스할 수 있습니다；
        #   2. 듀얼 데이터 소스: animadex.db 우선 읽기; 라이브러리 행 수가 부족하면 import/*.csv
        #      （임포트 스크립트가 실행 중이거나 ingest되지 않았으면 CSV를 메모리에 붓습니다 SQLite，
        #      그러므로 '가로로 다운로드 및」—— 이미지를 천천히 로드하고 검색은 즉시 사용 가능；
        #   3. 상속 분류: AnimaDex 내장 저작권/발색/瞳색/발장/성별
        #      분면을 바로 여기의 필터 옵션으로 만들고, 로컬 갤러리에 저장할 때도 여기에 떨어집니다
        #      「AnimaDex / <작품명>」두 단계 분류 내；
        #   4. 유발어는 필요에 따라 펼쳐집니다：~~hatsune_miku 이러한 인용문은 "프롬프트"에 실제로 나타나야 함」
        #      백엔드에서 먼저 확인하고, 몇만 개의 매핑을 한 번에 프론트엔드로 쏟아뜨리지 않습니다.；
        #   5. 사용자 폴더 링크: 데이터 디렉터리 감지 후 최대한 위치에 있습니다 <bsk_panel>/animadex-data
        #      링크 생성 (Windows용 융합 연결, POSIX용 심볼릭 링크)），
        #      리소스 관리자에서 직접 탐색하기 편리함. 링크 실패는 위 기능에 영향을 미치지 않음。
        # =====================================================================

        import sqlite3 as _sqlite3      # 이 섹션 전용, 플러그인 전역 가져오기를 오염시키지 않음

        AD_CFG_PATH = USER_PANEL_DIR / "animadex.json"
        AD_LINK_NAME = "animadex-data"
        AD_LINK_PATH = USER_PANEL_DIR / AD_LINK_NAME
        AD_DEFAULT_REPO_URL = "https://github.com/zetaneko/AnimaDex.git"
        AD_DB_NAME = "animadex.db"

        AD_TRAIT_FACETS = ("hair_color", "hair_length", "eye_color", "gender")
        AD_CHAR_FACETS = ("copyright",) + AD_TRAIT_FACETS
        AD_ARTIST_FACETS = ("score", "category")
        AD_FACET_LABELS = {
            "copyright": "작품", "hair_color": "발색", "hair_length": "발 길이",
            "eye_color": "瞳색", "gender": "성별", "score": "그림 점수",
            "category": "그림 작가 분류",
        }
        AD_SCORE_BUCKETS = (
            ("5", "50% 위", 0.50, 1.01),
            ("4", "40 - 50%", 0.40, 0.50),
            ("3", "30 - 40%", 0.30, 0.40),
            ("2", "20 - 30%", 0.20, 0.30),
            ("1", "20% 아래", 0.00, 0.20),
        )

        # AnimaDex/db.py와 일치: 핵심 레이블 → 분할
        AD_ILLEGAL_FS = '<>:"\\/|?*'
        AD_HAIR_COLOR_TAGS = {
            "aqua hair", "black hair", "blonde hair", "blue hair", "brown hair",
            "dark blue hair", "gradient hair", "green hair", "grey hair",
            "light blue hair", "light brown hair", "light green hair",
            "light purple hair", "multicolored hair", "orange hair", "pink hair",
            "purple hair", "red hair", "silver hair", "split-color hair",
            "streaked hair", "two-tone hair", "white hair",
        }
        AD_HAIR_LENGTH_TAGS = {
            "very short hair", "short hair", "medium hair", "long hair",
            "very long hair", "absurdly long hair",
        }
        AD_EYE_COLOR_TAGS = {
            "aqua eyes", "black eyes", "blue eyes", "brown eyes",
            "gradient eyes", "green eyes", "grey eyes", "multicolored eyes",
            "orange eyes", "pink eyes", "purple eyes", "red eyes",
            "two-tone eyes", "yellow eyes",
        }
        AD_GENDER_TAGS = {"1boy": "Male", "1girl": "Female",
                          "1other": "Ambiguous", "no humans": "Non-Human"}

        # 트리거 단어에 포함될 수 없는 문자 (프론트엔드 TRIGGER_BREAK와 일치), 저장할 때 대체해야 함
        AD_TRIGGER_BREAK = set(" \t\r\n~,，.。;；:：!！?？|、")

        # ---------------- 설정 ----------------

        def _ad_default_cfg():
            return {
                "data_dir": "",          # 공 = 자동 탐지
                "repo_dir": "",
                "repo_url": AD_DEFAULT_REPO_URL,
                "auto_link": True,
                "expand_enabled": True,  # ~~slug 실행 시 펼쳐짐 여부
                "expand_mode": "trigger_tags",   # trigger_tags | tags | trigger
                "category_mode": "copyright",    # copyright | source | flat
                "page_size": 60,
                "source": "auto",        # auto | local | online（데이터 소스）
                "online_base": "https://animadex.net",   # 온라인 검색 인터페이스 루트 주소
                "online_cache_mb": 512,  # 온라인 썸네일 캐시 최대값
                "online_cache_enabled": True,   # 온라인 미리보기 자동 저장 여부
                "install_scoring": False,       # 설치 스크립트 --with-scoring
                "install_generation": False,    # 설치 스크립트 --with-generation
            }

        def _ad_read_cfg():
            cfg = _ad_default_cfg()
            try:
                if AD_CFG_PATH.exists():
                    with open(AD_CFG_PATH, "r", encoding="utf-8") as f:
                        d = json.load(f)
                    if isinstance(d, dict):
                        for k, v in d.items():
                            if k in cfg:
                                cfg[k] = v
            except Exception:
                pass
            return cfg

        def _ad_write_cfg(cfg):
            try:
                USER_PANEL_DIR.mkdir(parents=True, exist_ok=True)
                tmp = str(AD_CFG_PATH) + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(cfg, f, ensure_ascii=False, indent=2)
                os.replace(tmp, str(AD_CFG_PATH))
            except Exception as e:
                print(f"[ComfyUI Panel] animadex.json 쓰기 실패: {e}")
            return cfg

        # ---------------- 작은 도구 ----------------

        def _ad_is_windows():
            return os.name == "nt"

        def _ad_platform():
            if _ad_is_windows():
                return "windows"
            if sys.platform == "darwin":
                return "macos"
            return "linux"

        def _ad_sanitize_filename(name):
            """복제 AnimaDex의 온디스크 파일 이름 규칙 (유발어) → 파일명）。"""
            cleaned = "".join("_" if c in AD_ILLEGAL_FS else c for c in (name or ""))
            return cleaned.rstrip(" .") or "unnamed"

        def _ad_cap_word(word):
            for i, ch in enumerate(word):
                if ch.isalpha():
                    return word[:i] + ch.upper() + word[i + 1:]
            return word

        def _ad_titlecase(text):
            return " ".join(_ad_cap_word(w) for w in (text or "").split(" "))

        def _ad_trait_label(tag):
            return _ad_titlecase((tag or "").rsplit(" ", 1)[0])

        def _ad_traits_from_tags(core):
            """핵심 레이블 → (facet, value, label)，규칙은 AnimaDex/db.py와 일치。"""
            out = []
            for t in (x.strip() for x in (core or "").split(",")):
                if not t:
                    continue
                if t in AD_HAIR_COLOR_TAGS:
                    out.append(("hair_color", t, _ad_trait_label(t)))
                elif t in AD_HAIR_LENGTH_TAGS:
                    out.append(("hair_length", t, _ad_trait_label(t)))
                elif t in AD_EYE_COLOR_TAGS:
                    out.append(("eye_color", t, _ad_trait_label(t)))
                elif t in AD_GENDER_TAGS:
                    out.append(("gender", t, AD_GENDER_TAGS[t]))
            return out

        def _ad_trigger_of(slug):
            """slug → 인용할 수 있는 트리거 단어 (슬래시/대체 문자는 인용을 잘라냅니다).）。"""
            s = "".join("_" if c in AD_TRIGGER_BREAK else c for c in (slug or ""))
            s = re.sub(r"_+", "_", s).strip("_")
            return s

        def _ad_like(token):
            esc = (token or "").replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            return "%" + esc + "%"

        def _ad_parse_query(q):
            return [t for t in re.split(r"\s+", (q or "").strip()) if t]

        # ---------------- 온라인 데이터 소스 (animadex.net 공개 읽기 전용 인터페이스).） ----------------
        # 몇만 개의 캐릭터를 위해 20GB 데이터 패키지를 다운로드하는 방법이 아니면: 공식 웹사이트의 공개 검색을 사용하세요
        # API, 이미지 필요에 따라 가져오고 디스크 캐시에 저장. 주요 사항：
        #   · 요청은 User-Agent를 포함해야 하며, 그렇지 않으면 403；
        #   · 응답에 CORS 헤드가 없으면 브라우저 fetch가 차단되므로 항상 백엔드로 전달；
        #   · animadex.net 계열 도메인만 인식 —— 어떤 URL 프록시로도 변환되지 않도록 합니다（SSRF）；
        #   · 온라인 인터페이스 페이지당 고정 36(page_size 전달해도 무용, 단면 지원은 단일 값만)。

        AD_ONLINE_UA = "bsk-animadex/1 (+https://github.com/zetaneko/AnimaDex)"
        AD_ONLINE_PAGE = 36
        AD_ONLINE_CACHE_DIR = USER_PANEL_DIR / "animadex-cache"
        AD_ONLINE_THUMBS = AD_ONLINE_CACHE_DIR / "thumbs"
        _AD_ONLINE = {"search": {}, "lookup": {}, "facets": {},
                      "total": 0, "err": "", "cn": None, "cache": (0, 0, 0)}

        # 중국어 → 영어. 일반적으로 사용되는 (시리즈 + 인기 있는 캐릭터), 사용자는 획득
        # <bsk_panel>/animadex_cn.json 덮기 / 보충：{"중국어명": "english name"}
        AD_CN_ALIAS = {
            # —— 작품 / 시리즈
            "원신": "genshin impact", "붕괴: 신성철의 망원경": "honkai star rail",
            "성경열철도": "honkai star rail", "별철": "honkai star rail",
            "붕괴3": "honkai impact 3rd", "붕괴 3": "honkai impact 3rd",
            "붕괴": "honkai", "Zero Zero Zero": "zenless zone zero", "zzz": "zenless zone zero",
            "명태": "wuthering waves", "碧蓝航선": "azur lane",
            "아르키메들": "arknights", "방배": "arknights",
            "울트라 데이터베이스": "blue archive", "碧蓝파일案": "blue archive",
            "소녀앞선": "girls frontline", "공주 연결": "princess connect",
            "경쟁말": "umamusume", "말": "umamusume",
            "아이돌 마스터": "idolmaster", "운명의 칭호 지정": "fate grand order",
            "fgo": "fate grand order", "함대 수집": "kantai collection",
            "함대collection": "kantai collection", "쉽c": "kantai collection",
            "쉽娘": "kantai collection", "전함 소녀": "warship girls",
            "승리 여신": "nikke", "니키": "nikke", "포켓몬": "pokemon",
            "포켓몬스터": "pokemon", "귀멸의 칼날": "demonslayer", "귀멸": "demonslayer",
            "주술회전": "jujutsu kaisen", "진격의 거인": "attack on titan",
            "신세기 에반게리온": "neon genesis evangelion", "eva": "evangelion",
            "나루토": "naruto", "나루토": "naruto", "원피스": "one piece",
            "해적": "one piece", "데몬": "bleach", "용주": "dragon ball",
            "칠룡주": "dragon ball", "파이널 판타지": "final fantasy",
            "젤다": "zelda", "동쪽": "touhou", "동쪽project": "touhou",
            "초음": "hatsune miku", "vocaloid": "vocaloid",
            # —— 원신
            "뢰전장군": "raiden shogun", "뢰신": "raiden shogun",
            "장군": "raiden shogun", "뢰전": "raiden shogun",
            "종리": "zhongli", "이시": "zhongli", "摩拉克斯": "zhongli",
            "원디": "venti", "풍신": "venti", "바바토스": "venti",
            "Naxidada": "nahida", "초신": "nahida", "푸닝": "furina",
            "물신": "furina", "芙芙": "furina", "호두": "hu tao",
            "Gan Yu": "ganyu", "เฉิงเหริน": "keqing", "신리 린화": "kamisato ayaka",
            "연화": "kamisato ayaka", "쇼무야": "yoimiya", "야오청시": "yae miko",
            "신자": "yae miko", "포토가 유메": "kaedehara kazuha", "만야": "kaedehara kazuha",
            "야란": "yelan", "신미 린인": "kamisato ayato", "칼리": "klee",
            "시치": "qiqi", "건반": "jean gunnhildr", "모나": "mona",
            "디루크": "diluc", "시외": "xiao", "신혜": "shenhe",
            "아라타키 이토": "arataki itto", "구키닌": "kuki shinobu",
            "스트레이더": "wanderer", "흩어져": "wanderer", "에일하선": "alhaitham",
            "나비아": "navia", "클라린데": "clorinde", "시그먼트": "sigewinne",
            "레이오스리": "wriothesley", "그 윌리엇": "neuvillette",
            "나비에르": "neuvillette", "니롤": "nilou", "세노": "cyno",
            "티나리": "tighnari", "다다리아": "tartaglia", "공자": "tartaglia",
            "장미": "xiangling", "행추": "xingqiu", "피에스엘": "fischl",
            "반닛트": "bennett", "중요": "chongyun", "영광": "ningguang",
            "위성항복": "beidou", "매화색": "yanfei", "로사리야": "rosaria",
            "카이아": "kaeya", "리사": "lisa", "안버": "amber",
            "바바라": "barbara", "설탕": "sucrose", "디오나": "diona",
            "노에일": "noelle", "뢰제": "razor", "유라": "eula",
            "아베르도": "albedo", "코रल 가든 마인 해": "sangonomiya kokomi",
            "심해": "sangonomiya kokomi", "지원마": "thoma", "조유": "sayu",
            "오로": "gorou", "Yun Jin": "yun jin", "구자사로": "kujou sara",
            "팔루산": "faruzan", "칸노인": "shikanoin heizou",
            "Kiyora Kiyora": "kirara", "린닛트": "lynette", "림니": "lyney",
            "페미니": "freminet", "레이이라": "layla", "カヴィ": "kaveh",
            "백출": "baizhu", "칸티스": "candace", "멀티 로라": "dori",
            "코레이": "collei", "瑶瑶": "yaoyao", "형": "lumine", "공": "aether",
            "파몬": "paimon",
            # —— 성경열철도
            "염색": "acheron", "류링": "firefly", "샘": "firefly",
            "미러 스트림": "jingliu", "카푸카": "kafka", "연랑": "silver wolf",
            "폭죽": "sparkle", "검은 백조": "black swan", "줄무늬 독수리": "robin",
            "날": "blade", "블로니아": "bronya", "히어": "seele",
            "단항": "dan heng", "3월 7일": "march 7th", "키키": "himeko",
            "발트": "welt", "백로": "bailu", "취성": "tingyun",
            "공중": "yukong", "수양": "sushang", "청작": "qingque",
            "경원": "jing yuan", "언경": "yanqing", "로사체": "luocha",
            "푸현": "fu xuan", "玲可": "lynx", "계내분": "guinaifen",
            "토파": "topaz", "연지": "argenti", "藿藿": "huohuo",
            "한아": "hanya", "눈의 옷": "xueyi", "진리 의사": "dr. ratio",
            "황금": "aventurine", "파티오": "boothill", "윈리": "yunli",
            "지오추": "jiaqiu", "비쪼": "feixiao", "灵砂": "lingsha",
            "Random Break": "rappa", "잊혀진 사람": "fugue", "遐엽": "castorice",
            "아그리애야": "aglaea", "티박스": "tribbie", "무적": "mydei",
            "백어": "phainon", "그때 여름": "anaxa",
            # —— 붕괴 3
            "기야나": "kiana kaslana", "뢰인": "raiden mei",
            "Meayi": "raiden mei", "야에중사": "yae sakura",
            "Bloonyia Zaychik": "bronya zaychik", "델리사": "theresa apocalypse",
            "푸화": "fu hua", "리타": "rita rossweisse",
            "요람델": "durandal", "에리히야": "elysia",
            # —— 동쪽
            "보리 님": "hakurei reimu", "림멍": "hakurei reimu",
            "무리사": "kirisame marisa", "마리사": "kirisame marisa",
            "시스루 야": "izayoi sakuya", "사야": "izayoi sakuya",
            "레밀리아": "remilia scarlet", "플란도루": "flandre scarlet",
            "바유": "yakumo yukari", "여신상": "yakumo ran",
            "혼령요몽": "konpaku youmu", "요몽": "konpaku youmu",
            "서행사 유유자": "saigyouji yuyuko", "유유자": "saigyouji yuyuko",
            "Shimei Maru Bun": "shameimaru aya", "동풍구 매초": "kochiya sanae",
            "초기화": "kochiya sanae", "고명지인식": "komeiji satori",
            "고명지연": "komeiji koishi", "기루노": "cirno",
            "Red Mei Ling": "hong meiling", "벨선": "reisen udongein inaba",
            "인판테": "inaba tewi", "후지워라 메구히미": "fujiwara no mokou",
            "봉래산 혜야": "houraisan kaguya", "휘예": "houraisan kaguya",
            "그 이전의 황제": "hinanawi tenshi", "성백련": "hijiri byakuren",
            "이부스키 추향": "ibuki suika", "츠키미 하센": "ibaraki kasen",
            "파큐리": "patchouli knowledge", "앨리스": "alice margatroid",
            # —— Fate / 형곡
            "알트리오스": "artoria pendragon", "오왕": "artoria pendragon",
            "기르기메시크": "gilgamesh", "금빛빛": "gilgamesh",
            "에사카 린": "tosaka rin", "간토 쿄": "matou sakura",
            "위궁스턴스": "emiya shirou", "위궁": "emiya",
            "정덕": "jeanne d'arc", "니로": "nero claudius",
            "야모전": "tamamo no mae", "스카하": "scathach",
            "큐큐린": "cu chulainn", "메린": "merlin", "모건": "morgan le fay",
            "이리야": "illyasviel von einzbern",
            # —— 기타 인기
            "초음미래": "hatsune miku", "미러 벨": "kagamine rin",
            "미러 링크": "kagamine len", "바바토스": "megurine luka",
            "카구메 요코": "kaname madoka", "샤오미야옌": "akemi homura",
            "미츠키 사야카": "miki sayaka", "바마미": "tomoe mami",
            "사쿠라 아츠코": "sakura kyouko", "미사키 고자카": "misaka mikoto",
            "포저": "misaka mikoto", "야시나": "asuna",
            "고하킨리": "itsuka kotori", "시즈카 쿠도": "tokisaki kurumi",
            "狂三": "tokisaki kurumi", "아미야": "amiya",
            "텍사스": "texas", "엔젤": "exusiai", "연회": "silverash",
            "에야팔라": "eyjafjalla", "세레야": "saria", "별곰": "hoshiguma",
            "스카티": "skadi", "케일시": "kal'tsit", "기업": "enterprise",
            "오르건 왕자": "prinz eugen", "俾斯麦": "bismarck",
            "벨파스트": "belfast", "눈보라": "fubuki", "도우": "shimakaze",
            "다요": "yamato", "무장": "musashi", "눈 바람": "yukikaze",
            "비": "shigure", "백자": "shiroko", "별야": "hoshino",
            "일나": "hina", "猫又": "nekomiya mana", "주이연": "zhu yuan",
            "이련": "ellen joe", "니코": "nicole demara", "안비": "anby demara",
            "현재": "jinhsi", "장거리": "changli", "수호자": "shorekeeper",
            "나무": "camellya", "불 피우지 않도록": "jiyan", "음림": "yinlin",
        }

        def _ad_online_base(cfg=None):
            cfg = cfg or _ad_read_cfg()
            return (cfg.get("online_base") or "https://animadex.net").rstrip("/")

        def _ad_has_local_data():
            """AnimaDex 패키지가 로컬에 있는지 (10초 캐시)）。
            _ad_resolve() 스캔을 탐지하고 "현재 데이터 소스"를 매번 쿼리할 때마다 질문합니다，
            캐시가 없으면 페이지를 넘을 때마다 디스크 전체 검색。"""
            now = time.time()
            hit = _AD_ONLINE.get("resolve")
            if hit and now - hit[0] < 10:
                return hit[1]
            val = _ad_resolve()
            _AD_ONLINE["resolve"] = (now, bool(val))
            _AD_ONLINE["resolve_dir"] = val
            return bool(val)

        def _ad_local_dir():
            """위와 동일, 경로도 함께 기록하고, 한 문장 다시 스캔하지 마세요.。"""
            _ad_has_local_data()
            return _AD_ONLINE.get("resolve_dir") or ""

        def _ad_online_active(cfg=None):
            """현재 활성화된 데이터 소스：'online' / 'local'。"""
            cfg = cfg or _ad_read_cfg()
            src = (cfg.get("source") or "auto").strip().lower()
            if src == "online":
                return "online"
            if src == "local":
                return "local"
            # auto：로컬에 데이터가 있으면 로컬을 사용하고, 없으면 온라인 (다운로드 불필요)）
            return "local" if _ad_has_local_data() else "online"

        def _ad_url_host_ok(url):
            """애드미네кс.네트워크 도메인만 허용하여 이 프록시가 임의의 URL로 전달되는 것을 방지。"""
            try:
                from urllib.parse import urlparse
                host = (urlparse(url).hostname or "").lower()
            except Exception:
                return False
            if not host:
                return False
            return host == "animadex.net" or host.endswith(".animadex.net")

        def _ad_http_bytes(url, timeout=15):
            import urllib.request
            req = urllib.request.Request(
                url, headers={"User-Agent": AD_ONLINE_UA, "Accept": "*/*"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                if r.status != 200:
                    raise RuntimeError("HTTP %s" % r.status)
                return r.read(), r.headers.get("Content-Type", "")

        def _ad_online_json(path, params=None, timeout=20):
            import urllib.parse
            url = _ad_online_base() + path
            if params:
                url += "?" + urllib.parse.urlencode(
                    [(k, v) for k, v in params.items() if v not in (None, "")])
            raw, _ct = _ad_http_bytes(url, timeout)
            return json.loads(raw.decode("utf-8", "replace"))

        def _ad_cn_alias(q):
            """중국어 키워드 → 영문 이름; 중국어 또는 일치하지 않으면 빈 문자열 (원래대로 검색）。"""
            s = (q or "").strip()
            if not s or all(ord(c) < 128 for c in s):
                return ""
            table = _AD_ONLINE.get("cn")
            if not isinstance(table, dict):
                table = dict(AD_CN_ALIAS)
                try:
                    p = USER_PANEL_DIR / "animadex_cn.json"
                    if p.exists():
                        with open(p, "r", encoding="utf-8") as f:
                            d = json.load(f)
                        if isinstance(d, dict):
                            for k, v in d.items():
                                if isinstance(v, str) and v.strip():
                                    table[str(k).strip().lower()] = v.strip()
                except Exception:
                    pass
                _AD_ONLINE["cn"] = table
            low = s.lower()
            if low in table:
                return table[low]
            for k, v in table.items():
                if len(k) >= 2 and k in low:
                    return v
            return ""

        def _ad_online_item(mode, r):
            """선택 항목 → 로컬 _ad_query와 동일한 필드 (온라인 이미지 주소 추가)）。"""
            slug = r.get("slug") or ""
            if mode == "artists":
                trig = (r.get("trigger") or "").strip()
                return {
                    "key": "a:" + slug, "slug": slug,
                    "name": r.get("name") or slug, "sub": "",
                    "trigger": trig, "prompt": trig, "tags": [],
                    "count": r.get("count") or 0, "url": r.get("url") or "",
                    "score": r.get("score"), "thumb": "", "full": "", "v": 0,
                    "thumb_url": r.get("thumb_url") or "",
                    "img_url": r.get("img_url") or "",
                    "online": True,
                }
            tags = [str(t).strip() for t in (r.get("tags") or []) if str(t).strip()]
            trig = (r.get("trigger") or "").strip()
            return {
                "key": "c:" + slug, "slug": slug,
                "name": r.get("name") or slug,
                "sub": r.get("copyright_name") or r.get("copyright") or "",
                "copyright": r.get("copyright") or "",
                "copyright_name": r.get("copyright_name") or "",
                "trigger": trig,
                "prompt": _ad_prompt_of(trig, ",".join(tags)),
                "tags": tags, "count": r.get("count") or 0,
                "url": r.get("url") or "", "score": None,
                "thumb": "", "full": "", "v": 0,
                "thumb_url": r.get("thumb_url") or "",
                "img_url": r.get("img_url") or "",
                "online": True,
            }

        def _ad_online_page(path, params, n):
            """온라인 페이지 n번째 가져오기 (5분 캐시). 페이지당 고정 36개。"""
            now = time.time()
            key = (path, tuple(sorted((k, str(v)) for k, v in params.items())))
            bucket = _AD_ONLINE["search"].setdefault(key, {})
            hit = bucket.get(n)
            if hit and now - hit[0] < 300:
                return hit[1]
            if len(_AD_ONLINE["search"]) > 24:
                try:
                    _AD_ONLINE["search"].pop(next(iter(_AD_ONLINE["search"])))
                except Exception:
                    _AD_ONLINE["search"].clear()
                bucket = _AD_ONLINE["search"].setdefault(key, {})
            d = _ad_online_json(path, dict(params, page=n))
            data = {
                "items": d.get("results") or [],
                "total": d.get("total") or 0,
                "pages": d.get("pages") or 1,
            }
            bucket[n] = (now, data)
            _AD_ONLINE["err"] = ""
            return data

        def _ad_online_search(mode, q, filters, sort, page, page_size):
            """온라인 검색. _ad_query와 동일한 구조를 반환, 프런트엔드 분기 없음。"""
            mode = "artists" if mode == "artists" else "characters"
            page = max(1, int(page or 1))
            size = max(1, min(AD_ONLINE_PAGE, int(page_size or AD_ONLINE_PAGE)))
            api_q = _ad_cn_alias(q) or (q or "")
            params = {"q": api_q, "sort": "az" if sort == "az" else "count"}
            # 온라인 세분 파라미터는 단일 값만 허용하며, 다중 선택 시 첫 번째 값을 가져옵니다 (프론트엔드 온라인 모드도 단일 선택).）
            if mode == "characters":
                for f in AD_CHAR_FACETS:
                    vals = filters.get(f) or []
                    if vals:
                        params[f] = vals[0]
            path = ("/api/artists/search" if mode == "artists"
                    else "/api/characters/search")

            start = (page - 1) * size
            first = start // AD_ONLINE_PAGE + 1
            last = (start + size - 1) // AD_ONLINE_PAGE + 1
            items, total, pages = [], 0, 1
            try:
                for n in range(first, last + 1):
                    d = _ad_online_page(path, params, n)
                    total, pages = d["total"], d["pages"]
                    items.extend(d["items"])
                    if n < last and n >= pages:
                        break
            except Exception as e:
                _AD_ONLINE["err"] = str(e)
                return {"ok": False, "error": str(e), "items": [], "total": 0}
            rows = items[start - (first - 1) * AD_ONLINE_PAGE:][:size]
            return {
                "ok": True,
                "items": [_ad_online_item(mode, r) for r in rows],
                "total": total, "page": page,
                "pages": max(1, int(math.ceil(total / float(size)))),
                "source": "online", "mode": mode,
                "translated": api_q if api_q != (q or "") else None,
            }

        def _ad_online_facets(mode, facet, q, limit=60):
            """온라인 데이터 소스: 로컬 세트 (라이브러리 / CSV / 링크 / 저장소) 모두 무관）。"""
            if mode != "characters" or facet not in AD_CHAR_FACETS:
                return {"ok": True, "items": []}
            limit = max(1, min(300, int(limit or 60)))
            now = time.time()
            cached = _AD_ONLINE["facets"].get(facet)
            if not cached or now - cached[0] > 3600:
                try:
                    d = _ad_online_json("/api/characters/facets", {}, timeout=15)
                    _AD_ONLINE["total"] = d.get("total") or _AD_ONLINE["total"]
                    node = (d.get("facets") or {}).get(facet) or {}
                    cached = (now, node.get("values") or [])
                    _AD_ONLINE["facets"][facet] = cached
                    _AD_ONLINE["err"] = ""
                except Exception as e:
                    _AD_ONLINE["err"] = str(e)
                    return {"ok": False, "error": str(e), "items": []}
            rows = cached[1]
            kw = (q or "").strip().lower()
            if kw:
                rows = [v for v in rows
                        if kw in str(v.get("label", "")).lower()
                        or kw in str(v.get("value", "")).lower()]
            return {"ok": True, "items": [
                {"value": v.get("value"),
                 "label": v.get("label") or v.get("value"),
                 "n": v.get("count") or 0}
                for v in rows[:limit]]}

        def _ad_online_lookup(mode, slug):
            """slug / 이름으로 정확히 하나 검색합니다(24시간 캐시). 검색 실패 시 반환 None。"""
            key = str(slug or "").strip()
            if not key:
                return None
            modes = ([mode] if mode in ("characters", "artists")
                     else ["characters", "artists"])
            now = time.time()
            for m in modes:
                hit = _AD_ONLINE["lookup"].get((m, key))
                if hit and now - hit[0] < 86400:
                    return hit[1]
            for m in modes:
                path = ("/api/artists/search" if m == "artists"
                        else "/api/characters/search")
                try:
                    d = _ad_online_json(
                        path, {"q": key, "page": 1, "sort": "count"}, timeout=15)
                except Exception as e:
                    _AD_ONLINE["err"] = str(e)
                    continue
                for r in d.get("results") or []:
                    if str(r.get("slug") or "").lower() != key.lower():
                        continue
                    item = _ad_online_item(m, r)
                    item["core_tags"] = ",".join(item["tags"])
                    _AD_ONLINE["lookup"][(m, key)] = (time.time(), item)
                    _AD_ONLINE["err"] = ""
                    return item
            return None

        def _ad_online_expand(names, mode=""):
            """온라인 버전 필요에 따라 펼쳐짐: _ad_expand와 동형의 (map, info)。"""
            out, info = {}, {}
            names = [str(n).strip() for n in (names or []) if str(n).strip()]
            modes = ([mode] if mode in ("characters", "artists")
                     else ["characters", "artists"])
            left = list(dict.fromkeys(names))
            for m in modes:
                if not left:
                    break
                for n in list(left):
                    row = _ad_online_lookup(m, n)
                    if not row:
                        continue
                    body = (row.get("prompt") or "").strip()
                    if not body:
                        continue
                    out[n] = body
                    info[n] = {
                        "slug": row.get("slug"), "mode": m,
                        "name": row.get("name"), "sub": row.get("sub") or "",
                        "thumb": "", "full": "", "prompt": body,
                        "thumb_url": row.get("thumb_url") or "",
                        "img_url": row.get("img_url") or "",
                    }
                    left.remove(n)
            return out, info

        def _ad_online_trim_cache():
            """썸네일 캐시 초과 시 수정 시간을 기준으로 오래된 것부터 삭제。"""
            try:
                cap = int(_ad_read_cfg().get("online_cache_mb") or 512)
            except Exception:
                cap = 512
            cap = max(64, min(8192, cap)) * 1024 * 1024
            try:
                files = [(p.stat().st_mtime, p.stat().st_size, p)
                         for p in AD_ONLINE_THUMBS.iterdir() if p.is_file()]
            except Exception:
                return
            total = sum(s for _m, s, _p in files)
            if total <= cap:
                return
            files.sort()
            for _m, s, p in files:
                if total <= cap:
                    break
                try:
                    p.unlink()
                    total -= s
                except Exception:
                    pass

        def _ad_online_image_ext(url):
            ext = ".webp"
            low = (url or "").split("?")[0].lower()
            for e in (".png", ".jpg", ".jpeg", ".gif", ".webp"):
                if low.endswith(e):
                    ext = e
                    break
            return ext

        def _ad_online_cache_path(url):
            import hashlib
            return AD_ONLINE_THUMBS / (
                hashlib.md5((url or "").encode("utf-8")).hexdigest()
                + _ad_online_image_ext(url))

        def _ad_online_cache_on():
            """로컬에 자동으로 작은 이미지를 저장할지 여부(끄면 네트워크를 통해 항상 진행)）。"""
            return bool(_ad_read_cfg().get("online_cache_enabled", True))

        def _ad_online_image_bytes(url, timeout=20):
            """바이트만 추출하고 디스크에 쓰지 않음; (bytes, content_type) 반환; 실패 시 반환 (None, None)。"""
            if not url or not _ad_url_host_ok(url):
                return None, None
            try:
                raw, ctype = _ad_http_bytes(url, timeout)
            except Exception as e:
                _AD_ONLINE["err"] = str(e)
                return None, None
            if not raw:
                return None, None
            return raw, (ctype or ("image/" + _ad_online_image_ext(url).lstrip(".")))

        def _ad_online_image_file(url, timeout=20):
            """URL을 사용하여 원격 이미지를 캐시하고, 로컬 파일 경로를 반환합니다; 실패하면 None。

            이것은 '로컬 갤러리에 저장'만 사용합니다 —— 즐겨찾기는 반드시 현실 파일로 만들어야 함，
            따라서 online_cache_enabled 스위치에 영향을 받지 않습니다. 그림이 따라가는 경로를 보세요.。
            """
            if not url or not _ad_url_host_ok(url):
                return None
            try:
                AD_ONLINE_THUMBS.mkdir(parents=True, exist_ok=True)
            except Exception:
                return None
            fp = _ad_online_cache_path(url)
            try:
                if fp.exists() and fp.stat().st_size > 0:
                    return fp
            except Exception:
                pass
            raw, _ct = _ad_online_image_bytes(url, timeout)
            if not raw:
                return None
            try:
                tmp = str(fp) + ".part"
                with open(tmp, "wb") as f:
                    f.write(raw)
                os.replace(tmp, str(fp))
                _ad_online_trim_cache()
                return fp
            except Exception as e:
                _AD_ONLINE["err"] = str(e)
                return None

        def _ad_online_image_get(url, timeout=20):
            """이미지 추출 (보기용): 캐시 히트 시 디스크에서 직접 읽고, 캐시 미스 시 네트워크 사용。

            끄기 '자동 캐시' 모드**디스크에 새 그래프 쓰지 않음**，이미 캐시된 데이터 사용
            （저장 로컬에서 속도를 높일 수 있습니다. 공간을 확보하려면 '캐시 비우기' 버튼을 사용하세요.。
            (바이트, 콘텐츠 유형) 반환, 실패 시 반환 (None, None)。
            """
            if not url or not _ad_url_host_ok(url):
                return None, None
            ctype = "image/" + _ad_online_image_ext(url).lstrip(".")
            fp = _ad_online_cache_path(url)
            try:
                if fp.exists() and fp.stat().st_size > 0:
                    with open(fp, "rb") as f:
                        return f.read(), ctype
            except Exception:
                pass
            raw, r_ct = _ad_online_image_bytes(url, timeout)
            if not raw:
                return None, None
            if _ad_online_cache_on():
                try:
                    AD_ONLINE_THUMBS.mkdir(parents=True, exist_ok=True)
                    tmp = str(fp) + ".part"
                    with open(tmp, "wb") as f:
                        f.write(raw)
                    os.replace(tmp, str(fp))
                    _ad_online_trim_cache()
                except Exception:
                    pass        # 캐시 쓰기 실패는 이미지 보기 영향 없음
            return raw, (r_ct or ctype)

        def _ad_online_cache_clear():
            """온라인 미리보기 캐시 삭제, 삭제된 양 표시。"""
            n = b = 0
            try:
                for p in AD_ONLINE_THUMBS.iterdir():
                    if not p.is_file():
                        continue
                    try:
                        b += p.stat().st_size
                        p.unlink()
                        n += 1
                    except Exception:
                        pass
            except Exception:
                pass
            _AD_ONLINE["cache"] = (0, 0, 0)
            return {"files": n, "bytes": b}

        def _ad_online_cache_stats(refresh=False):
            """캐시 디렉토리 통계 (장수 / 바이트수), 60초에 한 번 실행。"""
            now = time.time()
            cached = _AD_ONLINE.get("cache") or (0, 0, 0)
            if not refresh and cached[0] and now - cached[0] < 60:
                return {"files": cached[1], "bytes": cached[2]}
            files = bytes_ = 0
            try:
                for p in AD_ONLINE_THUMBS.iterdir():
                    if not p.is_file():
                        continue
                    try:
                        files += 1
                        bytes_ += p.stat().st_size
                    except Exception:
                        pass
            except Exception:
                pass
            _AD_ONLINE["cache"] = (now, files, bytes_)
            return {"files": files, "bytes": bytes_}

        async def _ad_run(fn, *args, **kwargs):
            """%s %d {name} 
 <tag> ComfyUI LoRA GPU API BSK.。"""
            import asyncio
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, lambda: fn(*args, **kwargs))

        def _ad_online_probe():
            """탐색 온라인 인터페이스: 총 수 취득 + 확인 중。"""
            try:
                d = _ad_online_json("/api/characters/facets", {}, timeout=10)
                _AD_ONLINE["total"] = d.get("total") or 0
                _AD_ONLINE["err"] = ""
            except Exception as e:
                _AD_ONLINE["err"] = str(e)
            return _AD_ONLINE["err"] == ""

        def _ad_prompt_of(trigger, core_tags, mode=None):
            """설정에 따라 삽입/확장할 프롬프트 생성。"""
            trig = (trigger or "").strip().strip(",")
            core = (core_tags or "").strip()
            m = mode or _ad_read_cfg().get("expand_mode") or "trigger_tags"
            if m == "tags":
                return core
            if m == "trigger":
                return trig
            if trig and core:
                # 중복 제거: core_tags는 종종 trigger에 포함된 내용 포함
                parts, seen = [], set()
                for p in (trig + ", " + core).split(","):
                    s = p.strip().lower()
                    if not s or s in seen:
                        continue
                    seen.add(s)
                    parts.append(p.strip())
                return ", ".join(parts)
            return trig or core

        # ---------------- 디렉토리 파싱 / 탐지 ----------------

        def _ad_data_dir_from_repo(repo):
            """repo config.toml의 [paths].data_dir을 읽음 (repo 루트 경로를 상대적으로 파싱)）。"""
            try:
                p = Path(repo) / "config.toml"
                if not p.exists():
                    return None
                txt = p.read_text(encoding="utf-8", errors="replace")
                m = re.search(r'^\s*data_dir\s*=\s*["\']?([^"\'\r\n#]+?)["\']?\s*$',
                              txt, re.M)
                if not m:
                    return None
                val = m.group(1).strip()
                if not val:
                    return None
                return str((Path(repo) / val).resolve())
            except Exception:
                return None

        def _ad_repo_guesses():
            """가능한 저장소 위치 (데이터 디렉터리 역추적용)）。"""
            base = Path(folder_paths.base_path)
            cfg = _ad_read_cfg()
            out = []
            if cfg.get("repo_dir"):
                out.append(cfg["repo_dir"])
            for p in (
                base / "AnimaDex",
                base.parent / "AnimaDex",
                Path.home() / "AnimaDex",
                USER_PANEL_DIR / "AnimaDex",
            ):
                out.append(str(p))
            return out

        def _ad_detect_candidates():
            """AnimaDex 데이터 디렉터리의 후보를 자동으로 탐지합니다 (가능성에 따라 정렬, 중복 제거)）。"""
            out, seen = [], set()

            def add(p):
                if not p:
                    return
                try:
                    p = Path(str(p).strip()).expanduser()
                    key = str(p.resolve()).lower() if p.exists() else str(p).lower()
                except Exception:
                    key = str(p).lower()
                if key in seen:
                    return
                seen.add(key)
                out.append(str(p))

            cfg = _ad_read_cfg()
            add(cfg.get("data_dir"))

            # 기존 링크는 데이터 디렉토리 자체를 가리킵니다
            if AD_LINK_PATH.exists():
                try:
                    add(os.path.realpath(str(AD_LINK_PATH)))
                except Exception:
                    pass

            # 리포지토리 위치에서 역추적
            for repo in _ad_repo_guesses():
                add(_ad_data_dir_from_repo(repo))
                add(Path(repo) / ".." / "animadex-data")

            base = Path(folder_paths.base_path)
            for p in (
                base / "animadex-data",
                base / "AnimaDex" / "animadex-data",
                USER_PANEL_DIR / "animadex-data",
                base.parent / "animadex-data",
                base.parent / "AnimaDex" / "animadex-data",
                Path.home() / "animadex-data",
                Path.home() / "AnimaDex" / "animadex-data",
            ):
                add(p)
            return out

        def _ad_is_data_dir(p):
            """AnimaDex 데이터 디렉터리처럼 아니야? 라이브러리나 CSV가 있으면 모두 해당。"""
            try:
                p = Path(p)
                if not p.is_dir():
                    return False
                if (p / AD_DB_NAME).exists():
                    return True
                if (p / "import" / "characters.csv").exists():
                    return True
                if (p / "characters" / "thumbs").is_dir():
                    return True
                if (p / "characters").is_dir() and (p / "artists").is_dir():
                    return True
            except Exception:
                pass
            return False

        def _ad_resolve():
            """현재 사용 중인 데이터 디렉토리를 확인: 구성 → 자동 감지. 찾을 수 없음. ''。"""
            cfg = _ad_read_cfg()
            if cfg.get("data_dir"):
                p = str(cfg["data_dir"]).strip()
                if p and _ad_is_data_dir(p):
                    return os.path.normpath(p)
            for cand in _ad_detect_candidates():
                if _ad_is_data_dir(cand):
                    return os.path.normpath(cand)
            return ""

        # ---------------- 데이터 연결 (DB 우선, CSV 백업)） ----------------

        _AD_CACHE = {"sig": "", "ts": 0, "conn": None, "source": "none",
                     "error": "", "counts": {}, "n_db": 0, "n_csv": 0}
        _AD_LOCK = _threading.Lock()

        def _ad_signature(data_dir):
            parts = [data_dir]
            for rel in (AD_DB_NAME,
                        os.path.join("import", "characters.csv"),
                        os.path.join("import", "artists.csv")):
                p = Path(data_dir) / rel
                try:
                    st = p.stat()
                    parts.append("%s|%d|%d" % (rel, int(st.st_mtime), st.st_size))
                except Exception:
                    parts.append(rel + "|-")
            return "&".join(parts)

        def _ad_csv_row_count(path):
            try:
                n = 0
                with open(path, "rb") as f:
                    for _ in f:
                        n += 1
                return max(0, n - 1)          # 제거 헤더
            except Exception:
                return 0

        def _ad_mem_schema():
            return """
                CREATE TABLE characters (
                    character      TEXT PRIMARY KEY,
                    copyright      TEXT NOT NULL,
                    name           TEXT NOT NULL,
                    name_lower     TEXT NOT NULL,
                    copyright_name TEXT NOT NULL,
                    trigger        TEXT NOT NULL,
                    core_tags      TEXT NOT NULL,
                    count          INTEGER NOT NULL DEFAULT 0,
                    url            TEXT NOT NULL DEFAULT '',
                    imgname        TEXT NOT NULL,
                    thumbname      TEXT NOT NULL,
                    search_blob    TEXT NOT NULL,
                    image_version  INTEGER NOT NULL DEFAULT 0);
                CREATE INDEX ad_c_count ON characters(count DESC, name_lower);
                CREATE INDEX ad_c_name  ON characters(name_lower);
                CREATE INDEX ad_c_cp    ON characters(copyright);
                CREATE TABLE traits (
                    character TEXT NOT NULL, facet TEXT NOT NULL,
                    value     TEXT NOT NULL, label TEXT NOT NULL,
                    PRIMARY KEY (character, facet, value)) WITHOUT ROWID;
                CREATE INDEX ad_t_facet ON traits(facet, value);
                CREATE TABLE artists (
                    artist        TEXT PRIMARY KEY,
                    name          TEXT NOT NULL,
                    name_lower    TEXT NOT NULL,
                    trigger       TEXT NOT NULL,
                    count         INTEGER NOT NULL DEFAULT 0,
                    url           TEXT NOT NULL DEFAULT '',
                    imgname       TEXT NOT NULL,
                    thumbname     TEXT NOT NULL,
                    search_blob   TEXT NOT NULL,
                    score         REAL,
                    image_version INTEGER NOT NULL DEFAULT 0);
                CREATE INDEX ad_a_count ON artists(count DESC, name_lower);
                CREATE INDEX ad_a_name  ON artists(name_lower);
                CREATE TABLE artist_categories (
                    artist TEXT NOT NULL, category TEXT NOT NULL,
                    PRIMARY KEY (artist, category)) WITHOUT ROWID;
            """

        def _ad_build_memdb(data_dir):
            """저장 import/*.csv 메모리에 SQLite를 삽입하여 "불러오기"도 검색 가능하게 함。"""
            import csv as _csv
            con = _sqlite3.connect(":memory:")
            con.row_factory = _sqlite3.Row
            con.executescript(_ad_mem_schema())

            cpath = Path(data_dir) / "import" / "characters.csv"
            rows, trows = [], []
            try:
                with open(cpath, "r", encoding="utf-8", errors="replace", newline="") as f:
                    for row in _csv.DictReader(f):
                        ch = (row.get("character") or "").strip()
                        if not ch:
                            continue
                        trig = (row.get("trigger") or "").strip()
                        cp = (row.get("copyright") or "").strip()
                        core = (row.get("core_tags") or "").strip()
                        if ", " in trig:
                            nm, cpn = trig.split(", ", 1)
                        else:
                            nm, cpn = trig, ""
                        name = _ad_titlecase(nm) if nm else _ad_titlecase(ch.replace("_", " "))
                        cpname = (_ad_titlecase(cpn) if cpn
                                  else _ad_titlecase(cp.replace("_", " ")))
                        try:
                            cnt = int(row.get("count") or 0)
                        except Exception:
                            cnt = 0
                        stem = _ad_sanitize_filename(trig)
                        rows.append((ch, cp, name, name.lower(), cpname, trig, core,
                                     cnt, (row.get("url") or "").strip(),
                                     stem + ".png", stem + ".webp",
                                     " ".join((ch, cp, trig, core)).lower(), 0))
                        for facet, val, lab in _ad_traits_from_tags(core):
                            trows.append((ch, facet, val, lab))
            except Exception as e:
                print(f"[ComfyUI Panel] AnimaDex CSV(캐릭터) 읽기 실패: {e}")

            apath = Path(data_dir) / "import" / "artists.csv"
            arows = []
            try:
                with open(apath, "r", encoding="utf-8", errors="replace", newline="") as f:
                    for row in _csv.DictReader(f):
                        ar = (row.get("artist") or "").strip()
                        if not ar:
                            continue
                        trig = (row.get("trigger") or "").strip() or ar.replace("_", " ")
                        try:
                            cnt = int(row.get("count") or 0)
                        except Exception:
                            cnt = 0
                        try:
                            score = float(row.get("score") or 0) or None
                        except Exception:
                            score = None
                        stem = _ad_sanitize_filename(trig)
                        arows.append((ar, _ad_titlecase(trig), trig.lower(), trig, cnt,
                                      (row.get("url") or "").strip(),
                                      stem + ".png", stem + ".webp",
                                      " ".join((ar, trig)).lower(), score, 0))
            except Exception as e:
                print(f"[ComfyUI Panel] AnimaDex CSV(드로잉) 읽기 실패: {e}")

            if rows:
                con.executemany(
                    "INSERT OR REPLACE INTO characters VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
            if trows:
                con.executemany("INSERT OR REPLACE INTO traits VALUES (?,?,?,?)", trows)
            if arows:
                con.executemany(
                    "INSERT OR REPLACE INTO artists VALUES (?,?,?,?,?,?,?,?,?,?,?)", arows)
            con.commit()
            return con

        def _ad_conn(force=False):
            """사용 가능한 연결을 획득했습니다. DB는 읽기 전용으로 열리고, CSV 모드는 메모리 데이터베이스를 반환합니다.。"""
            data_dir = _ad_resolve()
            if not data_dir:
                _AD_CACHE.update(conn=None, source="none",
                                 error="AnimaDex 데이터 디렉터리를 찾을 수 없습니다.")
                return None, ""
            sig = _ad_signature(data_dir)
            with _AD_LOCK:
                cached = _AD_CACHE.get("conn")
                if (not force) and cached is not None and _AD_CACHE.get("sig") == sig \
                        and (time.time() - _AD_CACHE.get("ts", 0)) < 30:
                    return cached, data_dir

                db_path = Path(data_dir) / AD_DB_NAME
                cpath = Path(data_dir) / "import" / "characters.csv"
                n_csv = _ad_csv_row_count(cpath) if cpath.exists() else 0
                n_db = 0
                conn = None

                if db_path.exists():
                    try:
                        # as_uri() 이미 file:///... 형식이면 쿼리 매개변수를 직접 붙여넣으세요.
                        uri = Path(db_path).as_uri() + "?mode=ro"
                        probe = _sqlite3.connect(uri, uri=True, timeout=5)
                        n_db = probe.execute(
                            "SELECT COUNT(*) FROM characters").fetchone()[0]
                        # 라이브러리 행 수가 따라가지 않아요 CSV → ingest가 아직 완료되지 않았습니다, 사용 CSV
                        if n_db >= max(n_csv, 1):
                            conn = probe
                            conn.row_factory = _sqlite3.Row
                        else:
                            probe.close()
                    except Exception as e:
                        print(f"[ComfyUI Panel] AnimaDex 읽기 전용 데이터베이스 연결 실패: {e}")

                if conn is None and cpath.exists() and n_csv > 0:
                    try:
                        conn = _ad_build_memdb(data_dir)
                    except Exception as e:
                        print(f"[ComfyUI Panel] AnimaDex CSV 메모리 라이브러리 빌드 실패: {e}")

                old = _AD_CACHE.get("conn")
                _AD_CACHE.update(
                    sig=sig, ts=time.time(), conn=conn,
                    source=("db" if (conn is not None and n_db >= max(n_csv, 1)) else
                            ("csv" if conn is not None else "none")),
                    n_db=n_db, n_csv=n_csv, data_dir=data_dir,
                    error="" if conn is not None else "데이터가 없습니다: 사용 가능한 데이터베이스가 없거나 CSV")
                if old is not None and old is not conn:
                    try:
                        old.close()
                    except Exception:
                        pass
                return conn, data_dir

        # ---------------- 통계 / 가져오기 진행률 ----------------

        _AD_STAT_CACHE = {"key": "", "ts": 0, "data": {}}

        def _ad_stats(data_dir, force=False):
            """폴더 수준 통계 (썸네일 수 / 원본 수 / 다운로드 여부). 캐시 15s。"""
            key = data_dir
            if not force and _AD_STAT_CACHE.get("key") == key \
                    and (time.time() - _AD_STAT_CACHE.get("ts", 0)) < 15:
                return _AD_STAT_CACHE["data"]
            out = {"thumbs": 0, "images": 0, "artist_thumbs": 0, "parts": 0}
            try:
                td = Path(data_dir) / "characters" / "thumbs"
                if td.is_dir():
                    names = os.listdir(td)
                    out["thumbs"] = len([n for n in names if not n.endswith(".part")])
                    out["parts"] += len([n for n in names if n.endswith(".part")])
                idir = Path(data_dir) / "characters" / "images"
                if idir.is_dir():
                    names = os.listdir(idir)
                    out["images"] = len([n for n in names if not n.endswith(".part")])
                    out["parts"] += len([n for n in names if n.endswith(".part")])
                atd = Path(data_dir) / "artists" / "thumbs"
                if atd.is_dir():
                    out["artist_thumbs"] = len(
                        [n for n in os.listdir(atd) if not n.endswith(".part")])
            except Exception:
                pass
            _AD_STAT_CACHE.update(key=key, ts=time.time(), data=out)
            return out

        # ---------------- 크로스 플랫폼 디렉토리 링크 ----------------

        def _ad_make_dir_link(src, dst):
            """데이터 디렉터리 링크를 사용자 디렉터리로 연결합니다. Windows에서는 junction(관리자 권한 불필요)）。"""
            import subprocess as _sp
            if not src or not os.path.isdir(src):
                return {"ok": False, "created": False, "kind": "no-source",
                        "detail": "소스 디렉토리 없음: %s" % src}
            if os.path.lexists(dst):
                try:
                    if (os.path.normcase(os.path.realpath(dst)) ==
                            os.path.normcase(os.path.realpath(src))):
                        return {"ok": True, "created": False, "kind": "linked",
                                "detail": "이미 연결되었습니다"}
                except Exception:
                    pass
                return {"ok": False, "created": False, "kind": "occupied",
                        "detail": "대상 위치가 이미 사용 중입니다：%s" % dst}
            try:
                USER_PANEL_DIR.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                return {"ok": False, "created": False, "kind": "io",
                        "detail": "사용자 디렉토리 생성 불가: %s" % e}

            if _ad_is_windows():
                last = ""
                for flag, kind in (("/J", "junction"), ("/D", "symlink")):
                    try:
                        r = _sp.run(["cmd", "/c", "mklink", flag, str(dst), str(src)],
                                    capture_output=True, text=True, timeout=30)
                        if r.returncode == 0 and os.path.lexists(dst):
                            return {"ok": True, "created": True, "kind": kind}
                        last = ((r.stdout or "") + " " + (r.stderr or "")).strip()
                    except Exception as e:
                        last = str(e)
                try:
                    os.symlink(str(src), str(dst), target_is_directory=True)
                    return {"ok": True, "created": True, "kind": "symlink"}
                except Exception as e:
                    return {"ok": False, "created": False, "kind": "failed",
                            "detail": "Windows 링크 생성 실패 (junction / 심볼릭 링크 모두 작동하지 않음)）: %s | %s"
                                      % (last, e)}
            try:
                os.symlink(str(src), str(dst), target_is_directory=True)
                return {"ok": True, "created": True, "kind": "symlink"}
            except Exception as e:
                return {"ok": False, "created": False, "kind": "failed", "detail": str(e)}

        def _ad_link_info():
            info = {"path": str(AD_LINK_PATH), "exists": False, "ok": False,
                    "kind": "", "detail": "", "target": ""}
            try:
                info["exists"] = os.path.lexists(str(AD_LINK_PATH))
                if info["exists"]:
                    info["ok"] = True
                    info["kind"] = "junction/symlink"
                    info["target"] = os.path.realpath(str(AD_LINK_PATH))
                    info["detail"] = "링크"
            except Exception as e:
                info["detail"] = str(e)
            return info

        def _ad_unlink():
            """"링크"만 추출하고 실제 디렉터리 삭제하지 않음 —— 사용자 데이터 삭제는 허용되지 않음。"""
            import subprocess as _sp
            p = str(AD_LINK_PATH)
            if not os.path.lexists(p):
                return True
            if os.path.islink(p):
                os.unlink(p)
                return True
            if _ad_is_windows():
                # junction：os.path.islink 반환값이 False지만 rmdir은 파일만 삭제
                # 연결 노드는 자체적이며 대상 디렉토리의 파일을 건드리지 않습니다。
                if os.path.isdir(p):
                    r = _sp.run(["cmd", "/c", "rmdir", p],
                                capture_output=True, text=True, timeout=20)
                    if r.returncode == 0 and not os.path.lexists(p):
                        return True
                    os.rmdir(p)
                    return True
            raise OSError("목표는 링크가 아니므로 삭제를 거부합니다(실제 디렉터리 삭제를 우연히 삭제하는 것을 두려워합니다)）：%s" % p)

        def _ad_ensure_link(data_dir, force=False):
            """확인하여 사용자 디렉터리 링크가 data_dir。

            force=False（자동: 링크가 이미 존재하지만 다른 곳으로 가도 변경 없음 —— 사용자 가능
            직접 변경하면 혼란스러워질 수 있습니다。
            force=True（사용자가 명시적으로 "설치/링크"를 클릭하면, 이를 변경할 수 있도록 허용。
            """
            if not data_dir:
                return _ad_link_info()
            info = _ad_link_info()
            if info["ok"]:
                try:
                    same = (os.path.normcase(info["target"]) ==
                            os.path.normcase(os.path.realpath(data_dir)))
                except Exception:
                    same = False
                if same:
                    return info
                if not force:
                    info["detail"] = ("다른 곳에 링크되었습니다 (%s), 자동으로 대상이 변경되지 않습니다" %
                                      info["target"])
                    return info
                try:
                    _ad_unlink()
                except Exception as e:
                    return {"ok": False, "created": False, "kind": "failed",
                            "path": str(AD_LINK_PATH),
                            "detail": "구 링크 삭제 불가: %s" % e}
            res = _ad_make_dir_link(data_dir, str(AD_LINK_PATH))
            out = {"path": str(AD_LINK_PATH),
                   "exists": os.path.lexists(str(AD_LINK_PATH)),
                   "ok": bool(res.get("ok")),
                   "kind": res.get("kind", ""),
                   "detail": res.get("detail", ""),
                   "target": ""}
            if out["ok"]:
                try:
                    out["target"] = os.path.realpath(str(AD_LINK_PATH))
                except Exception:
                    pass
            return out

        # ---------------- 백그라운드 작업 (복제 / 설치, 전체적으로 블록킹 없음)） ----------------

        _AD_TASKS = {}
        _AD_TASKS_LOCK = _threading.Lock()
        _AD_TASK_SEQ = [0]
        _AD_LOG_MAX = 400

        def _ad_task_new(kind, title):
            with _AD_TASKS_LOCK:
                _AD_TASK_SEQ[0] += 1
                tid = "ad%d" % _AD_TASK_SEQ[0]
                _AD_TASKS[tid] = {
                    "id": tid, "kind": kind, "title": title,
                    "state": "running",     # running | done | error | cancelled
                    "phase": "준비 중", "log": [], "progress": 0,
                    "started": time.time(), "ended": 0, "error": "", "result": {},
                }
                return tid

        def _ad_task(tid):
            with _AD_TASKS_LOCK:
                return _AD_TASKS.get(tid)

        def _ad_task_log(tid, line, phase=None, progress=None):
            with _AD_TASKS_LOCK:
                t = _AD_TASKS.get(tid)
                if not t:
                    return
                s = str(line).rstrip()
                if s:
                    t["log"].append(s)
                    if len(t["log"]) > _AD_LOG_MAX:
                        del t["log"][:len(t["log"]) - _AD_LOG_MAX]
                if phase:
                    t["phase"] = phase
                if progress is not None:
                    t["progress"] = max(0, min(100, int(progress)))

        def _ad_task_finish(tid, state, error="", result=None):
            with _AD_TASKS_LOCK:
                t = _AD_TASKS.get(tid)
                if not t:
                    return
                t["state"] = state
                t["ended"] = time.time()
                t["error"] = str(error or "")
                if result:
                    t["result"] = result
                if state == "done":
                    t["progress"] = 100
                    t["phase"] = "완료"

        def _ad_run_stream(tid, cmd, cwd, env=None, timeout=3600):
            """명령어를 실행하여 stdout/stderr를 한 줄씩 작업 로그에 푸시。"""
            import subprocess as _sp
            _ad_task_log(tid, "$ %s   (cwd=%s)" % (" ".join(cmd), cwd), phase="실행 중")
            popen = _sp.Popen(cmd, cwd=cwd, env=env,
                              stdout=_sp.PIPE, stderr=_sp.STDOUT,
                              text=True, errors="replace", bufsize=1)
            t = _ad_task(tid)
            while True:
                if t and t.get("state") == "cancelling":
                    try:
                        popen.terminate()
                    except Exception:
                        pass
                    return -1, "취소됨"
                line = popen.stdout.readline()
                if line:
                    _ad_task_log(tid, line)
                    continue
                if popen.poll() is not None:
                    break
                time.sleep(0.05)
            try:
                popen.stdout.close()
            except Exception:
                pass
            return popen.returncode, ""

        def _ad_git_exe():
            import shutil as _sh
            return _sh.which("git") or ""

        def _ad_clone_core(tid, url, target, branch=""):
            """git clone。반환 (ok, error)。"""
            git = _ad_git_exe()
            if not git:
                return False, "Git을 찾을 수 없습니다. 먼저 Git을 설치하고 추가하십시오 PATH"
            target = os.path.abspath(target)
            parent = os.path.dirname(target) or "."
            try:
                os.makedirs(parent, exist_ok=True)
            except Exception as e:
                return False, "디렉터리 생성 불가 %s: %s" % (parent, e)
            if os.path.isdir(os.path.join(target, ".git")):
                _ad_task_log(tid, "디렉토리 안에 이미 저장소가 있으므로 클론을 건너띔：" + target)
                return True, ""
            if os.path.isdir(target) and os.listdir(target):
                return False, "대상 디렉토리가 존재하고 비어 있지 않음：" + target
            cmd = [git, "clone", "--progress"]
            if branch:
                cmd += ["-b", branch]
            cmd += [url, target]
            env = dict(os.environ)
            env["GIT_TERMINAL_PROMPT"] = "0"      # 패스워드 인터랙션에 갇히지 마세요
            env["GIT_ASKPASS"] = ""
            _ad_task_log(tid, "클론 시작 %s → %s" % (url, target),
                         phase="복제 중", progress=5)
            code, err = _ad_run_stream(tid, cmd, parent, env=env)
            if code != 0:
                return False, err or (
                    "git 종료 코드 %s(저장소 주소가 잘못되었거나 네트워크 연결이 안됨)？）" % code)
            _ad_task_log(tid, "클론 완료：" + target, progress=40)
            return True, ""

        def _ad_clone_worker(tid, url, target, branch=""):
            try:
                ok, err = _ad_clone_core(tid, url, target, branch)
                if not ok:
                    _ad_task_finish(tid, "error", err)
                    return
                _ad_task_log(tid, "클론 완료", phase="완료", progress=100)
                _ad_task_finish(tid, "done",
                                result={"repo": os.path.abspath(target)})
            except Exception as e:
                import traceback
                traceback.print_exc()
                _ad_task_finish(tid, "error", str(e))

        def _ad_install_core(tid, repo, with_scoring, with_generation):
            """설치 스크립트 실행 + 데이터 디렉토리 + 연결. 반환 (ok, error, data_dir)。"""
            repo = os.path.abspath(repo)
            if not os.path.isdir(repo):
                return False, "저장소 디렉터리가 존재하지 않습니다：" + repo, ""

            cfg = _ad_read_cfg()
            cfg["repo_dir"] = repo
            _ad_write_cfg(cfg)

            if _ad_is_windows():
                script = Path(repo) / "install.bat"
                cmd = ["cmd", "/c", "install.bat"]
            else:
                script = Path(repo) / "install.sh"
                cmd = ["bash", "install.sh"]
            if not script.exists():
                return False, "설치 스크립트가 없습니다：" + str(script), ""
            if not _ad_is_windows():
                try:
                    os.chmod(str(script), 0o755)
                except Exception:
                    pass
            if with_scoring:
                cmd.append("--with-scoring")
            if with_generation:
                cmd.append("--with-generation")

            _ad_task_log(tid, "시작 설치 (백그라운드 실행, 인터페이스 멈춤 없음)）：" + " ".join(cmd),
                         phase="설치 중", progress=45)
            code, err = _ad_run_stream(tid, cmd, repo)
            if code != 0:
                return False, err or ("설치 스크립트 종료 코드 %s, 로그를 참조" % code), ""

            _ad_task_log(tid, "스크립트 실행 완료, 데이터 디렉토리 검색 시작…",
                         phase="데이터 디렉토리 대기", progress=70)
            data_dir = _ad_wait_data_dir(tid, repo, timeout=600)
            if not data_dir:
                return (False,
                        "설치 완료되었지만 데이터 디렉토리가 아직 생성되지 않았습니다(고급 설정에서 수동으로 경로를 지정할 수 있습니다)）",
                        "")

            cfg = _ad_read_cfg()
            cfg["data_dir"] = data_dir
            cfg["repo_dir"] = repo
            _ad_write_cfg(cfg)
            _ad_task_log(tid, "데이터 폴더：" + data_dir,
                         phase="링크 데이터 디렉토리", progress=85)

            # 링크 실패도 설치 실패로 계산되지 않음——단지 리소스 관리자에서 직접 뒤지는 편의성만 부족합니다。
            # 여기 사용 force=True：사용자가 새 위치에 설치되었으므로 링크는 그 방향으로 따라 변경되어야 합니다.。
            if cfg.get("auto_link", True):
                res = _ad_ensure_link(data_dir, force=True)
                if res.get("ok"):
                    _ad_task_log(tid, "연결됨 사용자 디렉터리：%s → %s"
                                 % (AD_LINK_PATH, data_dir))
                else:
                    _ad_task_log(tid, "（프롬프트 링크가 사용자 디렉터리로 실패했지만, 사용에 영향을 미치지 않음：%s"
                                 % res.get("detail", ""))
            try:
                _ad_conn(True)
            except Exception:
                pass
            return True, "", data_dir

        def _ad_install_worker(tid, repo, with_scoring, with_generation):
            """설치 스크립트 실행 (블로킹 없음), 설치 완료 후 데이터 디렉터리 생성 및 링크。"""
            try:
                ok, err, data_dir = _ad_install_core(
                    tid, repo, with_scoring, with_generation)
                if not ok:
                    _ad_task_finish(tid, "error", err,
                                    result={"repo": repo, "data_dir": data_dir})
                    return
                _ad_task_log(tid, "완료, 사용할 수 있습니다", phase="완료", progress=100)
                _ad_task_finish(tid, "done",
                                result={"repo": repo, "data_dir": data_dir})
            except Exception as e:
                import traceback
                traceback.print_exc()
                _ad_task_finish(tid, "error", str(e))

        def _ad_setup_local_worker(tid, url, target, with_scoring, with_generation):
            """한 번 설치: 복제 → 설치 스크립트 실행 → 자동 연결. 사용자에게만 디렉토리를 제공。"""
            try:
                ok, err = _ad_clone_core(tid, url, target)
                if not ok:
                    _ad_task_finish(tid, "error", err)
                    return
                ok, err, data_dir = _ad_install_core(
                    tid, target, with_scoring, with_generation)
                if not ok:
                    _ad_task_finish(tid, "error", err)
                    return
                _ad_task_log(tid, "완료, 사용할 수 있습니다", phase="완료", progress=100)
                _ad_task_finish(tid, "done",
                                result={"repo": target, "data_dir": data_dir})
            except Exception as e:
                import traceback
                traceback.print_exc()
                _ad_task_finish(tid, "error", str(e))

        def _ad_wait_data_dir(tid, repo, timeout=600):
            """polling 등 데이터 디렉터리 생성 (install.sh 5단계에서 생성)）。"""
            deadline = time.time() + timeout
            tried = 0
            while time.time() < deadline:
                t = _ad_task(tid)
                if t and t.get("state") == "cancelling":
                    return ""
                cands = []
                dd = _ad_data_dir_from_repo(repo)
                if dd:
                    cands.append(dd)
                cands.append(str(Path(repo) / ".." / "animadex-data"))
                cands.append(str(Path(repo) / "animadex-data"))
                for c in cands:
                    if _ad_is_data_dir(c):
                        return os.path.normpath(c)
                tried += 1
                if tried % 5 == 1:
                    _ad_task_log(tid, "데이터 디렉토리를 아직 보지 못했습니다, 계속 기다립니다…（%ds）"
                                 % int(deadline - time.time()))
                time.sleep(2)
            return ""

        # ---------------- 쿼리 ----------------

        def _ad_build_where(mode, q, filters):
            where, params = [], []
            for token in _ad_parse_query(q):
                where.append("search_blob LIKE ? ESCAPE '\\'")
                params.append(_ad_like(token))
            if mode == "characters":
                chosen = filters.get("copyright")
                if chosen:
                    where.append("copyright IN (%s)" % ",".join("?" * len(chosen)))
                    params.extend(chosen)
                for facet in AD_TRAIT_FACETS:
                    chosen = filters.get(facet)
                    if chosen:
                        where.append(
                            "character IN (SELECT character FROM traits "
                            "WHERE facet = ? AND value IN (%s))"
                            % ",".join("?" * len(chosen)))
                        params.append(facet)
                        params.extend(chosen)
            else:
                sel = filters.get("score")
                if sel:
                    buckets = [b for b in AD_SCORE_BUCKETS if b[0] in sel]
                    if buckets:
                        where.append("(" + " OR ".join(
                            "(score >= ? AND score < ?)" for _ in buckets) + ")")
                        for _, _, lo, hi in buckets:
                            params += [lo, hi]
                cats = filters.get("category")
                if cats:
                    where.append(
                        "artist IN (SELECT artist FROM artist_categories "
                        "WHERE category IN (%s))" % ",".join("?" * len(cats)))
                    params.extend(cats)
            return (" WHERE " + " AND ".join(where)) if where else "", params

        def _ad_query(mode, q, filters, sort, page, page_size):
            conn, data_dir = _ad_conn()
            if conn is None:
                return {"ok": False, "error": _AD_CACHE.get("error") or "데이터 사용 불가",
                        "items": [], "total": 0}
            page = max(1, int(page or 1))
            page_size = max(1, min(240, int(page_size or 60)))
            where, params = _ad_build_where(mode, q, filters)

            if mode == "artists":
                table, cols = "artists", ("artist", "name", "trigger", "count",
                                          "url", "score", "imgname", "thumbname",
                                          "image_version")
            else:
                table, cols = "characters", ("character", "copyright", "name",
                                             "copyright_name", "trigger", "core_tags",
                                             "count", "url", "imgname", "thumbname",
                                             "image_version")
            sel = ",".join(cols)
            try:
                total = conn.execute(
                    "SELECT COUNT(*) FROM %s%s" % (table, where), params).fetchone()[0]
                order = ("name_lower" if sort == "az"
                         else ("count" if sort == "count" else "count DESC, name_lower"))
                if sort == "newest":
                    order = "count DESC, name_lower"
                rows = conn.execute(
                    "SELECT %s FROM %s%s ORDER BY %s LIMIT ? OFFSET ?"
                    % (sel, table, where, order),
                    params + [page_size, (page - 1) * page_size]).fetchall()
            except Exception as e:
                return {"ok": False, "error": str(e), "items": [], "total": 0}

            items = []
            for r in rows:
                d = {k: r[k] for k in r.keys()}
                if mode == "artists":
                    prompt = (d.get("trigger") or "").strip()
                    items.append({
                        "key": "a:" + d["artist"],
                        "slug": d["artist"],
                        "name": d.get("name") or d["artist"],
                        "sub": "",
                        "trigger": d.get("trigger") or "",
                        "prompt": prompt,
                        "tags": [],
                        "count": d.get("count") or 0,
                        "url": d.get("url") or "",
                        "score": d.get("score"),
                        "thumb": d.get("thumbname") or "",
                        "full": d.get("imgname") or "",
                        "v": d.get("image_version") or 0,
                    })
                else:
                    tags = [t.strip() for t in (d.get("core_tags") or "").split(",") if t.strip()]
                    items.append({
                        "key": "c:" + d["character"],
                        "slug": d["character"],
                        "name": d.get("name") or d["character"],
                        "sub": d.get("copyright_name") or d.get("copyright") or "",
                        "copyright": d.get("copyright") or "",
                        "copyright_name": d.get("copyright_name") or "",
                        "trigger": d.get("trigger") or "",
                        "prompt": _ad_prompt_of(d.get("trigger"), d.get("core_tags")),
                        "tags": tags,
                        "count": d.get("count") or 0,
                        "url": d.get("url") or "",
                        "thumb": d.get("thumbname") or "",
                        "full": d.get("imgname") or "",
                        "v": d.get("image_version") or 0,
                    })
            return {
                "ok": True, "items": items, "total": total,
                "page": page, "pages": max(1, int(math.ceil(total / float(page_size)))),
                "source": _AD_CACHE.get("source", ""), "mode": mode,
            }

        def _ad_facets(mode, facet, q, limit=60):
            conn, _dd = _ad_conn()
            if conn is None:
                return {"ok": False, "error": _AD_CACHE.get("error") or "데이터 사용 불가",
                        "items": []}
            limit = max(1, min(300, int(limit or 60)))
            q = (q or "").strip().lower()
            try:
                if mode == "artists":
                    if facet == "score":
                        return {"ok": True, "items": [
                            {"value": b[0], "label": b[1], "n": 0}
                            for b in AD_SCORE_BUCKETS]}
                    if facet == "category":
                        w = "WHERE category LIKE ? ESCAPE '\\'" if q else ""
                        rows = conn.execute(
                            "SELECT category AS value, category AS label, COUNT(*) AS n "
                            "FROM artist_categories %s GROUP BY category "
                            "ORDER BY n DESC, label LIMIT ?" % w,
                            ([_ad_like(q)] if q else []) + [limit]).fetchall()
                        return {"ok": True, "items": [
                            {"value": r["value"], "label": r["label"], "n": r["n"]}
                            for r in rows]}
                    return {"ok": True, "items": []}

                if facet == "copyright":
                    w = ("WHERE (copyright LIKE ? ESCAPE '\\' "
                         "OR copyright_name LIKE ? ESCAPE '\\')") if q else ""
                    rows = conn.execute(
                        "SELECT copyright AS value, copyright_name AS label, "
                        "COUNT(*) AS n FROM characters %s "
                        "GROUP BY copyright ORDER BY n DESC, label LIMIT ?" % w,
                        ([_ad_like(q), _ad_like(q)] if q else []) + [limit]).fetchall()
                    return {"ok": True, "items": [
                        {"value": r["value"],
                         "label": r["label"] or r["value"], "n": r["n"]}
                        for r in rows]}

                if facet in AD_TRAIT_FACETS:
                    w = ("AND (value LIKE ? ESCAPE '\\' OR label LIKE ? ESCAPE '\\')") if q else ""
                    rows = conn.execute(
                        "SELECT value, label, COUNT(*) AS n FROM traits "
                        "WHERE facet = ? %s GROUP BY value, label "
                        "ORDER BY n DESC, label LIMIT ?" % w,
                        [facet] + ([_ad_like(q), _ad_like(q)] if q else []) + [limit]
                    ).fetchall()
                    return {"ok": True, "items": [
                        {"value": r["value"], "label": r["label"] or r["value"],
                         "n": r["n"]} for r in rows]}
            except Exception as e:
                return {"ok": False, "error": str(e), "items": []}
            return {"ok": True, "items": []}

        def _ad_lookup(mode, slug):
            if _ad_online_active() == "online":
                return _ad_online_lookup(mode, slug)
            conn, _dd = _ad_conn()
            if conn is None or not slug:
                return None
            try:
                if mode == "artists":
                    r = conn.execute(
                        "SELECT artist, name, trigger, count, url, score, "
                        "imgname, thumbname, image_version FROM artists "
                        "WHERE artist = ? LIMIT 1", (slug,)).fetchone()
                    if not r:
                        return None
                    return {
                        "mode": "artists", "slug": r["artist"],
                        "name": r["name"] or r["artist"], "sub": "",
                        "copyright": "", "copyright_name": "",
                        "trigger": r["trigger"] or "", "core_tags": "",
                        "prompt": (r["trigger"] or "").strip(),
                        "tags": [], "count": r["count"] or 0, "url": r["url"] or "",
                        "score": r["score"], "thumb": r["thumbname"] or "",
                        "full": r["imgname"] or "", "v": r["image_version"] or 0,
                    }
                r = conn.execute(
                    "SELECT character, copyright, name, copyright_name, trigger, "
                    "core_tags, count, url, imgname, thumbname, image_version "
                    "FROM characters WHERE character = ? LIMIT 1", (slug,)).fetchone()
                if not r:
                    return None
                tags = [t.strip() for t in (r["core_tags"] or "").split(",") if t.strip()]
                return {
                    "mode": "characters", "slug": r["character"],
                    "name": r["name"] or r["character"],
                    "sub": r["copyright_name"] or r["copyright"] or "",
                    "copyright": r["copyright"] or "",
                    "copyright_name": r["copyright_name"] or "",
                    "trigger": r["trigger"] or "", "core_tags": r["core_tags"] or "",
                    "prompt": _ad_prompt_of(r["trigger"], r["core_tags"]),
                    "tags": tags, "count": r["count"] or 0, "url": r["url"] or "",
                    "score": None, "thumb": r["thumbname"] or "",
                    "full": r["imgname"] or "", "v": r["image_version"] or 0,
                }
            except Exception:
                return None

        def _ad_expand(names, mode=""):
            """필요에 따라 펼치기: 힌트에서 실제로 나타난 이름만 확인。

            반환 (map, info)：
              map  이름 → 플러그인 작성자 버튼 크기가 약간 작음）
              info 이름 → {thumb, full, name, sub, mode}（편집기 썸네일 컨트롤 사용）
            info 선택적 추가 정보이며, 맵의 계약은 변경되지 않음。
            """
            if _ad_online_active() == "online":
                return _ad_online_expand(names, mode)
            conn, _dd = _ad_conn()
            out, info = {}, {}
            if conn is None or not names:
                return out, info
            names = [str(n).strip() for n in names if str(n).strip()]
            if not names:
                return out, info
            modes = [mode] if mode in ("characters", "artists") else ["characters", "artists"]
            left = list(dict.fromkeys(names))
            for m in modes:
                if not left:
                    break
                table = "characters" if m == "characters" else "artists"
                key = "character" if m == "characters" else "artist"
                # 아티스트 테이블에 core_tags 열이 없으므로 모드별로 열을 선택하여 전체 SQL이 중단되지 않도록
                # sub 디스플레이용(작품명/작가), 작가가 작품이 없으면 빈 문자열을 자리 표시합니다.
                extra = (", core_tags, copyright_name AS sub"
                         if m == "characters" else ", '' AS sub") + \
                        ", imgname, thumbname"
                # 왼쪽과 평행한 '소문자' + 공백 형식, 일치에 사용 name_lower，
                # 이렇게 ~~Hatsune_Miku / ~~hatsune_miku 모두 맞을 수 있다
                norm = [x.replace("_", " ").strip().lower() for x in left]
                try:
                    ph = ",".join("?" * len(left))
                    rows = conn.execute(
                        "SELECT %s, name, trigger%s FROM %s "
                        "WHERE %s IN (%s) OR name_lower IN (%s)"
                        % (key, extra, table, key, ph, ph),
                        left + norm).fetchall()
                except Exception as e:
                    print(f"[ComfyUI Panel] AnimaDex 쿼리 실패 확장: {e}")
                    rows = []
                for r in rows:
                    slug = r[key]
                    body = (_ad_prompt_of(r["trigger"], r["core_tags"])
                            if m == "characters" else (r["trigger"] or "").strip())
                    # 타겟은 slug일 수도 있고 '이름' 형식일 수도 있으며, 두 형식 모두 쓸 수 있음
                    hit = None
                    for cand in (slug, str(slug).replace(" ", "_"),
                                 str(slug).replace("_", " ")):
                        if cand in left:
                            hit = cand
                            break
                    if hit is None:
                        low = str(slug).replace("_", " ").strip().lower()
                        for cand in list(left):
                            if cand.replace("_", " ").strip().lower() == low:
                                hit = cand
                                break
                    if hit is None:
                        # name_lower 일치하는 경우: 규칙 형식에 해당하는 원래 형식으로 되돌림
                        low = str(r["trigger"]).strip().lower()
                        for cand in list(left):
                            if cand.replace("_", " ").strip().lower() == low:
                                hit = cand
                                break
                    if hit is not None:
                        out[hit] = body
                        try:
                            info[hit] = {
                                "slug": slug,
                                "mode": m,
                                "name": r["name"] or slug,
                                "sub": r["sub"] or ("" if m == "artists" else ""),
                                "thumb": r["thumbname"] or "",
                                "full": r["imgname"] or "",
                                "prompt": body,
                            }
                        except Exception:
                            info[hit] = {"slug": slug, "mode": m, "thumb": "",
                                         "full": "", "name": slug, "sub": "",
                                         "prompt": body}
                        left.remove(hit)
            return out, info

        # ---------------- 로컬 이미지 라이브러리에 저장 (카테고리 상속)） ----------------

        def _ad_favorite(mode, slugs, category_mode=""):
            cfg = _ad_read_cfg()
            cmode = category_mode or cfg.get("category_mode") or "copyright"
            category = "AnimaDex"
            added, skipped, failed = [], [], []
            for slug in slugs:
                row = _ad_lookup(mode, slug)
                if not row:
                    failed.append({"slug": slug, "error": "데이터에 이 항목이 없습니다"})
                    continue
                trigger = _ad_trigger_of(slug)
                prompt = (row.get("prompt") or "").strip()
                if not trigger or not prompt:
                    failed.append({"slug": slug, "error": "유발어 또는 프롬프트가 비어 있습니다"})
                    continue

                subcategory = ""
                if cmode == "copyright":
                    subcategory = row.get("copyright_name") or row.get("copyright") or ""
                    if not subcategory:
                        subcategory = "화가" if mode == "artists" else "역할"
                elif cmode == "source":
                    subcategory = "화가" if mode == "artists" else "역할"

                # 동일 트리거 단어가 있는 로컬 항목을 건너뛰고, 사용자가 쓴 것을 덮어쓰지 마세요.
                dup = None
                try:
                    _idx, items = _pg_collect()
                    for it in items:
                        if (it.get("trigger") or "").strip() == trigger:
                            dup = it
                            break
                except Exception:
                    pass
                if dup:
                    skipped.append({"slug": slug, "id": dup.get("id")})
                    continue

                try:
                    _pg_ensure_dirs()
                    item_id = _pg_unique_id(_pg_slug(trigger))
                    # 미리보는 이미지를 AnimaDex의 기존 webp로 직접 복사하고 다시 인코딩하지 않습니다.
                    thumb_src = None
                    data_dir = _ad_resolve()
                    if data_dir and row.get("thumb"):
                        sub = "artists" if mode == "artists" else "characters"
                        cand = Path(data_dir) / sub / "thumbs" / row["thumb"]
                        if cand.exists():
                            thumb_src = cand
                    if thumb_src is None and row.get("thumb_url"):
                        # 온라인 데이터 소스: 먼저 원격 썸네일을 캐시 디렉토리로 가져오고, 이미지 라이브러리에 복사합니다
                        thumb_src = _ad_online_image_file(row["thumb_url"])
                    if thumb_src:
                        shutil.copyfile(str(thumb_src),
                                        str(PGALLERY_IMAGES / (item_id + ".webp")))
                    else:
                        continue     # 아직 다운되지 않은 항목은 처리하지 않고, 완료 후 클릭

                    _pg_write_txt(PGALLERY_IMAGES / (item_id + ".txt"), trigger, prompt)

                    idx = _pg_load_index()
                    m = idx.setdefault("meta", {}).setdefault(item_id, {})
                    m["category"] = category
                    m["subcategory"] = subcategory
                    m["trigger"] = trigger
                    m["mtime"] = int(time.time())
                    m["source"] = "animadex"
                    cats = idx.setdefault("categories", [])
                    if category not in cats:
                        cats.append(category)
                    subs = idx.setdefault("subcategories", {}).setdefault(category, [])
                    if subcategory and subcategory not in subs:
                        subs.append(subcategory)
                    _pg_save_index(idx)
                    _pg_make_thumb(PGALLERY_IMAGES / (item_id + ".webp"), item_id)
                    added.append({"slug": slug, "id": item_id, "trigger": trigger,
                                  "category": category, "subcategory": subcategory})
                except Exception as e:
                    failed.append({"slug": slug, "error": str(e)})
            return {"added": added, "skipped": skipped, "failed": failed}

        # ---------------- 라우팅 ----------------

        @server.routes.get("/bsk/animadex/status")
        async def animadex_status(request):
            """데이터 상태: 경로 / 링크 / 데이터 소스 / 개수 / 가져오기 여부。"""
            try:
                force = request.query.get("refresh") in ("1", "true", "yes")
                cfg = _ad_read_cfg()

                # 복사 불가 (읽기 전용 / 권한 부족 / 디스크 꽉 참)이면 플러그인 디렉토리 원 경로로 되돌림，
                # 직접 온라인 뷰포트 상태를 되돌립니다. 프런트엔드에서 분기할 필요가 없습니다.。
                if _ad_online_active(cfg) == "online":
                    if force or not _AD_ONLINE.get("total"):
                        await _ad_run(_ad_online_probe)
                    return web.json_response({
                        "success": True,
                        "platform": _ad_platform(),
                        "is_windows": _ad_is_windows(),
                        "data_dir": _ad_local_dir(),
                        # 온라인 모드에서는 로컬 데이터 패키지를 탐지할 수 없습니다. 디스크를 낭비하지 마세요.；
                        # 로컬 데이터를 설치하면 "로컬" 쪽에서 자동으로 목록이 표시됩니다.
                        "exists": True,
                        "candidates": [],
                        "link": _ad_link_info(),
                        "source": "online",
                        "source_mode": cfg.get("source") or "auto",
                        "counts": {
                            "characters": _AD_ONLINE.get("total") or 0,
                            "artists": 0, "csv_characters": 0, "csv_artists": 0,
                            "thumbs": 0, "images": 0, "artist_thumbs": 0,
                        },
                        "importing": False,
                        "built": {},
                        "repo": {"dir": cfg.get("repo_dir") or "",
                                 "exists": False, "has_venv": False,
                                 "has_config": False, "has_install": False},
                        "has_git": bool(_ad_git_exe()),
                        "online": {
                            "base": _ad_online_base(cfg),
                            "ok": not bool(_AD_ONLINE.get("err")),
                            "total": _AD_ONLINE.get("total") or 0,
                            "error": _AD_ONLINE.get("err") or "",
                            "cache": _ad_online_cache_stats(force),
                            "cache_enabled": _ad_online_cache_on(),
                        },
                        "settings": {
                            "source": cfg.get("source", "auto"),
                            "online_base": _ad_online_base(cfg),
                            "online_cache_mb": cfg.get("online_cache_mb", 512),
                            "online_cache_enabled": cfg.get("online_cache_enabled", True),
                            "install_scoring": cfg.get("install_scoring", False),
                            "install_generation": cfg.get("install_generation", False),
                            "auto_link": cfg.get("auto_link", True),
                            "expand_enabled": cfg.get("expand_enabled", True),
                            "expand_mode": cfg.get("expand_mode", "trigger_tags"),
                            "category_mode": cfg.get("category_mode", "copyright"),
                            "page_size": cfg.get("page_size", 60),
                            "repo_url": cfg.get("repo_url", AD_DEFAULT_REPO_URL),
                        },
                        "error": _AD_ONLINE.get("err") or "",
                    })

                if force:
                    try:
                        _ad_conn(True)
                    except Exception:
                        pass
                    _ad_stats(_ad_resolve(), True)

                data_dir = _ad_resolve()
                cfg0 = _ad_read_cfg()
                link = (_ad_ensure_link(data_dir, force=False)
                        if (data_dir and cfg0.get("auto_link", True))
                        else _ad_link_info())

                conn, _dd = _ad_conn()
                source = _AD_CACHE.get("source", "none")
                error = _AD_CACHE.get("error", "")
                chars = artists = 0
                built = {}
                if conn is not None:
                    try:
                        chars = conn.execute(
                            "SELECT COUNT(*) FROM characters").fetchone()[0]
                        artists = conn.execute(
                            "SELECT COUNT(*) FROM artists").fetchone()[0]
                        if source == "db":
                            for r in conn.execute(
                                    "SELECT key, value FROM meta").fetchall():
                                built[r["key"]] = r["value"]
                    except Exception as e:
                        error = error or str(e)

                st = _ad_stats(data_dir) if data_dir else {}
                n_csv_c = _AD_CACHE.get("n_csv", 0)
                importing = bool(data_dir) and (
                    (n_csv_c > max(chars, 0)) or
                    (st.get("thumbs", 0) < max(chars, n_csv_c))
                )

                repo = cfg.get("repo_dir") or ""
                repo_info = {"dir": repo, "exists": False, "has_venv": False,
                             "has_config": False, "has_install": False}
                # 설정 경로가 우선이지만 유효하지 않아도 계속 추측 ——
                # 사용자가 저장소를 옮기거나 삭제한 후에는 그것이 사라지지 않도록 해야 합니다.。
                for cand in _ad_repo_guesses():
                    if cand and os.path.isdir(cand):
                        p = Path(cand)
                        repo_info = {
                            "dir": str(p),
                            "exists": True,
                            "has_venv": (p / ".venv").is_dir()
                                        or (p / "venv").is_dir(),
                            "has_config": (p / "config.toml").exists(),
                            "has_install": (p / "install.sh").exists()
                                           or (p / "install.bat").exists(),
                        }
                        break

                return web.json_response({
                    "success": True,
                    "platform": _ad_platform(),
                    "is_windows": _ad_is_windows(),
                    "data_dir": data_dir,
                    "exists": bool(data_dir),
                    "candidates": [c for c in _ad_detect_candidates()
                                   if _ad_is_data_dir(c)][:10],
                    "link": link,
                    "source": source,
                    "counts": {
                        "characters": chars, "artists": artists,
                        "csv_characters": n_csv_c,
                        "csv_artists": (_ad_csv_row_count(
                            Path(data_dir) / "import" / "artists.csv")
                            if data_dir and (Path(data_dir) / "import" / "artists.csv").exists()
                            else 0),
                        "thumbs": st.get("thumbs", 0),
                        "images": st.get("images", 0),
                        "artist_thumbs": st.get("artist_thumbs", 0),
                    },
                    "importing": importing,
                    "built": built,
                    "repo": repo_info,
                    "has_git": bool(_ad_git_exe()),
                    "source_mode": cfg.get("source") or "auto",
                    "online": {
                        "base": _ad_online_base(cfg),
                        "ok": not bool(_AD_ONLINE.get("err")),
                        "total": _AD_ONLINE.get("total") or 0,
                        "error": _AD_ONLINE.get("err") or "",
                        "cache": _ad_online_cache_stats(),
                        "cache_enabled": _ad_online_cache_on(),
                    },
                    "settings": {
                        "source": cfg.get("source", "auto"),
                        "online_base": _ad_online_base(cfg),
                        "online_cache_mb": cfg.get("online_cache_mb", 512),
                        "online_cache_enabled": cfg.get("online_cache_enabled", True),
                        "install_scoring": cfg.get("install_scoring", False),
                        "install_generation": cfg.get("install_generation", False),
                        "auto_link": cfg.get("auto_link", True),
                        "expand_enabled": cfg.get("expand_enabled", True),
                        "expand_mode": cfg.get("expand_mode", "trigger_tags"),
                        "category_mode": cfg.get("category_mode", "copyright"),
                        "page_size": cfg.get("page_size", 60),
                        "repo_url": cfg.get("repo_url", AD_DEFAULT_REPO_URL),
                    },
                    "error": error,
                })
            except Exception as e:
                import traceback
                traceback.print_exc()
                return web.json_response({"success": False, "error": str(e)})

        @server.routes.get("/bsk/animadex/list")
        async def animadex_list(request):
            """페이지 검색。?mode=characters|artists&q=&page=&sort=&<facet>=a,b&src=local|online"""
            try:
                mode = request.query.get("mode", "characters")
                if mode not in ("characters", "artists"):
                    mode = "characters"
                filters = {}
                for f in (("copyright",) + AD_TRAIT_FACETS if mode == "characters"
                          else AD_ARTIST_FACETS):
                    raw = request.query.get(f, "")
                    vals = [v.strip() for v in raw.split(",") if v.strip()]
                    if vals:
                        filters[f] = vals
                cfg = _ad_read_cfg()
                src = (request.query.get("src") or "").strip().lower()
                if src not in ("local", "online"):
                    src = _ad_online_active(cfg)
                args = (
                    mode,
                    request.query.get("q", ""),
                    filters,
                    request.query.get("sort", "count"),
                    int(request.query.get("page", 1) or 1),
                    int(request.query.get("size", cfg.get("page_size", 60)) or 60),
                )
                # 온라인은 네트워크를 사용, 스레드 풀을 잃고, 이벤트 루프를 막지 마세요
                res = (await _ad_run(_ad_online_search, *args) if src == "online"
                       else _ad_query(*args))
                res["success"] = res.pop("ok")
                if not res["success"]:
                    res["items"] = []
                return web.json_response(res)
            except Exception as e:
                return web.json_response(
                    {"success": False, "error": str(e), "items": [], "total": 0})

        @server.routes.get("/bsk/animadex/facets")
        async def animadex_facets(request):
            """면 후보：?mode=&facet=copyright|hair_color|...&q=&src=local|online"""
            try:
                cfg = _ad_read_cfg()
                src = (request.query.get("src") or "").strip().lower()
                if src not in ("local", "online"):
                    src = _ad_online_active(cfg)
                args = (
                    request.query.get("mode", "characters"),
                    request.query.get("facet", "copyright"),
                    request.query.get("q", ""),
                    int(request.query.get("limit", 60) or 60),
                )
                res = (await _ad_run(_ad_online_facets, *args) if src == "online"
                       else _ad_facets(*args))
                res["success"] = res.pop("ok")
                res.setdefault("items", [])
                return web.json_response(res)
            except Exception as e:
                return web.json_response({"success": False, "error": str(e), "items": []})

        @server.routes.get("/bsk/animadex/online/image")
        async def animadex_online_image(request):
            """선택된 이미지의 선：?url=<원격 주소>。
            캐시된 데이터는 디스크에서 직접 읽고, 캐시되지 않은 데이터는 가져와 캐시를 켜둔 경우 저장합니다.。"""
            url = (request.query.get("url") or "").strip()
            if not url:
                return web.Response(status=400, text="url required")
            if not _ad_url_host_ok(url):
                return web.Response(status=400, text="bad host")
            raw, ctype = await _ad_run(_ad_online_image_get, url)
            if not raw:
                return web.Response(status=502, text="fetch failed",
                                    headers={"Cache-Control": "no-store"})
            return web.Response(body=raw, content_type=ctype or "image/webp",
                                headers={"Cache-Control": "public, max-age=86400"})

        @server.routes.post("/bsk/animadex/online/cache/clear")
        async def animadex_online_cache_clear(request):
            """온라인 미리보기 캐시를 지웁니다 (사용자가 수동으로 클릭하며 '자동 캐시' 스위치와 관련이 없습니다).）。"""
            try:
                res = await _ad_run(_ad_online_cache_clear)
                return web.json_response({
                    "success": True, "cleared": res,
                    "cache": _ad_online_cache_stats(True),
                })
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        @server.routes.post("/bsk/animadex/online/ping")
        async def animadex_online_ping(request):
            """온라인 인터페이스 통신 상태 확인: 총 개수 및 캐시 사용량 반환。"""
            try:
                ok = await _ad_run(_ad_online_probe)
                return web.json_response({
                    "success": True, "ok": ok,
                    "base": _ad_online_base(),
                    "total": _AD_ONLINE.get("total") or 0,
                    "error": _AD_ONLINE.get("err") or "",
                    "cache": _ad_online_cache_stats(True),
                })
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        @server.routes.get("/bsk/animadex/image")
        async def animadex_image(request):
            """직접 AnimaDex의 이미지 읽기。?mode=characters&name=<thumbname>&size=thumb"""
            try:
                data_dir = _ad_resolve()
                if not data_dir:
                    return web.Response(status=404, text="no data dir")
                mode = request.query.get("mode", "characters")
                if mode not in ("characters", "artists", "copyrights"):
                    return web.Response(status=400, text="bad mode")
                name = os.path.basename(request.query.get("name", "") or "")
                size = request.query.get("size", "thumb")
                if not name or name in (".", "..") or "/" in name or "\\" in name:
                    return web.Response(status=400, text="bad name")

                if mode == "copyrights":
                    rel = Path("copyrights") / "thumbs" / name
                elif size == "full":
                    rel = Path(mode) / "images" / name
                else:
                    rel = Path(mode) / "thumbs" / name
                src = Path(data_dir) / rel
                if not src.exists() and size == "full":
                    # 전체 크기 이미지가 다운로드되지 않음 (import 시 추가되지 않음) --with-images）→ 취소 미리보기
                    src = Path(data_dir) / mode / "thumbs" / Path(name).with_suffix(".webp").name
                if not src.exists() or not src.is_file():
                    return web.Response(status=404, text="not found",
                                        headers={"Cache-Control": "no-store"})

                ext = src.suffix.lower()
                ctype = {".webp": "image/webp", ".png": "image/png",
                         ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                         ".gif": "image/gif", ".bmp": "image/bmp"}.get(
                             ext, "application/octet-stream")
                with open(src, "rb") as f:
                    data = f.read()
                return web.Response(body=data, content_type=ctype, headers={
                    "Cache-Control": "public, max-age=86400"})
            except Exception as e:
                return web.Response(status=500, text=str(e))

        @server.routes.post("/bsk/animadex/expand")
        async def animadex_expand(request):
            """필요에 따라 펼치기 트리거：body {names:[...], mode?} → {map:{이름: 프롬프트}}"""
            try:
                data = await request.json()
                names = data.get("names") or []
                if isinstance(names, str):
                    names = [names]
                names = [str(n) for n in names][:200]
                m, info = _ad_expand(names, (data.get("mode") or "").strip())
                return web.json_response(
                    {"success": True, "map": m, "info": info})
            except Exception as e:
                return web.json_response({"success": False, "error": str(e), "map": {}})

        @server.routes.post("/bsk/animadex/favorite")
        async def animadex_favorite(request):
            """AnimaDex 항목을 로컬 프롬프트 갤러리에 저장 (썸네일 복사) + 쓰다 txt + 상속 분류）。"""
            try:
                data = await request.json()
                mode = data.get("mode", "characters")
                if mode not in ("characters", "artists"):
                    mode = "characters"
                slugs = data.get("slugs") or ([data.get("slug")] if data.get("slug") else [])
                slugs = [str(s) for s in slugs][:200]
                if not slugs:
                    return web.json_response(
                        {"success": False, "error": "선택할 항목이 지정되지 않았습니다"})
                res = _ad_favorite(mode, slugs, data.get("category_mode") or "")
                return web.json_response({"success": True, **res})
            except Exception as e:
                import traceback
                traceback.print_exc()
                return web.json_response({"success": False, "error": str(e)})

        @server.routes.post("/bsk/animadex/config")
        async def animadex_config(request):
            """설정 저장; 편리하게 데이터 디렉토리 지정 / 링크 생성 / 링크 끊기。"""
            try:
                data = await request.json()
                cfg = _ad_read_cfg()
                for k in ("data_dir", "repo_dir", "repo_url", "expand_mode",
                          "category_mode", "source", "online_base"):
                    if k in data and isinstance(data[k], str):
                        cfg[k] = data[k].strip()
                for k in ("auto_link", "expand_enabled", "online_cache_enabled",
                          "install_scoring", "install_generation"):
                    if k in data and isinstance(data[k], bool):
                        cfg[k] = data[k]
                if "page_size" in data:
                    try:
                        cfg["page_size"] = max(12, min(240, int(data["page_size"])))
                    except Exception:
                        pass
                if "online_cache_mb" in data:
                    try:
                        cfg["online_cache_mb"] = max(
                            64, min(8192, int(data["online_cache_mb"])))
                    except Exception:
                        pass
                # 데이터 소스/데이터 디렉토리를 변경하면 온라인 측의 단기 캐시를 지웁니다. 이전 결과를 사용하여 속임수를 피하기 위해.
                if ("source" in data or "online_base" in data or "data_dir" in data):
                    _AD_ONLINE["search"] = {}
                    _AD_ONLINE["facets"] = {}
                    _AD_ONLINE["lookup"] = {}
                    _AD_ONLINE["resolve"] = None
                    _AD_ONLINE["resolve_dir"] = ""
                    _AD_ONLINE["err"] = ""

                action = (data.get("action") or "").strip()
                link_res = None
                if action == "link":
                    # 사용자가 명시적으로 선택한 → force：기존 링크를 여기로 변경 허용
                    src = (cfg.get("data_dir") or _ad_resolve() or "").strip()
                    link_res = _ad_ensure_link(src, force=True)
                elif action == "unlink":
                    try:
                        _ad_unlink()
                        link_res = {"ok": True, "created": False, "kind": "unlinked"}
                    except Exception as e:
                        link_res = {"ok": False, "kind": "failed", "detail": str(e)}
                elif action == "setup_local":
                    # 한 번 설치: 복제 + 설치 스크립트 실행 + 자동 연결, 단 폴더 위치만 확인
                    if not _ad_git_exe():
                        return web.json_response({
                            "success": False,
                            "error": "Git을 찾을 수 없습니다. 먼저 Git을 설치하고 추가하십시오 PATH"})
                    target = (data.get("dir") or "").strip()
                    if not target:
                        base = Path(folder_paths.base_path)
                        target = str(base.parent / "AnimaDex")
                    target = os.path.abspath(os.path.expanduser(target))
                    url = (data.get("url") or cfg.get("repo_url")
                           or AD_DEFAULT_REPO_URL).strip()
                    tid = _ad_task_new("setup_local", "AnimaDex 설치 (로컬 데이터 패키지）")
                    _threading.Thread(
                        target=_ad_setup_local_worker,
                        args=(tid, url, target,
                              bool(cfg.get("install_scoring")),
                              bool(cfg.get("install_generation"))),
                        daemon=True, name="bsk-animadex-setup-local").start()

                    cfg["source"] = "local"          # 완료되면 로컬 사용
                    cfg["repo_dir"] = target
                    cfg["repo_url"] = url
                    _ad_write_cfg(cfg)
                    _AD_ONLINE["resolve"] = None
                    _AD_ONLINE["resolve_dir"] = ""
                    return web.json_response({
                        "success": True, "task": tid, "dir": target,
                        "action_result": {"kind": "setup_local", "task": tid},
                    })

                _ad_write_cfg(cfg)
                try:
                    _ad_conn(True)          # 폴더를 변경하면 연결을 즉시 재구성합니다.
                except Exception:
                    pass
                return web.json_response({
                    "success": True,
                    "settings": cfg,
                    "data_dir": _ad_resolve(),
                    "link": _ad_link_info(),
                    "action_result": link_res,
                })
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        @server.routes.get("/bsk/animadex/browse")
        async def animadex_browse(request):
            """경로 선택기용 디렉토리 목록。?path=<dir>&mode=dirs|all"""
            try:
                raw = (request.query.get("path") or "").strip()
                roots = []
                if raw:
                    roots.append(os.path.abspath(os.path.expanduser(raw)))
                else:
                    base = Path(folder_paths.base_path)
                    for p in (base, base.parent, Path.home(),
                              Path.home() / "Desktop", USER_PANEL_DIR):
                        try:
                            if p and Path(p).is_dir():
                                roots.append(str(Path(p).resolve()))
                        except Exception:
                            pass
                    seen, uniq = set(), []
                    for r in roots:
                        if r.lower() not in seen:
                            seen.add(r.lower())
                            uniq.append(r)
                    roots = uniq[:1]
                target = roots[0] if roots else str(Path.home())
                if not os.path.isdir(target):
                    return web.json_response(
                        {"success": False, "error": "폴더가 없습니다: %s" % target})

                entries = []
                try:
                    for name in sorted(os.listdir(target)):
                        if name.startswith(".") and name not in (".venv",):
                            continue
                        full = os.path.join(target, name)
                        try:
                            isdir = os.path.isdir(full)
                        except Exception:
                            continue
                        entries.append({"name": name, "path": full, "dir": isdir,
                                        "is_data": isdir and _ad_is_data_dir(full)})
                except Exception as e:
                    return web.json_response({"success": False, "error": str(e)})

                parent = os.path.dirname(target.rstrip(os.sep)) or target
                return web.json_response({
                    "success": True, "path": target,
                    "parent": parent if parent != target else "",
                    "entries": entries,
                })
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        @server.routes.post("/bsk/animadex/clone")
        async def animadex_clone(request):
            """백그라운드 git clone (블로킹 없음)）。body {url, dir, branch}"""
            try:
                data = await request.json()
                url = (data.get("url") or AD_DEFAULT_REPO_URL).strip()
                target = (data.get("dir") or "").strip()
                if not target:
                    base = Path(folder_paths.base_path)
                    target = str(base.parent / "AnimaDex")
                target = os.path.abspath(os.path.expanduser(target))
                if not _ad_git_exe():
                    return web.json_response(
                        {"success": False, "error": "Git을 찾을 수 없습니다. 먼저 Git을 설치하고 PATH에 추가하십시오. PATH"})
                tid = _ad_task_new("clone", "복제 AnimaDex")
                _threading.Thread(
                    target=_ad_clone_worker,
                    args=(tid, url, target, (data.get("branch") or "").strip()),
                    daemon=True, name="bsk-animadex-clone").start()

                cfg = _ad_read_cfg()
                cfg["repo_dir"] = target
                cfg["repo_url"] = url
                _ad_write_cfg(cfg)
                return web.json_response({"success": True, "task": tid, "dir": target})
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        @server.routes.post("/bsk/animadex/install")
        async def animadex_install(request):
            """백그라운드에서 설치 스크립트 실행 (블로킹 없음), 설치 완료 후 자동으로 데이터 디렉터리 생성 및 링크。"""
            try:
                data = await request.json()
                repo = (data.get("repo") or "").strip()
                if not repo:
                    cfg = _ad_read_cfg()
                    repo = cfg.get("repo_dir") or ""
                if not repo:
                    for cand in _ad_repo_guesses():
                        if cand and os.path.isdir(cand):
                            repo = cand
                            break
                if not repo or not os.path.isdir(repo):
                    return web.json_response(
                        {"success": False, "error": "AnimaDex 저장소 디렉터리를 찾을 수 없습니다. 먼저 클론하거나 수동으로 지정하십시오."})
                tid = _ad_task_new("install", "설치 AnimaDex")
                _threading.Thread(
                    target=_ad_install_worker,
                    args=(tid, repo, bool(data.get("with_scoring")),
                          bool(data.get("with_generation"))),
                    daemon=True, name="bsk-animadex-install").start()
                return web.json_response({"success": True, "task": tid, "repo": repo})
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        @server.routes.get("/bsk/animadex/task")
        async def animadex_task(request):
            """백그라운드 작업 진행률。?id=<tid>；id가 없으면 가장 최근 항목을 반환합니다。"""
            try:
                tid = (request.query.get("id") or "").strip()
                with _AD_TASKS_LOCK:
                    if not tid and _AD_TASKS:
                        tid = list(_AD_TASKS.keys())[-1]
                    t = _AD_TASKS.get(tid)
                    if not t:
                        return web.json_response(
                            {"success": False, "error": "이 작업이 없습니다"})
                    from_idx = int(request.query.get("from", 0) or 0)
                    return web.json_response({
                        "success": True,
                        "id": t["id"], "kind": t["kind"], "title": t["title"],
                        "state": t["state"], "phase": t["phase"],
                        "progress": t["progress"], "error": t["error"],
                        "result": t["result"],
                        "log": t["log"][from_idx:],
                        "total_log": len(t["log"]),
                    })
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        @server.routes.post("/bsk/animadex/task/cancel")
        async def animadex_task_cancel(request):
            """백그라운드 작업 취소 요청 (자식 프로세스에서 다음 행을 읽을 때 종료)）。"""
            try:
                data = await request.json()
                tid = (data.get("id") or "").strip()
                with _AD_TASKS_LOCK:
                    t = _AD_TASKS.get(tid)
                    if not t:
                        return web.json_response(
                            {"success": False, "error": "이 작업이 없습니다"})
                    if t["state"] == "running":
                        t["state"] = "cancelling"
                        t["phase"] = "취소 중"
                return web.json_response({"success": True})
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        # -----------------------------------------

        print(f"[ComfyUI Panel] ✓ API routes registered successfully!")
        print(f"[ComfyUI Panel] ✓ Endpoints:")
        print(f"[ComfyUI Panel]   - GET  /comfyui_panel/test")
        print(f"[ComfyUI Panel]   - GET  /comfyui_panel/hardware_stats")
        print(f"[ComfyUI Panel]   - GET  /comfyui_panel/runtime_stats  (통합 인터페이스: 하드웨어+큐)")
        print(f"[ComfyUI Panel]   - GET  /comfyui_panel/default_avatar")
        print(f"[ComfyUI Panel]   - GET  /comfyui_panel/random_seed")
        print(f"[ComfyUI Panel]   - POST /comfyui_panel/random_seed")
        print(f"[ComfyUI Panel]   - GET  /comfyui_panel/input_files")
        print(f"[ComfyUI Panel]   - GET  /comfyui_panel/output_files")
        print(f"[ComfyUI Panel]   - GET  /comfyui_panel/list_models")
        print(f"[ComfyUI Panel]   - GET  /comfyui_panel/workflow_files")
        print(f"[ComfyUI Panel]   - GET  /comfyui_panel/read_workflow")
        print(f"[ComfyUI Panel]   - POST /comfyui_panel/upload_crop")
        print(f"[ComfyUI Panel]   - POST /comfyui_panel/crop_image")
        print(f"[ComfyUI Panel]   - POST /comfyui_panel/delete_file")
        print(f"[ComfyUI Panel]   - POST /comfyui_panel/save_config")
        print(f"[ComfyUI Panel]   - GET  /comfyui_panel/list_configs")
        print(f"[ComfyUI Panel]   - GET  /comfyui_panel/load_config")
        print(f"[ComfyUI Panel]   - POST /comfyui_panel/delete_config")
        print(f"[ComfyUI Panel]   - GET  /comfyui_panel/list_categories")
        print(f"[ComfyUI Panel]   - POST /comfyui_panel/create_category")
        print(f"[ComfyUI Panel]   - POST /comfyui_panel/delete_category")
        print(f"[ComfyUI Panel]   - GET  /comfyui_panel/list_configs_v2")
        print(f"[ComfyUI Panel]   - GET  /comfyui_panel/load_config_v2")
        print(f"[ComfyUI Panel]   - GET  /comfyui_panel/templates/list")
        print(f"[ComfyUI Panel]   - GET  /comfyui_panel/templates/load")
        print(f"[ComfyUI Panel]   - POST /comfyui_panel/templates/save")
        print(f"[ComfyUI Panel]   - POST /comfyui_panel/templates/delete")
        print(f"[ComfyUI Panel]   - GET  /comfyui_panel/prompt_configs/list")
        print(f"[ComfyUI Panel]   - GET  /comfyui_panel/prompt_configs/load")
        print(f"[ComfyUI Panel]   - POST /comfyui_panel/prompt_configs/save")
        print(f"[ComfyUI Panel]   - POST /comfyui_panel/prompt_configs/delete")
        print(f"[ComfyUI Panel]   - GET  /comfyui_panel/multilora_configs/list")
        print(f"[ComfyUI Panel]   - GET  /comfyui_panel/multilora_configs/load")
        print(f"[ComfyUI Panel]   - POST /comfyui_panel/multilora_configs/save")
        print(f"[ComfyUI Panel]   - POST /comfyui_panel/multilora_configs/delete")
        print(f"[ComfyUI Panel]   - GET  /comfyui_panel/multilora_configs/thumbnail")
        print(f"[ComfyUI Panel]   - POST /comfyui_panel/multilora_configs/embed_image")
        print(f"[ComfyUI Panel]   - GET  /comfyui_panel/multilora_configs/list_categories")
        print(f"[ComfyUI Panel]   - POST /comfyui_panel/multilora_configs/create_category")
        print(f"[ComfyUI Panel]   - POST /comfyui_panel/multilora_configs/delete_category")
        print(f"[ComfyUI Panel]   - POST /comfyui_panel/save_config_v2")
        print(f"[ComfyUI Panel]   - POST /comfyui_panel/upload_config_file")
        print(f"[ComfyUI Panel]   - POST /comfyui_panel/separate_mask")
        print(f"[ComfyUI Panel]   - POST /comfyui_panel/json_to_png")
        print(f"[ComfyUI Panel]   - POST /rgba_editor/save_channel")
        print(f"[ComfyUI Panel]   - GET  /comfyui_panel/thumbnail")
        print(f"[ComfyUI Panel]   - POST /comfyui_panel/clear_thumbnails")
        print(f"[ComfyUI Panel]   - GET  /comfyui_panel/output_caption")
        print(f"[ComfyUI Panel]   - POST /comfyui_panel/output_caption")
        print(f"[ComfyUI Panel]   - POST /comfyui_panel/convert_workflow")
        print(f"[ComfyUI Panel]   - GET  /comfyui_panel/convert_workflow")
        print(f"[ComfyUI Panel]   - GET  /comfyui_panel/bsk_settings")
        print(f"[ComfyUI Panel]   - POST /comfyui_panel/bsk_settings")
        print(f"[ComfyUI Panel]   - GET  /comfyui_panel/panel_bg/list    (패널 배경 이미지: 목록+현재 설정)")
        print(f"[ComfyUI Panel]   - GET  /comfyui_panel/panel_bg/file    (패널 배경 이미지: 원본)")
        print(f"[ComfyUI Panel]   - GET  /comfyui_panel/panel_bg/thumb   (패널 배경 이미지: WebP 썸네일)")
        print(f"[ComfyUI Panel]   - POST /comfyui_panel/panel_bg/upload  (패널 배경 이미지: 업로드)")
        print(f"[ComfyUI Panel]   - POST /comfyui_panel/panel_bg/delete  (패널 배경 이미지: 삭제)")
        print(f"[ComfyUI Panel]   - GET  /bsk/pgallery/list       (프롬프트 갤러리: 항목+분류)")
        print(f"[ComfyUI Panel]   - GET  /bsk/pgallery/triggers   (프롬프트 카드 라이브러리: 트리거 단어→프롬프트 매핑)")
        print(f"[ComfyUI Panel]   - GET  /bsk/pgallery/image      (프롬프트 이미지 라이브러리: 이미지/소개 그림)")
        print(f"[ComfyUI Panel]   - POST /bsk/pgallery/save       (프롬프트 라이브러리: 추가/업데이트)")
        print(f"[ComfyUI Panel]   - POST /bsk/pgallery/delete     (프롬프트 갤러리: 삭제)")
        print(f"[ComfyUI Panel]   - POST /bsk/pgallery/categories (프롬프트 갤러리: 분류 관리)")
        print(f"[ComfyUI Panel]   - GET  /bsk/animadex/status    (AnimaDex: 파일명 접두사] 저장 경로)")
        print(f"[ComfyUI Panel]   - GET  /bsk/animadex/list      (AnimaDex: 페이지 검색+면 필터링)")
        print(f"[ComfyUI Panel]   - GET  /bsk/animadex/facets    (AnimaDex: 면 후보)")
        print(f"[ComfyUI Panel]   - GET  /bsk/animadex/image     (AnimaDex: 이미지 직접 읽기)")
        print(f"[ComfyUI Panel]   - POST /bsk/animadex/expand    (AnimaDex: 유발어는 필요에 따라 펼쳐집니다)")
        print(f"[ComfyUI Panel]   - POST /bsk/animadex/favorite  (AnimaDex: 로컬 이미지 라이브러리에 저장)")
        print(f"[ComfyUI Panel]   - POST /bsk/animadex/config    (AnimaDex: 설정/경로/링크)")
        print(f"[ComfyUI Panel]   - GET  /bsk/animadex/browse    (AnimaDex: 목차 선택)")
        print(f"[ComfyUI Panel]   - POST /bsk/animadex/clone     (AnimaDex: 저장소를 복제하고 백그라운드에서 실행하세요)")
        print(f"[ComfyUI Panel]   - POST /bsk/animadex/install   (AnimaDex: 설치, 백그라운드 차단 없음)")
        print(f"[ComfyUI Panel]   - GET  /bsk/animadex/task      (AnimaDex: 백그라운드 작업 진행률)")
        # 독립 패널 페이지 라우팅 등록 - ComfyUI 기본 프런트를 로드하지 않고도 액세스 가능
        panel_dir = os.path.dirname(os.path.abspath(__file__))

        @server.routes.get("/x")  # x: 1. 라우팅 경로 수정
        async def serve_panel_redirect(request):
            """꼬리가 있는 URL로 리디렉션하여 상대 경로가 올바르게 해석되도록 합니다."""
            return web.HTTPFound("/x/")  # x: 2. 재정위 대상 수정

        @server.routes.get("/x/")  # x: 3. 라우팅 경로 수정
        async def serve_panel_page(request):
            """독립 HTML 페이지를 제공하여 서버 주소와 카드 스크립트 목록을 자동으로 주입합니다."""
            html_path = os.path.join(panel_dir, "comfyui_panel.html")
            with open(html_path, 'r', encoding='utf-8') as f:
                html = f.read()
            # 자동 연결 플래그 주입: ComfyUI 서버에서 제공될 때 직접 연결, 수동 입력 불필요
            # 주의: 기존에 있는 곳에서 <script> 블록 내 주입, 사용 불가 <script> 레이블 포장
            # 현재 페이지 경로에서 /x/ 접미사를 제거하여 서버 주소로 사용하고, 역방향 프록시 및 jupyter-server-proxy
            inject_code = 'window.COMFYUI_PANEL_AUTO_CONNECT = true; window.COMFYUI_PANEL_SERVER_URL = location.origin + location.pathname.replace(/\\/x\\/?$/, "");'
            html = html.replace(
                'window.COMFYUI_PANEL_STANDALONE = true;',
                'window.COMFYUI_PANEL_STANDALONE = true;\n    ' + inject_code
            )
            # js/cards/ 폴더 자동 스캔 및 카드 스크립트 목록에 주입
            # 새 카드 JS 파일 추가 후 수동으로 수정할 필요가 없습니다. comfyui_panel.html
            cards_dir = os.path.join(panel_dir, "js", "cards")
            card_scripts = []
            if os.path.isdir(cards_dir):
                for fname in sorted(os.listdir(cards_dir)):
                    if fname.endswith('.js') and not fname.startswith('.'):
                        # 파일 수정 시간을 버전 파괴 매개변수로 추가하여 브라우저/CDN이 오래된 JS를 캐싱하여 코드가 적용되지 않도록 방지합니다.
                        _mtime = int(os.path.getmtime(os.path.join(cards_dir, fname)))
                        card_scripts.append(f"'./js/cards/{fname}?t={_mtime}'")
            if card_scripts:
                card_scripts_str = ',\n        '.join(card_scripts)
                html = html.replace(
                    "];  // END_SCRIPTS",
                    f",\n        {card_scripts_str}\n      ];  // END_SCRIPTS"
                )
            return web.Response(text=html, content_type='text/html')

        @server.routes.get("/x/{path:.*}")  # x: 4. 수정 정적 리소스 경로
        async def serve_panel_static(request):
            """패널에 필요한 정적 리소스 (js/css/이미지 등)）"""
            path = request.match_info.get("path", "")
            file_path = os.path.join(panel_dir, path)
            if os.path.isfile(file_path) and not path.startswith(".."):
                return web.FileResponse(file_path)
            return web.Response(text="Not Found", status=404)

        print("[ComfyUI Panel] ✓ Panel page registered: http://localhost:8188/x/")  # x: 5. 프린트 정보 수정

        # ========== 플러그인 설치 API（git clone + zip 압축 해제 다운로드） ==========
        # GitHub 속도 소스 (gh-proxy): 국내 직접 연결 GitHub는 자주 액세스할 수 없습니다. 플러그인 설치 및 종속성 매핑 파싱
        # 기본적으로 가속원을 사용합니다 (플러그인 의존성 패널 상단 드롭다운에서 원래 원으로 되돌릴 수 있음), 요청의 mirror 파라미터로 제어됨。
        # 여기서만 URL 변환만 수행하고, 프론트엔드/워크플로우 구성에 저장된 것은 항상 원본 GitHub 주소입니다.。
        GH_PROXY_PREFIX = "https://gh-proxy.org/"
        _GH_HOST_PREFIXES = (
            "https://github.com/",
            "https://raw.githubusercontent.com/",
            "https://codeload.github.com/",
            "https://objects.githubusercontent.com/",
        )

        def _strip_gh_mirror(url):
            """가속도 주소를 GitHub 직접 링크로 되돌립니다 (플러그인 자체로 덮은 접두사를 제거합니다)）。"""
            if url.startswith(GH_PROXY_PREFIX):
                rest = url[len(GH_PROXY_PREFIX):]
                if rest.startswith("https://"):
                    return rest
            return url

        def _apply_gh_mirror(url, mirror="proxy"):
            """mirror='proxy'(github 직접 링크를 래핑하여 속도 프리픽스 적용; 'direct' 또는 GitHub 주소가 아닌 경우 원본 그대로 반환。"""
            if not url or mirror == "direct" or url.startswith(GH_PROXY_PREFIX):
                return url
            return GH_PROXY_PREFIX + url if url.startswith(_GH_HOST_PREFIXES) else url

        @server.routes.post("/comfyui_panel/install_plugin_git")
        async def install_plugin_git(request):
            """git clone을 통해 플러그인을 custom_nodes 디렉토리에 설치합니다."""
            try:
                import subprocess
                import asyncio

                data = await request.json()
                git_url = data.get("git_url", "").strip()
                plugin_name = data.get("plugin_name", "").strip()
                force = bool(data.get("force", False))
                mirror = data.get("mirror", "proxy")

                if not git_url:
                    return web.json_response({"success": False, "error": "git_url 비어 있을 수 없음"})

                # 프론트엔드에 가속 URL을 직접 전달할 수 있습니다: 먼저 GitHub 직연결으로 복원한 다음 아래의 검증 및 플러그인 이름 추론을 수행합니다.
                git_url = _strip_gh_mirror(git_url)

                # URL 안전성 검증: https://github.com/ 또는 https://gitee.com/과 같은 신뢰할 수 있는 소스만 허용
                allowed_prefixes = ("https://github.com/", "https://gitlab.com/", "https://gitee.com/", "https://bitbucket.org/")
                if not any(git_url.startswith(p) for p in allowed_prefixes):
                    return web.json_response({"success": False, "error": f"지원되지 않는 git 저장소 주소, 허용: {', '.join(allowed_prefixes)}"})

                # custom_nodes 디렉터리 가져오기
                custom_nodes_dir = folder_paths.get_folder_paths("custom_nodes")
                if not custom_nodes_dir:
                    custom_nodes_dir = os.path.join(os.path.dirname(folder_paths.__file__), "custom_nodes")
                else:
                    custom_nodes_dir = custom_nodes_dir[0]

                if not os.path.isdir(custom_nodes_dir):
                    return web.json_response({"success": False, "error": "custom_nodes 폴더가 없습니다"})

                # 플러그인 이름 추론: git URL 마지막 부분에서 추출
                if not plugin_name:
                    url_path = git_url.rstrip("/")
                    if url_path.endswith(".git"):
                        url_path = url_path[:-4]
                    plugin_name = os.path.basename(url_path)

                # 보안 검사: 플러그인 이름에 경로 구분자를 포함할 수 없음
                if "/" in plugin_name or "\\" in plugin_name or ".." in plugin_name:
                    return web.json_response({"success": False, "error": "플러그인 이름에 불법 문자 포함"})

                target_dir = os.path.join(custom_nodes_dir, plugin_name)

                # 이미 존재하는지 확인
                if os.path.isdir(target_dir):
                    if force:
                        # 강제 재설치: 디렉토리 유효성 여부와 관계없이 먼저 정리 후 계속 설치
                        _remove_broken_plugin_dir(target_dir)
                        if os.path.isdir(target_dir):
                            try:
                                import shutil
                                shutil.rmtree(target_dir, ignore_errors=True)
                            except Exception:
                                pass
                        if os.path.isdir(target_dir):
                            return web.json_response({"success": False, "error": f"플러그인 디렉토리가 존재하고 정리할 수 없습니다: {plugin_name}，수동으로 삭제 후 다시 시도해주세요"})
                    elif not _is_valid_plugin_dir(target_dir):
                        # 디렉터리가 존재하지만 유효하지 않음 (git clone 실패로 인한 빈 디렉터리 또는 불완전한 디렉터리): 자동 정리 후 설치를 계속합니다.
                        removed = _remove_broken_plugin_dir(target_dir)
                        if removed:
                            print(f"[ComfyUI Panel] 손상된 플러그인 디렉터리 {plugin_name}，자동 정리 및 재설치")
                        else:
                            return web.json_response({"success": False, "error": f"플러그인 디렉토리가 존재하고 정리할 수 없습니다: {plugin_name}，수동으로 삭제 후 다시 시도해주세요"})
                    else:
                        return web.json_response({"success": False, "error": f"플러그인 이미 존재: {plugin_name}", "already_exists": True})

                # 실행 git clone（mirror='proxy' 시간 흐름 속도 가속, 플러그인 이름은 여전히 직접 링크로 추론하며, 디렉토리 이름에 영향을 주지 않음）
                clone_url = _apply_gh_mirror(git_url, mirror)
                cmd = ["git", "clone", "--depth", "1", clone_url, target_dir]
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await process.communicate()

                if process.returncode != 0:
                    error_msg = stderr.decode("utf-8", errors="replace").strip()
                    # 가속원 실패 시, 원본 가속원으로 돌아가도록 안내 (드롭다운은 '플러그인 의존성' 패널 상단)）
                    mirror_hint = "" if clone_url == git_url else "（현재는 가속원입니다. '플러그인 의존성' 패널 상단으로 되돌려 다시 시도할 수 있습니다）"
                    # 잠재적으로 불완전한 디렉터리를 정리합니다: clone 실패 시 git가 비어 있지 않은 .git 잔해를 썼을 수 있습니다.，
                    # 직접 rmtree 후에도 삭제 확인 및 잔류 여부 확인"가짜 유효"시스템 리소스 모니터링 구성 요소 - panel-config 내에 위치
                    if os.path.isdir(target_dir):
                        import shutil
                        shutil.rmtree(target_dir, ignore_errors=True)
                        # 최저점: 여전히 (권한 등) 잔류하고 유효하지 않다고 판단되면 강제로 다시 정리
                        if os.path.isdir(target_dir) and not _is_valid_plugin_dir(target_dir):
                            try:
                                shutil.rmtree(target_dir, ignore_errors=True)
                            except Exception:
                                pass
                    if os.path.isdir(target_dir):
                        return web.json_response({"success": False, "error": f"git clone 실패 및 잔여 디렉토리 삭제 불가: {plugin_name}，수동으로 삭제 후 다시 시도해주세요: {error_msg}{mirror_hint}"})
                    return web.json_response({"success": False, "error": f"git clone 실패: {error_msg}{mirror_hint}"})

                return web.json_response({"success": True, "plugin_name": plugin_name, "path": target_dir})

            except Exception as e:
                import traceback
                traceback.print_exc()
                return web.json_response({"success": False, "error": str(e)})

        @server.routes.post("/comfyui_panel/install_plugin_zip")
        async def install_plugin_zip(request):
            """URL에서 zip 압축 파일을 다운로드하고 custom_nodes 디렉터리에 압축을 풉니다"""
            try:
                import subprocess
                import asyncio
                import zipfile
                import tempfile
                import shutil
                import urllib.request

                data = await request.json()
                zip_url = data.get("zip_url", "").strip()
                force = bool(data.get("force", False))
                mirror = data.get("mirror", "proxy")

                if not zip_url:
                    return web.json_response({"success": False, "error": "zip_url 비어 있을 수 없음"})

                # 허용: 프론트에서 직접 가속 URL 전달: 먼저 직접 링크로 되돌린 후 검증 및 파일명 추론 수행
                zip_url = _strip_gh_mirror(zip_url)

                # URL 안전성 검증
                allowed_prefixes = ("https://",)
                if not any(zip_url.startswith(p) for p in allowed_prefixes):
                    return web.json_response({"success": False, "error": "https://로 시작하는 다운로드 주소만 허용."})

                # custom_nodes 디렉터리 가져오기
                custom_nodes_dir = folder_paths.get_folder_paths("custom_nodes")
                if not custom_nodes_dir:
                    custom_nodes_dir = os.path.join(os.path.dirname(folder_paths.__file__), "custom_nodes")
                else:
                    custom_nodes_dir = custom_nodes_dir[0]

                if not os.path.isdir(custom_nodes_dir):
                    return web.json_response({"success": False, "error": "custom_nodes 폴더가 없습니다"})

                # URL에서 압축 파일 이름 추출 (확장자 제거), 롤백 플러그인 이름으로 사용
                url_path = zip_url.rstrip("/")
                url_filename = os.path.basename(url_path)
                for ext in [".zip", ".tar.gz", ".tgz", ".tar.bz2"]:
                    if url_filename.endswith(ext):
                        url_filename = url_filename[:-len(ext)]
                        break
                fallback_name = url_filename or "unknown_plugin"

                # zip 파일을 임시 디렉토리로 다운로드 (500MB 제한)）
                MAX_ZIP_SIZE = 500 * 1024 * 1024
                with tempfile.TemporaryDirectory() as tmp_dir:
                    zip_path = os.path.join(tmp_dir, "download.zip")

                    # urllib를 사용하여 다운로드 (재지정 지원)）；mirror='proxy' 시 github 직링크이동속도 증가
                    dl_url = _apply_gh_mirror(zip_url, mirror)
                    try:
                        urllib.request.urlretrieve(dl_url, zip_path)
                    except Exception as dl_err:
                        hint = "" if dl_url == zip_url else "（현재는 가속원입니다. '플러그인 의존성' 패널 상단으로 되돌려 다시 시도할 수 있습니다）"
                        return web.json_response({"success": False, "error": f"다운로드 실패: {str(dl_err)}{hint}"})

                    # 파일 크기 확인
                    zip_size = os.path.getsize(zip_path)
                    if zip_size > MAX_ZIP_SIZE:
                        return web.json_response({"success": False, "error": f"압축 파일 너무 큼 ({zip_size / 1024 / 1024:.1f}MB)，제한 초과 ({MAX_ZIP_SIZE // 1024 // 1024}MB)"})

                    # zip 파일인지 검증
                    if not zipfile.is_zipfile(zip_path):
                        return web.json_response({"success": False, "error": "다운로드된 파일이 유효한 zip 압축 파일이 아님"})

                    # 임시 디렉토리로 압축 해제
                    extract_dir = os.path.join(tmp_dir, "extracted")
                    os.makedirs(extract_dir, exist_ok=True)

                    with zipfile.ZipFile(zip_path, 'r') as zf:
                        # 안전 검사: ZIP 슬립 공격 방지
                        total_extracted = 0
                        for member in zf.namelist():
                            member_path = os.path.join(extract_dir, member)
                            if not os.path.commonpath([extract_dir, os.path.abspath(member_path)]) == extract_dir:
                                return web.json_response({"success": False, "error": "압축 파일에 안전하지 않은 경로가 포함되어 있습니다 (zip slip 검출 실패).）"})
                            total_extracted += zf.getinfo(member).file_size
                        if total_extracted > MAX_ZIP_SIZE * 3:
                            return web.json_response({"success": False, "error": f"압축 후 총 크기가 너무 큼 ({total_extracted / 1024 / 1024:.1f}MB)"})
                        zf.extractall(extract_dir)

                    # 디렉토리 구조를 분석하여 플러그인 루트 디렉토리 및 플러그인 이름을 결정
                    # 규칙：
                    #   - 압축 파일 내에 하위 폴더가 포함되어 있습니다. → 해당 하위 폴더 이름으로 플러그인 디렉토리 이름 지정
                    #   - 압축 파일 루트 디렉토리가 플러그인입니다 (자식 폴더로 묶이지 않음)） → 압축 파일 이름으로 플러그인 디렉토리 이름 지정
                    plugin_source = None
                    plugin_name = None

                    root_items = os.listdir(extract_dir)
                    root_has_init = "__init__.py" in root_items
                    root_has_py = any(f.endswith(".py") for f in root_items if os.path.isfile(os.path.join(extract_dir, f)))
                    root_dirs = [d for d in root_items if os.path.isdir(os.path.join(extract_dir, d))]

                    if root_has_init or root_has_py:
                        # 경우 1: 루트 디렉토리가 플러그인 디렉토리 (압축 파일 루트 디렉토리에 플러그인 파일 포함)）
                        # → 압축 파일 이름으로 플러그인 디렉토리 이름 지정
                        plugin_source = extract_dir
                        plugin_name = fallback_name
                    elif len(root_dirs) == 1:
                        # 상황 2: 하나의 자식 디렉토리만 존재하면 플러그인 디렉토리
                        # → 이 하위 디렉터리 이름을 플러그인 디렉터리 이름으로 사용
                        plugin_source = os.path.join(extract_dir, root_dirs[0])
                        plugin_name = root_dirs[0]
                    else:
                        # 상황 3: 여러 개의 하위 디렉터리에서 __init__.py를 포함하는 것을 찾습니다.
                        for d in root_dirs:
                            sub_path = os.path.join(extract_dir, d)
                            if os.path.isfile(os.path.join(sub_path, "__init__.py")):
                                plugin_source = sub_path
                                plugin_name = d
                                break

                    if not plugin_source:
                        return web.json_response({"success": False, "error": "플러그인의 루트 디렉토리 구조를 확인할 수 없습니다. 수동으로 설치해주세요."})

                    # 보안 검사
                    if "/" in plugin_name or "\\" in plugin_name or ".." in plugin_name:
                        return web.json_response({"success": False, "error": f"플러그인 이름에 불법 문자 포함: {plugin_name}"})

                    target_dir = os.path.join(custom_nodes_dir, plugin_name)

                    # 이미 존재하는지 확인
                    if os.path.isdir(target_dir):
                        if force:
                            # 강제 재설치: 디렉토리 유효성 여부와 관계없이 먼저 정리 후 계속 설치
                            _remove_broken_plugin_dir(target_dir)
                            if os.path.isdir(target_dir):
                                try:
                                    import shutil
                                    shutil.rmtree(target_dir, ignore_errors=True)
                                except Exception:
                                    pass
                            if os.path.isdir(target_dir):
                                return web.json_response({"success": False, "error": f"플러그인 디렉토리가 존재하고 정리할 수 없습니다: {plugin_name}，수동으로 삭제 후 다시 시도해주세요"})
                        elif not _is_valid_plugin_dir(target_dir):
                            # 디렉토리, 원본 이미지 자료는 HuggingFace 데이터셋에서 가져옴
                            removed = _remove_broken_plugin_dir(target_dir)
                            if removed:
                                print(f"[ComfyUI Panel] 손상된 플러그인 디렉터리 {plugin_name}，자동 정리 및 재설치")
                            else:
                                return web.json_response({"success": False, "error": f"플러그인 디렉토리가 존재하고 정리할 수 없습니다: {plugin_name}，수동으로 삭제 후 다시 시도해주세요"})
                        else:
                            return web.json_response({"success": False, "error": f"플러그인 이미 존재: {plugin_name}", "already_exists": True})

                    # 복사 대상 custom_nodes
                    shutil.copytree(plugin_source, target_dir)

                return web.json_response({"success": True, "plugin_name": plugin_name, "path": target_dir})

            except Exception as e:
                import traceback
                traceback.print_exc()
                return web.json_response({"success": False, "error": str(e)})

        @server.routes.get("/comfyui_panel/check_plugin_installed")
        async def check_plugin_installed(request):
            """지정 플러그인 설치 여부 확인 (또한 .disabled 비활성화 디렉토리 확인)）

            필드로 돌아가기：
              - installed: 플러그인 디렉터리 존재 및 유효
              - broken: 디렉토리가 존재하지만 유효하지 않음 (설치 실패 잔재)）
              - disabled: 플러그인이 비활성화 상태 (`.disabled` 하위 디렉터리 또는 `.disabled` 접미사 디렉터리 위치)）
              - matched_name: 실제 일치한 디렉토리 이름
              - path: 플러그인 디렉토리 경로
            """
            try:
                plugin_name = request.rel_url.query.get("plugin_name", "").strip()
                git_url = request.rel_url.query.get("git_url", "").strip()

                if not plugin_name and not git_url:
                    return web.json_response({"success": False, "error": "plugin_name git_url과 최소 1개 필요"})

                custom_nodes_dir = folder_paths.get_folder_paths("custom_nodes")
                if not custom_nodes_dir:
                    custom_nodes_dir = os.path.join(os.path.dirname(folder_paths.__file__), "custom_nodes")
                else:
                    custom_nodes_dir = custom_nodes_dir[0]

                disabled_dir = os.path.join(custom_nodes_dir, '.disabled')

                # 모든 후보 디렉터리 이름 수집
                candidates = set()

                # 후보 1: plugin_name을 폴더 이름으로 직접 사용 (오래된 로직 호환)）
                if plugin_name:
                    if "/" not in plugin_name and "\\" not in plugin_name and ".." not in plugin_name:
                        candidates.add(plugin_name)

                # git_url에서 디렉토리 이름 추론 (가장 신뢰할 수 있는 방법)）
                if git_url:
                    url_path = git_url.rstrip("/")
                    if url_path.endswith(".git"):
                        url_path = url_path[:-4]
                    repo_name = os.path.basename(url_path)
                    if repo_name:
                        candidates.add(repo_name)

                def _result_for_dir(plugin_dir, name, disabled):
                    """결과를 일관되게 구성합니다: 디렉토리가 존재하면 유효성 검사를 기반으로 installed 또는 broken"""
                    if not _is_valid_plugin_dir(plugin_dir):
                        return web.json_response({"success": True, "installed": False, "broken": True,
                                                  "disabled": disabled, "path": plugin_dir, "matched_name": name})
                    return web.json_response({"success": True, "installed": True, "broken": False,
                                              "disabled": disabled, "path": plugin_dir, "matched_name": name})

                # 1) 정확히 일치하는 활성화 경로：custom_nodes/<name>
                for name in candidates:
                    plugin_dir = os.path.join(custom_nodes_dir, name)
                    if os.path.isdir(plugin_dir):
                        return _result_for_dir(plugin_dir, name, False)

                # 2) 정확한 일치 비활성화 경로：custom_nodes/.disabled/<name>（Manager 스타일）
                for name in candidates:
                    disabled_subfolder = os.path.join(disabled_dir, name)
                    if os.path.isdir(disabled_subfolder):
                        return _result_for_dir(disabled_subfolder, name, True)

                # 3) 정확한 일치 비활성화 경로：custom_nodes/<name>.disabled（구식 접미사）
                for name in candidates:
                    suffix_path = os.path.join(custom_nodes_dir, name + '.disabled')
                    if os.path.isdir(suffix_path):
                        return _result_for_dir(suffix_path, name, True)

                # 정확한 일치 미명중, 대소문자 구별 없이 시도 (사용자 정의 폴더 이름 대소문자 차이 고려)）
                if candidates and os.path.isdir(custom_nodes_dir):
                    candidate_lowers = {c.lower() for c in candidates}
                    try:
                        entries = os.listdir(custom_nodes_dir)
                        # 4) 대소문자를 구분하지 않는 일치 여부 경로 활성화
                        for entry in entries:
                            if entry == '.disabled' or entry.endswith('.disabled'):
                                continue
                            if entry.lower() in candidate_lowers:
                                plugin_dir = os.path.join(custom_nodes_dir, entry)
                                if os.path.isdir(plugin_dir):
                                    return _result_for_dir(plugin_dir, entry, False)
                        # 5) 대소문자 구별 없는 매칭 비활성화, 경로(기존 확장자) 무시 <name>.disabled）
                        for entry in entries:
                            if entry == '.disabled':
                                continue
                            if entry.endswith('.disabled') and entry.lower()[:-len('.disabled')] in candidate_lowers:
                                suffix_path = os.path.join(custom_nodes_dir, entry)
                                if os.path.isdir(suffix_path):
                                    real_name = entry[:-len('.disabled')]
                                    return _result_for_dir(suffix_path, real_name, True)
                        # 6) 대소문자 구별 없는 일치, 경로 매칭 비활성화 (Manager 스타일) .disabled/<name>）
                        if os.path.isdir(disabled_dir):
                            for entry in os.listdir(disabled_dir):
                                if entry.lower() in candidate_lowers:
                                    sub = os.path.join(disabled_dir, entry)
                                    if os.path.isdir(sub):
                                        return _result_for_dir(sub, entry, True)
                    except OSError:
                        pass

                return web.json_response({"success": True, "installed": False, "broken": False, "disabled": False,
                                          "path": None, "candidates": list(candidates)})

            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        @server.routes.get("/comfyui_panel/list_custom_nodes")
        async def list_custom_nodes(request):
            """custom_nodes 디렉토리에서 설치된 플러그인을 나열합니다"""
            try:
                custom_nodes_dir = folder_paths.get_folder_paths("custom_nodes")
                if not custom_nodes_dir:
                    custom_nodes_dir = os.path.join(os.path.dirname(folder_paths.__file__), "custom_nodes")
                else:
                    custom_nodes_dir = custom_nodes_dir[0]

                plugins = []
                if os.path.isdir(custom_nodes_dir):
                    for name in sorted(os.listdir(custom_nodes_dir)):
                        plugin_path = os.path.join(custom_nodes_dir, name)
                        if os.path.isdir(plugin_path) and not name.startswith("."):
                            has_init = os.path.isfile(os.path.join(plugin_path, "__init__.py"))
                            plugins.append({"name": name, "has_init": has_init})

                return web.json_response({"success": True, "plugins": plugins})

            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        # ---------- 플러그인 디렉터리 유효성 검사 ----------
        def _is_valid_plugin_dir(plugin_dir):
            """디렉토리가 있는지 확인"유효"설치된 플러그인 폴더。

            ComfyUI 플러그인 = 폴더에 __init__.py가 존재합니다 (git clone 또는 zip 설치 시 생성)）。
            __init__.py만 인식: 존재하면 유효한 플러그인, 존재하지 않으면 (빈 디렉토리 또는 git clone 실패로 남은)
            반쪽 .git 잔해)는 항상 유효하지 않은 (손상) 것으로 간주하여 자동 정리 트리거하고 재설치 시 영향을 받지 않도록
            already_exists intercept 및 check_plugin_installed 오탐 방지。

            참고: .git 디렉터리 판단에 의존하지 않음——git clone 참조 슬롯이 가득 찼습니다. 채울 공간이 없습니다
            .git 디렉토리, 이대로 판단 플러그인 유효하지 않음。
            """
            if not plugin_dir or not os.path.isdir(plugin_dir):
                return False
            return os.path.isfile(os.path.join(plugin_dir, "__init__.py"))

        def _remove_broken_plugin_dir(plugin_dir):
            """손상/불완전 플러그인 디렉토리 삭제 (유효하지 않을 경우에만)）。
            True는 삭제되었음을 나타내고, False는 디렉토리가 유효함을 (삭제되지 않았음을) 나타내거나 삭제 실패를 나타냅니다.。"""
            if not plugin_dir or not os.path.isdir(plugin_dir):
                return False
            if _is_valid_plugin_dir(plugin_dir):
                return False
            try:
                import shutil
                shutil.rmtree(plugin_dir, ignore_errors=True)
                return not os.path.isdir(plugin_dir)
            except Exception:
                return False

        # ---------- 플러그인 Python 의존성 설치（pip） ----------
        # 디자인 포인트 (가장 큰 함정은 환경이 잘못된 경우입니다. 여기서는 근본적으로 회피)）：
        #  1) 인터프리터 일괄 사용 sys.executable —— 즉"현재 ComfyUI 실행 중인 이 Python"，
        #     다시 사용 "-m pip" 내장된 pip를 호출합니다. PATH에 있는 것을 사용하지 마세요. python/pip，
        #     그렇지 않으면 시스템 Python, 다른 venv, conda 환경이 서로 섞일 수 있습니다。
        #     （venv / conda / ComfyUI python_embeded의 세 가지 경우 모두 적용）
        #  2) 자 프로세스 stdout+stderr 병합 프롬프트에서 카메라 제어는 기본적으로 활성화되지 않았습니다"절대 줄 번호"점진적 추출，
        #     언제든지 중단하고, 이미지 소스를 변경하여 다시 로드할 수 있으며, 페이지를 다시 열지 않아도 됩니다.。
        #  3) 오래된 pip 버전은 --break-system-packages를 인식하지 못합니다. 다시 시도할 필요가 없습니다.。
        _PIP_MIRRORS = [
            {"id": "tsinghua", "name": "청화", "url": "https://pypi.tuna.tsinghua.edu.cn/simple"},
            {"id": "aliyun", "name": "阿里云", "url": "https://mirrors.aliyun.com/pypi/simple/"},
            {"id": "ustc", "name": "중과", "url": "https://pypi.mirrors.ustc.edu.cn/simple/"},
            {"id": "tencent", "name": "텐센트 클라우드", "url": "https://mirrors.cloud.tencent.com/pypi/simple"},
            {"id": "huawei", "name": "Huawei Cloud", "url": "https://repo.huaweicloud.com/repository/pypi/simple"},
            {"id": "douban", "name": "두반", "url": "https://pypi.doubanio.com/simple/"},
            {"id": "official", "name": "PyPI 공식", "url": "https://pypi.org/simple"},
        ]
        _PIP_DEFAULT_MIRROR = "tsinghua"
        _PIP_TASKS = {}          # task_id -> 작업 사전
        _PIP_TASK_SEQ = [0]
        _PIP_LOG_LIMIT = 5000

        def _pip_custom_nodes_dir():
            """custom_nodes 디렉토리 (플러그인 설치/활성화 인터페이스와 동일한 값)）"""
            try:
                d = folder_paths.get_folder_paths("custom_nodes")
            except Exception:
                d = None
            if not d:
                d = os.path.join(os.path.dirname(folder_paths.__file__), "custom_nodes")
            else:
                d = d[0]
            return d

        def _pip_find_plugin_dir(plugin_name, git_url="", path=""):
            """플러그인 디렉터리 위치, 활성화/비활성화된 상태 모두 지원(.disabled 디렉터리 또는 확장자)。

            반환 (plugin_dir, matched_name), 찾을 수 없으면 반환 (None, None)。
            """
            # 프론트엔드가 절대 경로를 가져온 경우 바로 사용합니다 (가장 빠르고 정확)）
            if path and os.path.isdir(path):
                return path, os.path.basename(os.path.normpath(path))

            cn = _pip_custom_nodes_dir()
            disabled_dir = os.path.join(cn, ".disabled")
            candidates = []
            if plugin_name and "/" not in plugin_name and "\\" not in plugin_name and ".." not in plugin_name:
                candidates.append(plugin_name)
            if git_url:
                up = git_url.rstrip("/")
                if up.endswith(".git"):
                    up = up[:-4]
                repo = os.path.basename(up)
                if repo and repo not in candidates:
                    candidates.append(repo)
            if not candidates:
                return None, None

            for name in candidates:                                   # 활성화：custom_nodes/<name>
                p = os.path.join(cn, name)
                if os.path.isdir(p):
                    return p, name
            for name in candidates:                                   # 비활성화：custom_nodes/.disabled/<name>
                p = os.path.join(disabled_dir, name)
                if os.path.isdir(p):
                    return p, name
            for name in candidates:                                   # 비활성화：custom_nodes/<name>.disabled
                p = os.path.join(cn, name + ".disabled")
                if os.path.isdir(p):
                    return p, name

            lowers = {c.lower(): c for c in candidates}               # 대소문자 구분 없는 fallback
            try:
                for entry in sorted(os.listdir(cn)):
                    if entry == ".disabled":
                        continue
                    low = entry.lower()
                    key = low[:-len(".disabled")] if low.endswith(".disabled") else low
                    if key in lowers:
                        p = os.path.join(cn, entry)
                        if os.path.isdir(p):
                            return p, entry
                if os.path.isdir(disabled_dir):
                    for entry in sorted(os.listdir(disabled_dir)):
                        if entry.lower() in lowers:
                            p = os.path.join(disabled_dir, entry)
                            if os.path.isdir(p):
                                return p, entry
            except OSError:
                pass
            return None, None

        def _pip_list_requirement_files(plugin_dir):
            """플러그인 디렉토리의 requirements*.txt（requirements.txt 항상 1위）"""
            if not plugin_dir or not os.path.isdir(plugin_dir):
                return []
            try:
                names = [n for n in os.listdir(plugin_dir)
                         if n.lower().startswith("requirements") and n.lower().endswith(".txt")
                         and os.path.isfile(os.path.join(plugin_dir, n))]
            except OSError:
                return []
            names.sort(key=lambda n: (0 if n.lower() == "requirements.txt" else 1, n.lower()))
            return names

        def _pip_scan_all_plugins():
            """custom_nodes (포함 .disabled) 내 모든 지정된 requirements*.txt 플러그인, 프런트엔드 드롭다운 전환"""
            out = []
            try:
                cn = _pip_custom_nodes_dir()
                for base_dir, base_disabled in ((cn, False), (os.path.join(cn, ".disabled"), True)):
                    if not os.path.isdir(base_dir):
                        continue
                    for entry in sorted(os.listdir(base_dir)):
                        if entry.startswith("."):
                            continue
                        p = os.path.join(base_dir, entry)
                        if not os.path.isdir(p):
                            continue
                        files = _pip_list_requirement_files(p)
                        if not files:
                            continue
                        name, disabled = entry, base_disabled
                        if name.lower().endswith(".disabled"):
                            name = name[: -len(".disabled")]
                            disabled = True
                        out.append({"name": name, "dir": p, "files": files, "disabled": disabled})
            except OSError:
                pass
            out.sort(key=lambda x: x["name"].lower())
            return out

        def _pip_read_text(path, limit=400000):
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    return f.read(limit)
            except Exception:
                return ""

        def _pip_env_info():
            """Current ComfyUI running interpreter information (the only authoritative source sys.executable）"""
            import platform
            exe = sys.executable or ""
            base_prefix = getattr(sys, "base_prefix", sys.prefix)
            parent = os.path.basename(os.path.dirname(exe)).lower() if exe else ""
            try:
                comfyui_dir = folder_paths.base_path
            except Exception:
                comfyui_dir = os.path.abspath("./")
            return {
                "exe": exe,
                "version": platform.python_version(),
                "prefix": sys.prefix,
                "base_prefix": base_prefix,
                "is_venv": sys.prefix != base_prefix,
                "is_embeded": parent in ("python_embeded", "python_embedded"),
                "comfyui_dir": comfyui_dir,
                "cwd": os.getcwd(),
            }

        async def _pip_version_text():
            """검색 'sys.executable -m pip --version'; pip 누락 시 빈 문자열"""
            exe = sys.executable or "python"
            try:
                import asyncio as _aio
                proc = await _aio.create_subprocess_exec(
                    exe, "-m", "pip", "--version",
                    stdout=_aio.subprocess.PIPE, stderr=_aio.subprocess.STDOUT)
                out, _ = await _aio.wait_for(proc.communicate(), timeout=25)
                txt = (out or b"").decode("utf-8", errors="replace").strip()
                return txt if proc.returncode == 0 else ""
            except Exception:
                return ""

        def _pip_mirror_by_id(mid):
            for m in _PIP_MIRRORS:
                if m["id"] == mid:
                    return m
            for m in _PIP_MIRRORS:
                if m["id"] == _PIP_DEFAULT_MIRROR:
                    return m
            return _PIP_MIRRORS[0]

        def _pip_host_of(url):
            try:
                from urllib.parse import urlparse
                return urlparse(url).hostname or ""
            except Exception:
                return ""

        def _pip_task_dir():
            """임시 requirements 저장 디렉토리: ComfyUI/user/default/bsk_pip(1일 전의 오래된 파일도 함께 정리).）"""
            import time as _t
            try:
                base_path = folder_paths.base_path
            except Exception:
                base_path = os.path.abspath("./")
            d = os.path.join(base_path, "user", "default", "bsk_pip")
            try:
                os.makedirs(d, exist_ok=True)
                now = _t.time()
                for n in os.listdir(d):
                    if not n.startswith("requirements_") or not n.endswith(".txt"):
                        continue
                    fp = os.path.join(d, n)
                    try:
                        if now - os.path.getmtime(fp) > 86400:
                            os.remove(fp)
                    except OSError:
                        pass
            except Exception:
                pass
            return d

        def _pip_cmd_display(cmd):
            """명령어 배열을 복사 가능한 문자열로 합치기(공백 포함 경로 따옴표)）"""
            parts = []
            for a in cmd:
                s = str(a)
                parts.append(s if (" " not in s and '"' not in s) else '"%s"' % s.replace('"', '\\"'))
            return " ".join(parts)

        def _pip_level_of(line):
            low = (line or "").lower()
            if "error" in low or "fatal" in low or "failed" in low:
                return "error"
            if "warning" in low or "deprecat" in low:
                return "warn"
            if "successfully installed" in low or "satisfied" in low:
                return "ok"
            return "info"

        def _pip_append_log(task, text, level=None):
            for s in str(text).replace("\r\n", "\n").replace("\r", "\n").split("\n"):
                task["lines"].append({"s": s, "l": level or _pip_level_of(s)})
            over = len(task["lines"]) - _PIP_LOG_LIMIT
            if over > 0:
                task["lines"] = task["lines"][over:]
                task["offset"] += over

        async def _pip_run_task(task_id, cwd, env):
            """진짜 pip 실행: 로그 실시간으로 작업 버퍼에 쓰기, 종료 후 정지 상태"""
            import asyncio as _aio
            task = _PIP_TASKS.get(task_id)
            if not task:
                return
            cmd = task["cmd"]
            exe = cmd[0]
            # python_embeded는 pip가 필요하지 않게 설정되어 있습니다. 먼저 ensurepip를 자동 실행하여 사용자에게 pip를 보여주지 않습니다."pip 모듈 없음"완전히 멍하다
            if not await _pip_version_text():
                _pip_append_log(task, "pip를 감지하지 못했습니다, 먼저 실행: %s -m ensurepip" % exe, "warn")
                try:
                    p = await _aio.create_subprocess_exec(
                        exe, "-m", "ensurepip", "--default-pip",
                        stdout=_aio.subprocess.PIPE, stderr=_aio.subprocess.STDOUT,
                        cwd=cwd, env=env)
                    out, _ = await p.communicate()
                    _pip_append_log(task, (out or b"").decode("utf-8", errors="replace"))
                except Exception as e:
                    _pip_append_log(task, "ensurepip 실행 실패: %s" % e, "error")

            kwargs = {"start_new_session": True} if os.name != "nt" else {}
            try:
                proc = await _aio.create_subprocess_exec(
                    *cmd, cwd=cwd, env=env,
                    stdout=_aio.subprocess.PIPE, stderr=_aio.subprocess.STDOUT, **kwargs)
            except Exception as e:
                task["status"] = "error"
                task["returncode"] = -1
                _pip_append_log(task, "pip 시작 실패: %s" % e, "error")
                return

            task["proc"] = proc
            try:
                while True:
                    raw = await proc.stdout.readline()
                    if not raw:
                        break
                    _pip_append_log(task, raw.decode("utf-8", errors="replace").rstrip("\r\n"))
            except Exception as e:
                _pip_append_log(task, "설치 로그를 읽는 데 실패했습니다: %s" % e, "error")
            try:
                await proc.wait()
            except Exception:
                pass

            rc = proc.returncode
            task["returncode"] = rc
            if task.get("cancelled"):
                task["status"] = "cancelled"
            elif rc == 0:
                task["status"] = "done"
            else:
                task["status"] = "error"
            _pip_append_log(task, "---- 설치 완료 (종료 코드) %s）----" % rc, "ok" if rc == 0 else "error")
            if rc == 0:
                _pip_append_log(task, "종속성은 설치되어 있어야 하며, ComfyUI를 다시 시작한 후에 노드가 새 종속성을 로드。", "warn")

        @server.routes.get("/comfyui_panel/plugin_requirements")
        async def plugin_requirements(request):
            """읽기 플러그인 종속성 목록 정보。

            query: plugin / git_url / path / file
            plugin 및 path가 없을 때만 반환 all=True 의"설치 가능한 의존성 플러그인"목록 (프론트엔드 드롭다운 전환)）。
            """
            try:
                plugin = request.rel_url.query.get("plugin", "").strip()
                git_url = request.rel_url.query.get("git_url", "").strip()
                path = request.rel_url.query.get("path", "").strip()
                file_name = request.rel_url.query.get("file", "").strip()

                base = {
                    "success": True,
                    "mirrors": _PIP_MIRRORS,
                    "default_mirror": _PIP_DEFAULT_MIRROR,
                    "python": _pip_env_info(),
                    "pip_version": await _pip_version_text(),
                    "plugins": _pip_scan_all_plugins(),
                }
                if not plugin and not path:
                    return web.json_response(dict(base, all=True, found=False))

                plugin_dir, matched = _pip_find_plugin_dir(plugin, git_url, path)
                if not plugin_dir:
                    return web.json_response(dict(base, all=False, found=False,
                                                  error="플러그인 디렉터리를 찾을 수 없습니다. 먼저 플러그인을 설치하십시오"))
                files = _pip_list_requirement_files(plugin_dir)
                if file_name and file_name not in files:
                    file_name = ""
                if not file_name and files:
                    file_name = files[0]
                content = _pip_read_text(os.path.join(plugin_dir, file_name)) if file_name else ""

                running = None
                for tid, t in _PIP_TASKS.items():
                    if t.get("status") == "running" and (t.get("dir") == plugin_dir
                                                         or (matched and t.get("plugin") == matched)):
                        running = {"task_id": tid, "mirror": t.get("mirror", "")}
                        break

                return web.json_response(dict(base, all=False, found=True,
                                              plugin=matched or plugin, matched_name=matched or "",
                                              dir=plugin_dir, files=files, file=file_name,
                                              content=content, running=running))
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        @server.routes.post("/comfyui_panel/plugin_pip_install")
        async def plugin_pip_install(request):
            """플러그인 디렉토리로의 requirements.txt 설치 Python 의존성 (변경 가능한 이미지 소스 / 수정된 목록 사용)）"""
            try:
                import asyncio as _aio
                import time as _t

                data = await request.json()
                plugin = (data.get("plugin") or "").strip()
                git_url = (data.get("git_url") or "").strip()
                path = (data.get("path") or "").strip()
                file_name = (data.get("file") or "").strip()
                mirror_id = (data.get("mirror") or _PIP_DEFAULT_MIRROR).strip()
                upgrade = bool(data.get("upgrade"))
                extra_args = (data.get("extra_args") or "").strip()
                content = data.get("content")

                plugin_dir, matched = _pip_find_plugin_dir(plugin, git_url, path)
                if not plugin_dir:
                    return web.json_response({"success": False, "error": "플러그인 디렉터리를 찾을 수 없습니다. 먼저 플러그인을 설치하십시오"})

                # 이미 실행 중인 작업이 동일 플러그인 디렉토리에 있습니다 → Direct reuse (frontend continues to pull logs), avoid repeated installations fighting each other
                for tid, t in _PIP_TASKS.items():
                    if t.get("status") == "running" and t.get("dir") == plugin_dir:
                        return web.json_response({"success": True, "task_id": tid, "reused": True,
                                                  "cmd": _pip_cmd_display(t["cmd"])})

                if file_name and ("/" in file_name or "\\" in file_name or ".." in file_name):
                    return web.json_response({"success": False, "error": "유효하지 않은 의존성 파일 이름"})

                req_path = None
                if isinstance(content, str):
                    if not content.strip():
                        return web.json_response({"success": False, "error": "의존 목록이 비어 있어 설치할 수 없음"})
                    # 화면 내용과 디스크 불일치 → 임시 파일에 쓰기, 플러그인 디렉터리 오염 방지 (cwd는 여전히 플러그인 디렉터리, 상대 경로도 동일하게 적용)）
                    disk_target = os.path.join(plugin_dir, file_name or "requirements.txt")
                    disk_txt = _pip_read_text(disk_target) if os.path.isfile(disk_target) else None
                    if disk_txt is None or content.replace("\r\n", "\n") != disk_txt.replace("\r\n", "\n"):
                        req_path = os.path.join(_pip_task_dir(),
                                                "requirements_%d.txt" % int(_t.time() * 1000))
                        try:
                            with open(req_path, "w", encoding="utf-8", newline="\n") as f:
                                f.write(content)
                        except Exception as e:
                            return web.json_response({"success": False, "error": "임시 의존성 목록 작성 실패: %s" % e})

                if not req_path:
                    req_path = os.path.join(plugin_dir, file_name or "requirements.txt")
                    if not os.path.isfile(req_path):
                        alt = _pip_list_requirement_files(plugin_dir)
                        if not alt:
                            return web.json_response({"success": False,
                                                      "error": "플러그인 디렉토리에 requirements.txt가 없으므로 의존성을 설치할 필요가 없습니다."})
                        req_path = os.path.join(plugin_dir, alt[0])

                mirror = _pip_mirror_by_id(mirror_id)
                exe = sys.executable or "python"
                cmd = [exe, "-m", "pip", "install", "-r", req_path, "-i", mirror["url"]]
                host = _pip_host_of(mirror["url"])
                if host:
                    cmd += ["--trusted-host", host]
                if upgrade:
                    cmd += ["--upgrade"]
                cmd += ["--disable-pip-version-check"]
                if extra_args:
                    try:
                        import shlex
                        cmd += shlex.split(extra_args)
                    except Exception:
                        pass

                # 역사 작업 재활용: 1시간이 넘는 작업은 모두 삭제하고, 최근 10개의 완료된 작업만 유지합니다.。
                # 청소하지 않으면 각 작업 최대 5000줄 로그가 남고, 메모리가 많이 소모됨。
                try:
                    now = _t.time()
                    finished = [(k, v.get("started", now)) for k, v in _PIP_TASKS.items()
                                if v.get("status") != "running"]
                    for k, st in finished:
                        if now - st > 3600:
                            _PIP_TASKS.pop(k, None)
                    finished = sorted([(k, v.get("started", 0)) for k, v in _PIP_TASKS.items()
                                       if v.get("status") != "running"], key=lambda x: x[1])
                    for k, _st in finished[:-10]:
                        _PIP_TASKS.pop(k, None)
                except Exception:
                    pass

                _PIP_TASK_SEQ[0] += 1
                task_id = "pip_%d_%d" % (int(_t.time()), _PIP_TASK_SEQ[0])
                env = os.environ.copy()
                env["PYTHONIOENCODING"] = "utf-8"     # 중국 Windows에서 pip 출력은 GBK이며, UTF-8로 강제하여 깨짐을 방지합니다.
                env["PYTHONUTF8"] = "1"
                env["PYTHONDONTWRITEBYTECODE"] = "1"
                env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"

                task = {"id": task_id, "plugin": matched or plugin, "dir": plugin_dir,
                        "req": req_path, "cmd": cmd, "lines": [], "offset": 0,
                        "status": "running", "returncode": None, "proc": None,
                        "cancelled": False, "mirror": mirror["id"], "started": _t.time()}
                _PIP_TASKS[task_id] = task

                info = _pip_env_info()
                _pip_append_log(task, "$ " + _pip_cmd_display(cmd), "cmd")
                _pip_append_log(task, "인터프리터: %s" % info["exe"])
                _pip_append_log(task, "Python : %s%s%s" % (info["version"],
                                                            "（가상 환경）" if info["is_venv"] else "",
                                                            "（휴대용）" if info["is_embeded"] else ""))
                _pip_append_log(task, "미러 소스: %s (%s)" % (mirror["name"], mirror["url"]))
                _pip_append_log(task, "작업 디렉토리: %s" % plugin_dir)
                _pip_append_log(task, "의존성 파일: %s" % req_path)

                _aio.create_task(_pip_run_task(task_id, plugin_dir, env))
                return web.json_response({"success": True, "task_id": task_id,
                                          "cmd": _pip_cmd_display(cmd), "plugin": matched or plugin})
            except Exception as e:
                import traceback
                traceback.print_exc()
                return web.json_response({"success": False, "error": str(e)})

        @server.routes.get("/comfyui_panel/plugin_pip_log")
        async def plugin_pip_log(request):
            """증분 다운로드 설치 로그：from = 프론트엔드에서 소비된 절대 행 번호"""
            try:
                tid = (request.rel_url.query.get("task_id") or "").strip()
                task = _PIP_TASKS.get(tid)
                if not task:
                    return web.json_response({"success": False, "error": "작업이 존재하지 않거나 삭제됨"})
                try:
                    frm = int(request.rel_url.query.get("from", "0"))
                except Exception:
                    frm = 0
                start = max(0, frm - task["offset"])
                chunk = task["lines"][start:] if start < len(task["lines"]) else []
                return web.json_response({
                    "success": True,
                    "status": task["status"],
                    "returncode": task["returncode"],
                    "lines": chunk,
                    "next": task["offset"] + len(task["lines"]),
                    "plugin": task.get("plugin", ""),
                    "mirror": task.get("mirror", ""),
                })
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        @server.routes.post("/comfyui_panel/plugin_pip_cancel")
        async def plugin_pip_cancel(request):
            """실행 중인 의존성 설치를 중단합니다(종속 프로세스 트리를 죽이고, pip 다운로드 중단 시 다른 저장소로 변경하여 다시 설치할 수 있습니다).）"""
            try:
                data = await request.json()
                tid = (data.get("task_id") or "").strip()
                task = _PIP_TASKS.get(tid)
                if not task:
                    return web.json_response({"success": False, "error": "작업이 존재하지 않음"})
                if task["status"] != "running":
                    return web.json_response({"success": True, "status": task["status"], "noop": True})

                task["cancelled"] = True
                _pip_append_log(task, "중단 요청을 수신했습니다. pip 프로세스를 종료 중입니다....", "warn")
                proc = task.get("proc")
                if proc is not None:
                    try:
                        if os.name == "nt":
                            import subprocess as _sp
                            _sp.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                                    stdout=_sp.DEVNULL, stderr=_sp.DEVNULL, timeout=15)
                        else:
                            import signal as _sig
                            try:
                                os.killpg(os.getpgid(proc.pid), _sig.SIGKILL)
                            except Exception:
                                proc.kill()
                    except Exception as e:
                        _pip_append_log(task, "프로세스 종료 실패: %s" % e, "error")
                return web.json_response({"success": True})
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        # ---------- 지우기 (E) - 원상복구 ----------
        def _get_plugin_desc_config_path():
            """플러그인 기능 소개 구성 파일 경로: ComfyUI/user/default/bsk_plugin_descriptions.json"""
            try:
                base_path = folder_paths.base_path
            except Exception:
                base_path = os.path.abspath('./')
            cfg_dir = os.path.join(base_path, "user", "default")
            os.makedirs(cfg_dir, exist_ok=True)
            return os.path.join(cfg_dir, "bsk_plugin_descriptions.json")

        def _read_plugin_descriptions():
            """읽기 플러그인 설명 구성 파일, 반환 {plugin_name: description} dict"""
            path = _get_plugin_desc_config_path()
            if not os.path.isfile(path):
                return {}
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    return {str(k): str(v) for k, v in data.items()}
            except Exception:
                pass
            return {}

        def _write_plugin_descriptions(desc_map):
            """플러그인 소개 파일 템플릿 작성"""
            path = _get_plugin_desc_config_path()
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(desc_map, f, indent=2, ensure_ascii=False)

        def _extract_pyproject_info(plugin_path):
            """플러그인 디렉토리의 pyproject.toml에서 기능 정보를 추출합니다
            반환 {"description": "", "display_name": "", "repository": ""}（파일이 없거나 파싱 실패 시 빈 값으로 반환）
            """
            result = {"description": "", "display_name": "", "repository": ""}
            toml_path = os.path.join(plugin_path, "pyproject.toml")
            if not os.path.isfile(toml_path):
                return result
            try:
                # 우선으로 사용 tomllib (Python 3.11+)
                import tomllib
                with open(toml_path, 'rb') as f:
                    data = tomllib.load(f)
                proj = data.get("project", {}) if isinstance(data, dict) else {}
                if isinstance(proj, dict):
                    result["description"] = str(proj.get("description", "") or "")
                    urls = proj.get("urls", {})
                    if isinstance(urls, dict):
                        # 일반 키: Repository / repo / Homepage
                        result["repository"] = str(urls.get("Repository", "") or urls.get("repository", "") or urls.get("Homepage", "") or "")
                tool_comfy = data.get("tool", {}).get("comfy", {}) if isinstance(data, dict) else {}
                if isinstance(tool_comfy, dict):
                    result["display_name"] = str(tool_comfy.get("DisplayName", "") or "")
                return result
            except Exception:
                pass
            # 롤백: 간단한 정규식 파싱 (tomllib 또는 파싱 실패 시 호환)）
            try:
                with open(toml_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                import re
                # [프로젝트] 구간에서 추출 description
                m = re.search(r'description\s*=\s*"((?:[^"\\]|\\.)*)"', content)
                if m:
                    result["description"] = m.group(1).encode().decode('unicode_escape', errors='replace')
                m2 = re.search(r'DisplayName\s*=\s*"((?:[^"\\]|\\.)*)"', content)
                if m2:
                    result["display_name"] = m2.group(1).encode().decode('unicode_escape', errors='replace')
                m3 = re.search(r'Repository\s*=\s*"((?:[^"\\]|\\.)*)"', content)
                if m3:
                    result["repository"] = m3.group(1).encode().decode('unicode_escape', errors='replace')
            except Exception:
                pass
            return result

        @server.routes.get("/comfyui_panel/list_all_plugins")
        async def list_all_plugins(request):
            """모든 custom_nodes 플러그인과 활성화/비활성화 상태를 나열합니다. + 기능 소개

            비활성화 메커니즘 (ComfyUI-Manager와 호환)）：
            1. 서브 디렉토리 스타일：custom_nodes/.disabled/<plugin>   （Manager 기본）
            2. 확장자 스타일：  custom_nodes/<plugin>.disabled      （오래된 방식）
            ComfyUI 로드 시 .disabled로 끝나는 항목을 건너뛰므로 두 가지 방법 모두 유효합니다.。

            기능 소개 출처 우선순위：
            1. 사용자 설정 파일 bsk_plugin_descriptions.json (우선)）
            2. 플러그인 디렉토리의 pyproject.toml [project].description
            """
            try:
                custom_nodes_dir = folder_paths.get_folder_paths("custom_nodes")
                if not custom_nodes_dir:
                    custom_nodes_dir = os.path.join(os.path.dirname(folder_paths.__file__), "custom_nodes")
                else:
                    custom_nodes_dir = custom_nodes_dir[0]

                # 사용자 설정 기능 소개 (우선순위 최고)）
                user_desc_map = _read_plugin_descriptions()

                plugins = []

                def _build_plugin_entry(real_name, full_path, enabled, disabled_via):
                    """개별 플러그인 항목 구성, 기능 설명 포함"""
                    entry = {
                        "name": real_name,
                        "enabled": enabled,
                        "has_init": os.path.isfile(os.path.join(full_path, '__init__.py')),
                        "disabled_via": disabled_via,
                        # 손상된 표시: 디렉토리가 존재하지만 __init__.py가 없거나 유효한 .git이 없음 (git clone 실패 잔류)）
                        "broken": not _is_valid_plugin_dir(full_path),
                    }
                    description = ""
                    display_name = ""
                    repository = ""
                    desc_source = ""
                    # 우선순위 1: 사용자 설정 파일
                    if real_name in user_desc_map:
                        description = user_desc_map[real_name]
                        desc_source = "config"
                    # 우선순위2: pyproject.toml
                    if not description:
                        info = _extract_pyproject_info(full_path)
                        if info["description"]:
                            description = info["description"]
                            desc_source = "pyproject"
                        display_name = info["display_name"]
                        repository = info["repository"]
                    entry["description"] = description
                    entry["description_source"] = desc_source
                    entry["display_name"] = display_name
                    entry["repository"] = repository
                    return entry

                if os.path.isdir(custom_nodes_dir):
                    # 1) custom_nodes 아래에 있는 플러그인 (활성화 또는 이전식 .disabled 확장자)）
                    for name in sorted(os.listdir(custom_nodes_dir)):
                        if name in ('.disabled', '__pycache__'):
                            continue
                        if name.startswith('.'):
                            continue
                        full = os.path.join(custom_nodes_dir, name)
                        if not os.path.isdir(full):
                            continue
                        # 구식 비활성화：<plugin>.disabled
                        if name.endswith('.disabled'):
                            real_name = name[:-len('.disabled')]
                            plugins.append(_build_plugin_entry(real_name, full, False, "suffix"))
                            continue
                        plugins.append(_build_plugin_entry(name, full, True, ""))

                    # 2) .disabled 자식 디렉토리에서 사용 중인 플러그인 (Manager 스타일)）
                    disabled_dir = os.path.join(custom_nodes_dir, '.disabled')
                    if os.path.isdir(disabled_dir):
                        for name in sorted(os.listdir(disabled_dir)):
                            if name == '__pycache__' or name.startswith('.'):
                                continue
                            full = os.path.join(disabled_dir, name)
                            if not os.path.isdir(full):
                                continue
                            plugins.append(_build_plugin_entry(name, full, False, "subfolder"))

                return web.json_response({"success": True, "plugins": plugins})

            except Exception as e:
                import traceback
                traceback.print_exc()
                return web.json_response({"success": False, "error": str(e)})

        @server.routes.post("/comfyui_panel/save_plugin_description")
        async def save_plugin_description(request):
            """플러그인 기능 설명을 사용자 구성 파일에 저장
            요청 본문: {"name": "<plugin>", "description": "<설명 텍스트>"}
            """
            try:
                data = await request.json()
                name = (data.get("name") or "").strip()
                description = data.get("description", "")
                if not name:
                    return web.json_response({"success": False, "error": "name 비어 있을 수 없음"})
                if "/" in name or "\\" in name or ".." in name:
                    return web.json_response({"success": False, "error": "유효하지 않은 플러그인 이름"})

                desc_map = _read_plugin_descriptions()
                if isinstance(description, str) and description.strip():
                    desc_map[name] = description.strip()
                else:
                    # 공개 소개가 있으면 해당 항목 삭제 (pyproject 기본값으로 복원）
                    desc_map.pop(name, None)
                _write_plugin_descriptions(desc_map)
                return web.json_response({"success": True, "name": name, "description": desc_map.get(name, "")})
            except Exception as e:
                import traceback
                traceback.print_exc()
                return web.json_response({"success": False, "error": str(e)})

        @server.routes.post("/comfyui_panel/import_plugin_descriptions")
        async def import_plugin_descriptions(request):
            """플러그인 기능 소개 (배치 임포트/병합)
            요청 본문: {"descriptions": {"<plugin_name>": "<소개>", ...}}
            병합 로직: 가져온 항목의 설명이 비어 있지 않을 때만 현재 설정을 업데이트/새로 만듭니다. 기존에 존재하고 가져온 항목이 비어 있으면 원래 값을 유지합니다.。
            반환 업데이트 수량 및 최종 설정。
            """
            try:
                data = await request.json()
                incoming = data.get("descriptions", {})
                if not isinstance(incoming, dict):
                    return web.json_response({"success": False, "error": "descriptions 객체여야 함"})

                desc_map = _read_plugin_descriptions()
                updated = 0
                for name, desc in incoming.items():
                    name = str(name).strip()
                    if not name or "/" in name or "\\" in name or ".." in name:
                        continue
                    desc = str(desc).strip() if desc else ""
                    if not desc:
                        # 빈 소개 건너뛰기, 기존 덮어쓰지
                        continue
                    if desc_map.get(name, "") != desc:
                        desc_map[name] = desc
                        updated += 1
                _write_plugin_descriptions(desc_map)
                return web.json_response({"success": True, "updated": updated, "total": len(desc_map)})
            except Exception as e:
                import traceback
                traceback.print_exc()
                return web.json_response({"success": False, "error": str(e)})

        @server.routes.post("/comfyui_panel/clear_plugin_descriptions")
        async def clear_plugin_descriptions(request):
            """모든 플러그인 기능 설명의 사용자 지정 설정을 지우고 기본(pyproject) 내용으로 복원"""
            try:
                desc_map = _read_plugin_descriptions()
                count = len(desc_map)
                _write_plugin_descriptions({})
                return web.json_response({"success": True, "cleared": count})
            except Exception as e:
                import traceback
                traceback.print_exc()
                return web.json_response({"success": False, "error": str(e)})

        @server.routes.post("/comfyui_panel/set_plugin_enabled")
        async def set_plugin_enabled(request):
            """특정 플러그인 활성화 / 비활성화 (폴더 이동을 통해) custom_nodes/.disabled/）

            요청 본문: {"name": "<plugin>", "enable": true/false}
            """
            try:
                import shutil

                data = await request.json()
                name = (data.get("name") or "").strip()
                enable = bool(data.get("enable", False))

                if not name:
                    return web.json_response({"success": False, "error": "name 비어 있을 수 없음"})
                # 보안: 경로 구분자/특수 디렉토리 금지
                if "/" in name or "\\" in name or ".." in name or name in ('.disabled', '__pycache__'):
                    return web.json_response({"success": False, "error": "유효하지 않은 플러그인 이름"})

                # 보호: 핵심 플러그인 비활성화 금지 (자체) / ComfyUI-Manager）
                protected = ('comfyui-bsk_ui', 'comfyui-manager')
                if not enable and name.lower() in protected:
                    return web.json_response({"success": False, "error": f"핵심 플러그인 비활성화 금지: {name}"})

                custom_nodes_dir = folder_paths.get_folder_paths("custom_nodes")
                if not custom_nodes_dir:
                    custom_nodes_dir = os.path.join(os.path.dirname(folder_paths.__file__), "custom_nodes")
                else:
                    custom_nodes_dir = custom_nodes_dir[0]

                if not os.path.isdir(custom_nodes_dir):
                    return web.json_response({"success": False, "error": "custom_nodes 폴더가 없습니다"})

                disabled_dir = os.path.join(custom_nodes_dir, '.disabled')
                enabled_path = os.path.join(custom_nodes_dir, name)
                enabled_path_suffix = enabled_path + '.disabled'  # 구형 비활성화 경로
                disabled_subfolder_path = os.path.join(disabled_dir, name)

                if enable:
                    # 활성화: .disabled 하위 디렉터리에서 또는 <name>.disabled 이동 <name>
                    src = None
                    if os.path.isdir(disabled_subfolder_path):
                        src = disabled_subfolder_path
                    elif os.path.isdir(enabled_path_suffix):
                        src = enabled_path_suffix
                    if src is None:
                        # 이미 활성화되었을 수 있음
                        if os.path.isdir(enabled_path):
                            return web.json_response({"success": True, "enabled": True, "name": name,
                                                      "already_enabled": True, "path": enabled_path})
                        return web.json_response({"success": False, "error": f"사용 중지된 플러그인이 없습니다: {name}"})
                    if os.path.exists(enabled_path):
                        return web.json_response({"success": False, "error": f"대상 디렉토리가 존재함: {name}"})
                    shutil.move(src, enabled_path)
                    return web.json_response({"success": True, "enabled": True, "name": name, "path": enabled_path})
                else:
                    # 비활성화: 부터 <name> 이동 .disabled/<name>
                    if not os.path.isdir(enabled_path):
                        if os.path.isdir(disabled_subfolder_path) or os.path.isdir(enabled_path_suffix):
                            return web.json_response({"success": True, "enabled": False, "name": name,
                                                      "already_disabled": True})
                        return web.json_response({"success": False, "error": f"플러그인 찾을 수 없음: {name}"})
                    os.makedirs(disabled_dir, exist_ok=True)
                    if os.path.exists(disabled_subfolder_path):
                        return web.json_response({"success": False, "error": f".disabled 중복된 동일 이름 존재: {name}"})
                    shutil.move(enabled_path, disabled_subfolder_path)
                    return web.json_response({"success": True, "enabled": False, "name": name,
                                              "path": disabled_subfolder_path})

            except Exception as e:
                import traceback
                traceback.print_exc()
                return web.json_response({"success": False, "error": str(e)})

        @server.routes.post("/comfyui_panel/cleanup_broken_plugins")
        async def cleanup_broken_plugins(request):
            """손상된 플러그인 디렉터리 모두 삭제합니다 (디렉터리가 존재하지만 __init__.py가 없거나 유효하지 않은 경우). .git）。

            일반적으로 git clone / zip 설치 실패의 잔여 디렉토리, 후속 재설치를 방지。
            반환 {"success": True, "removed": ["name1", ...], "failed": ["name2", ...]}
            """
            try:
                custom_nodes_dir = folder_paths.get_folder_paths("custom_nodes")
                if not custom_nodes_dir:
                    custom_nodes_dir = os.path.join(os.path.dirname(folder_paths.__file__), "custom_nodes")
                else:
                    custom_nodes_dir = custom_nodes_dir[0]

                removed = []
                failed = []
                if os.path.isdir(custom_nodes_dir):
                    for name in sorted(os.listdir(custom_nodes_dir)):
                        if name in ('.disabled', '__pycache__') or name.startswith('.'):
                            continue
                        full = os.path.join(custom_nodes_dir, name)
                        if not os.path.isdir(full):
                            continue
                        # 오래된 .disabled 접미사 폴더 건너뛰기 (별도의 플러그인으로 개별 평가)）
                        if name.endswith('.disabled'):
                            continue
                        if not _is_valid_plugin_dir(full):
                            if _remove_broken_plugin_dir(full):
                                removed.append(name)
                            else:
                                failed.append(name)

                # 동시에 .disabled 하위 디렉터리에서 손상된 플러그인을 정리합니다
                disabled_dir = os.path.join(custom_nodes_dir, '.disabled')
                if os.path.isdir(disabled_dir):
                    for name in sorted(os.listdir(disabled_dir)):
                        if name == '__pycache__' or name.startswith('.'):
                            continue
                        full = os.path.join(disabled_dir, name)
                        if not os.path.isdir(full):
                            continue
                        if not _is_valid_plugin_dir(full):
                            if _remove_broken_plugin_dir(full):
                                removed.append(name + " (disabled)")
                            else:
                                failed.append(name + " (disabled)")

                if removed:
                    print(f"[ComfyUI Panel] 삭제된 손상 플러그인 디렉토리: {', '.join(removed)}")
                return web.json_response({"success": True, "removed": removed, "failed": failed})
            except Exception as e:
                import traceback
                traceback.print_exc()
                return web.json_response({"success": False, "error": str(e)})

        @server.routes.get("/comfyui_panel/resolve_plugin_source")
        async def resolve_plugin_source(request):
            """노드 이름으로 플러그인의 GitHub 주소를 파싱하고, ComfyUI-Manager 로컬 캐시를 우선 사용합니다."""
            try:
                import json as _json
                import hashlib
                import urllib.request
                import glob as _glob
                import time

                node_names = request.rel_url.query.get("node_names", "").strip()
                # mirror='proxy'(기본) 데이터 매핑을 가속화 다운로드 소스 사용 (중국 직연 raw.githubusercontent 실패 가능성 높음)）
                mirror = request.rel_url.query.get("mirror", "proxy")
                if not node_names:
                    return web.json_response({"success": False, "error": "node_names 비어 있을 수 없음"})

                node_list = [n.strip() for n in node_names.split(",") if n.strip()]

                ext_map = None

                # ===== 우선순위 1: ComfyUI-Manager 로컬 캐시 읽기 시도 =====
                # Manager 캐시 디렉토리: ComfyUI/user/__manager/cache/
                try:
                    comfy_root = os.path.dirname(folder_paths.__file__)
                    manager_cache_dir = os.path.join(comfy_root, "user", "__manager", "cache")
                    if os.path.isdir(manager_cache_dir):
                        # 우선 extension-node-map.json (가장 완전한 매핑)）
                        cache_files = _glob.glob(os.path.join(manager_cache_dir, "*_extension-node-map.json"))
                        if cache_files:
                            cache_file = cache_files[0]  # 첫 번째 일치하는 것
                            cache_age = time.time() - os.path.getmtime(cache_file)
                            if cache_age < 86400 * 7:  # 7일 효과
                                try:
                                    with open(cache_file, 'r', encoding='utf-8') as f:
                                        ext_map = _json.load(f)
                                    print(f"[ComfyUI Panel] ComfyUI-Manager 캐싱 사용: {os.path.basename(cache_file)}")
                                except Exception:
                                    pass
                except Exception:
                    pass

                # ===== 우선순위 2: 플러그인 자체 캐시 =====
                if ext_map is None:
                    cache_dir = os.path.join(os.path.dirname(__file__), "cache")
                    os.makedirs(cache_dir, exist_ok=True)
                    cache_file = os.path.join(cache_dir, "extension-node-map.json")

                    use_cache = False
                    if os.path.isfile(cache_file):
                        cache_age = time.time() - os.path.getmtime(cache_file)
                        if cache_age < 86400:  # 24 hours
                            use_cache = True

                    if use_cache:
                        try:
                            with open(cache_file, 'r', encoding='utf-8') as f:
                                ext_map = _json.load(f)
                        except Exception:
                            ext_map = None

                # ===== 우선순위 3: GitHub에서 원격 다운로드 =====
                if ext_map is None:
                    remote_url = _apply_gh_mirror(
                        "https://raw.githubusercontent.com/ltdrdata/ComfyUI-Manager/main/extension-node-map.json", mirror)
                    try:
                        req = urllib.request.Request(remote_url, headers={"User-Agent": "ComfyUI-Panel/1.0"})
                        with urllib.request.urlopen(req, timeout=15) as resp:
                            raw_data = resp.read()
                        ext_map = _json.loads(raw_data.decode("utf-8"))
                        # 자체 캐시에 저장
                        cache_dir2 = os.path.join(os.path.dirname(__file__), "cache")
                        os.makedirs(cache_dir2, exist_ok=True)
                        with open(os.path.join(cache_dir2, "extension-node-map.json"), 'w', encoding='utf-8') as f:
                            f.write(raw_data.decode("utf-8"))
                    except Exception as e:
                        return web.json_response({"success": False, "error": f"매핑 데이터 다운로드 실패: {str(e)}"})

                if not ext_map:
                    return web.json_response({"success": False, "error": "매핑 데이터 없음"})

                # 빌드 역방향 검색：node_name -> [{url, title}]
                results = {}
                for node_name in node_list:
                    matches = []
                    for url, info in ext_map.items():
                        if isinstance(info, list) and len(info) > 0 and isinstance(info[0], list):
                            if node_name in info[0]:
                                # 추출 플러그인 이름
                                plugin_name = url.rstrip("/")
                                if plugin_name.endswith(".git"):
                                    plugin_name = plugin_name[:-4]
                                plugin_name = os.path.basename(plugin_name)
                                matches.append({
                                    "source": url,
                                    "plugin_name": plugin_name,
                                    "source_type": "github" if "github.com" in url else "zip_url"
                                })
                    if matches:
                        # github.com 주소 우선 선택
                        github_matches = [m for m in matches if m["source_type"] == "github"]
                        results[node_name] = github_matches if github_matches else matches

                return web.json_response({"success": True, "results": results})

            except Exception as e:
                import traceback
                traceback.print_exc()
                return web.json_response({"success": False, "error": str(e)})

        return True

    except Exception as e:
        print(f"[ComfyUI Panel] ✗ Failed to register API: {e}")
        import traceback
        traceback.print_exc()
        return False


try:
    import folder_paths
    import server

    if hasattr(server, 'PromptServer') and hasattr(server.PromptServer, 'instance'):
        setup_api()
    else:
        print("[ComfyUI Panel] Waiting for server initialization...")
        import comfy.model_management
        if hasattr(server.PromptServer, 'instance'):
            setup_api()
        else:
            print("[ComfyUI Panel] Server instance not available, API registration deferred")

except ImportError as e:
    print(f"[ComfyUI Panel] Import error: {e}")
    print("[ComfyUI Panel] Not running in ComfyUI environment, API registration skipped")

# ========== 노드/ 디렉터리에서 노드 자동 감지 ==========
import importlib
_nodes_dir = os.path.join(os.path.dirname(__file__), 'nodes')
if os.path.isdir(_nodes_dir):
    for f in sorted(os.listdir(_nodes_dir)):
        if f.endswith('.py') and not f.startswith('_'):
            module_name = f[:-3]
            try:
                module = importlib.import_module(f'.nodes.{module_name}', package=__package__)
                if hasattr(module, 'NODE_CLASS_MAPPINGS'):
                    NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)
                if hasattr(module, 'NODE_DISPLAY_NAME_MAPPINGS'):
                    NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)
                print(f"[ComfyUI Panel] 노드 모듈 로드: {module_name}")
            except Exception as e:
                print(f"[ComfyUI Panel] 노드 모듈 로드 실패 {module_name}: {e}")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

print(f"[ComfyUI Panel] Extension loaded (v{VERSION})")
