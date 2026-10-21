# SPDX-FileCopyrightText: 2026 灰暗x
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
BSKQwenImageRefSet - Qwen Image 2.1 참조 갤러리 (구조화된 텍스트) → N 이미지 출력）

용도：
  TextEncodeQwenImage21 (ComfyUI 내장 노드)에 참조 이미지를 대량으로 공급합니다. 이 노드의 이미지 입력은
  Autogrow 동적 슬롯(image_1 .. image_16), 수동으로 하나씩 LoadImage 연결하는 것은 번거로움; 이 노드는
  텍스트로 슬롯과 파일을 설명하고, 한 번에 10개의 IMAGE를 직접 드래그 앤 드롭합니다.。

텍스트 형식 (줄 단위)）：
    1 bsk_1.png
    2 sub/bsk_2.png
  · 행 시작 숫자 = 출력 번호 (1..MAX_SLOTS), 뒤에 input 디렉토리 내의 상대 파일명；
    숫자와 파일 이름 사이에 "공백" 또는 ": / ："로 구분 (피하기 "2.png" 이러한 파일 이름은 번호로 오인함）。
  · 행首에 숫자를 쓰지 않음 → 출력 순서대로 자동으로 다음 빈 슬롯에 할당。
  · # 주석으로 시작; 빈 줄은 무시; 중복 번호는 마지막 번호 적용。

출력：
  MAX_SLOTS개의 IMAGE를 사용합니다. 이미지 슬롯을 구성하여 해당 이미지(RGB)를 출력하고, 구성되지 않거나 파일이 없는 슬롯은...
  출력 None —— 후속 Autogrow 슬롯에 빈 텐서가 전송되지 않으며 TextEncodeQwenImage21 내부에서
  `if image is None: continue` 직접 건너뛰므로 오류가 발생하지 않으며, 빈 참조 이미지를 생성하지 않음。

확대 (선택 사항)）：
  resize_enabled 열고, 각 그림을 '자신의 긴 변'에 따라 동일 비율로 크기 조정합니다. long_edge，
  두 개의 가장자리를 각각 divisible_by_32 ? 32 : 8의 배수로 정렬합니다 (최소값은 동일한 배수).）。
  기본적으로 닫혀 있으며 TextEncodeQwenImage21의 해상도 처리로 통합합니다.。

주의: MAX_SLOTS를 16으로 변경하면 TextEncodeQwenImage21의 모든 슬롯을 커버할 수 있습니다.。
"""

import os
import re
import hashlib

# 출력 엔드포인트 수 (TextEncodeQwenImage21은 실제 16개의 Autogrow 슬롯이 있으며, UI 요구사항에 따라 가져옴) 10）
MAX_SLOTS = 10

# 행 시작 번호: 공백 또는 ': / ：'과 파일 이름으로 구분, 방지 "2.png" 잘못 판별됨 "2 . png"
_ROW_RE = re.compile(r'^\s*(\d{1,2})(?:\s+|\s*[:：]\s*)(\S.*?)\s*$')

_IMG_EXT = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif', '.tif', '.tiff')


def _input_dir():
    try:
        import folder_paths
        return folder_paths.get_input_directory()
    except Exception:
        return os.path.abspath(os.path.join('.', 'input'))


def _resolve_path(name):
    """입력 디렉토리의 상대 파일 이름을 절대 경로로 해석합니다. 오버플로우(../) 반환 None。"""
    if not name:
        return None
    base = os.path.abspath(_input_dir())
    rel = name.replace('\\', '/').lstrip('/')
    p = os.path.abspath(os.path.join(base, rel))
    if p != base and not p.startswith(base + os.sep):
        return None
    return p


def parse_image_list(text, max_slots=MAX_SLOTS):
    """구조화된 텍스트 파싱 → [(slot0based, filename), ...]（슬롯 기준으로 오름차순, 중복 제거）。"""
    slots = {}
    auto = 0
    for raw in (text or '').splitlines():
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        m = _ROW_RE.match(line)
        if m:
            idx = int(m.group(1))
            name = m.group(2).strip().strip('"').strip("'").strip()
            if idx < 1 or idx > max_slots or not name:
                continue
            i = idx - 1
        else:
            name = line.strip('"').strip("'").strip()
            if not name:
                continue
            while auto < max_slots and auto in slots:
                auto += 1
            if auto >= max_slots:
                continue
            i = auto
        slots[i] = name
    return sorted(slots.items())


def _align(v, multiple):
    """변을 여러 배수로 정렬 (최소값 multiple）。"""
    return max(multiple, int(round(float(v) / multiple)) * multiple)


def resize_long_edge(img, long_edge, divisible_by_32=True):
    """각 이미지의 긴 변을 동일 비율로 확대: 긴 변 → long_edge，양쪽 모두 32 또는 8의 배수로 정렬。

    원본 이미지 변경 없을 때 동일한 객체 반환 (불필요한 복사 방지)）。
    """
    from PIL import Image

    multiple = 32 if divisible_by_32 else 8
    target = int(long_edge or 0)
    if target <= 0:
        return img
    w, h = img.size
    long_side = max(w, h)
    if long_side <= 0:
        return img
    scale = float(_align(target, multiple)) / float(long_side)
    new_w = _align(w * scale, multiple)
    new_h = _align(h * scale, multiple)
    if (new_w, new_h) == (w, h):
        return img
    resample = getattr(Image, 'LANCZOS', None) or Image.BICUBIC
    return img.resize((new_w, new_h), resample)


def _load_one(path, resize_enabled=False, long_edge=1536, divisible_by_32=True):
    """이미지 한 장 읽기 → ComfyUI IMAGE 텐서 [1,H,W,3]（0..1, RGB）。"""
    from PIL import Image, ImageOps
    import numpy as np
    import torch

    img = Image.open(path)
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass
    if img.mode == 'I':
        img = img.point(lambda i: i * (1.0 / 255.0))
    img = img.convert('RGB')
    if resize_enabled:
        img = resize_long_edge(img, long_edge, divisible_by_32)
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


class BSKQwenImageRefSet:
    """Qwen Image 2.1 참조 갤러리: 구조화된 텍스트 → 10 로 IMAGE。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_list": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "한 줄에 하나씩: 파일 이름 (예: 1 bsk_1.png); 번호가 없으면 자동으로 할당"
                }),
                "resize_enabled": ("BOOLEAN", {
                    "default": False,
                    "label_on": "활성화",
                    "label_off": "사용 중지",
                    "tooltip": "활성화되면 각 그림의 긴 변을 동일 비율로 조정합니다 long_edge"
                }),
                "long_edge": ("INT", {
                    "default": 1536,
                    "min": 64,
                    "max": 8192,
                    "step": 8,
                    "tooltip": "확대된 긴 변 픽셀 (실제 값은 32/8 배수 단위로 맞춰짐)）"
                }),
                "divisible_by_32": ("BOOLEAN", {
                    "default": True,
                    "label_on": "32",
                    "label_off": "8",
                    "tooltip": "가로 정렬 배수: 켜짐=32，관=8"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",) * MAX_SLOTS
    RETURN_NAMES = tuple("그림%d" % (i + 1) for i in range(MAX_SLOTS))
    FUNCTION = "execute"
    CATEGORY = "bsk_UI/이미지"
    DESCRIPTION = (
        "구조화된 텍스트를 한 번에 여러 IMAGE 출력, TextEncodeQwenImage21의 참조 이미지 슬롯 사용。"
        "각 행 '번호 파일명'(파일은 input 폴더에 위치); 미설정 슬롯 출력 None, 하류는 안전하게 건너듸。"
    )

    def execute(self, image_list, resize_enabled=False, long_edge=1536, divisible_by_32=True):
        outs = [None] * MAX_SLOTS
        missing = []
        for i, name in parse_image_list(image_list):
            path = _resolve_path(name)
            if not path or not os.path.isfile(path):
                missing.append(name)
                continue
            try:
                outs[i] = _load_one(path, resize_enabled, long_edge, divisible_by_32)
            except Exception as e:
                missing.append("%s (%s)" % (name, e))
        if missing:
            print("[BSKQwenImageRefSet] 다음 항목이 로드되지 않음 (출력은 None）: %s" % missing)
        if resize_enabled:
            print("[BSKQwenImageRefSet] 긴 변 확대: 목표 %d, 배수 %d"
                  % (int(long_edge), 32 if divisible_by_32 else 8))
        return tuple(outs)

    @classmethod
    def IS_CHANGED(cls, image_list, resize_enabled=False, long_edge=1536, divisible_by_32=True):
        """입력 텍스트가 변하지 않았지만 파일이 동일한 이름으로 덮어씌워졌을 때도 다시 실행합니다: [키] mtime+size 지문 생성。"""
        try:
            h = hashlib.md5()
            h.update(("%s|%s|%s" % (bool(resize_enabled), int(long_edge),
                                    32 if divisible_by_32 else 8)).encode("utf-8"))
            for i, name in parse_image_list(image_list):
                path = _resolve_path(name)
                try:
                    st = os.stat(path)
                    h.update(("%d|%s|%d|%d" % (i, name, st.st_mtime_ns, st.st_size)).encode("utf-8"))
                except Exception:
                    h.update(("%d|%s|-" % (i, name)).encode("utf-8"))
            return h.hexdigest()
        except Exception:
            return float("NaN")


NODE_CLASS_MAPPINGS = {
    "BSKQwenImageRefSet": BSKQwenImageRefSet,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BSKQwenImageRefSet": "BSK Qwen참고 갤러리",
}
