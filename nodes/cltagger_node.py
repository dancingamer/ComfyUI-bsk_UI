# SPDX-License-Identifier: AGPL-3.0-or-later
#
# BSK Tagger —— ComfyUI-bsk_UI 의 로컬 ONNX 타격 노드。
# Copyright (C) 2026 灰暗x (B상) @灰暗x). ComfyUI-bsk_UI의 일부，
# AGPL-3.0-or-later로 배포。
#
# 본 파일은 두 가지 유형의 로컬 ONNX 라벨러를 모두 지원하며, 사전 처리/라벨 형식/후처리 로직을 엄격하게 구분합니다.：
#
#  - CLTagger: 모델 출처 HF "cella110n/cl_tagger" (자 디렉토리 cl_tagger_1_02, ONNX)，
#    레이블이 JSON입니다 /255 -> HWC->CHW -> BGR -> (x-0.5)/0.5。
#    이미지 전처리 및 태그 후처리 로직은 오픈 소스 프로젝트에서 파생됨 "cl_tagger"
#    (https://github.com/Cirius0310/cl_tagger, Apache License 2.0)。Apache-2.0 전체 텍스트
#    본 저장소 LICENSES/Apache-2.0.txt 참조. 이 모델은 "SmilingWolf/wd-eva02-large-tagger-v3"
#    미세 조정됨。
#
#  - WD14: 모델은 SmilingWolf의 wd-tagger-v3 시리즈에서 제공（wd-vit / wd-convnext /
#    wd-eva02-large / wd-swinv2），레이블은 selected_tags.csv(category 열) 입니다：
#    0 = rating 출발점，4 = character 출발점）；프리프로세싱을 직접 float32 + RGB->BGR，
#    **255로 나누지 않고, 평균 정규화 하지 않음**（CLTagger와 완전히 반대이므로, 반드시 구별하십시오.）。
#    이 노드의 WD14 추론 부분은 / 파생된 오픈소스 프로젝트를 참조합니다 "ComfyUI-WD14-Tagger"
#    (https://github.com/pythongosssss/ComfyUI-WD14-Tagger, MIT License,
#    Copyright (c) 2024 pythongosssss)；MIT 라이선스 전체 내용은 이 저장소 참조 LICENSES/MIT.txt。
#    CSV 레이블 형식 약관은 SmilingWolf 커뮤니티 모델에서 비롯되었습니다.。
#
# 모델 가중치는 이 플러그인에 포함되지 않으며, 처음 사용할 때 Hugging Face에서 실행 시간에 다운로드합니다.。
# 다운로드는 기본적으로 HF 미러 https://hf-mirror.com을 사용하며, HF_ENDPOINT 환경 변수를 통해 덮어쓸 수 있습니다.。
"""
BSK_Tagger - 길이 검사 실패 또는 즉시 멈춤. X-Script-Name은 마운트 전용 접미사이며, 저희가 통일하여 지정합니다.。

CLTagger:
- 8 카테고리: Rating / Quality는 argmax를 사용하고, General / Meta / Model은 사용합니다 general_threshold；
  Character / Copyright / Artist 용 character_threshold。
- 사전 처리: 흰색 정사각형 pad -> 448 BICUBIC -> /255 -> HWC->CHW -> RGB->BGR -> (x-0.5)/0.5。

WD14 (SmilingWolf wd-tagger-v3 시리즈):
- 카테고리：rating (argmax) + general (임계값) + character (임계값, CSV 카테고리 구분자 사용。
- 장면을 줄여서 max_resolution -> 모델 크기로 줄임 -> 백색 배경 pad -> RGB->BGR（무 정규화）。

numpy / Pillow / onnxruntime에만 의존하며, 표준 라이브러리가 다운로드하고, huggingface_hub 의 의존성을 줄입니다.。
"""

import os
import re
import csv
import json
import time
import sys
import base64
import io
import subprocess
import threading
import atexit
import urllib.request
import urllib.error

import numpy as np
from PIL import Image

try:
    import onnxruntime as ort
except Exception:  # noqa: BLE001
    ort = None

try:
    import folder_paths
    _MODELS_ROOT = folder_paths.models_dir
except Exception:  # noqa: BLE001
    _MODELS_ROOT = os.path.join(os.path.abspath("./"), "models")

# ---------------------------------------------------------------------------
# 상수 및 경로
# ---------------------------------------------------------------------------
DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"

# --- CLTagger 설정 ---
CLTAGGER_REPO_ID = "cella110n/cl_tagger"
CLTAGGER_SUBDIR = "cl_tagger_1_02"
CLTAGGER_ONNX_REMOTE = f"{CLTAGGER_SUBDIR}/model.onnx"
CLTAGGER_TAGMAP_REMOTE = f"{CLTAGGER_SUBDIR}/tag_mapping.json"
CLTAGGER_DIR = os.path.join(_MODELS_ROOT, "cltagger")
CLTAGGER_ONNX_PATH = os.path.join(CLTAGGER_DIR, "model.onnx")
CLTAGGER_TAGMAP_PATH = os.path.join(CLTAGGER_DIR, "tag_mapping.json")

# --- WD14 설정 (v3 시리즈만)）---
WD14_MODELS = {
    "wd-vit-tagger-v3": "SmilingWolf/wd-vit-tagger-v3",
    "wd-convnext-tagger-v3": "SmilingWolf/wd-convnext-tagger-v3",
    "wd-eva02-large-tagger-v3": "SmilingWolf/wd-eva02-large-tagger-v3",
    "wd-swinv2-tagger-v3": "SmilingWolf/wd-swinv2-tagger-v3",
}
WD14_DIR = os.path.join(_MODELS_ROOT, "wd14")

# Register exclusive folder type, provide panel "Model Download" API precise write to this node to read directory
# （model_type 전 "bsk_wd14" / "bsk_cltagger" 아래 디렉토리로 파싱, models_dir 사용자 정의 설정에 영향을 받지 않음）
try:
    import folder_paths as _fp
    _fp.add_model_folder_path("bsk_wd14", os.path.join(_MODELS_ROOT, "wd14"))
    _fp.add_model_folder_path("bsk_cltagger", os.path.join(_MODELS_ROOT, "cltagger"))
except Exception:  # noqa: BLE001
    pass

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tiff", ".tif"}

# 실행 시 캐싱: (tagger_type, model_key)로 구분, 전환 시 오류 발생 방지
_SESSIONS = {}


# ---------------------------------------------------------------------------
# 모델 다운로드 (국내 직연 원 우선, 실패 회퇴 HF 이미지, 재 회퇴 공식 원)）
# ---------------------------------------------------------------------------
# 국내 직 연결 우선 원본: 카드 '수동 모델 다운로드' 기본 주소와 일치。
# 선택된 모델이 여기에 맞으면 (즉, 카드 기본 주소에 해당하는 모델), 자동으로 다운로드할 때 중국 직 링크를 우선 사용합니다.
# （ModelScope，속도가 빠르고 타임아웃되지 않도록 합니다); 중국 소스 실패 시에만 HF 이미지 체인을 되돌립니다.。
# 타격 조건 key = (tagger_type, wd14_model or None)。
# 참고: URL에 지정된 파일 이름과 백엔드에 저장된 파일 이름이 다를 수 있습니다 (예 model_optimized.onnx → model.onnx），
# 저장 파일 이름은 호출 지점의 filename에 의해 결정되며, 다운로드 소스만 처리합니다。
MODEL_DOMESTIC_URLS = {
    ("cltagger", None): {
        "cl_tagger_1_02/model.onnx": "https://modelscope.cn/models/nikoovo/cl-tagger/resolve/master/cl_tagger_1_01/model_optimized.onnx",
        "cl_tagger_1_02/tag_mapping.json": "https://modelscope.cn/models/nikoovo/cl-tagger/resolve/master/cl_tagger_1_02/tag_mapping.json",
    },
    ("wd14", "wd-vit-tagger-v3"): {
        "model.onnx": "https://modelscope.cn/models/shiertier/wd14_tagger/resolve/master/wd-vit-tagger-v3.onnx",
        "selected_tags.csv": "https://modelscope.cn/models/shiertier/wd14_tagger/resolve/master/wd-vit-tagger-v3.csv",
    },
}


def _download_one(url, dest):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": "bsk_UI-tagger/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        total = int(resp.headers.get("Content-Length", 0) or 0)
        got = 0
        last_pct = -1
        with open(tmp, "wb") as f:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                got += len(chunk)
                if total > 0:
                    pct = int(got * 100 / total)
                    if pct != last_pct and pct % 5 == 0:
                        last_pct = pct
                        print(f"[BSK_Tagger] 다운로드 {os.path.basename(dest)}: {pct}% "
                              f"({got // (1 << 20)}MB/{total // (1 << 20)}MB)")
    os.replace(tmp, dest)
    print(f"[BSK_Tagger] 저장됨: {dest}")


def _ensure_model_paths(tagger_type, wd14_model):
    """ONNX 파일과 태그 파일을 위치에 올리고, 누락된 경우 HF 이미지에서 다운로드합니다. 반환 (onnx_path, tagmap_path)。"""
    if tagger_type == "wd14":
        if wd14_model not in WD14_MODELS:
            raise ValueError(f"[BSK_Tagger] 알 수 없는 wd14 모델: {wd14_model}")
        repo_id = WD14_MODELS[wd14_model]
        model_dir = os.path.join(WD14_DIR, wd14_model)
        onnx_path = os.path.join(model_dir, "model.onnx")
        tagmap_path = os.path.join(model_dir, "selected_tags.csv")
        need = []
        if not os.path.isfile(onnx_path):
            need.append(("model.onnx", onnx_path))
        if not os.path.isfile(tagmap_path):
            need.append(("selected_tags.csv", tagmap_path))
    else:
        repo_id = CLTAGGER_REPO_ID
        model_dir = CLTAGGER_DIR
        onnx_path = CLTAGGER_ONNX_PATH
        tagmap_path = CLTAGGER_TAGMAP_PATH
        need = []
        if not os.path.isfile(onnx_path):
            need.append((CLTAGGER_ONNX_REMOTE, onnx_path))
        if not os.path.isfile(tagmap_path):
            need.append((CLTAGGER_TAGMAP_REMOTE, tagmap_path))

    if not need:
        return onnx_path, tagmap_path

    label = (wd14_model if tagger_type == "wd14" else "cltagger")
    # 국내 직연 우선 원: 타겟 카드 기본 주소에 해당하는 모델일 때 우선 사용, 실패 시 HF 미러 체인 사용
    domestic = MODEL_DOMESTIC_URLS.get(
        (tagger_type, wd14_model if tagger_type == "wd14" else None), {}
    )
    # 후보 이미지: 사용자 지정 HF_ENDPOINT 우선, 다음으로 기본 hf-mirror, 마지막으로 공식 소스 복구。
    # 단일 미러 (특히 hf-mirror)는 일부 네트워크에서 다운로드할 수 없어 순서대로 롤백하여 성공률을 높입니다.。
    user_ep = os.getenv("HF_ENDPOINT")
    candidate_eps = []
    if user_ep:
        candidate_eps.append(user_ep.strip().rstrip("/"))
    candidate_eps.append(DEFAULT_HF_ENDPOINT)
    candidate_eps.append("https://huggingface.co")
    endpoints = []
    for e in candidate_eps:
        if e not in endpoints:
            endpoints.append(e)
    print(f"[BSK_Tagger] 모델이 누락되었습니다. 다운로드를 시작합니다 {label} (국내 직 연결 우선, 이미지 롤백: {', '.join(endpoints)}) ...")
    for remote, dest in need:
        # 후보 URL: 국내 직렬 원 우선, 일치 시 국내 링크 사용 (빠르고 타임아웃 방지)）
        urls = []
        if remote in domestic:
            urls.append(domestic[remote])
        for ep in endpoints:
            urls.append(f"{ep}/{repo_id}/resolve/main/{remote}")
        last_err = None
        for url in urls:
            try:
                _download_one(url, dest)
                break
            except (urllib.error.URLError, urllib.error.HTTPError, OSError) as err:
                last_err = err
                print(f"[BSK_Tagger] 다운로드 실패: {url}\n  오류: {err}，다음 시도")
        else:
            raise RuntimeError(
                f"[BSK_Tagger] 다운로드 실패: {remote}\n"
                f"마지막 오류: {last_err}\n"
                f"제안：① 카드 '수동 모델 다운로드'에 유효 주소 붙여넣기 (예: ModelScope 온라인 페이스) "
                f"wd-vit-tagger-v3.onnx）후점 다운로드；② 또 HF_ENDPOINT 환경 변수를 설정하여 이미지를 지정할 수 있습니다.；"
                f"③ 또는 수동으로 {remote} 배치 {os.path.dirname(dest)}/ 아래。"
            ) from last_err
    return onnx_path, tagmap_path


# ---------------------------------------------------------------------------
# 라벨 매핑 파싱
# ---------------------------------------------------------------------------
class _LabelData:
    __slots__ = ("names", "rating", "general", "artist", "character",
                 "copyright", "meta", "quality", "model")

    def __init__(self, names, rating, general, artist, character,
                 copyright_, meta, quality, model):
        self.names = names
        self.rating = rating
        self.general = general
        self.artist = artist
        self.character = character
        self.copyright = copyright_
        self.meta = meta
        self.quality = quality
        self.model = model


def _load_tag_mapping(mapping_path):
    """CLTagger의 JSON 태그 매핑을 분석합니다。"""
    with open(mapping_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "idx_to_tag" in data:
        idx_to_tag = {int(k): v for k, v in data["idx_to_tag"].items()}
        tag_to_category = data["tag_to_category"]
    elif isinstance(data, dict):
        try:
            di = {int(k): v for k, v in data.items()}
            idx_to_tag = {idx: d["tag"] for idx, d in di.items()}
            tag_to_category = {d["tag"]: d["category"] for d in di.values()}
        except (KeyError, ValueError) as e:
            raise ValueError(f"[BSK_Tagger] 지원되지 않는 tag_mapping 형식: {e}")
    else:
        raise ValueError("[BSK_Tagger] 지원되지 않는 tag_mapping 형식: 딕셔너리여야 함。")

    names = [None] * (max(idx_to_tag.keys()) + 1)
    rating, general, artist, character = [], [], [], []
    copyright_tags, meta, quality, model_name = [], [], [], []
    for idx, tag in idx_to_tag.items():
        if idx >= len(names):
            names.extend([None] * (idx - len(names) + 1))
        names[idx] = tag
        category = tag_to_category.get(tag, "Unknown")
        idx_int = int(idx)
        if category == "Rating":
            rating.append(idx_int)
        elif category == "General":
            general.append(idx_int)
        elif category == "Artist":
            artist.append(idx_int)
        elif category == "Character":
            character.append(idx_int)
        elif category == "Copyright":
            copyright_tags.append(idx_int)
        elif category == "Meta":
            meta.append(idx_int)
        elif category == "Quality":
            quality.append(idx_int)
        elif category == "Model":
            model_name.append(idx_int)

    return _LabelData(
        names=names,
        rating=np.array(rating, dtype=np.int64),
        general=np.array(general, dtype=np.int64),
        artist=np.array(artist, dtype=np.int64),
        character=np.array(character, dtype=np.int64),
        copyright_=np.array(copyright_tags, dtype=np.int64),
        meta=np.array(meta, dtype=np.int64),
        quality=np.array(quality, dtype=np.int64),
        model=np.array(model_name, dtype=np.int64),
    )


def _load_wd14_csv(csv_path):
    """WD14의 selected_tags.csv를 파싱하여 반환 (tags, general_index, character_index)。

    category 열 약속（SmilingWolf）：0=rating 출발점，4=character 출발점。
    general_index / character_index 테마 토큰 색상 (프롬프트 라이브러리 편집 상자) category。
    """
    tags = []
    general_index = None
    character_index = None
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        next(reader)  # 헤더 건너뛰기
        for row in reader:
            if len(row) < 3:
                continue
            if general_index is None and row[2] == "0":
                general_index = reader.line_num - 2
            elif character_index is None and row[2] == "4":
                character_index = reader.line_num - 2
            tags.append(row[1])
    return tags, general_index, character_index


# ---------------------------------------------------------------------------
# 사전 처리 (두 세트, 엄격하게 구분）
# ---------------------------------------------------------------------------
def _pil_ensure_rgb(image):
    if image.mode not in ("RGB", "RGBA"):
        image = image.convert("RGBA") if "transparency" in image.info else image.convert("RGB")
    if image.mode == "RGBA":
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])
        image = background
    return image


def _pil_pad_square(image):
    w, h = image.size
    if w == h:
        return image
    new_size = max(w, h)
    new_image = Image.new(image.mode, (new_size, new_size), (255, 255, 255))
    new_image.paste(image, ((new_size - w) // 2, (new_size - h) // 2))
    return new_image


def _preprocess_cltagger(image, target_size=(448, 448)):
    """CLTagger: /255 -> HWC->CHW -> BGR -> (x-0.5)/0.5，범위 [-1,1]。"""
    image = _pil_ensure_rgb(image)
    image = _pil_pad_square(image)
    resized = image.resize(target_size, Image.BICUBIC)
    arr = np.array(resized, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)          # HWC -> CHW
    arr = arr[::-1, :, :]                 # RGB -> BGR
    mean = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(3, 1, 1)
    arr = (arr - mean) / std
    arr = np.expand_dims(arr, axis=0)
    return np.ascontiguousarray(arr, dtype=np.float32)


def _preprocess_wd14(image, target_size, max_resolution):
    """WD14: 직접 float32 + RGB->BGR，**없음 /255, 없음 평균화 정규화**（범위 [0,255]）。"""
    image = image.convert("RGB")
    # 추론 전 스케일링: 가장 긴 변의 길이가 max_resolution을 초과하지 않도록 하며, 가로 세비율을 유지합니다.
    if max_resolution and max_resolution > 0:
        max_side = max(image.size)
        if max_side > max_resolution:
            ratio = max_resolution / max_side
            new_size = tuple(int(round(x * ratio)) for x in image.size)
            image = image.resize(new_size, Image.LANCZOS)
    height = target_size[0]
    ratio = float(height) / max(image.size)
    new_size = tuple(int(round(x * ratio)) for x in image.size)
    image = image.resize(new_size, Image.LANCZOS)
    square = Image.new("RGB", (height, height), (255, 255, 255))
    square.paste(image, ((height - new_size[0]) // 2, (height - new_size[1]) // 2))
    arr = np.array(square).astype(np.float32)
    arr = arr[:, :, ::-1]                  # RGB -> BGR
    arr = np.expand_dims(arr, axis=0)
    return np.ascontiguousarray(arr, dtype=np.float32)


def _stable_sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


# ---------------------------------------------------------------------------
# 후처리(두 세트, 엄격하게 구분)）
# ---------------------------------------------------------------------------
def _get_cltags(probs, labels, gen_threshold, char_threshold):
    """CLTagger 8 클리어 후처리。"""
    result = {
        "rating": [], "general": [], "character": [],
        "copyright": [], "artist": [], "meta": [],
        "quality": [], "model": [],
    }

    def _argmax_category(indices, key):
        if len(indices) == 0:
            return
        valid = indices[indices < len(probs)]
        if len(valid) == 0:
            return
        cat_probs = probs[valid]
        local = int(np.argmax(cat_probs))
        g = valid[local]
        if g < len(labels.names) and labels.names[g] is not None:
            result[key].append((labels.names[g], float(cat_probs[local])))

    _argmax_category(labels.rating, "rating")
    _argmax_category(labels.quality, "quality")

    category_map = {
        "general": (labels.general, gen_threshold),
        "character": (labels.character, char_threshold),
        "copyright": (labels.copyright, char_threshold),
        "artist": (labels.artist, char_threshold),
        "meta": (labels.meta, gen_threshold),
        "model": (labels.model, gen_threshold),
    }
    for category, (indices, threshold) in category_map.items():
        if len(indices) == 0:
            continue
        valid = indices[indices < len(probs)]
        if len(valid) == 0:
            continue
        cat_probs = probs[valid]
        mask = cat_probs >= threshold
        sel_local = np.where(mask)[0]
        if len(sel_local) == 0:
            continue
        sel_global = valid[sel_local]
        sel_probs = cat_probs[sel_local]
        for g, p in zip(sel_global, sel_probs):
            if g < len(labels.names) and labels.names[g] is not None:
                result[category].append((labels.names[g], float(p)))

    for k in result:
        result[k] = sorted(result[k], key=lambda x: x[1], reverse=True)
    return result


_META_EXCLUDE = ("id", "commentary", "request", "mismatch")
_NBOYGIRL_RE = re.compile(r"[1-6]\+?(girl|boy)s?")


def _format_default(predictions, exclude_meta, exclude_rating, exclude_quality):
    out = []
    if predictions.get("rating") and not exclude_rating:
        out.append(predictions["rating"][0][0].replace("_", " "))
    if predictions.get("quality") and not exclude_quality:
        out.append(predictions["quality"][0][0].replace("_", " "))
    categories = ["artist", "character", "copyright", "general", "meta", "model"]
    for category in categories:
        if category == "meta" and exclude_meta:
            continue
        for tag, _ in predictions.get(category, []):
            if category == "meta" and any(p in tag.lower() for p in _META_EXCLUDE):
                continue
            out.append(tag.replace("_", " "))
    return ", ".join(out)


def _format_novelai(predictions):
    out = []
    for category in ["character", "copyright", "artist", "general"]:
        for tag, _ in predictions.get(category, []):
            out.append(tag.replace("_", " "))
    boys = [t for t in out if "girl" not in t and _NBOYGIRL_RE.search(t)]
    girls = [t for t in out if "girl" in t and _NBOYGIRL_RE.search(t)]
    others = [t for t in out if not _NBOYGIRL_RE.search(t)]
    return ", ".join(boys + girls + others)


def _get_wd14_tags(probs, tags, general_index, character_index,
                   gen_threshold, char_threshold, exclude_tags, replace_underscore,
                   trailing_comma):
    """WD14 후처리：rating(argmax) + general(임계값) + character(임계값)。"""
    gi = general_index if general_index is not None else 0
    ci = character_index if character_index is not None else len(tags)
    if len(probs) != len(tags):
        print(f"[BSK_Tagger][WD14] ⚠ 차원 불일치: probs={len(probs)} tags={len(tags)} "
              f"(probs 해야 (N,) 확률 벡터, type={type(probs)})")
    result = list(zip(tags, probs))
    general = [item for item in result[gi:ci] if item[1] > gen_threshold]
    character = [item for item in result[ci:] if item[1] > char_threshold]
    _top = sorted(result, key=lambda x: x[1], reverse=True)[:5]
    print(f"[BSK_Tagger][WD14] tags={len(tags)} gi(원시)={general_index} ci(원시)={character_index} "
          f"gen_th={gen_threshold} char_th={char_threshold} | 맞중 general={len(general)} character={len(character)}")
    print(f"[BSK_Tagger][WD14] top5={[(t, round(float(p), 3)) for t, p in _top]}")
    selected = character + general
    remove = [s.strip().lower() for s in exclude_tags.split(",") if s.strip()]
    if remove:
        selected = [t for t in selected if t[0].lower() not in remove]
    out = []
    for t, _ in selected:
        if replace_underscore:
            t = t.replace("_", " ")
        out.append(t)
    if trailing_comma:
        return "".join(t + ", " for t in out)
    return ", ".join(out)


# ---------------------------------------------------------------------------
# ONNX session（느린 초기화 + 유형으로 캐싱）
# ---------------------------------------------------------------------------
_ORT_ENV_DIAGNOSED = False


def _diagnose_ort_env():
    """일회성 ONNXRuntime 환경 진단, CPU 대신 GPU 사용 이유 파악.。"""
    global _ORT_ENV_DIAGNOSED
    if _ORT_ENV_DIAGNOSED:
        return
    _ORT_ENV_DIAGNOSED = True
    print("=" * 70)
    print("[BSK_Tagger][진단] ===== onnxruntime 환경 검사 =====")
    if ort is None:
        print("[BSK_Tagger][진단] onnxruntime 미 설치 (import 실패). 설치해주세요. onnxruntime-gpu。")
        print("=" * 70)
        return
    # 1) 현재 가져온 onnxruntime은 어떤 패키지?
    try:
        print(f"[BSK_Tagger][진단] ort.__version__ = {getattr(ort, '__version__', '?')}")
        print(f"[BSK_Tagger][진단] ort.__file__    = {getattr(ort, '__file__', '?')}")
    except Exception as e:  # noqa: BLE001
        print(f"[BSK_Tagger][진단] ort 버전/경로 읽기 실패: {e}")
    # 2) 실제 환경에 설치된 onnxruntime 배포판은 무엇입니까?→남은 CPU）
    try:
        import importlib.metadata as _md
        found = []
        for dist in _md.distributions():
            try:
                name = dist.metadata["Name"]
            except Exception:  # noqa: BLE001
                name = None
            if name and name.lower().startswith("onnxruntime"):
                found.append(f"{name}=={dist.version}")
        if found:
            print(f"[BSK_Tagger][진단] onnxruntime 배포판 설치됨: {found}")
            names = [f.split('==')[0].lower() for f in found]
            has_cpu = "onnxruntime" in names
            has_gpu = "onnxruntime-gpu" in names
            if has_cpu and has_gpu:
                print("[BSK_Tagger][진단] ⚠ 온nxruntime(CPU)와 함께 설치 onnxruntime-gpu，"
                      "공통 네임스페이스 사용 시 충돌 발생, CUDA EP 사용 불가 문제 발생 가능。"
                      "pip uninstall onnxruntime을 통해 순수한 CPU 버전을 제거하십시오. 유지합니다. onnxruntime-gpu。")
            elif has_cpu and not has_gpu:
                print("[BSK_Tagger][진단] ⚠ onnxruntime(CPU 버전)만 설치되어 있으며... onnxruntime-gpu，"
                      "따라서 GPU를 사용할 수 없습니다. 설치：pip install onnxruntime-gpu")
        else:
            print("[BSK_Tagger][진단] importlib.metadata에서 onnxruntime 배포판 정보를 찾을 수 없음。")
    except Exception as e:  # noqa: BLE001
        print(f"[BSK_Tagger][진단] 설치 배포판 목록 실패: {e}")
    # 3) onnxruntime 장치 및 사용 가능성 확인 provider
    try:
        print(f"[BSK_Tagger][진단] ort.get_device()             = {ort.get_device()}")
    except Exception as e:  # noqa: BLE001
        print(f"[BSK_Tagger][진단] ort.get_device() 실패: {e}")
    try:
        avail = ort.get_available_providers()
        print(f"[BSK_Tagger][진단] ort.get_available_providers() = {avail}")
        if "CUDAExecutionProvider" not in avail:
            if "CANNExecutionProvider" in avail:
                if _cann_via_worker():
                    print("[BSK_Tagger][진단] ✓ CANNExecutionProvider 사용 가능; 프로세스 로드됨 torch_npu，"
                          "독립형 서브 프로세스 worker를 사용하여 昇腾 NPU(동 프로세스 경쟁 방지) ACL）。")
                else:
                    print("[BSK_Tagger][진단] ✓ CANNExecutionProvider 사용 가능, 이 노드는 우선적으로 시먼티를 사용합니다. NPU。")
            elif "DmlExecutionProvider" in avail:
                print("[BSK_Tagger][진단] ✓ DmlExecutionProvider 사용 가능, 이 노드는 우선 모델 자체 사용 DirectML。")
            else:
                print("[BSK_Tagger][진단] ✗ GPU/NPU 가속 EP(CUDA/CANN/DML 모두 사용 불가능). 가능한 원인：")
                print("[BSK_Tagger][진단] a) onnxruntime(CPU 버전)을 사용 중, 비 onnxruntime-gpu；")
                print("[BSK_Tagger][진단] b) GPU 버전 설치 시 CUDA/cuDNN DLL 로드 실패"
                      "（CUDA 버전과 onnxruntime-gpu 요구 사항이 일치하지 않거나 cuDNN/zlib이 누락되었거나 PATH를 찾을 수 없습니다. DLL）；")
                print("[BSK_Tagger][진단] c) onnxruntime-gpu 버전과 호스트 CUDA 대 버전 일치 여부 확인"
                      "（ORT를 사용하려면 CUDA 12.x가 필요하며, 이 기계에 설치되어 있습니다. 11.x）。")
        else:
            print("[BSK_Tagger][진단] ✓ CUDAExecutionProvider 사용 가능, 이 노드는 우선 사용。")
    except Exception as e:  # noqa: BLE001
        print(f"[BSK_Tagger][진단] ort.get_available_providers() 실패: {e}")
    print("=" * 70)


def _ort_session_options():
    """SessionOptions 구성: 스레드 수를 명시적으로 지정하여 클라우드 호스트 코어 바인딩 오류 방지。

    일부 클라우드 환경에서 CPUID를 인식할 수 없습니다. CPU vendor（"Unknown CPU vendor. cpuinfo_vendor value: 15"），
    ORT 기본적으로 토폴로지에 따라 스레드 수를 자동으로 계산하고 pthread_setaffinity_np를 사용하여 코어를 바인딩하며, 계산된 마스크가 유효하지 않습니다. →
    "pthread_setaffinity_np failed ... error code: 22 Invalid argument"。
    intra/inter_op_num_threads를 명시적으로 지정하면 ORT에서 더 이상 친화성 설정이 불필요함 (공식 권장, 오류 메시지 참조)）。
    BSK_TAGGER_ORT_THREADS 환경 변수를 사용하여 스레드 수를 덮어씁니다.；log_severity_level=3 부록
    cpuid vendor 경고 제거。
    """
    so = ort.SessionOptions()
    try:
        n = os.getenv("BSK_TAGGER_ORT_THREADS", "").strip()
        n = int(n) if n.isdigit() and int(n) > 0 else 0
        if n <= 0:
            n = os.cpu_count() or 4
        n = max(1, min(n, 64))
        so.intra_op_num_threads = n
        so.inter_op_num_threads = n
    except Exception:  # noqa: BLE001
        pass
    try:
        so.log_severity_level = 3  # ORT_LOGGING_ERROR：오류 이상만 유지
    except Exception:  # noqa: BLE001
        pass
    return so


def _providers():
    _diagnose_ort_env()
    available = set(ort.get_available_providers())
    order = []
    if "CUDAExecutionProvider" in available:
        order.append("CUDAExecutionProvider")
    if "DmlExecutionProvider" in available:
        order.append("DmlExecutionProvider")
    if "CANNExecutionProvider" in available and not _cann_via_worker():
        # Huawei昇腾 NPU: 기본적으로 가상 장치 0을 사용합니다 (ASCEND_RT_VISIBLE_DEVICES 제약 대상)）
        # 주의: 이 프로세스가 torch_npu(onnxruntime-cann과 경쟁)를 로드한 경우 ACL），in-process CANN
        # 사용할 수 없습니다. CANN은 별도 자식 프로세스 worker(_get_cann_worker)에서 사용합니다.。
        order.append(("CANNExecutionProvider", {"device_id": 0}))
    order.append("CPUExecutionProvider")
    return order


def _detect_accelerator():
    """best-effort 호환되는 하드웨어 가속기 플랫폼을 탐지하고 반환합니다 'nvidia' | 'amd' | 'ascend' | 'none'。
    진단용 프롬프트에만 사용됩니다. 어떤 이상 탐지라도 'none'으로 되돌려 추론 프로세스에 영향을 주지 않습니다.。"""
    import shutil
    try:
        import torch
    except Exception:  # noqa: BLE001
        torch = None

    # 1) NVIDIA（nvidia-smi 드라이버가 설치된 Windows/Linux에서는 일반적으로 존재합니다）
    if shutil.which("nvidia-smi"):
        return "nvidia"
    if torch is not None and getattr(torch, "cuda", None) is not None \
            and torch.cuda.is_available() and not getattr(torch.version, "hip", None):
        return "nvidia"

    # 2) AMD ROCm
    if shutil.which("rocm-smi") or shutil.which("rocminfo"):
        return "amd"
    if torch is not None and getattr(torch.version, "hip", None):
        return "amd"
    if os.getenv("ROCM_HOME") or os.getenv("HIP_VISIBLE_DEVICES") is not None:
        return "amd"

    # 3) Huawei Sentient Intelligence NPU
    if shutil.which("npu-smi") or shutil.which("ascend-smi"):
        return "ascend"
    if os.path.exists("/dev/davinci0"):
        return "ascend"
    if os.getenv("ASCEND_HOME") or os.getenv("ASCEND_HOME_PATH") or os.getenv("ASCEND_OPP_PATH"):
        return "ascend"

    return "none"


def _detect_cuda_version():
    """best-effort 본인 CUDA runtime 버전을 탐지하고 '12.4'와 같이 반환 None。"""
    try:
        import torch
        if getattr(torch, "version", None) and torch.version.cuda:
            return torch.version.cuda
    except Exception:  # noqa: BLE001
        pass
    import shutil
    import subprocess
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(
                ["nvidia-smi"], capture_output=True, text=True, timeout=10
            ).stdout
            for line in out.splitlines():
                if "CUDA Version" in line:
                    return line.split("CUDA Version:")[-1].split("|")[0].strip()
        except Exception:  # noqa: BLE001
            pass
    return None


def _cpu_inference_advice(actual, requested):
    """실제 CPU 추론을 사용할 때, 본인 머신 가속기 플랫폼에 맞는 해결책을 제시. 출력만, 동작은 변경 없음。"""
    hw = _detect_accelerator()
    print("=" * 70)
    print(f"[BSK_Tagger][진단] ===== 실제 provider={actual[0]}，호환되는 하드웨어 가속기를 사용하여 해결 방법을 제안합니다 =====")
    print(f"[BSK_Tagger][진단] 감지된 가속기 플랫폼: {hw}")
    _ep_name = lambda p: p[0] if isinstance(p, tuple) else p
    requested_non_cpu = [_ep_name(p) for p in requested if _ep_name(p) != "CPUExecutionProvider"]
    if requested_non_cpu:
        print(f"[BSK_Tagger][진단] 요청된 가속 EP={requested_non_cpu}，하지만 모두 선택되지 않았습니다 (롤백) CPU）。")

    # 설치됨
    try:
        import importlib.metadata as _md
        found = []
        for dist in _md.distributions():
            try:
                name = dist.metadata["Name"]
            except Exception:  # noqa: BLE001
                name = None
            if name and name.lower().startswith("onnxruntime"):
                found.append(f"{name}=={dist.version}")
        names = [f.split('==')[0].lower() for f in found]
    except Exception:  # noqa: BLE001
        found, names = [], []
    has_cpu = "onnxruntime" in names
    has_gpu = "onnxruntime-gpu" in names

    # 순수 CPU 환경: CPU 추론은 예상되는 동작이며, 사용자가 N 카드 라이브러리를 설치하도록 오도하지 않아야 함
    if hw == "none":
        print("[BSK_Tagger][진단] 본래 GPU/NPU 가속기가 감지되지 않음，")
        print("          현재 CPU 추론은 예상되는 동작이며 GPU 버전을 설치할 필요가 없습니다 onnxruntime。")
        if found:
            print(f"[BSK_Tagger][진단] 설치됨: {', '.join(found)}")
        print("[BSK_Tagger][진단] 가속기를 사용하려 할 때, 해당 드라이버/런타임 라이브러리가 설치되어 있고 프로세스가 보이도록 확인。")
        print("=" * 70)
        return

    if has_cpu and has_gpu:
        print("[BSK_Tagger][진단] onnxruntime(CPU) 및 onnxruntime-gpu 동시 설치 감지, 충돌 감지 → 먼저 CPU 버전 제거：")
        print("           pip uninstall -y onnxruntime")
    elif has_cpu and not has_gpu:
        print("[BSK_Tagger][진단] onnxruntime(CPU 버전)만 설치되어 있어 현재 가속기와 일치하지 않습니다. → 먼저 삭제하세요：")
        print("           pip uninstall -y onnxruntime")
    else:
        print("[BSK_Tagger][진단] 호스트 가속기 매칭되지 않은 onnxruntime 변형 설치되지 않음。")

    if hw == "nvidia":
        cuda_ver = _detect_cuda_version()
        print("[BSK_Tagger][진단] 본 기는 NVIDIA이며, 본 기 CUDA와 일치하는 설치 onnxruntime-gpu：")
        if cuda_ver and cuda_ver.startswith("12"):
            print("           pip install onnxruntime-gpu")
            print(f"           （호스트 탐지 CUDA {cuda_ver}，최신 onnxruntime-gpu 지원. CUDA 12）")
        elif cuda_ver and cuda_ver.startswith("11"):
            print('           pip install "onnxruntime-gpu==1.18.*"')
            print(f"           （호스트 탐지 CUDA {cuda_ver}，ORT≥1.19 CUDA 11을 지원하지 않습니다. 1.18.x）")
        else:
            print("           pip install onnxruntime-gpu")
            if not cuda_ver:
                print("           （js/video_studio/ 스크립트 디렉토리 확인, 플러그인 파일이 완전히 배포되었는지 확인하십시오.，")
                print("            다시 선택 onnxruntime-gpu 버전，")
                print("            자세한 내용 참조 https://onnxruntime.ai/docs/execution-providers/CUDA-EP.html）")
    elif hw == "amd":
        print("[BSK_Tagger][진단] AMD(ROCm) 기기입니다: 공식 onnxruntime-gpu wheel이 내장되어 있습니다. ROCm EP，")
        print("          먼저 GPU 버전을 설치하세요：")
        print("           pip install onnxruntime-gpu")
        print("           ORT가 ROCm EP를 선택하도록 환경 변수를 설정합니다(예:)：HIP_VISIBLE_DEVICES=0")
        print("           （loras 출력 디렉토리 획득 (환경 변수 덮어쓰기 지원)；")
        print("            자세한 내용 참조 https://onnxruntime.ai/docs/execution-providers/ROCm-EP.html）")
    elif hw == "ascend":
        print("[BSK_Tagger][진단] 현 기기: 텐센트 昇腾 NPU, 공식 PyPI onnxruntime 지원 안 함，")
        print("          Huawei Custom Edition onnxruntime (ACL Execution Provider 포함)를 사용해야 하며, CANN 도구 체인이 설치되어 있는지 확인하십시오.：")
        print("           - 참조 화웨이 문서 설치 일치 CANN 버전 맞춤 onnxruntime")
        print("           - CANN 런타임 라이브러리가 LD_LIBRARY_PATH(Linux)/PATH(Windows)에서 실행되고 현재 프로세스에서 접근 가능함을 확인")
        print("           - 상세 내용은 Bytedance onnxruntime 호환성 설명 참조（ACL EP）")
    print("[BSK_Tagger][진단] 설치 후 ComfyUI를 재시작하고 이 노드를 다시 실행하면 CPU 외 provider가 나타나면 활성화됩니다。")
    print("=" * 70)


def _ensure_session(tagger_type, wd14_model):
    key = (tagger_type, wd14_model)
    if key in _SESSIONS:
        print(f"[BSK_Tagger] 캐시 재사용 session ({tagger_type}/{wd14_model})，Skip model loading")
        return _SESSIONS[key]
    if ort is None:
        raise RuntimeError(
            "[BSK_Tagger] onnxruntime이 설치되지 않았으므로 추론할 수 없습니다. onnxruntime을 설치하십시오 "
            "onnxruntime-gpu 재시작 ComfyUI。"
        )
    onnx_path, tagmap_path = _ensure_model_paths(tagger_type, wd14_model)
    providers = _providers()
    print(f"[BSK_Tagger] 생성 ONNX session ({tagger_type}/{wd14_model})，providers={providers}")
    session = ort.InferenceSession(
        onnx_path, sess_options=_ort_session_options(), providers=providers)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    input_shape = session.get_inputs()[0].shape
    input_type = session.get_inputs()[0].type
    print(f"[BSK_Tagger] 모델 입력: name={input_name} shape={input_shape} type={input_type}")
    if tagger_type == "wd14":
        labels = _load_wd14_csv(tagmap_path)
    else:
        labels = _load_tag_mapping(tagmap_path)
    _SESSIONS[key] = (session, input_name, output_name, labels, input_shape)
    actual = session.get_providers()
    print(f"[BSK_Tagger] 준비 완료 ({tagger_type}/{wd14_model})，"
          f"실제 provider={actual[0]} | session.providers={actual}")
    if actual and actual[0] == "CPUExecutionProvider":
        # 실제로 CPU 추론 사용 중: 로컬 가속기 플랫폼에 따라 맞춤형 제안
        # （N 카드를 바인딩하지 않음, AMD/昇腾 사용자에게 N 카드 라이브러리 설치를 잘못 유도하지 않도록）
        print(f"[BSK_Tagger][진단] ⚠ 실제 우선 선택 provider={actual[0]}，가속기가 사용되지 않았으므로 속도가 현저히 느려짐。")
        _cpu_inference_advice(actual, providers)
    return _SESSIONS[key]


def _infer_size(session):
    """ONNX에서 shape를 입력하여 사각형의 변의 길이를 추론합니다 (NCHW/NHWC 및 동적 shape와 호환), 기본값으로 설정합니다. 448。

    모델 입력은 일반적으로 [N, C, H, W] 또는 [N, H, W, C]이며, 정사각형이어야 합니다. H==W。
    배치 차원은 1/None이고, 채널 차원은 모두 3입니다. 1과 3를 제외하고 남은 정수 차원의 최대값을 측면으로 사용합니다.，
    채널 차원(3)을 공간 변의 길이로 잘못 해석하지 않도록 합니다 (Conv가 수신되는 {3,3} 실패）。
    """
    try:
        shape = session.get_inputs()[0].shape
        candidates = [int(v) for v in shape if isinstance(v, int) and v not in (1, 3)]
        if candidates:
            side = max(candidates)
            return (side, side)
    except Exception:  # noqa: BLE001
        pass
    return (448, 448)


# ---------------------------------------------------------------------------
# 昇腾 CANN 서브 프로세스 워커 (torch_npu와 동일 프로세스에서 ACL을 두고 경쟁하는 것을 방지, 참조) opendatalab/MinerU#2547）
# ---------------------------------------------------------------------------
def _torch_npu_present():
    """best-effort 현재 프로세스가 torch_npu를 로드했는지 감지합니다. 하위 프로세스를 실행할지 여부를 결정하는 데 사용됩니다. worker。"""
    try:
        import torch
    except Exception:  # noqa: BLE001
        return False
    try:
        return getattr(torch, "npu", None) is not None and torch.npu.is_available()
    except Exception:  # noqa: BLE001
        return False


_cann_via_worker_cache = None


def _cann_via_worker():
    """CANN 사용 가능; 프로세스 로드됨 torch_npu → 독립적인 자식 프로세스 worker를 통해서만 사용 가능 NPU。"""
    global _cann_via_worker_cache
    if _cann_via_worker_cache is None:
        try:
            avail = "CANNExecutionProvider" in set(ort.get_available_providers())
            _cann_via_worker_cache = avail and _torch_npu_present()
        except Exception:  # noqa: BLE001
            _cann_via_worker_cache = False
    return _cann_via_worker_cache


class _CANNWorker:
    """자식 프로세스: torch-free 환경에서 onnxruntime-cann을 사용하여 ONNX 추론 실행。"""

    def __init__(self):
        self.script = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "cltagger_cann_worker.py"
        )
        if not os.path.isfile(self.script):
            raise RuntimeError(f"[BSK_Tagger][CANN worker] worker 스크립트 찾을 수 없음: {self.script}")
        self._lock = threading.Lock()
        self._proc = subprocess.Popen(
            [sys.executable, self.script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,
            text=True,
            bufsize=1,
            encoding="utf-8",
        )
        # 워커 핸드셰이크 대기 (워커 시작 후 출력 {"type":"ready"}）；프로세스가 조기에 종료되면 readline이 반환 ""
        ready = self._proc.stdout.readline()
        if not ready:
            raise RuntimeError("[BSK_Tagger][CANN worker] 자식 프로세스 시작 실패 (ready 핸드셰이크 수신 안 됨)）")
        # 종료 시 자식 프로세스 정리, 유령 프로세스 남용 방지 NPU
        atexit.register(self._cleanup)

    def _cleanup(self):
        if self._proc.poll() is None:
            self._proc.terminate()

    def _read_line(self):
        line = self._proc.stdout.readline()
        if not line:
            raise RuntimeError("[BSK_Tagger][CANN worker] 자식 프로세스 연결 중단")
        return line

    def infer(self, tagger_type, wd14_model, model_path, pil_image, max_resolution):
        buf = io.BytesIO()
        pil_image.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        req = {
            "type": "infer",
            "tagger_type": tagger_type,
            "wd14_model": wd14_model,
            "model_path": model_path,
            "max_resolution": max_resolution,
            "image_b64": img_b64,
        }
        with self._lock:
            self._proc.stdin.write(json.dumps(req) + "\n")
            self._proc.stdin.flush()
            resp = json.loads(self._read_line())
        if not resp.get("ok"):
            raise RuntimeError(f"[BSK_Tagger][CANN worker] {resp.get('error')}")
        print(f"[BSK_Tagger] CANN worker 추론 provider={resp.get('provider')} "
              f"耗시={resp.get('elapsed'):.2f}s")
        return resp["scores"]


_cann_worker_singleton = None
_cann_worker_lock = threading.Lock()


def _get_cann_worker():
    """CANN 워커 (싱글톤) 반환; 사용 불가 시 None 반환 (호출자 롤백 CPU）。"""
    global _cann_worker_singleton
    if not _cann_via_worker():
        return None
    with _cann_worker_lock:
        if _cann_worker_singleton is None:
            try:
                _cann_worker_singleton = _CANNWorker()
            except Exception as e:  # noqa: BLE001
                print(f"[BSK_Tagger][CANN worker 실행 실패, 프로세스 내로 진행 CPU] {e}")
                _cann_worker_singleton = False  # 마킹 실패, 반복 시도 방지
        return _cann_worker_singleton if _cann_worker_singleton else None


def _postprocess(scores, tagger_type, labels, gen_threshold, char_threshold,
                sort_mode, exclude_meta, exclude_rating, exclude_quality,
                replace_underscore, trailing_comma, exclude_tags):
    """모델 출력 점수를 후처리합니다(내부 경로와 일치; 점수는 워커가 반환하는 (1,N) 중첩 리스트입니다).）。"""
    outputs = np.asarray(scores, dtype=np.float32)
    if np.isnan(outputs).any() or np.isinf(outputs).any():
        outputs = np.nan_to_num(outputs, nan=0.0, posinf=1.0, neginf=0.0)
    if tagger_type == "wd14":
        # WD14 ONNX 출력은 이미 확률([0,1])이며, 참조 구현과 일치: 수정하지 않음 sigmoid；
        # outputs[0] 확률 벡터 (N,), 직접 전달 (구현 참조) probs=run(...)[0] 후 zip(tags, probs[0])）
        tags, gi, ci = labels
        return _get_wd14_tags(outputs[0], tags, gi, ci, gen_threshold, char_threshold,
                              exclude_tags, replace_underscore, trailing_comma)
    probs = _stable_sigmoid(outputs[0])
    predictions = _get_cltags(probs, labels, gen_threshold, char_threshold)
    if sort_mode == "novelai":
        return _format_novelai(predictions)
    return _format_default(predictions, exclude_meta, exclude_rating, exclude_quality)


def _tag_image(pil_image, tagger_type, wd14_model, gen_threshold, char_threshold,
               sort_mode, exclude_meta, exclude_rating, exclude_quality,
               replace_underscore, trailing_comma,
               exclude_tags, max_resolution):
    onnx_path, tagmap_path = _ensure_model_paths(tagger_type, wd14_model)
    # CANN 자식 프로세스 경로: 본 프로세스에서 torch_npu 로드 중 in-process CANN이 ACL 실패를 쟁탈함，
    # 독립적인 서브 프로세스 worker(torch-free)를 통해 사용해야 함 NPU。
    worker = _get_cann_worker()
    if worker is not None:
        try:
            scores = worker.infer(tagger_type, wd14_model, onnx_path, pil_image, max_resolution)
            labels = (_load_wd14_csv(tagmap_path) if tagger_type == "wd14"
                      else _load_tag_mapping(tagmap_path))
            return _postprocess(scores, tagger_type, labels, gen_threshold, char_threshold,
                                sort_mode, exclude_meta, exclude_rating, exclude_quality,
                                replace_underscore, trailing_comma, exclude_tags)
        except Exception as e:  # noqa: BLE001
            print(f"[BSK_Tagger][CANN worker 추론 실패, 되돌리기 프로세스 내 CPU] {e}")
    # —— 프로세스 내 경로(CANN 없음 또는 워커 사용/실패 시) CPU）——
    session, input_name, output_name, labels, input_shape = _ensure_session(tagger_type, wd14_model)
    size = _infer_size(session)
    if tagger_type == "wd14":
        tensor = _preprocess_wd14(pil_image, size, max_resolution)
    else:
        tensor = _preprocess_cltagger(pil_image, size)
    print(f"[BSK_Tagger] 추론 전: 실제 텐서 shape={tensor.shape} dtype={tensor.dtype} "
          f"min={tensor.min():.3f} max={tensor.max():.3f} | 모델 기대값 shape={input_shape}")
    # 배치 유효성 검사: 모델이 배치 차원을 원하지 않는 경우（rank=3），그렇지 않으면 필수입니다 rank=4
    # 입력 차원 수 (항상 4 또는 3) - 개수를 나타내지 않음"비 None 차원"——ダイナミック shape [None,3,None,None]
    # 잘못 계산될 것 rank=1，추가 오류 squeeze/expand。
    expected_rank = len(input_shape) if input_shape else 4
    if tensor.ndim != expected_rank:
        if tensor.ndim == 4 and expected_rank == 3:
            tensor = np.squeeze(tensor, axis=0)
            print(f"[BSK_Tagger] 배치 제거됨 -> shape={tensor.shape}")
        elif tensor.ndim == 3 and expected_rank == 4:
            tensor = np.expand_dims(tensor, axis=0)
            print(f"[BSK_Tagger] 배치 위 보완됨 -> shape={tensor.shape}")
        else:
            raise ValueError(
                f"[BSK_Tagger] 텐서 rank({tensor.ndim}) 모델 기대값 rank({expected_rank}) 정렬 불가: "
                f"실제 {tensor.shape} vs 기대 {input_shape}")
    _t0 = time.time()
    outputs = session.run([output_name], {input_name: tensor})[0]
    print(f"[BSK_Tagger] 추론 시간 {time.time() - _t0:.2f}s | provider={session.get_providers()[0]}")
    if np.isnan(outputs).any() or np.isinf(outputs).any():
        outputs = np.nan_to_num(outputs, nan=0.0, posinf=1.0, neginf=0.0)
    if tagger_type == "wd14":
        tags, gi, ci = labels
        return _get_wd14_tags(outputs[0], tags, gi, ci, gen_threshold, char_threshold,
                              exclude_tags, replace_underscore, trailing_comma)
    probs = _stable_sigmoid(outputs[0])
    predictions = _get_cltags(probs, labels, gen_threshold, char_threshold)
    if sort_mode == "novelai":
        return _format_novelai(predictions)
    return _format_default(predictions, exclude_meta, exclude_rating, exclude_quality)


def _apply_trigger(tags, trigger_word):
    tw = (trigger_word or "").strip()
    if not tw:
        return tags
    if not tw.endswith(","):
        tw = tw + ","
    return f"{tw} {tags}" if tags else tw


def _batch_directory(directory, tagger_type, wd14_model, gen_threshold, char_threshold,
                     sort_mode, exclude_meta, exclude_rating, exclude_quality,
                     replace_underscore, trailing_comma,
                     exclude_tags, max_resolution, trigger_word):
    if not os.path.isdir(directory):
        print(f"[BSK_Tagger] 배치 디렉토리 없음: {directory}")
        return 0
    files = sorted(
        f for f in os.listdir(directory)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
    )
    if not files:
        print(f"[BSK_Tagger] 폴더 내에 이미지가 없음: {directory}")
        return 0

    print(f"[BSK_Tagger] 배치 태깅 {len(files)} 장 -> {directory}")
    count = 0
    for name in files:
        src = os.path.join(directory, name)
        txt = os.path.join(directory, os.path.splitext(name)[0] + ".txt")
        try:
            with Image.open(src) as img:
                tags = _tag_image(img, tagger_type, wd14_model, gen_threshold,
                                  char_threshold, sort_mode, exclude_meta,
                                  exclude_rating, exclude_quality,
                                  replace_underscore, trailing_comma, exclude_tags,
                                  max_resolution)
            with open(txt, "w", encoding="utf-8") as f:
                f.write(_apply_trigger(tags, trigger_word))
            count += 1
            print(f"[BSK_Tagger] [{count}/{len(files)}] {name} -> {os.path.basename(txt)}")
        except Exception as e:  # noqa: BLE001
            print(f"[BSK_Tagger] 처리 실패 {name}: {e}")
    print(f"[BSK_Tagger] 배치 완료: {count}/{len(files)}")
    return count


# ---------------------------------------------------------------------------
# ComfyUI 노드
# ---------------------------------------------------------------------------
class BSK_Tagger:
    """길이 검사 실패 또는 즉시 멈춤. X-Script-Name은 마운트 전용 접미사이며, 저희가 통일하여 지정합니다.。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "tagger_type": (["cltagger", "wd14"], {"default": "cltagger"}),
                "wd14_model": (list(WD14_MODELS.keys()), {"default": "wd-vit-tagger-v3"}),
                "general_threshold": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.05}),
                "character_threshold": ("FLOAT", {"default": 0.60, "min": 0.0, "max": 1.0, "step": 0.05}),
                "sort_mode": (["default", "novelai"], {"default": "default"}),
                "exclude_meta": ("BOOLEAN", {"default": True}),
                "exclude_rating": ("BOOLEAN", {"default": True}),
                "exclude_quality": ("BOOLEAN", {"default": True}),
                "replace_underscore": ("BOOLEAN", {"default": False}),
                "trailing_comma": ("BOOLEAN", {"default": False}),
                "exclude_tags": ("STRING", {"default": "", "multiline": False}),
                "max_resolution": ("INT", {"default": 1280, "min": 64, "max": 4096, "step": 64}),
                "trigger_word": ("STRING", {"default": "", "multiline": False}),
                "batch_dir": ("STRING", {"default": "", "multiline": False}),
                "enabled": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("tags",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "execute"
    OUTPUT_NODE = True
    CATEGORY = "bsk_UI/Tagging"
    DESCRIPTION = ("로컬 ONNX 이미지 라벨링, CLTagger (cella110n) 지원 WD14 "
                   "(SmilingWolf wd-tagger-v3 시리즈). 두 종류의 태그링기 사전 처리/태그 형식은 엄격하게 구분；"
                   "초기 모델은 HF 이미지에서 자동으로 다운로드。")

    def execute(self, image, tagger_type="cltagger", wd14_model="wd-vit-tagger-v3",
                general_threshold=0.35, character_threshold=0.60,
                sort_mode="default", exclude_meta=True, exclude_rating=True,
                exclude_quality=True, replace_underscore=False,
                trailing_comma=False, exclude_tags="", max_resolution=1280,
                trigger_word="",
                batch_dir="", enabled=True):
        # 배치 모드
        if batch_dir and os.path.isdir(batch_dir):
            _batch_directory(batch_dir, tagger_type, wd14_model, general_threshold,
                             character_threshold, sort_mode, exclude_meta,
                             exclude_rating, exclude_quality,
                             replace_underscore, trailing_comma, exclude_tags,
                             max_resolution, trigger_word)
            return {"ui": {"tags": ["[배치 모드] 콘솔 출력 참조。"]}, "result": ([""],)}

        if not enabled or image is None:
            return {"ui": {"tags": [""]}, "result": ([""],)}

        try:
            tensor = np.array(image * 255, dtype=np.uint8)
            if tensor.size == 0 or tensor.shape[0] == 0:
                return {"ui": {"tags": [""]}, "result": ([""],)}

            tags_list = []
            for i in range(tensor.shape[0]):
                pil = Image.fromarray(tensor[i])
                tags = _tag_image(pil, tagger_type, wd14_model, general_threshold,
                                  character_threshold, sort_mode, exclude_meta,
                                  exclude_rating, exclude_quality,
                                  replace_underscore, trailing_comma, exclude_tags,
                                  max_resolution)
                tags_list.append(_apply_trigger(tags, trigger_word))
            return {"ui": {"tags": tags_list}, "result": (tags_list,)}
        except Exception as e:  # noqa: BLE001
            print(f"[BSK_Tagger] 태깅 오류: {e}")
            return {"ui": {"tags": [str(e)]}, "result": ([""],)}


NODE_CLASS_MAPPINGS = {
    "BSK_Tagger": BSK_Tagger,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BSK_Tagger": "BSK 이미지 라벨러",
}
