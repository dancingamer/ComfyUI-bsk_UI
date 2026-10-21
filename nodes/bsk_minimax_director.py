# SPDX-License-Identifier: AGPL-3.0-or-later
"""
bsk_MiniMaxDirector —— MiniMax H3 직접 작성 배치 노드。

설계 원칙 (MD/MiniMaxH3_Director_기술 원리 및 BSK 적응 분석 보고서 참조).md）：
- 100% ComfyUI 핵심 공식 노드를 재사용하여 기본 기능을 완료：
    * comfy_extras.nodes_minimax_h3.MiniMaxH3ImageToVideo（t2v / i2v / fl2v Condition editor + 키프레임 마운트）
    * comfy_extras.nodes_minimax_h3.MiniMaxH3ReferenceToVideo（임의의 위치 참조 이미지/비디오 + <Picture i>/<Video k> 레이블）
    * comfy_extras.nodes_minimax_h3.MiniMaxH3SigmaShift（shift clone）
    * nodes.KSampler / nodes.VAEDecode（샘플링 + 비디오 VAE 디코딩, 구식 인스턴스 메서드）
    * comfy_extras.nodes_lt.LTXVSeparateAVLatent（오디오 및 비디오 잠재력을 분리, 새로운 스타일 execute）
    * comfy_extras.nodes_audio.VAEDecodeAudio（오디오 VAE 디코딩, 새로운 스타일 실행, audio_vae 선택 사항）
- 이 노드는 단지"타임라인 JSON 오케스트레이션 레이어"：프론트엔드 buildMiniMaxTimelineData에서 전달된
  timeline_data（사용자 계약 v6), 모드에 따라 공식 노드를 사용하여 샘플링 실행+디코딩, 출력 images / audio list。

듀얼 모드 분기 (MiniMax 하위 실제 능력에 대응)）：
- mode == 'concat'（모드 A·시작 및 끝 프레임 연결）
      다수 그룹 헤드 및 꼬리 반복: 각 그룹별 판정 task_key(t2v/i2v/fl2v)，
      MiniMaxH3ImageToVideo(first/last frame 쌍의 하드 앵커)를 실행합니다. 프롬프트만 입력합니다. t2v。
      데이터에서 td["groups"]：[{task_key, firstFrame, lastFrame, frames, prompt, prevLastFrame}]。
- mode == 'ref'（모드 B: 여러 참조 자료 / 이전 호환）
      《MiniMaxH3ReferenceToVideo》（이미지/비디오/오디오 참조, 프롬프트 내 포함 <Picture i>/<Video k>/<Audio j> 레이블）。
      데이터에서 td["references"]（구형) 또는 td["ref_images"]/["ref_videos"]/["ref_video_audios"]/["ref_audios"]（새）。
- mode == 'fl2v'/'i2v'/'t2v'（단일 단계 퇴화, concat 한 그룹과 동일）

프론트엔드 timeline_data 계약 (v6, v5의 ref 필드 후방향 호환)）：
{
  "version": 6,
  "width": int, "height": int, "frameRate": int,
  "mode": "concat" | "ref" | "t2v" | "i2v" | "fl2v",
  // 모드A·concat：
  "groups": [ {"task_key":"fl2v"|"i2v"|"t2v", "firstFrame":"path"|null,
               "lastFrame":"path"|null, "prevLastFrame":"path"|null,
               "frames": int, "prompt": str} ],
  // 모드 B·ref（새）：
  "ref_image_size": "match"|"max",
  "ref_images": ["path", ...] (≤9), "ref_videos": ["path", ...] (≤3),
  "ref_video_audios": ["path", ...] (≤3, 숫자 접미사 매칭), "ref_audios": ["path", ...] (≤3),
  "prompt": "포함 <Picture i>/<Video k>/<Audio j> 레이블",
  // 샘플링 (선택 사항, 이전 버전의 상위 층 스칼라 지원; 참조 _build_sampling_stages）：
  //   sampling.stage1 = {samplerName, scheduler, steps, denoise, shiftVideo, shiftAudio, sigmas}
  //   sampling.stage2 = {enabled, mode:'independent'|'split', upscale, scaleBy, method, ...동일 stage1}
  //   mode='split' → 두 구간 공용 총 sigma 시퀀스 (총 단계 수)=두 단계의 합, 첫 번째 단계 수로 자동으로 분할；
  //   upscale=true → 샘플링 전만 업스케일링, 비디오 latent (조건 내 키프레임/참조 시각 latent 동일 비율 동기화)）。
  // 오래된 v5 호환：
  "global": {"prompt": str},
  "firstFrame": "path"|null, "lastFrame": "path"|null,
  "references": [ {"kind":"image"|"video","file":"path","slot":int,"tag":str} ]
}
"""

from __future__ import annotations

import json
import os
import re

import torch

import comfy.model_management as model_management

from comfy_extras.nodes_minimax_h3 import (
    MiniMaxH3ImageToVideo,
    MiniMaxH3ReferenceToVideo,
    MiniMaxH3SigmaShift,
)
# BSK 단일 프레임 내장 노드 (nodes_minimax_h3.py에서 공식 노드 파생, 지원) length=1 단 프레임 출력）。
# bsk_minimax_h3_frame.py 파일 헤더의 저작권 및 개선 사항 참조。
# 주의: 이 파일은 루트 __init__.py를 importlib.util로 절대 경로로 로드합니다 (모듈 이름
# "bsk_minimax_director"），동일 디렉토리의 bsk_minimax_h3_frame.py가 자동으로 입력되지 않음 sys.path，
# 배열이 ref_images보다 짧을 수 있습니다 (프론트엔드 trimNullTail 기능 사용). "No module named" 오류。
import importlib.util as _ilu
import os as _os
_bsk_frame_path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "bsk_minimax_h3_frame.py")
_bsk_frame_spec = _ilu.spec_from_file_location("bsk_minimax_h3_frame", _bsk_frame_path)
_bsk_frame = _ilu.module_from_spec(_bsk_frame_spec)
_bsk_frame_spec.loader.exec_module(_bsk_frame)
BskImageToVideo = _bsk_frame.BskMiniMaxH3ImageToVideo
BskRefToVideo = _bsk_frame.BskMiniMaxH3ReferenceToVideo
BSK_LOW_FRAME_FRAMES = _bsk_frame.BSK_LOW_FRAME_FRAMES
import base64
import io as pyio
import time

from PIL import Image, ImageOps

from nodes import VAEDecode
from comfy_extras.nodes_audio import VAEDecodeAudio
from comfy_extras.nodes_custom_sampler import (
    RandomNoise,
    BasicGuider,
    KSamplerSelect,
    BasicScheduler,
    SamplerCustomAdvanced,
)
import comfy.patcher_extension
from server import PromptServer


# 전체 시간선 모드 / 작업 유형 한국화된 열거형 → 노드 내부 영어 유형
TASK_TYPE_MAP = {
    "t2v": "text_to_video",
    "text_to_video": "text_to_video",
    "텍스트에서 비디오": "text_to_video",
    "i2v": "image_to_video",
    "image_to_video": "image_to_video",
    "비디오 생성": "image_to_video",
    "fl2v": "first_last_to_video",
    "first_last_to_video": "first_last_to_video",
    "시작/종료 프레임 비디오": "first_last_to_video",
    "ref": "reference_to_video",
    "reference_to_video": "reference_to_video",
    "참조 생 비디오": "reference_to_video",
}


def _resolve_task_type(raw: str) -> str:
    if not raw:
        return "first_last_to_video"
    raw = str(raw).strip().lower()
    head = raw.split(" ")[0].split("—")[0].strip()
    for k, v in TASK_TYPE_MAP.items():
        if head == k or raw.startswith(k) or k in raw:
            return v
    return "first_last_to_video"


# ---------- 상수 제약 조건 (공식과 일치, 실제 테스트를 통해 확인, 오해의 소지가 없습니다)）----------
MIN_FL2V_FRAMES = 5          # fl2v 최소 프레임 = 5
MIN_I2V_FRAMES = 9           # i2v 최소 프레임 = 9
MIN_T2V_FRAMES = 5           # t2v 최소 프레임 = 5
MIN_DIM = 32                 # 크기 하한선
MAX_DIM = 8192               # 크기 상한선
DIM_STEP = 32                # 크기 step
# 단일 프레임/초단 프레임 화이트리스트: 비디오 모드에서 강제 ≥MIN_*，하지만 이 컬렉션 내 프레임 수는 사용됨"단 프레임 출력"（이미지 시퀀스 패턴），
# 직접 전달하여 공식 노드에서 temporal_shape([length]) 참여, 원본 지원 length=1/5，그리드 정렬 안 함。
LOW_FRAME_FRAMES = (1, 5)


# fl2v/i2v 하드 락 프롬프트 접두사 (참조 플러그인 복제) fl2v_timeline.py reinforce_fl2v_prompt）。
# 시작 및 종료 프레임을 프롬프트 바디 앞에 단단히 고정하여 문맥에서 잘리지 않도록 한다。
FLF_PROMPT_PREFIX = "[First frame: the start of the video.] [Last frame: the end of the video.] "
FLF_PROMPT_SUFFIX = ""
I2V_PROMPT_PREFIX = "[First frame: the start of the video.] "
I2V_PROMPT_SUFFIX = ""


def reinforce_fl2v_prompt(prompt: str, has_end_frame: bool) -> str:
    """복제 참조 플러그인 fl2v_timeline.py L225 reinforce_fl2v_prompt: 시작/종료 프레임은 body 이전에 단단히 잠금 해제합니다.。

    has_end_frame=True  → 프레임 시작/종료 모드 (fl2v), FLF 접두사 사용；
    has_end_frame=False → 프론트엔드 형식 워크플로우에서 패널 설정, 노드。
    반환 패키징된 prompt。
    """
    prompt = (prompt or "").strip()
    if has_end_frame:
        if not prompt.startswith(FLF_PROMPT_PREFIX):
            prompt = FLF_PROMPT_PREFIX + prompt
        if FLF_PROMPT_SUFFIX and not prompt.endswith(FLF_PROMPT_SUFFIX):
            prompt = prompt + FLF_PROMPT_SUFFIX
    else:
        if not prompt.startswith(I2V_PROMPT_PREFIX):
            prompt = I2V_PROMPT_PREFIX + prompt
        if I2V_PROMPT_SUFFIX and not prompt.endswith(I2V_PROMPT_SUFFIX):
            prompt = prompt + I2V_PROMPT_SUFFIX
    return prompt


def _align_frames(length: int, minimum: int) -> int:
    """공식 격자에 따라 총 프레임 수 맞춤：>= minimum 그리고 17k+5（MiniMax H3 latent 시간 차원 요구）。

    공식 `align_frame_count`：while n%17 != 5: n+=1，최소 5。
    """
    if length < minimum:
        length = minimum
    while length % 17 != 5:
        length += 1
    return length


def _align_dim(v: int) -> int:
    v = int(round(v))
    v = max(MIN_DIM, min(MAX_DIM, v))
    return max(MIN_DIM, (v // DIM_STEP) * DIM_STEP)


def _parse_sigma_list(raw):
    """사용자 정의 sigma 입력 파싱 → torch.FloatTensor；유효하지 않음/빈 경우 반환 None。

    쉼표/세미콜론/공백/새 줄로 구분된 십진 부동 소수점 숫자를 받습니다 (예 "1.2,0.9,0.5,0.2,0"）。
    약속 (공식 Sigmas 입력과 동일)）：
    - 숫자는**절대 sigma**（H3 shift=12 시간 0.03~14.6），퍼센트가 아님；
    - 내림차순으로 자동 정렬 (샘플링 단일성 수렴 보장); 음수/NaN/Inf는 모두 유효하지 않음；
    - 항목이 0이 아니면 자동으로 0으로 보완합니다(그렇지 않으면 샘플링이 절반 멈추고 화면에 노이즈가 남아 있습니다).）；
    - 최소 2개의 유효한 값이 유효한 것으로 간주 (1개의 값 + 0 채우기만 한 번의 노이즈 제거）。
    """
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)):
        parts = [str(x) for x in raw]
    else:
        s = str(raw).strip()
        if not s:
            return None
        parts = [p for p in re.split(r"[,;\s]+", s) if p]
    vals = []
    for p in parts:
        try:
            v = float(p)
        except (TypeError, ValueError):
            return None
        if v != v or v in (float("inf"), float("-inf")) or v < 0:
            return None
        vals.append(v)
    if len(vals) < 2:
        return None
    vals.sort(reverse=True)
    if vals[-1] != 0.0:
        vals.append(0.0)
    return torch.tensor(vals, dtype=torch.float32)


def _build_sigmas(model_shifted, scheduler, steps, denoise, raw_sigmas):
    """샘플링 시그마 시퀀스 생성: 사용자 지정 시그마 우선, 그렇지 않으면 스케줄러/단계/노이즈 계산에 따라。

    프론트엔드 ‘사용자 정의 sigma vs 스케줄러’와 동일한 의미론적 일관성을 유지합니다: raw_sigmas가 유효하면 무시합니다. scheduler/steps/denoise。
    torch.FloatTensor 반환; 설정 무효（denoise<=0 완료) None 반환(호출자 건너띔 해당 단계)）。
    """
    custom = _parse_sigma_list(raw_sigmas)
    if custom is not None:
        print("[bsk_MiniMaxDirector][sampling] 사용자 정의 sigma: "
              f"{[round(float(x), 4) for x in custom.tolist()]}", flush=True)
        return custom
    try:
        d = float(denoise if denoise is not None else 1.0)
    except (TypeError, ValueError):
        d = 1.0
    d = max(0.0, min(1.0, d))
    if d <= 0.0:
        return None
    n = max(1, int(steps or 20))
    sig = _out(BasicScheduler.execute(model_shifted, str(scheduler or "simple"), n, d))[0]
    if sig is None or len(sig) < 2:
        return None
    return sig


def _align_latent_even(v) -> int:
    """latent Align to even number（H3 DiT patch=(1,2,2)，홀수 차원 patchify reshape는 직접 멈춥니다）。"""
    v = int(v)
    return max(2, v - (v % 2))


def _upscale_video_latent(video, scale_by, method):
    """비디오 latent [B,24,T,h,w]의 공간 크기를 scale_by로 확대(h/w는 짝수, 상한) MAX_DIM//16）。

    단지 "비디오" 컴포넌트에만 사용: H3 오디오 latent는 자체적인 형태를 가지며, 오디오를 확대하면 오디오 트랙이 섞이고, 다운스트림 형태 검사 실패。
    """
    if video is None or scale_by is None:
        return video
    try:
        s = float(scale_by)
    except (TypeError, ValueError):
        return video
    if s <= 0.0 or abs(s - 1.0) < 1e-6:
        return video
    h, w = int(video.shape[-2]), int(video.shape[-1])
    max_lat = max(2, MAX_DIM // 16)
    nh = min(max_lat, _align_latent_even(round(h * s)))
    nw = min(max_lat, _align_latent_even(round(w * s)))
    if nh == h and nw == w:
        return video
    import comfy.utils
    out = comfy.utils.common_upscale(video.float(), nw, nh, str(method or "bilinear"), "disabled")
    print(f"[bsk_MiniMaxDirector][upscale] 비디오 latent {w}x{h} -> {nw}x{nh} "
          f"(x{s:g}, {method or 'bilinear'})", flush=True)
    return out.to(dtype=video.dtype)


def _upscale_av_latent(latent, scale_by, method):
    """오른쪽 상단 닫기 버튼 - 오른쪽 상단에 밀착**비디오**컴포넌트, 오디오 컴포넌트(및 오디오 사이드 노이즈 마스크)는 원본대로 유지합니다。

    H3 의 latent 은 NestedTensor((video, audio))입니다. nested 하지 않을 경우 전체 확대(일반 비디오 모델 의미)로 퇴화합니다.）。
    """
    if not isinstance(latent, dict):
        return latent
    samples = latent.get("samples")
    out = dict(latent)
    if samples is not None and getattr(samples, "is_nested", False):
        import comfy.nested_tensor
        streams = list(samples.unbind())
        if streams:
            streams[0] = _upscale_video_latent(streams[0], scale_by, method)
        out["samples"] = comfy.nested_tensor.NestedTensor(streams)
    else:
        out["samples"] = _upscale_video_latent(samples, scale_by, method)
    nm = latent.get("noise_mask")
    if nm is not None and getattr(nm, "is_nested", False):
        import comfy.nested_tensor
        mstreams = list(nm.unbind())
        if mstreams:
            mstreams[0] = _upscale_video_latent(mstreams[0], scale_by, method)
        out["noise_mask"] = comfy.nested_tensor.NestedTensor(mstreams)
    return out


def _upscale_conditioning_visuals(positive, scale_by, method):
    """conditioning에서 minimax_keyframes / minimax_refs의**시각** latent 동일 배율 확대。

    동기화가 필요한 이유: 이러한 키프레임/참조 latent는 첫 번째 캔버스 인코딩으로 생성됩니다; latent를 확장할 때 해당 값에 상관없이，
    앵커/참조와 확대된 캔버스 비율이 일치하지 않습니다 (구도 오류, 얼굴 왜곡). 커딩 업스케일의 방법을 따릅니다.。
    단순 시각 측면만 사용: 참조 블록의 latent와 latent_h/latent_w(모델은 후자를 기준으로 RoPE 행 레이아웃 계산)）；
    audio_latent / ref_audio_t / resolved_frame_index / latent_t 변경 없음。
    """
    if positive is None or scale_by is None:
        return positive
    if not isinstance(positive, (list, tuple)):
        return positive
    try:
        if abs(float(scale_by) - 1.0) < 1e-6:
            return positive
    except (TypeError, ValueError):
        return positive
    # conditioning 형상과 같음 [[cond_tensor, {값...}], ...]；값 딕셔너리 / 목록이 여러 번 공유될 수 있음 → memo 반복 증폭 방지
    done = {}
    out = []
    for pair in positive:
        try:
            cond_t, values = pair[0], pair[1]
        except (TypeError, IndexError, KeyError):
            out.append(pair)
            continue
        if not isinstance(values, dict):
            out.append(pair)
            continue
        new_values = dict(values)
        for key in ("minimax_keyframes", "minimax_refs"):
            items = values.get(key)
            if not isinstance(items, list) or id(items) in done:
                continue
            done[id(items)] = True
            new_items = []
            for it in items:
                if not isinstance(it, dict):
                    new_items.append(it)
                    continue
                new_it = dict(it)
                if isinstance(it.get("latent"), torch.Tensor):
                    new_it["latent"] = _upscale_video_latent(it["latent"], scale_by, method)
                for mk in ("latent_h", "latent_w"):
                    if isinstance(it.get(mk), int):
                        new_it[mk] = _align_latent_even(round(it[mk] * float(scale_by)))
                new_items.append(new_it)
            new_values[key] = new_items
        out.append([cond_t, new_values])
    return out


def _normalize_crop(crop, iw, ih):
    """저장 crop={x,y,w,h} 원래 이미지 픽셀 정수 좌표로 정규화 (원점 왼쪽 상단)）。

    - 비율 모드: x/y/w/h 모두 [0,1] 범위 내에서 (원본 이미지 크기에 상대적), iw/ih로 곱하여 픽셀 값으로 되돌림。
      （프론트엔드 vs_crop.js에 저장된 비율 좌표이므로 백엔드에서 여기에서 일관되게 변환。）
    - 픽셀 모드: x/y/w/h는 원래 이미지 픽셀 값 (오래된 계약 호환)）。
    유효 범위로 자르기 (불법/오류 시)；w/h<=0 None 반환 (호출자는 잘라내기를 무시)）。
    """
    if not crop or not isinstance(crop, dict):
        return None
    try:
        x = float(crop["x"]); y = float(crop["y"])
        w = float(crop["w"]); h = float(crop["h"])
    except (KeyError, TypeError, ValueError):
        return None
    if w <= 0 or h <= 0:
        return None
    # 비율 판별: 가로 세로 중 하나 <=1 비율 좌표로 간주 (실제 <1px 잘라내기는 무의미하며, 이 경계를 무시합니다）
    if w <= 1.0 or h <= 1.0:
        x = x * iw; y = y * ih; w = w * iw; h = h * ih
    x = int(round(x)); y = int(round(y)); w = int(round(w)); h = int(round(h))
    x = max(0, min(x, iw - 1)); y = max(0, min(y, ih - 1))
    w = max(1, min(w, iw - x)); h = max(1, min(h, ih - y))
    return (x, y, w, h)


def _apply_crop(img: "Image.Image", crop) -> "Image.Image":
    """누르다 crop={x,y,w,h} PIL 이미지 잘라내기; 좌표 비율 지원(0~1)또는 원본 픽셀, 원점 왼쪽 상단。

    크롭 후 호출하는 측에서도 32배수로 정렬하여 리사이즈하므로, 여기서는 정수 나누기 처리가 필요하지 않습니다.。
    """
    iw, ih = img.size
    rc = _normalize_crop(crop, iw, ih)
    if rc is None:
        return img
    x, y, w, h = rc
    return img.crop((x, y, x + w, y + h))


def _apply_crop_tensor(t: torch.Tensor, crop) -> torch.Tensor:
    """For [B,H,W,3] / [H,W,3] tensors by crop={x,y,w,h} 크롭; 좌표 지원 비율(0~1)또는。"""
    if t.dim() == 3:
        t = t.unsqueeze(0)
    _, H, W, _ = t.shape
    rc = _normalize_crop(crop, W, H)
    if rc is None:
        return t
    x, y, w, h = rc
    return t[:, y:y + h, x:x + w, :]


def _limit_pixels(t: torch.Tensor, max_pixels: int) -> torch.Tensor:
    """[B,H,W,3] / [H,W,3] 텐서의 등방화 피라미드 스케일링을 픽셀 양으로 조정합니다. ≤ max_pixels（단순 감소）。

    다중 참조 ref 이미지 프리 스케일링에 사용: 공식 ref 노드는 ref 토큰을 UNET에 단계별 샘플링 시 주입합니다.，
    이미지 크기가 클수록 각 단계가 느려집니다 (승수 효과). 사용자가 제한을 설정하면 초과된 이미지는 이 단계에서 등비율로 다운샘플링됩니다.
    （lanczos，공식의 _resize 알고리즘과 동일), 공식 노드의 match 스케일 조정도 동일 min(1.0,...)
    하위만 감소, 두 레벨 겹침 후 최종 크기는 엄격하게 초과하지 않음 max_pixels。
    """
    if t is None or max_pixels <= 0:
        return t
    if t.dim() == 3:
        t = t.unsqueeze(0)
    if t.dim() != 4:
        return t
    _, H, W, _ = t.shape
    if W * H <= max_pixels:
        return t
    import math
    import comfy.utils
    s = math.sqrt(max_pixels / (W * H))
    nw = max(1, round(W * s))
    nh = max(1, round(H * s))
    samples = t.movedim(-1, 1)  # [B,C,H,W]
    samples = comfy.utils.common_upscale(samples, nw, nh, "lanczos", "disabled")
    print(f"[bsk_MiniMaxDirector][crop] ref 이미지 미리 확대 {W}x{H} -> {nw}x{nh} (≤{max_pixels}px)", flush=True)
    return samples.movedim(1, -1)


def _align_ref_to_official(t: torch.Tensor, target_w: int, target_h: int, ref_image_size: str = "match") -> torch.Tensor:
    """ref 이미지를 공식 MiniMaxH3ReferenceToVideo 최종 출력 크기로 강제로 조정합니다.」。

    공식 노드 매치 분기：
        scale = min(1.0, sqrt((width*height)/(w*h)))        # 면적 비율, 하락만 하지 상승하지 않음
        tw = max(32, round(w*scale/32)*32); th = max(32, round(h*scale/32)*32)
    다음으로 플러그인 디렉토리 **floor**（내림 연산(floor)을 사용하여 32비트 정렬을 보장합니다(round와 달리) tw*th ≤ target_w*target_h
    （공식 기저 면적). 따라서 공식 노드는 이 함수 출력을 다시 계산합니다 scale'=min(1, sqrt(완전/(tw*th)))=1，
    그리고 tw/th 이미 32배수입니다 → round 변경 없음 → **공식 균등, 최종 크기 완전 확정**。
    프론트엔드가 동일한 알고리즘(computeFinalSize)을 시뮬레이션하고 크롭 상자를 흡수하여 '봐야 다는 것'을 구현합니다.」。
    ref_image_size 매개변수 정렬 공식: 'max'는 짧은 변 2048, 'match'(기본값)는 생성 면적 사용。
    """
    if t is None:
        return t
    if t.dim() == 3:
        t = t.unsqueeze(0)
    if t.dim() != 4:
        return t
    import math
    import comfy.utils
    _, H, W, _ = t.shape
    if ref_image_size == "max":
        scale = min(1.0, 2048 / min(W, H))
    else:
        scale = min(1.0, math.sqrt(max(1, target_w * target_h) / max(1, W * H)))
    fw = max(32, math.floor(W * scale / 32) * 32)
    fh = max(32, math.floor(H * scale / 32) * 32)
    if fw == W and fh == H:
        return t
    samples = t.movedim(-1, 1)  # [B,C,H,W]
    samples = comfy.utils.common_upscale(samples, fw, fh, "lanczos", "disabled")
    print(f"[bsk_MiniMaxDirector][crop] ref 이미지 정렬 공식 최종 크기 {W}x{H} -> {fw}x{fh}", flush=True)
    return samples.movedim(1, -1)


def _auto_crop_for_target(iw: int, ih: int, ar: float):
    """최대 내부 사각형의 비율을 최대화 (원래 픽셀 정수, 중앙）——전면 기본 자르기 상자와 동일 기하。

    크롭 데이터 없을 시 기본 사용, 반환 {x,y,w,h}；파라미터 불법 반환 None。
    """
    if not iw or not ih or not ar or ar <= 0:
        return None
    if iw / ih >= ar:
        h = ih
        w = int(round(h * ar))
    else:
        w = iw
        h = int(round(w / ar))
    if w <= 0 or h <= 0:
        return None
    return {
        "x": max(0, int(round((iw - w) / 2))),
        "y": max(0, int(round((ih - h) / 2))),
        "w": w,
        "h": h,
    }


def _crop_or_default(t: torch.Tensor, crop, tw, th) -> torch.Tensor:
    """명시적 크롭으로 자르기; 크롭이 없으면 대상 비율(가로:세로)에 맞춰 내접 사각형을 바닥에 맞게 자릅니다.。

    공식 MiniMaxH3ImageToVideo / ReferenceToVideo는 잘라내지 않은 전체 그림에 대해 stretch(스트레칭)을 적용하며, 자료 비율
    목표와 일치하지 않으면 전체적으로 변형(예: 가로 이미지를 세로 이미지로 강제로 압축). 이 백업 보장 "이미지/비디오는 항상"
    자르기 좌표". 주의: 프레임 이어받기 (이전 그룹의 마지막 프레임 텐서)은 대상 비율에 있으며, 호출자는 이 함수에 목표 비율을 전달해서는 안 됩니다.。
    """
    if t is None:
        return None
    if crop:
        return _apply_crop_tensor(t, crop)
    if t.dim() == 3:
        t = t.unsqueeze(0)
    if t.dim() == 4 and t.shape[1] and t.shape[2] and tw and th:
        _, H, W, _ = t.shape
        ac = _auto_crop_for_target(W, H, tw / th)
        if ac:
            print(f"[bsk_MiniMaxDirector][crop] 절단 데이터가 없으므로 자동으로 대상 비율 내에 맞게 프레임으로 연결합니다 {ac}（입력 {W}x{H}，목표 {tw}x{th}）",
                  flush=True)
            return _apply_crop_tensor(t, ac)
    return t


def _load_image_tensor(image, _crop=None) -> torch.Tensor:
    """이미지 입력을 [1,H,W,3] float32 텐서로 변환（0~1）。

    torch.Tensor [B,H,W,3] 및 파일 경로 문자열 지원 / ComfyUI annotated path。
    _crop：선택 {x,y,w,h}（원래 픽셀, 원점 왼쪽 상단), 리사이즈 전 래프팅。
    주의: 프레임 덤프는 텐서 형식으로 전달 시에도 자르기를 통과합니다 (이전 Tensor 분기에서 _crop 무시) bug）。
    """
    if isinstance(image, torch.Tensor):
        t = image
        if t.dim() == 3:
            t = t.unsqueeze(0)
        if _crop:
            t = _apply_crop_tensor(t, _crop)
        return t.float()
    if isinstance(image, str) and image:
        import folder_paths
        import numpy as np
        from PIL import Image
        path = folder_paths.get_annotated_filepath(image)
        img = Image.open(path).convert("RGB")
        if _crop:
            img = _apply_crop(img, _crop)
        # 32자리 정렬 중단: 공식 MiniMaxH3ImageToVideo 최종적으로 사용 _resize(...,"disabled"/"center")
        # first/last_frame을 목표 크기로 직접 조정 width×height，중간 크기 모두 정확하게 늘림 수정。
        # 기존의 w/h는 각각 독립적 //32*32 전면 잠금 크롭 비율을 파괴함（560x896→544x896，0.625→0.607），
        # 어떤 비율이든 두 32배 비율로 정확하게 표현할 수 없습니다——크롭 후 원본 크기 유지는 가장 정확함。
        arr = np.array(img).astype(np.float32) / 255.0
        return torch.from_numpy(arr)[None, ...]
    raise TypeError(f"bsk_MiniMaxDirector: 지원되지 않는 이미지 유형 {type(image)}")


def _load_video_frames(video_file: str, _crop=None) -> torch.Tensor:
    """참조 비디오를 [T,H,W,3] float32 텐서로 로드합니다（0~1），vs_ltx_patch의 이미지/비디오 로드 경로 재사용。

    형태 [T, H, W, 3] 반환. _crop: 선택 사항 {x,y,w,h}，각 프레임에 대해 동일한 영역을 잘라내기。
    """
    import folder_paths
    from PIL import Image
    import numpy as np

    if video_file.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
        # 싱글 이미지 1 프레임 비디오로 사용
        return _load_image_tensor(video_file, _crop=_crop)

    # 비디오: torchvision.io.read_video는 최신 버전에서 사용됩니다.(torchvision>=0.17)제거됨，
    # PyAV 디코딩을 사용하고 있으며 (환경이 설치됨 av), torchvision(구 버전)으로 되돌립니다. decord。
    import folder_paths
    path = folder_paths.get_annotated_filepath(video_file)

    # 1) PyAV（우선순위, 버전 간 호환성 안정적）
    try:
        import av
        container = av.open(path)
        frames = []
        for frame in container.decode(video=0):
            img = frame.to_image()  # PIL.Image, RGB
            frames.append(np.asarray(img, dtype=np.float32) / 255.0)
        container.close()
        if not frames:
            raise RuntimeError("비디오에서 어떤 프레임도 디코딩되지 않음")
        arr = np.stack(frames, axis=0)  # [T, H, W, 3]
        if _crop:
            ih, iw = arr.shape[1], arr.shape[2]
            rc = _normalize_crop(_crop, iw, ih)
            if rc is not None:
                x, y, w, h = rc
                arr = arr[:, y:y + h, x:x + w, :]
        return torch.from_numpy(arr)
    except Exception as e:
        # 2) 되돌리기: 이전 버전 torchvision.io.read_video
        try:
            import torchvision
            vframes, audio, info = torchvision.io.read_video(path, pts_unit="sec")
            arr = vframes.cpu().numpy().astype(np.float32) / 255.0
            if _crop:
                ih, iw = arr.shape[1], arr.shape[2]
                rc = _normalize_crop(_crop, iw, ih)
                if rc is not None:
                    x, y, w, h = rc
                    arr = arr[:, y:y + h, x:x + w, :]
            return torch.from_numpy(arr)
        except Exception:
            pass
        # 3) 회퇴：decord
        try:
            from decord import VideoReader, cpu
            vr = VideoReader(path, ctx=cpu(0))
            arr = vr.get_batch(list(range(len(vr)))).asnumpy().astype(np.float32) / 255.0
            if _crop:
                ih, iw = arr.shape[1], arr.shape[2]
                rc = _normalize_crop(_crop, iw, ih)
                if rc is not None:
                    x, y, w, h = rc
                    arr = arr[:, y:y + h, x:x + w, :]
            return torch.from_numpy(arr)
        except Exception:
            pass
        raise RuntimeError(f"bsk_MiniMaxDirector: 참조 비디오 로드 실패 {video_file}: {e}")


def _load_audio(audio_file: str) -> dict:
    """참조 오디오를 공식 MiniMaxH3ReferenceToVideo._encode_ref_audio가 기대하는 형식으로 로드。

    반환 {"waveform": Tensor[B,C,L], "sample_rate": int}（B=1）。
    """
    import folder_paths
    import torchaudio
    path = folder_paths.get_annotated_filepath(audio_file)
    waveform, sample_rate = torchaudio.load(path)  # waveform: [C, L]
    waveform = waveform.unsqueeze(0)               # [B, C, L]
    return {"waveform": waveform, "sample_rate": int(sample_rate)}


def _out(o):
    """호환 새로운 스타일 io.NodeOutput 및 이전 스타일 tuple 반환 언패킹。"""
    return o.args if hasattr(o, "args") else o


def _sanitize_audio_waveform(seg_audio: dict) -> dict:
    """Wash NaN/Inf from audio waveform (Ascend fp16 AudioVAE value issue)）。

    - 파형 깨끗함 (CUDA 등 정상 환경)）→ 원래대로 반환, 0행 변화 없음。
    - 포함 NaN/Inf → 예외 샘플링을 0(무음)으로 설정하고 전체적으로 [-1,1] 범위로 클램핑하여 다운스트림을 보장
      SaveVideo AAC 인코딩이 충돌하지 않도록 (avcodec_send_frame 반환) 22）。
    반환 입력과 동일한 구조 {"waveform", "sample_rate"} dict。
    """
    wf = seg_audio.get("waveform")
    if wf is None:
        return seg_audio
    bad = torch.isnan(wf) | torch.isinf(wf)
    if not bool(bad.any().item()):
        return seg_audio
    n_bad = int(bad.sum().item())
    total = int(wf.numel())
    print(f"[bsk_MiniMaxDirector][audio] 파형 포함 {n_bad}/{total} NaN/Inf 샘플링, 음소거화됨", flush=True)
    wf = torch.nan_to_num(wf, nan=0.0, posinf=0.0, neginf=0.0).clamp(-1.0, 1.0)
    return {"waveform": wf, "sample_rate": seg_audio.get("sample_rate")}


# 소요 시간 누적 (그룹 간 연결 덧셈 누적 동일 항목), process는 매 시작 시 초기화；
# process 종료 시 분할 + 총계 소요 시간은 bsk_director_timing 이벤트로 프론트엔드로 푸시됩니다
# （프론트엔드에서 히스토리 바 우측 정렬 표시, 총 소요 시간은 백엔드 기준, 프론트엔드 닫힘/리프레시 시 데이터 손실 방지）。
_TIMING_STATS: dict = {}
_TIMING_PROC_START = None
# 현재 process의 run_id: process 시작 시 생성, 마다 kj_preview_override / bsk_vs_stage
# 이벤트는 프론트엔드로 브로드캐스트되어 새로운 생성과 이전 생성의 차이점을 구분(이전 실행의 지연 프레임/지연 상태가 새로운 미리보기에 포함되지 않도록)）。
_CURRENT_RUN_ID: str | None = None


def _push_stage(stage: str, detail: str = "", unique_id=None):
    """현재 프런트엔드로 생성 단계 상태(bsk_vs_stage)를 전송하고, 실패하면 무음으로 처리합니다.。

    stage 간결한 동사 구문 (예“참조 자료 로드/샘플링 중/VAE 비디오 디코딩”），detail 보충 설명。
    프로세스 미리보 프레임을 받기 전까지 표시합니다“현재 실행 중인 것은 무엇”，프레임 수신 후 자동 숨기기。
    """
    try:
        if PromptServer is None:
            return
        ps = PromptServer.instance
        ps.send_sync("bsk_vs_stage", {
            "run_id": _CURRENT_RUN_ID,
            "stage": stage,
            "detail": detail,
            "unique_id": str(unique_id) if unique_id is not None else None,
            "ts": time.time(),
        }, None)  # 브로드캐스트: kj_preview_override와 일치하여 패널/멀티 클라이언트 모두 수신
    except Exception:
        pass


def _timing_norm(label: str) -> str:
    """정규화 소요 시간 레이블: 괄호 작업 접미사 제거 (예: CLIP 조건부)(fl2v)'→'CLIP 조건 탭 '생성', 그룹 간 합산 가능。"""
    i = label.find("(")
    return label[:i].strip() if i > 0 else label


def _timing(label: str, start, extra: str = ""):
    """시간 소요 통계: 인쇄 + _TIMING_STATS에 누적하고 bsk_director_timing 증분 이벤트(increment event)를 전송합니다.。"""
    ms = (time.perf_counter() - start) * 1000.0
    _TIMING_STATS[_timing_norm(label)] = _TIMING_STATS.get(_timing_norm(label), 0.0) + ms
    print(f"[bsk_MiniMaxDirector][timing] {label}: {ms:.1f} ms{extra}", flush=True)
    _push_timing()


def _push_timing():
    """현재 누적 항목 분할 + 총 소요 시간 푸시 프론트엔드 (실패는 무음, 노드 실행에 영향을 미치지 않음)）。"""
    try:
        if PromptServer is None:
            return
        ps = PromptServer.instance
        cid = getattr(ps, "client_id", None)
        if cid is None:
            return
        # process 총 소요 시간 자체를 제외합니다(별도로 total_ms로 표시)）
        items = [{"label": k, "ms": round(v, 1)}
                 for k, v in _TIMING_STATS.items() if k != "process 총 소요 시간"]
        total_ms = None
        if _TIMING_PROC_START is not None:
            total_ms = round((time.perf_counter() - _TIMING_PROC_START) * 1000.0, 1)
        ps.send_sync("bsk_director_timing", {
            "items": items,
            "total_ms": total_ms,
        }, cid)
    except Exception:
        pass


def _unload_clip_model(clip):
    """CLIP 인코딩이 완료되면 CLIP 모델을 완전히 제거합니다(기본적으로 제거되지 않음; 메모리가 부족하면 프론트 엔드에서 선택하여 켜기)）。

    comfy.model_management.unload_model_and_clones만 제거합니다 CLIP patcher，
    CPU RAM을 직접 해제하고, 이후 세그먼트/노드가 필요하면 자동으로 다시 로드합니다.。
    """
    try:
        if getattr(clip, "patcher", None) is not None:
            model_management.unload_model_and_clones(clip.patcher, unload_additional_models=False)
    except Exception as e:
        print(f"[bsk_MiniMaxDirector][unload] CLIP 언로드 실패: {e}", flush=True)


# ============== 중간 결과물 (CLIP 조건 데이터 캐시)） ==============
# 각 생성물 = 두 개의 동일 이름 파일이 위치해 있습니다 user/default/bsk_video_studio/intermediates/：
#   <name>.json    —— 파라미터/설정 (프론트엔드 제출 시 작성, 단일 키 릴리즈 가능)）
#   <name>.bskdata —— CLIP 인코딩된 positive/latent (본 모듈에 쓰여진)，torch.save）
# .bskdata 구조：
#   {"format": "bsk_intermediate_v1", "entries": [{"positive": cond, "latent": lat}, ...]}
#   concat 여러 세트의 시작/종료 프레임 시 entries 길이 = 그룹 수; ref / 단일 씬 모드 = 1 줄。
INTERMEDIATE_SUBDIR = "intermediates"


def _intermediate_dir() -> str:
    """중간 결과 디렉토리 (VS_INTERMEDIATE_DIR과 일치)）。"""
    try:
        import folder_paths
        base = os.path.join(folder_paths.get_user_directory(), "default", "bsk_video_studio")
    except Exception:
        base = os.path.join(os.getcwd(), "user", "default", "bsk_video_studio")
    d = os.path.join(base, INTERMEDIATE_SUBDIR)
    try:
        os.makedirs(d, exist_ok=True)
    except Exception:
        pass
    return d


def _intermediate_path(name: str, ext: str = ".bskdata") -> str:
    """안전한 결과 경로 결합 (경로 침투 방지), 절대 경로 반환。"""
    safe = os.path.basename(str(name or "").strip())
    if not safe:
        return ""
    return os.path.join(_intermediate_dir(), safe + ext)


def _to_cpu(obj):
    """텐서를 CPU로 이동(detach); NestedTensor를 사용합니다. .cpu()。"""
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_cpu(x) for x in obj)
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    if getattr(obj, "is_nested", False) and hasattr(obj, "cpu"):
        try:
            return obj.cpu()
        except Exception:
            return obj
    return obj


def _to_device(obj, device):
    """텐서를 지정된 장치로 재배치(샘플링 전 사용)하고, NestedTensor를 사용하여 .to(device=...)。"""
    if device is None:
        return obj
    if isinstance(obj, torch.Tensor):
        return obj.to(device=device)
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_device(x, device) for x in obj)
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    if getattr(obj, "is_nested", False) and hasattr(obj, "to"):
        try:
            return obj.to(device=device)
        except Exception:
            return obj
    return obj


def _intermediate_save_all(name: str, entries: list) -> str:
    """저장 entries=[(positive, latent), ...] 저장 <dir>/<name>.bskdata（CPU 텐서）。"""
    path = _intermediate_path(name)
    if not path:
        return ""
    torch.save({
        "format": "bsk_intermediate_v1",
        "entries": [{"positive": p, "latent": l} for p, l in entries],
    }, path)
    return path


def _intermediate_load_all(name: str, device=None):
    """읽기 <dir>/<name>.bskdata，[(positive, latent), ...]; 누락/손상된 경우 반환합니다. None。

    텐서 먼저 CPU에 로드하고 필요에 따라 장치로 이동 (긍정은 샘플링 장치에 있어야 함)；
    latent CPU도 안전하게 유지——공식 빈 latent 본질적으로 CPU, 샘플러 내부에서 이동）。
    """
    path = _intermediate_path(name)
    if not path or not os.path.isfile(path):
        return None
    try:
        data = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"[bsk_MiniMaxDirector][intermediate] 로드 실패 {path}: {e}", flush=True)
        return None
    raw = data.get("entries") if isinstance(data, dict) else None
    if not isinstance(raw, list) or not raw:
        return None
    entries = []
    for e in raw:
        if not isinstance(e, dict):
            return None
        p = e.get("positive")
        l = e.get("latent")
        if p is None or l is None:
            return None
        entries.append((p, l))
    if device is not None:
        entries = [(_to_device(p, device), _to_device(l, device)) for p, l in entries]
    return entries


def _push_intermediate_saved(name: str, count: int):
    """프론트엔드로 중간 결과물이 저장되었음을 알립니다(왼쪽 목록 새로 고침용), 실패 시 무음으로 처리。"""
    try:
        if PromptServer is None:
            return
        ps = PromptServer.instance
        cid = getattr(ps, "client_id", None)
        if cid is None:
            return
        ps.send_sync("bsk_intermediate_saved", {"name": name, "count": count}, cid)
    except Exception:
        pass


def _save_video_frames_png(images, prefix: str, subfolder: str = "") -> list:
    """이미지 시퀀스 모드: [T,H,W,3] 텐서（0~1）프레임별로 PNG로 저장하여 output 폴더에 저장합니다.。

    [B,T,H,W,3](5D) 입력 호환, 배치 차원 자동 압축, 반환
    [{"filename": str, "subfolder": str, "type": "output"}, ...]（및 /view?type=output 계약 일치）。
    실패 시 프레임별 건너뛰기, 전체 예외 반환 빈 목록, 샘플링 프로세스 중단 방지。
    """
    if images is None:
        return []
    try:
        t = images
        if t.dim() == 5:
            t = t.squeeze(0)
        if t.dim() != 4 or t.shape[3] not in (1, 3, 4):
            print(f"[bsk_MiniMaxDirector][imageSeq] 프레임별로 저장 불가 images: shape={tuple(t.shape)}", flush=True)
            return []
    except Exception as e:
        print(f"[bsk_MiniMaxDirector][imageSeq] 프레임 텐서 검증 실패: {e}", flush=True)
        return []
    try:
        import numpy as np
        import folder_paths
        out_dir = folder_paths.get_output_directory()
        os.makedirs(out_dir, exist_ok=True)
        arr = t.detach().float().cpu().numpy()
        n = arr.shape[0]
        digits = max(4, len(str(n)))
        saved = []
        for i in range(n):
            f = arr[i]
            if f.ndim != 3 or f.shape[2] not in (3, 4):
                continue
            pil = Image.fromarray((np.clip(f, 0.0, 1.0) * 255.0).astype("uint8"))
            if pil.mode != "RGB":
                pil = pil.convert("RGB")
            fn = f"{prefix}_{i:0{digits}d}.png"
            pil.save(os.path.join(out_dir, fn), format="PNG")
            saved.append({"filename": fn, "subfolder": subfolder, "type": "output"})
        if saved:
            print(f"[bsk_MiniMaxDirector][imageSeq] 저장됨 {len(saved)} PNG 프레임 -> {out_dir}", flush=True)
        return saved
    except Exception as e:
        print(f"[bsk_MiniMaxDirector][imageSeq] PNG 프레임 저장 실패: {e}", flush=True)
        return []


def _push_image_seq_saved(files: list):
    """프론트엔드 이미지 시퀀스 저장 알림(파일 목록 포함, 썸네일 그리드 렌더링용), 실패는 무시。"""
    try:
        if PromptServer is None:
            return
        ps = PromptServer.instance
        cid = getattr(ps, "client_id", None)
        if cid is None:
            return
        ps.send_sync("bsk_image_seq_saved", {"files": files, "count": len(files)}, cid)
    except Exception:
        pass


class BskMiniMaxDirector:
    """MiniMax H3 감독 노드 구성 (자체 작성, 공식 기반 재사용, 양방향)：fl2v/i2v/t2v + ref）。"""

    CATEGORY = "BSK/MiniMax"
    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("images", "audio")
    OUTPUT_IS_LIST = (True, True)
    FUNCTION = "process"

    @classmethod
    def INPUT_TYPES(cls):
        # 드롭다운을 4개의 중국어 주석이 포함된 옵션으로 간소화; 동시에 이전 열거 값을 합법 옵션으로 함께 나열。
        # 이유: ComfyUI의 combo 검증은 process 이전에 발생합니다 (_resolve_task_type는 덮어쓰지 않음)），
        # 기존 워크플로우/패널 템플릿/패널 설정 저장의 task_type은 약칭(예시)입니다. "t2v"），목록에 없으면
        # Value not in list 오류가 발생하고 실행을 거부합니다. UI에 추가된 이전 값은 호환성을 위한 것이며, 프론트엔드 제출 시에는 항상 기록됩니다 td.mode。
        task_type_options = [
            "fl2v — 시작/종료 프레임 비디오(First-Last Frame)",
            "i2v — 비디오 생성(Image to Video)",
            "t2v — 텍스트에서 비디오(Text to Video)",
            "ref — 참조 생 비디오(Reference To Video)",
            # ---- 구 enum 호환 (역사적 워크플로우가 이러한 값을 저장할 수 있습니다)）----
            "fl2v", "i2v", "t2v", "ref",
            "first_last_to_video", "image_to_video", "text_to_video", "reference_to_video",
        ]
        return {
            "required": {
                "model": ("MODEL",),
                "vae": ("VAE",),
                "timeline_data": ("STRING", {"default": "{}", "multiline": True}),
            },
            "optional": {
                # clip 선택 사항 활성화 시, "중간 결과 사용" 기능이 활성화되면 CLIP 연결이 끊어집니다.，
                # 백엔드는 데이터 파일에서 조건 직접 읽기, CLIP 노드 필요 없음。
                "clip": ("CLIP",),
                # 비디오에는 항상 오디오가 있습니다: v2v/rv2v(참조 경로에 비디오 참조)가 필요합니다 audio_vae；
                # 순수한 t2v/fl2v/i2v, 연결되지 않으면 오디오 없이 무음（seg_audio=None）。
                "audio_vae": ("VAE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "task_type": (task_type_options, {"default": "fl2v — 시작/종료 프레임 비디오(First-Last Frame)"}),
                # 일반 노드 복원하여 자르기 좌표를）：
                #   unload_clip  —— CLIP 인코딩이 완료되면 CLIP을 완전히 언로드하고 후속 샘플링을 위해 GPU 메모리를 확보합니다；
                #   unload_model —— UNET 샘플링 완료 후, VAE 디코딩 전 완전히 언로드 UNET。
                # 그녀는 녹색 하트 패턴이 있는 머그컵을 들고 있었고, 마치 여유로운 애프터눈 티타임을 즐기는 것처럼 보였다。
                "unload_clip": ("BOOLEAN", {"default": False, "label_on": "アンロードCLIP", "label_off": "보존CLIP"}),
                "unload_model": ("BOOLEAN", {"default": False, "label_on": "アンロードUNET", "label_off": "보존UNET"}),
                # ===== 중간 결과물 (우선 timeline_data.intermediate 읽기; 노드 입력은 수동 워크플로우 백업)）=====
                "use_intermediate": ("BOOLEAN", {"default": True, "label_on": "원래 너비 사용", "label_off": "사용하지 마세요"}),
                "only_intermediate": ("BOOLEAN", {"default": False, "label_on": "중간 결과물만 생성", "label_off": "완전 생성"}),
                "intermediate_name": ("STRING", {"default": "", "multiline": False}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    def process(self, model, vae, timeline_data, audio_vae=None, clip=None, seed=0, task_type="first_last_to_video",
                unload_clip=False, unload_model=False, unique_id=None,
                use_intermediate=True, only_intermediate=False, intermediate_name=""):
        global _TIMING_PROC_START, _CURRENT_RUN_ID
        _proc_start = time.perf_counter()
        _TIMING_STATS.clear()
        _TIMING_PROC_START = _proc_start
        _CURRENT_RUN_ID = str(time.time_ns())
        print(f"[BSK] MiniMaxDirector 수신 seed={seed} unload_clip={unload_clip} unload_model={unload_model} "
              f"intermediate: use={use_intermediate} only={only_intermediate} name={intermediate_name}", flush=True)
        _push_stage("생성 준비", "생성 작업 초기화…", unique_id)
        # 생성 캐시: 이 실행의 시작을 표시(상태 running), 프런트엔드가 세션 간에 미리보기/결과를 복원할 수 있습니다.
        self._runs_begin(unique_id)

        try:
            td = json.loads(timeline_data) if isinstance(timeline_data, str) else (timeline_data or {})
        except Exception:
            td = {}

        # ---- 중간 생성물 모드 파싱 (timeline_data.intermediate는 권위 있으며, 노드 입력은 기본 수동 워크플로우입니다)）----
        inter_mode = {
            "use": False,
            "only": False,
            "name": "",
            "_saves": [],      # only 모드: 누적 보관할 (positive, latent)（CPU）
            "_entries": None,  # use 모드: 로드된 결과 항목 [(positive, latent), ...]
        }
        _inter_td = td.get("intermediate")
        if isinstance(_inter_td, dict):
            inter_mode["use"] = bool(_inter_td.get("use", False))
            inter_mode["only"] = bool(_inter_td.get("only", False))
            inter_mode["name"] = str(_inter_td.get("name") or "").strip()
        else:
            # 오래된 계약 / 수동 워크플로우 기본 사용：td.useIntermediate / td.onlyIntermediate / td.intermediateName
            inter_mode["use"] = bool(td.get("useIntermediate", bool(use_intermediate)))
            inter_mode["only"] = bool(td.get("onlyIntermediate", bool(only_intermediate)))
            inter_mode["name"] = str(td.get("intermediateName") or intermediate_name or "").strip()
        if inter_mode["use"] and inter_mode["only"]:
            inter_mode["use"] = False  # 상호 배타적: 생성 시 '사용' 무시(산출물 미존재).）
        if inter_mode["only"] and not inter_mode["name"]:
            inter_mode["name"] = "int_" + time.strftime("%Y%m%d_%H%M%S")  # 자동 이름 지정, 저장 가능 보장
        inter_mode["_fps"] = float(td.get("frameRate") or 24)
        if inter_mode["use"] or inter_mode["only"]:
            print(f"[bsk_MiniMaxDirector][intermediate] use={inter_mode['use']} only={inter_mode['only']} name={inter_mode['name']}", flush=True)

        width = _align_dim(int(td.get("width", 768) or 768))
        height = _align_dim(int(td.get("height", 480) or 480))
        total_frames = int(td.get("totalFrames", 97) or 97)
        global_prompt = (td.get("global", {}) or {}).get("prompt", "") or ""
        # 프론트엔드로 정확한 mode (v6 계약)를 유도하고, timelineMode / global.taskType / 연결을 되돌립니다. task_type
        raw_mode = (td.get("mode")
                    or td.get("timelineMode")
                    or (td.get("global", {}) or {}).get("taskType")
                    or task_type)
        mode = _resolve_task_type(raw_mode)
        # ref 모드 v6/v5에서 mode는 이미 'ref'입니다. 프론트엔드에서 이전 버전의 timelineMode를 사용하고 references가 포함되어 있다면 'ref'로 처리합니다.
        references = td.get("references") or []
        if mode != "reference_to_video" and references:
            mode = "reference_to_video"
        print(f"[bsk_MiniMaxDirector] mode={mode} received totalFrames={total_frames} width={width} height={height} frameRate={td.get('frameRate')}", flush=True)

        # 샘플 초점(프론트엔드 확장 가능, 기본 안정)）
        steps = int(td.get("steps", 30) or 30)
        cfg = float(td.get("cfg", 3.5) or 3.5)
        shift_video = float(td.get("shiftVideo", 12.0) or 12.0)
        shift_audio = float(td.get("shiftAudio", 3.0) or 3.0)
        sampler_name = td.get("samplerName", "euler")
        scheduler = td.get("scheduler", "simple")
        # 샘플링 구간（1~2 단): timeline_data.sampling 권한, 기본으로 위에서 파싱된 상위 스칼라(이전 버전 프론트엔드에 대한 호환성)）。
        # Second paragraph = 첫 번째 부분의 latent를 계속 사용하여 노이즈 제거（denoise<1 중간 sigma 이어지거나 사용자 정의 sigma로 덮어쓰기），
        # "두 번째 샘플링 디테일 강화" 및 "사용자 지정 sigma 억제 고동력 찢어짐 감쇠" 두 가지 용도로 사용됩니다.。
        sampling_stages = self._build_sampling_stages(
            td, steps, sampler_name, scheduler, shift_video, shift_audio)
        if len(sampling_stages) > 1:
            _s1, _s2 = sampling_stages[0], sampling_stages[1]
            print("[bsk_MiniMaxDirector][sampling] 두 번 샘플링: "
                  f"s1={_s1['samplerName']}/{_s1['scheduler']}/steps={_s1['steps']}/denoise={_s1['denoise']}"
                  f" -> s2={_s2['samplerName']}/{_s2['scheduler']}/steps={_s2['steps']}/denoise={_s2['denoise']}"
                  f"{' [사용자 정의 sigma]' if _s2['sigmas'] else ''}", flush=True)
        # 생성 과정 미리보기 위치: 'off' 끄기; 'low'/'medium'/'high' 미리보기 해상도 제어 (프론트엔드 buildMiniMaxTimelineData 쓰기)）
        preview_quality = td.get("previewQuality", "off")
        # 생성 과정 미리보기 프레임/프레임 수 (정렬 KJ 모델 오버라이드 KJ) preview_fps / preview_frames）：
        # - preview_fps    프레임 레이트 재생, 프로젝트 `frameRate` (24/30...), 프런트엔드가 올바른 속도로 재생되도록 보장。
        # - preview_frames 각 단계 디코딩 시 비디오 잠재 변수의 프레임 수 제한（>= latent 등지 NPU 하드웨어 모니터링）。
        #   기본적으로 프로젝트 총 프레임 수를 가져옵니다 (참조 워크플로우와 정렬) preview_frames=240 총 프레임 수와 동일하게 설정하고 [1,128]로 볼륨 팽창을 방지합니다.。
        try:
            preview_fps = int(td.get("frameRate") or 24)
        except Exception:
            preview_fps = 24
        try:
            preview_frames = int(td.get("previewFrames") or total_frames)
        except Exception:
            preview_frames = int(total_frames)
        preview_frames = max(1, min(128, preview_frames))
        preview_fps = max(1, min(60, preview_fps))
        # 이미지 시퀀스 모드 (프론트엔드 timeline_data.imageMode): 비디오 합성하지 않고, 샘플링 후 각 프레임을 PNG로 output 디렉토리에 저장합니다.，
        # 또한 bsk_image_seq_saved 이벤트로 파일 목록을 프론트엔드로 전달합니다 (프론트엔드는 썸네일 그리드를 렌더링하고, 클릭하여 대형 이미지를 확인합니다)）。
        image_mode = bool(td.get("imageMode", False))

        all_images: list = []
        all_audio: list = []

        # ---------- 상위 레이어 모드 분기 ----------
        if mode == "reference_to_video":
            # 경로 B: 자료 참조 많이 (모드 B / 이전 버전) ref）
            all_images, all_audio = self._run_refs(
                td, references, clip, vae, audio_vae, model, width, height,
                total_frames, global_prompt, steps, cfg, shift_video, shift_audio,
                sampler_name, scheduler, seed, preview_quality, unique_id,
                unload_clip=unload_clip, unload_model=unload_model,
                inter_mode=inter_mode,
                preview_fps=preview_fps, preview_frames=preview_frames,
                sampling_stages=sampling_stages,
            )
        elif td.get("groups"):
            # 경로 A-1: 모드 A, 여러 세트의 시작/종료 프레임 반복 연결（concat）
            all_images, all_audio = self._run_concat(
                td, clip, vae, audio_vae, model, width, height,
                global_prompt, steps, cfg, shift_video, shift_audio,
                sampler_name, scheduler, seed, preview_quality, unique_id,
                unload_clip=unload_clip, unload_model=unload_model,
                inter_mode=inter_mode,
                preview_fps=preview_fps, preview_frames=preview_frames,
                sampling_stages=sampling_stages,
            )
        else:
            # 경로 A-2: mode가 fl2v/i2v/t2v의 단일 구간 역화 (구 버전 v5 단일 경로)）
            images, seg_audio = self._run_one_fl2v_group(
                first_frame_path=td.get("firstFrame"),
                last_frame_path=(td.get("lastFrame") if mode == "first_last_to_video" else None),
                task_key=("fl2v" if mode == "first_last_to_video"
                          else ("i2v" if mode == "image_to_video" else "t2v")),
                frames=total_frames, prompt=td.get("prompt") or global_prompt or "",
                clip=clip, vae=vae, audio_vae=audio_vae, model=model, width=width, height=height,
                steps=steps, cfg=cfg, shift_video=shift_video, shift_audio=shift_audio,
                sampler_name=sampler_name, scheduler=scheduler, seed=seed,
                preview_quality=preview_quality, unique_id=unique_id,
                unload_clip=unload_clip, unload_model=unload_model,
                inter_mode=inter_mode, group_index=0,
                preview_fps=preview_fps, preview_frames=preview_frames,
                sampling_stages=sampling_stages,
            )
            all_images.append(images)
            all_audio.append(seg_audio)

        # ---- 중간 결과물만 생성: 모든 항목의 조건 데이터를 일괄 저장 ----
        # 중간 결과 모드만 실행하고 CLIP 기반 조건 샘플링을 하지 않으므로 (UNet은 GPU 메모리에 올라가지 않음)**모델 제거 안 함**：
        # 중간 결과물을 일괄적으로 미리 계산할 때 다음 그룹/다음 구성은 동일한 CLIP을 계속 재사용합니다. 이 경우 언로드하면 반복적으로 다시 로드해야 합니다.；
        # 메모리 부족은 ComfyUI 모델 자가 관리를 통해 처리합니다. unload_clip / unload_model 스위치는 이 모드에서 무시됨。
        if inter_mode["only"]:
            if inter_mode["_saves"]:
                _is = time.perf_counter()
                path = _intermediate_save_all(inter_mode["name"], inter_mode["_saves"])
                _push_intermediate_saved(inter_mode["name"], len(inter_mode["_saves"]))
                print(f"[bsk_MiniMaxDirector][intermediate] 중간 결과물 저장됨 {inter_mode['name']} "
                      f"(entries={len(inter_mode['_saves'])}) -> {path}", flush=True)
            else:
                print("[bsk_MiniMaxDirector][intermediate] 중간 결과물만 생성, 조건 데이터 생성 없음, 저장 건너듸", flush=True)

        # ---- 이미지 시퀀스 모드: 각 프레임을 PNG로 저장하고 비디오를 생성하지 않습니다 ----
        # 샘플 결과가 all_images(list, 보통 길이=1；concat 단일 비디오로 병합됨, 프레임별로 디스크에 기록 후
        # bsk_image_seq_saved 이벤트를 전송(파일 목록 포함)하여 프런트엔드에서 썸네일 그리드 렌더링。
        # "단일 중간 결과물 생성"과 상호 배타적 (only 모드에서 all_images는 빈 흑단, 실제 프레임을 저장할 수 없음)）。
        if image_mode and not inter_mode["only"]:
            _iseq = time.perf_counter()
            prefix = "bsk_frames_" + time.strftime("%Y%m%d_%H%M%S")
            all_saved: list = []
            multi = len(all_images) > 1
            for vi, imgs in enumerate(all_images):
                seg_prefix = f"{prefix}_{vi}" if multi else prefix
                all_saved.extend(_save_video_frames_png(imgs, seg_prefix))
            _push_image_seq_saved(all_saved)
            _timing("PNG 프레임 저장", _iseq, f" (frames={len(all_saved)})")
            if not all_saved:
                print("[bsk_MiniMaxDirector][imageSeq] 이미지 시퀀스 모드: PNG 프레임에 저장 안 함", flush=True)
            # 이미지 시퀀스 모드는 비디오를 생성하지 않습니다: 1 프레임을 흑색 이미지로 대체하여 반환 값을 방지하고 워크플로우에 남아있는 문제를 방지합니다.
            # SaveVideo 저장 노드가 전체 비디오를 다시 쓰기 위해 (프론트엔드 제출 시 저장 체인 끊김, 여기는 이중 보험)）。
            all_images = [self._placeholder_images(height, width)]
            all_audio = [self._silent_audio(1, inter_mode.get("_fps", 24))]

        _timing("process 총 소요 시간", _proc_start, f" (mode={mode})")
        # 캐시 생성: 샘플링 완료 (최종 결과는 프론트엔드에서 /history를 통해 확인)）
        self._runs_finish(unique_id)
        return (all_images, all_audio)

    @staticmethod
    def _placeholder_images(height, width):
        """중간 생성물 모드만 생성합니다: 1 프레임의 빈 검은색 이미지를 반환 ([1,H,W,3])하여 다운스트림 SaveVideo의 입력 부족 문제를 방지합니다。"""
        return torch.zeros([1, int(height), int(width), 3], dtype=torch.float32)

    @staticmethod
    def _silent_audio(frames=1, fps=24, sample_rate=44100):
        """중간 결과물 모드만 생성: 무음 오디오 (전체 0 볼트파형) 반환, concat 병합 / SaveVideo 하위 노드에서 기본 처리。"""
        n = max(1, int(round((int(frames) or 1) / max(1, int(fps) or 24) * sample_rate)))
        return {"waveform": torch.zeros([1, 1, n], dtype=torch.float32), "sample_rate": int(sample_rate)}

    def _resolve_conditioning(self, inter_mode, idx, clip, vae, prompt,
                              width, height, length, first_frame, last_frame,
                              ref_kwargs=None):
        """반환 (positive, latent)。

        - use 모드（inter_mode["use"] 그리고 이름이 있는 경우: 데이터 파일에서 idx로 미리 계산된 조건 파일을 로드합니다 (건너뛰기) CLIP）；
        - only 패턴: 정상적인 CLIP 조건 생성 실행 및 결과(CPU) 누적 inter_mode["_saves"]；
        - 일반 모드: 정상적으로 CLIP 조건 생성。
        """
        name = inter_mode.get("name") or ""
        if inter_mode.get("use") and name:
            entries = inter_mode.get("_entries")
            if entries is None:
                entries = _intermediate_load_all(name, device=None)
                if entries is None:
                    raise RuntimeError(
                        f"bsk_MiniMaxDirector: 중간 결과 데이터 파일이 없거나 손상됨: {name}.bskdata"
                        f"（먼저 왼쪽의 '중간 결과물' 목록에서 결과물이 생성되었는지 확인하거나 닫으세요“원래 너비 사용”）"
                    )
                inter_mode["_entries"] = entries
            if idx >= len(entries):
                raise RuntimeError(
                    f"bsk_MiniMaxDirector: 중간 생성물 {name} 전문적인 초고화질 애니메이션 피규어"
                    f"（결과 {len(entries)} 그룹, 현재 필요한 1번째 {idx + 1} 조）"
                )
            positive, latent = entries[idx]
            return positive, latent

        # ---- 정상 / only: 연결해야 함 CLIP ----
        if clip is None:
            raise RuntimeError(
                "bsk_MiniMaxDirector: Current workflow is not connected to CLIP and 'Use intermediate output」"
                "（CLIP에 연결하거나, 왼쪽의 '중간 결과물' 목록에서 데이터 파일을 선택하십시오.）"
            )
        # 단 프레임 / 초 단 프레임（length in {1,5}）BSK 내장 노드: 원시 지원 length=1 단 프레임 출력，
        # 공식적인 temporal_shape가 1프레임을 5프레임으로 강제로 정렬하도록 방지합니다. 다른 오버레이 프레임은 공식 노드를 통해 진행합니다.。
        _use_bsk = length in BSK_LOW_FRAME_FRAMES
        if ref_kwargs is not None:
            if _use_bsk:
                cond_out = _out(BskRefToVideo.execute(**ref_kwargs))
            else:
                cond_out = _out(MiniMaxH3ReferenceToVideo.execute(**ref_kwargs))
        else:
            if _use_bsk:
                cond_out = _out(BskImageToVideo.execute(
                    clip=clip, vae=vae, prompt=prompt,
                    width=width, height=height, length=length,
                    first_frame=first_frame, last_frame=last_frame,
                ))
            else:
                cond_out = _out(MiniMaxH3ImageToVideo.execute(
                    clip=clip, vae=vae, prompt=prompt,
                    width=width, height=height, length=length,
                    first_frame=first_frame, last_frame=last_frame,
                ))
        positive, latent = cond_out[0], cond_out[1]
        if inter_mode.get("only"):
            inter_mode["_saves"].append((_to_cpu(positive), _to_cpu(latent)))
        return positive, latent

    # ---------- 모드 A-1: 여러 세트의 시작/종료 프레임 반복 연결（concat）----------
    def _run_concat(self, td, clip, vae, audio_vae, model, width, height,
                    global_prompt, steps, cfg, shift_video, shift_audio,
                    sampler_name, scheduler, seed, preview_quality, unique_id,
                    unload_clip=False, unload_model=False, inter_mode=None,
                    preview_fps=24, preview_frames=97, sampling_stages=None):
        """순회 td["groups"]，각 그룹별로 task_key를 판단하고, 각 그룹마다 MiniMaxH3ImageToVideo 샘플링을 수행하며, 시간 축을 따라 concat。

        group = {
          "task_key": "t2v"|"i2v"|"fl2v",
          "firstFrame": "path"|null, "lastFrame": "path"|null,
          "firstFrameAuto": bool,        # 첫 번째 그룹이 아닌 경우 자체적으로 첫 번째 프레임이 없는 경우, 백엔드에서 이전 그룹의 실제 마지막 프레임 텐서를 이어받음
          "frames": int, "prompt": str
        }
        continuity（실행 중 프레임 백필: i2v 그룹이 자체적으로 첫 프레임이 없고 첫 그룹이 아닌 경우 및 활성화된 경우 firstFrameAuto，
        이전 세트의 실제 생성된 마지막 프레임 텐서 사용（images[-1]，0~1）본 그룹의 첫 프레임；
        첫 번째도 캡처를 켜도 캡처하지 않음 (이전 프레임 무시); 자체적으로 첫 프레임 이미지는 항상 우선 (캡처 무시)）。
        프레임은 생성된 마지막 프레임 텐서를 받습니다(정렬된 크기), 어떤 꼬리 프레임 슬롯 이미지도 사용하지 않고 크롭도 수행하지 않습니다.。
        """
        groups = td.get("groups") or []
        all_images: list = []
        all_audio: list = []
        prev_last_tensor = None  # 이전 그룹에서 생성한 비디오의 마지막 프레임 텐서（[H,W,3] 0~1），실시간으로 이어받음
        for gi, g in enumerate(groups):
            task_key = (g.get("task_key") or "t2v").lower()
            # 각 그룹 시작 시, 현재 그룹 번호/총 그룹 수를 표시하고 "그룹 n/G : 비율"을 보여줍니다.」
            if PromptServer is not None and getattr(PromptServer.instance, "client_id", None):
                try:
                    PromptServer.instance.send_sync(
                        "bsk_concat_group",
                        {"index": gi, "total": len(groups), "task_key": task_key},
                        PromptServer.instance.client_id
                    )
                except Exception as e:
                    print(f"[bsk_MiniMaxDirector][concat] 그룹 번호 이벤트 전송 실패: {e}")
            own_ff = g.get("firstFrame")          # 자신의 첫 번째 프레임 자재 (경로 또는 None）
            lf = g.get("lastFrame")
            auto_on = bool(g.get("firstFrameAuto"))  # 활성화하시겠습니까?“이전 그룹 테일프레임 수용”
            is_first = (gi == 0)

            # ---- 프레임 시작 소스 결정 (견고성 우선)） ----
            # 1) 첫 프레임 이미지 보유: 수용 여부와 관계없이 항상 자체 첫 프레임 사용 (최우선)）。
            # 2) 자신에게 프레임 없음 + 非제일조 + 활성화 이어받기 + 이전 그룹의 마지막 프레임 사용 가능: 이전 그룹의 실제 마지막 프레임 이어짐。
            # 3) 첫 번째 그룹: 연결을 켜도 이전 그룹을 이어받지 않으므로, 존재하지 않는 데이터로 프레임 초기화하지 마십시오。
            ff = own_ff
            ff_source = "self_image" if own_ff else "none"
            if not ff and auto_on and not is_first and prev_last_tensor is not None:
                ff = prev_last_tensor          # 텐서 직접 전송, _load_image_tensor는 텐서 입력 지원
                ff_source = "inherited_last"

            # ---- 디버그 로그: 상태 수신 + 첫 프레임 데이터 수신 여부 ----
            print(f"[bsk_MiniMaxDirector][concat] group[{gi}] "
                  f"task_key={task_key} is_first={is_first} auto_inherit={auto_on} "
                  f"has_own_first={bool(own_ff)} prev_last_available={prev_last_tensor is not None} "
                  f"-> first_frame_source={ff_source} "
                  f"received_first_data={ff is not None}")

            images, seg_audio = self._run_one_fl2v_group(
                first_frame_path=ff,
                last_frame_path=lf,
                first_frame_crop=g.get("firstFrameCrop"),
                last_frame_crop=g.get("lastFrameCrop"),
                task_key=task_key,
                frames=g.get("frames"),
                prompt=g.get("prompt") or global_prompt or "",
                clip=clip, vae=vae, audio_vae=audio_vae, model=model, width=width, height=height,
                steps=steps, cfg=cfg, shift_video=shift_video, shift_audio=shift_audio,
                sampler_name=sampler_name, scheduler=scheduler, seed=seed,
                preview_quality=preview_quality, unique_id=unique_id,
                unload_clip=unload_clip, unload_model=unload_model,
                inter_mode=inter_mode, group_index=gi,
                preview_fps=preview_fps, preview_frames=preview_frames,
                sampling_stages=sampling_stages,
            )
            # 이 그룹에서 생성된 비디오의 마지막 프레임을 저장하고 다음 그룹에서 실행할 때 이어받습니다(이미지는 [T,H,W,3] 0~1 텐서）
            if images is not None:
                try:
                    prev_last_tensor = images[-1].detach().float().cpu().contiguous()
                except Exception as e:
                    print(f"[bsk_MiniMaxDirector][concat] group[{gi}] 마지막 프레임 실패, 다음 그룹 수용 불가: {e}")
                    prev_last_tensor = None
            else:
                prev_last_tensor = None
            all_images.append(images)
            all_audio.append(seg_audio)

        # 각 구간 이미지를 시간 차원에 따라 통합(dim=0)병합, 오디오 파형 샘플 차원 따라(dim=2)붙이기。
        # 반환 목록 길이=1，유지 OUTPUT_IS_LIST=(True,True) 계약 (프론트엔드 순회는 1개의 비디오만 획득)）。
        if not all_images:
            return [], []
        import torch
        merged_images = torch.cat(all_images, dim=0)
        # 오디오 병합: 각 구간 샘플링 레이트 일치 후 파형 연결
        merged_audio = all_audio[0]
        if len(all_audio) > 1:
            sample_rate = all_audio[0].get("sample_rate")
            for seg_audio in all_audio[1:]:
                if seg_audio.get("sample_rate") != sample_rate:
                    raise ValueError(
                        "bsk_MiniMaxDirector: concat 각 단편 오디오 샘플링률이 일치하지 않아 단일 비디오로 합칠 수 없습니다"
                    )
            wavs = [a.get("waveform") for a in all_audio if a.get("waveform") is not None]
            if wavs:
                merged_waveform = torch.cat(wavs, dim=2)
                merged_audio = {"waveform": merged_waveform, "sample_rate": sample_rate}
        return [merged_images], [merged_audio]

    # ---------- 단일 fl2v/i2v/t2v 샘플링 (모드 A 공용 단위）----------
    def _run_one_fl2v_group(self, first_frame_path, last_frame_path, task_key,
                            frames, prompt, clip, vae, audio_vae, model, width, height,
                            steps, cfg, shift_video, shift_audio,
                            sampler_name, scheduler, seed, preview_quality, unique_id,
                            first_frame_crop=None, last_frame_crop=None,
                            unload_clip=False, unload_model=False,
                            inter_mode=None, group_index=0,
                            preview_fps=24, preview_frames=97, sampling_stages=None):
        """공식 MiniMaxH3ImageToVideo 단일 샘플링 (듀얼 하드 앵커 포인트)）。

        - fl2v：first_frame + last_frame 듀얼 하드 앵커 (공식 내부 latent[:,0]/[:, -1] 하드 앵커)）。
        - i2v：단 first_frame。
        - t2v：프레임 없음。
        prompt 강화된 fl2v_prompt로 프레임 시작/종료 제약 설정 (꼬리 프레임 있음)=fl2v 프리픽스, 그렇지 않으면 i2v 프리픽스）。
        frames 누르다 17k+5 격자 정렬 (min 의 task_key）。

        중간 생성물：
        - use 패턴: 시작/종료 프레임 로드 안 함, CLIP 실행 안 함, 데이터 파일에서 group_index 기준으로 직접 읽기；
        - only 모드: CLIP 정상 실행, 결과 누적 inter_mode["_saves"]，1 프레임의 자리 표시기 검은색 이미지를 반환합니다 (샘플링 건너띡니다)）。
        """
        inter_mode = inter_mode or {"use": False, "only": False, "name": "",
                                    "_saves": [], "_entries": None}
        _use_inter = bool(inter_mode.get("use") and inter_mode.get("name"))
        _only_inter = bool(inter_mode.get("only"))
        _gstart = time.perf_counter()
        _push_stage("로드 시작 프레임", "첫 프레임/마지막 프레임 자료 읽기…", unique_id)
        if _use_inter:
            # 사용 중간 결과: 조건 데이터 캐시됨, CLIP 인코딩 실행 불필요; 하지만 시작/종료 프레임 고정은 샘플링 단계에 전달해야 함：
            # - 자체 첫 번째 프레임 자료: 프런트엔드에서 first_frame_path(경로)를 삽입하여 정상적으로 로드합니다.；
            # - 프레임 수용: _run_concat 상류가 firstFrameAuto로 결정하여 이전 그룹의 실제 마지막 프레임 텐서를 사용
            #   first_frame_path 전들어오세요, 여기직접沿용（torch.Tensor 건너뛰기자르기，크기이미정렬）；
            # - 마지막 프레임은 동일하게 입력 값을 사용하고 fl2v 이중 고정 앵커를 유지합니다. + prompt 프리픽스 use 섹션에도 적용。
            first_frame = None
            if first_frame_path is not None:
                first_frame = (first_frame_path if isinstance(first_frame_path, torch.Tensor)
                               else _load_image_tensor(first_frame_path))
            last_frame = None
            if last_frame_path is not None:
                last_frame = (last_frame_path if isinstance(last_frame_path, torch.Tensor)
                              else _load_image_tensor(last_frame_path))
            has_end = last_frame is not None
            seg_frames = int(frames or 97)
        else:
            # 인터프레임 소스에서 이전 그룹에서 생성된 마지막 프레임 텐서(torch.Tensor)의 크기를 사용합니다.，
            # 첫 번째 프레임 크롭 슬롯을 적용해서는 안 됩니다(first_frame_crop는 이 그룹의 첫 번째 프레임 이미지에만 유의)）。
            _ff_crop = None if isinstance(first_frame_path, torch.Tensor) else first_frame_crop
            first_frame = None
            if first_frame_path is not None:
                first_frame = _load_image_tensor(first_frame_path)
                # 바닥 자르기: 명시적 자르기 우선; 자르기 없음 및 연속 프레임이 아닌 경우, 목표 비율 내에 래핑 박스로 자동 자르기，
                # 그렇지 않으면 공식 노드가 전체 원본 이미지를 대상 해상도로 늘려 비율이 맞지 않아 왜곡될 수 있습니다 (예: 가로 이미지가 세로 이미지로 압축됨)）。
                if not isinstance(first_frame_path, torch.Tensor):
                    first_frame = _crop_or_default(first_frame, _ff_crop, width, height)
            last_frame = None
            if last_frame_path is not None:
                last_frame = _crop_or_default(_load_image_tensor(last_frame_path), last_frame_crop, width, height)
            _timing("리소스 로딩 (프레임 시작/끝))", _gstart)
            has_end = last_frame is not None
            # task_key 정규화: fl2v는 반드시 꼬리 프레임이 있어야 함, fl2v만 있는 경우 퇴화 i2v
            if task_key == "fl2v" and not has_end:
                task_key = "i2v"
            elif task_key == "i2v" and has_end:
                # 사용자가 꼬리 프레임 이미지를 명시적으로 추가함 → 중간 실패 시에도 일부 파일이 포함된 것을 생성합니다 + fl2v prompt 프리피 확정
                task_key = "fl2v"
            min_frames = (MIN_FL2V_FRAMES if task_key == "fl2v"
                          else (MIN_I2V_FRAMES if task_key == "i2v" else MIN_T2V_FRAMES))
            # 단일 프레임/매우 짧은 프레임 (LOW_FRAME_FRAMES) 사용"단 프레임 출력"：격자 정렬 하한선 없이 직접 전달 1/5，
            # 공식 노드 temporal_shape([length]) 원본 지원 length=1/5（견 _empty_av_latent）。
            seg_frames = (int(frames) if int(frames or 97) in LOW_FRAME_FRAMES
                          else _align_frames(int(frames or 97), min_frames))
            prompt = reinforce_fl2v_prompt(prompt or "", has_end_frame=has_end)

        _cstart = time.perf_counter()
        positive, latent = self._resolve_conditioning(
            inter_mode, group_index, clip, vae, prompt,
            width, height, seg_frames, first_frame, last_frame,
        )
        _timing(f"CLIP Condition editor({task_key})", _cstart,
                " [중간 생성물]" if (_use_inter or _only_inter) else "")
        _push_stage("샘플링 준비", "SigmaShift / 샘플러 초기화…", unique_id)
        if _only_inter:
            # 중간 결과물만 생성: UNet 샘플링/디코딩 건너뛰기, 플레이스홀더 프레임 반환 + 오디오 음소거를 다운스트림 SaveVideo에 전달합니다.
            print(f"[bsk_MiniMaxDirector][intermediate] group[{group_index}] 조건을 계산하고 샘플링을 건너뜁니다(중간 결과물만 생성)）", flush=True)
            return self._placeholder_images(height, width), self._silent_audio(1, inter_mode.get("_fps", 24))
        if _use_inter:
            # 데이터 파일에서 로드된 조건은 기본적으로 CPU에서 실행되며, 샘플링 전에 모델 장치로 이동합니다 (공식 경로의 positive는 자체적으로 장치에 존재합니다)）
            dev = getattr(model, "load_device", None)
            if dev is not None:
                positive = _to_device(positive, dev)
                latent = _to_device(latent, dev)
        # CLIP 소모됨: 메모리 부족 시 즉시 완전 삭제, 후속 샘플링/디코딩 메모리 확보 (반복 구간에서 구간별 자동 재로딩)；
        # only 패턴은 프로세스 끝에 언로드하여 여러 구간 간 반복적인 재로딩을 방지합니다.）
        if unload_clip and not _only_inter:
            _u0 = time.perf_counter()
            _unload_clip_model(clip)
            _timing("アンロード CLIP", _u0)
        # 공식 MiniMaxH3ImageToVideo.execute 는 (positive, latent) 두 개의 요소를 반환합니다(독립적인 negative 조건이 없습니다).，
        # 네이티브 링크 사용 SamplerCustomAdvanced + BasicGuider(positive)，의존 안 함 negative/cfg）。
        _push_stage("샘플링 중", "비디오 프레임 생성 중…", unique_id)

        # 각 프레임 그룹의 미리보기 프레임 수는 해당 프레임 그룹의 비디오 프레임 수와 정렬됩니다(concat 각 구간의 길이는 다를 수 있으며, 프로젝트 총 프레임 수를 사용할 수 없습니다)），
        # 보장 해당 그룹 프리뷰 지속 시간 = 이 비디오 길이 (프레임 수/fps), 재생 속도와 비디오 일치。
        group_pf = max(1, min(128, int(seg_frames or preview_frames)))
        # SigmaShift（shift_video/shift_audio）샘플 구간 반복은 _sample_decode 내에서 구간별로 실행，
        # 따라서 원본 모델 (shift되지 않음)을 전달하고 각 부분은 각각 clone。
        images, seg_audio = self._sample_decode(
            model=model, seed=seed, steps=steps, cfg=cfg,
            sampler_name=sampler_name, scheduler=scheduler,
            shift_video=shift_video, shift_audio=shift_audio,
            positive=positive, latent=latent, vae=vae, audio_vae=audio_vae,
            audio_latent_needed=True,
            unique_id=unique_id, preview_quality=preview_quality,
            unload_model=unload_model,
            preview_fps=preview_fps, preview_frames=group_pf,
            sampling_stages=sampling_stages,
        )
        _timing(f"싱글 그룹 총 소요 시간({task_key})", _gstart)
        return images, seg_audio

    # ---------- 모드 B / 이전 ref: 여러 참조 자료 참조 ----------
    def _run_refs(self, td, references, clip, vae, audio_vae, model, width, height,
                  total_frames, global_prompt, steps, cfg, shift_video, shift_audio,
                  sampler_name, scheduler, seed, preview_quality, unique_id,
                  unload_clip=False, unload_model=False, inter_mode=None,
                  preview_fps=24, preview_frames=97, sampling_stages=None):
        """MiniMaxH3ReferenceToVideo：이미지/비디오/오디오 멀티모달 참조。

        v6 새 계약：td["ref_images"]/["ref_videos"]/["ref_video_audios"]/["ref_audios"]（배열 + slot From index）；
        오래된 v5 호환：td["references"]（포함 kind/slot/file）。
        수량 보호: 이미지≤9、비디오≤3、각별한 오디오/독립 오디오≤3（공식 모델 사양 상한）。

        중간 결과물: use 모드에서는 참조 자료 로딩 및 CLIP을 건너뜁고, only 모드에서는 계산 후 저장하고 대체 프레임을 반환합니다.。
        """
        inter_mode = inter_mode or {"use": False, "only": False, "name": "",
                                    "_saves": [], "_entries": None}
        _use_inter = bool(inter_mode.get("use") and inter_mode.get("name"))
        _only_inter = bool(inter_mode.get("only"))
        # 공식 ref 노드는 audio_vae(비디오/오디오 참조 또는 생성 소리 모두 필요); 순수 이미지 참조도 연결 audio_vae。
        # use 모드 조건이 캐시됨, 더 이상 audio_vae로 조건 생성 필요 (하지만 디코딩 오디오는 여전히 필요, 누락되면 무음으로 오디오 없음)）。
        if audio_vae is None and not _use_inter:
            raise ValueError(
                "bsk_MiniMaxDirector: mode=ref（참조 생 비디오) 연결이 필요합니다 audio_vae。"
                "MiniMax H3 latent 내장 오디오 채널, 생성된 비디오는 항상 소리 있음。"
            )
        ref_images_map: dict = {}
        ref_videos_map: dict = {}
        ref_video_audios_map: dict = {}
        ref_audios_map: dict = {}
        # 참조 이미지 미리 확대 최대값 (픽셀，0=제한 없음/공식 매치 바닥): 사용자는 공식 바닥 값보다 더 작은 임계값을 설정할 수 있습니다
        # ref 토큰 수를 제어합니다(각 단계 샘플링 시 DiT 전향에 참여하여 추론 시간이 직접 영향을 미칩니다)）。
        ref_image_max_pixels = int(td.get("ref_image_max_pixels") or 0)
        _rstart = time.perf_counter()
        _push_stage("참조 자료 로드", "읽기 다중 모드 참조 자료…", unique_id)

        def _load_ref_image(f, crop):
            """로드 ref 이미지: 명시적 크롭 시에는 잘라내고, 크롭이 없으면 전체 이미지를 유지 (공식 ref 노드 등각 비율 조정, 하옵션만)，
            변형되지 않음, 첫 번째 및 마지막 프레임과 같이 대상 비율에 맞게 내부 자르기를 하지 않아도 됨）→ 선택적 프리 스케일링 →
            공식 최종 크기 정렬(32 floor 정렬, 공식 수신 후 동일, 보이는 대로 얻기),）。"""
            t = _load_image_tensor(f)
            if crop:
                t = _apply_crop_tensor(t, crop)
            if ref_image_max_pixels > 0:
                t = _limit_pixels(t, ref_image_max_pixels)
            t = _align_ref_to_official(t, width, height, td.get("ref_image_size", "match"))
            return t

        def _load_ref_video(f, crop):
            """참조 비디오 로드: 명시적 크롭 시 프레임별로 자르기; 크롭 없음 시 전체 이미지 유지 (공식 adapt_canvas 등비율 스케일링)）。"""
            t = _load_video_frames(f)
            if crop:
                t = _apply_crop_tensor(t, crop)
            return t

        if not _use_inter:
            # 새 v6 ref_* 배열 (인덱스 즉 슬롯); ref_image_crops/ref_video_crops 동일 인덱스 선택 가능 자르기
            for i, f in enumerate((td.get("ref_images") or [])[:9]):
                if f:
                    # crops 변환 그라데이션 프레임 (::before) - 호버 시 표시, 블러 효과 제거하여 성능 향상. null），
                    # 범위 초과 안전한 값 가져오기, 희소 슬롯 배열 피하기 IndexError。
                    _crops = td.get("ref_image_crops") or []
                    crop = _crops[i] if i < len(_crops) else None
                    ref_images_map[f"ref_image_{i}"] = _load_ref_image(f, crop)
            for i, f in enumerate((td.get("ref_videos") or [])[:3]):
                if f:
                    _crops = td.get("ref_video_crops") or []
                    crop = _crops[i] if i < len(_crops) else None
                    ref_videos_map[f"ref_video_{i}"] = _load_ref_video(f, crop)
            for i, f in enumerate((td.get("ref_video_audios") or [])[:3]):
                if f:
                    ref_video_audios_map[f"ref_video_audio_{i}"] = _load_audio(f)
            for i, f in enumerate((td.get("ref_audios") or [])[:3]):
                if f:
                    ref_audios_map[f"ref_audio_{i}"] = _load_audio(f)

            # 구 v5 참조 호환（kind + slot）
            for ref in (references or []):
                kind = ref.get("kind")
                file = ref.get("file")
                slot = int(ref.get("slot", 0))
                if not file:
                    continue
                if kind == "image":
                    ref_images_map[f"ref_image_{slot}"] = _load_ref_image(file, ref.get("crop"))
                elif kind == "video":
                    ref_videos_map[f"ref_video_{slot}"] = _load_ref_video(file, ref.get("crop"))

        _timing("리소스 로드", _rstart,
                f" (img={len(ref_images_map)} vid={len(ref_videos_map)} aud={len(ref_audios_map) + len(ref_video_audios_map)})")
        _push_stage("인코딩 참조 조건", "참조 이미지/비디오/오디오 특징 인코딩…", unique_id)
        # 단일 프레임/매우 짧은 프레임 (LOW_FRAME_FRAMES) 사용"단 프레임 출력"：ref 노드 조건 없이 참조 안커점 추가，
        # Native Support length=1/5，그리드 정렬 하한선에 따라 진행 안 함。
        seg_frames = (int(total_frames) if int(total_frames or 97) in LOW_FRAME_FRAMES
                      else _align_frames(total_frames, MIN_FL2V_FRAMES))
        prompt = td.get("prompt") or global_prompt or ""
        ref_image_size = td.get("ref_image_size", "match")

        _cstart = time.perf_counter()
        if _use_inter:
            positive, latent = self._resolve_conditioning(
                inter_mode, 0, clip, vae, prompt, width, height, seg_frames, None, None,
            )
        else:
            positive, latent = self._resolve_conditioning(
                inter_mode, 0, clip, vae, prompt, width, height, seg_frames, None, None,
                ref_kwargs=dict(
                    clip=clip, vae=vae, audio_vae=audio_vae, prompt=prompt,
                    width=width, height=height, length=seg_frames,
                    ref_image_size=ref_image_size,
                    ref_images=ref_images_map or None,
                    ref_videos=ref_videos_map or None,
                    ref_video_audios=ref_video_audios_map or None,
                    ref_audios=ref_audios_map or None,
                ),
            )
        _timing("CLIP Condition editor(ref)", _cstart, " [중간 생성물]" if (_use_inter or _only_inter) else "")
        _push_stage("샘플링 준비", "SigmaShift / 샘플러 초기화…", unique_id)
        if _only_inter:
            # 중간 결과물만 생성: UNet 샘플링/디코딩 건너뛰기, 플레이스홀더 프레임 반환 + 음소거 오디오
            print("[bsk_MiniMaxDirector][intermediate] ref 경로 계산 조건, 샘플링 건너뛰기 (중간 결과물만 생성)）", flush=True)
            return [self._placeholder_images(height, width)], [self._silent_audio(1, inter_mode.get("_fps", 24))]
        if _use_inter:
            dev = getattr(model, "load_device", None)
            if dev is not None:
                positive = _to_device(positive, dev)
                latent = _to_device(latent, dev)
        # CLIP 메모리 부족 시 즉시 완전히 언로드 (only 모드에서는 프로세스 끝에 언로드).）
        if unload_clip and not _only_inter:
            _u0 = time.perf_counter()
            _unload_clip_model(clip)
            _timing("アンロード CLIP", _u0)
        # At MM:SS.mmm, 접두사 삽입. [Shot N]을 추가하여 완전한 씬 블록을 구성하고, '주체 참조'와 함께 사용하여 특정 시점에 주체의 동작을 설명합니다.）
        _push_stage("샘플링 중", "비디오 프레임 생성 중…", unique_id)

        # SigmaShift（shift_video/shift_audio）샘플 구간 반복은 _sample_decode 내에서 구간별로 실행，
        # 따라서 원본 모델 (shift되지 않음)을 전달하고 각 부분은 각각 clone。
        images, seg_audio = self._sample_decode(
            model=model, seed=seed, steps=steps, cfg=cfg,
            sampler_name=sampler_name, scheduler=scheduler,
            shift_video=shift_video, shift_audio=shift_audio,
            positive=positive, latent=latent, vae=vae, audio_vae=audio_vae,
            audio_latent_needed=True,
            unique_id=unique_id, preview_quality=preview_quality,
            unload_model=unload_model,
            preview_fps=preview_fps, preview_frames=preview_frames,
            sampling_stages=sampling_stages,
        )
        _timing("ref 총 소요 시간", _rstart)
        return [images], [seg_audio]

    # ================= 생성 미리보기/결과 캐시 (세션 간 복원 가능)） =================
    # 대상: 과정 프레임 + 루트 디렉터리가 하나 있으며, 직접 사용 user/default/bsk_video_studio/runs/，
    # 프론트엔드 '어떤 방식으로든 비디오 워크스테이션 재개방'이 마지막 프로세스 미리보기와 최종 결과로 복원되도록 합니다.。
    # - 프로세스 프레임 key = unique_id（감독 노드 인스턴스 ID, process 내에서 직접 얻을 수 있으며 prompt_id）。
    # - 최종 결과(비디오 파일)는 이 노드에 저장되지 않으며, 프론트엔드에서 검색합니다. ComfyUI /history/<prompt_id>。
    # - prompt_id 프론트엔드에서 성공적으로 제출된 후 POST /vs_runs/track을 덮어쓰고 있습니다 (process 내에서 가져올 수 없음) prompt_id）。
    _RUN_MAX_KEEP = 5          # 최근 N회만 보존, 무한 증가 방지
    _RUN_FLUSH_SEC = 0.5       # 메모리 버퍼 데진션 간격 (초)）
    _RUNS_BUF = {}             # unique_id -> {"frames":[...], "last_flush":float, "meta":dict}

    @staticmethod
    def _runs_dir() -> str:
        """위치 캐시 디렉토리: user/default/bsk_video_studio/runs/ output/bsk_vs_runs）。"""
        try:
            import folder_paths
            base = os.path.join(folder_paths.get_user_directory(), "default", "bsk_video_studio")
        except Exception:
            base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bsk_video_studio")
        d = os.path.join(base, "runs")
        try:
            os.makedirs(d, exist_ok=True)
        except Exception:
            pass
        return d

    @classmethod
    def _runs_path(cls, unique_id: str) -> str:
        return os.path.join(cls._runs_dir(), f"{unique_id}.json")

    @classmethod
    def _runs_last_path(cls) -> str:
        return os.path.join(cls._runs_dir(), "last.json")

    @classmethod
    def _runs_ensure_dir(cls):
        try:
            os.makedirs(cls._runs_dir(), exist_ok=True)
        except Exception:
            pass

    @classmethod
    def _runs_write_run(cls, unique_id: str, meta: dict, frames: list, flush: bool = False):
        """증량 쓰기 <unique_id>.json（프로세스 프레임 포함）。flush=True 강제 디스크에 쓰기, 그렇지 않으면 의존성 디바운스 버퍼。

        추가된 첫 번째 프레임의 ‘전역 단일 조화 지수 i’(프레임 추가 시 반환) -1）。
        i 지속등은 해당 프레임의 frames 배열 인덱스이며, 모듈과 프런트 사이의 유일한 권위 있는 포인터입니다.：
        WebSocket 실시간 프레임과 HTTP 증분 보간 프레임 모두 동일한 i를 사용하며, 프런트엔드는 i를 기준으로 포인터를 진행합니다.
        「한 프레임만 렌더링하고, 링크가 끊어지면 다른 링크를 '무난하게 보완'하여, 오류가 발생하기 쉬운 프레임 수를 비교하지 않습니다.。
        """
        buf = cls._RUNS_BUF.get(unique_id)
        now = time.time()
        base = -1
        if buf is None:
            buf = {"frames": list(frames), "last_flush": 0.0, "meta": dict(meta)}
            cls._RUNS_BUF[unique_id] = buf
            if frames:
                base = 0
        else:
            if frames:
                base = len(buf["frames"])
                buf["frames"].extend(frames)
            buf["meta"].update(meta)
        if base >= 0:
            # 타임스탬프: frames 와 buf["frames"] 동일한 dict 객체이며, 이 위치에 할당하면 버퍼 및 디스크에 기록된 내용이 동기화됩니다.
            for k, fr in enumerate(frames):
                try:
                    fr["i"] = base + k
                except Exception:
                    pass
        if not flush and (now - buf["last_flush"]) < cls._RUN_FLUSH_SEC:
            return base
        buf["last_flush"] = now
        try:
            cls._runs_ensure_dir()
            payload = {
                "unique_id": unique_id,
                "prompt_id": buf["meta"].get("prompt_id") or "",
                "run_id": buf["meta"].get("run_id") or _CURRENT_RUN_ID,
                "status": buf["meta"].get("status", "running"),
                "frames": buf["frames"],
                "result": buf["meta"].get("result"),
                "updated_at": now,
            }
            tmp = cls._runs_path(unique_id) + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
            os.replace(tmp, cls._runs_path(unique_id))
        except Exception as e:
            print(f"[bsk_MiniMaxDirector][runs] 캐시 쓰기 실패 {unique_id}: {e}", flush=True)
        return base

    @classmethod
    def _runs_write_last(cls, unique_id: str, prompt_id: str = "", status: str = "running"):
        try:
            cls._runs_ensure_dir()
            payload = {"unique_id": unique_id, "prompt_id": prompt_id or "",
                       "status": status, "updated_at": time.time()}
            tmp = cls._runs_last_path() + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
            os.replace(tmp, cls._runs_last_path())
        except Exception as e:
            print(f"[bsk_MiniMaxDirector][runs] last.json 쓰기 실패: {e}", flush=True)

    @classmethod
    def _runs_prune(cls, keep: int = None):
        """최근 keep 번 실행 파일만 유지 (last.json 제외), 무한 증가 방지。"""
        keep = keep or cls._RUN_MAX_KEEP
        try:
            d = cls._runs_dir()
            files = [f for f in os.listdir(d)
                     if f.endswith(".json") and f != "last.json"]
            def _mtime(fn):
                try:
                    return os.path.getmtime(os.path.join(d, fn))
                except Exception:
                    return 0
            files.sort(key=_mtime, reverse=True)
            for fn in files[keep:]:
                try:
                    os.remove(os.path.join(d, fn))
                except Exception:
                    pass
        except Exception:
            pass

    @classmethod
    def _runs_track_prompt(cls, unique_id: str, prompt_id: str):
        """프론트엔드 제출이 성공적으로 완료되면 prompt_id를 last.json 및 해당 run 파일에 씁니다.。"""
        if not unique_id:
            return
        try:
            lp = cls._runs_last_path()
            data = {}
            if os.path.isfile(lp):
                with open(lp, "r", encoding="utf-8") as f:
                    data = json.load(f)
            if data.get("unique_id") == unique_id:
                data["prompt_id"] = prompt_id or ""
                cls._runs_write_last(unique_id, prompt_id, data.get("status", "running"))
            rp = cls._runs_path(unique_id)
            if os.path.isfile(rp):
                with open(rp, "r", encoding="utf-8") as f:
                    rd = json.load(f)
                rd["prompt_id"] = prompt_id or ""
                with open(rp, "w", encoding="utf-8") as f:
                    json.dump(rd, f, ensure_ascii=False)
        except Exception as e:
            print(f"[bsk_MiniMaxDirector][runs] track_prompt 실패 {unique_id}: {e}", flush=True)

    @classmethod
    def _runs_begin(cls, unique_id: str):
        """process 시작: 이번 실행 초기화 (상태 running), 기록 last。"""
        if not unique_id:
            return
        cls._RUNS_BUF.pop(unique_id, None)
        cls._runs_write_run(unique_id, {"status": "running", "prompt_id": "",
                                        "run_id": _CURRENT_RUN_ID}, [], flush=True)
        cls._runs_write_last(unique_id, "", "running")
        cls._runs_prune()

    @classmethod
    def _runs_finish(cls, unique_id: str, status: str = "sampled"):
        """process 정상 종료 (샘플링 완료): 디스크 상태 저장 + 메모리 버퍼를 지웁니다. 결과는 전방향에서 확인합니다 /history。"""
        if not unique_id:
            return
        # 프레임이 프레임별로 누적됨 _RUNS_BUF[unique_id]["frames"] None일 경우 단일 구간으로 퇴화합니다 (기존 동작과 동일)，
        # 그렇지 않으면 _runs_write_run이 buf["frames"].extend(buf["frames"]) 자기 복제 시 프레임 수 두 배로 증가。
        cls._runs_write_run(unique_id, {"status": status, "run_id": _CURRENT_RUN_ID}, [], flush=True)
        cls._RUNS_BUF.pop(unique_id, None)
        lp = cls._runs_last_path()
        if os.path.isfile(lp):
            try:
                with open(lp, "r", encoding="utf-8") as f:
                    d = json.load(f)
                if d.get("unique_id") == unique_id:
                    cls._runs_write_last(unique_id, d.get("prompt_id", ""), status)
            except Exception:
                pass

    @classmethod
    def _runs_fail(cls, unique_id: str):
        """process 예외: 태그 failed。"""
        if not unique_id:
            return
        # 동 _runs_finish_: 프레임이 버퍼에 쌓여 있으며, 여기서는 상태만 업데이트하고 더 이상 프레임 자체 복제 방지 기능을 전파하지 않습니다.。
        cls._runs_write_run(unique_id, {"status": "failed", "run_id": _CURRENT_RUN_ID}, [], flush=True)
        cls._RUNS_BUF.pop(unique_id, None)
        lp = cls._runs_last_path()
        if os.path.isfile(lp):
            try:
                with open(lp, "r", encoding="utf-8") as f:
                    d = json.load(f)
                if d.get("unique_id") == unique_id:
                    cls._runs_write_last(unique_id, d.get("prompt_id", ""), "failed")
            except Exception:
                pass

    # ================= 캐시 모듈 종료 =================

    @staticmethod
    def _build_sampling_stages(td, steps, sampler_name, scheduler, shift_video, shift_audio):
        """timeline_data.sampling을 샘플링 구간 목록(길이 1 또는 2）。

        계약 (프론트엔드 vs_submit.buildMiniMaxTimelineData 쓰기; 기본적으로 상위 층의 이전 스칼라로 되돌아가며, 호환성을 유지한다)）：
        sampling = {
          "stage1": {"samplerName", "scheduler", "steps", "denoise", "shiftVideo", "shiftAudio", "sigmas"},
          "stage2": {"enabled": bool, 백엔드 API 변환 성공, 노드 수 "mode", "upscale", "scaleBy", "method"}
        }
        - sigmas 비어 있지 않음 → 사용자 정의 sigma(scheduler/steps/denoise 무시, 둘 다 상호 배타적) 사용）；
        - 그렇지 않으면 BasicScheduler(scheduler, steps, denoise)를 사용하여 생성；
        - stage1.denoise 항상 1.0 (완전 노이즈 제거)）；stage2.denoise<1 중간 sigma 이어짐 (기본값) 0.4）；
        - stage2.mode：
            "split"（단계 연속）——두 단락이 하나의 총 sigma 시퀀스를 공유합니다(총 단계 수). = stage1.steps + stage2.steps，
                stage1.steps 기준으로 자동 분할 (두 번째 구간 시작점) == 첫 번째 단계 종료, 근사 변위 없음(); 이 모드에서는
                stage2 Scheduler/shift follows stage1 (scale must be consistent), denoise is meaningless；
            "independent"（기본）——stage2 자신의 스케줄러 사용 + denoise 시작 지점 결정 (0.4 정교화 / 1.0 처음부터 재실행）；
        - stage2.upscale：샘플링 전 두 번째로**비디오** latent scaleBy 배율(오디오 구성 요소 및 조건 내 오디오는 변경되지 않음)），
          method ∈ nearest-exact/bilinear/area/bicubic/bislerp；조건의 키프레임/참조 시각 latent 동일 배율 동기화。
        """
        def _seg(src, default_denoise):
            src = src if isinstance(src, dict) else {}

            def _num(key, fallback):
                v = src.get(key)
                return float(fallback if v is None or v == "" else v)

            return {
                "samplerName": str(src.get("samplerName") or sampler_name or "euler"),
                "scheduler": str(src.get("scheduler") or scheduler or "simple"),
                "steps": int(src.get("steps") or steps or 20),
                "denoise": _num("denoise", default_denoise),
                "shiftVideo": _num("shiftVideo", shift_video),
                "shiftAudio": _num("shiftAudio", shift_audio),
                "sigmas": src.get("sigmas") or "",
                "mode": str(src.get("mode") or "independent"),
                "upscale": bool(src.get("upscale")),
                "scaleBy": _num("scaleBy", 1.5),
                "method": str(src.get("method") or "bilinear"),
            }

        sp = td.get("sampling") if isinstance(td.get("sampling"), dict) else {}
        stages = [_seg(sp.get("stage1"), 1.0)]
        # 첫 번째는 항상 완전한 노이즈 제거: 잠재 공간은 비어 있음 latent，denoise<1 시그마 중간에서 시작 → 결과는 반드시 파괴됨
        stages[0]["denoise"] = 1.0
        s2 = sp.get("stage2") if isinstance(sp.get("stage2"), dict) else {}
        if s2 and bool(s2.get("enabled")):
            stages.append(_seg(s2, 0.4))
        return stages

    def _sample_decode(self, model, seed, steps, cfg, sampler_name, scheduler,
                       shift_video, shift_audio,
                       positive, latent, vae, audio_vae, audio_latent_needed,
                       unique_id=None, preview_quality="off", unload_model=False,
                       preview_fps=24, preview_frames=97, sampling_stages=None):
        """공유: 샘플링（1~2 변） → 비디오 디코딩 → 오디오 디코딩. 반환 (images, seg_audio)。

        ⚠️ 샘플링은 MiniMax H3 원본 링크를 통해 진행해야 합니다.（SamplerCustomAdvanced + BasicGuider +
        KSamplerSelect + BasicScheduler + RandomNoise），**사용 불가 comfy.sample.sample**：
        H3 의 latent는 av-pack NestedTensor(비디오+오디오 패키징 (포함되지 않음), 그리고 comfy.sample.sample /
        comfy.samplers.sample 그 세트의 구식 샘플링은 latent_image에 대해 일반 텐서를만 지원합니다 (내부적으로 의도적으로 디자인되어 FindWindowW가 찾을 수 없게 되어 단일 인스턴스 활성화가 작동하지 않도록 합니다).
        NestedTensor에 대해 .dim() 호출/다차원 인덱싱 → 크래시), 그리고 출력을 생성하지 않습니다 {"samples":...} dict
        To downstream。

        H3 원시 링크는 cfg / negative가 없습니다(BasicGuider 내부) cfg=1.0 단 self-guidance），
        cfg 파라미터는 이 링크에서 무의미하며 무시됨。

        ⚠️ 충돌 복구（2026-08-06）：**사용하지 마세요**LTXVSeparateAVLatent을 호출하여 오디오와 비디오를 미리 분리합니다. latent。
        공식 VAEDecode.decode (nodes.py L330-333) 및 vae_decode_audio（nodes_audio.py
        L98-101）각자 내부적으로 NestedTensor에 대해 unbind()[0] / unbind()[-1]를 사용하여 비디오/오디오 컴포넌트를 분리했습니다.。
        일반 5D 텐서로 먼저 분리한 다음 전달하면 두 번 처리됩니다. unbind →
        "too many indices for tensor of dimension 5"。sampled dict를 직접 둘 다에 제공。

        ⚠️ 입력 유형: VAEDecode.decode(vae, samples)와 VAEDecodeAudio.execute(audio_vae, samples)
        의 샘플 파라미터 모두 포함 "samples" key dict(내부 samples["samples"] 취하다 NestedTensor），
        전체 샘플링 딕셔너리를 전송하는 것이 아닙니다 sampled["samples"]。

        샘플링 구간 (2026-09-12 추가, 기존 단 구간 호환)）：
        - sampling_stages CSV에서 해당 항목이 없는 경우：shift/sampler/scheduler/steps/denoise=1）；
        - 여러 구간을 순차적으로 실행: N번째 구간은 N-1번째 구간의 출력 latent를 입력으로 사용(H3의) av-pack NestedTensor
          두 번째 샘플링은 공식 링크에서 원시적으로 지원합니다: comfy.sample.fix_empty_latent_channels와 prepare_noise
          모두 is_nested에 대한 별도 분기 처리）；
        - 각 구간은 sampler / sigma / shift를 각각 지정할 수 있으며, '이중 샘플링 디테일 강조' 또는
          「사용자 지정 sigma 역추적 고동력 찢어짐 방지」；
        - 두 종류의 두 단계 관계（stage2.mode）：
            · "split"「단계 연속」——두 단락이 하나의 총 sigma 시퀀스를 공유합니다(총 단계 수).=두 단계 수의 합 (첫 번째 단계 수대로)
              사용자 정의 스타일 오프닝 (원래 출력, 일반적으로 상단 드롭다운 사용)"시간 절약 분할 실행"；
            · "independent"「독립」——두 번째 단락은 자체 스케줄러 사용 + denoise 시작 지점 (0.4 수정 / 1.0 재실행）；
        - 두 번째 단계는 샘플링 전에 비디오 latent에 공간 확대를 수행할 수 있습니다(stage2.upscale/scaleBy/method, 비디오 컴포넌트만 사용).，
          조건의 키프레임/참조 시각 latent 동일 배율 동기화）→ 생태적 주류 '잠재 공간 확대 + 낮은 시그마 두 채널, 고화질 복구；
        - 각 덩어리에 대해 shift가 있는 모델(MiniMaxH3SigmaShift)을 복제하고 공유 BaseModel，
          unload_model은 마지막 구간의 클론을 함께 해제。

        생성 과정 미리보기：preview_quality != 'off' 샘플링 전에 각 구간의 모델에 걸치
        OUTER_SAMPLE wrapper，각 단계마다 x0 비디오 랭귀지를 여러 프레임으로 디코딩 (우선 Latent2RGB 선형 변환)，
        진 VAE로 되돌립니다), 렌더링하여 애니메이션 WebP로 저장하고, 통해 PromptServer.send_sync(
        "kj_preview_override", payload, None) 모든 클라이언트에 브로드캐스트 (재개/새 패널 포함)）。
        프론트엔드 panel.js가 해당 이벤트를 전달함 → vs_submit.js → vs_preview.js showKjPreview（링크가 준비되었습니다）。
        """
        # 원래 워크플로우 링크 복제 (노드) 127/128/129/130/131）：
        #   RandomNoise(seed) → noise
        #   BasicGuider(model, positive) → guider（없다 uncond）
        #   KSamplerSelect(sampler_name) → sampler
        #   BasicScheduler(model, scheduler, steps, denoise) → sigmas（사용자 정의 sigma 시 직접 적용）
        #   SamplerCustomAdvanced(noise, guider, sampler, sigmas, latent) → {samples: NestedTensor}
        stages = [s for s in (sampling_stages or [{
            "samplerName": sampler_name, "scheduler": scheduler, "steps": steps,
            "denoise": 1.0, "shiftVideo": shift_video, "shiftAudio": shift_audio, "sigmas": "",
        }]) if isinstance(s, dict)]
        # 단계 연속（stage2.mode == 'split'）：두 단락이 하나의 총 sigma 시퀀스를 공유합니다(총 단계 수). = 두 단계 수의 합，
        # stage1.steps에 따라 자동으로 분할 → 두 번째 시작점 == 첫 번째 종료 지점 (정확한 연결, 근사 오차 없음)）。
        # 스케일은 첫 번째 구간의 shift/스케줄러에 의해 결정되므로, 이 모드에서는 두 번째 구간의 shift/스케줄러가 첫 번째 구간을 따릅니다 (frontend는 grayed out 힌트 표시)）。
        # 어떤 단계에서 사용자 지정 sigma를 입력하면 자동으로 분할되지 않습니다 (사용자 지정 sigma는 명시적인 사용자 의도입니다)）。
        chain_sigmas = None
        if (len(stages) > 1 and str(stages[1].get("mode") or "") == "split"
                and not stages[0].get("sigmas") and not stages[1].get("sigmas")):
            _cstart = time.perf_counter()
            _chain_model = _out(MiniMaxH3SigmaShift.execute(
                model,
                float(stages[0].get("shiftVideo") or shift_video or 12.0),
                float(stages[0].get("shiftAudio") or shift_audio or 3.0)))[0]
            _total = int(stages[0].get("steps") or 20) + int(stages[1].get("steps") or 10)
            chain_sigmas = _build_sigmas(_chain_model, stages[0].get("scheduler"), _total, 1.0, None)
            if chain_sigmas is None or len(chain_sigmas) < 3:
                chain_sigmas = None
            else:
                stages[1]["scheduler"] = stages[0]["scheduler"]
                stages[1]["shiftVideo"] = stages[0]["shiftVideo"]
                stages[1]["shiftAudio"] = stages[0]["shiftAudio"]
                stages[1]["denoise"] = 1.0
                _cut = min(len(chain_sigmas) - 2, max(1, int(stages[0].get("steps") or 20)))
                print("[bsk_MiniMaxDirector][sampling] 단계 연속: 총 단계 "
                      f"{stages[0]['steps']}+{stages[1]['steps']}（스케줄러 {stages[0]['scheduler']}），"
                      f"분할점 sigma={float(chain_sigmas[_cut]):.4f}", flush=True)
            _timing("계속 시그마 계획", _cstart)

        _dstart = time.perf_counter()
        noise = RandomNoise.execute(seed)[0]
        _timing("샘플링 전위(noise)", _dstart)

        sampled = None
        latent_in = latent
        for si, st in enumerate(stages):
            tag = "" if si == 0 else f"(제{si + 1}변)"
            # Second segment (configurable) enlarge video latent before sampling = 「잠재 공간 확대 + 샘플링 '후 행 hires fix。
            # 단순히 비디오 컴포넌트만 확대(오디오 컴포넌트는 원본 그대로), 조건 내의 키프레임/참조 시각 latent 동일 해상도로 동기화，
            # 그렇지 않으면 앵커/참조와 확대된 캔버스 비율이 일치하지 않음 (구도 꼬임, 얼굴 붕괴)）。
            if si > 0 and st.get("upscale"):
                _ustart = time.perf_counter()
                _sc = float(st.get("scaleBy") or 1.5)
                _um = str(st.get("method") or "bilinear")
                _push_stage("잠재 공간 확대", f"비디오 latent x{_sc:g}（{_um}）…", unique_id)
                latent_in = _upscale_av_latent(latent_in, _sc, _um)
                positive = _upscale_conditioning_visuals(positive, _sc, _um)
                _timing("잠재 공간 확대", _ustart, f" (x{_sc:g} {_um})")

            # 각 섹션을 복제 (원래 모델 사용, 두 번째 패치 중첩 방지)）
            _mstart = time.perf_counter()
            model_shifted = _out(MiniMaxH3SigmaShift.execute(
                model,
                float(st.get("shiftVideo") or shift_video or 12.0),
                float(st.get("shiftAudio") or shift_audio or 3.0)))[0]
            _timing(f"SigmaShift{tag}", _mstart)

            sampler = KSamplerSelect.execute(st.get("samplerName") or "euler")[0]
            if chain_sigmas is not None:
                # 단계 연속: 첫 번째 구간 s1 단계, 두 번째 구간 해당 sigma에서부터 달리기 0
                _n1 = min(len(chain_sigmas) - 1, max(1, int(stages[0].get("steps") or 20)))
                sigmas = chain_sigmas[:_n1 + 1] if si == 0 else chain_sigmas[_n1:]
            else:
                sigmas = _build_sigmas(model_shifted, st.get("scheduler"), st.get("steps"),
                                       st.get("denoise"), st.get("sigmas"))
            if sigmas is None:
                if si == 0:
                    raise RuntimeError(
                        "bsk_MiniMaxDirector: 샘플링 시그마가 비어 있음 (사용자 정의 시그마 무효 또는 denoise<=0），불가능한 샘플링")
                print(f"[bsk_MiniMaxDirector][sampling] 제{si + 1}sigma가 비어 있어 건너뜁니다", flush=True)
                break

            # 생성 과정 미리보기 래퍼를 연결합니다(멱등: 고정 키를 사용하여 반복적으로 연결하면 겹쳐지므로 먼저 제거한 다음 연결）
            # guider 생성 전에 반드시 래핑해야 함: guider는 model_shifted 참조를 보유하고, 샘플링 시 이를 통해
            # prepare_model_patcher model.wrappers를 model_options에 동기화하면 wrapper가 활성화됨。
            # 설계 원칙: 워크플로우가 항상 멈추지 않도록 KJ 미리보기 노드, 이 wrapper는 무조건적인 유일한 미리보기 소스。
            if (preview_quality and preview_quality != "off" and unique_id is not None):
                _attach_minimax_preview(model_shifted, unique_id, vae, sigmas, preview_quality,
                                        preview_fps=preview_fps, preview_frames=preview_frames)

            guider = BasicGuider.execute(model_shifted, positive)[0]
            _sstart = time.perf_counter()
            sampled = SamplerCustomAdvanced.execute(noise, guider, sampler, sigmas, latent_in)[0]
            _timing(f"UNET 샘플링{tag}", _sstart, f" (steps={len(sigmas) - 1})")
            latent_in = sampled

        if sampled is None:
            raise RuntimeError("bsk_MiniMaxDirector: 트리거 단어, 반복 10회")
        _push_stage("VAE 비디오 디코딩", "샘플링 완료, 비디오 프레임 디코딩 중…", unique_id)
        # sampled 이다 {"samples": NestedTensor} dict。VAEDecode.decode / VAEDecodeAudio.execute
        # 각자 내부에서 NestedTensor에 대해 unbind()를 수행하여 텐서를 분리하지 않고 샘플링된 dict를 직접 전달합니다. 미리 분리할 필요가 없습니다.。

        # 샘플링 완료, VAE 디코딩 전: 메모리 부족 시 UNET 완전히 제거(SigmaShift 클론과 원 model
        # 공유 동일 BaseModel, unload_model_and_clones는 함께 언로드합니다, GPU 메모리를 디코더에 할당합니다。
        if unload_model:
            _u0 = time.perf_counter()
            try:
                model_management.unload_model_and_clones(model_shifted, unload_additional_models=False)
                _timing("アンロード UNET", _u0)
            except Exception as e:
                print(f"[bsk_MiniMaxDirector][unload] UNET 언로드 실패: {e}", flush=True)

        _vstart = time.perf_counter()
        vae_dec = VAEDecode()
        images = vae_dec.decode(vae, sampled)[0]
        _timing("VAE 비디오 디코딩", _vstart)

        seg_audio = None
        if audio_vae is not None:
            try:
                _push_stage("VAE 오디오 디코딩", "비디오 디코딩 완료, 오디오 디코딩 중…", unique_id)
                _astart = time.perf_counter()
                ad = _out(VAEDecodeAudio.execute(audio_vae, sampled))
                seg_audio = ad[0] if ad else None
                _timing("VAE 오디오 디코딩", _astart)
                # absolute는 stage의 축소에 따라 조정되지 않으며, 어떤 max-height도 이미지 하반부를 잘라냅니다
                # SaveVideo 붕괴. 방어적 세척: 비정상 샘플만 음소거, 정상 환경에 영향 없음。
                if seg_audio is not None:
                    seg_audio = _sanitize_audio_waveform(seg_audio)
            except Exception:
                seg_audio = None
        return images, seg_audio


# ---------- 생성 프로세스 미리보기: OUTER_SAMPLE wrapper(자체 구현, KJ 플러그인에 의존하지 않음)）----------
# 원리 일치 ComfyUI-KJNodes의 ModelPreviewOverrideKJ: 각 단계에서 x0 비디오 컴포넌트를 분해 JPEG，
# kj_preview_override 이벤트를 프론트로 전송합니다(프론트엔드 링크) panel.js → vs_submit.js →
# vs_preview.js showKjPreview 모두 준비 완료）。
# 디코딩 전략(2026-08-11 수정): 우선 Latent2RGB 선형 변환(latent_rgb_factors, CPU 경량).，
# 腾 칩 플랫폼 안정적, ComfyUI 원시 get_previewer / KJ와 동일 경로, rgb_factors가 없을 때만 되돌림
# 진 VAE 디코딩——원래 구현에서는 각 단계마다 vae.decode를 전체 해상도로 디코딩했습니다. 昇腾 fp16 VAE 값 문제로 인해 검은 화면/정적 화면이 발생했습니다.
# 실패 (프레임 없음) 및 각 단계 소요 시간 ~13s 샘플링 속도 3배 느리게 설정。
_PREVIEW_WRAPPER_KEY = "bsk_minimax_preview"

# 기어 → 미리보기 최대 변위(픽셀). low 384 / medium 512 / high 0(확대되지 않음, 원본 해상도)）。
_PREVIEW_MAX_RES = {"low": 384, "medium": 512, "high": 0}


def _decode_x0_frames(vae, x0, max_res, latent_format=None, latent_shapes=None, max_frames=97):
    """특정 단계의 x0 비디오 구성 요소를 "이 단계의 전체 비디오의 여러 프레임" PIL 목록으로 디코딩하고, KJ의 동적 미리보기 메커니즘과 일치시킵니다.。

    디코딩 아이디어 참조 ComfyUI-KJNodes (kijai, GPL-3.0) ModelPreviewOverrideKJ：
    각 단계에서 해당 단계의 예측된 전체 비디오 잠재 변수를 여러 프레임으로 디코딩하여 독립적으로 애니메이션(webp/mp4)으로 인코딩합니다.
    preview_fps 재생, 프론트엔드 <img>/<video> 다중 프레임 자동 반복 → 시각적으로는 '이 단계 비디오 콘텐츠'의 동적 미리보기，
    단프레임 정적, 또한 단계를 건너는 노이즈 프레임 축적 (축적은 프레임 속도 불일치/재생 속도 증가)）。

    이 함수는 '이 단계의 다중 프레임을 디코딩'합니다: 우선 Latent2RGB 선형 변환（latent_rgb_factors，
    CPU 경량, 昇腾 안정적이며 ComfyUI의 원시 Latent2RGBPreviewer / KJ와 동일한 경로에 있습니다.），
    rgb_factors가 없을 때만 True VAE 디코딩으로 되돌아감. 코드는 독립적으로 구현. list[(PIL.Image, w, h)]를 반환하고, 실패하면 []。
    """
    # 일관된 비디오 쿼티（av-pack NestedTensor → unbind()[0]）reshape 되도록 5D
    try:
        if getattr(x0, "is_nested", False):
            x0 = x0.unbind()[0]
        # 3D flattened av-pack（[B,3,C*T*H*W]）→ latent_shapes[0] reshape 실행 5D
        # [B,C,T,H,W]，KJ와 정렬 _normalize_packed_x0
        if x0.ndim == 3 and latent_shapes and len(latent_shapes) > 0:
            target = latent_shapes[0]
            if len(target) >= 3:
                cut = 1
                for d in target[1:]:
                    cut *= int(d)
                x0 = x0[:, :, :cut].reshape([x0.shape[0]] + list(target)[1:])
        if x0.ndim != 5:
            return []
        x0 = x0[0]  # [C,T,H,W]
    except Exception as e:
        print(f"[bsk_MiniMaxDirector][preview] x0 비디오 양량 실패: {e}", flush=True)
        return []

    T = x0.shape[1]
    # 항상 latent의 T개의 시간 프레임을 균등하게 샘플링/확장하여 max_frames 프레임으로 만듭니다（max_frames=preview_frames）。
    # 핵심 수정 (2026-08-13): MiniMax H3의 latent 시간 차원 T는 일반적으로 프로젝트 총 프레임 수보다 훨씬 작습니다 (VAE 시간 압축)，
    # 예: 97 프레임 비디오 latent는 ~25 프레임).  만약 T<max_frames 범위(T)만 취함, 애니메이션은 T 프레임 수，
    # preview_fps로 재생하는 반복 시간 = T/fps << 비디오 총 길이 = 총 프레임 수/fps, 미리보기는"비디오보다 2배 빠름"。
    # T 프레임을 균등하게 재샘플링하여 max_frames 프레임으로 만듭니다(T 시간 인덱스가 자연스럽게 반복되고, 구현"내용 유지, 총 지속 시간 맞춤"），
    # 반복 시간 설정 = max_frames/preview_fps = 비디오 총 길이, 재생 속도와 최종 비디오 일치。
    import numpy as _np
    if max_frames <= 1:
        pick = [0]
    else:
        pick = [int(round(i)) for i in _np.linspace(0, T - 1, max_frames)]
        # 중복 제거 T>=max_frames 시간 동안 프레임 손실 없음；T<max_frames 중복 인덱스를 유지하여 프레임 수를 보충합니다
        if len(pick) > max_frames:
            pick = pick[:max_frames]

    out = []

    # 경로 A: Latent2RGB 선형 변환 (선택 사항). [C,T,H,W] 전체에 대해 선형 변환을 수행합니다 (KJ에 맞춰 정렬).
    # _decode_video_frames_l2rgb：x0[0] → movedim(0,-1) → linear → clamp → uint8），프레임별 비교
    # 반복이 더 효율적이고 의미가 일치합니다. MiniMax H3 latent 24ch, latent_rgb_factors는 24×3（없다 reshape）。
    if latent_format is not None:
        try:
            rgb_factors = getattr(latent_format, "latent_rgb_factors", None)
            if rgb_factors is not None:
                reshape = getattr(latent_format, "latent_rgb_factors_reshape", None)
                x_rgb = x0
                if reshape is not None:
                    x_rgb = reshape(x_rgb)
                factors = torch.tensor(rgb_factors, dtype=torch.float32).transpose(0, 1)  # [3,C]
                bias = None
                rgb_bias = getattr(latent_format, "latent_rgb_factors_bias", None)
                if rgb_bias is not None:
                    bias = torch.tensor(rgb_bias, dtype=torch.float32)
                # x_rgb: [C,T,H,W] → movedim(0,-1) → [T,H,W,C]
                xm = x_rgb.detach().float().cpu().movedim(0, -1)  # [T,H,W,C]
                rgb = torch.nn.functional.linear(xm, factors, bias=bias)  # [T,H,W,3]
                rgb = rgb.add_(1.0).mul_(127.5).clamp_(0, 255).to(torch.uint8)
                rgb_np = rgb.numpy()
                for fi in pick:
                    pil = Image.fromarray(rgb_np[fi], "RGB")
                    if max_res and max_res > 0 and (pil.width > max_res or pil.height > max_res):
                        pil = ImageOps.contain(pil, (max_res, max_res), Image.LANCZOS)
                    out.append((pil, pil.width, pil.height))
                return out
        except Exception as e:
            print(f"[bsk_MiniMaxDirector][preview] Latent2RGB 다중 프레임 미리보기 실패, 롤백 VAE: {e}", flush=True)

    # 경로 B: True VAE 디코딩 멀티 프레임 (롤백)）
    try:
        # 중복 제거 후 한 번만 디코딩하고, pick 순서대로 재사용하여 중복 방지 T<<max_frames 시간 프레임 수일 때 전체 시간 프레임을 가져옴
        uniq_pick = list(dict.fromkeys(pick))
        idx_of = {v: k for k, v in enumerate(uniq_pick)}
        frames_latent = x0[:, uniq_pick, :, :]  # [C, n, H, W]
        with torch.no_grad():
            img = vae.decode(frames_latent)  # [B,n,H,W,3] 또는 [B,n,C,H,W]
        if img.ndim == 5:
            img = img.reshape(-1, img.shape[-3], img.shape[-2], img.shape[-1])
        for fi in pick:
            img0 = img[idx_of[fi]].detach().cpu().float()
            if img0.dim() == 3 and img0.shape[0] in (1, 3):
                img0 = img0.movedim(0, -1)  # CHW → HWC
            img0 = img0.clamp(0.0, 1.0)
            pil = Image.fromarray((img0.numpy() * 255).astype("uint8"))
            if pil.mode != "RGB":
                pil = pil.convert("RGB")
            if max_res and max_res > 0 and (pil.width > max_res or pil.height > max_res):
                pil = ImageOps.contain(pil, (max_res, max_res), Image.LANCZOS)
            out.append((pil, pil.width, pil.height))
        return out
    except Exception as e:
        print(f"[bsk_MiniMaxDirector][preview] VAE 멀티 프레임 프리뷰 디코딩 실패: {e}", flush=True)
        return []


def _encode_animated_webp(frames_pil, fps, quality=85):
    """여러 프레임의 PIL을 애니메이션 WebP의 base64로 변환합니다 (프론트엔드) <img> 다중 프레임 자동 반복 = 동적 프리뷰）。
    정렬 KJ _encode_animated_webp：save_all + 프레임별 duration，loop=0 무한 루프。
    단프레임 시에도 WebP (브라우저) <img> 또한 정적 webp를 표시할 수 있으며, 의미가 통일되어 JPEG 브랜치를 줄입니다.）。
    """
    if not frames_pil:
        return None, 0, 0
    durations = max(1, int(round(1000.0 / max(1, fps))))
    w = frames_pil[0].width
    h = frames_pil[0].height
    buf = pyio.BytesIO()
    try:
        frames_pil[0].save(
            buf, format="WEBP", save_all=True, append_images=frames_pil[1:],
            duration=durations, loop=0, minimize_size=True, lossless=False, quality=quality,
            method=4,
        )
    except Exception:
        # 일부 구형 Pillow 라이브러리는 method 매개변수를 지원하지 않으므로, 되돌아가서 처리합니다. method
        buf = pyio.BytesIO()
        frames_pil[0].save(
            buf, format="WEBP", save_all=True, append_images=frames_pil[1:],
            duration=durations, loop=0, minimize_size=True, lossless=False, quality=quality,
        )
    return base64.b64encode(buf.getvalue()).decode("ascii"), w, h


def _attach_minimax_preview(model_patcher, node_id, vae, sigmas, preview_quality,
                            preview_fps=24, preview_frames=97):
    """model_patcher에 OUTER_SAMPLE wrapper를 추가하고, 각 단계마다 kj_preview_override 이벤트를 전송。

    사전 뷰 메커니즘은 ComfyUI-KJNodes (kijai, GPL-3.0) ModelPreviewOverrideKJ의 "동적 비디오 사전 뷰"」：
    - 각 단계에서 ‘이 단계’ x0 비디오 잠재 변수를 여러 프레임(균일 샘플링 preview_frames 시간 프레임)으로 디코딩합니다.）；
    - 각 단계별로 애니메이션으로 독립 코딩 WebP（duration=1000/fps），preview_fps 기준으로 재생；
    - 프론트엔드 <img> 다중 프레임 자동 반복 = 이 단계 비디오 콘텐츠의 동적 미리보기 (올바른 프레임률, 단계 간 누적 아님)）。
    코드는 독립적으로 구현(wrapper는 OUTER_SAMPLE를 통해 ComfyUI patcher_extension을 감싸서 재사용) 공개 API）。

    멱등성: 먼저 동일한 key의 이전 wrapper를 제거한 후 추가하고, 중복 샘플링 시 중복이 쌓이는 것을 방지합니다。
    wrapper 서명 정렬 comfy.patcher_extension.WrapperExecutor：
        __call__(executor, noise, latent_image, sampler, sigmas, denoise_mask,
                 callback, disable_pbar, seed, latent_shapes)
    callback 6번째 위치 매개변수(index 5), 자신에게 대체 new_callback，
    new_callback 먼저 원래 콜백을 호출하고 (ComfyUI 기본 미리보기/진행률 표시줄 유지), 그런 다음 kj 이벤트를 보냅니다.。

    설계 원칙: 워크플로우가 항상 멈추지 않도록 KJ 미리보기 노드, 이 wrapper는 무조건적인 유일한 미리보기 소스。

    preview_fps / preview_frames process에서 timeline_data.frameRate / totalFrames에서 파생，
    예보기 재생 속도가 최종 비디오와 일치하도록 보장합니다(프레임 속도 동기화), 프레임 수 동기화 프로젝트 총 프레임 수(최대 128）。
    """
    max_res = _PREVIEW_MAX_RES.get(preview_quality, 384)
    nid = str(node_id) if node_id is not None else None
    sigmas_list = sigmas.detach().cpu().tolist() if sigmas is not None else []
    total_steps = max(0, len(sigmas_list) - 1)
    # Latent2RGB Preview use: Take the model latent_format (including latent_rgb_factors) to avoid each step's true VAE decoding
    latent_format = getattr(getattr(model_patcher, "model", None), "latent_format", None)
    state = {"last_time": None, "step_ms_window": [], "last_x0": None, "warned": False,
             "preview_frames": preview_frames, "preview_fps": preview_fps}

    # 발사 step=0 초기화 메시지 (프론트엔드 showKjPreview) step===0 표시"준비 중"）
    if nid is not None:
        try:
            PromptServer.instance.send_sync("kj_preview_override", {
                "node_id": nid, "run_id": _CURRENT_RUN_ID,
                "step": 0, "total": total_steps,
                "sigma": sigmas_list[0] if sigmas_list else None,
                "sigmas": sigmas_list,
                "fps": preview_fps,
            }, None)  # client_id=None 모든 클라이언트에 브로드캐스트 (재개/새 패널 포함)）
        except Exception:
            pass

    def new_callback(step, x0, x, total_steps_, latent_shapes=None):
        if nid is None or PromptServer is None:
            return
        try:
            # 1) 이 단계 x0 비디오 잠재 변수 디코딩은 "이 단계 전체 비디오" 다프레임 (KJ 정렬: 각 단계 독립 애니메이션，
            #    단계 간 누적 안 함, 프레임 속도 오류/재생 속도 빠름 방지）
            decoded = _decode_x0_frames(vae, x0, max_res, latent_format, latent_shapes,
                                        max_frames=state["preview_frames"])
            if not decoded:
                if not state["warned"]:
                    state["warned"] = True
                    print(f"[bsk_MiniMaxDirector][preview] step={step + 1}/{total_steps_} 사전 보기 디코딩 실패, 프로세스 프레임 없음 (후속 단계 중복 출력 안 함)）", flush=True)
                return
            # 2) 이 단계는 독립적으로 애니메이션 WebP로 인코딩(preview_fps에 따라 재생, 단일 프레임 시에도 WebP 정적 프레임 출력)）
            frames_pil = [pil for pil, _, _ in decoded]
            webp_b64, w, h = _encode_animated_webp(frames_pil, state["preview_fps"])
            if webp_b64 is None:
                return

            # 3) 시간 소요 통계
            now = time.perf_counter()
            step_ms = None
            if state["last_time"] is not None:
                step_ms = (now - state["last_time"]) * 1000.0
                wnd = state["step_ms_window"]
                wnd.append(step_ms)
                if len(wnd) > 8:
                    wnd.pop(0)
            state["last_time"] = now
            avg_step_ms = (sum(state["step_ms_window"]) / len(state["step_ms_window"])) if state["step_ms_window"] else None

            # 4) delta：상위 단계 x0 대비 변화
            delta_v = None
            try:
                x0_cpu = x0.detach().float().cpu()
                if getattr(x0, "is_nested", False):
                    x0_cpu = x0_cpu.unbind()[0]
                prev = state["last_x0"]
                state["last_x0"] = x0_cpu
                if prev is not None and prev.shape == x0_cpu.shape:
                    diff = x0_cpu - prev
                    delta_v = (diff.norm() / max(1, diff.numel()) ** 0.5).item()
            except Exception:
                pass

            # 5) 디스크에 쓰기: 이 단계의 애니메이션 webp (fps 메타데이터 포함)를 메모 버퍼에 추가 (디할림 디스크 쓰기)），
            #    프론트엔드가 세션 복원 시 올바른 프레임 속도로 미리보기 재생。
            #    해당 프레임의 전역 인덱스 동시에 복구 i —— WS 실시간 프레임과 HTTP 증분 프레임 보수 프레임이 공유하며, 이를 위한 idempotent 포인터를 사용합니다.。
            rec = {
                "node_id": nid,
                "run_id": _CURRENT_RUN_ID,
                "step": step + 1, "total": total_steps_,
                "mime": "image/webp", "image": webp_b64, "w": w, "h": h,
                "step_ms": step_ms, "avg_step_ms": avg_step_ms,
                "fps": state["preview_fps"],
                "i": -1,
            }
            try:
                base_i = BskMiniMaxDirector._runs_write_run(nid, {}, [rec])
                if isinstance(base_i, int) and base_i >= 0:
                    rec["i"] = base_i
            except Exception:
                pass

            # 6) 브로드캐스트 모든 클라이언트（client_id=None）：패널/멀티 클라이언트에서 실시간 미리보기 프레임 수신
            sigma_val = sigmas_list[step] if 0 <= step < len(sigmas_list) else None
            try:
                PromptServer.instance.send_sync("kj_preview_override", {
                    "node_id": nid,
                    "run_id": rec["run_id"],
                    "i": rec["i"],
                    "image": webp_b64,
                    "mime": "image/webp",
                    "w": w, "h": h,
                    "step": step + 1,
                    "total": total_steps_,
                    "sigma": sigma_val,
                    "sigmas": None,
                    "delta": delta_v,
                    "step_ms": step_ms,
                    "avg_step_ms": avg_step_ms,
                    "fps": state["preview_fps"],
                }, None)
            except Exception:
                pass
        except Exception:
            pass

    def outer_sample_wrapper(executor, *args, **kwargs):
        args = list(args)
        cb_idx = 5  # outer_sample(noise, latent_image, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes)
        orig_cb = args[cb_idx] if len(args) > cb_idx else kwargs.get("callback")
        ls_idx = 9  # latent_shapes：3D flattened av-pack reshape 5D로 되돌리기 필요한 (정렬 KJ）
        latent_shapes_arg = args[ls_idx] if len(args) > ls_idx else kwargs.get("latent_shapes")

        def chained_cb(step, x0, x, total_steps):
            # 먼저 원래 콜백을 호출하고 ComfyUI 기본 미리보기를 유지합니다 + 진행 표시줄）
            if orig_cb is not None:
                orig_cb(step, x0, x, total_steps)
            new_callback(step, x0, x, total_steps, latent_shapes_arg)

        if len(args) > cb_idx:
            args[cb_idx] = chained_cb
        else:
            kwargs["callback"] = chained_cb
        return executor(*args, **kwargs)

    # 멱등성: 먼저 동일한 키의 기존 wrapper를 제거한 다음 새 wrapper를 추가합니다(중복 샘플링/클론 시나리오).
    try:
        model_patcher.remove_wrappers_with_key(
            comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, _PREVIEW_WRAPPER_KEY)
    except Exception:
        pass
    model_patcher.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, _PREVIEW_WRAPPER_KEY, outer_sample_wrapper)


# ---------- ComfyUI 단일 파일 노드 등록 (호환 새 스타일) io.ComfyNode）----------
try:
    NODE_CLASS_MAPPINGS = {"bsk_MiniMaxDirector": BskMiniMaxDirector}
    NODE_DISPLAY_NAME_MAPPINGS = {"bsk_MiniMaxDirector": "BSK MiniMax 감독"}
except Exception:
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}
