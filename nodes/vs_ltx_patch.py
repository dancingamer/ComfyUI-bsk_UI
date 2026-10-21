# SPDX-License-Identifier: AGPL-3.0-or-later
"""
BSK Video Studio — 실행 중 monkey-patch 레이어(자르기) + 음량）。

책임: WhatDreamsCost-ComfyUI/ltx_director.py 소스 코드를 수정하지 않고，
실행 중에 모듈 수준 함수 래핑：

1) `_load_image_tensor` / `_load_video_tensor`
   프론트엔드로 전달된 `seg["bskCrop"]`（원래 이미지 픽셀 좌표) 정확한 자르기，
   그 후 원 플러그인에 의해 `_resize_image` 등비율로 대상 해상도로 조정。

2) `_build_combined_audio`
   이 함수(ltx_director.py L585-734)는 각 단락을 해독한 후
   `out_waveform[...] += clip_waveform`（L728）순수한 additive 합，
   어떤 보정 계수도 없고, clamp 없음. 이 패치는 프론트엔드로 전달된
   `audioSegments[i]["bskGain"]`（선형 곱셈，1.0=원래 음량) 음량 적용，
   그리고 믹싱 결과에서 컷오프 보호를 수행합니다。

발동: 프론트에서 제출 전 호출 `GET /bsk/vs/ensure_crop_patch` Lazy loading。
이 라우트는 ComfyUI가 완전히 시작된 후에 실행되며, 지연 위험이 있음。

좌표 시스템(프론트엔드 Canvas 미리보기와 완전히 일치해야 함)）：
  x, y : 비디오: 정상적으로 합성하고 MP4 비디오 파일 저장
  w, h : 자르기 상자 너비 및 높이, 원본 자재 픽셀 정수
  비율: w:h는 params.width:params.height와 항상 같아야 합니다 (프론트엔드 잠금)）

기본: seg 없음 "bskCrop" 필드 → 크롭하지 않음; 없음 "bskGain" 필드 → 볼륨 변경 안 함。
두 노드에 교환할 수 없는 연결이 있습니다。
"""

import json
import os
import sys

try:
    import folder_paths
except Exception:  # ComfyUI 외부 (예: 단위 테스트)를 껍데기로 낮추어 가져오기 시 붕괴 방지
    folder_paths = None

_MARK = "__bsk_crop_patched__"
_MARK_AUDIO = "__bsk_gain_patched__"

# Normalize when peak value exceeds the threshold after mixing, to prevent multi-segment clipping
_PEAK_LIMIT = 1.0


def _apply_crop(tensor, crop):
    """픽셀 좌표로 텐서 자르기, 좌표 경계 클램핑。

    디버깅: 잘라내기 여부, 원본 이미지 크기, 활성화 잘라내기 사각형, 잘라낸 후 해상도 출력。
    """
    if not crop:
        return tensor
    try:
        N, H, W, _ = tensor.shape
    except Exception:
        return tensor

    orig_w, orig_h = W, H
    x = int(crop.get("x", 0))
    y = int(crop.get("y", 0))
    w = int(crop.get("w", 0))
    h = int(crop.get("h", 0))

    # 경계 클리핑 전 요청 좌표 (오류 표시용)）
    req = (x, y, w, h)
    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))
    w = max(1, min(w, W - x))
    h = max(1, min(h, H - y))

    cropped = tensor[:, y:y + h, x:x + w, :]
    print(
        f"[BSK crop] APPLY_CROP: src={os.path.basename(str(crop.get('_src', '')))} "
        f"orig={orig_w}x{orig_h} requested={req} "
        f"clamped=(x={x},y={y},w={w},h={h}) -> cropped={w}x{h}",
        flush=True,
    )
    return cropped


def _seg_gain(seg):
    """증가율 (선형 배수) 읽기. 기본/유효하지 않음 → 1.0。제한까지 [0, 4]。"""
    if not isinstance(seg, dict):
        return 1.0
    raw = seg.get("bskGain", None)
    if raw is None:
        return 1.0
    try:
        g = float(raw)
    except (TypeError, ValueError):
        return 1.0
    if g != g:  # NaN
        return 1.0
    return max(0.0, min(4.0, g))


def install_audio_gain(module):
    """포장 `_build_combined_audio`，audioSegments[].bskGain(멱등)）。

    구현 핵심: 원 함수는 입력 timeline_data **JSON 문자열**그리고 내부에서 완료
    디코딩 / 재샘플링 / 자르기 / additive 혼합. 중복 구현을 피하기 위해 복잡한 로직을 사용합니다.，
    이곳에서 증폭 값으로 오디오 세그먼트를 그룹화하고, 각 그룹에 대해 '해당 그룹 세그먼트만 포함'을 timeline JSON
    원래 함수 호출 후 각 그룹 결과에 각 증폭률 곱하여 합산。

    - 모든 Segment Boost는 1.0 → 직접 투사 원 함수, 추가 비용 없음。
    - override_audio=True motionSegments(이 프로젝트는 lock false), 참여하지 않음。
    """
    if getattr(module, _MARK_AUDIO, False):
        return False
    if not hasattr(module, "_build_combined_audio"):
        return False

    _orig_audio = module._build_combined_audio

    def _patched_build_combined_audio(timeline_data_str, start_frame, duration_frames,
                                      frame_rate, override_audio=False):
        # override_audio 분기 사용 motionSegments, 본 프로젝트는 활성화되지 않았으며, 직접 전달
        if override_audio or not timeline_data_str:
            return _orig_audio(timeline_data_str, start_frame, duration_frames,
                               frame_rate, override_audio)

        try:
            data = json.loads(timeline_data_str)
        except Exception:
            return _orig_audio(timeline_data_str, start_frame, duration_frames,
                               frame_rate, override_audio)

        if not isinstance(data, dict) or data.get("retakeMode", False):
            return _orig_audio(timeline_data_str, start_frame, duration_frames,
                               frame_rate, override_audio)

        segs = data.get("audioSegments") or []
        if not isinstance(segs, list) or not segs:
            return _orig_audio(timeline_data_str, start_frame, duration_frames,
                               frame_rate, override_audio)

        # 증강 모듈로 그룹화; 전체 1.0은 개입이 필요 없습니다
        groups = {}
        for seg in segs:
            g = _seg_gain(seg)
            groups.setdefault(g, []).append(seg)

        if len(groups) == 1 and abs(next(iter(groups)) - 1.0) < 1e-6:
            return _orig_audio(timeline_data_str, start_frame, duration_frames,
                               frame_rate, override_audio)

        import torch  # 지연 로드하여 환경에 영향을 미치지 않도록 torch 로드 방지

        mixed = None
        sample_rate = 44100
        for gain, group_segs in groups.items():
            if gain <= 0.0:
                continue  # 무음 그룹을 직접 건너뛰고 디코딩 과정을 생략합니다
            sub = dict(data)
            sub["audioSegments"] = group_segs
            try:
                part = _orig_audio(json.dumps(sub), start_frame, duration_frames,
                                   frame_rate, override_audio)
            except Exception:
                continue
            if not isinstance(part, dict) or part.get("waveform") is None:
                continue
            sample_rate = part.get("sample_rate", sample_rate)
            wf = part["waveform"]
            contrib = wf if abs(gain - 1.0) < 1e-6 else wf * gain
            if mixed is None:
                mixed = contrib.clone() if hasattr(contrib, "clone") else contrib
            else:
                n = min(mixed.shape[-1], contrib.shape[-1])
                mixed[..., :n] += contrib[..., :n]

        if mixed is None:
            # 모든 그룹이 건너뛰어집니다 (예: 모든 음소거): 동일 길이의 음소거를 반환합니다.
            base = _orig_audio(timeline_data_str, start_frame, duration_frames,
                               frame_rate, override_audio)
            if isinstance(base, dict) and base.get("waveform") is not None:
                return {"waveform": torch.zeros_like(base["waveform"]),
                        "sample_rate": base.get("sample_rate", sample_rate)}
            return base

        # 파형보호: 피크 초과 한계 시 전체 가우쉬화(상대적인 강도 관계 유지)）
        try:
            peak = float(mixed.abs().max())
            if peak > _PEAK_LIMIT:
                mixed = mixed * (_PEAK_LIMIT / peak)
        except Exception:
            pass

        return {"waveform": mixed, "sample_rate": sample_rate}

    module._build_combined_audio = _patched_build_combined_audio
    setattr(module, _MARK_AUDIO, True)
    return True


def _save_crop_debug(tensor, seg_id, kind):
    """크롭 및 스케일 조정된 텐서를 ComfyUI 출력 디렉터리에 저장하여 수동으로 확인합니다。

    원래 텐서 저장하지 않음 (메모리에 저장하여 다른 슬롯/시간선 블록 재사용）。
    단일 프레임 모드 (i2v), I2V 접두사 사용 None。
    """
    try:
        import torch
        out_dir = folder_paths.get_output_directory()
        (counter, filename, full_path, _) = folder_paths.get_save_image_path(
            f"BSK_crop_{kind}_{seg_id or 'na'}", out_dir
        )
        arr = (tensor[0].detach().cpu().numpy() * 255.0).clip(0, 255).astype("uint8")
        from PIL import Image as _PIL
        _PIL.fromarray(arr).save(full_path)
        print(f"[BSK crop] SAVED {kind} crop -> {full_path} (shape={tuple(tensor.shape)})", flush=True)
        return full_path
    except Exception as e:
        print(f"[BSK crop] SAVE_FAILED {kind} seg={seg_id}: {e}", flush=True)
        return None


def install(module):
    """대상 ltx_director 모듈에 크롭 패치(멱등) 설치.。

    주의: 이 저장소 nodes/bsk_ltx_director.py는 실제로 생성에 참여하는 모듈입니다.（_load_image_tensor
    시간선 로드 프로세스에서 호출됨), 동일하게 크롭 패치를 설치해야 하며 건너뛸 수 없습니다.。
    """
    if getattr(module, _MARK, False):
        return False
    if not hasattr(module, "_load_image_tensor") or not hasattr(module, "_load_video_tensor"):
        return False

    _orig_img = module._load_image_tensor
    _orig_vid = module._load_video_tensor

    def _patched_load_image_tensor(seg):
        t = _orig_img(seg)
        crop = seg.get("bskCrop") if isinstance(seg, dict) else None
        if crop:
            print(f"[BSK crop] IMAGE seg={seg.get('id')} will crop, orig_tensor={tuple(t.shape)}", flush=True)
            t = _apply_crop(t, crop)
            _save_crop_debug(t, seg.get("id"), "img")
        else:
            print(f"[BSK crop] IMAGE seg={seg.get('id')} NO crop", flush=True)
        return t

    def _patched_load_video_tensor(seg, frame_rate):
        t = _orig_vid(seg, frame_rate)
        crop = seg.get("bskCrop") if isinstance(seg, dict) else None
        if crop:
            print(f"[BSK crop] VIDEO seg={seg.get('id')} will crop, orig_tensor={tuple(t.shape)}", flush=True)
            t = _apply_crop(t, crop)
            _save_crop_debug(t, seg.get("id"), "vid")
        else:
            print(f"[BSK crop] VIDEO seg={seg.get('id')} NO crop", flush=True)
        return t

    module._load_image_tensor = _patched_load_image_tensor
    module._load_video_tensor = _patched_load_video_tensor
    setattr(module, _MARK, True)
    return True


def ensure_installed():
    """sys.modules에서 로드된 ltx_director 모듈을 찾아서 모든 패치(커팅)를 설치합니다. + 음량）。

    두 가지 실행 환경과 호환됩니다：
      - 사용자가 WhatDreamsCost를 설치한 ltx_director(모듈 이름에 포함) "whatdreamscost"）；
      - 사용자가 이 저장소 복제본 nodes/bsk_ltx_director.py를 직접 사용합니다(실제 참여 생성, 동일해야 함). patch）。

    모든 endswith("ltx_director") 이미 _load_image_tensor를 구현하여 로드된 모듈 모두 자르기 설치 patch，
    멱등 마커로 중복 설치 방지。
    """
    patched = []
    errors = []
    for name, mod in list(sys.modules.items()):
        if not (name.endswith("ltx_director") and mod is not None):
            continue
        if not hasattr(mod, "_load_image_tensor"):
            continue
        try:
            crop_new = install(mod)
        except Exception as e:
            errors.append(f"crop install failed on {name}: {e}")
            continue
        try:
            gain_new = install_audio_gain(mod)
        except Exception as e:
            errors.append(f"gain install failed on {name}: {e}")
            continue
        patched.append({
            "module": name,
            "crop": "installed" if crop_new else "already patched",
            "gain": "installed" if gain_new else "already patched",
        })
        print(f"[BSK crop] ensure_installed -> patched module={name} "
              f"crop={'installed' if crop_new else 'already'} "
              f"gain={'installed' if gain_new else 'already'}", flush=True)

    if not patched:
        return {"ok": False,
                "error": "ltx_director module not found in sys.modules (target plugin not loaded)",
                "details": errors}
    return {"ok": True, "patched": patched, "errors": errors}
