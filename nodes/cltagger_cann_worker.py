# SPDX-License-Identifier: AGPL-3.0-or-later
#
# BSK Tagger —— onnxruntime-cann 추론 자 프로세스 worker。
# Copyright (C) 2026 灰暗x (B상) @灰暗x). ComfyUI-bsk_UI의 일부，AGPL-3.0-or-later로 배포。
#
# 존재 목적: ComfyUI 메인 프로세스가 torch_npu, torch_npu는 onnxruntime-cann와 동일 프로세스
# 昇腾 ACL 런타임 경쟁(aclInit/aclFinalize 충돌, 참조) opendatalab/MinerU#2547），원인 CANN EP
# 생성 실패 시 CPU 복귀. 이 워커는**로드 안 함 torch_npu** 독립적인 자식 프로세스, NPU 독점 실행 ONNX 추론，
# stdin/stdout의 JSON 행 프로토콜을 통해 메인 프로세스와 통신하여 동일 프로세스 ACL 충돌을 우회합니다.。
#
# 프로토콜 (각 행은 JSON, 요청/응답 일대일 대응)）：
#   요청: {"type":"infer","tagger_type":"wd14"|"cltagger","model_path":str,
#          "max_resolution":int,"image_b64":str(PNG base64)}
#   응답: {"ok":true,"scores":[[...]],"provider":str,"elapsed":float}
#       | {"ok":false,"error":str}
#   실행 핸드셰이크: 첫 번째 줄 출력 {"type":"ready"}
#
# 주의: 이 파일의 모든 로그는 stderr에 작성해야 하며 stdout으로 출력해서는 안 됩니다. 그렇지 않으면 JSON 프로토콜이 손상됩니다.。

import sys
import io
import os
import json
import time
import base64

import numpy as np
from PIL import Image
import onnxruntime as ort


def _ort_session_options():
    """cltagger_node._ort_session_options와 동일한 원천: 명시적인 스레드 수를 통해 클라우드 호스트 코어 바인딩 오류를 피합니다.
    （pthread_setaffinity_np EINVAL），log_severity_level=3 cpuid vendor 경고 제거。
    worker 독립된 서브 프로세스이므로 여기에서 경량 복사본을 유지합니다(주 파일 가져오지 않음, 추가 종속성 방지)）。"""
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


# ---------------------------------------------------------------------------
# 사전 처리 (nodes/cltagger_node.py 내의) _preprocess_* 엄격하게 일치；
# 메인 파일 프리처리 변경 사항이 있으면 반드시 동기화해야 합니다. 그렇지 않으면 NPU 추론 결과와 CPU가 일치하지 않습니다.）
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


# model_path -> (session, desc)，desc 최종 선택된 구성 이름 (로그용)）
_SESSIONS = {}


def _infer_size(session):
    try:
        shape = session.get_inputs()[0].shape
        candidates = [int(v) for v in shape if isinstance(v, int) and v not in (1, 3)]
        if candidates:
            side = max(candidates)
            return (side, side)
    except Exception:  # noqa: BLE001
        pass
    return (448, 448)


def _warmup(session, tagger_type="wd14", max_resolution=1280):
    """따뜻한 물 그림을 한 번 실행하여 미리 CANN 그래프 컴파일 트리거。

    실제 추론과 사용해야 함**완전히 동일**preprocessor 구성 preheat 텐서: cltagger는 EVA02 ViT，
    ONNX 입력 형식: NCHW [N,3,H,W], H/W는 동적 심볼 차원. 이전 구현처럼 심볼 차원 모두 채우지 않음 1，
    생성될 것 1×1 가짜 이미지를 patch-embed 컨볼루션에 입력합니다 (kernel 14) -> Invalid input shape: {1,1}，
    모든 CANN 설정 및 CPU 기본 예열이 모두 실패하여 결국 예열되지 않은 CPU 세션으로 실제 실행
    추론(실측 51초). _infer_size로 실제 가로 세를 얻어 해당 가로 세의 정사각형을 생성하여 표준 전처리 실행，
    동적 차원에 합법적인 크기 유지, CANN 그래프 엔진/단일 연산자 모두 정상적으로 컴파일 가능。
    """
    side = _infer_size(session)[0]
    white = Image.new("RGB", (side, side), (255, 255, 255))
    if tagger_type == "wd14":
        tensor = _preprocess_wd14(white, (side, side), max_resolution)
    else:
        tensor = _preprocess_cltagger(white, (side, side))

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    input_shape = session.get_inputs()[0].shape
    expected_rank = len(input_shape) if input_shape else 4
    if tensor.ndim != expected_rank:
        if tensor.ndim == 4 and expected_rank == 3:
            tensor = np.squeeze(tensor, axis=0)
        elif tensor.ndim == 3 and expected_rank == 4:
            tensor = np.expand_dims(tensor, axis=0)
        else:
            raise ValueError(
                f"[CANN worker] 텐서 사전 가열 rank({tensor.ndim}) 모델 기대값 rank({expected_rank}) 정렬 불가")

    session.run([output_name], {input_name: tensor})


def _ensure_session(model_path, tagger_type="wd14"):
    if model_path in _SESSIONS:
        return _SESSIONS[model_path]

    # CANN 그래프 엔진 일부 모델의 ONNX 그래프 컴파일 실패 (CANNGRAPH failure / GRAPH_FAILED)，
    # cltagger와 같은 ViT 내보내기의 예시（NCHW + 동적 높이/너비(dynamic height/width), WD14는 정상적으로 컴파일됩니다.。
    # 순서대로 설정을 시도하고 첫 번째 프리히트를 실행합니다（=처음 run = 그래픽 컴파일) 성공자 캐시 재사용：
    #   1) 기본 그래프 엔진      —— WD14 이 경로를 따르십시오 (실제 NPU 0.01s, 동작 변동 없음)）
    #   2) 자식 탭 분할 활성화    —— CANN이 실행할 수 있는 서브그래프 자동 분리, 나머지는 버림 CPU
    #   3) 닫기 엔진      —— 문서에 제공된 GRAPH_FAILED 회피 수단, 단일 연산자 추론으로 복귀
    # 모두 실패 시 CPU 사용（=현재, 붕괴되지 않음, 충돌하지 않음）。
    base = {"device_id": 0}
    configs = [
        ("CANN", {}),
        ("CANN(subgraph)", {"enable_cann_subgraph": "1"}),
        ("CANN(single-op)", {"enable_cann_graph": "0"}),
    ]

    last_err = None
    for desc, opts in configs:
        so = dict(base)
        so.update(opts)
        try:
            sess = ort.InferenceSession(
                model_path,
                sess_options=_ort_session_options(),
                providers=[("CANNExecutionProvider", so), "CPUExecutionProvider"],
            )
            _warmup(sess, tagger_type)
            _SESSIONS[model_path] = (sess, desc)
            sys.stderr.write(
                f"[CANN worker] 모델 {model_path} 사용 설정 {desc} (opts={opts})\n")
            return _SESSIONS[model_path]
        except Exception as e:  # noqa: BLE001
            last_err = e
            sys.stderr.write(f"[CANN worker] 설정 {desc} 예열 실패, 다음 항목 시도: {e}\n")
            continue

    # 모든 CANN 설정이 실패함 -> 순수한 CPU로 기본 설정, 최소한 기능 사용 가능, 절대 충돌하지 않음
    try:
        sess = ort.InferenceSession(
            model_path, sess_options=_ort_session_options(),
            providers=["CPUExecutionProvider"])
        _warmup(sess, tagger_type)
        desc = "CPU"
        _SESSIONS[model_path] = (sess, desc)
        sys.stderr.write(
            f"[CANN worker] 모델 {model_path} CPU 롤백 (모든 CANN 설정 실패): {last_err}）\n")
        return _SESSIONS[model_path]
    except Exception as e:  # noqa: BLE001
        sys.stderr.write(f"[CANN worker] CPU만으로도 최저 실패: {e}\n")
        # 극단적인 경우: CPU로도 생성할 수 없으면 세션 캐시를 만들어 호출자가 예외 스택을 가져올 수 있음
        sess = ort.InferenceSession(
            model_path, sess_options=_ort_session_options(),
            providers=["CPUExecutionProvider"])
        _SESSIONS[model_path] = (sess, "CPU")
        return _SESSIONS[model_path]


def _infer(req):
    model_path = req["model_path"]
    tagger_type = req.get("tagger_type", "wd14")
    max_resolution = req.get("max_resolution", 1280)
    pil = Image.open(io.BytesIO(base64.b64decode(req["image_b64"]))).convert("RGB")

    sess, desc = _ensure_session(model_path, tagger_type)
    size = _infer_size(sess)

    if tagger_type == "wd14":
        tensor = _preprocess_wd14(pil, size, max_resolution)
    else:
        tensor = _preprocess_cltagger(pil, size)

    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    input_shape = sess.get_inputs()[0].shape
    expected_rank = len(input_shape) if input_shape else 4
    if tensor.ndim != expected_rank:
        if tensor.ndim == 4 and expected_rank == 3:
            tensor = np.squeeze(tensor, axis=0)
        elif tensor.ndim == 3 and expected_rank == 4:
            tensor = np.expand_dims(tensor, axis=0)
        else:
            raise ValueError(
                f"[CANN worker] 텐서 rank({tensor.ndim}) 모델 기대값 rank({expected_rank}) 정렬 불가")

    t0 = time.time()
    outputs = sess.run([output_name], {input_name: tensor})[0]
    elapsed = time.time() - t0
    return outputs.tolist(), desc, elapsed


def main():
    sys.stdout.write(json.dumps({"type": "ready"}) + "\n")
    sys.stdout.flush()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception as e:  # noqa: BLE001
            sys.stdout.write(json.dumps({"ok": False, "error": f"bad request: {e}"}) + "\n")
            sys.stdout.flush()
            continue
        try:
            scores, provider, elapsed = _infer(req)
            sys.stdout.write(json.dumps({
                "ok": True, "scores": scores, "provider": provider, "elapsed": elapsed
            }) + "\n")
            sys.stdout.flush()
        except Exception as e:  # noqa: BLE001
            sys.stderr.write(f"[CANN worker] infer 실패: {e}\n")
            sys.stdout.write(json.dumps({"ok": False, "error": str(e)}) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
