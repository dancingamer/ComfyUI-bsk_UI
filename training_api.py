# SPDX-FileCopyrightText: 2026 灰暗x
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
ComfyUI 훈련 도구 백엔드 API
sd-scripts (kohya-ss)를 기반으로 하는 모델 훈련 백엔드.

기능:
- 훈련 이미지 디렉토리 스캔
- WD14 배치 태깅
- 훈련 설정 생성 (TOML)
- 훈련 프로세스 관리 (accelerate launch)
- 학습 상태 모니터링 (손실/단계/로그))
- 훈련 샘플 이미지 및 모델 관리
"""

import os
import sys
import json
import subprocess
import threading
import time
import re
import random

import math
import importlib
import signal as signal_module
import shutil
import tempfile
import asyncio
from pathlib import Path

# 토크나이저 병렬 실행을 비활성화하여 포크 경고를 방지합니다
os.environ["TOKENIZERS_PARALLELISM"] = "false"
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# sd-scripts 경로 검색 (우선순위))
_PLUGIN_DIR = os.path.dirname(os.path.realpath(__file__))
_CUSTOM_NODES_DIR = os.path.dirname(_PLUGIN_DIR)

def _detect_comfyui_root():
    """사용자 정의 드롭다운 생성 실패, 원시 컨트롤로 되돌림"""
    # 1. 환경 변수 우선
    env_root = os.environ.get('COMFYUI_ROOT', '')
    if env_root and os.path.isdir(env_root):
        return env_root
    # 2. custom_nodes에서 유도합니다.
    if os.path.basename(_CUSTOM_NODES_DIR) == "custom_nodes":
        return os.path.dirname(_CUSTOM_NODES_DIR)
    # 3. 현재 디렉토리
    return _CUSTOM_NODES_DIR

_COMFYUI_ROOT = _detect_comfyui_root()

_SD_SCRIPTS_SEARCH_PATHS = [
    os.path.join(_PLUGIN_DIR, "sd-scripts"),          # 플러그인 디렉토리 내
    os.path.join(_CUSTOM_NODES_DIR, "sd-scripts"),    # custom_nodes 동급
    os.path.join(_COMFYUI_ROOT, "sd-scripts"),         # ComfyUI 루트 디렉토리
    os.path.join(os.path.dirname(_COMFYUI_ROOT), "sd-scripts"),  # ComfyUI 상위 디렉토리
]

# sd-scripts 전방향에 대한 복제 목표를 추천합니다 (단순히 참고용)"복제"드롭다운 목록）：
# 플러그인 디렉토리 및 ComfyUI 루트 디렉토리 포함 안 함 (훈련 스크립트 ComfyUI/플러그인에 클론되지 않도록 방지)），
# 기본 항목은 ComfyUI와 동일한 디렉토리입니다 (ComfyUI가 /workspace/comfyui에 설치된 경우, 해당 디렉토리에 복제합니다). /workspace/sd-scripts）。
_SD_SCRIPTS_CLONE_TARGETS = [
    os.path.join(os.path.dirname(_COMFYUI_ROOT), "sd-scripts"),  # ComfyUI와 동일 레벨 (기본/추천)）
    os.path.join(_CUSTOM_NODES_DIR, "sd-scripts"),               # custom_nodes 동일 레벨 (옵션)）
]
_PREFERRED_CLONE_TARGET = _SD_SCRIPTS_CLONE_TARGETS[0]

def _get_train_data_dir():
    """훈련 데이터 루트 디렉터리 가져오기 (환경 변수 덮어쓰기 지원)）"""
    env_dir = os.environ.get('TRAIN_DATA_DIR', '')
    if env_dir:
        return env_dir
    return os.path.join(os.path.dirname(_COMFYUI_ROOT), "train_data")

def _get_reg_data_dir():
    """기본 정규화 이미지 디렉토리 획득 (훈련 자료 디렉토리와 인접하여 관리 용이)）

    reg 서브 디렉토리(train_data 아래 기본 위치); TRAINING_REG_DIR 환경 변수를 통해 덮어쓸 수 있습니다.。
    이 디렉터리는 train_data 내에 위치하며, 허용된 루트 백업 목록에 있음。
    """
    env_dir = os.environ.get('TRAINING_REG_DIR', '')
    if env_dir:
        return env_dir
    return os.path.join(_get_train_data_dir(), "reg")

def _get_loras_dir():
    """입력 디렉토리의 최신 이미지 획득 (수정 시간 기준 정렬)）"""
    env_dir = os.environ.get('TRAINING_LORAS_DIR', '')
    if env_dir:
        return env_dir
    return os.path.join(_COMFYUI_ROOT, "models", "loras")

# ===== Path security check =====
# 요소의 z-index 무효(CSS 규격은 적용되지 않음)）
_ALLOWED_IMAGE_ROOTS = None  # 지연 초기화

def _get_allowed_image_roots():
    """허용된 미디어 경로의 루트 디렉토리 목록 가져오기"""
    global _ALLOWED_IMAGE_ROOTS
    if _ALLOWED_IMAGE_ROOTS is not None:
        return _ALLOWED_IMAGE_ROOTS
    roots = [_get_train_data_dir()]
    # 출력 디렉토리(샘플 이미지도 포함)에 추가하여 delete_image 화이트리스트 방지
    output_dir = _training_state.get("output_dir")
    if output_dir and os.path.isdir(output_dir):
        roots.append(output_dir)
    # 환경 변수 확장 백 목록, 경로 구분자를 사용하여 여러 디렉터리 분리
    extra = os.environ.get('TRAINING_ALLOWED_DIRS', '')
    if extra:
        for d in extra.split(os.pathsep):
            d = d.strip()
            if d and os.path.isdir(d):
                roots.append(d)
    # 중복 제거 및 정규화
    _ALLOWED_IMAGE_ROOTS = list(set(os.path.realpath(r) for r in roots if os.path.isdir(r)))
    return _ALLOWED_IMAGE_ROOTS

def _is_path_allowed(path, allowed_roots=None):
    """허용된 루트 디렉토리 내 경로인지 확인 (경로 트래버스 방지)）
    
    Args:
        path: 검토할 경로
        allowed_roots: 허용된 루트 디렉토리 목록, None이면 기본 백 목록을 사용합니다.
    Returns:
        (bool, str): (허용 여부, 정규화된 실제 경로)
    """
    if not path:
        return False, ''
    # 정규화: 심볼 링크 및 .. 
    real_path = os.path.realpath(path)
    roots = allowed_roots or _get_allowed_image_roots()
    for root in roots:
        # 경로는 루트 디렉토리 내에 있어야 함 (루트 디렉토리 자체 또는 하위 경로)）
        if real_path == root or real_path.startswith(root + os.sep):
            return True, real_path
    return False, real_path

def _validate_image_dir(dir_path):
    """유효한 자재 폴더 경로를 확인합니다
    
    Returns:
        (bool, str, str): (Is it valid? Normalize the path, error message)
    """
    if not dir_path or not dir_path.strip():
        return False, '', 'Path cannot be empty'
    dir_path = dir_path.strip()
    # 경로 탐색 확인
    if '..' in dir_path.split(os.sep) or '..' in dir_path.split('/'):
        return False, '', 'Path cannot contain ..'
    # 백 목록 내에 있는지 확인합니다
    allowed, real_path = _is_path_allowed(dir_path)
    if not allowed:
        return False, real_path, '경로가 허용된 범위 내에 없습니다'
    # 디렉토리 존재 여부 확인
    if not os.path.isdir(real_path):
        return False, real_path, '폴더가 없습니다'
    return True, real_path, ''

def _find_sd_scripts():
    """sd-scripts 디렉토리 검색"""
    # 1. 환경 변수 우선
    env_path = os.environ.get('SD_SCRIPTS_DIR', '')
    if env_path and os.path.isdir(env_path) and os.path.isfile(os.path.join(env_path, "train_network.py")):
        return env_path
    # 2. 기존 검색 경로
    for path in _SD_SCRIPTS_SEARCH_PATHS:
        if os.path.isdir(path) and os.path.isfile(os.path.join(path, "train_network.py")):
            return path
    return None

SD_SCRIPTS_DIR = _find_sd_scripts()

# 상태 영구 저장 경로
_STATE_FILE = os.path.join(os.path.dirname(os.path.realpath(__file__)), "training_configs", "_training_state.json")
_state_lock = threading.Lock()

def _load_state():
    """파일에서 학습 상태 로드"""
    default = {
        "status": "idle",
        "process": None,
        "config_path": None,
        "current_step": 0,
        "total_steps": 0,
        "current_loss": 0.0,
        "current_epoch": 0,
        "log_lines": [],
        "saved_models": [],
        "sample_images": [],
        "output_dir": None,
        "start_time": None,
        "loss_history": [],
        "script_name": "",
    }
    try:
        if os.path.exists(_STATE_FILE):
            with open(_STATE_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
            # 다시 복원 비실행 지속성 필드
            for k in ("current_step", "total_steps", "current_loss", "current_epoch",
                      "saved_models", "output_dir", "loss_history", "script_name", "max_train_epochs"):
                if k in saved:
                    default[k] = saved[k]
            # 상대 경로 수정output_dir
            if default["output_dir"] and not os.path.isabs(default["output_dir"]):
                if SD_SCRIPTS_DIR:
                    default["output_dir"] = os.path.abspath(os.path.join(SD_SCRIPTS_DIR, default["output_dir"]))
            # 마지막 실행/일시 중지인 경우 표시interrupted
            if saved.get("status") in ("running", "paused"):
                default["status"] = "interrupted"
            else:
                default["status"] = saved.get("status", "idle")
    except Exception:
        pass
    return default

def _save_state():
    """저장된 훈련 상태로 파일"""
    try:
        os.makedirs(os.path.dirname(_STATE_FILE), exist_ok=True)
        serializable = {k: v for k, v in _training_state.items() if k != "process"}
        with open(_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False)
    except Exception:
        pass

# 훈련 프로세스 상태
_training_state = _load_state()

# 태깅 상태
_tagging_state = {
    "running": False,
    "current": 0,
    "total": 0,
    "process": None,
    "log_lines": [],
    "current_subdir": "",  # 현재 처리 중인 하위 디렉토리
    "cancel_requested": False,  # 요청 취소
}

# 생성 정규화 이미지 상태
_gen_reg_state = {
    "running": False,
    "current": 0,
    "total": 0,
    "process": None,
    "log_lines": [],
    "current_subdir": "",  # 현재 처리 중인 하위 디렉토리
    "cancel_requested": False,  # 요청 취소
}

# 이미지 확장자
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff', '.tif', '.avif'}

# 이미지 전처리 상태
_preprocess_state = {
    "running": False,
    "current": 0,
    "total": 0,
    "log_lines": [],
    "cancel_requested": False,
}

# ===== 정규화 이미지 지원 함수 생성 =====
def _skip_first_tag(prompt):
    """프롬프트의 첫 번째 태그를 건너뛰고(쉼표 또는 마침표로 구분), 트리거 단어가 정규식 그래프에 포함되지 않도록 합니다."""
    if not prompt:
        return prompt
    idx = -1
    for ch in ['，', ',', '。', '.']:
        i = prompt.find(ch)
        if i >= 0 and (idx < 0 or i < idx):
            idx = i
    if idx < 0:
        return prompt.strip()
    return prompt[idx + 1:].strip()


def _round_down_to_multiple(value, multiple):
    """내림차순으로 multiple의 배수로 조정합니다 (anima/sdxl은 높이/너비를 32의 배수로 요구)）。"""
    value = int(value)
    if value <= 0:
        return multiple
    return max(multiple, (value // multiple) * multiple)


def _find_image_recursive(root):
    """하위 디렉토리에서 첫 번째 이미지 파일을 재귀적으로 찾습니다 (가장 최근 수정된 것을 가져오고), 찾을 수 없으면 반환 None。"""
    root = str(root)
    if not os.path.isdir(root):
        return None
    candidates = []
    for dirpath, _dirs, filenames in os.walk(root):
        for fn in filenames:
            if os.path.splitext(fn)[1].lower() in IMAGE_EXTENSIONS:
                candidates.append(os.path.join(dirpath, fn))
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: os.path.getmtime(p))[-1]


def _build_gen_reg_cmd(model_family, script_path, python_path, prompt, params, paths):
    """추론 명령 구성. params 포함 steps/width/height/cfg/seed/sampler/precision/attn_mode/tmp_dir；
    paths 포함 base_model/dit/vae/text_encoder"""
    p = params
    py = python_path or sys.executable
    if model_family == 'sdxl':
        cmd = [
            py, script_path,
            "--prompt", prompt,
            "--ckpt", paths["base_model"],
            "--outdir", p["tmp_dir"],
            "--W", str(p["width"]),
            "--H", str(p["height"]),
            "--steps", str(p["steps"]),
            "--sampler", p["sampler"],
            "--scale", str(p["cfg"]),
            "--seed", str(p["seed"]),
            "--images_per_prompt", "1",
        ]
        if paths.get("vae"):
            cmd += ["--vae", paths["vae"]]
        if p["precision"] == "bf16":
            cmd.append("--bf16")
        else:
            cmd.append("--fp16")
    else:  # anima
        cmd = [
            py, script_path,
            "--dit", paths["dit"],
            "--vae", paths["vae"],
            "--text_encoder", paths["text_encoder"],
            "--prompt", prompt,
            "--image_size", str(p["height"]), str(p["width"]),
            "--infer_steps", str(p["steps"]),
            "--guidance_scale", str(p["cfg"]),
            "--seed", str(p["seed"]),
            # 참고: anima는 --save_path를 '디렉토리'로 간주하며, 실제로는 저장됨
            # <save_path>/<timestamp>/<image>。직접 전달 tmp_dir，
            # 재귀적으로 _collect_gen_output 수집。
            "--save_path", p["tmp_dir"],
            "--output_type", "images",
            "--attn_mode", p["attn_mode"],
        ]
    return cmd


def _collect_gen_output(model_family, tmp_dir, out_path):
    """임시 디렉토리에서 생성된 이미지를 수집하고 이름을 변경합니다 out_path（png）

    anima --save_path를 디렉토리로 사용, 실제 이미지는 tmp_dir/<timestamp>/<image>，
    따라서 먼저 재귀적으로 검색하고; sdxl은 tmp_dir 루트에 직접 쓰고, 루트 디렉토리에서 되돌아감。
    """
    if model_family != 'sdxl':
        found = _find_image_recursive(tmp_dir)
        if found and found != out_path:
            shutil.move(found, out_path)
            return True
    imgs = [os.path.join(tmp_dir, f) for f in os.listdir(tmp_dir)
            if os.path.isfile(os.path.join(tmp_dir, f))
            and os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS]
    if not imgs:
        return False
    shutil.move(sorted(imgs)[0], out_path)
    return True

def _build_gen_reg_jobs(tasks, images_per_prompt, seed, width, height,
                         steps, cfg, sampler, precision, attn_mode,
                         write_caption, skip_first_tag,
                         random_per_image=False):
    """자식 디렉토리 작업들을 개별 작업으로 평면화하여 미리 계산 prompt / out_path / seed。

    캡션이 없는 이미지 건너뛰기 skip=True）。반환된 attempted 목록
    실제로 이미지를 출력해야 하는 작업, 시드 값은 시도 순서대로 증가(기존 로직과 동일)）。
    """
    jobs = []
    gen_idx = 0
    for t in tasks:
        imgs = [f for f in sorted(os.listdir(t["src"]))
                if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
                and os.path.isfile(os.path.join(t["src"], f))]
        seq = t["seq"]
        for img in imgs:
            base = os.path.splitext(img)[0]
            cap = ""
            for ext in ['.txt', '.caption', '.tags']:
                cp = os.path.join(t["src"], base + ext)
                if os.path.isfile(cp):
                    with open(cp, 'r', encoding='utf-8') as fh:
                        cap = fh.read().strip()
                    break
            prompt = _skip_first_tag(cap) if skip_first_tag else cap
            if not prompt:
                jobs.append({"skip": True, "img": img,
                             "target": t["target"], "prefix": t["prefix"]})
                continue
            for _k in range(images_per_prompt):
                out_name = f"{t['prefix']}_{seq:05d}.png"
                out_path = os.path.join(t["target"], out_name)
                # 각 이미지 무작위: 64비트 무서명 랜덤 시드 생성（[0, 2^64-1]），
                # torch.manual_seed의 완전한 64비트 값은 각 이미지에 대해 독립적이고 무관합니다.
                cur_seed = random.randint(0, 2**64 - 1) if random_per_image else seed + gen_idx
                jobs.append({
                    "skip": False, "img": img, "out_name": out_name,
                    "out_path": out_path, "target": t["target"],
                    "prefix": t["prefix"], "prompt": prompt, "seed": cur_seed,
                })
                seq += 1
                gen_idx += 1
    return jobs


def _run_gen_reg_one_load(script, python_path, sd_scripts_dir,
                           dit, vae, text_encoder, attn_mode, precision,
                           width, height, steps, cfg, sampler,
                           jobs, state):
    """일회성 스크립트를 사용하여 모든 정규화 그래프를 생성합니다 (각 그래프를 매번 로드하지 않음)）。

    jobs: [{img, out_path, out_name, target, prefix, prompt, seed}]
    스크립트에서 모델을 한 번 로드하고, 이미지 출력을 반복하며 stdout을 통해 진행 상황을 표시합니다.：
      [REG_SUBDIR] <name> / [REG_SKIP] <img> / [REG_GEN] <img> -> <out>
      [REG_DONE] <done>/<total> / [REG_CANCEL] / [REG_ERROR] <msg>
    """
    py = python_path or sys.executable
    cancel_file = os.path.join(
        tempfile.gettempdir(), f"bsk_genreg_cancel_{os.getpid()}.tmp")
    try:
        if os.path.exists(cancel_file):
            os.remove(cancel_file)
    except OSError:
        pass
    cfg_data = {
        "dit": dit, "vae": vae, "text_encoder": text_encoder,
        "attn_mode": attn_mode, "precision": precision,
        "width": width, "height": height,
        "steps": steps, "cfg": cfg,
        "sampler_name": sampler, "scheduler": "simple",
        "negative_prompt": "",
        "cancel_file": cancel_file,
        "tasks": [
            {"img": j["img"], "out_path": j["out_path"],
             "out_name": j["out_name"], "target": j["target"],
             "prefix": j["prefix"], "prompt": j["prompt"], "seed": j["seed"]}
            for j in jobs
        ],
    }
    cfg_path = tempfile.mkstemp(prefix="bsk_genreg_cfg_", suffix=".json")[1]
    try:
        with open(cfg_path, "w", encoding="utf-8") as fh:
            json.dump(cfg_data, fh, ensure_ascii=False)

        state["log_lines"].append(
            f"[존재 [단일 스크립트 생성] {len(jobs)} 장(모델은 한 번만 로드됩니다)）")
        cmd = [py, script, "--config", cfg_path]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True,
                                encoding="utf-8", errors="replace",
                                cwd=sd_scripts_dir)
        state["process"] = proc
        done = 0
        for line in proc.stdout:
            line = line.rstrip()
            if not line:
                continue
            if line.startswith("[REG_SUBDIR]"):
                state["current_subdir"] = line[len("[REG_SUBDIR]"):].strip()
            elif line.startswith("[REG_DONE]"):
                try:
                    done = int(line[len("[REG_DONE]"):].strip().split("/")[0])
                except ValueError:
                    pass
                state["current"] = done
            state["log_lines"].append(line)
            if len(state["log_lines"]) > 200:
                state["log_lines"] = state["log_lines"][-200:]
        proc.wait()
        state["process"] = None
        if proc.returncode != 0 and not state.get("cancel_requested"):
            state["log_lines"].append(
                f"[오류] 단일 스크립트 생성 중 예기치 않은 종료 (returncode={proc.returncode})")
    finally:
        try:
            os.remove(cfg_path)
        except OSError:
            pass
        try:
            if os.path.exists(cancel_file):
                os.remove(cancel_file)
        except OSError:
            pass


# ===== 의존성 검사 =====
# (package_name, import_name, category)
# category: "required" / "recommended" / "optional"
TRAINING_DEPS = [
    ("accelerate",       "accelerate",       "required"),
    ("diffusers",        "diffusers",        "required"),
    ("toml",             "toml",             "required"),
    ("voluptuous",       "voluptuous",       "required"),
    ("huggingface-hub",  "huggingface_hub",  "required"),
    ("ftfy",             "ftfy",             "required"),
    ("opencv-python",    "cv2",              "required"),
    ("bitsandbytes",     "bitsandbytes",     "required"),
    ("lion-pytorch",     "lion_pytorch",     "recommended"),
    ("tensorboard",      "tensorboard",      "recommended"),
    ("schedulefree",     "schedulefree",     "required"),
    ("pytorch-optimizer","pytorch_optimizer","required"),
    ("prodigyopt",       "prodigyopt",       "required"),
    ("prodigy-plus-schedule-free", "prodigyplus", "required"),
    ("rich",             "rich",             "required"),
    ("imagesize",        "imagesize",        "required"),
]

    # ("onnxruntime-gpu",  "onnxruntime",      "recommended"),
# 설치 상태 캐시
_deps_check_cache = None
_deps_install_lock = threading.Lock()


def _check_training_deps():
    """훈련 도구 종속성 설치 여부 확인, 누락된 목록 반환"""
    global _deps_check_cache
    if _deps_check_cache is not None:
        return _deps_check_cache

    missing = []
    installed = []
    for pkg_name, import_name, category in TRAINING_DEPS:
        try:
            importlib.import_module(import_name)
            installed.append({"package": pkg_name, "import": import_name, "category": category, "installed": True})
        except ImportError:
            missing.append({"package": pkg_name, "import": import_name, "category": category, "installed": False})

    _deps_check_cache = {"missing": missing, "installed": installed, "all_ok": len(missing) == 0}
    return _deps_check_cache


def _install_training_deps(packages, pip_args=None):
    """지정된 의존 패키지 설치, 반환(subprocess, output_collector)"""
    cmd = [sys.executable, "-m", "pip", "install", "-i", "https://mirrors.cloud.tencent.com/pypi/simple"]
    if pip_args:
        cmd.extend(pip_args)
    cmd.extend(packages)

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace"
    )

    # 스레드를 사용하여 실시간으로 stdout을 소비하여 PIPE 버퍼가 가득 차서 pip가 멈추는 것을 방지합니다.
    output_lines = []
    def _reader():
        try:
            for line in proc.stdout:
                output_lines.append(line)
        except Exception:
            pass

    reader_thread = threading.Thread(target=_reader, daemon=True)
    reader_thread.start()

    return proc, output_lines


# ===== bsk-training-deps.txt 기반 버전 감지 의존성 분석 =====
# 데이터 소스: 본 파일과 동일 디렉토리에 있는 bsk-training-deps.txt (유일한 목록))
_DEPS_TXT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bsk-training-deps.txt")
# 무거운 패키지: 및 CUDA/드라이버 강하게바인딩, 단검사미스, 아니참및자동アップグレード(잘못 설치 방지버전파괴환경)
HEAVY_PACKAGES = {
    "torch", "torchvision", "torchaudio", "torchsde",
    "onnxruntime", "onnxruntime-gpu", "onnxruntime-cann",
}


def _parse_deps_file():
    """파싱 bsk-training-deps.txt。
    반환 항목 목록 [{"name","spec","raw","heavy"}, ...]。
    packaging 라이브러리 또는 파일이 존재하지 않을 경우 None 반환 (호출자 이전 로직으로 되돌림))。
    """
    try:
        from packaging.requirements import Requirement
    except Exception:
        return None
    if not os.path.exists(_DEPS_TXT):
        return None
    entries = []
    try:
        with open(_DEPS_TXT, "r", encoding="utf-8") as f:
            for line in f:
                s = line.split("#", 1)[0].strip()   # 행 종료 주석 제거
                if not s:
                    continue
                try:
                    req = Requirement(s)
                except Exception:
                    continue
                # 환경 표시(예 ; sys_platform==...)표에 있는 설명자와 완전히 일치합니다
                if req.marker is not None:
                    try:
                        if not req.marker.evaluate():
                            continue
                    except Exception:
                        pass
                entries.append({
                    "name": req.name,
                    "spec": req.specifier,
                    "raw": s,
                    "heavy": req.name.lower() in HEAVY_PACKAGES,
                })
    except Exception:
        return None
    return entries


def _spec_satisfied(installed_ver, spec):
    """최저 버전 의미: 설치된 버전 >= 하한 제약은 만족으로 간주。
    고 버전 다운그레이드 안 함 (무시) < / <= 상한 제한), 구현"고 버전이 설치됨"。
    """
    if spec is None or len(spec) == 0:
        return True
    try:
        from packaging.version import Version
    except Exception:
        return True
    try:
        iv = Version(installed_ver)
    except Exception:
        return True   # 파싱 불가, 설치 버전 보수적으로 만족 처리, 다운그레이드 유발 안함
    for s in spec:
        op = s.operator
        try:
            rv = Version(s.version)
        except Exception:
            continue
        if op in ("==", "===", "~=", ">="):
            if iv < rv:
                return False
        elif op == ">":
            if iv <= rv:
                return False
        elif op == "!=":
            if iv == rv:
                return False
        # "<" / "<=" 상한선 제약: 무시(고 버전 변경 없음))
    return True


def _analyze_deps():
    """설치된 버전과 비교 bsk-training-deps.txt:
      - 설치되지 않음 / 버전 낮음 → 설치 필요(install_list/install_packages)
      - 충족됨 → 건너뛰기(skip_list)
      - 무거운 패키지 누락 → 경고만 표시(heavy_missing), 자동 업그레이드 안 함
    파싱 불가 (패키징/파일 누락) 반환 None。
    """
    import importlib.metadata as _md
    entries = _parse_deps_file()
    if entries is None:
        return None
    install_list, install_packages, skip_list, heavy_missing = [], [], [], []
    for e in entries:
        try:
            iv = _md.version(e["name"])
        except Exception:
            iv = None
        if e["heavy"]:
            if iv is None:
                heavy_missing.append(e["name"])
            else:
                skip_list.append(f"{e['name']} {iv}")
            continue
        if iv is None:
            install_list.append(f"{e['raw']} (설치되지 않음)")
            install_packages.append(e["raw"])
        elif not _spec_satisfied(iv, e["spec"]):
            install_list.append(f"{e['name']} {iv} → {e['raw']} (アップグレード)")
            install_packages.append(e["raw"])
        else:
            skip_list.append(f"{e['name']} {iv}")
    return {
        "install_list": install_list,
        "install_packages": install_packages,
        "skip_list": skip_list,
        "heavy_missing": heavy_missing,
    }


# ===== 이미지 전처리 =====
def _preprocess_images_worker(dir_path, enable_resize, target_resolution, num_threads):
    """이미지 프리프로세싱 워크스레드
    
    Args:
        dir_path: 대상 폴더 경로
        enable_resize: 확대 여부 활성화
        target_resolution: 대상 해상도(가로))
        num_threads: Thread count
    """
    import cv2
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import tempfile
    import shutil
    
    try:
        # 폴더 이름으로 접두사 가져오기
        folder_name = os.path.basename(dir_path)
        
        # 모든 이미지 파일 수집
        image_files = []
        for f in os.listdir(dir_path):
            ext = os.path.splitext(f)[1].lower()
            if ext in IMAGE_EXTENSIONS:
                image_files.append(os.path.join(dir_path, f))
        
        if not image_files:
            _preprocess_state["log_lines"].append("[완료] 이미지 파일이 없습니다")
            _preprocess_state["running"] = False
            return
        
        _preprocess_state["total"] = len(image_files)
        _preprocess_state["log_lines"].append(f"[시작] 총 발견 {len(image_files)} 이미지 템플릿")
        
        # 새 파일 매핑 생성 (기존 경로 -> 새 경로)
        # 임시 파일 이름 사용으로 충돌 방지
        file_mapping = {}  # {old_path: (temp_path, final_path)}
        temp_dir = tempfile.mkdtemp(prefix="preprocess_")
        
        try:
            # 1단계: 모든 파일은 먼저 변환/확대 축소하고 임시 디렉토리에 저장합니다
            def process_single_image(img_path, index):
                """단일 이미지 처리"""
                if _preprocess_state["cancel_requested"]:
                    return None
                
                try:
                    # 이미지 읽기
                    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                    if img is None:
                        return {"status": "error", "file": img_path, "error": "이미지 읽기 실패"}
                    
                    # 확대
                    if enable_resize and target_resolution > 0:
                        h, w = img.shape[:2]
                        max_dim = max(w, h)
                        if max_dim != target_resolution:
                            scale = target_resolution / max_dim
                            new_w = int(w * scale)
                            new_h = int(h * scale)
                            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    
                    # 새 파일 이름 생성: folder_00001.png
                    new_name = f"{folder_name}_{index:05d}.png"
                    temp_path = os.path.join(temp_dir, new_name)
                    
                    # 저장PNG
                    success = cv2.imwrite(temp_path, img)
                    if not success:
                        return {"status": "error", "file": img_path, "error": "저장 실패"}
                    
                    return {
                        "status": "success",
                        "old_path": img_path,
                        "temp_path": temp_path,
                        "new_name": new_name
                    }
                except Exception as e:
                    return {"status": "error", "file": img_path, "error": str(e)}
            
            # 다중 스레드 처리
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = {executor.submit(process_single_image, img_path, i+1): img_path 
                          for i, img_path in enumerate(image_files)}
                
                for future in as_completed(futures):
                    if _preprocess_state["cancel_requested"]:
                        executor.shutdown(wait=False, cancel_futures=True)
                        break
                    
                    result = future.result()
                    _preprocess_state["current"] += 1
                    
                    if result is None:
                        continue
                    
                    if result["status"] == "success":
                        file_mapping[result["old_path"]] = (result["temp_path"], result["new_name"])
                        _preprocess_state["log_lines"].append(
                            f"[{_preprocess_state['current']}/{_preprocess_state['total']}] "
                            f"{os.path.basename(result['old_path'])} -> {result['new_name']}"
                        )
                    else:
                        _preprocess_state["log_lines"].append(
                            f"[오류] {os.path.basename(result['file'])}: {result.get('error', '알 수 없는 오류')}"
                        )
            
            if _preprocess_state["cancel_requested"]:
                _preprocess_state["log_lines"].append("[취소] 사용자 운영 취소")
                _preprocess_state["running"] = False
                return
            
            # 2단계: 원본 이미지 삭제 및 새 파일 이동
            _preprocess_state["log_lines"].append("[이름 바꾸기] 시작 파일 대체...")
            
            # 먼저 모든 원본 이미지(PNG 포함)를 삭제하세요)
            for old_path in file_mapping.keys():
                try:
                    if os.path.exists(old_path):
                        os.remove(old_path)
                except Exception as e:
                    _preprocess_state["log_lines"].append(f"[경고] 삭제 불가능 {os.path.basename(old_path)}: {e}")
            
            # 새로운 파일을 대상 디렉토리로 이동
            for old_path, (temp_path, new_name) in file_mapping.items():
                try:
                    final_path = os.path.join(dir_path, new_name)
                    shutil.move(temp_path, final_path)
                except Exception as e:
                    _preprocess_state["log_lines"].append(f"[오류] 이동 실패 {new_name}: {e}")
            
            _preprocess_state["log_lines"].append(
                f"[완료] 성공 처리 {len(file_mapping)} 이미지 템플릿"
            )
            
        finally:
            # 정리 임시 폴더
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
        
    except Exception as e:
        _preprocess_state["log_lines"].append(f"[오류] {str(e)}")
    finally:
        _preprocess_state["running"] = False


def setup_training_api(app=None, config=None):
    """훈련 도구 API 경로 등록
    
    매개변수:
      app: aiohttp.web.Application 예시
           - ComfyUI 모드: 입력 PromptServer.instance.app
           - 독립 모드: 자체 구축된 Application
      config: 선택 구성 딕셔너리, 독립 모드 시 경로 구성 전달
    """
    from aiohttp import web
    
    if app is None:
        # ComfyUI 내용(임시 주석/수정 가능, 설치 시 이 텍스트 사용)
        try:
            from server import PromptServer
            app = PromptServer.instance.app
        except (ImportError, AttributeError):
            raise RuntimeError("PromptServer를 가져오려면 app 매개변수를 제공해주세요. aiohttp Application")

    try:


# ===== 의존성 검사 =====
        _install_process = None  # 설치 프로세스 참조
        _install_output = []    # 설치 출력 수집
        _install_packages = []  # 현재 설치된 패키지 목록

        async def default_paths(request):
            """기본 경로 정보를 반환합니다(ComfyUI 루트 디렉토리, 기본 자료 경로, Python 경로, sd-scripts 경로 등)）"""
            try:
                train_data_dir = _get_train_data_dir()
                # 자동으로 기본 자료 디렉토리를 생성합니다
                os.makedirs(train_data_dir, exist_ok=True)
                # 현재 Python 실행 파일 경로 획득 (가상 환경 확인)）
                python_path = sys.executable
                # sd-scripts 실제 경로 가져오기
                sd_scripts_dir = SD_SCRIPTS_DIR or ""
                return web.json_response({
                    "success": True,
                    "comfyui_root": _COMFYUI_ROOT,
                    "train_data_dir": train_data_dir,
                    "reg_data_dir": _get_reg_data_dir(),
                    "python_path": python_path,
                    "sd_scripts_path": sd_scripts_dir,
                    "default_models_dir": os.path.join(_COMFYUI_ROOT, "models"),
                    "allowed_dirs": _get_allowed_image_roots(),
                })
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        async def validate_image_dir(request):
            """유효한 자재 폴더 경로를 확인합니다"""
            try:
                data = await request.json()
                dir_path = data.get("dir", "").strip()
                valid, real_path, error = _validate_image_dir(dir_path)
                return web.json_response({
                    "success": valid,
                    "real_path": real_path,
                    "error": error,
                })
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        async def check_deps(request):
            """훈련 도구 종속성이 설치되었는지 확인합니다"""
            try:
                # 강제 캐시 새로고침 (패널 열 때마다 재검출)）
                global _deps_check_cache, SD_SCRIPTS_DIR
                _deps_check_cache = None
                result = _check_training_deps()
                # 버전 인식 분석 (bsk-training-deps.txt를 데이터 소스로 사용))
                smart = _analyze_deps()
                if smart is not None:
                    result = dict(result)   # 오염 방지 _deps_check_cache
                    result["smart"] = smart
                    # all_ok: 설치/업데이트 불필요 및 무거운 패키지 누락 없음
                    result["all_ok"] = (len(smart["install_list"]) == 0
                                        and len(smart["heavy_missing"]) == 0)
                # sd-scripts 경로를 다시 검색합니다.
                SD_SCRIPTS_DIR = _find_sd_scripts()
                result["sd_scripts_found"] = SD_SCRIPTS_DIR is not None
                result["sd_scripts_path"] = SD_SCRIPTS_DIR or ""
                # 드롭다운 복제는 추천 대상만 제공 (플러그인 디렉토리/ComfyUI 루트 디렉토리 제외)）
                result["sd_scripts_search_paths"] = _SD_SCRIPTS_CLONE_TARGETS
                result["preferred_clone_target"] = _PREFERRED_CLONE_TARGET
                return web.json_response({"success": True, **result})
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        async def install_deps(request):
            """필요한 종속성을 설치합니다"""
            nonlocal _install_process, _install_output, _install_packages
            try:
                data = await request.json()
                packages = data.get("packages", [])
                category_filter = data.get("category", "required")  # required / recommended / all
                use_smart = data.get("smart", False)
                details = None

                if not packages:
                    # 지능형 모드: bsk-training-deps.txt를 데이터 소스로 사용하고, 설치되지 않았거나 버전이 낮은 패키지를 설치합니다(무거운 패키제외).)
                    analysis = _analyze_deps() if use_smart else None
                    if analysis is not None:
                        packages = analysis["install_packages"]
                        details = {
                            "install_list": analysis["install_list"],
                            "skip_list": analysis["skip_list"],
                            "heavy_missing": analysis["heavy_missing"],
                        }
                    else:
                        # 롤백: 카테고리별로 기존 하드코딩된 검사 패키지를 선택합니다
                        deps = _check_training_deps()
                        if category_filter == "all":
                            packages = [d["package"] for d in deps["missing"]]
                        elif category_filter == "recommended":
                            packages = [d["package"] for d in deps["missing"]
                                        if d["category"] in ("required", "recommended")]
                        else:  # required only
                            packages = [d["package"] for d in deps["missing"]
                                        if d["category"] == "required"]

                if not packages:
                    return web.json_response({"success": True, "message": "모든 의존성 충족됨",
                                              "smart": use_smart, "details": details})

                # 중복 설치 방지
                if _install_process and _install_process.poll() is None:
                    return web.json_response({"success": False, "error": "설치 중입니다. 기다려 주세요"})

                _install_process, _install_output = _install_training_deps(packages)
                _install_packages = packages
                # 생성 수동 설치 명령어 백업
                manual_cmd = f'"{sys.executable}" -m pip install -i https://mirrors.cloud.tencent.com/pypi/simple {" ".join(packages)}'
                return web.json_response({"success": True, "packages": packages,
                                          "manual_command": manual_cmd,
                                          "smart": use_smart, "details": details})
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        async def install_status(request):
            """설치 진행률 쿼리"""
            nonlocal _install_process, _install_output, _install_packages
            if _install_process is None:
                return web.json_response({"success": True, "status": "idle"})

            # 수동 설치 명령 생성 (항상 반환, 사용자 복사 용이)）
            manual_cmd = ""
            if _install_packages:
                manual_cmd = f'"{sys.executable}" -m pip install -i https://mirrors.cloud.tencent.com/pypi/simple {" ".join(_install_packages)}'

            # 현재 수집된 출력 가져오기
            current_output = "".join(_install_output) if _install_output else ""

            if _install_process.poll() is None:
                # 여전히 실행 중
                return web.json_response({
                    "success": True,
                    "status": "installing",
                    "output": current_output[-2000:] if current_output else "",
                    "manual_command": manual_cmd,
                })

            # 설치 완료
            returncode = _install_process.returncode
            _install_process = None

            # 설치 후 의존성 캐시를 삭제하고 다음 검사 시 다시 스캔합니다.
            global _deps_check_cache
            _deps_check_cache = None

            return web.json_response({
                "success": True,
                "status": "completed",
                "returncode": returncode,
                "output": current_output[-2000:] if current_output else "",
                "ok": returncode == 0,
                    "manual_command": manual_cmd,
                })

        async def clone_sd_scripts(request):
            """sd-scripts 저장소를 지정된 디렉토리로 복제합니다 (훈련 스크립트가 누락된 경우 사용）

            프론트엔드 _cloneSdScripts / _installDeps 호출, 요청 본문 {repo, target}。
            반환 {success:true, path} 또는 {success:false, error}。
            """
            try:
                data = await request.json()
            except Exception:
                return web.json_response({"success": False, "error": "요청 본문은 유효하지 않습니다 JSON"})

            repo = (data.get("repo") or "").strip()
            target = (data.get("target") or "").strip()
            if not repo:
                return web.json_response({"success": False, "error": "저장소 주소 누락 (repo)"})
            if not target:
                return web.json_response({"success": False, "error": "목적 경로가 없습니다 (target)"})

            target = os.path.abspath(os.path.expanduser(target))
            # 기본 보안 검사: 절대 경로여야 하며, 시스템 루트 디렉터리에 위치해서는 안 됨
            if not os.path.isabs(target):
                return web.json_response({"success": False, "error": f"목표 경로가 비어있거나 절대 경로여야 함: {target}"})
            if target in ("/", "\\", "C:\\", os.path.sep):
                return web.json_response({"success": False, "error": "목적 경로가 너무 위험하여 거부되었습니다"})

            # 대상 이미지가 존재하고 비어 있지 않음 → 거부 (git clone이 덮어쓰기를 허용하지 않음))
            if os.path.exists(target):
                if os.listdir(target):
                    return web.json_response({"success": False, "error": f"대상 디렉터리 이미 존재하고 비어있지 않음: {target}"})
                # 빈 디렉토리: git은 이미 존재하는 디렉토리에 클론할 수 없습니다. 먼저 제거하십시오
                try:
                    os.rmdir(target)
                except OSError:
                    pass

            # 폴더 만들기
            parent = os.path.dirname(target)
            try:
                os.makedirs(parent, exist_ok=True)
            except Exception as e:
                return web.json_response({"success": False, "error": f"상위 디렉토리 생성 불가 {parent}: {e}"})

            # 실행 git clone (비동기, 이벤트 루프 차단))
            try:
                proc = await asyncio.create_subprocess_exec(
                    "git", "clone", repo, target,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                )
            except FileNotFoundError:
                return web.json_response({"success": False, "error": "git 명령어를 찾을 수 없습니다. Git이 설치되어 있는지 확인하고 PATH에 있는지 확인하세요."})
            except Exception as e:
                return web.json_response({"success": False, "error": f"이름을 지정한 후 자료가 자동으로 프로젝트 폴더로 이동합니다: {e}"})

            stdout, _ = await proc.communicate()
            out_text = stdout.decode("utf-8", "replace") if stdout else ""
            if proc.returncode != 0:
                # 복제 실패 시 잠재적으로 남은 빈 디렉토리 정리
                if os.path.isdir(target) and not os.listdir(target):
                    try:
                        os.rmdir(target)
                    except OSError:
                        pass
                return web.json_response({
                    "success": False,
                    "error": f"git clone 실패 (exit={proc.returncode}):\n{out_text[-2000:]}",
                })

            # 복제 성공 후 sd-scripts 경로를 다시 검색하여 후속 종속성 검출이 가능하도록 합니다.
            global SD_SCRIPTS_DIR
            SD_SCRIPTS_DIR = _find_sd_scripts()
            return web.json_response({"success": True, "path": target})

        # ===== 데이터 준비 =====
        async def scan_images(request):
            """훈련 이미지 디렉토리 스캔"""
            try:
                data = await request.json()
                dir_path = data.get("dir", "").strip()
                if not dir_path or not os.path.isdir(dir_path):
                    return web.json_response({"success": False, "error": "폴더가 없습니다"})

                images = []
                total_size = 0
                widths, heights = [], []
                tagged_count = 0

                for root, dirs, filenames in os.walk(dir_path):
                    for f in filenames:
                        ext = os.path.splitext(f)[1].lower()
                        if ext not in IMAGE_EXTENSIONS:
                            continue
                        file_path = os.path.join(root, f)
                        if not os.path.isfile(file_path):
                            continue
                        stat = os.stat(file_path)
                        rel_dir = os.path.relpath(root, dir_path)
                        images.append({
                            "name": f if rel_dir == "." else os.path.join(rel_dir, f),
                            "size": stat.st_size,
                            "mtime": stat.st_mtime,
                        })
                        total_size += stat.st_size

                        # 검사 동일 이름 레이블 파일이 있는지
                        txt_path = os.path.join(root, os.path.splitext(f)[0] + ".txt")
                        if os.path.exists(txt_path):
                            tagged_count += 1

                        # 이미지 크기 가져오기 (샘플링 초반 몇 장만 가져와 속도 개선))
                        if len(images) <= 20:
                            try:
                                from PIL import Image
                                with Image.open(file_path) as img:
                                    widths.append(img.width)
                                    heights.append(img.height)
                            except:
                                pass

                avg_w = int(sum(widths) / len(widths)) if widths else 0
                avg_h = int(sum(heights) / len(heights)) if heights else 0

                return web.json_response({
                    "success": True,
                    "images": images,
                    "stats": {
                        "count": len(images),
                        "tagged": tagged_count,
                        "total_size": total_size,
                        "avgWidth": avg_w,
                        "avgHeight": avg_h,
                    }
                })
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        async def scan_models(request):
            """스캔 가능한 모델 파일, folder_paths 등록된 디렉토리 및 fallback 검색 지원"""
            try:
                model_type = request.query.get("type", "checkpoints")
                # 사용자 정의 모델 검색 디렉토리 (프론트엔드"모델 검색 폴더"입력 상자 (Input Box)
                custom_search_dir = (request.query.get("model_search_dir") or "").strip()
                files = []
                model_dirs = []  # 여러 디렉토리 검색 지원

                # 특수 유형을 ComfyUI 알려진 하위 디렉토리로 매핑 (유형별로 스캔)）
                type_aliases = {
                    "diffusion_models": ["diffusion_models", "unet", "checkpoints"],
                    "text_encoders": ["text_encoders", "clip"],
                    "clip": ["clip", "text_encoders"],
                    "t5_tokenizer": ["t5_tokenizer", "text_encoders"],
                    "vae": ["vae"],
                    "checkpoints": ["checkpoints"],
                    "loras": ["loras"],
                }
                search_types = type_aliases.get(model_type, [model_type])

                if custom_search_dir and os.path.isdir(custom_search_dir):
                    # 사용자 지정 디렉토리: 모델 유형별로 해당 하위 디렉토리로 스캔하여 드롭다운 목록을 유형별로 분류
                    for st in search_types:
                        c = os.path.join(custom_search_dir, st)
                        if os.path.isdir(c) and c not in model_dirs:
                            model_dirs.append(c)
                else:
                    # 1. folder_paths로 등록된 폴더를 우선 사용 (여러 경로가 반환될 수 있음)）
                    try:
                        import folder_paths
                        paths = folder_paths.get_folder_paths(model_type)
                        if paths:
                            model_dirs.extend([p for p in paths if os.path.isdir(p)])
                    except (ImportError, IndexError, KeyError):
                        pass

                    # 3. fallback: 여러 가능한 위치 검색
                    if not model_dirs:
                        candidates = []
                        for st in search_types:
                            candidates.extend([
                                os.path.join(_COMFYUI_ROOT, "models", st),
                                os.path.join(os.path.dirname(_COMFYUI_ROOT), "models", st),
                                os.path.join(os.path.dirname(os.path.realpath(__file__)), "models", st),
                            ])
                        for c in candidates:
                            if os.path.isdir(c) and c not in model_dirs:
                                model_dirs.append(c)

                # 4. 모든 모델 디렉토리 스캔
                seen = set()
                for model_dir in model_dirs:
                    if not os.path.isdir(model_dir):
                        continue
                    for root, dirs, filenames in os.walk(model_dir):
                        # Skip hidden directory
                        dirs[:] = [d for d in dirs if not d.startswith('.')]
                        for f in filenames:
                            ext = os.path.splitext(f)[1].lower()
                            if ext not in ['.ckpt', '.safetensors', '.pt', '.pth', '.bin']:
                                continue
                            file_path = os.path.join(root, f)
                            # 자리 차지 파일을 건너띡니다 (0바이트 또는 특별한 이름)）
                            if f.startswith('put_') and f.endswith('_here'):
                                continue
                            # 중복 제거
                            abs_path = os.path.abspath(file_path)
                            if abs_path in seen:
                                continue
                            seen.add(abs_path)
                            rel_path = os.path.relpath(file_path, model_dir)
                            try:
                                fsize = os.path.getsize(file_path)
                            except OSError:
                                fsize = 0
                            files.append({
                                "name": f,
                                "path": file_path,
                                "rel_path": rel_path,
                                "size": fsize,
                            })

                files.sort(key=lambda x: x["path"].lower())
                primary_dir = model_dirs[0] if model_dirs else ""
                return web.json_response({"success": True, "files": files, "count": len(files), "model_dir": primary_dir, "model_dirs": model_dirs})
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        # ===== Tagging =====
        async def start_tagging(request):
            """WD14 배치 마킹 시작
            두 가지 모드 지원:
            1. 구형 모드: dir + trigger_word (역호환, 자식 디렉토리 재귀 스캔)
            2. 새 모드: images_dir + subdirs [{path, trigger_word}] (각 하위 디렉터리별 트리거)
            """
            try:
                data = await request.json()
                # 태그 유형 및 매개변수 (호환 이전 필드) model / threshold）
                tagger_type = data.get("tagger_type", "cltagger")
                if tagger_type not in ("cltagger", "wd14"):
                    tagger_type = "cltagger"
                wd14_model = data.get("wd14_model") or data.get("model") or "wd-vit-tagger-v3"
                general_threshold = float(data.get("general_threshold", data.get("threshold", 0.35)))
                character_threshold = float(data.get("character_threshold", 0.60))
                sort_mode = data.get("sort_mode", "default")
                exclude_meta = data.get("exclude_meta", True)
                exclude_tags = data.get("exclude_tags", "")
                trailing_comma = data.get("trailing_comma", False)
                replace_underscore = data.get("replace_underscore", True)
                overwrite = data.get("overwrite", False)
                max_resolution = int(data.get("max_resolution", 1280))
                enable_resize = data.get("enable_resize", True)

                # 내장된 BSK 타그 CLI를 우선 사용합니다 (CLTagger / WD14 두 종류의 타그기 사용 지원).）
                bsk_cli = os.path.join(_PLUGIN_DIR, "bsk_tagger_cli.py")
                if os.path.exists(bsk_cli):
                    cli_path = bsk_cli
                    use_old_wd14 = False
                else:
                    # 최저값: 이전 ComfyUI-WD14-Tagger 플러그인 (지원 wd14）
                    wd14_plugin_dir = None
                    if os.path.isdir(_CUSTOM_NODES_DIR):
                        for d in os.listdir(_CUSTOM_NODES_DIR):
                            if d.lower() == "comfyui-wd14-tagger":
                                wd14_plugin_dir = os.path.join(_CUSTOM_NODES_DIR, d)
                                break
                    if not wd14_plugin_dir:
                        env_wd14 = os.environ.get('WD14_TAGGER_DIR', '')
                        if env_wd14 and os.path.isfile(os.path.join(env_wd14, "wd14_tagger_cli.py")):
                            wd14_plugin_dir = env_wd14
                    if not wd14_plugin_dir:
                        return web.json_response({"success": False, "error": "라벨링 도구를 찾을 수 없습니다: bsk_tagger_cli.py를 플러그인 루트 디렉토리에 넣거나 ComfyUI-WD14-Tagger 플러그인을 설치하십시오。"})
                    cli_path = os.path.join(wd14_plugin_dir, "wd14_tagger_cli.py")
                    if not os.path.exists(cli_path):
                        return web.json_response({"success": False, "error": f"wd14_tagger_cli.py 존재하지 않음 {wd14_plugin_dir}"})
                    if tagger_type != "wd14":
                        tagger_type = "wd14"  # 구 CLI만 지원 wd14
                    use_old_wd14 = True
                    wd14_models_dir = os.path.join(wd14_plugin_dir, "models")

                # 빌드 서브 디렉토리 라벨링 작업 목록
                subdirs = data.get("subdirs", None)
                if subdirs and isinstance(subdirs, list) and len(subdirs) > 0:
                    # 새 모드: 하위 디렉토리별로 태깅
                    tagging_tasks = []
                    for sd in subdirs:
                        sub_path = sd.get("path", "").strip()
                        tw = sd.get("trigger_word", "").strip()
                        if sub_path and os.path.isdir(sub_path):
                            tagging_tasks.append({"path": sub_path, "trigger_word": tw})

                    if not tagging_tasks:
                        return web.json_response({"success": False, "error": "유효하지 않은 하위 디렉터리 없음"})

                    # 전체 이미지 수 통계 (각 하위 디렉터리, 재귀 없음))
                    total_count = 0
                    for task in tagging_tasks:
                        cnt = sum(1 for f in os.listdir(task["path"])
                                  if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
                                  and os.path.isfile(os.path.join(task["path"], f)))
                        total_count += cnt
                else:
                    # 구 모드: 단일 디렉토리 재귀 태깅 (역호환))
                    dir_path = data.get("dir", "").strip()
                    trigger_word = data.get("trigger_word", "").strip()
                    if not dir_path or not os.path.isdir(dir_path):
                        return web.json_response({"success": False, "error": "폴더가 없습니다"})

                    tagging_tasks = [{"path": dir_path, "trigger_word": trigger_word, "recursive": True}]

                    # 모든 이미지 통계 (재귀))
                    total_count = 0
                    for root, dirs, filenames in os.walk(dir_path):
                        for f in filenames:
                            if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS:
                                total_count += 1

                if total_count == 0:
                    return web.json_response({"success": False, "error": "이미지 파일이 없음"})

                # 마킹 프로세스 시작
                _tagging_state["running"] = True
                _tagging_state["current"] = 0
                _tagging_state["total"] = total_count
                _tagging_state["log_lines"] = []
                _tagging_state["process"] = None
                _tagging_state["current_subdir"] = ""
                _tagging_state["cancel_requested"] = False

                # 백그라운드 스레드가 순서대로 하위 디렉터리를 처리합니다
                def _monitor_tagging():
                    tagged_total = 0
                    for task in tagging_tasks:
                        if _tagging_state["cancel_requested"]:
                            _tagging_state["log_lines"].append("[취소] 사용자가 라벨링 작업을 취소했습니다")
                            break

                        subdir_name = os.path.basename(task["path"])
                        _tagging_state["current_subdir"] = subdir_name

                        if use_old_wd14:
                            # 구 버전 ComfyUI-WD14-Tagger CLI (단 wd14）
                            cmd = [
                                sys.executable, cli_path,
                                "--dir", task["path"],
                                "--model", wd14_model,
                                "--threshold", str(general_threshold),
                                "--character_threshold", str(character_threshold),
                                "--models_dir", wd14_models_dir,
                            ]
                            if task.get("trigger_word"):
                                cmd.extend(["--trigger_word", task["trigger_word"]])
                            if not replace_underscore:
                                cmd.append("--no_replace_underscore")
                            if overwrite:
                                cmd.append("--overwrite")
                            if task.get("recursive"):
                                cmd.append("--recursive")
                            cmd.extend(["--max_resolution", str(max_resolution)])
                            if not enable_resize:
                                cmd.append("--no_resize")
                        else:
                            # 새 BSK 라벨링 CLI（CLTagger / WD14）
                            cmd = [
                                sys.executable, cli_path,
                                "--tagger_type", tagger_type,
                                "--general_threshold", str(general_threshold),
                                "--character_threshold", str(character_threshold),
                                "--sort_mode", sort_mode,
                                "--max_resolution", str(max_resolution),
                                "--dir", task["path"],
                            ]
                            if tagger_type == "wd14":
                                cmd.extend(["--wd14_model", wd14_model])
                            if not exclude_meta:
                                cmd.append("--no_exclude_meta")
                            if not replace_underscore:
                                cmd.append("--no_replace_underscore")
                            if trailing_comma:
                                cmd.append("--trailing_comma")
                            if exclude_tags:
                                cmd.extend(["--exclude_tags", exclude_tags])
                            if task.get("trigger_word"):
                                cmd.extend(["--trigger_word", task["trigger_word"]])
                            if overwrite:
                                cmd.append("--overwrite")
                            if task.get("recursive"):
                                cmd.append("--recursive")

                        _tagging_state["log_lines"].append(f"[자 디렉토리] {task['path']} (trigger: {task.get('trigger_word', '없다')})")
                        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                                text=True, encoding='utf-8', errors='replace')
                        _tagging_state["process"] = proc

                        for line in proc.stdout:
                            line = line.rstrip()
                            _tagging_state["log_lines"].append(line)
                            if len(_tagging_state["log_lines"]) > 200:
                                _tagging_state["log_lines"] = _tagging_state["log_lines"][-200:]
                            if "OK:" in line or "tagged" in line.lower() or "processed" in line.lower():
                                tagged_total += 1
                                _tagging_state["current"] = tagged_total
                            # 취소 요청 확인
                            if _tagging_state["cancel_requested"]:
                                proc.terminate()
                                try:
                                    proc.wait(timeout=5)
                                except subprocess.TimeoutExpired:
                                    proc.kill()
                                _tagging_state["log_lines"].append(f"[취소] 종료됨 {subdir_name}")
                                break

                        if _tagging_state["cancel_requested"]:
                            break

                        proc.wait()
                        if proc.returncode != 0 and not _tagging_state["cancel_requested"]:
                            _tagging_state["log_lines"].append(f"[경고] 하위 디렉토리 {task['path']} 오류 코드 반환: {proc.returncode}")

                    _tagging_state["running"] = False
                    _tagging_state["process"] = None
                    _tagging_state["current_subdir"] = ""

                threading.Thread(target=_monitor_tagging, daemon=True).start()

                return web.json_response({"success": True, "total": total_count, "subdir_count": len(tagging_tasks)})
            except Exception as e:
                _tagging_state["running"] = False
                _tagging_state["process"] = None
                return web.json_response({"success": False, "error": str(e)})

        async def tagging_status(request):
            """라벨링 진행률 쿼리"""
            return web.json_response({
                "success": True,
                "running": _tagging_state["running"],
                "current": _tagging_state["current"],
                "total": _tagging_state["total"],
                "current_subdir": _tagging_state.get("current_subdir", ""),
                "log_lines": _tagging_state["log_lines"][-50:],
            })

        async def stop_tagging(request):
            """현재 라벨링 중지/취소"""
            try:
                if not _tagging_state["running"]:
                    return web.json_response({"success": False, "error": "레이블링 진행 중 없음"})

                _tagging_state["cancel_requested"] = True
                proc = _tagging_state.get("process")
                if proc and proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()

                _tagging_state["running"] = False
                _tagging_state["process"] = None
                return web.json_response({"success": True})
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        # ===== 생성 정규화 이미지 (기반 소재) caption） =====
        async def start_gen_reg(request):
            """재료 기반 caption을 생성하여 이미지 정규화（anima / sdxl）"""
            try:
                data = await request.json()
                model_family = (data.get("model_family") or "anima").strip()
                material_dir = (data.get("material_dir") or "").strip()
                reg_dir = (data.get("reg_dir") or "").strip()
                sd_scripts_dir = (data.get("sd_scripts_dir") or SD_SCRIPTS_DIR or "").strip()
                python_path = (data.get("python_path") or "").strip() or sys.executable

                base_model = (data.get("base_model") or "").strip()
                dit = (data.get("dit") or "").strip()
                vae = (data.get("vae") or "").strip()
                text_encoder = (data.get("text_encoder") or "").strip()

                steps = int(data.get("steps", 25))
                width = int(data.get("width", 1024))
                height = int(data.get("height", 1024))
                # anima/sdxl 가로 세로가 32의 배수여야 하며, 정수형으로 내림하여 오버플로우 오류가 발생하는 것을 방지합니다
                width = _round_down_to_multiple(width, 32)
                height = _round_down_to_multiple(height, 32)
                cfg = float(data.get("cfg", 7.5))
                # 프론트엔드는 문자열로 시드를 전달합니다(64비트 시드가 JS 안전 정수 범위 초과), int()를 사용하여 정확하게 파싱합니다.；
                # or 42로 기본 빈 문자열 사용, 방지 int("") 던 ValueError
                seed = int(data.get("seed") or 42)
                images_per_prompt = max(1, int(data.get("images_per_prompt", 1)))
                sampler = (data.get("sampler") or "euler_a").strip()
                precision = (data.get("precision") or "fp16").strip()
                attn_mode = (data.get("attn_mode") or "torch").strip()
                # 각 이미지별 독립적인 64비트 시드: 전면에서 전달된 고정 시드를 무시합니다. seed
                random_per_image = bool(data.get("random_per_image", False))
                write_caption = bool(data.get("write_caption", True))
                skip_first_tag = bool(data.get("skip_first_tag", True))

                # 검증
                if not material_dir or not os.path.isdir(material_dir):
                    return web.json_response({"success": False, "error": "리소스 폴더가 존재하지 않습니다. 먼저 리소스 페이지에서 리소스 경로를 설정하십시오."})
                if not reg_dir:
                    return web.json_response({"success": False, "error": "정규화 디렉토리가 설정되지 않았습니다. 정규화 페이지에서 경로를 선택/입력하십시오."})
                if not sd_scripts_dir or not os.path.isdir(sd_scripts_dir):
                    return web.json_response({"success": False, "error": f"sd-scripts 폴더가 없습니다: {sd_scripts_dir or '(공)'}。파라미터 페이지에서 훈련 스크립트 경로가 올바르게 감지/설치되었는지 확인해주세요."})
                if model_family == 'sdxl':
                    if not base_model or not os.path.isfile(base_model):
                        return web.json_response({"success": False, "error": f"SDXL 기본 모델(ckpt) 설정되지 않음 또는 존재하지 않음: {base_model}。파라미터 페이지에서 기본 모델을 설정하세요"})
                    script_name = "sdxl_gen_img.py"
                else:
                    model_family = 'anima'
                    if not dit or not os.path.isfile(dit):
                        return web.json_response({"success": False, "error": f"anima DiT 모델이 설정되지 않았거나 존재하지 않습니다: {dit}。파라미터 페이지에서 확산 모델을 설정하세요"})
                    if not text_encoder or not os.path.isfile(text_encoder):
                        return web.json_response({"success": False, "error": f"anima 텍스트 인코더가 설정되지 않았거나 존재하지 않습니다: {text_encoder}。Please go to the parameters page to set the text encoder"})
                    script_name = "anima_minimal_inference.py"
                script_path = os.path.join(sd_scripts_dir, script_name)
                if not os.path.isfile(script_path):
                    return web.json_response({"success": False, "error": f"추론 스크립트가 존재하지 않음: {script_path}"})

                os.makedirs(reg_dir, exist_ok=True)

                # 자료 서브 디렉토리를 스캔합니다 (N_name 규격)）
                reg_basename = os.path.basename(os.path.normpath(reg_dir))
                subdirs = []
                for entry in sorted(os.listdir(material_dir)):
                    full = os.path.join(material_dir, entry)
                    if not os.path.isdir(full):
                        continue
                    if entry.startswith('.') or entry == '__pycache__':
                        continue
                    # 정규화 출력 디렉토리를 건너뛰어 reg을 자료 디렉토리로 인식하지 않도록 합니다.
                    if full == reg_dir or entry == reg_basename:
                        continue
                    subdirs.append((entry, full))
                # 리소스 폴더가 서브 폴더가 없는 평면 구조인 경우 전체를 그룹으로 취급，suffix=reg
                if not subdirs:
                    subdirs = [('', material_dir)]

                # 작업 생성: (src, target, prefix, start_seq)
                tasks = []
                for name, full in subdirs:
                    if name:
                        m = re.match(r'^\d+_(.+)$', name)
                        suffix = m.group(1) if m else name
                    else:
                        suffix = 'reg'
                    target = os.path.join(reg_dir, f"1_{suffix}")
                    os.makedirs(target, exist_ok=True)
                    start_seq = 0
                    for f in os.listdir(target):
                        pm = re.match(re.escape(suffix) + r'_(\d+)$', os.path.splitext(f)[0])
                        if pm:
                            start_seq = max(start_seq, int(pm.group(1)) + 1)
                    tasks.append({"src": full, "target": target, "prefix": suffix, "seq": start_seq})

                # 탐색기 스크립트의 가용성 확인 (anima_train 모듈 필요)，
                # (anima_minimal_inference.py와 동일한 출처의 Anima 프레임워크). 사용 가능한 경우
                # 모델은 한 번만 로드하여 모든 이미지를 순회, 각 이미지를 반복적으로 로드하지 않음。
                one_load_ok = False
                one_load_script = os.path.join(sd_scripts_dir, "gen_reg_anima.py")
                if os.path.isfile(one_load_script):
                    try:
                        _probe = subprocess.run(
                            [python_path or sys.executable, "-c", "import anima_train"],
                            cwd=sd_scripts_dir, capture_output=True,
                            text=True, timeout=30,
                        )
                        one_load_ok = (_probe.returncode == 0)
                    except Exception:
                        one_load_ok = False
                    if not one_load_ok:
                        _gen_reg_state["log_lines"] = [
                            "[알림] anima_train 모듈이 감지되지 않았습니다. 셔츠별 자식 프로세스 생성으로 되돌아갑니다"
                            "（모델을 다시 로드하여 느림）",
                        ]

                _gen_reg_state.update({
                    "running": True, "current": 0, "total": 0,
                    "log_lines": _gen_reg_state.get("log_lines", []), "process": None,
                    "current_subdir": "",
                    "cancel_requested": False,
                })

                def _monitor():
                    tmp_dir = tempfile.mkdtemp(prefix="bsk_genreg_")
                    try:
                        # 평면화된 작업: 각 이미지별 사전 계산 prompt / out_path / seed
                        jobs = _build_gen_reg_jobs(
                            tasks, images_per_prompt, seed, width, height,
                            steps, cfg, sampler, precision, attn_mode,
                            write_caption, skip_first_tag,
                            random_per_image=random_per_image,
                        )
                        attempted = [j for j in jobs if not j.get("skip")]
                        _gen_reg_state["total"] = len(attempted)

                        # 일회성 스크립트를 우선 사용합니다: 모델은 한 번만 로드하고 모든 맵을 반복합니다
                        one_load_script = os.path.join(sd_scripts_dir, "gen_reg_anima.py")
                        if one_load_ok and attempted and os.path.isfile(one_load_script):
                            _run_gen_reg_one_load(
                                one_load_script, python_path, sd_scripts_dir,
                                dit, vae, text_encoder, attn_mode, precision,
                                width, height, steps, cfg, sampler,
                                attempted, _gen_reg_state,
                            )
                            return

                        # 되돌립니다: 각 프로세스별로（anima_minimal_inference.py / sdxl_gen_img.py）
                        done = 0
                        gen_idx = 0
                        for t in tasks:
                            if _gen_reg_state["cancel_requested"]:
                                _gen_reg_state["log_lines"].append("[취소] 사용자 생성 취소")
                                break
                            _gen_reg_state["current_subdir"] = os.path.basename(t["target"])
                            _gen_reg_state["log_lines"].append(
                                f"[자 디렉토리] {t['src']} -> {t['target']} (prefix={t['prefix']}, start={t['seq']})")
                            imgs = [f for f in sorted(os.listdir(t["src"]))
                                    if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
                                    and os.path.isfile(os.path.join(t["src"], f))]
                            seq = t["seq"]
                            for img in imgs:
                                if _gen_reg_state["cancel_requested"]:
                                    break
                                base = os.path.splitext(img)[0]
                                cap = ""
                                for ext in ['.txt', '.caption', '.tags']:
                                    cp = os.path.join(t["src"], base + ext)
                                    if os.path.isfile(cp):
                                        with open(cp, 'r', encoding='utf-8') as fh:
                                            cap = fh.read().strip()
                                        break
                                prompt = _skip_first_tag(cap) if skip_first_tag else cap
                                if not prompt:
                                    _gen_reg_state["log_lines"].append(f"[건너뛰기] {img} 유효하지 않음 caption")
                                    continue
                                for _k in range(images_per_prompt):
                                    if _gen_reg_state["cancel_requested"]:
                                        break
                                    out_name = f"{t['prefix']}_{seq:05d}.png"
                                    out_path = os.path.join(t["target"], out_name)
                                    # 지우 임시 폴더
                                    for f in os.listdir(tmp_dir):
                                        try:
                                            os.remove(os.path.join(tmp_dir, f))
                                        except OSError:
                                            pass
                                    cur_seed = random.randint(0, 2**64 - 1) if random_per_image else seed + gen_idx
                                    params = {
                                        "steps": steps, "width": width, "height": height,
                                        "cfg": cfg, "seed": cur_seed, "sampler": sampler,
                                        "precision": precision, "attn_mode": attn_mode,
                                        "tmp_dir": tmp_dir,
                                    }
                                    paths = {
                                        "base_model": base_model, "dit": dit, "vae": vae,
                                        "text_encoder": text_encoder,
                                    }
                                    cmd = _build_gen_reg_cmd(model_family, script_path, python_path, prompt, params, paths)
                                    _gen_reg_state["log_lines"].append(
                                        f"[생성] {img} -> {out_name} (seed={cur_seed}) prompt={prompt[:60]}")
                                    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                                            stderr=subprocess.STDOUT, text=True,
                                                            encoding='utf-8', errors='replace',
                                                            cwd=sd_scripts_dir)
                                    _gen_reg_state["process"] = proc
                                    for line in proc.stdout:
                                        line = line.rstrip()
                                        _gen_reg_state["log_lines"].append(line)
                                        if len(_gen_reg_state["log_lines"]) > 200:
                                            _gen_reg_state["log_lines"] = _gen_reg_state["log_lines"][-200:]
                                    proc.wait()
                                    _gen_reg_state["process"] = None
                                    if proc.returncode != 0:
                                        _gen_reg_state["log_lines"].append(
                                            f"[오류] 생성 실패: {img} (returncode={proc.returncode})")
                                        gen_idx += 1
                                        continue
                                    if _collect_gen_output(model_family, tmp_dir, out_path):
                                        if write_caption:
                                            cap_path = os.path.splitext(out_path)[0] + '.txt'
                                            with open(cap_path, 'w', encoding='utf-8') as fh:
                                                fh.write(prompt)
                                        seq += 1
                                        done += 1
                                        gen_idx += 1
                                        _gen_reg_state["current"] = done
                                    else:
                                        _gen_reg_state["log_lines"].append(
                                            f"[오류] 생성 출력 찾을 수 없음: {img}")
                                        gen_idx += 1
                    except Exception as e:
                        _gen_reg_state["log_lines"].append(f"[예외] {e}")
                    finally:
                        shutil.rmtree(tmp_dir, ignore_errors=True)
                        _gen_reg_state["running"] = False
                        _gen_reg_state["process"] = None
                        _gen_reg_state["current_subdir"] = ""

                threading.Thread(target=_monitor, daemon=True).start()
                return web.json_response({"success": True, "total": 0})
            except Exception as e:
                _gen_reg_state["running"] = False
                _gen_reg_state["process"] = None
                return web.json_response({"success": False, "error": str(e)})

        async def gen_reg_status(request):
            """정규 표현 그래프 생성 쿼리 진행률"""
            return web.json_response({
                "success": True,
                "running": _gen_reg_state["running"],
                "current": _gen_reg_state["current"],
                "total": _gen_reg_state["total"],
                "current_subdir": _gen_reg_state.get("current_subdir", ""),
                "log_lines": _gen_reg_state["log_lines"][-50:],
            })

        async def stop_gen_reg(request):
            """현재 생성 중지/취소"""
            try:
                if not _gen_reg_state["running"]:
                    return web.json_response({"success": False, "error": "생성 진행 중 없음"})
                _gen_reg_state["cancel_requested"] = True
                # 알림이 한 번 실행되는 스크립트가 깔끔하게 종료됩니다 (_run_gen_reg_one_load와 동일) pid）
                try:
                    _cf = os.path.join(
                        tempfile.gettempdir(), f"bsk_genreg_cancel_{os.getpid()}.tmp")
                    with open(_cf, "w", encoding="utf-8") as _fh:
                        _fh.write("1")
                except OSError:
                    pass
                proc = _gen_reg_state.get("process")
                if proc and proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                _gen_reg_state["running"] = False
                _gen_reg_state["process"] = None
                return web.json_response({"success": True})
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        # ===== 이미지 전처리 =====
        async def start_preprocess(request):
            """시작 이미지 전처리"""
            try:
                data = await request.json()
                dir_path = data.get("dir", "").strip()
                enable_resize = data.get("enable_resize", False)
                target_resolution = int(data.get("target_resolution", 1024))
                num_threads = int(data.get("num_threads", 4))
                
                if not dir_path or not os.path.isdir(dir_path):
                    return web.json_response({"success": False, "error": "폴더가 없습니다"})
                
                # 보안 검사
                allowed, real_path = _is_path_allowed(dir_path)
                if not allowed:
                    return web.json_response({"success": False, "error": "경로가 허용되지 않습니다"})
                
                if _preprocess_state["running"]:
                    return web.json_response({"success": False, "error": "기존 전처리 작업이 실행 중"})
                
                # 상태 재설정
                _preprocess_state["running"] = True
                _preprocess_state["current"] = 0
                _preprocess_state["total"] = 0
                _preprocess_state["log_lines"] = []
                _preprocess_state["cancel_requested"] = False
                
                # 백그라운드 스레드 시작
                threading.Thread(
                    target=_preprocess_images_worker,
                    args=(real_path, enable_resize, target_resolution, num_threads),
                    daemon=True
                ).start()
                
                return web.json_response({"success": True})
            except Exception as e:
                _preprocess_state["running"] = False
                return web.json_response({"success": False, "error": str(e)})
        
        async def preprocess_status(request):
            """미리 처리 진행률 조회"""
            return web.json_response({
                "success": True,
                "running": _preprocess_state["running"],
                "current": _preprocess_state["current"],
                "total": _preprocess_state["total"],
                "log_lines": _preprocess_state["log_lines"][-50:],
            })
        
        async def stop_preprocess(request):
            """사전 처리 중지"""
            try:
                if not _preprocess_state["running"]:
                    return web.json_response({"success": False, "error": "현재 사전 처리 없음"})
                
                _preprocess_state["cancel_requested"] = True
                return web.json_response({"success": True})
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        # ===== 설정 관리 =====
        async def generate_config(request):
            """TOML 훈련 구성 파일 생성"""
            try:
                data = await request.json()
                toml_content = data.get("toml", "")
                name = data.get("name", "training_config")

                config_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "training_configs")
                os.makedirs(config_dir, exist_ok=True)

                safe_name = os.path.basename(name)
                if not safe_name.endswith(".toml"):
                    safe_name += ".toml"

                config_path = os.path.join(config_dir, safe_name)
                with open(config_path, "w", encoding="utf-8") as f:
                    f.write(toml_content)

                return web.json_response({"success": True, "path": config_path})
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        async def save_config(request):
            """훈련 설정 저장"""
            try:
                data = await request.json()
                name = data.get("name", "training_config")
                toml = data.get("toml", "")

                config_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "training_configs")
                os.makedirs(config_dir, exist_ok=True)

                safe_name = os.path.basename(name)
                if not safe_name.endswith(".toml"):
                    safe_name += ".toml"

                config_path = os.path.join(config_dir, safe_name)
                with open(config_path, "w", encoding="utf-8") as f:
                    f.write(toml)

                return web.json_response({"success": True, "name": safe_name})
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        async def list_configs(request):
            """저장된 훈련 설정 목록"""
            try:
                config_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "training_configs")
                if not os.path.isdir(config_dir):
                    return web.json_response({"success": True, "files": []})

                files = [f for f in os.listdir(config_dir) if f.endswith(".toml")]
                files.sort()
                return web.json_response({"success": True, "files": files})
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        async def load_config(request):
            """훈련 설정 로드"""
            try:
                name = request.query.get("name", "")
                if not name:
                    return web.json_response({"success": False, "error": "파일 이름이 없습니다"})

                safe_name = os.path.basename(name)
                config_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "training_configs")
                config_path = os.path.join(config_dir, safe_name)

                if not os.path.exists(config_path):
                    return web.json_response({"success": False, "error": "설정 파일이 존재하지 않음"})

                with open(config_path, "r", encoding="utf-8") as f:
                    content = f.read()

                return web.json_response({"success": True, "toml": content})
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        async def save_params(request):
            """훈련 매개변수 설정을 JSON 파일로 저장합니다."""
            try:
                data = await request.json()
                name = data.get("name", "").strip()
                params = data.get("params", {})
                if not name:
                    return web.json_response({"success": False, "error": "설정 이름을 입력하십시오"})

                config_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "training_configs")
                os.makedirs(config_dir, exist_ok=True)

                safe_name = os.path.basename(name)
                if not safe_name.endswith(".json"):
                    safe_name += ".json"

                config_path = os.path.join(config_dir, safe_name)
                with open(config_path, "w", encoding="utf-8") as f:
                    json.dump(params, f, ensure_ascii=False, indent=2)

                return web.json_response({"success": True, "name": safe_name, "path": config_path})
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        async def list_params(request):
            """저장된 학습 파라미터 구성 목록"""
            try:
                config_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "training_configs")
                if not os.path.isdir(config_dir):
                    return web.json_response({"success": True, "files": []})

                files = [f for f in os.listdir(config_dir) if f.endswith(".json") and not f.startswith("_")]
                files.sort()
                # 각 구성의 이름 정보를 읽습니다
                configs = []
                for f in files:
                    fpath = os.path.join(config_dir, f)
                    try:
                        with open(fpath, "r", encoding="utf-8") as fp:
                            d = json.load(fp)
                        configs.append({
                            "file": f,
                            "name": f.replace(".json", ""),
                            "outputName": d.get("outputName", ""),
                            "modelFamily": d.get("modelFamily", ""),
                        })
                    except Exception:
                        configs.append({"file": f, "name": f.replace(".json", ""), "outputName": "", "modelFamily": ""})
                return web.json_response({"success": True, "configs": configs})
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        async def load_params(request):
            """훈련 매개변수 설정 로드"""
            try:
                name = request.query.get("name", "")
                if not name:
                    return web.json_response({"success": False, "error": "파일 이름이 없습니다"})

                safe_name = os.path.basename(name)
                if not safe_name.endswith(".json"):
                    safe_name += ".json"

                config_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "training_configs")
                config_path = os.path.join(config_dir, safe_name)

                if not os.path.exists(config_path):
                    return web.json_response({"success": False, "error": "설정 파일이 존재하지 않음"})

                with open(config_path, "r", encoding="utf-8") as f:
                    content = f.read()

                return web.json_response({"success": True, "params": json.loads(content), "name": safe_name})
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        async def delete_params(request):
            """훈련 매개변수 설정 삭제"""
            try:
                data = await request.json()
                name = data.get("name", "").strip()
                if not name:
                    return web.json_response({"success": False, "error": "파일 이름이 없습니다"})

                safe_name = os.path.basename(name)
                if not safe_name.endswith(".json"):
                    safe_name += ".json"

                config_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "training_configs")
                config_path = os.path.join(config_dir, safe_name)

                if os.path.exists(config_path):
                    os.remove(config_path)

                return web.json_response({"success": True})
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        # ===== 훈련 프로세스 관리 =====
        async def start_training(request):
            """훈련 프로세스 시작"""
            try:
                data = await request.json()
                config_toml = data.get("config_toml", "")
                script_name = data.get("script_name", "")  # 프론트엔드 명시적 전달
                mixed_precision = data.get("mixed_precision", "fp16")
                output_dir = data.get("output_dir", "")

                if not config_toml:
                    return web.json_response({"success": False, "error": "훈련 설정을 누락했습니다"})

                if _training_state["status"] == "running":
                    return web.json_response({"success": False, "error": "현재 훈련 중"})

                # sd-scripts 디렉토리 확인
                if not os.path.isdir(SD_SCRIPTS_DIR):
                    return web.json_response({"success": False, "error": f"sd-scripts 폴더가 없습니다: {SD_SCRIPTS_DIR}"})

                # 설정을 임시 파일로 저장
                config_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "training_configs")
                os.makedirs(config_dir, exist_ok=True)
                config_path = os.path.join(config_dir, "_current_training.toml")
                with open(config_path, "w", encoding="utf-8") as f:
                    f.write(config_toml)

                # sample_prompts.txt 저장 (프론트엔드에서 내용 전달 시)）
                sample_prompts_content = data.get("sample_prompts", "")
                sample_prompts_path = os.path.join(config_dir, "sample_prompts.txt")
                if sample_prompts_content:
                    with open(sample_prompts_path, "w", encoding="utf-8") as f:
                        f.write(sample_prompts_content)
                    # TOML의 sample_prompts 경로를 절대 경로로 대체
                    config_toml = config_toml.replace(
                        'sample_prompts = "sample_prompts.txt"',
                        f'sample_prompts = "{sample_prompts_path.replace(os.sep, "/")}"'
                    )
                    # 재작성 TOML 파일
                    with open(config_path, "w", encoding="utf-8") as f:
                        f.write(config_toml)
                # TOML이 sample_prompts.txt를 참조했지만 내용이 비어 있으면 기존 파일 삭제
                elif os.path.exists(sample_prompts_path):
                    os.remove(sample_prompts_path)

                # 훈련 스크립트 확인
                if not script_name:
                    script_name = "train_network.py"
                script_path = os.path.join(SD_SCRIPTS_DIR, script_name)
                if not os.path.exists(script_path):
                    return web.json_response({"success": False, "error": f"훈련 스크립트가 존재하지 않습니다: {script_name}"})

                # build accelerate launch 명령
                cmd = ["accelerate", "launch"]

                # GPU 선택
                gpu_ids = data.get("gpu_ids", "").strip()
                if gpu_ids:
                    cmd.extend(["--gpu_ids", gpu_ids])

                # 멀티 GPU
                num_processes = int(data.get("num_processes", 1))
                if num_processes > 1:
                    cmd.extend(["--num_processes", str(num_processes)])

                # 멀티 머신
                num_machines = int(data.get("num_machines", 1))
                if num_machines > 1:
                    cmd.extend(["--num_machines", str(num_machines)])
                    main_process_ip = data.get("main_process_ip", "")
                    if main_process_ip:
                        cmd.extend(["--main_process_ip", main_process_ip])
                    main_process_port = int(data.get("main_process_port", 29500))
                    cmd.extend(["--main_process_port", str(main_process_port)])
                    machine_rank = int(data.get("machine_rank", 0))
                    cmd.extend(["--machine_rank", str(machine_rank)])

                # DeepSpeed
                use_deepspeed = data.get("use_deepspeed", False)
                if use_deepspeed:
                    cmd.append("--use_deepspeed")
                    zero_stage = int(data.get("zero_stage", 2))
                    cmd.extend(["--zero_stage", str(zero_stage)])
                    offload_opt = data.get("offload_optimizer_device", "")
                    if offload_opt:
                        cmd.extend(["--offload_optimizer_device", offload_opt])
                    offload_param = data.get("offload_param_device", "")
                    if offload_param:
                        cmd.extend(["--offload_param_device", offload_param])

                # 혼합 정밀도
                cmd.append(f"--mixed_precision={mixed_precision}")

                # DDP 최적화
                ddp_timeout = data.get("ddp_timeout", "")
                if ddp_timeout:
                    cmd.extend(["--ddp_timeout", str(int(float(ddp_timeout) * 60))])  # 분당 초 변환
                if data.get("ddp_gradient_as_bucket_view", False):
                    cmd.append("--gradient_as_bucket_view")
                if data.get("ddp_static_graph", False):
                    cmd.append("--static_graph")

                # CPU Thread count
                cmd.extend(["--num_cpu_threads_per_process", "1"])

                # 스크립트 및 설정
                cmd.extend([script_path, "--config_file", config_path])

                # output_dir 파싱: 프론트엔드로 전달된 경우 우선 사용하고, 그렇지 않으면 TOML 파싱
                if not output_dir:
                    m = re.search(r'output_dir\s*=\s*"([^"]+)"', config_toml)
                    if m:
                        output_dir = m.group(1)
                if not output_dir:
                    output_dir = os.path.join(SD_SCRIPTS_DIR, "output")

                # output_dir이 절대 경로인지 확인 (sd-scripts 디렉터리 상대적으로 해석))
                if not os.path.isabs(output_dir):
                    output_dir = os.path.abspath(os.path.join(SD_SCRIPTS_DIR, output_dir))

                # 다시 훈련 시작
                _training_state["status"] = "running"
                _training_state["current_step"] = 0
                # 최대 학습 단계 수를 TOML에서 파싱하려고 시도합니다. total_steps
                total_steps = 0
                m = re.search(r'max_train_steps\s*=\s*(\d+)', config_toml)
                if m:
                    total_steps = int(m.group(1))
                _training_state["total_steps"] = total_steps
                _training_state["current_loss"] = 0.0
                _training_state["current_epoch"] = 0
                # max_train_epochs 파싱하여 후속 추정 total_steps
                m_epochs = re.search(r'max_train_epochs\s*=\s*(\d+)', config_toml)
                _training_state["max_train_epochs"] = int(m_epochs.group(1)) if m_epochs else 0
                # 현재 aiohttp 단일 스레드 모드이며 독립적인 다운로드 명령이 없습니다(aria2 또는 자동 모드로 전환하여 복사 명령을 사용해 주세요). total_steps
                if total_steps == 0 and _training_state["max_train_epochs"] > 0:
                    num_images = len(_training_state.get("images", []))
                    m_bs = re.search(r'train_batch_size\s*=\s*(\d+)', config_toml)
                    m_ga = re.search(r'gradient_accumulation_steps\s*=\s*(\d+)', config_toml)
                    batch_size = int(m_bs.group(1)) if m_bs else 1
                    grad_acc = int(m_ga.group(1)) if m_ga else 1
                    if num_images > 0:
                        import math
                        steps_per_epoch = math.ceil(num_images / batch_size / grad_acc)
                        _training_state["total_steps"] = _training_state["max_train_epochs"] * steps_per_epoch
                _training_state["log_lines"] = []
                _training_state["saved_models"] = []
                _training_state["sample_images"] = []
                _training_state["loss_history"] = []
                _training_state["config_path"] = config_path
                _training_state["output_dir"] = output_dir
                # 훈련 시작 시 output_dir 최종 결정: 경로 백업 캐시 초기화，
                # 그렇지 않으면 _ALLOWED_IMAGE_ROOTS는 단계 1(이때 output_dir은 아직 None)에서 처음
                # 경로 유효성 검사 시 캐싱되어 후속 삭제 샘플 이미지(output_dir 아래)가 판별됨"경로가 허용되지 않습니다"
                _ALLOWED_IMAGE_ROOTS = None
                _training_state["script_name"] = script_name
                _training_state["start_time"] = time.time()
                _save_state()
                # 프로세스를 시작합니다 (GPU 선택을 지원하도록 환경 변수를 설정)）
                env = os.environ.copy()
                if gpu_ids:
                    env["ASCEND_RT_VISIBLE_DEVICES"] = gpu_ids
                    env["CUDA_VISIBLE_DEVICES"] = gpu_ids

                # 현재 Python 환경에서 강제로 사용 accelerate
                python_dir = os.path.dirname(sys.executable)
                if sys.platform == "win32":
                    bin_dir = os.path.join(python_dir, "Scripts")
                else:
                    bin_dir = os.path.join(python_dir, "bin")

                # 현재 Python의 bin 디렉토리를 PATH 맨 앞에 추가합니다
                old_path = env.get("PATH", "")
                env["PATH"] = bin_dir + os.pathsep + old_path

                # 잠재적으로 방해되는 Python 환경 변수를 지우십시오
                # 기존 PYTHONPATH 유지, NPU 관련 경로도 포함 확인
                old_pythonpath = env.get("PYTHONPATH", "")

                # 현재 기능 설명이 없으며, 이중 클릭하여 추가
                ascend_home = env.get('ASCEND_HOME', '/usr/local/Ascend/ascend-toolkit/latest')
                if not os.path.exists(ascend_home):
                    ascend_home = '/usr/local/Ascend/ascend-toolkit/latest'

                npu_paths = []
                tbe_path = os.path.join(ascend_home, 'opp', 'built-in', 'op_impl', 'ai_core', 'tbe')
                python_site = os.path.join(ascend_home, 'python', 'site-packages')
                if os.path.exists(tbe_path):
                    npu_paths.append(tbe_path)
                if os.path.exists(python_site):
                    npu_paths.append(python_site)

                # 모든 경로 병합
                all_paths = []
                if old_pythonpath:
                    all_paths.extend(old_pythonpath.split(os.pathsep))
                all_paths.extend(npu_paths)

                # 중복 제거 및 순서 유지
                unique_paths = []
                for p in all_paths:
                    if p and p not in unique_paths:
                        unique_paths.append(p)

                if unique_paths:
                    env["PYTHONPATH"] = os.pathsep.join(unique_paths)

                env.pop("PYTHONHOME", None)
                # 터미널 열 너비를 강제로 설정하여 rich/logging 라이브러리가 PIPE를 감지하여 각 줄을 단일 문자 줄 바꿈으로 분할하는 것을 방지합니다.
                env["COLUMNS"] = "200"

                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding='utf-8',
                    errors='replace',
                    env=env,
                    cwd=SD_SCRIPTS_DIR,
                )
                _training_state["process"] = proc

                # 백엔드 스레드가 출력을 읽습니다
                def _read_output():
                    for line in proc.stdout:
                        line = line.rstrip()
                        if not line:
                            continue
                        _training_state["log_lines"].append(line)
                        # 최근 500줄 유지
                        if len(_training_state["log_lines"]) > 500:
                            _training_state["log_lines"] = _training_state["log_lines"][-500:]

                        # 파싱 step/loss
                        _parse_training_line(line)

                    proc.wait()
                    if _training_state["status"] == "running":
                        _training_state["status"] = "completed"
                    _save_state()

                threading.Thread(target=_read_output, daemon=True).start()

                return web.json_response({"success": True, "script": script_name})
            except Exception as e:
                _training_state["status"] = "error"
                _save_state()
                return web.json_response({"success": False, "error": str(e)})

        async def training_status(request):
            """훈련 상태 쿼리"""
            # 저장된 모델 목록 동적 가져오기 (파일 시스템 스캔))
            output_dir = _training_state.get("output_dir")
            saved_models_detail = []
            if output_dir and os.path.isdir(output_dir):
                for f in os.listdir(output_dir):
                    ext = os.path.splitext(f)[1].lower()
                    if ext in ['.safetensors', '.pt', '.ckpt']:
                        fp = os.path.join(output_dir, f)
                        if os.path.isfile(fp):
                            saved_models_detail.append({
                                "name": f,
                                "path": f,
                                "size": os.path.getsize(fp),
                                "created": os.path.getctime(fp),
                            })
                saved_models_detail.sort(key=lambda x: x.get("name", ""))

            return web.json_response({
                "success": True,
                "status": _training_state["status"],
                "current_step": _training_state["current_step"],
                "total_steps": _training_state["total_steps"],
                "current_loss": _training_state["current_loss"],
                "current_epoch": _training_state["current_epoch"],
                "max_train_epochs": _training_state.get("max_train_epochs", 0),
                "elapsed": time.time() - _training_state["start_time"] if _training_state["start_time"] else 0,
                "loss_history": _training_state.get("loss_history", [])[-200:],
                "saved_models": _training_state.get("saved_models", []),
                "saved_models_detail": saved_models_detail,
                "output_dir": _training_state.get("output_dir", ""),
                "script_name": _training_state.get("script_name", ""),
            })

        async def training_logs(request):
            """훈련 로그 가져오기"""
            try:
                tail = int(request.query.get("tail", "50"))
                lines = _training_state["log_lines"][-tail:]
                return web.json_response({"success": True, "lines": lines})
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        async def pause_training(request):
            """훈련 일시 중지 (프로세스 신호)"""
            try:
                if _training_state["status"] != "running":
                    return web.json_response({"success": False, "error": "훈련이 실행 중 아님"})

                proc = _training_state.get("process")
                if not proc or proc.poll() is not None:
                    return web.json_response({"success": False, "error": "훈련 프로세스가 존재하지 않습니다"})

                # Windows: CREATE_SUSPENDED는 작동하지 않으므로 SIGBREAK를 사용하여 일시 중지합니다.
                # 실제 솔루션: Windows에서 프로세스 트리 전체를 일시 중지
                try:
                    if sys.platform == "win32":
                        # Windows: 디버그 API를 통해 모든 스레드 일시 중지
                        import ctypes
                        kernel32 = ctypes.windll.kernel32
                        PROCESS_SUSPEND_RESUME = 0x0800
                        handle = kernel32.OpenProcess(PROCESS_SUSPEND_RESUME, False, proc.pid)
                        if handle:
                            # NtSuspendProcess
                            ntdll = ctypes.windll.ntdll
                            ntdll.NtSuspendProcess(handle)
                            kernel32.CloseHandle(handle)
                        else:
                            return web.json_response({"success": False, "error": "프로세스 일시 중단이 불가능합니다(Windows 권한 부족))"})
                    else:
                        os.kill(proc.pid, signal_module.SIGSTOP)
                except Exception as e:
                    return web.json_response({"success": False, "error": f"일시 중지 실패: {e}"})

                _training_state["status"] = "paused"
                _save_state()
                return web.json_response({"success": True})
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        async def resume_training(request):
            """훈련 복원"""
            try:
                if _training_state["status"] != "paused":
                    return web.json_response({"success": False, "error": "훈련이 중지되지 않음"})

                proc = _training_state.get("process")
                if not proc or proc.poll() is not None:
                    return web.json_response({"success": False, "error": "훈련 프로세스가 종료되었습니다"})

                try:
                    if sys.platform == "win32":
                        import ctypes
                        kernel32 = ctypes.windll.kernel32
                        PROCESS_SUSPEND_RESUME = 0x0800
                        handle = kernel32.OpenProcess(PROCESS_SUSPEND_RESUME, False, proc.pid)
                        if handle:
                            ntdll = ctypes.windll.ntdll
                            ntdll.NtResumeProcess(handle)
                            kernel32.CloseHandle(handle)
                    else:
                        os.kill(proc.pid, signal_module.SIGCONT)
                except Exception as e:
                    return web.json_response({"success": False, "error": f"복원 실패: {e}"})

                _training_state["status"] = "running"
                _save_state()
                return web.json_response({"success": True})
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        async def stop_training(request):
            """훈련 중단 - psutil을 사용하여 프로세스 트리 재귀 종료"""
            try:
                proc = _training_state.get("process")
                killed_pids = []
                if proc and proc.poll() is None:
                    pid = proc.pid
                    
                    # psutil(가장 안정적))
                    if HAS_PSUTIL:
                        try:
                            parent = psutil.Process(pid)
                            # 마스크 디버깅 도구가 활성화되었습니다
                            children = parent.children(recursive=True)
                            killed_pids.append(f"psutil: found {len(children)} children")
                            
                            # 먼저 모든 자식 프로세스 종료
                            for child in children:
                                try:
                                    child.terminate()
                                    killed_pids.append(f"terminate child: {child.pid}")
                                except psutil.NoSuchProcess:
                                    pass
                                except Exception as e:
                                    # terminate실패 시kill
                                    try:
                                        child.kill()
                                        killed_pids.append(f"kill child: {child.pid}")
                                    except Exception:
                                        pass
                            
                            # 자식 프로세스 종료 대기 (최대 3초))
                            gone, alive = psutil.wait_procs(children, timeout=3)
                            for p in alive:
                                try:
                                    p.kill()
                                    killed_pids.append(f"force kill child: {p.pid}")
                                except Exception:
                                    pass
                            
                            # 상위 프로세스 종료
                            try:
                                parent.terminate()
                                parent.wait(timeout=5)
                                killed_pids.append("parent terminated")
                            except psutil.TimeoutExpired:
                                parent.kill()
                                killed_pids.append("parent killed")
                            except psutil.NoSuchProcess:
                                killed_pids.append("parent already gone")
                                
                        except psutil.NoSuchProcess:
                            killed_pids.append("process not found")
                        except Exception as e:
                            killed_pids.append(f"psutil error: {e}")
                    
                    # 대체 방법: 기존 taskkill/wmic 방법
                    if not killed_pids or "error" in str(killed_pids):
                        if sys.platform == "win32":
                            # Windows: taskkill프로세스 트리 살려내기
                            try:
                                result = subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                                             capture_output=True, timeout=10, text=True)
                                killed_pids.append(f"taskkill: {result.returncode}")
                            except Exception as e:
                                killed_pids.append(f"taskkill failed: {e}")
                            
                            # 모든 하위 프로세스 찾아서 죽임
                            try:
                                find_cmd = f'wmic process where "ParentProcessId={pid}" get ProcessId'
                                result = subprocess.run(find_cmd, shell=True, capture_output=True, timeout=5, text=True)
                                for line in result.stdout.strip().split('\n'):
                                    line = line.strip()
                                    if line and line.isdigit():
                                        child_pid = int(line)
                                        try:
                                            subprocess.run(["taskkill", "/F", "/PID", str(child_pid)],
                                                         capture_output=True, timeout=3)
                                            killed_pids.append(f"wmic child: {child_pid}")
                                        except Exception:
                                            pass
                            except Exception as e:
                                killed_pids.append(f"wmic failed: {e}")
                        else:
                            # Linux/Mac: SIGTERM을 프로세스 그룹으로 전송
                            try:
                                os.killpg(os.getpgid(pid), signal_module.SIGTERM)
                                killed_pids.append("killpg SIGTERM")
                            except Exception:
                                proc.terminate()
                                killed_pids.append("terminate")
                        
                        # 프로세스 종료 대기
                        try:
                            proc.wait(timeout=10)
                            killed_pids.append("wait ok")
                        except subprocess.TimeoutExpired:
                            proc.kill()
                            killed_pids.append("force kill")

                _training_state["status"] = "idle"
                _training_state["process"] = None
                _save_state()
                return web.json_response({"success": True, "killed": killed_pids})
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        async def training_samples(request):
            """훈련 샘플 이미지 목록 가져오기 - output_dir/sample 하위 디렉터리 스캔"""
            try:
                output_dir = _training_state.get("output_dir")
                if not output_dir or not os.path.isdir(output_dir):
                    return web.json_response({"success": True, "images": [], "sample_dir": ""})

                # 우선 샘플 하위 디렉터 스캔 후 output_dir 자체 스캔
                sample_dirs = []
                sample_subdir = os.path.join(output_dir, "sample")
                if os.path.isdir(sample_subdir):
                    sample_dirs.append(sample_subdir)
                sample_dirs.append(output_dir)

                samples = []
                for scan_dir in sample_dirs:
                    if not os.path.isdir(scan_dir):
                        continue
                    for f in os.listdir(scan_dir):
                        ext = os.path.splitext(f)[1].lower()
                        if ext not in IMAGE_EXTENSIONS:
                            continue
                        file_path = os.path.join(scan_dir, f)
                        if not os.path.isfile(file_path):
                            continue
                        # 중복 방지
                        if any(s["path"] == file_path for s in samples):
                            continue
                        samples.append({
                            "name": f,
                            "path": file_path,
                            "mtime": os.path.getmtime(file_path),
                            "size": os.path.getsize(file_path),
                        })

                samples.sort(key=lambda x: x["name"])
                return web.json_response({"success": True, "images": samples, "sample_dir": sample_subdir if os.path.isdir(sample_subdir) else ""})
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        async def training_samples_diff(request):
            """Incremental get training sample image changes - Compare client known file fingerprint"""
            try:
                data = await request.json()
                known = data.get("known", {})  # {filename: {mtime: float, size: int}}

                output_dir = _training_state.get("output_dir")
                if not output_dir or not os.path.isdir(output_dir):
                    return web.json_response({"success": True, "added": [], "removed": [], "unchanged": []})

                # 현재 디렉토리 스캔 (training_samples 로직과 동일)）
                sample_dirs = []
                sample_subdir = os.path.join(output_dir, "sample")
                if os.path.isdir(sample_subdir):
                    sample_dirs.append(sample_subdir)
                sample_dirs.append(output_dir)

                current_files = {}  # {filename: {path, mtime, size}}
                for scan_dir in sample_dirs:
                    if not os.path.isdir(scan_dir):
                        continue
                    for f in os.listdir(scan_dir):
                        ext = os.path.splitext(f)[1].lower()
                        if ext not in IMAGE_EXTENSIONS:
                            continue
                        file_path = os.path.join(scan_dir, f)
                        if not os.path.isfile(file_path):
                            continue
                        if f in current_files:
                            continue  # 중복 방지
                        stat = os.stat(file_path)
                        current_files[f] = {
                            "name": f,
                            "path": file_path,
                            "mtime": stat.st_mtime,
                            "size": stat.st_size,
                        }

                # 지문 비교
                added = []
                removed = []
                unchanged = []

                for name, info in current_files.items():
                    if name not in known:
                        added.append(info)
                    else:
                        k = known[name]
                        if abs(info["mtime"] - k.get("mtime", 0)) > 0.5 or info["size"] != k.get("size"):
                            added.append(info)  # 지문 변화는 업데이트로 간주
                        else:
                            unchanged.append(name)

                for name in known:
                    if name not in current_files:
                        removed.append(name)

                return web.json_response({
                    "success": True,
                    "added": added,
                    "removed": removed,
                    "unchanged": unchanged,
                })
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        async def training_models(request):
            """저장된 모델 목록 가져오기"""
            try:
                output_dir = _training_state.get("output_dir")
                if not output_dir or not os.path.isdir(output_dir):
                    return web.json_response({"success": True, "models": []})

                models = []
                for f in os.listdir(output_dir):
                    ext = os.path.splitext(f)[1].lower()
                    if ext in ['.safetensors', '.pt', '.ckpt']:
                        file_path = os.path.join(output_dir, f)
                        if not os.path.isfile(file_path):
                            continue
                        models.append({
                            "name": f,
                            "path": file_path,
                            "size": os.path.getsize(file_path),
                            "mtime": os.path.getmtime(file_path),
                        })

                models.sort(key=lambda x: x["mtime"], reverse=True)
                return web.json_response({"success": True, "models": models})
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        async def copy_to_loras(request):
            """복사 모델 파일 ComfyUI models/loras 폴더로"""
            try:
                import shutil
                data = await request.json()
                filename = data.get("path", "").strip()
                # 안전: 파일 이름만 허용, 경로 탐색 금지
                if not filename or '/' in filename or '\\' in filename or '..' in filename:
                    return web.json_response({"success": False, "error": "Invalid filename"})
                output_dir = _training_state.get("output_dir", "")
                if not output_dir:
                    return web.json_response({"success": False, "error": "No output directory"})
                model_path = os.path.join(output_dir, filename)
                model_path = os.path.normpath(model_path)
                if not model_path.startswith(os.path.normpath(output_dir)):
                    return web.json_response({"success": False, "error": "Access denied"})
                if not os.path.isfile(model_path):
                    return web.json_response({"success": False, "error": f"파일이 존재하지 않습니다: {filename}"})

                # 대상 디렉토리: LoRA 출력 디렉토리 (환경 변수 덮어쓰기 지원).）
                loras_dir = _get_loras_dir()
                os.makedirs(loras_dir, exist_ok=True)

                # 대상 파일 이름
                filename = os.path.basename(model_path)
                target_path = os.path.join(loras_dir, filename)

                # 대상 파일이 존재하면 접미사 추가
                if os.path.exists(target_path):
                    base, ext = os.path.splitext(filename)
                    i = 1
                    while os.path.exists(os.path.join(loras_dir, f"{base}_{i}{ext}")):
                        i += 1
                    target_path = os.path.join(loras_dir, f"{base}_{i}{ext}")

                shutil.copy2(model_path, target_path)
                return web.json_response({
                    "success": True,
                    "target_path": target_path,
                    "message": f"복사됨 {os.path.basename(target_path)}"
                })
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        async def download_model(request):
            """모델 파일 다운로드"""
            try:
                filename = request.query.get("path", "")
                # 안전: 파일 이름만 허용, 경로 탐색 금지
                if not filename or '/' in filename or '\\' in filename or '..' in filename:
                    return web.Response(status=400, text="Invalid filename")
                output_dir = _training_state.get("output_dir", "")
                if not output_dir:
                    return web.Response(status=400, text="No output directory")
                model_path = os.path.join(output_dir, filename)
                # 확인된 경로가 output_dir 내에 있는지 확인합니다
                model_path = os.path.normpath(model_path)
                if not model_path.startswith(os.path.normpath(output_dir)):
                    return web.Response(status=403, text="Access denied")
                if not os.path.isfile(model_path):
                    return web.Response(status=404, text="File not found")

                ext = os.path.splitext(model_path)[1].lower()
                if ext not in ['.safetensors', '.pt', '.ckpt', '.pth', '.bin']:
                    return web.Response(status=400, text="Not a model file")

                return web.FileResponse(model_path, headers={
                    "Content-Disposition": f'attachment; filename="{os.path.basename(model_path)}"'
                })
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        async def tensorboard_data(request):
            """TensorBoard 로그 데이터 읽기"""
            try:
                # 로그 디렉토리 가져오기
                output_dir = _training_state.get("output_dir")
                if not output_dir:
                    output_dir = os.path.join(SD_SCRIPTS_DIR or ".", "output")
                logging_dir = os.path.join(output_dir, "logs")

                if not os.path.isdir(logging_dir):
                    return web.json_response({"success": True, "data": {}, "message": "로그 디렉토리 없음"})

                # 최신 로그 하위 디렉토리 찾음
                log_subdirs = [d for d in os.listdir(logging_dir) if os.path.isdir(os.path.join(logging_dir, d))]
                if not log_subdirs:
                    return web.json_response({"success": True, "data": {}, "message": "무 로그 파일"})

                log_subdirs.sort(reverse=True)
                latest_log_dir = os.path.join(logging_dir, log_subdirs[0])

                # 이벤트 파일 찾기
                event_files = []
                for root, dirs, files in os.walk(latest_log_dir):
                    for f in files:
                        if f.startswith("events.out.tfevents."):
                            event_files.append(os.path.join(root, f))

                if not event_files:
                    return web.json_response({"success": True, "data": {}, "message": "무 이벤트 파일"})

                # tensorboard 라이브러리를 사용하여 분석을 시도합니다
                try:
                    from tensorboard.backend.event_processing import event_accumulator
                    all_data = {}

                    for event_file in event_files:
                        try:
                            ea = event_accumulator.EventAccumulator(event_file)
                            ea.Reload()
                            tags = ea.Tags().get('scalars', [])

                            for tag in tags:
                                events = ea.Scalars(tag)
                                if tag not in all_data:
                                    all_data[tag] = []
                                for e in events:
                                    all_data[tag].append({
                                        "step": e.step,
                                        "value": e.value,
                                        "wall_time": e.wall_time,
                                    })
                        except Exception:
                            continue

                    return web.json_response({"success": True, "data": all_data, "log_dir": latest_log_dir})

                except ImportError:
                    # tensorboard설치되지 않음, 팁 반환
                    return web.json_response({
                        "success": True,
                        "data": {},
                        "message": "필요 설치tensorboard: pip install tensorboard",
                        "log_dir": latest_log_dir,
                    })

            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        # ===== 서브 디렉토리 관리 =====
        async def scan_subdirs(request):
            """이미지 자료 경로 하위 디렉터리(N_name 규격 준수) 및 삭제된 디렉터리 스캔"""
            try:
                data = await request.json()
                dir_path = data.get("dir", "").strip()
                if not dir_path:
                    return web.json_response({"success": False, "error": "Path cannot be empty"})
                allowed, real_path = _is_path_allowed(dir_path)
                if not allowed:
                    return web.json_response({"success": False, "error": "경로가 허용되지 않습니다"})
                if not os.path.isdir(real_path):
                    return web.json_response({"success": False, "error": "폴더가 없습니다"})
                dir_path = real_path

                subdirs = []
                for entry in os.listdir(dir_path):
                    full = os.path.join(dir_path, entry)
                    if not os.path.isdir(full):
                        continue
                    if entry.startswith('.') or entry == '__pycache__':
                        continue
                    # 이미지 수량 통계
                    count = sum(1 for f in os.listdir(full)
                                if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
                                and os.path.isfile(os.path.join(full, f)))
                    subdirs.append({"name": entry, "count": count})

                subdirs.sort(key=lambda x: x["name"].lower())

                # 전체 자료 저장소 파일의 링크를 프로젝트 폴더에 연결
                removed = []
                archived_dir = os.path.join(dir_path, ".archived")
                if os.path.isdir(archived_dir):
                    for entry in os.listdir(archived_dir):
                        full = os.path.join(archived_dir, entry)
                        if os.path.isdir(full):
                            count = sum(1 for f in os.listdir(full)
                                        if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
                                        and os.path.isfile(os.path.join(full, f)))
                            removed.append({"name": entry, "count": count})
                    removed.sort(key=lambda x: x["name"].lower())

                return web.json_response({"success": True, "subdirs": subdirs, "removed": removed})
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        async def scan_subdir_images(request):
            """지정된 하위 디렉터리의 이미지 목록 스캔"""
            try:
                data = await request.json()
                dir_path = data.get("dir", "").strip()
                if not dir_path:
                    return web.json_response({"success": False, "error": "Path cannot be empty"})
                allowed, real_path = _is_path_allowed(dir_path)
                if not allowed:
                    return web.json_response({"success": False, "error": "경로가 허용되지 않습니다"})
                if not os.path.isdir(real_path):
                    return web.json_response({"success": False, "error": "폴더가 없습니다"})
                dir_path = real_path

                images = []
                for f in os.listdir(dir_path):
                    ext = os.path.splitext(f)[1].lower()
                    if ext not in IMAGE_EXTENSIONS:
                        continue
                    file_path = os.path.join(dir_path, f)
                    if not os.path.isfile(file_path):
                        continue
                    stat = os.stat(file_path)
                    images.append({
                        "name": f,
                        "path": file_path,
                        "size": stat.st_size,
                    })
                images.sort(key=lambda x: x["name"].lower())
                return web.json_response({"success": True, "images": images})
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        async def delete_image(request):
            """지정된 이미지 파일 삭제"""
            try:
                data = await request.json()
                file_path = data.get("path", "").strip()
                if not file_path or not os.path.isfile(file_path):
                    return web.json_response({"success": False, "error": "파일이 존재하지 않습니다"})
                # 보안 검사: 경로가 허용 범위 내에 있는지 확인
                allowed, _ = _is_path_allowed(file_path)
                if not allowed:
                    return web.json_response({"success": False, "error": "경로가 허용되지 않습니다"})
                os.remove(file_path)
                # 해당 레이블 파일 동시에 삭제（.txt/.caption）
                for ext in ['.txt', '.caption', '.tags']:
                    tag_path = os.path.splitext(file_path)[0] + ext
                    if os.path.isfile(tag_path):
                        os.remove(tag_path)
                return web.json_response({"success": True})
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        async def read_caption(request):
            """이미지 관련 프롬프트 파일 읽기(.txt)"""
            try:
                data = await request.json()
                file_path = data.get("path", "").strip()
                if not file_path:
                    return web.json_response({"success": False, "error": "Path cannot be empty"})
                allowed, real_path = _is_path_allowed(file_path)
                if not allowed:
                    return web.json_response({"success": False, "error": "경로가 허용되지 않습니다"})
                # .txt 및 .caption 확장명 시도
                base = os.path.splitext(real_path)[0]
                for ext in ['.txt', '.caption', '.tags']:
                    cap_path = base + ext
                    if os.path.isfile(cap_path):
                        with open(cap_path, 'r', encoding='utf-8') as f:
                            return web.json_response({"success": True, "caption": f.read(), "file": cap_path})
                return web.json_response({"success": True, "caption": "", "file": ""})
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        async def save_caption(request):
            """이미지 저장에 대한 프롬프트 파일(.txt)"""
            try:
                data = await request.json()
                file_path = data.get("path", "").strip()
                caption = data.get("caption", "")
                if not file_path:
                    return web.json_response({"success": False, "error": "Path cannot be empty"})
                allowed, real_path = _is_path_allowed(file_path)
                if not allowed:
                    return web.json_response({"success": False, "error": "경로가 허용되지 않습니다"})
                # 저장 .txt
                cap_path = os.path.splitext(real_path)[0] + '.txt'
                with open(cap_path, 'w', encoding='utf-8') as f:
                    f.write(caption)
                return web.json_response({"success": True, "file": cap_path})
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        async def add_subdir(request):
            """서브 디렉토리 추가, 자동으로 1_ 접두사 추가"""
            try:
                data = await request.json()
                dir_path = data.get("dir", "").strip()
                name = data.get("name", "").strip()
                if not dir_path or not name:
                    return web.json_response({"success": False, "error": "파라미터 불완전"})
                # 보안 검사
                allowed, real_path = _is_path_allowed(dir_path)
                if not allowed:
                    return web.json_response({"success": False, "error": "경로가 허용되지 않습니다"})
                dir_path = real_path

                # 자동 접두사 추가
                if not re.match(r'^\d+_', name):
                    name = '1_' + name

                full_path = os.path.join(dir_path, name)
                if os.path.exists(full_path):
                    return web.json_response({"success": False, "error": f"메뉴가 이미 존재합니다: {name}"})

                os.makedirs(full_path, exist_ok=True)
                return web.json_response({"success": True, "name": name})
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        async def create_image_dir(request):
            """리소스/정규화 루트 디렉토리를 만듭니다 (허용된 화이트리스트 루트 디렉토리 내에 있어야 합니다)）"""
            try:
                data = await request.json()
                dir_path = data.get("dir", "").strip()
                if not dir_path:
                    return web.json_response({"success": False, "error": "Path cannot be empty"})
                # 보안 검사: 백 목록 루트 디렉토리 내에서 (경로 탐색 방지)）
                allowed, real_path = _is_path_allowed(dir_path)
                if not allowed:
                    return web.json_response({"success": False, "error": "경로가 허용된 범위 내에 없습니다"})
                os.makedirs(real_path, exist_ok=True)
                return web.json_response({"success": True, "real_path": real_path})
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        async def remove_subdir(request):
            """서브 디렉토리 제거 .archived"""
            try:
                data = await request.json()
                dir_path = data.get("dir", "").strip()
                name = data.get("name", "").strip()
                if not dir_path or not name:
                    return web.json_response({"success": False, "error": "파라미터 불완전"})
                # 보안 검사
                allowed, real_path = _is_path_allowed(dir_path)
                if not allowed:
                    return web.json_response({"success": False, "error": "경로가 허용되지 않습니다"})
                dir_path = real_path

                src = os.path.join(dir_path, name)
                if not os.path.isdir(src):
                    return web.json_response({"success": False, "error": "폴더가 없습니다"})

                archived_dir = os.path.join(dir_path, ".archived")
                os.makedirs(archived_dir, exist_ok=True)
                dst = os.path.join(archived_dir, name)

                # 대상 파일이 존재하면 접미사 추가
                if os.path.exists(dst):
                    i = 1
                    while os.path.exists(f"{dst}_{i}"):
                        i += 1
                    dst = f"{dst}_{i}"

                import shutil
                shutil.move(src, dst)
                return web.json_response({"success": True})
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        async def move_images(request):
            """리소스 디렉토리 내에서 이미지 파일 이동 (서브 디렉토리 간)）"""
            try:
                import shutil
                data = await request.json()
                dir_path = data.get("dir", "").strip()
                source_subdir = data.get("source_subdir", "").strip()
                target_subdir = data.get("target_subdir", "").strip()
                files = data.get("files", [])
                if not dir_path or not files:
                    return web.json_response({"success": False, "error": "파라미터 불완전"})
                # 보안 검사
                allowed, real_path = _is_path_allowed(dir_path)
                if not allowed:
                    return web.json_response({"success": False, "error": "경로가 허용되지 않습니다"})
                dir_path = real_path

                src_dir = os.path.join(dir_path, source_subdir) if source_subdir else dir_path
                dst_dir = os.path.join(dir_path, target_subdir) if target_subdir else dir_path
                for d in (src_dir, dst_dir):
                    os.makedirs(d, exist_ok=True)

                moved = 0
                errors = []
                for fname in files:
                    src = os.path.join(src_dir, fname)
                    dst = os.path.join(dst_dir, fname)
                    if not os.path.isfile(src):
                        errors.append(fname)
                        continue
                    try:
                        if os.path.exists(dst):
                            # 목표가 이미 존재합니다, 건너듸합니다
                            errors.append(fname + " (목표가 이미 존재합니다)")
                            continue
                        shutil.move(src, dst)
                        # 동시에 해당 caption 파일 이동 (.txt / .caption)
                        for ext in (".txt", ".caption"):
                            src_cap = src + ext
                            dst_cap = dst + ext
                            if os.path.exists(src_cap):
                                shutil.move(src_cap, dst_cap)
                        moved += 1
                    except Exception as ex:
                        errors.append(fname + ": " + str(ex))

                return web.json_response({
                    "success": True,
                    "moved": moved,
                    "errors": errors,
                    "error": "; ".join(errors) if errors else None,
                })
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        async def rename_subdir(request):
            """Rename subfolder"""
            try:
                import shutil
                data = await request.json()
                dir_path = data.get("dir", "").strip()
                name = data.get("name", "").strip()
                new_name = data.get("new_name", "").strip()
                archived = data.get("archived", False)
                if not dir_path or not name or not new_name:
                    return web.json_response({"success": False, "error": "파라미터 불완전"})
                # 보안 검사
                allowed, real_path = _is_path_allowed(dir_path)
                if not allowed:
                    return web.json_response({"success": False, "error": "경로가 허용되지 않습니다"})
                dir_path = real_path

                if archived:
                    src = os.path.join(dir_path, ".archived", name)
                    dst = os.path.join(dir_path, ".archived", new_name)
                else:
                    src = os.path.join(dir_path, name)
                    dst = os.path.join(dir_path, new_name)

                if not os.path.isdir(src):
                    return web.json_response({"success": False, "error": "폴더가 없습니다"})
                if os.path.exists(dst):
                    return web.json_response({"success": False, "error": f"대상 이름이 존재함: {new_name}"})

                shutil.move(src, dst)
                return web.json_response({"success": True, "name": new_name})
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        async def restore_subdir(request):
            """.archived에서 하위 디렉터리를 복원합니다."""
            try:
                data = await request.json()
                dir_path = data.get("dir", "").strip()
                name = data.get("name", "").strip()
                if not dir_path or not name:
                    return web.json_response({"success": False, "error": "파라미터 불완전"})
                # 보안 검사
                allowed, real_path = _is_path_allowed(dir_path)
                if not allowed:
                    return web.json_response({"success": False, "error": "경로가 허용되지 않습니다"})
                dir_path = real_path

                src = os.path.join(dir_path, ".archived", name)
                if not os.path.isdir(src):
                    return web.json_response({"success": False, "error": "삭제된 디렉터리가 존재하지 않습니다"})

                dst = os.path.join(dir_path, name)
                if os.path.exists(dst):
                    return web.json_response({"success": False, "error": f"대상 디렉토리가 존재함: {name}"})

                import shutil
                shutil.move(src, dst)
                return web.json_response({"success": True})
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        # ===== 파일 업로드 =====
        async def upload_files(request):
            """이미지/폴더/zip을 자재 디렉토리로 업로드"""
            try:
                import shutil
                import zipfile
                from aiohttp import web as aio_web

                data = await request.post()
                base_dir = data.get("baseDir", "").strip()
                target_subdir = data.get("targetSubdir", "").strip()
                mode = data.get("mode", "images")

                if not base_dir or not os.path.isdir(base_dir):
                    return web.json_response({"success": False, "error": "재료 경로가 유효하지 않습니다"})

                # 보안 검사: base_dir이 허용 범위 내에 있는지 확인
                allowed, real_base_dir = _is_path_allowed(base_dir)
                if not allowed:
                    return web.json_response({"success": False, "error": "경로가 허용되지 않습니다"})

                files = data.getall("files", [])
                if not files:
                    return web.json_response({"success": False, "error": "파일 없음"})

                uploaded_count = 0

                for file_obj in files:
                    filename = file_obj.filename
                    content = file_obj.file.read()

                    if filename.lower().endswith('.zip'):
                        # zip처리: 임시 디렉토리로 압축, 구조 분석 후 이동
                        import tempfile
                        with tempfile.TemporaryDirectory() as tmp_dir:
                            zip_path = os.path.join(tmp_dir, filename)
                            with open(zip_path, 'wb') as zf:
                                zf.write(content)
                            try:
                                def _fix_zip_name(info):
                                    """zip 내 파일 이름 디코딩 수정(중국어 등 비 ASCII 파일 이름 깨짐 처리, __init__.py 방식)）。"""
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
                                with zipfile.ZipFile(zip_path, 'r') as zf:
                                    for _info in zf.infolist():
                                        _info.filename = _fix_zip_name(_info)
                                    zf.extractall(tmp_dir)
                            except zipfile.BadZipFile:
                                continue

                            # 압축된 구조 분석
                            # 이미지 파일 찾기
                            extracted_images = []
                            for root, dirs, fnames in os.walk(tmp_dir):
                                for f in fnames:
                                    if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS:
                                        extracted_images.append(os.path.join(root, f))

                            if not extracted_images:
                                continue

                            # 판단: 이미지 자식 디렉토리 내에 있는지
                            subdir_images = {}  # subdir_name -> [image_paths]
                            for img_path in extracted_images:
                                rel = os.path.relpath(img_path, tmp_dir)
                                parts = rel.replace('\\', '/').split('/')
                                if len(parts) > 1:
                                    # 상위 폴더
                                    parent = parts[0]
                                    if parent not in subdir_images:
                                        subdir_images[parent] = []
                                    subdir_images[parent].append(img_path)
                                else:
                                    # 루트 디렉토리에 직접
                                    if '_root' not in subdir_images:
                                        subdir_images['_root'] = []
                                    subdir_images['_root'].append(img_path)

                            for subdir_name, img_paths in subdir_images.items():
                                if subdir_name == '_root':
                                    # 선택된 하위 디렉토리에 배치
                                    if target_subdir:
                                        dest_dir = os.path.join(base_dir, target_subdir)
                                    else:
                                        # 자동 생성
                                        dest_dir = os.path.join(base_dir, "1_concept")
                                        if not os.path.exists(dest_dir):
                                            os.makedirs(dest_dir, exist_ok=True)
                                else:
                                    # 자식 디렉토리 생성, 자동으로 접두사 추가
                                    safe_name = os.path.basename(subdir_name)
                                    if not re.match(r'^\d+_', safe_name):
                                        safe_name = '1_' + safe_name
                                    dest_dir = os.path.join(base_dir, safe_name)

                                os.makedirs(dest_dir, exist_ok=True)
                                for img_path in img_paths:
                                    dest = os.path.join(dest_dir, os.path.basename(img_path))
                                    if not os.path.exists(dest):
                                        shutil.copy2(img_path, dest)
                                        uploaded_count += 1
                                    # 동 이름 복사txt/json
                                    for ext in ['.txt', '.json']:
                                        sidecar = os.path.splitext(img_path)[0] + ext
                                        if os.path.exists(sidecar):
                                            sidecar_dest = os.path.join(dest_dir, os.path.basename(sidecar))
                                            if not os.path.exists(sidecar_dest):
                                                shutil.copy2(sidecar, sidecar_dest)

                    elif mode == 'folder':
                        # 폴더 업로드: webkitdirectory는 상대 경로를 유지합니다.
                        # file_obj.filename 가능성이다 "subdir/image.png"
                        rel_path = filename.replace('\\', '/')
                        parts = rel_path.split('/')
                        ext = os.path.splitext(parts[-1])[1].lower()

                        if ext not in IMAGE_EXTENSIONS and ext not in ['.txt', '.json']:
                            continue

                        if len(parts) > 1:
                            # 하위 폴더
                            folder_name = parts[0]
                            if not re.match(r'^\d+_', folder_name):
                                folder_name = '1_' + folder_name
                            dest_dir = os.path.join(base_dir, folder_name)
                        else:
                            # 직접 파일
                            if target_subdir:
                                dest_dir = os.path.join(base_dir, target_subdir)
                            else:
                                dest_dir = os.path.join(base_dir, "1_concept")
                                if not os.path.exists(dest_dir):
                                    os.makedirs(dest_dir, exist_ok=True)

                        os.makedirs(dest_dir, exist_ok=True)
                        dest = os.path.join(dest_dir, parts[-1])
                        if not os.path.exists(dest):
                            with open(dest, 'wb') as f:
                                f.write(content)
                            if ext in IMAGE_EXTENSIONS:
                                uploaded_count += 1

                    else:
                        # 일반 이미지 업로드
                        ext = os.path.splitext(filename)[1].lower()
                        if ext not in IMAGE_EXTENSIONS:
                            continue

                        if target_subdir:
                            dest_dir = os.path.join(base_dir, target_subdir)
                        else:
                            dest_dir = os.path.join(base_dir, "1_concept")
                            if not os.path.exists(dest_dir):
                                os.makedirs(dest_dir, exist_ok=True)

                        os.makedirs(dest_dir, exist_ok=True)
                        dest = os.path.join(dest_dir, filename)
                        if not os.path.exists(dest):
                            with open(dest, 'wb') as f:
                                f.write(content)
                            uploaded_count += 1

                return web.json_response({"success": True, "uploaded": uploaded_count})
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        # ===== 이미지 프록시 =====
        async def serve_image(request):
            """대리 이미지 파일 액세스, 지원 thumb=1 생성 썸네일，webp=1 전체 크기 WebP 미리보기"""
            try:
                img_path = request.query.get("path", "")
                thumb = request.query.get("thumb", "")
                webp = request.query.get("webp", "")
                if not img_path or not os.path.isfile(img_path):
                    return web.Response(status=404, text="Not found")

                ext = os.path.splitext(img_path)[1].lower()
                if ext not in IMAGE_EXTENSIONS:
                    return web.Response(status=400, text="Not an image")

                # 보안 검사: 캐시를 의존하지 않고 동적으로 구성된 허용된 루트 디렉토리를 생성합니다）
                allowed_roots = [os.path.realpath(r) for r in [_get_train_data_dir()] if os.path.isdir(r)]
                extra_env = os.environ.get('TRAINING_ALLOWED_DIRS', '')
                if extra_env:
                    for d in extra_env.split(os.pathsep):
                        d = d.strip()
                        if d and os.path.isdir(d):
                            allowed_roots.append(os.path.realpath(d))
                output_dir = _training_state.get("output_dir")
                if output_dir and os.path.isdir(output_dir):
                    real_output = os.path.realpath(output_dir)
                    if real_output not in allowed_roots:
                        allowed_roots.append(real_output)
                allowed, _ = _is_path_allowed(img_path, allowed_roots=allowed_roots if allowed_roots else None)
                if not allowed:
                    return web.Response(status=403, text="Forbidden")

                if thumb:
                    try:
                        from PIL import Image
                        import io
                        img = Image.open(img_path)
                        img.thumbnail((256, 256), Image.LANCZOS)
                        if img.mode not in ('RGB', 'L'):
                            img = img.convert('RGB')
                        buf = io.BytesIO()
                        img.save(buf, format='WEBP', quality=80)
                        buf.seek(0)
                        img.close()
                        return web.Response(
                            body=buf.read(),
                            content_type='image/webp',
                            headers={'Cache-Control': 'max-age=30'},
                        )
                    except Exception:
                        return web.FileResponse(img_path)

                if webp:
                    try:
                        from PIL import Image
                        import io
                        img = Image.open(img_path)
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
                        buf = io.BytesIO()
                        img.save(buf, format='WEBP', quality=85, method=6)
                        buf.seek(0)
                        img.close()
                        return web.Response(
                            body=buf.read(),
                            content_type='image/webp',
                            headers={'Cache-Control': 'max-age=300'},
                        )
                    except Exception:
                        return web.FileResponse(img_path)

                return web.FileResponse(img_path)
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        # ===== 목차 탐색 =====
        async def browse_dir(request):
            """디렉토리 구조를 탐색하고 하위 디렉토리 목록을 반환합니다 (허용 범위 내에서)）"""
            try:
                data = await request.json()
                dir_path = data.get("dir", "").strip()
                if not dir_path:
                    # 반환 허용 루트 디렉토리 목록
                    roots = _get_allowed_image_roots()
                    dirs = [{"name": os.path.basename(r) or r, "path": r, "type": "dir"} for r in roots]
                    return web.json_response({"success": True, "dirs": dirs, "current": ""})
                # 보안 검사: 브라우징 범위 제한
                allowed, real_path = _is_path_allowed(dir_path)
                if not allowed:
                    return web.json_response({"success": False, "error": "경로가 허용되지 않습니다"})
                dir_path = real_path
                if not os.path.isdir(dir_path):
                    return web.json_response({"success": False, "error": "폴더가 없습니다"})

                dirs = []
                try:
                    for entry in os.listdir(dir_path):
                        full = os.path.join(dir_path, entry)
                        if os.path.isdir(full):
                            # 숨겨진 디렉토리 및 시스템 디렉토리 건너뛰기
                            if entry.startswith('.') or entry == '__pycache__':
                                continue
                            dirs.append({"name": entry, "path": full, "type": "dir"})
                except PermissionError:
                    pass

                dirs.sort(key=lambda x: x["name"].lower())
                return web.json_response({"success": True, "dirs": dirs, "current": dir_path})
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        # ===== output 디렉토리의 LoRA 파일과 훈련 상태 디렉토리를 나열합니다 =====
        async def list_output_files(request):
            """output 디렉토리의 LoRA/모델 파일과 훈련 상태 디렉토리를 나열합니다"""
            try:
                output_dir = _training_state.get("output_dir", "")
                if not output_dir or not os.path.isdir(output_dir):
                    return web.json_response({"success": True, "loras": [], "states": []})
                loras = []
                states = []
                for f in os.listdir(output_dir):
                    fp = os.path.join(output_dir, f)
                    if os.path.isfile(fp):
                        ext = os.path.splitext(f)[1].lower()
                        if ext in ('.safetensors', '.pt', '.ckpt'):
                            loras.append({"name": f, "path": fp})
                    elif os.path.isdir(fp) and f.endswith('-state'):
                        states.append({"name": f, "path": fp})
                loras.sort(key=lambda x: x["name"].lower())
                states.sort(key=lambda x: x["name"].lower())
                return web.json_response({"success": True, "loras": loras, "states": states})
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        # ===== 샘플 이미지 파일 서비스 =====
        async def sample_image(request):
            """훈련 샘플 이미지 가져오기"""
            try:
                img_path = request.query.get("path", "")
                if not img_path or not os.path.isfile(img_path):
                    return web.Response(status=404, text="Not found")

                ext = os.path.splitext(img_path)[1].lower()
                if ext not in IMAGE_EXTENSIONS:
                    return web.Response(status=400, text="Not an image")

                from aiohttp import web as aio_web
                return web.FileResponse(img_path)
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        # ===== T5 Tokenizer 관리 =====
        async def t5_tokenizer_status(request):
            """T5 토크나이저 다운로드 여부 확인, 경로 및 파일 목록 반환"""
            try:
                # 가능한 t5_tokenizer 디렉터리 검색.
                search_paths = [
                    os.path.join(_COMFYUI_ROOT, "models", "t5_tokenizer"),
                    os.path.join(os.path.dirname(_COMFYUI_ROOT), "models", "t5_tokenizer"),
                    os.path.join(os.path.dirname(os.path.realpath(__file__)), "models", "t5_tokenizer"),
                    os.path.join(SD_SCRIPTS_DIR or "", "configs", "t5_old"),
                ]
                # 또한 검색 folder_paths
                try:
                    import folder_paths
                    fp_paths = folder_paths.get_folder_paths("t5_tokenizer")
                    if fp_paths:
                        search_paths = [p for p in fp_paths if os.path.isdir(p)] + search_paths
                except (ImportError, IndexError, KeyError):
                    pass

                found_path = None
                found_files = []
                required_files = ["spiece.model", "tokenizer_config.json", "special_tokens_map.json"]

                for sp in search_paths:
                    if not sp or not os.path.isdir(sp):
                        continue
                    existing = [f for f in required_files if os.path.isfile(os.path.join(sp, f))]
                    if len(existing) >= 2:  # 최소 2개의 코어 파일이 있으면 유효하다고 간주합니다
                        found_path = sp
                        all_files = os.listdir(sp)
                        found_files = [f for f in all_files if os.path.isfile(os.path.join(sp, f))]
                        break

                # sd-scripts의 내장된 것 확인 t5_old
                builtin_path = os.path.join(SD_SCRIPTS_DIR or "", "configs", "t5_old") if SD_SCRIPTS_DIR else None
                builtin_available = False
                if builtin_path and os.path.isdir(builtin_path):
                    builtin_files = os.listdir(builtin_path)
                    if "spiece.model" in builtin_files:
                        builtin_available = True

                return web.json_response({
                    "success": True,
                    "found": found_path is not None,
                    "path": found_path or "",
                    "files": found_files,
                    "builtin_available": builtin_available,
                    "builtin_path": builtin_path or "",
                    "search_paths": [p for p in search_paths if p],
                })
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        async def download_t5_tokenizer(request):
            """T5 토크나이저(google/t5-v1_1-xxl) 다운로드 models/t5_tokenizer/
            여러 이미지 소스 지원: auto(기본, hf-mirror 우선 백트레일)), hf-mirror, huggingface, modelscope
            """
            try:
                data = await request.json()
                source = data.get("source", "auto") if data else "auto"  # auto / hf-mirror / huggingface / modelscope

                # 목표 디렉토리 확인 (환경 변수 덮어쓰기 지원)）
                t5_tokenizer_dir = os.environ.get('T5_TOKENIZER_DIR', '')
                if t5_tokenizer_dir:
                    target_dir = t5_tokenizer_dir
                else:
                    target_dir = os.path.join(_COMFYUI_ROOT, "models", "t5_tokenizer")
                os.makedirs(target_dir, exist_ok=True)

                # 다운로드할 파일 목록
                t5_files = ["spiece.model", "tokenizer_config.json", "special_tokens_map.json"]

                # 모두 존재하는지 확인합니다
                existing = [f for f in t5_files if os.path.isfile(os.path.join(target_dir, f))]
                if len(existing) == len(t5_files):
                    return web.json_response({"success": True, "message": "T5 tokenizer 존재함", "path": target_dir})

                # ---- 이미지 소스 설정 ----
                # HF 미러 엔드포인트 목록 (우선순위별))
                HF_MIRRORS = [
                    ("hf-mirror", "https://hf-mirror.com"),
                    ("huggingface", "https://huggingface.co"),
                ]

                # 빌드 시도 순서
                if source == "auto":
                    # 자동 모드: hf-mirror 우선, 공식으로 롤백
                    mirror_order = HF_MIRRORS
                elif source == "hf-mirror":
                    mirror_order = [HF_MIRRORS[0]]
                elif source == "huggingface":
                    mirror_order = [HF_MIRRORS[1]]
                elif source == "modelscope":
                    mirror_order = []  # ModelScope 독립적인 로직으로 진행
                else:
                    mirror_order = HF_MIRRORS

                # ---- 방법 1: huggingface_hub 다운로드 (이미지 엔드포인트 지원)) ----
                if mirror_order:
                    try:
                        from huggingface_hub import hf_hub_download
                    except ImportError:
                        return web.json_response({"success": False, "error": "필요 설치 huggingface_hub: pip install huggingface-hub"})

                    last_error = None
                    for mirror_name, endpoint in mirror_order:
                        try:
                            for fname in t5_files:
                                if os.path.isfile(os.path.join(target_dir, fname)):
                                    continue
                                downloaded = hf_hub_download(
                                    repo_id="google/t5-v1_1-xxl",
                                    filename=fname,
                                    local_dir=target_dir,
                                    endpoint=endpoint,
                                )
                                # hf_hub_download 서브 디렉터리 다운로드 가능, 대상 디렉터리로 이동
                                if downloaded and os.path.isfile(downloaded):
                                    target_file = os.path.join(target_dir, fname)
                                    if downloaded != target_file and not os.path.exists(target_file):
                                        import shutil
                                        shutil.copy2(downloaded, target_file)

                            return web.json_response({
                                "success": True,
                                "message": f"T5 tokenizer 다운로드 완료 (via {mirror_name})",
                                "path": target_dir,
                                "source": mirror_name,
                            })
                        except Exception as e:
                            last_error = e
                            continue

                    # 모든 HF 이미지가 실패했습니다
                    if source in ("hf-mirror", "huggingface"):
                        # 사용자가 단일 소스를 지정하여 오류가 발생합니다
                        return web.json_response({"success": False, "error": f"에서 {source} 다운로드 실패: {last_error}"})

                # ---- 방법 2: ModelScope 다운로드 ----
                if source in ("auto", "modelscope"):
                    try:
                        from modelscope.hub.file_download import model_file_download
                        from modelscope.hub.snapshot_download import snapshot_download

                        # ModelScope google/t5-v1_1-xxl에 해당하는 모델을 선택합니다. ID
                        # 사용자가 ModelScope에 이 모델을 업로드했을 수 있으며, 일반적 ID:
                        ms_repo_candidates = [
                            "iic/t5-v1_1-xxl",           # 마다 공식에서 있을 수 있음
                            "AI-ModelScope/t5-v1_1-xxl",  # 커뮤니티가 있을 수 있음
                        ]
                        # 사용자 환경 변수를 통해 지정도 허용 ModelScope repo
                        ms_repo = os.environ.get("T5_MODELSCOPE_REPO", "")
                        if ms_repo:
                            ms_repo_candidates.insert(0, ms_repo)

                        ms_last_error = None
                        for ms_repo_id in ms_repo_candidates:
                            try:
                                ms_dir = snapshot_download(
                                    ms_repo_id,
                                    cache_dir=target_dir,
                                    allow_patterns=t5_files,
                                )
                                # 캐시 디렉토리에서 파일 복사하기 target_dir
                                for fname in t5_files:
                                    src_file = os.path.join(ms_dir, fname)
                                    dst_file = os.path.join(target_dir, fname)
                                    if os.path.isfile(src_file) and not os.path.isfile(dst_file):
                                        import shutil
                                        shutil.copy2(src_file, dst_file)

                                # 검증
                                final_existing = [f for f in t5_files if os.path.isfile(os.path.join(target_dir, f))]
                                if len(final_existing) == len(t5_files):
                                    return web.json_response({
                                        "success": True,
                                        "message": f"T5 tokenizer 다운로드 완료 (via ModelScope: {ms_repo_id})",
                                        "path": target_dir,
                                        "source": "modelscope",
                                    })
                            except Exception as e:
                                ms_last_error = e
                                continue

                        if source == "modelscope":
                            return web.json_response({"success": False, "error": f"ModelScope에서 다운로드 실패했습니다.: {ms_last_error}"})

                    except ImportError:
                        if source == "modelscope":
                            return web.json_response({"success": False, "error": "필요 설치 modelscope: pip install modelscope"})

                # ---- 방법 3: 직접 HTTP 다운로드 (최종적으로 적용).) ----
                try:
                    import urllib.request
                    import ssl

                    # hf-mirror에서 직접 HTTP 다운로드 시도
                    direct_urls = [
                        # hf-mirror 직링크
                        f"https://hf-mirror.com/google/t5-v1_1-xxl/resolve/main/{fname}"
                        for fname in t5_files
                        if not os.path.isfile(os.path.join(target_dir, fname))
                    ]

                    # SSL 인증 없이 컨텍스트 생성 (일부 환경에서 인증서 문제))
                    ssl_ctx = ssl.create_default_context()

                    all_ok = True
                    for url in direct_urls:
                        fname = url.split("/")[-1]
                        target_file = os.path.join(target_dir, fname)
                        if os.path.isfile(target_file):
                            continue
                        try:
                            req = urllib.request.Request(url, headers={"User-Agent": "ComfyUI-Training/1.0"})
                            with urllib.request.urlopen(req, context=ssl_ctx, timeout=120) as resp:
                                with open(target_file, "wb") as f:
                                    f.write(resp.read())
                        except Exception as e:
                            all_ok = False
                            break

                    if all_ok:
                        final_existing = [f for f in t5_files if os.path.isfile(os.path.join(target_dir, f))]
                        if len(final_existing) == len(t5_files):
                            return web.json_response({
                                "success": True,
                                "message": "T5 tokenizer hф-mirror 직접 다운로드)",
                                "path": target_dir,
                                "source": "direct",
                            })
                except Exception:
                    pass

                # 모두 실패
                hint = "모든 다운로드 소스가 실패했습니다. 권장：\n"
                hint += "1. 환경 변수 설정 HF_ENDPOINT=https://hf-mirror.com 재시작 ComfyUI\n"
                hint += "2. 자동 완성 설정을 엽니다: 자신의 프롬프트 완성 어휘 라이브러리를 관리합니다(새로 만들기/ 가져오기/ 내보내기/ 다중 선택 활성화), 기본 내장 완성 기능을 비활성화할 수 있습니다. spiece.model/tokenizer_config.json/special_tokens_map.json\n"
                hint += "3. 다운로드 파일을 저장합니다: " + target_dir
                return web.json_response({"success": False, "error": hint})
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        # ===== 지능형 반복 계산 =====
        async def calc_repeats(request):
            """목표 단계 수 계산num_repeats"""
            try:
                data = await request.json()
                target_steps = int(data.get("target_steps", 2000))
                image_count = int(data.get("image_count", 1))
                batch_size = int(data.get("batch_size", 1))
                max_epochs = int(data.get("max_epochs", 10))

                if image_count <= 0 or batch_size <= 0 or max_epochs <= 0:
                    return web.json_response({"success": False, "error": "파라미터는 크기보다 큼0"})

                num_repeats = math.ceil(target_steps * batch_size / (image_count * max_epochs))
                actual_steps = image_count * num_repeats * max_epochs // batch_size

                return web.json_response({
                    "success": True,
                    "num_repeats": num_repeats,
                    "actual_steps": actual_steps,
                    "target_steps": target_steps,
                    "deviation": abs(actual_steps - target_steps) / target_steps if target_steps > 0 else 0,
                })
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        # ===== 하드웨어 모니터링 API (독립 모드 필수, ComfyUI 모드 __init__.py에서 제공)） =====
        async def hardware_stats(request):
            """하드웨어 사용률 가져오기 (독립 모드）"""
            try:
                stats = {
                    "success": True,
                    "cpu_utilization": -1,
                    "ram_total": 0,
                    "ram_used": 0,
                    "ram_percent": 0,
                    "gpu_available": False,
                    "gpus": [],
                }
                if HAS_PSUTIL:
                    stats["cpu_utilization"] = psutil.cpu_percent(interval=0)
                    ram = psutil.virtual_memory()
                    stats["ram_total"] = ram.total
                    stats["ram_used"] = ram.used
                    stats["ram_percent"] = ram.percent

                # pynvml을 사용하여 정확한 GPU 사용률 및 메모리 정보를 우선 사용합니다.
                pynvml_ok = False
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    device_count = pynvml.nvmlDeviceGetCount()
                    for i in range(device_count):
                        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                        gpu_info = {"index": i}

                        # GPU 이름
                        try:
                            name = pynvml.nvmlDeviceGetName(handle)
                            gpu_info["name"] = name.decode('utf-8', errors='ignore') if isinstance(name, bytes) else name
                        except:
                            gpu_info["name"] = f"GPU {i}"

                        # GPU 사용률
                        try:
                            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                            gpu_info["gpu_utilization"] = util.gpu
                            gpu_info["memory_utilization"] = util.memory
                        except:
                            gpu_info["gpu_utilization"] = -1
                            gpu_info["memory_utilization"] = -1

                        # 메모리 정보 (실제 사용량, PyTorch 할당 제외)）
                        try:
                            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                            gpu_info["vram_total"] = mem.total
                            gpu_info["vram_used"] = mem.used
                            gpu_info["vram_percent"] = (mem.used / mem.total * 100) if mem.total > 0 else 0
                        except:
                            gpu_info["vram_total"] = 0
                            gpu_info["vram_used"] = 0
                            gpu_info["vram_percent"] = 0

                        # 온도
                        try:
                            gpu_info["temperature"] = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                        except:
                            gpu_info["temperature"] = -1

                        stats["gpus"].append(gpu_info)
                    stats["gpu_available"] = True
                    pynvml.nvmlShutdown()
                    pynvml_ok = True
                except ImportError:
                    pass
                except Exception:
                    pass

                # pynvml 사용할 수 없는 경우 torch.cuda로 되돌립니다(메모리 데이터 부정확, PyTorch 할당만 통계)）
                if not pynvml_ok:
                    try:
                        import torch
                        if torch.cuda.is_available():
                            for i in range(torch.cuda.device_count()):
                                props = torch.cuda.get_device_properties(i)
                                vram_used = torch.cuda.memory_allocated(i)
                                stats["gpus"].append({
                                    "index": i,
                                    "name": props.name,
                                    "gpu_utilization": -1,
                                    "memory_utilization": -1,
                                    "vram_total": props.total_memory,
                                    "vram_used": vram_used,
                                    "vram_percent": vram_used / props.total_memory * 100 if props.total_memory > 0 else 0,
                                    "temperature": -1,
                                })
                            stats["gpu_available"] = True
                    except ImportError:
                        pass
                return web.json_response(stats)
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)})

        # ===== 일관된 라우팅 등록 =====
        routes = [
            ('GET',  '/comfyui_panel/training/default_paths', default_paths),
            ('POST', '/comfyui_panel/training/validate_image_dir', validate_image_dir),
            ('GET',  '/comfyui_panel/training/check_deps', check_deps),
            ('POST', '/comfyui_panel/training/install_deps', install_deps),
            ('GET',  '/comfyui_panel/training/install_status', install_status),
            ('POST', '/comfyui_panel/training/clone_sd_scripts', clone_sd_scripts),
            ('POST', '/comfyui_panel/training/scan_images', scan_images),
            ('GET',  '/comfyui_panel/training/scan_models', scan_models),
            ('POST', '/comfyui_panel/training/start_tagging', start_tagging),
            ('GET',  '/comfyui_panel/training/tagging_status', tagging_status),
            ('POST', '/comfyui_panel/training/stop_tagging', stop_tagging),
            # 생성 정규화 이미지 (기반 소재) caption）
            ('POST', '/comfyui_panel/training/start_gen_reg', start_gen_reg),
            ('GET',  '/comfyui_panel/training/gen_reg_status', gen_reg_status),
            ('POST', '/comfyui_panel/training/stop_gen_reg', stop_gen_reg),
            ('POST', '/comfyui_panel/training/generate_config', generate_config),
            ('POST', '/comfyui_panel/training/save_config', save_config),
            ('GET',  '/comfyui_panel/training/list_configs', list_configs),
            ('GET',  '/comfyui_panel/training/load_config', load_config),
            ('POST', '/comfyui_panel/training/save_params', save_params),
            ('GET',  '/comfyui_panel/training/list_params', list_params),
            ('GET',  '/comfyui_panel/training/load_params', load_params),
            ('POST', '/comfyui_panel/training/delete_params', delete_params),
            ('POST', '/comfyui_panel/training/start', start_training),
            ('GET',  '/comfyui_panel/training/status', training_status),
            ('GET',  '/comfyui_panel/training/logs', training_logs),
            ('POST', '/comfyui_panel/training/pause', pause_training),
            ('POST', '/comfyui_panel/training/resume', resume_training),
            ('POST', '/comfyui_panel/training/stop', stop_training),
            ('GET',  '/comfyui_panel/training/samples', training_samples),
            ('POST', '/comfyui_panel/training/samples_diff', training_samples_diff),
            ('GET',  '/comfyui_panel/training/models', training_models),
            ('POST', '/comfyui_panel/training/copy_to_loras', copy_to_loras),
            ('GET',  '/comfyui_panel/training/download_model', download_model),
            ('GET',  '/comfyui_panel/training/tensorboard_data', tensorboard_data),
            ('POST', '/comfyui_panel/training/scan_subdirs', scan_subdirs),
            ('POST', '/comfyui_panel/training/scan_subdir_images', scan_subdir_images),
            ('POST', '/comfyui_panel/training/delete_image', delete_image),
            ('POST', '/comfyui_panel/training/read_caption', read_caption),
            ('POST', '/comfyui_panel/training/save_caption', save_caption),
            ('POST', '/comfyui_panel/training/add_subdir', add_subdir),
            ('POST', '/comfyui_panel/training/create_image_dir', create_image_dir),
            ('POST', '/comfyui_panel/training/remove_subdir', remove_subdir),
            ('POST', '/comfyui_panel/training/rename_subdir', rename_subdir),
            ('POST', '/comfyui_panel/training/restore_subdir', restore_subdir),
            ('POST', '/comfyui_panel/training/move_images', move_images),
            ('POST', '/comfyui_panel/training/upload_files', upload_files),
            ('GET',  '/comfyui_panel/training/image', serve_image),
            ('POST', '/comfyui_panel/training/browse_dir', browse_dir),
            ('GET',  '/comfyui_panel/training/list_output_files', list_output_files),
            ('GET',  '/comfyui_panel/training/sample_image', sample_image),
            ('GET',  '/comfyui_panel/training/t5_tokenizer_status', t5_tokenizer_status),
            ('POST', '/comfyui_panel/training/download_t5_tokenizer', download_t5_tokenizer),
            ('POST', '/comfyui_panel/training/calc_repeats', calc_repeats),
            ('GET',  '/comfyui_panel/hardware_stats', hardware_stats),
            # 이미지 전처리
            ('POST', '/comfyui_panel/training/start_preprocess', start_preprocess),
            ('GET',  '/comfyui_panel/training/preprocess_status', preprocess_status),
            ('POST', '/comfyui_panel/training/stop_preprocess', stop_preprocess),
        ]

        for method, path, handler in routes:
            app.router.add_route(method, path, handler)

        print("[ComfyUI Training] 훈련 API 등록됨")

    except Exception as e:
        print(f"[ComfyUI Training] API 등록 실패: {e}")


def _parse_training_line(line):
    """훈련 출력 행 파싱, step/loss 등 지표 추출"""
    # sd-scripts 일반 출력 형식:
    # "N epoch, M/total steps, loss: X.XXXX, ..."
    # "step N: loss X.XXXX"
    # "N/total steps, loss X.XXXX"
    # "[N/X] loss: X.XXXX"

    # 형식1: "N epoch, M/total steps" (sd-scripts표준 형식)
    epoch_step_match = re.search(r'(\d+)\s*epoch\s*,\s*(\d+)/(\d+)\s+steps', line, re.IGNORECASE)
    if epoch_step_match:
        _training_state["current_epoch"] = int(epoch_step_match.group(1))
        _training_state["current_step"] = int(epoch_step_match.group(2))
        _training_state["total_steps"] = int(epoch_step_match.group(3))

    # 형식2: "N/total steps"
    elif not epoch_step_match:
        total_match = re.search(r'(\d+)/(\d+)\s+steps', line, re.IGNORECASE)
        if total_match:
            _training_state["current_step"] = int(total_match.group(1))
            _training_state["total_steps"] = int(total_match.group(2))

    # 2b 형식: tqdm 진행률 표시줄 "steps: 50%|██| 550/1100 [...]"
    if not epoch_step_match:
        tqdm_match = re.search(r'\|\s*(\d+)/(\d+)\s*\[', line)
        if tqdm_match:
            _training_state["current_step"] = int(tqdm_match.group(1))
            if int(tqdm_match.group(2)) > 0:
                _training_state["total_steps"] = int(tqdm_match.group(2))

    # 형식3: "step N"
    step_match = re.search(r'step\s+(\d+)', line, re.IGNORECASE)
    if step_match:
        _training_state["current_step"] = int(step_match.group(1))

    # loss (다양한 형식)
    loss_match = re.search(r'loss[:\s]+([0-9]*\.?[0-9]+)', line, re.IGNORECASE)
    if loss_match:
        loss_val = float(loss_match.group(1))
        _training_state["current_loss"] = loss_val
        # 손실 기록
        if "loss_history" not in _training_state:
            _training_state["loss_history"] = []
        _training_state["loss_history"].append({
            "step": _training_state["current_step"],
            "loss": loss_val,
            "epoch": _training_state["current_epoch"],
        })
        # 보존 최근 2000개 포인트
        if len(_training_state["loss_history"]) > 2000:
            _training_state["loss_history"] = _training_state["loss_history"][-2000:]

    # epoch (독립 실행)
    epoch_match = re.search(r'epoch\s+(\d+)', line, re.IGNORECASE)
    if epoch_match and not epoch_step_match:
        _training_state["current_epoch"] = int(epoch_match.group(1))

    # total_steps가 0이면 epoch 및 steps를 기반으로 추정합니다
    # (max_train_epochs 설정 시 max_train_steps 설정 필요)
    if _training_state["total_steps"] == 0 and _training_state["current_epoch"] > 1 and _training_state["current_step"] > 0:
        steps_per_epoch = _training_state["current_step"] / _training_state["current_epoch"]
        max_epochs = _training_state.get("max_train_epochs", 0)
        if max_epochs > 0:
            _training_state["total_steps"] = int(steps_per_epoch * max_epochs)

    # 모델 저장 확인
    if "saving model" in line.lower() or "saving checkpoint" in line.lower():
        model_match = re.search(r'[\w/\\.-]+\.(safetensors|pt|ckpt)', line)
        if model_match:
            model_name = model_match.group(0)
            # 상대 경로인 경우 연결output_dir
            if not os.path.isabs(model_name):
                output_dir = _training_state.get("output_dir", "")
                if output_dir:
                    model_name = os.path.join(output_dir, os.path.basename(model_name))
            _training_state["saved_models"].append(model_name)
            _save_state()

    # 샘플 이미지 생성 확인
    if "sample" in line.lower() and IMAGE_EXTENSIONS:
        sample_match = re.search(r'[\w/\\.-]+\.(png|jpg|jpeg|webp)', line, re.IGNORECASE)
        if sample_match:
            _training_state["sample_images"].append(sample_match.group(0))


# ============================================================
#  대상 비디오 편집의 원 비디오
# ============================================================

def _collect_npu_stats():
    """昇腾 NPU 사용률 및 HBM 메모리 정보 수집。

    반환 형식은 pynvml과 일치합니다(BSK_HardwareStats.getStatus를 직접 panel에 넣기 위해). result['gpus']）:
        [
            {
                'index': 0,
                'name': 'Ascend XXXX',
                'gpu_utilization': 75,       # NPU 사용률 계산 (%)
                'memory_utilization': 0,     # （NPU 이 기능은 고정입니다 0）
                'vram_total': 64424509440,   # HBM 총량 (bytes)
                'vram_used': 32212254720,    # HBM 이미 사용 (bytes)
                'vram_percent': 50.0,        # HBM 사용률 (%)
                'temperature': 65,           # 온도 (°C)
            },
            ...
        ]

    우선순위:
     1. torch_npu — torch_npu를 통해 HBM, NPU 사용률 및 온도를 가져옴 (acl 직접 호출 방지, 예방) CANN C++ 층 abort）
     2. npu-smi info（명령줄）— CANN 툴체인 내장
     3. 모두 사용할 수 없는 경우 빈 목록 반환
    """
    result = []

    # ── 방법 1: torch_npu (HBM/효율/온도 모두 torch_npu로 사용하고, 직접 호출하지 않음) acl）──
    try:
        import torch
        import torch_npu  # noqa: F401

        npu_count = torch.npu.device_count()
        if npu_count > 0:
            # ── 주의: acl.init()를 직접 호출하면 이 환경에서 트리거됩니다 CANN C++ 층
            #    std::logic_error (basic_string null) 그리고 전체 프로세스를 종료합니다.。
            #    torch_npu 인터페이스를 사용하여 활용률/온도를 가져오고 안전하며 불필요합니다. abort。
            try:
                for i in range(npu_count):
                    try:
                        # ── HBM 메모리 (통과 torch.npu.mem_get_info）──
                        free_bytes, total_bytes = torch.npu.mem_get_info(i)
                        mem_used = total_bytes - free_bytes
                        vram_percent = (mem_used / total_bytes * 100) if total_bytes > 0 else 0

                        # ── NPU 사용률 (통과) torch_npu.npu.utilization）──
                        gpu_util = -1
                        try:
                            gpu_util = int(round(torch_npu.npu.utilization(i)))
                        except Exception:
                            pass  # 자르기 상자 조정 완료, 잘라내기 필요 표시 -1

                        # ── 이름 ──
                        try:
                            name = torch_npu.npu.get_device_name(i)
                        except Exception:
                            name = f"Ascend NPU {i}"

                        # ── 온도 ──
                        temp = -1
                        try:
                            temp = int(round(torch.npu.get_device_temperature(i)))
                        except Exception:
                            pass

                        result.append({
                            'index': i,
                            'name': name,
                            'gpu_utilization': gpu_util,
                            'memory_utilization': 0,
                            'vram_total': int(total_bytes),
                            'vram_used': int(mem_used),
                            'vram_percent': round(vram_percent, 1),
                            'temperature': temp,
                        })
                    except Exception as e:
                        print(f"[BSK] NPU device {i} query failed: {e}")
            except Exception as e:
                print(f"[BSK] NPU collect error: {e}")
    except ImportError:
        pass  # torch_npu 설치되지 않았습니다, 방법을 확인 2
    except Exception as e:
        print(f"[BSK] torch_npu collect_npu_stats error: {e}")

    # ── 방법 2: npu-smi 정보(명령줄 복귀）────────────────────────
    if not result:
        try:
            proc = subprocess.run(
                ['npu-smi', 'info', '-m'],
                capture_output=True, text=True, timeout=10
            )
            if proc.returncode != 0:
                # 인수 시 args (사용 가능한 경우
                proc = subprocess.run(
                    ['npu-smi', 'info', '-t', 'total'],
                    capture_output=True, text=True, timeout=10
                )
            if proc.returncode == 0 and proc.stdout.strip():
                # npu-smi 테이블 출력 파싱
                lines = proc.stdout.strip().split('\n')
                for line in lines[1:]:  # 헤더 건너뛰기
                    parts = [p.strip() for p in line.split('|')]
                    if len(parts) < 5:
                        continue
                    try:
                        idx = int(parts[0])
                    except ValueError:
                        continue
                    name = parts[1] if len(parts) > 1 else f"NPU {idx}"
                    try:
                        hbm_total = int(float(parts[2]) * 1024 * 1024) if parts[2] not in ('-', 'N/A') else 0
                    except (ValueError, IndexError):
                        hbm_total = 0
                    try:
                        hbm_used = int(float(parts[3]) * 1024 * 1024) if parts[3] not in ('-', 'N/A') else 0
                    except (ValueError, IndexError):
                        hbm_used = 0
                    try:
                        util = int(parts[4]) if parts[4] not in ('-', 'N/A') else 0
                    except (ValueError, IndexError):
                        util = 0
                    result.append({
                        'index': idx,
                        'name': name,
                        'gpu_utilization': util,
                        'memory_utilization': 0,
                        'vram_total': hbm_total,
                        'vram_used': hbm_used,
                        'vram_percent': round(hbm_used / hbm_total * 100, 1) if hbm_total > 0 else 0,
                        'temperature': -1,
                    })
        except FileNotFoundError:
            pass  # npu-smi PATH에 없음
        except Exception as e:
            print(f"[BSK] npu-smi fallback error: {e}")

    return result
