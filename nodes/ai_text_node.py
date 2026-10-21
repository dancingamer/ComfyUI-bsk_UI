# SPDX-FileCopyrightText: 2026 灰暗x
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
BSK_AIText - OpenAI 호환 인터페이스 호출하여 텍스트 생성

기능：
  - 두 개의 텍스트 입력: rule(규칙 프롬프트, system), text(입력 텍스트, prompt) user）
  - execute 현재 전역 설정 user/default/ai_text_config.json(API Key / Base URL / Model 등)，
    워크플로우로 이동하지 않는 JSON을 사용하고, 표준 라이브러리 urllib를 사용하여 OpenAI 호환성을 호출합니다 /v1/chat/completions
  - AI 반환 내용은 STRING으로 출력합니다. 실패 시 (키 없음 / 네트워크 오류 / HTTP 오류)는 접두사 붙은 오류 텍스트를 반환하며, 프로그램이 종료되지 않습니다.
  - 기본 deepseek

주의: 표준 라이브러리만 사용하며 새로운 의존성을 추가하지 않습니다.。
"""
 
import os
import re
import json
import urllib.request
import urllib.error
 
try:
    import folder_paths
    _BASE_DIR = folder_paths.base_path
except Exception:
    _BASE_DIR = os.path.abspath('./')

CONFIG_DIR = os.path.join(_BASE_DIR, "user", "default")
CONFIG_PATH = os.path.join(CONFIG_DIR, "ai_text_config.json")

DEFAULT_CONFIG = {
    "provider": "deepseek",
    "api_key": "",
    "base_url": "https://api.deepseek.com",
    "model": "deepseek-chat",
    "temperature": 0.7,
    "max_tokens": 1024,
    "vision": "auto",
}

# 서비스 제공업체 프리셋 (프리셋 선택 시 자동으로 채워짐 base_url + model）
PROVIDER_PRESETS = {
    "deepseek": {"base_url": "https://api.deepseek.com", "model": "deepseek-chat"},
    "openai":   {"base_url": "https://api.openai.com/v1", "model": "gpt-4o-mini"},
    "moonshot": {"base_url": "https://api.moonshot.cn/v1", "model": "moonshot-v1-8k"},
}


# vision 모델 이름 문자열 힌트 (소문자 매칭), 이미지 지원 여부 판단 '자동' 모드에 사용
VISION_MODEL_HINTS = [
    "vision", "gpt-4o", "gpt-4-turbo", "gpt-4.1", "gpt-4.5",
    "qwen-vl", "qwen2-vl", "qwen2.5-vl", "qwen3-vl",
    "glm-4v", "glm-4.1v",
    "claude-3", "claude-opus", "claude-sonnet", "claude-haiku",
    "gemini", "llava", "moondream", "pixtral", "step-1v", "step-1.5v",
    "kimi-vl", "deepseek-vl", "janus", "internvl", "minicpm-v",
    "phi-3-vision", "phi-4-multimodal", "yi-vl", "doubao-vision", "bailing-vl",
    "agnes",
]

# 일반 VL 명명 기반 처리: xxx-vl / xxx_vl / xxx-vl:free / vl-xxx / vision 등 (프롬프트 테이블 미포함 모델)）
_VL_PATTERN = re.compile(r"(^|[^a-z])(vl|vlm|vision)([^a-z]|$)", re.IGNORECASE)


def _looks_like_vision_model(model_name):
    """모델 이름이 이미지 입력을 지원하는 모델과 같은지: 먼저 프롬프트 테이블을 일치시키고, 일반 VL 이름을 사용하여 기본적으로 매핑합니다.。"""
    name = str(model_name or "").lower()
    if any(h in name for h in VISION_MODEL_HINTS):
        return True
    return bool(_VL_PATTERN.search(name))


def chat_endpoint(base_url):
    """base_url을 chat/completions 엔드포인트로 일반화。

    사용자가 완전 엔드포인트를 직접 붙여넣을 수 있음 (/chat/completions 또는 /completions），
    그렇지 않으면 합쳐서 .../chat/completions/chat/completions。
    """
    u = str(base_url or "").strip().rstrip("/")
    low = u.lower()
    for suffix in ("/chat/completions", "/completions"):
        if low.endswith(suffix):
            u = u[: -len(suffix)].rstrip("/")
            break
    return u + "/chat/completions"


def _tensor_to_base64(image):
    """ComfyUI IMAGE tensor [B,H,W,C]（0-1, RGB）→ PNG base64 문자열。"""
    import io
    import base64
    import numpy as np
    from PIL import Image
    arr = (255.0 * image[0].detach().cpu().numpy()).clip(0, 255).astype("uint8")
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def load_config():
    """전역 설정을 읽고 기본값과 병합(비어 있지 않은 필드만 유지하여 빈 문자열로 기본값을 덮어쓰지 않음)）。"""
    cfg = dict(DEFAULT_CONFIG)
    try:
        if os.path.isfile(CONFIG_PATH):
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                for k in DEFAULT_CONFIG:
                    v = data.get(k)
                    if v is not None and v != "":
                        cfg[k] = v
    except Exception:
        pass
    return cfg


class BSK_AIText:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "rule": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "규칙 기반 프롬프트 (시스템 메시지로 사용)）"
                }),
                "text": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "입력 텍스트(사용자 메시지로, 규칙과 병합하여 전송) AI）"
                }),
                "enabled": ("BOOLEAN", {
                    "default": True,
                    "label_on": "활성화",
                    "label_off": "비활성화"
                }),
            },
            "optional": {
                "image": ("IMAGE",),
                "send_image": ("BOOLEAN", {"default": False, "label_on": "이미지 인식", "label_off": "Plain text"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("AI회신",)
    FUNCTION = "execute"
    CATEGORY = "bsk_UI/AI텍스트"
    DESCRIPTION = "각 AI 제공업체의 OpenAI 호환 인터페이스 호출, 규칙에 따라 프롬프트 제공 + 텍스트 입력을 통해 텍스트를 생성합니다. API 설정은 ComfyUI 사용자 디렉토리에 저장되며 워크플로우에 진입하지 않습니다.）。"

    def execute(self, rule, text, enabled, image=None, send_image=False):
        # 비활성화 시 직접 전달된 입력 텍스트(system 프롬프트 제외)를 호출하지 않습니다. AI
        if not enabled:
            return (text if text is not None else "",)

        cfg = load_config()

        api_key = (cfg.get("api_key") or "").strip()
        if not api_key:
            return ("[AI텍스트 노드] 오류: API 키가 구성되지 않았습니다. 노드 카드의 '설정' 패널에서 채우고 저장하십시오。",)

        base_url = (cfg.get("base_url") or "").strip()
        if not base_url:
            return ("[AI텍스트 노드] 오류: 구성되지 않음 Base URL。",)

        model = (cfg.get("model") or "").strip()
        if not model:
            return ("[AI텍스트 노드] 오류: 모델 이름 구성되지 않음。",)

        # ---- 이미지와 동일한 이름의 txt 파일이 함께 삭제됩니다 ----
        has_img = image is not None and getattr(image, 'shape', (0,))[0] > 0
        img_hash = ""
        if has_img:
            try:
                import hashlib
                _arr = image[0].detach().cpu().numpy()
                img_hash = hashlib.md5(_arr.tobytes()).hexdigest()
            except Exception:
                img_hash = ""

        vision_mode = str(cfg.get("vision", "auto")).lower()
        auto_vision = _looks_like_vision_model(model)
        if vision_mode == "on":
            vision_enabled = True
        elif vision_mode == "off":
            vision_enabled = False
        else:
            vision_enabled = auto_vision

        want_image = bool(send_image)
        if has_img and want_image and not vision_enabled:
            return ("[AI전역 설정 user/default/ai_text_config.json(API Key / Base URL / Model 등)을 읽습니다 gpt-4o）。",)

        messages = []
        if rule and str(rule).strip():
            messages.append({"role": "system", "content": str(rule)})
        if has_img and want_image and vision_enabled:
            user_parts = []
            if text and str(text).strip():
                user_parts.append({"type": "text", "text": str(text)})
            try:
                b64 = _tensor_to_base64(image)
            except Exception as e:
                return ("[AI텍스트 노드] 이미지 인코딩 실패: %s" % str(e),)
            user_parts.append({
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64," + b64}
            })
            messages.append({"role": "user", "content": user_parts})
        elif text and str(text).strip():
            messages.append({"role": "user", "content": str(text)})
        else:
            # 사용자 메시지를 보낼 수 없습니다: API에 시스템만 있는 불법 요청을 보내지 마십시오(업스트림에서 오류가 발생합니다). No user query）
            if has_img and not want_image:
                return ("[AI텍스트 노드] '인식'이 닫히고 텍스트가 입력되지 않은 경우: '인식'을 닫으면 이미지가 더 이상 전송되지 않습니다. '입력 텍스트'를 입력하거나 '인식' 버튼을 다시 켜서 이미지를 전송하십시오.。",)
            return ("[AI텍스트 노드] 오류: 규칙 및 입력 텍스트 모두 비어 있음, 내용 전송 불가。",)

        try:
            temperature = float(cfg.get("temperature", 0.7))
        except Exception:
            temperature = 0.7
        try:
            max_tokens = int(cfg.get("max_tokens", 1024))
        except Exception:
            max_tokens = 1024

        # 캐시: 규칙 + 텍스트 + 출력에 영향을 미치는 설정이 동일하면 이전 AI 결과값을 직접 재사용하여 중복 호출 방지
        cache_key = (
            str(rule) if rule is not None else "",
            str(text) if text is not None else "",
            model, base_url, temperature, max_tokens, img_hash, want_image,
        )
        if getattr(self, "_cache_key", None) == cache_key and getattr(self, "_cache_val", None) is not None:
            return (self._cache_val,)

        url = chat_endpoint(base_url)
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        def _post(_max_tokens):
            _req = urllib.request.Request(
                url,
                data=json.dumps(dict(payload, max_tokens=_max_tokens)).encode("utf-8"),
                method="POST")
            _req.add_header("Content-Type", "application/json")
            _req.add_header("Authorization", "Bearer " + api_key)
            with urllib.request.urlopen(_req, timeout=120) as _resp:
                return json.loads(_resp.read().decode("utf-8", "ignore"))

        def _pick(_data):
            """취 (본문, finish_reason, 추론 텍스트); parts 리스트 형태의 공급자 content 호환。"""
            _choices = _data.get("choices") or []
            if not _choices:
                return "", "", ""
            _ch = _choices[0]
            _msg = _ch.get("message") or {}
            _content = _msg.get("content")
            if isinstance(_content, list):
                _content = "".join(p.get("text", "") for p in _content if isinstance(p, dict))
            if _content is None:
                _content = ""
            return (str(_content).strip(),
                    str(_ch.get("finish_reason") or ""),
                    str(_msg.get("reasoning") or ""))

        try:
            content, finish_reason, reasoning = _pick(_post(max_tokens))
            # 추론 모델의 reasoning과 본문이 공통으로 사용하는 max_tokens: 추론이 예산을 모두 소모하면 본문이 비어 있습니다.（finish_reason=length），
            # 예산을 높여 한 번 다시 시도하여 무음 반환을 방지합니다
            if not content and finish_reason == "length":
                retry_tokens = min(max(max_tokens * 4, 4096), 32768)
                if retry_tokens > max_tokens:
                    content, finish_reason, reasoning = _pick(_post(retry_tokens))
            if not content:
                return ("[AI텍스트 노드] 모델이 본문을 반환하지 않음（finish_reason=%s，추론 내용 %d 자，max_tokens=%d）。"
                        "이 모델은 먼저 추론을 출력하고, 그 다음 텍스트를 출력해야 합니다. 설정에서 Max Tokens를 높이거나 다른 추론 모델을 사용하십시오.。"
                        % (finish_reason or "unknown", len(reasoning), max_tokens),)
            # 본문이 있는 성공 결과만 캐시 (빈 본문은 캐시되지 않음, 계속 재현)）
            self._cache_key = cache_key
            self._cache_val = content
            return (self._cache_val,)
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode("utf-8", "ignore")[:400]
            except Exception:
                pass
            return ("[AI텍스트 노드] HTTP 오류 %s: %s" % (e.code, detail),)
        except urllib.error.URLError as e:
            return ("[AI텍스트 노드] 호출 실패: %s" % str(getattr(e, "reason", e)),)
        except Exception as e:
            return ("[AI새로 추가: 키프레임 위치 정보: %s" % str(e),)


NODE_CLASS_MAPPINGS = {
    "BSK_AIText": BSK_AIText,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BSK_AIText": "BSK AI텍스트 생성",
}
