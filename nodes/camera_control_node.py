# SPDX-FileCopyrightText: 2026 灰暗x
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
CameraControlNode - 시각화된 카메라 프롬프트 제어 노드 (재구성 버전), 이 노드를 참조할 경우 출처를 명시하십시오

설계 (참고 카메라 제어 노드_솔루션).md）：
1. 방향 (Azimuth)은 pos_x에 의해 결정 az = pos_x * π 분해 4개 방향 성분으로, 따라“합”정규화 후 분할
   방위 예산 = azimuth.weight × 극향 게이트 (게이트=1 시간 방향 가중치, 단 한 |pos_y|→1 완전히 위/아래로 이동할 때
   문 제어→0，앞, 뒤, 좌, 우로 출력하지 않습니다. 방향 가중치는 앵글과 관련이 있으며, 위, 아래 앵글의 크기 조정에 영향을 받지 않습니다.）；
   가중치 < azimuth.deadzone_ratio(기본 0.2) 방향 출력 안 함。
2. 기본값 입력 행 스타일 - 워크플로우의 값을 표시하는 데 사용 = |pos_y| × (1 + extra_master × elevation.extra)，방위가 멀어짐
   （pos_y=0 시간 높이=0，±1 시=1，瓜분 100%）；거리（Distance，pos_z）위하여 1D 상호 배제기어、독립우부앙，
   기본 가중치 고정 1.0, 최종 가중치 = clamp(1 + extra_master × distance.extra, 0.1, 10)。
3. 기울림(Tilt, roll)이 tilt.deadzone 초과 시 dutch_tag 출력, 가중치는 동일。
4. 추가 프롬프트(렌즈/심도/촬영/구도/스타일)를 활성화 스위치 제어에 사용 가능하며, 사용자 정의 텍스트를 입력할 수 있습니다.。
5. 네거티브 프롬프트는 출력하지 않습니다 (삭제됨).
6. 이 노드는 프롬프트 텍스트의 유일한 권위 있는 어셈블러입니다; 프론트엔드 computePrompt는 드래그 앤 드롭 실시간 미리보기에만 사용되며, 알고리즘은 이 파일과 일치해야 합니다。

참고: 롤은 프런트 엔드 캔버스에서 노드 입력으로 사용（Shift+휠 / , 스크롤 키) tilt。
"""

import json
import math


# ============ 기본 설정 (프론트엔드 카드 DEFAULT_CONFIG와 일치해야 함)） ============
DEFAULT_CONFIG = {
    "weight_min": 0.1,
    "weight_max": 10.0,
    "no_weight": False,
    "no_weight_threshold": 0.5,
    "azimuth": {
        "enabled": True,
        "weight": 10.0,
        "deadzone_ratio": 0.2,
        "directions": {
            "front": {"tag": "from front", "enabled": True},
            "back":  {"tag": "from behind", "enabled": True},
            "left":  {"tag": "from right", "enabled": True},
            "right": {"tag": "from left", "enabled": True},
        },
    },
    "elevation": {
        "enabled": True,
        "extra": 10.0,
        "eye_peak": 3.0,
        "categories": {
                "bird": {"tag": "directly above, from above, aerial view,", "enabled": True},
                "high": {"tag": "high angle, from above", "enabled": True},
                "eye":  {"tag": "eye-level", "enabled": True},
                "low":  {"tag": "low angle, from below,", "enabled": True},
                "worm": {"tag": "directly below", "enabled": True},
        },
    },
    "distance": {
        "enabled": True,
        "extra": 0.0,
        "categories": {
            "ecu":    {"tag": "extreme close-up", "enabled": True},
            "cu":     {"tag": "close-up", "enabled": True},
            "medium": {"tag": "medium shot", "enabled": True},
            "full":   {"tag": "full body", "enabled": True},
            "wide":   {"tag": "wide shot", "enabled": True},
        },
    },
    "tilt": {
        "enabled": True,
        "deadzone": 0.15,
        "extra": 0.0,
        "dutch_tag": "dutch angle",
    },
    "extra_master": 1.0,
    "wheel_step": 0.0003,   # 유산 필드: 휠로 거리를 조정하는 기능은 폐지됨(오른쪽 클릭하여 드래그), 호환성을 위해 유지
    "drag_step": 0.004,     # 마우스 오른쪽 클릭 드래그 거리: 1px당 z 값, 제한 [0.0005, 0.02]
    "extras": {
        "lens":        {"enabled": False, "value": "85mm lens"},
        "dof":         {"enabled": False, "value": "shallow depth of field", "weight": 1.3},
        "movement":    {"enabled": False, "value": "handheld camera"},
        "composition": {"enabled": False, "value": "rule of thirds"},
        "style":       {"enabled": False, "value": "cinematic"},
    },
}

DEFAULT_CONFIG_JSON = json.dumps(DEFAULT_CONFIG, ensure_ascii=False)

# 구형 스키마 태그 키(역사 버전 유산, 로드 시 마이그레이션 및 삭제 필요).）
LEGACY_KEYS = {"two_tier", "axes", "negative"}

# 파일위의 z 영역: 파일 내 가중치는 z 영역 시작점(0% 추가 가중치)에서 끝점(100% 추가 가중치)까지 z 값에 따라 선형적으로 상승합니다.)。
# 파일이 다를 경우 다음 파일은 0%에서 재시작 (즉"다음 프롬프트로 건너가"）。
DIST_RANGES = {
    "ecu":    (0.7, 1.0),
    "cu":     (0.2, 0.7),
    "medium": (-0.2, 0.2),
    "full":   (-0.7, -0.2),
    "wide":   (-1.0, -0.7),
}

# 전면/전신/원근: 거리가 멀수록(z가 작을수록) 가중치는 커서 파일 내 frac를 반대로 계산합니다.；
# 클로즈업/근경은 z 값이 클수록 가중치가 큽니다。
DIST_FAR_STRONGER = {"medium", "full", "wide"}


class CameraControlNode:
    """카메라 프롬프트 제어 노드 시각화"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pos_x": ("FLOAT", {
                    "default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01,
                    "label": "좌우 (X)",
                }),
                "pos_y": ("FLOAT", {
                    "default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01,
                    "label": "상/하 (Y)",
                }),
                "pos_z": ("FLOAT", {
                    "default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01,
                    "label": "앞뒤 (Z)",
                }),
                "roll": ("FLOAT", {
                    "default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01,
                    "label": "롤 (Roll)",
                }),
                "config": ("STRING", {
                    "multiline": True,
                    "default": DEFAULT_CONFIG_JSON,
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("카메라 프롬프트",)
    FUNCTION = "execute"
    CATEGORY = "bsk_UI/카메라"
    DESCRIPTION = "시각화 제어 카메라 위치, 해당 카메라 프롬프트 텍스트 (방향 2D 비율 분배) + 높이/거리/기울기 + 렌즈/심도/줌/구도/스타일）"

    # ---------- 도구 ----------
    @staticmethod
    def _fmt_weight(w):
        # 두 자리 소수 정밀도, frontend fmtWeight와 일치 (항상 두 자리 소수 자릿수)）
        return f"{round(float(w), 2):.2f}"

    @staticmethod
    def _split_tags(tag):
        """쉼표로 구분된 여러 태그를 목록으로 분할 (공백 제거)）。"""
        return [t.strip() for t in str(tag).split(",") if t.strip()]

    @classmethod
    def _emit_weighted(cls, tag, w):
        """tag(들)을 분할하여 여러 (tag:weight) 조각으로 만듭니다. 동일 속성을 가진 tag는 동일한 가중치를 공유합니다.。"""
        return [f"({t}:{cls._fmt_weight(w)})" for t in cls._split_tags(tag)]

    @classmethod
    def _emit_plain(cls, tag):
        """무 가중치 모드: 태그만 ( :가중치 없이), 여러 개의 쉼표로 구분된 태그는 각 문장이 됨。"""
        return cls._split_tags(tag)

    @staticmethod
    def _merge_defaults(cfg, base):
        """base의 기본값을 재귀적으로 cfg에 누락된 필드를 채웁니다 (오래된 설정 호환, 오래된 axes/negative 필드는 무시)）。"""
        for k, v in base.items():
            if k not in cfg:
                cfg[k] = v
            elif isinstance(v, dict) and isinstance(cfg[k], dict):
                CameraControlNode._merge_defaults(cfg[k], v)
        return cfg

    @classmethod
    def _has_legacy_keys(cls, obj):
        """재귀적으로 이전 스키마 마커 키 포함 여부 확인（two_tier/axes/negative）。"""
        if not isinstance(obj, dict):
            return False
        for k, v in obj.items():
            if k in LEGACY_KEYS:
                return True
            if isinstance(v, dict) and cls._has_legacy_keys(v):
                return True
        return False

    @classmethod
    def _strip_legacy_keys(cls, obj):
        """재귀적으로 이전 스키마 태그 키를 삭제합니다.。"""
        if not isinstance(obj, dict):
            return
        for k in list(obj.keys()):
            if k in LEGACY_KEYS:
                del obj[k]
            elif isinstance(obj[k], dict):
                cls._strip_legacy_keys(obj[k])

    @classmethod
    def _load_config(cls, raw):
        """config JSON 파싱 실패 시 기본 설정으로 되돌리고, 기본 값으로 누락된 필드를 채웁니다.。
        구형 스키마 설정 (two_tier/axes/negative 포함)을 마이그레이션합니다: 하한/상한 제한을 새 기본값으로 되돌리고 기존 필드를 삭제합니다.，
        weight_min과 같은 이전 값을 사용하여 점수 가중치를 Clamp(기울)하지 마십시오. 1。"""
        if not raw:
            return json.loads(DEFAULT_CONFIG_JSON)
        try:
            cfg = json.loads(raw)
        except Exception:
            cfg = json.loads(DEFAULT_CONFIG_JSON)
        if cls._has_legacy_keys(cfg):
            cfg["weight_min"] = DEFAULT_CONFIG["weight_min"]
            cfg["weight_max"] = DEFAULT_CONFIG["weight_max"]
            cls._strip_legacy_keys(cfg)
        cls._merge_defaults(cfg, json.loads(DEFAULT_CONFIG_JSON))
        return cfg

    # 수직 축은 1차원 배타적 축입니다: 정면/상향/하향은 상반되며 동시에 나타날 수 없습니다(모델은 충돌 신호를 받음)），
    # 모든 시간에서 한 번만 출력。y>0=항공 뷰 (카메라 위에서 내려다봄), 프론트엔드 극성 일치。
    # eye 유일한가요?“중심 파일”，축 0 이상：y=0 즉 완전히 평视; 0 축 이하(y<0)항상 위를 바라봐，
    # 모든 작업을 low/worm에 맡기고 eye-level 설명을 더 이상 사용하지 않습니다.——그렇지 않으면“아래에서 위로 보기”말할 것이다“평행 시선”。
    ELEV_EYE_MAX = 0.2  # eye 파일 상위, 하위 파일과 일치

    @staticmethod
    def _elevation_key(y):
        if y > 0.7:
            return "bird"
        if y > CameraControlNode.ELEV_EYE_MAX:
            return "high"
        if y >= 0:
            return "eye"
        if y >= -0.7:
            return "low"
        return "worm"

    @classmethod
    def _elevation_weight(cls, cfg, y, ek):
        """높이 가중치: bird/high/low/worm“극단형”파일, eye 는“중심형”파일, 두 가지 단일 방향이 반대，
        공유 불가 |y| 공식——그것은 얻을 것이다“가장 눈높이 시 eye-level 가중치는 0, 완전히 출력하지 않음”의황미결과。
        극단형: 수평에서 벗어날수록 해당 특징이 더 뚜렷하고, 가중치는 |y| 단조 증가; 그리고 가중치는 단독으로 의존 |y|，
                변경 교차 파일 자동 연속（high→bird 위에서 공유, 전환 점이 튀지 않도록）。
        중심형 eye: y 값이 0에 가까울수록 정시, 가중치가 커지고, eye_peak(y=0 완전 평면) 선형 감쇠까지
                기본 가중치 1.0(y=ELEV_EYE_MAX，전경 보기로 전환됩니다. 방향과 가중치 방향이 일치합니다。"""
        elev_cfg = cfg.get("elevation") or {}
        if ek == "eye":
            peak = float(elev_cfg.get("eye_peak", 3.0))
            t = max(0.0, min(1.0, float(y) / cls.ELEV_EYE_MAX))
            return peak + (1.0 - peak) * t
        extra_master = float(cfg.get("extra_master", 1.0))
        elev_extra = float(elev_cfg.get("extra", 0.0))
        k = 1.0 + extra_master * elev_extra
        if k <= 0:
            return 0.0  # 스타일 보충(영어, 선택 사항): 광원, 색조, 화질 등
        return abs(float(y)) * k

    @staticmethod
    def _distance_key(z):
        if z > 0.7:
            return "ecu"
        if z > 0.2:
            return "cu"
        if z >= -0.2:
            return "medium"
        if z >= -0.7:
            return "full"
        return "wide"

    @classmethod
    def _distance_parts(cls, cfg, z):
        """거리 가중치: 내부 폴더 가중치 = clamp(1 + frac × extra_master × distance.extra, weight_min, weight_max)。"""
        key = cls._distance_key(float(z))
        cat = (cfg["distance"].get("categories") or {}).get(key)
        if not cat or not cat.get("tag") or not cat.get("enabled", True):
            return []
        extra_master = float(cfg.get("extra_master", 1.0))
        extra = float(cfg["distance"].get("extra", 0.0))
        wmin = float(cfg.get("weight_min", 0.1))
        wmax = float(cfg.get("weight_max", 10.0))
        start, end = DIST_RANGES[key]
        if key in DIST_FAR_STRONGER:
            frac = max(0.0, min(1.0, (end - float(z)) / (end - start)))  # 멀수록 가중치가 큽니다
        else:
            frac = max(0.0, min(1.0, (float(z) - start) / (end - start)))  # 가까울수록 가중치가 큽니다
        w = 1.0 + frac * extra_master * extra
        w = min(wmax, max(wmin, w))
        return cls._emit_weighted(cat["tag"], w)

    @classmethod
    def _weighted_tilt(cls, cfg):
        tilt = cfg.get("tilt", {})
        if not tilt.get("enabled", True):
            return []
        extra_master = float(cfg.get("extra_master", 1.0))
        extra = float(tilt.get("extra", 0.0))
        wmax = float(cfg.get("weight_max", 10.0))
        w = 1.0 + extra_master * extra
        w = min(wmax, max(0.1, w))
        return cls._emit_weighted(tilt.get("dutch_tag", ""), w)

    def execute(self, pos_x, pos_y, pos_z, roll, config):
        cfg = self._load_config(config)
        if cfg.get("no_weight"):
            return (self._compute_prompt_no_weight(float(pos_x), float(pos_y), float(pos_z), float(roll), cfg),)
        parts = []

        wmin = float(cfg.get("weight_min", 0.1))
        wmax = float(cfg.get("weight_max", 5.0))
        dz = float(cfg.get("azimuth", {}).get("deadzone_ratio", 0.2))

        # ---------- 방위(Azimuth) 2D 비율 분배 ----------
        if cfg.get("azimuth", {}).get("enabled", True):
            az = float(pos_x) * math.pi
            front = max(0.0, math.cos(az))
            back = max(0.0, -math.cos(az))
            right = max(0.0, math.sin(az))
            left = max(0.0, -math.sin(az))
            # 총 합에 대해 정규화: 네 방향으로 100% 가중치 분할 (azimuth.weight 총 가중치 예산)）
            s = front + back + left + right
            if s > 0:
                front /= s
                back /= s
                left /= s
                right /= s
            # 위치 총 예산 = azimuth.weight × 극향 게이트: 방향 가중치는 링 어드각도와 관련됨，
            # 기본 예산은 azimuth.weight (상하 각도 스케일링에 영향 없음); 카메라가 가까울 때만 적용“완전히 위/아래를 가리킵니다”
            # (|pos_y|→1) 시간 수평 투영이 0에 수렴하고, 문컨→0，앞, 뒤, 좌, 우 출력 안 함。
            AZ_POLE = 0.9
            az_gate = max(0.0, min(1.0, (1.0 - abs(float(pos_y))) / (1.0 - AZ_POLE)))
            az_budget = float(cfg["azimuth"]["weight"]) * az_gate
            for name, ratio in (("front", front), ("back", back), ("left", left), ("right", right)):
                dir_cfg = cfg["azimuth"]["directions"].get(name, {})
                if not dir_cfg.get("enabled", True):
                    continue
                w = ratio * az_budget
                if ratio <= 0 or w < dz:
                    continue
                w = min(wmax, max(wmin, w))
                tag = dir_cfg.get("tag", "")
                parts.extend(self._emit_weighted(tag, w))

        # ---------- 높이(Elevation) 단일 모드 ----------
        # 가중치 방향은 템플릿 유형에 따라 결정됩니다 (극단형 증분 / 중심형 eye 감소), _elevation_weight 참조。
        if cfg.get("elevation", {}).get("enabled", True):
            elev_key = self._elevation_key(float(pos_y))
            elev_cat = (cfg["elevation"].get("categories") or {}).get(elev_key)
            if elev_cat and elev_cat.get("tag") and elev_cat.get("enabled", True):
                # 높이 재사용 없음, 방향 사각지대: 방향 사각지대는“방향 비율”임계값, 높이에 적용하면 가장 가까운 수평 값만
                # 그 부분(정면 시선, 초기 가속도) 모두 제거하여 pos_y≈0 시간 높이 단어 완전 누락。
                ew = self._elevation_weight(cfg, float(pos_y), elev_key)
                if ew > 0:
                    ew = min(wmax, max(wmin, ew))
                    parts.extend(self._emit_weighted(elev_cat["tag"], ew))

        # ---------- 거리 (Distance, 층내 가중치는 z 값에 따라 선형 상승)） ----------
        if cfg.get("distance", {}).get("enabled", True):
            parts.extend(self._distance_parts(cfg, float(pos_z)))

        # ---------- 기울기（Tilt） ----------
        if cfg.get("tilt", {}).get("enabled", True):
            if abs(float(roll)) >= float(cfg["tilt"]["deadzone"]):
                parts.extend(self._weighted_tilt(cfg))

        # ---------- 추가 카메라 프롬프트 ----------
        extras = cfg.get("extras", {})
        for key in ("lens", "dof", "movement", "composition", "style"):
            e = extras.get(key)
            if not e or not e.get("enabled"):
                continue
            val = (e.get("value") or "").strip()
            if not val:
                continue
            if key == "dof":
                w = e.get("weight", 1.3)
                parts.append(f"({val}:{self._fmt_weight(w)})")
            else:
                parts.append(val)

        result = ", ".join(parts)
        if result:
            result += ","
        return (result,)

    @classmethod
    def _compute_prompt_no_weight(cls, pos_x, pos_y, pos_z, roll, cfg):
        """무 가중치 모드: 태그만 출력( :가중치 없이), 프롬프트 가중치를 지원하지 않는 모델과 호환됩니다.。
        방향은 주 방향만 출력; 부방향은 비율이 있을 때만 출력 >= 임계값만 출력 (임계값 조정 가능, 기본값 0.5=중선 전환）。
        높이/거리 값을 등급별로 출력하며, 기본 등급(눈/중간)은 출력하지 않습니다.。"""
        parts = []
        thr = float(cfg.get("no_weight_threshold", 0.5))
        az = cfg.get("azimuth", {})
        if az.get("enabled", True):
            a = float(pos_x) * math.pi
            front = max(0.0, math.cos(a))
            back = max(0.0, -math.cos(a))
            right = max(0.0, math.sin(a))
            left = max(0.0, -math.sin(a))
            s = front + back + left + right
            if s > 0:
                front /= s
                back /= s
                left /= s
                right /= s
            AZ_POLE = 0.9
            az_gate = max(0.0, min(1.0, (1.0 - abs(float(pos_y))) / (1.0 - AZ_POLE)))
            if az_gate > 0:
                dirs = (("front", front), ("back", back), ("left", left), ("right", right))
                dom = None
                dom_r = -1.0
                for name, ratio in dirs:
                    d = az.get("directions", {}).get(name, {})
                    if not d.get("enabled", True):
                        continue
                    if ratio > dom_r:
                        dom_r = ratio
                        dom = name
                if dom is not None and dom_r > 0:
                    parts.extend(cls._emit_plain(az["directions"][dom].get("tag", "")))
                for name, ratio in dirs:
                    if name == dom:
                        continue
                    d = az.get("directions", {}).get(name, {})
                    if not d.get("enabled", True):
                        continue
                    if ratio >= thr:
                        parts.extend(cls._emit_plain(d.get("tag", "")))
        # 높이(eye 기본 정면 파일 출력 안 함)）
        if cfg.get("elevation", {}).get("enabled", True):
            ek = cls._elevation_key(float(pos_y))
            if ek != "eye":
                ecat = (cfg["elevation"].get("categories") or {}).get(ek)
                if ecat and ecat.get("tag") and ecat.get("enabled", True):
                    parts.extend(cls._emit_plain(ecat["tag"]))
        # 거리 (medium 기본 중경도 출력 안 함)）
        if cfg.get("distance", {}).get("enabled", True):
            dk = cls._distance_key(float(pos_z))
            if dk != "medium":
                dcat = (cfg["distance"].get("categories") or {}).get(dk)
                if dcat and dcat.get("tag") and dcat.get("enabled", True):
                    parts.extend(cls._emit_plain(dcat["tag"]))
        # 스라잉 (순수) tag）
        if cfg.get("tilt", {}).get("enabled", True) and abs(float(roll)) >= float(cfg["tilt"]["deadzone"]):
            parts.extend(cls._emit_plain(cfg["tilt"].get("dutch_tag", "")))
        # 추가 프롬프트 (dof도 순수) tag）
        extras = cfg.get("extras", {})
        for key in ("lens", "dof", "movement", "composition", "style"):
            e = extras.get(key)
            if not e or not e.get("enabled"):
                continue
            val = (e.get("value") or "").strip()
            if val:
                parts.append(val)
        result = ", ".join(parts)
        return result + "," if result else ""


NODE_CLASS_MAPPINGS = {
    "CameraControlNode": CameraControlNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "CameraControlNode": "BSK_카메라 제어",
}
