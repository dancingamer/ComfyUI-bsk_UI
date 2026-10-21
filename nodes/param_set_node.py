# SPDX-FileCopyrightText: 2026 灰暗x
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
BSK 파라미터 세트 노드 (ParamSet)

10 그룹 파라미터 고정, 각 그룹 = 제목(titleN) + 값(valueN) + 데이터 유형(typeN), 출력 outN。
- 데이터 유형 typeN ∈ {BOOLEAN, INT, FLOAT, STRING}，백엔드에서 valueN을 강제로 변환하는 방법을 결정합니다。
- valueN 선언하다 STRING（forceInput=False）：표준 ComfyUI 프론트엔드와 유사
  bsk_MultiLoraLoaderWithPath 의 부울/플로트와 동일하게 직접 렌더링 노드 내부의 편집 가능한 텍스트 상자，
  단지 입력 엔드포인트가 아닌 경우; 노드에 입력된 값은 워크플로우 내보내기(widgets_values)와 함께 변경되며, 백엔드는 typeN을 통해 파싱합니다.。
  자신의 패널 프론트엔드 (카드)도 typeN에 따라 렌더링 스위치/숫자 입력 상자/텍스트 입력 상자 후 문자열로 작성합니다.。
- 상위 연결될 때: 문자열 입력은 여전히 허용 "*" Type output (such as this node outN or other wildcard output nodes)）
  의 연결, 백엔드에서 비문자 객체 원본 그대로 통과 (typeN 무시）。
- 백엔드 판별: valueN이 문자열 → 편집 상자/카드에서 typeN으로 강제 변환；
  문자열이 아닌 (그리고) None）→ 하위 객체, 직접 연결。
"""

class BSK_ParamSet:
    MAX = 10
    TYPES = ("BOOLEAN", "INT", "FLOAT", "STRING")

    @classmethod
    def INPUT_TYPES(cls):
        optional = {}
        for i in range(1, cls.MAX + 1):
            optional[f"title{i}"] = ("STRING", {"default": "", "forceInput": False, "multiline": False})
            optional[f"value{i}"] = ("STRING", {"default": "", "forceInput": False, "tooltip": "파라미터 값: 연결되지 않았을 때 노드에서 직접 편집; STRING 유형은 표준 ComfyUI 프론트엔드 렌더링을 텍스트 상자로 보장하고 워크플로우 내보내기와 함께 작동합니다. 백엔드에서 typeN으로 강제 변환합니다."})
            optional[f"type{i}"] = (cls.TYPES, {"default": "INT"})
        return {"required": {}, "optional": optional}

    RETURN_TYPES = ("*",) * MAX
    RETURN_NAMES = tuple(f"out{i}" for i in range(1, MAX + 1))
    FUNCTION = "pass_through"
    CATEGORY = "bsk/utilities"
    DESCRIPTION = "파라미터 세트: 10개 그룹 (제목/값/유형), 각 그룹은 변환된 값을 출력합니다; 패널 카드만 연결된 그룹을 확장하여 표시。"

    @staticmethod
    def _coerce(value, type_name):
        """value 문자열 (ComfyUI 노드 텍스트 상자/패널에서)）→ type_name으로 강제 변환; 그렇지 않으면 (상류 객체/None）→ 직접 또는 기본에 연결。"""
        if value is None:
            value = ""
        if isinstance(value, str):
            s = value.strip()
            if s == "":
                if type_name == "INT":
                    return 0
                if type_name == "FLOAT":
                    return 0.0
                if type_name == "BOOLEAN":
                    return False
                return ""
            try:
                if type_name == "INT":
                    return int(float(s))
                if type_name == "FLOAT":
                    return float(s)
                if type_name == "BOOLEAN":
                    return s.lower() in ("1", "true", "yes", "y", "t")
                if type_name == "STRING":
                    return s
            except (TypeError, ValueError):
                return value
            return value
        # 문자열이 아닌: 상류 연결 객체, 원본 그대로 통과
        return value

    def pass_through(self, **kwargs):
        result = []
        for i in range(1, self.MAX + 1):
            v = kwargs.get(f"value{i}", "")
            ty = kwargs.get(f"type{i}", "STRING")
            result.append(self._coerce(v, ty))
        return tuple(result)


NODE_CLASS_MAPPINGS = {
    "BSK_ParamSet": BSK_ParamSet,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BSK_ParamSet": "BSK 파라미터 세트",
}
