# SPDX-FileCopyrightText: 2026 灰暗x
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
PromptFormatNode - 프롬프트 포맷팅 노드
기능：
1. 프롬프트 텍스트를 입력하고 영어 쉼표로 구분
2. Anima 형식 활성화 시 각 프래그먼트 형식 지정 @화자 이름 형식：
   - wlop           -> @wlop
   - (wlop:1.3)     -> (@wlop:1.3)   （괄호로 가중치 있음）
   - (@wlop:1.3)    -> (@wlop:1.3)   （중복 제거 @，괄호 가중치 유지）
   - artist:wlop    -> @wlop
   - (artist:wlop:0.96) -> (@wlop:0.96)
   - (matcha)       -> @matcha       （가중치 괄호 없음）
"""

import re


class PromptFormatNode:
    """프롬프트 포맷팅 노드"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt_text": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "일반 형식의 화상 체인 예시: (artist:asdf:0.96), asdfasdf, 는 Anima 지원 형식으로 포맷됩니다. @asdf, @asdfasdf"
                }),
                "anima_format": ("BOOLEAN", {
                    "default": False,
                    "label_on": "Anima",
                    "label_off": "Anima"
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("프롬프트 텍스트 형식화",)
    FUNCTION = "execute"
    CATEGORY = "bsk_UI/프롬프트"
    DESCRIPTION = "프롬프트 형식화 지원, Anima 모델 아티스트 체인 형식 출력 지원."

    def execute(self, prompt_text, anima_format):
        """
        프롬프트 포맷 실행

        매개변수：
            prompt_text: 프롬프트 텍스트
            anima_format: Anima 형식 아티스트 스트링 출력 여부
        """
        if not prompt_text or not prompt_text.strip():
            return ("",)

        if not anima_format:
            return (prompt_text,)

        # 쉼표로 분할 (괄호 중첩 고려)）
        parts = self._split_by_comma(prompt_text)
        formatted_parts = []
        for part in parts:
            part = part.strip()
            if not part:
                continue
            base_name, weight = self._extract_artist_info(part)
            if base_name:
                if weight:
                    # 가중치 있음：(@name:weight)
                    formatted_parts.append("(" + "@" + base_name + ":" + weight + ")")
                else:
                    # 가중치 없음：@name
                    formatted_parts.append("@" + base_name)

        result = ", ".join(formatted_parts)
        # 끝에 쉼표로 끝나도록 합니다
        if result and not result.rstrip().endswith(","):
            result += ","

        return (result,)

    def _split_by_comma(self, text):
        """쉼표로 구분, 괄호 중첩 고려"""
        parts = []
        depth = 0
        current = ""
        for ch in text:
            if ch in ('(', '{'):
                depth += 1
                current += ch
            elif ch in (')', '}'):
                depth -= 1
                current += ch
            elif ch == ',' and depth == 0:
                parts.append(current)
                current = ""
            else:
                current += ch
        if current.strip():
            parts.append(current)
        return parts

    def _extract_artist_info(self, text):
        """
        비용 및 주의사항은 아래에 각 항목별로 명확히 작성, 이미 겪었기 때문.)。
        반환 (base_name, weight_str) 또는 (None, None)。

        다양한 형식 처리：
          wlop           -> ("wlop", None)
          (wlop:1.3)     -> ("wlop", "1.3")
          (@wlop:1.3)    -> ("wlop", "1.3")
          artist:wlop    -> ("wlop", None)
          (artist:wlop:0.96) -> ("wlop", "0.96")
          (matcha)       -> ("matcha", None)
          @wlop          -> ("wlop", None)
        """
        t = text.strip()
        if not t:
            return None, None

        # 1) 외부 괄호 (xxx:weight) 제거 {xxx:weight}
        weight = None
        if (t.startswith('(') and t.endswith(')')) or \
           (t.startswith('{') and t.endswith('}')):
            inner = t[1:-1].strip()
            colon_pos = self._find_outer_colon(inner)
            if colon_pos > 0:
                after = inner[colon_pos + 1:].strip()
                if re.match(r'^-?\d+\.?\d*$', after):
                    # 가중치 있음：(name:weight)
                    weight = after
                    t = inner[:colon_pos].strip()
                else:
                    # 무게 없음 괄호：(matcha) -> matcha
                    t = inner

        # 2) 제거 아티스트: 접두사
        t = re.sub(r'^artist\s*: ?', '', t, flags=re.IGNORECASE).strip()

        # 3) 기존 제거 @ 접두사 (피하기) @@）
        if t.startswith('@'):
            t = t[1:].strip()

        # 4) 일반 괄호(역할이 없는)를 정리하지만, 역참조 괄호는 유지 \( ... \)
        #    황혼에서 한 오후로 빛이 따라 변하는 모습(조명 변화를 명시하여 모델이 따라하도록 합니다)
        ESC_OPEN = '\x00ESC_OPEN\x00'
        ESC_CLOSE = '\x00ESC_CLOSE\x00'
        t = t.replace('\\(', ESC_OPEN).replace('\\)', ESC_CLOSE)
        t = re.sub(r'[()]', '', t)
        t = t.replace(ESC_OPEN, '\\(').replace(ESC_CLOSE, '\\)')

        # 5) 불필요한 공백을 정리하고 단일 공백으로 병합합니다
        t = re.sub(r'\s+', ' ', t).strip()

        return t if t else None, weight

    def _find_outer_colon(self, text):
        """가장 바깥쪽(괄호 깊이 0)의 마지막 콜론 위치 찾기"""
        depth = 0
        last_colon = -1
        for i, c in enumerate(text):
            if c in ('(', '{'):
                depth += 1
            elif c in (')', '}'):
                depth -= 1
            elif c == ':' and depth == 0:
                last_colon = i
        return last_colon


NODE_CLASS_MAPPINGS = {
    "PromptFormatNode": PromptFormatNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptFormatNode": "BSK_Anima형식 화가 순",
}
