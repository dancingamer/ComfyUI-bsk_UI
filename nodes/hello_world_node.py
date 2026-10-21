# SPDX-FileCopyrightText: 2026 灰暗x
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
HelloWorldNode - bsk_UI 사용자 정의 카드 테스트 노드
사용자 지정 카드 인터페이스 노드를 만드는 방법을 시연합니다
"""

class HelloWorldNode:
    """단순 테스트 노드를 사용하여 사용자 지정 카드 렌더링과 함께 표시합니다"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "중간 생성물 사용!", "multiline": False}),
                "color": ("STRING", {"default": "#4CAF50"}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("출력 텍스트",)
    FUNCTION = "execute"
    CATEGORY = "bsk_UI/테스트"
    DESCRIPTION = "사용자 정의 카드 기능을 보여주는 간단한 테스트 노드"

    def execute(self, text, color):
        return (text,)


NODE_CLASS_MAPPINGS = {
    "HelloWorldNode": HelloWorldNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HelloWorldNode": "BSK 안녕하세요 세상",
}
