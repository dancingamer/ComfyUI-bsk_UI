# SPDX-FileCopyrightText: 2026 灰暗x
# SPDX-License-Identifier: AGPL-3.0-or-later
import os

class TxtPromptBatchNode:
    """출력 건너뛰기 전 N개의 프롬프트 후 전체 파일 내용을 (쉼표로 구분, 쉼표로 끝맺음)）"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory": ("STRING", {"default": "input/prompts"}),
                "file_name": ("STRING", {"default": ""}),
                "skip_count": ("INT", {"default": 0, "min": 0, "step": 1}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "execute"
    CATEGORY = "bsk_UI/텍스트"

    def execute(self, directory, file_name, skip_count):
        if not file_name:
            return ("",)
        file_path = os.path.join(directory, file_name)
        if not os.path.isfile(file_path):
            return ("",)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            print(f"[TxtPromptBatchNode] 읽기 실패: {e}")
            return ("",)

        # 마지막에 쉼표가 있는지 확인합니다.）
        if content and not content.rstrip().endswith(","):
            content = content.rstrip() + ","

        # 프롬프트 분할, skip_count 개의 건너뺭
        prompts = [p.strip() for p in content.split(",") if p.strip()]
        if skip_count > 0:
            prompts = prompts[skip_count:]

        # 재조합, 끝에 쉼표 유지
        result = ",".join(prompts)
        if result and not result.endswith(","):
            result += ","
        return (result,)

NODE_CLASS_MAPPINGS = {"TxtPromptBatchNode": TxtPromptBatchNode}
NODE_DISPLAY_NAME_MAPPINGS = {"TxtPromptBatchNode": "BSK TXT 프롬프트 (처리）"}