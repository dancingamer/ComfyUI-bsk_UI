"""
BSK_TextGenerate / BSK_TextGenerateLTX2Prompt

ComfyUI 핵심 노드에서 수정 comfy_extras/nodes_textgen.py
（상류 프로젝트: ComfyUI, Apache License 2.0, Copyright ComfyUI 프로젝트 및 기여자）。

이 파일은 상류의 comfy_api.latest 동적 스키마(io.DynamicCombo)를 수정합니다.
표준 ComfyUI INPUT_TYPES를 사용하여 노드가 bsk_UI 패널/편집기에서 원시 렌더링 및 직렬화되도록 합니다.
（DynamicCombo 상위 요소에서 중첩된 하위 입력이 손실되는 문제를 방지합니다.）。
핵심 호출 로직 (clip.tokenize / clip.generate / clip.decode)은 상위와 일치함。

============================================================================
상류 라이선스 선언 (Apache License 2.0, 주요 내용)）：
  - 본 파생 작품은 동일 라이선스 사용（Apache-2.0）。
  - 업스트림 저작권: Copyright (c) ComfyUI 프로젝트 및 기여자。
  - 본 파일을 배포할 경우, 상기 저작권 및 라이선스 선언을 유지해야 합니다。
  - 본 파일은 상류 상표/이름에 대한 사용 허가를 의미하지 않습니다。
============================================================================
"""

# 참고: 이 노드는 tokenize/generate/decode를 제공하는 특별한 CLIP(즉 ComfyUI
# 텍스트 생성 모델 로더가 생성한 clip, 예: Gemma 클래스 텍스트 생성 모델). 이 로더는
# 실행 환경(ComfyUI 핵심) 제공; 본 파일은 모델 로딩 로직을 중복 구현하지 않음。


# ----------------------------------------------------------------------------
# LTX2 프롬프트 강화에 사용되는 시스템 프롬프트(상류에서 수정) nodes_textgen.py，Apache-2.0）
# ----------------------------------------------------------------------------
LTX2_T2V_SYSTEM_PROMPT = """You are a Creative Assistant. Given a user's raw input prompt describing a scene or concept, expand it into a detailed video generation prompt with specific visuals and integrated audio to guide a text-to-video model.
#### Guidelines
- Strictly follow all aspects of the user's raw input: include every element requested (style, visuals, motions, actions, camera movement, audio).
    - If the input is vague, invent concrete details: lighting, textures, materials, scene settings, etc.
        - For characters: describe gender, clothing, hair, expressions. DO NOT invent unrequested characters.
- Use active language: present-progressive verbs ("is walking," "speaking"). If no action specified, describe natural movements.
- Maintain chronological flow: use temporal connectors ("as," "then," "while").
- Audio layer: Describe complete soundscape (background audio, ambient sounds, SFX, speech/music when requested). Integrate sounds chronologically alongside actions. Be specific (e.g., "soft footsteps on tile"), not vague (e.g., "ambient sound is present").
- Speech (only when requested):
    - For ANY speech-related input (talking, conversation, singing, etc.), ALWAYS include exact words in quotes with voice characteristics (e.g., "The man says in an excited voice: 'You won't believe what I just saw!'").
    - Specify language if not English and accent if relevant.
- Style: Include visual style at the beginning: "Style: <style>, <rest of prompt>." Default to cinematic-realistic if unspecified. Omit if unclear.
- Visual and audio only: NO non-visual/auditory senses (smell, taste, touch).
- Restrained language: Avoid dramatic/exaggerated terms. Use mild, natural phrasing.
    - Colors: Use plain terms ("red dress"), not intensified ("vibrant blue," "bright red").
    - Lighting: Use neutral descriptions ("soft overhead light"), not harsh ("blinding light").
    - Facial features: Use delicate modifiers for subtle features (i.e., "subtle freckles").

#### Important notes:
- Analyze the user's raw input carefully. In cases of FPV or POV, exclude the description of the subject whose POV is requested.
- Camera motion: DO NOT invent camera motion unless requested by the user.
- Speech: DO NOT modify user-provided character dialogue unless it's a typo.
- No timestamps or cuts: DO NOT use timestamps or describe scene cuts unless explicitly requested.
- Format: DO NOT use phrases like "The scene opens with...". Start directly with Style (optional) and chronological scene description.
- Format: DO NOT start your response with special characters.
- DO NOT invent dialogue unless the user mentions speech/talking/singing/conversation.
- If the user's raw input prompt is highly detailed, chronological and in the requested format: DO NOT make major edits or introduce new elements. Add/enhance audio descriptions if missing.

#### Output Format (Strict):
- Single continuous paragraph in natural language (English).
- NO titles, headings, prefaces, code fences, or Markdown.
- If unsafe/invalid, return original user prompt. Never ask questions or clarifications.

Your output quality is CRITICAL. Generate visually rich, dynamic prompts with integrated audio for high-quality video generation.

#### Example
Input: "A woman at a coffee shop talking on the phone"
Output:
Style: realistic with cinematic lighting. In a medium close-up, a woman in her early 30s with shoulder-length brown hair sits at a small wooden table by the window. She wears a cream-colored turtleneck sweater, holding a white ceramic coffee cup in one hand and a smartphone to her ear with the other. Ambient cafe sounds fill the space—espresso machine hiss, quiet conversations, gentle clinking of cups. The woman listens intently, nodding slightly, then takes a sip of her coffee and sets it down with a soft clink. Her face brightens into a warm smile as she speaks in a clear, friendly voice, 'That sounds perfect! I'd love to meet up this weekend. How about Saturday afternoon?' She laughs softly—a genuine chuckle—and shifts in her chair. Behind her, other patrons move subtly in and out of focus. 'Great, I'll see you then,' she concludes cheerfully, lowering the phone.
"""

LTX2_I2V_SYSTEM_PROMPT = """You are a Creative Assistant. Given a user's raw input prompt describing a scene or concept, expand it into a detailed video generation prompt with specific visuals and integrated audio to guide a text-to-video model.
You are a Creative Assistant writing concise, action-focused image-to-video prompts. Given an image (first frame) and user Raw Input Prompt, generate a prompt to guide video generation from that image.

#### Guidelines:
- Analyze the Image: Identify Subject, Setting, Elements, Style and Mood.
- Follow user Raw Input Prompt: Include all requested motion, actions, camera movements, audio, and details. If in conflict with the image, prioritize user request while maintaining visual consistency (describe transition from image to user's scene).
- Describe only changes from the image: Don't reiterate established visual details. Inaccurate descriptions may cause scene cuts.
- Active language: Use present-progressive verbs ("is walking," "speaking"). If no action specified, describe natural movements.
- Chronological flow: Use temporal connectors ("as," "then," "while").
- Audio layer: Describe complete soundscape throughout the prompt alongside actions—NOT at the end. Align audio intensity with action tempo. Include natural background audio, ambient sounds, effects, speech or music (when requested). Be specific (e.g., "soft footsteps on tile") not vague (e.g., "ambient sound").
- Speech (only when requested): Provide exact words in quotes with character's visual/voice characteristics (e.g., "The tall man speaks in a low, gravelly voice"), language if not English and accent if relevant. If general conversation mentioned without text, generate contextual quoted dialogue. (i.e., "The man is talking" input -> the output should include exact spoken words, like: "The man is talking in an excited voice saying: 'You won't believe what I just saw!' His hands gesture expressively as he speaks, eyebrows raised with enthusiasm. The ambient sound of a quiet room underscores his animated speech.")
- Style: Include visual style at beginning: "Style: <style>, <rest of prompt>." If unclear, omit to avoid conflicts.
- Visual and audio only: Describe only what is seen and heard. NO smell, taste, or tactile sensations.
- Restrained language: Avoid dramatic terms. Use mild, natural, understated phrasing.

#### Important notes:
- Camera motion: DO NOT invent camera motion/movement unless requested by the user. Make sure to include camera motion only if specified in the input.
- Speech: DO NOT modify or alter the user's provided character dialogue in the prompt, unless it's a typo.
- No timestamps or cuts: DO NOT use timestamps or describe scene cuts unless explicitly requested.
- Objective only: DO NOT interpret emotions or intentions - describe only observable actions and sounds.
- Format: DO NOT use phrases like "The scene opens with..." / "The video starts...". Start directly with Style (optional) and chronological scene description.
- Format: Never start output with punctuation marks or special characters.
- DO NOT invent dialogue unless the user mentions speech/talking/singing/conversation.
- Your performance is CRITICAL. High-fidelity, dynamic, correct, and accurate prompts with integrated audio descriptions are essential for generating high-quality video. Your goal is flawless execution of these rules.

#### Output Format (Strict):
- Single concise paragraph in natural English. NO titles, headings, prefaces, sections, code fences, or Markdown.
- If unsafe/invalid, return original user prompt. Never ask questions or clarifications.

#### Example output:
Style: realistic - cinematic - The woman glances at her watch and smiles warmly. She speaks in a cheerful, friendly voice, "I think we're right on time!" In the background, a café barista prepares drinks at the counter. The barista calls out in a clear, upbeat tone, "Two cappuccinos ready!" The sound of the espresso machine hissing softly blends with gentle background chatter and the light clinking of cups on saucers.
"""


class BSK_TextGenerate:
    """
    역추적/프롬프트 생성 노드 (ComfyUI 핵심에서 수정). TextGenerate）。

    상위와 차이점: 상위는 comfy_api.latest의 io.DynamicCombo로 설명 sampling_mode
    그리고 동적 서브 입력; 이 노드는 표준 INPUT_TYPES로 temperature/top_k/...를 향상시킴
    항상 존재하는 선택적 상위 입력으로, bsk_UI 패널/편집기가 원시적으로 렌더링 및 직렬화될 수 있습니다.。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "prompt": ("STRING", {
                    "multiline": True,
                    "dynamicPrompts": True,
                    "default": ""
                }),
                "max_length": ("INT", {"default": 512, "min": 1, "max": 32768}),
                "sampling_mode": (["on", "off"], {"default": "on"}),
            },
            "optional": {
                "image": ("IMAGE",),
                "video": ("IMAGE", {"tooltip": "Video frames as image batch. Assumed to be 24 FPS; subsampled to 1 FPS internally."}),
                "audio": ("AUDIO",),
                "thinking": ("BOOLEAN", {"default": False, "tooltip": "Operate in thinking mode if the model supports it."}),
                "use_default_template": ("BOOLEAN", {"default": True, "tooltip": "Use the built in system prompt/template if the model has one.", "advanced": True}),
                # —— 원 DynamicCombo 서브 입력, 상위로 증분 optional（sampling_mode="off" 시작 do_sample=False 무시）——
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.01, "max": 2.0, "step": 0.000001}),
                "top_k": ("INT", {"default": 64, "min": 0, "max": 1000}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),
                "min_p": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}),
                "repetition_penalty": ("FLOAT", {"default": 1.05, "min": 0.0, "max": 5.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "presence_penalty": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 5.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("generated_text",)
    FUNCTION = "execute"
    CATEGORY = "bsk_UI/텍스트 생성"
    DESCRIPTION = "이미지/비디오/오디오 및 텍스트 프롬프트 기반으로 텍스트 생성 모델 (특수 CLIP)을 호출하여 프롬프트 생성 또는 역추적. ComfyUI 핵심에서 영감 TextGenerate（Apache-2.0）。"

    def execute(self, clip, prompt, max_length, sampling_mode,
                image=None, video=None, audio=None,
                thinking=False, use_default_template=True,
                temperature=0.7, top_k=64, top_p=0.95, min_p=0.05,
                repetition_penalty=1.05, seed=0, presence_penalty=0.0):
        # 상위 노드 실행과 일치하는 호출 로직(인수 출처를 dict에서 명시적 인수로 변경)）
        tokens = clip.tokenize(
            prompt,
            image=image,
            skip_template=not use_default_template,
            min_length=1,
            thinking=thinking,
            video=video,
            audio=audio,
        )

        do_sample = sampling_mode == "on"

        generated_ids = clip.generate(
            tokens,
            do_sample=do_sample,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            seed=seed,
        )

        generated_text = clip.decode(generated_ids)
        return (generated_text,)


class BSK_TextGenerateLTX2Prompt(BSK_TextGenerate):
    """
    LTX2 힌트 노드 (ComfyUI 핵심에서수정) TextGenerateLTX2Prompt）。
    모델로 전송하기 전에 이미지 존재 여부에 따라 LTX2 시스템 프롬프트 자동 병합（T2V / I2V）。
    """

    @classmethod
    def INPUT_TYPES(cls):
        # 부모 클래스와 완전히 동일, 분류/표시 이름만 다름
        return super().INPUT_TYPES()

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("generated_text",)
    FUNCTION = "execute"
    CATEGORY = "bsk_UI/텍스트 생성"
    DESCRIPTION = "BSK_TextGenerate TextGenerateLTX2Prompt（Apache-2.0）。"

    def execute(self, clip, prompt, max_length, sampling_mode,
                image=None, video=None, audio=None,
                thinking=False, use_default_template=True,
                temperature=0.7, top_k=64, top_p=0.95, min_p=0.05,
                repetition_penalty=1.05, seed=0, presence_penalty=0.0):
        if image is None:
            formatted_prompt = f"<start_of_turn>system\n{LTX2_T2V_SYSTEM_PROMPT.strip()}<end_of_turn>\n<start_of_turn>user\nUser Raw Input Prompt: {prompt}.<end_of_turn>\n<start_of_turn>model\n"
        else:
            formatted_prompt = f"<start_of_turn>system\n{LTX2_I2V_SYSTEM_PROMPT.strip()}<end_of_turn>\n<start_of_turn>user\n\n<image_soft_token>\n\nUser Raw Input Prompt: {prompt}.<end_of_turn>\n<start_of_turn>model\n"
        return super().execute(
            clip, formatted_prompt, max_length, sampling_mode,
            image=image, video=video, audio=audio,
            thinking=thinking, use_default_template=use_default_template,
            temperature=temperature, top_k=top_k, top_p=top_p, min_p=min_p,
            repetition_penalty=repetition_penalty, seed=seed, presence_penalty=presence_penalty,
        )


NODE_CLASS_MAPPINGS = {
    "BSK_TextGenerate": BSK_TextGenerate,
    "BSK_TextGenerateLTX2Prompt": BSK_TextGenerateLTX2Prompt,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BSK_TextGenerate": "BSK 역방향 프롬프트(TextGenerate)",
    "BSK_TextGenerateLTX2Prompt": "BSK LTX2프롬프트 강화",
}
