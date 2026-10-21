# SPDX-License-Identifier: AGPL-3.0-or-later
"""
BSK_UnloadModels — 재렌더링 작업(예: VideoVAE 디코딩)을 수행하기 전에 정확하게 언로드합니다
대규모 모델(UNET/CLIP)을 지정하여 ComfyUI의 smart-memory가 모델을 이동하는 것을 방지합니다.
CPU RAM EXIF UserComment 내장 (메타데이터 비활성화되지 않았고 스위치가 켜져 있는 경우에만)。

메커니즘: 직접 호출 comfy.model_management.unload_model_and_clones(patcher, False)，
지정된 패처만 제거하고, 나머지 모델 (다른 것 포함, AudioVAE 등)은 메모리에 유지합니다.。
모델이 언로드되면 ModelPatcher.detach()를 통해 완전히 해제 (메모리)+RAM 동시에 해제, 이동 없음 RAM）。
"""
import comfy.model_management as model_management


class BSK_UnloadModels:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "unload_model": ("BOOLEAN", {"default": True, "label_on": "アンロードUNET", "label_off": "보존UNET"}),
                "unload_clip": ("BOOLEAN", {"default": False, "label_on": "アンロードCLIP", "label_off": "보존CLIP"}),
            },
            "optional": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("samples",)
    FUNCTION = "unload"
    CATEGORY = "bsk_UI/비디오"
    OUTPUT_NODE = True

    def unload(self, samples, unload_model, unload_clip, model=None, clip=None):
        # 명시적으로 지정된 모델만 제거; 다른 모델 (또 다른, AudioVAE 등)은 메모리에 유지。
        if unload_model and model is not None:
            model_management.unload_model_and_clones(model, unload_additional_models=False)
        if unload_clip and clip is not None and getattr(clip, "patcher", None) is not None:
            model_management.unload_model_and_clones(clip.patcher, unload_additional_models=False)
        return (samples,)


NODE_CLASS_MAPPINGS = {
    "BSK_UnloadModels": BSK_UnloadModels,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "BSK_UnloadModels": "BSK 모델 정확한 삭제",
}
