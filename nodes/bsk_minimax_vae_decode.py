# nodes/bsk_minimax_vae_decode.py
# BSK MiniMax H3 분할 VAE 디코딩 (NestedTensor 호환성 문제를 해결하여 VAEDecodeTiled의 공식 VAE 디코딩을 지원합니다)）
import torch


class BSK_MiniMaxVAEDecodeTiled:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "vae": ("VAE",),
                "tile_size": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 32, "advanced": True}),
                "overlap": ("INT", {"default": 64, "min": 0, "max": 4096, "step": 32, "advanced": True}),
                "temporal_size": ("INT", {"default": 64, "min": 8, "max": 4096, "step": 4,
                                          "tooltip": "비디오 VAE 한 번 디코딩 프레임 수", "advanced": True}),
                "temporal_overlap": ("INT", {"default": 8, "min": 4, "max": 4096, "step": 4, "advanced": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "decode"
    CATEGORY = "bsk_UI/비디오"

    def decode(self, vae, samples, tile_size, overlap=64, temporal_size=64, temporal_overlap=8):
        if tile_size < overlap * 4:
            overlap = tile_size // 4
        if temporal_size < temporal_overlap * 2:
            temporal_overlap = temporal_overlap // 2

        temporal_compression = vae.temporal_compression_decode()
        if temporal_compression is not None:
            temporal_size = max(2, temporal_size // temporal_compression)
            temporal_overlap = max(1, min(temporal_size // 2, temporal_overlap // temporal_compression))
        else:
            temporal_size = None
            temporal_overlap = None

        compression = vae.spacial_compression_decode()

        # === 핵심 수정: MiniMaxH3 샘플링 출력 comfy.nested_tensor.NestedTensor ===
        # It is not a torch.Tensor subclass, and the current environment has no torch.is_nested() function.，
        # 그래서 comfy의 NestedTensor 유형을 사용 + .is_nested 속성 판별。
        # NestedTensor.tensors 다음"시간 분할"저장된 일반 텐서 목록 (각 시간 간격이 다를 수 있음)），
        # 전체적으로 펼치면 메모리를 한 번에 모두 소비하여 cgroup 50GB 벽에 부딪혀 OOM Killer에 의해 죽습니다。
        # 각 시간 간격별로 개별적으로 decode_tiled를 수행하고, 각 덩어리를 한 번만 확장하여 피크 메모리를 단일 덩어리 수준으로 줄인 다음 시간 차원으로 합쳐서 구성。
        samples = dict(samples)
        latent = samples["samples"]

        def _decode_one(piece):
            return vae.decode_tiled(
                piece,
                tile_x=tile_size // compression,
                tile_y=tile_size // compression,
                overlap=overlap // compression,
                tile_t=temporal_size,
                overlap_t=temporal_overlap,
            )

        if getattr(latent, "is_nested", False):
            # piece 모양: [B, 24, T_i, H_lat, W_lat] -> 픽셀 [B, 3, T_i_out, H, W]
            frames = [_decode_one(p) for p in latent.unbind()]
            images = torch.cat(frames, dim=2)  # 시간 축에 따라 연결
            del frames
        else:
            images = _decode_one(latent)

        if len(images.shape) == 5:  # 병합 배치 차원
            images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        return (images,)


NODE_CLASS_MAPPINGS = {"BSK_MiniMaxVAEDecodeTiled": BSK_MiniMaxVAEDecodeTiled}
NODE_DISPLAY_NAME_MAPPINGS = {"BSK_MiniMaxVAEDecodeTiled": "BSK MiniMax 블록 디코딩"}
