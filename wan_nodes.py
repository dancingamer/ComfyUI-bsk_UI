"""
WanVideo 관련 노드
=======================
이미지 -> 비디오 인코딩 및 프레임 마스크 관련 노드 포함。

이 파일은 기반으로 함 ComfyUI-WanVideoWrapper
(https://github.com/kijai/ComfyUI-WanVideoWrapper, 저자: kijai)에서 수정，
이 프로젝트는 Apache License 2.0으로 게시되었습니다. 수정 날짜：2026-07-18。

저작권 정보：
- 원래 코드 Copyright kijai，Licensed under the Apache License, Version 2.0
  (http://www.apache.org/licenses/LICENSE-2.0)
- 본 수정 버전은 2026년 灰暗x에 의해 ComfyUI-bsk_UI (AGPL-3.0)의 일부로 배포되었으며, 저작권이 있습니다.。

법적 요구 사항 또는 서면 동의가 없는 경우 본 문서는"원래 모습"제공합니다, 명시적 또는
최신 버전: B상을 통해만 Apache License 2.0。
"""

import torch
import numpy as np
try:
    import cv2  # 프롬프트 조각에서 추출 (아티스트 기본 이름, 가중치)
except ImportError:  # opencv-python 설치되지 않았을 때 노드 로딩에 영향을 미치지 않음
    cv2 = None
import gc
import math
from comfy import model_management as mm
from comfy.utils import common_upscale

# 상수 정의
PATCH_SIZE = (1, 2, 2)
device = mm.get_torch_device()
offload_device = mm.unet_offload_device()


#region I2V encode with frame selection
class bsk_WanVideoImageToVideoEncodeSelective:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "width": ("INT", {"default": 832, "min": 64, "max": 8096, "step": 8}),
            "height": ("INT", {"default": 480, "min": 64, "max": 8096, "step": 8}),
            "num_frames": ("INT", {"default": 81, "min": 1, "max": 10000, "step": 4}),
            "noise_aug_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.001}),
            "start_latent_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}),
            "end_latent_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}),
            "force_offload": ("BOOLEAN", {"default": True}),
            
            # 선택적 출력 파라미터
            "output_mode": (["single_frame", "frame_range", "all_frames", "none"], {
                "default": "all_frames",
            }),
            "target_frame": ("INT", {"default": 40, "min": 0, "max": 10000}),
            "frame_range_start": ("INT", {"default": 20, "min": 0, "max": 10000}),
            "frame_range_end": ("INT", {"default": 60, "min": 0, "max": 10000}),
            "other_frames_stop_step": ("INT", {"default": 5, "min": 0, "max": 100}),
            "output_frames_as_list": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "vae": ("WANVAE",),
                "clip_embeds": ("WANVIDIMAGE_CLIPEMBEDS",),
                "start_image": ("IMAGE",),
                "end_image": ("IMAGE",),
                "keyframe1": ("IMAGE",),
                "keyframe2": ("IMAGE",),
                "keyframe3": ("IMAGE",),
                "keyframe1_latent_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}),
                "keyframe2_latent_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}),
                "keyframe3_latent_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}),
                "control_embeds": ("WANVIDIMAGE_EMBEDS",),
                "fun_or_fl2v_model": ("BOOLEAN", {"default": True}),
                "temporal_mask": ("MASK",),
                "extra_latents": ("LATENT",),
                "tiled_vae": ("BOOLEAN", {"default": False}),
                "add_cond_latents": ("ADD_COND_LATENTS",),
                "augment_empty_frames": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "empty_frame_pad_image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS", "WANVIDFRAMEMASK")
    RETURN_NAMES = ("image_embeds", "frame_mask")
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"

    def _calculate_keyframe_positions(self, num_frames, has_start, has_kf1, has_kf2, has_kf3, has_end):
        """
        모든 키프레임의 정확한 위치 계산
        비디오 프레임을 평균적으로 4분할하여 각 키 프레임이 위치해야 하는 위치를 계산
        
        dict 반환, 각 키프레임 위치 인덱스 (프레임 레벨)）
        """
        positions = {}
        
        # 계산 세그먼트 크기 (비디오를 4개로 나눔)）
        # num_frames - 1 사용 (프레임 인덱스는 0부터 시작)
        if num_frames <= 1:
            segment_size = 0
        else:
            segment_size = (num_frames - 1) / 4.0
        
        # 각 주요 프레임 위치 계산
        positions['start'] = 0
        positions['keyframe1'] = round(segment_size * 1)  # 25%위치
        positions['keyframe2'] = round(segment_size * 2)  # 50%위치
        positions['keyframe3'] = round(segment_size * 3)  # 75%위치
        positions['end'] = num_frames - 1  # 마지막 프레임
        
        # 확인하세요 위치가 유효한 범위 내에 있습니다.
        if num_frames > 1:
            positions['keyframe1'] = min(positions['keyframe1'], num_frames - 2)
            positions['keyframe2'] = min(positions['keyframe2'], num_frames - 2)
            positions['keyframe3'] = min(positions['keyframe3'], num_frames - 2)
        
        # 위치 순서 정확히 (겹치지 않도록)）
        occupied = set()
        if has_start:
            occupied.add(positions['start'])
        
        if has_kf1:
            # 첫 번째 사용 가능한 위치를 찾습니다
            while positions['keyframe1'] in occupied and positions['keyframe1'] < num_frames - 1:
                positions['keyframe1'] += 1
            occupied.add(positions['keyframe1'])
        
        if has_kf2:
            while positions['keyframe2'] in occupied and positions['keyframe2'] < num_frames - 1:
                positions['keyframe2'] += 1
            occupied.add(positions['keyframe2'])
        
        if has_kf3:
            while positions['keyframe3'] in occupied and positions['keyframe3'] < num_frames - 1:
                positions['keyframe3'] += 1
            occupied.add(positions['keyframe3'])
        
        return positions

    def process(self, width, height, num_frames, noise_aug_strength, 
                start_latent_strength, end_latent_strength, force_offload,
                output_mode="all_frames", target_frame=40, 
                frame_range_start=20, frame_range_end=60,
                other_frames_stop_step=5, output_frames_as_list=False,
                **kwargs):
        
        vae = kwargs.get('vae')
        if vae is None:
            raise ValueError("VAE is required for image encoding.")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        H, W = height, width

        lat_h = H // vae.upsampling_factor
        lat_w = W // vae.upsampling_factor

        num_frames = ((num_frames - 1) // 4) * 4 + 1
        
        # ===== 기존 코드 로직 (변경 없음）=====
        has_start = kwargs.get('start_image') is not None
        has_kf1 = kwargs.get('keyframe1') is not None
        has_kf2 = kwargs.get('keyframe2') is not None
        has_kf3 = kwargs.get('keyframe3') is not None
        has_end = kwargs.get('end_image') is not None
        
        fun_or_fl2v_model = kwargs.get('fun_or_fl2v_model', False)
        if not has_start and has_end:
            fun_or_fl2v_model = True
        
        # 키 프레임 위치 계산
        kf_positions = self._calculate_keyframe_positions(
            num_frames, has_start, has_kf1, has_kf2, has_kf3, has_end
        )
        
        # 수집 주요 프레임
        keyframes = []
        if has_start:
            keyframes.append(('start', kf_positions['start'], kwargs.get('start_image'), start_latent_strength))
        if has_kf1:
            keyframes.append(('keyframe1', kf_positions['keyframe1'], kwargs.get('keyframe1'), kwargs.get('keyframe1_latent_strength', 1.0)))
        if has_kf2:
            keyframes.append(('keyframe2', kf_positions['keyframe2'], kwargs.get('keyframe2'), kwargs.get('keyframe2_latent_strength', 1.0)))
        if has_kf3:
            keyframes.append(('keyframe3', kf_positions['keyframe3'], kwargs.get('keyframe3'), kwargs.get('keyframe3_latent_strength', 1.0)))
        if has_end:
            keyframes.append(('end', kf_positions['end'], kwargs.get('end_image'), end_latent_strength))
        
        keyframes.sort(key=lambda x: x[1])
        
        # 잘못된 분할 매개변수
        concatenated = self._prepare_image_sequence(
            keyframes, num_frames, H, W, device, vae.dtype, noise_aug_strength,
            fun_or_fl2v_model, kwargs.get('empty_frame_pad_image')
        )
        
        # ===== VAE 인코딩 =====
        vae.to(device)
        
        # 모델 유형 확인 (transformer를 통해) in_dim）
        # transformer = model.diffusion_model if hasattr(model, 'diffusion_model') else None
        is_i2v = True
        

        if is_i2v:
            # I2V 모델: 마스크가 있는 입력을 생성해야 합니다
            # 1. 먼저 이미지 시퀀스 인코딩을 실행하세요
            y = vae.encode([concatenated], device, end_=(has_end and not fun_or_fl2v_model), 
                        tiled=kwargs.get('tiled_vae', False))[0]  # [16, T, H, W]
            
            # 2. 마스크 생성 (4 채널)
            mask = torch.ones(4, y.shape[1], lat_h, lat_w, device=device, dtype=y.dtype)
            
            # 3. 마스크와 이미지 연결 latent -> [20, T, H, W]
            image_cond = torch.cat([mask, y], dim=0)
            
            # 4. I2V 모델의 경우, image_embeds는 image_cond
            final_embeds = image_cond
            
            # 5. max_seq_len 계산 (샘플러 필요))
            patches_per_frame = lat_h * lat_w // (PATCH_SIZE[1] * PATCH_SIZE[2])
            frames_per_stride = (num_frames - 1) // 4 + (2 if has_end and not fun_or_fl2v_model else 1)
            max_seq_len = frames_per_stride * patches_per_frame
        else:
            # T2V 모델: 직접 사용 인코딩 결과
            final_embeds = vae.encode([concatenated], device, end_=(has_end and not fun_or_fl2v_model), 
                                    tiled=kwargs.get('tiled_vae', False))[0]
            max_seq_len = math.ceil((lat_h * lat_w) / 4 * final_embeds.shape[1])
        
        
        # 처리 extra_latents
        extra_latents = kwargs.get('extra_latents')
        has_ref = False
        if extra_latents is not None:
            samples = extra_latents["samples"].squeeze(0)
            if is_i2v:
                # I2V에서 extra_latents는 image_cond의 일부여야 합니다
                if image_cond is not None:
                    image_cond = torch.cat([samples, image_cond], dim=1)
                    final_embeds = image_cond
            else:
                final_embeds = torch.cat([samples, final_embeds], dim=1)
            num_frames += samples.shape[1] * 4
            has_ref = True
        
        # ===== 프레임 마스크 생성 =====
        full_denoise_frames = set()
        if output_mode == "single_frame":
            full_denoise_frames.add(target_frame)
        elif output_mode == "frame_range":
            for f in range(frame_range_start, min(frame_range_end + 1, num_frames)):
                full_denoise_frames.add(f)
        else:  # all_frames 또는 none
            for f in range(num_frames):
                full_denoise_frames.add(f)
        
        # latent 공간 인덱스로 변환
        full_denoise_latents = set()
        for frame_idx in full_denoise_frames:
            latent_idx = frame_idx // 4
            full_denoise_latents.add(latent_idx)
        
        # 생성 프레임 마스크
        frame_mask = torch.zeros(num_frames, dtype=torch.bool)
        for f in full_denoise_frames:
            frame_mask[f] = True
        
        # 잠재 마스크 생성
        latent_frame_mask = torch.zeros((num_frames + 3) // 4, dtype=torch.bool)
        for l in full_denoise_latents:
            latent_frame_mask[l] = True
        
        # ===== 구축 image_embeds =====
        image_embeds = {
            # I2V 모델의 경우 제공해야 하는 image_cond
            # "image_cond": image_cond.cpu() if image_cond is not None else None,
            "image_embeds": final_embeds.cpu(),  # 이것이 핵심입니다! I2V, 여기서는 [20, T, H, W]
            "clip_context": kwargs.get('clip_embeds', {}).get("clip_embeds") if kwargs.get('clip_embeds') else None,
            "negative_clip_context": kwargs.get('clip_embeds', {}).get("negative_clip_embeds") if kwargs.get('clip_embeds') else None,
            "num_frames": num_frames,
            "lat_h": lat_h,
            "lat_w": lat_w,
            "fun_or_fl2v_model": fun_or_fl2v_model,
            "has_ref": has_ref,
            
            # 선택적 샘플링 파라미터
            "frame_mask": frame_mask.cpu(),
            "latent_frame_mask": latent_frame_mask.cpu(),
            "other_frames_stop_step": other_frames_stop_step,
            "output_mode": output_mode,
            "target_frame": target_frame,
            "frame_range_start": frame_range_start,
            "frame_range_end": frame_range_end,
            "output_frames_as_list": output_frames_as_list,
            "vae": vae,
            
            # 라벨 모델 유형 지정
            "is_i2v": is_i2v,
        }
        
        # 생성 미리보기 프레임
        # preview_frames = torch.zeros(1, 64, 64, 3)
        # if output_mode != "none" and final_latent is not None:
        #     try:
        #         preview_frames = self._generate_preview_frames(
        #             final_latent if final_latent is not None else image_cond,
        #             vae, device, output_mode, target_frame, 
        #             frame_range_start, frame_range_end,
        #             num_frames, output_frames_as_list
        #         )
        #     except Exception as e:
        #         print(f"Preview generation failed: {e}")
        
        if force_offload:
            vae.model.to(torch.device("cpu"))
        
        return (image_embeds,  frame_mask)
    
    # ... 기타 보조 방법 유지 ...
    #endregion
    def _prepare_image_sequence(self, keyframes, num_frames, H, W, device, dtype, 
                                noise_aug_strength, fun_or_fl2v_model, empty_frame_pad_image):
        """이미지 시퀀스 준비 - 올바른 범위 사용 [-1, 1]"""
        import torch
        from comfy.utils import common_upscale
        
        prepared_frames = {}
        for name, pos, img, strength in keyframes:
            if img is not None:
                img = img[..., :3]
                
                # 크기 조정
                if img.shape[1] != H or img.shape[2] != W:
                    resized = common_upscale(img.movedim(-1, 1), W, H, "lanczos", "disabled").movedim(0, 1)
                else:
                    resized = img.permute(3, 0, 1, 2)
                
                # ComfyUI IMAGE는 [0,1] 범위, 다음으로 변환 [-1,1]
                resized = resized * 2 - 1
                
                if noise_aug_strength > 0.0:
                    sigma = torch.ones((resized.shape[0],)).to(device, dtype) * noise_aug_strength
                    noise = torch.randn_like(resized) * sigma[:, None, None, None]
                    resized = resized + noise
                    resized = torch.clamp(resized, -1.0, 1.0)
                
                prepared_frames[pos] = resized.to(device, dtype=dtype)
        
        # -1로 채우기 ([-1,1] 범위 내의 검은색은 -1）
        sequence = torch.full((3, num_frames, H, W), -1.0, device=device, dtype=dtype)
        
        for pos, frame_tensor in prepared_frames.items():
            num_frame_images = frame_tensor.shape[1]
            end_pos = min(pos + num_frame_images, num_frames)
            sequence[:, pos:end_pos] = frame_tensor[:, :end_pos-pos]
        
        return sequence
    
    def _generate_preview_frames(self, latents, vae, device, output_mode, target_frame,
                                 frame_range_start, frame_range_end, num_frames, as_list):
        """생성 미리보기 프레임"""
        import torch
        preview_frames = []
        
        vae.to(device)
        
        if output_mode == "single_frame":
            # 대상 프레임만 디코딩
            latent_idx = target_frame // 4
            if latent_idx < latents.shape[1]:
                frame_latent = latents[:, latent_idx:latent_idx+1].to(device)
                decoded = vae.decode(frame_latent.unsqueeze(0), device=device, tiled=False)[0]
                # [C, T, H, W] -> [H, W, C]
                frame = decoded[:, 0].permute(1, 2, 0).cpu()
                frame = (frame + 1) / 2  # [-1,1] -> [0,1]
                preview_frames = [frame]
        
        elif output_mode == "frame_range":
            # 디코딩 범위 내의 모든 프레임
            for frame_idx in range(frame_range_start, min(frame_range_end + 1, num_frames)):
                latent_idx = frame_idx // 4
                if latent_idx < latents.shape[1]:
                    frame_latent = latents[:, latent_idx:latent_idx+1].to(device)
                    decoded = vae.decode(frame_latent.unsqueeze(0), device=device, tiled=False)[0]
                    frame = decoded[:, 0].permute(1, 2, 0).cpu()
                    frame = (frame + 1) / 2
                    preview_frames.append(frame)
        
        else:  # all_frames
            # 모든 프레임 디코딩 (하지만 핵심 프레임만 추출하여 너무 많은）
            step = max(1, num_frames // 10)  # 최대 10프레임 표시
            for frame_idx in range(0, num_frames, step):
                latent_idx = frame_idx // 4
                if latent_idx < latents.shape[1]:
                    frame_latent = latents[:, latent_idx:latent_idx+1].to(device)
                    decoded = vae.decode(frame_latent.unsqueeze(0), device=device, tiled=False)[0]
                    frame = decoded[:, 0].permute(1, 2, 0).cpu()
                    frame = (frame + 1) / 2
                    preview_frames.append(frame)
        
        if as_list:
            # 목록으로 돌아가기
            return preview_frames
        else:
            # 붙여서 비디오 만들기
            if not preview_frames:
                return torch.zeros(1, 64, 64, 3)
            
            if len(preview_frames) == 1:
                return preview_frames[0].unsqueeze(0)
            else:
                return torch.stack(preview_frames, dim=0)


#region 프레임 마스크 생성 노드
class bsk_WanVideoFrameMask:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "num_frames": ("INT", {"default": 81, "min": 1, "max": 10000}),
                "mode": (["all", "single", "range", "custom"], {"default": "all"}),
                "single_frame": ("INT", {"default": 40, "min": 0}),
                "range_start": ("INT", {"default": 20, "min": 0}),
                "range_end": ("INT", {"default": 60, "min": 0}),
                "invert": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "custom_indices": ("STRING", {
                    "default": "0,10,20,30",
                    "tooltip": "쉼표로 구분된 프레임 인덱스 목록"
                }),
            }
        }
    
    RETURN_TYPES = ("WANVIDFRAMEMASK", "IMAGE")
    RETURN_NAMES = ("frame_mask", "visualization")
    FUNCTION = "create_mask"
    CATEGORY = "WanVideoWrapper"
    
    def create_mask(self, num_frames, mode, single_frame, range_start, range_end,
                   invert=False, custom_indices=""):
        import torch
        import numpy as np
        
        mask = torch.zeros(num_frames, dtype=torch.bool)
        
        if mode == "all":
            mask[:] = True
        elif mode == "single":
            if 0 <= single_frame < num_frames:
                mask[single_frame] = True
        elif mode == "range":
            start = max(0, range_start)
            end = min(num_frames - 1, range_end)
            mask[start:end+1] = True
        elif mode == "custom":
            try:
                indices = [int(x.strip()) for x in custom_indices.split(",")]
                for idx in indices:
                    if 0 <= idx < num_frames:
                        mask[idx] = True
            except:
                pass
        
        if invert:
            mask = ~mask
        
        # 시각화 이미지 생성
        vis_height = 64
        vis_width = num_frames * 2  # 프레임당 2픽셀 폭
        
        vis = np.ones((vis_height, vis_width, 3), dtype=np.float32)
        
        for i in range(num_frames):
            x_start = i * 2
            x_end = (i + 1) * 2
            if mask[i]:
                # 선택된 프레임이 녹색으로 표시됩니다
                vis[:, x_start:x_end] = [0, 1, 0]
            else:
                # 선택되지 않은 프레임이 회색으로 표시
                vis[:, x_start:x_end] = [0.3, 0.3, 0.3]
        
        # 프레임 번호 표시 (cv2 사용 불가 시 건너뺌)）
        if cv2 is not None:
            for i in range(0, num_frames, 10):
                x = i * 2
                cv2.putText(vis, str(i), (x, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (1, 1, 1), 1)
        
        return (mask, torch.from_numpy(vis).unsqueeze(0))
#endregion


#region I2V encode
class bsk_WanVideoImageToVideoEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "width": ("INT", {"default": 832, "min": 64, "max": 8096, "step": 8, "tooltip": "Width of the image to encode"}),
            "height": ("INT", {"default": 480, "min": 64, "max": 8096, "step": 8, "tooltip": "Height of the image to encode"}),
            "num_frames": ("INT", {"default": 81, "min": 1, "max": 10000, "step": 4, "tooltip": "Number of frames to encode"}),
            "noise_aug_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.001, "tooltip": "Strength of noise augmentation, helpful for I2V where some noise can add motion and give sharper results"}),
            "start_latent_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001, "tooltip": "Additional latent multiplier for start frame, helpful for I2V where lower values allow for more motion"}),
            "end_latent_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001, "tooltip": "Additional latent multiplier for end frame, helpful for I2V where lower values allow for more motion"}),
            "force_offload": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "vae": ("WANVAE",),
                "clip_embeds": ("WANVIDIMAGE_CLIPEMBEDS", {"tooltip": "Clip vision encoded image"}),
                "start_image": ("IMAGE", {"tooltip": "Start frame image (position 0)"}),
                "end_image": ("IMAGE", {"tooltip": "End frame image (last position)"}),
                # 새로 추가된 3개의 중간 키프레임
                "keyframe1": ("IMAGE", {"tooltip": "Keyframe 1 (position at ~25% of video)"}),
                "keyframe2": ("IMAGE", {"tooltip": "Keyframe 2 (position at ~50% of video)"}),
                "keyframe3": ("IMAGE", {"tooltip": "Keyframe 3 (position at ~75% of video)"}),
                # 각 키프레임의latent strength
                "keyframe1_latent_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001, "tooltip": "Latent strength for keyframe 1"}),
                "keyframe2_latent_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001, "tooltip": "Latent strength for keyframe 2"}),
                "keyframe3_latent_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001, "tooltip": "Latent strength for keyframe 3"}),
                "control_embeds": ("WANVIDIMAGE_EMBEDS", {"tooltip": "Control signal for the Fun -model"}),
                "fun_or_fl2v_model": ("BOOLEAN", {"default": True, "tooltip": "Enable when using official FLF2V or Fun model"}),
                "temporal_mask": ("MASK", {"tooltip": "mask"}),
                "extra_latents": ("LATENT", {"tooltip": "Extra latents to add to the input front, used for Skyreels A2 reference images"}),
                "tiled_vae": ("BOOLEAN", {"default": False, "tooltip": "Use tiled VAE encoding for reduced memory use"}),
                "add_cond_latents": ("ADD_COND_LATENTS", {"advanced": True, "tooltip": "Additional cond latents WIP"}),
                "augment_empty_frames": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "EXPERIMENTAL: Augment empty frames with the difference to the start image to force more motion"}),
                "empty_frame_pad_image": ("IMAGE", {"tooltip": "Use this image to pad empty frames instead of gray, used with SVI-shot and SVI 2.0 LoRAs"}),
            }
        }

    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS",)
    RETURN_NAMES = ("image_embeds",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"

    def _calculate_keyframe_positions(self, num_frames, has_start, has_kf1, has_kf2, has_kf3, has_end):
        """
        모든 키프레임의 정확한 위치 계산
        비디오 프레임을 평균적으로 4분할하여 각 키 프레임이 위치해야 하는 위치를 계산
        
        dict 반환, 각 키프레임 위치 인덱스 (프레임 레벨)）
        """
        positions = {}
        
        # 계산 세그먼트 크기 (비디오를 4개로 나눔)）
        # num_frames - 1 사용 (프레임 인덱스는 0부터 시작)
        if num_frames <= 1:
            segment_size = 0
        else:
            segment_size = (num_frames - 1) / 4.0
        
        # 각 주요 프레임 위치 계산
        positions['start'] = 0
        positions['keyframe1'] = round(segment_size * 1)  # 25%위치
        positions['keyframe2'] = round(segment_size * 2)  # 50%위치
        positions['keyframe3'] = round(segment_size * 3)  # 75%위치
        positions['end'] = num_frames - 1  # 마지막 프레임
        
        # 확인하세요 위치가 유효한 범위 내에 있습니다.
        if num_frames > 1:
            positions['keyframe1'] = min(positions['keyframe1'], num_frames - 2)
            positions['keyframe2'] = min(positions['keyframe2'], num_frames - 2)
            positions['keyframe3'] = min(positions['keyframe3'], num_frames - 2)
        
        # 위치 순서 정확히 (겹치지 않도록)）
        occupied = set()
        if has_start:
            occupied.add(positions['start'])
        
        if has_kf1:
            # 첫 번째 사용 가능한 위치를 찾습니다
            while positions['keyframe1'] in occupied and positions['keyframe1'] < num_frames - 1:
                positions['keyframe1'] += 1
            occupied.add(positions['keyframe1'])
        
        if has_kf2:
            while positions['keyframe2'] in occupied and positions['keyframe2'] < num_frames - 1:
                positions['keyframe2'] += 1
            occupied.add(positions['keyframe2'])
        
        if has_kf3:
            while positions['keyframe3'] in occupied and positions['keyframe3'] < num_frames - 1:
                positions['keyframe3'] += 1
            occupied.add(positions['keyframe3'])
        
        return positions

    def add_noise_to_reference_video(self, image, ratio=None):
        sigma = torch.ones((image.shape[0],)).to(image.device, image.dtype) * ratio
        image_noise = torch.randn_like(image) * sigma[:, None, None, None]
        image_noise = torch.where(image==-1, torch.zeros_like(image), image_noise)
        image = image + image_noise
        return image

    def _prepare_keyframe_image(self, image, H, W, device, dtype, noise_aug_strength):
        """
        주요 프레임 이미지를 준비합니다: 크기 조정, 정규화, 노이즈 추가
        모양이 [C, T, H, W]인 텐서를 반환합니다.
        """
        if image is None:
            return None, 0
        
        image = image[..., :3]
        if image.shape[1] != H or image.shape[2] != W:
            resized = common_upscale(image.movedim(-1, 1), W, H, "lanczos", "disabled").movedim(0, 1)
        else:
            resized = image.permute(3, 0, 1, 2)  # C, T, H, W
        
        resized = resized * 2 - 1  # 정규화 to[-1, 1]
        
        if noise_aug_strength > 0.0:
            resized = self.add_noise_to_reference_video(resized, ratio=noise_aug_strength)
        
        return resized.to(device, dtype=dtype), resized.shape[1]  # 이미지 및 프레임 수를 반환합니다

    def process(self, width, height, num_frames, force_offload, noise_aug_strength, 
                start_latent_strength, end_latent_strength, 
                start_image=None, end_image=None, 
                keyframe1=None, keyframe2=None, keyframe3=None,
                keyframe1_latent_strength=1.0, keyframe2_latent_strength=1.0, keyframe3_latent_strength=1.0,
                control_embeds=None, fun_or_fl2v_model=False,
                temporal_mask=None, extra_latents=None, clip_embeds=None, tiled_vae=False, 
                add_cond_latents=None, vae=None, augment_empty_frames=0.0, empty_frame_pad_image=None):

        if vae is None:
            raise ValueError("VAE is required for image encoding.")
        H = height
        W = width

        lat_h = H // vae.upsampling_factor
        lat_w = W // vae.upsampling_factor

        num_frames = ((num_frames - 1) // 4) * 4 + 1
        
        # 어떤 키프레임이 존재하는지 감지합니다
        has_start = start_image is not None
        has_kf1 = keyframe1 is not None
        has_kf2 = keyframe2 is not None
        has_kf3 = keyframe3 is not None
        has_end = end_image is not None
        
        # 만약 종료 프레임만 있고 시작 프레임이 없다면 fun_or_fl2v_model 모드를 활성화하세요
        if not has_start and has_end:
            fun_or_fl2v_model = True
        
        # 모든 주요 프레임 위치 계산
        kf_positions = self._calculate_keyframe_positions(
            num_frames, has_start, has_kf1, has_kf2, has_kf3, has_end
        )
        
        # 모든 존재하는 키프레임 정보를 수집합니다: (name, position, image, strength)
        keyframes = []
        if has_start:
            keyframes.append(('start', kf_positions['start'], start_image, start_latent_strength))
        if has_kf1:
            keyframes.append(('keyframe1', kf_positions['keyframe1'], keyframe1, keyframe1_latent_strength))
        if has_kf2:
            keyframes.append(('keyframe2', kf_positions['keyframe2'], keyframe2, keyframe2_latent_strength))
        if has_kf3:
            keyframes.append(('keyframe3', kf_positions['keyframe3'], keyframe3, keyframe3_latent_strength))
        if has_end:
            keyframes.append(('end', kf_positions['end'], end_image, end_latent_strength))
        
        # 위치로 정렬
        keyframes.sort(key=lambda x: x[1])
        
        # ========== 핵심 수정: 원래 코드와 호환되는 마스크와 프레임 시퀀스 구축 ==========
        
        # 계산 기본 프레임 수 (원래 로직과 일치)）
        two_ref_images = has_start and has_end
        base_frames = num_frames + (1 if two_ref_images and not fun_or_fl2v_model else 0)
        
        # ========== 구축mask ==========
        if temporal_mask is None:
            mask = torch.zeros(1, base_frames, lat_h, lat_w, device=device, dtype=vae.dtype)
            # 시작 프레임 설정mask
            if has_start:
                mask[:, 0:start_image.shape[0]] = 1
            # 종료 프레임 설정mask
            if has_end:
                mask[:, -end_image.shape[0]:] = 1
            # 중간 주요 프레임 설정mask
            for name, pos, img, strength in keyframes:
                if name not in ['start', 'end'] and img is not None:
                    pos_in_mask = min(pos, mask.shape[1] - 1)
                    end_pos = min(pos + img.shape[0], mask.shape[1])
                    mask[:, pos_in_mask:end_pos] = 1
        else:
            mask = common_upscale(temporal_mask.unsqueeze(1).to(device), lat_w, lat_h, "nearest", "disabled").squeeze(1)
            if mask.shape[0] > base_frames:
                mask = mask[:base_frames]
            elif mask.shape[0] < base_frames:
                mask = torch.cat([mask, torch.zeros(base_frames - mask.shape[0], lat_h, lat_w, device=device)])
            mask = mask.unsqueeze(0).to(device, vae.dtype)

        pixel_mask = mask.clone()

        # ========== 원래 로직에 따라 마스크 확장 처리 ==========
        # Repeat first frame mask 4 times
        start_mask_repeated = torch.repeat_interleave(mask[:, 0:1], repeats=4, dim=1)
        
        # 처리 종료 프레임 마스크 (원래 로직과 일치)）
        if has_end and not fun_or_fl2v_model:
            end_mask_repeated = torch.repeat_interleave(mask[:, -1:], repeats=4, dim=1)
            mask = torch.cat([start_mask_repeated, mask[:, 1:-1], end_mask_repeated], dim=1)
        else:
            mask = torch.cat([start_mask_repeated, mask[:, 1:]], dim=1)

        # Reshape mask into groups of 4 frames -> [C, T, H, W]
        # mask shape: [1, T*4, H, W] -> [1, T, 4, H, W] -> [4, T, H, W]
        mask = mask.view(1, mask.shape[1] // 4, 4, lat_h, lat_w)
        mask = mask.movedim(1, 2)[0]  # C, T, H, W

        # ========== 이미지 프레임 시퀀스 준비 ==========
        # 프레임 시작 준비
        if has_start:
            resized_start_image, start_num_frames = self._prepare_keyframe_image(
                start_image, H, W, device, vae.dtype, noise_aug_strength)
        
        # 프레임 종료 준비
        if has_end:
            resized_end_image, end_num_frames = self._prepare_keyframe_image(
                end_image, H, W, device, vae.dtype, noise_aug_strength)
        
        # 중간 키프레임을 준비하세요
        prepared_keyframes = []
        for name, pos, img, strength in keyframes:
            if name not in ['start', 'end'] and img is not None:
                prepared_img, num_imgs = self._prepare_keyframe_image(
                    img, H, W, device, vae.dtype, noise_aug_strength)
                prepared_keyframes.append((name, pos, prepared_img, strength, num_imgs))

        # ========== 완전한 프레임 시퀀스 구축 (원래 로직과 호환)）==========
        if has_start and not has_end:
            zero_frames = torch.zeros(3, num_frames - start_image.shape[0], H, W, device=device, dtype=vae.dtype)
            concatenated = torch.cat([resized_start_image, zero_frames], dim=1)
            del resized_start_image, zero_frames
        elif not has_start and has_end:
            zero_frames = torch.zeros(3, num_frames - end_image.shape[0], H, W, device=device, dtype=vae.dtype)
            concatenated = torch.cat([zero_frames, resized_end_image], dim=1)
            del zero_frames
        elif not has_start and not has_end:
            concatenated = torch.zeros(3, num_frames, H, W, device=device, dtype=vae.dtype)
        else:
            # 시작 프레임과 종료 프레임이 있음
            if fun_or_fl2v_model:
                zero_frames = torch.zeros(3, num_frames - (start_image.shape[0] + end_image.shape[0]), H, W, device=device, dtype=vae.dtype)
            else:
                zero_frames = torch.zeros(3, num_frames - 1, H, W, device=device, dtype=vae.dtype)
            concatenated = torch.cat([resized_start_image, zero_frames, resized_end_image], dim=1)
            del resized_start_image, zero_frames

        # ========== 중간 키프레임을 올바른 위치에 배치 ==========
        for name, pos, prepared_img, strength, num_imgs in prepared_keyframes:
            # 확인하세요 위치가 유효한 범위 내에 있습니다.
            end_pos = min(pos + num_imgs, num_frames)
            actual_pos = min(pos, num_frames - 1)
            
            if prepared_img.shape[1] <= num_frames - actual_pos:
                concatenated[:, actual_pos:actual_pos + prepared_img.shape[1]] = prepared_img
            else:
                concatenated[:, actual_pos:num_frames] = prepared_img[:, :num_frames - actual_pos]

        # ========== 처리empty_frame_pad_image ==========
        if empty_frame_pad_image is not None:
            pad_img = empty_frame_pad_image.clone()[..., :3]
            if pad_img.shape[1] != H or pad_img.shape[2] != W:
                pad_img = common_upscale(pad_img.movedim(-1, 1), W, H, "lanczos", "disabled").movedim(1, -1)
            pad_img = (pad_img.movedim(-1, 0) * 2 - 1).to(device, dtype=vae.dtype)

            num_pad_frames = pad_img.shape[1]
            num_target_frames = concatenated.shape[1]
            if num_pad_frames < num_target_frames:
                pad_img = torch.cat([pad_img, pad_img[:, -1:].expand(-1, num_target_frames - num_pad_frames, -1, -1)], dim=1)
            else:
                pad_img = pad_img[:, :num_target_frames]

            frame_is_empty = (pixel_mask[0].mean(dim=(-2, -1)) < 0.5)[:concatenated.shape[1]].clone()
            
            # 시작 프레임과 종료 프레임을 비워두지 마십시오
            if has_start:
                frame_is_empty[:start_image.shape[0]] = False
            if has_end:
                frame_is_empty[-end_image.shape[0]:] = False
            # 중간 주요 프레임을 비어있게 설정
            for name, pos, prepared_img, strength, num_imgs in prepared_keyframes:
                end_pos = min(pos + num_imgs, num_frames)
                frame_is_empty[pos:end_pos] = False

            concatenated[:, frame_is_empty] = pad_img[:, frame_is_empty]

        mm.soft_empty_cache()
        gc.collect()

        # ========== VAE인코딩 ==========
        vae.to(device)
        y = vae.encode([concatenated], device, end_=(has_end and not fun_or_fl2v_model), tiled=tiled_vae)[0]
        del concatenated

        # ========== 처리extra_latents ==========
        has_ref = False
        if extra_latents is not None:
            samples = extra_latents["samples"].squeeze(0)
            y = torch.cat([samples, y], dim=1)
            mask = torch.cat([torch.ones_like(mask[:, 0:samples.shape[1]]), mask], dim=1)
            num_frames += samples.shape[1] * 4
            has_ref = True
        
        # ========== 적용latent strength ==========
        # 시작 프레임
        y[:, :1] *= start_latent_strength
        # End frame
        y[:, -1:] *= end_latent_strength
        
        # 중간 키프레임의latent strength
        for name, pos, prepared_img, strength, num_imgs in prepared_keyframes:
            # latent내 위치 = 프레임 위치 // 4
            latent_pos = pos // 4
            if latent_pos < y.shape[1]:
                y[:, latent_pos:latent_pos+1] *= strength
        
        if augment_empty_frames > 0.0:
            frame_is_empty = (mask[0].mean(dim=(-2, -1)) < 0.5).view(1, -1, 1, 1)
            y = y[:, :1] + (y - y[:, :1]) * ((augment_empty_frames+1) * frame_is_empty + ~frame_is_empty)

        # Calculate maximum sequence length
        patches_per_frame = lat_h * lat_w // (PATCH_SIZE[1] * PATCH_SIZE[2])
        frames_per_stride = (num_frames - 1) // 4 + (2 if has_end and not fun_or_fl2v_model else 1)
        max_seq_len = frames_per_stride * patches_per_frame

        if add_cond_latents is not None:
            add_cond_latents["ref_latent_neg"] = vae.encode(torch.zeros(1, 3, 1, H, W, device=device, dtype=vae.dtype), device)

        if force_offload:
            vae.model.to(offload_device)
            mm.soft_empty_cache()
            gc.collect()

        # end_image 준비 (존재하는 경우)）
        resized_end_image_out = None
        if has_end:
            resized_end_image_out, _ = self._prepare_keyframe_image(end_image, H, W, device, vae.dtype, 0.0)

        # 키프레임 위치 정보 반환에 사용
        keyframe_info = {
            'positions': kf_positions,
            'has_keyframes': {
                'start': has_start,
                'keyframe1': has_kf1,
                'keyframe2': has_kf2,
                'keyframe3': has_kf3,
                'end': has_end
            }
        }

        image_embeds = {
            "image_embeds": y.cpu(),
            "clip_context": clip_embeds.get("clip_embeds", None) if clip_embeds is not None else None,
            "negative_clip_context": clip_embeds.get("negative_clip_embeds", None) if clip_embeds is not None else None,
            "max_seq_len": max_seq_len,
            "num_frames": num_frames,
            "lat_h": lat_h,
            "lat_w": lat_w,
            "control_embeds": control_embeds["control_embeds"] if control_embeds is not None else None,
            "end_image": resized_end_image_out,
            "fun_or_fl2v_model": fun_or_fl2v_model,
            "has_ref": has_ref,
            "add_cond_latents": add_cond_latents,
            "mask": mask.cpu(),
            "keyframe_info": keyframe_info,  # 의존성 목록 없음, 명령 없음
        }

        return (image_embeds,)
#endregion


# 내보내기 노드 클래스
NODE_CLASS_MAPPINGS = {
    "bsk_WanVideoImageToVideoEncodeSelective": bsk_WanVideoImageToVideoEncodeSelective,
    "bsk_WanVideoFrameMask": bsk_WanVideoFrameMask,
    "bsk_WanVideoImageToVideoEncode": bsk_WanVideoImageToVideoEncode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "bsk_WanVideoImageToVideoEncodeSelective": "bsk_WanVideoImageToVideoEncode프레임 선택",
    "bsk_WanVideoFrameMask": "bsk_WanVideoFrameMask",
    "bsk_WanVideoImageToVideoEncode": "bsk_wan3중간 프레임",
}
