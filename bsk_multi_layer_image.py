# SPDX-FileCopyrightText: 2026 灰暗x
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
BSK 다중 레이어 이미지 로드 노드
유형: 다운스트림 예상 유형을 우선 사용하고, 그렇지 않으면 inN 유형을 유지 (직통)
각 레이어는 하나의 이미지 입력만 필요, 마스크는 alpha 채널에서 자동 추출
"""

import os
import hashlib
import torch
import numpy as np
from PIL import Image, ImageOps
from typing import List, Tuple, Optional, Dict, Any

import folder_paths
import node_helpers


class BSK_MultiLayerImageLoader:
    """
    다중 레이어 이미지 로드 노드
    - 고정 5개 레이어
    - 각 레이어는 하나의 이미지 입력을 필요로 합니다
    - 이미지에 알파 채널이 있으면 자동으로 마스크로 추출합니다.
    - 이미지에 알파 채널이 없으면 흰색 마스크를 반환합니다.
    
    작동 방식은 원본 LoadImage 노드와 유사합니다.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        # 모든 입력 파일 가져오기
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        files = folder_paths.filter_files_content_types(files, ["image"])
        
        # 입력 파라미터 구성 - 모든 레이어는 선택 사항입니다.
        inputs = {"required": {}, "optional": {}}
        
        # 각 레이어에 이미지 파일 선택 (이미지 업로드 기능 포함) - 모두 선택 가능으로 설정
        for i in range(1, 6):  # 5레이어
            inputs["optional"][f"layer_{i}"] = (
                sorted(files),
                {"image_upload": True}
            )
        
        return inputs
    
    # 고정 출력: 각 레이어 출력 이미지 및 마스크
    RETURN_TYPES = (
        "IMAGE", "MASK",  # 레이어1
        "IMAGE", "MASK",  # 레이어2
        "IMAGE", "MASK",  # 레이어3
        "IMAGE", "MASK",  # 레이어4
        "IMAGE", "MASK",  # 레이어5
    )
    RETURN_NAMES = (
        "이미지1", "마스크1",
        "이미지2", "마스크2",
        "이미지3", "마스크3",
        "이미지4", "마스크4",
        "이미지5", "마스크5",
    )
    OUTPUT_IS_LIST = (False,) * 10
    FUNCTION = "load"
    CATEGORY = "BSK/image"
    OUTPUT_NODE = False
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # 모든 이미지 파일의 해시 값을 확인
        hash_list = []
        for i in range(1, 6):
            image_file = kwargs.get(f"layer_{i}", "")
            if image_file:
                image_path = folder_paths.get_annotated_filepath(image_file)
                if os.path.exists(image_path):
                    m = hashlib.sha256()
                    with open(image_path, 'rb') as f:
                        m.update(f.read())
                    hash_list.append(m.digest().hex())
        return ",".join(hash_list) if hash_list else ""
    
    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        # All layers are optional, empty or invalid values will be processed as default image
        # 파일 이름 입력만 유효성 검사, 모든 레이어에 입력이 필요하지 않도록 강제하지 않습니다.
        for i in range(1, 6):
            image_file = kwargs.get(f"layer_{i}", "")
            # 일부 환경에서 UTF-8 표시가 없는 UTF-8 패키지인 경우, 원본으로 직접 반환）
            if image_file and not folder_paths.exists_annotated_filepath(image_file):
                print(f"[BSK MultiLayer] Warning: layer_{i}: '{image_file}' not found, will use default empty image")
        return True
    
    def load(self, **kwargs):
        """
        모든 레이어 로드
        각 레이어 출력 이미지 및 마스크
        마스크는 이미지의 알파 채널에서 자동으로 추출
        빈 값 또는 유효하지 않은 파일은 기본 이미지 및 마스크 반환
        """
        results = []
        
        for i in range(1, 6):  # 5레이어
            image_file = kwargs.get(f"layer_{i}", "")
            
            # 파일 유효성 확인
            is_valid = False
            if image_file:
                try:
                    is_valid = folder_paths.exists_annotated_filepath(image_file)
                except:
                    is_valid = False
            
            if is_valid:
                image_tensor, mask_tensor = self._load_image_with_mask(image_file)
            else:
                # 빈 값 또는 유효하지 않은 파일: 기본 이미지 및 마스크를 반환합니다
                if image_file:
                    print(f"[BSK MultiLayer] layer_{i}: using default empty image (file not found: {image_file})")
                else:
                    print(f"[BSK MultiLayer] layer_{i}: using default empty image (no input)")
                image_tensor = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
                mask_tensor = torch.ones((1, 64, 64), dtype=torch.float32)
            
            results.append(image_tensor)
            results.append(mask_tensor)
        
        return tuple(results)
    
    def _load_image_with_mask(self, image: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        이미지 로드 및 마스크 추출
        LoadImage와 유사한 구현
        
        Returns:
            (image_tensor, mask_tensor)
        """
        image_path = folder_paths.get_annotated_filepath(image)
        
        try:
            img = node_helpers.pillow(Image.open, image_path)
            
            output_images = []
            output_masks = []
            w, h = None, None
            
            # 다중 프레임 이미지 처리 (예 GIF）
            from PIL import ImageSequence
            for i in ImageSequence.Iterator(img):
                i = node_helpers.pillow(ImageOps.exif_transpose, i)
                
                # 16비트 이미지 처리
                if i.mode == 'I':
                    i = i.point(lambda i: i * (1 / 255))
                
                # 변환 RGB
                image_rgb = i.convert("RGB")
                
                if len(output_images) == 0:
                    w = image_rgb.size[0]
                    h = image_rgb.size[1]
                
                if image_rgb.size[0] != w or image_rgb.size[1] != h:
                    continue
                
                # 변환 tensor
                image_array = np.array(image_rgb).astype(np.float32) / 255.0
                image_tensor = torch.from_numpy(image_array)[None,]
                
                # 마스크 추출
                if 'A' in i.getbands():
                    # 알파 채널에서 마스크 추출
                    mask_array = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                    mask_tensor = 1. - torch.from_numpy(mask_array)
                elif i.mode == 'P' and 'transparency' in i.info:
                    # 컬러 팔레트 모드 투명도
                    mask_array = np.array(i.convert('RGBA').getchannel('A')).astype(np.float32) / 255.0
                    mask_tensor = 1. - torch.from_numpy(mask_array)
                else:
                    # No transparency, use a full black mask (the mask indicates complete visibility)）
                    mask_tensor = torch.zeros((h, w), dtype=torch.float32, device="cpu")
                
                output_images.append(image_tensor)
                output_masks.append(mask_tensor.unsqueeze(0))
                
                # MPO 포맷은 첫 번째 프레임만 처리
                if img.format == "MPO":
                    break
            
            # 여러 프레임을 병합하거나 단일 프레임 반환
            if len(output_images) > 1:
                output_image = torch.cat(output_images, dim=0)
                output_mask = torch.cat(output_masks, dim=0)
            else:
                output_image = output_images[0]
                output_mask = output_masks[0]
            
            print(f"[BSK MultiLayer] 이미지 로드: {image}, 크기: {output_image.shape}")
            return (output_image, output_mask)
            
        except Exception as e:
            print(f"[BSK MultiLayer] 이미지 로드 실패: {image}, 오류: {e}")
            return (
                torch.zeros((1, 64, 64, 3), dtype=torch.float32),
                torch.ones((1, 64, 64), dtype=torch.float32)
            )


# 노드 등록 정보
NODE_CLASS_MAPPINGS = {
    "BSK_MultiLayerImageLoader": BSK_MultiLayerImageLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BSK_MultiLayerImageLoader": "BSK 다중 레이어 이미지 로더",
}
