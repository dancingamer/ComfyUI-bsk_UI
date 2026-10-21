# SPDX-FileCopyrightText: 2026 灰暗x
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
BSK 자르기 반환 노드（BSK_CropMerge）

BSK_LoadImage와 함께 사용：
  - 수신 BSK_LoadImage의 "원래 이미지" (크롭/스케일 전 전체 이미지) 및 "크롭 데이터"」。
  - 샘플러 VAE 디코딩 결과 재그리기（patch_image）。
  - 패치 강제 리사이즈를 크롭 사각형 크기로 되돌려 원본 이미지에 다시 붙여넣으세요함 (crop_x, crop_y)，
    중간에 스케일링/VAE의 /8 정렬을 거쳐도 원본 이미지 픽셀을 엄격하게 정렬。
  - 엣지 페더링(feather), 정적 픽셀 간의 딱딱한 연결을 방지합니다; 원본 이미지 경계선과 겹치는 경계는 페더링되지 않습니다.，
    이미지 가장자리 점진적 흐림 방지。
  - 선택적 마스크 입력: 마스크 영역(흐림 후 부드럽게 연출)에서만 혼합, 마스크 외에는 원본 픽셀 유지，
    fp32 정밀도를 사용하고, fp16 훈련 시 활성화하여 문제를 방지합니다。

스위치 및 오류 복구 (사용됨)"정상 모드 ↔ 재그리기 모드"전환）：
  - original_image / patch_image None일 수 있음 (연결되지 않음)）。
  - enabled False 또는 patch_image가 None일 때：
    노드가 직접 진입"투명 전송"분기, 리사이즈/흐림/합성 계산을 하지 않고 반환합니다 original_image
    （8의 정수 배수이므로 두 가지 단계 모두 VAE의 /8 제약 조건을 충족하며, 엄격도는 다릅니다.。
  - 참고: enabled/None은 이 노드를 제로 비용으로 실행할 뿐, 업스트림 샘플러가 다시 그리기 방지할 수 없습니다. ComfyUI
    실행 (ComfyUI 전체 커밋 그래프 전체 실행, 스위치가 상류로 건너가지 않음).
    계산, 워크플로우에서 bypass (우회)하여 자식 서브플롯 다시 그리기。
"""

import torch
import torch.nn.functional as F


class BSK_CropMerge:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "crop_data": ("BSK_CROP_DATA",),
                "feather": ("INT", {
                    "default": 32, "min": 0, "max": 1024, "step": 1,
                    "tooltip": "연결 범위 (원래 이미지 픽셀 좌표계)）。0 = 딱딱한 테이프 되돌리기。",
                }),
                "enabled": ("BOOLEAN", {
                    "default": True, "label_on": "회붙여넣으세요", "label_off": "투명 전송",
                    "tooltip": "종료 시 원본 이미지를 직접 전달하고, 어떤 계산도 수행하지 않음 (정상 모드）。",
                }),
            },
            "optional": {
                "original_image": ("IMAGE",),
                "patch_image": ("IMAGE",),
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("IMAGE", "MASK")
    FUNCTION = "merge"
    CATEGORY = "image"

    def merge(self, crop_data=None, feather=32, enabled=True,
              original_image=None, patch_image=None, mask=None):
        # ---- 직접 분기: 끄거나 패치 없음, 기본 맵으로 직접 반환, 후처리 계산 없음 ----
        if (not enabled) or (patch_image is None) or (not isinstance(patch_image, torch.Tensor)):
            if isinstance(original_image, torch.Tensor):
                B, H, W, _ = original_image.shape
                alpha = torch.zeros((B, H, W), device=original_image.device,
                                    dtype=original_image.dtype)
                return (original_image, alpha)
            if isinstance(patch_image, torch.Tensor):
                B, H, W, _ = patch_image.shape
                alpha = torch.ones((B, H, W), device=patch_image.device,
                                   dtype=patch_image.dtype)
                return (patch_image, alpha)
            # 둘 다 없음: 구성 오류 시 None 반환, 하류 처리 담당
            return (None, None)

        # ---- 활성화되어 있고 패치가 존재하면 원본 이미지를 기반으로 합니다 ----
        if not isinstance(original_image, torch.Tensor):
            # 기본 맵이 없을 경우 퇴화: 직접 반환 patch
            B, H, W, _ = patch_image.shape
            alpha = torch.ones((B, H, W), device=patch_image.device,
                               dtype=patch_image.dtype)
            return (patch_image, alpha)

        device = original_image.device
        original_image = original_image.to(device)
        patch_image = patch_image.to(device)
        if isinstance(mask, torch.Tensor):
            mask = mask.to(device)

        B_o, H, W, C = original_image.shape

        # ---- 파싱 잘라내기 사각형 ----
        if isinstance(crop_data, dict):
            cx = int(crop_data.get("crop_x", 0))
            cy = int(crop_data.get("crop_y", 0))
            cw = int(crop_data.get("crop_w", 0))
            ch = int(crop_data.get("crop_h", 0))
        else:
            cx, cy, cw, ch = 0, 0, 0, 0

        # 잘라내지 않음（crop_w/h == 0）：디미지하여 전체 이미지 사각형으로 변경, 동작은 리사이즈 후 전체 이미지 대체와 동일
        if cw <= 0 or ch <= 0:
            cx, cy, cw, ch = 0, 0, W, H

        # clamp 원래 이미지 범위, 경계 방지
        cx = max(0, min(cx, W - 1))
        cy = max(0, min(cy, H - 1))
        cw = min(cw, W - cx)
        ch = min(ch, H - cy)
        if cw <= 0 or ch <= 0:
            # 유효하지 않은 사각형: 원본 그대로 반환, 충돌 방지
            empty = torch.zeros((B_o, H, W), device=device, dtype=original_image.dtype)
            return (original_image, empty)

        # ---- patch resize를 래스터 크기에 엄격하게 정렬 ----
        patch = patch_image[0:1] if patch_image.shape[0] >= 1 else patch_image
        ph, pw = patch.shape[1], patch.shape[2]
        if ph != ch or pw != cw:
            patch_r = patch.permute(0, 3, 1, 2)  # (1, C, h, w)
            patch_r = F.interpolate(patch_r, size=(ch, cw), mode="bicubic", align_corners=False)
            patch_r = patch_r.permute(0, 2, 3, 1)  # (1, ch, cw, C)
        else:
            patch_r = patch

        # ---- 빌드 혼합 가중치 alpha (ch, cw) ----
        alpha = self._build_alpha(mask, ch, cw, feather, cx, cy, W, H, device)
        alpha = alpha.to(device).clamp(0.0, 1.0)

        # ---- 합성 ----
        out = original_image.clone()
        region = out[:, cy:cy + ch, cx:cx + cw, :]
        a = alpha.view(1, ch, cw, 1)  # B로 브로드캐스팅 및 C

        pc = patch_r.shape[3]
        if pc != C:
            mc = min(pc, C)
            region = patch_r[:, :, :, :mc] * a[..., :mc] + region[:, :, :, :mc] * (1.0 - a[..., :mc])
            out[:, cy:cy + ch, cx:cx + cw, :mc] = region
        else:
            region = patch_r * a + region * (1.0 - a)
            out[:, cy:cy + ch, cx:cx + cw, :] = region

        # 전체 크기 alpha 출력 (체인 두 단계 수정 용)）
        full_alpha = torch.zeros((B_o, H, W), device=device, dtype=original_image.dtype)
        full_alpha[:, cy:cy + ch, cx:cx + cw] = alpha

        return (out, full_alpha)

    def _build_alpha(self, mask, ch, cw, feather, cx, cy, W, H, device):
        f = max(int(feather), 0)

        # 기본 가중치
        if isinstance(mask, torch.Tensor):
            m = mask.float()
            if m.dim() == 3:
                m = m[0:1]  # (1, Hm, Wm)
            m = m.unsqueeze(0)  # (1, 1, Hm, Wm)
            m = F.interpolate(m, size=(ch, cw), mode="bilinear", align_corners=False)
            m = m.squeeze(0).squeeze(0)  # (ch, cw)
            if f > 0:
                sigma = max(f / 3.0, 0.5)
                m = self._gaussian_blur(m.unsqueeze(0).unsqueeze(0), sigma).squeeze(0).squeeze(0)
            base = m.clamp(0.0, 1.0)
        else:
            base = torch.ones((ch, cw), device=device, dtype=torch.float32)

        # 에지 가드: 원본 이미지 경계와 겹치는 가장자리는 펜할 없음 (이미지 경계가 흐릿해지는 것을 방지)）
        ramp = torch.ones((ch, cw), device=device, dtype=torch.float32)

        # 왼쪽 경계
        if cx != 0 and f > 0:
            L = min(f, cw)
            t = torch.linspace(0.0, 1.0, L, device=device)
            ramp[:, :L] = ramp[:, :L] * t.unsqueeze(0)
        # 우측 경계
        if cx + cw < W and f > 0:
            L = min(f, cw)
            t = torch.linspace(1.0, 0.0, L, device=device)
            ramp[:, cw - L:] = ramp[:, cw - L:] * t.unsqueeze(0)
        # 상단 경계
        if cy != 0 and f > 0:
            L = min(f, ch)
            t = torch.linspace(0.0, 1.0, L, device=device)
            ramp[:L, :] = ramp[:L, :] * t.unsqueeze(1)
        # 하단 경계
        if cy + ch < H and f > 0:
            L = min(f, ch)
            t = torch.linspace(1.0, 0.0, L, device=device)
            ramp[ch - L:, :] = ramp[ch - L:, :] * t.unsqueeze(1)

        return (base * ramp).clamp(0.0, 1.0)

    @staticmethod
    def _gaussian_blur(x, sigma):
        """x: (1, 1, H, W) -> 분리 가능한 가우시안 블러（reflect padding）"""
        ksize = int(sigma * 6.0) + 1
        if ksize % 2 == 0:
            ksize += 1
        if ksize <= 1:
            return x
        ax = torch.arange(-ksize // 2 + 1, ksize // 2 + 1, dtype=torch.float32, device=x.device)
        g = torch.exp(-(ax ** 2) / (2.0 * sigma ** 2))
        g = g / g.sum()
        kernel = (g[:, None] * g[None, :]).unsqueeze(0).unsqueeze(0)  # (1, 1, ks, ks)
        pad = ksize // 2
        x = F.pad(x, (pad, pad, pad, pad), mode="reflect")
        return F.conv2d(x, kernel, padding=0)


NODE_CLASS_MAPPINGS = {
    "BSK_CropMerge": BSK_CropMerge,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BSK_CropMerge": "BSK 자르기 반사 (무작위 병합)）",
}
