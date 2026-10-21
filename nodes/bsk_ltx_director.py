"""
bsk_LTXDirector —— ComfyUI-bsk_UI에서 WhatDreamsCost/Koolook의 LTX Director 기능을 재현하십시오.。

기능: 프롬프트 전달 시간 분할 프롬프트 + 이미지 키프레임 안내 + 오디오 합성, Wan 및 LTX 두 아키텍처 호환。

이 파일은 상류의 것을 병합했습니다 prompt_relay.py / patches.py / _relay_overrides.py / ltx_director.py，
bsk_UI 규격에 따라 클래식 INPUT_TYPES 스타일로 재작성합니다(필요없음) io.Schema / define_schema）。

참조 소스 (읽기 전용)）：
  custom_nodes/ComfyUI-Koolook/forks/whatdreamscost_koolook/versions/v1_3_9/

이 파일은 기반으로 함 WhatDreamsCost-ComfyUI
(https://github.com/WhatDreamsCost/WhatDreamsCost-ComfyUI, 저자: WhatDreamsCost) 수정됨，
이 프로젝트는 GNU General Public License v3로 게시되었습니다. 수정 날짜：2026-07-18。

저작권 정보：
- 원래 코드 Copyright WhatDreamsCost，Licensed under the GNU General Public License v3
  (https://www.gnu.org/licenses/gpl-3.0.html)
- 본 수정 버전은 2026년 灰暗x에 의해 ComfyUI-bsk_UI (AGPL-3.0)의 일부로 배포되었으며, 저작권이 있습니다.。
  전체 ComfyUI-bsk_UI 프로젝트는 AGPL-3.0로 배포됩니다; 본 파일은 GPL-3.0 호환 범위 내에 있습니다
  프로젝트 전체와 함께 AGPL-3.0로 배포。

본 프로그램은 자유 소프트웨어입니다. GNU 일반 공통 라이선스(제 3판)에 따라 재배포 및/또는 수정할 수 있습니다.。
이 프로그램은 따라"원래 모습"제공, 보장 없음。
"""

# ============================================================
# 첫 번째 부분: 가져오기
# ============================================================

import logging
import json
import base64
import io as _io
import math
import os
import re
import types
from typing import Optional

import numpy as np
import torch
import av
from PIL import Image

import folder_paths
import comfy.model_management
import comfy.ldm.modules.attention
import comfy.utils
import node_helpers

log = logging.getLogger(__name__)

# 사용자 정의 유형 GUIDE_DATA: INPUT_TYPES 스타일에서 입력/출력 유형 이름 문자열이 일치하면 연결. ComfyUI 핵심에서 영감。
# 직접 문자열 사용 "GUIDE_DATA"，custom_type 등록 불필요 (ComfyUI 타입 시스템은 이름으로 매칭)）。
GUIDE_DATA_TYPE = "GUIDE_DATA"


# ============================================================
# 두 번째 부분: Prompt Relay 알고리즘 핵심 (prompt_relay.py에서 이식)）
# ============================================================

def build_temporal_cost(q_token_idx, Lq, Lk, device, dtype, tokens_per_frame):
    """Gaussian penalty matrix [Lq, Lk] for video cross-attention (integer frame indexing)."""
    offset = torch.zeros(Lq, Lk, device=device, dtype=dtype)
    query_frames = torch.arange(Lq, device=device, dtype=torch.long) // tokens_per_frame

    for seg in q_token_idx:
        local = seg["local_token_idx"].to(device=device)
        d = (query_frames.float()[:, None] - seg["midpoint"]).abs()
        strength = seg.get("strength", 1.0)
        cost = strength * (torch.relu(d - seg["window"]) ** 2) / (2 * seg["sigma"] ** 2)
        offset[:, local] = cost.to(offset.dtype)

    return offset


def build_temporal_cost_scaled(q_token_idx, Lq, Lk, device, dtype, latent_frames):
    """Penalty matrix for queries that don't map to integer frames (e.g. LTXAV audio tokens)."""
    offset = torch.zeros(Lq, Lk, device=device, dtype=dtype)
    query_frames = torch.arange(Lq, device=device, dtype=torch.float32) * latent_frames / Lq

    for seg in q_token_idx:
        local = seg["local_token_idx"].to(device=device)
        d = (query_frames[:, None] - seg["midpoint"]).abs()
        sigma_a = seg.get("sigma_audio", seg["sigma"])
        window_a = seg.get("window_audio", seg["window"])
        strength_a = seg.get("strength_audio", 1.0)
        cost = strength_a * (torch.relu(d - window_a) ** 2) / (2 * sigma_a ** 2)
        offset[:, local] = cost.to(offset.dtype)

    return offset


def create_mask_fn(q_token_idx, fallback_tokens_per_frame, latent_frames):
    """Closure: mask_fn(q, k, transformer_options) -> additive mask or None."""
    cache = {}
    max_token_idx = max(int(seg["local_token_idx"].max().item()) for seg in q_token_idx) + 1

    def mask_fn(q, k, transformer_options):
        Lq, Lk = q.shape[1], k.shape[1]

        if Lq == Lk:
            return None

        # Only apply on conditional pass — not unconditional (negative prompt)
        cond_or_uncond = transformer_options.get("cond_or_uncond", [])
        if 1 in cond_or_uncond and 0 not in cond_or_uncond:
            return None

        grid_sizes = transformer_options.get("grid_sizes", None)
        video_tpf = int(grid_sizes[1]) * int(grid_sizes[2]) if grid_sizes is not None else fallback_tokens_per_frame
        video_lq = latent_frames * video_tpf

        # Skip cross-modal attention — text keys are padded to a fixed length ≥ max_token_idx and != video_lq
        if Lk == video_lq or Lk < max_token_idx:
            return None

        mode = "video" if Lq == video_lq else "scaled"

        key = (Lq, Lk, mode, q.device)
        if key not in cache:
            if mode == "video":
                cost = build_temporal_cost(q_token_idx, Lq, Lk, q.device, q.dtype, video_tpf)
            else:
                cost = build_temporal_cost_scaled(q_token_idx, Lq, Lk, q.device, q.dtype, latent_frames)
            log.info(
                "[PromptRelay] Built penalty matrix (%s): Lq=%d, Lk=%d, nonzero=%d/%d",
                mode, Lq, Lk, (cost > 0).sum().item(), cost.numel(),
            )
            cache[key] = -cost

        return cache[key].to(q.dtype)

    return mask_fn


def build_segments(token_ranges, segment_lengths, epsilon=1e-3, relay_options=None):
    """Per-segment metadata for the temporal penalty.

    relay_options (optional dict) overrides per-stream knobs:
        video_strength, video_window_scale,
        audio_epsilon, audio_strength, audio_window_scale
    Audio knobs only affect architectures whose cross-attention takes the scaled
    (non-integer-frame) path — currently LTX audio_attn2.

    sigma is computed per segment using the Prompt-Relay paper formula
    sigma = (L - w_eff) / (2 * sqrt(ln(1/epsilon))) so the penalty reaches the
    threshold epsilon at each segment boundary regardless of segment length.
    """
    sigma_fallback = 0.1448

    opts = relay_options or {}
    v_strength = opts.get("video_strength", 1.0)
    v_window_scale = opts.get("video_window_scale", 1.0)
    a_epsilon = opts.get("audio_epsilon")
    a_strength = opts.get("audio_strength", 1.0)
    a_window_scale = opts.get("audio_window_scale", 1.0)

    denom_v = 2.0 * math.sqrt(math.log(1.0 / epsilon)) if (0 < epsilon < 1) else None
    if a_epsilon is not None and 0 < a_epsilon < 1:
        denom_a = 2.0 * math.sqrt(math.log(1.0 / a_epsilon))
    else:
        denom_a = denom_v

    if relay_options:
        log.info(
            "[PromptRelay] Advanced options active — video: strength=%.3f window_scale=%.3f | "
            "audio: epsilon=%s strength=%.3f window_scale=%.3f",
            v_strength, v_window_scale,
            f"{a_epsilon:.4f}" if a_epsilon is not None else "inherit",
            a_strength, a_window_scale,
        )

    q_token_idx = []
    frame_cursor = 0

    for (tok_start, tok_end), L in zip(token_ranges, segment_lengths):
        if L <= 0:
            frame_cursor += L
            continue
        midpoint = (2 * frame_cursor + L) // 2
        base_window = max(L // 2 - 2, 0)
        eff_window_v = max(base_window * v_window_scale, 0.0)
        eff_window_a = max(base_window * a_window_scale, 0.0)
        sigma_v = (L - eff_window_v) / denom_v if (denom_v is not None and L > eff_window_v) else sigma_fallback
        sigma_a = (L - eff_window_a) / denom_a if (denom_a is not None and L > eff_window_a) else sigma_fallback
        log.info(
            "[PromptRelay] seg L=%d w_v=%.2f sigma_v=%.4f | w_a=%.2f sigma_a=%.4f (strength v=%.2f a=%.2f)",
            L, eff_window_v, sigma_v, eff_window_a, sigma_a, v_strength, a_strength,
        )
        q_token_idx.append({
            "local_token_idx": torch.arange(tok_start, tok_end),
            "midpoint": midpoint,
            "window": eff_window_v,
            "sigma": sigma_v,
            "strength": v_strength,
            "window_audio": eff_window_a,
            "sigma_audio": sigma_a,
            "strength_audio": a_strength,
        })
        frame_cursor += L

    return q_token_idx


def get_raw_tokenizer(clip):
    """Extract the raw SPiece/HF tokenizer from a ComfyUI CLIP object."""
    tokenizer_wrapper = clip.tokenizer
    for attr_name in dir(tokenizer_wrapper):
        if attr_name.startswith("_"):
            continue
        inner = getattr(tokenizer_wrapper, attr_name, None)
        if inner is not None and hasattr(inner, "tokenizer"):
            return inner.tokenizer

    raise RuntimeError(
        f"Could not find raw tokenizer on CLIP object. "
        f"Known attributes: {[a for a in dir(tokenizer_wrapper) if not a.startswith('_')]}"
    )


def map_token_indices(raw_tokenizer, global_prompt, local_prompts):
    """Tokenize global + space-prefixed locals; return (full_prompt, per-local token ranges).

    Uses incremental tokenization to avoid SentencePiece context-dependency issues.
    """
    prefixed_locals = [" " + lp for lp in local_prompts]
    full_prompt = global_prompt + "".join(prefixed_locals)
    has_eos = getattr(raw_tokenizer, "add_eos", False)
    eos_adj = 1 if has_eos else 0

    prev_len = len(raw_tokenizer(global_prompt)["input_ids"]) - eos_adj
    token_ranges = []
    built = global_prompt

    for plp in prefixed_locals:
        built += plp
        cur_len = len(raw_tokenizer(built)["input_ids"]) - eos_adj
        if cur_len <= prev_len:
            raise ValueError(f"Local prompt produced no tokens: '{plp.strip()}'")
        token_ranges.append((prev_len, cur_len))
        prev_len = cur_len

    return full_prompt, token_ranges


def distribute_segment_lengths(num_segments, latent_frames, specified_lengths=None):
    """Validate or auto-distribute segment frame counts, capped to fit within latent_frames."""
    if specified_lengths:
        if len(specified_lengths) != num_segments:
            raise ValueError(
                f"Number of segment_lengths ({len(specified_lengths)}) "
                f"must match number of local prompts ({num_segments})"
            )
        lengths = specified_lengths
    else:
        # ceil division — matches reference implementation
        step = -(-latent_frames // num_segments)
        lengths = [step] * num_segments

    effective = []
    cursor = 0
    for L in lengths:
        end = min(cursor + L, latent_frames)
        effective.append(max(end - cursor, 0))
        cursor = end
    return effective


# ---- parse_relay_overrides（_relay_overrides.py_에서 이식）----

# Allowlist of recognised override keys and their coercion type.
RELAY_OVERRIDE_KEYS: dict = {
    "video_strength": float,
    "video_window_scale": float,
    "audio_strength": float,
    "audio_window_scale": float,
    "audio_epsilon": float,
}


_PAIR_LINE_RE = re.compile(
    r'^\s*(?P<key>"?[A-Za-z_][A-Za-z0-9_]*"?)\s*(?::|=)\s*(?P<value>[^,#/]+?)\s*,?\s*$'
)


def _parse_loose_relay_overrides(raw: str) -> Optional[dict]:
    """Parse Comfy Text Multiline-friendly override snippets."""
    text = str(raw or "").strip()
    if not text:
        return None

    decoder = json.JSONDecoder()
    idx = 0
    objects = []
    try:
        while idx < len(text):
            while idx < len(text) and text[idx].isspace():
                idx += 1
            if idx >= len(text):
                break
            obj, next_idx = decoder.raw_decode(text, idx)
            if not isinstance(obj, dict):
                raise ValueError
            objects.append(obj)
            idx = next_idx
        if objects:
            merged = {}
            for obj in objects:
                merged.update(obj)
            return merged
    except (json.JSONDecodeError, ValueError):
        pass

    loose = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", "//")):
            continue
        if stripped in ("{", "}"):
            continue
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError:
                obj = None
            if isinstance(obj, dict):
                loose.update(obj)
                continue
        if "{" in stripped or "}" in stripped:
            return None
        match = _PAIR_LINE_RE.match(stripped.strip("{} \t"))
        if not match:
            return None
        key = match.group("key").strip().strip('"')
        value = match.group("value").strip().strip('"').strip("'")
        loose[key] = value

    return loose or None


def parse_relay_overrides(s: str) -> Optional[dict]:
    """Parse the relay_overrides multiline-string input.

    Returns a dict of validated {knob_name: float} overrides, or None when input
    is empty/whitespace (meaning "use upstream defaults"). Raises ValueError on
    non-empty input that doesn't parse.
    """
    if not s or not s.strip():
        return None

    try:
        opts = json.loads(s)
    except (json.JSONDecodeError, TypeError) as exc:
        opts = _parse_loose_relay_overrides(s)
        if opts is None:
            raise ValueError(
                "relay_overrides input is not valid JSON or a supported "
                f"key/value block ({type(exc).__name__}: {exc}). Empty "
                "the field to use upstream defaults, paste a JSON object "
                'like {"video_strength": 10.0}, or use '
                "video_strength: 10.0."
            ) from exc

    if not isinstance(opts, dict):
        raise ValueError(
            "relay_overrides must be a JSON object, got "
            f"{type(opts).__name__}. Empty the field to use upstream "
            'defaults, or paste a JSON object like '
            '{"video_strength": 10.0}.'
        )

    cleaned = {}
    unknown = []

    for key, value in opts.items():
        if key.startswith("_"):
            # `_`-prefixed keys are reserved as inline JSON comments — ignored at parse time.
            continue
        if key not in RELAY_OVERRIDE_KEYS:
            unknown.append(key)
            continue

        coerce = RELAY_OVERRIDE_KEYS[key]
        if isinstance(value, bool):
            raise ValueError(
                f"relay_overrides[{key!r}] must be a number, got "
                f"{value!r} (bool). Example: "
                f'{{"{key}": 10.0}}'
            )
        try:
            cleaned[key] = coerce(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"relay_overrides[{key!r}] must be a number, got "
                f"{value!r} ({type(value).__name__}). Example: "
                f'{{"{key}": 10.0}}'
            ) from exc

    if unknown:
        log.warning(
            "[PromptRelay] relay_overrides has unknown keys (typo?): %s. "
            "Recognised keys: %s. Use `_`-prefixed keys for inline "
            "comments.",
            sorted(unknown),
            sorted(RELAY_OVERRIDE_KEYS),
        )

    if not cleaned:
        return None

    log.info("[PromptRelay] Loaded relay_overrides from node input: %s", cleaned)
    return cleaned


# ============================================================
# 제3부: 모델 패치 (patches.py에서 이식)）
# ============================================================

def _masked_attention(q, k, v, heads, mask, transformer_options={}, **kwargs):
    # Bypass wrap_attn (sage/etc may ignore masks) by calling attention_pytorch directly.
    return comfy.ldm.modules.attention.attention_pytorch(
        q, k, v, heads, mask=mask,
        _inside_attn_wrapper=True,
        transformer_options=transformer_options,
        **kwargs,
    )


def _wan_t2v_forward(self, mask_fn, x, context, transformer_options={}, **kwargs):
    q = self.norm_q(self.q(x))
    k = self.norm_k(self.k(context))
    v = self.v(context)

    mask = mask_fn(q, k, transformer_options)
    if mask is not None:
        x = _masked_attention(q, k, v, heads=self.num_heads, mask=mask,
                              transformer_options=transformer_options)
    else:
        x = comfy.ldm.modules.attention.optimized_attention(
            q, k, v, heads=self.num_heads, transformer_options=transformer_options,
        )
    return self.o(x)


def _wan_i2v_forward(self, mask_fn, x, context, context_img_len, transformer_options={}, **kwargs):
    context_img = context[:, :context_img_len]
    context_text = context[:, context_img_len:]

    q = self.norm_q(self.q(x))

    k_img = self.norm_k_img(self.k_img(context_img))
    v_img = self.v_img(context_img)
    img_x = comfy.ldm.modules.attention.optimized_attention(
        q, k_img, v_img, heads=self.num_heads, transformer_options=transformer_options,
    )

    k = self.norm_k(self.k(context_text))
    v = self.v(context_text)

    mask = mask_fn(q, k, transformer_options)
    if mask is not None:
        x = _masked_attention(q, k, v, heads=self.num_heads, mask=mask,
                              transformer_options=transformer_options)
    else:
        x = comfy.ldm.modules.attention.optimized_attention(
            q, k, v, heads=self.num_heads, transformer_options=transformer_options,
        )

    return self.o(x + img_x)


def _ltx_forward(self, mask_fn, x, context=None, mask=None, pe=None, k_pe=None, transformer_options={}):
    from comfy.ldm.lightricks.model import apply_rotary_emb

    is_self_attn = context is None
    context = x if is_self_attn else context

    q = self.q_norm(self.to_q(x))
    k = self.k_norm(self.to_k(context))
    v = self.to_v(context)

    if pe is not None:
        q = apply_rotary_emb(q, pe)
        k = apply_rotary_emb(k, pe if k_pe is None else k_pe)

    if not is_self_attn:
        temporal_mask = mask_fn(q, k, transformer_options)
        if temporal_mask is not None:
            mask = temporal_mask if mask is None else mask + temporal_mask

    if mask is None:
        out = comfy.ldm.modules.attention.optimized_attention(
            q, k, v, self.heads, attn_precision=self.attn_precision,
            transformer_options=transformer_options,
        )
    else:
        out = _masked_attention(q, k, v, self.heads, mask=mask,
                                attn_precision=self.attn_precision,
                                transformer_options=transformer_options)

    if self.to_gate_logits is not None:
        gate_logits = self.to_gate_logits(x)
        b, t, _ = out.shape
        out = out.view(b, t, self.heads, self.dim_head)
        out = out * (2.0 * torch.sigmoid(gate_logits)).unsqueeze(-1)
        out = out.view(b, t, self.heads * self.dim_head)

    return self.to_out(out)


class _CrossAttnPatch:
    """Descriptor that binds (impl, mask_fn) as a method onto a cross-attn module."""

    def __init__(self, impl, mask_fn):
        self.impl = impl
        self.mask_fn = mask_fn

    def __get__(self, obj, objtype=None):
        impl, mask_fn = self.impl, self.mask_fn

        def wrapped(self_module, *args, **kwargs):
            return impl(self_module, mask_fn, *args, **kwargs)

        return types.MethodType(wrapped, obj)


def detect_model_type(model):
    """Return (arch, patch_size, temporal_stride) for latent geometry.

    temporal_stride is the VAE's pixel→latent temporal compression factor,
    used to convert user-facing pixel frame counts to latent frames.
    """
    diff_model = model.model.diffusion_model

    if hasattr(diff_model, "patch_size") and not hasattr(diff_model, "patchifier"):
        return "wan", tuple(diff_model.patch_size), 4

    if hasattr(diff_model, "patchifier"):
        return "ltx", (1, 1, 1), int(diff_model.vae_scale_factors[0])

    raise ValueError(
        f"Unsupported model type: {type(diff_model).__name__}. "
        f"Currently supports Wan and LTX models."
    )


def _check_unpatched(model_clone, key):
    if key in getattr(model_clone, "object_patches", {}):
        raise RuntimeError(
            f"PromptRelay: cross-attention forward at '{key}' is already patched by "
            "another node (e.g. KJNodes NAG). Stacking is not supported — remove the "
            "conflicting node."
        )


def apply_patches(model_clone, arch, mask_fn):
    diffusion_model = model_clone.get_model_object("diffusion_model")

    if arch == "wan":
        from comfy.ldm.wan.model import WanI2VCrossAttention
        for idx, block in enumerate(diffusion_model.blocks):
            key = f"diffusion_model.blocks.{idx}.cross_attn.forward"
            _check_unpatched(model_clone, key)
            cross_attn = block.cross_attn
            impl = _wan_i2v_forward if isinstance(cross_attn, WanI2VCrossAttention) else _wan_t2v_forward
            model_clone.add_object_patch(key, _CrossAttnPatch(impl, mask_fn).__get__(cross_attn, cross_attn.__class__))
        return

    if arch == "ltx":
        for idx, block in enumerate(diffusion_model.transformer_blocks):
            for attr in ("attn2", "audio_attn2"):
                module = getattr(block, attr, None)
                if module is None:
                    continue
                key = f"diffusion_model.transformer_blocks.{idx}.{attr}.forward"
                _check_unpatched(model_clone, key)
                model_clone.add_object_patch(key, _CrossAttnPatch(_ltx_forward, mask_fn).__get__(module, module.__class__))
        return

    raise ValueError(f"Unknown model arch: {arch}")


# ============================================================
# 제4 부분: 이미지 처리
# ============================================================

def _load_image_tensor(seg: dict) -> torch.Tensor:
    """Decode an image from the ComfyUI input folder (if imageFile provided) or fallback to base64
    to a ComfyUI-style image tensor of shape [1, H, W, 3], float32 in [0, 1]."""
    if seg.get("imageFile"):
        file_path = os.path.join(folder_paths.get_input_directory(), seg["imageFile"])
        if os.path.exists(file_path):
            img = Image.open(file_path).convert("RGB")
            arr = np.array(img, dtype=np.float32) / 255.0
            return torch.from_numpy(arr).unsqueeze(0)

    b64_str = seg.get("imageB64", "")
    if not b64_str or b64_str.startswith("/view?"):
        return torch.zeros((1, 512, 512, 3), dtype=torch.float32)

    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]

    try:
        img_bytes = base64.b64decode(b64_str)
        img = Image.open(_io.BytesIO(img_bytes)).convert("RGB")
        arr = np.array(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0)
    except Exception:
        return torch.zeros((1, 512, 512, 3), dtype=torch.float32)


def _resize_image(tensor: torch.Tensor, target_w: int, target_h: int, method: str, divisible_by: int) -> torch.Tensor:
    """Resize a [1, H, W, 3] float32 tensor to target dimensions using the given method,
    then snap the final dimensions to be divisible by `divisible_by`."""
    from PIL import Image as _PilImage

    def snap(val, div):
        return max(div, (val // div) * div)

    tw = snap(target_w, divisible_by)
    th = snap(target_h, divisible_by)

    img_np = (tensor[0].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    pil = _PilImage.fromarray(img_np)
    src_w, src_h = pil.size

    if method == "stretch to fit":
        resized = pil.resize((tw, th), _PilImage.LANCZOS)

    elif method == "maintain aspect ratio":
        ratio = min(tw / src_w, th / src_h)
        new_w = int(src_w * ratio)
        new_h = int(src_h * ratio)
        new_w = snap(new_w, divisible_by)
        new_h = snap(new_h, divisible_by)
        resized = pil.resize((new_w, new_h), _PilImage.LANCZOS)

    elif method == "pad":
        ratio = min(tw / src_w, th / src_h)
        new_w = snap(int(src_w * ratio), divisible_by)
        new_h = snap(int(src_h * ratio), divisible_by)
        inner = pil.resize((new_w, new_h), _PilImage.LANCZOS)
        resized = _PilImage.new("RGB", (tw, th), (0, 0, 0))
        resized.paste(inner, ((tw - new_w) // 2, (th - new_h) // 2))

    elif method == "crop":
        ratio = max(tw / src_w, th / src_h)
        new_w = int(src_w * ratio)
        new_h = int(src_h * ratio)
        inner = pil.resize((new_w, new_h), _PilImage.LANCZOS)
        left = (new_w - tw) // 2
        top = (new_h - th) // 2
        resized = inner.crop((left, top, left + tw, top + th))

    else:
        resized = pil.resize((tw, th), _PilImage.LANCZOS)

    arr = np.array(resized, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def _compress_image(tensor: torch.Tensor, crf: int) -> torch.Tensor:
    """Apply H.264 compression artefacts to a [1, H, W, 3] float32 tensor (ComfyUI image format).
    crf=0 means no compression. Uses PyAV to encode/decode a single frame in-memory."""
    if crf == 0:
        return tensor
    img = tensor[0]  # [H, W, 3]
    # Dimensions must be even for H.264
    h = (img.shape[0] // 2) * 2
    w = (img.shape[1] // 2) * 2
    img_np = (img[:h, :w] * 255.0).byte().cpu().numpy()  # uint8 [H, W, 3]

    try:
        buf = _io.BytesIO()
        container = av.open(buf, mode="w", format="mp4")
        stream = container.add_stream("libx264", rate=1)
        stream.width = w
        stream.height = h
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": str(crf), "preset": "ultrafast"}
        frame = av.VideoFrame.from_ndarray(img_np, format="rgb24")
        for pkt in stream.encode(frame):
            container.mux(pkt)
        for pkt in stream.encode(None):
            container.mux(pkt)
        container.close()

        buf.seek(0)
        container_r = av.open(buf, mode="r")
        decoded = None
        for frame_r in container_r.decode(video=0):
            decoded = frame_r.to_ndarray(format="rgb24")  # [H, W, 3]
            break
        container_r.close()

        if decoded is None:
            return tensor
        arr = torch.from_numpy(decoded.astype(np.float32) / 255.0).to(tensor.device, tensor.dtype)
        # Re-embed into original tensor shape (may have been cropped by even-rounding)
        out = tensor.clone()
        out[0, :h, :w] = arr
        return out
    except Exception as e:
        log.warning("[PromptRelay] img_compression encode/decode failed: %s", e)
        return tensor


# ============================================================
# 제5 부분: 오디오 합성
# ============================================================

DEFAULT_AUDIO_TRANSCRIPT_PROMPT_TEMPLATE = (
    'The character faces the camera and says "{text}". '
    "The mouth, jaw, and lips clearly form the spoken words."
)
DEFAULT_AUDIO_TRANSCRIPT_PAUSE_PROMPT = (
    "The character pauses between spoken words. "
    "The mouth closes naturally while the face stays alive and attentive."
)


def _frame_from_seconds(value: float, frame_rate: float) -> int:
    return max(0, int(round(float(value) * float(frame_rate))))


def _first_timeline_image_file(timeline: dict) -> str:
    for seg in timeline.get("segments", []):
        image_file = seg.get("imageFile")
        if image_file:
            return image_file
    return ""


def _build_audio_transcript_spans(
    phrases,
    *,
    duration_frames,
    frame_rate,
    prompt_template=DEFAULT_AUDIO_TRANSCRIPT_PROMPT_TEMPLATE,
    pause_prompt=DEFAULT_AUDIO_TRANSCRIPT_PAUSE_PROMPT,
    pause_threshold_seconds=0.2,
):
    clean_phrases = []
    for phrase in phrases:
        text = str(phrase.get("text", "")).strip()
        if not text:
            continue
        try:
            start = float(phrase.get("start", 0.0))
            end = float(phrase.get("end", start))
        except (TypeError, ValueError):
            continue
        clean_phrases.append({"start": start, "end": max(start, end), "text": text})

    clean_phrases.sort(key=lambda item: item["start"])

    spans = []
    cursor = 0
    pause_threshold = _frame_from_seconds(pause_threshold_seconds, frame_rate)
    for phrase in clean_phrases:
        start = min(duration_frames, _frame_from_seconds(phrase["start"], frame_rate))
        end = min(duration_frames, max(start + 1, _frame_from_seconds(phrase["end"], frame_rate)))
        if start >= duration_frames:
            continue
        if start - cursor >= pause_threshold:
            spans.append({"start": cursor, "length": start - cursor, "prompt": pause_prompt})
        elif start > cursor and spans:
            spans[-1]["length"] += start - cursor
        spans.append(
            {
                "start": start,
                "length": max(1, end - start),
                "prompt": prompt_template.format(text=phrase["text"]),
            }
        )
        cursor = end

    if duration_frames > cursor:
        spans.append({"start": cursor, "length": duration_frames - cursor, "prompt": pause_prompt})

    return [span for span in spans if int(span["length"]) > 0]


def _apply_audio_transcript_json(
    timeline_data: str,
    local_prompts: str,
    segment_lengths: str,
    audio_transcript_json: str,
    *,
    duration_frames: int,
    frame_rate: float,
):
    if not str(audio_transcript_json or "").strip():
        return timeline_data, local_prompts, segment_lengths

    try:
        transcript = json.loads(audio_transcript_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"audio_transcript_json is not valid JSON: {exc}") from exc

    phrases = transcript.get("phrases")
    if not isinstance(phrases, list):
        raise ValueError("audio_transcript_json must contain a phrases list.")
    if not phrases:
        raise ValueError("audio_transcript_json.phrases must not be empty.")

    timeline = json.loads(timeline_data) if str(timeline_data or "").strip() else {}
    if not isinstance(timeline, dict):
        timeline = {}

    spans = _build_audio_transcript_spans(
        phrases,
        duration_frames=int(duration_frames),
        frame_rate=float(frame_rate),
    )
    image_file = _first_timeline_image_file(timeline)
    speech_segments = [
        {
            "id": f"speech_{idx + 1:03d}",
            "type": "image",
            "start": int(span["start"]),
            "length": int(span["length"]),
            "prompt": span["prompt"],
            **({"imageFile": image_file} if image_file else {}),
        }
        for idx, span in enumerate(spans)
    ]
    if not timeline.get("segments"):
        timeline["segments"] = speech_segments
    else:
        log.warning(
            "[AudioTranscript] Preserving existing timeline_data segments; "
            "audio_transcript_json updated Prompt Relay local_prompts and "
            "segment_lengths only."
        )
    timeline.setdefault("audioSegments", [])

    return (
        json.dumps(timeline, ensure_ascii=False, separators=(",", ":")),
        " | ".join(str(span["prompt"]) for span in spans),
        ",".join(str(int(span["length"])) for span in spans),
    )


def _build_combined_audio(timeline_data_str: str, duration_frames: int, frame_rate: float) -> dict:
    """Parses timeline JSON, loads/trims audio directly from memory using PyAV,
    and aligns to a global timeline yielding ComfyUI's format.
    Output length explicitly mimics the timeline's duration_frames length."""
    target_sr = 44100
    total_samples = max(1, int(math.ceil(duration_frames / frame_rate * target_sr)))
    empty_audio = {"waveform": torch.zeros((1, 2, total_samples), dtype=torch.float32), "sample_rate": target_sr}

    if not timeline_data_str:
        return empty_audio

    try:
        data = json.loads(timeline_data_str)
        audio_segs = data.get("audioSegments", [])
    except Exception:
        return empty_audio

    if not audio_segs:
        return empty_audio

    out_waveform = torch.zeros((2, total_samples), dtype=torch.float32)

    for seg in audio_segs:
        buffer = None
        if seg.get("audioFile"):
            file_path = os.path.join(folder_paths.get_input_directory(), seg["audioFile"])
            if os.path.exists(file_path):
                with open(file_path, "rb") as f:
                    buffer = _io.BytesIO(f.read())

        if not buffer and seg.get("audioB64"):
            b64 = seg.get("audioB64")
            if "," in b64:
                b64 = b64.split(",", 1)[1]
            try:
                audio_bytes = base64.b64decode(b64)
                buffer = _io.BytesIO(audio_bytes)
            except Exception:
                pass

        if not buffer:
            continue

        try:
            clip_frames = []

            # Use PyAV to decode directly from memory buffer
            with av.open(buffer) as container:
                stream = container.streams.audio[0]

                # Setup resampler to ensure output is 44.1kHz, Stereo, Float32 Planar
                resampler = av.AudioResampler(
                    format='fltp',
                    layout='stereo',
                    rate=target_sr,
                )

                for frame in container.decode(stream):
                    for resampled_frame in resampler.resample(frame):
                        # to_ndarray() on fltp gives shape (channels, samples)
                        arr = resampled_frame.to_ndarray()
                        clip_frames.append(torch.from_numpy(arr))

                # Flush the resampler to get any remaining samples
                for resampled_frame in resampler.resample(None):
                    arr = resampled_frame.to_ndarray()
                    clip_frames.append(torch.from_numpy(arr))

            if not clip_frames:
                continue

            # Concatenate all frame blocks along the samples dimension (dim 1)
            waveform = torch.cat(clip_frames, dim=1)  # Shape: [2, total_clip_samples]

            # Calculate interactive trim boundaries
            trim_start_frames = float(seg.get("trimStart", 0))
            length_frames = float(seg.get("length", 1))
            start_frames = float(seg.get("start", 0))

            start_sample_src = int(trim_start_frames / frame_rate * target_sr)
            length_samples = int(length_frames / frame_rate * target_sr)
            end_sample_src = start_sample_src + length_samples

            if start_sample_src < 0:
                start_sample_src = 0
            if end_sample_src > waveform.shape[1]:
                end_sample_src = waveform.shape[1]

            actual_length = end_sample_src - start_sample_src
            if actual_length <= 0:
                continue

            # Extract the correct segment of the audio
            clip_waveform = waveform[:, start_sample_src:end_sample_src]

            # Position onto the timeline
            start_sample_dst = int(start_frames / frame_rate * target_sr)

            if start_sample_dst >= out_waveform.shape[1]:
                continue

            end_sample_dst = start_sample_dst + actual_length

            # Clip any trailing overflow so we don't index past the timeline bounds
            if end_sample_dst > out_waveform.shape[1]:
                actual_length = out_waveform.shape[1] - start_sample_dst
                clip_waveform = clip_waveform[:, :actual_length]
                end_sample_dst = start_sample_dst + actual_length

            if actual_length <= 0:
                continue

            # Additive composite (allows clips overlapping to sum together naturally)
            out_waveform[:, start_sample_dst:end_sample_dst] += clip_waveform

        except Exception as e:
            log.warning("[PromptRelay] Audio process error for segment %s: %s", seg.get("fileName"), e)
            continue

    return {"waveform": out_waveform.unsqueeze(0), "sample_rate": target_sr}


# ============================================================
# 제6장: 잠재 공간 도구
# ============================================================

def _convert_to_latent_lengths(pixel_lengths, temporal_stride, latent_frames):
    """Convert pixel-space segment lengths to integer latent-space lengths using the
    largest-remainder method. Targets the full `latent_frames` when the pixel sum looks
    like full coverage (within one stride of latent_frames * stride). Otherwise targets
    round(total_pixel / temporal_stride) so partial-coverage timelines stay partial.
    """
    if not pixel_lengths:
        return []
    total_pixel = sum(pixel_lengths)
    if total_pixel <= 0:
        return [1] * len(pixel_lengths)

    naive_total = max(1, round(total_pixel / temporal_stride))
    target_total = min(latent_frames, naive_total)
    # Within one frame of full → user clearly intended full coverage; pin to latent_frames.
    if target_total >= latent_frames - 1:
        target_total = latent_frames

    exact = [p * target_total / total_pixel for p in pixel_lengths]
    result = [int(e) for e in exact]
    diff = target_total - sum(result)
    if diff > 0:
        order = sorted(range(len(exact)), key=lambda i: -(exact[i] - int(exact[i])))
        for k in range(diff):
            result[order[k % len(order)]] += 1

    # Ensure every segment has ≥ 1 latent frame (steal from the largest if needed).
    for i in range(len(result)):
        if result[i] < 1:
            max_idx = max(range(len(result)), key=lambda j: result[j])
            if result[max_idx] > 1:
                result[max_idx] -= 1
                result[i] = 1

    return result


# ============================================================
# Part 7: Encoding Main Flow
# ============================================================

def _encode_relay(model, clip, latent, global_prompt, local_prompts, segment_lengths, epsilon, relay_overrides=""):
    for name, val in (("global_prompt", global_prompt),
                      ("local_prompts", local_prompts),
                      ("segment_lengths", segment_lengths)):
        if val is None:
            raise ValueError(
                f"PromptRelay: '{name}' arrived as None. "
                "Likely causes: a stale workflow JSON saved with null, the timeline "
                "editor's web extension failing to load, or an upstream node returning None. "
                "Set the field to an empty string or fix the upstream connection."
            )

    # Split prompts but do NOT filter out empty ones yet, so we can detect them
    locals_list = [p.strip() for p in local_prompts.split("|")]

    # Check if any specific segment is empty
    for p in locals_list:
        if not p:
            raise ValueError("There is a segment on the timeline missing a prompt!")

    if not locals_list or (len(locals_list) == 1 and not locals_list[0]):
        raise ValueError("At least one local prompt is required.")

    arch, patch_size, temporal_stride = detect_model_type(model)

    samples = latent["samples"]
    latent_frames = samples.shape[2]
    if samples.shape[3] <= 0 or samples.shape[4] <= 0:
        raise ValueError("Prompt Relay: latent spatial dimensions are zero or invalid.")
    tokens_per_frame = (samples.shape[3] // patch_size[1]) * (samples.shape[4] // patch_size[2])

    parsed_lengths = None
    if segment_lengths.strip():
        pixel_lengths = [int(float(x.strip())) for x in segment_lengths.split(",") if x.strip()]
        parsed_lengths = _convert_to_latent_lengths(pixel_lengths, temporal_stride, latent_frames)

    raw_tokenizer = get_raw_tokenizer(clip)
    full_prompt, token_ranges = map_token_indices(raw_tokenizer, global_prompt, locals_list)

    log.info("[PromptRelay] Global: tokens [0:%d] (%d tokens)", token_ranges[0][0], token_ranges[0][0])
    for i, (s, e) in enumerate(token_ranges):
        log.info("[PromptRelay] Segment %d: tokens [%d:%d] (%d tokens)", i, s, e, e - s)

    conditioning = clip.encode_from_tokens_scheduled(clip.tokenize(full_prompt))

    effective_lengths = distribute_segment_lengths(len(locals_list), latent_frames, parsed_lengths)

    log.info(
        "[PromptRelay] Latent: %d frames, %d tokens/frame, segments: %s",
        latent_frames, tokens_per_frame, effective_lengths,
    )

    q_token_idx = build_segments(token_ranges, effective_lengths, epsilon, parse_relay_overrides(relay_overrides))
    mask_fn = create_mask_fn(q_token_idx, tokens_per_frame, latent_frames)

    patched = model.clone()
    apply_patches(patched, arch, mask_fn)

    return patched, conditioning


# ============================================================
# 제8 부분: 노드 정의
# ============================================================

class bsk_LTXDirector:
    """WYSIWYG timeline variant — segments and lengths come from a visual editor in the node UI.

    bsk_UI 패널에서 타임라인 카드판으로 세그먼트 프롬프트/이미지 가이드/오디오 편집, 백엔드 실행
    Prompt Relay 인코딩 + 이미지 가이드 데이터 구축 + 오디오 합성. Wan 및 LTX 두 아키텍처 호환。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "확산 모델"}),
                "clip": ("CLIP", {"tooltip": "텍스트 인코더"}),
                "global_prompt": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "모든 파일이 이 필드의 옵션에 나타납니다(추가 항목 허용, 예시 포함)）。",
                }),
                "duration_frames": ("INT", {
                    "default": 120, "min": 1, "max": 10000, "step": 1,
                    "tooltip": "시간 축 총 프레임 수 (픽셀 공간), 편집기 시각적 눈금에만 사용。",
                }),
                "duration_seconds": ("FLOAT", {
                    "default": 5.0, "min": 0.1, "max": 1000.0, "step": 0.01,
                    "tooltip": "총 시간 (초) - 프레임과 동기화）。",
                }),
                "timeline_data": ("STRING", {
                    "default": "",
                    "tooltip": "타임라인 JSON (편집기 자동 유지, 수동으로 채우지 마세요).）。",
                }),
                "local_prompts": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "세그먼트 프롬프트, 사용 | 분리 (편집기 자동 유지）。",
                }),
                "segment_lengths": ("STRING", {
                    "default": "",
                    "tooltip": "세그먼트 길이, 쉼표로 구분된 픽셀 프레임 수 (에디터 자동 유지)）。",
                }),
                "epsilon": ("FLOAT", {
                    "default": 0.001, "min": 0.0001, "max": 0.99, "step": 0.0001,
                    "tooltip": "경계선 부드러움：0.001=하드 컷 (논문 기본)），0.5+=그라데이션。",
                }),
                "guide_strength": ("STRING", {
                    "default": "",
                    "tooltip": "이미지 가이드 강도, 쉼표로 구분 (에디터 자동 유지)）。",
                }),
            },
            "optional": {
                "frame_rate": ("FLOAT", {
                    "default": 24.0, "min": 1.0, "max": 240.0, "step": 1.0,
                    "tooltip": "프레임 속도, 시간 표시 및 오디오 정렬에 영향을 미칩니다。",
                }),
                "display_mode": (["frames", "seconds"], {
                    "default": "seconds",
                    "tooltip": "눈금자 표시 모드: 프레임 또는 초. 내부 저장 값은 항상 픽셀 프레임。",
                }),
                "custom_width": ("INT", {
                    "default": 0, "min": 0, "max": 8192, "step": 1,
                    "tooltip": "이미지 대상 너비，0=원래 높이 사용。",
                }),
                "custom_height": ("INT", {
                    "default": 0, "min": 0, "max": 8192, "step": 1,
                    "tooltip": "이미지 대상 높이，0=사용률 모니터링 클래스。",
                }),
                "resize_method": (["maintain aspect ratio", "stretch to fit", "pad", "crop"], {
                    "default": "maintain aspect ratio",
                    "tooltip": "이미지 확대 방식。",
                }),
                "divisible_by": ("INT", {
                    "default": 32, "min": 1, "max": 256, "step": 1,
                    "tooltip": "크기 정렬 인자, LTX 기본값 32。",
                }),
                "img_compression": ("INT", {
                    "default": 0, "min": 0, "max": 100, "step": 1,
                    "tooltip": "H.264 CRF 압축，0=현재 환경에 맞지 않으면 건너듸합니다。",
                }),
                "use_custom_audio": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "True=타임라인 오디오 생성 사용 audio_latent；False=빈 오디오 생성 latent。",
                }),
                "audio_vae": ("VAE", {"tooltip": "오디오 VAE(audio_latent 생성 시 필요）"}),
                "optional_latent": ("LATENT", {"tooltip": "외부로 latent 전달, 자동으로 생성된 빈 값 덮어쓰기 LTXV latent。"}),
                "relay_overrides": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "JSON dict，고급 relay 파라미터 덮어쓰기. 지원되는 키: video_strength, video_window_scale, audio_strength, audio_window_scale, audio_epsilon. 예：{\"video_strength\": 10.0}",
                }),
                "audio_transcript_json": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "음성 녹취 JSON, local_prompts/segment_lengths/timeline_data를 자동으로 구성합니다. 예시：{\"phrases\":[{\"start\":0,\"end\":0.5,\"text\":\"What?\"}]}",
                }),
            },
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "LATENT", "LATENT", GUIDE_DATA_TYPE, "FLOAT", "AUDIO")
    RETURN_NAMES = ("model", "positive", "video_latent", "audio_latent", "guide_data", "frame_rate", "combined_audio")
    FUNCTION = "execute"
    CATEGORY = "bsk_UI/LTX"
    DESCRIPTION = (
        "Prompt Relay 시간 분할 프롬프트 + 이미지 키프레임 안내 + 오디오 합성。"
        "bsk_UI 패널에서 시간축 카드 편집을 지원하며, Wan 및 LTX。"
    )

    def execute(self, model, clip, global_prompt, duration_frames, duration_seconds,
                timeline_data, local_prompts, segment_lengths, guide_strength="", epsilon=1e-3,
                frame_rate=24.0, display_mode="seconds",
                custom_width=0, custom_height=0, resize_method="maintain aspect ratio",
                divisible_by=32, img_compression=0, audio_vae=None, optional_latent=None,
                use_custom_audio=False, relay_overrides="", audio_transcript_json=""):

        # 1. [[선택 사항] audio_transcript_json 자동 채우기
        timeline_data, local_prompts, segment_lengths = _apply_audio_transcript_json(
            timeline_data,
            local_prompts,
            segment_lengths,
            audio_transcript_json,
            duration_frames=int(duration_frames),
            frame_rate=float(frame_rate),
        )

        # 2. guide_data 생성 (latent보다 먼저, 파생 출력 크기)）
        guide_data = {"images": [], "insert_frames": [], "strengths": [], "frame_rate": float(frame_rate)}
        derived_w, derived_h = int(custom_width), int(custom_height)
        try:
            tdata = json.loads(timeline_data) if timeline_data else {}
            img_segs = [
                s for s in tdata.get("segments", [])
                if s.get("type", "image") == "image"
                and (s.get("imageFile") or s.get("imageB64"))
                and int(s.get("start", 0)) < duration_frames  # exclude segments fully outside duration
            ]
            img_segs.sort(key=lambda s: s["start"])

            strengths = []
            if guide_strength.strip():
                strengths = [float(x.strip()) for x in guide_strength.split(",") if x.strip()]

            for idx, seg in enumerate(img_segs):
                tensor = _load_image_tensor(seg)

                # Apply resize
                src_h, src_w = tensor.shape[1], tensor.shape[2]

                def snap(val, div):
                    return max(div, (val // div) * div)

                if custom_width > 0 and custom_height > 0:
                    # Both dimensions set — apply selected resize_method (pad, crop, stretch, maintain AR)
                    tensor = _resize_image(tensor, custom_width, custom_height, resize_method, divisible_by)
                elif custom_width > 0:
                    # Width only — scale height from AR, snap both, then resize to exact dimensions
                    tgt_w = snap(custom_width, divisible_by)
                    tgt_h = snap(int(src_h * tgt_w / src_w), divisible_by)
                    tensor = _resize_image(tensor, tgt_w, tgt_h, "stretch to fit", divisible_by)
                elif custom_height > 0:
                    # Height only — scale width from AR, snap both, then resize to exact dimensions
                    tgt_h = snap(custom_height, divisible_by)
                    tgt_w = snap(int(src_w * tgt_h / src_h), divisible_by)
                    tensor = _resize_image(tensor, tgt_w, tgt_h, "stretch to fit", divisible_by)
                else:
                    # Both zero — keep original dimensions, just snap to divisible_by
                    tensor = _resize_image(tensor, src_w, src_h, "maintain aspect ratio", divisible_by)

                # Apply compression
                if img_compression > 0:
                    tensor = _compress_image(tensor, img_compression)

                # Record dimensions of the first processed image for latent generation
                if idx == 0:
                    derived_h = tensor.shape[1]
                    derived_w = tensor.shape[2]

                strength = strengths[idx] if idx < len(strengths) else 1.0
                guide_data["images"].append(tensor)
                guide_data["insert_frames"].append(int(seg["start"]))
                guide_data["strengths"].append(float(strength))

            # If no images were loaded from the timeline, create a dummy image at strength 0
            # to prevent artifacts in text-to-video mode.
            if not guide_data["images"]:
                w = derived_w if derived_w > 0 else 768
                h = derived_h if derived_h > 0 else 512
                w = (w // 32) * 32
                h = (h // 32) * 32

                dummy_image = torch.zeros((1, h, w, 3), dtype=torch.float32)
                guide_data["images"].append(dummy_image)
                guide_data["insert_frames"].append(0)
                guide_data["strengths"].append(0.0)

                derived_w = w
                derived_h = h
        except Exception as e:
            log.warning("[PromptRelay] Could not build guide_data: %s", e)

        # 3. video_latent을 생성합니다 (외부 latent가 없는 경우 자동으로 빈 것으로 생성). LTXV latent）
        # LTXV includes an extra terminal pixel frame. Prompt Relay segment
        # lengths may sum to `duration_frames`; `_convert_to_latent_lengths`
        # clamps near-full coverage back to the full latent-frame count.
        ltxv_length = int(duration_frames) + 1
        if optional_latent is None:
            latent_w = max(32, (derived_w // 32) * 32)
            latent_h = max(32, (derived_h // 32) * 32)
            # LTXV temporal: ((length - 1) // 8) + 1 latent frames; invert to get pixel frames -> length
            latent_t = ((ltxv_length - 1) // 8) + 1
            samples = torch.zeros(
                [1, 128, latent_t, latent_h // 32, latent_w // 32],
                device=comfy.model_management.intermediate_device(),
            )
            latent = {"samples": samples}
            log.info(
                "[PromptRelay] Auto-generated LTXV latent: %dx%d, %d pixel frames (%d latent frames)",
                latent_w, latent_h, ltxv_length, latent_t,
            )
        else:
            latent = optional_latent

        # 4. Prompt Relay 인코딩
        patched, conditioning = _encode_relay(
            model, clip, latent, global_prompt, local_prompts, segment_lengths, epsilon,
            relay_overrides=relay_overrides,
        )

        # 5. 오디오 합성
        audio_out = _build_combined_audio(timeline_data, ltxv_length, float(frame_rate))

        # 6. 생성 audio_latent
        audio_latent = {}

        if audio_vae is not None:
            # Helper to generate empty latent
            def get_empty_latent():
                # Support both raw AudioVAE objects and ComfyUI VAE wrappers.
                inner = getattr(audio_vae, "first_stage_model", audio_vae)
                z_channels = audio_vae.latent_channels
                audio_freq = inner.latent_frequency_bins
                num_audio_latents = inner.num_of_latents_from_frames(ltxv_length, float(frame_rate))
                audio_latents = torch.zeros(
                    (1, z_channels, num_audio_latents, audio_freq),
                    device=comfy.model_management.intermediate_device(),
                )
                return {"samples": audio_latents, "type": "audio"}

            if use_custom_audio:
                try:
                    if audio_out is not None:
                        # 1. Encode audio waveform into latent space
                        waveform = audio_out["waveform"]
                        if waveform.ndim == 2:
                            waveform = waveform.unsqueeze(0)
                        if waveform.ndim != 3:
                            raise ValueError(
                                f"Expected custom audio waveform with 2 or 3 dims, got shape {tuple(waveform.shape)}"
                            )

                        # Wrapped ComfyUI VAE expects (batch, samples, channels);
                        # raw AudioVAE expects a dict with waveform in (batch, channels, samples).
                        if hasattr(audio_vae, "first_stage_model"):
                            latent_samples = audio_vae.encode(waveform.movedim(1, -1))
                        else:
                            latent_samples = audio_vae.encode({
                                "waveform": waveform,
                                "sample_rate": audio_out["sample_rate"],
                            })

                        if latent_samples.numel() == 0:
                            raise ValueError("Encoded audio latent is empty (0 elements).")

                        # 2. Create solid mask with value 0.0 (0 means keep/use conditioning, 1 means generate noise)
                        mask = torch.full(
                            (1, latent_samples.shape[-2], latent_samples.shape[-1]),
                            0.0,
                            dtype=torch.float32,
                            device=comfy.model_management.intermediate_device()
                        )

                        # 3. Set Latent Noise Mask
                        audio_latent = {
                            "samples": latent_samples,
                            "type": "audio",
                            "noise_mask": mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1]))
                        }
                        log.info("[PromptRelay] Generated custom audio latent with noise mask (value=0.0).")
                    else:
                        raise ValueError("No audio waveform to encode.")
                except Exception as e:
                    log.error("[PromptRelay] Failed to generate custom audio latent: %s", e)
                    raise e
            else:
                # Generate empty latent
                try:
                    audio_latent = get_empty_latent()
                    log.info("[PromptRelay] Auto-generated empty audio latent.")
                except Exception as e:
                    log.error("[PromptRelay] Could not generate empty audio latent: %s", e)
                    raise e

        # 7. 반환
        return (patched, conditioning, latent, audio_latent, guide_data, float(frame_rate), audio_out)


NODE_CLASS_MAPPINGS = {
    "bsk_LTXDirector": bsk_LTXDirector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "bsk_LTXDirector": "BSK LTX Director",
}
