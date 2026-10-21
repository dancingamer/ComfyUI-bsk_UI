#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 灰暗x
# SPDX-License-Identifier: AGPL-3.0-or-later
"""정규 표현 그래프 생성（anima）— 모델을 한 번에 로드하고, 각 캡션에 대해 대응되는 그림을 한 장씩 생성합니다.。

training_api.start_gen_reg를 호출하여: 전달 --config <json>。
설정 필드：
  dit, vae, text_encoder, attn_mode, precision,
  width, height, steps, cfg, sampler_name, scheduler, negative_prompt,
  cancel_file,
  tasks: [{img, out_path, out_name, target, prefix, prompt, seed}]

진행률은 stdout을 통해 반환 (training_api 파싱)）：
  [REG_SUBDIR] <name> / [REG_SKIP] <img> / [REG_GEN] <img> -> <out>
  [REG_DONE] <done>/<total> / [REG_CANCEL] / [REG_ERROR] <msg>

의존성: anima_train (anima_minimal_inference.py와 동일한 출처의 Anima 프레임워크 모듈)）。
모델은 한 번만 로드하고, 모든 이미지를 출력하여 한 장씩 반복적으로 로드하지 않습니다.。
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import torch

_THIS = Path(__file__).resolve().parent
if str(_THIS) not in sys.path:
    sys.path.insert(0, str(_THIS))

try:
    import anima_train as _T
except Exception as _e:  # noqa: BLE001
    sys.stderr.write(f"[REG_ERROR] 불가능한 가져오기 anima_train: {_e}\n")
    sys.exit(2)


def _canceled(cancel_file: str) -> bool:
    return bool(cancel_file) and os.path.exists(cancel_file)


def _sample(model, vae_m, qwen_model, qwen_tok, t5_tok, *, prompt,
           height, width, steps, cfg_scale, negative_prompt,
           sampler_name, scheduler, device, dtype):
    """호출 샘플이미지, sampler_name/scheduler 및와 호환 sampler= 두 가지 서명。"""
    try:
        return _T.sample_image(
            model, vae_m, qwen_model, qwen_tok, t5_tok,
            prompt=prompt, height=height, width=width,
            steps=steps, cfg_scale=cfg_scale,
            negative_prompt=negative_prompt or None,
            sampler_name=sampler_name, scheduler=scheduler,
            device=device, dtype=dtype,
        )
    except TypeError:
        # sample_image의 기본 사용을 지원합니다 sampler= sampler_name/scheduler 변형 아님
        return _T.sample_image(
            model, vae_m, qwen_model, qwen_tok, t5_tok,
            prompt=prompt, height=height, width=width,
            steps=steps, cfg_scale=cfg_scale,
            negative_prompt=negative_prompt or None,
            sampler=sampler_name, device=device, dtype=dtype,
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as fh:
        cfg = json.load(fh)

    dit = cfg["dit"]
    vae = cfg["vae"]
    text_encoder = cfg["text_encoder"]
    attn_mode = cfg.get("attn_mode", "torch")
    precision = cfg.get("precision", "fp16")
    width = int(cfg.get("width", 1024))
    height = int(cfg.get("height", 1024))
    steps = int(cfg.get("steps", 25))
    cfg_scale = float(cfg.get("cfg", 7.5))
    sampler_name = cfg.get("sampler_name", "er_sde")
    scheduler = cfg.get("scheduler", "simple")
    negative_prompt = cfg.get("negative_prompt", "")
    cancel_file = cfg.get("cancel_file", "")
    tasks = cfg.get("tasks") or []

    use_flash = attn_mode in ("flash", "flash_attn")
    use_xformers = attn_mode == "xformers"
    dtype = (
        torch.bfloat16 if precision == "bf16"
        else torch.float16 if precision == "fp16"
        else torch.float32
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    repo_root = _T.find_diffusion_pipe_root()
    bases = [Path.cwd(), _THIS, repo_root]
    dit = _T.resolve_path_best_effort(dit, bases)
    vae = _T.resolve_path_best_effort(vae, bases)
    text_encoder = _T.resolve_path_best_effort(text_encoder, bases)

    sys.stderr.write("DiT 모델 로드...\n")
    model = _T.load_anima_model(dit, device, dtype, repo_root, flash_attn=use_flash)
    if use_xformers:
        _T.enable_xformers(model)
    sys.stderr.write("ロード VAE...\n")
    vae_m = _T.load_vae(vae, device, dtype, repo_root)
    sys.stderr.write("텍스트 인코더 로딩...\n")
    qwen_model, qwen_tok, t5_tok = _T.load_text_encoders(
        text_encoder, None, device, dtype,
    )
    model.eval()

    total = len(tasks)
    done = 0
    for idx, task in enumerate(tasks):
        if _canceled(cancel_file):
            print("[REG_CANCEL]")
            break
        img = task.get("img", "")
        out_path = Path(task["out_path"])
        prompt = task.get("prompt", "")
        seed = int(task.get("seed", 0))
        target = task.get("target", "")
        if target:
            print(f"[REG_SUBDIR] {Path(target).name}")
        if not prompt:
            print(f"[REG_SKIP] {img} 유효하지 않음 caption")
            continue
        torch.manual_seed(seed)
        random.seed(seed)
        print(f"[REG_GEN] {img} -> {out_path.name} (seed={seed})")
        try:
            image = _sample(
                model, vae_m, qwen_model, qwen_tok, t5_tok,
                prompt=prompt, height=height, width=width,
                steps=steps, cfg_scale=cfg_scale,
                negative_prompt=negative_prompt,
                sampler_name=sampler_name, scheduler=scheduler,
                device=device, dtype=dtype,
            )
            out_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(out_path)
            cap_path = out_path.with_suffix(".txt")
            cap_path.write_text(prompt, encoding="utf-8")
            done += 1
            print(f"[REG_DONE] {done}/{total}")
        except Exception as e:  # noqa: BLE001
            print(f"[REG_ERROR] 생성 실패 {img}: {e}")
    print(f"[REG_DONE] {done}/{total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
