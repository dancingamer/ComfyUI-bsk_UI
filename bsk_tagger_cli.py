#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 灰暗x
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
BSK Tagger CLI —— BSK_Tagger 노드 (nodes/cltagger_node.py) 독립 명령줄 태깅 도구。

설계 원칙: 노드 내의 사전/사후 처리/추론 로직 재사용 (단일 사실 출처)），
알고리즘 중복 구현 X; 이 스크립트는 파라미터 파싱, 디렉토리 스캔, .txt 작성, 진행 상황 출력，
스크립트 위치 디렉토리를 nodes/에 추가하여 시스템 경로에 포함시키고 노드 모듈 임포트 가능하게 함。

지원 CLTagger (cella110n/cl_tagger) 및 WD14 (SmilingWolf wd-tagger-v3 시리즈) 태그러。

본 스크립트는 AGPL-3.0-or-later 라이선스로 배포되지만, 그 태깅 알고리즘(사전 처리 / 태그 형식 / 후처리)은 다음 오픈 소스 프로젝트에서 파생되었습니다.，
기존 라이선스 및 저작권 선언은 함께 유지하십시오：

  - CLTagger Tagging 알고리즘은 오픈 소스 프로젝트에서 파생되었습니다 "cl_tagger"
    (https://github.com/Cirius0310/cl_tagger, Apache License 2.0)。Apache-2.0 전체 내용 저장소 참조 LICENSES/Apache-2.0.txt。
    모델 가중치에서 HuggingFace "cella110n/cl_tagger"（Fine-tune self "SmilingWolf/wd-eva02-large-tagger-v3"）。
  - WD14 추론 부분은 / 파생된 오픈 소스 프로젝트 참조 "ComfyUI-WD14-Tagger"
    (https://github.com/pythongosssss/ComfyUI-WD14-Tagger, MIT License, Copyright (c) 2024 pythongosssss)；
    MIT 라이선스 전체 내용은 REPOSITORY/LICENSES/MIT.txt에 있습니다. CSV 태그 형식은 SmilingWolf 커뮤니티 모델에서 유래했습니다.。

모델 가중치는 이 플러그인에 포함되지 않으며, 처음 사용할 때 Hugging Face에서 실행 시간에 다운로드합니다.。
훈련 도구의 ‘메시지 페이지 라벨링’ 기능이 자식 프로세스 방식으로 호출됨。

사용법:
    python bsk_tagger_cli.py --dir /path/to/images --tagger_type cltagger
    python bsk_tagger_cli.py --dir /path/to/images --tagger_type wd14 --wd14_model wd-vit-tagger-v3
    python bsk_tagger_cli.py --dir /path/to/images --recursive --trigger_word "1girl"
    python bsk_tagger_cli.py --list_models
"""

import argparse
import os
import sys

# 스크립트가 임의의 CWD에서 노드 모듈 임포트 가능: 스크립트 자체 디렉토리 + nodes/ 자 디렉토리
# （개발 환경 노드는 루트 또는 nodes/에, 패키지 노드는 nodes/에 게시합니다. 둘 다 덮어씁니다.）
_HERE = os.path.dirname(os.path.abspath(__file__))
for _cand in (_HERE, os.path.join(_HERE, "nodes")):
    if os.path.isdir(_cand) and _cand not in sys.path:
        sys.path.insert(0, _cand)

try:
    from cltagger_node import (  # noqa: E402
        _tag_image,
        _apply_trigger,
        WD14_MODELS,
        IMAGE_EXTENSIONS,
        CLTAGGER_DIR,
        WD14_DIR,
    )
except Exception as _imp_err:  # noqa: BLE001
    print(f"[BSK_Tagger] 노드 모듈 가져오기 불가 cltagger_node: {_imp_err}")
    print("[BSK_Tagger] 노드 파일 nodes/cltagger_node.py 존재 여부 확인 및 설치 numpy / Pillow。")
    sys.exit(1)


def _list_models():
    print("CLTagger: 고정 모델 (cella110n/cl_tagger)")
    print(f"  목차: {CLTAGGER_DIR}")
    cl_ok = os.path.isfile(os.path.join(CLTAGGER_DIR, "model.onnx"))
    print(f"  상태: {'준비 완료' if cl_ok else '처음 실행 시 자동으로 다운로드'}")
    print("WD14 모델 (SmilingWolf wd-tagger-v3 시리즈)):")
    for k in WD14_MODELS:
        d = os.path.join(WD14_DIR, k)
        mark = "설치됨" if os.path.isfile(os.path.join(d, "model.onnx")) else "다운로드되지 않음"
        print(f"  - {k} [{mark}]")


def _process_directory(directory, tagger_type, wd14_model, gen_threshold,
                        char_threshold, sort_mode, exclude_meta,
                        exclude_rating, exclude_quality, replace_underscore,
                        trailing_comma, exclude_tags, max_resolution, trigger_word,
                        recursive, overwrite):
    """이미지 일괄 태깅 폴더, 각 이미지에 동일 이름 생성 .txt。"""
    from PIL import Image

    files = []
    if recursive:
        for root, _dirs, names in os.walk(directory):
            for f in sorted(names):
                if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS:
                    files.append(os.path.join(root, f))
    else:
        for f in sorted(os.listdir(directory)):
            full = os.path.join(directory, f)
            if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS and os.path.isfile(full):
                files.append(full)

    if not files:
        print(f"[BSK_Tagger] 이미지 찾을 수 없음: {directory}")
        return

    tag_label = (wd14_model if tagger_type == "wd14" else "cltagger")
    print(f"[BSK_Tagger] 함께 {len(files)} 이미지 템플릿 | Labeler={tagger_type} 모델={tag_label}")
    print("-" * 60)

    ok = skip = fail = 0
    for i, src in enumerate(files, 1):
        txt = os.path.splitext(src)[0] + ".txt"
        if os.path.exists(txt) and not overwrite:
            print(f"[{i}/{len(files)}] SKIP: {os.path.basename(src)}")
            skip += 1
            continue
        try:
            with Image.open(src) as img:
                tags = _tag_image(img, tagger_type, wd14_model, gen_threshold,
                                  char_threshold, sort_mode, exclude_meta,
                                  exclude_rating, exclude_quality,
                                  replace_underscore, trailing_comma, exclude_tags,
                                  max_resolution)
            final = _apply_trigger(tags, trigger_word)
            with open(txt, "w", encoding="utf-8") as f:
                f.write(final)
            ok += 1
            rel = os.path.relpath(src, directory)
            print(f"[{i}/{len(files)}] OK: {rel}")
        except Exception as e:  # noqa: BLE001
            fail += 1
            print(f"[{i}/{len(files)}] FAIL: {os.path.basename(src)} - {e}")

    print("-" * 60)
    print(f"[BSK_Tagger] 완료! 성공:{ok} 건너뛰기:{skip} 실패:{fail}")


def main():
    parser = argparse.ArgumentParser(
        description="BSK Tagger CLI - 배치 이미지 태그 역추적 도구 (CLTagger / WD14)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  %(prog)s --dir /path/to/images --tagger_type cltagger
  %(prog)s --dir /path/to/images --tagger_type wd14 --wd14_model wd-vit-tagger-v3
  %(prog)s --dir /path/to/images --recursive --trigger_word "1girl" --overwrite
  %(prog)s --list_models
        """,
    )
    parser.add_argument("--dir", "-d", help="대상 이미지 디렉토리 경로")
    parser.add_argument("--recursive", "-r", action="store_true", help="재귀적으로 하위 디렉토리 스캔")
    parser.add_argument("--tagger_type", "-tt", choices=["cltagger", "wd14"],
                        default="cltagger", help="태그 유형 (기본값): cltagger)")
    parser.add_argument("--wd14_model", "-m", default="wd-vit-tagger-v3",
                        help="WD14 모델 키 (단 tagger_type=wd14 활성화)")
    parser.add_argument("--general_threshold", "-t", type=float, default=0.35,
                        help="일반적인 레이블 임계값 (기본): 0.35)")
    parser.add_argument("--character_threshold", "-ct", type=float, default=0.60,
                        help="캐릭터 태그 임계값 (기본값: 0.60)")
    parser.add_argument("--sort_mode", choices=["default", "novelai"], default="default",
                        help="정렬/형식 모드 (기본/novelai, 주로 사용됨) CLTagger)")
    parser.add_argument("--exclude_meta", dest="exclude_meta", action="store_true", default=True,
                        help="Meta 클래스 태그(CLTagger, 기본 활성화))")
    parser.add_argument("--no_exclude_meta", dest="exclude_meta", action="store_false",
                        help="메타 클래스 태그 제외 안 함")
    parser.add_argument("--exclude_rating", dest="exclude_rating", action="store_true", default=True,
                        help="Rating 태그 (CLTagger, 기본적으로 활성화))")
    parser.add_argument("--no_exclude_rating", dest="exclude_rating", action="store_false",
                        help="아니제외 Rating 레이블")
    parser.add_argument("--exclude_quality", dest="exclude_quality", action="store_true", default=True,
                        help="Quality 태그(CLTagger, 기본적으로 활성화)를 제외합니다.)")
    parser.add_argument("--no_exclude_quality", dest="exclude_quality", action="store_false",
                        help="품질 태그 제외 안 함")
    parser.add_argument("--replace_underscore", dest="replace_underscore",
                        action="store_true", default=True, help="밑줄을 공백으로 대체 (기본 활성화))")
    parser.add_argument("--no_replace_underscore", dest="replace_underscore",
                        action="store_false", help="밑줄 대체 안 함")
    parser.add_argument("--trailing_comma", action="store_true", default=False,
                        help="각 라벨 뒤에 쉼표 추가")
    parser.add_argument("--exclude_tags", "-e", default="", help="제외된 태그, 쉼표로 구분")
    parser.add_argument("--trigger_word", "-tw", default="", help="유발 단어, txt 맨 앞에 추가")
    parser.add_argument("--max_resolution", type=int, default=1280,
                        help="추론 전 이미지 최대 해상도(가로) 확대 (기본: 1280)")
    parser.add_argument("--overwrite", "-o", action="store_true", help="기존 txt 파일 덮어쓰기")
    parser.add_argument("--list_models", action="store_true", help="사용 가능한 모델 목록 표시 후 종료")

    args = parser.parse_args()

    if args.list_models:
        _list_models()
        return

    if not args.dir:
        parser.error("the following arguments are required: --dir/-d")

    _process_directory(
        directory=args.dir,
        tagger_type=args.tagger_type,
        wd14_model=args.wd14_model,
        gen_threshold=args.general_threshold,
        char_threshold=args.character_threshold,
        sort_mode=args.sort_mode,
        exclude_meta=args.exclude_meta,
        exclude_rating=args.exclude_rating,
        exclude_quality=args.exclude_quality,
        replace_underscore=args.replace_underscore,
        trailing_comma=args.trailing_comma,
        exclude_tags=args.exclude_tags,
        max_resolution=args.max_resolution,
        trigger_word=args.trigger_word,
        recursive=args.recursive,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
