> Copyright (C) 2026 B상 / @灰暗x
>
> 이 저장소는 AGPL-3.0 라이선스를 따릅니다. 수정하거나 재배포할 수 있지만, 저작권 고지와 `LICENSE` 파일을 유지해야 하며 파생물도 AGPL-3.0 조건을 따라야 합니다. 제3자 파생 코드와 리소스는 각 파일 헤더와 `LICENSES/`의 라이선스를 따릅니다.

<h1 align="center">ComfyUI-bsk_UI</h1>

<p align="center">
  <a href="https://space.bilibili.com/26779709" target="_blank">@灰暗x</a>의 ComfyUI_bsk_UI를 기반으로 한 한국어 작업본
</p>

## 이 저장소에 대하여

이 저장소는 원본 [ikusag-png/ComfyUI_bsk_UI](https://github.com/ikusag-png/ComfyUI_bsk_UI)를 기반으로, 이 PC에서 사용 중인 한국어 화면 문구와 수정 사항을 보존하기 위해 분리한 작업본입니다. 원본 저장소와는 별도의 git 저장소이며, 다른 PC에서도 같은 워크플로를 열 수 있도록 `BSK_카메라 제어` 노드를 포함합니다.

Anima 카메라 비교 워크플로에서 쓰는 노드는 `CameraControlNode`이고, 화면에는 `BSK_카메라 제어`로 표시됩니다. 이 노드는 카메라 방향을 텍스트 프롬프트로 만들어 줄 뿐, LoRA를 직접 불러오지는 않습니다.

## 설치

ComfyUI의 `custom_nodes` 폴더에서 이 저장소를 클론합니다.

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/dancingamer/ComfyUI-bsk_UI.git
```

폴더 이름은 `ComfyUI-bsk_UI`로 둡니다. 설치 후 ComfyUI를 다시 시작하면 노드가 등록됩니다.

생성 중 미리보기를 사용하려면 다음 파일을 `ComfyUI/models/vae_approx`에 넣고 ComfyUI 시작 인자에 `--preview-method taesd`를 추가합니다.

- [taew2_1.safetensors](https://www.modelscope.cn/models/Kijai/WanVideo_comfy/resolve/master/taew2_1.safetensors)
- [taew2_2.safetensors](https://www.modelscope.cn/models/Kijai/WanVideo_comfy/resolve/master/taew2_2.safetensors)

## Anima 카메라 워크플로에서 사용하기

`BSK_카메라 제어`는 카메라 문장만 만듭니다. 그림에 실제로 반영하려면 다음 연결이 필요합니다.

1. `Anima/anima_bsk_camera_control.safetensors`를 LoRA 프리셋에 추가합니다. 권장값은 모델 0.8, CLIP 0.8입니다.
2. `BSK_카메라 제어`의 `카메라 프롬프트` 출력을 `카메라 선택` 노드의 `bsk` 입력에 연결합니다.
3. `카메라 선택`의 출력을 `Anima LoRA 프롬프트`의 `camera_prompt` 입력에 연결합니다.
4. 장면 문장은 `Anima LoRA 프롬프트`의 입력칸에 적고, 아래의 `적용될 프롬프트`에서 카메라 문장과 트리거가 합쳐졌는지 확인합니다.

자세한 연결 예시는 [ComfyUI-Anima3DPoseControl](https://github.com/dancingamer/ComfyUI-Anima3DPoseControl) 저장소의 README와 `workflows/AnimaPoseCameraV1.json`을 참고하세요.

## 주요 기능

- ComfyUI 안에서 사용할 수 있는 bsk_UI 패널
- 프롬프트 라이브러리와 프롬프트 편집 도구
- 워크플로 불러오기와 관리 기능
- `BSK_카메라 제어` 노드
- H3, LTX, Qwen, WD14 태깅, 훈련 관련 보조 기능
- `comfyui_panel.html`을 직접 열어 사용하는 독립 패널 모드

## 사용법

ComfyUI를 시작한 뒤 화면 왼쪽 위에 추가된 bsk_UI 버튼을 눌러 패널을 엽니다. 설정에서 ComfyUI API 형식의 워크플로 JSON을 불러와 사용할 수 있습니다.

원본 플러그인의 자세한 사용 설명과 영상 튜토리얼은 제작자 Bilibili 채널을 참고하세요.

- Bilibili: [@灰暗x](https://space.bilibili.com/26779709)

## 변경 기록

- 한국어 화면 문구와 README를 정리했습니다.
- Anima 카메라 비교 워크플로에서 쓰는 `BSK_카메라 제어` 노드를 보존했습니다.
- 원본 저장소와 분리해 `dancingamer/ComfyUI-bsk_UI` 저장소로 관리합니다.
- 원본 260313 변경: 프롬프트 편집 상자의 텍스트 정렬 문제를 수정하고, `comfyui_panel.html` 독립 모드를 추가했습니다.

## 라이선스

Copyright (C) 2026 [@灰暗x](https://space.bilibili.com/26779709)

이 프로젝트는 GNU Affero General Public License v3.0(AGPL-3.0) 라이선스를 따릅니다. 배포하거나 네트워크 서비스로 제공할 때는 라이선스 조건에 따라 소스 코드를 공개해야 하며, 원저작자 표시와 라이선스 고지를 유지해야 합니다.

원저작자는 수정본을 폐쇄형 상업 제품으로 재포장해 판매하는 행위를 금지한다고 명시했습니다. 자세한 내용은 `LICENSE`와 원본 프로젝트의 고지를 확인하세요.

## 감사와 제3자 라이선스

이 프로젝트는 여러 오픈소스 프로젝트와 리소스를 참고하거나 포함합니다. 각 항목의 라이선스는 해당 프로젝트와 `LICENSES/` 파일을 따릅니다.

- [ComfyUI](https://github.com/Comfy-Org/ComfyUI) - GPL-3.0. 실행 호스트 플랫폼.
- [ComfyUI-Manager](https://github.com/ltdrdata/ComfyUI-Manager) - GPL-3.0. 의존성 검색과 설치 기능.
- [Crystools](https://github.com/crystian/ComfyUI-Crystools) - MIT. 하드웨어 모니터링 기능 참고.
- [Comfyroll Studio](https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes) - GPL-3.0. 백엔드 다중 LoRA 로딩 노드 참고.
- [ComfyUI-WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper) - Apache-2.0. `wan_nodes.py` 기반.
- [WhatDreamsCost-ComfyUI / ComfyUI-Koolook](https://github.com/WhatDreamsCost/WhatDreamsCost-ComfyUI) - GPL-3.0. `bsk_ltx_director.py`와 `nodes/vs_ltx_patch.py` 관련 참고.
- [sd-scripts](https://github.com/kohya-ss/sd-scripts) - Apache-2.0. 훈련 관련 스크립트 참고.
- [MiniMax H3](https://github.com/MiniMax-AI/MiniMax-H3) - MiniMax H3 Community License Agreement. H3 비디오 워크스테이션 관련 노드와 프롬프트 편집 구조 참고.
- [CodeMirror 5](https://codemirror.net/5/) - MIT. 프롬프트 편집기 코드 편집기.
- Google Fonts의 Playfair Display, Noto Sans SC - SIL Open Font License 1.1.
- [cl_tagger](https://github.com/Cirius0310/cl_tagger) - Apache-2.0. `BSK_Tagger` 태깅 알고리즘 기반.
- [ComfyUI-WD14-Tagger](https://github.com/pythongosssss/ComfyUI-WD14-Tagger) - MIT. WD14 태그 노드 기반.
- [AnimaDex](https://github.com/zetaneko/AnimaDex) - MIT. 프롬프트 갤러리용 데이터셋 연동.
- [SVG Repo](https://www.svgrepo.com/) - 일부 SVG 아이콘 출처.
- [Unsplash](https://unsplash.com/photos/a-dog-and-a-cat-laying-in-the-grass-ouo1hbizWwo) - 일부 예시 이미지 출처.

## 연락처

- 원저작자 Bilibili: [@灰暗x](https://space.bilibili.com/26779709)
- 이 한국어 작업본: [dancingamer/ComfyUI-bsk_UI](https://github.com/dancingamer/ComfyUI-bsk_UI)
