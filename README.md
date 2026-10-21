> **© 2026 B상 @어둡습니다 ｜ 사용 약관：AGPL-3.0**

## 이 복사본

이 폴더는 원본 [ikusag-png/ComfyUI_bsk_UI](https://github.com/ikusag-png/ComfyUI_bsk_UI)를 받아 화면 문구를 한국어로 고친 작업본입니다. 원본 저장소와는 연결되어 있지 않습니다. 라이선스는 AGPL-3.0이고, 위 저작권 표시와 `LICENSE`는 유지합니다.

Anima 카메라 비교에서 쓰는 `BSK_카메라 제어`는 이 복사본의 `CameraControlNode`입니다. 다른 PC에서는 이 저장소를 클론합니다.

```text
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/dancingamer/ComfyUI-bsk_UI.git
```

폴더 이름은 `ComfyUI-bsk_UI`입니다. ComfyUI를 다시 시작하면 노드가 등록됩니다. 이 노드는 카메라 문장만 만듭니다. 그림에 반영하려면 프리셋에 `Anima/anima_bsk_camera_control.safetensors`를 올리고, 문장은 `Anima LoRA 프롬프트`의 `camera_prompt`로 넣습니다. 연결 방법은 [ComfyUI-Anima3DPoseControl](https://github.com/dancingamer/ComfyUI-Anima3DPoseControl) README를 보면 됩니다.
> 이 플러그인은 **AGPL-3.0** 게시합니다. 자유롭게 사용, 수정, 재배포할 수 있지만,**저작권 선언을 유지해야 하며 `LICENSE` 파일**；이 코드를 기반으로 한 파생 작품 또는 재배포，**전체적으로 AGPL-3.0 라이선스에 계속 유지해야 합니다.**（소스 코드를 폐쇄적으로 사용하거나 수정하지 않고 상업적으로 사용하지 않는 경우.**저작권 및 라이선스 정보를 삭제해서는 안 됩니다**。타사 파생 코드(wan_nodes.py / bsk_ltx_director.py / text_generate_node.py / cltagger_node.py / wd14 등)는 각 라이선스에 따라 서명되며, 자세한 내용은 각 파일 헤더 및 `LICENSES/` 목차。
>
<h1 align="center">ComfyUI_bsk_UI</h1>
<!-- 또는 스타일 속성 사용 -->
<h2 align="center">
  <a href="https://space.bilibili.com/26779709" target="_blank">
    <img src="https://www.bilibili.com/favicon.ico" width="20" height="20" alt="B상" style="vertical-align: middle;">
    @灰暗x
  </a>
</h2>
제 최 B상 공간에서 최신 버전을 가져오세요,
**🌟 소개 | Introduction**
ComfyUI_bsk_UI ComfyUI를 위해 현대적이고 사용자 친화적인 인터랙티브 인터페이스입니다.。
ComfyUI 원본 프론트엔드의 번거로운 작업을 완전히 바꾸어 AI 그림 작업 흐름 구성이 더 효율적이고 편안하게 가능하도록 함。

ComfyUI 기본 인터페이스의 끔찍한 상호 작용 경험에 지치셨나요？

캔버스 점을 드래그하여 확대/축소하면 손이 아프세요? 노드 연결선을 눈을 굴리도록 만듭니다？

bsk_UI 하면이다위하여준비됐어?의해결책！

ComfyUI_bsk_UI is a modern and user-friendly interface for ComfyUI,
designed to liberate you from the cumbersome and complicated operation style of the original frontend.
Create your AI art workflows more efficiently and comfortably!

![alt text](aa.jpg)
![alt text](bb.jpg)
✨ **핵심 특징** | Key Features

🎯 **간단한 작업** - 복잡한 노드 작업에서 벗어나 워크플로우 구성이 직관적이고 간단해짐

🖱️ **부드러운 상호 작용** - 캔바스를 드래그하여 크기를 조정할 필요가 없습니다. 단축키만 사용하면 대부분의 작업을 완료할 수 있으며, 더 이상"손이 아파"

🎨 **명확한 인터페이스** - 새롭게 설계된 시각 레이아웃, 중요한 정보가 한눈에 보이는 전체 화면 프롬프트 작성 기능, 강력한 워크플로우 및 프롬프트 라이브러리 저장 기능,

🔧 **유연한 설정** - API 워크플로우 로드 지원, 창작 환경 자유롭게 커스터마이징

✨ **독립 모드** - comfyui_panel.html을 열면 ComfyUI 프런드에서 분리하여 독립적으로 실행할 수 있습니다..

📦 **설치 방법 | Installation**
본 저장소 복제하여 ComfyUI의 custom_nodes 디렉토리에 저장：

bash

```text
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/dancingamer/ComfyUI-bsk_UI.git
```

생성 과정 미리보기 활성화를 위해 다음 두 파일을 다운로드하여 저장하십시오./ComfyUI/models/vae_approx

[taew2_1.safetensors](https://www.modelscope.cn/models/Kijai/WanVideo_comfy/resolve/master/taew2_1.safetensors)

[taew2_2.safetensors](https://www.modelscope.cn/models/Kijai/WanVideo_comfy/resolve/master/taew2_2.safetensors)

또한 ComfyUI 시작 매개변수에 추가 --preview-method taesd
ComfyUI 시작, 인터페이스 상단 왼쪽에서 새롭게 추가된 버튼을 클릭하여 bsk_UI 패널을 엽니다.

설정 버튼을 클릭하여 ComfyUI로 내보낸 API 워크플로우의 JSON 파일을 로드하고 탐색을 시작！

🚀 **사용 지침 | Usage Guide**
상세 사용 튜토리얼 및 기술 공유, B상 공간 팔로우 환영：

👉 Bilibili:  [@灰暗x](https://space.bilibili.com/26779709)

정기적으로 비디오 튜토리얼을 업데이트하여, 당신이 AnimaDex를 사용하도록 가르쳐 드리겠습니다 bsk_UI！

💡 창작 의도 | Why I Built This
ComfyUI 정말 강력한 도구이지만, 기본 인터랙션 경험은 참을 수가 없습니다. 거의 매번 사용 시마다 참을 줄을 알아야 합니다.：

노드 레이아웃이 꽉 잡히지 않아 다른 노드가 정렬되지 않아 프롬프트 작성 및 생성 결과 확인이 매우 엉터리 상태로 진행됨,

결과 관찰 및 프롬프트 텍스트 작성 시, 캔버스 드래그 앤 드롭 크기 조정 시 마우스가 고장날 정도로 잦음,

그래서 제가 진짜 유용한 ComfyUI 프론트엔드를 직접 만들어 보기로 했습니다. 이것이 bsk_UI의 탄생 이야기입니다.。

있다그것을 당신이 할 수 있습니다이节약많은 체력을 얻으면 더 좋음의사용경험. 모두운영이다가능아니움직이면아니동, 능누르다일개키절아니누르다제이개.


**업데이트 로그:**

260313: 프롬프트 편집 상자 텍스트 정렬 문제 수정, 일부 개선 사항은 사용자 경험을 향상시키기 위한 것입니다.,
        독립 모드가 추가되었습니다. comfiui_panel.html을 열어 플러그인이 설치된 ComfyUI에 연결하여 즉시 프론트엔드 경험을 제공합니다..


📄 **오픈 소스 라이선스 | License**
Copyright © 2026 [@灰暗x](https://space.bilibili.com/26779709)

본 프로젝트는 GNU Affero General Public License v3.0 (AGPLv3) 라이선스 하에 오픈 소스입니다.。

이는 엄격한 오픈 소스 라이선스입니다. 이 프로젝트가 항상 자유롭고 개방되도록 유지하기를 바랍니다. 누구도 또는 어떤 조직도 이를 폐쇄적인 상업적 용도로 사용하지 않도록 합니다.。

🙏 **감사합니다 | Acknowledgements**

이 프로젝트에서 참조, 사용 또는 기반으로 한 제3자 저작권 리소스에 감사드립니다. 각 리소스의 라이선스를 존중합니다.：

• **ComfyUI**: https://github.com/Comfy-Org/ComfyUI （GPL-3.0 License）— 이 프로젝트 실행 호스트 플랫폼；`nodes/text_generate_node.py` 개선됨 (기본 노드에서) `nodes_textgen.py`（Apache-2.0）。
• **ComfyUI-Manager**: https://github.com/ltdrdata/ComfyUI-Manager （GPL-3.0 License）— 플러그인 의존성 검색 및 설치。
• **Crystools**: https://github.com/crystian/ComfyUI-Crystools （MIT License, Copyright (c) 2023 Crystian）— 하드웨어 모니터링 기능은 구현을 참조했습니다。
• **Comfyroll Studio**: https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes （GPL-3.0 License）— 백엔드 다중 LoRA 로딩 노드 참조 구현。
• **ComfyUI-WanVideoWrapper**: https://github.com/kijai/ComfyUI-WanVideoWrapper （Apache-2.0 License）— `wan_nodes.py` 그것을 기반으로 수정됨。
• **WhatDreamsCost-ComfyUI / ComfyUI-Koolook**: https://github.com/WhatDreamsCost/WhatDreamsCost-ComfyUI （GPL-3.0 License）— `bsk_ltx_director.py` WhatDreamsCost의 LTX Director를 기반으로 재현 (참조) Koolook fork `v1_3_9`）；`nodes/vs_ltx_patch.py` 실행 중 monkey-patch 하여 `ltx_director.py` 자르기 및 볼륨 구현。
• **sd-scripts**: https://github.com/kohya-ss/sd-scripts （Apache-2.0 License）— 훈련 관련 스크립트 참조。
• **MiniMax H3**: https://github.com/MiniMax-AI/MiniMax-H3 （MiniMax H3 Community License Agreement, Copyright © 2026 MiniMax, All Rights Reserved.）— H3 비디오 워크스테이션（`nodes/bsk_minimax_director.py`）공식 노드 재사용 `MiniMaxH3ImageToVideo` / `MiniMaxH3ReferenceToVideo` / `MiniMaxH3SigmaShift`；H3 프롬프트 편집기의 구조와 연결 규칙은 공식 skill 프로젝트 [skills](https://github.com/MiniMax-AI/MiniMax-H3/tree/main/skills)를 참조했습니다. 자세한 내용은 다음을 참조하십시오. `LICENSES/MiniMax-H3.txt`。
• **CodeMirror 5**: https://codemirror.net/5/ （MIT License）— 프롬프트 팁 편집기에서 사용하는 코드 편집기 커널 (한국어 지원). `js/codemirror/`）。
• **Google Fonts**: Playfair Display 및 Noto Sans SC（SIL Open Font License 1.1）— 인터페이스 제목과 본문에 사용된 글꼴, 실행 시 Google Fonts에서 로드。
• **cl_tagger**: https://github.com/Cirius0310/cl_tagger （Apache License 2.0）— 이미지 라벨링 노드（`nodes/cltagger_node.py`，노드 `BSK_Tagger`）이 CLTagger의 태깅 알고리즘은 이 프로젝트에서 파생되었습니다. 모델 가중치는 HuggingFace [cella110n/cl_tagger](https://huggingface.co/cella110n/cl_tagger)에서 가져왔으며(cella110n/cl_tagger에서 미세 조정됨) [SmilingWolf/wd-eva02-large-tagger-v3](https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3)）。
• **ComfyUI-WD14-Tagger**: https://github.com/pythongosssss/ComfyUI-WD14-Tagger （MIT License）— WD14 태그 노드는 이 프로젝트에서 파생되었습니다. WD14 모델은 [SmilingWolf](https://huggingface.co/SmilingWolf)의 wd-tagger-v3 시리즈에서 가져왔습니다. 두 가지 가중치는 실행 중에 다운로드되며 플러그인에 포함되지 않습니다. Apache-2.0 전체는 다음을 참조하십시오. `LICENSES/Apache-2.0.txt`，MIT 전체 텍스트 참조 `LICENSES/MIT.txt`。
• **AnimaDex**: https://github.com/zetaneko/AnimaDex （MIT License, Copyright (c) 2026 AnimaDex contributors）— 프롬프트 갤러리에는 추출된 애니메이션 캐릭터/아티스트 데이터셋만 읽기 전용으로 연결됩니다. 측면(작품/색상/눈 색 등) 및 파일 이름 규칙은 구현 방식에 맞춰 호환성을 갖도록 설정되었습니다. 이 플러그인에는 소스 코드가 포함되어 있지 않으며, 읽기 전용으로만 작동합니다. `animadex.db` / `import/*.csv`；설정의 '복제 / 설치'는 AnimaDex 자체를 호출 `install.sh` / `install.bat`（%s %d {name} 
 <tag> ComfyUI LoRA GPU API BSK. `LICENSES/AnimaDex.txt`。
• **SVG Repo**: https://www.svgrepo.com/ — 일부 SVG 아이콘은 이 웹사이트에서 공유한 아이콘을 사용합니다.：
 * Lasso Select & Paint Bucket: MIT License (Author: Esri, Denali Design)
 * Pen Icon: CC BY License (Author: Solar Icons)
 * Source: SVG Repo (https://www.svgrepo.com/)
• **Unsplash**: Photo by Andrew S on Unsplash https://unsplash.com/photos/a-dog-and-a-cat-laying-in-the-grass-ouo1hbizWwo
      
이것은 당신이 | This means you are free to:

✅ - 를 모든 시나리오에서 이 소프트웨어 사용 | Use the software for any purpose

✅ 수정 - 코드를 수정하여 요구 사항을 충족하십시오 | Modify the code to fit your needs

✅ 공유 - 타인에게 공유 | Redistribute copies

✅ 네트워크 서비스 - 네트워크를 통해 이 소프트웨어 서비스를 제공합니다 | Run a public instance/network service

저RAM 모드 | But you must:

🔓 오픈 소스 - 본 소프트웨어를 배포하거나 네트워크 서비스를 제공하는 경우 모든 소스 코드 공개 필수 | Disclose your source code when you distribute the software or run a public instance

📝 저작권 보존 - 원본 저작권 선언을 반드시 유지해야 합니다. | Retain the original copyright notice

⚖️ 동일 프로토콜 - 수정된 코드는 동일한 AGPLv3 프로토콜 사용 | Apply the same AGPLv3 license to your modifications

특별 선언 | Important Note
본 계약은 누구에게나 이 소프트웨어(수정 버전 포함)를 비공개 상업적 용도로 사용하는 것을 명시적으로 금지합니다. 소프트웨어를 변경하여 패키징하여 판매하려는 모든 시도는 저작권 침해 행위가 됩니다.。

This license explicitly prohibits using this software (including modified versions) for closed-source commercial purposes. Any attempt to repackage and sell this software under a different name will constitute copyright infringement.

📬 저자 연락 | Contact
Bilibili: [@灰暗x](https://space.bilibili.com/26779709)

Issues: GitHub 저장소에 이슈를 제출하여 문제 또는 제안을 제공해 주십시오.

이 프로젝트가 도움이 되셨다면 B상에서 팔로우하고 지원해 주세요.！

If you find this project helpful, please consider following me on Bilibili!

⭐ Star 역시 격려를 해 | A star is also a great encouragement ⭐
