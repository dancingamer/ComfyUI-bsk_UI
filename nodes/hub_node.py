# SPDX-FileCopyrightText: 2026 灰暗x
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
BSK 수집기 노드 (Hub Node)
블로킹된 작업을 스레드 풀에 던지고 aiohttp의 이벤트 루프를 막지 마십시오.。

특징：
- 기본적으로 1 그룹 입력+출력, 직접（out{i} = in{i}）。
- 어느 엔드포인트에 연결되면 프론트엔드(js/hub_node.js)는 자동으로 새로운 입력 그룹을 끝에 추가합니다.+출력。
- End point is "*"（자유롭게유형），어떤 노드와도 연결 가능데이터 유형；연결후표시실제유형。
- 백엔드 최대 그룹 선언 optional "*" 연결되지 않은 그룹은 None(직통 건너뜁니다).）。

프론트엔드 동적 추가/삭제 로직은 js/hub_node.js (js/string_merge_dynamic.js의 검증된 패턴을 따름)에서 확인하십시오.）。
"""

class BSK_Hub:
    MAX = 16  # 사전 선언 한도를 설정하여 일상적인 사용에 충분합니다

    @classmethod
    def INPUT_TYPES(cls):
        optional = {}
        for i in range(1, cls.MAX + 1):
            # "*" = 어떤 유형이든, 어떤 출력과 연결 가능
            # optional + default=None：연결되지 않았을 때 이 그룹은 비어 있으며, 노드는 실행 가능 (빈 엔드포인트로 오류 발생하지 않음)）
            optional[f"in{i}"] = ("*", {"default": None, "forceInput": False})
        return {"required": {}, "optional": optional}

    RETURN_TYPES = ("*",) * MAX
    RETURN_NAMES = tuple(f"out{i}" for i in range(1, MAX + 1))
    FUNCTION = "pass_through"
    CATEGORY = "bsk/utilities"
    DESCRIPTION = "수집기/버스: 임의의 엔드포인트에 대한 입력/출력 그룹 자동 증가/감소, 출력 직접 통신 매핑 입력。"

    def pass_through(self, **kwargs):
        # in1..inMAX 순서대로 원본 반환; 연결되지 않은 경우 None
        # （프론트만 넣기"사용 중"피부는 어두운 색 (텍스트는 밝게), 이 창은 독립적인 겹쳐진 레이어입니다.）
        return tuple(kwargs.get(f"in{i}", None) for i in range(1, self.MAX + 1))


NODE_CLASS_MAPPINGS = {
    "BSK_Hub": BSK_Hub,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "BSK_Hub": "BSK 수신기",
}
