# SPDX-FileCopyrightText: 2026 灰暗x
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
BatchRandomPromptNode - 다중 랜덤 프롬프트 생성 노드
기능：
1. 여러 줄 프롬프트 입력 지원, 자동 주석 필터링
2. 두 가지 무작위 모드 지원: 순서 무작위, 가중치 무작위
3. 트리거 메커니즘을 통해 매번 실행 보장
4. artists 디렉토리 파일에서 프롬프트를 읽어 합칩니다.
"""

import random
import os


class BatchRandomPromptNode:
    """다중 랜덤 프롬프트 생성 노드"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt_text": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "여러 줄 프롬프트를 입력합니다\n# 시작 행동 주석\n인라인 주석 지원：aa, bb, # cc, dd"
                }),
                "trigger": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 999999
                }),
                "shuffle_order": ("BOOLEAN", {
                    "default": False,
                    "label_on": "순서 무작위",
                    "label_off": "순서 섞기"
                }),
                "random_count": ("BOOLEAN", {
                    "default": False,
                    "label_on": "수량 무작위",
                    "label_off": "고정 수량"
                }),
                "random_weight": ("BOOLEAN", {
                    "default": False,
                    "label_on": "가중치 무작위",
                    "label_off": "고정 가중치"
                }),
                "weight_min": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.01,
                    "max": 10.0,
                    "step": 0.1
                }),
                "weight_max": ("FLOAT", {
                    "default": 1.9,
                    "min": 0.01,
                    "max": 10.0,
                    "step": 0.1
                }),
            },
            "optional": {
                "weight_increment": ("BOOLEAN", {
                    "default": False,
                    "label_on": "가중치 증가",
                    "label_off": "가중치 증가"
                }),
                "weight_step": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.01,
                    "max": 10.0,
                    "step": 0.01
                }),
                "weight_only_has": ("BOOLEAN", {
                    "default": False,
                    "label_on": "오직 가중치",
                    "label_off": "오직 가중치"
                }),
                "seed_str": ("STRING", {
                    "default": "0",
                    "multiline": False
                }),
                "auto_seed": ("BOOLEAN", {
                    "default": True,
                    "label_on": "자동 시드",
                    "label_off": "자동 시드"
                }),
                "dedup_prompt": ("BOOLEAN", {
                    "default": False,
                    "label_on": "중복 제거",
                    "label_off": "중복 제거"
                }),
                "enable_file_prompt": ("BOOLEAN", {
                    "default": False,
                    "label_on": "파일 프롬프트",
                    "label_off": "파일 프롬프트"
                }),
                "artist_file": ("STRING", {
                    "default": "",
                    "multiline": False
                }),
                "file_prompt_max_count": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 9999,
                    "step": 1
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("랜덤 프롬프트", "유발값")
    FUNCTION = "execute"
    CATEGORY = "bsk_UI/프롬프트"
    DESCRIPTION = "프롬프트 단어를 무작위로 생성하고 순서 무작위, 수량 무작위 및 가중치 무작위 지원"
    
    def execute(self, prompt_text, trigger, shuffle_order, random_count, 
                random_weight, weight_min, weight_max,
                weight_increment=False, weight_step=0.1,
                weight_only_has=False,
                seed_str="0", auto_seed=False, dedup_prompt=False,
                enable_file_prompt=False, artist_file="", file_prompt_max_count=0):
        """
        랜덤 프롬프트 생성 실행
        
        매개변수：
            prompt_text: 다중 줄 프롬프트 텍스트
            trigger: 트리거 값 (실행 트리거）
            shuffle_order: 순서 무작위 여부
            random_count: 수량 무작위 여부
            random_weight: 가중치 무작위 여부
            weight_min: 가중치 최소값
            weight_max: 가중치 최대값
            weight_increment: 가중치 증가 여부 (활성화되면 무작위 가중치를 비활성화하고, 실행할 때마다 단계별 값으로 가중치가 증가)）
            weight_step: 점진적 가중치 증가 값
            weight_only_has: 이미 존재하는 가중치 프롬프트에 대한 무작위 가중치 적용
            seed_str: 랜덤 시드 (문자열, 정확도 손실 방지)）
            auto_seed: 매번 실행 시 새 시드 자동 생성
            dedup_prompt: 제거 텍스트 기반 중복 프롬프트, 가중치 버전 우선 보존
        """
        # 설정 랜덤 시드, 결과 재현 가능하게
        if seed_str is not None:
            try:
                seed_val = int(str(seed_str).strip())
                random.seed(seed_val)
            except (ValueError, TypeError):
                random.seed(0)
        
        # 주석 필터링, 유효한 프롬프트 키워드 추출
        prompts = self._filter_prompts(prompt_text)
        
        # 파일에서 프롬프트 텍스트 읽어 합침
        if enable_file_prompt and artist_file:
            file_prompts = self._read_artist_file_prompts(artist_file)
            if file_prompts:
                # 최대 개수를 설정하고 0보다 크면 무작위로 선택합니다
                if file_prompt_max_count > 0 and len(file_prompts) > file_prompt_max_count:
                    file_prompts = random.sample(file_prompts, file_prompt_max_count)
                prompts.extend(file_prompts)
        
        if not prompts:
            return ("", 0)
        
        # 중복 제거: 텍스트 기반으로 그룹화, 가중치가 있는 버전 우선 유지
        if dedup_prompt:
            prompts = self._dedup_prompts(prompts)
        
        result = prompts
        
        # 순서 무작위
        if shuffle_order:
            random.shuffle(result)
        
        # 수량 무작위
        if random_count and len(result) > 1:
            count = random.randint(1, len(result))
            result = random.sample(result, count)
        
        # 가중치 증가 상태 업데이트(종료 시 마킹, 다음 시작 시 최소값으로 재설정)）
        node_id = id(self)
        # 인스턴스 레벨 상태 딕셔너리가 존재하도록 보장합니다 (_apply_increment_weight도 확인하며, 미리 초기화하여 179행 접근 오류 방지)）
        if not hasattr(self, '_increment_state'):
            self._increment_state = {}
        if not weight_increment:
            if node_id in self._increment_state:
                self._increment_state[node_id]["was_enabled"] = False

        # 가중치 증가 모드(임의 가중치 우선, 활성화 시 임의 가중치 무시）
        if weight_increment:
            result = self._apply_increment_weight(result, weight_min, weight_max, weight_step, weight_only_has)
        # 가중치 무작위
        elif random_weight:
            result = self._apply_weight(result, weight_min, weight_max, weight_only_has)
        
        # 출력 결과: 마지막에 영어 쉼표가 있는지 확인
        prompt_result = ', '.join(result)
        if prompt_result and not prompt_result.rstrip().endswith(','):
            prompt_result += ','
        
        # 유발 값 출력 계산
        # trigger=True 그리고 auto_seed=True → Output random integer
        # trigger=True 그리고 auto_seed=False → 출력 트리거 값
        # trigger=False → 출력 0
        trigger_active = trigger is not None and trigger > 0
        if trigger_active and auto_seed:
            trigger_output = random.randint(0, 2147483647)
        elif trigger_active:
            trigger_output = trigger
        else:
            trigger_output = 0
        
        return (prompt_result, trigger_output)
    
    def _filter_prompts(self, text):
        """
        주석 필터링, 유효한 프롬프트 키워드 추출
        MultilineStringEditorNode의 필터링 로직 재사용
        
        필터 전략：
        1. 전체 주석: 줄 시작에#，행 건너뛰기
        2. 내장 주석: 내장#，필터 포함#프롬프트 텍스트
        3. 줄 바꿈: 건너띡
        """
        lines = text.split('\n')
        all_prompts = []
        
        for line in lines:
            stripped = line.lstrip()
            
            # 전 행 주석
            if stripped.startswith('#'):
                continue
            
            # 공백
            if not stripped:
                continue
            
            # 행 내 주석
            if '#' in line:
                filtered_line = self._filter_inline_comments(line)
                if filtered_line.strip():
                    # 콤마로 분할 추출 프롬프트
                    prompts = [p.strip() for p in filtered_line.split(',') if p.strip()]
                    all_prompts.extend(prompts)
            else:
                # 콤마로 분할 추출 프롬프트
                prompts = [p.strip() for p in line.split(',') if p.strip()]
                all_prompts.extend(prompts)
        
        return all_prompts
    
    def _filter_inline_comments(self, line):
        """
        필터 내부 포함된#프롬프트(콤마로 구분)）
        원래 쉼표 구조 유지
        """
        parts = []
        current = ""
        for char in line:
            if char == ',':
                parts.append((current, True))
                current = ""
            else:
                current += char
        parts.append((current, False))
        
        filtered_parts = []
        for content, has_comma in parts:
            stripped = content.lstrip()
            if stripped.startswith('#'):
                if filtered_parts and has_comma:
                    filtered_parts[-1] = (filtered_parts[-1][0], True)
                continue
            else:
                filtered_parts.append((content, has_comma))
        
        result = ""
        for content, has_comma in filtered_parts:
            result += content
            if has_comma:
                result += ","
        
        return result
    
    def _apply_weight(self, prompts, weight_min, weight_max, weight_only_has=False):
        """
        가중치 무작위: 각 프롬프트에 무작위 가중치 할당
        기존 가중치 문법을 지능적으로 처리하여 (text:weight) 형식으로 통일하여 출력
        weight_only_has: 기존 가중치 프롬프트에만 무작위 가중치를 적용하고, 가중치가 없는 것은 원래대로 유지합니다
        """
        weighted = []
        for prompt in prompts:
            p = prompt.strip()
            has_weight = self._has_weight(p)
            if weight_only_has and not has_weight:
                weighted.append(p)
                continue
            new_weight = round(random.uniform(weight_min, weight_max), 2)
            pure_text = self._strip_all_weight(p)
            weighted.append(f"({pure_text}:{new_weight:.2f})")
        return weighted

    def _apply_increment_weight(self, prompts, weight_min, weight_max, weight_step, weight_only_has=False):
        """
        가중치 증가: 각 실행마다 가중치를 단계별로 증가
        weight_min에서 시작하여 weight_step만큼 증가시키고, weight_max를 초과하면 재설정합니다. weight_min
        가중치 증가 시 재설정 weight_min
        직접 가중치를 설정하고 프롬프트의 원래 가중치를 고려하지 않습니다
        """
        node_id = id(self)
        # 인스턴스 레벨 상태 딕셔너리가 존재해야 함
        if not hasattr(self, '_increment_state'):
            self._increment_state = {}
        state = self._increment_state.get(node_id, {"weight": weight_min, "was_enabled": False})
        
        # If you just enabled increment (from off to on), reset weights to minimum
        if not state.get("was_enabled", False):
            state["weight"] = weight_min
            state["was_enabled"] = True
        
        current_weight = round(state["weight"], 4)
        
        # 다음 가중치 계산
        next_weight = round(current_weight + weight_step, 4)
        if next_weight > weight_max:
            next_weight = weight_min
        state["weight"] = next_weight
        self._increment_state[node_id] = state
        
        print(f"[BatchRandomPrompt] 가중치 증가: node={node_id}, 현재 가중치={current_weight:.2f}, 다음 가중치={next_weight:.2f}")
        
        weighted = []
        for prompt in prompts:
            p = prompt.strip()
            has_weight = self._has_weight(p)
            if weight_only_has and not has_weight:
                weighted.append(p)
                continue
            pure_text = self._strip_all_weight(p)
            weighted.append(f"({pure_text}:{current_weight:.2f})")
        return weighted

    def _has_weight(self, text):
        """
        확인 중인 프롬프트의 가중치 문법이 있는지 확인
        형식 검사：(text:weight), {text}, text:weight
        """
        import re
        text = text.strip()
        if not text:
            return False
        # 괄호 안에 가중치 지정 (text:weight)
        if text.startswith('(') and text.endswith(')'):
            inner = text[1:-1]
            if self._find_outer_colon(inner) > 0:
                return True
            # 순수 괄호 강화 (text)도 가중치로 간주 (강도 1.1배)）
            return True
        # 중괄호 {text}
        if text.startswith('{') and text.endswith('}'):
            return True
        # 원시 가중치 text:1.5
        if re.match(r'^.+?:(-?\d+\.?\d*)\s*$', text):
            return True
        return False

    def _strip_all_weight(self, text):
        """
        재귀적으로 모든 가중치 문법을 제거하고 순수한 텍스트를 반환합니다.
        처리 형식：
        - (text:weight) → 재귀 처리 text
        - (text) → 재귀 처리 text
        - {text} → 재귀 처리 text
        - text:weight（원시 가중치）→ 추출 text
        - Plain text → 원樣 반환
        """
        import re
        text = text.strip()

        if not text:
            return text

        # 탈피중괄호 {text}
        if text.startswith('{') and text.endswith('}'):
            return self._strip_all_weight(text[1:-1])

        # Remove parentheses
        if text.startswith('(') and text.endswith(')'):
            inner = text[1:-1]
            # 가장 바깥쪽 (깊이 0)의 마지막 콜론 찾기
            colon_pos = self._find_outer_colon(inner)
            if colon_pos > 0:
                # (text:weight) 형식, 텍스트 부분 추출
                inner_text = inner[:colon_pos]
                # inner_text 내재된 중첩이 있을 수 있음, 재귀 처리
                return self._strip_all_weight(inner_text)
            else:
                # (text) 단순 괄호, 재귀적으로 내부 처리
                return self._strip_all_weight(inner)

        # 비디오 워크스테이션 미리보기/결과 캐시 복원）
        bare_match = re.match(r'^(.+?):(-?\d+\.?\d*)\s*$', text)
        if bare_match:
            return self._strip_all_weight(bare_match.group(1))

        # Plain text
        return text

    def _dedup_prompts(self, prompts):
        """
        제거 텍스트만 있는 중복 프롬프트
        전략: 순수 텍스트로 그룹화하고, 동일 그룹에서 우선적으로 가중치 문법이 있는 버전을 유지
        예: aa와 (aa:0.1) 텍스트 모두 aa로 유지합니다 (aa:0.1)
        """
        seen = {}
        result = []
        for p in prompts:
            pure = self._strip_all_weight(p.strip()).lower()
            has_w = self._has_weight(p.strip())
            if pure not in seen:
                seen[pure] = {'text': p, 'has_weight': has_w}
                result.append(p)
            else:
                # 존재함: 현재 버전이 이미 가중치를 가지고 있다면 해당 버전이 가중치를 가지고 있지 않으면 대체
                if has_w and not seen[pure]['has_weight']:
                    idx = result.index(seen[pure]['text'])
                    result[idx] = p
                    seen[pure] = {'text': p, 'has_weight': has_w}
        return result

    def _find_outer_colon(self, text):
        """
        텍스트에서 가장 바깥쪽 (괄호 깊이 0)의 마지막 콜론 위치를 찾습니다.
        반환 위치 인덱스, 찾을 수 없으면 반환-1
        """
        depth = 0
        last_colon = -1
        for i, c in enumerate(text):
            if c == '(' or c == '{':
                depth += 1
            elif c == ')' or c == '}':
                depth -= 1
            elif c == ':' and depth == 0:
                last_colon = i
        return last_colon
    
    @staticmethod
    def _get_artist_dir():
        """artists 디렉토리 경로 가져오기"""
        plugin_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(plugin_dir, "artists")
    
    @classmethod
    def get_artist_file_list(cls):
        """artists 디렉토리 아래의 .txt 파일 목록을 가져옵니다"""
        artist_dir = cls._get_artist_dir()
        if not os.path.isdir(artist_dir):
            return []
        files = []
        for f in sorted(os.listdir(artist_dir)):
            if f.lower().endswith('.txt') and os.path.isfile(os.path.join(artist_dir, f)):
                files.append(f)
        return files
    
    def _read_artist_file_prompts(self, filename):
        """
        artists 폴더에서 지정된 파일을 읽고 프롬프트 목록으로 형식화
        규칙：
        1. 줄 바르기 전에 영어 쉼표가 없으면 쉼표를 추가하세요
        2. 각 쉼표로 구분된 프롬프트 뒤에 공백이 있는지 확인
        3. 쉼표로 구분된 프롬프트 목록으로 돌아가기
        """
        artist_dir = self._get_artist_dir()
        filepath = os.path.join(artist_dir, filename)
        
        # 보안 검사: 경로 교차 방지
        real_path = os.path.realpath(filepath)
        real_dir = os.path.realpath(artist_dir)
        if not real_path.startswith(real_dir + os.sep) and real_path != real_dir:
            print(f"[BatchRandomPrompt] 보안 경고: 경로 교차 검출, 접근 거부 {filename}")
            return []
        
        if not os.path.isfile(filepath):
            return []
        
        # 파일 크기 제한 (최대 5MB）
        if os.path.getsize(filepath) > 5 * 1024 * 1024:
            print(f"[BatchRandomPrompt] 파일이 너무 큽니다, 건너듸: {filename}")
            return []
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
        except (IOError, UnicodeDecodeError) as e:
            print(f"[BatchRandomPrompt] 파일 읽기 실패: {filename}, 오류: {e}")
            return []
        
        if not content.strip():
            return []
        
        # 포맷: 줄 바꿈을 쉼표로 대체(줄 바꿈 전 쉼표가 없는 경우)）
        lines = content.split('\n')
        formatted_parts = []
        for i, line in enumerate(lines):
            stripped = line.rstrip()
            if not stripped:
                continue
            # 마지막 줄이 아니면 줄 끝에 영어 쉼표가 없으면 쉼표를 추가합니다
            if i < len(lines) - 1 and not stripped.rstrip().endswith(','):
                stripped = stripped + ','
            formatted_parts.append(stripped)
        
        formatted_text = ' '.join(formatted_parts)
        
        # 쉼표 뒤에 공백이 있는지 확인하세요
        import re
        formatted_text = re.sub(r',\s*', ', ', formatted_text)
        
        # 콤마로 분할 추출 프롬프트
        prompts = [p.strip() for p in formatted_text.split(',') if p.strip()]
        return prompts


NODE_CLASS_MAPPINGS = {
    "BatchRandomPromptNode": BatchRandomPromptNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BatchRandomPromptNode": "BSK 배치 랜덤 프롬프트",
}
