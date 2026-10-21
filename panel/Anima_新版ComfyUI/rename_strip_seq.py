# SPDX-FileCopyrightText: 2026 灰暗x
# SPDX-License-Identifier: AGPL-3.0-or-later
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""파일 이름에서 확장자 제거 " (n)" 순서(예 " (6)"、" (2)"）。직접 실행 시 적용。"""

import os
import re

# 파일 이름의 끝(확장자 이전) " (숫자)"，예시 " (6)" / " (2)"
PATTERN = re.compile(r" \(\d+\)$")

TARGET_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    if not os.path.isdir(TARGET_DIR):
        print(f"폴더가 없습니다: {TARGET_DIR}")
        return

    renamed = 0
    for name in sorted(os.listdir(TARGET_DIR)):
        base, ext = os.path.splitext(name)
        new_base = PATTERN.sub("", base)
        if new_base == base:
            continue  # 일치하지 않음, 건너듸
        new_name = new_base + ext
        old_path = os.path.join(TARGET_DIR, name)
        new_path = os.path.join(TARGET_DIR, new_name)

        if os.path.exists(new_path):
            print(f"[Skip] Target exists: {name} -> {new_name}")
            continue

        os.rename(old_path, new_path)
        print(f"[이름 변경] {name} -> {new_name}")
        renamed += 1

    if renamed == 0:
        print("이름 변경할 파일 없음。")
    else:
        print(f"\n공동 변경 이름 {renamed} 파일。")


if __name__ == "__main__":
    main()
