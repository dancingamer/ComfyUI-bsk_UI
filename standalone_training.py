#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 灰暗x
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
훈련 도구 독립 실행기 시작
ComfyUI에서 분리되어 사용자 정의 HTTP 서버에서 훈련 도구 프런트를 실행합니다.

사용법:
  python standalone_training.py                    # 기본 포트 8189
  python standalone_training.py --port 9090        # 포트 지정
  python standalone_training.py --sd-scripts /path/to/sd-scripts
  python standalone_training.py --comfyui-root /path/to/ComfyUI
  python standalone_training.py --no-browser       # 브라우저 자동 열기 안 함
"""

import argparse
import asyncio
import os
import sys
import webbrowser
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='훈련 도구 독립 실행기 시작')
    parser.add_argument('--port', type=int, default=8189, help='HTTP 포트 (기본): 8189)')
    parser.add_argument('--host', default='127.0.0.1', help='리스닝 주소 (기본값: 127.0.0.1)')
    parser.add_argument('--sd-scripts', default='', help='sd-scripts 목차 경로')
    parser.add_argument('--comfyui-root', default='', help='ComfyUI 루트 디렉토리 (모델 검색용))')
    parser.add_argument('--train-data', default='', help='훈련 데이터 루트 디렉토리')
    parser.add_argument('--loras-dir', default='', help='loras 디렉토리 출력')
    parser.add_argument('--no-browser', action='store_true', help='브라우저 자동 열기 안 함')
    args = parser.parse_args()

    # 환경 변수 설정 (training_api.py가 읽음)）
    if args.comfyui_root:
        os.environ['COMFYUI_ROOT'] = args.comfyui_root
    if args.sd_scripts:
        os.environ['SD_SCRIPTS_DIR'] = args.sd_scripts
    if args.train_data:
        os.environ['TRAIN_DATA_DIR'] = args.train_data
    if args.loras_dir:
        os.environ['TRAINING_LORAS_DIR'] = args.loras_dir

    # training_api.py가 가져올 수 있도록 보장
    plugin_dir = Path(__file__).parent.resolve()
    sys.path.insert(0, str(plugin_dir))

    from aiohttp import web
    from training_api import setup_training_api

    app = web.Application()

    # 훈련 API 경로 등록
    setup_training_api(app=app)

    # 정적 파일 서비스: 제공 프런트 엔드 페이지
    async def serve_index(request):
        """독립 훈련 페이지 제공"""
        html_path = plugin_dir / 'training_standalone.html'
        if html_path.exists():
            return web.FileResponse(str(html_path))
        return web.Response(text='훈련 페이지를 찾을 수 없습니다. training_standalone.html이 존재하는지 확인하십시오.', status=404)

    app.router.add_route('GET', '/', serve_index)
    app.router.add_route('GET', '/training', serve_index)

    # 정적 파일 라우팅: /js/*, /assets/* 등
    js_dir = plugin_dir / 'js'
    if js_dir.exists():
        app.router.add_static('/js', str(js_dir))

    assets_dir = plugin_dir / 'assets'
    if assets_dir.exists():
        app.router.add_static('/assets', str(assets_dir))

    # 시작
    url = f'http://{args.host}:{args.port}'
    print('=' * 50)
    print('훈련 도구 독립 모드')
    print('=' * 50)
    print(f'주소: {url}')
    print(f'플러그인 디렉토리: {plugin_dir}')
    print(f'ComfyUI 루트 디렉토리: {os.environ.get("COMFYUI_ROOT", "설정되지 않음")}')
    print(f'sd-scripts 목차: {os.environ.get("SD_SCRIPTS_DIR", "자동 검색")}')
    print(f'훈련 데이터 디렉토리: {os.environ.get("TRAIN_DATA_DIR", "기본")}')
    print('=' * 50)

    if not args.no_browser:
        # 브라우저 시작 지연, 서버 시작 대기
        async def open_browser(app):
            await asyncio.sleep(0.5)
            webbrowser.open(url)
        app.on_startup.append(open_browser)

    web.run_app(app, host=args.host, port=args.port)


if __name__ == '__main__':
    main()
