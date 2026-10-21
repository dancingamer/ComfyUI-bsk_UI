#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 灰暗x
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
BSK 이미지 동기화 보호 프로그램 (재구성)）
============================

모니터 현지 이미지 파일 변경, 자동 업로드 ComfyUI 및 프런트엔드 갱신。

핵심 논리：
- 연결 후 단일 스레드 폴링 시작, 매초 바인딩 파일 변경 감지。
- 변화 감지 시 즉시 바인딩 검색 및 업로드, 복잡한 댐핑 없음。
- 제로 제3자 의존성 유지, 표준 라이브러리 사용。

사용법：
    python image_watchdog.py                       # GUI 모드
    python image_watchdog.py --url http://127.0.0.1:8188  # 명령줄 모드
"""

import os
import sys
import json
import time
import threading
import argparse
import hashlib
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import urllib.request
import urllib.error
import urllib.parse

# 설정 상수
PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(PLUGIN_DIR, "watch_config.json")

DEFAULT_BASE_URL = "http://127.0.0.1:8188"
DEFAULT_POLL_INTERVAL = 1.0  # 초당 한 번 검사
DEFAULT_REQUEST_TIMEOUT = 30
MAX_RETRIES = 3  # 업로드 실패 재시도 횟수

# 디버그 스위치
VERBOSE = False


def load_config():
    """설정 파일 로드"""
    try:
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        print(f"[설정] 설정 로드 실패: {e}")
    return {}


def save_config(cfg):
    """설정 파일 저장"""
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[설정 저장 실패: {e}")


class ComfyUIClient:
    """ComfyUI HTTP 클라이언트"""
    
    def __init__(self, base_url, timeout=DEFAULT_REQUEST_TIMEOUT):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
    
    def _request(self, path, method="GET", data=None):
        """HTTP 요청 전송"""
        url = f"{self.base_url}{path}"
        headers = {"Content-Type": "application/json"}
        
        req = urllib.request.Request(
            url,
            data=json.dumps(data).encode("utf-8") if data else None,
            headers=headers,
            method=method
        )
        
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as e:
            raise ConnectionError(f"연결할 수 없음 {url}: {e}")
        except json.JSONDecodeError as e:
            raise ValueError(f"응답 형식 오류: {e}")
    
    def get_watch_map(self):
        """바인딩 매핑 가져오기"""
        return self._request("/comfyui_panel/get_watch_map", method="GET")
    
    def notify_image_update(self, node_id, filename):
        """이미지 업데이트 알림"""
        return self._request(
            "/comfyui_panel/notify_image_update",
            method="POST",
            data={"node_id": node_id, "filename": filename}
        )
    
    def get_node_image(self, node_id):
        """노드 현재 그래프 다운로드, 바이트 단위 반환; 노드에 그래프가 없거나 오류 발생 시 반환 None"""
        url = f"{self.base_url}/comfyui_panel/get_node_image?node_id={urllib.parse.quote(str(node_id))}"
        try:
            with urllib.request.urlopen(url, timeout=self.timeout) as resp:
                if resp.status == 200:
                    return resp.read()
                return None
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            raise ConnectionError(f"노드 다운로드 {node_id} 이미지 실패: {e}")
        except urllib.error.URLError as e:
            raise ConnectionError(f"연결할 수 없음 {url}: {e}")
    
    def upload_image(self, filepath, node_id, remote_name):
        """이미지 파일 업로드"""
        boundary = "----bskwatchdogboundary"
        
        with open(filepath, "rb") as f:
            file_data = f.read()
        
        # 구축 multipart/form-data
        crlf = b"\r\n"
        body = b""
        
        # 파일 필드
        body += (f"--{boundary}\r\n").encode()
        body += f'Content-Disposition: form-data; name="image"; filename="{remote_name}"\r\n'.encode()
        body += b"Content-Type: application/octet-stream\r\n\r\n"
        body += file_data
        body += crlf
        
        # filename 필드
        body += (f"--{boundary}\r\n").encode()
        body += b'Content-Disposition: form-data; name="filename"\r\n\r\n'
        body += remote_name.encode()
        body += crlf
        
        # overwrite 필드
        body += (f"--{boundary}\r\n").encode()
        body += b'Content-Disposition: form-data; name="overwrite"\r\n\r\n'
        body += b"true"
        body += crlf
        
        # 종료 경계
        body += (f"--{boundary}--\r\n").encode()
        
        headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
        url = f"{self.base_url}/comfyui_panel/upload_crop"
        
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))


class FileWatcher:
    """파일 변화 감지기"""
    
    def __init__(self, client, download_dir, log_callback=None):
        self.client = client
        self.download_dir = download_dir
        self.log = log_callback or print
        self.running = False
        self.thread = None
        
        # 파일 상태 캐시: 경로 -> (mtime, size, hash)
        self.file_stats = {}
        
        # 바인딩 매핑 캐시
        self.watch_map = {}
        
        # 통계 정보
        self.total_uploads = 0
        self.failed_uploads = 0
    
    def start(self):
        """시작 감지"""
        if self.running:
            self.log("모듈 실행 중", "warn")
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._poll_loop, daemon=True)
        self.thread.start()
        self.log("검출기 시작됨", "ok")
    
    def stop(self):
        """검지 중지"""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        self.log("검출기 중지됨", "info")
    
    def _poll_loop(self):
        """주 핵심 루프"""
        self.log("루프 실행 중", "debug")
        
        while self.running:
            start_time = time.time()
            
            try:
                # 1. 최신 바인딩 쿼리
                self._fetch_watch_map()
                
                # 2. 검출 각 바인딩 파일의 변경
                self._check_all_files()
                
            except Exception as e:
                self.log(f"검출기 루프 예외: {e}", "err")
            
            # 프롬프트 팁 라이브러리 확장 실패, 원본으로 제출
            elapsed = time.time() - start_time
            sleep_time = max(0, DEFAULT_POLL_INTERVAL - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        self.log("루프 종료", "debug")
    
    def _fetch_watch_map(self):
        """바인딩 매핑 가져오기"""
        try:
            response = self.client.get_watch_map()
            new_map = response.get("map", {})
            
            # 새 바인드 있는지 확인
            if new_map != self.watch_map:
                added = set(new_map.keys()) - set(self.watch_map.keys())
                removed = set(self.watch_map.keys()) - set(new_map.keys())
                
                if added:
                    self.log(f"새로 바인딩: {added}", "info")
                if removed:
                    self.log(f"제거 바인딩: {removed}", "info")
                
                self.watch_map = new_map
                
        except Exception as e:
            self.log(f"바인딩 매핑 가져오기 실패: {e}", "err")
    
    def _check_all_files(self):
        """모든 바인딩 노드에 해당하는 로컬 이미지 파일 확인"""
        if not self.watch_map:
            return
        
        for node_id, entry in self.watch_map.items():
            if isinstance(entry, dict):
                remote_name = entry.get("remote_name", "") or f"bsk_source_{node_id}.png"
            else:
                remote_name = f"bsk_source_{node_id}.png"
            local_path = os.path.join(self.download_dir, remote_name)
            try:
                self._check_single_file(node_id, local_path, remote_name)
            except Exception as e:
                self.log(f"노드 검사 {node_id} 실패: {e}", "err")
    
    def _check_single_file(self, node_id, local_path, remote_name):
        """단일 로컬 이미지 파일을 감지합니다 (local_path는 호스트 프로세스에 의해 다운로드 디렉토리에 연결됨).）"""
        # 파일 존재 여부 확인
        if not os.path.exists(local_path):
            self.log(f"파일이 존재하지 않습니다: {local_path} (먼저 보호 프로그램에서 검색 및 다운로드)", "warn")
            return
        
        # 파일 상태 가져오기
        try:
            stat = os.stat(local_path)
            mtime = stat.st_mtime
            size = stat.st_size
        except OSError as e:
            self.log(f"파일 상태 가져오기 불가: {local_path} - {e}", "err")
            return
        
        # 장치 계산 (cuda 일반적으로 더 빠르지만 메모리 사용량 증가)）
        try:
            file_hash = self._compute_file_hash(local_path)
        except Exception as e:
            self.log(f"계산 파일 해시 실패: {e}", "warn")
            file_hash = None
        
        # 파일 변경 여부 확인
        cached_stat = self.file_stats.get(local_path)
        changed = False
        
        if cached_stat:
            cached_mtime, cached_size, cached_hash = cached_stat
            # 해시 비교 우선, 해시 없으면 mtime 및 size
            if file_hash:
                changed = (file_hash != cached_hash)
            else:
                changed = (mtime != cached_mtime or size != cached_size)
        else:
            # 초기 감지, 변경으로 간주 (초기 업로드)）
            changed = True
        
        # 캐시 업데이트
        self.file_stats[local_path] = (mtime, size, file_hash)
        
        # 변화가 있으면 업로드
        if changed:
            self._upload_file(node_id, local_path, remote_name)
    
    def _compute_file_hash(self, filepath):
        """파일 해시 값 계산 (블록 단위 계산)）"""
        hasher = hashlib.md5()
        with open(filepath, "rb") as f:
            while True:
                chunk = f.read(65536)  # 64KB chunks
                if not chunk:
                    break
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def _upload_file(self, node_id, local_path, remote_name):
        """파일 업로드"""
        filename = os.path.basename(local_path)
        self.log(f"변화 감지, 업로드 준비: {filename} -> 노드 {node_id}", "info")
        
        # 재시도 메커니즘
        for attempt in range(MAX_RETRIES):
            try:
                result = self.client.upload_image(local_path, node_id, remote_name)
                
                if result.get("success"):
                    self.total_uploads += 1
                    self.log(
                        f"업로드 성공: {filename} -> {remote_name} (노드 {node_id})",
                        "ok"
                    )
                    
                    # 프런트엔드 갱신 알림
                    try:
                        self.client.notify_image_update(node_id, remote_name)
                    except Exception as e:
                        self.log(f"프런트엔드 갱신 실패: {e}", "warn")
                    
                    return
                else:
                    error = result.get("error", "알 수 없는 오류")
                    self.log(f"업로드 실패 (시도) {attempt + 1}/{MAX_RETRIES}): {error}", "err")
                    
            except Exception as e:
                self.log(f"업로드 예외 (시도) {attempt + 1}/{MAX_RETRIES}): {e}", "err")
            
            # 잠시 후 재시도
            if attempt < MAX_RETRIES - 1:
                time.sleep(1.0)
        
        # 모든 재시도 실패
        self.failed_uploads += 1
        self.log(f"최종 업로드 실패: {filename}", "err")


class BSKWatchdogGUI:
    """BSK 이미지 동기 보존 프로그램 GUI"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("BSK 이미지 동기 보존 프로그램")
        self.root.geometry("700x500")
        
        # 초기화 상태
        self.client = None
        self.watcher = None
        self.connected = False
        
        # 다운로드 디렉토리 동기화 (기본적으로 프로세스 실행 디렉토리 아래) images）
        self.download_dir = os.path.join(os.getcwd(), "images")
        
        # 설정 로드
        self.config = load_config()
        
        # 구축 UI
        self._build_ui()
        
        # 저장된 설정 로드
        self._load_saved_config()
    
    def _build_ui(self):
        """건축 사용자 인터페이스"""
        # 메인 컨테이너
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 연결 영역
        self._build_connection_area(main_frame)
        
        # 파라미터 영역
        self._build_params_area(main_frame)
        
        # 상태 영역
        self._build_status_area(main_frame)
        
        # 바인딩 목록 영역
        self._build_bindings_area(main_frame)
        
        # 로그 영역
        self._build_log_area(main_frame)
    
    def _build_connection_area(self, parent):
        """건축 연결 영역"""
        frame = ttk.LabelFrame(parent, text="ComfyUI 연결", padding=5)
        frame.pack(fill=tk.X, pady=(0, 10))
        
        # 주소 입력
        ttk.Label(frame, text="주소:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.url_var = tk.StringVar(value=DEFAULT_BASE_URL)
        url_entry = ttk.Entry(frame, textvariable=self.url_var, width=50)
        url_entry.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=2)
        
        # 연결 버튼
        self.connect_btn = ttk.Button(frame, text="연결", command=self._on_connect_click)
        self.connect_btn.grid(row=0, column=2, padx=5)
        
        # 상태 레이블
        self.status_var = tk.StringVar(value="○ 연결되지 않음")
        self.status_label = ttk.Label(frame, textvariable=self.status_var, foreground="gray")
        self.status_label.grid(row=0, column=3, padx=10)
        
        frame.columnconfigure(1, weight=1)
    
    def _build_params_area(self, parent):
        """빌드 매개변수 영역"""
        frame = ttk.LabelFrame(parent, text="파라미터 설정", padding=5)
        frame.pack(fill=tk.X, pady=(0, 10))
        
        # 간격 검사
        ttk.Label(frame, text="검사 간격(초)):").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.poll_var = tk.DoubleVar(value=DEFAULT_POLL_INTERVAL)
        poll_spin = ttk.Spinbox(
            frame,
            from_=0.5,
            to=10.0,
            increment=0.5,
            textvariable=self.poll_var,
            width=10
        )
        poll_spin.grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        
        # 시간 초과 설정
        ttk.Label(frame, text="요청 시간 초과(초)):").grid(row=0, column=2, sticky=tk.W, padx=(20, 5))
        self.timeout_var = tk.IntVar(value=DEFAULT_REQUEST_TIMEOUT)
        timeout_spin = ttk.Spinbox(
            frame,
            from_=10,
            to=120,
            increment=10,
            textvariable=self.timeout_var,
            width=10
        )
        timeout_spin.grid(row=0, column=3, sticky=tk.W, padx=5, pady=2)
        
        # 재시도 횟수
        ttk.Label(frame, text="실패 재시도 횟수:").grid(row=0, column=4, sticky=tk.W, padx=(20, 5))
        self.retry_var = tk.IntVar(value=MAX_RETRIES)
        retry_spin = ttk.Spinbox(
            frame,
            from_=1,
            to=10,
            increment=1,
            textvariable=self.retry_var,
            width=10
        )
        retry_spin.grid(row=0, column=5, sticky=tk.W, padx=5, pady=2)
    
    def _build_status_area(self, parent):
        """빌드 상태 영역"""
        frame = ttk.LabelFrame(parent, text="실행 상태", padding=5)
        frame.pack(fill=tk.X, pady=(0, 10))
        
        # 통계 정보
        self.stats_var = tk.StringVar(value="대기 연결...")
        ttk.Label(frame, textvariable=self.stats_var).pack(anchor=tk.W)
    
    def _build_bindings_area(self, parent):
        """동기화 디렉토리 설정 영역을 구성합니다 + 현재 동기화 노드 목록"""
        # 동기화 폴더 설정 영역
        dframe = ttk.LabelFrame(parent, text="동기 디렉토리(가드 프로세스 다운로드/로컬 이미지 모니터링)）", padding=5)
        dframe.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(dframe, text="목차:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.dir_var = tk.StringVar(value=self.download_dir)
        ttk.Entry(dframe, textvariable=self.dir_var, width=40).grid(
            row=0, column=1, sticky=tk.EW, padx=5, pady=2)
        ttk.Button(dframe, text="선택", command=self._choose_dir).grid(row=0, column=2, padx=2)
        ttk.Button(dframe, text="열기", command=self._open_dir).grid(row=0, column=3, padx=2)
        ttk.Button(dframe, text="감지 및 다운로드", command=self._do_detect).grid(row=0, column=4, padx=5)
        dframe.columnconfigure(1, weight=1)
        
        # 현재 동기화 노드 목록 영역
        frame = ttk.LabelFrame(parent, text="현재 동기화 노드", padding=5)
        frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        self.bindings_text = scrolledtext.ScrolledText(frame, height=6, state=tk.DISABLED)
        self.bindings_text.pack(fill=tk.BOTH, expand=True)
    
    def _build_log_area(self, parent):
        """빌드 로그 영역"""
        frame = ttk.LabelFrame(parent, text="로그 출력", padding=5)
        frame.pack(fill=tk.BOTH, expand=True)
        
        # 로그 표시
        self.log_text = scrolledtext.ScrolledText(
            frame,
            height=8,
            state=tk.DISABLED,
            bg="#1e1e1e",
            fg="#d0d0d0",
            font=("Consolas", 9)
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # 초기화 로그 버튼
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Button(btn_frame, text="로그 비움", command=self._clear_log).pack(side=tk.RIGHT)
        ttk.Button(btn_frame, text="로그 복사", command=self._copy_log).pack(side=tk.RIGHT, padx=5)
    
    def _load_saved_config(self):
        """저장된 설정 로드"""
        base_url = self.config.get("base_url", DEFAULT_BASE_URL)
        poll = self.config.get("poll", DEFAULT_POLL_INTERVAL)
        timeout = self.config.get("timeout", DEFAULT_REQUEST_TIMEOUT)
        retry = self.config.get("retry", MAX_RETRIES)
        
        self.url_var.set(base_url)
        self.poll_var.set(poll)
        self.timeout_var.set(timeout)
        self.retry_var.set(retry)
    
    def _save_current_config(self):
        """현재 설정 저장"""
        config = {
            "base_url": self.url_var.get().strip(),
            "poll": self.poll_var.get(),
            "timeout": self.timeout_var.get(),
            "retry": self.retry_var.get()
        }
        save_config(config)
        self.config = config
    
    def _log(self, message, level="info"):
        """추가 로그 메시지"""
        timestamp = time.strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] [{level.upper()}] {message}\n"
        
        # 색상 정의
        colors = {
            "ok": "#7ec699",
            "err": "#ff6b6b",
            "warn": "#ffd166",
            "info": "#cfcfcf",
            "debug": "#888888"
        }
        
        # 로그 삽입
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, formatted_message)
        self.log_text.tag_configure(level, foreground=colors.get(level, "#cfcfcf"))
        self.log_text.tag_add(level, f"{int(self.log_text.index(tk.END).split('.')[0]) - 2}.end", tk.END)
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)
        
        # 업데이트 상태
        self._update_status()
    
    def _clear_log(self):
        """로그 비움"""
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.configure(state=tk.DISABLED)
    
    def _copy_log(self):
        """로그를 클립보드로 복사"""
        content = self.log_text.get(1.0, tk.END)
        self.root.clipboard_clear()
        self.root.clipboard_append(content)
        self._log("로그가 클립보드에 복사됨", "info")
    
    def _update_status(self):
        """상태 업데이트"""
        if self.connected and self.watcher:
            stats = f"연결 | 업로드 성공: {self.watcher.total_uploads} | 실패: {self.watcher.failed_uploads}"
        else:
            stats = "연결되지 않음"
        
        self.stats_var.set(stats)
    
    def _update_bindings_list(self):
        """업데이트 바인딩 목록"""
        if not self.watcher or not self.watcher.watch_map:
            content = "동기화 노드가 없음. 먼저 프론트엔드 이미지 카드에서 '동기화'를 클릭」。"
        else:
            lines = []
            for node_id, entry in self.watcher.watch_map.items():
                if isinstance(entry, dict):
                    local_path = entry.get("local_path") or "（로컬 디렉토리 보호）"
                    remote_name = entry.get("remote_name", f"bsk_source_{node_id}.png")
                else:
                    local_path = "（로컬 디렉토리 보호）"
                    remote_name = f"bsk_source_{node_id}.png"
                
                lines.append(f"노드 {node_id}: {local_path} -> {remote_name}")
            
            content = "\n".join(lines)
        
        self.bindings_text.configure(state=tk.NORMAL)
        self.bindings_text.delete(1.0, tk.END)
        self.bindings_text.insert(1.0, content)
        self.bindings_text.configure(state=tk.DISABLED)
    
    def _choose_dir(self):
        """동기 다운로드 디렉터리 선택"""
        d = filedialog.askdirectory(initialdir=self.download_dir or os.getcwd())
        if d:
            self.download_dir = d
            self.dir_var.set(d)
            if self.watcher:
                self.watcher.download_dir = d
    
    def _open_dir(self):
        """파일 관리자에서 동기화 폴더 열기"""
        d = self.dir_var.get().strip() or self.download_dir
        if not os.path.isdir(d):
            os.makedirs(d, exist_ok=True)
        try:
            os.startfile(d)
        except Exception:
            import subprocess
            if sys.platform == "darwin":
                subprocess.run(["open", d])
            else:
                subprocess.run(["xdg-open", d])
    
    def _do_detect(self):
        """동기화된 노드 확인, 로컬 디렉토리로 다운로드하고 모니터링 시작"""
        if not self.connected or not self.client:
            messagebox.showwarning("알림", "먼저 연결해주세요 ComfyUI")
            return
        try:
            resp = self.client.get_watch_map()
            m = resp.get("map", {})
        except Exception as e:
            self._log(f"동기화 노드 가져오기 실패: {e}", "err")
            return
        if not m:
            self._log("동기화된 노드가 없습니다. 먼저 프론트엔드 이미지 카드에서 '동기화'를 클릭하십시오.」", "warn")
            return
        os.makedirs(self.download_dir, exist_ok=True)
        for node_id, entry in m.items():
            if isinstance(entry, dict):
                remote_name = entry.get("remote_name", "") or f"bsk_source_{node_id}.png"
            else:
                remote_name = f"bsk_source_{node_id}.png"
            local_path = os.path.join(self.download_dir, remote_name)
            try:
                data = self.client.get_node_image(node_id)
                if data is None:
                    self._log(f"노드 {node_id} 현재 그림 없음, 건너듸 (먼저 카드에 이미지를 업로드)）", "warn")
                    continue
                with open(local_path, "wb") as f:
                    f.write(data)
                # 기본값 지문이 기록되어 첫 번째 판독 오류로 인한 불필요한 업로드를 방지합니다.
                if self.watcher is not None:
                    stat = os.stat(local_path)
                    self.watcher.file_stats[local_path] = (
                        stat.st_mtime, stat.st_size, self.watcher._compute_file_hash(local_path)
                    )
                self._log(f"다운로드된 노드 {node_id} 그림 -> {local_path}", "ok")
            except Exception as e:
                self._log(f"노드 다운로드 {node_id} 이미지 실패: {e}", "err")
        self._update_bindings_list()
    
    def _on_connect_click(self):
        """연결/해제 버튼 클릭 처리"""
        if self.connected:
            self._disconnect()
        else:
            self._connect()
    
    def _connect(self):
        """연결 생성"""
        base_url = self.url_var.get().strip()
        if not base_url:
            messagebox.showwarning("알림", "请입력 ComfyUI 주소")
            return
        
        self._log(f"연결 중 {base_url}...", "info")
        self.connect_btn.configure(state=tk.DISABLED)
        
        try:
            # 클라이언트 생성
            self.client = ComfyUIClient(
                base_url,
                timeout=self.timeout_var.get()
            )
            
            # 연결 테스트
            response = self.client.get_watch_map()
            if "map" not in response:
                raise ValueError("유효하지 않은 서버 응답")
            
            # 연결 성공
            self.connected = True
            self._log("연결 성공", "ok")
            
            # 업데이트 UI
            self.status_var.set("● 연결")
            self.status_label.configure(foreground="green")
            self.connect_btn.configure(text="끊다", state=tk.NORMAL)
            
            # 설정 저장
            self._save_current_config()
            
            # 모니터 생성 및 시작
            self.watcher = FileWatcher(
                self.client,
                self.download_dir,
                log_callback=lambda msg, level: self._log(msg, level)
            )
            self.watcher.start()
            
            # 정기 업데이트 바인딩 목록
            self._start_bindings_update()
            
        except Exception as e:
            self._log(f"연결 실패: {e}", "err")
            self.status_var.set("○ 연결 실패")
            self.status_label.configure(foreground="red")
            self.connect_btn.configure(state=tk.NORMAL)
    
    def _disconnect(self):
        """연결 끊기"""
        if self.watcher:
            self.watcher.stop()
            self.watcher = None
        
        self.client = None
        self.connected = False
        
        # 업데이트 UI
        self.status_var.set("○ 연결되지 않음")
        self.status_label.configure(foreground="gray")
        self.connect_btn.configure(text="연결")
        
        self._log("연결 끊기", "info")
        self._update_status()
    
    def _start_bindings_update(self):
        """바인딩 목록 정기 업데이트 시작"""
        if not self.connected:
            return
        
        self._update_bindings_list()
        self.root.after(2000, self._start_bindings_update)
    
    def run(self):
        """실행 애플리케이션"""
        # 창 닫기 처리 설정
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        
        # 자동 연결 시도 (저장된 경우 URL）
        if self.config.get("base_url"):
            self.root.after(500, self._auto_connect)
        
        self.root.mainloop()
    
    def _auto_connect(self):
        """자동 연결"""
        if not self.connected:
            self._connect()
    
    def _on_close(self):
        """창문 닫힘 처리"""
        if messagebox.askokcancel("종료", "BSK 이미지 동기 보호 프로그램을 종료하시겠습니까?？"):
            if self.watcher:
                self.watcher.stop()
            self.root.destroy()


def run_cli(base_url, poll, timeout, retry):
    """명령줄 모드"""
    print("=" * 60)
    print("BSK 이미지 동기화 보호 프로그램 (명령줄 모드）")
    print(f"  ComfyUI 주소: {base_url}")
    print(f"  간격 검사: {poll} 초")
    print(f"  요청 시간 초과: {timeout} 초")
    print(f"  실패 재시도: {retry} 회")
    print("=" * 60)
    
    # 클라이언트 생성
    client = ComfyUIClient(base_url, timeout)
    
    # 연결 테스트
    try:
        response = client.get_watch_map()
        print(f"✓ ComfyUI에 연결됨, 현재 바인딩 수: {len(response.get('map', {}))}")
    except Exception as e:
        print(f"✗ 연결할 수 없음 ComfyUI: {e}")
        sys.exit(1)
    
    # 생성 감시자
    watcher = FileWatcher(client, log_callback=lambda msg, level: print(f"[{level}] {msg}"))
    watcher.start()
    
    print("감시 중...（Ctrl+C 종료）")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n중지 중...")
        watcher.stop()
        print("중지됨")


def main():
    """프로그램 진입"""
    parser = argparse.ArgumentParser(description="BSK 이미지 동기 보존 프로그램")
    parser.add_argument("--url", help="ComfyUI 주소 (제공 시 명령줄 모드로 진입)）")
    parser.add_argument("--poll", type=float, default=DEFAULT_POLL_INTERVAL, help="검사 간격(초))")
    parser.add_argument("--timeout", type=int, default=DEFAULT_REQUEST_TIMEOUT, help="요청 시간 초과(초))")
    parser.add_argument("--retry", type=int, default=MAX_RETRIES, help="실패 재시도 횟수")
    
    args = parser.parse_args()
    
    if args.url:
        run_cli(args.url, args.poll, args.timeout, args.retry)
    else:
        app = BSKWatchdogGUI()
        app.run()


if __name__ == "__main__":
    main()
