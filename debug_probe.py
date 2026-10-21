# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 灰暗x
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
ComfyUI-bsk_UI 디버그 핀 (Debug Probe)
======================================

진단용"VSCode pty가 끊어지고, 이미지가 미리보기되지 않으며, 구간이 지나면 복구"문제。

디자인 원칙:
1. 독립적인 백그라운드 스레드에서 샘플링을 수행하며, ComfyUI의 aiohttp 이벤트 루프에 의존하지 않습니다.
   (메인 서비스가 멈춰도 핀은 계속 실행되며, 멈춘 순간 상태를 기록합니다)
2. 독립 HTTP 서비스 (포트 8229), 표준 라이브러리 사용 http.server
   (메인 서비스가 멈추어도 http://localhost:8229/를 통해 실시간 상태 확인 가능)
3. 가벼움: 카운터는 로크리스 또는 매우 짧은 로크 사용, 샘플링 간격 2초, 새로운 리소스 소비 도입 안 함
4. 예외를 모두 섭취하고, 메인 프로세스에 영향을 주지 않습니다

샘플링 지표:
- 프로세스 FD 수 (/proc/self/fd)        → FD 누수 지표
- Thread count (/proc/self/status)         → 스레드 누수 지표
- TCP ESTABLISHED 연결 수             → 연결 누수 지표
- CPU% / RSS 메모리                    → 리소스 사용량
- 각 엔드포인트 요청 횟수 + 속도              → 요청 빈도 지표
- runtime_stats 최근 응답 지난 초     → 이벤트 루프 탐지
  (약 > 10s 설명 이벤트 루프가 멈췄습니다)

로그: /workspace/출력/debug_probe.jsonl (JSONL, 최근 5000줄 유지))

사용:
  브라우저 방문 http://localhost:8229/         → 실시간 캡처 (HTML)
  브라우저 방문 http://localhost:8229/snapshot → 실시간 캡처 (JSON)
  브라우저 방문 http://localhost:8229/tail?n=100 → 최근 N개의 로그
  curl http://localhost:8229/snapshot | python -m json.tool
"""

import os
import sys
import time
import json
import threading
import socket
from datetime import datetime
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from collections import deque

# ========== 전역 상태 (스레드 안전, 극단적인 잠금 사용)) ==========

_lock = threading.Lock()

# 엔드포인트 요청 카운터: {endpoint: {"total": int, "first_ts": float, "last_ts": float}}
_counters = {}

# 샘플링 히스토리 (최근 5000개 유지))
_history = deque(maxlen=5000)

# 최근 샘플링 샷
_latest = None

# 표시되지 않음: Python은 기본적으로 cp437로 디코딩하며, cp437 디코딩은 완전 역영역이며, 사용.)
_last_thread_details = None

# 스레드 상세 수집 구성 ( /thread_config 인터페이스를 통해 동적으로 수정 가능))
# thread_enabled: 스위치, 기본적으로 꺼짐(False), 필요할 때 수동으로 켜기
# thread_interval_s: 수집 간격(초), 기본 10초, 범위 2~300
_thread_config = {
    "enabled": False,
    "interval_s": 10,
}
_thread_config_lock = threading.Lock()
# 마지막 수집 스레드 세부 정보의 타임스탬프, 간격에 따라 다시 수집할지 여부를 판단하는 데 사용합니다.
_last_thread_sample_ts = 0.0

# 센서가 시작되었습니까?
_started = False

# 로그 파일 경로
_LOG_FILE = os.path.join(
    os.environ.get("COMFYUI_OUTPUT", "/workspace/출력"),
    "debug_probe.jsonl",
)

# HTTP 포트
# 기본 8229:8199는 ComfyUI 설치기(bsk_comfyui_installer.py)의 기본 포트입니다.，
# 두 개 모두 로컬에 실행되며, 동일한 포트를 사용하면 충돌합니다 (후발이 지게 됩니다)）。
_PROBE_PORT = int(os.environ.get("BSK_DEBUG_PROBE_PORT", "8229"))


def _ensure_log_dir():
    try:
        d = os.path.dirname(_LOG_FILE)
        if d and not os.path.exists(d):
            os.makedirs(d, exist_ok=True)
    except Exception:
        pass


# ========== API (__init__.py 엔드포인트 호출용)) ==========

def incr(endpoint):
    """특정 엔드포인트의 요청 수를 늘립니다. 매우 빠른 경로로 설계하고 실패하면 무시합니다。

    __init__.py의 API 엔드포인트에서 호출합니다.: debug_probe.incr("runtime_stats")
    """
    try:
        now = time.time()
        with _lock:
            c = _counters.get(endpoint)
            if c is None:
                c = {"total": 0, "first_ts": now, "last_ts": now}
                _counters[endpoint] = c
            c["total"] += 1
            c["last_ts"] = now
    except Exception:
        pass


def mark_loop_alive():
    """마크 이벤트 루프 생존 (실행 시간 통계 엔드포인트 성공 반환 전 호출))。

    탐침이 이 타임스탬프를 사용하여 이벤트 루프가 멈추었는지 판단:
    mark 후 10초 이상 새로운 mark가 없으면 이벤트 루프가 차단된 것으로 간주。
    """
    incr("__loop_alive__")


# ========== 시스템 지표 샘플링 (Linux /proc, 의존성 없음) psutil) ==========

def _read_fd_count():
    """현재 프로세스에서 열린 FD 수 읽기"""
    try:
        return len(os.listdir("/proc/self/fd"))
    except Exception:
        return -1


def _read_thread_count():
    """현재 프로세스 스레드 수"""
    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("Threads:"):
                    return int(line.split()[1])
    except Exception:
        pass
    return -1


def _read_rss_mb():
    """현재 프로세스 RSS 메모리 읽기 (MB)"""
    try:
        # /proc/self/statm 필드: size resident shared text lib data dt (단위: 페이지))
        with open("/proc/self/statm", "r") as f:
            fields = f.read().split()
            resident_pages = int(fields[1])
            page_size = os.sysconf("SC_PAGE_SIZE")
            return resident_pages * page_size / 1024 / 1024  # MB
    except Exception:
        pass
    return -1


def _read_cpu_percent(prev_times, prev_ts):
    """CPU% 계산 프로세스 ( /proc/self/stat을 기반으로) utime+stime)

    반환 (cpu_percent, new_total_jiffies, new_ts):
      처음 호출 cpu_percent=0.0
      후속 호출 반환 (현재 샘플링 주기의) 단일 코어 CPU 비율
    """
    try:
        with open("/proc/self/stat", "r") as f:
            fields = f.read().split()
            # utime=필드13, stime=필드 14 (인덱스는 0부터 시작))
            utime = int(fields[13])
            stime = int(fields[14])
        total_jiffies = utime + stime
        hz = os.sysconf("SC_CLK_TCK")
        now = time.time()
        if prev_times is not None and prev_ts > 0:
            dt_jiffies = total_jiffies - prev_times
            dt_sec = now - prev_ts
            if dt_sec > 0:
                pct = (dt_jiffies / hz) / dt_sec * 100.0  # 싱글 코어 퍼센트
                return pct, total_jiffies, now
        return 0.0, total_jiffies, now
    except Exception:
        pass
    return 0.0, prev_times, prev_ts


def _read_tcp_established():
    """현재 프로세스의 ESTABLISHED TCP 연결 수 (근사치, 통해) /proc/net/tcp + fd 역검색)"""
    try:
        # 방법: /proc/self/fd에서 socket의 개수(대략적, 완전 정확하지 않지만 유용))
        count = 0
        fd_dir = "/proc/self/fd"
        for fd_name in os.listdir(fd_dir):
            try:
                target = os.readlink(os.path.join(fd_dir, fd_name))
                if target.startswith("socket:"):
                    count += 1
            except Exception:
                pass
        return count
    except Exception:
        return -1


def _read_thread_details():
    """수집 스레드 상세 정보: 이름별 통계 그룹화 + threading 모듈 생성된 스레드 나열

    반환:
      {
        "by_name": {"ThreadName": count, ...},   # 이름별 그룹 계산
        "named_total": int,                       # threading 모듈 보이는 스레드 수
        "native_total": int,                      # /proc 원래 스레드 수를 읽습니다
        "top_names": [("name", count), ...]       # 가장 많은 수의 상위 15개 이름
      }
    """
    result = {"by_name": {}, "named_total": 0, "native_total": 0, "top_names": []}
    try:
        # 1. threading 패널에 보이는 스레드 (이름 있음))
        by_name = {}
        for t in threading.enumerate():
            name = t.name or "(unnamed)"
            by_name[name] = by_name.get(name, 0) + 1
        result["by_name"] = by_name
        result["named_total"] = len(by_name)

        # 2. 원시 스레드 총수 (/proc, C 확장으로 생성된 Python 이름 없는 스레드 포함))
        result["native_total"] = _read_thread_count()

        # 3. top 15
        sorted_names = sorted(by_name.items(), key=lambda x: -x[1])[:15]
        result["top_names"] = [{"name": n, "count": c} for n, c in sorted_names]
    except Exception:
        pass
    return result


# ========== 샘플링 메인 루프 ==========

def _sample_once(prev_cpu_times, prev_cpu_ts, sample_idx=0):
    """샘플링 한 번 실행, 반환 (snapshot_dict, new_prev_cpu_times, new_prev_cpu_ts)

    sample_idx: 샘플 번호, 저주파 지표 수집 빈도 제어 사용
    """
    now = time.time()

    # 데이터 수집 지표
    fd_count = _read_fd_count()
    thread_count = _read_thread_count()
    rss_mb = _read_rss_mb()
    tcp_sock = _read_tcp_established()

    cpu_percent, new_cpu_times, new_cpu_ts = _read_cpu_percent(prev_cpu_times, prev_cpu_ts)

    # 데이터 포인트 카운트 샷 캡처
    with _lock:
        counters_snap = {}
        for ep, c in _counters.items():
            counters_snap[ep] = {
                "total": c["total"],
                "first_ts": c["first_ts"],
                "last_ts": c["last_ts"],
                "last_ago": round(now - c["last_ts"], 2),
            }
        # 계산 각 엔드포인트 최근 60초 속도
        # (total 차이로 근사값을 사용하고, 정확한 속도는 히스토리를 기반으로 합니다. 여기서는 단순화합니다.)
        counters_snapshot = dict(counters_snap)

    # 이벤트 루프 활성화: __loop_alive__ last_ago
    loop_alive = counters_snapshot.get("__loop_alive__", {})
    loop_last_ago = loop_alive.get("last_ago", 999999)
    loop_alive_flag = loop_last_ago < 10.0  # 10s 응답이 생존한다고 간주합니다

    # 스레드 상세 정보: _thread_config에 의해 제어됩니다
    # - enabled=False: 수집하지 않음 (thread_details=None, HTML 닫기 힌트 표시)
    # - enabled=True: interval_s 간격으로 샘플링 (간격 미달 시 None 반환, 루프가 캐시로 채움))
    thread_details = None
    with _thread_config_lock:
        td_enabled = _thread_config["enabled"]
        td_interval = _thread_config["interval_s"]
    if td_enabled and (now - _last_thread_sample_ts) >= td_interval:
        thread_details = _read_thread_details()
        # 참고: _last_thread_sample_ts 업데이트는 호출자에서 수행되며, 여기서는 단지 판단합니다.

    snapshot = {
        "ts": now,
        "ts_str": datetime.now().strftime("%H:%M:%S"),
        "fd": fd_count,
        "threads": thread_count,
        "rss_mb": round(rss_mb, 1) if rss_mb >= 0 else -1,
        "cpu_pct": round(cpu_percent, 1) if isinstance(cpu_percent, (int, float)) else -1,
        "tcp_sock": tcp_sock,
        "loop_alive": loop_alive_flag,
        "loop_last_ago_s": round(loop_last_ago, 1) if loop_last_ago < 999999 else -1,
        "counters": counters_snapshot,
    }
    if thread_details is not None:
        snapshot["thread_details"] = thread_details

    return snapshot, new_cpu_times, new_cpu_ts


def _sampler_loop():
    """샘플링 메인 루프 (데몬 스레드))"""
    global _latest, _last_thread_details, _last_thread_sample_ts
    _ensure_log_dir()
    prev_cpu_times = None
    prev_cpu_ts = 0.0

    # 이전 라운드의 total, 속도 계산에 사용
    prev_totals = {}

    sample_idx = 0
    while True:
        try:
            snapshot, prev_cpu_times, prev_cpu_ts = _sample_once(
                prev_cpu_times, prev_cpu_ts, sample_idx
            )
            sample_idx += 1

            # 스레드 상세 처리:
            # - 활성화되고 이번 라운드 수집 → 캐시 업데이트 + 타임스탬프
            # - 시작되었으나 이번 라운드 간격 미만 → 재생 캐시로 채우기)
            # - 닫기 → 캐시 삭제, 반환하지 않음
            with _thread_config_lock:
                td_enabled = _thread_config["enabled"]

            if snapshot.get("thread_details") is not None:
                _last_thread_details = snapshot["thread_details"]
                _last_thread_sample_ts = snapshot["ts"]
            elif td_enabled and _last_thread_details is not None:
                snapshot["thread_details"] = _last_thread_details
            elif not td_enabled:
                # 스위치 켜고 끄기: 캐시 삭제, 캡처에 이전 데이터 포함되지 않도록 합니다.
                _last_thread_details = None
                snapshot.pop("thread_details", None)

            # 각 엔드포인트의 최근 샘플링 주기의 속도 (초/초))
            now = snapshot["ts"]
            with _lock:
                rate_info = {}
                for ep, c in _counters.items():
                    prev_total = prev_totals.get(ep, 0)
                    delta = c["total"] - prev_total
                    rate_info[ep] = round(delta / 2.0, 2)  # 2s 샘플링 간격
                    prev_totals[ep] = c["total"]
            snapshot["rates_per_s"] = rate_info

            with _lock:
                _latest = snapshot
                _history.append(snapshot)

            # JSONL 로그 쓰기 (추가 모드))
            try:
                with open(_LOG_FILE, "a", encoding="utf-8") as f:
                    f.write(json.dumps(snapshot, ensure_ascii=False) + "\n")
            except Exception:
                pass

        except Exception as e:
            # Probe 자체 오류는 메인 프로세스에 영향을 주지 않습니다.
            try:
                sys.stderr.write(f"[debug_probe] sampler error: {e}\n")
            except Exception:
                pass

        time.sleep(2.0)


# ========== 독립 HTTP 서비스 (포트 8229) ==========

class _ProbeHandler(BaseHTTPRequestHandler):
    """스캐너 HTTP 요청 프로세서"""

    def log_message(self, *args):
        # 조용히, 액세스 로그 출력을 방지합니다 (오염 방지) stdout/pty)
        pass

    def _send_json(self, data, code=200):
        body = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html):
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        """POST 요청 처리 (현재만 /thread_config)"""
        try:
            if self.path.startswith("/thread_config"):
                return self._handle_thread_config()
            self._send_json({"error": "unknown POST endpoint"}, 404)
        except Exception as e:
            self._send_json({"error": str(e)}, 500)

    def do_GET(self):
        try:
            if self.path.startswith("/snapshot"):
                with _lock:
                    snap = _latest.copy() if _latest else {}
                self._send_json(snap)
                return
            if self.path.startswith("/tail"):
                # /tail?n=100
                n = 100
                if "?" in self.path:
                    import urllib.parse
                    qs = urllib.parse.parse_qs(self.path.split("?", 1)[1])
                    try:
                        n = int(qs.get("n", ["100"])[0])
                    except Exception:
                        n = 100
                with _lock:
                    items = list(_history)[-n:]
                self._send_json(items)
                return

            if self.path.startswith("/reset"):
                with _lock:
                    global _counters
                    _counters = {}
                self._send_json({"success": True, "msg": "counters reset"})
                return

            if self.path.startswith("/thread_config"):
                return self._handle_thread_config()

            # 기본: HTML 실시간 패널
            self._send_html(self._render_html())
        except Exception as e:
            self._send_json({"error": str(e)}, 500)

    def _handle_thread_config(self):
        """처리 스레드 상세 설정: GET 현재 설정 반환, POST 설정 수정

        GET  /thread_config
        POST /thread_config  body: {"enabled": true/false, "interval_s": 10}
        """
        import urllib.parse
        if self.command == "POST":
            try:
                length = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(length) if length > 0 else b"{}"
                data = json.loads(raw.decode("utf-8") or "{}")
            except Exception as e:
                self._send_json({"success": False, "error": f"invalid json: {e}"}, 400)
                return

            with _thread_config_lock:
                if "enabled" in data:
                    _thread_config["enabled"] = bool(data["enabled"])
                if "interval_s" in data:
                    try:
                        v = int(data["interval_s"])
                        # 제한 범위 2~300 초
                        _thread_config["interval_s"] = max(2, min(300, v))
                    except Exception:
                        pass
                resp = dict(_thread_config)
            # 닫을 때 캐시를 지우세요
            if not resp["enabled"]:
                global _last_thread_details
                _last_thread_details = None
            self._send_json({"success": True, "config": resp})
            return

        # GET: 반환 현재 설정
        with _thread_config_lock:
            resp = dict(_thread_config)
        self._send_json({"success": True, "config": resp})
        return

    def _render_html(self):
        with _lock:
            snap = _latest.copy() if _latest else {}
            hist = list(_history)[-60:]  # 최근 60개 (2분))
        if not snap:
            return "<html><body><h3>Debug Probe 샘플링 시작 대기... (2초 후 갱신))</h3><meta http-equiv='refresh' content='2'></body></html>"

        # 트렌드 데이터
        fd_series = [h.get("fd", 0) for h in hist]
        thr_series = [h.get("threads", 0) for h in hist]
        sock_series = [h.get("tcp_sock", 0) for h in hist]
        loop_series = [1 if h.get("loop_alive") else 0 for h in hist]

        # 엔드포인트 카운터
        counters = snap.get("counters", {})
        rates = snap.get("rates_per_s", {})
        rows = []
        for ep in sorted(counters.keys()):
            if ep == "__loop_alive__":
                continue
            c = counters[ep]
            r = rates.get(ep, 0)
            rows.append(
                f"<tr><td>{ep}</td><td>{c['total']}</td><td>{r}</td>"
                f"<td>{c['last_ago']}</td></tr>"
            )

        loop_color = "#4ade80" if snap.get("loop_alive") else "#f87171"
        loop_text = "생존" if snap.get("loop_alive") else f"<b>멈춤? 마지막 응답 시점 {snap.get('loop_last_ago_s','?')}s</b>"

        # 스레드 상세 카드: _thread_config에 의해 제어됨 (스위치) + 간격)
        with _thread_config_lock:
            td_enabled = _thread_config["enabled"]
            td_interval = _thread_config["interval_s"]
        td = snap.get("thread_details")

        if td_enabled and td and td.get("top_names"):
            # 켜짐 + 데이터가 있습니다: 테이블 표시 + 닫기 버튼 + 간격 설정
            named = td.get("named_total", 0)
            native = td.get("native_total", 0)
            unnamed = native - named if native >= named else 0
            thread_rows = "".join(
                f"<tr><td>{item['name']}</td><td>{item['count']}</td></tr>"
                for item in td["top_names"]
            )
            thread_card = f"""
<div class="card">
  <h3 style="display:flex;justify-content:space-between;align-items:center;">
    <span>스레드 상세 정보 (실행 중, 각 {td_interval}s 새로고침)</span>
    <span>
      <input type="number" id="td-interval" value="{td_interval}" min="2" max="300" step="1"
             style="width:60px;background:#1e1e1e;color:#ddd;border:1px solid #444;border-radius:4px;padding:2px 4px;" title="새로 고침 간격(초))">
      <button onclick="tdApplyInterval()" style="background:#3a3a3a;color:#ddd;border:1px solid #555;border-radius:4px;padding:3px 8px;cursor:pointer;">적용</button>
      <button onclick="tdToggle(false)" style="background:#f87171;color:#fff;border:none;border-radius:4px;padding:3px 8px;cursor:pointer;margin-left:4px;">닫기</button>
    </span>
  </h3>
  <p style="color:#888;font-size:12px">Naming Thread: {named} | 원시 스레드 총수: {native} | 이름 없음(C 확장): {unnamed}</p>
  <table>
    <tr><th>스레드 이름</th><th>수량</th></tr>
    {thread_rows}
  </table>
  <p style="color:#888;font-size:11px;margin-top:8px">일반적인 출처: aiohttp 작업 스레드 / ThreadPoolExecutor / torch 추론 스레드 / watchdog / bsk-*</p>
</div>"""
        elif td_enabled:
            # 켜져 있지만 현재 데이터가 없음 (초기 샘플링 대기))
            thread_card = f"""
<div class="card">
  <h3 style="display:flex;justify-content:space-between;align-items:center;">
    <span>스레드 상세 정보 (실행 중, 각 {td_interval}s 새로고침)</span>
    <span>
      <button onclick="tdToggle(false)" style="background:#f87171;color:#fff;border:none;border-radius:4px;padding:3px 8px;cursor:pointer;">닫기</button>
    </span>
  </h3>
  <p style="color:#888;font-size:12px">초기 샘플링 대기 (약 {td_interval}s 내에 나타나)...</p>
</div>"""
        else:
            # 닫기: 표시 켜기 버튼 + 간격 설정
            thread_card = f"""
<div class="card">
  <h3 style="display:flex;justify-content:space-between;align-items:center;">
    <span>스레드 상세 <span style="color:#888;font-size:12px">(기본적으로 닫힘)</span></span>
    <span>
      <input type="number" id="td-interval" value="{td_interval}" min="2" max="300" step="1"
             style="width:60px;background:#1e1e1e;color:#ddd;border:1px solid #444;border-radius:4px;padding:2px 4px;" title="새로 고침 간격(초))">
      <button onclick="tdApplyInterval()" style="background:#3a3a3a;color:#ddd;border:1px solid #555;border-radius:4px;padding:3px 8px;cursor:pointer;">애플리케이션 간격</button>
      <button onclick="tdToggle(true)" style="background:#4ade80;color:#1e1e1e;border:none;border-radius:4px;padding:3px 8px;cursor:pointer;margin-left:4px;">켜짐</button>
    </span>
  </h3>
  <p style="color:#888;font-size:12px">클릭"켜짐"샘플링 스레드 상세 정보 시작 (약간 오버헤드 발생, 기본값) {td_interval}s 한 번)</p>
</div>"""

        return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<meta http-equiv="refresh" content="3">
<title>BSK Debug Probe</title>
<style>
  body {{ font-family: monospace; margin: 20px; background:#1e1e1e; color:#ddd; }}
  .card {{ background:#2a2a2a; padding:15px; border-radius:8px; margin-bottom:15px; }}
  .metric {{ display:inline-block; margin:0 20px 10px 0; }}
  .metric .v {{ font-size:28px; font-weight:bold; }}
  .metric .l {{ font-size:12px; color:#888; }}
  table {{ border-collapse:collapse; width:100%; }}
  td,th {{ border:1px solid #444; padding:6px 10px; text-align:left; }}
  th {{ background:#333; }}
  .spark {{ height:40px; width:100%; }}
  .warn {{ color:#fbbf24; }}
  .ok {{ color:#4ade80; }}
  .bad {{ color:#f87171; }}
</style></head><body>
<h2>BSK UI Debug Probe</h2>
<p style="color:#888;font-size:12px">샘플링 시간: {snap.get('ts_str','?')} | 자동 3초 갱신 | 포트 {_PROBE_PORT}</p>

<div class="card">
  <div class="metric"><div class="v" style="color:{'#fbbf24' if snap.get('fd',0)>500 else '#4ade80'}">{snap.get('fd','?')}</div><div class="l">FD 숫자 (&gt;800 경고 누출)</div></div>
  <div class="metric"><div class="v" style="color:{'#fbbf24' if snap.get('threads',0)>100 else '#4ade80'}">{snap.get('threads','?')}</div><div class="l">스레드 수 (&gt;100 주의)</div></div>
  <div class="metric"><div class="v" style="color:{'#fbbf24' if snap.get('tcp_sock',0)>100 else '#4ade80'}">{snap.get('tcp_sock','?')}</div><div class="l">socket FD 수</div></div>
  <div class="metric"><div class="v">{snap.get('rss_mb','?')}</div><div class="l">RSS 메모리 (MB)</div></div>
  <div class="metric"><div class="v">{snap.get('cpu_pct','?')}</div><div class="l">CPU% (シングルコア)</div></div>
</div>

<div class="card">
  <div class="metric"><div class="v" style="color:{loop_color}">{loop_text}</div><div class="l">이벤트 루프 상태 (runtime_stats 응답))</div></div>
</div>

{thread_card}

<div class="card">
  <h3>엔드포인트 요청 통계</h3>
  <table>
    <tr><th>종점</th><th>누적</th><th>속도(회)/s)</th><th>최근 호출 거리(s)</th></tr>
    {''.join(rows) if rows else '<tr><td colspan=4>데이터 없음</td></tr>'}
  </table>
</div>

<div class="card">
  <h3>추세 (최근 2분)</h3>
  <p>FD: {min(fd_series) if fd_series else 0} ~ {max(fd_series) if fd_series else 0}</p>
  <p>스레드: {min(thr_series) if thr_series else 0} ~ {max(thr_series) if thr_series else 0}</p>
  <p>socket: {min(sock_series) if sock_series else 0} ~ {max(sock_series) if sock_series else 0}</p>
  <p>이벤트 루프 생존: {sum(loop_series)}/{len(loop_series)} 샘플링 횟수</p>
</div>

<div class="card" style="font-size:12px;color:#888">
  JSON 인터페이스: <a href="/snapshot" style="color:#6cb6ff">/snapshot</a> |
  <a href="/tail?n=100" style="color:#6cb6ff">/tail?n=100</a> |
  <a href="/reset" style="color:#6cb6ff">/reset (리셋 카운터)</a><br>
  로그 파일: {_LOG_FILE}
</div>

<script>
// 스레드 상세 스위치 + 간격 설정 (fetch 호출 /thread_config, 페이지 새로고침 안 함))
function tdToggle(enable) {{
  fetch('/thread_config', {{
    method: 'POST',
    headers: {{'Content-Type': 'application/json'}},
    body: JSON.stringify({{enabled: enable}})
  }}).then(r => r.json()).then(d => {{
    console.log('thread_config:', d);
    // 페이지를 일시적으로 늦게 갱신 UI
    setTimeout(() => location.reload(), 200);
  }}).catch(e => alert('설정 실패: ' + e));
}}

function tdApplyInterval() {{
  const el = document.getElementById('td-interval');
  if (!el) return;
  let v = parseInt(el.value) || 10;
  v = Math.max(2, Math.min(300, v));
  el.value = v;
  fetch('/thread_config', {{
    method: 'POST',
    headers: {{'Content-Type': 'application/json'}},
    body: JSON.stringify({{interval_s: v}})
  }}).then(r => r.json()).then(d => {{
    console.log('interval set:', d);
    setTimeout(() => location.reload(), 200);
  }}).catch(e => alert('설정 실패: ' + e));
}}
</script>
</body></html>"""


def _http_server_loop():
    """독립 HTTP 서비스 (daemon 스레드))"""
    try:
        server = ThreadingHTTPServer(
            ("0.0.0.0", _PROBE_PORT), _ProbeHandler
        )
        server.serve_forever()
    except Exception as e:
        try:
            sys.stderr.write(f"[debug_probe] HTTP server failed on port {_PROBE_PORT}: {e}\n")
        except Exception:
            pass


# ========== 시작점 (멱등)) ==========

def start():
    """디버깅 팁 시작 (멱등, 안전하게 여러 번 호출 가능))"""
    global _started
    if _started:
        return
    _started = True
    try:
        # 샘플링 스레드 시작
        t1 = threading.Thread(target=_sampler_loop, daemon=True, name="bsk-debug-sampler")
        t1.start()
        # HTTP 서비스 스레드 시작
        t2 = threading.Thread(target=_http_server_loop, daemon=True, name="bsk-debug-http")
        t2.start()
        print(f"[BSK Debug Probe] 시작됨 → http://localhost:{_PROBE_PORT}/  (로그: {_LOG_FILE})")
    except Exception as e:
        try:
            sys.stderr.write(f"[debug_probe] start failed: {e}\n")
        except Exception:
            pass


def get_snapshot():
    """최신 샷 가져오기 (다른 모듈 프로그램 액세스용))"""
    with _lock:
        return _latest.copy() if _latest else {}
