"""Repeatable long-running resource audit for the FestVox desktop GUI.

The profiler deliberately uses the real Qt window and bundled diphone engine,
but redirects configuration writes to a temporary directory and replaces the
physical sound device with a no-op player.  Source voicebanks are read-only.

Run from this directory with the GUI dependencies installed::

    python resource_soak.py --cycles 30 --output rendered_audio/prompt0.json
"""
from __future__ import annotations

import argparse
import copy
import ctypes
import gc
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import tracemalloc
from dataclasses import asdict, dataclass
from typing import Optional, Sequence


MIB = 1024 * 1024
DEFAULT_THRESHOLDS = {
    "rss_slope_mib_per_cycle": 1.0,
    "private_slope_mib_per_cycle": 1.0,
    "python_slope_mib_per_cycle": 0.5,
    "thread_growth": 2,
    "handle_growth": 2,
    "child_growth": 0,
    "temp_file_growth": 1,
    "temp_growth_mib": 1.0,
    "idle_cpu_percent": 5.0,
    "qt_widget_slope_per_cycle": 2.0,
    "diphone_database_count": 2,
    "diphone_cache_files": 128,
    # Two resident voices, each bounded at 64 MiB decoded + 32 MiB slices.
    "diphone_cache_mib": 192.0,
    "undo_command_count": 64,
    "stale_phrase_preview_count": 0,
}


@dataclass(frozen=True)
class ResourceSnapshot:
    label: str
    cycle: int
    elapsed_seconds: float
    rss_bytes: Optional[int]
    private_bytes: Optional[int]
    virtual_bytes: Optional[int]
    cpu_seconds: float
    thread_count: int
    python_thread_count: int
    handle_count: Optional[int]
    child_process_count: Optional[int]
    temp_file_count: int
    temp_bytes: int
    python_allocated_bytes: Optional[int]
    python_peak_bytes: Optional[int]
    qt_widget_count: Optional[int]
    gpu_memory_bytes: Optional[int]
    gpu_status: str
    logical_resources: dict


class _FILETIME(ctypes.Structure):
    _fields_ = [("low", ctypes.c_ulong), ("high", ctypes.c_ulong)]


class _PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivateUsage", ctypes.c_size_t),
    ]


class _PROCESSENTRY32W(ctypes.Structure):
    _fields_ = [
        ("dwSize", ctypes.c_ulong),
        ("cntUsage", ctypes.c_ulong),
        ("th32ProcessID", ctypes.c_ulong),
        ("th32DefaultHeapID", ctypes.c_size_t),
        ("th32ModuleID", ctypes.c_ulong),
        ("cntThreads", ctypes.c_ulong),
        ("th32ParentProcessID", ctypes.c_ulong),
        ("pcPriClassBase", ctypes.c_long),
        ("dwFlags", ctypes.c_ulong),
        ("szExeFile", ctypes.c_wchar * 260),
    ]


def _filetime_seconds(value: _FILETIME) -> float:
    ticks = (int(value.high) << 32) | int(value.low)
    return ticks / 10_000_000.0


def _windows_process_metrics():
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    kernel32.CreateToolhelp32Snapshot.restype = ctypes.c_void_p
    kernel32.GetProcessHandleCount.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
    kernel32.GetProcessTimes.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(_FILETIME), ctypes.POINTER(_FILETIME),
        ctypes.POINTER(_FILETIME), ctypes.POINTER(_FILETIME)]
    kernel32.Process32FirstW.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(_PROCESSENTRY32W)]
    kernel32.Process32NextW.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(_PROCESSENTRY32W)]
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    psapi.GetProcessMemoryInfo.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(_PROCESS_MEMORY_COUNTERS_EX),
        ctypes.c_ulong]
    process = kernel32.GetCurrentProcess()
    counters = _PROCESS_MEMORY_COUNTERS_EX()
    counters.cb = ctypes.sizeof(counters)
    if not psapi.GetProcessMemoryInfo(
            process, ctypes.byref(counters), counters.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    handles = ctypes.c_ulong()
    if not kernel32.GetProcessHandleCount(process, ctypes.byref(handles)):
        raise ctypes.WinError(ctypes.get_last_error())
    created, exited, kernel, user = (_FILETIME() for _ in range(4))
    if not kernel32.GetProcessTimes(
            process, ctypes.byref(created), ctypes.byref(exited),
            ctypes.byref(kernel), ctypes.byref(user)):
        raise ctypes.WinError(ctypes.get_last_error())

    pid = os.getpid()
    thread_count, children = 0, 0
    snapshot = kernel32.CreateToolhelp32Snapshot(0x00000002, 0)
    invalid = ctypes.c_void_p(-1).value
    if snapshot != invalid:
        entry = _PROCESSENTRY32W()
        entry.dwSize = ctypes.sizeof(entry)
        if kernel32.Process32FirstW(snapshot, ctypes.byref(entry)):
            while True:
                if int(entry.th32ProcessID) == pid:
                    thread_count = int(entry.cntThreads)
                if int(entry.th32ParentProcessID) == pid:
                    children += 1
                if not kernel32.Process32NextW(snapshot, ctypes.byref(entry)):
                    break
        kernel32.CloseHandle(snapshot)
    return {
        "rss": int(counters.WorkingSetSize),
        "private": int(counters.PrivateUsage),
        # PROCESS_MEMORY_COUNTERS_EX has no process virtual-address total.
        "virtual": None,
        "cpu": _filetime_seconds(kernel) + _filetime_seconds(user),
        "threads": thread_count or threading.active_count(),
        "handles": int(handles.value),
        "children": children,
    }


def _proc_process_metrics():
    status = {}
    status_path = Path("/proc/self/status")
    for line in status_path.read_text(encoding="utf-8").splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            status[key] = value.strip()

    def kib(name):
        text = status.get(name, "0 kB").split()[0]
        return int(text) * 1024

    children = set()
    for task in Path("/proc/self/task").glob("*/children"):
        try:
            children.update(task.read_text().split())
        except OSError:
            pass
    return {
        "rss": kib("VmRSS"),
        "private": kib("RssAnon"),
        "virtual": kib("VmSize"),
        "cpu": time.process_time(),
        "threads": int(status.get("Threads", threading.active_count())),
        "handles": len(list(Path("/proc/self/fd").iterdir())),
        "children": len(children),
    }


def process_metrics():
    try:
        if sys.platform == "win32":
            return _windows_process_metrics()
        if Path("/proc/self/status").is_file():
            return _proc_process_metrics()
    except (OSError, ValueError):
        pass
    return {
        "rss": None,
        "private": None,
        "virtual": None,
        "cpu": time.process_time(),
        "threads": threading.active_count(),
        "handles": None,
        "children": None,
    }


def _temp_usage():
    temp_root = Path(tempfile.gettempdir())
    candidates = list(temp_root.glob("festvox_gui_play*.wav"))
    exchange = temp_root / "festvox_gui_wsl"
    if exchange.is_dir():
        candidates.extend(path for path in exchange.rglob("*") if path.is_file())
    total = 0
    for path in candidates:
        try:
            total += path.stat().st_size
        except OSError:
            pass
    return len(candidates), total


def _gpu_usage():
    executable = shutil.which("nvidia-smi")
    if not executable:
        return None, "nvidia-smi unavailable"
    try:
        proc = subprocess.run(
            [executable, "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
            creationflags=(subprocess.CREATE_NO_WINDOW
                           if sys.platform == "win32" else 0),
        )
    except (OSError, subprocess.SubprocessError) as error:
        return None, "nvidia-smi failed: %s" % error
    if proc.returncode != 0:
        return None, "nvidia-smi exit %d" % proc.returncode
    total_mib = 0
    found = False
    for line in proc.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 2 or parts[0] != str(os.getpid()):
            continue
        try:
            total_mib += int(parts[1])
            found = True
        except ValueError:
            pass
    return (total_mib * MIB if found else 0), (
        "measured" if found else "no GPU allocation for this process")


def take_snapshot(label, cycle, started, qt_widget_count=None, include_gpu=False,
                  logical_resources=None):
    metrics = process_metrics()
    current, peak = (tracemalloc.get_traced_memory()
                     if tracemalloc.is_tracing() else (None, None))
    temp_files, temp_bytes = _temp_usage()
    gpu_bytes, gpu_status = _gpu_usage() if include_gpu else (
        None, "not sampled this cycle")
    return ResourceSnapshot(
        label=str(label), cycle=int(cycle),
        elapsed_seconds=round(time.monotonic() - started, 6),
        rss_bytes=metrics["rss"], private_bytes=metrics["private"],
        virtual_bytes=metrics["virtual"], cpu_seconds=metrics["cpu"],
        thread_count=int(metrics["threads"]),
        python_thread_count=threading.active_count(),
        handle_count=metrics["handles"],
        child_process_count=metrics["children"],
        temp_file_count=temp_files, temp_bytes=temp_bytes,
        python_allocated_bytes=current, python_peak_bytes=peak,
        qt_widget_count=qt_widget_count,
        gpu_memory_bytes=gpu_bytes, gpu_status=gpu_status,
        logical_resources=dict(logical_resources or {}),
    )


def _gui_resource_counts(window):
    if window is None:
        return {}
    backend = getattr(window, "backend", None)
    databases = list(getattr(backend, "_dbs", {}).values())
    cache_files = 0
    cache_bytes = 0
    for database in databases:
        if hasattr(database, "cache_info"):
            info = database.cache_info()
            cache_files += int(info.get("files") or 0)
            cache_bytes += int(info.get("total_bytes", info.get("bytes")) or 0)
    preview_count = 0
    preview_bytes = 0
    stale_previews = 0
    for state in getattr(window, "sentences", []):
        previews = state.get("phrase_previews") or {}
        phrase_ids = {str(row.get("id") or "")
                      for row in (state.get("phrases") or [])}
        stale_previews += sum(
            1 for key in previews if str(key) not in phrase_ids)
        preview_count += len(previews)
        for samples, _rate in previews.values():
            preview_bytes += int(getattr(samples, "nbytes", 0))
    from PyQt5 import QtWidgets
    return {
        "diphone_database_count": len(databases),
        "diphone_cache_files": cache_files,
        "diphone_cache_bytes": cache_bytes,
        "sustain_cache_entries": len(getattr(backend, "_sustains", {})),
        "undo_command_count": int(window.undo_stack.count()),
        "phrase_preview_count": preview_count,
        "phrase_preview_bytes": preview_bytes,
        "stale_phrase_preview_count": stale_previews,
        "transient_dialog_count": len(window.findChildren(
            QtWidgets.QDialog)),
        "transient_menu_count": len(window.findChildren(QtWidgets.QMenu)),
        "playback_timer_active": bool(window._playback_timer.isActive()),
        "playback_finish_timer_active": bool(
            window._playback_finish_timer.isActive()),
    }


def _slope(values: Sequence[Optional[int]]) -> Optional[float]:
    rows = [(index, float(value)) for index, value in enumerate(values)
            if value is not None]
    if len(rows) < 2:
        return None
    x_mean = sum(row[0] for row in rows) / len(rows)
    y_mean = sum(row[1] for row in rows) / len(rows)
    denominator = sum((row[0] - x_mean) ** 2 for row in rows)
    if denominator <= 0:
        return 0.0
    return sum((x - x_mean) * (y - y_mean) for x, y in rows) / denominator


def analyze_snapshots(snapshots, warmup=3, thresholds=None):
    limits = dict(DEFAULT_THRESHOLDS)
    limits.update(thresholds or {})
    cycles = [row for row in snapshots if row.label == "cycle"]
    measured = cycles[min(max(0, int(warmup)), max(0, len(cycles) - 2)):]

    def slope_mib(field):
        value = _slope([getattr(row, field) for row in measured])
        return None if value is None else value / MIB

    slopes = {
        "rss_mib_per_cycle": slope_mib("rss_bytes"),
        "private_mib_per_cycle": slope_mib("private_bytes"),
        "python_mib_per_cycle": slope_mib("python_allocated_bytes"),
        "threads_per_cycle": _slope([
            row.thread_count for row in measured]),
        "handles_per_cycle": _slope([
            row.handle_count for row in measured]),
        "qt_widgets_per_cycle": _slope([
            row.qt_widget_count for row in measured]),
    }
    start = next((row for row in snapshots if row.label == "startup"),
                 snapshots[0])
    final = snapshots[-1]

    def growth(field):
        left, right = getattr(start, field), getattr(final, field)
        return None if left is None or right is None else right - left

    deltas = {
        "rss_bytes": growth("rss_bytes"),
        "private_bytes": growth("private_bytes"),
        "python_allocated_bytes": growth("python_allocated_bytes"),
        "thread_count": growth("thread_count"),
        "handle_count": growth("handle_count"),
        "child_process_count": growth("child_process_count"),
        "temp_file_count": growth("temp_file_count"),
        "temp_bytes": growth("temp_bytes"),
        "qt_widget_count": growth("qt_widget_count"),
    }
    # Startup-to-idle deltas include one-time Qt, audio, and FFT runtime
    # initialization. Leak gates use the post-warmup cycle span instead, while
    # the lifetime deltas above remain in the report for diagnosis.
    steady_start = measured[0] if measured else (cycles[0] if cycles else start)
    steady_final = measured[-1] if measured else (cycles[-1] if cycles else final)

    def steady_growth(field):
        left = getattr(steady_start, field)
        right = getattr(steady_final, field)
        return None if left is None or right is None else right - left

    steady_state_deltas = {
        "thread_count": steady_growth("thread_count"),
        "handle_count": steady_growth("handle_count"),
        "child_process_count": steady_growth("child_process_count"),
    }
    checks = {}

    def check(name, value, limit):
        checks[name] = {
            "value": value, "limit": limit,
            "passed": None if value is None else value <= limit,
        }

    check("rss_slope_mib_per_cycle", slopes["rss_mib_per_cycle"],
          limits["rss_slope_mib_per_cycle"])
    check("private_slope_mib_per_cycle", slopes["private_mib_per_cycle"],
          limits["private_slope_mib_per_cycle"])
    check("python_slope_mib_per_cycle", slopes["python_mib_per_cycle"],
          limits["python_slope_mib_per_cycle"])
    check("qt_widget_slope_per_cycle", slopes["qt_widgets_per_cycle"],
          limits["qt_widget_slope_per_cycle"])
    check("thread_growth", steady_state_deltas["thread_count"],
          limits["thread_growth"])
    check("handle_growth", steady_state_deltas["handle_count"],
          limits["handle_growth"])
    check("child_growth", deltas["child_process_count"],
          limits["child_growth"])
    check("temp_file_growth", deltas["temp_file_count"],
          limits["temp_file_growth"])
    temp_growth_mib = (None if deltas["temp_bytes"] is None else
                       deltas["temp_bytes"] / MIB)
    check("temp_growth_mib", temp_growth_mib, limits["temp_growth_mib"])
    idle_rows = [row for row in snapshots
                 if row.label in {"idle_start", "idle_end"}]
    idle_cpu_percent = None
    if len(idle_rows) >= 2:
        wall = idle_rows[-1].elapsed_seconds - idle_rows[0].elapsed_seconds
        cpu = idle_rows[-1].cpu_seconds - idle_rows[0].cpu_seconds
        if wall > 0:
            idle_cpu_percent = max(0.0, cpu / wall * 100.0)
    check("idle_cpu_percent", idle_cpu_percent,
          limits["idle_cpu_percent"])
    logical_rows = [row.logical_resources for row in cycles
                    if row.logical_resources]
    logical_maxima = {
        key: max(float(row.get(key) or 0) for row in logical_rows)
        for key in (
            "diphone_database_count", "diphone_cache_files",
            "diphone_cache_bytes", "sustain_cache_entries",
            "undo_command_count", "phrase_preview_count",
            "phrase_preview_bytes", "stale_phrase_preview_count",
            "transient_dialog_count", "transient_menu_count")
    } if logical_rows else {}
    check("diphone_database_count",
          logical_maxima.get("diphone_database_count"),
          limits["diphone_database_count"])
    check("diphone_cache_files", logical_maxima.get("diphone_cache_files"),
          limits["diphone_cache_files"])
    cache_mib = (None if not logical_rows else
                 logical_maxima.get("diphone_cache_bytes", 0) / MIB)
    check("diphone_cache_mib", cache_mib, limits["diphone_cache_mib"])
    check("undo_command_count", logical_maxima.get("undo_command_count"),
          limits["undo_command_count"])
    check("stale_phrase_preview_count",
          logical_maxima.get("stale_phrase_preview_count"),
          limits["stale_phrase_preview_count"])
    active_timers = [
        row.cycle for row in cycles
        if row.logical_resources.get("playback_timer_active") or
        row.logical_resources.get("playback_finish_timer_active")
    ]
    checks["stopped_playback_timers"] = {
        "value": active_timers, "limit": [], "passed": not active_timers,
    }
    failed = [name for name, row in checks.items() if row["passed"] is False]
    return {
        "passed": not failed,
        "failed_checks": failed,
        "warmup_cycles_excluded": min(int(warmup), len(cycles)),
        "measured_cycle_count": len(measured),
        "slopes": slopes,
        "deltas": deltas,
        "steady_state_deltas": steady_state_deltas,
        "checks": checks,
        "thresholds": limits,
        "idle_cpu_percent": idle_cpu_percent,
        "logical_maxima": logical_maxima,
    }


class _NullPlayer:
    mode = "soak-no-device"

    def __init__(self):
        self.play_count = 0
        self.stop_count = 0

    def play(self, _samples, _rate):
        self.play_count += 1

    def stop(self):
        self.stop_count += 1

    def shutdown(self):
        self.stop()


def _select_data(combo, value):
    index = combo.findData(value)
    if index >= 0:
        combo.setCurrentIndex(index)
        return True
    return False


def _select_voice(widget, value):
    for index in range(widget.count()):
        if widget.item(index).text() == value:
            widget.setCurrentRow(index)
            return True
    return False


def _drain_qt(app):
    app.processEvents()
    from PyQt5 import QtCore
    QtCore.QCoreApplication.sendPostedEvents(
        None, QtCore.QEvent.DeferredDelete)
    app.processEvents()


def _idle_qt(seconds):
    from PyQt5 import QtCore
    loop = QtCore.QEventLoop()
    QtCore.QTimer.singleShot(
        max(1, int(round(max(0.0, float(seconds)) * 1000.0))), loop.quit)
    loop.exec_()


def _voice_databases(tool_dir: Path):
    result = {}
    root = tool_dir / "generated_voices"
    if root.is_dir():
        for folder in sorted(root.iterdir()):
            if (folder / "dic" / "diphone_index.json").is_file():
                result[folder.name] = str(folder)
    return result


def _soak_phone_sequences(voice_root, limit=3):
    metadata = json.loads(
        (Path(voice_root) / "dic" / "diphone_index.json")
        .read_text(encoding="utf-8"))
    adjacency = {}
    for name in sorted((metadata.get("index") or {})):
        if "-" not in name or "__" in name:
            continue
        left, right = name.split("-", 1)
        adjacency.setdefault(left, []).append(right)
    results = []

    def walk(phone, path, seen):
        if len(results) >= int(limit) or len(path) > 8:
            return
        for following in adjacency.get(phone, []):
            if len(results) >= int(limit):
                return
            if following == "pau" and path:
                sequence = tuple(path)
                if sequence not in results:
                    results.append(sequence)
                continue
            if following == "pau" or following in seen:
                continue
            walk(following, path + [following], seen | {following})

    walk("pau", [], set())
    if not results:
        raise RuntimeError(
            "The soak voice has no complete pau-to-pau diphone path.")
    return results


def run_gui_soak(cycles=20, warmup=3, idle_seconds=5.0,
                 enable_tracemalloc=False, reload_each_cycle=False):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    tool_dir = Path(__file__).resolve().parent
    gui_dir = tool_dir / "festvox_gui"
    if str(gui_dir) not in sys.path:
        sys.path.insert(0, str(gui_dir))
    if str(tool_dir) not in sys.path:
        sys.path.insert(0, str(tool_dir))
    import festvox_core as fc
    import festvox_gui as fg

    voices = _voice_databases(tool_dir)
    if not voices:
        raise RuntimeError(
            "No generated diphone voice with dic/diphone_index.json was "
            "found under festvox/generated_voices.")
    first_name, first_path = next(iter(voices.items()))
    phone_sequences = _soak_phone_sequences(first_path)
    # Two aliases exercise voice switching and backend LRU behavior without
    # reading from any second source bank.
    soak_voices = {
        "soak_a_" + first_name: first_path,
        "soak_b_" + first_name: first_path,
    }
    app = fg.QtWidgets.QApplication.instance() or fg.QtWidgets.QApplication([])
    old_config_path = fg.CONFIG_PATH
    config_temp = tempfile.TemporaryDirectory(prefix="festvox_soak_config_")
    fg.CONFIG_PATH = str(Path(config_temp.name) / "config.json")
    cfg = copy.deepcopy(fc.DEFAULT_CONFIG)
    cfg.update({
        "engine": "diphone",
        "festvox_config": str(tool_dir / "festvox.json"),
        "extra_voicebanks": soak_voices,
        "default_language": "English",
        "default_text": "this is a resource soak test",
        "undo_limit": 64,
    })
    started = time.monotonic()
    if enable_tracemalloc:
        tracemalloc.start()
        trace_start = tracemalloc.take_snapshot()
    else:
        tracemalloc.stop()
        trace_start = None
    snapshots = [take_snapshot("process_start", -1, started, include_gpu=True)]
    failures = []
    window = None
    player = _NullPlayer()
    try:
        window = fg.MainWindow(cfg)
        window.player.shutdown()
        window.player = player
        _select_data(window.engine, "diphone")
        _select_data(window.lang, "en")
        app.processEvents()
        snapshots.append(take_snapshot(
            "startup", -1, started, len(app.allWidgets()), include_gpu=True,
            logical_resources=_gui_resource_counts(window)))
        if enable_tracemalloc:
            trace_start = tracemalloc.take_snapshot()
        names = list(soak_voices)
        for cycle in range(max(1, int(cycles))):
            _select_voice(window.voicebank, names[cycle % len(names)])
            _select_data(window.input_mode, "phones")
            window.text.setText(" ".join(
                phone_sequences[cycle % len(phone_sequences)]))
            result = window._generate_current(
                confirm_replace=False, show_error=False)
            if result is None:
                failures.append({
                    "cycle": cycle,
                    "operation": "generate",
                    "error": str(window._last_generation_error),
                })
            else:
                window.on_play()
                app.processEvents()
                window.on_stop()
            window._begin_sentence_batch(2, "Soak")
            window._request_batch_cancel()
            window._end_sentence_batch()
            _drain_qt(app)
            gc.collect()
            _drain_qt(app)
            snapshots.append(take_snapshot(
                "cycle", cycle, started, len(app.allWidgets()),
                include_gpu=(cycle == int(cycles) - 1),
                logical_resources=_gui_resource_counts(window)))
            if reload_each_cycle:
                window.backend.reload_festvox_config()
                window._populate_from_backend()
                _select_data(window.engine, "diphone")
                _select_data(window.lang, "en")
                _drain_qt(app)
    finally:
        if window is not None:
            window.close()
            window.deleteLater()
            _drain_qt(app)
            window = None
        gc.collect()
        _drain_qt(app)
        snapshots.append(take_snapshot(
            "idle_start", int(cycles), started,
            len(app.allWidgets()), include_gpu=True))
        _idle_qt(idle_seconds)
        gc.collect()
        _drain_qt(app)
        snapshots.append(take_snapshot(
            "idle_end", int(cycles), started,
            len(app.allWidgets()), include_gpu=True))
        fg.CONFIG_PATH = old_config_path
        config_temp.cleanup()
    analysis = analyze_snapshots(snapshots, warmup=warmup)
    trace_growth = []
    if enable_tracemalloc:
        trace_end = tracemalloc.take_snapshot()
        for row in trace_end.compare_to(trace_start, "lineno")[:20]:
            frame = row.traceback[0]
            trace_growth.append({
                "file": str(frame.filename), "line": int(frame.lineno),
                "size_diff_bytes": int(row.size_diff),
                "count_diff": int(row.count_diff),
            })
        tracemalloc.stop()
    if failures:
        analysis["passed"] = False
        analysis["failed_checks"].append("workflow_failures")
    return {
        "schema_version": 1,
        "platform": sys.platform,
        "python": sys.version,
        "cycles": int(cycles),
        "warmup": int(warmup),
        "idle_seconds": float(idle_seconds),
        "tracemalloc_enabled": bool(enable_tracemalloc),
        "reload_each_cycle": bool(reload_each_cycle),
        "voice_source": first_path,
        "phone_sequences": [list(row) for row in phone_sequences],
        "source_bank_modified": False,
        "workflows": [
            "generation", "playback_start_stop", "batch_cancellation",
            "voice_alias_switching", "persistent_warm_cache",
        ] + (["backend_model_reload"] if reload_each_cycle else []),
        "player": {
            "mode": player.mode,
            "play_count": player.play_count,
            "stop_count": player.stop_count,
        },
        "failures": failures,
        "analysis": analysis,
        "tracemalloc_top_growth": trace_growth,
        "snapshots": [asdict(row) for row in snapshots],
    }


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycles", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--idle-seconds", type=float, default=5.0)
    parser.add_argument(
        "--tracemalloc", action="store_true",
        help="enable slower Python allocation attribution (changes RSS timing)")
    parser.add_argument(
        "--reload-each-cycle", action="store_true",
        help="also stress explicit backend/model invalidation after each cycle")
    parser.add_argument("--output", type=Path)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    report = run_gui_soak(
        args.cycles, args.warmup, args.idle_seconds,
        enable_tracemalloc=args.tracemalloc,
        reload_each_cycle=args.reload_each_cycle)
    payload = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
        print(args.output)
    print(json.dumps({
        "passed": report["analysis"]["passed"],
        "failed_checks": report["analysis"]["failed_checks"],
        "slopes": report["analysis"]["slopes"],
        "deltas": report["analysis"]["deltas"],
        "workflow_failures": len(report["failures"]),
    }, indent=2))
    return 0 if report["analysis"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
