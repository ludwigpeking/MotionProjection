"""Enumerate monitors on Windows so the projector window lands on the right one."""

import ctypes
from ctypes import wintypes
from dataclasses import dataclass


@dataclass
class Monitor:
    left: int
    top: int
    width: int
    height: int
    is_primary: bool


def _make_process_dpi_aware():
    """Without this, Windows reports monitor rectangles in scaled units and the
    fullscreen window ends up the wrong size on a 125% / 150% display."""
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


class _MonitorInfo(ctypes.Structure):
    _fields_ = [
        ("size_bytes", wintypes.DWORD),
        ("monitor_rectangle", wintypes.RECT),
        ("work_rectangle", wintypes.RECT),
        ("flags", wintypes.DWORD),
    ]


def list_monitors():
    _make_process_dpi_aware()
    user32 = ctypes.windll.user32
    monitors = []

    callback_type = ctypes.WINFUNCTYPE(
        ctypes.c_int, wintypes.HMONITOR, wintypes.HDC,
        ctypes.POINTER(wintypes.RECT), wintypes.LPARAM,
    )

    def on_monitor(monitor_handle, device_context, rectangle_pointer, user_data):
        info = _MonitorInfo()
        info.size_bytes = ctypes.sizeof(_MonitorInfo)
        user32.GetMonitorInfoW(monitor_handle, ctypes.byref(info))
        rectangle = info.monitor_rectangle
        monitors.append(Monitor(
            left=rectangle.left,
            top=rectangle.top,
            width=rectangle.right - rectangle.left,
            height=rectangle.bottom - rectangle.top,
            is_primary=bool(info.flags & 1),
        ))
        return 1

    user32.EnumDisplayMonitors(None, None, callback_type(on_monitor), 0)
    return monitors


def pick_projector_monitor(monitor_index, fallback_width, fallback_height):
    """Return (monitor, is_real_second_screen)."""
    monitors = list_monitors()
    for index, monitor in enumerate(monitors):
        primary_tag = " primary" if monitor.is_primary else ""
        print(f"[displays] monitor {index}: {monitor.width}x{monitor.height} at "
              f"({monitor.left},{monitor.top}){primary_tag}")

    if monitor_index is not None and 0 <= monitor_index < len(monitors):
        return monitors[monitor_index], True

    for monitor in monitors:
        if not monitor.is_primary:
            return monitor, True

    print("[displays] no second monitor found; projector output will open as a "
          f"{fallback_width}x{fallback_height} window on the primary screen")
    return Monitor(0, 0, fallback_width, fallback_height, True), False
