"""Fullscreen output window placed on the projector monitor."""

import ctypes
import ctypes.wintypes

import cv2
import numpy

import config

_GWL_STYLE = -16
_WS_POPUP = 0x80000000
_WS_VISIBLE = 0x10000000
_HWND_TOPMOST = -1
_SWP_SHOWWINDOW = 0x0040
_SWP_FRAMECHANGED = 0x0020
_SW_RESTORE = 9


class ProjectorWindow:
    def __init__(self, monitor, fullscreen, window_name="projector", background_fraction=None):
        self.window_name = window_name
        self.monitor = monitor
        self.fullscreen = fullscreen
        self.width = monitor.width
        self.height = monitor.height
        self.placement_checked = False
        if background_fraction is None:
            background_fraction = getattr(config, "projector_background_fraction", 0.0)
        self.background_level = numpy.uint8(round(255 * background_fraction))
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.moveWindow(window_name, monitor.left, monitor.top)
        if fullscreen:
            # OpenCV picks the monitor the window is currently on, so move first.
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.resizeWindow(window_name, self.width // 2, self.height // 2)
        self.close_requested = False
        self.shown_once = False
        cv2.setMouseCallback(window_name, self._on_mouse)
        print(f"[projector] {self.width}x{self.height} fullscreen={fullscreen}; "
              "to close it: double-click or right-click the projector window, or Alt+F4 on it")

    def _on_mouse(self, event, x, y, flags, user_data):
        if event in (cv2.EVENT_LBUTTONDBLCLK, cv2.EVENT_RBUTTONDOWN):
            self.close_requested = True

    def _stop_if_closed(self):
        """The person at the computer must always be able to get rid of this window. A
        double-click or right-click on it, or closing it (Alt+F4), ends the tool that owns
        it: SystemExit runs the tool's own clean-up (camera, bridge, recorder)."""
        closed = False
        if self.shown_once:
            try:
                closed = cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1
            except cv2.error:
                closed = True
        if self.close_requested or closed:
            try:
                cv2.destroyWindow(self.window_name)
            except cv2.error:
                pass
            raise SystemExit("[projector] the projector window was closed by hand; stopping")

    def show(self, image):
        self._stop_if_closed()
        if self.background_level > 0:
            # A faint floor instead of pure black: the beam's extent stays visible on the scene.
            image = numpy.maximum(image, self.background_level)
        cv2.imshow(self.window_name, image)
        self.shown_once = True
        if self.fullscreen and not self.placement_checked:
            # The window exists only after the first imshow. OpenCV's fullscreen request
            # is not always honoured (for instance when the process was started with a
            # minimized initial window state); verify and enforce it with Win32 directly.
            cv2.waitKey(1)
            self._enforce_placement()

    def _enforce_placement(self):
        user32 = ctypes.windll.user32
        handle = user32.FindWindowW(None, self.window_name)
        if not handle:
            return
        rectangle = ctypes.wintypes.RECT()
        user32.GetWindowRect(handle, ctypes.byref(rectangle))
        placed = (rectangle.left == self.monitor.left and rectangle.top == self.monitor.top
                  and rectangle.right - rectangle.left == self.width and rectangle.bottom - rectangle.top == self.height)
        if not placed:
            user32.ShowWindow(handle, _SW_RESTORE)
            user32.SetWindowLongW(handle, _GWL_STYLE, _WS_POPUP | _WS_VISIBLE)
            user32.SetWindowPos(handle, _HWND_TOPMOST, self.monitor.left, self.monitor.top, self.width, self.height,
                                _SWP_SHOWWINDOW | _SWP_FRAMECHANGED)
            user32.GetWindowRect(handle, ctypes.byref(rectangle))
        self.placement_checked = True
        print(f"[projector] window at ({rectangle.left},{rectangle.top}) "
              f"{rectangle.right - rectangle.left}x{rectangle.bottom - rectangle.top}"
              + ("" if placed else " (corrected with Win32)"))
