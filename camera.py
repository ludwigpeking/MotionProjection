"""Webcam capture on a background thread with timestamped frames.

The capture thread keeps a short history of (timestamp, frame). The main loop
takes the newest frame for tracking, and the servo picks reference and
measurement frames by timestamp, so timing does not depend on how fast the
main loop happens to run.
"""

import collections
import threading
import time

import cv2


class Camera:
    def __init__(self, config):
        # Media Foundation exposes real exposure control on this machine;
        # DirectShow silently ignores it and reports -1.
        self.capture = cv2.VideoCapture(config.camera_index, cv2.CAP_MSMF)
        if not self.capture.isOpened():
            raise RuntimeError(f"could not open camera index {config.camera_index} "
                               "(is another program, or an earlier run, still using it?)")

        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.camera_capture_width_pixels)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.camera_capture_height_pixels)
        self.capture.set(cv2.CAP_PROP_FPS, config.camera_target_frames_per_second)

        if config.camera_lock_exposure:
            # Media Foundation convention: 0 = manual, 1 = automatic.
            self.capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
            self.capture.set(cv2.CAP_PROP_EXPOSURE, config.camera_manual_exposure_value)
            self.capture.set(cv2.CAP_PROP_AUTO_WB, 0)
            if config.camera_manual_gain_value is not None:
                self.capture.set(cv2.CAP_PROP_GAIN, config.camera_manual_gain_value)

        first_frame = None
        for _ in range(10):
            ok, first_frame = self.capture.read()
            if ok and first_frame is not None:
                break
        if first_frame is None:
            raise RuntimeError("camera opened but delivered no frames")

        self.frame_height, self.frame_width = first_frame.shape[:2]
        auto_exposure_reported = self.capture.get(cv2.CAP_PROP_AUTO_EXPOSURE)
        print(f"[camera] {self.frame_width}x{self.frame_height} "
              f"auto_exposure={auto_exposure_reported} "
              f"exposure={self.capture.get(cv2.CAP_PROP_EXPOSURE)} "
              f"gain={self.capture.get(cv2.CAP_PROP_GAIN)}")
        if config.camera_lock_exposure and auto_exposure_reported != 0.0:
            print("[camera] WARNING: driver may have ignored the manual exposure request; "
                  "lock exposure in the camera vendor app if the servo finds no dots")

        self.frames = collections.deque(maxlen=config.camera_frame_history_length)
        self.frames.append((time.perf_counter(), first_frame))
        self.condition = threading.Condition()
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def _capture_loop(self):
        while self.running:
            ok, frame = self.capture.read()
            if not ok or frame is None:
                time.sleep(0.005)
                continue
            timestamp = time.perf_counter()
            with self.condition:
                self.frames.append((timestamp, frame))
                self.condition.notify_all()

    def latest(self):
        """Newest (timestamp, frame)."""
        with self.condition:
            return self.frames[-1]

    def frame_after(self, time_seconds, timeout_seconds=1.0):
        """Block until a frame with timestamp > time_seconds exists; return (timestamp, frame) or None."""
        deadline = time.perf_counter() + timeout_seconds
        with self.condition:
            while True:
                for timestamp, frame in self.frames:
                    if timestamp > time_seconds:
                        return timestamp, frame
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    return None
                self.condition.wait(remaining)

    def latest_after(self, time_seconds, timeout_seconds=1.0):
        """Block until the newest frame is newer than time_seconds, then return
        it: (timestamp, frame) or None. Skips any frames in between, so a slow
        consumer never falls behind real time."""
        deadline = time.perf_counter() + timeout_seconds
        with self.condition:
            while True:
                timestamp, frame = self.frames[-1]
                if timestamp > time_seconds:
                    return timestamp, frame
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    return None
                self.condition.wait(remaining)

    def recent_frames(self):
        """Snapshot of the history, oldest first."""
        with self.condition:
            return list(self.frames)

    def release(self):
        self.running = False
        self.thread.join(timeout=1.0)
        self.capture.release()
