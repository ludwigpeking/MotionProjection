"""Raw camera recording: the colour frames exactly as delivered, no overlays.

Files go to recordings/kinect_<date>_<time>.avi (Motion JPEG, always playable
on Windows and with every frame kept intact). The frame rate written is the
rate the camera delivers when recording starts, so playback runs at real time.
"""

import datetime
import os

import cv2


class Recorder:
    def __init__(self, directory="recordings"):
        self.directory = directory
        self.writer = None
        self.path = None
        self.frame_count = 0
        self.started_at = None

    @property
    def active(self):
        return self.writer is not None

    def start(self, frame_width, frame_height, frames_per_second):
        os.makedirs(self.directory, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(self.directory, f"kinect_{stamp}.avi")
        rate = float(frames_per_second) if frames_per_second and frames_per_second >= 5 else 30.0
        self.writer = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*"MJPG"), rate, (frame_width, frame_height))
        if not self.writer.isOpened():
            self.writer = None
            print(f"[record] could not open {self.path} for writing", flush=True)
            return False
        self.frame_count = 0
        self.started_at = datetime.datetime.now()
        print(f"[record] started {self.path} at {rate:.0f} fps", flush=True)
        return True

    def write(self, frame_bgr):
        if self.writer is None:
            return
        self.writer.write(frame_bgr)
        self.frame_count += 1

    def stop(self):
        if self.writer is None:
            return
        self.writer.release()
        self.writer = None
        seconds = (datetime.datetime.now() - self.started_at).total_seconds()
        print(f"[record] stopped {self.path}: {self.frame_count} frames, {seconds:.0f} s", flush=True)

    def status_text(self):
        if self.writer is None:
            return ""
        seconds = (datetime.datetime.now() - self.started_at).total_seconds()
        return f"REC {seconds:.0f} s, {self.frame_count} frames"
