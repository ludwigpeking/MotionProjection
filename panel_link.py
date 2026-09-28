"""Link between a running Kinect tool and the web panel (web_panel.py).

The tool publishes its debug view as a JPEG file the panel serves to the
browser, and polls a command file the panel writes ("quit", "lead+", ...).
Both are plain files in the diagnostics directory, so a tool started from the
terminal without the panel behaves exactly as before.
"""

import json
import os
import struct
import time

import cv2
import numpy

LIVE_VIEW_FILE_NAME = "live_view.jpg"
PROJECTOR_VIEW_FILE_NAME = "live_projector_view.jpg"
CLOUD_FILE_NAME = "live_cloud.bin"
STATE_FILE_NAME = "live_state.json"
COMMAND_FILE_NAME = "panel_command.txt"
CLOUD_MAGIC = 0x434C4F55       # "CLOU"


class PanelLink:
    def __init__(self, directory, enabled, publish_period_seconds=0.1, projector_view_period_seconds=0.4):
        self.enabled = enabled
        self.live_view_path = os.path.join(directory, LIVE_VIEW_FILE_NAME)
        self.projector_view_path = os.path.join(directory, PROJECTOR_VIEW_FILE_NAME)
        self.cloud_path = os.path.join(directory, CLOUD_FILE_NAME)
        self.state_path = os.path.join(directory, STATE_FILE_NAME)
        self.last_state_time = 0.0
        self.command_path = os.path.join(directory, COMMAND_FILE_NAME)
        self.publish_period_seconds = publish_period_seconds
        self.projector_view_period_seconds = projector_view_period_seconds
        self.last_publish_time = 0.0
        self.last_projector_view_time = 0.0
        if enabled:
            os.makedirs(directory, exist_ok=True)
            self.command()      # discard a stale command from an earlier run

    def projector_view_due(self):
        """True when it is time to render and publish the next projector view (rendering costs ~30 ms)."""
        return self.enabled and time.perf_counter() - self.last_projector_view_time >= self.projector_view_period_seconds

    def publish(self, view_bgr, force=False):
        """Write the camera view as the live JPEG, at most every publish_period_seconds unless forced."""
        if not self.enabled:
            return
        now = time.perf_counter()
        if not force and now - self.last_publish_time < self.publish_period_seconds:
            return
        self.last_publish_time = now
        self._write_jpeg(view_bgr, self.live_view_path)

    def publish_projector_view(self, view_bgr):
        if not self.enabled:
            return
        self.last_projector_view_time = time.perf_counter()
        self._write_jpeg(view_bgr, self.projector_view_path)

    def publish_state(self, state, force=False):
        """Small JSON with the tool's live settings (lead, prediction, radius, recording) for the page."""
        if not self.enabled:
            return
        now = time.perf_counter()
        if not force and now - self.last_state_time < 0.25:
            return
        self.last_state_time = now
        temporary_path = self.state_path + ".tmp"
        with open(temporary_path, "w") as file:
            json.dump(state, file)
        try:
            os.replace(temporary_path, self.state_path)
        except OSError:
            pass

    def publish_cloud(self, points_3d, colours_bgr, focal_x, focal_y, centre_x, centre_y, image_width, image_height):
        """The depth cloud for the 3D view: valid points only, float32 xyz in metres (the
        Kinect frame: x to the image's right, y up, z forward) followed by uint8 RGB.
        Header: magic, count, fx, fy, cx, cy, width, height (uint32/float32)."""
        if not self.enabled:
            return
        points_3d = numpy.asarray(points_3d, dtype=numpy.float32).reshape(-1, 3)
        colours_bgr = numpy.asarray(colours_bgr, dtype=numpy.uint8).reshape(-1, 3)
        valid = numpy.isfinite(points_3d[:, 2]) & (points_3d[:, 2] > 0)
        points_3d = points_3d[valid]
        colours_rgb = colours_bgr[valid][:, ::-1]
        header = struct.pack("<IIffffII", CLOUD_MAGIC, len(points_3d), float(focal_x), float(focal_y),
                             float(centre_x), float(centre_y), int(image_width), int(image_height))
        temporary_path = self.cloud_path + ".tmp"
        with open(temporary_path, "wb") as file:
            file.write(header)
            file.write(points_3d.tobytes())
            file.write(numpy.ascontiguousarray(colours_rgb).tobytes())
        try:
            os.replace(temporary_path, self.cloud_path)
        except OSError:
            pass

    @staticmethod
    def _write_jpeg(view_bgr, path):
        ok, encoded = cv2.imencode(".jpg", view_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            return
        temporary_path = path + ".tmp"
        with open(temporary_path, "wb") as file:
            file.write(encoded.tobytes())
        try:
            os.replace(temporary_path, path)
        except OSError:
            pass

    def command(self):
        """The pending command from the panel, or None. The command file is consumed."""
        if not self.enabled or not os.path.exists(self.command_path):
            return None
        try:
            with open(self.command_path) as file:
                text = file.read().strip()
            os.remove(self.command_path)
        except OSError:
            return None
        return text or None
