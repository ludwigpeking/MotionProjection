"""Kinect v2 through libfreenect2 (freenect2_bridge.exe) instead of the Microsoft driver.

Same interface as kinect_camera.KinectCamera, so the tools do not care which
backend runs. The bridge publishes each frame into a named shared-memory
mapping (see freenect2_bridge/freenect2_bridge.cpp); this class starts the
bridge, polls the mapping on a background thread and keeps a timestamped
history of colour frames together with the depth registered onto the colour
image ("bigdepth": one depth value per colour pixel, in millimetres).

3D points are in the colour camera's frame, in metres, right-handed like the
SDK's camera space (x to the image's right, y UP, z forward):
    x = (u - cx) / fx * z,   y = -(v - cy) / fy * z,   z = depth
with the colour intrinsics the sensor reports. Anything calibrated against
these coordinates must be recalibrated after switching backends.
"""

import collections
import ctypes
import mmap
import os
import struct
import subprocess
import sys
import threading
import time

import cv2
import numpy

MAPPING_NAME = "Local\\freenect2_bridge_frames"
STOP_EVENT_NAME = "Local\\freenect2_bridge_stop"
HEADER_FORMAT = "<IIIIdIIIII4f9fIIII32s"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)
MAGIC = 0x46324B42
EVENT_MODIFY_STATE = 0x0002


class Freenect2Camera:
    def __init__(self, config):
        self.bridge_process = None
        self.bridge_path = config.freenect2_bridge_path
        if not os.path.isabs(self.bridge_path):      # relative to the project, whatever the working directory
            self.bridge_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.bridge_path)
        self.stop_event = ctypes.windll.kernel32.CreateEventW(None, True, False, STOP_EVENT_NAME)
        ctypes.windll.kernel32.ResetEvent(self.stop_event)
        self._start_bridge(config)
        self.mapping = self._open_mapping(timeout_seconds=config.kinect_startup_timeout_seconds)
        header = self._read_header()
        (self.frame_width, self.frame_height, self.bigdepth_height, self.depth_width, self.depth_height,
         self.colour_focal_x, self.colour_focal_y, self.colour_centre_x, self.colour_centre_y,
         self.colour_offset, self.bigdepth_offset, self.depth_offset, self.total_bytes, self.pipeline) = (
            header["colour_width"], header["colour_height"], header["bigdepth_height"], header["depth_width"],
            header["depth_height"], header["colour_fx"], header["colour_fy"], header["colour_cx"], header["colour_cy"],
            header["colour_offset"], header["bigdepth_offset"], header["depth_offset"], header["total_bytes"],
            header["pipeline"])
        self.mapping = mmap.mmap(-1, self.total_bytes, tagname=MAPPING_NAME, access=mmap.ACCESS_WRITE)

        self.frames = collections.deque(maxlen=config.camera_frame_history_length)
        self.depth_frames = collections.deque(maxlen=config.camera_frame_history_length)
        self.depth_sample_radius_pixels = config.kinect_depth_sample_radius_pixels
        self.condition = threading.Condition()
        self.running = True
        self.exposure_seconds = None            # libfreenect2 does not report the colour exposure
        self.gain = None
        self.dropped_frame_count = 0
        self.captured_frame_count = 0
        self.last_error = None
        self.stage_seconds = collections.defaultdict(float)
        self._mapped_depth_id = None
        self._mapped_points = None
        self._mapped_depth_frame_id = None
        self._depth_frame_colour_coordinates = None
        self._depth_frame_points = None
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

        waited_seconds = 0.0
        while self.frame_after(0.0, timeout_seconds=2.0) is None:
            waited_seconds += 2.0
            print(f"[freenect2] waiting for the first frame ({waited_seconds:.0f} s)")
            if self.bridge_process is not None and self.bridge_process.poll() is not None:
                raise RuntimeError(f"freenect2_bridge exited with code {self.bridge_process.returncode}")
            if waited_seconds >= config.kinect_startup_timeout_seconds:
                self.release()
                raise RuntimeError("freenect2_bridge delivered no frames")
        print(f"[freenect2] colour {self.frame_width}x{self.frame_height}, depth {self.depth_width}x{self.depth_height}, "
              f"pipeline {self.pipeline}, colour focal {self.colour_focal_x:.0f}/{self.colour_focal_y:.0f} px, "
              f"centre ({self.colour_centre_x:.0f},{self.colour_centre_y:.0f})")

    # --------------------------------------------------------------- bridge

    def _start_bridge(self, config):
        # A bridge left behind by a tool that was killed keeps the mapping alive but may be
        # stuck; only adopt one whose frame counter is still advancing, otherwise replace it.
        try:
            probe = mmap.mmap(-1, HEADER_SIZE, tagname=MAPPING_NAME, access=mmap.ACCESS_READ)
            magic = struct.unpack_from("<I", probe, 0)[0]
            if magic == MAGIC:
                count_before = struct.unpack_from("<I", probe, 12)[0]
                time.sleep(1.5)
                count_after = struct.unpack_from("<I", probe, 12)[0]
                probe.close()
                if count_after - count_before >= 5:
                    print("[freenect2] a live bridge is already running; using it")
                    return
                print(f"[freenect2] a stale bridge is running ({count_after - count_before} frames in 1.5 s); "
                      "asking it to stop")
            else:
                probe.close()
        except (OSError, ValueError):
            pass
        # Stop any bridge through its named event and wait for it to exit: a killed bridge
        # leaves the sensor's streams open, and the next one then receives no frames.
        ctypes.windll.kernel32.SetEvent(self.stop_event)
        deadline = time.perf_counter() + 8.0
        while time.perf_counter() < deadline:
            alive = subprocess.run(["tasklist", "/FI", "IMAGENAME eq freenect2_bridge.exe"], capture_output=True, text=True)
            if "freenect2_bridge.exe" not in alive.stdout:
                break
            time.sleep(0.3)
        else:
            subprocess.run(["taskkill", "/F", "/IM", "freenect2_bridge.exe"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print("[freenect2] the old bridge ignored the stop request and was killed")
        ctypes.windll.kernel32.ResetEvent(self.stop_event)
        time.sleep(1.0)
        if not os.path.exists(self.bridge_path):
            raise RuntimeError(f"{self.bridge_path} not found: build freenect2_bridge first (see README)")
        self.bridge_process = subprocess.Popen([self.bridge_path, config.freenect2_pipeline], cwd=os.path.dirname(self.bridge_path),
                                               stdout=sys.stdout, stderr=subprocess.STDOUT)

    def _open_mapping(self, timeout_seconds):
        deadline = time.perf_counter() + timeout_seconds
        while time.perf_counter() < deadline:
            try:
                mapping = mmap.mmap(-1, HEADER_SIZE, tagname=MAPPING_NAME, access=mmap.ACCESS_READ)
                if struct.unpack_from("<I", mapping, 0)[0] == MAGIC:
                    return mapping
                mapping.close()
            except (OSError, ValueError):
                pass
            if self.bridge_process is not None and self.bridge_process.poll() is not None:
                raise RuntimeError(f"freenect2_bridge exited with code {self.bridge_process.returncode} before publishing")
            time.sleep(0.2)
        raise RuntimeError("freenect2_bridge did not publish its frame mapping")

    def _read_header(self):
        values = struct.unpack_from(HEADER_FORMAT, self.mapping, 0)
        keys = ["magic", "header_bytes", "sequence", "frame_count", "timestamp", "colour_width", "colour_height",
                "bigdepth_height", "depth_width", "depth_height", "colour_fx", "colour_fy", "colour_cx", "colour_cy",
                "ir_fx", "ir_fy", "ir_cx", "ir_cy", "ir_k1", "ir_k2", "ir_k3", "ir_p1", "ir_p2",
                "colour_offset", "bigdepth_offset", "depth_offset", "total_bytes", "pipeline"]
        header = dict(zip(keys, values))
        header["pipeline"] = header["pipeline"].split(b"\0", 1)[0].decode("ascii", "replace")
        return header

    def _capture_loop(self):
        last_sequence = 0
        colour_bytes = self.frame_width * self.frame_height * 4
        bigdepth_bytes = self.frame_width * self.bigdepth_height * 4
        while self.running:
            sequence = struct.unpack_from("<I", self.mapping, 8)[0]
            if sequence == last_sequence or sequence % 2 == 1:
                time.sleep(0.002)
                continue
            stage_started = time.perf_counter()
            colour_bgra = numpy.frombuffer(self.mapping, dtype=numpy.uint8, count=colour_bytes,
                                           offset=self.colour_offset).reshape(self.frame_height, self.frame_width, 4)
            frame_bgr = cv2.cvtColor(colour_bgra, cv2.COLOR_BGRA2BGR)
            bigdepth = numpy.frombuffer(self.mapping, dtype=numpy.float32, count=bigdepth_bytes // 4,
                                        offset=self.bigdepth_offset).reshape(self.bigdepth_height, self.frame_width)
            depth_millimetres = bigdepth[1:1 + self.frame_height].copy()      # rows 0 and 1081 are padding
            timestamp = time.perf_counter()
            if struct.unpack_from("<I", self.mapping, 8)[0] != sequence:
                self.dropped_frame_count += 1            # overwritten while we copied: torn frame, skip it
                continue
            last_sequence = sequence
            self.stage_seconds["copy"] += time.perf_counter() - stage_started
            with self.condition:
                self.frames.append((timestamp, frame_bgr))
                self.depth_frames.append((timestamp, depth_millimetres))
                self.captured_frame_count += 1
                self.condition.notify_all()

    def capture_report(self):
        count = max(self.captured_frame_count, 1)
        return (f"[freenect2] {self.captured_frame_count} frames captured, {self.dropped_frame_count} torn, "
                f"thread {'alive' if self.thread.is_alive() else 'DEAD'}; copy {self.stage_seconds['copy'] / count * 1000:.1f} ms per frame")

    # ------------------------------------------------- camera.Camera interface

    def latest(self):
        with self.condition:
            return self.frames[-1]

    def frame_after(self, time_seconds, timeout_seconds=1.0):
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
        deadline = time.perf_counter() + timeout_seconds
        with self.condition:
            while True:
                if self.frames:
                    timestamp, frame = self.frames[-1]
                    if timestamp > time_seconds:
                        return timestamp, frame
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    return None
                self.condition.wait(remaining)

    def recent_frames(self):
        with self.condition:
            return list(self.frames)

    def frames_per_second(self):
        with self.condition:
            if len(self.frames) < 2:
                return 0.0
            oldest_timestamp = self.frames[0][0]
            newest_timestamp = self.frames[-1][0]
        if time.perf_counter() - newest_timestamp > 1.0:
            return 0.0
        span = newest_timestamp - oldest_timestamp
        return (len(self.frames) - 1) / span if span > 0 else 0.0

    def release(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=2.0)
        if self.bridge_process is not None:
            ctypes.windll.kernel32.SetEvent(self.stop_event)
            try:
                self.bridge_process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self.bridge_process.terminate()
        try:
            self.mapping.close()
        except Exception:
            pass

    # ------------------------------------------------------------------ depth

    def depth_for(self, timestamp):
        """(timestamp, depth in millimetres per colour pixel) captured with that colour frame."""
        with self.condition:
            if not self.depth_frames:
                return None
            return min(self.depth_frames, key=lambda entry: abs(entry[0] - timestamp))

    def median_depth(self, frame_count, timeout_seconds=15.0):
        stack = []
        last_timestamp = time.perf_counter()
        deadline = last_timestamp + timeout_seconds
        while len(stack) < frame_count:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            result = self.frame_after(last_timestamp, timeout_seconds=remaining)
            if result is None:
                break
            last_timestamp = result[0]
            depth_entry = self.depth_for(last_timestamp)
            if depth_entry is not None:
                stack.append(depth_entry[1])
        if not stack:
            return None
        stacked = numpy.stack(stack)
        stacked[~numpy.isfinite(stacked) | (stacked <= 0)] = numpy.nan
        with numpy.errstate(all="ignore"):
            median = numpy.nanmedian(stacked, axis=0)
        return numpy.nan_to_num(median, nan=0.0).astype(numpy.float32)

    def map_depth_to_colour_points(self, depth_millimetres):
        """(frame_height, frame_width, 3) float32 3D points in metres for every colour pixel; NaN where unknown."""
        if self._mapped_depth_id == id(depth_millimetres) and self._mapped_points is not None:
            return self._mapped_points
        depth_metres = numpy.asarray(depth_millimetres, dtype=numpy.float32) / 1000.0
        depth_metres[~numpy.isfinite(depth_metres) | (depth_metres <= 0)] = numpy.nan
        columns = numpy.arange(self.frame_width, dtype=numpy.float32)
        rows = numpy.arange(self.frame_height, dtype=numpy.float32)
        x_factor = (columns - self.colour_centre_x) / self.colour_focal_x
        # y points UP (image rows go down): with x to the image's right in the mirrored
        # colour image and z forward, y up makes the frame right-handed, as the SDK's was.
        # A left-handed frame fits one plane perfectly and reflects everything off it.
        y_factor = -(rows - self.colour_centre_y) / self.colour_focal_y
        points = numpy.empty((self.frame_height, self.frame_width, 3), dtype=numpy.float32)
        points[:, :, 0] = depth_metres * x_factor[None, :]
        points[:, :, 1] = depth_metres * y_factor[:, None]
        points[:, :, 2] = depth_metres
        self._mapped_depth_id = id(depth_millimetres)
        self._mapped_points = points
        return points

    def _map_depth_frame(self, depth_millimetres):
        """(colour coordinates (N, 2), 3D points (N, 3)) on a 4-pixel grid of the colour image:
        the same shape of data the SDK backend gives per depth pixel, for the projector view."""
        if self._mapped_depth_frame_id == id(depth_millimetres) and self._depth_frame_colour_coordinates is not None:
            return self._depth_frame_colour_coordinates, self._depth_frame_points
        points = self.map_depth_to_colour_points(depth_millimetres)[::4, ::4]
        rows, columns = numpy.mgrid[0:self.frame_height:4, 0:self.frame_width:4]
        coordinates = numpy.stack([columns.ravel(), rows.ravel()], axis=1).astype(numpy.float32)
        self._mapped_depth_frame_id = id(depth_millimetres)
        self._depth_frame_colour_coordinates = coordinates
        self._depth_frame_points = points.reshape(-1, 3)
        return self._depth_frame_colour_coordinates, self._depth_frame_points

    def points_3d_at(self, depth_millimetres, colour_pixels, sample_radius_pixels=None, return_depth_spread=False):
        """3D points (metres, y up) at colour pixels: the depth is the median of the valid
        depths in a window around each pixel; NaN where the window holds none. All pixels
        are looked up at once (468 landmarks cost about a millisecond)."""
        radius = self.depth_sample_radius_pixels if sample_radius_pixels is None else sample_radius_pixels
        pixels = numpy.asarray(colour_pixels, dtype=numpy.float64).reshape(-1, 2)
        points = numpy.full((len(pixels), 3), numpy.nan, dtype=numpy.float64)
        spreads = numpy.full(len(pixels), numpy.nan, dtype=numpy.float64)
        depth = numpy.asarray(depth_millimetres, dtype=numpy.float32)
        if len(pixels) > 0:
            columns = numpy.round(pixels[:, 0]).astype(numpy.int64)
            rows = numpy.round(pixels[:, 1]).astype(numpy.int64)
            inside_image = (columns >= 0) & (columns < self.frame_width) & (rows >= 0) & (rows < self.frame_height)
            offsets = numpy.arange(-radius, radius + 1)
            window_rows = rows[:, None, None] + offsets[None, :, None]                # (N, side, 1)
            window_columns = columns[:, None, None] + offsets[None, None, :]          # (N, 1, side)
            within = ((window_rows >= 0) & (window_rows < depth.shape[0])
                      & (window_columns >= 0) & (window_columns < depth.shape[1]))
            window_rows = numpy.clip(window_rows, 0, depth.shape[0] - 1)
            window_columns = numpy.clip(window_columns, 0, depth.shape[1] - 1)
            values = depth[window_rows, window_columns].astype(numpy.float64).reshape(len(pixels), -1)
            valid = within.reshape(len(pixels), -1) & numpy.isfinite(values) & (values > 0)
            values[~valid] = numpy.nan
            has_depth = inside_image & valid.any(axis=1)
            if has_depth.any():
                with numpy.errstate(all="ignore"):
                    z = numpy.nanmedian(values[has_depth], axis=1) / 1000.0
                    spreads[has_depth] = (numpy.nanmax(values[has_depth], axis=1)
                                          - numpy.nanmin(values[has_depth], axis=1)) / 1000.0
                x = pixels[has_depth, 0]
                y = pixels[has_depth, 1]
                points[has_depth, 0] = (x - self.colour_centre_x) / self.colour_focal_x * z
                points[has_depth, 1] = -(y - self.colour_centre_y) / self.colour_focal_y * z     # y up, see map_depth_to_colour_points
                points[has_depth, 2] = z
        if return_depth_spread:
            return points, spreads
        return points
