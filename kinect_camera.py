"""Kinect v2 colour + depth capture on a background thread, with timestamped frames.

Drop-in replacement for camera.Camera: latest(), frame_after(), latest_after(),
recent_frames(), frame_width, frame_height and release() behave the same, so
the coded-dot scan and the face tracker run unchanged on the Kinect's colour
image. On top of that every colour frame is captured together with the depth
frame of the same instant (one multi-source frame), and points_3d_at() turns
colour pixels into metric 3D points (metres, Kinect camera space) through the
SDK's coordinate mapper.

The Kinect v2 colour camera has no manual exposure in SDK 2.0: it is always
automatic. Frame differencing over a few hundred milliseconds still works
because the projected dots are small compared with the frame.

Needs the Kinect SDK 2.0 runtime (Kinect20.dll) and pykinect2 with the
patches applied by patch_pykinect2.py.
"""

import collections
import ctypes
import threading
import time

import cv2
import numpy
from pykinect2 import PyKinectV2

_WAIT_OBJECT_0 = 0
_WAIT_TIMEOUT = 0x102
_TIMESPAN_TICKS_PER_SECOND = 10_000_000.0


class KinectCamera:
    def __init__(self, config):
        self.sensor = ctypes.POINTER(PyKinectV2.IKinectSensor)()
        result = ctypes.windll.kinect20.GetDefaultKinectSensor(ctypes.byref(self.sensor))
        if result != 0 or not self.sensor:
            raise RuntimeError("no Kinect v2 sensor found (is the Kinect SDK 2.0 runtime installed?)")
        self.sensor.Open()
        self.mapper = self.sensor.CoordinateMapper

        color_description = self.sensor.ColorFrameSource.FrameDescription
        depth_description = self.sensor.DepthFrameSource.FrameDescription
        self.frame_width = color_description.Width
        self.frame_height = color_description.Height
        self.depth_width = depth_description.Width
        self.depth_height = depth_description.Height

        self.color_buffer_capacity = ctypes.c_uint(self.frame_width * self.frame_height * 4)
        self.color_buffer = (ctypes.c_ubyte * self.color_buffer_capacity.value)()
        self.depth_buffer_capacity = ctypes.c_uint(self.depth_width * self.depth_height)
        self.depth_buffer = (ctypes.c_ushort * self.depth_buffer_capacity.value)()

        self.reader = self.sensor.OpenMultiSourceFrameReader(
            PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Depth)
        self.frame_event = self.reader.SubscribeMultiSourceFrameArrived()
        self.close_event = ctypes.windll.kernel32.CreateEventW(None, False, False, None)
        self.wait_handles = (ctypes.c_void_p * 2)(self.close_event, self.frame_event)

        self.frames = collections.deque(maxlen=config.camera_frame_history_length)
        self.depth_frames = collections.deque(maxlen=config.camera_frame_history_length)
        self.depth_sample_radius_pixels = config.kinect_depth_sample_radius_pixels
        self.condition = threading.Condition()
        self.running = True
        self.exposure_seconds = None
        self.gain = None
        self.dropped_frame_count = 0
        self.captured_frame_count = 0
        self.last_error = None
        self.stage_seconds = collections.defaultdict(float)
        self._mapped_depth_id = None
        self._mapped_points = None
        self._camera_space_points = (PyKinectV2._CameraSpacePoint * (self.frame_width * self.frame_height))()
        self._mapped_depth_frame_id = None
        self._depth_frame_colour_coordinates = None
        self._depth_frame_points = None
        self._depth_colour_space_points = (PyKinectV2._ColorSpacePoint * (self.depth_width * self.depth_height))()
        self._depth_camera_space_points = (PyKinectV2._CameraSpacePoint * (self.depth_width * self.depth_height))()
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

        # The sensor needs a few seconds after Open() before the first frame arrives, and
        # the stream sometimes stalls for tens of seconds right after a previous client closed.
        waited_seconds = 0.0
        while self.frame_after(0.0, timeout_seconds=2.0) is None:
            waited_seconds += 2.0
            print(f"[kinect] waiting for the first frame ({waited_seconds:.0f} s; the stream stalls at times, "
                  "Ctrl+C to give up)")
            if waited_seconds >= config.kinect_startup_timeout_seconds:
                report = self.capture_report()
                self.release()
                raise RuntimeError(f"Kinect opened but delivered no frames within {waited_seconds:.0f} s "
                                   "(is the KinectMonitor service running, and the sensor plugged into USB 3?) "
                                   f"available={bool(self.sensor.IsAvailable)} {report}")
        print(f"[kinect] colour {self.frame_width}x{self.frame_height}, depth {self.depth_width}x{self.depth_height}, "
              f"colour exposure {self.exposure_seconds * 1000.0 if self.exposure_seconds else float('nan'):.1f} ms "
              f"(automatic, not lockable on the Kinect v2), gain {self.gain}")

    def capture_report(self):
        """One line: frames captured, dropped, and the capture thread's time per frame per stage."""
        count = max(self.captured_frame_count, 1)
        stages = ", ".join(f"{name} {seconds / count * 1000:.1f} ms" for name, seconds in self.stage_seconds.items())
        return (f"[kinect] {self.captured_frame_count} frames captured, {self.dropped_frame_count} dropped, "
                f"thread {'alive' if self.thread.is_alive() else 'DEAD'}; per frame: {stages}"
                + (f"; last error: {self.last_error}" if self.last_error else ""))

    # ------------------------------------------------------------ capture

    def _capture_loop(self):
        # Seconds spent per stage, for diagnostics (self.stage_seconds / self.captured_frame_count).
        while self.running:
            stage_started = time.perf_counter()
            wait_result = ctypes.windll.kernel32.WaitForMultipleObjects(2, self.wait_handles, False, 500)
            self.stage_seconds["wait"] += time.perf_counter() - stage_started
            if wait_result == _WAIT_OBJECT_0 or not self.running:
                break
            if wait_result == _WAIT_TIMEOUT:
                continue
            stage_started = time.perf_counter()
            try:
                event_data = self.reader.GetMultiSourceFrameArrivedEventData(self.frame_event)
                multi_source_frame = event_data.FrameReference.AcquireFrame()
            except Exception as error:
                self.dropped_frame_count += 1
                self.last_error = f"multi-source frame: {error}"
                continue
            timestamp = time.perf_counter()
            try:
                color_frame = multi_source_frame.colorFrameReference.AcquireFrame()
                depth_frame = multi_source_frame.depthFrameReference.AcquireFrame()
            except Exception as error:
                # One of the two streams had no frame in this multi-source frame; skip it.
                self.dropped_frame_count += 1
                self.last_error = f"colour/depth frame: {error}"
                continue
            self.stage_seconds["acquire"] += time.perf_counter() - stage_started
            try:
                stage_started = time.perf_counter()
                color_frame.CopyConvertedFrameDataToArray(self.color_buffer_capacity, self.color_buffer,
                                                          PyKinectV2.ColorImageFormat_Bgra)
                self.stage_seconds["copy_colour"] += time.perf_counter() - stage_started
                stage_started = time.perf_counter()
                depth_frame.CopyFrameDataToArray(self.depth_buffer_capacity, self.depth_buffer)
                self.stage_seconds["copy_depth"] += time.perf_counter() - stage_started
                if self.exposure_seconds is None:
                    settings = color_frame.ColorCameraSettings
                    self.exposure_seconds = settings.ExposureTime / _TIMESPAN_TICKS_PER_SECOND
                    self.gain = settings.Gain
            except Exception as error:
                self.dropped_frame_count += 1
                self.last_error = f"frame copy: {error}"
                continue
            finally:
                color_frame = None
                depth_frame = None
                multi_source_frame = None
                event_data = None

            stage_started = time.perf_counter()
            color_bgra = numpy.frombuffer(self.color_buffer, dtype=numpy.uint8).reshape(
                self.frame_height, self.frame_width, 4)
            frame_bgr = cv2.cvtColor(color_bgra, cv2.COLOR_BGRA2BGR)
            depth_millimetres = numpy.frombuffer(self.depth_buffer, dtype=numpy.uint16).reshape(
                self.depth_height, self.depth_width).copy()
            self.stage_seconds["convert"] += time.perf_counter() - stage_started
            with self.condition:
                self.frames.append((timestamp, frame_bgr))
                self.depth_frames.append((timestamp, depth_millimetres))
                self.captured_frame_count += 1
                self.condition.notify_all()

    # ---------------------------------------------- camera.Camera interface

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
        """Block until the newest frame is newer than time_seconds, then return it."""
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
        """Snapshot of the colour history, oldest first."""
        with self.condition:
            return list(self.frames)

    def frames_per_second(self):
        """Arrival rate over the recent history; 0 when the newest frame is more than a second old (a stall)."""
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
        ctypes.windll.kernel32.SetEvent(self.close_event)
        self.thread.join(timeout=2.0)
        try:
            self.reader.UnsubscribeMultiSourceFrameArrived(self.frame_event)
        except Exception:
            pass
        try:
            self.sensor.Close()
        except Exception:
            pass

    # ---------------------------------------------------------------- depth

    def depth_for(self, timestamp):
        """The depth frame captured together with the colour frame of this timestamp
        (or the nearest one in time). Returns (timestamp, depth_millimetres) or None."""
        with self.condition:
            if not self.depth_frames:
                return None
            best = min(self.depth_frames, key=lambda entry: abs(entry[0] - timestamp))
            return best

    def median_depth(self, frame_count, timeout_seconds=15.0):
        """Per-pixel median of the next frame_count depth frames, holes ignored.
        Static scenes only. Returns depth in millimetres with 0 where no frame had
        depth, or None when the stream delivered nothing within timeout_seconds."""
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
                stack.append(depth_entry[1].astype(numpy.float32))
        if not stack:
            return None
        stacked = numpy.stack(stack)
        stacked[stacked == 0] = numpy.nan
        with numpy.errstate(all="ignore"):
            median = numpy.nanmedian(stacked, axis=0)
        return numpy.nan_to_num(median, nan=0.0).astype(numpy.uint16)

    def map_depth_to_colour_points(self, depth_millimetres):
        """3D point (metres, Kinect camera space) for every colour pixel: an
        (frame_height, frame_width, 3) float32 array, non-finite where the depth is
        unknown. ~13 ms for the full frame; the array is valid until the next call."""
        if self._mapped_depth_id == id(depth_millimetres) and self._mapped_points is not None:
            return self._mapped_points
        depth_contiguous = numpy.ascontiguousarray(depth_millimetres, dtype=numpy.uint16)
        depth_pointer = depth_contiguous.ctypes.data_as(ctypes.POINTER(ctypes.c_ushort))
        self.mapper.MapColorFrameToCameraSpace(depth_contiguous.size, depth_pointer,
                                               self.frame_width * self.frame_height, self._camera_space_points)
        points = numpy.ctypeslib.as_array(self._camera_space_points).view(numpy.float32).reshape(
            self.frame_height, self.frame_width, 3)
        self._mapped_depth_id = id(depth_millimetres)
        self._mapped_points = points
        return points

    def _map_depth_frame(self, depth_millimetres):
        """(colour coordinates (D, 2), 3D points (D, 3)) of every depth pixel, D = depth_width * depth_height.
        Much cheaper than mapping the whole colour frame (~3 ms), enough for a few query pixels."""
        if self._mapped_depth_frame_id == id(depth_millimetres) and self._depth_frame_colour_coordinates is not None:
            return self._depth_frame_colour_coordinates, self._depth_frame_points
        depth_contiguous = numpy.ascontiguousarray(depth_millimetres, dtype=numpy.uint16)
        depth_pointer = depth_contiguous.ctypes.data_as(ctypes.POINTER(ctypes.c_ushort))
        self.mapper.MapDepthFrameToColorSpace(depth_contiguous.size, depth_pointer,
                                              depth_contiguous.size, self._depth_colour_space_points)
        self.mapper.MapDepthFrameToCameraSpace(depth_contiguous.size, depth_pointer,
                                               depth_contiguous.size, self._depth_camera_space_points)
        colour_coordinates = numpy.ctypeslib.as_array(self._depth_colour_space_points).view(numpy.float32).reshape(-1, 2)
        points = numpy.ctypeslib.as_array(self._depth_camera_space_points).view(numpy.float32).reshape(-1, 3)
        self._mapped_depth_frame_id = id(depth_millimetres)
        self._depth_frame_colour_coordinates = colour_coordinates
        self._depth_frame_points = points
        return colour_coordinates, points

    def points_3d_at(self, depth_millimetres, colour_pixels, sample_radius_pixels=None, return_depth_spread=False):
        """3D points (N, 3) in metres for colour pixels (N, 2): the median of the
        depth samples that land within a square window (colour pixels) around each
        pixel. NaN rows where the window has no depth. Depth pixels are ~4 colour
        pixels apart, so the default radius gathers a handful of samples.
        With return_depth_spread, also the z range (metres) inside each window: a
        large spread means the pixel sits on a depth edge (silhouette) and its 3D
        point is unreliable."""
        radius = self.depth_sample_radius_pixels if sample_radius_pixels is None else sample_radius_pixels
        pixels = numpy.asarray(colour_pixels, dtype=numpy.float64).reshape(-1, 2)
        points = numpy.full((len(pixels), 3), numpy.nan, dtype=numpy.float64)
        spreads = numpy.full(len(pixels), numpy.nan, dtype=numpy.float64)
        if len(pixels) > 50:
            # Many queries (the calibration): one full colour-frame map is cheaper than many masks.
            mapped = self.map_depth_to_colour_points(depth_millimetres)
            for index, (x, y) in enumerate(pixels):
                column = int(round(x))
                row = int(round(y))
                if not (0 <= column < self.frame_width and 0 <= row < self.frame_height):
                    continue
                window = mapped[max(0, row - radius):row + radius + 1,
                                max(0, column - radius):column + radius + 1].reshape(-1, 3)
                finite = window[numpy.isfinite(window[:, 2])]
                if len(finite) > 0:
                    points[index] = numpy.median(finite, axis=0)
                    spreads[index] = float(finite[:, 2].max() - finite[:, 2].min())
        else:
            colour_coordinates, depth_points = self._map_depth_frame(depth_millimetres)
            for index, (x, y) in enumerate(pixels):
                near = ((numpy.abs(colour_coordinates[:, 0] - x) <= radius)
                        & (numpy.abs(colour_coordinates[:, 1] - y) <= radius))
                samples = depth_points[near]
                samples = samples[numpy.isfinite(samples[:, 2])]
                if len(samples) > 0:
                    points[index] = numpy.median(samples, axis=0)
                    spreads[index] = float(samples[:, 2].max() - samples[:, 2].min())
        if return_depth_spread:
            return points, spreads
        return points


def open_camera(config):
    """The Kinect through the backend named in config.kinect_backend."""
    if getattr(config, "kinect_backend", "sdk") == "freenect2":
        from freenect2_camera import Freenect2Camera
        return Freenect2Camera(config)
    return KinectCamera(config)
