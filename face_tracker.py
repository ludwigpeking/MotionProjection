"""MediaPipe face landmarks with one-euro smoothing and short-horizon prediction."""

import math
import os
import urllib.request

import cv2
import mediapipe
import numpy
from mediapipe.tasks import python as mediapipe_python
from mediapipe.tasks.python import vision


def ensure_model_downloaded(model_path, model_url):
    if os.path.exists(model_path):
        return
    print(f"[face] downloading landmarker model to {model_path} ...")
    urllib.request.urlretrieve(model_url, model_path)
    print("[face] download complete")


def _smoothing_factor(time_delta_seconds, cutoff_hertz):
    time_constant_seconds = 1.0 / (2.0 * math.pi * cutoff_hertz)
    return 1.0 / (1.0 + time_constant_seconds / time_delta_seconds)


class OneEuroFilter:
    """Vectorised one-euro filter (Casiez et al. 2012) over an array of positions."""

    def __init__(self, min_cutoff_hertz, speed_coefficient, derivative_cutoff_hertz):
        self.min_cutoff_hertz = min_cutoff_hertz
        self.speed_coefficient = speed_coefficient
        self.derivative_cutoff_hertz = derivative_cutoff_hertz
        self.reset()

    def reset(self):
        self.previous_time_seconds = None
        self.previous_position = None
        self.previous_raw_position = None
        self.previous_velocity = None

    def update(self, position, time_seconds):
        if self.previous_position is None:
            self.previous_time_seconds = time_seconds
            self.previous_position = position.copy()
            self.previous_raw_position = position.copy()
            self.previous_velocity = numpy.zeros_like(position)
            return position.copy(), self.previous_velocity.copy()

        time_delta_seconds = max(time_seconds - self.previous_time_seconds, 1e-3)
        # Velocity from raw samples, not from the smoothed value: the smoothed
        # value lags, and that lag would otherwise leak into the prediction.
        raw_velocity = (position - self.previous_raw_position) / time_delta_seconds
        self.previous_raw_position = position.copy()
        velocity_alpha = _smoothing_factor(time_delta_seconds, self.derivative_cutoff_hertz)
        velocity = velocity_alpha * raw_velocity + (1.0 - velocity_alpha) * self.previous_velocity

        speed = numpy.linalg.norm(velocity, axis=-1, keepdims=True)
        cutoff_hertz = self.min_cutoff_hertz + self.speed_coefficient * speed
        position_alpha = _smoothing_factor(time_delta_seconds, cutoff_hertz)
        smoothed_position = position_alpha * position + (1.0 - position_alpha) * self.previous_position

        self.previous_time_seconds = time_seconds
        self.previous_position = smoothed_position
        self.previous_velocity = velocity
        return smoothed_position, velocity


class FaceTracker:
    def __init__(self, config, frame_width, frame_height):
        ensure_model_downloaded(config.face_landmarker_model_path, config.face_landmarker_model_url)
        options = vision.FaceLandmarkerOptions(
            base_options=mediapipe_python.BaseOptions(model_asset_path=config.face_landmarker_model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.smoothing_filter = OneEuroFilter(
            config.smoothing_min_cutoff_hertz,
            config.smoothing_speed_coefficient,
            config.smoothing_derivative_cutoff_hertz,
        )
        self.prediction_lead_seconds = config.prediction_lead_seconds
        self.prediction_deadband_pixels_per_second = getattr(config, "prediction_deadband_pixels_per_second", 0.0)
        self.prediction_full_pixels_per_second = getattr(config, "prediction_full_pixels_per_second", 1.0)
        self.last_prediction_gain = 0.0
        self.last_timestamp_milliseconds = -1
        self.last_raw_landmarks = None
        self.last_raw_landmarks_3d = None
        self.last_smoothed_z = None
        self.last_velocity = None
        self.downscale_factor = config.face_detection_downscale_factor

    def detect_raw(self, frame_bgr, time_seconds):
        """Unsmoothed landmarks as an (N, 2) pixel array, or None. Does not touch the smoothing filter."""
        timestamp_milliseconds = int(time_seconds * 1000.0)
        if timestamp_milliseconds <= self.last_timestamp_milliseconds:
            timestamp_milliseconds = self.last_timestamp_milliseconds + 1
        self.last_timestamp_milliseconds = timestamp_milliseconds

        if self.downscale_factor > 1:
            frame_bgr = cv2.resize(frame_bgr, None, fx=1.0 / self.downscale_factor, fy=1.0 / self.downscale_factor,
                                   interpolation=cv2.INTER_AREA)
        frame_rgb = numpy.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        image = mediapipe.Image(image_format=mediapipe.ImageFormat.SRGB, data=frame_rgb)
        result = self.landmarker.detect_for_video(image, timestamp_milliseconds)
        if not result.face_landmarks:
            self.last_raw_landmarks = None
            self.last_raw_landmarks_3d = None
            return None
        # MediaPipe z is in the same normalised units as x (relative depth,
        # roughly centred on the face); scaled by the frame width like x.
        self.last_raw_landmarks_3d = numpy.array(
            [[landmark.x * self.frame_width, landmark.y * self.frame_height, landmark.z * self.frame_width]
             for landmark in result.face_landmarks[0]],
            dtype=numpy.float64,
        )
        self.last_raw_landmarks = self.last_raw_landmarks_3d[:, :2].copy()
        return self.last_raw_landmarks

    @staticmethod
    def face_size(landmarks):
        """Distance between the outer eye corners, in pixels: proportional to 1 / depth."""
        return float(numpy.linalg.norm(landmarks[263] - landmarks[33]))

    def process(self, frame_bgr, time_seconds):
        """Return (smoothed_landmarks, predicted_landmarks) as (N, 2) pixel arrays, or (None, None)."""
        raw_landmarks = self.detect_raw(frame_bgr, time_seconds)
        if raw_landmarks is None:
            self.smoothing_filter.reset()
            return None, None
        smoothed_3d, velocity_3d = self.smoothing_filter.update(self.last_raw_landmarks_3d, time_seconds)
        smoothed_landmarks = smoothed_3d[:, :2]
        velocity = velocity_3d[:, :2]
        self.last_velocity = velocity
        self.last_smoothed_z = smoothed_3d[:, 2]
        # Prediction only when the face really moves: velocity estimates are noisy even at
        # rest, and multiplied by the lead that noise becomes a visible wobble.
        speed = float(numpy.median(numpy.linalg.norm(velocity, axis=1)))
        gain = numpy.clip((speed - self.prediction_deadband_pixels_per_second)
                          / max(1e-6, self.prediction_full_pixels_per_second - self.prediction_deadband_pixels_per_second),
                          0.0, 1.0)
        self.last_prediction_gain = float(gain)
        predicted_landmarks = smoothed_landmarks + velocity * (self.prediction_lead_seconds * gain)
        return smoothed_landmarks, predicted_landmarks

    def close(self):
        self.landmarker.close()
