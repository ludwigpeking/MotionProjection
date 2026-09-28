"""Blinking border markers: the projector's own edges, measured live.

Eight small dots sit at the projector's border (corners and edge midpoints).
They blink in a repeating cycle of slots, each slot_seconds long:

    slot 0: all off (reference)   slot 1: all on
    slots 2..5: only the markers whose ID code has bit 0, 1, 2, 3 set

Camera frames are attributed to slots by their timestamp, allowing for the
projector-to-camera latency. Once a cycle has a frame for every slot, the
markers are found by differencing all-on against all-off (far more sensitive
than looking for faint static dots) and identified by reading their bits, so
no prediction, seed or previous state is needed. A homography fitted to the
identified markers is the live wall-plane mapping; it refreshes every cycle.
"""

import math

import cv2
import numpy

from dot_detector import detect_all_dots

SLOT_ALL_OFF = 0
SLOT_ALL_ON = 1
FIRST_BIT_SLOT = 2


def marker_projector_points(width, height, inset):
    left, right = inset, width - 1 - inset
    top, bottom = inset, height - 1 - inset
    middle_x, middle_y = width / 2.0, height / 2.0
    return numpy.array([
        [left, top], [middle_x, top], [right, top],
        [right, middle_y], [right, bottom], [middle_x, bottom],
        [left, bottom], [left, middle_y],
    ], dtype=numpy.float64)


def _sample_disc_peak(blurred_image, x, y, radius):
    left = int(max(0, x - radius))
    top = int(max(0, y - radius))
    right = int(min(blurred_image.shape[1], x + radius + 1))
    bottom = int(min(blurred_image.shape[0], y + radius + 1))
    if right <= left or bottom <= top:
        return 0.0
    return float(blurred_image[top:bottom, left:right].max())


class BlinkingBorderTracker:
    def __init__(self, config, projector_points, latency_none_before, latency_full_after):
        self.projector_points = projector_points
        self.marker_count = len(projector_points)
        self.bit_count = max(1, math.ceil(math.log2(self.marker_count + 1)))   # codes 1..marker_count
        self.slot_count = FIRST_BIT_SLOT + self.bit_count
        self.slot_seconds = config.border_marker_slot_seconds
        self.radius = config.border_marker_radius_pixels
        self.brightness = config.border_marker_brightness
        self.latency_none_before = latency_none_before
        self.latency_full_after = latency_full_after
        self.min_peak = config.border_marker_min_peak_difference
        self.max_area = config.border_marker_max_area_pixels
        self.sample_radius = config.border_marker_sample_radius_pixels
        self.min_margin = config.border_marker_code_min_margin
        self.min_found = config.border_marker_min_found
        self.ransac_threshold = config.servo_ransac_reprojection_threshold_pixels

        self.cycle_frames = {}          # slot -> gray frame, for the cycle being collected
        self.cycle_index = None
        self.homography = None
        self.camera_positions = {}      # marker index -> (x, y) from the last decoded cycle
        self.found_count = 0
        self.last_decode_time = None

    # ------------------------------------------------------------ drawing

    def _slot_at(self, time_seconds):
        slot_position = time_seconds / self.slot_seconds
        cycle = int(slot_position // self.slot_count)
        slot = int(slot_position) % self.slot_count
        return cycle, slot

    def markers_in_slot(self, slot):
        if slot == SLOT_ALL_OFF:
            return []
        if slot == SLOT_ALL_ON:
            return list(range(self.marker_count))
        bit = slot - FIRST_BIT_SLOT
        return [index for index in range(self.marker_count) if ((index + 1) >> bit) & 1]

    def draw(self, image, time_seconds):
        _, slot = self._slot_at(time_seconds)
        for index in self.markers_in_slot(slot):
            x, y = self.projector_points[index]
            cv2.circle(image, (int(round(x)), int(round(y))), self.radius, (self.brightness,) * 3, -1, cv2.LINE_AA)

    # ------------------------------------------------------------ observing

    def observe(self, gray, timestamp):
        """Feed a camera frame. Returns a fresh homography when a cycle completes, else None."""
        # The frame shows what was drawn about one latency earlier; find the slot
        # that was fully visible at this timestamp.
        drawn_at_latest = timestamp - self.latency_full_after      # slot must have started by then
        drawn_at_earliest = timestamp - self.latency_none_before   # and not ended before this
        cycle_a, slot_a = self._slot_at(drawn_at_latest)
        cycle_b, slot_b = self._slot_at(drawn_at_earliest)
        if (cycle_a, slot_a) != (cycle_b, slot_b):
            return None     # straddles a slot change: unusable
        if self.cycle_index != cycle_a:
            self.cycle_index = cycle_a
            self.cycle_frames = {}
        # Average every usable frame of the slot: the markers are faint on the wall.
        accumulated, count = self.cycle_frames.get(slot_a, (None, 0))
        gray_float = gray.astype(numpy.float32)
        accumulated = gray_float if accumulated is None else accumulated + gray_float
        self.cycle_frames[slot_a] = (accumulated, count + 1)
        if len(self.cycle_frames) < self.slot_count or slot_a != self.slot_count - 1:
            return None
        homography = self._decode()
        self.cycle_frames = {}
        self.last_decode_time = timestamp
        return homography

    def _slot_image(self, slot):
        accumulated, count = self.cycle_frames[slot]
        return (accumulated / count).clip(0, 255).astype(numpy.uint8)

    def _decode(self):
        reference = self._slot_image(SLOT_ALL_OFF)
        all_on = self._slot_image(SLOT_ALL_ON)
        blobs, _ = detect_all_dots(reference, all_on, self.min_peak, self.max_area,
                                   opening_radius_pixels=1, max_negative_ratio=0.6)
        blurred_reference = cv2.GaussianBlur(reference, (5, 5), 0).astype(numpy.float32)
        blurred_all_on = cv2.GaussianBlur(all_on, (5, 5), 0).astype(numpy.float32)
        blurred_bits = [cv2.GaussianBlur(self._slot_image(FIRST_BIT_SLOT + bit), (5, 5), 0).astype(numpy.float32)
                        for bit in range(self.bit_count)]

        decoded = {}
        for x, y, _ in blobs:
            dark = _sample_disc_peak(blurred_reference, x, y, self.sample_radius)
            bright = _sample_disc_peak(blurred_all_on, x, y, self.sample_radius)
            span = bright - dark
            if span < self.min_peak:
                continue
            code = 0
            worst_margin = float("inf")
            for bit, blurred_bit in enumerate(blurred_bits):
                level = (_sample_disc_peak(blurred_bit, x, y, self.sample_radius) - dark) / span
                worst_margin = min(worst_margin, abs(level - 0.5))
                if level > 0.5:
                    code |= 1 << bit
            index = code - 1
            if worst_margin < self.min_margin or index < 0 or index >= self.marker_count:
                continue
            if index in decoded and decoded[index][2] >= worst_margin:
                continue
            decoded[index] = (x, y, worst_margin)

        self.camera_positions = {index: (x, y) for index, (x, y, _) in decoded.items()}
        self.found_count = len(decoded)
        if self.found_count < self.min_found:
            return None
        camera_points = numpy.array([[x, y] for x, y, _ in decoded.values()])
        projector_points = numpy.array([self.projector_points[index] for index in decoded])
        homography, inlier_mask = cv2.findHomography(camera_points, projector_points, cv2.RANSAC, self.ransac_threshold)
        if homography is None or inlier_mask is None or int(inlier_mask.sum()) < self.min_found:
            return None
        if not numpy.all(numpy.isfinite(homography)):
            return None
        self.homography = homography
        return homography
