"""Find a projected pure-red disc in a camera frame."""

import cv2
import numpy


def find_red_disc(frame_bgr, center, search_radius, min_area):
    """Centroid of the largest strongly red blob near center, or None. The
    projected landmark disc is pure red; skin is reddish but far less so."""
    height, width = frame_bgr.shape[:2]
    left = int(max(0, center[0] - search_radius))
    top = int(max(0, center[1] - search_radius))
    right = int(min(width, center[0] + search_radius))
    bottom = int(min(height, center[1] + search_radius))
    if right <= left or bottom <= top:
        return None
    window = frame_bgr[top:bottom, left:right].astype(numpy.int16)
    blue, green, red = window[:, :, 0], window[:, :, 1], window[:, :, 2]
    red_mask = ((red - green > 70) & (red - blue > 70) & (red > 120)).astype(numpy.uint8)
    count, labels, statistics, centroids = cv2.connectedComponentsWithStats(red_mask, connectivity=8)
    if count < 2:
        return None
    largest = 1 + int(numpy.argmax(statistics[1:, cv2.CC_STAT_AREA]))
    if statistics[largest, cv2.CC_STAT_AREA] < min_area:
        return None
    return float(left + centroids[largest][0]), float(top + centroids[largest][1])


DOT_COLOURS_BGR = {"red": (0, 0, 255), "green": (0, 255, 0), "blue": (255, 0, 0)}
DOT_CHANNEL_INDEX = {"blue": 0, "green": 1, "red": 2}


def measure_projected_dot(frame_bgr, center, search_radius, min_area, colour="green", contrast_threshold=40,
                           brightness_threshold=100):
    """Looks for the projected dot near center. Returns (dot centre or None, strongest colour
    contrast found in the search window).

    The dot adds light of one projector colour to whatever it lands on. On skin in a lit
    room that does not make the pixel green: it makes it greener than the skin around it.
    So the measure is the colour's dominance (its channel minus the stronger of the other
    two) relative to the median dominance of the search window, which is the skin's own.
    A blob counts only if it is compact (roughly as wide as tall, mostly filled), which
    rejects streaks and edges. The strongest contrast is returned even when no dot is
    accepted, so a log shows by how much a detection was missed."""
    height, width = frame_bgr.shape[:2]
    left = int(max(0, center[0] - search_radius))
    top = int(max(0, center[1] - search_radius))
    right = int(min(width, center[0] + search_radius))
    bottom = int(min(height, center[1] + search_radius))
    if right <= left or bottom <= top:
        return None, 0.0
    window = frame_bgr[top:bottom, left:right].astype(numpy.int16)
    channel_index = DOT_CHANNEL_INDEX[colour]
    chosen = window[:, :, channel_index]
    others = numpy.delete(window, channel_index, axis=2).max(axis=2)
    dominance = chosen - others
    contrast = dominance - numpy.median(dominance)
    smoothed_contrast = cv2.blur(contrast.astype(numpy.float32), (5, 5))       # a single noisy pixel is not a dot
    strongest_contrast = float(smoothed_contrast.max())
    mask = ((smoothed_contrast > contrast_threshold) & (chosen > brightness_threshold)).astype(numpy.uint8)
    # A bright dot saturates the camera at its centre (all channels full, no dominant one) and
    # leaves a coloured ring: closing the mask fills that hole so the ring counts as one disc.
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
    count, labels, statistics, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    best_weight = 0.0
    best_centre = None
    for label_index in range(1, count):
        area = statistics[label_index, cv2.CC_STAT_AREA]
        blob_width = statistics[label_index, cv2.CC_STAT_WIDTH]
        blob_height = statistics[label_index, cv2.CC_STAT_HEIGHT]
        if area < min_area:
            continue
        if max(blob_width, blob_height) > 2.2 * min(blob_width, blob_height):
            continue
        if area < 0.45 * blob_width * blob_height:
            continue
        rows, columns = numpy.nonzero(labels == label_index)
        weights = numpy.maximum(contrast[rows, columns].astype(numpy.float64), 1.0)     # the saturated centre still counts
        total_weight = float(weights.sum())
        if total_weight > best_weight:
            best_weight = total_weight
            best_centre = (float(left + (columns * weights).sum() / total_weight),
                           float(top + (rows * weights).sum() / total_weight))
    return best_centre, strongest_contrast


def find_projected_dot(frame_bgr, center, search_radius, min_area, colour="green"):
    """Centre of the projected dot near center, or None (see measure_projected_dot)."""
    return measure_projected_dot(frame_bgr, center, search_radius, min_area, colour)[0]
