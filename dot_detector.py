"""Find projected dots by differencing a dots-on camera frame against a dots-off frame.

Three things separate a dot from everything else that changes between frames:
  * a dot only ever makes the surface brighter, so only the positive
    difference is searched, and a candidate is rejected if a comparable
    *negative* difference sits next to it (a moving edge brightens one side
    and darkens the other; a dot never darkens anything);
  * a dot is a filled disc, so a morphological opening with a disc slightly
    smaller than the dot removes thin edges (hair outline, jaw line) and
    leaves the dot intact;
  * a dot is compact, so blobs that are too large or too elongated are
    rejected as motion streaks.
"""

import math

import cv2
import numpy

# Why the most recent candidate was rejected; read it after a None result.
last_rejection_reason = ""


def _reject(reason):
    global last_rejection_reason
    last_rejection_reason = reason
    return None


def _difference_images(off_gray, on_gray, opening_radius_pixels):
    """Return (positive, negative) differences, both blurred; positive is also opened."""
    positive = cv2.GaussianBlur(cv2.subtract(on_gray, off_gray), (5, 5), 0)
    negative = cv2.GaussianBlur(cv2.subtract(off_gray, on_gray), (5, 5), 0)
    if opening_radius_pixels > 0:
        kernel_size = 2 * opening_radius_pixels + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        positive = cv2.morphologyEx(positive, cv2.MORPH_OPEN, kernel)
    return positive, negative


def _brightest_compact_blob(positive_window, negative_window, min_peak_difference,
                            max_blob_area_pixels, max_negative_ratio):
    """Return (x, y) of the intensity-weighted centroid of the blob at the brightest
    pixel, in window coordinates, or None if it does not look like a dot."""
    if positive_window.size == 0:
        return _reject("empty window")

    peak_value = float(positive_window.max())
    background_level = float(numpy.median(positive_window))
    if peak_value - background_level < min_peak_difference:
        return _reject(f"peak {peak_value:.0f} - background {background_level:.0f} < {min_peak_difference}")

    threshold = background_level + 0.5 * (peak_value - background_level)
    binary = (positive_window >= threshold).astype(numpy.uint8)
    component_count, labels, statistics, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if component_count < 2:
        return _reject("no component")

    peak_row, peak_column = numpy.unravel_index(int(positive_window.argmax()), positive_window.shape)
    peak_label = labels[peak_row, peak_column]
    if peak_label == 0:
        return _reject("peak not in a component")

    blob_area = statistics[peak_label, cv2.CC_STAT_AREA]
    blob_width = statistics[peak_label, cv2.CC_STAT_WIDTH]
    blob_height = statistics[peak_label, cv2.CC_STAT_HEIGHT]
    if blob_area > max_blob_area_pixels:
        return _reject(f"peak {peak_value:.0f} at ({peak_column},{peak_row}): area {blob_area} > {max_blob_area_pixels}")
    aspect_ratio = blob_width / max(blob_height, 1)
    if aspect_ratio < 0.4 or aspect_ratio > 2.5:
        return _reject(f"peak {peak_value:.0f} at ({peak_column},{peak_row}): aspect {aspect_ratio:.2f} ({blob_width}x{blob_height})")

    # Motion check: look for a darkening of similar strength near the blob.
    blob_left = statistics[peak_label, cv2.CC_STAT_LEFT]
    blob_top = statistics[peak_label, cv2.CC_STAT_TOP]
    margin = max(blob_width, blob_height)
    neighbourhood = negative_window[
        max(0, blob_top - margin):blob_top + blob_height + margin,
        max(0, blob_left - margin):blob_left + blob_width + margin,
    ]
    if neighbourhood.size and float(neighbourhood.max()) > max_negative_ratio * peak_value:
        return _reject(f"peak {peak_value:.0f} at ({peak_column},{peak_row}): "
                       f"negative {float(neighbourhood.max()):.0f} next to it (motion)")

    component_mask = labels == peak_label
    weights = positive_window.astype(numpy.float64) * component_mask
    weight_sum = weights.sum()
    if weight_sum <= 0:
        return _reject("zero weight")
    rows, columns = numpy.mgrid[0:positive_window.shape[0], 0:positive_window.shape[1]]
    centroid_x = float((columns * weights).sum() / weight_sum)
    centroid_y = float((rows * weights).sum() / weight_sum)
    return centroid_x, centroid_y


def detect_gated_dots(off_gray, on_gray, expected_points, search_radius_pixels,
                      min_peak_difference, max_blob_area_pixels,
                      opening_radius_pixels, max_negative_ratio):
    """Look for one dot near each expected camera point.

    Returns (found, positive_difference) where found is a list of
    (expected_index, (x, y)) for every dot that was detected.
    """
    positive, negative = _difference_images(off_gray, on_gray, opening_radius_pixels)
    frame_height, frame_width = positive.shape
    found = []
    for expected_index, (expected_x, expected_y) in enumerate(expected_points):
        left = int(max(0, expected_x - search_radius_pixels))
        top = int(max(0, expected_y - search_radius_pixels))
        right = int(min(frame_width, expected_x + search_radius_pixels))
        bottom = int(min(frame_height, expected_y + search_radius_pixels))
        if right <= left or bottom <= top:
            continue
        blob = _brightest_compact_blob(positive[top:bottom, left:right], negative[top:bottom, left:right],
                                       min_peak_difference, max_blob_area_pixels, max_negative_ratio)
        if blob is None:
            continue
        found.append((expected_index, (left + blob[0], top + blob[1])))
    return found, positive


def detect_all_dots(off_gray, on_gray, min_peak_difference, max_blob_area_pixels,
                    opening_radius_pixels, max_negative_ratio, max_count=200):
    """Find every dot-like blob in on - off. Returns (list of (x, y, peak), positive_difference)."""
    positive, negative = _difference_images(off_gray, on_gray, opening_radius_pixels)
    binary = (positive >= min_peak_difference).astype(numpy.uint8)
    component_count, labels, statistics, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    blobs = []
    for label in range(1, component_count):
        area = statistics[label, cv2.CC_STAT_AREA]
        width = statistics[label, cv2.CC_STAT_WIDTH]
        height = statistics[label, cv2.CC_STAT_HEIGHT]
        if area < 4 or area > max_blob_area_pixels:
            continue
        aspect_ratio = width / max(height, 1)
        if aspect_ratio < 0.4 or aspect_ratio > 2.5:
            continue
        left = statistics[label, cv2.CC_STAT_LEFT]
        top = statistics[label, cv2.CC_STAT_TOP]
        component_mask = labels[top:top + height, left:left + width] == label
        window = positive[top:top + height, left:left + width].astype(numpy.float64) * component_mask
        peak = float(window.max())
        margin = max(width, height)
        neighbourhood = negative[max(0, top - margin):top + height + margin,
                                 max(0, left - margin):left + width + margin]
        if neighbourhood.size and float(neighbourhood.max()) > max_negative_ratio * peak:
            continue
        weight_sum = window.sum()
        if weight_sum <= 0:
            continue
        rows, columns = numpy.mgrid[0:height, 0:width]
        blobs.append((left + float((columns * window).sum() / weight_sum),
                      top + float((rows * window).sum() / weight_sum), peak))
    blobs.sort(key=lambda blob: -blob[2])
    return blobs[:max_count], positive


def _greedy_unique_assignment(distances, tolerance_pixels):
    """Pair rows (expected) with columns (blobs), closest first, each used once."""
    assignment = []
    used_rows = set()
    used_columns = set()
    order = numpy.dstack(numpy.unravel_index(numpy.argsort(distances, axis=None), distances.shape))[0]
    for row, column in order:
        if distances[row, column] > tolerance_pixels:
            break
        if row in used_rows or column in used_columns:
            continue
        used_rows.add(row)
        used_columns.add(column)
        assignment.append((int(row), int(column)))
    return assignment


def match_dot_pattern(blobs, expected_points, tolerance_pixels, min_matches, max_offset_pixels,
                      scale_range=(0.7, 1.4), max_rotation_degrees=25.0):
    """Identify which blob is which aimed dot.

    Every pair of aimed points matched to a pair of blobs defines a similarity
    (scale, rotation, translation); the hypothesis under which the most aimed
    points land on a blob wins, provided its scale, rotation and translation
    are plausible. This tolerates a mapping that is off in scale or rotation
    while still refusing to line the pattern up with the wrong dots.

    blobs: list of (x, y, peak). expected_points: (N, 2) array.
    Returns (found, offset) with found = [(expected_index, (x, y)), ...] and
    offset = the hypothesis translation, or ([], None).
    """
    if len(blobs) < 2 or len(expected_points) < 2:
        return [], None
    blob_positions = numpy.array([[x, y] for x, y, _ in blobs], dtype=numpy.float64)
    expected = numpy.asarray(expected_points, dtype=numpy.float64)
    expected_count = len(expected)
    blob_count = len(blob_positions)
    max_rotation_radians = math.radians(max_rotation_degrees)

    best_assignment = []
    best_offset = None
    best_score = (0, float("inf"))    # (match count, total distance): more matches, then tighter
    for first_expected in range(expected_count):
        for second_expected in range(first_expected + 1, expected_count):
            expected_vector = expected[second_expected] - expected[first_expected]
            expected_length = numpy.linalg.norm(expected_vector)
            if expected_length < 1e-6:
                continue
            for first_blob in range(blob_count):
                for second_blob in range(blob_count):
                    if first_blob == second_blob:
                        continue
                    blob_vector = blob_positions[second_blob] - blob_positions[first_blob]
                    blob_length = numpy.linalg.norm(blob_vector)
                    scale = blob_length / expected_length
                    if not (scale_range[0] <= scale <= scale_range[1]):
                        continue
                    rotation = math.atan2(blob_vector[1], blob_vector[0]) - math.atan2(expected_vector[1], expected_vector[0])
                    rotation = (rotation + math.pi) % (2 * math.pi) - math.pi
                    if abs(rotation) > max_rotation_radians:
                        continue
                    cosine, sine = scale * math.cos(rotation), scale * math.sin(rotation)
                    rotation_matrix = numpy.array([[cosine, -sine], [sine, cosine]])
                    translation = blob_positions[first_blob] - rotation_matrix @ expected[first_expected]
                    offset = translation + rotation_matrix @ expected.mean(axis=0) - expected.mean(axis=0)
                    if numpy.linalg.norm(offset) > max_offset_pixels:
                        continue
                    transformed = expected @ rotation_matrix.T + translation
                    distances = numpy.linalg.norm(transformed[:, None, :] - blob_positions[None, :, :], axis=2)
                    assignment = _greedy_unique_assignment(distances, tolerance_pixels)
                    total_distance = sum(distances[row, column] for row, column in assignment)
                    if (len(assignment), -total_distance) > (best_score[0], -best_score[1]):
                        best_score = (len(assignment), total_distance)
                        best_assignment = assignment
                        best_offset = offset

    if best_score[0] < min_matches:
        return [], None
    found = [(expected_index, (float(blob_positions[blob_index][0]), float(blob_positions[blob_index][1])))
             for expected_index, blob_index in best_assignment]
    return found, best_offset


def detect_single_dot(off_gray, on_gray, min_peak_difference, max_blob_area_pixels,
                      opening_radius_pixels, max_negative_ratio, accept_rectangle=None):
    """Look for exactly one dot, anywhere in the frame or only inside
    accept_rectangle = (left, top, right, bottom) in camera pixels."""
    positive, negative = _difference_images(off_gray, on_gray, opening_radius_pixels)
    frame_height, frame_width = positive.shape
    if accept_rectangle is None:
        left, top, right, bottom = 0, 0, frame_width, frame_height
    else:
        left = int(max(0, accept_rectangle[0]))
        top = int(max(0, accept_rectangle[1]))
        right = int(min(frame_width, accept_rectangle[2]))
        bottom = int(min(frame_height, accept_rectangle[3]))
    blob = _brightest_compact_blob(positive[top:bottom, left:right], negative[top:bottom, left:right],
                                   min_peak_difference, max_blob_area_pixels, max_negative_ratio)
    if blob is not None:
        blob = (left + blob[0], top + blob[1])
    return blob, positive
