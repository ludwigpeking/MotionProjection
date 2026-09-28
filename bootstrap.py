"""Initial calibration with temporally coded dots, compensated for face motion.

Frames shown: black (reference), all dots on, then one frame per ID bit with
only the dots whose bit is 1. A dot's position comes from the all-on frame;
each bit is read as "closer to the all-on level than to the black level" at
that position. No markers, no decoding of shapes.

The person moves during the ~3 s capture, so for blobs inside the face box
every sample position is carried into the other frames with a similarity
transform fitted between that frame's MediaPipe landmarks and the all-on
frame's, and the reference frame is warped the same way before differencing.
Wall dots are static and need none of that.

Dots inside the face box are at face depth, which is the plane the mask is
drawn on, so they are preferred for the fit whenever there are enough.
"""

import math
import os
import time

import cv2
import numpy

from dot_detector import detect_all_dots


def _show_and_capture(projector_window, camera, face_tracker, image, config, overlay=None):
    """Show image, wait for it to be fully visible, then return
    (mean gray float32 of the next frames, raw landmarks of the first one or None)."""
    if overlay is not None:
        image = overlay(image.copy())
    projector_window.show(image)
    cv2.waitKey(1)
    shown_at = time.perf_counter()
    next_frame_after = shown_at + config.latency_full_after_seconds + config.bootstrap_extra_settle_seconds
    accumulated = None
    captured = 0
    landmarks = None
    for _ in range(config.bootstrap_average_frames):
        result = camera.frame_after(next_frame_after, timeout_seconds=config.bootstrap_frame_timeout_seconds)
        if result is None:
            break
        timestamp, frame = result
        next_frame_after = timestamp
        if captured == 0:
            landmarks = face_tracker.detect_raw(frame, timestamp)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(numpy.float32)
        accumulated = gray if accumulated is None else accumulated + gray
        captured += 1
        cv2.waitKey(1)
    if captured == 0:
        return None, None
    return accumulated / captured, landmarks


def _face_rectangle(landmarks, expand_fraction):
    left, top = landmarks.min(axis=0)
    right, bottom = landmarks.max(axis=0)
    width = right - left
    height = bottom - top
    return (left - expand_fraction * width, top - expand_fraction * height,
            right + expand_fraction * width, bottom + expand_fraction * height)


def _inside(rectangle, x, y):
    return rectangle[0] <= x <= rectangle[2] and rectangle[1] <= y <= rectangle[3]


def _similarity_between(landmarks_from, landmarks_to):
    """2x3 similarity mapping points in the 'from' frame to the 'to' frame, or None."""
    if landmarks_from is None or landmarks_to is None:
        return None
    similarity, _ = cv2.estimateAffinePartial2D(landmarks_from.astype(numpy.float32),
                                                landmarks_to.astype(numpy.float32))
    return similarity


def _grid_points(width, height, margin_fraction, columns, rows, offset_fraction=(0.0, 0.0), edge_clearance_pixels=0):
    """Grid of projector points. offset_fraction shifts the whole grid by that fraction of
    the spacing between columns and between rows; points pushed closer to the image edge
    than edge_clearance_pixels are left out."""
    column_positions = numpy.linspace(margin_fraction * width, (1 - margin_fraction) * width, columns)
    row_positions = numpy.linspace(margin_fraction * height, (1 - margin_fraction) * height, rows)
    column_spacing = column_positions[1] - column_positions[0] if columns > 1 else 0.0
    row_spacing = row_positions[1] - row_positions[0] if rows > 1 else 0.0
    points = []
    for projector_y in row_positions + offset_fraction[1] * row_spacing:
        for projector_x in column_positions + offset_fraction[0] * column_spacing:
            if (edge_clearance_pixels <= projector_x <= width - 1 - edge_clearance_pixels
                    and edge_clearance_pixels <= projector_y <= height - 1 - edge_clearance_pixels):
                points.append((float(projector_x), float(projector_y)))
    return points


def _dots_image(width, height, points, selected_ids, radius):
    image = numpy.zeros((height, width, 3), dtype=numpy.uint8)
    for dot_id in selected_ids:
        x, y = points[dot_id]
        cv2.circle(image, (int(round(x)), int(round(y))), radius, (255, 255, 255), -1, cv2.LINE_AA)
    return image


def _sample_disc_peak(blurred_image, x, y, radius):
    """Brightest value of a blurred image within a disc: tolerant to a few
    pixels of error in where the dot is expected."""
    left = int(max(0, x - radius))
    top = int(max(0, y - radius))
    right = int(min(blurred_image.shape[1], x + radius + 1))
    bottom = int(min(blurred_image.shape[0], y + radius + 1))
    if right <= left or bottom <= top:
        return 0.0
    rows, columns = numpy.mgrid[top:bottom, left:right]
    inside = (rows - y) ** 2 + (columns - x) ** 2 <= radius ** 2
    if not inside.any():
        return 0.0
    return float(blurred_image[top:bottom, left:right][inside].max())


def _blur(image):
    return cv2.GaussianBlur(image, (5, 5), 0)


def _sample_annulus_median(blurred_image, x, y, inner_radius, outer_radius):
    """Median of a ring around (x, y): the local background in that same frame,
    so a dot's level is judged against the skin actually under it right now."""
    left = int(max(0, x - outer_radius))
    top = int(max(0, y - outer_radius))
    right = int(min(blurred_image.shape[1], x + outer_radius + 1))
    bottom = int(min(blurred_image.shape[0], y + outer_radius + 1))
    if right <= left or bottom <= top:
        return 0.0
    rows, columns = numpy.mgrid[top:bottom, left:right]
    distance_squared = (rows - y) ** 2 + (columns - x) ** 2
    ring = (distance_squared > inner_radius ** 2) & (distance_squared <= outer_radius ** 2)
    if not ring.any():
        return 0.0
    return float(numpy.median(blurred_image[top:bottom, left:right][ring]))


def run_bootstrap(camera, projector_window, face_tracker, config, debug_window_name=None, overlay=None,
                  grid_columns=None, grid_rows=None, dot_radius=None, opening_radius=None,
                  sample_radius=None, return_all=False, grid_offset_fraction=(0.0, 0.0), static_scene=False):
    """Return a list of ((camera_x, camera_y), (projector_x, projector_y)) pairs,
    restricted to the face when enough dots landed there."""
    width = projector_window.width
    height = projector_window.height
    grid_columns = grid_columns or config.bootstrap_grid_columns
    grid_rows = grid_rows or config.bootstrap_grid_rows
    opening_radius = config.bootstrap_opening_radius_pixels if opening_radius is None else opening_radius
    sample_radius = config.bootstrap_code_sample_radius_pixels if sample_radius is None else sample_radius
    radius = dot_radius or config.bootstrap_dot_radius_pixels
    points = _grid_points(width, height, config.bootstrap_grid_margin_fraction, grid_columns, grid_rows,
                          grid_offset_fraction, edge_clearance_pixels=radius)
    dot_count = len(points)
    bit_count = max(1, math.ceil(math.log2(dot_count + 1)))   # codes 1..dot_count
    radius = dot_radius or config.bootstrap_dot_radius_pixels
    black = numpy.zeros((height, width, 3), dtype=numpy.uint8)
    print(f"[bootstrap] {dot_count} coded dots, {bit_count} bits, {2 + bit_count} frames; hold still ...")

    # Warm up: the first presentation of a fullscreen window can be slow.
    projector_window.show(overlay(black.copy()) if overlay is not None else black)
    cv2.waitKey(1)
    camera.frame_after(time.perf_counter() + 0.5, timeout_seconds=2.0)

    all_on_image = _dots_image(width, height, points, range(dot_count), radius)
    for attempt in range(3):
        reference, reference_landmarks = _show_and_capture(projector_window, camera, face_tracker, black, config, overlay)
        all_on, all_on_landmarks = _show_and_capture(projector_window, camera, face_tracker, all_on_image, config, overlay)
        all_on_landmarks_3d = face_tracker.last_raw_landmarks_3d if all_on_landmarks is not None else None
        if reference is None or all_on is None:
            continue
        # Verify the dots really were captured: a person moving also brightens
        # pixels, but only dots make many compact blobs.
        probe_blobs, _ = detect_all_dots(
            reference.clip(0, 255).astype(numpy.uint8), all_on.clip(0, 255).astype(numpy.uint8),
            config.bootstrap_min_peak_difference, config.bootstrap_max_blob_area_pixels,
            opening_radius, config.dot_max_negative_ratio)
        if len(probe_blobs) >= config.bootstrap_min_blobs_for_valid_capture:
            break
        print(f"[bootstrap] all-on frame shows only {len(probe_blobs)} blobs; waiting longer and retrying")
        camera.frame_after(time.perf_counter() + 0.4 * (attempt + 1), timeout_seconds=3.0)
    else:
        print("[bootstrap] the camera never saw the dots; is the projector showing the second screen?")
        return [], []
    bit_frames = []
    for bit in range(bit_count):
        ones = [dot_id for dot_id in range(dot_count) if ((dot_id + 1) >> bit) & 1]
        bit_frames.append(_show_and_capture(projector_window, camera, face_tracker,
                                            _dots_image(width, height, points, ones, radius), config, overlay))
    projector_window.show(overlay(black.copy()) if overlay is not None else black)
    cv2.waitKey(1)

    if reference is None or all_on is None or any(frame is None for frame, _ in bit_frames):
        print("[bootstrap] camera delivered no frames")
        return [], []

    face_rectangle = None
    if all_on_landmarks is not None:
        face_rectangle = _face_rectangle(all_on_landmarks, config.bootstrap_face_box_expand_fraction)

    # Reference for blob detection: inside the face box, the black frame warped
    # to follow the face as seen in the all-on frame.
    detection_reference = reference
    if face_rectangle is not None:
        to_all_on = _similarity_between(reference_landmarks, all_on_landmarks)
        if to_all_on is not None:
            warped = cv2.warpAffine(reference, to_all_on, (reference.shape[1], reference.shape[0]),
                                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            detection_reference = reference.copy()
            left, top, right, bottom = [int(value) for value in face_rectangle]
            left, top = max(0, left), max(0, top)
            right, bottom = min(reference.shape[1], right), min(reference.shape[0], bottom)
            detection_reference[top:bottom, left:right] = warped[top:bottom, left:right]

    reference_gray = detection_reference.clip(0, 255).astype(numpy.uint8)
    all_on_gray = all_on.clip(0, 255).astype(numpy.uint8)
    blobs, difference = detect_all_dots(
        reference_gray, all_on_gray,
        config.bootstrap_min_peak_difference, config.bootstrap_max_blob_area_pixels,
        opening_radius, config.dot_max_negative_ratio)
    print(f"[bootstrap] {len(blobs)} blobs in the all-on frame")

    landmark_flags = "".join("y" if landmarks is not None else "-" for _, landmarks in bit_frames)
    print(f"[bootstrap] face landmarks: reference {'y' if reference_landmarks is not None else '-'}, "
          f"all-on {'y' if all_on_landmarks is not None else '-'}, bit frames {landmark_flags}")

    # A projected dot stays where the projector's ray hits, so its camera
    # position is the same in every frame even when the face moves under it.
    # Sample every frame at the blob position; the dark level comes from the
    # reference warped to the all-on face position (same skin under the dot).
    blurred_reference = _blur(detection_reference)
    blurred_all_on = _blur(all_on)
    blurred_bit_frames = [_blur(frame) for frame, _ in bit_frames]
    if static_scene:
        # Nothing moves, so every frame is compared with the black reference itself: what the
        # pattern added at the dot. A dot blurred by defocus (a near object while the projector
        # is focused far away) reads correctly this way; a ring around it would lie inside the blur.
        added_by_all_on = _blur(all_on - detection_reference)
        added_by_bit_frames = [_blur(frame - detection_reference) for frame, _ in bit_frames]
    decoded = {}
    rejected_weak = 0
    rejected_ambiguous = 0
    ring_inner = sample_radius + 2
    ring_outer = sample_radius + 7
    for x, y, _ in blobs:
        on_face = face_rectangle is not None and _inside(face_rectangle, x, y)
        # Each frame is judged against its own local background (a ring around the
        # dot), so skin sliding under a fixed dot while the head moves cannot flip a bit.
        if static_scene:
            bright_level = _sample_disc_peak(added_by_all_on, x, y, sample_radius)
            dark_level = 0.0
        else:
            bright_level = _sample_disc_peak(blurred_all_on, x, y, sample_radius)             - _sample_annulus_median(blurred_all_on, x, y, ring_inner, ring_outer)
            dark_level = _sample_disc_peak(blurred_reference, x, y, sample_radius)             - _sample_annulus_median(blurred_reference, x, y, ring_inner, ring_outer)
        span = bright_level - dark_level
        if span < config.bootstrap_min_peak_difference:
            rejected_weak += 1
            continue

        code = 0
        worst_margin = float("inf")
        levels = []
        for bit, blurred_bit_frame in enumerate(blurred_bit_frames):
            if static_scene:
                bit_level = _sample_disc_peak(added_by_bit_frames[bit], x, y, sample_radius)
            else:
                bit_level = _sample_disc_peak(blurred_bit_frame, x, y, sample_radius)                 - _sample_annulus_median(blurred_bit_frame, x, y, ring_inner, ring_outer)
            level = (bit_level - dark_level) / span
            levels.append(level)
            worst_margin = min(worst_margin, abs(level - 0.5))
            if level > 0.5:
                code |= 1 << bit
        dot_id = code - 1     # codes start at 1 so an all-dark reading is never a valid dot
        if on_face and config.bootstrap_save_debug_images:
            print(f"[bootstrap]   face blob ({x:.0f},{y:.0f}): dark {dark_level:.0f} bright {bright_level:.0f} "
                  f"levels {' '.join(f'{level:.2f}' for level in levels)} -> id {dot_id}")
        if worst_margin < config.bootstrap_code_min_margin or dot_id < 0 or dot_id >= dot_count:
            rejected_ambiguous += 1
            continue
        if dot_id in decoded and decoded[dot_id][2] >= worst_margin:
            continue
        decoded[dot_id] = (x, y, worst_margin, on_face)

    def depth_at(x, y):
        """MediaPipe z of the landmark nearest to a camera point (face dots only)."""
        if all_on_landmarks_3d is None:
            return None
        nearest = int(numpy.argmin(numpy.hypot(all_on_landmarks_3d[:, 0] - x, all_on_landmarks_3d[:, 1] - y)))
        return float(all_on_landmarks_3d[nearest, 2])

    all_pairs = [((x, y), points[dot_id], depth_at(x, y) if on_face else None)
                 for dot_id, (x, y, _, on_face) in sorted(decoded.items())]
    face_pairs = [pair for pair, (_, (_, _, _, on_face)) in zip(all_pairs, sorted(decoded.items())) if on_face]
    print(f"[bootstrap] {len(all_pairs)} dots decoded ({len(face_pairs)} on the face); "
          f"rejected {rejected_weak} weak, {rejected_ambiguous} ambiguous")

    if debug_window_name is not None or config.bootstrap_save_debug_images:
        view = cv2.applyColorMap(difference, cv2.COLORMAP_INFERNO)
        for x, y, _ in blobs:
            cv2.circle(view, (int(x), int(y)), 10, (90, 90, 90), 1)
        for (x, y), (projector_x, projector_y), _ in all_pairs:
            color = (0, 255, 0) if (face_rectangle is not None and _inside(face_rectangle, x, y)) else (255, 200, 0)
            cv2.circle(view, (int(x), int(y)), 12, color, 2)
            cv2.putText(view, f"{projector_x:.0f},{projector_y:.0f}", (int(x) + 12, int(y) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        if face_rectangle is not None:
            cv2.rectangle(view, (int(face_rectangle[0]), int(face_rectangle[1])),
                          (int(face_rectangle[2]), int(face_rectangle[3])), (0, 200, 255), 2)
        cv2.putText(view, f"bootstrap: {len(all_pairs)} decoded, {len(face_pairs)} on face", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        if debug_window_name is not None:
            cv2.imshow(debug_window_name, view)
            cv2.waitKey(1)
        if config.bootstrap_save_debug_images:
            os.makedirs(config.debug_image_directory, exist_ok=True)
            cv2.imwrite(os.path.join(config.debug_image_directory, "bootstrap_decoded.png"), view)
            cv2.imwrite(os.path.join(config.debug_image_directory, "bootstrap_all_on.png"), all_on_gray)
            cv2.imwrite(os.path.join(config.debug_image_directory, "bootstrap_reference.png"),
                        reference.clip(0, 255).astype(numpy.uint8))
            for bit, (frame, _) in enumerate(bit_frames):
                cv2.imwrite(os.path.join(config.debug_image_directory, f"bootstrap_bit{bit}.png"),
                            frame.clip(0, 255).astype(numpy.uint8))

    wall_pairs = [pair for pair, (_, (_, _, _, on_face)) in zip(all_pairs, sorted(decoded.items())) if not on_face]
    if config.bootstrap_save_debug_images:
        import json
        with open(os.path.join(config.debug_image_directory, "bootstrap_pairs.json"), "w") as file:
            json.dump([{"camera": [float(x), float(y)], "projector": [float(v) for v in points[dot_id]],
                        "on_face": bool(on_face), "z": depth_at(x, y) if on_face else None}
                       for dot_id, (x, y, _, on_face) in sorted(decoded.items())], file, indent=1)
    if return_all:
        print(f"[bootstrap] returning all {len(all_pairs)} correspondences (dense scan)")
        return all_pairs, wall_pairs
    if len(face_pairs) >= 4:
        face_camera = numpy.array([pair[0] for pair in face_pairs], dtype=numpy.float64)
        face_projector = numpy.array([pair[1] for pair in face_pairs], dtype=numpy.float64)
        _, consistency_mask = cv2.findHomography(face_camera, face_projector, cv2.RANSAC, 8.0)
        consistent = int(consistency_mask.sum()) if consistency_mask is not None else 0
        print(f"[bootstrap] face dot self-consistency: {consistent}/{len(face_pairs)} agree with one plane to 8 px")
    if len(face_pairs) >= config.bootstrap_min_face_pairs:
        print(f"[bootstrap] using the {len(face_pairs)} face correspondences ({len(wall_pairs)} wall dots for the beam tracker)")
        return face_pairs, wall_pairs
    if face_rectangle is None:
        print("[bootstrap] NO FACE DETECTED during the scan: sit facing the camera, hands down, hold still")
    else:
        print(f"[bootstrap] FACE OUTSIDE THE BEAM: only {len(face_pairs)} dots hit the face box "
              f"({len(all_pairs)} decoded elsewhere) - tilt the projector so the face is inside the cyan quad, then press b")
    print("[bootstrap] not enough face hits; using every decoded dot (wall plane)")
    return all_pairs, wall_pairs
