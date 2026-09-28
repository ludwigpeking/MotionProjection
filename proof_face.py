"""Face geometry proof: project a marker at a face landmark, measure where it lands.

For each chosen landmark: read the current landmarks (with MediaPipe z), map
the landmark to a projector pixel with the estimator (with or without the
per-landmark depth term), project a small disc there, capture off and on
frames, find the disc in the camera, and compare with the landmark's position
in the on frame. The error per landmark, in camera pixels, is the evidence.
"""

import time

import cv2
import numpy

from dot_detector import detect_all_dots


def _capture_average(camera, face_tracker, shown_at, settle_seconds, average_frames):
    """Mean gray of a few frames after settle, plus the raw landmarks of the first."""
    next_after = shown_at + settle_seconds
    accumulated = None
    captured = 0
    landmarks = None
    for _ in range(average_frames):
        result = camera.frame_after(next_after, timeout_seconds=2.0)
        if result is None:
            break
        timestamp, frame = result
        next_after = timestamp
        if captured == 0:
            landmarks = face_tracker.detect_raw(frame, timestamp)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(numpy.float32)
        accumulated = gray if accumulated is None else accumulated + gray
        captured += 1
        cv2.waitKey(1)
    if captured == 0:
        return None, None
    return (accumulated / captured).clip(0, 255).astype(numpy.uint8), landmarks


def run_face_proof(camera, projector_window, face_tracker, estimator, config, landmark_indices,
                   use_depth, debug_window_name=None):
    """Return list of (landmark_index, aimed_camera_point, measured_point or None, error or None)."""
    width, height = projector_window.width, projector_window.height
    black = numpy.zeros((height, width, 3), dtype=numpy.uint8)
    settle = config.latency_full_after_seconds + config.bootstrap_extra_settle_seconds
    previous_state = estimator.depth_model_enabled
    estimator.depth_model_enabled = use_depth
    results = []
    label = "with depth" if use_depth else "no depth"
    print(f"[proof-face] {label}: {len(landmark_indices)} landmarks ...")
    for landmark_index in landmark_indices:
        projector_window.show(black)
        cv2.waitKey(1)
        off, landmarks_off = _capture_average(camera, face_tracker, time.perf_counter(), settle, 2)
        if off is None or landmarks_off is None:
            results.append((landmark_index, None, None, None))
            print(f"[proof-face]   landmark {landmark_index}: no face")
            continue
        landmarks_3d = face_tracker.last_raw_landmarks_3d
        face_size = face_tracker.face_size(landmarks_off)
        aim = landmarks_off[landmark_index]
        depth = landmarks_3d[landmark_index, 2] if use_depth else None
        projector_point = estimator.map_points([aim], face_size, None if depth is None else [depth])[0]
        if not (0 <= projector_point[0] < width and 0 <= projector_point[1] < height):
            results.append((landmark_index, aim, None, None))
            print(f"[proof-face]   landmark {landmark_index}: outside the beam")
            continue
        marker = black.copy()
        cv2.circle(marker, (int(round(projector_point[0])), int(round(projector_point[1]))),
                   config.proof_marker_radius_pixels, (255, 255, 255), -1, cv2.LINE_AA)
        projector_window.show(marker)
        cv2.waitKey(1)
        on, landmarks_on = _capture_average(camera, face_tracker, time.perf_counter(), settle, 2)
        if on is None:
            results.append((landmark_index, aim, None, None))
            continue
        # Compare against where the landmark is in the on frame (the head may have moved a little).
        target = landmarks_on[landmark_index] if landmarks_on is not None else aim
        blobs, _ = detect_all_dots(off, on, config.pointer_min_difference, config.bootstrap_max_blob_area_pixels,
                                   config.scan_opening_radius_pixels, config.dot_max_negative_ratio)
        candidates = [(numpy.hypot(x - target[0], y - target[1]), x, y) for x, y, _ in blobs]
        candidates = [item for item in candidates if item[0] <= config.pointer_search_radius_pixels]
        if not candidates:
            results.append((landmark_index, target, None, None))
            print(f"[proof-face]   landmark {landmark_index}: marker not seen")
            continue
        _, measured_x, measured_y = min(candidates)
        measured = numpy.array([measured_x, measured_y])
        error = float(numpy.linalg.norm(measured - target))
        results.append((landmark_index, target, measured, error))
        print(f"[proof-face]   landmark {landmark_index}: aimed camera ({target[0]:.0f},{target[1]:.0f}) "
              f"z {landmarks_3d[landmark_index, 2]:+.0f} -> projector ({projector_point[0]:.0f},{projector_point[1]:.0f}); "
              f"seen at ({measured[0]:.0f},{measured[1]:.0f}); error {error:.1f} px")
        if debug_window_name is not None:
            view = cv2.cvtColor(on, cv2.COLOR_GRAY2BGR)
            cv2.drawMarker(view, (int(target[0]), int(target[1])), (0, 255, 255), cv2.MARKER_CROSS, 30, 2)
            cv2.circle(view, (int(measured[0]), int(measured[1])), 10, (0, 0, 255), 2)
            cv2.putText(view, f"{label}: landmark {landmark_index} error {error:.1f} px", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imshow(debug_window_name, view)
            cv2.waitKey(1)
    projector_window.show(black)
    cv2.waitKey(1)
    estimator.depth_model_enabled = previous_state
    errors = [item[3] for item in results if item[3] is not None]
    if errors:
        print(f"[proof-face] {label}: {len(errors)}/{len(results)} measured, mean {numpy.mean(errors):.1f} px, "
              f"median {numpy.median(errors):.1f} px, max {numpy.max(errors):.1f} px")
    else:
        print(f"[proof-face] {label}: nothing measured")
    return results


def draw_face_proof(frame_bgr, results_with_depth, results_without, path):
    view = frame_bgr.copy()
    for results, color, label, row in ((results_without, (0, 165, 255), "no depth", 30),
                                       (results_with_depth, (0, 255, 0), "with depth", 60)):
        errors = [item[3] for item in results if item[3] is not None]
        summary = (f"{label}: {len(errors)}/{len(results)} measured, mean {numpy.mean(errors):.1f} px, max {numpy.max(errors):.1f} px"
                   if errors else f"{label}: nothing measured")
        cv2.putText(view, summary, (10, row), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 5)
        cv2.putText(view, summary, (10, row), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
        for landmark_index, target, measured, error in results:
            if target is None:
                continue
            x, y = int(target[0]), int(target[1])
            cv2.drawMarker(view, (x, y), (0, 255, 255), cv2.MARKER_CROSS, 16, 1)
            if measured is not None:
                cv2.circle(view, (int(measured[0]), int(measured[1])), 6, color, 2)
                cv2.line(view, (x, y), (int(measured[0]), int(measured[1])), color, 1)
    cv2.imwrite(path, view)
    return view
