"""Proof sweep: is camera -> projector geometrically right, depth included?

Held-out camera points are chosen across the scanned scene (between measured
dots). For each, a ring is projected at the point the scene map gives, and
the camera measures where the ring actually appears. The offset between the
chosen point and the measured ring centre, in camera pixels, is the error.
"""

import os
import time

import cv2
import numpy

from dot_detector import detect_all_dots


def choose_test_points(scene_map, frame_width, frame_height, count, min_dot_distance, max_dot_distance, rng):
    """Spread test points over the scanned area, each near measured dots but not on one."""
    camera_points = scene_map.camera_points
    left, top = camera_points.min(axis=0)
    right, bottom = camera_points.max(axis=0)
    chosen = []
    attempts = 0
    while len(chosen) < count and attempts < 5000:
        attempts += 1
        candidate = numpy.array([rng.uniform(left, right), rng.uniform(top, bottom)])
        if not (0 <= candidate[0] < frame_width and 0 <= candidate[1] < frame_height):
            continue
        nearest = scene_map.nearest_distance(candidate)
        if nearest < min_dot_distance or nearest > max_dot_distance:
            continue
        if any(numpy.linalg.norm(candidate - point) < 80 for point in chosen):
            continue
        if scene_map.query(candidate) is None:
            continue
        chosen.append(candidate)
    return chosen


def _capture_after(camera, shown_at, settle_seconds, average_frames):
    next_after = shown_at + settle_seconds
    accumulated = None
    captured = 0
    for _ in range(average_frames):
        result = camera.frame_after(next_after, timeout_seconds=2.0)
        if result is None:
            break
        timestamp, frame = result
        next_after = timestamp
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(numpy.float32)
        accumulated = gray if accumulated is None else accumulated + gray
        captured += 1
        cv2.waitKey(1)
    if captured == 0:
        return None
    return (accumulated / captured).clip(0, 255).astype(numpy.uint8)


def run_proof(camera, projector_window, scene_map, config, test_points, debug_window_name=None):
    """Return a list of (camera_point, projector_point, measured_point or None, error_px or None)."""
    width, height = projector_window.width, projector_window.height
    black = numpy.zeros((height, width, 3), dtype=numpy.uint8)
    settle = config.latency_full_after_seconds + config.bootstrap_extra_settle_seconds
    results = []
    print(f"[proof] testing {len(test_points)} held-out points ...")
    for index, camera_point in enumerate(test_points):
        projector_point = scene_map.query(camera_point)
        if projector_point is None:
            results.append((camera_point, None, None, None))
            continue
        ring_image = black.copy()
        center = (int(round(projector_point[0])), int(round(projector_point[1])))
        cv2.circle(ring_image, center, config.proof_marker_radius_pixels, (255, 255, 255), -1, cv2.LINE_AA)

        projector_window.show(black)
        cv2.waitKey(1)
        off = _capture_after(camera, time.perf_counter(), settle, 2)
        projector_window.show(ring_image)
        cv2.waitKey(1)
        on = _capture_after(camera, time.perf_counter(), settle, 2)
        if off is None or on is None:
            results.append((camera_point, projector_point, None, None))
            continue
        # Same detector as the scan: the blob nearest to the chosen point.
        blobs, _ = detect_all_dots(off, on, config.pointer_min_difference, config.bootstrap_max_blob_area_pixels,
                                   config.scan_opening_radius_pixels, config.dot_max_negative_ratio)
        radius = config.pointer_search_radius_pixels
        candidates = [(numpy.hypot(x - camera_point[0], y - camera_point[1]), x, y) for x, y, _ in blobs]
        candidates = [item for item in candidates if item[0] <= radius]
        if not candidates:
            results.append((camera_point, projector_point, None, None))
            print(f"[proof]   point {index}: camera ({camera_point[0]:.0f},{camera_point[1]:.0f}) -> marker not seen")
            continue
        _, measured_x, measured_y = min(candidates)
        measured = numpy.array([measured_x, measured_y])
        error = float(numpy.linalg.norm(measured - camera_point))
        results.append((camera_point, projector_point, measured, error))
        print(f"[proof]   point {index}: camera ({camera_point[0]:.0f},{camera_point[1]:.0f}) -> "
              f"projector ({projector_point[0]:.0f},{projector_point[1]:.0f}); ring seen at "
              f"({measured[0]:.0f},{measured[1]:.0f}); error {error:.1f} px")
        if debug_window_name is not None:
            view = cv2.cvtColor(on, cv2.COLOR_GRAY2BGR)
            cv2.drawMarker(view, (int(camera_point[0]), int(camera_point[1])), (0, 255, 255), cv2.MARKER_CROSS, 30, 2)
            cv2.circle(view, (int(measured[0]), int(measured[1])), 10, (0, 0, 255), 2)
            cv2.putText(view, f"proof point {index}: error {error:.1f} px", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imshow(debug_window_name, view)
            cv2.waitKey(1)
    projector_window.show(black)
    cv2.waitKey(1)

    errors = [item[3] for item in results if item[3] is not None]
    if errors:
        print(f"[proof] {len(errors)} of {len(results)} points measured: mean error {numpy.mean(errors):.1f} px, "
              f"median {numpy.median(errors):.1f} px, max {numpy.max(errors):.1f} px")
    else:
        print("[proof] no point could be measured")
    return results


def draw_proof(frame_bgr, results, scene_map, path):
    view = frame_bgr.copy()
    for x, y in scene_map.camera_points:
        cv2.circle(view, (int(x), int(y)), 2, (90, 90, 90), -1)
    for camera_point, projector_point, measured, error in results:
        x, y = int(camera_point[0]), int(camera_point[1])
        cv2.drawMarker(view, (x, y), (0, 255, 255), cv2.MARKER_CROSS, 24, 2)
        if measured is not None:
            cv2.circle(view, (int(measured[0]), int(measured[1])), 8, (0, 0, 255), 2)
            cv2.line(view, (x, y), (int(measured[0]), int(measured[1])), (0, 0, 255), 1)
            cv2.putText(view, f"{error:.1f}px", (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        else:
            cv2.putText(view, "n/a", (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
    errors = [item[3] for item in results if item[3] is not None]
    summary = (f"proof: {len(errors)}/{len(results)} measured, mean {numpy.mean(errors):.1f} px, max {numpy.max(errors):.1f} px"
               if errors else "proof: nothing measured")
    cv2.putText(view, summary, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5)
    cv2.putText(view, summary, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    cv2.imwrite(path, view)
    return view
