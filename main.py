"""Feedback projection mapping: paint a panda onto a tracked face.

Loop per camera frame:
  camera -> MediaPipe landmarks -> smooth + predict -> panda in camera space
  -> warp with the current homography -> projector.

Servo cycle (every servo_period_seconds):
  1. Freeze the mask and note when that frozen image was shown (T0).
  2. A little later show the same frozen image plus small dots at the
     projector positions the current mapping predicts for a few face
     landmarks; note when the dots were first shown (Td).
  3. Reference frame: the last camera frame read after T0 + full latency
     (frozen mask fully visible) and before Td + no-show latency (dots not yet
     visible). Measurement frame: the first camera frame read after
     Td + full latency (dots fully visible).
  4. Unfreeze the mask as soon as no later render can reach the measurement
     frame, difference the two frames, find where the dots really landed, and
     update the mapping from those correspondences.

Keys (with either window focused):
  q / Esc  quit          b  re-run bootstrap      s  save homography
  m  toggle projection   v  toggle servo          a  toggle always-on alignment dots
  l  landmark mode: project the face landmarks as dots instead of the panda
  k  toggle the border markers (live wall-plane mapping)   c  toggle debug overlays
  r  forget face correspondences
"""

import argparse
import collections
import time

import cv2
import numpy

import config
from bootstrap import run_bootstrap
from border_markers import BlinkingBorderTracker, marker_projector_points
from camera import Camera
from displays import pick_projector_monitor
from dot_detector import detect_all_dots, match_dot_pattern
from face_tracker import FaceTracker
from homography_servo import HomographyEstimator
from panda_mask import render_panda_mask
from projector import ProjectorWindow
from proof import choose_test_points, draw_proof, run_proof
from proof_face import draw_face_proof, run_face_proof
from red_disc import find_red_disc
from face_mesh_render import fit_projection_matrix, render_face_mesh, render_wireframe, triangulate
from scene_map import SceneMap
from face_mesh_export import export_face_mesh, load_painted_texture, render_painted_texture

DEBUG_WINDOW_NAME = "debug"
PROJECTOR_VIEW_WINDOW_NAME = "projector view (3D mesh)"


def draw_dots(projector_image, points, radius, brightness, color=None):
    if color is None:
        color = (brightness, brightness, brightness)
    for x, y in points:
        cv2.circle(projector_image, (int(round(x)), int(round(y))), radius, color, -1, cv2.LINE_AA)


def align_reference_to_face(reference_gray, reference_landmarks, measurement_landmarks, expand_fraction):
    """Warp the reference frame so the face in it sits where the face is in the
    measurement frame, inside the face box only. Projected dots stay put in the
    camera when the face moves, so aligning the skin leaves only the dots."""
    if reference_landmarks is None or measurement_landmarks is None:
        return reference_gray
    similarity, _ = cv2.estimateAffinePartial2D(reference_landmarks.astype(numpy.float32),
                                                measurement_landmarks.astype(numpy.float32))
    if similarity is None:
        return reference_gray
    warped = cv2.warpAffine(reference_gray, similarity, (reference_gray.shape[1], reference_gray.shape[0]),
                            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    left, top = measurement_landmarks.min(axis=0)
    right, bottom = measurement_landmarks.max(axis=0)
    width, height = right - left, bottom - top
    left = int(max(0, left - expand_fraction * width))
    top = int(max(0, top - expand_fraction * height))
    right = int(min(reference_gray.shape[1], right + expand_fraction * width))
    bottom = int(min(reference_gray.shape[0], bottom + expand_fraction * height))
    aligned = reference_gray.copy()
    aligned[top:bottom, left:right] = warped[top:bottom, left:right]
    return aligned


def build_render_homography(homography_camera_to_projector, render_scale_factor):
    canvas_to_camera = numpy.diag([1.0 / render_scale_factor, 1.0 / render_scale_factor, 1.0])
    return homography_camera_to_projector @ canvas_to_camera


def projector_coverage_in_camera(homography_camera_to_projector, projector_width, projector_height):
    """Camera-space quad of the projector's frame: where the projector can paint at all."""
    try:
        projector_to_camera = numpy.linalg.inv(homography_camera_to_projector)
    except numpy.linalg.LinAlgError:
        return None
    corners = numpy.array([[0, 0], [projector_width, 0], [projector_width, projector_height], [0, projector_height]],
                          dtype=numpy.float64).reshape(-1, 1, 2)
    quad = cv2.perspectiveTransform(corners, projector_to_camera).reshape(-1, 2)
    if not numpy.all(numpy.isfinite(quad)) or numpy.abs(quad).max() > 1e5:
        return None
    return quad


def compose_debug_view(frame, landmarks, servo_indices, last_gating_points, last_found,
                       last_difference, search_radius, status_lines, coverage_quad=None, projected_indices=None,
                       clean=False):
    view = frame.copy()
    if clean:
        # Only the projected landmarks and the status text.
        landmarks_only = landmarks
        landmarks = None
        last_gating_points = None
        last_difference = None
        coverage_quad = None
        if landmarks_only is not None and projected_indices is not None:
            for landmark_index, (x, y) in zip(projected_indices, landmarks_only[projected_indices]):
                x, y = int(x), int(y)
                cv2.circle(view, (x, y), 12, (255, 0, 255), 2)
                cv2.line(view, (x - 18, y), (x + 18, y), (255, 0, 255), 2)
                cv2.line(view, (x, y - 18), (x, y + 18), (255, 0, 255), 2)
        projected_indices = None
    if landmarks is not None and projected_indices is not None:
        # The landmarks being projected: bold magenta crosshairs, labelled when few.
        for landmark_index, (x, y) in zip(projected_indices, landmarks[projected_indices]):
            x, y = int(x), int(y)
            cv2.circle(view, (x, y), 12, (255, 0, 255), 2)
            cv2.line(view, (x - 18, y), (x + 18, y), (255, 0, 255), 2)
            cv2.line(view, (x, y - 18), (x, y + 18), (255, 0, 255), 2)
            if len(projected_indices) <= 8:
                cv2.putText(view, f"landmark {landmark_index}", (x + 16, y - 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    if coverage_quad is not None:
        cv2.polylines(view, [coverage_quad.astype(numpy.int32).reshape(-1, 1, 2)], True, (255, 220, 0), 2)
        cv2.putText(view, "projector coverage", (int(coverage_quad[0][0]) + 6, int(coverage_quad[0][1]) + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 220, 0), 2)
    if landmarks is not None:
        for x, y in landmarks[servo_indices]:
            cv2.circle(view, (int(x), int(y)), 4, (0, 255, 0), 1)
    if last_gating_points is not None:
        for x, y in last_gating_points:
            cv2.circle(view, (int(x), int(y)), int(search_radius), (0, 200, 255), 1)
        for expected_index, (found_x, found_y) in last_found:
            expected_x, expected_y = last_gating_points[expected_index]
            cv2.circle(view, (int(found_x), int(found_y)), 5, (0, 0, 255), 2)
            cv2.line(view, (int(expected_x), int(expected_y)), (int(found_x), int(found_y)), (0, 0, 255), 1)
    if last_difference is not None:
        inset_width = 320
        inset_height = int(last_difference.shape[0] * inset_width / last_difference.shape[1])
        inset = cv2.applyColorMap(cv2.resize(last_difference, (inset_width, inset_height)), cv2.COLORMAP_INFERNO)
        view[0:inset_height, view.shape[1] - inset_width:] = inset
    for line_index, line in enumerate(status_lines):
        cv2.putText(view, line, (10, 28 + 26 * line_index), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(view, line, (10, 28 + 26 * line_index), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return view


class ServoCycle:
    """State of one in-flight measurement cycle."""

    def __init__(self, frozen_mask_image, sent_projector_points):
        self.frozen_mask_image = frozen_mask_image
        self.sent_projector_points = sent_projector_points
        self.frozen_shown_at = None      # T0: when the frozen image was first shown
        self.dots_shown_at = None        # Td: when the dots were first shown
        self.reference_gray = None
        self.reference_landmarks = None
        self.reference_timestamp = None


def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--seconds", type=float, default=None,
                                 help="quit automatically after this many seconds (for unattended tests)")
    argument_parser.add_argument("--alignment-dots", action="store_true",
                                 help="start with permanent dots on the servo landmarks")
    argument_parser.add_argument("--no-servo", action="store_true", help="start with the servo off")
    argument_parser.add_argument("--landmarks", action="store_true",
                                 help="project the face landmarks as dots instead of the panda (alignment debug)")
    argument_parser.add_argument("--dot-indices", type=str, default=None,
                                 help="comma-separated landmark indices to project in landmark mode, e.g. 4 for the nose tip")
    argument_parser.add_argument("--dot-radius", type=int, default=config.landmark_dot_radius_pixels,
                                 help="radius in projector pixels of the projected landmark dots")
    argument_parser.add_argument("--dot-color", type=str, default="white",
                                 help="white, red, green or blue for the projected landmark dots")
    argument_parser.add_argument("--scan", action="store_true",
                                 help="dense coded scan of the whole scene at start (key g rescans); the pointer uses it")
    argument_parser.add_argument("--prove", action="store_true",
                                 help="after the scan, project rings at held-out points and measure the error")
    argument_parser.add_argument("--face-mesh", action="store_true",
                                 help="fit the projector as a camera (P) from bootstrap face dots and project the live "
                                      "camera texture of the face through MediaPipe's 3D mesh")
    argument_parser.add_argument("--export-face", action="store_true",
                                 help="export the face mesh as a UV-unwrapped OBJ + baked texture once a face is tracked (key e)")
    argument_parser.add_argument("--paint", type=str, default=None,
                                 help="PNG painted on export/face_texture.png's UV layout; projected through the live mesh (needs P)")
    argument_parser.add_argument("--prove-face", action="store_true",
                                 help="after the bootstrap, project markers at face landmarks with and without the "
                                      "depth term and measure the error per landmark (geometry only, no panda)")
    argument_parser.add_argument("--pointer", action="store_true",
                                 help="geometry test: project a ring at the camera point under the mouse (debug window)")
    argument_parser.add_argument("--clean", action="store_true",
                                 help="hide the tracking overlays in the debug window (key c toggles)")
    arguments = argument_parser.parse_args()
    dot_colors = {"white": (255, 255, 255), "red": (0, 0, 255), "green": (0, 255, 0), "blue": (255, 0, 0)}
    landmark_dot_color = dot_colors.get(arguments.dot_color, (255, 255, 255))
    clean_view = arguments.clean
    landmark_projection_indices = config.landmark_projection_indices
    if arguments.dot_indices:
        landmark_projection_indices = [int(value) for value in arguments.dot_indices.split(",")]
        arguments.landmarks = True

    monitor, is_second_screen = pick_projector_monitor(
        config.projector_monitor_index,
        config.projector_fallback_width_pixels,
        config.projector_fallback_height_pixels,
    )
    camera = Camera(config)
    projector = ProjectorWindow(monitor, fullscreen=is_second_screen)
    cv2.namedWindow(DEBUG_WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(DEBUG_WINDOW_NAME, 960, 540)

    pointer = {"position": None, "wall_homography": None, "scene_map": None, "frames": {}, "measured": None,
               "offset": None, "last_measure": 0.0, "correction": numpy.zeros(2), "correction_anchor": None}

    def on_mouse(event, x, y, flags, user_data):
        if event == cv2.EVENT_MOUSEMOVE or event == cv2.EVENT_LBUTTONDOWN:
            pointer["position"] = (float(x), float(y))

    cv2.setMouseCallback(DEBUG_WINDOW_NAME, on_mouse)
    if arguments.face_mesh:
        cv2.resizeWindow(DEBUG_WINDOW_NAME, 1600, 450)     # camera view | projector 3D view, side by side

    face_tracker = FaceTracker(config, camera.frame_width, camera.frame_height)
    estimator = HomographyEstimator(config, camera.frame_width, camera.frame_height,
                                    projector.width, projector.height)

    marker_points = marker_projector_points(projector.width, projector.height, config.border_marker_inset_pixels)
    border_tracker = BlinkingBorderTracker(config, marker_points,
                                           config.latency_none_before_seconds, config.latency_full_after_seconds)
    border_markers_enabled = config.border_markers_enabled

    def overlay_markers(image):
        if border_markers_enabled:
            border_tracker.draw(image, time.perf_counter())
        return image

    mesh = {"projection": None, "face_dot_pairs": [], "saved": False}

    def fit_face_projection():
        """Projector as a camera: P from the bootstrap face dots (camera x, y, MediaPipe z -> projector u, v)."""
        pairs = mesh["face_dot_pairs"]
        if len(pairs) < 6:
            print(f"[face-mesh] only {len(pairs)} face dots with depth; at least 6 needed to fit P")
            mesh["projection"] = None
            return
        points_3d = numpy.array([[pair[0][0], pair[0][1], pair[2]] for pair in pairs])
        projector_points = numpy.array([pair[1] for pair in pairs])
        projection, inliers, rms = fit_projection_matrix(points_3d, projector_points)
        if projection is None:
            print("[face-mesh] P fit failed")
            mesh["projection"] = None
            return
        mesh["projection"] = projection
        mesh["saved"] = False
        z_values = points_3d[:, 2]
        print(f"[face-mesh] P fitted from {int(inliers.sum())}/{len(pairs)} face dots, rms {rms:.1f} px, "
              f"z range {z_values.min():.0f}..{z_values.max():.0f}")

    def bootstrap_estimator():
        for attempt in range(config.bootstrap_attempts):
            pairs, wall_pairs = run_bootstrap(camera, projector, face_tracker, config, DEBUG_WINDOW_NAME)
            landmarks = face_tracker.last_raw_landmarks
            bootstrap_face_size = face_tracker.face_size(landmarks) if landmarks is not None else 0.0
            wall_homography = None
            if len(wall_pairs) >= 8:
                wall_homography, wall_inliers = cv2.findHomography(
                    numpy.array([pair[0] for pair in wall_pairs]), numpy.array([pair[1] for pair in wall_pairs]),
                    cv2.RANSAC, config.servo_ransac_reprojection_threshold_pixels)
                if wall_homography is not None and estimator.is_sane(wall_homography):
                    border_tracker.homography = wall_homography
                    pointer["wall_homography"] = wall_homography
                    print(f"[main] wall-plane mapping from {int(wall_inliers.sum())} wall dots (until the markers take over)")
                else:
                    wall_homography = None
            if estimator.set_bootstrap_pairs(pairs, bootstrap_face_size, wall_homography):
                mesh["face_dot_pairs"] = [pair for pair in pairs if len(pair) > 2 and pair[2] is not None]
                fit_face_projection()
                return
            print(f"[main] bootstrap attempt {attempt + 1} failed; "
                  + ("retrying, hold still" if attempt + 1 < config.bootstrap_attempts else "giving up (press b to retry)"))

    def dense_scan_and_proof():
        pairs, _ = run_bootstrap(camera, projector, face_tracker, config, DEBUG_WINDOW_NAME,
                                 grid_columns=config.scan_grid_columns, grid_rows=config.scan_grid_rows,
                                 dot_radius=config.scan_dot_radius_pixels,
                                 opening_radius=config.scan_opening_radius_pixels,
                                 sample_radius=config.scan_code_sample_radius_pixels, return_all=True)
        if len(pairs) < 10:
            print("[scan] too few dots decoded; no scene map")
            return
        pointer["scene_map"] = SceneMap(pairs, config.scan_map_neighbours, config.scan_map_max_neighbour_distance_pixels,
                                        config.scan_map_outlier_pixels)
        print(f"[scan] scene map with {len(pointer['scene_map'])} measured correspondences "
              f"({pointer['scene_map'].dropped_outliers} dropped as inconsistent with their neighbours)")
        if arguments.prove:
            rng = numpy.random.default_rng(int(time.time()))
            test_points = choose_test_points(pointer["scene_map"], camera.frame_width, camera.frame_height,
                                             config.proof_point_count, config.proof_min_dot_distance_pixels,
                                             config.scan_map_max_neighbour_distance_pixels * 0.6, rng)
            results = run_proof(camera, projector, pointer["scene_map"], config, test_points, DEBUG_WINDOW_NAME)
            _, latest_frame = camera.latest()
            draw_proof(latest_frame, results, pointer["scene_map"], f"{config.debug_image_directory}/proof.png")
            print(f"[proof] wrote {config.debug_image_directory}/proof.png")

    if not estimator.load(config.homography_file_path):
        bootstrap_estimator()
    if arguments.scan:
        dense_scan_and_proof()

    def face_proof():
        if not estimator.is_valid():
            print("[proof-face] no mapping")
            return
        indices = config.face_proof_landmark_indices
        without = run_face_proof(camera, projector, face_tracker, estimator, config, indices, False, DEBUG_WINDOW_NAME)
        with_depth = run_face_proof(camera, projector, face_tracker, estimator, config, indices, True, DEBUG_WINDOW_NAME)
        _, latest_frame = camera.latest()
        draw_face_proof(latest_frame, with_depth, without, f"{config.debug_image_directory}/proof_face.png")
        print(f"[proof-face] wrote {config.debug_image_directory}/proof_face.png")

    if arguments.prove_face:
        face_proof()
        if not estimator.is_valid():
            print("[main] bootstrap did not yield a usable mapping; point the projector at a "
                  "surface the camera can see and press 'b' to retry")

    servo_enabled = config.servo_enabled_at_start and not arguments.no_servo
    mask_enabled = not (arguments.prove_face or arguments.face_mesh or arguments.paint)   # geometry/paint runs: no panda
    painted_texture = load_painted_texture(arguments.paint) if arguments.paint else None
    export_pending = arguments.export_face

    def export_now():
        if face_tracker.last_raw_landmarks_3d is None:
            print("[export] no face tracked")
            return
        _, latest_frame = camera.latest()
        landmarks_3d = numpy.column_stack([smoothed_landmarks, face_tracker.last_smoothed_z])             if smoothed_landmarks is not None else face_tracker.last_raw_landmarks_3d
        obj_path = export_face_mesh(config.export_directory, landmarks_3d, latest_frame, landmarks_3d[:, :2], mesh["projection"])
        print(f"[export] wrote {obj_path} (+ face_mesh.mtl, face_texture.png, face_mesh_state.npz)")
    landmark_mode = arguments.landmarks
    alignment_dots_always_on = arguments.alignment_dots
    servo_indices = config.servo_landmark_indices

    none_before = config.latency_none_before_seconds
    full_after = config.latency_full_after_seconds
    margin = config.servo_timing_margin_seconds
    # Dots go on late enough that a reference frame (frozen mask fully visible,
    # dots not yet) is guaranteed to exist: a window of at least 2 camera frames.
    dots_delay_seconds = (full_after - none_before) + margin + 0.07
    # The measurement frame can contain renders up to (full - none) after Td.
    unfreeze_after_dots_seconds = (full_after - none_before) + margin

    cycle = None
    last_cycle_end = time.perf_counter()
    cycle_count = 0
    failed_cycles_in_a_row = 0
    landmark_offsets = {}
    last_live_update = -1e9
    sent_disc_history = collections.deque(maxlen=60)   # (time shown, projector point) of the red disc
    last_seen_disc = None
    last_passive_update = 0.0
    passive_warning = ""
    last_gating_points = None
    last_found = []
    last_difference = None
    last_search_radius = config.servo_search_radius_initial_pixels
    last_mean_offset = float("nan")

    frame_count = 0
    fps_window_start = time.perf_counter()
    frames_per_second = 0.0
    run_start = time.perf_counter()
    last_status_print = run_start
    last_processed_timestamp = 0.0

    render_scale = config.render_scale_factor
    canvas_width = int(camera.frame_width * render_scale)
    canvas_height = int(camera.frame_height * render_scale)
    # Frames this loop has processed, with their landmarks: (timestamp, gray, raw_landmarks).
    processed_frames = collections.deque(maxlen=config.camera_frame_history_length)
    black_projector_image = numpy.zeros((projector.height, projector.width, 3), dtype=numpy.uint8)

    try:
        while True:
            result = camera.latest_after(last_processed_timestamp, timeout_seconds=1.0)
            if result is None:
                cv2.waitKey(1)
                continue
            frame_timestamp, frame = result
            last_processed_timestamp = frame_timestamp
            now = time.perf_counter()
            smoothed_landmarks, predicted_landmarks = face_tracker.process(frame, frame_timestamp)
            face_available = smoothed_landmarks is not None
            face_size = face_tracker.face_size(smoothed_landmarks) if face_available else 0.0
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            raw_landmarks = face_tracker.last_raw_landmarks
            processed_frames.append((frame_timestamp, gray, raw_landmarks))

            # ------------------------------------------------ live wall-plane mapping from the border markers
            live_base_note = "beam markers: off"
            if border_markers_enabled:
                live_homography = border_tracker.observe(gray, frame_timestamp)
                if live_homography is not None and estimator.is_sane(live_homography):
                    estimator.set_base(live_homography)
                    last_live_update = now
                cycle_seconds = border_tracker.slot_seconds * border_tracker.slot_count
                stale = now - last_live_update > 3 * cycle_seconds
                live_base_note = f"beam markers: {border_tracker.found_count}/{len(marker_points)}" \
                    + (" (LOST - mapping frozen)" if stale else f" (H refreshed {now - last_live_update:.0f} s ago)")

            if export_pending and face_available and frame_count > 30:
                export_pending = False
                export_now()

            # ------------------------------------------------ render panda
            projector_image = numpy.zeros((projector.height, projector.width, 3), dtype=numpy.uint8)
            outside_beam_note = ""
            pointer_note = ""
            if arguments.pointer:
                # Geometry test: a blinking ring at the mapping of the camera point under the mouse,
                # measured back in the camera by differencing the on and off phases.
                pointer_homography = pointer["wall_homography"]
                if pointer_homography is None and estimator.is_valid():
                    pointer_homography = estimator.base_homography
                scene_map = pointer["scene_map"]
                if pointer["position"] is not None and (scene_map is not None or pointer_homography is not None):
                    mouse = numpy.array(pointer["position"], dtype=numpy.float64)

                    def open_loop_map(camera_point):
                        mapped = scene_map.query(camera_point) if scene_map is not None else None
                        if mapped is None and pointer_homography is not None:
                            mapped = cv2.perspectiveTransform(
                                numpy.asarray(camera_point, dtype=numpy.float64).reshape(1, 1, 2), pointer_homography).reshape(2)
                        return mapped

                    target = open_loop_map(mouse)
                    mapping_used = "scan map" if (scene_map is not None and scene_map.query(mouse) is not None) else "homography"
                    if target is None:
                        target = numpy.array([-1000.0, -1000.0])
                    # Closed loop: the measured landing error, converted to projector pixels through the
                    # local Jacobian of the mapping, corrects the aim. Converges in a few blinks on any surface.
                    if pointer["correction_anchor"] is not None and numpy.linalg.norm(mouse - pointer["correction_anchor"]) > config.pointer_correction_reset_pixels:
                        pointer["correction"] = numpy.zeros(2)
                        pointer["offset"] = None
                    pointer["correction_anchor"] = mouse.copy()
                    target = target + pointer["correction"]
                    mapping_used += f" + feedback ({pointer['correction'][0]:+.0f},{pointer['correction'][1]:+.0f})"
                    ring_on = int(now // config.pointer_blink_seconds) % 2 == 0
                    if ring_on:
                        center = (int(round(target[0])), int(round(target[1])))
                        cv2.circle(projector_image, center, config.pointer_ring_radius_pixels, (255, 255, 255),
                                   config.pointer_ring_thickness_pixels, cv2.LINE_AA)
                        cv2.circle(projector_image, center, 6, (255, 255, 255), -1, cv2.LINE_AA)
                    # Attribute this camera frame to a blink phase (allowing for latency).
                    phase_a = int((frame_timestamp - full_after) // config.pointer_blink_seconds)
                    phase_b = int((frame_timestamp - none_before) // config.pointer_blink_seconds)
                    if phase_a == phase_b:
                        pointer["frames"][phase_a % 2] = (frame_timestamp, gray)
                    frames = pointer["frames"]
                    if (0 in frames and 1 in frames and abs(frames[0][0] - frames[1][0]) < 2 * config.pointer_blink_seconds
                            and now - pointer["last_measure"] >= config.pointer_blink_seconds):
                        pointer["last_measure"] = now
                        difference = cv2.GaussianBlur(cv2.subtract(frames[0][1], frames[1][1]), (5, 5), 0)
                        radius = config.pointer_search_radius_pixels
                        left, top = int(max(0, mouse[0] - radius)), int(max(0, mouse[1] - radius))
                        right, bottom = int(min(camera.frame_width, mouse[0] + radius)), int(min(camera.frame_height, mouse[1] + radius))
                        window = difference[top:bottom, left:right].astype(numpy.float64)
                        window[window < config.pointer_min_difference] = 0.0
                        weight_sum = window.sum()
                        if weight_sum > 0:
                            rows, columns = numpy.mgrid[top:bottom, left:right]
                            measured = (float((columns * window).sum() / weight_sum), float((rows * window).sum() / weight_sum))
                            pointer["measured"] = measured
                            pointer["offset"] = (measured[0] - mouse[0], measured[1] - mouse[1])
                            # Jacobian of camera -> projector at the mouse, by finite differences.
                            base = open_loop_map(mouse)
                            step_x = open_loop_map(mouse + numpy.array([8.0, 0.0]))
                            step_y = open_loop_map(mouse + numpy.array([0.0, 8.0]))
                            if base is not None and step_x is not None and step_y is not None:
                                jacobian = numpy.column_stack([(step_x - base) / 8.0, (step_y - base) / 8.0])
                                error_camera = numpy.array(pointer["offset"])
                                if numpy.linalg.norm(error_camera) < config.pointer_correction_max_error_pixels:
                                    pointer["correction"] = pointer["correction"] - config.pointer_correction_gain * (jacobian @ error_camera)
                            print(f"[pointer] mouse ({mouse[0]:.0f},{mouse[1]:.0f}) ring seen at ({measured[0]:.0f},{measured[1]:.0f}) "
                                  f"error {numpy.hypot(*pointer['offset']):.1f} px; correction now "
                                  f"({pointer['correction'][0]:+.0f},{pointer['correction'][1]:+.0f}) projector px")
                        else:
                            pointer["measured"] = None
                            pointer["offset"] = None
                    pointer_note = (f"pointer [{mapping_used}]: camera ({mouse[0]:.0f},{mouse[1]:.0f}) -> "
                                    f"projector ({target[0]:.0f},{target[1]:.0f})")
                    if pointer["offset"] is not None:
                        pointer_note += (f"; ring seen at ({pointer['measured'][0]:.0f},{pointer['measured'][1]:.0f}), "
                                         f"offset ({pointer['offset'][0]:+.0f},{pointer['offset'][1]:+.0f}) px = "
                                         f"{numpy.hypot(*pointer['offset']):.0f} px")
                    else:
                        pointer_note += "; ring not seen"
                else:
                    pointer_note = "pointer: move the mouse over this window" \
                        if (pointer_homography is not None or scene_map is not None) else "pointer: no mapping yet"
            if arguments.pointer:
                pass    # the ring is the only projection in the geometry test
            elif arguments.face_mesh:
                if mesh["projection"] is not None and face_available:
                    # Geometry: predicted x, y with smoothed z. Texture: the current frame at the smoothed landmarks.
                    landmarks_3d = numpy.column_stack([predicted_landmarks, face_tracker.last_smoothed_z])
                    if mesh.get("triangles") is None or frame_count % config.face_mesh_retriangulate_every_frames == 0:
                        mesh["triangles"] = triangulate(smoothed_landmarks, camera.frame_width, camera.frame_height)
                    triangles = mesh["triangles"]
                    # Shown in its own debug window, not projected (the projector stays dark in this mode).
                    projector_view = render_face_mesh(frame, smoothed_landmarks, landmarks_3d, mesh["projection"],
                                                      triangles, projector.width, projector.height,
                                                      config.face_mesh_texture_gain)
                    wireframe = render_wireframe(landmarks_3d, mesh["projection"], triangles, projector.width, projector.height)
                    projector_view_with_mesh = cv2.addWeighted(projector_view, 1.0, wireframe, 0.35, 0)
                    mesh["latest_view"] = projector_view_with_mesh
                    if not mesh["saved"]:
                        mesh["saved"] = True
                        cv2.imwrite(f"{config.debug_image_directory}/face_mesh_projector_view.png", projector_view_with_mesh)
                        cv2.imwrite(f"{config.debug_image_directory}/face_mesh_wireframe.png", wireframe)
                        print("[face-mesh] saved face_mesh_projector_view.png and face_mesh_wireframe.png")
            elif painted_texture is not None:
                if mesh["projection"] is not None and face_available:
                    landmarks_3d = numpy.column_stack([predicted_landmarks, face_tracker.last_smoothed_z])
                    projector_image = render_painted_texture(painted_texture, landmarks_3d, mesh["projection"],
                                                             projector.width, projector.height)
            elif mask_enabled and predicted_landmarks is not None and estimator.is_valid():
                if landmark_mode:
                    # Alignment debug: the landmarks themselves, mapped straight into projector space.
                    projected_points = estimator.map_points(predicted_landmarks[landmark_projection_indices], face_size,
                                                           face_tracker.last_smoothed_z[landmark_projection_indices])
                    draw_dots(projector_image, projected_points,
                              arguments.dot_radius, config.landmark_dot_brightness, landmark_dot_color)
                    outside = [landmark_index for landmark_index, (x, y) in zip(landmark_projection_indices, projected_points)
                               if not (0 <= x < projector.width and 0 <= y < projector.height)]
                    outside_beam_note = f"OUTSIDE BEAM: landmarks {outside}" if outside else ""
                    if len(projected_points) == 1 and landmark_dot_color == (0, 0, 255):
                        # Remember what was sent, so the disc seen ~latency later can be paired with it.
                        sent_disc_history.append((now, projected_points[0].copy()))
                else:
                    # Every landmark mapped with its own depth, then the panda drawn in projector space.
                    projector_landmarks = estimator.map_points(predicted_landmarks, face_size, face_tracker.last_smoothed_z)
                    projector_image = render_panda_mask(projector.width, projector.height, projector_landmarks,
                                                        config.face_fill_brightness, config.mouth_brightness)

            # ------------------------------------------------ passive check on the red disc
            passive_note = ""
            if (config.passive_disc_tracking_enabled and landmark_mode and face_available and estimator.is_valid()
                    and len(landmark_projection_indices) == 1 and landmark_dot_color == (0, 0, 255)):
                aimed_landmark = smoothed_landmarks[landmark_projection_indices[0]]
                seen_disc = find_red_disc(frame, aimed_landmark, config.passive_disc_search_radius_pixels,
                                          config.passive_disc_min_area_pixels)
                last_seen_disc = seen_disc
                velocity = face_tracker.last_velocity
                speed = float(numpy.linalg.norm(velocity[landmark_projection_indices[0]])) if velocity is not None else 0.0
                if seen_disc is not None:
                    disc_offset = numpy.hypot(seen_disc[0] - aimed_landmark[0], seen_disc[1] - aimed_landmark[1])
                    passive_note = f"red disc {disc_offset:.0f} px from landmark"
                    # Pair the disc with what was sent one latency ago; only while the head is still.
                    sent_then = None
                    for sent_time, sent_point in sent_disc_history:
                        if sent_time <= frame_timestamp - full_after:
                            sent_then = sent_point
                    if (sent_then is not None and speed < config.passive_disc_max_speed_pixels_per_second
                            and now - last_passive_update >= config.passive_disc_update_period_seconds):
                        last_passive_update = now
                        if config.passive_disc_correction_enabled:
                            estimator.add_face_pairs([((seen_disc[0], seen_disc[1]), tuple(sent_then))], face_size)
                            passive_note += " (correcting)"
                        if disc_offset > config.passive_disc_warning_pixels:
                            passive_warning = f"PROJECTION OFF TARGET by {disc_offset:.0f} px - projector or camera moved? (b = re-bootstrap)"
                        else:
                            passive_warning = ""
                else:
                    passive_note = "red disc not seen in camera"

            if alignment_dots_always_on and face_available and estimator.is_valid():
                draw_dots(projector_image, estimator.map_points(smoothed_landmarks[servo_indices], face_size,
                                                                face_tracker.last_smoothed_z[servo_indices]),
                          config.servo_dot_radius_pixels, config.servo_dot_brightness)

            # ------------------------------------------------ servo cycle
            servo_active = servo_enabled and estimator.is_valid() and face_available
            if not servo_active:
                cycle = None
            elif cycle is None:
                converged = (estimator.face_pair_count() >= config.servo_face_points_for_tight_gate
                             and last_mean_offset < config.servo_converged_offset_pixels)
                period = config.servo_period_converged_seconds if converged else config.servo_period_seconds
                if now - last_cycle_end >= period:
                    frozen_image = black_projector_image if config.servo_hide_mask_during_cycle else projector_image.copy()
                    cycle = ServoCycle(frozen_image, None)

            drawing_dots_this_frame = False
            if cycle is not None:
                if cycle.dots_shown_at is None:
                    drawing_dots_this_frame = (cycle.frozen_shown_at is not None
                                               and now >= cycle.frozen_shown_at + dots_delay_seconds)
                    mask_frozen = True
                else:
                    drawing_dots_this_frame = True
                    mask_frozen = now < cycle.dots_shown_at + unfreeze_after_dots_seconds

                if mask_frozen:
                    projector_image = cycle.frozen_mask_image.copy()
                if drawing_dots_this_frame:
                    if cycle.sent_projector_points is None:
                        # Aim at the predicted positions at the moment the dots go up, exactly
                        # as the mask is aimed: then any trailing along the motion direction is
                        # uncorrected latency, which the lead auto-tune removes.
                        cycle.sent_projector_points = estimator.map_points(
                            predicted_landmarks[servo_indices], face_size, face_tracker.last_smoothed_z[servo_indices])
                    draw_dots(projector_image, cycle.sent_projector_points,
                              config.servo_dot_radius_pixels, config.servo_dot_brightness)

                if cycle.dots_shown_at is not None:
                    if cycle.reference_gray is None and now >= cycle.dots_shown_at + none_before:
                        candidates = [entry for entry in processed_frames
                                      if cycle.frozen_shown_at + full_after <= entry[0] <= cycle.dots_shown_at + none_before]
                        if candidates:
                            cycle.reference_timestamp, cycle.reference_gray, cycle.reference_landmarks = candidates[-1]
                        else:
                            window_start = cycle.frozen_shown_at + full_after
                            window_end = cycle.dots_shown_at + none_before
                            recent = [f"{1000 * (entry[0] - cycle.frozen_shown_at):.0f}" for entry in processed_frames
                                      if entry[0] >= cycle.frozen_shown_at - 0.1]
                            print(f"[servo] no clean reference frame: window {1000 * (window_start - cycle.frozen_shown_at):.0f}"
                                  f"..{1000 * (window_end - cycle.frozen_shown_at):.0f} ms after freeze, "
                                  f"frames at {' '.join(recent)} ms; skipping cycle")
                            cycle = None
                            last_cycle_end = now

                    if cycle is not None and cycle.reference_gray is not None:
                        measurement = None
                        if frame_timestamp >= cycle.dots_shown_at + full_after:
                            measurement = (frame_timestamp, gray, raw_landmarks)
                        if measurement is None and now > cycle.dots_shown_at + full_after + 0.5:
                            print("[servo] no measurement frame arrived; skipping cycle")
                            cycle = None
                            last_cycle_end = now
                        elif measurement is not None:
                            measurement_gray = measurement[1]
                            aligned_reference = align_reference_to_face(
                                cycle.reference_gray, cycle.reference_landmarks, measurement[2],
                                config.servo_face_box_expand_fraction)
                            gating_points = smoothed_landmarks[servo_indices]
                            mapping_is_rough = estimator.face_pair_count() < config.servo_face_points_for_tight_gate
                            last_search_radius = config.servo_search_radius_initial_pixels if mapping_is_rough \
                                else config.servo_search_radius_pixels
                            # Look for every dot-like blob around the aimed points, then
                            # match the whole aimed pattern to them under one translation.
                            region_left = int(max(0, gating_points[:, 0].min() - last_search_radius))
                            region_top = int(max(0, gating_points[:, 1].min() - last_search_radius))
                            region_right = int(min(camera.frame_width, gating_points[:, 0].max() + last_search_radius))
                            region_bottom = int(min(camera.frame_height, gating_points[:, 1].max() + last_search_radius))
                            region_blobs, region_difference = detect_all_dots(
                                aligned_reference[region_top:region_bottom, region_left:region_right],
                                measurement_gray[region_top:region_bottom, region_left:region_right],
                                config.servo_min_peak_difference,
                                config.servo_max_blob_area_pixels,
                                config.servo_opening_radius_pixels,
                                config.dot_max_negative_ratio,
                            )
                            blobs = [(x + region_left, y + region_top, peak) for x, y, peak in region_blobs]
                            difference = numpy.zeros_like(measurement_gray)
                            difference[region_top:region_bottom, region_left:region_right] = region_difference
                            raw_offsets = f"{len(blobs)} blobs"
                            found, pattern_offset = match_dot_pattern(
                                blobs, gating_points,
                                config.servo_pattern_tolerance_pixels,
                                config.servo_min_consistent_dots,
                                config.servo_max_plausible_offset_pixels,
                                config.servo_pattern_scale_range,
                                config.servo_pattern_max_rotation_degrees)
                            if not found and not mapping_is_rough:
                                # Converged mapping: two dots under a small pure translation are
                                # unambiguous enough (the head turned so the others are hidden).
                                found, pattern_offset = match_dot_pattern(
                                    blobs, gating_points,
                                    config.servo_converged_two_dot_tolerance_pixels, 2,
                                    config.servo_converged_two_dot_max_offset_pixels,
                                    (0.97, 1.03), 3.0)
                            rejected_count = len(blobs) - len(found)
                            pairs = [(camera_point, tuple(cycle.sent_projector_points[expected_index]))
                                     for expected_index, camera_point in found]
                            if pairs:
                                depths = [float(face_tracker.last_smoothed_z[servo_indices[expected_index]])
                                          for expected_index, _ in found]
                                estimator.add_face_pairs(pairs, face_size, depths)
                            for expected_index, camera_point in found:
                                landmark_offsets.setdefault(servo_indices[expected_index], []).append(
                                    float(numpy.hypot(camera_point[0] - gating_points[expected_index][0],
                                                      camera_point[1] - gating_points[expected_index][1])))
                            if cycle_count % 5 == 4:
                                table = "  ".join(f"{index}:{numpy.mean(values):.0f}px(n{len(values)})"
                                                  for index, values in sorted(landmark_offsets.items()))
                                print(f"[proof-face] depth model {'ON' if estimator.depth_model_enabled else 'OFF'}; "
                                      f"mean landing offset per landmark: {table}")
                            offsets = [numpy.hypot(camera_point[0] - gating_points[expected_index][0],
                                                   camera_point[1] - gating_points[expected_index][1])
                                       for expected_index, camera_point in found]
                            last_mean_offset = float(numpy.mean(offsets)) if offsets else float("nan")

                            lag_note = ""
                            velocity = face_tracker.last_velocity
                            if found and velocity is not None:
                                mean_velocity = velocity[servo_indices].mean(axis=0)
                                speed = float(numpy.linalg.norm(mean_velocity))
                                if speed >= config.lead_autotune_min_speed_pixels_per_second \
                                        and len(found) >= config.lead_autotune_min_dots:
                                    offset_vectors = numpy.array([[camera_point[0] - gating_points[expected_index][0],
                                                                   camera_point[1] - gating_points[expected_index][1]]
                                                                  for expected_index, camera_point in found])
                                    trailing = -float(offset_vectors.mean(axis=0) @ (mean_velocity / speed))
                                    lag_error_seconds = trailing / speed
                                    lag_note = f", speed {speed:.0f} px/s, lag error {1000 * lag_error_seconds:+.0f} ms"
                                    if config.lead_autotune_enabled:
                                        face_tracker.prediction_lead_seconds = float(numpy.clip(
                                            face_tracker.prediction_lead_seconds + config.lead_autotune_gain * lag_error_seconds,
                                            0.0, config.lead_autotune_max_seconds))
                                        lag_note += f" -> lead {1000 * face_tracker.prediction_lead_seconds:.0f} ms"
                            last_gating_points = gating_points
                            last_found = found
                            last_difference = difference
                            cycle_count += 1
                            if config.bootstrap_save_debug_images and cycle_count <= 30:
                                prefix = f"{config.debug_image_directory}/cycle_{cycle_count:02d}"
                                annotated = cv2.applyColorMap(difference, cv2.COLORMAP_INFERNO)
                                cv2.rectangle(annotated, (region_left, region_top), (region_right, region_bottom),
                                              (0, 120, 160), 1)
                                for expected_index, (x, y) in enumerate(gating_points):
                                    cv2.circle(annotated, (int(x), int(y)), 6, (0, 200, 255), 1)
                                    cv2.putText(annotated, str(expected_index), (int(x) + 8, int(y) + 4),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)
                                for x, y, peak in blobs:
                                    cv2.circle(annotated, (int(x), int(y)), 10, (140, 140, 140), 1)
                                for expected_index, (found_x, found_y) in found:
                                    cv2.circle(annotated, (int(found_x), int(found_y)), 10, (0, 255, 0), 2)
                                    aim_x, aim_y = gating_points[expected_index]
                                    cv2.line(annotated, (int(aim_x), int(aim_y)), (int(found_x), int(found_y)), (0, 255, 0), 1)
                                # Where the current mapping would put the aim points (sent dots) in the camera: none;
                                # but show where the dots were sent, mapped back through the mapping for reference.
                                cv2.putText(annotated, f"cycle {cycle_count}: {len(found)} matched of {len(blobs)} blobs",
                                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                                cv2.imwrite(prefix + "_positive.png", annotated)
                            print(f"[servo] cycle {cycle_count}: {len(found)}/{len(servo_indices)} dots seen, "
                                  f"mean landing offset {last_mean_offset:.1f} px, model {estimator.model_name}, "
                                  f"residual {estimator.residual_rms_pixels:.1f} px, "
                                  f"face pairs {estimator.face_pair_count()}, face size {face_size:.0f}, "
                                  f"depth slope ({estimator.translation_slope[0]:.1f},{estimator.translation_slope[1]:.1f}), "
                                  f"ref->meas {1000 * (measurement[0] - cycle.reference_timestamp):.0f} ms"
                                  + f", {raw_offsets}, {rejected_count} unmatched{lag_note}")
                            cycle = None
                            last_cycle_end = now
                            drawing_dots_this_frame = False
                            failed_cycles_in_a_row = 0 if found else failed_cycles_in_a_row + 1
                            mapping_never_converged = estimator.face_pair_count() < config.servo_face_points_for_tight_gate
                            reacquire = (failed_cycles_in_a_row >= config.servo_reacquire_after_failed_cycles
                                         and mapping_never_converged) \
                                or failed_cycles_in_a_row >= config.servo_reacquire_after_failed_cycles_converged
                            if reacquire:
                                print("[servo] lost the dots; re-running the coded bootstrap")
                                bootstrap_estimator()
                                failed_cycles_in_a_row = 0
                                last_cycle_end = time.perf_counter()

            overlay_markers(projector_image)
            projector.show(projector_image)

            # ------------------------------------------------ debug view
            frame_count += 1
            if now - fps_window_start >= 1.0:
                frames_per_second = frame_count / (now - fps_window_start)
                frame_count = 0
                fps_window_start = now
            phase = "idle" if cycle is None else ("dots" if cycle.dots_shown_at is not None else "frozen")
            status_lines = [
                f"{frames_per_second:4.1f} fps   face: {'yes' if face_available else 'no'}   "
                f"{'landmarks' if landmark_mode else 'panda'}: {'on' if mask_enabled else 'off'}   "
                f"servo: {'on' if servo_enabled else 'off'} ({phase})   face size {face_size:.0f}   "
                f"lead {1000 * face_tracker.prediction_lead_seconds:.0f} ms",
                f"mapping: {estimator.model_name if estimator.is_valid() else 'NONE'}   inliers {estimator.inlier_count}   "
                f"residual {estimator.residual_rms_pixels:.1f} px   face pairs {estimator.face_pair_count()}   "
                f"bootstrap pairs {len(estimator.bootstrap_pairs)}",
                f"last cycle: {len(last_found)}/{len(servo_indices)} dots seen, landing offset {last_mean_offset:.1f} px",
            ]
            if outside_beam_note:
                status_lines.append(outside_beam_note)
            status_lines[1] += f"   {live_base_note}"
            if passive_note:
                status_lines.append(passive_note)
            if pointer_note:
                status_lines.append(pointer_note)
            warnings = []
            if passive_warning:
                warnings.append(passive_warning)
            if not estimator.is_valid():
                warnings.append("NO MAPPING (H) - press b to bootstrap: face inside the beam, hold still 5 s")
            elif not face_available:
                warnings.append("NO FACE DETECTED")
            elif servo_enabled and failed_cycles_in_a_row >= config.servo_warn_after_failed_cycles:
                warnings.append(f"SERVO LOST THE DOTS ({failed_cycles_in_a_row} cycles): mapping may be stale - press b")
            coverage_quad = None
            if estimator.is_valid():
                coverage_quad = projector_coverage_in_camera(
                    estimator.homography_for_size(face_size), projector.width, projector.height)
            debug_view = compose_debug_view(
                frame, smoothed_landmarks, servo_indices, last_gating_points, last_found,
                last_difference, last_search_radius, status_lines, coverage_quad,
                landmark_projection_indices if landmark_mode else None, clean_view)
            if arguments.pointer and pointer["position"] is not None:
                mouse_x, mouse_y = int(pointer["position"][0]), int(pointer["position"][1])
                cv2.line(debug_view, (mouse_x - 25, mouse_y), (mouse_x + 25, mouse_y), (0, 255, 255), 2)
                cv2.line(debug_view, (mouse_x, mouse_y - 25), (mouse_x, mouse_y + 25), (0, 255, 255), 2)
                cv2.circle(debug_view, (mouse_x, mouse_y), 40, (0, 255, 255), 1)
                if pointer["measured"] is not None:
                    measured_x, measured_y = int(pointer["measured"][0]), int(pointer["measured"][1])
                    cv2.circle(debug_view, (measured_x, measured_y), 12, (0, 0, 255), 2)
                    cv2.line(debug_view, (mouse_x, mouse_y), (measured_x, measured_y), (0, 0, 255), 2)
            if last_seen_disc is not None and landmark_mode:
                cv2.circle(debug_view, (int(last_seen_disc[0]), int(last_seen_disc[1])), 14, (0, 0, 255), 2)
                cv2.putText(debug_view, "disc seen", (int(last_seen_disc[0]) + 16, int(last_seen_disc[1]) + 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
            for warning_index, warning in enumerate(warnings):
                position = (10, debug_view.shape[0] - 24 - 40 * warning_index)
                cv2.putText(debug_view, warning, position, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 6)
                cv2.putText(debug_view, warning, position, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            if arguments.face_mesh:
                panel_height = debug_view.shape[0]
                right_panel = mesh.get("latest_view")
                if right_panel is None:
                    right_panel = numpy.zeros((projector.height, projector.width, 3), dtype=numpy.uint8)
                    cv2.putText(right_panel, "3D view: waiting for P (needs >= 6 face dots)", (40, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 4)
                right_panel = cv2.resize(right_panel, (int(projector.width * panel_height / projector.height), panel_height))
                cv2.putText(right_panel, "projector viewpoint (3D mesh)", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
                cv2.putText(right_panel, "projector viewpoint (3D mesh)", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                debug_view = numpy.hstack([debug_view, right_panel])
            cv2.imshow(DEBUG_WINDOW_NAME, debug_view)
            if now - last_status_print >= 2.0 and cycle is None and now - last_cycle_end > 0.5:
                last_status_print = now
                print(f"[status] {status_lines[0]}")
                if config.bootstrap_save_debug_images:
                    cv2.imwrite(f"{config.debug_image_directory}/debug_view.png", debug_view)
                    cv2.imwrite(f"{config.debug_image_directory}/camera_frame.png", frame)
                    cv2.imwrite(f"{config.debug_image_directory}/projector_image.png",
                                cv2.resize(projector_image, (960, 540)))

            # ------------------------------------------------ keys
            key = cv2.waitKey(1) & 0xFF
            shown_at = time.perf_counter()
            if cycle is not None:
                if cycle.frozen_shown_at is None:
                    cycle.frozen_shown_at = shown_at
                if drawing_dots_this_frame and cycle.dots_shown_at is None:
                    cycle.dots_shown_at = shown_at

            if key in (ord("q"), 27):
                break
            if arguments.seconds is not None and now - run_start > arguments.seconds:
                print(f"[main] {arguments.seconds:.0f} s elapsed; quitting")
                break
            elif key == ord("b"):
                bootstrap_estimator()
                cycle = None
            elif key == ord("s"):
                estimator.save(config.homography_file_path)
            elif key == ord("m"):
                mask_enabled = not mask_enabled
            elif key == ord("l"):
                landmark_mode = not landmark_mode
            elif key == ord("c"):
                clean_view = not clean_view
            elif key == ord("g"):
                dense_scan_and_proof()
            elif key == ord("f"):
                face_proof()
            elif key == ord("e"):
                export_now()
            elif key == ord("z"):
                estimator.depth_model_enabled = not estimator.depth_model_enabled
                estimator.clear_face_pairs()
                landmark_offsets.clear()
                print(f"[main] depth model {'ON' if estimator.depth_model_enabled else 'OFF'}; face correspondences cleared")
            elif key == ord("k"):
                border_markers_enabled = not border_markers_enabled
            elif key in (ord("["), ord("]")):
                step = config.prediction_lead_step_seconds * (1 if key == ord("]") else -1)
                face_tracker.prediction_lead_seconds = max(0.0, face_tracker.prediction_lead_seconds + step)
                print(f"[main] prediction lead {1000 * face_tracker.prediction_lead_seconds:.0f} ms")
            elif key == ord("v"):
                servo_enabled = not servo_enabled
            elif key == ord("a"):
                alignment_dots_always_on = not alignment_dots_always_on
            elif key == ord("r"):
                estimator.clear_face_pairs()
    finally:
        if estimator.is_valid() and estimator.face_pair_count() > 0:
            estimator.save(config.homography_file_path)
        face_tracker.close()
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
