"""Calibrate the projector against the Kinect v2: metric 3D -> projector pixels.

1. Measures the projector -> Kinect colour latency with black/white flashes.
2. Runs the coded-dot scan (bootstrap.run_bootstrap) with the Kinect colour
   image as the camera: every decoded dot is a (colour pixel, projector pixel) pair.
3. Looks up each dot's 3D point in a temporally medianed depth frame.
4. Fits the projector as a camera (projector_calibration.fit_projector), holding
   a fraction of the dots out of the fit to report an honest error, and saves
   projector_calibration.json for kinect_red_dot.py.

The scene must not move during the ~10 s: sit still in the beam or step out.
Either way the beam should hit surfaces at clearly different depths (wall,
chair, desk, person); a single plane cannot pin down the projector's pose.

    .venv\\Scripts\\python.exe kinect_calibrate.py               # full run
    .venv\\Scripts\\python.exe kinect_calibrate.py --skip-latency  # keep config's latency numbers
"""

import argparse
import json
import os
import time

import cv2
import numpy

import config
from bootstrap import run_bootstrap
from displays import pick_projector_monitor
from face_tracker import FaceTracker
from kinect_camera import open_camera
from panel_link import PanelLink
from projector import ProjectorWindow
from projector_calibration import ProjectorCalibration, fit_projector, fit_projector_focal_and_pose, fit_projector_pose
from projector_view import render_projector_view

DEBUG_WINDOW_NAME = "kinect calibration"


def _small_gray(frame_bgr):
    small = cv2.resize(frame_bgr, (480, 270), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(numpy.float32)


def wait_for_steady_stream(camera, needed_frames=20, timeout_seconds=30.0):
    """Block until the Kinect delivered needed_frames within one second (a stall is over)."""
    deadline = time.perf_counter() + timeout_seconds
    while time.perf_counter() < deadline:
        count_before = camera.captured_frame_count
        camera.frame_after(time.perf_counter() + 1.0, timeout_seconds=2.0)
        if camera.captured_frame_count - count_before >= needed_frames:
            return True
    print("[calibration] the Kinect stream did not recover within 30 s")
    return False


def measure_latency(camera, projector, toggles):
    """How long after a change is shown the Kinect colour frames still show nothing
    of it, and after how long they show all of it: (none_before_seconds, full_after_seconds),
    or (None, None) when the Kinect does not see the projector at all. The brightness
    is measured only inside the beam (the pixels the white frame lit up), so the
    step is large even in a bright room."""
    black = numpy.zeros((projector.height, projector.width, 3), dtype=numpy.uint8)
    white = numpy.full((projector.height, projector.width, 3), 255, dtype=numpy.uint8)
    projector.show(black)
    cv2.waitKey(1)
    camera.frame_after(time.perf_counter() + 1.0, timeout_seconds=3.0)
    last_none_times = []
    first_full_times = []
    for toggle in range(toggles):
        baseline_frames = []
        last_timestamp = time.perf_counter()
        while len(baseline_frames) < 5:
            result = camera.frame_after(last_timestamp, timeout_seconds=2.0)
            if result is None:
                break
            last_timestamp, frame = result
            baseline_frames.append(_small_gray(frame))
        if not baseline_frames:
            continue
        baseline_image = numpy.mean(baseline_frames, axis=0)

        projector.show(white)
        cv2.waitKey(1)
        shown_at = time.perf_counter()
        samples = []
        last_timestamp = shown_at
        while time.perf_counter() - shown_at < 1.0:
            result = camera.frame_after(last_timestamp, timeout_seconds=1.0)
            if result is None:
                break
            last_timestamp, frame = result
            samples.append((last_timestamp - shown_at, _small_gray(frame)))
        projector.show(black)
        cv2.waitKey(1)
        camera.frame_after(time.perf_counter() + 1.0, timeout_seconds=3.0)

        if len(samples) < 3:
            continue
        # The beam: pixels the last frames lit by a clear margin.
        lit_image = numpy.mean([image for _, image in samples[-3:]], axis=0)
        beam_mask = (lit_image - baseline_image) > 20.0
        if beam_mask.mean() < 0.002:
            print(f"[latency] flash {toggle + 1}: the Kinect barely saw the white frame "
                  f"({beam_mask.mean() * 100:.2f}% of pixels lit); is the projector pointing into the Kinect's view?")
            continue
        if toggle == 0:
            # Where the beam falls in the Kinect view: the lit pixels tinted on the latest frame.
            _, latest_frame = camera.latest()
            beam_view = latest_frame.copy()
            full_mask = cv2.resize(beam_mask.astype(numpy.uint8), (beam_view.shape[1], beam_view.shape[0]),
                                   interpolation=cv2.INTER_NEAREST).astype(bool)
            beam_view[full_mask] = (0.5 * beam_view[full_mask] + numpy.array([0, 128, 128])).astype(numpy.uint8)
            cv2.putText(beam_view, "projector beam as the Kinect sees it (tinted)", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            os.makedirs(config.debug_image_directory, exist_ok=True)
            config.kinect_save_diagnostic_images and cv2.imwrite(os.path.join(config.debug_image_directory, "kinect_beam.png"), beam_view)
        baseline = float(baseline_image[beam_mask].mean())
        curve = [(elapsed, float(image[beam_mask].mean())) for elapsed, image in samples]
        peak = max(value for _, value in curve)
        step = peak - baseline
        rise_threshold = baseline + 0.15 * step
        full_threshold = baseline + 0.85 * step
        last_none = 0.0
        first_full = None
        for elapsed, value in curve:
            if first_full is not None:
                break
            if value < rise_threshold:
                last_none = elapsed
            elif value >= full_threshold:
                first_full = elapsed
        if first_full is None:
            first_full = curve[-1][0]
        last_none_times.append(last_none)
        first_full_times.append(first_full)
        print(f"[latency] flash {toggle + 1}: beam covers {beam_mask.mean() * 100:.0f}% of the Kinect image, "
              f"brightness {baseline:.0f} -> {peak:.0f}; nothing until {last_none * 1000:.0f} ms, "
              f"complete at {first_full * 1000:.0f} ms")
    if not last_none_times:
        return None, None
    # Medians: one flash can be stretched by a stream stall or the colour auto-exposure settling.
    return float(numpy.median(last_none_times)), float(numpy.median(first_full_times))


def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--skip-latency", action="store_true",
                                 help="do not measure the projector -> Kinect latency; keep the config values")
    argument_parser.add_argument("--allow-window", action="store_true",
                                 help="run even without a second monitor (dry run in a window; the result is meaningless)")
    argument_parser.add_argument("--scans", type=int, default=config.kinect_calibration_scans,
                                 help="number of coded scans pooled into one fit; sit at a different distance for each")
    argument_parser.add_argument("--pause", type=int, default=config.kinect_calibration_pause_seconds,
                                 help="seconds shown as a countdown on the projector before each scan")
    argument_parser.add_argument("--panel", action="store_true",
                                 help="driven by web_panel.py: the live view also goes to the browser")
    argument_parser.add_argument("--fit-intrinsics", action="store_true",
                                 help="fit the projector's focal length, principal point and distortion too, instead of "
                                      "using the fixed values in config (needs objects at several depths in the beam)")
    arguments = argument_parser.parse_args()
    panel = PanelLink(config.debug_image_directory, arguments.panel)

    monitor, is_second_screen = pick_projector_monitor(
        config.projector_monitor_index, config.projector_fallback_width_pixels, config.projector_fallback_height_pixels)
    if not is_second_screen and not arguments.allow_window:
        raise SystemExit("[calibration] no second monitor: connect the projector in extend mode first "
                         "(or pass --allow-window for a dry run)")

    camera = open_camera(config)
    projector = ProjectorWindow(monitor, fullscreen=is_second_screen)
    face_tracker = FaceTracker(config, camera.frame_width, camera.frame_height)
    previous_calibration = None
    if os.path.exists(config.kinect_calibration_file_path):
        previous_calibration = ProjectorCalibration.load(config.kinect_calibration_file_path)
    cv2.namedWindow(DEBUG_WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(DEBUG_WINDOW_NAME, 960, 540)
    black = numpy.zeros((projector.height, projector.width, 3), dtype=numpy.uint8)

    try:
        latency_none_before = config.latency_none_before_seconds
        latency_full_after = config.latency_full_after_seconds
        if not arguments.skip_latency:
            measured_none_before, measured_full_after = measure_latency(camera, projector, config.kinect_latency_test_toggles)
            if measured_none_before is None:
                print("[latency] could not measure; keeping the config values")
            else:
                latency_none_before = max(0.0, measured_none_before - 0.01)
                latency_full_after = measured_full_after + 0.02
                print(f"[latency] projector -> Kinect: nothing before {latency_none_before:.3f} s, "
                      f"complete after {latency_full_after:.3f} s (config webcam values: "
                      f"{config.latency_none_before_seconds:.2f} / {config.latency_full_after_seconds:.2f})")
                config.latency_none_before_seconds = latency_none_before
                config.latency_full_after_seconds = latency_full_after

        # The scan code reads these two settings from config; the calibration has its own values
        # because its scene is static (see config.kinect_calibration_average_frames).
        config.bootstrap_average_frames = config.kinect_calibration_average_frames
        config.bootstrap_min_peak_difference = config.kinect_calibration_min_peak_difference
        config.bootstrap_extra_settle_seconds = config.kinect_calibration_extra_settle_seconds
        all_points_3d = []
        all_projector_points = []
        all_camera_pixels = []
        for scan_index in range(arguments.scans):
            if arguments.scans > 1:
                message = (f"scan {scan_index + 1} of {arguments.scans}: sit in the beam at a "
                           f"{'new' if scan_index else ''} distance with the dots on your face and chest, then hold still")
                print(f"[calibration] {message}")
                countdown_started = time.perf_counter()
                last_timestamp = 0.0
                shown_remaining = None
                while True:
                    remaining = arguments.pause - int(time.perf_counter() - countdown_started)
                    if remaining <= 0:
                        break
                    if remaining != shown_remaining:
                        shown_remaining = remaining
                        notice = black.copy()
                        cv2.putText(notice, message, (60, projector.height // 2 - 40), cv2.FONT_HERSHEY_SIMPLEX,
                                    1.3, (255, 255, 255), 3)
                        cv2.putText(notice, f"starting in {remaining} s", (60, projector.height // 2 + 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3)
                        projector.show(notice)
                    # Live Kinect view in the debug window while the person gets into position.
                    result = camera.latest_after(last_timestamp, timeout_seconds=0.2)
                    if result is not None:
                        last_timestamp, frame = result
                        live = cv2.resize(frame, (960, 540), interpolation=cv2.INTER_AREA)
                        cv2.putText(live, f"{message} - {remaining} s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                    (0, 255, 255), 2)
                        cv2.imshow(DEBUG_WINDOW_NAME, live)
                        panel.publish(live)
                        if previous_calibration is not None and panel.projector_view_due():
                            depth_entry = camera.depth_for(last_timestamp)
                            if depth_entry is not None:
                                projector_view = render_projector_view(camera, depth_entry[1], frame,
                                                                       previous_calibration, 960, 540)
                                cv2.putText(projector_view, "projector's-eye view (previous calibration)", (10, 24),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                                panel.publish_projector_view(projector_view)
                    cv2.waitKey(1)
                projector.show(black)
                cv2.waitKey(1)
                camera.frame_after(time.perf_counter() + 0.5, timeout_seconds=3.0)
            pairs = []
            for attempt in range(2):
                grid_offset = config.kinect_calibration_grid_offsets[scan_index % len(config.kinect_calibration_grid_offsets)]
                pairs, _ = run_bootstrap(camera, projector, face_tracker, config, DEBUG_WINDOW_NAME,
                                         grid_columns=config.kinect_calibration_grid_columns,
                                         grid_rows=config.kinect_calibration_grid_rows,
                                         dot_radius=config.scan_dot_radius_pixels,
                                         opening_radius=config.scan_opening_radius_pixels,
                                         sample_radius=config.scan_code_sample_radius_pixels, return_all=True,
                                         grid_offset_fraction=grid_offset, static_scene=True)
                projector.show(black)
                cv2.waitKey(1)
                if len(pairs) >= config.kinect_calibration_min_points // 2:
                    break
                print(f"[calibration] scan {scan_index + 1}: only {len(pairs)} dots decoded"
                      + ("; waiting for the Kinect stream and retrying, hold still" if attempt == 0 else "; skipped"))
                wait_for_steady_stream(camera)
            if len(pairs) < config.kinect_calibration_min_points // 2:
                continue
            depth_millimetres = camera.median_depth(config.kinect_calibration_depth_frames)
            if depth_millimetres is None:
                raise SystemExit("[calibration] the Kinect delivered no depth frames after the scan "
                                 "(stream stalled: is the CPU saturated by another program?)")
            camera_pixels = numpy.array([pair[0] for pair in pairs], dtype=numpy.float64)
            projector_points = numpy.array([pair[1] for pair in pairs], dtype=numpy.float64)
            points_3d, depth_spread = camera.points_3d_at(depth_millimetres, camera_pixels, return_depth_spread=True)
            has_depth = numpy.isfinite(points_3d[:, 2])
            on_edge = has_depth & (depth_spread > config.kinect_calibration_max_depth_spread_metres)
            usable = has_depth & ~on_edge
            print(f"[calibration] scan {scan_index + 1}: {len(pairs)} dots decoded, {int(has_depth.sum())} with depth, "
                  f"{int(on_edge.sum())} on a depth edge (dropped), {int(usable.sum())} usable; "
                  f"depth {points_3d[usable, 2].min():.2f} .. {points_3d[usable, 2].max():.2f} m"
                  if usable.any() else f"[calibration] scan {scan_index + 1}: no usable dots")
            all_points_3d.append(points_3d[usable])
            all_projector_points.append(projector_points[usable])
            all_camera_pixels.append(camera_pixels[usable])
        if not all_points_3d:
            raise SystemExit("[calibration] no usable scan")
        points_3d = numpy.vstack(all_points_3d)
        projector_points = numpy.vstack(all_projector_points)
        camera_pixels = numpy.vstack(all_camera_pixels)
        if len(points_3d) < config.kinect_calibration_min_points:
            raise SystemExit(f"[calibration] only {len(points_3d)} usable dots; at least "
                             f"{config.kinect_calibration_min_points} needed")
        os.makedirs(config.debug_image_directory, exist_ok=True)
        points_path = os.path.join(config.debug_image_directory, "kinect_calibration_points.json")
        with open(points_path, "w") as file:
            json.dump({"points_3d": points_3d.tolist(), "projector_points": projector_points.tolist(),
                       "camera_pixels": camera_pixels.tolist(),
                       "projector_width": projector.width, "projector_height": projector.height,
                       "latency_none_before_seconds": latency_none_before,
                       "latency_full_after_seconds": latency_full_after}, file)
        print(f"[calibration] raw correspondences saved to {points_path} (refit offline with fit_calibration_points.py)")
        depth_span = float(points_3d[:, 2].max() - points_3d[:, 2].min())
        print(f"[calibration] Kinect depth of the dots: {points_3d[:, 2].min():.2f} .. {points_3d[:, 2].max():.2f} m "
              f"(span {depth_span:.2f} m)")
        if depth_span < config.kinect_calibration_min_depth_span_metres:
            print("[calibration] WARNING: the dots lie almost on one plane; put something (a chair, yourself) "
                  "at a different distance inside the beam for a well-conditioned fit")

        # Honest error: fit without a held-out subset, measure on it, then refit on everything.
        rng = numpy.random.default_rng(int(time.time()))
        order = rng.permutation(len(points_3d))
        holdout_count = int(round(config.kinect_calibration_holdout_fraction * len(points_3d)))
        holdout = order[:holdout_count]
        training = order[holdout_count:]
        def fit(fit_points_3d, fit_projector_points, min_points):
            if config.kinect_projector_use_fixed_intrinsics and not arguments.fit_intrinsics:
                if config.kinect_projector_fit_focal:
                    return fit_projector_focal_and_pose(
                        fit_points_3d, fit_projector_points, projector.width, projector.height,
                        config.kinect_calibration_outlier_pixels, config.kinect_projector_focal_pixels,
                        config.kinect_projector_principal_point, config.kinect_projector_distortion, min_points,
                        config.kinect_projector_focal_search_fraction, config.kinect_projector_focal_near_depth_fraction,
                        config.kinect_projector_focal_min_near_dots)
                return fit_projector_pose(fit_points_3d, fit_projector_points, projector.width, projector.height,
                                          config.kinect_calibration_outlier_pixels, config.kinect_projector_focal_pixels,
                                          config.kinect_projector_principal_point, config.kinect_projector_distortion,
                                          min_points)
            return fit_projector(fit_points_3d, fit_projector_points, projector.width, projector.height,
                                 config.kinect_calibration_outlier_pixels, min_points)

        partial, _ = fit(points_3d[training], projector_points[training], config.kinect_calibration_min_points // 2)
        holdout_rms = float("nan")
        if partial is not None and holdout_count > 0:
            holdout_errors = partial.reprojection_errors(points_3d[holdout], projector_points[holdout])
            holdout_rms = float(numpy.sqrt(numpy.mean(holdout_errors ** 2)))
            print(f"[calibration] held-out {holdout_count} dots: rms {holdout_rms:.2f} px, "
                  f"median {numpy.median(holdout_errors):.2f} px, max {holdout_errors.max():.2f} px")

        calibration, inliers = fit(points_3d, projector_points, config.kinect_calibration_min_points)
        if calibration is None:
            raise SystemExit("[calibration] the fit failed (too few consistent dots)")
        calibration.holdout_rms_pixels = holdout_rms
        calibration.latency_none_before_seconds = latency_none_before
        calibration.latency_full_after_seconds = latency_full_after
        calibration.prediction_lead_seconds = latency_full_after + config.kinect_prediction_lead_extra_seconds
        calibration.save(config.kinect_calibration_file_path)
        print(f"[calibration] {calibration.inlier_count}/{calibration.point_count} dots fit to rms "
              f"{calibration.rms_pixels:.2f} projector px; {calibration.describe()}")
        print(f"[calibration] prediction lead for the red dot: {calibration.prediction_lead_seconds * 1000:.0f} ms")
        print(f"[calibration] saved {config.kinect_calibration_file_path}")

        latest_timestamp, latest_frame = camera.latest()
        depth_entry = camera.depth_for(latest_timestamp)
        if depth_entry is not None:
            projector_view = render_projector_view(camera, depth_entry[1], latest_frame, calibration, 960, 540)
            cv2.putText(projector_view, "projector's-eye view (new calibration)", (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2)
            panel.publish_projector_view(projector_view)
        view = latest_frame.copy()
        errors = calibration.reprojection_errors(points_3d, projector_points)
        for pixel, error, inlier in zip(camera_pixels, errors, inliers):
            color = (0, 220, 0) if inlier else (0, 0, 255)
            cv2.circle(view, (int(pixel[0]), int(pixel[1])), 10, color, 2)
            cv2.putText(view, f"{error:.1f}", (int(pixel[0]) + 10, int(pixel[1]) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(view, f"projector calibration: {calibration.inlier_count}/{calibration.point_count} dots, "
                          f"rms {calibration.rms_pixels:.2f} px, held-out rms {holdout_rms:.2f} px",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        os.makedirs(config.debug_image_directory, exist_ok=True)
        snapshot_path = os.path.join(config.debug_image_directory, "kinect_calibration.png")
        config.kinect_save_diagnostic_images and cv2.imwrite(snapshot_path, view)
        if config.kinect_save_diagnostic_images:
            print(f"[calibration] residual map saved to {snapshot_path}")
        cv2.imshow(DEBUG_WINDOW_NAME, view)
        panel.publish(cv2.resize(view, (960, 540), interpolation=cv2.INTER_AREA), force=True)
        cv2.waitKey(1500)
    finally:
        face_tracker.close()
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
