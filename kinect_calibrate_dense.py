"""Complete projector calibration from a static scene, with a dense Gray-code scan.

Set the scene up first: objects at several distances inside the beam, from as near as
the face will be (0.6 .. 0.9 m) to a far wall, and nobody moving in it. Then:

    .venv\\Scripts\\python.exe kinect_calibrate_dense.py

1. Measures the projector -> Kinect latency with black/white flashes.
2. Takes the median of several Kinect depth frames: the 3D point of every colour pixel.
3. Shows Gray-code stripe patterns (gray_code_scan.py) and reads, for every camera pixel,
   which projector cell lights it. One correspondence per cell: projector pixel, 3D point.
4. Fits the complete projector model (projector_calibration.fit_projector_complete):
   focal length, lens centre, distortion, rotation, translation. Nothing is assumed
   except the starting values.
5. Checks the model on cells that were kept out of the fit, per distance band, and saves
   projector_calibration.json.

The scan takes about a minute. Keys in the debug window: q / Esc to abort.
"""

import argparse
import json
import os
import time

import cv2
import numpy

import config
import gray_code_scan
from displays import pick_projector_monitor
from kinect_calibrate import measure_latency, wait_for_steady_stream
from kinect_camera import open_camera
from panel_link import PanelLink
from projector import ProjectorWindow
from projector_calibration import fit_projector_complete, report_errors_by_distance
from projector_view import render_projector_view

DEBUG_WINDOW_NAME = "kinect dense calibration"


def residual_view(frame, camera_pixels, errors, outlier_pixels, title):
    """The camera image with every correspondence drawn: green within the limit, red beyond."""
    view = cv2.resize(frame, (960, 540), interpolation=cv2.INTER_AREA)
    scale = 960.0 / frame.shape[1]
    for (column, row), error in zip(camera_pixels, errors):
        colour = (0, 220, 0) if error < outlier_pixels else (0, 0, 255)
        cv2.circle(view, (int(column * scale), int(row * scale)), 2, colour, -1)
    band = view[0:34, :]
    band[:] = (band * 0.35).astype(numpy.uint8)
    cv2.putText(view, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return view


def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--skip-latency", action="store_true",
                                 help="do not measure the projector -> Kinect latency; keep the config values")
    argument_parser.add_argument("--panel", action="store_true", help="publish the views to the web panel, open no window")
    arguments = argument_parser.parse_args()
    panel = PanelLink(config.debug_image_directory, arguments.panel)
    show_window = not arguments.panel

    monitor, is_second_screen = pick_projector_monitor(
        config.projector_monitor_index, config.projector_fallback_width_pixels, config.projector_fallback_height_pixels)
    if not is_second_screen:
        raise SystemExit("[dense] no second screen found: the projector must be connected as an extended display")
    camera = open_camera(config)
    projector = ProjectorWindow(monitor, fullscreen=True, background_fraction=0.0)
    if show_window:
        cv2.namedWindow(DEBUG_WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(DEBUG_WINDOW_NAME, 960, 540)
    black = numpy.zeros((projector.height, projector.width, 3), dtype=numpy.uint8)

    try:
        projector.show(black)
        cv2.waitKey(1)
        wait_for_steady_stream(camera, needed_frames=5)
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
                      f"complete after {latency_full_after:.3f} s")
        projector.show(black)
        cv2.waitKey(1)
        camera.frame_after(time.perf_counter() + latency_full_after + 0.3, timeout_seconds=5.0)

        depth_millimetres = camera.median_depth(config.kinect_calibration_depth_frames)
        if depth_millimetres is None:
            raise SystemExit("[dense] the Kinect delivered no depth frames")
        points_3d_image = camera.map_depth_to_colour_points(depth_millimetres).astype(numpy.float64)
        _, reference_frame = camera.latest()
        reference_frame = reference_frame.copy()

        def progress(shown, total):
            view = cv2.resize(reference_frame, (960, 540), interpolation=cv2.INTER_AREA)
            band = view[0:34, :]
            band[:] = (band * 0.35).astype(numpy.uint8)
            cv2.putText(view, f"dense scan: pattern {shown} of {total} (keep the scene still)", (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            panel.publish(view)
            if show_window:
                cv2.imshow(DEBUG_WINDOW_NAME, view)
            if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27) or panel.command() == "quit":
                raise SystemExit("[dense] aborted")

        print(f"[dense] scanning: stripes down to {config.kinect_dense_finest_stripe_pixels} projector px, "
              f"{config.kinect_dense_average_frames} frames per pattern")
        scan_started = time.perf_counter()
        scan_result = gray_code_scan.scan(
            projector, camera, config.kinect_dense_finest_stripe_pixels,
            latency_full_after + config.kinect_dense_extra_settle_seconds, config.kinect_dense_average_frames,
            config.bootstrap_frame_timeout_seconds, config.kinect_dense_minimum_beam_step,
            config.kinect_dense_minimum_bit_contrast_fraction, progress)
        if scan_result is None:
            raise SystemExit("[dense] the Kinect stream stalled during the scan; run it again")
        print(f"[dense] scan took {time.perf_counter() - scan_started:.0f} s; the beam covers "
              f"{100.0 * scan_result['beam'].mean():.0f}% of the Kinect image")
        bit_count = scan_result["bit_count"]
        both = numpy.minimum(scan_result["column_reliable_bits"], scan_result["row_reliable_bits"])[scan_result["beam"]]
        for level in range(config.kinect_dense_coarsest_level + 1):
            print(f"[dense] beam pixels readable down to {config.kinect_dense_finest_stripe_pixels * 2 ** level:3d} px cells: "
                  f"{100.0 * (both >= bit_count - level).mean():.0f}%")

        cells = gray_code_scan.cell_correspondences(
            scan_result, points_3d_image, config.kinect_dense_finest_stripe_pixels, projector.width, projector.height,
            config.kinect_dense_coarsest_level, camera.colour_focal_x / config.kinect_projector_focal_pixels,
            config.kinect_dense_minimum_cell_fill, config.kinect_calibration_max_depth_spread_metres,
            config.kinect_dense_maximum_depth_spread_fraction)
        points_3d, projector_points, camera_pixels = cells["points_3d"], cells["projector_points"], cells["camera_pixels"]
        print(f"[dense] {len(points_3d)} cells with a 3D point; by level "
              + ", ".join(f"{config.kinect_dense_finest_stripe_pixels * 2 ** level} px: {int((cells['level'] == level).sum())}"
                          for level in range(config.kinect_dense_coarsest_level + 1)))
        if len(points_3d) < config.kinect_calibration_min_points:
            raise SystemExit(f"[dense] only {len(points_3d)} cells; at least {config.kinect_calibration_min_points} needed")
        histogram, edges = numpy.histogram(points_3d[:, 2], bins=numpy.arange(0.4, 4.41, 0.4))
        print("[dense] cells by distance: " + ", ".join(f"{low:.1f}-{high:.1f} m: {count}"
                                                       for count, low, high in zip(histogram, edges[:-1], edges[1:]) if count))

        os.makedirs(config.debug_image_directory, exist_ok=True)
        points_path = os.path.join(config.debug_image_directory, "kinect_calibration_points.json")
        with open(points_path, "w") as file:
            json.dump({"points_3d": points_3d.tolist(), "projector_points": projector_points.tolist(),
                       "camera_pixels": camera_pixels.tolist(), "cell_level": cells["level"].tolist(),
                       "projector_width": projector.width, "projector_height": projector.height,
                       "latency_none_before_seconds": latency_none_before,
                       "latency_full_after_seconds": latency_full_after, "method": "gray code cells"}, file)
        print(f"[dense] correspondences saved to {points_path}")

        def fit(fit_points, fit_pixels, quiet):
            return fit_projector_complete(fit_points, fit_pixels, projector.width, projector.height,
                                          config.kinect_calibration_outlier_pixels, config.kinect_projector_focal_pixels,
                                          config.kinect_projector_principal_point, config.kinect_projector_distortion,
                                          config.kinect_calibration_min_points, quiet=quiet)

        # Honest error: fit without a held-out fifth, measure on it, then fit on everything.
        order = numpy.random.default_rng(int(time.time())).permutation(len(points_3d))
        holdout_count = int(round(config.kinect_calibration_holdout_fraction * len(points_3d)))
        holdout, training = order[:holdout_count], order[holdout_count:]
        holdout_rms = float("nan")
        partial, _ = fit(points_3d[training], projector_points[training], quiet=True)
        if partial is not None and holdout_count > 0:
            holdout_errors = report_errors_by_distance(partial, points_3d[holdout], projector_points[holdout],
                                                       f"{holdout_count} cells kept out of the fit")
            holdout_rms = float(numpy.sqrt(numpy.mean(numpy.minimum(holdout_errors, 50.0) ** 2)))

        calibration, inliers = fit(points_3d, projector_points, quiet=False)
        if calibration is None:
            raise SystemExit("[dense] the fit failed")
        errors = report_errors_by_distance(calibration, points_3d, projector_points, "all cells")
        calibration.holdout_rms_pixels = holdout_rms
        calibration.latency_none_before_seconds = latency_none_before
        calibration.latency_full_after_seconds = latency_full_after
        calibration.prediction_lead_seconds = latency_full_after + config.kinect_prediction_lead_extra_seconds
        calibration.save(config.kinect_calibration_file_path)
        print(f"[dense] {calibration.inlier_count}/{calibration.point_count} cells within "
              f"{config.kinect_calibration_outlier_pixels:.0f} px, rms {calibration.rms_pixels:.2f} px; {calibration.describe()}")
        print(f"[dense] saved {config.kinect_calibration_file_path}")

        view = residual_view(reference_frame, camera_pixels, errors, config.kinect_calibration_outlier_pixels,
                             f"dense calibration: {calibration.inlier_count}/{calibration.point_count} cells, "
                             f"rms {calibration.rms_pixels:.2f} px, focal {calibration.camera_matrix[0, 0]:.0f} px")
        panel.publish(view)
        projector_view = render_projector_view(camera, depth_millimetres, reference_frame, calibration, 960, 540)
        cv2.putText(projector_view, "projector's-eye view (new calibration)", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2)
        panel.publish_projector_view(projector_view)
        if show_window:
            cv2.imshow(DEBUG_WINDOW_NAME, view)
            cv2.waitKey(1500)
    finally:
        projector.show(black)
        cv2.waitKey(1)
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
