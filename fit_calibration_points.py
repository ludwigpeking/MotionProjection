"""Refit the projector model from the raw correspondences saved by kinect_calibrate.py
(diagnostics/kinect_calibration_points.json), without touching the sensor.

    .venv\\Scripts\\python.exe fit_calibration_points.py            # report only
    .venv\\Scripts\\python.exe fit_calibration_points.py --save     # also overwrite projector_calibration.json
"""

import argparse
import json
import os
import time

import numpy

import config
from projector_calibration import fit_projector, fit_projector_focal_and_pose, fit_projector_pose


def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--points", type=str,
                                 default=os.path.join(config.debug_image_directory, "kinect_calibration_points.json"))
    argument_parser.add_argument("--save", action="store_true", help="write the refitted model to the calibration file")
    argument_parser.add_argument("--outlier-pixels", type=float, default=config.kinect_calibration_outlier_pixels)
    argument_parser.add_argument("--fit-intrinsics", action="store_true",
                                 help="fit focal length, principal point and distortion too (default: fixed from config)")
    arguments = argument_parser.parse_args()

    def fit(fit_points_3d, fit_projector_points, min_points, width, height):
        if config.kinect_projector_use_fixed_intrinsics and not arguments.fit_intrinsics:
            if config.kinect_projector_fit_focal:
                return fit_projector_focal_and_pose(
                    fit_points_3d, fit_projector_points, width, height, arguments.outlier_pixels,
                    config.kinect_projector_focal_pixels, config.kinect_projector_principal_point,
                    config.kinect_projector_distortion, min_points, config.kinect_projector_focal_search_fraction,
                    config.kinect_projector_focal_near_depth_fraction, config.kinect_projector_focal_min_near_dots)
            return fit_projector_pose(fit_points_3d, fit_projector_points, width, height, arguments.outlier_pixels,
                                      config.kinect_projector_focal_pixels, config.kinect_projector_principal_point,
                                      config.kinect_projector_distortion, min_points)
        return fit_projector(fit_points_3d, fit_projector_points, width, height, arguments.outlier_pixels, min_points)

    with open(arguments.points) as file:
        data = json.load(file)
    points_3d = numpy.array(data["points_3d"], dtype=numpy.float64)
    projector_points = numpy.array(data["projector_points"], dtype=numpy.float64)
    width, height = data["projector_width"], data["projector_height"]
    print(f"[refit] {len(points_3d)} correspondences, Kinect depth {points_3d[:, 2].min():.2f} .. {points_3d[:, 2].max():.2f} m")

    rng = numpy.random.default_rng(int(time.time()))
    order = rng.permutation(len(points_3d))
    holdout_count = int(round(config.kinect_calibration_holdout_fraction * len(points_3d)))
    holdout, training = order[:holdout_count], order[holdout_count:]
    partial, _ = fit(points_3d[training], projector_points[training], config.kinect_calibration_min_points // 2,
                     width, height)
    holdout_rms = float("nan")
    if partial is not None and holdout_count > 0:
        holdout_errors = partial.reprojection_errors(points_3d[holdout], projector_points[holdout])
        holdout_rms = float(numpy.sqrt(numpy.mean(holdout_errors ** 2)))
        print(f"[refit] held-out {holdout_count} dots: rms {holdout_rms:.2f} px, median {numpy.median(holdout_errors):.2f} px, "
              f"max {holdout_errors.max():.2f} px")

    calibration, inliers = fit(points_3d, projector_points, config.kinect_calibration_min_points, width, height)
    if calibration is None:
        raise SystemExit("[refit] the fit failed")
    errors = calibration.reprojection_errors(points_3d, projector_points)
    print(f"[refit] {calibration.inlier_count}/{calibration.point_count} inliers, rms {calibration.rms_pixels:.2f} px, "
          f"median error over all dots {numpy.median(errors):.2f} px; {calibration.describe()}")
    print(f"[refit] projector-frame depth of the inliers {calibration.depth_range_metres[0]:.2f} .. "
          f"{calibration.depth_range_metres[1]:.2f} m")
    if arguments.save:
        calibration.holdout_rms_pixels = holdout_rms
        calibration.latency_none_before_seconds = data.get("latency_none_before_seconds")
        calibration.latency_full_after_seconds = data.get("latency_full_after_seconds")
        if calibration.latency_full_after_seconds is not None:
            calibration.prediction_lead_seconds = (calibration.latency_full_after_seconds
                                                   + config.kinect_prediction_lead_extra_seconds)
        calibration.save(config.kinect_calibration_file_path)
        print(f"[refit] saved {config.kinect_calibration_file_path}")


if __name__ == "__main__":
    main()
