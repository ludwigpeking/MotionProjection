"""Proof of the Kinect projector calibration, independent of face tracking.

Red discs are projected one at a time at a grid of projector positions. Each
disc is found in the Kinect colour image, its 3D point is read from the depth
frame, and that point is sent back through the calibration. The distance
between where the disc was drawn and where the model says it is, in projector
pixels, is the model's error on whatever surface the disc hit (wall or face).

    .venv\\Scripts\\python.exe kinect_prove.py
    .venv\\Scripts\\python.exe kinect_prove.py --columns 6 --rows 4 --radius 24
"""

import argparse
import os
import time

import cv2
import numpy

import config
from displays import pick_projector_monitor
from kinect_camera import KinectCamera
from projector import ProjectorWindow
from projector_calibration import ProjectorCalibration
from red_disc import find_red_disc


def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--columns", type=int, default=6)
    argument_parser.add_argument("--rows", type=int, default=4)
    argument_parser.add_argument("--radius", type=int, default=20, help="disc radius in projector pixels")
    argument_parser.add_argument("--calibration", type=str, default=config.kinect_calibration_file_path)
    arguments = argument_parser.parse_args()

    calibration = ProjectorCalibration.load(arguments.calibration)
    monitor, is_second_screen = pick_projector_monitor(
        config.projector_monitor_index, config.projector_fallback_width_pixels, config.projector_fallback_height_pixels)
    camera = KinectCamera(config)
    projector = ProjectorWindow(monitor, fullscreen=is_second_screen)
    black = numpy.zeros((projector.height, projector.width, 3), dtype=numpy.uint8)
    settle_seconds = (calibration.latency_full_after_seconds or config.latency_full_after_seconds) + 0.15

    margin = 0.08
    grid = [(float(x), float(y))
            for y in numpy.linspace(margin * projector.height, (1 - margin) * projector.height, arguments.rows)
            for x in numpy.linspace(margin * projector.width, (1 - margin) * projector.width, arguments.columns)]
    results = []
    _, overview = camera.latest()
    overview = overview.copy()
    try:
        projector.show(black)
        cv2.waitKey(1)
        camera.frame_after(time.perf_counter() + 1.0, timeout_seconds=5.0)
        for projector_x, projector_y in grid:
            canvas = black.copy()
            cv2.circle(canvas, (int(round(projector_x)), int(round(projector_y))), arguments.radius, (0, 0, 255), -1,
                       cv2.LINE_AA)
            projector.show(canvas)
            cv2.waitKey(1)
            result = camera.frame_after(time.perf_counter() + settle_seconds, timeout_seconds=10.0)
            if result is None:
                print(f"[prove] ({projector_x:.0f},{projector_y:.0f}): no frame")
                continue
            timestamp, frame = result
            centre = (frame.shape[1] / 2.0, frame.shape[0] / 2.0)
            disc = find_red_disc(frame, centre, max(frame.shape) , config.kinect_red_dot_min_area_pixels)
            if disc is None:
                print(f"[prove] ({projector_x:.0f},{projector_y:.0f}): disc not seen (outside the Kinect view or too dim)")
                continue
            depth_entry = camera.depth_for(timestamp)
            point_3d = camera.points_3d_at(depth_entry[1], [disc])[0]
            if not numpy.isfinite(point_3d[2]):
                print(f"[prove] ({projector_x:.0f},{projector_y:.0f}): seen at camera ({disc[0]:.0f},{disc[1]:.0f}) but no depth there")
                continue
            reprojected = calibration.project(point_3d)[0]
            error = float(numpy.hypot(reprojected[0] - projector_x, reprojected[1] - projector_y))
            results.append((projector_x, projector_y, disc, point_3d, reprojected, error))
            print(f"[prove] drawn ({projector_x:.0f},{projector_y:.0f}) seen at camera ({disc[0]:.0f},{disc[1]:.0f}) "
                  f"depth {point_3d[2]:.2f} m -> model says ({reprojected[0]:.0f},{reprojected[1]:.0f}): "
                  f"error {error:.1f} px")
            cv2.circle(overview, (int(disc[0]), int(disc[1])), 14, (0, 0, 255), 2)
            cv2.putText(overview, f"{error:.0f}px @{point_3d[2]:.1f}m", (int(disc[0]) + 14, int(disc[1]) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        projector.show(black)
        cv2.waitKey(1)
    finally:
        camera.release()
        cv2.destroyAllWindows()

    if not results:
        print("[prove] no disc was measured")
        return
    errors = numpy.array([entry[5] for entry in results])
    depths = numpy.array([entry[3][2] for entry in results])
    print(f"[prove] {len(results)} discs: error mean {errors.mean():.1f} px, median {numpy.median(errors):.1f} px, "
          f"max {errors.max():.1f} px")
    near = depths < 1.5
    if near.any():
        print(f"[prove] discs nearer than 1.5 m ({int(near.sum())}): mean error {errors[near].mean():.1f} px; "
              f"further: {errors[~near].mean() if (~near).any() else float('nan'):.1f} px")
    os.makedirs(config.debug_image_directory, exist_ok=True)
    path = os.path.join(config.debug_image_directory, "kinect_proof.png")
    cv2.imwrite(path, overview)
    print(f"[prove] overview saved to {path}")


if __name__ == "__main__":
    main()
