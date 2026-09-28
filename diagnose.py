"""Capture what the camera sees for a few known projector images and save them.

Run this when the bootstrap or servo finds nothing. It writes to ./diagnostics:
  black.png, white.png, grid.png, dot.png            camera frames
  white_minus_black.png, grid_minus_black.png,
  dot_minus_black.png                                difference images
and prints brightness statistics for each.
"""

import os
import time

import cv2
import numpy

import config
from camera import Camera
from displays import pick_projector_monitor
from projector import ProjectorWindow

OUTPUT_DIRECTORY = "diagnostics"
SETTLE_FRAMES = 12


def capture_after(projector, camera, image, label):
    projector.show(image)
    cv2.waitKey(1)
    shown_at = time.perf_counter()
    frame = None
    for _ in range(SETTLE_FRAMES):
        cv2.waitKey(1)
        result = camera.frame_after(shown_at, timeout_seconds=1.0)
        if result is not None:
            shown_at, frame = result
    cv2.imwrite(os.path.join(OUTPUT_DIRECTORY, f"{label}.png"), frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    print(f"[diagnose] {label:6s}: camera mean brightness {gray.mean():6.1f}  "
          f"min {gray.min():3d}  max {gray.max():3d}")
    return gray


def report_difference(off_gray, on_gray, label):
    difference = cv2.GaussianBlur(cv2.subtract(on_gray, off_gray), (5, 5), 0)
    cv2.imwrite(os.path.join(OUTPUT_DIRECTORY, f"{label}_minus_black.png"),
                cv2.applyColorMap(difference, cv2.COLORMAP_INFERNO))
    peak_row, peak_column = numpy.unravel_index(int(difference.argmax()), difference.shape)
    bright_pixel_count = int((difference > config.bootstrap_min_peak_difference).sum())
    print(f"[diagnose] {label:6s} - black: peak {int(difference.max()):3d} at camera "
          f"({peak_column},{peak_row})  median {float(numpy.median(difference)):.1f}  "
          f"pixels above {config.bootstrap_min_peak_difference}: {bright_pixel_count}")
    return difference


def main():
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    monitor, is_second_screen = pick_projector_monitor(
        config.projector_monitor_index,
        config.projector_fallback_width_pixels,
        config.projector_fallback_height_pixels,
    )
    camera = Camera(config)
    projector = ProjectorWindow(monitor, fullscreen=is_second_screen)
    cv2.namedWindow("diagnose", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("diagnose", 960, 540)

    width, height = projector.width, projector.height
    black = numpy.zeros((height, width, 3), dtype=numpy.uint8)
    white = numpy.full((height, width, 3), 255, dtype=numpy.uint8)

    grid = black.copy()
    margin = config.bootstrap_grid_margin_fraction
    for projector_y in numpy.linspace(margin * height, (1 - margin) * height, config.bootstrap_grid_rows):
        for projector_x in numpy.linspace(margin * width, (1 - margin) * width, config.bootstrap_grid_columns):
            cv2.circle(grid, (int(projector_x), int(projector_y)), config.bootstrap_dot_radius_pixels,
                       (255, 255, 255), -1, cv2.LINE_AA)

    dot = black.copy()
    cv2.circle(dot, (width // 2, height // 2), config.bootstrap_dot_radius_pixels, (255, 255, 255), -1, cv2.LINE_AA)

    print("[diagnose] keep still; capturing 4 projector states ...")
    black_gray = capture_after(projector, camera, black, "black")
    black_gray_again = capture_after(projector, camera, black, "black2")
    white_gray = capture_after(projector, camera, white, "white")
    grid_gray = capture_after(projector, camera, grid, "grid")
    dot_gray = capture_after(projector, camera, dot, "dot")

    print()
    report_difference(black_gray, black_gray_again, "black2")
    white_difference = report_difference(black_gray, white_gray, "white")
    report_difference(black_gray, grid_gray, "grid")
    dot_difference = report_difference(black_gray, dot_gray, "dot")

    print()
    lit_fraction = float((white_difference > 10).mean())
    print(f"[diagnose] fraction of camera frame lit by full white: {lit_fraction:.2f} "
          "(0 = camera does not see the projection at all)")
    print(f"[diagnose] images saved in ./{OUTPUT_DIRECTORY}")

    projector.show(black)
    side_by_side = numpy.hstack([
        cv2.applyColorMap(white_difference, cv2.COLORMAP_INFERNO),
        cv2.applyColorMap(dot_difference, cv2.COLORMAP_INFERNO),
    ])
    cv2.putText(side_by_side, "white - black            |            dot - black",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.imshow("diagnose", side_by_side)
    cv2.waitKey(3000)
    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
