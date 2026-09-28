"""Static geometry test of the Kinect projector calibration: project at a point the camera sees.

The debug window shows the live Kinect colour view. Click a point (a spot on the
wall, the chair, your own nose while still): its 3D point comes from the depth
frame, the calibration turns it into a projector pixel, and a red disc is
projected there. The Kinect then sees where the disc really landed; the window
draws the target (green), the landing (red cross) and the distance between them
in camera pixels and in millimetres at that depth.

--grid steps through fixed camera points automatically and prints the landing
error for each (unattended proof on static surfaces).

    .venv\\Scripts\\python.exe kinect_pointer.py            # click to aim
    .venv\\Scripts\\python.exe kinect_pointer.py --grid     # automatic sweep

Keys: q / Esc quit, s snapshot.
"""

import argparse
import math
import os
import time

import cv2
import numpy

import config
from displays import pick_projector_monitor
from kinect_camera import open_camera
from panel_link import PanelLink
from recorder import Recorder
from projector_view import (beam_coverage_mask, depth_image, draw_projector_marker, publish_cloud,
                            render_projector_view, tint_beam_coverage)
from projector import ProjectorWindow
from dot_detector import detect_single_dot
from projector_calibration import ProjectorCalibration

DEBUG_WINDOW_NAME = "kinect pointer (click to aim)"
PROJECTOR_VIEW_WINDOW_NAME = "projector view"
KINECT_COLOUR_FOCAL_PIXELS = 1064.0     # measured in this project's self-test


def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--grid", action="store_true", help="sweep fixed camera points instead of clicking")
    argument_parser.add_argument("--radius", type=int, default=30, help="disc radius in projector pixels")
    argument_parser.add_argument("--calibration", type=str, default=config.kinect_calibration_file_path)
    argument_parser.add_argument("--seconds", type=float, default=None)
    argument_parser.add_argument("--panel", action="store_true",
                                 help="driven by web_panel.py: no debug window, view and commands go through files")
    arguments = argument_parser.parse_args()
    panel = PanelLink(config.debug_image_directory, arguments.panel)
    recorder = Recorder()          # raw colour frames on request from the panel, no overlays
    show_window = not arguments.panel

    calibration = ProjectorCalibration.load(arguments.calibration)
    focal = calibration.camera_matrix[0, 0]
    print(f"[pointer] {calibration.describe()}")
    print(f"[pointer] projector field of view {2 * math.degrees(math.atan(calibration.projector_width / (2 * focal))):.1f} deg "
          f"wide; Kinect colour {2 * math.degrees(math.atan(1920 / (2 * KINECT_COLOUR_FOCAL_PIXELS))):.1f} deg")
    monitor, is_second_screen = pick_projector_monitor(
        config.projector_monitor_index, config.projector_fallback_width_pixels, config.projector_fallback_height_pixels)
    camera = open_camera(config)
    projector = ProjectorWindow(monitor, fullscreen=is_second_screen)
    canvas = numpy.zeros((projector.height, projector.width, 3), dtype=numpy.uint8)
    if show_window:
        cv2.namedWindow(DEBUG_WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(DEBUG_WINDOW_NAME, 960, 540)
    os.makedirs(config.debug_image_directory, exist_ok=True)
    view_scale = camera.frame_width / 960.0

    state = {"target": None, "target_3d": None, "projector_pixel": None, "aimed_at": 0.0, "frames_after_settle": 0,
             "layer": config.camera_view_layer,
             "reference_gray": None}

    def on_mouse(event, x, y, flags, user_data):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["target"] = (x * view_scale, y * view_scale)
            state["target_3d"] = None
            state["frames_after_settle"] = 0

    if show_window:
        cv2.setMouseCallback(DEBUG_WINDOW_NAME, on_mouse)

    grid_targets = []
    if arguments.grid:
        # Candidate camera points everywhere; keep those with depth whose projection
        # falls inside the projector image (the beam is much narrower than the Kinect view).
        camera.frame_after(time.perf_counter() + 0.5, timeout_seconds=10.0)
        first_timestamp, _ = camera.latest()
        first_depth = camera.depth_for(first_timestamp)[1]
        candidates = [(column_fraction * camera.frame_width, row_fraction * camera.frame_height)
                      for row_fraction in numpy.linspace(0.15, 0.85, 8)
                      for column_fraction in numpy.linspace(0.1, 0.9, 12)]
        candidate_points = camera.points_3d_at(first_depth, candidates)
        for candidate, point_3d in zip(candidates, candidate_points):
            if not numpy.isfinite(point_3d[2]):
                continue
            px, py = calibration.project(point_3d)[0]
            if 2 * arguments.radius <= px < projector.width - 2 * arguments.radius \
                    and 2 * arguments.radius <= py < projector.height - 2 * arguments.radius:
                grid_targets.append((candidate, float(point_3d[2])))
        # Nearest first: the near surfaces are the ones the wall-dominated scan tells least about.
        grid_targets.sort(key=lambda entry: entry[1])
        print(f"[pointer] {len(grid_targets)} of {len(candidates)} candidate points have depth and lie inside the beam; "
              f"depths {', '.join(f'{depth:.1f}' for _, depth in grid_targets)} m")
        grid_targets = [candidate for candidate, _ in grid_targets]
    grid_index = 0
    grid_results = []
    settle_seconds = (calibration.latency_full_after_seconds or config.latency_full_after_seconds) + 0.3

    start_time = time.perf_counter()
    last_timestamp = 0.0
    snapshot_count = 0
    frame_count = 0
    beam_mask = None
    latest_depth = None
    landing_history = []
    try:
        while True:
            result = camera.latest_after(last_timestamp, timeout_seconds=0.5)
            if result is None:
                # Stream stalled: say so in the window and keep the keys alive.
                stalled_seconds = time.perf_counter() - last_timestamp if last_timestamp else 0.0
                stalled_view = numpy.zeros((540, 960, 3), dtype=numpy.uint8)
                cv2.putText(stalled_view, f"Kinect stream stalled for {stalled_seconds:.0f} s (q to quit)", (10, 270),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                if show_window:
                    cv2.imshow(DEBUG_WINDOW_NAME, stalled_view)
                panel.publish(stalled_view)
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27) or panel.command() == "quit":
                    break
                continue
            timestamp, frame = result
            recorder.write(frame)
            last_timestamp = timestamp
            now = time.perf_counter()

            # Advance on frames seen after the disc settled, not on wall-clock time: a
            # stalled Kinect stream then only delays the sweep instead of skipping points.
            if arguments.grid and (state["target"] is None or state["frames_after_settle"] >= 6):
                if state["target"] is not None:
                    if landing_history:
                        grid_results.append((state["target"], state["target_3d"], landing_history[-1]))
                    elif state["target_3d"] is None:
                        print(f"[pointer]   camera ({state['target'][0]:.0f},{state['target'][1]:.0f}): skipped, no depth")
                    else:
                        print(f"[pointer]   camera ({state['target'][0]:.0f},{state['target'][1]:.0f}) depth "
                              f"{state['target_3d'][2]:.2f} m: disc never seen by the Kinect (dark surface, or a stream stall)")
                if grid_index >= len(grid_targets):
                    break
                state["target"] = grid_targets[grid_index]
                state["target_3d"] = None
                state["aimed_at"] = now
                state["frames_after_settle"] = 0
                grid_index += 1

            status_lines = [f"Kinect {camera.frames_per_second():.0f} fps, {camera.dropped_frame_count} dropped - "
                            "click a point: the projector puts a red disc there (green tint = reachable)"]
            warning = ""
            landing = None
            if state["target"] is not None:
                if state["target_3d"] is None:
                    depth_entry = camera.depth_for(timestamp)
                    point_3d = camera.points_3d_at(depth_entry[1], [state["target"]],
                                                   sample_radius_pixels=2 * config.kinect_depth_sample_radius_pixels)[0]
                    if numpy.isfinite(point_3d[2]):
                        state["target_3d"] = point_3d
                        state["projector_pixel"] = calibration.project(point_3d)[0]
                        state["aimed_at"] = now
                        # This frame still shows no disc: the reference for the differencing.
                        state["reference_gray"] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        landing_history = []
                        canvas[:] = 0
                        px, py = state["projector_pixel"]
                        inside_beam = 0 <= px < projector.width and 0 <= py < projector.height
                        if inside_beam:
                            cv2.circle(canvas, (int(round(px)), int(round(py))), arguments.radius, (0, 0, 255), -1,
                                       cv2.LINE_AA)
                        projector.show(canvas)
                        cv2.waitKey(1)
                        print(f"[pointer] aimed at camera ({state['target'][0]:.0f},{state['target'][1]:.0f}) -> 3D "
                              f"({point_3d[0]:.2f}, {point_3d[1]:.2f}, {point_3d[2]:.2f}) m -> projector ({px:.0f},{py:.0f})"
                              + ("" if inside_beam else
                                 f"  OUTSIDE the projector image by {max(0, -px, px - projector.width):.0f} px horizontally, "
                                 f"{max(0, -py, py - projector.height):.0f} px vertically: the beam does not reach that point"))
                    else:
                        warning = "no depth at the clicked point"
                        print(f"[pointer] aimed at camera ({state['target'][0]:.0f},{state['target'][1]:.0f}): no depth there")
                if state["target_3d"] is not None:
                    point_3d = state["target_3d"]
                    px, py = state["projector_pixel"]
                    inside = 0 <= px < projector.width and 0 <= py < projector.height
                    status_lines = [
                        f"target camera ({state['target'][0]:.0f},{state['target'][1]:.0f}) -> 3D "
                        f"({point_3d[0] * 100:.1f}, {point_3d[1] * 100:.1f}, {point_3d[2] * 100:.1f}) cm",
                        f"-> projector ({px:.0f},{py:.0f}){'' if inside else '  OUTSIDE THE PROJECTOR IMAGE'}",
                    ]
                    if timestamp - state["aimed_at"] > settle_seconds:
                        state["frames_after_settle"] += 1
                        # Where did the disc land? Difference against the frame taken before it was
                        # shown, so a dim disc on a dark surface counts as well as a bright one on a wall.
                        target_x, target_y = state["target"]
                        landing, _ = detect_single_dot(
                            state["reference_gray"], cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                            config.servo_min_peak_difference, 6000, config.servo_opening_radius_pixels,
                            config.dot_max_negative_ratio,
                            accept_rectangle=(target_x - 400, target_y - 400, target_x + 400, target_y + 400))
                        if landing is not None:
                            offset_pixels = float(numpy.hypot(landing[0] - state["target"][0], landing[1] - state["target"][1]))
                            offset_millimetres = offset_pixels * point_3d[2] / KINECT_COLOUR_FOCAL_PIXELS * 1000.0
                            landing_history.append((landing, offset_pixels, offset_millimetres))
                            status_lines.append(f"disc landed {offset_pixels:.1f} camera px = {offset_millimetres:.0f} mm "
                                                f"from the target (dx {landing[0] - state['target'][0]:+.0f}, "
                                                f"dy {landing[1] - state['target'][1]:+.0f} px)")
                        else:
                            status_lines.append("disc not seen by the Kinect")

            # One layer at a time in the left view, chosen from the panel: colour, depth or beam.
            view = cv2.resize(frame, (960, 540), interpolation=cv2.INTER_AREA)
            if state["layer"] == "depth":
                # The depth captured with this very frame, never an older one.
                depth_entry_for_view = camera.depth_for(timestamp)
                if depth_entry_for_view is not None:
                    view = depth_image(depth_entry_for_view[1][::2, ::2])
            elif state["layer"] == "beam" and beam_mask is not None:
                tint_beam_coverage(view, beam_mask, strength=0.45)
            if state["target"] is not None:
                cv2.circle(view, (int(state["target"][0] / view_scale), int(state["target"][1] / view_scale)), 10, (0, 255, 0), 2)
            if landing is not None:
                centre = (int(landing[0] / view_scale), int(landing[1] / view_scale))
                cv2.drawMarker(view, centre, (0, 0, 255), cv2.MARKER_CROSS, 24, 2)
                cv2.line(view, (int(state["target"][0] / view_scale), int(state["target"][1] / view_scale)), centre,
                         (0, 200, 255), 1)
            band = view[0:8 + 22 * len(status_lines), :]
            band[:] = (band * 0.35).astype(numpy.uint8)
            for line_index, line in enumerate(status_lines):
                cv2.putText(view, line, (10, 24 + 22 * line_index), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            if warning:
                cv2.putText(view, warning, (10, 528), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if show_window:
                cv2.imshow(DEBUG_WINDOW_NAME, view)
            panel.publish_state({"tool": "kinect_pointer", "recording": recorder.active, "recording_text": recorder.status_text()})
            if recorder.active:
                cv2.putText(view, recorder.status_text(), (760, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            panel.publish(view)

            # The projector's-eye view: the depth cloud through the calibration, with the aimed pixel marked.
            if panel.projector_view_due() or (show_window and frame_count % 6 == 0):
                depth_entry = camera.depth_for(timestamp)
                if depth_entry is not None:
                    beam_mask = beam_coverage_mask(camera, depth_entry[1], calibration, 960, 540)
                    latest_depth = depth_entry[1]
                    projector_view = render_projector_view(camera, depth_entry[1], frame, calibration, 960, 540,
                                                           colour_by_depth=config.projector_view_colour_by_depth)
                    if state["target_3d"] is not None:
                        draw_projector_marker(projector_view, state["projector_pixel"], calibration)
                    cv2.putText(projector_view, "projector's-eye view" + (" coloured by depth (blue 0.4 m .. red 3 m, grey = no depth)"
                                                                          if config.projector_view_colour_by_depth
                                                                          else " (depth cloud through the calibration)"),
                                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    if show_window:
                        cv2.imshow(PROJECTOR_VIEW_WINDOW_NAME, projector_view)
                    panel.publish_projector_view(projector_view)
                    publish_cloud(panel, camera, depth_entry[1], frame)
            frame_count += 1

            key = cv2.waitKey(1) & 0xFF
            panel_command = panel.command()
            if panel_command == "record":
                if recorder.active:
                    recorder.stop()
                else:
                    recorder.start(camera.frame_width, camera.frame_height, camera.frames_per_second())
            if panel_command == "quit":
                key = ord("q")
            elif panel_command == "snapshot":
                key = ord("s")
            elif panel_command is not None and panel_command.startswith("aim "):
                # "aim <x fraction> <y fraction>" from a click on the live image in the browser.
                parts = panel_command.split()
                if len(parts) == 3:
                    state["target"] = (float(parts[1]) * camera.frame_width, float(parts[2]) * camera.frame_height)
                    state["target_3d"] = None
                    state["frames_after_settle"] = 0
            elif panel_command is not None and panel_command.startswith("layer "):
                layer = panel_command.split(maxsplit=1)[1].strip()
                if layer in ("colour", "depth", "beam"):
                    state["layer"] = layer
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                path = os.path.join(config.debug_image_directory, f"kinect_pointer_{snapshot_count:03d}.png")
                config.kinect_save_diagnostic_images and cv2.imwrite(path, view)
                snapshot_count += 1
                print(f"[pointer] snapshot {path}")
            if arguments.grid and landing is not None and len(landing_history) == 3:
                path = os.path.join(config.debug_image_directory, f"kinect_pointer_grid_{grid_index - 1:02d}.png")
                config.kinect_save_diagnostic_images and cv2.imwrite(path, view)
            if arguments.seconds is not None and now - start_time > arguments.seconds:
                break
    finally:
        recorder.stop()
        projector.show(canvas * 0)
        cv2.waitKey(1)
        camera.release()
        cv2.destroyAllWindows()

    if grid_results:
        print("[pointer] grid sweep:")
        for target, point_3d, (landing, offset_pixels, offset_millimetres) in grid_results:
            print(f"[pointer]   camera ({target[0]:.0f},{target[1]:.0f}) depth {point_3d[2]:.2f} m: "
                  f"landed {offset_pixels:.1f} px = {offset_millimetres:.0f} mm off")
        offsets = numpy.array([entry[2][2] for entry in grid_results])
        print(f"[pointer] {len(grid_results)} points measured: median {numpy.median(offsets):.0f} mm, max {offsets.max():.0f} mm")


if __name__ == "__main__":
    main()
