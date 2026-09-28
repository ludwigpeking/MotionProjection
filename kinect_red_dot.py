"""Project a red dot onto a moving face, with the depth measured by the Kinect v2.

Per Kinect frame: MediaPipe finds the landmark in the colour image, the depth
frame captured in the same instant gives the landmark's metric 3D point, a
one-euro filter smooths it and predicts it ahead by the measured latency, and
the projector calibration (kinect_calibrate.py) turns it into a projector pixel.
No homography and no servo: the geometry was measured once, in 3D.

The red disc is also looked for in the Kinect image, so the debug view shows
how far it lands from the landmark (in camera pixels); while the head is still
that distance is the pure geometric error.

    .venv\\Scripts\\python.exe kinect_red_dot.py                 # nose tip
    .venv\\Scripts\\python.exe kinect_red_dot.py --landmark 9    # between the brows
    .venv\\Scripts\\python.exe kinect_red_dot.py --seconds 30    # unattended run

Keys: q / Esc quit, [ and ] prediction lead -/+ 20 ms, - and + dot radius,
p toggle prediction, s save a snapshot of the debug view.
"""

import argparse
import os
import time

import cv2
import numpy

import config
from displays import pick_projector_monitor
from face_tracker import FaceTracker, OneEuroFilter
from kinect_camera import open_camera
from panel_link import PanelLink
from recorder import Recorder
from projector_view import draw_projector_marker, publish_cloud, render_projector_view
from projector import ProjectorWindow
from projector_calibration import ProjectorCalibration
from red_disc import DOT_COLOURS_BGR, measure_projected_dot

DEBUG_WINDOW_NAME = "kinect red dot"


def compose_debug_view(frame, landmark_pixel, disc_pixel, status_lines, warning):
    view = cv2.resize(frame, (960, 540), interpolation=cv2.INTER_AREA)
    scale = 960.0 / frame.shape[1]
    if landmark_pixel is not None:
        centre = (int(landmark_pixel[0] * scale), int(landmark_pixel[1] * scale))
        cv2.circle(view, centre, 9, (0, 255, 0), 2)
    if disc_pixel is not None:
        centre = (int(disc_pixel[0] * scale), int(disc_pixel[1] * scale))
        cv2.drawMarker(view, centre, (0, 0, 255), cv2.MARKER_CROSS, 18, 2)
        if landmark_pixel is not None:
            cv2.line(view, (int(landmark_pixel[0] * scale), int(landmark_pixel[1] * scale)), centre, (0, 200, 255), 1)
    # Dark bands behind the text so it stays readable over a bright window.
    band = view[0:8 + 22 * len(status_lines), :]
    band[:] = (band * 0.35).astype(numpy.uint8)
    for line_index, line in enumerate(status_lines):
        cv2.putText(view, line, (10, 24 + 22 * line_index), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    if warning:
        bottom_band = view[500:540, :]
        bottom_band[:] = (bottom_band * 0.35).astype(numpy.uint8)
        cv2.putText(view, warning, (10, 528), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return view


def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--seconds", type=float, default=None, help="quit by itself after this long")
    argument_parser.add_argument("--landmark", type=int, default=config.kinect_red_dot_landmark_index,
                                 help="MediaPipe landmark index to put the dot on (4 = nose tip)")
    argument_parser.add_argument("--radius", type=int, default=config.kinect_red_dot_radius_pixels,
                                 help="dot radius in projector pixels")
    argument_parser.add_argument("--lead", type=float, default=None,
                                 help="prediction lead in seconds (default: from the calibration file)")
    argument_parser.add_argument("--calibration", type=str, default=config.kinect_calibration_file_path)
    argument_parser.add_argument("--no-predict", action="store_true", help="no motion prediction: dot at the smoothed point")
    argument_parser.add_argument("--panel", action="store_true",
                                 help="driven by web_panel.py: no debug window, view and commands go through files")
    arguments = argument_parser.parse_args()
    panel = PanelLink(config.debug_image_directory, arguments.panel)
    recorder = Recorder()          # raw colour frames on request from the panel, no overlays
    show_window = not arguments.panel

    if not os.path.exists(arguments.calibration):
        raise SystemExit(f"[red-dot] {arguments.calibration} not found: run kinect_calibrate.py first")
    calibration = ProjectorCalibration.load(arguments.calibration)
    print(f"[red-dot] calibration from {calibration.calibrated_at}: rms {calibration.rms_pixels:.2f} px, "
          f"held-out rms {calibration.holdout_rms_pixels:.2f} px, {calibration.describe()}")
    prediction_lead_seconds = arguments.lead
    if prediction_lead_seconds is None:
        prediction_lead_seconds = calibration.prediction_lead_seconds or config.prediction_lead_seconds
    predict = not arguments.no_predict
    dot_radius = arguments.radius
    landmark_index = arguments.landmark

    monitor, is_second_screen = pick_projector_monitor(
        config.projector_monitor_index, config.projector_fallback_width_pixels, config.projector_fallback_height_pixels)
    if monitor.width != calibration.projector_width or monitor.height != calibration.projector_height:
        print(f"[red-dot] WARNING: projector is {monitor.width}x{monitor.height} but the calibration was made at "
              f"{calibration.projector_width}x{calibration.projector_height}")
    camera = open_camera(config)
    projector = ProjectorWindow(monitor, fullscreen=is_second_screen)
    face_tracker = FaceTracker(config, camera.frame_width, camera.frame_height)
    smoothing_filter = OneEuroFilter(config.smoothing_min_cutoff_hertz, config.smoothing_speed_coefficient,
                                     config.smoothing_derivative_cutoff_hertz)
    if show_window:
        cv2.namedWindow(DEBUG_WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(DEBUG_WINDOW_NAME, 960, 540)
    os.makedirs(config.debug_image_directory, exist_ok=True)
    panel_key_for_command = {"quit": ord("q"), "lead+": ord("]"), "lead-": ord("["), "radius+": ord("+"),
                             "radius-": ord("-"), "predict": ord("p"), "snapshot": ord("s")}

    canvas = numpy.zeros((projector.height, projector.width, 3), dtype=numpy.uint8)
    start_time = time.perf_counter()
    last_timestamp = 0.0
    last_point_3d = None
    frame_count = 0
    face_frame_count = 0
    depth_frame_count = 0
    disc_offsets_still = []
    disc_offsets_all = []
    still_offsets_millimetres = []           # (right, down) of the dot relative to the landmark, head still
    dot_colour_bgr = DOT_COLOURS_BGR[config.kinect_dot_colour]
    offsets_path = os.path.join(config.debug_image_directory, config.kinect_dot_offsets_file_name)
    offsets_file = open(offsets_path, "w")
    offsets_file.write("time_seconds,landmark_x_metres,landmark_y_metres,landmark_z_metres,speed_metres_per_second,"
                       "prediction_gain,projector_x_pixels,projector_y_pixels,dot_drawn,strongest_colour_contrast,"
                       "landmark_column_pixels,landmark_row_pixels,"
                       "dot_column_pixels,dot_row_pixels,offset_right_millimetres,offset_down_millimetres,"
                       "evidence_frame_index,evidence_left_pixels,evidence_top_pixels\n")
    # Raw pixels around the landmark for every logged frame in which the dot is drawn: what the
    # camera really saw, without any overlay, to measure the landing offline.
    evidence_writer = None
    evidence_frame_count = 0
    evidence_half_width = config.kinect_dot_evidence_half_width_pixels
    evidence_half_height = config.kinect_dot_evidence_half_height_pixels
    if config.kinect_dot_evidence_file_name:
        evidence_path = os.path.join(config.debug_image_directory, config.kinect_dot_evidence_file_name)
        evidence_writer = cv2.VideoWriter(evidence_path, cv2.VideoWriter_fourcc(*"MJPG"), 10.0,
                                          (2 * evidence_half_width, 2 * evidence_half_height))
    frame_times = []
    snapshot_count = 0
    last_snapshot_time = start_time
    saved_snapshots = []
    # Per-second log: does the stream stall when the face is lost, or when the loop is slow?
    log_path = os.path.join(config.debug_image_directory, "kinect_red_dot_log.csv")
    log_file = open(log_path, "w")
    log_file.write("second,kinect_frames_captured,kinect_frames_dropped,loops,face_found,disc_seen,loop_ms_mean,loop_ms_max\n")
    log_second_started = start_time
    log_captured_before = camera.captured_frame_count
    log_loops = 0
    log_face = 0
    log_disc = 0
    log_loop_times = []

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
            loop_started = time.perf_counter()
            frame_count += 1

            landmarks = face_tracker.detect_raw(frame, timestamp)
            landmark_pixel = None
            point_3d = None
            projector_pixel = None
            disc_pixel = None
            speed = 0.0
            prediction_gain = 0.0
            offset_millimetres = None
            strongest_contrast = 0.0
            warning = ""
            depth_found = False
            if landmarks is not None:
                face_frame_count += 1
                landmark_pixel = landmarks[landmark_index]
                depth_entry = camera.depth_for(timestamp)
                if depth_entry is not None:
                    point_3d = camera.points_3d_at(depth_entry[1], [landmark_pixel])[0]
                    if not numpy.isfinite(point_3d[2]):
                        point_3d = camera.points_3d_at(depth_entry[1], [landmark_pixel],
                                                       sample_radius_pixels=3 * config.kinect_depth_sample_radius_pixels)[0]
                    depth_found = bool(numpy.isfinite(point_3d[2]))
                if depth_found:
                    depth_frame_count += 1
                    last_point_3d = point_3d
                elif last_point_3d is not None:
                    point_3d = last_point_3d
                    warning = "no Kinect depth at the landmark; holding the last 3D point"
                else:
                    point_3d = None
                    warning = "no Kinect depth at the landmark yet"
                if point_3d is not None:
                    smoothed, velocity = smoothing_filter.update(point_3d.reshape(1, 3), timestamp)
                    speed = float(numpy.linalg.norm(velocity))
                    # Dead band: no prediction at rest (velocity noise x lead = wobble), full lead when moving.
                    prediction_gain = float(numpy.clip(
                        (speed - config.prediction_deadband_metres_per_second)
                        / max(1e-6, config.prediction_full_metres_per_second - config.prediction_deadband_metres_per_second),
                        0.0, 1.0))
                    aimed = smoothed + velocity * (prediction_lead_seconds * prediction_gain) if predict else smoothed
                    if calibration.depths(aimed)[0] > 0.0:
                        projector_pixel = calibration.project(aimed)[0]
                disc_pixel, strongest_contrast = measure_projected_dot(
                    frame, landmark_pixel, config.kinect_red_dot_search_radius_pixels,
                    config.kinect_red_dot_min_area_pixels, config.kinect_dot_colour)
                if point_3d is not None and projector_pixel is not None:
                    # Every frame with a face is logged, dot seen or not: a run without detections then
                    # still says whether the dot was drawn at all and how faint it looked to the camera.
                    dot_drawn = (0 <= projector_pixel[0] < projector.width and 0 <= projector_pixel[1] < projector.height)
                    dot_columns = ("nan,nan,nan,nan" if disc_pixel is None else
                                   f"{disc_pixel[0]:.1f},{disc_pixel[1]:.1f},"
                                   f"{(disc_pixel[0] - landmark_pixel[0]) * point_3d[2] / camera.colour_focal_x * 1000.0:.1f},"
                                   f"{(disc_pixel[1] - landmark_pixel[1]) * point_3d[2] / camera.colour_focal_x * 1000.0:.1f}")
                    evidence_columns = "-1,0,0"
                    if evidence_writer is not None and dot_drawn:
                        evidence_left = int(numpy.clip(landmark_pixel[0] - evidence_half_width, 0,
                                                       frame.shape[1] - 2 * evidence_half_width))
                        evidence_top = int(numpy.clip(landmark_pixel[1] - evidence_half_height, 0,
                                                      frame.shape[0] - 2 * evidence_half_height))
                        evidence_writer.write(numpy.ascontiguousarray(
                            frame[evidence_top:evidence_top + 2 * evidence_half_height,
                                  evidence_left:evidence_left + 2 * evidence_half_width]))
                        evidence_columns = f"{evidence_frame_count},{evidence_left},{evidence_top}"
                        evidence_frame_count += 1
                    offsets_file.write(
                        f"{timestamp - start_time:.3f},{point_3d[0]:.4f},{point_3d[1]:.4f},{point_3d[2]:.4f},{speed:.4f},"
                        f"{prediction_gain:.2f},{projector_pixel[0]:.1f},{projector_pixel[1]:.1f},{int(dot_drawn)},"
                        f"{strongest_contrast:.0f},{landmark_pixel[0]:.1f},{landmark_pixel[1]:.1f},{dot_columns},"
                        f"{evidence_columns}\n")
                    offsets_file.flush()
                if disc_pixel is not None:
                    offset = float(numpy.hypot(disc_pixel[0] - landmark_pixel[0], disc_pixel[1] - landmark_pixel[1]))
                    disc_offsets_all.append(offset)
                    if speed < config.kinect_red_dot_still_speed_metres_per_second:
                        disc_offsets_still.append(offset)
                    if point_3d is not None and projector_pixel is not None:
                        # Camera pixels to millimetres at the landmark's distance: right and down as the camera sees them.
                        millimetres_per_pixel = point_3d[2] / camera.colour_focal_x * 1000.0
                        offset_millimetres = ((disc_pixel[0] - landmark_pixel[0]) * millimetres_per_pixel,
                                              (disc_pixel[1] - landmark_pixel[1]) * millimetres_per_pixel)
                        if depth_found and speed < config.kinect_red_dot_still_speed_metres_per_second:
                            still_offsets_millimetres.append(offset_millimetres)
            else:
                smoothing_filter.reset()
                warning = "no face"

            canvas[:] = 0
            if projector_pixel is not None:
                x, y = projector_pixel
                if -dot_radius <= x < projector.width + dot_radius and -dot_radius <= y < projector.height + dot_radius:
                    cv2.circle(canvas, (int(round(x)), int(round(y))), dot_radius, dot_colour_bgr, -1, cv2.LINE_AA)
                else:
                    warning = "landmark outside the projector image"
            projector.show(canvas)

            loop_seconds = time.perf_counter() - loop_started
            frame_times.append(loop_seconds)
            if len(frame_times) > 60:
                frame_times.pop(0)
            elapsed = time.perf_counter() - start_time
            log_loops += 1
            log_face += int(landmarks is not None)
            log_disc += int(disc_pixel is not None)
            log_loop_times.append(loop_seconds)
            if time.perf_counter() - log_second_started >= 1.0:
                captured_now = camera.captured_frame_count
                log_file.write(f"{int(elapsed)},{captured_now - log_captured_before},{camera.dropped_frame_count},"
                               f"{log_loops},{log_face},{log_disc},{numpy.mean(log_loop_times) * 1000:.0f},"
                               f"{numpy.max(log_loop_times) * 1000:.0f}\n")
                log_file.flush()
                log_second_started = time.perf_counter()
                log_captured_before = captured_now
                log_loops = log_face = log_disc = 0
                log_loop_times = []
            status_lines = [
                f"Kinect {camera.frames_per_second():.0f} fps, loop {numpy.mean(frame_times) * 1000:.0f} ms, "
                f"lead {prediction_lead_seconds * 1000:.0f} ms{'' if predict else ' (off)'}, radius {dot_radius} px",
                (f"landmark {landmark_index}: 3D ({point_3d[0] * 100:.1f}, {point_3d[1] * 100:.1f}, {point_3d[2] * 100:.1f}) cm, "
                 f"speed {speed * 100:.0f} cm/s" if point_3d is not None else f"landmark {landmark_index}: no 3D point"),
                (f"projector ({projector_pixel[0]:.0f}, {projector_pixel[1]:.0f})" if projector_pixel is not None
                 else "projector: nothing drawn"),
                ((f"dot lands {offset_millimetres[0]:+.0f} mm right, {offset_millimetres[1]:+.0f} mm down of the landmark"
                  if offset_millimetres is not None else f"dot seen {disc_offsets_all[-1]:.1f} px from the landmark")
                 if disc_pixel is not None else f"{config.kinect_dot_colour} dot not seen in the Kinect image "
                                                f"(strongest colour contrast {strongest_contrast:.0f}, needs 40)"),
                (f"head still, {len(still_offsets_millimetres)} frames: mean "
                 f"{numpy.mean([offset[0] for offset in still_offsets_millimetres]):+.0f} mm right, "
                 f"{numpy.mean([offset[1] for offset in still_offsets_millimetres]):+.0f} mm down"
                 if still_offsets_millimetres else "head still: no measurement yet (face the Kinect inside the beam and hold still)"),
            ]
            view = compose_debug_view(frame, landmark_pixel, disc_pixel, status_lines, warning)
            if show_window:
                cv2.imshow(DEBUG_WINDOW_NAME, view)
            panel.publish_state({"tool": "red_dot", "lead_milliseconds": round(prediction_lead_seconds * 1000),
                                 "prediction": bool(predict), "radius_pixels": int(dot_radius),
                                 "recording": recorder.active, "recording_text": recorder.status_text()})
            if recorder.active:
                cv2.putText(view, recorder.status_text(), (760, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            panel.publish(view)
            if panel.projector_view_due():
                depth_entry = camera.depth_for(timestamp)
                if depth_entry is not None:
                    projector_view = render_projector_view(camera, depth_entry[1], frame, calibration, 960, 540)
                    if projector_pixel is not None:
                        draw_projector_marker(projector_view, projector_pixel, calibration)
                    cv2.putText(projector_view, "projector's-eye view", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (255, 255, 255), 2)
                    panel.publish_projector_view(projector_view)
                    publish_cloud(panel, camera, depth_entry[1], frame)

            now = time.perf_counter()
            if config.kinect_red_dot_snapshot_period_seconds > 0 and now - last_snapshot_time >= config.kinect_red_dot_snapshot_period_seconds:
                last_snapshot_time = now
                snapshot_path = os.path.join(config.debug_image_directory, f"kinect_red_dot_{snapshot_count:03d}.png")
                cv2.imwrite(snapshot_path, view)
                saved_snapshots.append(snapshot_path)
                snapshot_count += 1

            key = cv2.waitKey(1) & 0xFF
            panel_command = panel.command()
            if panel_command == "record":
                if recorder.active:
                    recorder.stop()
                else:
                    recorder.start(camera.frame_width, camera.frame_height, camera.frames_per_second())
            if panel_command in panel_key_for_command:
                key = panel_key_for_command[panel_command]
            if key in (ord("q"), 27):
                break
            if key == ord("["):
                prediction_lead_seconds = max(0.0, prediction_lead_seconds - 0.02)
            if key == ord("]"):
                prediction_lead_seconds += 0.02
            if key in (ord("-"), ord("_")):
                dot_radius = max(2, dot_radius - 2)
            if key in (ord("+"), ord("=")):
                dot_radius += 2
            if key == ord("p"):
                predict = not predict
            if key == ord("s"):
                snapshot_path = os.path.join(config.debug_image_directory, f"kinect_red_dot_manual_{snapshot_count:03d}.png")
                cv2.imwrite(snapshot_path, view)
                saved_snapshots.append(snapshot_path)
                snapshot_count += 1
            if arguments.seconds is not None and elapsed >= arguments.seconds:
                break
    finally:
        recorder.stop()
        projector.show(canvas * 0)
        cv2.waitKey(1)
        log_file.close()
        offsets_file.close()
        if evidence_writer is not None:
            evidence_writer.release()
        print(camera.capture_report())
        face_tracker.close()
        camera.release()
        cv2.destroyAllWindows()

    print(f"[red-dot] {frame_count} frames, face in {face_frame_count}, Kinect depth at the landmark in {depth_frame_count}")
    print(f"[red-dot] per-second log: {log_path}")
    if still_offsets_millimetres:
        still = numpy.array(still_offsets_millimetres)
        print(f"[red-dot] head still, {len(still)} frames: the dot lands {still[:, 0].mean():+.1f} mm right and "
              f"{still[:, 1].mean():+.1f} mm down of the landmark (spread {still[:, 0].std():.1f} / {still[:, 1].std():.1f} mm)")
    print(f"[red-dot] per-frame offsets: {offsets_path} (analyse with analyse_dot_offsets.py)")
    if disc_offsets_all:
        print(f"[red-dot] disc seen in {len(disc_offsets_all)} frames: mean offset {numpy.mean(disc_offsets_all):.1f} px "
              f"(all), {numpy.mean(disc_offsets_still) if disc_offsets_still else float('nan'):.1f} px "
              f"({len(disc_offsets_still)} still frames)")
    else:
        print("[red-dot] the disc was never seen in the Kinect image")
    if saved_snapshots:
        print(f"[red-dot] snapshots: {', '.join(saved_snapshots)}")


if __name__ == "__main__":
    main()
