"""Project onto the face through its live 3D mesh, with the Kinect's depth.

MediaPipe gives 468 landmarks in the colour image; the Kinect's depth turns each
into a metric 3D point; the projector calibration turns those into projector
pixels; the mesh renderers (face_mesh_render.py, face_mesh_export.py) warp an
image onto the projected mesh. Three modes, switchable from the panel:

  texture   the camera's own picture of the face projected back onto it (feedback)
  paint     a texture painted in the canonical MediaPipe UV layout (export_test/face_texture.png,
            or a Blender-painted copy of it) projected through the live mesh
  wire      the mesh as green lines

    .venv\\Scripts\\python.exe kinect_face_mesh.py [--mode paint] [--paint <png>]

Keys: q / Esc quit, 1 2 3 mode, [ ] lead, s snapshot.
"""

import argparse
import os
import time

import cv2
import numpy

import config
from displays import pick_projector_monitor
from face_mesh_export import load_canonical_topology, load_painted_texture, render_painted_texture
from face_mesh_render import render_face_mesh, render_wireframe
from projector_view import depth_image


def depth_entry_for_view(camera, timestamp):
    return camera.depth_for(timestamp)
from face_tracker import FaceTracker
from kinect_camera import open_camera
from panel_link import PanelLink
from projector import ProjectorWindow
from projector_calibration import ProjectorCalibration
from recorder import Recorder

DEBUG_WINDOW_NAME = "kinect face mesh"
MODES = ("texture", "paint", "wire", "off")     # off = projector blank (compare the face without projection)
MESH_VERTEX_COUNT = 468


class FaceDepthSmoother:
    """The face's distance from the Kinect: the median depth over the landmarks that have
    one, smoothed over time. A sudden jump is followed only when it persists, so a frame
    with a bad depth patch does not move the whole mesh."""

    def __init__(self, smoothing_factor, jump_metres, jump_frames):
        self.smoothing_factor = smoothing_factor
        self.jump_metres = jump_metres
        self.jump_frames = jump_frames
        self.depth = None
        self.pending_jump_count = 0

    def update(self, measured_depth):
        if measured_depth is None:
            return self.depth
        if self.depth is None:
            self.depth = measured_depth
        elif abs(measured_depth - self.depth) > self.jump_metres:
            self.pending_jump_count += 1
            if self.pending_jump_count >= self.jump_frames:
                self.depth = measured_depth
                self.pending_jump_count = 0
        else:
            self.pending_jump_count = 0
            self.depth += self.smoothing_factor * (measured_depth - self.depth)
        return self.depth


def measure_face_depth(camera, depth_millimetres, landmark_pixels):
    """Median Kinect depth over the landmarks in metres, how many landmarks had depth, and
    every landmark's own Kinect depth (NaN where the depth image has no value)."""
    points = camera.points_3d_at(depth_millimetres, landmark_pixels)
    valid = numpy.isfinite(points[:, 2])
    if valid.sum() < 20:
        return None, int(valid.sum()), points[:, 2]
    return float(numpy.median(points[valid, 2])), int(valid.sum()), points[:, 2]


FACE_RELIEF_LIMIT_METRES = 0.15         # no point of a face is further than this from the face's median depth


class ReliefScaleEstimator:
    """How deep the face really is, relative to MediaPipe's guess. MediaPipe's landmark z
    is a learned relative depth whose scale is only roughly that of x, so the mesh's
    nose-to-cheek relief can be off by tens of percent, which a projector looking from the
    side turns into a mesh wider than the head. The Kinect measures the real depth at every
    landmark it sees: the factor that best maps the MediaPipe relief onto the Kinect relief
    is fitted each frame (landmarks on the silhouette, whose depth pixel hits the
    background, are rejected) and smoothed over time."""

    def __init__(self, smoothing_factor, limits, outlier_metres, min_landmarks):
        self.smoothing_factor = smoothing_factor
        self.limits = limits
        self.outlier_metres = outlier_metres
        self.min_landmarks = min_landmarks
        self.scale = None
        self.fitted_landmark_count = 0

    def update(self, mediapipe_relief_metres, kinect_depth_metres):
        valid = numpy.isfinite(kinect_depth_metres)
        if valid.sum() < self.min_landmarks:
            return self.scale
        guess = mediapipe_relief_metres[valid] - numpy.median(mediapipe_relief_metres[valid])
        measured = kinect_depth_metres[valid] - numpy.median(kinect_depth_metres[valid])
        on_face = numpy.abs(measured) < FACE_RELIEF_LIMIT_METRES        # background hits at the silhouette are metres off
        guess, measured = guess[on_face], measured[on_face]
        scale = None
        for _ in range(3):                                   # least squares, re-fitted without the outliers
            if len(guess) < self.min_landmarks or numpy.sum(guess * guess) <= 0.0:
                return self.scale
            scale = float(numpy.sum(guess * measured) / numpy.sum(guess * guess))
            keep = numpy.abs(measured - scale * guess) < self.outlier_metres
            guess, measured = guess[keep], measured[keep]
        self.fitted_landmark_count = int(len(guess))
        scale = min(max(scale, self.limits[0]), self.limits[1])
        if self.scale is None:
            self.scale = scale
        else:
            self.scale += self.smoothing_factor * (scale - self.scale)
        return self.scale


def vertex_neighbours(triangles, vertex_count):
    """Row-normalised adjacency matrix (vertex_count x vertex_count): row i averages the
    vertices that share a triangle edge with vertex i. Built once; one matrix product
    per smoothing pass."""
    adjacency = numpy.zeros((vertex_count, vertex_count), dtype=numpy.float64)
    for a, b, c in triangles:
        adjacency[a, b] = adjacency[a, c] = 1.0
        adjacency[b, a] = adjacency[b, c] = 1.0
        adjacency[c, a] = adjacency[c, b] = 1.0
    degrees = adjacency.sum(axis=1)
    isolated = degrees == 0
    adjacency[isolated, isolated] = 1.0             # a vertex without neighbours averages to itself
    degrees[isolated] = 1.0
    return adjacency / degrees[:, None]


def smooth_over_mesh(values, averaging_matrix, passes, weight):
    """Laplacian smoothing of a per-vertex quantity over the mesh: each vertex moves a
    fraction of the way to the mean of its neighbours. Removes single-vertex spikes
    without flattening the face."""
    smoothed = numpy.array(values, dtype=numpy.float64)
    for _ in range(passes):
        smoothed += weight * (averaging_matrix @ smoothed - smoothed)
    return smoothed


def landmarks_to_3d(camera, landmark_pixels, relative_depth_pixels, face_depth_metres, neighbours=None,
                    relief_scale=1.0):
    """Metric 3D for every landmark: MediaPipe's smoothed face shape (x, y in pixels, z
    relative in the same units) placed at the Kinect-measured face distance, its relief
    scaled by relief_scale (the Kinect-measured correction to MediaPipe's depth guess).
    The shape is complete and steady; the distance and the relief come from the depth
    camera."""
    metres_per_pixel = face_depth_metres / camera.colour_focal_x
    relative = relative_depth_pixels - numpy.median(relative_depth_pixels)
    if neighbours is not None:
        relative = smooth_over_mesh(relative, neighbours, config.face_mesh_depth_smoothing_passes,
                                    config.face_mesh_depth_smoothing_weight)
    z = face_depth_metres + relative * metres_per_pixel * relief_scale
    x = (landmark_pixels[:, 0] - camera.colour_centre_x) / camera.colour_focal_x * z
    y = -(landmark_pixels[:, 1] - camera.colour_centre_y) / camera.colour_focal_y * z
    return numpy.column_stack([x, y, z])


def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--mode", choices=MODES, default=config.face_mesh_start_mode)
    argument_parser.add_argument("--paint", type=str, default=config.face_paint_texture_path,
                                 help="PNG in the canonical MediaPipe UV layout for the paint mode")
    argument_parser.add_argument("--lead", type=float, default=None, help="prediction lead in seconds")
    argument_parser.add_argument("--calibration", type=str, default=config.kinect_calibration_file_path)
    argument_parser.add_argument("--panel", action="store_true")
    argument_parser.add_argument("--seconds", type=float, default=None)
    arguments = argument_parser.parse_args()
    panel = PanelLink(config.debug_image_directory, arguments.panel)
    recorder = Recorder()
    show_window = not arguments.panel

    if not os.path.exists(arguments.calibration):
        raise SystemExit(f"[face-mesh] {arguments.calibration} not found: run the calibration first")
    calibration = ProjectorCalibration.load(arguments.calibration)
    projector_position = calibration.position_metres()      # for back-face culling: only triangles facing the projector are drawn
    lead_seconds = arguments.lead if arguments.lead is not None else (calibration.prediction_lead_seconds
                                                                       or config.prediction_lead_seconds)
    mode = arguments.mode
    painted_texture = None
    if os.path.exists(arguments.paint):
        painted_texture = load_painted_texture(arguments.paint)
        print(f"[face-mesh] paint texture {arguments.paint} ({painted_texture.shape[1]}x{painted_texture.shape[0]})")
    elif mode == "paint":
        print(f"[face-mesh] no paint texture at {arguments.paint}; starting in texture mode")
        mode = "texture"

    monitor, is_second_screen = pick_projector_monitor(
        config.projector_monitor_index, config.projector_fallback_width_pixels, config.projector_fallback_height_pixels)
    camera = open_camera(config)
    projector = ProjectorWindow(monitor, fullscreen=is_second_screen)
    face_tracker = FaceTracker(config, camera.frame_width, camera.frame_height)
    face_tracker.prediction_lead_seconds = lead_seconds
    if show_window:
        cv2.namedWindow(DEBUG_WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(DEBUG_WINDOW_NAME, 960, 540)
    black = numpy.zeros((projector.height, projector.width, 3), dtype=numpy.uint8)
    projector.show(black)
    cv2.waitKey(1)

    started = time.perf_counter()
    last_timestamp = 0.0
    frame_count = 0
    _, triangles = load_canonical_topology()        # fixed topology: no facet flicker from re-triangulation
    neighbours = vertex_neighbours(triangles, MESH_VERTEX_COUNT)
    held_mesh = None
    held_mesh_time = 0.0
    last_smoothed_z = None
    predict = True
    layer = "mesh"                       # left pane: mesh (camera + tracked mesh), colour (camera only) or depth
    depth_smoother = FaceDepthSmoother(config.face_depth_smoothing_factor, config.face_depth_jump_metres,
                                       config.face_depth_jump_frames)
    relief_estimator = ReliefScaleEstimator(config.face_relief_scale_smoothing_factor, config.face_relief_scale_limits,
                                            config.face_relief_outlier_metres, config.face_relief_min_landmarks)
    render_times = []
    try:
        while True:
            result = camera.latest_after(last_timestamp, timeout_seconds=0.5)
            if result is None:
                stalled_view = numpy.zeros((540, 960, 3), dtype=numpy.uint8)
                cv2.putText(stalled_view, "Kinect stream stalled (q to quit)", (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (0, 0, 255), 2)
                if show_window:
                    cv2.imshow(DEBUG_WINDOW_NAME, stalled_view)
                panel.publish(stalled_view)
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27) or panel.command() == "quit":
                    break
                continue
            timestamp, frame = result
            recorder.write(frame)
            last_timestamp = timestamp
            frame_count += 1
            loop_started = time.perf_counter()

            smoothed, predicted = face_tracker.process(frame, timestamp)
            if smoothed is not None:
                # The landmarker returns 478 points (468 face + 10 iris); the canonical mesh has 468.
                smoothed = smoothed[:MESH_VERTEX_COUNT]
                predicted = predicted[:MESH_VERTEX_COUNT]
            projector_image = black
            valid_count = 0
            warning = ""
            if smoothed is None and held_mesh is not None and timestamp - held_mesh_time < config.face_mesh_hold_seconds:
                # MediaPipe dropped the face for a moment: keep the last mesh rather than blanking the projection.
                smoothed, predicted, last_smoothed_z = held_mesh
                warning = f"face lost, holding the mesh ({timestamp - held_mesh_time:.1f} s)"
            elif smoothed is not None:
                last_smoothed_z = face_tracker.last_smoothed_z[:MESH_VERTEX_COUNT]
                held_mesh = (smoothed, predicted, last_smoothed_z.copy())
                held_mesh_time = timestamp
            if smoothed is not None:
                depth_entry = camera.depth_for(timestamp)
                measured_depth, valid_count, landmark_depths = (measure_face_depth(camera, depth_entry[1], smoothed)
                                                                if depth_entry is not None else (None, 0, None))
                face_depth = depth_smoother.update(measured_depth)
                if face_depth is None:
                    warning = "no Kinect depth on the face yet"
                else:
                    if landmark_depths is not None:
                        relief_estimator.update(last_smoothed_z * (face_depth / camera.colour_focal_x), landmark_depths)
                    relief_scale = relief_estimator.scale if relief_estimator.scale is not None else 1.0
                    landmarks_3d = landmarks_to_3d(camera, predicted if predict else smoothed, last_smoothed_z, face_depth,
                                                   neighbours, relief_scale)
                    if mode == "off":
                        projector_image = black
                    elif mode == "paint" and painted_texture is not None:
                        projector_image = render_painted_texture(painted_texture, landmarks_3d, calibration,
                                                                 projector.width, projector.height, viewpoint=projector_position)
                    elif mode == "wire":
                        projector_image = render_wireframe(landmarks_3d, calibration, triangles, projector.width,
                                                           projector.height, viewpoint=projector_position)
                    else:
                        projector_image = render_face_mesh(frame, smoothed, landmarks_3d, calibration, triangles,
                                                           projector.width, projector.height,
                                                           config.face_mesh_texture_gain, viewpoint=projector_position)
            else:
                warning = "no face"
            projector.show(projector_image)
            render_times.append(time.perf_counter() - loop_started)
            if len(render_times) > 60:
                render_times.pop(0)

            # The debug view is only built when it will be shown or published: resizing the frame
            # and drawing the mesh on it costs several milliseconds that the projection does not need.
            if show_window or panel.view_due():
                if layer == "depth" and depth_entry_for_view(camera, timestamp) is not None:
                    view = depth_image(depth_entry_for_view(camera, timestamp)[1][::2, ::2])
                else:
                    view = cv2.resize(frame, (960, 540), interpolation=cv2.INTER_AREA)
                if layer == "mesh" and smoothed is not None and triangles is not None:
                    scaled = (smoothed / 2.0).astype(numpy.int32)
                    for triangle in triangles[::3]:
                        cv2.polylines(view, [scaled[triangle].reshape(-1, 1, 2)], True, (0, 200, 0), 1)
                band = view[0:56, :]
                band[:] = (band * 0.35).astype(numpy.uint8)
                cv2.putText(view, f"Kinect {camera.frames_per_second():.0f} fps, render {numpy.mean(render_times) * 1000:.0f} ms, "
                                  f"mode {mode}, lead {lead_seconds * 1000:.0f} ms {'on' if predict else 'OFF'}, "
                                  f"prediction gain {face_tracker.last_prediction_gain:.2f}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 255), 2)
                relief_text = (f", relief x{relief_estimator.scale:.2f} from {relief_estimator.fitted_landmark_count}"
                               if relief_estimator.scale is not None else ", relief x1.00 (not measured)")
                cv2.putText(view, (f"face distance {depth_smoother.depth:.2f} m from {valid_count} landmarks with depth{relief_text}"
                                   if depth_smoother.depth is not None else "face distance: not measured yet")
                                  + (f"   {warning}" if warning else ""),
                            (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255) if not warning else (0, 0, 255), 2)
                if recorder.active:
                    cv2.putText(view, recorder.status_text(), (760, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                if show_window:
                    cv2.imshow(DEBUG_WINDOW_NAME, view)
                panel.publish(view)
            if panel.projector_view_due():
                projector_view = cv2.resize(projector_image, (960, 540), interpolation=cv2.INTER_AREA)
                cv2.putText(projector_view, "projector image", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                panel.publish_projector_view(projector_view)
            panel.publish_state({"tool": "face_mesh", "mode": mode, "lead_milliseconds": round(lead_seconds * 1000),
                                 "prediction": bool(predict), "prediction_gain": round(face_tracker.last_prediction_gain, 2),
                                 "recording": recorder.active,
                                 "recording_text": recorder.status_text(), "paint_available": painted_texture is not None})

            key = cv2.waitKey(1) & 0xFF
            panel_command = panel.command()
            if panel_command == "quit":
                key = ord("q")
            elif panel_command == "lead+":
                key = ord("]")
            elif panel_command is not None and panel_command.startswith("layer "):
                requested = panel_command.split(maxsplit=1)[1].strip()
                if requested in ("mesh", "colour", "depth"):
                    layer = requested
            elif panel_command == "predict":
                predict = not predict
            elif panel_command == "lead-":
                key = ord("[")
            elif panel_command == "record":
                if recorder.active:
                    recorder.stop()
                else:
                    recorder.start(camera.frame_width, camera.frame_height, camera.frames_per_second())
            elif panel_command is not None and panel_command.startswith("mode "):
                requested = panel_command.split(maxsplit=1)[1].strip()
                if requested in MODES and (requested != "paint" or painted_texture is not None):
                    mode = requested
            if key in (ord("q"), 27):
                break
            if key == ord("["):
                lead_seconds = max(0.0, lead_seconds - 0.02)
                face_tracker.prediction_lead_seconds = lead_seconds
            if key == ord("]"):
                lead_seconds += 0.02
                face_tracker.prediction_lead_seconds = lead_seconds
            if key in (ord("1"), ord("2"), ord("3"), ord("4")):
                requested = MODES[key - ord("1")]
                if requested != "paint" or painted_texture is not None:
                    mode = requested
            if arguments.seconds is not None and time.perf_counter() - started > arguments.seconds:
                break
    finally:
        recorder.stop()
        projector.show(black)
        cv2.waitKey(1)
        print(camera.capture_report())
        face_tracker.close()
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
