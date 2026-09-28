"""Kinect-only check: the live colour view with frame rate and stall state, and the
depth image beside it. No projector, no tracking.

    .venv\\Scripts\\python.exe kinect_view.py            # OpenCV windows
    .venv\\Scripts\\python.exe kinect_view.py --panel    # through web_panel.py

Keys: q / Esc quit.
"""

import argparse
import time

import cv2
import numpy

import config
from kinect_camera import open_camera
from panel_link import PanelLink
from recorder import Recorder
from projector_view import depth_image, publish_cloud

COLOUR_WINDOW_NAME = "kinect colour"
DEPTH_WINDOW_NAME = "kinect depth"


def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--panel", action="store_true", help="driven by web_panel.py")
    arguments = argument_parser.parse_args()
    panel = PanelLink(config.debug_image_directory, arguments.panel)
    recorder = Recorder()          # raw colour frames on request from the panel, no overlays
    show_window = not arguments.panel

    camera = open_camera(config)
    if show_window:
        cv2.namedWindow(COLOUR_WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(COLOUR_WINDOW_NAME, 960, 540)
        cv2.namedWindow(DEPTH_WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(DEPTH_WINDOW_NAME, 512, 424)
    started = time.perf_counter()
    layer = "colour"                    # left pane: colour or depth, switched from the panel
    last_timestamp = 0.0
    stall_count = 0
    stalled_since = None
    last_report = started
    try:
        while True:
            result = camera.latest_after(last_timestamp, timeout_seconds=0.5)
            if result is None:
                if stalled_since is None:
                    stalled_since = time.perf_counter()
                    stall_count += 1
                    print(f"[view] stall {stall_count} begins at t={stalled_since - started:.1f} s", flush=True)
                stalled_view = numpy.zeros((540, 960, 3), dtype=numpy.uint8)
                cv2.putText(stalled_view, f"Kinect stream stalled for {time.perf_counter() - stalled_since:.0f} s "
                                          f"(stall {stall_count})", (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                if show_window:
                    cv2.imshow(COLOUR_WINDOW_NAME, stalled_view)
                panel.publish(stalled_view)
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27) or panel.command() == "quit":
                    break
                continue
            timestamp, frame = result
            recorder.write(frame)
            if stalled_since is not None:
                print(f"[view] stream resumes after {timestamp - stalled_since:.1f} s", flush=True)
                stalled_since = None
            last_timestamp = timestamp

            if layer == "depth":
                depth_entry_for_view = camera.depth_for(timestamp)
                view = (depth_image(depth_entry_for_view[1][::2, ::2]) if depth_entry_for_view is not None
                        else numpy.zeros((540, 960, 3), dtype=numpy.uint8))
            else:
                view = cv2.resize(frame, (960, 540), interpolation=cv2.INTER_AREA)
            band = view[0:56, :]
            band[:] = (band * 0.35).astype(numpy.uint8)
            cv2.putText(view, f"Kinect {camera.frames_per_second():.0f} fps, {camera.dropped_frame_count} dropped, "
                              f"{stall_count} stalls, exposure {camera.exposure_seconds * 1000 if camera.exposure_seconds else 0:.0f} ms",
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(view, f"running {time.perf_counter() - started:.0f} s, {camera.captured_frame_count} frames captured",
                        (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            if show_window:
                cv2.imshow(COLOUR_WINDOW_NAME, view)
            panel.publish_state({"tool": "kinect_view", "recording": recorder.active, "recording_text": recorder.status_text()})
            if recorder.active:
                cv2.putText(view, recorder.status_text(), (760, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            panel.publish(view)

            if panel.projector_view_due() or show_window:
                depth_entry = camera.depth_for(timestamp)
                if depth_entry is not None:
                    depth_millimetres = depth_entry[1]
                    depth_view = cv2.applyColorMap(numpy.clip(depth_millimetres / 4500.0 * 255.0, 0, 255).astype(numpy.uint8),
                                                   cv2.COLORMAP_JET)
                    depth_view[depth_millimetres == 0] = 0
                    depth_view = cv2.resize(depth_view, (960, 540), interpolation=cv2.INTER_NEAREST)
                    if hasattr(camera, "map_depth_to_colour_points"):
                        publish_cloud(panel, camera, depth_millimetres, frame)
                    valid_fraction = float((depth_millimetres > 0).mean())
                    cv2.putText(depth_view, f"depth: {valid_fraction * 100:.0f}% of pixels valid, blue near .. red 4.5 m",
                                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    if show_window:
                        cv2.imshow(DEPTH_WINDOW_NAME, depth_view)
                    panel.publish_projector_view(depth_view)

            if time.perf_counter() - last_report >= 10.0:
                last_report = time.perf_counter()
                print(f"[view] t={last_report - started:.0f} s: {camera.frames_per_second():.0f} fps now, "
                      f"{camera.captured_frame_count} frames, {stall_count} stalls", flush=True)
            key = cv2.waitKey(1) & 0xFF
            panel_command = panel.command()
            if panel_command == "record":
                if recorder.active:
                    recorder.stop()
                else:
                    recorder.start(camera.frame_width, camera.frame_height, camera.frames_per_second())
            if panel_command is not None and panel_command.startswith("layer "):
                requested = panel_command.split(maxsplit=1)[1].strip()
                if requested in ("colour", "depth"):
                    layer = requested
            if key in (ord("q"), 27) or panel_command == "quit":
                break
    finally:
        recorder.stop()
        print(camera.capture_report(), flush=True)
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
