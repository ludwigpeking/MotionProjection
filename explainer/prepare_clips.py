"""Cut the short clips the explainer shows from the material in ../recordings.

    ..\\.venv\\Scripts\\python.exe prepare_clips.py

Every clip is written to clips/<name>.mp4 at 15 frames per second and 640 pixels wide.
The landmark clip runs the project's own face tracker over a recorded take and draws
the 468 landmarks and the mesh on it.
"""

import os
import sys

import cv2
import numpy

PROJECT_DIRECTORY = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RECORDINGS_DIRECTORY = os.path.join(PROJECT_DIRECTORY, "recordings")
CLIPS_DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "clips")
CLIP_FRAMES_PER_SECOND = 15.0
CLIP_WIDTH_PIXELS = 640

# name: (recording, start seconds, length seconds, crop as (left, top, right, bottom) in source pixels or None)
PLAIN_CLIPS = {
    "result_lit": ("kinect_20260927_165901.avi", 20.0, 13.0, None),
    "result_dark": ("kinect_20260927_001213.avi", 0.0, 4.0, None),
    "blender_paint": ("_ (Unsaved) - Blender 5.1.1 2026-09-26 22-19-14.mp4", 6.0, 30.0, (0, 70, 1580, 1012)),
    "panel_scene": ("Projection Panel - Google Chrome 2026-09-26 21-56-35.mp4", 2.0, 12.0, (38, 622, 1388, 1080)),
}
LANDMARK_CLIP = ("landmarks", "kinect_20260926_214044.avi", 0.5, 10.0)


def open_writer(name, frame_width, frame_height):
    path = os.path.join(CLIPS_DIRECTORY, f"{name}.mp4")
    return cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), CLIP_FRAMES_PER_SECOND, (frame_width, frame_height)), path


def frames_of(recording, start_seconds, length_seconds):
    """Frames sampled at the clip rate from a recording of any frame rate."""
    capture = cv2.VideoCapture(os.path.join(RECORDINGS_DIRECTORY, recording))
    frame_count = int(round(length_seconds * CLIP_FRAMES_PER_SECOND))
    for frame_index in range(frame_count):
        capture.set(cv2.CAP_PROP_POS_MSEC, (start_seconds + frame_index / CLIP_FRAMES_PER_SECOND) * 1000.0)
        read_ok, frame = capture.read()
        if not read_ok:
            break
        yield frame
    capture.release()


def fit_to_clip(frame, crop):
    if crop is not None:
        left, top, right, bottom = crop
        frame = frame[top:bottom, left:right]
    clip_height = int(round(frame.shape[0] * CLIP_WIDTH_PIXELS / frame.shape[1] / 2.0)) * 2
    return cv2.resize(frame, (CLIP_WIDTH_PIXELS, clip_height), interpolation=cv2.INTER_AREA)


def write_plain_clip(name, recording, start_seconds, length_seconds, crop):
    writer = None
    written = 0
    for frame in frames_of(recording, start_seconds, length_seconds):
        small = fit_to_clip(frame, crop)
        if writer is None:
            writer, path = open_writer(name, small.shape[1], small.shape[0])
        writer.write(small)
        written += 1
    if writer is not None:
        writer.release()
    print(f"{name}: {written} frames")


def write_landmark_clip(name, recording, start_seconds, length_seconds):
    sys.path.insert(0, PROJECT_DIRECTORY)
    os.chdir(PROJECT_DIRECTORY)                       # the tracker's model path is relative to the project
    import config
    from face_mesh_export import load_canonical_topology
    from face_tracker import FaceTracker

    config.face_detection_downscale_factor = 1        # offline: full resolution, the face may be small
    _, triangles = load_canonical_topology()
    edges = set()
    for a, b, c in triangles:
        for first, second in ((a, b), (b, c), (c, a)):
            edges.add((min(first, second), max(first, second)))
    edges = numpy.array(sorted(edges))
    writer = None
    tracker = None
    written = 0
    found = 0
    for frame_index, frame in enumerate(frames_of(recording, start_seconds, length_seconds)):
        if tracker is None:
            tracker = FaceTracker(config, frame.shape[1], frame.shape[0])
        landmarks = tracker.detect_raw(frame, frame_index / CLIP_FRAMES_PER_SECOND)
        if landmarks is not None:
            found += 1
            points = numpy.round(landmarks[:468]).astype(numpy.int32)
            for first, second in edges:
                cv2.line(frame, tuple(points[first]), tuple(points[second]), (0, 220, 255), 1, cv2.LINE_AA)
            for point in points:
                cv2.circle(frame, tuple(point), 2, (255, 255, 255), -1, cv2.LINE_AA)
        small = fit_to_clip(frame, None)
        if writer is None:
            writer, path = open_writer(name, small.shape[1], small.shape[0])
        writer.write(small)
        written += 1
    if writer is not None:
        writer.release()
    print(f"{name}: {written} frames, face found in {found}")


def main():
    os.makedirs(CLIPS_DIRECTORY, exist_ok=True)
    for name, (recording, start_seconds, length_seconds, crop) in PLAIN_CLIPS.items():
        write_plain_clip(name, recording, start_seconds, length_seconds, crop)
    write_landmark_clip(*LANDMARK_CLIP)


if __name__ == "__main__":
    main()
