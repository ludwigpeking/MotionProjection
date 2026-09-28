"""Draw a panda face from MediaPipe landmarks, in the coordinate frame the
landmarks are given in. The caller warps the result into projector space.
"""

import math

import cv2
import numpy

FACE_OVAL_INDICES = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379,
    378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
    162, 21, 54, 103, 67, 109,
]
# "Left" and "right" are the subject's own left and right.
LEFT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
LEFT_EYEBROW_INDICES = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]
RIGHT_EYEBROW_INDICES = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
LOWER_LIP_INDICES = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]

FOREHEAD_INDEX = 10
CHIN_INDEX = 152
NOSE_TIP_INDEX = 4
RIGHT_JAW_INDEX = 234
LEFT_JAW_INDEX = 454

EYE_PATCH_SCALE_FACTOR = 1.75
EYE_PATCH_OUTWARD_SHIFT_FACTOR = 0.12    # fraction of eye width, toward the outer face
EAR_RADIUS_FACTOR = 0.24                 # fraction of face width
EAR_OUTLINE_THICKNESS_FACTOR = 0.035     # fraction of face width
NOSE_WIDTH_FACTOR = 0.20                 # fraction of face width
NOSE_HEIGHT_FACTOR = 0.13


def _as_integer_point(point):
    return int(round(point[0])), int(round(point[1]))


def _draw_eye_patch(canvas, landmarks, eye_indices, eyebrow_indices, face_center, black):
    points = landmarks[eye_indices + eyebrow_indices].astype(numpy.float32)
    (center_x, center_y), (axis_width, axis_height), angle_degrees = cv2.fitEllipse(points)
    eye_width = numpy.linalg.norm(landmarks[eye_indices[0]] - landmarks[eye_indices[8]])
    outward_direction = numpy.array([center_x, center_y]) - face_center
    outward_norm = numpy.linalg.norm(outward_direction)
    if outward_norm > 1e-6:
        outward_direction = outward_direction / outward_norm
    shifted_center = numpy.array([center_x, center_y]) + outward_direction * eye_width * EYE_PATCH_OUTWARD_SHIFT_FACTOR
    axes = (int(axis_width * EYE_PATCH_SCALE_FACTOR / 2), int(axis_height * EYE_PATCH_SCALE_FACTOR / 2))
    cv2.ellipse(canvas, _as_integer_point(shifted_center), axes, angle_degrees, 0, 360, black, -1, cv2.LINE_AA)


def render_panda_mask(canvas_width, canvas_height, landmarks, face_fill_brightness, mouth_brightness):
    """landmarks: (N, 2) array in canvas pixel coordinates. Returns a BGR uint8 canvas."""
    canvas = numpy.zeros((canvas_height, canvas_width, 3), dtype=numpy.uint8)
    white = (face_fill_brightness,) * 3
    black = (0, 0, 0)
    mouth_color = (mouth_brightness,) * 3

    forehead = landmarks[FOREHEAD_INDEX]
    chin = landmarks[CHIN_INDEX]
    right_jaw = landmarks[RIGHT_JAW_INDEX]
    left_jaw = landmarks[LEFT_JAW_INDEX]

    face_height = numpy.linalg.norm(forehead - chin)
    face_width = numpy.linalg.norm(left_jaw - right_jaw)
    if face_height < 2 or face_width < 2:
        return canvas

    up_direction = (forehead - chin) / face_height
    side_direction = (left_jaw - right_jaw) / face_width
    roll_degrees = math.degrees(math.atan2(side_direction[1], side_direction[0]))
    face_center = (forehead + chin) / 2.0

    # Head cap: a white region over the top of the head so the ears have a background.
    cap_center = face_center + up_direction * face_height * 0.30
    cap_axes = (int(face_width * 0.58), int(face_height * 0.62))
    cv2.ellipse(canvas, _as_integer_point(cap_center), cap_axes, roll_degrees, 0, 360, white, -1, cv2.LINE_AA)

    # Face itself, following the mesh outline.
    oval = landmarks[FACE_OVAL_INDICES].astype(numpy.int32).reshape(-1, 1, 2)
    cv2.fillPoly(canvas, [oval], white, cv2.LINE_AA)

    # Ears: black discs with a white rim so they read against dark hair.
    ear_radius = int(face_width * EAR_RADIUS_FACTOR)
    ear_outline_thickness = int(face_width * EAR_OUTLINE_THICKNESS_FACTOR)
    ear_offset_up = face_height * 0.55
    ear_offset_side = face_width * 0.48
    for side_sign in (-1.0, 1.0):
        ear_center = cap_center + up_direction * ear_offset_up + side_direction * side_sign * ear_offset_side
        cv2.circle(canvas, _as_integer_point(ear_center), ear_radius + ear_outline_thickness, white, -1, cv2.LINE_AA)
        cv2.circle(canvas, _as_integer_point(ear_center), ear_radius, black, -1, cv2.LINE_AA)

    # Eye patches.
    _draw_eye_patch(canvas, landmarks, LEFT_EYE_INDICES, LEFT_EYEBROW_INDICES, face_center, black)
    _draw_eye_patch(canvas, landmarks, RIGHT_EYE_INDICES, RIGHT_EYEBROW_INDICES, face_center, black)

    # Nose.
    nose_center = landmarks[NOSE_TIP_INDEX] - up_direction * face_height * 0.03
    nose_axes = (int(face_width * NOSE_WIDTH_FACTOR / 2), int(face_width * NOSE_HEIGHT_FACTOR / 2))
    cv2.ellipse(canvas, _as_integer_point(nose_center), nose_axes, roll_degrees, 0, 360, black, -1, cv2.LINE_AA)

    # Mouth: a line along the lower lip.
    mouth = landmarks[LOWER_LIP_INDICES].astype(numpy.int32).reshape(-1, 1, 2)
    mouth_thickness = max(2, int(face_width * 0.02))
    cv2.polylines(canvas, [mouth], False, mouth_color, mouth_thickness, cv2.LINE_AA)

    return canvas
