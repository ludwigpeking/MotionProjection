"""Video covers for "Projection Mapping on Moving Objects": one horizontal, one vertical.

    ..\\.venv\\Scripts\\python.exe make_covers.py

The design: the face as a floating mask, cut out along its outline. Its left half is the
real camera picture with the projected image on it; its right half is the tracked mesh,
drawn as a wireframe. Fading copies of the wireframe trail behind it (the object moves),
the projector's beam comes in from the corner, and the title stands beside or below in
heavy type. Writes covers/cover_horizontal.png (1920 x 1080) and
covers/cover_vertical.png (1080 x 1920).
"""

import argparse
import os
import sys

import cv2
import numpy
from PIL import Image, ImageDraw, ImageFilter, ImageFont

PROJECT_DIRECTORY = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
RECORDINGS_DIRECTORY = os.path.join(PROJECT_DIRECTORY, "recordings")
OUTPUT_DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "covers")
FONT_DIRECTORY = r"C:\Windows\Fonts"
TITLE_FONT_FILES = ["seguibl.ttf", "ariblk.ttf", "impact.ttf"]            # first one found is used
TEXT_FONT_FILES = ["bahnschrift.ttf", "segoeuib.ttf", "arialbd.ttf"]
CREDIT_FONT_FILES = ["msyhbd.ttc", "msyh.ttc", "simhei.ttf"]              # must hold Chinese characters

GROUND_DARK = (6, 8, 16)
GROUND_LIGHT = (26, 30, 66)
ACCENT_COLOUR = (255, 216, 61)            # warm yellow
BEAM_COLOUR = (255, 86, 200)              # the projector's magenta
MESH_COLOUR = (80, 232, 255)              # cyan
WHITE = (255, 255, 255)

# MediaPipe's landmarks along the outline of the face, in order
FACE_OUTLINE = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152,
                148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
NOSE_TIP = 1

# Per language: the title lines as (text, colour name, with motion echo), the fonts of the
# title, the small lines, and what is added to the file names.
WORDING = {
    "en": {"title": [("PROJECTION", "white", False), ("MAPPING", "white", False),
                     ("ON MOVING", "accent", True), ("OBJECTS", "accent", False)],
           "title_fonts": TITLE_FONT_FILES, "line_height": 0.98,
           "tag": "KINECT  +  PROJECTOR", "below": "the geometry, explained", "below_fonts": TEXT_FONT_FILES,
           "suffix": ""},
    "zh": {"title": [("移动物体上的", "accent", True), ("投影映射", "white", False)],
           "title_fonts": CREDIT_FONT_FILES, "line_height": 1.22,
           "tag": "KINECT  +  PROJECTOR", "below": "几何原理讲解  ·  Projection Mapping on Moving Objects",
           "below_fonts": CREDIT_FONT_FILES, "suffix": "_zh"},
}


def font(file_names, size):
    for file_name in file_names:
        path = os.path.join(FONT_DIRECTORY, file_name)
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def read_frame(recording, seconds):
    capture = cv2.VideoCapture(os.path.join(RECORDINGS_DIRECTORY, recording))
    capture.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000.0)
    read_ok, frame = capture.read()
    capture.release()
    if not read_ok:
        raise SystemExit(f"no frame at {seconds} s in {recording}")
    return frame


def in_project_directory(action):
    """The project's modules find their files relative to the project folder."""
    sys.path.insert(0, PROJECT_DIRECTORY)
    working_directory = os.getcwd()
    os.chdir(PROJECT_DIRECTORY)
    try:
        return action()
    finally:
        os.chdir(working_directory)


def find_landmarks(frame):
    def detect():
        import config
        from face_tracker import FaceTracker
        config.face_detection_downscale_factor = 1
        tracker = FaceTracker(config, frame.shape[1], frame.shape[0])
        landmarks = tracker.detect_raw(frame, 0.0)
        tracker.close()
        return landmarks
    landmarks = in_project_directory(detect)
    if landmarks is None:
        raise SystemExit("no face found in that frame: choose another with --recording and --seconds")
    return landmarks[:468]


def mesh_edges():
    def load():
        from face_mesh_export import load_canonical_topology
        return load_canonical_topology()[1]
    edges = set()
    for a, b, c in in_project_directory(load):
        for first, second in ((a, b), (b, c), (c, a)):
            edges.add((min(first, second), max(first, second)))
    return sorted(edges)


def graded(frame_bgr):
    """More contrast and colour, so the projected image glows."""
    image = frame_bgr.astype(numpy.float32) / 255.0
    image = numpy.clip((image - 0.10) * 1.30, 0.0, 1.0) ** 1.05
    hsv = cv2.cvtColor((image * 255).astype(numpy.uint8), cv2.COLOR_BGR2HSV).astype(numpy.float32)
    hsv[:, :, 1] = numpy.clip(hsv[:, :, 1] * 1.5, 0, 255)
    return cv2.cvtColor(hsv.astype(numpy.uint8), cv2.COLOR_HSV2BGR)


def background(width, height, light_centre):
    rows, columns = numpy.mgrid[0:height, 0:width].astype(numpy.float32)
    distance = numpy.hypot(columns - light_centre[0], rows - light_centre[1]) / (0.95 * max(width, height))
    mix = numpy.clip(1.0 - distance, 0.0, 1.0)[:, :, None] ** 1.6
    ground = numpy.array(GROUND_DARK, numpy.float32) * (1.0 - mix) + numpy.array(GROUND_LIGHT, numpy.float32) * mix
    image = Image.fromarray(ground.astype(numpy.uint8)).convert("RGBA")
    # a faint measuring grid: this is a film about geometry
    grid = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(grid)
    spacing = max(width, height) // 24
    for x in range(0, width, spacing):
        draw.line([(x, 0), (x, height)], fill=MESH_COLOUR + (16,), width=1)
    for y in range(0, height, spacing):
        draw.line([(0, y), (width, y)], fill=MESH_COLOUR + (16,), width=1)
    return Image.alpha_composite(image, grid)


def face_layers(frame, landmarks, edges, size, face_centre, face_height):
    """The photographic half, the wireframe, and a copy of the wireframe alone for the trails,
    all as RGBA layers of the cover's size."""
    width, height = size
    outline_top = landmarks[FACE_OUTLINE][:, 1].min()
    outline_bottom = landmarks[FACE_OUTLINE][:, 1].max()
    source_centre = numpy.array([landmarks[FACE_OUTLINE][:, 0].mean(), (outline_top + outline_bottom) / 2.0])
    scale = face_height / (outline_bottom - outline_top)
    placed = (landmarks - source_centre) * scale + numpy.array(face_centre)

    transform = numpy.array([[scale, 0.0, face_centre[0] - source_centre[0] * scale],
                             [0.0, scale, face_centre[1] - source_centre[1] * scale]])
    picture = cv2.warpAffine(graded(frame), transform, (width, height), flags=cv2.INTER_LANCZOS4)
    picture = cv2.addWeighted(picture, 1.6, cv2.GaussianBlur(picture, (0, 0), 3.0), -0.6, 0)        # crisper after enlarging

    outline = numpy.round(placed[FACE_OUTLINE]).astype(numpy.int32)
    inside = numpy.zeros((height, width), numpy.uint8)
    cv2.fillPoly(inside, [outline], 255, lineType=cv2.LINE_AA)
    inside = cv2.GaussianBlur(inside, (0, 0), max(2.0, face_height / 260.0)).astype(numpy.float32) / 255.0
    columns = numpy.arange(width, dtype=numpy.float32)[None, :]
    split_x = placed[NOSE_TIP][0]
    blend_width = face_height * 0.07
    photographic_side = numpy.clip((split_x + blend_width - columns) / (2.0 * blend_width), 0.0, 1.0)

    photo = numpy.dstack([cv2.cvtColor(picture, cv2.COLOR_BGR2RGB),
                          (inside * photographic_side * 255).astype(numpy.uint8)])
    photo_layer = Image.fromarray(photo, "RGBA")

    wire_layer = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(wire_layer)
    line_width = max(2, int(round(face_height / 380.0)))
    for first, second in edges:
        draw.line([tuple(placed[first]), tuple(placed[second])], fill=MESH_COLOUR + (235,), width=line_width)
    dot_radius = max(2, int(round(face_height / 230.0)))
    for x, y in placed:
        draw.ellipse([x - dot_radius, y - dot_radius, x + dot_radius, y + dot_radius], fill=WHITE + (255,))
    # strong on the wireframe half, a light veil over the photographic half
    strength = (1.0 - 0.72 * photographic_side) * numpy.ones((height, 1), numpy.float32)
    alpha = numpy.asarray(wire_layer.getchannel("A"), numpy.float32) * strength
    split_wire = wire_layer.copy()
    split_wire.putalpha(Image.fromarray(alpha.astype(numpy.uint8)))

    # the dark fill behind the wireframe half, so the grid does not show through the face
    fill = numpy.zeros((height, width, 4), numpy.uint8)
    fill[:, :, :3] = (8, 14, 30)
    fill[:, :, 3] = (inside * (1.0 - photographic_side) * 235).astype(numpy.uint8)
    return photo_layer, split_wire, wire_layer, Image.fromarray(fill, "RGBA"), placed


def shifted(layer, offset, opacity):
    moved = Image.new("RGBA", layer.size, (0, 0, 0, 0))          # pasted, not rolled: nothing re-enters at the far edge
    moved.paste(layer, (int(offset[0]), int(offset[1])))
    alpha = moved.getchannel("A").point(lambda value: int(value * opacity))
    moved.putalpha(alpha)
    return moved


def beam_layer(size, source, face_centre, face_height):
    width, height = size
    half = 0.58 * face_height
    cone = Image.new("RGBA", size, (0, 0, 0, 0))
    ImageDraw.Draw(cone).polygon([source, (face_centre[0], face_centre[1] - half), (face_centre[0], face_centre[1] + half)],
                                 fill=BEAM_COLOUR + (58,))
    cone = cone.filter(ImageFilter.GaussianBlur(radius=max(8, height // 90)))
    rays = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(rays)
    for fraction in numpy.linspace(-1.0, 1.0, 7):
        draw.line([source, (face_centre[0], face_centre[1] + fraction * half)], fill=BEAM_COLOUR + (110,),
                  width=max(2, height // 500))
    return Image.alpha_composite(cone, rays)


def fitted_font(file_names, text, maximum_width, start_size):
    size = start_size
    while size > 24:
        candidate = font(file_names, size)
        if candidate.getbbox(text)[2] - candidate.getbbox(text)[0] <= maximum_width:
            return candidate
        size -= 2
    return font(file_names, size)


def draw_title(image, left, top, maximum_width, start_size, wording):
    """Blocks of heavy type, each as wide as the column; the line that moves has echoes.
    Returns the y below the last line."""
    colours = {"white": WHITE, "accent": ACCENT_COLOUR}
    lines = [(text, colours[colour_name], with_echo) for text, colour_name, with_echo in wording["title"]]
    size = min(fitted_font(wording["title_fonts"], text, maximum_width, start_size).size for text, _, _ in lines)
    title_font = font(wording["title_fonts"], size)
    y = top
    for text, colour, with_echo in lines:
        box = title_font.getbbox(text)
        if with_echo:                                         # the word that moves leaves a trail
            echo = Image.new("RGBA", image.size, (0, 0, 0, 0))
            echo_draw = ImageDraw.Draw(echo)
            for step, opacity in ((3, 40), (2, 75), (1, 120)):
                echo_draw.text((left - step * size * 0.085 - box[0], y - box[1]), text, font=title_font,
                               fill=colour + (opacity,))
            image.alpha_composite(echo)
        draw = ImageDraw.Draw(image)
        shadow = max(4, size // 18)
        draw.text((left + shadow - box[0], y + shadow - box[1]), text, font=title_font, fill=(0, 0, 0, 200))
        draw.text((left - box[0], y - box[1]), text, font=title_font, fill=colour + (255,))
        y += int(size * wording["line_height"])
    return y, size


def make_cover(frame, landmarks, edges, orientation, wording):
    if orientation == "horizontal":
        size = (1920, 1080)
        face_centre, face_height = (1420.0, 540.0), 800.0
        beam_source = (2000.0, -120.0)
        trail_direction = numpy.array([1.0, 0.0])
        text_left, text_top, text_width, title_size = 84, 232, 900, 190
    else:
        size = (1080, 1920)
        face_centre, face_height = (540.0, 520.0), 800.0
        beam_source = (1200.0, -160.0)
        trail_direction = numpy.array([1.0, 0.0])
        text_left, text_top, text_width, title_size = 70, 1090, 940, 180
    width, height = size
    image = background(width, height, face_centre)
    photo_layer, split_wire, full_wire, fill, _ = face_layers(frame, landmarks, edges, size, face_centre, face_height)

    # the trail: where the face was a moment ago
    for step, opacity in ((3, 0.10), (2, 0.18), (1, 0.30)):
        image.alpha_composite(shifted(full_wire, trail_direction * step * face_height * 0.11, opacity))
    image.alpha_composite(beam_layer(size, beam_source, face_centre, face_height))

    # the projected light spills: a soft glow of the picture's own colours behind the face
    glow = photo_layer.filter(ImageFilter.GaussianBlur(radius=face_height / 14.0))
    glow.putalpha(glow.getchannel("A").point(lambda value: int(value * 0.85)))
    image.alpha_composite(glow)
    image.alpha_composite(fill)
    image.alpha_composite(photo_layer)
    wire_glow = split_wire.filter(ImageFilter.GaussianBlur(radius=max(3, face_height / 200.0)))
    image.alpha_composite(wire_glow)
    image.alpha_composite(split_wire)

    draw = ImageDraw.Draw(image)
    tag_font = font(TEXT_FONT_FILES, int(title_size * 0.24))
    draw.text((text_left, text_top - int(title_size * 0.46)), wording["tag"], font=tag_font, fill=MESH_COLOUR + (255,))
    below, used_size = draw_title(image, text_left, text_top, text_width, title_size, wording)
    draw = ImageDraw.Draw(image)
    bar_top = below + int(used_size * 0.14)
    draw.rectangle([text_left, bar_top, text_left + int(used_size * 1.7), bar_top + max(8, used_size // 16)],
                   fill=BEAM_COLOUR + (255,))
    below_font = fitted_font(wording["below_fonts"], wording["below"], text_width, int(title_size * 0.30))
    draw.text((text_left, bar_top + int(used_size * 0.22)), wording["below"], font=below_font, fill=WHITE + (255,))
    credit_font = font(CREDIT_FONT_FILES, int(title_size * 0.19))
    draw.text((text_left, height - int(title_size * 0.46)), "Richard Qian Li 陈腐粉碎机  ·  ITP, NYU", font=credit_font,
              fill=(200, 206, 222, 255))
    return image.convert("RGB")


def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--recording", type=str, default="kinect_20260927_165901.avi")
    argument_parser.add_argument("--seconds", type=float, default=27.0)
    argument_parser.add_argument("--language", choices=sorted(WORDING), default="en")
    arguments = argument_parser.parse_args()
    frame = read_frame(arguments.recording, arguments.seconds)
    landmarks = find_landmarks(frame)
    outline = landmarks[FACE_OUTLINE]
    print(f"face found: {outline[:, 1].max() - outline[:, 1].min():.0f} px tall in the recording")
    edges = mesh_edges()
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    for orientation in ("horizontal", "vertical"):
        wording = WORDING[arguments.language]
        cover = make_cover(frame, landmarks, edges, orientation, wording)
        path = os.path.join(OUTPUT_DIRECTORY, f"cover_{orientation}{wording['suffix']}.png")
        cover.save(path)
        upload_path = os.path.join(OUTPUT_DIRECTORY, f"cover_{orientation}{wording['suffix']}.jpg")      # YouTube takes at most 2 MB
        cover.save(upload_path, quality=92, optimize=True)
        print(f"{path}: {cover.size[0]} x {cover.size[1]}; {upload_path}: {os.path.getsize(upload_path) / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
