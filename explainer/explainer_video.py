"""The face-projection system explained: camera pixels to points in the room, points to
projector pixels, and the feedback loop around a moving face. One Manim scene, narrated
and subtitled section by section.

    ..\\.venv\\Scripts\\python.exe prepare_clips.py
    ..\\.venv\\Scripts\\python.exe build_audio.py
    ..\\.venv\\Scripts\\python.exe -m manim render -qm --disable_caching explainer_video.py SystemExplainer

Render only some sections while editing (PowerShell):
    $env:EXPLAINER_SECTIONS = "error,loop"

Every section adds its narration clip and then plays each animation at the moment the
voice reaches the matching phrase (self.at("phrase")), so picture and voice stay together.

Notation, as in the narration:
    depth camera pixel (x, y) with distance d      colour camera pixel (x', y')
    projector pixel (x'', y'')                     point in the room (X, Y, Z), metres
"""

import json
import os
import re

import cv2
import numpy
from manim import (BLACK, BLUE, DOWN, GREEN, GREY, GREY_B, LEFT, ORANGE, ORIGIN, RED, RIGHT, UP, WHITE, YELLOW,
                   RESAMPLING_ALGORITHMS, Arrow, Circle, Create, CurvedArrow, DashedLine, Dot, FadeIn, FadeOut,
                   GrowArrow, ImageMobject, Indicate, Line, Polygon, Rectangle, RoundedRectangle, Scene,
                   VGroup, VMobject, ValueTracker, Write, always_redraw, config, rate_functions)

from manim import MarkupText as ManimMarkupText
from manim import Text as ManimText

LANGUAGE = os.environ.get("EXPLAINER_LANGUAGE", "en")           # "en" or "zh"
if LANGUAGE == "zh":
    import language_zh
    NARRATION = language_zh.NARRATION
else:
    from narration import NARRATION

AUDIO_DIRECTORY = "audio" if LANGUAGE == "en" else f"audio_{LANGUAGE}"
CHINESE_FONT = "Microsoft YaHei"
UNTRANSLATED_TEXTS = set()
CLIPS_DIRECTORY = "clips"
IMAGES_DIRECTORY = "images"
NARRATION_DATA_PATH = os.path.join(AUDIO_DIRECTORY, "narration.json")
SUBTITLE_FILE_PATH = "system_explainer.srt" if LANGUAGE == "en" else f"system_explainer_{LANGUAGE}.srt"
CANONICAL_FACE_PATH = os.path.join("..", "canonical_face_model.obj")
CLIP_FRAMES_PER_SECOND = 15.0
AUTHOR_CREDITS = {"en": "Richard Qian Li 陈腐粉碎机  ·  ITP, NYU", "zh": "Richard Qian Li 陈腐粉碎机"}
AUTHOR_CREDIT = AUTHOR_CREDITS[LANGUAGE]


KINECT_COLOUR = BLUE
DEPTH_COLOUR = RED
PROJECTOR_COLOUR = ORANGE
FACE_COLOUR = YELLOW
WALL_COLOUR = GREY
POINT_COLOUR = GREEN
ERROR_COLOUR = RED

CAPTION_BAND_HEIGHT = 0.9
CONTENT_FLOOR = -4.0 + CAPTION_BAND_HEIGHT + 0.1        # nothing but subtitles below this height

config.background_color = "#101418"


# ---------------------------------------------------------------- language
def translated(text):
    """The on-screen text in the language of this build. Texts without letters (symbols,
    numbers, coordinates) and names are the same in every language."""
    if LANGUAGE != "zh":
        return text
    if text in language_zh.TEXTS:
        return language_zh.TEXTS[text]
    changed = text
    for fragment, replacement in language_zh.TEXT_FRAGMENTS:
        changed = changed.replace(fragment, replacement)
    if changed == text and re.search(r"[A-Za-z]{4,}", re.sub(r"<[^>]+>", "", text)):
        UNTRANSLATED_TEXTS.add(text)
    return changed


def Text(text, **settings):
    if LANGUAGE == "zh":
        settings.setdefault("font", CHINESE_FONT)
    return ManimText(translated(text), **settings)


def MarkupText(text, **settings):
    if LANGUAGE == "zh":
        settings.setdefault("font", CHINESE_FONT)
    return ManimMarkupText(translated(text), **settings)


def without_spaces(text):
    return re.sub(r"\s+", "", text)


# ---------------------------------------------------------------- small builders
def label(text, size=26, colour=WHITE):
    return Text(text, font_size=size, color=colour)


def formula(markup, size=32, colour=WHITE):
    return MarkupText(markup, font_size=size, color=colour)


def framed(group, colour=GREY_B, padding=0.25):
    frame = RoundedRectangle(corner_radius=0.12, width=group.width + 2 * padding, height=group.height + 2 * padding,
                             color=colour, stroke_width=2).move_to(group)
    return frame


def thin_arrow(start, end, colour=WHITE, tip_ratio=0.15):
    return Arrow(start, end, buff=0, color=colour, stroke_width=3, max_tip_length_to_length_ratio=tip_ratio)


def double_arrow(start, end, colour=WHITE):
    middle = (numpy.array(start) + numpy.array(end)) / 2.0
    return VGroup(thin_arrow(middle, start, colour, 0.12), thin_arrow(middle, end, colour, 0.12))


def top_view(origin, units_per_metre):
    """Returns a function placing (x metres to the right, z metres ahead) on the screen, seen from above."""
    def place(x_metres, z_metres):
        return numpy.array(origin) + numpy.array([x_metres * units_per_metre, z_metres * units_per_metre, 0.0])
    return place


def text_matrix(rows, size=24, colour=WHITE):
    """A matrix drawn from text entries and two square brackets (no LaTeX needed)."""
    entries = VGroup(*[MarkupText(cell, font_size=size, color=colour) for row in rows for cell in row])
    entries.arrange_in_grid(rows=len(rows), cols=len(rows[0]), buff=(0.3, 0.18))
    top, bottom = entries.get_top()[1] + 0.12, entries.get_bottom()[1] - 0.12
    left, right = entries.get_left()[0] - 0.15, entries.get_right()[0] + 0.15
    tick = 0.12
    left_bracket = VMobject(color=colour, stroke_width=2).set_points_as_corners(
        [[left + tick, top, 0], [left, top, 0], [left, bottom, 0], [left + tick, bottom, 0]])
    right_bracket = VMobject(color=colour, stroke_width=2).set_points_as_corners(
        [[right - tick, top, 0], [right, top, 0], [right, bottom, 0], [right - tick, bottom, 0]])
    return VGroup(left_bracket, entries, right_bracket)


def load_canonical_face():
    """Vertices (468, 3) in the canonical frame (x right, y up, z toward the viewer) and triangles (898, 3)."""
    vertices, triangles = [], []
    with open(CANONICAL_FACE_PATH) as file:
        for line in file:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "v":
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == "f":
                triangles.append([int(token.split("/")[0]) - 1 for token in parts[1:4]])
    return numpy.array(vertices), numpy.array(triangles)


def load_canonical_texture_coordinates():
    """(468, 2) texture coordinates of the canonical mesh: u to the right, v up, both 0 .. 1."""
    texture_coordinates, faces = [], []
    with open(CANONICAL_FACE_PATH) as file:
        for line in file:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "vt":
                texture_coordinates.append((float(parts[1]), float(parts[2])))
            elif parts[0] == "f":
                faces.append([(int(token.split("/")[0]) - 1, int(token.split("/")[1]) - 1) for token in parts[1:4]])
    texture_coordinates = numpy.array(texture_coordinates)
    per_vertex = numpy.zeros((468, 2))
    for face in faces:
        for vertex_index, texture_index in face:
            per_vertex[vertex_index] = texture_coordinates[texture_index]
    return per_vertex


def mesh_edges(vertices_2d, triangles, keep_triangles=None):
    """One VMobject holding every unique edge of the kept triangles."""
    edges = set()
    for index, (a, b, c) in enumerate(triangles):
        if keep_triangles is not None and not keep_triangles[index]:
            continue
        for first, second in ((a, b), (b, c), (c, a)):
            edges.add((min(first, second), max(first, second)))
    path = VMobject()
    for first, second in sorted(edges):
        path.start_new_path(numpy.array([vertices_2d[first][0], vertices_2d[first][1], 0.0]))
        path.add_line_to(numpy.array([vertices_2d[second][0], vertices_2d[second][1], 0.0]))
    return path


def rotation_about_y(angle_radians):
    cosine, sine = numpy.cos(angle_radians), numpy.sin(angle_radians)
    return numpy.array([[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]])


def drawn_projector(width=1.8):
    """A small projector seen from the front quarter: body, lens and a hint of light."""
    body = RoundedRectangle(corner_radius=0.12, width=width, height=width * 0.5, color=PROJECTOR_COLOUR, fill_opacity=0.25)
    lens = Circle(radius=width * 0.14, color=WHITE, fill_opacity=0.5).move_to(body.get_center() + LEFT * width * 0.25)
    inner = Circle(radius=width * 0.07, color=PROJECTOR_COLOUR, fill_opacity=0.9).move_to(lens)
    vents = VGroup(*[Line(UP * 0.12, DOWN * 0.12, color=PROJECTOR_COLOUR, stroke_width=2).move_to(
        body.get_center() + RIGHT * (0.15 + 0.12 * index) * width / 1.8) for index in range(4)])
    return VGroup(body, lens, inner, vents)


class VideoClip(ImageMobject):
    """A clip from clips/ playing inside the scene. It fades itself in and out through the
    alpha channel of the frame it shows, because Manim's own fades write into the pixel
    array that the clip replaces every frame."""

    def __init__(self, name, width_units, loop=True):
        capture = cv2.VideoCapture(os.path.join(CLIPS_DIRECTORY, f"{name}.mp4"))
        self.frames = []
        while True:
            read_ok, frame = capture.read()
            if not read_ok:
                break
            self.frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA))
        capture.release()
        if not self.frames:
            raise FileNotFoundError(f"clips/{name}.mp4 is missing or empty: run prepare_clips.py")
        super().__init__(self.frames[0])
        self.set_resampling_algorithm(RESAMPLING_ALGORITHMS["linear"])
        self.scale_to_fit_width(width_units)
        self.loop = loop
        self.clock_seconds = 0.0
        self.fade_seconds = 0.6
        self.fading_out_since = None
        self.show_frame()
        self.add_updater(VideoClip.advance)

    def show_frame(self):
        index = int(self.clock_seconds * CLIP_FRAMES_PER_SECOND)
        index = index % len(self.frames) if self.loop else min(index, len(self.frames) - 1)
        opacity = min(self.clock_seconds / self.fade_seconds, 1.0)
        if self.fading_out_since is not None:
            opacity = min(opacity, max(1.0 - (self.clock_seconds - self.fading_out_since) / self.fade_seconds, 0.0))
        frame = self.frames[index].copy()
        frame[:, :, 3] = int(round(255 * opacity))
        self.pixel_array = frame

    def advance(self, dt):
        # Manim recognises a time-based updater by a parameter named exactly "dt": the seconds since the last frame.
        self.clock_seconds += dt
        self.show_frame()

    def fade_out(self):
        self.fading_out_since = self.clock_seconds


class SystemExplainer(Scene):
    def setup(self):
        with open(NARRATION_DATA_PATH, encoding="utf-8") as file:
            self.narration_data = json.load(file)
        self.section_key = None
        self.section_started_at = 0.0
        self.section_starts = []
        self.video_clips = []

        self.caption_band = Rectangle(width=config.frame_width, height=CAPTION_BAND_HEIGHT, stroke_width=0,
                                      fill_color=BLACK, fill_opacity=0.75).to_edge(DOWN, buff=0)
        self.caption_band.set_z_index(20)
        self.caption_lines = []
        self.caption_clock_seconds = 0.0
        self.caption_shown_index = None
        self.caption_holder = VGroup()
        self.caption_holder.set_z_index(21)
        self.caption_holder.add_updater(self.update_caption)
        self.add(self.caption_band, self.caption_holder)

    # ------------------------------------------------------------ narration, subtitles, cues
    def update_caption(self, holder, dt):
        # Manim recognises a time-based updater by a parameter named exactly "dt": the seconds since the last frame.
        self.caption_clock_seconds += dt
        active_index = None
        for index, (start, end, _) in enumerate(self.caption_lines):
            if start <= self.caption_clock_seconds < end:
                active_index = index
                break
        if active_index != self.caption_shown_index:
            # Every line of the section stays in the holder and only its opacity changes: Manim lists
            # the mobjects to draw when an animation starts, so a line swapped in or out in the middle
            # of one would be drawn together with the line it replaced.
            if self.caption_shown_index is not None:
                self.caption_lines[self.caption_shown_index][2].set_opacity(0.0)
            if active_index is not None:
                self.caption_lines[active_index][2].set_opacity(1.0)
            self.caption_shown_index = active_index

    def narrate(self, key):
        self.add_sound(os.path.join(AUDIO_DIRECTORY, f"{key}.mp3"))
        self.section_key = key
        self.section_started_at = self.renderer.time
        self.section_starts.append((key, self.renderer.time))
        self.caption_lines = []
        for line in self.narration_data[key]["subtitles"]:
            caption_settings = {"font": CHINESE_FONT} if LANGUAGE == "zh" else {}
            caption = ManimText(line["text"], font_size=26, color=WHITE, **caption_settings).move_to(self.caption_band)
            caption.set_z_index(21)
            caption.set_opacity(0.0)
            self.caption_lines.append((line["start"], line["end"], caption))
        self.caption_clock_seconds = 0.0
        self.caption_shown_index = None
        self.caption_holder.submobjects = [caption for _, _, caption in self.caption_lines]

    def section_clock(self):
        return self.renderer.time - self.section_started_at

    def cue_seconds(self, phrase):
        """When the voice reaches a phrase: the start of the subtitle line that holds the
        phrase's first character, moved into the line in proportion to where the phrase sits.
        Spaces and line breaks are ignored, so a phrase may run across two subtitle lines."""
        if LANGUAGE == "zh":
            phrase = language_zh.CUES[self.section_key][phrase]
        lines = self.narration_data[self.section_key]["subtitles"]
        full_text = ""
        spans = []
        for line in lines:
            start_character = len(full_text)
            full_text += without_spaces(line["text"]).lower()
            spans.append((start_character, len(full_text), line))
        position = full_text.find(without_spaces(phrase).lower())
        if position < 0:
            raise ValueError(f"cue '{phrase}' is not in the narration of section '{self.section_key}'")
        for start_character, end_character, line in spans:
            if start_character <= position < end_character:
                fraction = (position - start_character) / float(end_character - start_character)
                return line["start"] + fraction * (line["end"] - line["start"])
        return 0.0

    def at(self, phrase, lead_seconds=0.3):
        """Wait until just before the voice says the phrase."""
        remaining = self.cue_seconds(phrase) - lead_seconds - self.section_clock()
        if remaining > 1.0 / config.frame_rate:
            self.wait(remaining)

    def finish_narration(self, tail_seconds=1.0):
        remaining = self.narration_data[self.section_key]["seconds"] - self.section_clock() + tail_seconds
        self.wait(max(remaining, 0.3))

    def play_clip(self, name, width_units, position, loop=True, outline_colour=GREY_B):
        clip = VideoClip(name, width_units, loop)
        clip.move_to(position)
        outline = Rectangle(width=clip.width, height=clip.height, color=outline_colour, stroke_width=2).move_to(clip)
        self.video_clips.append(clip)
        self.add(clip)
        self.play(Create(outline), run_time=0.6)
        return clip, outline

    def clear_scene(self):
        for clip in self.video_clips:
            clip.fade_out()
        kept = {self.caption_band, self.caption_holder, *self.video_clips}
        fading = [mobject for mobject in self.mobjects if mobject not in kept]
        if fading:
            self.play(*[FadeOut(mobject) for mobject in fading], run_time=0.7)
        else:
            self.wait(0.7)
        for clip in self.video_clips:
            clip.clear_updaters()
            self.remove(clip)
        self.video_clips = []

    def write_subtitle_file(self):
        def stamp(seconds):
            milliseconds = int(round(seconds * 1000.0))
            hours, milliseconds = divmod(milliseconds, 3_600_000)
            minutes, milliseconds = divmod(milliseconds, 60_000)
            whole_seconds, milliseconds = divmod(milliseconds, 1000)
            return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d},{milliseconds:03d}"
        entries = []
        for key, section_start in self.section_starts:
            for line in self.narration_data[key]["subtitles"]:
                entries.append((section_start + line["start"], section_start + line["end"], line["text"]))
        with open(SUBTITLE_FILE_PATH, "w", encoding="utf-8") as file:
            for number, (start, end, text) in enumerate(entries, start=1):
                file.write(f"{number}\n{stamp(start)} --> {stamp(end)}\n{text}\n\n")

    # ------------------------------------------------------------ the film
    def construct(self):
        wanted = [name.strip() for name in os.environ.get("EXPLAINER_SECTIONS", "").split(",") if name.strip()]
        for key, _ in NARRATION:
            if wanted and key not in wanted:
                continue
            getattr(self, f"{key}_section")()
        self.write_subtitle_file()
        for text in sorted(UNTRANSLATED_TEXTS):
            print(f"[language] no translation for: {text!r}")

    def title_section(self):
        self.narrate("title")
        title = Text("Projecting onto a moving face", font_size=50).to_edge(UP, buff=0.3)
        credit = ManimText(AUTHOR_CREDIT, font_size=24, color=GREY_B).next_to(title, DOWN, buff=0.15)
        self.play(Write(title), run_time=2.0)
        self.play(FadeIn(credit), run_time=0.8)
        clip, outline = self.play_clip("result_lit", 6.0, numpy.array([0.0, 0.45, 0.0]))
        watcher = label("Kinect: watches", 26, KINECT_COLOUR).next_to(outline, LEFT, buff=0.4)
        painter = label("projector: paints", 26, PROJECTOR_COLOUR).next_to(outline, RIGHT, buff=0.4)
        self.at("A Kinect depth camera")
        self.play(FadeIn(watcher), run_time=0.8)
        self.at("A small Optoma")
        self.play(FadeIn(painter), run_time=0.8)
        first = VGroup(label("camera pixel", 26, KINECT_COLOUR), thin_arrow(LEFT * 0.4, RIGHT * 0.4),
                       label("point in the room", 26, POINT_COLOUR)).arrange(RIGHT, buff=0.3)
        second = VGroup(thin_arrow(LEFT * 0.4, RIGHT * 0.4), label("projector pixel", 26, PROJECTOR_COLOUR)).arrange(RIGHT, buff=0.3)
        chain = VGroup(first, second).arrange(RIGHT, buff=0.3).move_to(numpy.array([0.0, -2.1, 0.0]))
        self.at("how a camera pixel")
        self.play(FadeIn(first), run_time=1.0)
        self.at("and how that point")
        self.play(FadeIn(second), run_time=1.0)
        self.finish_narration()
        self.clear_scene()

    def equipment_section(self):
        self.narrate("equipment")
        heading = label("The equipment", 34).to_edge(UP, buff=0.4)
        kinect_photo = ImageMobject(os.path.join(IMAGES_DIRECTORY, "kinect.jpg"))
        kinect_photo.set_resampling_algorithm(RESAMPLING_ALGORITHMS["linear"])
        kinect_photo.scale_to_fit_width(5.6).move_to(numpy.array([-3.5, 1.0, 0.0]))
        kinect_name = label("Kinect v2", 28, KINECT_COLOUR).next_to(kinect_photo, DOWN, buff=0.2)
        credit = label("photo: Evan-Amos, public domain", 14, GREY).next_to(kinect_photo, UP, buff=0.05).align_to(kinect_photo, RIGHT)
        self.play(FadeIn(heading), FadeIn(kinect_photo), FadeIn(kinect_name), FadeIn(credit), run_time=1.2)
        colour_line = label("colour camera   1920 × 1080", 24).next_to(kinect_name, DOWN, buff=0.25)
        depth_line = label("depth camera   512 × 424\ninfrared, measures distance by timing light", 22, DEPTH_COLOUR).next_to(colour_line, DOWN, buff=0.15)
        self.at("A colour camera")
        self.play(FadeIn(colour_line), run_time=0.8)
        self.at("infrared depth camera")
        self.play(FadeIn(depth_line), run_time=0.8)

        projector_photo_path = os.path.join(IMAGES_DIRECTORY, "projector.jpg")
        if os.path.exists(projector_photo_path):
            projector_picture = ImageMobject(projector_photo_path)
            projector_picture.set_resampling_algorithm(RESAMPLING_ALGORITHMS["linear"])
            projector_picture.scale_to_fit_width(4.2)
        else:
            projector_picture = drawn_projector(3.0)
        projector_picture.move_to(numpy.array([3.8, 1.0, 0.0]))
        projector_name = label("Optoma projector", 28, PROJECTOR_COLOUR).next_to(projector_picture, DOWN, buff=0.4)
        projector_name.set_y(kinect_name.get_y())
        projector_line = label("1920 × 1080   second screen", 24).next_to(projector_name, DOWN, buff=0.25)
        self.at("The projector is")
        self.play(FadeIn(projector_picture), FadeIn(projector_name), run_time=1.0)
        self.play(FadeIn(projector_line), run_time=0.8)
        self.finish_narration()
        self.clear_scene()

    def rig_section(self):
        self.narrate("rig")
        place = top_view([-2.2, -1.9, 0.0], 2.0)
        heading = label("The rig, from above", 30).to_corner(UP + LEFT, buff=0.4)
        self.play(FadeIn(heading), run_time=0.6)
        kinect = Rectangle(width=0.5, height=0.22, color=KINECT_COLOUR, fill_opacity=0.5).move_to(place(0, 0))
        kinect_label = label("Kinect", 24, KINECT_COLOUR).next_to(kinect, LEFT, buff=0.15)
        projector = Rectangle(width=0.4, height=0.3, color=PROJECTOR_COLOUR, fill_opacity=0.5).move_to(place(0.7, 0))
        projector_label = label("projector", 24, PROJECTOR_COLOUR).next_to(projector, RIGHT, buff=0.15)
        self.at("Kinect sits")
        self.play(FadeIn(kinect), FadeIn(kinect_label), run_time=0.8)
        self.at("projector stands")
        self.play(FadeIn(projector), FadeIn(projector_label), run_time=0.8)
        baseline = double_arrow(place(0.0, -0.17), place(0.7, -0.17))
        baseline_label = label("baseline b = 0.7 m", 22).next_to(baseline, DOWN, buff=0.08)
        self.at("baseline")
        self.play(FadeIn(baseline), FadeIn(baseline_label), run_time=0.8)

        face_position = place(0.25, 0.8)
        face = Circle(radius=0.24, color=FACE_COLOUR, fill_opacity=0.4).move_to(face_position)
        nose = Polygon(face_position + numpy.array([-0.08, -0.22, 0]), face_position + numpy.array([0.08, -0.22, 0]),
                       face_position + numpy.array([0.0, -0.36, 0]), color=FACE_COLOUR, fill_opacity=0.4)
        face_label = label("person", 24, FACE_COLOUR).next_to(face, RIGHT, buff=0.15)
        distance = double_arrow(place(-0.95, 0.0), place(-0.95, 0.8))
        distance_label = label("d = 0.8 m", 22).next_to(distance, LEFT, buff=0.08)
        self.at("person is at")
        self.play(FadeIn(face), FadeIn(nose), FadeIn(face_label), FadeIn(distance), FadeIn(distance_label), run_time=1.0)
        wall = Line(place(-2.0, 2.5), place(3.2, 2.5), color=WALL_COLOUR, stroke_width=6)
        wall_label = label("wall, 2.5 m", 22, WALL_COLOUR).next_to(wall, UP, buff=0.08)
        self.at("wall is further")
        self.play(Create(wall), FadeIn(wall_label), run_time=0.8)

        half_view = numpy.tan(numpy.radians(35.0))
        kinect_view = VGroup(Line(place(0, 0), place(-2.5 * half_view, 2.5), color=KINECT_COLOUR, stroke_opacity=0.5),
                             Line(place(0, 0), place(2.5 * half_view, 2.5), color=KINECT_COLOUR, stroke_opacity=0.5))
        aim = face_position - place(0.7, 0)
        aim_direction = aim / numpy.linalg.norm(aim)
        across = numpy.array([-aim_direction[1], aim_direction[0], 0.0])
        beam_half_width = numpy.tan(numpy.radians(14.0))
        beam = VGroup()
        for side in (-1.0, 1.0):
            direction = aim_direction + side * beam_half_width * across
            edge_reach = (place(0, 2.5)[1] - place(0, 0)[1]) / direction[1]
            beam.add(Line(place(0.7, 0), place(0.7, 0) + direction * edge_reach, color=PROJECTOR_COLOUR, stroke_opacity=0.5))
        self.at("never share")
        self.play(Create(kinect_view), Create(beam), run_time=1.5)
        note = label("no shared viewpoint:\npixels do not map to pixels", 24).move_to(numpy.array([4.4, 0.4, 0.0]))
        self.play(FadeIn(note), run_time=0.8)
        note_room = label("common ground: the room", 24, POINT_COLOUR).next_to(note, DOWN, buff=0.3)
        self.at("only common ground")
        self.play(FadeIn(note_room), run_time=0.8)
        self.finish_narration()
        self.clear_scene()

    def notation_section(self):
        self.narrate("notation")
        heading = label("Three images, one room", 30).to_edge(UP, buff=0.35)
        self.play(FadeIn(heading), run_time=0.6)

        def image_panel(width, height, colour, title, pixel_name, centre):
            frame = Rectangle(width=width, height=height, color=colour).move_to(centre)
            corner = frame.get_corner(UP + LEFT)
            column_arrow = thin_arrow(corner, corner + RIGHT * 1.1, colour, 0.2)
            row_arrow = thin_arrow(corner, corner + DOWN * 1.1, colour, 0.2)
            column_name = MarkupText(pixel_name[0], font_size=24, color=colour).next_to(column_arrow, UP, buff=0.05)
            row_name = MarkupText(pixel_name[1], font_size=24, color=colour).next_to(row_arrow, LEFT, buff=0.05)
            caption = label(title, 22, colour).next_to(frame, DOWN, buff=0.15)
            sample = Dot(frame.get_center() + RIGHT * 0.35 + DOWN * 0.2, color=colour, radius=0.06)
            sample_name = MarkupText(pixel_name[2], font_size=24, color=colour).next_to(sample, RIGHT, buff=0.1)
            return VGroup(frame, caption, sample, sample_name), VGroup(column_arrow, column_name), VGroup(row_arrow, row_name)

        depth_panel, depth_columns, depth_rows = image_panel(2.6, 2.15, DEPTH_COLOUR, "depth camera  512 × 424",
                                                             ("x", "y", "(x, y), d"), numpy.array([-4.6, 1.5, 0.0]))
        colour_panel, colour_columns, colour_rows = image_panel(3.6, 2.03, KINECT_COLOUR, "colour camera  1920 × 1080",
                                                                ("x′", "y′", "(x′, y′)"), numpy.array([-0.4, 1.5, 0.0]))
        projector_panel, projector_columns, projector_rows = image_panel(3.6, 2.03, PROJECTOR_COLOUR, "projector  1920 × 1080",
                                                                         ("x″", "y″", "(x″, y″)"), numpy.array([4.4, 1.5, 0.0]))
        self.at("x and y in the depth")
        self.play(FadeIn(depth_panel), FadeIn(depth_columns), FadeIn(depth_rows), run_time=1.0)
        self.at("x prime and y prime")
        self.play(FadeIn(colour_panel), FadeIn(colour_columns), FadeIn(colour_rows), run_time=1.0)
        self.at("x double prime")
        self.play(FadeIn(projector_panel), FadeIn(projector_columns), FadeIn(projector_rows), run_time=1.0)
        self.at("x counts columns")
        self.play(Indicate(depth_columns), Indicate(colour_columns), Indicate(projector_columns), run_time=1.2)
        self.at("y counts rows")
        self.play(Indicate(depth_rows), Indicate(colour_rows), Indicate(projector_rows), run_time=1.2)

        axes_origin = numpy.array([-1.6, -2.2, 0.0])
        right_axis = thin_arrow(axes_origin, axes_origin + RIGHT * 1.5, POINT_COLOUR, 0.15)
        up_axis = thin_arrow(axes_origin, axes_origin + UP * 1.3, POINT_COLOUR, 0.15)
        ahead_axis = thin_arrow(axes_origin, axes_origin + numpy.array([0.95, 0.7, 0.0]), POINT_COLOUR, 0.15)
        right_name = label("X  right", 24, POINT_COLOUR).next_to(right_axis, RIGHT, buff=0.1)
        up_name = label("Y  up", 24, POINT_COLOUR).next_to(up_axis, LEFT, buff=0.1)
        ahead_name = label("Z  ahead", 24, POINT_COLOUR).next_to(ahead_axis.get_end(), RIGHT, buff=0.1)
        camera_mark = Dot(axes_origin, color=KINECT_COLOUR, radius=0.08)
        room_caption = label("the room: (X, Y, Z) in metres,\nmeasured from the colour camera", 24, POINT_COLOUR).move_to(numpy.array([3.4, -1.6, 0.0]))
        self.at("room has one set")
        self.play(FadeIn(camera_mark), FadeIn(room_caption), run_time=1.0)
        self.at("X to the right")
        self.play(GrowArrow(right_axis), FadeIn(right_name), run_time=0.7)
        self.at("Y up")
        self.play(GrowArrow(up_axis), FadeIn(up_name), run_time=0.7)
        self.at("Z straight ahead")
        self.play(GrowArrow(ahead_axis), FadeIn(ahead_name), run_time=0.7)
        self.finish_narration()
        self.clear_scene()

    def pixel_to_point_section(self):
        self.narrate("pixel_to_point")
        heading = label("Colour pixel + distance = point   (seen from above)", 28).to_corner(UP + LEFT, buff=0.35)
        self.play(FadeIn(heading), run_time=0.6)
        lens_position = numpy.array([-5.2, -0.8, 0.0])
        focal_units = 1.5
        pixel_offset_units = 0.7
        distance_units = 5.6
        image_x = lens_position[0] + focal_units
        lens = Dot(lens_position, color=KINECT_COLOUR, radius=0.09)
        lens_label = label("lens centre", 20, KINECT_COLOUR).next_to(lens, LEFT, buff=0.1).shift(UP * 0.3)
        axis = DashedLine(lens_position, lens_position + RIGHT * 7.0, color=GREY, dash_length=0.15)
        axis_label = label("Z: straight ahead", 20, GREY).next_to(axis.get_end(), DOWN, buff=0.1).shift(LEFT * 0.9)
        image_plane = Line(numpy.array([image_x, lens_position[1] - 1.3, 0]), numpy.array([image_x, lens_position[1] + 1.5, 0]),
                           color=KINECT_COLOUR, stroke_width=4)
        image_label = label("image", 20, KINECT_COLOUR).next_to(image_plane, UP, buff=0.08)
        centre_mark = Dot(numpy.array([image_x, lens_position[1], 0]), color=WHITE, radius=0.05)
        centre_label = MarkupText("c<sub>x</sub>", font_size=22).next_to(centre_mark, DOWN + RIGHT, buff=0.06)
        pixel = Dot(numpy.array([image_x, lens_position[1] + pixel_offset_units, 0]), color=FACE_COLOUR, radius=0.08)
        pixel_label = MarkupText("pixel x′", font_size=22, color=FACE_COLOUR).next_to(pixel, LEFT, buff=0.1)
        direction = pixel.get_center() - lens_position
        direction = direction / numpy.linalg.norm(direction)
        ray = Line(lens_position, lens_position + direction * 7.0, color=FACE_COLOUR, stroke_opacity=0.7)
        self.at("pixel names a direction")
        self.play(FadeIn(lens), FadeIn(lens_label), Create(axis), FadeIn(axis_label), Create(image_plane), FadeIn(image_label), run_time=1.2)
        self.play(FadeIn(pixel), FadeIn(pixel_label), Create(ray), run_time=1.2)

        focal_arrow = double_arrow(numpy.array([lens_position[0], lens_position[1] - 1.55, 0]), numpy.array([image_x, lens_position[1] - 1.55, 0]))
        focal_label = label("f  (pixels)", 22).next_to(focal_arrow, DOWN, buff=0.06)
        self.at("focal length, f")
        self.play(FadeIn(focal_arrow), FadeIn(focal_label), run_time=0.8)
        offset_bracket = double_arrow(numpy.array([image_x + 0.22, lens_position[1], 0]), numpy.array([image_x + 0.22, lens_position[1] + pixel_offset_units, 0]), FACE_COLOUR)
        offset_label = MarkupText("x′ − c<sub>x</sub>", font_size=22, color=FACE_COLOUR).next_to(offset_bracket, RIGHT, buff=0.08)
        self.at("subtract the image centre")
        self.play(FadeIn(centre_mark), FadeIn(centre_label), FadeIn(offset_bracket), FadeIn(offset_label), run_time=1.0)

        equation_position = numpy.array([4.3, 2.4, 0.0])
        slope = formula("slope = (x′ − c<sub>x</sub>) / f", 28).move_to(equation_position)
        self.at("Divide by f")
        self.play(Write(slope), run_time=1.2)

        depth_x = lens_position[0] + distance_units
        point_position = lens_position + direction * (distance_units / direction[0])
        depth_plane = DashedLine(numpy.array([depth_x, lens_position[1] - 1.3, 0]), numpy.array([depth_x, 2.6, 0]), color=POINT_COLOUR, dash_length=0.12)
        distance_arrow = double_arrow(numpy.array([lens_position[0], lens_position[1] - 0.45, 0]), numpy.array([depth_x, lens_position[1] - 0.45, 0]), POINT_COLOUR)
        distance_label = label("distance Z  (metres)", 22, POINT_COLOUR).next_to(distance_arrow, DOWN, buff=0.06).shift(RIGHT * 1.6)
        room_point = Dot(point_position, color=POINT_COLOUR, radius=0.1)
        room_point_label = label("point", 22, POINT_COLOUR).next_to(room_point, UP + LEFT, buff=0.08)
        sideways = double_arrow(numpy.array([depth_x + 0.25, lens_position[1], 0]), numpy.array([depth_x + 0.25, point_position[1], 0]), POINT_COLOUR)
        sideways_label = label("X", 24, POINT_COLOUR).next_to(sideways, RIGHT, buff=0.08)
        self.at("Multiply by the distance")
        self.play(Create(depth_plane), FadeIn(distance_arrow), FadeIn(distance_label), run_time=1.0)
        self.play(FadeIn(room_point), FadeIn(room_point_label), FadeIn(sideways), FadeIn(sideways_label), run_time=1.0)
        equations = VGroup(
            formula("X = (x′ − c<sub>x</sub>) / f · Z", 28, POINT_COLOUR),
            formula("Y = −(y′ − c<sub>y</sub>) / f · Z", 28, POINT_COLOUR),
            formula("Z = distance", 28, POINT_COLOUR),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2).next_to(slope, DOWN, buff=0.35).align_to(slope, LEFT)
        self.play(Write(equations[0]), run_time=1.2)
        self.at("rows work the same")
        self.play(Write(equations[1]), run_time=1.2)
        rows_note = label("minus sign: rows count downward,\nY points up", 18, GREY_B).next_to(equations, DOWN, buff=0.3).align_to(equations, LEFT)
        self.at("with a minus sign")
        self.play(FadeIn(rows_note), run_time=0.8)
        self.at("pixel plus a distance")
        self.play(Write(equations[2]), run_time=0.8)
        self.play(Create(framed(VGroup(slope, equations), POINT_COLOUR)), run_time=0.8)
        self.finish_narration()
        self.clear_scene()

    def association_section(self):
        self.narrate("association")
        heading = label("Which colour pixel belongs to a depth pixel?   (seen from above)", 28).to_corner(UP + LEFT, buff=0.35)
        self.play(FadeIn(heading), run_time=0.6)
        depth_lens_position = numpy.array([-4.6, -2.3, 0.0])
        colour_lens_position = numpy.array([-2.6, -2.3, 0.0])
        image_height_units = 0.9                               # the image planes drawn in front of the lenses
        depth_lens = Dot(depth_lens_position, color=DEPTH_COLOUR, radius=0.09)
        colour_lens = Dot(colour_lens_position, color=KINECT_COLOUR, radius=0.09)
        depth_image = Line(depth_lens_position + numpy.array([-0.9, image_height_units, 0]), depth_lens_position + numpy.array([0.9, image_height_units, 0]), color=DEPTH_COLOUR, stroke_width=4)
        colour_image = Line(colour_lens_position + numpy.array([-0.9, image_height_units, 0]), colour_lens_position + numpy.array([0.9, image_height_units, 0]), color=KINECT_COLOUR, stroke_width=4)
        depth_name = label("depth lens", 20, DEPTH_COLOUR).next_to(depth_lens, DOWN, buff=0.12)
        colour_name = label("colour lens", 20, KINECT_COLOUR).next_to(colour_lens, DOWN, buff=0.12)
        offset = double_arrow(depth_lens_position + UP * 0.35, colour_lens_position + UP * 0.35)
        offset_name = label("5 cm", 20).next_to(offset, UP, buff=0.04)
        self.at("other lens")
        self.play(FadeIn(depth_lens), FadeIn(colour_lens), Create(depth_image), Create(colour_image),
                  FadeIn(depth_name), FadeIn(colour_name), run_time=1.2)
        self.play(FadeIn(offset), FadeIn(offset_name), run_time=0.8)

        steps = VGroup(
            MarkupText("1  lift:  (x, y), d  →  point", font_size=24, color=DEPTH_COLOUR),
            MarkupText("2  shift by the lens offset", font_size=24, color=WHITE),
            MarkupText("3  project:  point  →  (x′, y′)", font_size=24, color=KINECT_COLOUR),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to(numpy.array([3.2, 1.6, 0.0]))

        distance_units = ValueTracker(4.2)
        ray_slope = 0.22                                        # the depth pixel's ray: sideways per unit ahead
        def point_position():
            return depth_lens_position + numpy.array([ray_slope * distance_units.get_value(), distance_units.get_value(), 0.0])
        def depth_pixel_position():
            return depth_lens_position + numpy.array([ray_slope * image_height_units, image_height_units, 0.0])
        def colour_pixel_position():
            toward_point = point_position() - colour_lens_position
            return colour_lens_position + toward_point * (image_height_units / toward_point[1])
        depth_pixel = Dot(depth_pixel_position(), color=DEPTH_COLOUR, radius=0.08)
        depth_pixel_name = MarkupText("(x, y)", font_size=20, color=DEPTH_COLOUR).next_to(depth_pixel, LEFT, buff=0.1).shift(UP * 0.2)
        depth_ray = always_redraw(lambda: Line(depth_lens_position, point_position(), color=DEPTH_COLOUR, stroke_opacity=0.8))
        room_point = always_redraw(lambda: Dot(point_position(), color=POINT_COLOUR, radius=0.1))
        room_point_name = always_redraw(lambda: label("point", 20, POINT_COLOUR).next_to(room_point, UP, buff=0.1))
        distance_marker = always_redraw(lambda: double_arrow(depth_lens_position + LEFT * 1.3, depth_lens_position + LEFT * 1.3 + UP * distance_units.get_value(), DEPTH_COLOUR))
        distance_name = always_redraw(lambda: label("d", 22, DEPTH_COLOUR).next_to(distance_marker, LEFT, buff=0.06))
        self.at("First, the depth pixel")
        self.play(FadeIn(depth_pixel), FadeIn(depth_pixel_name), run_time=0.6)
        self.play(Create(depth_ray), FadeIn(distance_marker), FadeIn(distance_name), FadeIn(steps[0]), run_time=1.2)
        self.play(FadeIn(room_point), FadeIn(room_point_name), run_time=0.6)
        self.at("Second, the point is shifted")
        self.play(Indicate(offset), Indicate(offset_name), FadeIn(steps[1]), run_time=1.2)
        colour_ray = always_redraw(lambda: Line(point_position(), colour_lens_position, color=KINECT_COLOUR, stroke_opacity=0.8))
        colour_pixel = always_redraw(lambda: Dot(colour_pixel_position(), color=KINECT_COLOUR, radius=0.08))
        colour_pixel_name = always_redraw(lambda: MarkupText("(x′, y′)", font_size=20, color=KINECT_COLOUR).next_to(colour_pixel, RIGHT, buff=0.1).shift(UP * 0.2))
        self.at("Third, the point is projected")
        self.play(Create(colour_ray), FadeIn(steps[2]), run_time=1.2)
        self.play(FadeIn(colour_pixel), FadeIn(colour_pixel_name), run_time=0.6)

        parallax = label("same depth pixel, different distance:\nthe colour pixel moves", 22, FACE_COLOUR).next_to(steps, DOWN, buff=0.5).align_to(steps, LEFT)
        self.at("depends on distance")
        self.play(FadeIn(parallax), run_time=0.8)
        self.play(distance_units.animate.set_value(2.0), run_time=2.2, rate_func=rate_functions.ease_in_out_sine)
        self.play(distance_units.animate.set_value(5.2), run_time=2.2, rate_func=rate_functions.ease_in_out_sine)
        self.play(distance_units.animate.set_value(4.2), run_time=1.2, rate_func=rate_functions.ease_in_out_sine)
        result = label("every colour pixel gets its own distance", 21, POINT_COLOUR).next_to(parallax, DOWN, buff=0.4).align_to(steps, LEFT)
        self.at("Done for every depth pixel")
        self.play(FadeIn(result), run_time=0.8)
        self.finish_narration()
        self.clear_scene()

    def cloud_section(self):
        self.narrate("cloud")
        heading = label("The room as the system sees it: real data from the rig", 28).to_edge(UP, buff=0.35)
        self.play(FadeIn(heading), run_time=0.6)
        clip, outline = self.play_clip("panel_scene", 11.0, numpy.array([0.0, 0.7, 0.0]))
        legend = VGroup(
            label("depth cloud: one point (X, Y, Z) per pixel", 22, WHITE),
            label("Kinect", 22, "#4dd0e1"),
            label("projector and its beam", 22, "#e040fb"),
        ).arrange(RIGHT, buff=0.7).next_to(outline, DOWN, buff=0.2)
        self.at("depth cloud of the room")
        self.play(FadeIn(legend[0]), run_time=0.7)
        self.at("Kinect in cyan")
        self.play(FadeIn(legend[1]), run_time=0.7)
        self.at("projector in magenta")
        self.play(FadeIn(legend[2]), run_time=0.7)
        self.finish_narration()
        self.clear_scene()

    def projector_section(self):
        self.narrate("projector")
        heading = label("The projector: a camera running backwards", 28).to_corner(UP + LEFT, buff=0.35)
        self.play(FadeIn(heading), run_time=0.6)
        lens_position = numpy.array([5.6, -0.6, 0.0])
        image_x = 4.2
        image_plane = Line(numpy.array([image_x, -2.2, 0]), numpy.array([image_x, 1.0, 0]), color=PROJECTOR_COLOUR, stroke_width=4)
        lens = Dot(lens_position, color=PROJECTOR_COLOUR, radius=0.09)
        lens_label = label("projector lens", 20, PROJECTOR_COLOUR).next_to(lens, DOWN, buff=0.12)
        image_label = label("projector image", 20, PROJECTOR_COLOUR).next_to(image_plane, UP, buff=0.08)
        room_point = Dot(numpy.array([0.4, 0.9, 0.0]), color=POINT_COLOUR, radius=0.1)
        room_point_label = label("point (X, Y, Z)", 22, POINT_COLOUR).next_to(room_point, UP, buff=0.12)
        toward_point = room_point.get_center() - lens_position
        crossing = lens_position + toward_point * ((image_x - lens_position[0]) / toward_point[0])
        light = Line(lens_position, room_point.get_center(), color=PROJECTOR_COLOUR, stroke_opacity=0.8)
        pixel = Dot(crossing, color=FACE_COLOUR, radius=0.08)
        pixel_label = MarkupText("(x″, y″)", font_size=22, color=FACE_COLOUR).next_to(pixel, UP, buff=0.1).shift(RIGHT * 0.45)
        self.at("camera running backwards")
        self.play(FadeIn(lens), FadeIn(lens_label), Create(image_plane), FadeIn(image_label), run_time=1.0)
        self.play(FadeIn(room_point), FadeIn(room_point_label), Create(light), run_time=1.2)

        facts = VGroup(label("focal length f", 22), label("image centre (cx, cy)", 22),
                       label("pose: rotation R, translation t", 22)).arrange(DOWN, aligned_edge=LEFT, buff=0.18)
        facts.move_to(numpy.array([-4.4, 1.9, 0.0]))
        self.at("A focal length")
        self.play(FadeIn(facts[0]), run_time=0.6)
        self.at("an image centre")
        self.play(FadeIn(facts[1]), run_time=0.6)
        kinect_axes = VGroup(thin_arrow(ORIGIN, RIGHT * 0.8, KINECT_COLOUR, 0.25), thin_arrow(ORIGIN, UP * 0.8, KINECT_COLOUR, 0.25)).move_to(numpy.array([-1.2, -2.3, 0.0]))
        kinect_axes_label = label("Kinect frame", 18, KINECT_COLOUR).next_to(kinect_axes, LEFT, buff=0.1)
        projector_axes = VGroup(thin_arrow(ORIGIN, RIGHT * 0.8, PROJECTOR_COLOUR, 0.25), thin_arrow(ORIGIN, UP * 0.8, PROJECTOR_COLOUR, 0.25)).rotate(0.5).move_to(numpy.array([2.6, -2.3, 0.0]))
        projector_axes_label = label("projector frame", 18, PROJECTOR_COLOUR).next_to(projector_axes, RIGHT, buff=0.1)
        pose_arrow = thin_arrow(kinect_axes.get_right() + RIGHT * 0.15, projector_axes.get_left() + LEFT * 0.15, WHITE, 0.08)
        pose_label = label("R, t", 22).next_to(pose_arrow, UP, buff=0.06)
        self.at("and a pose")
        self.play(FadeIn(facts[2]), FadeIn(kinect_axes), FadeIn(kinect_axes_label), FadeIn(projector_axes), FadeIn(projector_axes_label), run_time=1.0)
        self.play(GrowArrow(pose_arrow), FadeIn(pose_label), run_time=0.8)

        equations = VGroup(
            formula("(X<sub>p</sub>, Y<sub>p</sub>, Z<sub>p</sub>) = R · (X, Y, Z) + t", 26),
            formula("x″ = f · X<sub>p</sub> / Z<sub>p</sub> + c<sub>x</sub>", 26, FACE_COLOUR),
            formula("y″ = f · Y<sub>p</sub> / Z<sub>p</sub> + c<sub>y</sub>", 26, FACE_COLOUR),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2).next_to(facts, DOWN, buff=0.5).align_to(facts, LEFT)
        self.at("Rotate and translate")
        self.play(Write(equations[0]), run_time=1.5)
        self.at("Divide by its distance")
        self.play(Write(equations[1]), Write(equations[2]), run_time=2.0)
        self.at("The result is")
        self.play(FadeIn(pixel), FadeIn(pixel_label), Create(framed(equations, FACE_COLOUR)), run_time=1.0)
        self.finish_narration()
        self.clear_scene()

    def coded_dots_section(self):
        self.narrate("coded_dots")
        heading = label("Calibration: the projector shows where its pixels land", 28).to_corner(UP + LEFT, buff=0.35)
        self.play(FadeIn(heading), run_time=0.6)
        screen = Rectangle(width=4.0, height=2.25, color=PROJECTOR_COLOUR).move_to(numpy.array([-4.4, 1.2, 0.0]))
        screen_label = label("projector image", 20, PROJECTOR_COLOUR).next_to(screen, UP, buff=0.08)
        dots = VGroup()
        for row in range(4):
            for column in range(7):
                dots.add(Dot(screen.get_corner(UP + LEFT) + numpy.array([0.35 + column * 0.55, -0.33 - row * 0.53, 0]), radius=0.06, color=WHITE))
        chosen_index = 2 * 7 + 4
        chosen = dots[chosen_index]
        self.at("grid of dots")
        self.play(Create(screen), FadeIn(screen_label), FadeIn(dots, lag_ratio=0.03), run_time=1.5)
        chosen_ring = Circle(radius=0.16, color=FACE_COLOUR).move_to(chosen)
        chosen_label = MarkupText("(x″, y″) = (1180, 620)", font_size=20, color=FACE_COLOUR).next_to(screen, DOWN, buff=0.12)
        self.at("known pixel")
        self.play(Create(chosen_ring), FadeIn(chosen_label), run_time=0.8)

        code_bits = [1, 0, 1, 1, 0, 1]
        strip = VGroup(*[Rectangle(width=0.42, height=0.42, color=GREY_B, stroke_width=2) for _ in code_bits]).arrange(RIGHT, buff=0.08)
        strip.move_to(numpy.array([-4.4, -1.35, 0.0]))
        strip_label = label("this dot, frame by frame: its code", 20, FACE_COLOUR).next_to(strip, DOWN, buff=0.12)
        self.at("blinks its own binary code")
        self.play(FadeIn(strip), FadeIn(strip_label), run_time=0.6)
        random_source = numpy.random.default_rng(7)
        for bit_index, bit in enumerate(code_bits):
            states = random_source.integers(0, 2, len(dots))
            states[chosen_index] = bit
            changes = [dot.animate.set_opacity(1.0 if state else 0.12) for dot, state in zip(dots, states)]
            cell_fill = strip[bit_index].animate.set_fill(WHITE if bit else BLACK, opacity=0.9 if bit else 0.0)
            self.play(*changes, cell_fill, run_time=0.35)
            bit_text = label(str(bit), 20, BLACK if bit else WHITE).move_to(strip[bit_index])
            self.add(bit_text)
            self.wait(0.25)
        self.play(*[dot.animate.set_opacity(1.0) for dot in dots], run_time=0.4)

        camera_frame = Rectangle(width=4.0, height=2.25, color=KINECT_COLOUR).move_to(numpy.array([0.9, 1.2, 0.0]))
        camera_label = label("colour camera image", 20, KINECT_COLOUR).next_to(camera_frame, UP, buff=0.08)
        seen = Dot(camera_frame.get_center() + numpy.array([0.7, 0.25, 0]), radius=0.08, color=WHITE)
        seen_ring = Circle(radius=0.18, color=FACE_COLOUR).move_to(seen)
        seen_label = MarkupText("seen at (x′, y′) = (1302, 418)", font_size=20, color=KINECT_COLOUR).next_to(camera_frame, DOWN, buff=0.12)
        decode = label("code 101101 → dot 32", 20, FACE_COLOUR).next_to(seen_label, DOWN, buff=0.12)
        self.at("camera sees a dot")
        self.play(Create(camera_frame), FadeIn(camera_label), FadeIn(seen), Create(seen_ring), FadeIn(seen_label), run_time=1.2)
        self.at("the code says")
        self.play(FadeIn(decode), run_time=0.8)
        room = MarkupText("depth → (X, Y, Z) = (0.28, 0.05, 0.79) m", font_size=20, color=POINT_COLOUR).next_to(decode, DOWN, buff=0.12)
        self.at("depth gives")
        self.play(FadeIn(room), run_time=0.8)

        pair = VGroup(
            label("one pair per dot", 24, WHITE),
            MarkupText("(X, Y, Z)  ↔  (x″, y″)", font_size=28, color=FACE_COLOUR),
            MarkupText("(0.28, 0.05, 0.79) m  ↔  (1180, 620)", font_size=18, color=GREY_B),
        ).arrange(DOWN, buff=0.18).move_to(numpy.array([5.0, 0.6, 0.0]))
        pair_frame = framed(pair, FACE_COLOUR, 0.15)
        self.at("each dot is one pair")
        self.play(FadeIn(pair), Create(pair_frame), run_time=1.2)
        self.finish_narration()
        self.clear_scene()

    def solve_section(self):
        self.narrate("solve")
        heading = label("Solving the projector model from the pairs", 28).to_corner(UP + LEFT, buff=0.35)
        self.play(FadeIn(heading), run_time=0.6)
        unknowns = VGroup(
            label("rotation R        3 numbers", 24),
            label("translation t     3 numbers", 24),
            label("focal length f    1 number", 24),
            label("7 unknowns", 26, FACE_COLOUR),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.16).move_to(numpy.array([-4.3, 1.9, 0.0]))
        self.at("seven unknowns")
        self.play(FadeIn(unknowns[:3], lag_ratio=0.3), run_time=1.5)
        self.play(FadeIn(unknowns[3]), run_time=0.6)
        equations_count = VGroup(
            MarkupText("each pair: 2 equations  (x″ and y″)", font_size=24),
            label("about 500 pairs: 1000 equations", 24),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.16).next_to(unknowns, DOWN, buff=0.3).align_to(unknowns, LEFT)
        self.at("Each pair gives two equations")
        self.play(FadeIn(equations_count[0]), run_time=0.8)
        self.at("A few hundred pairs")
        self.play(FadeIn(equations_count[1]), run_time=0.8)

        board = Rectangle(width=4.4, height=2.5, color=PROJECTOR_COLOUR).move_to(numpy.array([3.6, 1.7, 0.0]))
        board_label = label("projector image", 20, PROJECTOR_COLOUR).next_to(board, UP, buff=0.08)
        random_source = numpy.random.default_rng(11)
        measured_positions = []
        for row in range(4):
            for column in range(6):
                measured_positions.append(board.get_corner(UP + LEFT) + numpy.array([0.45 + column * 0.7, -0.4 - row * 0.57, 0]))
        measured_positions = numpy.array(measured_positions)
        outlier_indices = [5, 15]
        board_centre = board.get_center()
        def model_positions(misfit):
            """Where the model puts the dots: rotated, scaled and shifted away from the truth by the misfit."""
            angle = 0.18 * misfit
            scale = 1.0 + 0.22 * misfit
            rotation = numpy.array([[numpy.cos(angle), -numpy.sin(angle), 0], [numpy.sin(angle), numpy.cos(angle), 0], [0, 0, 1]])
            return (measured_positions - board_centre) @ rotation.T * scale + board_centre + numpy.array([0.25, -0.15, 0]) * misfit
        measured = VGroup(*[Dot(position, radius=0.055, color=WHITE) for position in measured_positions])
        for outlier_index in outlier_indices:
            measured[outlier_index].shift(numpy.array([random_source.uniform(0.3, 0.5), random_source.uniform(-0.4, 0.4), 0]))
        misfit = ValueTracker(1.0)
        def model_dots():
            positions = model_positions(misfit.get_value())
            group = VGroup()
            for position, measured_dot in zip(positions, measured):
                group.add(Line(measured_dot.get_center(), position, color=ERROR_COLOUR, stroke_width=2, stroke_opacity=0.7))
                group.add(Circle(radius=0.085, color=PROJECTOR_COLOUR, stroke_width=2.5).move_to(position))
            return group
        model = always_redraw(model_dots)
        legend = VGroup(VGroup(Dot(radius=0.055, color=WHITE), label("dots the projector drew", 18)).arrange(RIGHT, buff=0.12),
                        VGroup(Circle(radius=0.085, color=PROJECTOR_COLOUR, stroke_width=2.5), label("where the model puts them", 18, PROJECTOR_COLOUR)).arrange(RIGHT, buff=0.12),
                        ).arrange(DOWN, aligned_edge=LEFT, buff=0.1).next_to(board, DOWN, buff=0.12)
        self.at("the fit looks for")
        self.play(Create(board), FadeIn(board_label), FadeIn(measured), FadeIn(legend), run_time=1.0)
        self.add(model)
        self.play(misfit.animate.set_value(0.0), run_time=3.5, rate_func=rate_functions.ease_in_out_cubic)
        crosses = VGroup()
        for outlier_index in outlier_indices:
            centre = measured[outlier_index].get_center()
            crosses.add(Line(centre + numpy.array([-0.13, -0.13, 0]), centre + numpy.array([0.13, 0.13, 0]), color=ERROR_COLOUR, stroke_width=4),
                        Line(centre + numpy.array([-0.13, 0.13, 0]), centre + numpy.array([0.13, -0.13, 0]), color=ERROR_COLOUR, stroke_width=4))
        ignored = label("decoded wrongly: ignored", 18, ERROR_COLOUR).next_to(legend, DOWN, buff=0.1).align_to(legend, LEFT)
        self.at("ignores dots")
        self.play(Create(crosses), FadeIn(ignored), run_time=0.8)

        pixel_vector = text_matrix([["x″"], ["y″"], ["1"]], 22, FACE_COLOUR)
        intrinsics = text_matrix([["f", "0", "c<sub>x</sub>"], ["0", "f", "c<sub>y</sub>"], ["0", "0", "1"]], 22)
        pose = text_matrix([["r<sub>11</sub>", "r<sub>12</sub>", "r<sub>13</sub>", "t<sub>1</sub>"],
                            ["r<sub>21</sub>", "r<sub>22</sub>", "r<sub>23</sub>", "t<sub>2</sub>"],
                            ["r<sub>31</sub>", "r<sub>32</sub>", "r<sub>33</sub>", "t<sub>3</sub>"]], 22)
        room_vector = text_matrix([["X"], ["Y"], ["Z"], ["1"]], 22, POINT_COLOUR)
        matrix_row = VGroup(pixel_vector, label("∼", 28), intrinsics, pose, room_vector).arrange(RIGHT, buff=0.22)
        names = VGroup(label("K", 22, GREY_B).next_to(intrinsics, DOWN, buff=0.1), label("[ R | t ]", 22, GREY_B).next_to(pose, DOWN, buff=0.1))
        matrix_group = VGroup(matrix_row, names)
        matrix_group.move_to(numpy.array([-2.6, CONTENT_FLOOR + matrix_group.height / 2.0 + 0.05, 0.0]))
        scale_note = label("∼ : equal after dividing\nby the third row (the distance)", 18, GREY_B).next_to(matrix_group, RIGHT, buff=0.4)
        self.at("Written as one matrix")
        self.play(FadeIn(matrix_group), run_time=1.5)
        self.play(FadeIn(scale_note), run_time=0.8)
        self.finish_narration()
        self.clear_scene()

    def focal_section(self):
        self.narrate("focal")
        heading = label("Why the focal length matters: magnification", 28).to_corner(UP + LEFT, buff=0.35)
        self.play(FadeIn(heading), run_time=0.6)
        lens_position = numpy.array([-5.6, 0.2, 0.0])
        focal_units = 1.6
        object_distance_units = 7.2
        object_half_units = 1.3
        lens = Dot(lens_position, color=PROJECTOR_COLOUR, radius=0.09)
        lens_label = label("projector lens", 20, PROJECTOR_COLOUR).next_to(lens, UP, buff=0.25).shift(RIGHT * 0.3)
        axis = DashedLine(lens_position, lens_position + RIGHT * 8.2, color=GREY, dash_length=0.15)
        object_x = lens_position[0] + object_distance_units
        image_x = lens_position[0] + focal_units
        face_object = Line(numpy.array([object_x, lens_position[1] - object_half_units, 0]), numpy.array([object_x, lens_position[1] + object_half_units, 0]), color=FACE_COLOUR, stroke_width=6)
        object_label = label("face, size s", 22, FACE_COLOUR).next_to(face_object, RIGHT, buff=0.12).shift(UP * 0.5)
        image_half_units = object_half_units * focal_units / object_distance_units
        image_plane = Line(numpy.array([image_x, lens_position[1] - 1.0, 0]), numpy.array([image_x, lens_position[1] + 1.0, 0]), color=PROJECTOR_COLOUR, stroke_width=3, stroke_opacity=0.5)
        image_span = Line(numpy.array([image_x, lens_position[1] - image_half_units, 0]), numpy.array([image_x, lens_position[1] + image_half_units, 0]), color=FACE_COLOUR, stroke_width=7)
        image_label = label("pixels it covers", 20, FACE_COLOUR).next_to(image_plane, UP, buff=0.08)
        upper_ray = Line(lens_position, face_object.get_end(), color=PROJECTOR_COLOUR, stroke_opacity=0.7)
        lower_ray = Line(lens_position, face_object.get_start(), color=PROJECTOR_COLOUR, stroke_opacity=0.7)
        focal_arrow = double_arrow(numpy.array([lens_position[0], lens_position[1] - 1.3, 0]), numpy.array([image_x, lens_position[1] - 1.3, 0]))
        focal_label = label("f", 22).next_to(focal_arrow, DOWN, buff=0.06)
        distance_arrow = double_arrow(numpy.array([lens_position[0], lens_position[1] - 1.9, 0]), numpy.array([object_x, lens_position[1] - 1.9, 0]))
        distance_label = label("distance D", 22).next_to(distance_arrow, DOWN, buff=0.06)
        self.at("sets the magnification")
        self.play(FadeIn(lens), FadeIn(lens_label), Create(axis), Create(image_plane), FadeIn(focal_arrow), FadeIn(focal_label), run_time=1.2)
        self.at("An object of size s")
        self.play(Create(face_object), FadeIn(object_label), FadeIn(distance_arrow), FadeIn(distance_label), run_time=1.0)
        self.play(Create(upper_ray), Create(lower_ray), Create(image_span), FadeIn(image_label), run_time=1.2)
        relation = formula("pixels = f · s / D", 34, FACE_COLOUR).move_to(numpy.array([3.6, 3.1, 0.0]))
        self.at("covers f times s")
        self.play(Write(relation), run_time=1.2)
        example = VGroup(label("f = 2200 px,  s = 0.15 m,  D = 0.9 m", 22),
                         label("2200 × 0.15 / 0.9 ≈ 370 pixels", 24, FACE_COLOUR)).arrange(DOWN, aligned_edge=LEFT, buff=0.14).next_to(relation, DOWN, buff=0.25)
        self.at("A face fifteen centimetres")
        self.play(FadeIn(example), run_time=1.0)
        wrong = label("f wrong by 5%  →  image on the face 5% too big or too small", 22, ERROR_COLOUR).move_to(numpy.array([0.0, CONTENT_FLOOR + 0.2, 0.0]))
        self.at("If the model's focal length")
        self.play(FadeIn(wrong), run_time=0.8)
        self.finish_narration()
        self.clear_scene()

    def ambiguity_section(self):
        self.narrate("ambiguity")
        heading = label("The trap: a wall cannot fix the focal length", 28).to_corner(UP + LEFT, buff=0.35)
        self.play(FadeIn(heading), run_time=0.6)
        wall_height = 2.7
        near_object_height = 1.2
        fan_centre_x = 2.2
        wall = Line(numpy.array([-1.8, wall_height, 0]), numpy.array([6.4, wall_height, 0]), color=WALL_COLOUR, stroke_width=6)
        wall_label = label("wall", 22, WALL_COLOUR).next_to(wall, UP, buff=0.06)
        wall_dots = VGroup(*[Dot(numpy.array([x, wall_height, 0]), radius=0.07, color=POINT_COLOUR) for x in numpy.linspace(fan_centre_x - 2.4, fan_centre_x + 2.4, 7)])
        self.at("every dot lands on the wall")
        self.play(Create(wall), FadeIn(wall_label), FadeIn(wall_dots), run_time=1.0)
        near_projector = numpy.array([fan_centre_x, -0.7, 0.0])
        far_projector = numpy.array([fan_centre_x, -2.6, 0.0])
        def fan(source, colour):
            return VGroup(*[Line(source, dot.get_center(), color=colour, stroke_opacity=0.6) for dot in wall_dots])
        near_body = Rectangle(width=0.4, height=0.3, color=PROJECTOR_COLOUR, fill_opacity=0.5).move_to(near_projector)
        near_label = label("short focal, close", 22, PROJECTOR_COLOUR).next_to(near_body, RIGHT, buff=0.15)
        far_body = Rectangle(width=0.4, height=0.3, color=ERROR_COLOUR, fill_opacity=0.5).move_to(far_projector)
        far_label = label("long focal, far back", 22, ERROR_COLOUR).next_to(far_body, RIGHT, buff=0.15)
        self.at("short focal length")
        self.play(FadeIn(near_body), FadeIn(near_label), Create(fan(near_projector, PROJECTOR_COLOUR)), run_time=1.3)
        self.at("from a long one")
        self.play(FadeIn(far_body), FadeIn(far_label), Create(fan(far_projector, ERROR_COLOUR)), run_time=1.3)
        notes = VGroup(label("same dots on the wall", 24), label("different dots on a near object", 24, FACE_COLOUR),
                       label("near dots pin the focal length", 24, POINT_COLOUR)).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        notes.move_to(numpy.array([-4.4, 0.6, 0.0]))
        self.at("Both draw the same")
        self.play(FadeIn(notes[0]), Indicate(wall_dots, color=WHITE), run_time=1.0)

        near_object = Line(numpy.array([fan_centre_x - 1.7, near_object_height, 0]), numpy.array([fan_centre_x + 1.7, near_object_height, 0]), color=FACE_COLOUR, stroke_width=6)
        near_object_label = label("near object", 22, FACE_COLOUR).next_to(near_object, RIGHT, buff=0.12)
        def crossings(source, colour):
            group = VGroup()
            for dot in wall_dots:
                direction = dot.get_center() - source
                point = source + direction * ((near_object_height - source[1]) / direction[1])
                if abs(point[0] - fan_centre_x) <= 1.7:
                    group.add(Dot(point, radius=0.08, color=colour))
            return group
        near_crossings = crossings(near_projector, PROJECTOR_COLOUR)
        far_crossings = crossings(far_projector, ERROR_COLOUR)
        self.at("only disagree on a near object")
        self.play(Create(near_object), FadeIn(near_object_label), run_time=0.8)
        self.play(FadeIn(near_crossings), FadeIn(far_crossings), FadeIn(notes[1]), run_time=1.0)
        self.at("those dots pin")
        self.play(FadeIn(notes[2]), Indicate(near_crossings, color=POINT_COLOUR), run_time=1.2)
        self.finish_narration()
        self.clear_scene()

    def face_section(self):
        self.narrate("face")
        heading = label("The moving target: the face", 28).to_corner(UP + LEFT, buff=0.35)
        self.play(FadeIn(heading), run_time=0.6)
        self.at("MediaPipe finds")
        landmark_clip, landmark_outline = self.play_clip("landmarks", 4.6, numpy.array([-4.5, 1.55, 0.0]), outline_colour=KINECT_COLOUR)
        landmark_label = MarkupText("MediaPipe: 468 landmarks (x′, y′)", font_size=20, color=KINECT_COLOUR).next_to(landmark_outline, DOWN, buff=0.1)
        self.play(FadeIn(landmark_label), run_time=0.6)
        steps = VGroup(
            label("+ Kinect distance", 22, POINT_COLOUR),
            label("+ depth at the landmarks: relief", 22, POINT_COLOUR),
            label("= 468 points (X, Y, Z)", 24, WHITE),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2).move_to(numpy.array([0.15, 1.7, 0.0]))
        self.at("Kinect adds the distance")
        self.play(FadeIn(steps[0]), run_time=0.7)
        self.at("depth at the landmarks")
        self.play(FadeIn(steps[1]), run_time=0.7)
        self.at("each landmark becomes")
        self.play(FadeIn(steps[2]), run_time=0.7)

        vertices, triangles = load_canonical_face()
        scale = 0.105
        camera_frame = Rectangle(width=3.5, height=2.1, color=KINECT_COLOUR).move_to(numpy.array([-2.6, -1.85, 0.0]))
        camera_label = label("as the camera\nsees it", 18, KINECT_COLOUR).next_to(camera_frame, LEFT, buff=0.12)
        front = vertices[:, :2] * scale + camera_frame.get_center()[:2]
        front_mesh = mesh_edges(front, triangles).set_stroke(FACE_COLOUR, width=1.1, opacity=0.9)
        projector_frame = Rectangle(width=3.5, height=2.1, color=PROJECTOR_COLOUR).move_to(numpy.array([2.9, -1.85, 0.0]))
        projector_label = label("as the projector\nmust draw it", 18, PROJECTOR_COLOUR).next_to(projector_frame, RIGHT, buff=0.12)
        model_arrow = thin_arrow(camera_frame.get_right() + RIGHT * 0.15, projector_frame.get_left() + LEFT * 0.15, WHITE, 0.1)
        model_label = label("projector\nmodel", 18).next_to(model_arrow, UP, buff=0.06)
        rotated = vertices @ rotation_about_y(numpy.radians(-50.0)).T       # the projector looks from the side
        corners = rotated[triangles]
        normals = numpy.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
        facing_projector = normals[:, 2] > 0
        side = rotated[:, :2] * scale + projector_frame.get_center()[:2]
        visible_mesh = mesh_edges(side, triangles, facing_projector).set_stroke(FACE_COLOUR, width=1.1, opacity=0.9)
        hidden_mesh = mesh_edges(side, triangles, ~facing_projector).set_stroke(GREY, width=0.8, opacity=0.4)
        self.at("goes through the projector model")
        self.play(Create(camera_frame), FadeIn(camera_label), Create(front_mesh), run_time=1.5)
        self.at("That gives the mesh")
        self.play(GrowArrow(model_arrow), FadeIn(model_label), Create(projector_frame), FadeIn(projector_label), run_time=1.0)
        self.play(Create(hidden_mesh), Create(visible_mesh), run_time=1.8)
        self.at("face away from the projector")
        self.play(FadeOut(hidden_mesh), run_time=1.0)
        self.finish_narration()
        self.clear_scene()

    def texture_section(self):
        self.narrate("texture")
        heading = label("From a flat painting to the face: the UV map", 28).to_corner(UP + LEFT, buff=0.35)
        self.play(FadeIn(heading), run_time=0.6)
        clip, outline = self.play_clip("blender_paint", 6.6, numpy.array([-3.4, 0.9, 0.0]), outline_colour=FACE_COLOUR)
        clip_label = label("painting in Blender: the flat texture (left) and the mesh (right)", 18, FACE_COLOUR).next_to(outline, DOWN, buff=0.1)
        self.play(FadeIn(clip_label), run_time=0.6)

        vertices, triangles = load_canonical_face()
        texture_coordinates = load_canonical_texture_coordinates()
        square_side = 2.7
        square = Rectangle(width=square_side, height=square_side, color=FACE_COLOUR).move_to(numpy.array([2.1, 1.75, 0.0]))
        square_label = label("texture (U, V)", 20, FACE_COLOUR).next_to(square, UP, buff=0.08)
        flat = (texture_coordinates - 0.5) * square_side + square.get_center()[:2]
        flat_mesh = mesh_edges(flat, triangles).set_stroke(FACE_COLOUR, width=0.9, opacity=0.8)
        self.at("second address")
        self.play(Create(square), FadeIn(square_label), run_time=0.8)
        self.at("That layout is the U V map")
        self.play(Create(flat_mesh), run_time=2.0)
        unfolded = label("the face, unfolded", 20, GREY_B).next_to(square, DOWN, buff=0.08)
        self.at("the face, unfolded")
        self.play(FadeIn(unfolded), run_time=0.6)

        projector_frame = Rectangle(width=3.3, height=2.3, color=PROJECTOR_COLOUR).move_to(numpy.array([5.2, -1.6, 0.0]))
        projector_label = label("projector image", 20, PROJECTOR_COLOUR).next_to(projector_frame, UP, buff=0.08)
        rotated = vertices @ rotation_about_y(numpy.radians(-35.0)).T
        corners = rotated[triangles]
        normals = numpy.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
        facing_projector = normals[:, 2] > 0
        live = rotated[:, :2] * 0.115 + projector_frame.get_center()[:2]
        live_mesh = mesh_edges(live, triangles, facing_projector).set_stroke(FACE_COLOUR, width=0.9, opacity=0.8)
        self.at("At run time")
        self.play(Create(projector_frame), FadeIn(projector_label), Create(live_mesh), run_time=1.8)

        # The same few triangles, filled, in the flat texture and on the live mesh.
        visible = numpy.flatnonzero(facing_projector)
        areas = numpy.abs(numpy.cross(flat[triangles[visible, 1]] - flat[triangles[visible, 0]],
                                      flat[triangles[visible, 2]] - flat[triangles[visible, 0]]))
        chosen = visible[numpy.argsort(-areas)[[0, 8, 20]]]
        colours = [ERROR_COLOUR, POINT_COLOUR, KINECT_COLOUR]
        self.at("Each triangle of the square")
        for triangle_index, colour in zip(chosen, colours):
            triangle = triangles[triangle_index]
            flat_triangle = Polygon(*[numpy.array([flat[vertex][0], flat[vertex][1], 0.0]) for vertex in triangle],
                                    color=colour, fill_opacity=0.85, stroke_width=1.5)
            live_triangle = Polygon(*[numpy.array([live[vertex][0], live[vertex][1], 0.0]) for vertex in triangle],
                                    color=colour, fill_opacity=0.85, stroke_width=1.5)
            link = Line(flat_triangle.get_center(), live_triangle.get_center(), color=colour, stroke_width=1.5, stroke_opacity=0.6)
            self.play(FadeIn(flat_triangle), run_time=0.4)
            self.play(Create(link), FadeIn(live_triangle), run_time=0.8)
        note = label("same triangles, moved corners: the paint follows", 22, WHITE).move_to(numpy.array([-3.0, -2.3, 0.0]))
        self.at("Only the corners move")
        self.play(FadeIn(note), run_time=0.8)
        self.finish_narration()
        self.clear_scene()

    def error_section(self):
        self.narrate("error")
        heading = label("Where the error comes from   (seen from above)", 28).to_corner(UP + LEFT, buff=0.35)
        self.play(FadeIn(heading), run_time=0.6)
        place = top_view([-3.9, -2.2, 0.0], 4.4)
        face_distance_metres = 0.8
        depth_error_metres = 0.16                                   # drawn much larger than the real ~1 cm
        true_x_metres = 0.22
        baseline_metres = ValueTracker(0.7)

        kinect = Rectangle(width=0.36, height=0.18, color=KINECT_COLOUR, fill_opacity=0.5).move_to(place(0, 0))
        kinect_label = label("Kinect", 20, KINECT_COLOUR).next_to(kinect, LEFT, buff=0.12)
        def projector_position():
            return place(baseline_metres.get_value(), 0)
        projector = always_redraw(lambda: Rectangle(width=0.3, height=0.24, color=PROJECTOR_COLOUR, fill_opacity=0.5).move_to(projector_position()))
        projector_label = always_redraw(lambda: label("projector", 20, PROJECTOR_COLOUR).next_to(projector, RIGHT, buff=0.12))
        baseline = always_redraw(lambda: double_arrow(place(0.0, -0.09), place(baseline_metres.get_value(), -0.09)))
        baseline_label = always_redraw(lambda: label(f"baseline b = {baseline_metres.get_value():.1f} m", 20).next_to(baseline, DOWN, buff=0.06))
        surface = Line(place(-0.12, face_distance_metres), place(0.62, face_distance_metres), color=FACE_COLOUR, stroke_width=7)
        surface_label = label("the real face surface", 20, FACE_COLOUR).next_to(surface, RIGHT, buff=0.12)
        distance = double_arrow(place(-0.3, 0.0), place(-0.3, face_distance_metres))
        distance_label = label("d = 0.8 m", 20).next_to(distance, LEFT, buff=0.06)
        true_point = Dot(place(true_x_metres, face_distance_metres), radius=0.09, color=POINT_COLOUR)
        true_label = label("true point", 20, POINT_COLOUR).next_to(true_point, DOWN + LEFT, buff=0.08)
        camera_ray = Line(place(0, 0), true_point.get_center(), color=KINECT_COLOUR, stroke_opacity=0.8)
        self.play(FadeIn(kinect), FadeIn(kinect_label), FadeIn(projector), FadeIn(projector_label), FadeIn(baseline), FadeIn(baseline_label), run_time=1.0)
        self.at("Mostly from depth")
        self.play(Create(surface), FadeIn(surface_label), FadeIn(distance), FadeIn(distance_label), run_time=1.0)
        self.play(Create(camera_ray), FadeIn(true_point), FadeIn(true_label), run_time=1.0)

        believed_metres = face_distance_metres + depth_error_metres
        believed_position = place(true_x_metres * believed_metres / face_distance_metres, believed_metres)
        believed_point = Dot(believed_position, radius=0.09, color=ERROR_COLOUR)
        believed_label = label("believed point", 20, ERROR_COLOUR).next_to(believed_point, UP, buff=0.1)
        ray_extension = DashedLine(true_point.get_center(), believed_position, color=KINECT_COLOUR, dash_length=0.08)
        error_marker = double_arrow(place(-0.12, face_distance_metres) + LEFT * 0.25, place(-0.12, believed_metres) + LEFT * 0.25, ERROR_COLOUR)
        error_label = label("ΔZ", 22, ERROR_COLOUR).next_to(error_marker, LEFT, buff=0.06)
        self.at("Suppose the measured distance")
        self.play(FadeIn(error_marker), FadeIn(error_label), run_time=0.8)
        self.at("further along the camera's ray")
        self.play(Create(ray_extension), FadeIn(believed_point), FadeIn(believed_label), run_time=1.0)

        def landing_position():
            source = projector_position()
            toward_believed = believed_position - source
            return source + toward_believed * ((place(0, face_distance_metres)[1] - source[1]) / toward_believed[1])
        aim_line = always_redraw(lambda: DashedLine(landing_position(), believed_position, color=PROJECTOR_COLOUR, dash_length=0.08))
        light = always_redraw(lambda: Line(projector_position(), landing_position(), color=PROJECTOR_COLOUR, stroke_width=5))
        landing = always_redraw(lambda: Dot(landing_position(), radius=0.09, color=PROJECTOR_COLOUR))
        landing_label = always_redraw(lambda: label("light lands here", 20, PROJECTOR_COLOUR).next_to(landing, DOWN + RIGHT, buff=0.08))
        full_aim = DashedLine(projector_position(), believed_position, color=PROJECTOR_COLOUR, dash_length=0.08)
        self.at("aims the projector")
        self.play(Create(full_aim), run_time=1.2)
        self.at("the light stops")
        self.add(aim_line)
        self.remove(full_aim)
        self.add(light)
        self.wait(0.6)
        self.at("It lands beside")
        self.play(FadeIn(landing), FadeIn(landing_label), run_time=0.8)
        shift = always_redraw(lambda: Line(true_point.get_center(), landing_position(), color=ERROR_COLOUR, stroke_width=9))
        shift_label = label("shift", 20, ERROR_COLOUR).move_to(true_point.get_center() + UP * 0.32 + LEFT * 0.6)

        relation = formula("shift ≈ ΔZ · b / d", 34, ERROR_COLOUR).move_to(numpy.array([4.2, 2.7, 0.0]))
        self.at("The sideways shift")
        self.play(FadeIn(shift), FadeIn(shift_label), Write(relation), run_time=1.2)
        def numbers():
            shift_millimetres = 10.0 * baseline_metres.get_value() / face_distance_metres
            return VGroup(label(f"b = {baseline_metres.get_value():.1f} m,  d = 0.8 m", 22),
                          label(f"ΔZ = 1 cm  →  shift ≈ {shift_millimetres:.0f} mm", 24, ERROR_COLOUR),
                          ).arrange(DOWN, aligned_edge=LEFT, buff=0.14).next_to(relation, DOWN, buff=0.3)
        values = always_redraw(numbers)
        exaggerated = label("(ΔZ is drawn much larger than 1 cm)", 18, GREY_B).move_to(numpy.array([4.2, 0.95, 0.0]))
        self.at("With the projector seventy")
        self.play(FadeIn(values), FadeIn(exaggerated), run_time=1.0)
        self.at("Slide the projector")
        self.play(baseline_metres.animate.set_value(0.1), run_time=3.5, rate_func=rate_functions.ease_in_out_sine)
        closer = label("projector next to the Kinect:\ndepth errors barely show", 22, POINT_COLOUR).move_to(numpy.array([4.2, -1.0, 0.0]))
        self.play(FadeIn(closer), run_time=0.8)
        self.finish_narration()
        self.clear_scene()

    def loop_section(self):
        self.narrate("loop")
        heading = label("The feedback loop", 30).to_corner(UP + LEFT, buff=0.35)
        self.play(FadeIn(heading), run_time=0.6)
        loop_centre = numpy.array([0.0, 0.35, 0.0])
        horizontal_radius, vertical_radius = 4.9, 2.3
        stages = [
            ("the face moves", FACE_COLOUR, "The face moves"),
            ("colour camera\ncaptures a frame", KINECT_COLOUR, "colour camera captures"),
            ("MediaPipe: landmarks\n(x′, y′) = the target", KINECT_COLOUR, "MediaPipe finds"),
            ("depth camera: distance\n→ point (X, Y, Z)", POINT_COLOUR, "attaches a distance"),
            ("projector model\n→ pixel (x″, y″)", PROJECTOR_COLOUR, "projector model turns"),
            ("light lands\non the face", FACE_COLOUR, "light lands on the face"),
        ]
        angles = [numpy.pi / 2 - index * 2 * numpy.pi / len(stages) for index in range(len(stages))]
        nodes = []
        for (text, colour, _), angle in zip(stages, angles):
            position = loop_centre + numpy.array([horizontal_radius * numpy.cos(angle), vertical_radius * numpy.sin(angle), 0.0])
            caption = Text(text, font_size=20, color=colour)
            box = RoundedRectangle(corner_radius=0.12, width=3.1, height=0.95, color=colour, fill_color=BLACK, fill_opacity=0.6).move_to(position)
            caption.move_to(box)
            nodes.append(VGroup(box, caption))

        def edge_point(node, toward):
            """Where the line from a node's centre toward another point leaves the node's box."""
            box = node[0]
            direction = toward - box.get_center()
            direction = direction / numpy.linalg.norm(direction)
            reach = min(box.width / 2.0 / max(abs(direction[0]), 1e-6), box.height / 2.0 / max(abs(direction[1]), 1e-6))
            return box.get_center() + direction * (reach + 0.12)

        def link(first, second):
            return CurvedArrow(edge_point(first, second.get_center()), edge_point(second, first.get_center()),
                               angle=-0.45, color=GREY_B, stroke_width=3, tip_length=0.2)

        clip, outline = self.play_clip("result_dark", 3.3, loop_centre)
        previous = None
        for (_, _, cue), node in zip(stages, nodes):
            self.at(cue)
            animations = [FadeIn(node)]
            if previous is not None:
                animations.append(Create(link(previous, node)))
            self.play(*animations, run_time=0.9)
            previous = node
        closing_link = link(nodes[-1], nodes[0])
        self.at("the face moves again")
        self.play(Create(closing_link), Indicate(nodes[0], color=WHITE), run_time=1.2)
        runner = Dot(nodes[0].get_center(), radius=0.12, color=WHITE)
        rate = label("15 loops per second", 22, WHITE).next_to(outline, DOWN, buff=0.1)
        self.at("fifteen times a second")
        self.play(FadeIn(rate), FadeIn(runner), run_time=0.6)
        for _ in range(2):
            for node in nodes[1:] + [nodes[0]]:
                self.play(runner.animate.move_to(node.get_center()), run_time=0.22, rate_func=rate_functions.linear)
        self.play(FadeOut(runner), run_time=0.3)
        prediction = label("one pass takes a fraction of a second: aim where the face will be", 22, FACE_COLOUR).move_to(numpy.array([0.0, CONTENT_FLOOR + 0.05, 0.0]))
        self.at("One pass takes")
        self.play(FadeIn(prediction), run_time=0.8)
        self.finish_narration()
        self.clear_scene()

    def closing_section(self):
        self.narrate("closing")
        clip, outline = self.play_clip("result_dark", 6.0, numpy.array([0.0, 1.45, 0.0]))
        chain = VGroup(label("camera pixel", 26, KINECT_COLOUR), thin_arrow(LEFT * 0.4, RIGHT * 0.4),
                       label("point in the room", 26, POINT_COLOUR), thin_arrow(LEFT * 0.4, RIGHT * 0.4),
                       label("projector pixel", 26, PROJECTOR_COLOUR), thin_arrow(LEFT * 0.4, RIGHT * 0.4),
                       label("the face", 26, FACE_COLOUR)).arrange(RIGHT, buff=0.25).move_to(numpy.array([0.0, -0.85, 0.0]))
        self.play(FadeIn(chain, lag_ratio=0.2), run_time=3.0)
        credit = ManimText(AUTHOR_CREDIT, font_size=28, color=WHITE).move_to(numpy.array([0.0, -1.6, 0.0]))
        self.play(FadeIn(credit), run_time=1.0)
        self.finish_narration(tail_seconds=0.6)
        animation_credit = VGroup(
            Text("Animated with Manim, the mathematical animation library", font_size=22, color=GREY_B),
            Text("created by Grant Sanderson of 3Blue1Brown  (Manim Community edition)", font_size=22, color=GREY_B),
        ).arrange(DOWN, buff=0.12).move_to(numpy.array([0.0, -2.45, 0.0]))
        self.play(FadeIn(animation_credit), run_time=1.0)
        self.wait(4.5)
        self.clear_scene()
