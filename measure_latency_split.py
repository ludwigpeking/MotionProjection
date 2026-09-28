"""Where does the display delay come from: the camera, or the projector?

The calibration measures one number, from "picture shown" to "picture seen in a Kinect
frame". That number holds the graphics path, the projector's own processing, the camera's
exposure and the USB transfer together. This tool separates them with a reference: a
window on the computer's own screen, flashed at the same instant as the projector.

    .venv\\Scripts\\python.exe measure_latency_split.py

Set-up: the Kinect must see both the projector's beam (on a wall or any surface) and the
computer's screen. Turning the screen toward the Kinect is enough. Keep the scene still.

What it reports:
    reference screen   time until the Kinect sees the computer's screen change.
                       A laptop panel answers in about 10 ms, so this is close to the
                       camera's own delay: exposure, transfer, and the wait for a frame.
    projector          time until the Kinect sees the projector change.
    difference         what the projector path costs on top: the projector's processing
                       and the second display output.

Closing: q or Esc in either window, the close button of the reference window, or a
double-click on the projector window.
"""

import argparse
import random
import time

import cv2
import numpy

import config
from displays import pick_projector_monitor
from kinect_camera import open_camera
from panel_link import PanelLink
from projector import ProjectorWindow

REFERENCE_WINDOW_NAME = "latency reference (keep visible to the Kinect)"
REFERENCE_WIDTH, REFERENCE_HEIGHT = 900, 600
SMALL_WIDTH, SMALL_HEIGHT = 480, 270
MINIMUM_STEP = 12.0                    # brightness (0..255) a flash must add inside its region
MINIMUM_REGION_FRACTION = 0.002        # of the Kinect image


class Stopped(Exception):
    pass


def small_gray(frame):
    return cv2.cvtColor(cv2.resize(frame, (SMALL_WIDTH, SMALL_HEIGHT), interpolation=cv2.INTER_AREA),
                        cv2.COLOR_BGR2GRAY).astype(numpy.float32)


class Displays:
    """The projector and the reference window, switched together."""

    def __init__(self, projector, panel):
        self.projector = projector
        self.panel = panel
        self.black_projector = numpy.zeros((projector.height, projector.width, 3), dtype=numpy.uint8)
        self.white_projector = numpy.full((projector.height, projector.width, 3), 255, dtype=numpy.uint8)
        self.black_reference = numpy.zeros((REFERENCE_HEIGHT, REFERENCE_WIDTH, 3), dtype=numpy.uint8)
        self.white_reference = numpy.full((REFERENCE_HEIGHT, REFERENCE_WIDTH, 3), 255, dtype=numpy.uint8)
        cv2.namedWindow(REFERENCE_WINDOW_NAME, cv2.WINDOW_AUTOSIZE)          # an ordinary window with a close button
        cv2.moveWindow(REFERENCE_WINDOW_NAME, 60, 60)
        self.reference_shown = False

    def show(self, projector_white, reference_white):
        """Returns the moment both pictures were handed to the display system."""
        cv2.imshow(REFERENCE_WINDOW_NAME, self.white_reference if reference_white else self.black_reference)
        self.projector.show(self.white_projector if projector_white else self.black_projector)
        self.reference_shown = True
        self.pump()
        return time.perf_counter()

    def pump(self):
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27) or self.panel.command() == "quit":
            raise Stopped("stopped by key")
        if self.reference_shown and cv2.getWindowProperty(REFERENCE_WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            raise Stopped("the reference window was closed")


def frames_for(camera, displays, seconds, after):
    """(time since `after`, small grey image) of every frame of the next `seconds`."""
    samples = []
    last_timestamp = after
    while time.perf_counter() - after < seconds:
        result = camera.frame_after(last_timestamp, timeout_seconds=1.0)
        displays.pump()
        if result is None:
            continue
        last_timestamp, frame = result
        samples.append((last_timestamp - after, small_gray(frame)))
    return samples


def settled_image(camera, displays, projector_white, reference_white, latency_full_after):
    displays.show(projector_white, reference_white)
    samples = frames_for(camera, displays, 0.6, time.perf_counter() + latency_full_after + 0.4)
    if not samples:
        raise Stopped("the Kinect delivered no frames")
    return numpy.mean([image for _, image in samples], axis=0)


def crossing_time(samples, region, low, high):
    """When the region's brightness crossed half of its step, interpolated between the two
    frames around the crossing; None when it never did."""
    half = low + 0.5 * (high - low)
    previous_time, previous_value = 0.0, low
    for elapsed, image in samples:
        value = float(image[region].mean())
        if value >= half:
            if value == previous_value:
                return elapsed
            fraction = (half - previous_value) / (value - previous_value)
            return previous_time + float(numpy.clip(fraction, 0.0, 1.0)) * (elapsed - previous_time)
        previous_time, previous_value = elapsed, value
    return None


def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--flashes", type=int, default=10)
    argument_parser.add_argument("--panel", action="store_true", help="publish the view to the web panel")
    arguments = argument_parser.parse_args()
    panel = PanelLink(config.debug_image_directory, arguments.panel)
    monitor, is_second_screen = pick_projector_monitor(
        config.projector_monitor_index, config.projector_fallback_width_pixels, config.projector_fallback_height_pixels)
    if not is_second_screen:
        raise SystemExit("[latency] no second screen found: the projector must be connected as an extended display")
    camera = open_camera(config)
    projector = ProjectorWindow(monitor, fullscreen=True, background_fraction=0.0)
    displays = Displays(projector, panel)
    latency_full_after = config.latency_full_after_seconds
    try:
        dark = settled_image(camera, displays, False, False, latency_full_after)
        beam_lit = settled_image(camera, displays, True, False, latency_full_after)
        screen_lit = settled_image(camera, displays, False, True, latency_full_after)
        displays.show(False, False)
        beam_step = cv2.GaussianBlur(beam_lit - dark, (5, 5), 0)
        screen_step = cv2.GaussianBlur(screen_lit - dark, (5, 5), 0)
        screen_region = (screen_step > MINIMUM_STEP) & (screen_step > 2.0 * beam_step)
        beam_region = (beam_step > MINIMUM_STEP) & (beam_step > 2.0 * screen_step)
        print(f"[latency] the Kinect sees the projector's beam in {100.0 * beam_region.mean():.1f}% of its image "
              f"and the reference screen in {100.0 * screen_region.mean():.1f}%")

        _, frame = camera.latest()
        view = cv2.resize(frame, (960, 540), interpolation=cv2.INTER_AREA)
        for region, colour in ((beam_region, (255, 0, 255)), (screen_region, (0, 255, 255))):
            mask = cv2.resize(region.astype(numpy.uint8), (960, 540), interpolation=cv2.INTER_NEAREST).astype(bool)
            view[mask] = (0.5 * view[mask] + 0.5 * numpy.array(colour)).astype(numpy.uint8)
        cv2.putText(view, "magenta: projector beam   yellow: reference screen", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)
        panel.publish(view, force=True)

        if beam_region.mean() < MINIMUM_REGION_FRACTION:
            raise SystemExit("[latency] the Kinect does not see the projector's beam: point the projector into its view")
        if screen_region.mean() < MINIMUM_REGION_FRACTION:
            raise SystemExit("[latency] the Kinect does not see the reference window: turn the computer's screen toward "
                             "the Kinect, with the reference window visible on it")

        beam_low, beam_high = float(dark[beam_region].mean()), float(beam_lit[beam_region].mean())
        screen_low, screen_high = float(dark[screen_region].mean()), float(screen_lit[screen_region].mean())
        projector_times, reference_times = [], []
        for flash in range(arguments.flashes):
            displays.show(False, False)
            frames_for(camera, displays, 0.5 + random.uniform(0.0, 0.1), time.perf_counter())     # not in step with the camera
            shown_at = displays.show(True, True)
            samples = frames_for(camera, displays, 1.0, shown_at)
            projector_time = crossing_time(samples, beam_region, beam_low, beam_high)
            reference_time = crossing_time(samples, screen_region, screen_low, screen_high)
            if projector_time is None or reference_time is None:
                print(f"[latency] flash {flash + 1}: not seen (stream stall?)")
                continue
            projector_times.append(projector_time)
            reference_times.append(reference_time)
            print(f"[latency] flash {flash + 1}: reference screen {reference_time * 1000:4.0f} ms, "
                  f"projector {projector_time * 1000:4.0f} ms, difference {(projector_time - reference_time) * 1000:+4.0f} ms")
        displays.show(False, False)
        if len(projector_times) < 3:
            raise SystemExit("[latency] too few flashes were seen for a result")
        reference = numpy.array(reference_times) * 1000.0
        projected = numpy.array(projector_times) * 1000.0
        difference = projected - reference
        print(f"[latency] result over {len(difference)} flashes (median, with the spread between the quartiles):")
        for name, values in (("reference screen (about the camera's own delay)", reference),
                             ("projector", projected), ("difference (what the projector path adds)", difference)):
            low, middle, high = numpy.percentile(values, [25, 50, 75])
            print(f"[latency]   {name:48s} {middle:5.0f} ms   ({low:.0f} .. {high:.0f})")
        print(f"[latency] the Kinect delivered {camera.frames_per_second():.0f} frames per second: one frame is "
              f"{1000.0 / max(camera.frames_per_second(), 1.0):.0f} ms, which limits how finely any of this can be timed")
    except Stopped as stop:
        print(f"[latency] {stop}")
    finally:
        try:
            projector.show(displays.black_projector)
            cv2.waitKey(1)
        except SystemExit:
            pass
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
