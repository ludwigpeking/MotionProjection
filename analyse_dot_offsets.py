"""Systematic error of the projection, from the nose-dot test.

kinect_red_dot.py logs, for every frame in which the camera sees the projected dot, where
the dot landed relative to the landmark it was aimed at (diagnostics/kinect_dot_offsets.csv).
This script keeps the frames in which the head was still (no latency error in them) and
reports the constant part of the error and how it changes with the position in the room.

    .venv\\Scripts\\python.exe analyse_dot_offsets.py

For a useful result, hold still for a few seconds at several places: near and far,
left and right, higher and lower, always facing the Kinect inside the beam.
"""

import argparse
import os

import numpy

import config


def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--log", type=str,
                                 default=os.path.join(config.debug_image_directory, config.kinect_dot_offsets_file_name))
    argument_parser.add_argument("--still-speed", type=float, default=config.kinect_red_dot_still_speed_metres_per_second)
    arguments = argument_parser.parse_args()

    table = numpy.genfromtxt(arguments.log, delimiter=",", names=True)
    table = numpy.atleast_1d(table)
    if len(table) == 0:
        raise SystemExit("[offsets] the log holds no frames: the face was never found with a depth")
    drawn = table["dot_drawn"] > 0
    seen = numpy.isfinite(table["dot_column_pixels"])
    print(f"[offsets] {len(table)} frames with a face, over {table['time_seconds'].max() - table['time_seconds'].min():.0f} s; "
          f"dot drawn (landmark inside the projector image) in {int(drawn.sum())}; dot seen by the camera in {int(seen.sum())}")
    if drawn.any() and not seen.any():
        contrast = table["strongest_colour_contrast"][drawn]
        raise SystemExit(f"[offsets] the dot was drawn but never recognised: strongest colour contrast median "
                         f"{numpy.median(contrast):.0f}, highest {contrast.max():.0f} (a detection needs 40). "
                         "Dim the room or enlarge the dot.")
    if not drawn.any():
        raise SystemExit(f"[offsets] the landmark was never inside the projector image: aimed projector pixel "
                         f"x {table['projector_x_pixels'].min():.0f} .. {table['projector_x_pixels'].max():.0f}, "
                         f"y {table['projector_y_pixels'].min():.0f} .. {table['projector_y_pixels'].max():.0f} "
                         f"(image is {config.projector_fallback_width_pixels} x {config.projector_fallback_height_pixels})")
    table = table[seen]
    still = table[table["speed_metres_per_second"] < arguments.still_speed]
    print(f"[offsets] {len(still)} of them with the head still (under {arguments.still_speed * 100:.0f} cm/s)")
    if len(still) < 5:
        raise SystemExit("[offsets] too few still frames: hold still for a few seconds at each position")

    position = numpy.column_stack([still["landmark_x_metres"], still["landmark_y_metres"], still["landmark_z_metres"]])
    offset = numpy.column_stack([still["offset_right_millimetres"], still["offset_down_millimetres"]])
    print(f"[offsets] positions covered: X {position[:, 0].min():+.2f} .. {position[:, 0].max():+.2f} m, "
          f"Y {position[:, 1].min():+.2f} .. {position[:, 1].max():+.2f} m, Z {position[:, 2].min():.2f} .. {position[:, 2].max():.2f} m")
    print(f"[offsets] constant part: the dot lands {numpy.median(offset[:, 0]):+.1f} mm right and "
          f"{numpy.median(offset[:, 1]):+.1f} mm down of the landmark (spread {offset[:, 0].std():.1f} / {offset[:, 1].std():.1f} mm)")

    spans = position.max(axis=0) - position.min(axis=0)
    varied = spans > 0.08                                   # an axis along which the head moved at least 8 cm
    if not varied.any():
        print("[offsets] the head stayed in one place: no statement on how the error changes with position")
        return
    axis_names = numpy.array(["X (sideways)", "Y (height)", "Z (distance)"])
    centred = position[:, varied] - position[:, varied].mean(axis=0)
    design = numpy.column_stack([numpy.ones(len(still)), centred])
    for column, direction in ((0, "right"), (1, "down")):
        coefficients, _, _, _ = numpy.linalg.lstsq(design, offset[:, column], rcond=None)
        remaining = offset[:, column] - design @ coefficients
        slopes = ", ".join(f"{slope:+.0f} mm per metre of {name}" for slope, name in zip(coefficients[1:], axis_names[varied]))
        print(f"[offsets] {direction:5s}: {coefficients[0]:+.1f} mm at the middle of the covered positions; {slopes}; "
              f"unexplained {remaining.std():.1f} mm")
    print("[offsets] reading: a constant offset points at the landmark or the dot centre; an offset that grows with "
          "distance Z points at the projector's focal length or position; one that grows sideways points at its rotation")

    print("[offsets] per position (20 cm cells):")
    cells = {}
    for place, value in zip(position, offset):
        key = tuple(numpy.round(place / 0.2).astype(int))
        cells.setdefault(key, []).append(value)
    for key in sorted(cells):
        values = numpy.array(cells[key])
        print(f"           X {key[0] * 0.2:+.1f}  Y {key[1] * 0.2:+.1f}  Z {key[2] * 0.2:.1f} m: "
              f"{values[:, 0].mean():+6.1f} mm right, {values[:, 1].mean():+6.1f} mm down  ({len(values)} frames)")


if __name__ == "__main__":
    main()
