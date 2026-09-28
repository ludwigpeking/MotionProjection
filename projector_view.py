"""The projector's-eye view: the Kinect depth cloud, coloured by the Kinect colour
image, pushed through the projector calibration into projector pixel space.

What the projector would see if it were a camera. A wrong calibration shows as
the scene landing in the wrong part of the frame or at the wrong size; a disc
drawn at a projector pixel should sit on the right surface here.
"""

import cv2
import numpy


DEPTH_COLOUR_NEAR_METRES = 0.4
DEPTH_COLOUR_FAR_METRES = 3.0
NO_DEPTH_COLOUR = (70, 70, 70)     # grey: the sensor has no depth there (too close, occluded, or absorbing)


def depth_colours(depth_metres):
    """BGR colours for depths: blue near .. red far (JET) between the range limits,
    grey where the depth is unknown."""
    depth_metres = numpy.asarray(depth_metres, dtype=numpy.float32).ravel()
    known = numpy.isfinite(depth_metres) & (depth_metres > 0)
    scaled = numpy.zeros_like(depth_metres)
    scaled[known] = (depth_metres[known] - DEPTH_COLOUR_NEAR_METRES) / (DEPTH_COLOUR_FAR_METRES - DEPTH_COLOUR_NEAR_METRES)
    index = numpy.clip(scaled * 255.0, 0, 255).astype(numpy.uint8)
    colours = cv2.applyColorMap(index.reshape(-1, 1), cv2.COLORMAP_JET).reshape(-1, 3)
    colours[~known] = NO_DEPTH_COLOUR
    return colours


def depth_image(depth_millimetres):
    """The depth alone as an image: blue near, red far, grey where there is no depth, with a caption."""
    depth_metres = numpy.asarray(depth_millimetres, dtype=numpy.float32) / 1000.0
    image = depth_colours(depth_metres).reshape(depth_metres.shape[0], depth_metres.shape[1], 3).copy()
    cv2.putText(image, f"depth: blue {DEPTH_COLOUR_NEAR_METRES:.1f} m .. red {DEPTH_COLOUR_FAR_METRES:.1f} m, grey = no depth",
                (10, image.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return image


def depth_overlay(view, depth_millimetres, strength=0.45):
    """Blend the depth image over a camera view of the same aspect, in place: blue near, red
    far, grey where there is no depth. Depth edges that do not sit on the colour edges show
    a registration problem."""
    depth_metres = numpy.asarray(depth_millimetres, dtype=numpy.float32) / 1000.0
    colours = depth_colours(depth_metres).reshape(depth_metres.shape[0], depth_metres.shape[1], 3)
    colours = cv2.resize(colours, (view.shape[1], view.shape[0]), interpolation=cv2.INTER_NEAREST)
    view[:] = (view.astype(numpy.float32) * (1.0 - strength) + colours.astype(numpy.float32) * strength).astype(numpy.uint8)
    cv2.putText(view, f"depth overlay: blue {DEPTH_COLOUR_NEAR_METRES:.1f} m .. red {DEPTH_COLOUR_FAR_METRES:.1f} m, grey = no depth",
                (10, view.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return view


def render_projector_view(camera, depth_millimetres, frame_bgr, calibration, output_width, output_height,
                          colour_by_depth=False):
    """(output_height, output_width, 3) BGR image of the scene in projector coordinates,
    scaled from the projector's own resolution. Unseen areas stay black. With
    colour_by_depth the points are painted by their distance from the Kinect instead
    of by the camera image, which shows whether the 3D shape is plausible."""
    colour_coordinates, points_3d = camera._map_depth_frame(depth_millimetres)
    valid = numpy.isfinite(points_3d[:, 2]) & numpy.isfinite(colour_coordinates[:, 0])
    points = points_3d[valid].astype(numpy.float64)
    if colour_by_depth:
        colours = depth_colours(points[:, 2])
    else:
        colour_columns = numpy.clip(colour_coordinates[valid, 0].astype(int), 0, camera.frame_width - 1)
        colour_rows = numpy.clip(colour_coordinates[valid, 1].astype(int), 0, camera.frame_height - 1)
        colours = frame_bgr[colour_rows, colour_columns]

    # The projector magnifies the depth pixels about four times at the output size, so the
    # cloud is painted at a quarter of the output (where neighbouring depth pixels touch),
    # gaps are closed there, and the result is scaled up.
    render_width, render_height = output_width // 4, output_height // 4
    view = numpy.zeros((render_height, render_width, 3), dtype=numpy.uint8)
    if len(points) == 0:
        return cv2.resize(view, (output_width, output_height), interpolation=cv2.INTER_NEAREST)
    projected = calibration.project(points)
    depths = calibration.depths(points)
    scale_x = render_width / calibration.projector_width
    scale_y = render_height / calibration.projector_height
    columns = numpy.round(projected[:, 0] * scale_x).astype(int)
    rows = numpy.round(projected[:, 1] * scale_y).astype(int)
    inside = (depths > 0) & (columns >= 0) & (columns < render_width) & (rows >= 0) & (rows < render_height)
    columns, rows, depths, colours = columns[inside], rows[inside], depths[inside], colours[inside]
    # Far points first, near points last: the last write wins, so nearer surfaces occlude.
    order = numpy.argsort(-depths)
    view[rows[order], columns[order]] = colours[order]
    view = cv2.morphologyEx(view, cv2.MORPH_CLOSE, numpy.ones((3, 3), dtype=numpy.uint8))
    return cv2.resize(view, (output_width, output_height), interpolation=cv2.INTER_LINEAR)


def beam_coverage_mask(camera, depth_millimetres, calibration, view_width, view_height):
    """Boolean (view_height, view_width) mask of the camera view: True where the surface
    the Kinect sees there lies inside the projector's image, i.e. where a projected
    dot can land. Built from the depth cloud, so it follows the scene's shape."""
    colour_coordinates, points_3d = camera._map_depth_frame(depth_millimetres)
    valid = numpy.isfinite(points_3d[:, 2]) & numpy.isfinite(colour_coordinates[:, 0])
    points = points_3d[valid].astype(numpy.float64)
    coordinates = colour_coordinates[valid]
    mask_width, mask_height = view_width // 4, view_height // 4
    mask = numpy.zeros((mask_height, mask_width), dtype=numpy.uint8)
    if len(points) == 0:
        return numpy.zeros((view_height, view_width), dtype=bool)
    projected = calibration.project(points)
    depths = calibration.depths(points)
    inside = ((depths > 0) & (projected[:, 0] >= 0) & (projected[:, 0] < calibration.projector_width)
              & (projected[:, 1] >= 0) & (projected[:, 1] < calibration.projector_height))
    columns = numpy.round(coordinates[inside, 0] * mask_width / camera.frame_width).astype(int)
    rows = numpy.round(coordinates[inside, 1] * mask_height / camera.frame_height).astype(int)
    keep = (columns >= 0) & (columns < mask_width) & (rows >= 0) & (rows < mask_height)
    mask[rows[keep], columns[keep]] = 1
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, numpy.ones((3, 3), dtype=numpy.uint8))
    return cv2.resize(mask, (view_width, view_height), interpolation=cv2.INTER_NEAREST).astype(bool)


def tint_beam_coverage(view, mask, colour=(0, 200, 0), strength=0.25):
    """Tint the camera view where the projector can reach (in place)."""
    tinted = view[mask].astype(numpy.float32) * (1.0 - strength) + numpy.array(colour, dtype=numpy.float32) * strength
    view[mask] = tinted.astype(numpy.uint8)
    return view


CLOUD_STEP_PIXELS = 6


def publish_cloud(panel, camera, depth_millimetres, frame_bgr):
    """Hand the panel a subsampled coloured depth cloud for its 3D view."""
    points = camera.map_depth_to_colour_points(depth_millimetres)[::CLOUD_STEP_PIXELS, ::CLOUD_STEP_PIXELS]
    colours = frame_bgr[::CLOUD_STEP_PIXELS, ::CLOUD_STEP_PIXELS]
    panel.publish_cloud(points.reshape(-1, 3), colours.reshape(-1, 3),
                        getattr(camera, "colour_focal_x", 1064.0), getattr(camera, "colour_focal_y", 1064.0),
                        getattr(camera, "colour_centre_x", camera.frame_width / 2.0),
                        getattr(camera, "colour_centre_y", camera.frame_height / 2.0),
                        camera.frame_width, camera.frame_height)


def draw_projector_marker(view, projector_pixel, calibration, colour=(0, 0, 255), radius=12):
    """Ring at a projector pixel on a view rendered by render_projector_view."""
    scale_x = view.shape[1] / calibration.projector_width
    scale_y = view.shape[0] / calibration.projector_height
    centre = (int(round(projector_pixel[0] * scale_x)), int(round(projector_pixel[1] * scale_y)))
    cv2.circle(view, centre, radius, colour, 2)
    return view
