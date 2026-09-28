"""Dense structured-light scan of a static scene: which projector pixel lights each camera pixel.

The projector shows stripe patterns whose on/off sequence spells, in Gray code, the column
(then the row) of every projector stripe. Each pattern is shown together with its inverse,
and a camera pixel reads a bit as "pattern brighter than inverse", which needs no
brightness threshold and cancels the room's own light. A pixel whose two readings are
too close is unreliable for that bit.

The projector is focused at one distance, so on nearer objects fine stripes wash out
while coarse ones survive. The scan therefore works on several levels: level 0 uses
every bit (cells of finest_stripe_pixels), level 1 drops the finest bit (cells twice as
wide), and so on. A surface is described by the finest level whose cells are complete
there.

A correspondence is one cell: the centre of the projector cell, and the centroid of the
camera pixels that decoded to it (with their 3D points). A blurred cell's centroid is
still the cell's centre, because defocus blurs symmetrically.
"""

import math
import time

import cv2
import numpy


def gray_code_patterns(projector_width, projector_height, finest_stripe_pixels):
    """Returns (bit_count, column_patterns, row_patterns): lists of uint8 images, most
    significant bit first. Both axes use the same bit count, so a level means the same
    cell size along both."""
    largest_side = max(projector_width, projector_height)
    bit_count = max(1, math.ceil(math.log2(largest_side / float(finest_stripe_pixels))))
    column_stripe = numpy.arange(projector_width) // finest_stripe_pixels
    row_stripe = numpy.arange(projector_height) // finest_stripe_pixels
    column_gray = column_stripe ^ (column_stripe >> 1)
    row_gray = row_stripe ^ (row_stripe >> 1)
    column_patterns, row_patterns = [], []
    for bit in range(bit_count):
        shift = bit_count - 1 - bit
        column_bits = ((column_gray >> shift) & 1).astype(numpy.uint8) * 255
        row_bits = ((row_gray >> shift) & 1).astype(numpy.uint8) * 255
        column_patterns.append(numpy.repeat(column_bits[None, :], projector_height, axis=0))
        row_patterns.append(numpy.repeat(row_bits[:, None], projector_width, axis=1))
    return bit_count, column_patterns, row_patterns


def show_and_average(projector, camera, pattern_gray, settle_seconds, frame_count, frame_timeout_seconds):
    """Show a pattern, wait until it is fully visible to the camera, and return the mean of
    the next frames as float32 grey, or None when the camera stream stalled."""
    projector.show(cv2.cvtColor(pattern_gray, cv2.COLOR_GRAY2BGR))
    cv2.waitKey(1)
    next_frame_after = time.perf_counter() + settle_seconds
    accumulated = None
    captured = 0
    for _ in range(frame_count):
        result = camera.frame_after(next_frame_after, timeout_seconds=frame_timeout_seconds)
        cv2.waitKey(1)
        if result is None:
            return None
        next_frame_after, frame = result
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(numpy.float32)
        accumulated = gray if accumulated is None else accumulated + gray
        captured += 1
    return accumulated / captured


def scan(projector, camera, finest_stripe_pixels, settle_seconds, frame_count, frame_timeout_seconds,
         minimum_beam_step, minimum_bit_contrast_fraction, progress=None):
    """Runs the scan. Returns a dictionary:
        bit_count, beam (bool image: lit by the projector),
        column_index / row_index (int32 images: stripe index from all bits),
        column_reliable_bits / row_reliable_bits (uint8 images: leading bits read reliably)
    or None when the camera stream stalled for good."""
    bit_count, column_patterns, row_patterns = gray_code_patterns(projector.width, projector.height, finest_stripe_pixels)
    black = numpy.zeros((projector.height, projector.width), dtype=numpy.uint8)
    white = numpy.full((projector.height, projector.width), 255, dtype=numpy.uint8)
    total = 2 + 4 * bit_count
    shown = 0

    def capture(pattern):
        nonlocal shown
        for attempt in range(3):
            image = show_and_average(projector, camera, pattern, settle_seconds, frame_count, frame_timeout_seconds)
            if image is not None:
                shown += 1
                if progress is not None:
                    progress(shown, total)
                return image
            print(f"[scan] the camera stream stalled during a pattern; repeating it (attempt {attempt + 2})")
        return None

    dark = capture(black)
    lit = capture(white)
    if dark is None or lit is None:
        return None
    beam_step = cv2.GaussianBlur(lit - dark, (5, 5), 0)
    beam = beam_step > minimum_beam_step
    required_contrast = numpy.maximum(minimum_bit_contrast_fraction * beam_step, 0.5 * minimum_beam_step)

    result = {"bit_count": bit_count, "beam": beam, "beam_step": beam_step}
    for axis_name, patterns in (("column", column_patterns), ("row", row_patterns)):
        gray_value = numpy.zeros(dark.shape, dtype=numpy.int32)
        reliable_bits = numpy.zeros(dark.shape, dtype=numpy.uint8)
        still_reliable = beam.copy()
        for bit, pattern in enumerate(patterns):
            positive = capture(pattern)
            negative = capture(255 - pattern)
            if positive is None or negative is None:
                return None
            difference = cv2.GaussianBlur(positive - negative, (3, 3), 0)
            gray_value = (gray_value << 1) | (difference > 0).astype(numpy.int32)
            still_reliable &= numpy.abs(difference) >= required_contrast
            reliable_bits[still_reliable] = bit + 1
        binary = gray_value.copy()                      # Gray code to stripe index
        shift = 1
        while shift < bit_count:
            binary ^= binary >> shift
            shift <<= 1
        result[f"{axis_name}_index"] = binary
        result[f"{axis_name}_reliable_bits"] = reliable_bits
    projector.show(cv2.cvtColor(black, cv2.COLOR_GRAY2BGR))
    cv2.waitKey(1)
    return result


def cell_correspondences(scan_result, points_3d_image, finest_stripe_pixels, projector_width, projector_height,
                         coarsest_level, camera_pixels_per_projector_pixel, minimum_cell_fill,
                         maximum_depth_spread_metres, maximum_depth_spread_fraction):
    """Turns the per-pixel decode into one correspondence per projector cell.

    points_3d_image: (rows, columns, 3) metric point of every camera pixel, NaN where unknown.
    Returns a dictionary of arrays: projector_points (N, 2), camera_pixels (N, 2),
    points_3d (N, 3), level (N,), pixel_count (N,)."""
    bit_count = scan_result["bit_count"]
    image_height, image_width = scan_result["beam"].shape
    rows_image, columns_image = numpy.mgrid[0:image_height, 0:image_width]
    depth = points_3d_image[:, :, 2]
    covered = numpy.zeros((image_height, image_width), dtype=bool)      # pixels already described by a finer level
    collected = {"projector_points": [], "camera_pixels": [], "points_3d": [], "level": [], "pixel_count": []}

    for level in range(coarsest_level + 1):
        used_bits = bit_count - level
        if used_bits < 1:
            break
        cell_pixels = finest_stripe_pixels * (2 ** level)
        readable = ((scan_result["column_reliable_bits"] >= used_bits) & (scan_result["row_reliable_bits"] >= used_bits))
        if not readable.any():
            continue
        cell_column = scan_result["column_index"][readable] >> level
        cell_row = scan_result["row_index"][readable] >> level
        cells_across = (2 ** bit_count) >> level
        cell_id = cell_row.astype(numpy.int64) * cells_across + cell_column
        order = numpy.argsort(cell_id, kind="stable")
        sorted_id = cell_id[order]
        boundaries = numpy.flatnonzero(numpy.diff(sorted_id)) + 1
        starts = numpy.concatenate([[0], boundaries])
        ends = numpy.concatenate([boundaries, [len(sorted_id)]])
        pixel_rows = rows_image[readable][order]
        pixel_columns = columns_image[readable][order]
        pixel_depth = depth[readable][order]
        pixel_points = points_3d_image[readable][order]
        pixel_covered = covered[readable][order]
        expected_count = (cell_pixels * camera_pixels_per_projector_pixel) ** 2
        newly_covered_rows, newly_covered_columns = [], []
        for start, end in zip(starts, ends):
            count = end - start
            if count < minimum_cell_fill * expected_count:
                continue
            if pixel_covered[start:end].mean() > 0.5:
                continue
            identifier = int(sorted_id[start])
            projector_x = ((identifier % cells_across) + 0.5) * cell_pixels - 0.5
            projector_y = ((identifier // cells_across) + 0.5) * cell_pixels - 0.5
            if projector_x > projector_width - 1 or projector_y > projector_height - 1:
                continue
            cell_depth = pixel_depth[start:end]
            known = numpy.isfinite(cell_depth)
            if known.mean() < 0.8:
                continue
            low, middle, high = numpy.percentile(cell_depth[known], [10, 50, 90])
            if high - low > max(maximum_depth_spread_metres, maximum_depth_spread_fraction * middle):
                continue                                                  # the cell straddles a depth edge
            collected["projector_points"].append((projector_x, projector_y))
            collected["camera_pixels"].append((pixel_columns[start:end][known].mean(), pixel_rows[start:end][known].mean()))
            collected["points_3d"].append(pixel_points[start:end][known].mean(axis=0))
            collected["level"].append(level)
            collected["pixel_count"].append(count)
            newly_covered_rows.append(pixel_rows[start:end])
            newly_covered_columns.append(pixel_columns[start:end])
        if newly_covered_rows:
            covered[numpy.concatenate(newly_covered_rows), numpy.concatenate(newly_covered_columns)] = True
    return {"projector_points": numpy.array(collected["projector_points"], dtype=numpy.float64).reshape(-1, 2),
            "camera_pixels": numpy.array(collected["camera_pixels"], dtype=numpy.float64).reshape(-1, 2),
            "points_3d": numpy.array(collected["points_3d"], dtype=numpy.float64).reshape(-1, 3),
            "level": numpy.array(collected["level"], dtype=numpy.int32),
            "pixel_count": numpy.array(collected["pixel_count"], dtype=numpy.int32)}
