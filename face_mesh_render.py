"""The projector as a camera looking at MediaPipe's 3D face mesh.

A 3x4 projection matrix P maps a landmark's 3D coordinates (image x, y in
pixels and MediaPipe's relative z, also in pixel-like units) straight to a
projector pixel. P is fitted (DLT) from bootstrap face dots: each has a
measured projector pixel and the 3D coordinates of the nearest landmark.

With P, the face mesh (Delaunay triangulation of the landmarks) is rendered
from the projector's viewpoint: every triangle's texture from the camera
frame is warped to where P puts its three vertices. Projecting that image
puts the photo of the face back onto the face if P is right.
"""

import cv2
import numpy

import config


def _normalise(points):
    """Similarity normalisation for a well-conditioned DLT: returns (normalised, transform)."""
    points = numpy.asarray(points, dtype=numpy.float64)
    mean = points.mean(axis=0)
    scale = numpy.sqrt(points.shape[1]) / (numpy.linalg.norm(points - mean, axis=1).mean() + 1e-9)
    dimension = points.shape[1]
    transform = numpy.eye(dimension + 1)
    transform[:dimension, :dimension] *= scale
    transform[:dimension, dimension] = -scale * mean
    homogeneous = numpy.hstack([points, numpy.ones((len(points), 1))])
    return (homogeneous @ transform.T)[:, :dimension], transform


def _fit_affine_camera(points_3d, projector_points, depth_ridge):
    """Affine camera: u = a.x + b.y + c.z + d (8 parameters), as a 3x4 P with last row [0 0 0 1].
    A small ridge on the z coefficients keeps them sane when the depth range is tiny."""
    design = numpy.hstack([points_3d, numpy.ones((len(points_3d), 1))])
    ridge = numpy.zeros((1, 4))
    ridge[0, 2] = numpy.sqrt(depth_ridge)
    design_ridged = numpy.vstack([design, ridge])
    targets = numpy.vstack([projector_points, numpy.zeros((1, 2))])
    solution, _, _, _ = numpy.linalg.lstsq(design_ridged, targets, rcond=None)
    projection = numpy.zeros((3, 4))
    projection[:2, :] = solution.T
    projection[2, 3] = 1.0
    return projection


def _fit_projective_camera(points_3d, projector_points):
    world, world_transform = _normalise(points_3d)
    image, image_transform = _normalise(projector_points)
    rows = []
    for (x, y, z), (u, v) in zip(world, image):
        rows.append([x, y, z, 1, 0, 0, 0, 0, -u * x, -u * y, -u * z, -u])
        rows.append([0, 0, 0, 0, x, y, z, 1, -v * x, -v * y, -v * z, -v])
    _, _, vt = numpy.linalg.svd(numpy.array(rows))
    projection = numpy.linalg.inv(image_transform) @ vt[-1].reshape(3, 4) @ world_transform
    return projection / (numpy.linalg.norm(projection[2, :3]) + 1e-12)


def fit_projection_matrix(points_3d, projector_points, outlier_pixels=8.0, depth_ridge=1e-3):
    """Fit the projector as a camera looking at the face points.

    The face spans only a few centimetres of depth, so a full projective fit
    is ill-conditioned; the affine camera is fitted first (always well posed)
    and the projective one replaces it only if it clearly fits better.
    Returns (P, inlier_mask, rms) or (None, None, nan)."""
    points_3d = numpy.asarray(points_3d, dtype=numpy.float64)
    projector_points = numpy.asarray(projector_points, dtype=numpy.float64)
    count = len(points_3d)
    if count < 6:
        return None, None, float("nan")
    # RANSAC over minimal sets of 5: gross outliers (mis-decoded dots) must not
    # take part in the fit that judges them.
    rng = numpy.random.default_rng(0)
    best_inliers = None
    for _ in range(400):
        sample = rng.choice(count, size=min(5, count), replace=False)
        try:
            candidate = _fit_affine_camera(points_3d[sample], projector_points[sample], depth_ridge)
        except numpy.linalg.LinAlgError:
            continue
        errors = numpy.linalg.norm(project_points(candidate, points_3d) - projector_points, axis=1)
        candidate_inliers = errors < outlier_pixels
        if best_inliers is None or candidate_inliers.sum() > best_inliers.sum():
            best_inliers = candidate_inliers
    if best_inliers is None or best_inliers.sum() < 6:
        return None, None, float("nan")
    inliers = best_inliers
    for _ in range(2):
        projection = _fit_affine_camera(points_3d[inliers], projector_points[inliers], depth_ridge)
        errors = numpy.linalg.norm(project_points(projection, points_3d) - projector_points, axis=1)
        inliers = errors < outlier_pixels
        if inliers.sum() < 6:
            return None, None, float("nan")
    affine_rms = float(numpy.sqrt(numpy.mean(errors[inliers] ** 2)))
    if inliers.sum() >= 12:
        try:
            projective = _fit_projective_camera(points_3d[inliers], projector_points[inliers])
            projective_errors = numpy.linalg.norm(project_points(projective, points_3d) - projector_points, axis=1)
            projective_rms = float(numpy.sqrt(numpy.mean(projective_errors[inliers] ** 2)))
            if numpy.all(numpy.isfinite(projective_errors)) and projective_rms < 0.8 * affine_rms:
                return projective, inliers, projective_rms
        except numpy.linalg.LinAlgError:
            pass
    return projection, inliers, affine_rms


def project_points(projection, points_3d):
    """Projector pixels for 3D points: through a 3x4 matrix, or through a calibration
    object with a project() method (projector_calibration.ProjectorCalibration, which
    also applies the lens distortion)."""
    if hasattr(projection, "project"):
        return projection.project(points_3d)
    points_3d = numpy.asarray(points_3d, dtype=numpy.float64).reshape(-1, 3)
    homogeneous = numpy.hstack([points_3d, numpy.ones((len(points_3d), 1))]) @ projection.T
    depth = homogeneous[:, 2:3]
    depth[numpy.abs(depth) < 1e-9] = 1e-9
    return homogeneous[:, :2] / depth


def triangulate(landmarks_2d, width, height):
    """Delaunay triangles of the landmarks as index triples."""
    subdivision = cv2.Subdiv2D((0, 0, width + 1, height + 1))
    lookup = {}
    for index, (x, y) in enumerate(landmarks_2d):
        x = float(numpy.clip(x, 0, width - 1))
        y = float(numpy.clip(y, 0, height - 1))
        lookup[(round(x, 3), round(y, 3))] = index
        subdivision.insert((x, y))
    triangles = []
    for triangle in subdivision.getTriangleList():
        indices = []
        for corner in range(3):
            key = (round(float(triangle[2 * corner]), 3), round(float(triangle[2 * corner + 1]), 3))
            if key not in lookup:
                break
            indices.append(lookup[key])
        if len(indices) == 3:
            triangles.append(indices)
    return numpy.array(triangles, dtype=numpy.int32)


# The canonical MediaPipe mesh is wound so that, in the Kinect frame (x right, y up,
# z away from the camera), normal . (viewpoint - corner) is negative for a triangle
# that faces the viewpoint (98% of the canonical face, seen from its front).
CANONICAL_FRONT_SIGN = -1.0


def front_facing(landmarks_3d, triangles, viewpoint, front_sign=CANONICAL_FRONT_SIGN):
    """Which triangles face the viewpoint (the projector), from the mesh's fixed winding.
    Triangles facing away would otherwise be painted through onto the near side. The
    winding is a property of the canonical topology, so this holds however far the face
    is turned from the projector (a majority vote over the triangles would not)."""
    corners = numpy.asarray(landmarks_3d, dtype=numpy.float64)[triangles]      # (T, 3, 3)
    normals = numpy.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    toward_viewpoint = numpy.asarray(viewpoint, dtype=numpy.float64) - corners[:, 0]
    facing = numpy.einsum("ij,ij->i", normals, toward_viewpoint)
    return facing * front_sign > 0


def render_face_mesh(texture_bgr, texture_landmarks_2d, landmarks_3d, projection, triangles, width, height,
                     brightness_scale=1.0, viewpoint=None, render_scale=None):
    """Warp the face texture to where P puts the mesh, in one vectorised pass:
    rasterise triangle ids, the texture position of every covered projector pixel, then
    a single remap. Returns a BGR projector image. With a viewpoint (the projector's
    position) triangles facing away from it are not drawn.

    render_scale below 1 draws the mesh that much smaller and enlarges the result: the
    work falls with the square of the scale, and on a face, where a projector pixel is
    well under a millimetre, the difference does not show."""
    if render_scale is None:
        render_scale = getattr(config, "face_mesh_render_scale", 1.0)
    projected = project_points(projection, landmarks_3d)
    if not numpy.all(numpy.isfinite(projected)):
        return numpy.zeros((height, width, 3), dtype=numpy.uint8)
    texture = texture_bgr if brightness_scale == 1.0 else numpy.clip(texture_bgr * brightness_scale, 0, 255).astype(numpy.uint8)

    output = numpy.zeros((height, width, 3), dtype=numpy.uint8)
    destination_corners = projected[triangles].astype(numpy.float32)          # (T, 3, 2)
    source_corners = texture_landmarks_2d[triangles].astype(numpy.float32)    # (T, 3, 2)
    # Only triangles that blew up (a vertex behind the projector, a numerical failure) are
    # skipped; a face close to the projector legitimately has large triangles.
    extents = destination_corners.max(axis=1) - destination_corners.min(axis=1)
    usable = (extents[:, 0] < 4 * width) & (extents[:, 1] < 4 * height)
    if viewpoint is not None:
        usable &= front_facing(landmarks_3d, triangles, viewpoint)
    if not usable.any():
        return output

    # Work only inside the mesh's bounding box on the projector.
    box_left = int(max(0, numpy.floor(destination_corners[usable].min(axis=(0, 1))[0])))
    box_top = int(max(0, numpy.floor(destination_corners[usable].min(axis=(0, 1))[1])))
    box_right = int(min(width, numpy.ceil(destination_corners[usable].max(axis=(0, 1))[0]) + 1))
    box_bottom = int(min(height, numpy.ceil(destination_corners[usable].max(axis=(0, 1))[1]) + 1))
    if box_right <= box_left or box_bottom <= box_top:
        return output
    box_width, box_height = box_right - box_left, box_bottom - box_top
    drawn_width = max(1, int(round(box_width * render_scale)))
    drawn_height = max(1, int(round(box_height * render_scale)))
    scale_x = drawn_width / float(box_width)
    scale_y = drawn_height / float(box_height)

    # The triangles in the coordinates of the (possibly smaller) drawing.
    drawn_corners = (destination_corners - numpy.array([box_left, box_top], dtype=numpy.float32)) \
        * numpy.array([scale_x, scale_y], dtype=numpy.float32)
    rounded_corners = numpy.round(drawn_corners).astype(numpy.int32)

    # Triangle id per pixel of the drawing (-1 = not covered).
    id_image = numpy.full((drawn_height, drawn_width), -1, dtype=numpy.int32)
    for index in numpy.flatnonzero(usable):
        cv2.fillConvexPoly(id_image, rounded_corners[index], int(index))
    covered = id_image >= 0
    if not covered.any():
        return output

    # Per-triangle affine map from drawing coordinates to texture coordinates: [D 1] . A = S.
    ones = numpy.ones((len(triangles), 3, 1), dtype=numpy.float32)
    drawn_homogeneous = numpy.concatenate([drawn_corners, ones], axis=2)                 # (T, 3, 3)
    determinants = numpy.linalg.det(drawn_homogeneous)
    safe = numpy.abs(determinants) > 1e-6
    affine = numpy.zeros((len(triangles) + 1, 3, 2), dtype=numpy.float32)                # the extra row serves id -1
    affine[:-1][safe] = numpy.linalg.solve(drawn_homogeneous[safe], source_corners[safe])
    affine[-1, 2] = (-10.0, -10.0)                                                        # uncovered pixels read outside the texture

    # Texture position of every pixel: six coefficient images, combined with the pixel's
    # own column and row (no per-pixel matrices).
    coefficients = affine[id_image]                                                       # (H, W, 3, 2); id -1 picks the extra row
    pixel_columns = numpy.arange(drawn_width, dtype=numpy.float32)[None, :]
    pixel_rows = numpy.arange(drawn_height, dtype=numpy.float32)[:, None]
    map_x = coefficients[:, :, 0, 0] * pixel_columns + coefficients[:, :, 1, 0] * pixel_rows + coefficients[:, :, 2, 0]
    map_y = coefficients[:, :, 0, 1] * pixel_columns + coefficients[:, :, 1, 1] * pixel_rows + coefficients[:, :, 2, 1]
    drawn = cv2.remap(texture, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    if (drawn_width, drawn_height) != (box_width, box_height):
        drawn = cv2.resize(drawn, (box_width, box_height), interpolation=cv2.INTER_LINEAR)
    output[box_top:box_bottom, box_left:box_right] = drawn
    return output


def render_wireframe(landmarks_3d, projection, triangles, width, height, viewpoint=None):
    """The mesh as lines from the projector's viewpoint, on black; back faces skipped when a viewpoint is given."""
    output = numpy.zeros((height, width, 3), dtype=numpy.uint8)
    projected = project_points(projection, landmarks_3d)
    if viewpoint is not None:
        triangles = triangles[front_facing(landmarks_3d, triangles, viewpoint)]
    for triangle in triangles:
        corners = projected[triangle]
        if not numpy.all(numpy.isfinite(corners)):
            continue
        cv2.polylines(output, [numpy.round(corners).astype(numpy.int32).reshape(-1, 1, 2)], True, (0, 255, 0), 1, cv2.LINE_AA)
    return output
