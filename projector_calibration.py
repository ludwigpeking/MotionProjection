"""The projector as a camera: metric 3D (Kinect camera space, metres) -> projector pixels.

Fitted from the coded-dot scan seen by the Kinect: every decoded dot is one
(3D point, projector pixel) correspondence. A normalised DLT under RANSAC
gives the 3x4 projection; cv2.calibrateCamera then refines it into intrinsics,
two radial distortion terms and the pose, all from that single view (the dots
are spread over wall, chair, desk and person, so they are not coplanar).

Once fitted, any 3D point the Kinect measures maps to the projector exactly,
whatever its depth: no homography, no parallax model.
"""

import datetime
import json

import cv2
import numpy


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


def direct_linear_transform(points_3d, projector_points):
    """3x4 projection from >= 6 correspondences (normalised DLT)."""
    world, world_transform = _normalise(points_3d)
    image, image_transform = _normalise(projector_points)
    rows = []
    for (x, y, z), (u, v) in zip(world, image):
        rows.append([x, y, z, 1, 0, 0, 0, 0, -u * x, -u * y, -u * z, -u])
        rows.append([0, 0, 0, 0, x, y, z, 1, -v * x, -v * y, -v * z, -v])
    _, _, right_singular_vectors = numpy.linalg.svd(numpy.array(rows))
    projection = numpy.linalg.inv(image_transform) @ right_singular_vectors[-1].reshape(3, 4) @ world_transform
    return projection / (numpy.linalg.norm(projection[2, :3]) + 1e-12)


def project_through_matrix(projection, points_3d):
    points_3d = numpy.asarray(points_3d, dtype=numpy.float64).reshape(-1, 3)
    homogeneous = numpy.hstack([points_3d, numpy.ones((len(points_3d), 1))]) @ projection.T
    depth = homogeneous[:, 2:3].copy()
    depth[numpy.abs(depth) < 1e-9] = 1e-9
    return homogeneous[:, :2] / depth


class ProjectorCalibration:
    def __init__(self, camera_matrix, distortion, rotation_vector, translation_vector,
                 projector_width, projector_height):
        self.camera_matrix = numpy.asarray(camera_matrix, dtype=numpy.float64).reshape(3, 3)
        self.distortion = numpy.asarray(distortion, dtype=numpy.float64).reshape(-1)
        self.rotation_vector = numpy.asarray(rotation_vector, dtype=numpy.float64).reshape(3, 1)
        self.translation_vector = numpy.asarray(translation_vector, dtype=numpy.float64).reshape(3, 1)
        self.projector_width = int(projector_width)
        self.projector_height = int(projector_height)
        self.rms_pixels = float("nan")
        self.holdout_rms_pixels = float("nan")
        self.point_count = 0
        self.inlier_count = 0
        self.depth_range_metres = (float("nan"), float("nan"))
        self.focal_source = "assumed"          # "assumed" (from config) or "fitted from N near dots"
        self.latency_none_before_seconds = None
        self.latency_full_after_seconds = None
        self.prediction_lead_seconds = None
        self.calibrated_at = None

    def project(self, points_3d):
        """Projector pixels (N, 2) for 3D points (N, 3) in Kinect camera space (metres)."""
        points_3d = numpy.asarray(points_3d, dtype=numpy.float64).reshape(-1, 1, 3)
        projected, _ = cv2.projectPoints(points_3d, self.rotation_vector, self.translation_vector,
                                         self.camera_matrix, self.distortion)
        return projected.reshape(-1, 2)

    def depths(self, points_3d):
        """Distance of each point along the projector's optical axis (negative = behind the projector)."""
        rotation, _ = cv2.Rodrigues(self.rotation_vector)
        points_3d = numpy.asarray(points_3d, dtype=numpy.float64).reshape(-1, 3)
        return (points_3d @ rotation.T + self.translation_vector.reshape(1, 3))[:, 2]

    def reprojection_errors(self, points_3d, projector_points):
        projected = self.project(points_3d)
        targets = numpy.asarray(projector_points, dtype=numpy.float64).reshape(-1, 2)
        return numpy.linalg.norm(projected - targets, axis=1)

    def position_metres(self):
        """The projector's optical centre in Kinect camera space."""
        rotation, _ = cv2.Rodrigues(self.rotation_vector)
        return (-rotation.T @ self.translation_vector).reshape(3)

    def describe(self):
        focal_x, focal_y = self.camera_matrix[0, 0], self.camera_matrix[1, 1]
        centre_x, centre_y = self.camera_matrix[0, 2], self.camera_matrix[1, 2]
        return (f"focal {focal_x:.0f}/{focal_y:.0f} px, principal point ({centre_x:.0f},{centre_y:.0f}), "
                f"({self.focal_source}), radial distortion {numpy.array2string(self.distortion[:2], precision=3)}, "
                f"projector at {numpy.array2string(self.position_metres(), precision=2)} m in Kinect space")

    def to_dictionary(self):
        return {
            "projector_width": self.projector_width,
            "projector_height": self.projector_height,
            "camera_matrix": self.camera_matrix.tolist(),
            "distortion": self.distortion.tolist(),
            "rotation_vector": self.rotation_vector.reshape(3).tolist(),
            "translation_vector": self.translation_vector.reshape(3).tolist(),
            "rms_pixels": self.rms_pixels,
            "holdout_rms_pixels": self.holdout_rms_pixels,
            "point_count": self.point_count,
            "inlier_count": self.inlier_count,
            "depth_range_metres": list(self.depth_range_metres),
            "focal_source": self.focal_source,
            "latency_none_before_seconds": self.latency_none_before_seconds,
            "latency_full_after_seconds": self.latency_full_after_seconds,
            "prediction_lead_seconds": self.prediction_lead_seconds,
            "calibrated_at": self.calibrated_at,
        }

    def save(self, path):
        self.calibrated_at = datetime.datetime.now().isoformat(timespec="seconds")
        with open(path, "w") as file:
            json.dump(self.to_dictionary(), file, indent=1)

    @classmethod
    def load(cls, path):
        with open(path) as file:
            data = json.load(file)
        calibration = cls(data["camera_matrix"], data["distortion"], data["rotation_vector"],
                          data["translation_vector"], data["projector_width"], data["projector_height"])
        for key in ("rms_pixels", "holdout_rms_pixels", "point_count", "inlier_count",
                    "latency_none_before_seconds", "latency_full_after_seconds", "prediction_lead_seconds",
                    "calibrated_at", "focal_source"):
            if key in data:
                setattr(calibration, key, data[key])
        if "depth_range_metres" in data:
            calibration.depth_range_metres = tuple(data["depth_range_metres"])
        return calibration


def _decompose(projection):
    """(camera_matrix, rotation_vector, translation_vector) from a 3x4 projection, with
    positive focal lengths and a proper rotation (det +1)."""
    camera_matrix, rotation, camera_centre_homogeneous = cv2.decomposeProjectionMatrix(projection)[:3]
    camera_matrix = camera_matrix / camera_matrix[2, 2]
    rotation = rotation.copy()
    # The RQ decomposition is unique only up to the sign of each row of R (and the
    # matching column of K): choose the signs that make the focal lengths positive.
    for axis in (0, 1):
        if camera_matrix[axis, axis] < 0:
            camera_matrix[:, axis] *= -1.0
            rotation[axis, :] *= -1.0
    if numpy.linalg.det(rotation) < 0:
        rotation *= -1.0            # K[-R | -t] = -P: identical projections, proper rotation
    camera_matrix[0, 1] = 0.0       # the DLT's tiny skew is noise; the pinhole refinement needs zero skew
    camera_centre = (camera_centre_homogeneous[:3] / camera_centre_homogeneous[3]).reshape(3)
    translation = -rotation @ camera_centre
    rotation_vector, _ = cv2.Rodrigues(rotation)
    return camera_matrix, rotation_vector.reshape(3, 1), translation.reshape(3, 1)


def _ransac_projection(points_3d, projector_points, outlier_pixels, iterations=600, seed=0):
    count = len(points_3d)
    rng = numpy.random.default_rng(seed)
    best_inliers = None
    for _ in range(iterations):
        sample = rng.choice(count, size=6, replace=False)
        try:
            candidate = direct_linear_transform(points_3d[sample], projector_points[sample])
        except numpy.linalg.LinAlgError:
            continue
        errors = numpy.linalg.norm(project_through_matrix(candidate, points_3d) - projector_points, axis=1)
        inliers = errors < outlier_pixels
        if best_inliers is None or inliers.sum() > best_inliers.sum():
            best_inliers = inliers
    return best_inliers


def _root_mean_square(values):
    return float(numpy.sqrt(numpy.mean(numpy.square(values)))) if len(values) else float("nan")


def _pinhole_from_pose(camera_matrix, points_3d, projector_points, projector_width, projector_height,
                       outlier_pixels):
    """Pose for a fixed camera matrix by PnP under RANSAC; returns (calibration, inliers) or (None, None)."""
    try:
        found, rotation_vector, translation_vector, inlier_indices = cv2.solvePnPRansac(
            points_3d.astype(numpy.float64).reshape(-1, 1, 3), projector_points.astype(numpy.float64).reshape(-1, 1, 2),
            camera_matrix, None, iterationsCount=500, reprojectionError=outlier_pixels, confidence=0.999,
            flags=cv2.SOLVEPNP_EPNP)
    except cv2.error:
        return None, None
    if not found or inlier_indices is None or len(inlier_indices) < 6:
        return None, None
    calibration = ProjectorCalibration(camera_matrix, numpy.zeros(5), rotation_vector, translation_vector,
                                       projector_width, projector_height)
    errors = calibration.reprojection_errors(points_3d, projector_points)
    return calibration, errors < outlier_pixels


def _initial_pinhole(points_3d, projector_points, projector_width, projector_height, outlier_pixels):
    """Best zero-skew pinhole with pose, from a sweep over focal lengths and principal points
    (plus the DLT's own guess when it is sane). Robust to nearly planar scenes, where the
    DLT alone returns a camera with absurd skew."""
    candidates = []
    inliers = _ransac_projection(points_3d, projector_points, outlier_pixels)
    if inliers is not None and inliers.sum() >= 6:
        try:
            projection = direct_linear_transform(points_3d[inliers], projector_points[inliers])
            camera_matrix, _, _ = _decompose(projection)
            focal = 0.5 * (camera_matrix[0, 0] + camera_matrix[1, 1])
            if 0.3 * projector_width < focal < 6.0 * projector_width \
                    and -projector_width < camera_matrix[0, 2] < 2 * projector_width \
                    and -projector_height < camera_matrix[1, 2] < 3 * projector_height:
                candidates.append(numpy.array([[focal, 0.0, camera_matrix[0, 2]],
                                               [0.0, focal, camera_matrix[1, 2]],
                                               [0.0, 0.0, 1.0]]))
        except numpy.linalg.LinAlgError:
            pass
    # Throw ratios from 0.5 to 3 (focal 0.5..3 x width); principal point at the image centre,
    # the bottom edge (typical lens offset of a table-top projector) and the top edge.
    for focal_factor in (0.5, 0.7, 1.0, 1.4, 2.0, 3.0):
        for centre_y_factor in (0.5, 1.0, 0.0):
            candidates.append(numpy.array([[focal_factor * projector_width, 0.0, projector_width / 2.0],
                                           [0.0, focal_factor * projector_width, centre_y_factor * projector_height],
                                           [0.0, 0.0, 1.0]]))
    best, best_inliers, best_rms = None, None, float("inf")
    for camera_matrix in candidates:
        calibration, candidate_inliers = _pinhole_from_pose(camera_matrix, points_3d, projector_points,
                                                            projector_width, projector_height, outlier_pixels)
        if calibration is None:
            continue
        rms = _root_mean_square(calibration.reprojection_errors(points_3d, projector_points)[candidate_inliers])
        if best_inliers is None or candidate_inliers.sum() > best_inliers.sum() \
                or (candidate_inliers.sum() == best_inliers.sum() and rms < best_rms):
            best, best_inliers, best_rms = calibration, candidate_inliers, rms
    return best, best_inliers, best_rms


def depth_balanced_score(calibration, points_3d, projector_points, threshold_pixels, bin_metres=0.5):
    """Mean over 0.5 m depth bins of the fraction of dots the model fits within the
    threshold. A wall at one depth holds most dots; this score makes a model that
    also fits the few near dots (a person, a chair) beat one that fits the wall alone."""
    errors = calibration.reprojection_errors(points_3d, projector_points)
    bins = numpy.floor(points_3d[:, 2] / bin_metres).astype(int)
    fractions = []
    for bin_index in numpy.unique(bins):
        in_bin = bins == bin_index
        if in_bin.sum() >= 5:
            fractions.append(float((errors[in_bin] < threshold_pixels).mean()))
    return float(numpy.mean(fractions)) if fractions else 0.0


def fit_projector(points_3d, projector_points, projector_width, projector_height, outlier_pixels,
                  min_points=12, fix_principal_point=True):
    """Fit the projector model. Returns (calibration, inlier_mask) or (None, None)."""
    points_3d = numpy.asarray(points_3d, dtype=numpy.float64).reshape(-1, 3)
    projector_points = numpy.asarray(projector_points, dtype=numpy.float64).reshape(-1, 2)
    count = len(points_3d)
    if count < max(min_points, 6):
        return None, None
    stage_models = []

    # Coarse to fine: a scene that is mostly one plane (a wall) plus a few near points
    # (a person) offers a tempting model that fits only the plane very precisely; the
    # model that explains both groups is selected at a loose threshold first and the
    # threshold is tightened only after the refinement has absorbed the near points.
    coarse_pixels = 3.0 * outlier_pixels
    best, inliers, best_rms = _initial_pinhole(points_3d, projector_points, projector_width, projector_height,
                                               coarse_pixels)
    if best is None or inliers.sum() < 6:
        print("[calibration] no pinhole model fits the dots; giving up")
        return None, None
    print(f"[calibration] initial pinhole: {int(inliers.sum())}/{count} dots within {coarse_pixels:.0f} px, "
          f"rms {best_rms:.2f} px, focal {best.camera_matrix[0, 0]:.0f} px, "
          f"principal point ({best.camera_matrix[0, 2]:.0f},{best.camera_matrix[1, 2]:.0f})")

    # Non-linear refinement of intrinsics (square pixels), pose and two radial distortion
    # terms, from this single view. The principal point stays where the pinhole search put
    # it: with one dominant plane in the data, letting it float lets the solver trade it
    # against the focal length and the pose and wander to a wrong model that still fits
    # the plane.
    flags = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_FIX_K3
             | cv2.CALIB_ZERO_TANGENT_DIST)
    if fix_principal_point:
        flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
    stage_models.append((depth_balanced_score(best, points_3d, projector_points, 2.0 * outlier_pixels), best))
    for threshold in (coarse_pixels, 2.0 * outlier_pixels, outlier_pixels):
        errors = best.reprojection_errors(points_3d, projector_points)
        inliers = errors < threshold
        if inliers.sum() < 6:
            break
        best_rms = _root_mean_square(errors[inliers])
        best_score = depth_balanced_score(best, points_3d, projector_points, 2.0 * outlier_pixels)
        for _ in range(2):
            try:
                object_points = [points_3d[inliers].astype(numpy.float32).reshape(-1, 1, 3)]
                image_points = [projector_points[inliers].astype(numpy.float32).reshape(-1, 1, 2)]
                # A projector with lens shift has its principal point outside the image (below
                # it for a table-top projector). calibrateCamera insists the point lies inside
                # the image size it is given, which it otherwise only uses for the initial
                # guess, so hand it a virtual image large enough to contain the point.
                centre_x, centre_y = best.camera_matrix[0, 2], best.camera_matrix[1, 2]
                if centre_x <= 0 or centre_y <= 0:
                    print("[calibration] principal point at a negative coordinate; no non-linear refinement")
                    break
                virtual_size = (int(max(projector_width, 2 * centre_x + 1)),
                                int(max(projector_height, 2 * centre_y + 1)))
                _, refined_matrix, refined_distortion, rotation_vectors, translation_vectors = cv2.calibrateCamera(
                    object_points, image_points, virtual_size, best.camera_matrix.copy(), None, flags=flags)
            except cv2.error as error:
                print(f"[calibration] non-linear refinement failed at {threshold:.0f} px: {error}")
                break
            refined = ProjectorCalibration(refined_matrix, refined_distortion, rotation_vectors[0],
                                           translation_vectors[0], projector_width, projector_height)
            refined_errors = refined.reprojection_errors(points_3d, projector_points)
            refined_inliers = refined_errors < threshold
            refined_rms = _root_mean_square(refined_errors[refined_inliers])
            refined_score = depth_balanced_score(refined, points_3d, projector_points, 2.0 * outlier_pixels)
            # A refinement that fits the dominant plane a little better but the other
            # depths worse is the wrong basin: refuse it.
            if refined_score >= best_score - 0.02 and (refined_inliers.sum() > inliers.sum()
                                                       or (refined_inliers.sum() == inliers.sum()
                                                           and refined_rms < best_rms)):
                best, best_rms, inliers, best_score = refined, refined_rms, refined_inliers, refined_score
            else:
                break
        score = depth_balanced_score(best, points_3d, projector_points, 2.0 * outlier_pixels)
        stage_models.append((score, best))
        print(f"[calibration] within {threshold:.0f} px: {int(inliers.sum())}/{count} dots, rms {best_rms:.2f} px, "
              f"depth-balanced score {score:.2f}, focal {best.camera_matrix[0, 0]:.0f} px, principal point "
              f"({best.camera_matrix[0, 2]:.0f},{best.camera_matrix[1, 2]:.0f}), "
              f"projector at {numpy.array2string(best.position_metres(), precision=2)} m")

    # The tightest stage can slide into a model that fits the dominant plane alone;
    # keep the stage that fits all depths best.
    if stage_models:
        best_score, best = max(stage_models, key=lambda entry: entry[0])
        errors = best.reprojection_errors(points_3d, projector_points)
        inliers = errors < outlier_pixels
        best_rms = _root_mean_square(errors[inliers])
        print(f"[calibration] kept the stage with depth-balanced score {best_score:.2f}")

    best.rms_pixels = best_rms
    best.point_count = int(count)
    best.inlier_count = int(inliers.sum())
    depths = best.depths(points_3d[inliers])
    best.depth_range_metres = (float(depths.min()), float(depths.max()))
    return best, inliers


def fit_projector_pose(points_3d, projector_points, projector_width, projector_height, outlier_pixels,
                       focal_pixels, principal_point, distortion, min_points=12):
    """Pose only, with the projector's intrinsics known: PnP under RANSAC, then a
    Levenberg-Marquardt refinement on the inliers. Well posed even when every dot lies
    on one plane. Returns (calibration, inlier_mask) or (None, None)."""
    points_3d = numpy.asarray(points_3d, dtype=numpy.float64).reshape(-1, 3)
    projector_points = numpy.asarray(projector_points, dtype=numpy.float64).reshape(-1, 2)
    count = len(points_3d)
    if count < max(min_points, 6):
        return None, None
    camera_matrix = numpy.array([[focal_pixels, 0.0, principal_point[0]],
                                 [0.0, focal_pixels, principal_point[1]],
                                 [0.0, 0.0, 1.0]])
    distortion_vector = numpy.zeros(5)
    distortion_vector[:2] = distortion[:2]
    try:
        found, rotation_vector, translation_vector, inlier_indices = cv2.solvePnPRansac(
            points_3d.reshape(-1, 1, 3), projector_points.reshape(-1, 1, 2), camera_matrix, distortion_vector,
            iterationsCount=1000, reprojectionError=2.0 * outlier_pixels, confidence=0.999, flags=cv2.SOLVEPNP_EPNP)
    except cv2.error as error:
        print(f"[calibration] PnP failed: {error}")
        return None, None
    if not found or inlier_indices is None or len(inlier_indices) < 6:
        return None, None
    calibration = ProjectorCalibration(camera_matrix, distortion_vector, rotation_vector, translation_vector,
                                       projector_width, projector_height)
    inliers = calibration.reprojection_errors(points_3d, projector_points) < 2.0 * outlier_pixels
    for threshold in (2.0 * outlier_pixels, outlier_pixels):
        if inliers.sum() < 6:
            break
        rotation_vector, translation_vector = cv2.solvePnPRefineLM(
            points_3d[inliers].reshape(-1, 1, 3), projector_points[inliers].reshape(-1, 1, 2), camera_matrix,
            distortion_vector, calibration.rotation_vector, calibration.translation_vector)
        calibration = ProjectorCalibration(camera_matrix, distortion_vector, rotation_vector, translation_vector,
                                           projector_width, projector_height)
        errors = calibration.reprojection_errors(points_3d, projector_points)
        inliers = errors < threshold
    errors = calibration.reprojection_errors(points_3d, projector_points)
    inliers = errors < outlier_pixels
    if inliers.sum() < 6:
        return None, None
    calibration.rms_pixels = _root_mean_square(errors[inliers])
    calibration.point_count = int(count)
    calibration.inlier_count = int(inliers.sum())
    depths = calibration.depths(points_3d[inliers])
    calibration.depth_range_metres = (float(depths.min()), float(depths.max()))
    print(f"[calibration] pose with fixed intrinsics: {int(inliers.sum())}/{count} dots within {outlier_pixels:.0f} px, "
          f"rms {calibration.rms_pixels:.2f} px, projector at {numpy.array2string(calibration.position_metres(), precision=2)} m")
    return calibration, inliers


def _capped_cost(errors, cap_pixels):
    """Sum of squared errors with each dot's contribution capped: a few undecodable dots
    cannot steer the fit."""
    return float(numpy.sum(numpy.minimum(errors, cap_pixels) ** 2))


def _refine_pose(points_3d, projector_points, camera_matrix, distortion_vector, rotation_vector, translation_vector,
                 projector_width, projector_height):
    rotation_vector, translation_vector = cv2.solvePnPRefineLM(
        points_3d.reshape(-1, 1, 3), projector_points.reshape(-1, 1, 2), camera_matrix, distortion_vector,
        rotation_vector.copy(), translation_vector.copy())
    return ProjectorCalibration(camera_matrix, distortion_vector, rotation_vector, translation_vector,
                                projector_width, projector_height)


def fit_projector_focal_and_pose(points_3d, projector_points, projector_width, projector_height, outlier_pixels,
                                 focal_guess_pixels, principal_point, distortion, min_points=12,
                                 search_fraction=0.4, near_depth_fraction=0.6, min_near_dots=15):
    """Pose plus the projector's focal length. A wall alone cannot separate the focal
    length from the distance to the wall (a longer focal and a farther projector draw
    the same dots), so the search is only run when enough dots lie well in front of the
    bulk of the scan: dots closer than near_depth_fraction times the median dot depth.
    Without them the focal stays at focal_guess_pixels and only the pose is solved.

    The search is deterministic: for each candidate focal the pose is refined by
    Levenberg-Marquardt from the fixed-focal solution and every dot's capped squared
    error is summed; a coarse sweep is followed by finer ones around the best value.
    Returns (calibration, inlier_mask) or (None, None)."""
    points_3d = numpy.asarray(points_3d, dtype=numpy.float64).reshape(-1, 3)
    projector_points = numpy.asarray(projector_points, dtype=numpy.float64).reshape(-1, 2)
    start, inliers = fit_projector_pose(points_3d, projector_points, projector_width, projector_height, outlier_pixels,
                                        focal_guess_pixels, principal_point, distortion, min_points)
    if start is None:
        return None, None
    depths = points_3d[:, 2]
    # Under the wrong focal the near dots are exactly the ones that miss by tens of pixels,
    # so they are counted among every decodable dot, not among the fixed-focal inliers.
    decodable = start.reprojection_errors(points_3d, projector_points) < 20.0 * outlier_pixels
    near = decodable & (depths < near_depth_fraction * numpy.median(depths[decodable]))
    near_count = int(near.sum())
    if near_count < min_near_dots:
        print(f"[calibration] focal length kept at {focal_guess_pixels:.0f} px: only {near_count} dots closer than "
              f"{near_depth_fraction:.0%} of the median depth (need {min_near_dots}); sit in the beam for a scan")
        start.focal_source = "assumed"
        return start, inliers

    distortion_vector = numpy.zeros(5)
    distortion_vector[:2] = distortion[:2]
    cap_pixels = 2.0 * outlier_pixels

    def solve(focal):
        camera_matrix = numpy.array([[focal, 0.0, principal_point[0]],
                                     [0.0, focal, principal_point[1]],
                                     [0.0, 0.0, 1.0]])
        calibration = _refine_pose(points_3d, projector_points, camera_matrix, distortion_vector,
                                   start.rotation_vector, start.translation_vector, projector_width, projector_height)
        errors = calibration.reprojection_errors(points_3d, projector_points)
        return calibration, errors, _capped_cost(errors, cap_pixels)

    range_low, range_high = focal_guess_pixels * (1.0 - search_fraction), focal_guess_pixels * (1.0 + search_fraction)
    low, high = range_low, range_high
    best = None
    for step_pixels in (50.0, 10.0, 2.0):
        candidates = numpy.arange(low, high + step_pixels / 2, step_pixels)
        results = [(solve(float(focal)), float(focal)) for focal in candidates]
        (calibration, errors, cost), focal = min(results, key=lambda item: item[0][2])
        if best is None or cost < best[2]:
            best = (calibration, errors, cost, focal)
        low, high = focal - step_pixels, focal + step_pixels
    calibration, errors, _, focal = best
    if focal - range_low < 1.0 or range_high - focal < 1.0:
        print(f"[calibration] focal search hit the edge of its range at {focal:.0f} px: the near dots disagree with "
              f"the wall; keeping the assumed {focal_guess_pixels:.0f} px")
        start.focal_source = "assumed"
        return start, inliers

    inliers = errors < outlier_pixels
    if inliers.sum() < 6:
        return None, None
    calibration = _refine_pose(points_3d[inliers], projector_points[inliers], calibration.camera_matrix,
                               distortion_vector, calibration.rotation_vector, calibration.translation_vector,
                               projector_width, projector_height)
    errors = calibration.reprojection_errors(points_3d, projector_points)
    inliers = errors < outlier_pixels
    calibration.rms_pixels = _root_mean_square(errors[inliers])
    calibration.point_count = int(len(points_3d))
    calibration.inlier_count = int(inliers.sum())
    calibration.depth_range_metres = (float(depths[inliers].min()), float(depths[inliers].max()))
    calibration.focal_source = f"fitted from {near_count} near dots"

    # The check that matters for a face: at the near dots, how much wider does the model
    # draw the dot pattern than the projector really did? 1.00 = true to size.
    near_inliers = near & inliers
    measured = projector_points[near_inliers]
    modelled = calibration.project(points_3d[near_inliers])
    centre = measured.mean(axis=0)
    size_ratio = float(numpy.median(numpy.linalg.norm(modelled - centre, axis=1) /
                                    numpy.maximum(numpy.linalg.norm(measured - centre, axis=1), 1.0)))
    print(f"[calibration] focal length fitted: {focal:.0f} px (assumed {focal_guess_pixels:.0f}) from {near_count} near "
          f"dots; {int(inliers.sum())}/{len(points_3d)} dots within {outlier_pixels:.0f} px, rms {calibration.rms_pixels:.2f} px, "
          f"near-dot size ratio model/real {size_ratio:.3f}, projector at "
          f"{numpy.array2string(calibration.position_metres(), precision=2)} m")
    return calibration, inliers


# ---------------------------------------------------------------- complete model
COMPLETE_MODEL_PARAMETER_NAMES = ("focal", "centre_x", "centre_y", "distortion_k1", "distortion_k2",
                                  "rotation_1", "rotation_2", "rotation_3",
                                  "translation_1", "translation_2", "translation_3")


def _calibration_from_parameters(parameters, projector_width, projector_height):
    camera_matrix = numpy.array([[parameters[0], 0.0, parameters[1]],
                                 [0.0, parameters[0], parameters[2]],
                                 [0.0, 0.0, 1.0]])
    distortion_vector = numpy.zeros(5)
    distortion_vector[:2] = parameters[3:5]
    return ProjectorCalibration(camera_matrix, distortion_vector, parameters[5:8], parameters[8:11],
                                projector_width, projector_height)


def _residuals_and_jacobian(parameters, points_3d, projector_points):
    """Reprojection residuals (2N,) and their derivatives (2N, 11) by the model parameters."""
    camera_matrix = numpy.array([[parameters[0], 0.0, parameters[1]],
                                 [0.0, parameters[0], parameters[2]],
                                 [0.0, 0.0, 1.0]])
    distortion_vector = numpy.zeros(5)
    distortion_vector[:2] = parameters[3:5]
    projected, derivatives = cv2.projectPoints(points_3d.reshape(-1, 1, 3), parameters[5:8].reshape(3, 1),
                                               parameters[8:11].reshape(3, 1), camera_matrix, distortion_vector)
    residuals = (projected.reshape(-1, 2) - projector_points).reshape(-1)
    # Columns as OpenCV orders them: rotation (3), translation (3), focal x, focal y, centre x, centre y, k1, k2, p1, p2, k3
    jacobian = numpy.column_stack([derivatives[:, 6] + derivatives[:, 7], derivatives[:, 8], derivatives[:, 9],
                                   derivatives[:, 10], derivatives[:, 11], derivatives[:, 0:3], derivatives[:, 3:6]])
    return residuals, jacobian


def _robust_least_squares(parameters, free, points_3d, projector_points, point_weights, huber_pixels, iterations=60):
    """Levenberg-Marquardt over the free parameters with Huber weights: a point far from the
    model pulls with a force that stops growing, so wrong correspondences cannot drag the fit."""
    parameters = parameters.copy()
    free_indices = numpy.flatnonzero(free)
    damping = 1e-3

    def weighted_cost(residuals):
        errors = numpy.linalg.norm(residuals.reshape(-1, 2), axis=1)
        huber = numpy.where(errors <= huber_pixels, 0.5 * errors ** 2, huber_pixels * (errors - 0.5 * huber_pixels))
        return float(numpy.sum(point_weights * huber))

    residuals, jacobian = _residuals_and_jacobian(parameters, points_3d, projector_points)
    cost = weighted_cost(residuals)
    for _ in range(iterations):
        errors = numpy.linalg.norm(residuals.reshape(-1, 2), axis=1)
        robust = numpy.where(errors <= huber_pixels, 1.0, huber_pixels / numpy.maximum(errors, 1e-9))
        weights = numpy.repeat(point_weights * robust, 2)
        free_jacobian = jacobian[:, free_indices]
        normal = free_jacobian.T @ (free_jacobian * weights[:, None])
        gradient = free_jacobian.T @ (weights * residuals)
        improved = False
        relative_gain = 0.0
        for _ in range(8):
            try:
                step = numpy.linalg.solve(normal + damping * numpy.diag(numpy.diag(normal) + 1e-12), -gradient)
            except numpy.linalg.LinAlgError:
                damping *= 10.0
                continue
            candidate = parameters.copy()
            candidate[free_indices] += step
            candidate_residuals, candidate_jacobian = _residuals_and_jacobian(candidate, points_3d, projector_points)
            candidate_cost = weighted_cost(candidate_residuals)
            if numpy.isfinite(candidate_cost) and candidate_cost < cost:
                relative_gain = (cost - candidate_cost) / max(cost, 1e-12)
                parameters, residuals, jacobian, cost = candidate, candidate_residuals, candidate_jacobian, candidate_cost
                damping = max(damping * 0.3, 1e-9)
                improved = True
                break
            damping *= 10.0
        if not improved or relative_gain < 1e-9:
            break
    return parameters


def depth_balance_weights(points_3d, bin_metres=0.4):
    """Per-point weights that give every distance band the same total say. A wall holds
    most of the points of a scan; without this the fit serves the wall and the few near
    points, which are the ones that fix the focal length, barely count."""
    bins = numpy.floor(points_3d[:, 2] / bin_metres).astype(int)
    weights = numpy.zeros(len(points_3d))
    occupied = numpy.unique(bins)
    for value in occupied:
        members = bins == value
        weights[members] = 1.0 / members.sum()
    return weights * len(points_3d) / len(occupied)


def fit_projector_complete(points_3d, projector_points, projector_width, projector_height, outlier_pixels,
                           focal_guess_pixels, principal_point_guess, distortion_guess, min_points=40,
                           minimum_depth_ratio=1.8, quiet=False):
    """The complete projector model from correspondences spread over near and far surfaces:
    focal length, lens centre, two radial distortion terms, rotation and translation.

    Staged, each stage starting from the last: the pose alone with the guessed intrinsics
    (well posed whatever the scene), then the focal length, then the lens centre, then the
    distortion. Every stage is a robust least-squares fit with all distance bands weighted
    equally. The intrinsics are only released when the points span distances in a ratio of
    at least minimum_depth_ratio; a single wall cannot determine them.
    Returns (calibration, inlier_mask) or (None, None)."""
    points_3d = numpy.asarray(points_3d, dtype=numpy.float64).reshape(-1, 3)
    projector_points = numpy.asarray(projector_points, dtype=numpy.float64).reshape(-1, 2)
    if len(points_3d) < min_points:
        return None, None
    start, _ = fit_projector_pose(points_3d, projector_points, projector_width, projector_height, 3.0 * outlier_pixels,
                                  focal_guess_pixels, principal_point_guess, distortion_guess, min_points=12)
    if start is None:
        return None, None
    parameters = numpy.concatenate([[focal_guess_pixels, principal_point_guess[0], principal_point_guess[1],
                                     distortion_guess[0], distortion_guess[1]],
                                    start.rotation_vector.ravel(), start.translation_vector.ravel()])
    depth_low, depth_high = numpy.percentile(points_3d[:, 2], [2, 98])
    depth_ratio = depth_high / max(depth_low, 1e-6)
    point_weights = depth_balance_weights(points_3d)
    pose_only = numpy.array([False] * 5 + [True] * 6)
    stages = [("pose", pose_only)]
    if depth_ratio >= minimum_depth_ratio:
        with_focal = pose_only.copy()
        with_focal[0] = True
        with_centre = with_focal.copy()
        with_centre[1:3] = True
        with_distortion = with_centre.copy()
        with_distortion[3:5] = True
        stages += [("pose + focal", with_focal), ("+ lens centre", with_centre), ("+ distortion", with_distortion)]
    else:
        print(f"[calibration] the points span distances of {depth_low:.2f} .. {depth_high:.2f} m (ratio {depth_ratio:.1f}): "
              f"too little to fit the intrinsics, which stay as guessed; put an object near the projector into the beam")
    usable = numpy.ones(len(points_3d), dtype=bool)
    for name, free in stages:
        for huber_pixels in (3.0 * outlier_pixels, outlier_pixels):
            parameters = _robust_least_squares(parameters, free, points_3d[usable], projector_points[usable],
                                               point_weights[usable], huber_pixels)
        calibration = _calibration_from_parameters(parameters, projector_width, projector_height)
        errors = calibration.reprojection_errors(points_3d, projector_points)
        usable = errors < 5.0 * outlier_pixels                      # gross mistakes leave; doubtful points stay, down-weighted
        if not quiet:
            print(f"[calibration] stage {name:14s}: focal {parameters[0]:7.1f} px, lens centre ({parameters[1]:6.1f}, "
                  f"{parameters[2]:6.1f}), distortion {parameters[3]:+.3f} {parameters[4]:+.3f}, "
                  f"median error {numpy.median(errors):.2f} px, "
                  f"{int((errors < outlier_pixels).sum())}/{len(errors)} within {outlier_pixels:.0f} px")
    calibration = _calibration_from_parameters(parameters, projector_width, projector_height)
    errors = calibration.reprojection_errors(points_3d, projector_points)
    inliers = errors < outlier_pixels
    if inliers.sum() < min_points // 2:
        return None, None
    calibration.rms_pixels = _root_mean_square(errors[inliers])
    calibration.point_count = int(len(points_3d))
    calibration.inlier_count = int(inliers.sum())
    calibration.depth_range_metres = (float(points_3d[inliers, 2].min()), float(points_3d[inliers, 2].max()))
    calibration.focal_source = (f"complete model fitted from {len(points_3d)} points at {depth_low:.2f} .. {depth_high:.2f} m"
                                if len(stages) > 1 else "assumed")
    return calibration, inliers


def report_errors_by_distance(calibration, points_3d, projector_points, label, band_edges_metres=(0.0, 1.0, 2.0, 10.0)):
    """Prints the reprojection error per distance band: a model is only right when the near
    band and the far band are both small."""
    errors = calibration.reprojection_errors(points_3d, projector_points)
    parts = []
    for low, high in zip(band_edges_metres[:-1], band_edges_metres[1:]):
        members = (points_3d[:, 2] >= low) & (points_3d[:, 2] < high)
        if members.any():
            parts.append(f"{low:.0f}-{high:.0f} m: median {numpy.median(errors[members]):.2f} px, "
                         f"90% under {numpy.percentile(errors[members], 90):.1f} px ({int(members.sum())} points)")
    print(f"[calibration] {label}: " + "; ".join(parts))
    return errors
