"""Depth-aware estimate of the camera-pixel -> projector-pixel mapping.

The mapping between camera and projector for a plane at depth d is
H(d) = K_p (R + t n^T / d) K_c^-1: linear in 1/d. MediaPipe's face size in
pixels is proportional to 1/d, so the mapping is modelled as a fixed base
homography plus a projector-space translation that is linear in face size:

    p = T(t0 + t1 * (face_size - size_center)) . H_base . x

H_base comes from the coded-dot bootstrap. Every servo measurement is a true
correspondence (camera point where a dot was seen, projector point where it
was sent) tagged with the face size at that moment; t0 and t1 are refitted
from the recent ones by ridge-regularised least squares, so with all
measurements at one depth t1 stays near zero and the model degrades to a
plain shift, and as the person moves in depth the slope is learnt.
"""

import collections
import json

import cv2
import numpy


def _translation_matrix(shift):
    return numpy.array([[1.0, 0.0, shift[0]], [0.0, 1.0, shift[1]], [0.0, 0.0, 1.0]])


class HomographyEstimator:
    def __init__(self, config, camera_width, camera_height, projector_width, projector_height):
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.projector_width = projector_width
        self.projector_height = projector_height
        self.ransac_reprojection_threshold_pixels = config.servo_ransac_reprojection_threshold_pixels
        self.min_points_for_homography = config.servo_min_points_for_homography
        self.min_face_points_before_dropping_bootstrap = config.servo_min_face_points_before_dropping_bootstrap
        self.depth_slope_ridge = config.servo_depth_slope_ridge
        self.outlier_threshold_pixels = config.servo_outlier_threshold_pixels
        self.size_extrapolation_margin = config.servo_size_extrapolation_margin_pixels
        self.min_bootstrap_pairs = config.bootstrap_min_pairs_for_mapping
        self.depth_model_enabled = config.depth_model_enabled
        self.min_pairs_for_depth_fit = config.depth_model_min_face_dots
        self.depth_vector = numpy.zeros(2)
        self.size_minimum_seen = None
        self.size_maximum_seen = None

        # (camera_point, projector_point, face_size)
        self.bootstrap_pairs = []
        self.face_pairs = collections.deque(maxlen=config.servo_correspondence_window_size)

        self.base_homography = None
        self.translation_intercept = numpy.zeros(2)
        self.translation_slope = numpy.zeros(2)
        self.size_center = 0.0
        self.model_name = "none"
        self.inlier_count = 0
        self.residual_rms_pixels = float("nan")

    # ------------------------------------------------------------ input

    def _make_pair(self, camera_point, projector_point, face_size, depth=None):
        """Pairs store their residual against the base in force when they were
        measured, net of the per-landmark depth term when a depth is known.
        The remaining residual is the whole-face parallax shift."""
        residual = numpy.asarray(projector_point, dtype=numpy.float64) - self._map_base([camera_point])[0]
        if depth is not None and self.depth_model_enabled:
            residual = residual - self.depth_vector * depth
        return (tuple(camera_point), tuple(projector_point), float(face_size), residual, depth)

    def set_base(self, base_homography):
        """Replace the base mapping with a live measurement (border markers)."""
        if base_homography is not None and self.is_sane(base_homography):
            self.base_homography = base_homography
            if self.model_name in ("none", "loaded"):
                self.model_name = "live"

    def set_bootstrap_pairs(self, pairs, face_size, base_homography=None):
        """A fresh bootstrap replaces everything. With a wall-plane base from the
        border markers, the face pairs only set the parallax shift; otherwise
        the base itself is fitted from the pairs."""
        self.face_pairs.clear()
        self.bootstrap_pairs = []
        self.translation_intercept = numpy.zeros(2)
        self.translation_slope = numpy.zeros(2)
        self.size_center = face_size
        self.size_minimum_seen = None
        self.size_maximum_seen = None
        self.base_homography = None
        self.model_name = "none"
        self.inlier_count = 0
        self.residual_rms_pixels = float("nan")
        if len(pairs) < self.min_bootstrap_pairs:
            print(f"[servo] only {len(pairs)} bootstrap correspondences; "
                  f"at least {self.min_bootstrap_pairs} needed for a mapping")
            return False
        camera_points = numpy.array([pair[0] for pair in pairs], dtype=numpy.float64)
        projector_points = numpy.array([pair[1] for pair in pairs], dtype=numpy.float64)
        if base_homography is not None and self.is_sane(base_homography):
            self.base_homography = base_homography
            self.model_name = "live"
            inliers = numpy.ones(len(pairs), dtype=bool)
        else:
            fit = self._fit_base(camera_points, projector_points)
            if fit is None:
                print("[servo] bootstrap correspondences do not support a plausible mapping")
                return False
            self.base_homography, inliers, self.model_name = fit
        # Per-landmark depth term: fitted from the face dots that carry a MediaPipe z.
        self.depth_vector = numpy.zeros(2)
        depth_pairs = [pair for pair, inlier in zip(pairs, inliers) if inlier and len(pair) > 2 and pair[2] is not None]
        if self.depth_model_enabled and len(depth_pairs) >= self.min_pairs_for_depth_fit:
            residuals = numpy.array([numpy.asarray(pair[1], dtype=numpy.float64) - self._map_base([pair[0]])[0]
                                     for pair in depth_pairs])
            depths = numpy.array([pair[2] for pair in depth_pairs], dtype=numpy.float64)
            design = numpy.column_stack([numpy.ones_like(depths), depths])
            solution, _, rank, _ = numpy.linalg.lstsq(design, residuals, rcond=None)
            depth_spread = float(depths.max() - depths.min())
            if rank == 2 and depth_spread > 1e-6:
                self.depth_vector = solution[1]
                predicted = design @ solution
                depth_fit_rms = float(numpy.sqrt(numpy.mean(numpy.sum((residuals - predicted) ** 2, axis=1))))
                plain_rms = float(numpy.sqrt(numpy.mean(numpy.sum((residuals - residuals.mean(axis=0)) ** 2, axis=1))))
                print(f"[servo] depth term from {len(depth_pairs)} face dots (z spread {depth_spread:.0f}): "
                      f"vector ({self.depth_vector[0]:+.2f},{self.depth_vector[1]:+.2f}) px per unit z; "
                      f"face residual {plain_rms:.1f} px -> {depth_fit_rms:.1f} px with depth")
        self.bootstrap_pairs = [self._make_pair(pair[0], pair[1], face_size, pair[2] if len(pair) > 2 else None)
                                for pair, inlier in zip(pairs, inliers) if inlier]
        self._refit_translation()
        return True

    def add_face_pairs(self, pairs, face_size, depths=None):
        """pairs: [(camera_point, projector_point), ...]; depths: matching MediaPipe z values or None."""
        if self.base_homography is None:
            return
        if depths is None:
            depths = [None] * len(pairs)
        self.face_pairs.extend(self._make_pair(camera_point, projector_point, face_size, depth)
                               for (camera_point, projector_point), depth in zip(pairs, depths))
        if len(self.face_pairs) >= self.min_face_points_before_dropping_bootstrap and self.bootstrap_pairs:
            print("[servo] enough face correspondences; dropping bootstrap points")
            self.bootstrap_pairs = []
        self._refit_translation()

    def clear_face_pairs(self):
        self.face_pairs.clear()
        self._refit_translation()

    # -------------------------------------------------------------- fit

    def _fit_base(self, camera_points, projector_points):
        """Richest model the point count supports. Returns (homography, inlier_mask, name) or None."""
        count = len(camera_points)
        threshold = self.ransac_reprojection_threshold_pixels
        if count >= self.min_points_for_homography:
            candidate, inlier_mask = cv2.findHomography(camera_points, projector_points, cv2.RANSAC, threshold)
            if candidate is not None and inlier_mask is not None and self.is_sane(candidate) \
                    and int(inlier_mask.sum()) >= 4:
                return candidate, inlier_mask.ravel().astype(bool), "homography"
        if count >= 3:
            affine, inlier_mask = cv2.estimateAffine2D(camera_points, projector_points, ransacReprojThreshold=threshold)
            if affine is not None and inlier_mask is not None and int(inlier_mask.sum()) >= 3:
                candidate = numpy.vstack([affine, [0.0, 0.0, 1.0]])
                if self.is_sane(candidate):
                    return candidate, inlier_mask.ravel().astype(bool), "affine"
        if count >= 2:
            similarity, inlier_mask = cv2.estimateAffinePartial2D(camera_points, projector_points,
                                                                  ransacReprojThreshold=threshold)
            if similarity is not None and inlier_mask is not None and int(inlier_mask.sum()) >= 2:
                candidate = numpy.vstack([similarity, [0.0, 0.0, 1.0]])
                if self.is_sane(candidate):
                    return candidate, inlier_mask.ravel().astype(bool), "similarity"
        return None

    def _refit_translation(self):
        """Ridge least squares of residual = t0 + t1 * (face_size - size_center),
        with one outlier-rejection pass. Outlying face pairs are discarded."""
        if self.base_homography is None:
            return
        pairs = list(self.bootstrap_pairs) + list(self.face_pairs)
        if not pairs:
            self.translation_intercept = numpy.zeros(2)
            self.translation_slope = numpy.zeros(2)
            self.inlier_count = 0
            self.residual_rms_pixels = float("nan")
            return
        sizes = numpy.array([pair[2] for pair in pairs], dtype=numpy.float64) - self.size_center
        residuals = numpy.array([pair[3] for pair in pairs], dtype=numpy.float64)

        inliers = numpy.ones(len(pairs), dtype=bool)
        for _ in range(2):
            weight_sum = float(inliers.sum())
            if weight_sum == 0:
                break
            size_sum = float(sizes[inliers].sum())
            size_square_sum = float((sizes[inliers] ** 2).sum()) + self.depth_slope_ridge
            normal_matrix = numpy.array([[weight_sum, size_sum], [size_sum, size_square_sum]])
            intercept = numpy.zeros(2)
            slope = numpy.zeros(2)
            for axis in range(2):
                right_hand_side = numpy.array([residuals[inliers, axis].sum(),
                                               (sizes[inliers] * residuals[inliers, axis]).sum()])
                intercept[axis], slope[axis] = numpy.linalg.solve(normal_matrix, right_hand_side)
            predicted = intercept[None, :] + sizes[:, None] * slope[None, :]
            errors = numpy.linalg.norm(residuals - predicted, axis=1)
            inliers = errors < self.outlier_threshold_pixels

        self.translation_intercept = intercept
        self.translation_slope = slope
        observed_sizes = sizes[inliers] + self.size_center
        if len(observed_sizes):
            self.size_minimum_seen = float(observed_sizes.min())
            self.size_maximum_seen = float(observed_sizes.max())
        self.inlier_count = int(inliers.sum())
        self.residual_rms_pixels = float(numpy.sqrt(numpy.mean(errors[inliers] ** 2))) if self.inlier_count else float("nan")

        bootstrap_count = len(self.bootstrap_pairs)
        kept_face_pairs = [pair for pair, inlier in zip(list(self.face_pairs), inliers[bootstrap_count:]) if inlier]
        dropped = len(self.face_pairs) - len(kept_face_pairs)
        if dropped:
            self.face_pairs.clear()
            self.face_pairs.extend(kept_face_pairs)

    # ------------------------------------------------------------ query

    def _map_base(self, camera_points):
        points = numpy.asarray(camera_points, dtype=numpy.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(points, self.base_homography).reshape(-1, 2)

    def translation_for_size(self, face_size):
        """The depth slope is only trusted over the range of face sizes it was
        fitted on (plus a small margin); beyond that the shift is held constant
        rather than extrapolated, so a poorly determined slope cannot throw the
        mapping off when the person steps back."""
        if self.size_minimum_seen is not None:
            face_size = float(numpy.clip(face_size,
                                         self.size_minimum_seen - self.size_extrapolation_margin,
                                         self.size_maximum_seen + self.size_extrapolation_margin))
        return self.translation_intercept + self.translation_slope * (face_size - self.size_center)

    def homography_for_size(self, face_size):
        return _translation_matrix(self.translation_for_size(face_size)) @ self.base_homography

    def map_points(self, camera_points, face_size, depths=None):
        """Camera points -> projector points. depths: per-point MediaPipe z (adds the parallax term)."""
        points = numpy.asarray(camera_points, dtype=numpy.float64).reshape(-1, 1, 2)
        mapped = cv2.perspectiveTransform(points, self.homography_for_size(face_size)).reshape(-1, 2)
        if depths is not None and self.depth_model_enabled:
            mapped = mapped + numpy.asarray(depths, dtype=numpy.float64).reshape(-1, 1) * self.depth_vector[None, :]
        return mapped

    def is_valid(self):
        return self.base_homography is not None

    def face_pair_count(self):
        return len(self.face_pairs)

    def is_sane(self, homography):
        """Reject fits that map the camera frame to something absurd."""
        if not numpy.all(numpy.isfinite(homography)):
            return False
        corners = numpy.array([
            [0, 0], [self.camera_width, 0],
            [self.camera_width, self.camera_height], [0, self.camera_height],
        ], dtype=numpy.float64)
        mapped_corners = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), homography).reshape(-1, 2)
        if not numpy.all(numpy.isfinite(mapped_corners)):
            return False
        edges = numpy.roll(mapped_corners, -1, axis=0) - mapped_corners
        next_edges = numpy.roll(edges, -1, axis=0)
        cross_products = edges[:, 0] * next_edges[:, 1] - edges[:, 1] * next_edges[:, 0]
        if not (numpy.all(cross_products > 0) or numpy.all(cross_products < 0)):
            return False
        mapped_area = 0.5 * abs(numpy.sum(
            mapped_corners[:, 0] * numpy.roll(mapped_corners[:, 1], -1)
            - mapped_corners[:, 1] * numpy.roll(mapped_corners[:, 0], -1)))
        projector_area = self.projector_width * self.projector_height
        return 0.02 * projector_area < mapped_area < 50.0 * projector_area

    # ------------------------------------------------------------ files

    def save(self, path):
        if self.base_homography is None:
            return False
        with open(path, "w") as file:
            json.dump({
                "base_homography_camera_to_projector": self.base_homography.tolist(),
                "translation_intercept": self.translation_intercept.tolist(),
                "translation_slope": self.translation_slope.tolist(),
                "depth_vector": self.depth_vector.tolist(),
                "size_center": self.size_center,
                "size_range_seen": [self.size_minimum_seen, self.size_maximum_seen],
                "camera_size": [self.camera_width, self.camera_height],
                "projector_size": [self.projector_width, self.projector_height],
            }, file, indent=2)
        print(f"[servo] saved mapping to {path}")
        return True

    def load(self, path):
        try:
            with open(path) as file:
                data = json.load(file)
        except (OSError, json.JSONDecodeError):
            return False
        if data.get("camera_size") != [self.camera_width, self.camera_height] or \
                data.get("projector_size") != [self.projector_width, self.projector_height] or \
                "base_homography_camera_to_projector" not in data:
            print("[servo] saved mapping is for a different setup or format; ignoring")
            return False
        base = numpy.array(data["base_homography_camera_to_projector"], dtype=numpy.float64)
        if not self.is_sane(base):
            return False
        self.base_homography = base
        self.translation_intercept = numpy.array(data.get("translation_intercept", [0.0, 0.0]))
        self.translation_slope = numpy.array(data.get("translation_slope", [0.0, 0.0]))
        self.depth_vector = numpy.array(data.get("depth_vector", [0.0, 0.0]))
        self.size_center = float(data.get("size_center", 0.0))
        size_range = data.get("size_range_seen", [None, None])
        self.size_minimum_seen = size_range[0]
        self.size_maximum_seen = size_range[1]
        self.model_name = "loaded"
        self.inlier_count = 0
        self.residual_rms_pixels = float("nan")
        print(f"[servo] loaded mapping from {path}")
        return True
