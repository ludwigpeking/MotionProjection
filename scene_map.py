"""Dense camera -> projector map measured by a coded-dot scan.

Every decoded dot is one exact correspondence, at whatever depth the surface
there happens to be. A query point is mapped by a local affine fit through
its nearest measured dots, so no plane is assumed anywhere: the depth is in
the measurements. Where no dot landed nearby, the map says so.
"""

import numpy


class SceneMap:
    def __init__(self, pairs, neighbour_count, max_neighbour_distance_pixels, outlier_pixels=None):
        self.camera_points = numpy.array([pair[0] for pair in pairs], dtype=numpy.float64).reshape(-1, 2)
        self.projector_points = numpy.array([pair[1] for pair in pairs], dtype=numpy.float64).reshape(-1, 2)
        self.neighbour_count = neighbour_count
        self.max_neighbour_distance = max_neighbour_distance_pixels
        self.dropped_outliers = 0
        if outlier_pixels is not None:
            self._drop_outliers(outlier_pixels)

    def _drop_outliers(self, outlier_pixels):
        """Drop a dot whose projector coordinate disagrees with the local affine
        fitted through its neighbours (a mis-decoded ID or a stray blob)."""
        if len(self.camera_points) < self.neighbour_count + 2:
            return
        keep = numpy.ones(len(self.camera_points), dtype=bool)
        for index in range(len(self.camera_points)):
            distances = numpy.linalg.norm(self.camera_points - self.camera_points[index], axis=1)
            distances[index] = numpy.inf
            order = numpy.argsort(distances)[:self.neighbour_count + 2]
            if distances[order[-1]] > 3 * self.max_neighbour_distance:
                continue
            design = numpy.hstack([self.camera_points[order], numpy.ones((len(order), 1))])
            solution, _, rank, _ = numpy.linalg.lstsq(design, self.projector_points[order], rcond=None)
            if rank < 3:
                continue
            predicted = numpy.array([*self.camera_points[index], 1.0]) @ solution
            if numpy.linalg.norm(predicted - self.projector_points[index]) > outlier_pixels:
                keep[index] = False
        self.dropped_outliers = int((~keep).sum())
        self.camera_points = self.camera_points[keep]
        self.projector_points = self.projector_points[keep]

    def __len__(self):
        return len(self.camera_points)

    def query(self, camera_point):
        """Projector point for a camera point, or None where the scan has no data."""
        if len(self.camera_points) < 3:
            return None
        query = numpy.asarray(camera_point, dtype=numpy.float64)
        distances = numpy.linalg.norm(self.camera_points - query, axis=1)
        order = numpy.argsort(distances)[:self.neighbour_count]
        if distances[order[0]] > self.max_neighbour_distance:
            return None
        neighbours_camera = self.camera_points[order]
        neighbours_projector = self.projector_points[order]
        # Weighted least-squares affine: projector = A . [x, y, 1]
        weights = 1.0 / (distances[order] + 5.0)
        design = numpy.hstack([neighbours_camera, numpy.ones((len(order), 1))]) * weights[:, None]
        target = neighbours_projector * weights[:, None]
        solution, _, rank, _ = numpy.linalg.lstsq(design, target, rcond=None)
        if rank < 3:
            return None
        return numpy.array([query[0], query[1], 1.0]) @ solution

    def nearest_distance(self, camera_point):
        if len(self.camera_points) == 0:
            return float("inf")
        return float(numpy.linalg.norm(self.camera_points - numpy.asarray(camera_point), axis=1).min())
