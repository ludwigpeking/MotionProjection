"""Draw how well one homography explains the bootstrap's wall dots.

Reads diagnostics/bootstrap_pairs.json, fits a homography to the wall dots
(RANSAC), and draws each dot's residual as an arrow (magnified) on a black
image the size of the camera frame. A radial pattern of arrows means lens
distortion (camera or projector); random directions mean noise; a smooth
non-radial pattern means the dots are not on one plane.

Usage: python residual_map.py [pairs.json] [magnification]
"""

import json
import sys

import cv2
import numpy

pairs_path = sys.argv[1] if len(sys.argv) > 1 else "diagnostics/bootstrap_pairs.json"
magnification = float(sys.argv[2]) if len(sys.argv) > 2 else 8.0

with open(pairs_path) as file:
    pairs = json.load(file)
wall = [pair for pair in pairs if not pair["on_face"]]
camera_points = numpy.array([pair["camera"] for pair in wall], dtype=numpy.float64)
projector_points = numpy.array([pair["projector"] for pair in wall], dtype=numpy.float64)
print(f"{len(wall)} wall dots, {len(pairs) - len(wall)} face dots")

homography, inlier_mask = cv2.findHomography(camera_points, projector_points, cv2.RANSAC, 6.0)
inliers = inlier_mask.ravel().astype(bool)
inverse = numpy.linalg.inv(homography)
# Residual in camera pixels: where the projector dot should appear according to H, versus where it was seen.
predicted_camera = cv2.perspectiveTransform(projector_points.reshape(-1, 1, 2), inverse).reshape(-1, 2)
residuals = camera_points - predicted_camera
magnitudes = numpy.linalg.norm(residuals, axis=1)
print(f"homography inliers {inliers.sum()}/{len(wall)}; residual rms {numpy.sqrt(numpy.mean(magnitudes[inliers] ** 2)):.2f} px, "
      f"max {magnitudes[inliers].max():.1f} px (inliers), max {magnitudes.max():.1f} px (all)")

# Is the residual radial about the image centre (lens distortion signature)?
center = camera_points[inliers].mean(axis=0)
radial_direction = camera_points - center
radial_direction /= numpy.linalg.norm(radial_direction, axis=1, keepdims=True) + 1e-9
radial_component = numpy.sum(residuals * radial_direction, axis=1)
distance_from_center = numpy.linalg.norm(camera_points - center, axis=1)
if inliers.sum() > 3:
    correlation = numpy.corrcoef(distance_from_center[inliers], radial_component[inliers])[0, 1]
    print(f"radial component vs distance from centre: correlation {correlation:+.2f} "
          f"(near +/-1 = lens distortion; near 0 = not radial)")

image = numpy.zeros((720, 1280, 3), dtype=numpy.uint8)
for (x, y), residual, inlier in zip(camera_points, residuals, inliers):
    color = (0, 255, 0) if inlier else (0, 0, 255)
    end = (int(x + magnification * residual[0]), int(y + magnification * residual[1]))
    cv2.circle(image, (int(x), int(y)), 3, color, -1)
    cv2.arrowedLine(image, (int(x), int(y)), end, color, 1, tipLength=0.3)
cv2.putText(image, f"wall-dot residuals x{magnification:.0f}  (green = inlier, red = outlier)", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
cv2.imwrite("diagnostics/bootstrap_residuals.png", image)
print("wrote diagnostics/bootstrap_residuals.png")
