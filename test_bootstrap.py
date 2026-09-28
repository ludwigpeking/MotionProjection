"""Run only the bootstrap grid, fit a homography from it, and report. Exits on its own."""

import cv2

import config
from bootstrap import run_bootstrap
from camera import Camera
from displays import pick_projector_monitor
from face_tracker import FaceTracker
from homography_servo import HomographyEstimator
from projector import ProjectorWindow


def main():
    monitor, is_second_screen = pick_projector_monitor(
        config.projector_monitor_index,
        config.projector_fallback_width_pixels,
        config.projector_fallback_height_pixels,
    )
    camera = Camera(config)
    projector = ProjectorWindow(monitor, fullscreen=is_second_screen)
    cv2.namedWindow("debug", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("debug", 960, 540)

    face_tracker = FaceTracker(config, camera.frame_width, camera.frame_height)
    pairs, wall_pairs = run_bootstrap(camera, projector, face_tracker, config, "debug")
    estimator = HomographyEstimator(config, camera.frame_width, camera.frame_height,
                                    projector.width, projector.height)
    landmarks = face_tracker.last_raw_landmarks
    face_size = face_tracker.face_size(landmarks) if landmarks is not None else 0.0
    estimator.set_bootstrap_pairs(pairs, face_size)
    if estimator.is_valid():
        print(f"[test] mapping valid ({estimator.model_name}): inliers {estimator.inlier_count}/{len(pairs)}, "
              f"residual {estimator.residual_rms_pixels:.1f} px, face size {face_size:.0f} px")
        for camera_point, projector_point in pairs:
            mapped = estimator.map_points([camera_point], face_size)[0]
            error = ((mapped[0] - projector_point[0]) ** 2 + (mapped[1] - projector_point[1]) ** 2) ** 0.5
            print(f"[test]   projector ({projector_point[0]:6.0f},{projector_point[1]:6.0f}) "
                  f"predicted ({mapped[0]:6.0f},{mapped[1]:6.0f})  error {error:5.1f} px")
        estimator.save(config.homography_file_path)
    else:
        print("[test] homography NOT valid")

    cv2.waitKey(1500)
    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
