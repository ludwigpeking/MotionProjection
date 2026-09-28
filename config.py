"""All tunable settings for the feedback projection mapper.

Every value has a working default. Nothing here is a camera or projector
intrinsic: the mapping between the two is measured live by the servo loop.
"""

# ---------------------------------------------------------------- camera
camera_index = 0
camera_capture_width_pixels = 1280
camera_capture_height_pixels = 720
camera_target_frames_per_second = 30

# Auto exposure / auto white balance must be off, otherwise the projected
# light changes the camera gain and the frame differencing sees nothing.
camera_lock_exposure = True
camera_manual_exposure_value = -5       # log2(seconds): -5 = 1/32 s (one stop brighter than -6)
camera_manual_gain_value = None         # None keeps the driver default

# ------------------------------------------------------------- projector
projector_monitor_index = None          # None = first non-primary monitor
projector_fallback_width_pixels = 1920  # used only when no second monitor is found
projector_fallback_height_pixels = 1080

# ------------------------------------------------------------- rendering
render_scale_factor = 1.0               # panda canvas resolution relative to the camera frame (2.0 = sharper, slower)
landmark_dot_radius_pixels = 8          # in projector pixels, for the landmark debug projection
servo_warn_after_failed_cycles = 3      # debug view warns after this many cycles without dots

# Passive feedback from the projected content: with a single red landmark disc
# projected, the camera finds the disc every frame and compares it with the
# landmark. No flashing needed. Corrections are fed into the mapping like servo
# measurements, while the head is still.
passive_disc_tracking_enabled = True
passive_disc_correction_enabled = True
passive_disc_search_radius_pixels = 160
passive_disc_min_area_pixels = 25
passive_disc_max_speed_pixels_per_second = 40.0
passive_disc_update_period_seconds = 0.5
passive_disc_warning_pixels = 25.0
landmark_dot_brightness = 255
# Landmarks projected in the landmark debug mode: eye corners and lids, brows,
# nose bridge and tip, mouth corners and lips, chin, jaw, cheeks, forehead.
landmark_projection_indices = [
    33, 133, 159, 145, 263, 362, 386, 374,      # eyes
    70, 105, 107, 336, 334, 300,                # brows
    6, 4, 98, 327,                              # nose
    61, 291, 0, 17,                             # mouth
    152, 199, 234, 454, 50, 280, 10, 9,         # chin, jaw, cheeks, forehead
]
face_fill_brightness = 160              # white level of the panda fur (0..255); below 255 so the camera is not blown out
mouth_brightness = 60

# ---------------------------------------------------------- face tracking
face_landmarker_model_path = "face_landmarker.task"
face_landmarker_model_url = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
smoothing_min_cutoff_hertz = 4.0        # one-euro filter: lower = smoother when still, but more lag
smoothing_speed_coefficient = 0.05      # one-euro filter: higher = less lag when moving
smoothing_derivative_cutoff_hertz = 5.0 # velocity estimate responsiveness; too low = stale prediction
prediction_lead_seconds = 0.28          # extrapolate landmarks this far ahead: capture + loop + projector (~0.2 s)
prediction_lead_step_seconds = 0.02     # keys [ and ] change the lead by this much while running
# Automatic lead calibration: servo dots are aimed at the predicted landmark
# positions; while the head moves, how far they land behind the landmarks
# along the motion direction, divided by the speed, is the remaining latency.
lead_autotune_enabled = True
lead_autotune_min_speed_pixels_per_second = 120.0
lead_autotune_min_dots = 4              # only trust the lag estimate from a cycle with this many matched dots
lead_autotune_gain = 0.2                # fraction of the measured error applied per cycle
lead_autotune_max_seconds = 0.5
face_detection_downscale_factor = 2     # run MediaPipe on a frame this many times smaller (faster, slightly coarser)

# ---------------------------------------------------------------- timing
# Projector-to-camera latency, measured on this rig: a change shown at time T
# first appears (partially) in the camera frame read ~0.19-0.21 s later and is
# complete in the frame after that. So a frame read before T + 0.14 s shows
# nothing of it, and a frame read after T + 0.27 s shows all of it.
latency_none_before_seconds = 0.14
latency_full_after_seconds = 0.27
camera_frame_history_length = 45        # frames kept for picking reference/measurement frames

# ------------------------------------------------------------------ servo
servo_enabled_at_start = True
servo_period_seconds = 1.5              # one measurement cycle this often while converging
servo_period_converged_seconds = 4.0    # slower once the mapping is good (fewer visible flashes)
servo_converged_offset_pixels = 6.0     # "good" = mean landing offset below this, with enough face points
# During a cycle the mask is frozen for about
#   (full - none + margin) + (full - none + margin)  ~ 0.35 s
# so that the only change between reference and measurement frames is the dots.
servo_timing_margin_seconds = 0.05
# Landmarks the servo flashes dots on. Kept inside the face outline (not on the
# jaw edge) because the outline is where head motion produces the strongest
# false edges: forehead, between brows, left cheek, right cheek, chin, under
# left eye, under right eye.
servo_landmark_indices = [10, 9, 107, 336, 50, 280, 101, 330, 199, 234, 454]
servo_dot_radius_pixels = 14            # in projector pixels (about 5 px in the camera on this rig)
servo_dot_brightness = 255
servo_search_radius_pixels = 35         # gate around each expected landmark, in camera pixels
servo_search_radius_initial_pixels = 70 # wider gate while the mapping is still rough
servo_hide_mask_during_cycle = True     # blink the mask off while measuring: no projected edges in the frames
servo_face_box_expand_fraction = 0.15   # region in which the reference frame is warped to follow the face
servo_max_plausible_offset_pixels = 70.0  # a cycle whose dots land further than this from aim is not trusted
servo_face_points_for_tight_gate = 8    # switch to the tight gate after this many face correspondences
servo_min_points_for_homography = 8     # fewer face points: only shift the existing mapping
# All dots of one cycle land with nearly the same offset (parallax is a
# translation), so the aimed pattern is matched to the detected blobs under a
# single translation; a blob further than this from its shifted aim point does
# not count. At least servo_min_consistent_dots must match.
servo_pattern_tolerance_pixels = 18.0
servo_min_consistent_dots = 3           # dots aimed at the side of the face away from the projector rarely show
servo_pattern_scale_range = (0.85, 1.2) # plausible scale of the landed pattern relative to the aimed one
servo_pattern_max_rotation_degrees = 12.0
servo_reacquire_after_failed_cycles = 8 # misses in a row before re-running the bootstrap, if it never converged
servo_reacquire_after_failed_cycles_converged = 30  # misses in a row before re-bootstrapping a converged mapping
servo_converged_two_dot_tolerance_pixels = 10.0     # converged mapping: accept 2 dots under a small pure translation
servo_converged_two_dot_max_offset_pixels = 30.0
servo_min_peak_difference = 15          # minimum brightness step (0..255) to accept a dot
servo_max_blob_area_pixels = 600        # bigger blobs are motion streaks, not dots
servo_opening_radius_pixels = 2         # disc radius (camera px) of the opening that kills thin edges
servo_correspondence_window_size = 60   # most recent face correspondences kept for the fit
servo_depth_slope_ridge = 2000.0        # ridge weight on the depth slope: with little depth variation the slope stays ~0
servo_size_extrapolation_margin_pixels = 15.0  # beyond the fitted face-size range (+ margin) the shift is held constant
servo_outlier_threshold_pixels = 30.0   # a correspondence this far from the fitted model is discarded
servo_ransac_reprojection_threshold_pixels = 6.0
servo_min_face_points_before_dropping_bootstrap = 14

# A candidate is rejected if a darkening of more than this fraction of its own
# brightness sits right next to it: moving edges darken one side, dots never do.
dot_max_negative_ratio = 0.5

# -------------------------------------------------------------- bootstrap
bootstrap_grid_columns = 18             # coded-dot grid over the whole projector
bootstrap_grid_rows = 10
bootstrap_grid_margin_fraction = 0.05
bootstrap_dot_radius_pixels = 11
bootstrap_average_frames = 2
bootstrap_extra_settle_seconds = 0.12   # waited on top of the measured latency; a half-lit frame halves every dot
bootstrap_min_peak_difference = 12
bootstrap_max_blob_area_pixels = 4000
bootstrap_opening_radius_pixels = 2
bootstrap_code_sample_radius_pixels = 8 # disc around a blob centre searched for the dot when reading its ID bits
bootstrap_code_min_margin = 0.2         # each bit level (0 = black, 1 = all-on) must be this far from 0.5
bootstrap_min_blobs_for_valid_capture = 8   # fewer blobs in the all-on frame means the capture missed the dots
bootstrap_face_box_expand_fraction = 0.0    # camera face box grown by this fraction for selecting face dots
bootstrap_min_face_pairs = 4            # use face dots only when at least this many landed on the face
bootstrap_min_pairs_for_mapping = 4     # fewer decoded dots than this: no mapping (warn), never a 2-3 point guess
bootstrap_attempts = 2                  # the bootstrap is retried this many times before giving up
bootstrap_save_debug_images = False     # write on-frame and difference images per dot
debug_image_directory = "diagnostics"

# ------------------------------------------------------------- face mesh
export_directory = "export"           # OBJ + texture written here by --export-face / key e
face_mesh_texture_gain = 1.0            # brightness gain on the camera texture before projecting it back
face_mesh_retriangulate_every_frames = 15   # Delaunay is recomputed this often; the topology barely changes

# ------------------------------------------------------------ depth model
# Per-landmark parallax: projector = H . x + shift(face size) + depth_vector * z,
# with z the landmark's MediaPipe depth. depth_vector is fitted from the
# bootstrap's face dots. Key z toggles the term for an A/B comparison.
depth_model_enabled = True
depth_model_min_face_dots = 6
# Landmarks tested by --prove-face: forehead, brows, eye corners, nose bridge
# and tip, cheeks, mouth corners, chin - a spread of depths across the face.
face_proof_landmark_indices = [10, 9, 107, 336, 33, 263, 6, 4, 50, 280, 61, 291, 199]

# ------------------------------------------------------------ dense scan
# main.py --scan: a dense coded-dot scan measures camera->projector for the
# whole scene, depth included (each dot is an exact correspondence on whatever
# surface it hit). ~880 dots, 12 frames, about 4.5 s. Key g rescans.
scan_grid_columns = 24
scan_grid_rows = 14
scan_dot_radius_pixels = 11
scan_opening_radius_pixels = 2
scan_code_sample_radius_pixels = 6
scan_map_neighbours = 6                 # measured dots used for the local affine fit at a query point
scan_map_max_neighbour_distance_pixels = 60   # further than this from any dot: no data
scan_map_outlier_pixels = 12            # a dot disagreeing with its neighbours' local fit by more is dropped

# Proof sweep (--prove): rings are projected through the scan map at held-out
# camera points and measured back; the per-point error is the evidence.
proof_point_count = 12
proof_marker_radius_pixels = 14         # filled disc, like the scan dots: its centroid is unbiased
proof_min_dot_distance_pixels = 8       # a test point must not sit on a measured dot (held out)

# ----------------------------------------------------------- pointer test
# main.py --pointer: a blinking ring is projected at the mapping of the camera
# point under the mouse and measured back in the camera. Pure geometry test.
pointer_ring_radius_pixels = 100        # projector pixels
pointer_ring_thickness_pixels = 8
pointer_blink_seconds = 0.45
pointer_search_radius_pixels = 220      # camera px around the mouse where the ring is looked for
pointer_min_difference = 12
# Closed loop on the ring: each measured landing error corrects the aim
# (through the mapping's local Jacobian), so the ring converges onto the
# cursor on any surface without knowing its depth.
pointer_correction_gain = 0.8
pointer_correction_max_error_pixels = 200.0   # larger errors are treated as a wrong detection, not fed back
pointer_correction_reset_pixels = 40.0        # moving the mouse further than this starts a fresh correction

# --------------------------------------------------------- border markers
# Eight small white dots at the projector's border blink in a 6-slot code
# (all off, all on, 4 ID bits). Every cycle the camera re-finds and
# re-identifies them and refits the live wall-plane mapping. Key k toggles.
border_markers_enabled = False          # live H disabled by decision (2026-09-15); key k turns the blinking markers on
border_marker_inset_pixels = 30         # from the projector edge
border_marker_radius_pixels = 14
border_marker_brightness = 255
border_marker_slot_seconds = 0.45       # each blink slot; cycle = 6 slots = 2.7 s
border_marker_min_peak_difference = 7   # all-on minus all-off (slot-averaged) must exceed this at a marker
border_marker_max_area_pixels = 900
border_marker_sample_radius_pixels = 5
border_marker_code_min_margin = 0.2
border_marker_min_found = 4

# ---------------------------------------------------------------- kinect
# kinect_calibrate.py / kinect_red_dot.py use a Kinect v2 (colour + depth)
# instead of the webcam. The projector is then fitted as a camera looking at
# the Kinect's metric 3D points (projector_calibration.py), so depth is
# measured, not modelled.
kinect_depth_sample_radius_pixels = 4         # colour-pixel half-window whose finite 3D points are medianed for a landmark
kinect_calibration_file_path = "projector_calibration.json"
kinect_calibration_depth_frames = 10          # depth frames medianed for the 3D points of the scan dots (static scene)
kinect_calibration_outlier_pixels = 6.0       # projector px; a scan dot further from the fitted model is dropped
kinect_calibration_min_points = 40            # decoded dots with depth needed for a calibration
kinect_calibration_holdout_fraction = 0.2     # dots kept out of the fit so the reported error is honest
kinect_calibration_min_depth_span_metres = 0.3  # below this the scene is nearly a plane and the fit is ill-conditioned
kinect_calibration_scans = 4                  # coded scans pooled into one fit, each with its grid shifted (see below)
# The projector is focused on the far wall, so its dots are blurred on near objects, and a
# dense grid of blurred dots merges into one glow there. Each scan therefore shows a sparse
# grid, and the grid is shifted from scan to scan, so the scans together still cover the image
# densely. A blurred dot's centre is still the dot's centre.
kinect_calibration_grid_columns = 12
kinect_calibration_grid_rows = 7
kinect_calibration_grid_offsets = [(0.0, 0.0), (0.5, 0.0), (0.0, 0.5), (0.5, 0.5)]   # fractions of the dot spacing, one per scan in turn
kinect_calibration_extra_settle_seconds = 0.5   # waited after the measured latency before a pattern is photographed

# ------------------------------------------------ dense calibration (kinect_calibrate_dense.py)
# Gray-code stripes on a static scene: thousands of correspondences at every distance.
kinect_dense_finest_stripe_pixels = 16          # width of the finest stripe, in projector pixels
kinect_dense_coarsest_level = 3                 # cells up to 16 * 2**3 = 128 px are used where finer stripes wash out (defocus)
kinect_dense_average_frames = 6                 # camera frames averaged per pattern
kinect_dense_extra_settle_seconds = 0.3         # waited after the measured latency before a pattern is photographed
kinect_dense_minimum_beam_step = 10.0           # brightness (0..255) full white must add for a pixel to count as lit
kinect_dense_minimum_bit_contrast_fraction = 0.15   # pattern minus inverse, as a fraction of white minus black, for a reliable bit
kinect_dense_minimum_cell_fill = 0.4            # a cell counts when it holds this fraction of the pixels expected for its size
kinect_dense_maximum_depth_spread_fraction = 0.04   # a cell whose depths spread more than this fraction of its distance straddles an edge
kinect_calibration_pause_seconds = 8          # countdown on the projector before each scan (time to move to a new distance)
kinect_calibration_max_depth_spread_metres = 0.03  # a dot whose depth window spans more sits on a silhouette: dropped
# The calibration scene is static, so every scan pattern can be averaged over many camera frames.
# That lowers the camera noise enough to accept much fainter dots, which a lit room needs.
kinect_calibration_average_frames = 8         # camera frames averaged per scan pattern (the face bootstrap uses 2)
kinect_calibration_min_peak_difference = 5    # brightness step (0..255) a dot must add (the face bootstrap uses 12)
kinect_latency_test_toggles = 3               # black -> white flashes used to measure projector -> Kinect latency
kinect_red_dot_landmark_index = 4             # MediaPipe nose tip
kinect_red_dot_radius_pixels = 12             # projector px
kinect_red_dot_search_radius_pixels = 120     # camera px around the landmark where the projected disc is looked for
kinect_red_dot_min_area_pixels = 20
kinect_dot_colour = "green"                   # red | green | blue. Green: skin is never green, so the camera finds the dot reliably
kinect_dot_offsets_file_name = "kinect_dot_offsets.csv"   # per-frame log of where the dot lands relative to the landmark
kinect_dot_evidence_file_name = "kinect_dot_evidence.avi"   # raw camera crop around the landmark, one frame per logged row ("" = keep none)
kinect_dot_evidence_half_width_pixels = 260    # the crop reaches this far left and right of the landmark
kinect_dot_evidence_half_height_pixels = 190   # and this far above and below
kinect_red_dot_still_speed_metres_per_second = 0.05  # below this the head counts as still for the landing-error statistic
kinect_red_dot_snapshot_period_seconds = 0     # debug view saved to diagnostics/ this often (0 = never)
kinect_prediction_lead_extra_seconds = 0.03   # added to the measured projector -> Kinect latency: loop + render time

# ------------------------------------------------------------------ files
homography_file_path = "homography.json"

# A camera stream that stalls (the Kinect does, for ~5 s bursts) only delays the
# capture of a pattern that stays displayed; wait this long for a frame before giving up.
bootstrap_frame_timeout_seconds = 8.0
kinect_startup_timeout_seconds = 60      # give up on the first frame after this long (the stream stalls at times)
kinect_save_diagnostic_images = False    # calibration residual map, beam image, pointer snapshots: off unless wanted

# The projector's intrinsics are a property of the projector (zoom and lens do not
# change), measured once from well-conditioned scans (objects at 0.7-3 m):
# focal 2094-2137 px over five fits, principal point (950-960, 1258-1283).
# With them fixed, a calibration only has to solve the projector's pose, which
# even a single wall plane determines unambiguously. kinect_calibrate.py
# --fit-intrinsics refits them (needs objects at several depths in the beam).
kinect_projector_use_fixed_intrinsics = True
kinect_projector_focal_pixels = 2120.0           # starting value; refined by every scan that has near dots (below)
kinect_projector_principal_point = (960.0, 1280.0)
kinect_projector_distortion = (0.03, -0.05)      # radial k1, k2 (k3 and tangential terms zero)

# A wall plane fixes the pose but not the focal length: a longer focal with the projector
# placed farther back draws the same dots on the wall, yet draws a face at 0.8 m at the
# wrong size. Dots on a near object (you, sitting in the beam for one of the scans) break
# that tie, so the calibration fits the focal length whenever it has enough of them and
# reports the size ratio it achieves at those near dots.
kinect_projector_fit_focal = True
kinect_projector_focal_search_fraction = 0.4      # focal searched within +-40% of the starting value
kinect_projector_focal_near_depth_fraction = 0.6  # "near" dots: closer than this fraction of the median dot depth
kinect_projector_focal_min_near_dots = 15         # fewer near dots than this: focal stays at the starting value

# --------------------------------------------------------- kinect backend
# "sdk": Microsoft Kinect SDK 2.0 (kinect_camera.KinectCamera). On this laptop the
# Microsoft driver restarts the sensor every ~7 s on a faulty thermal reading.
# "freenect2": libfreenect2 through freenect2_bridge.exe (freenect2_camera.Freenect2Camera),
# which ignores that reading; needs the libusbK driver on the Kinect (Zadig).
kinect_backend = "freenect2"                     # "sdk" = Microsoft driver (restarts the sensor every 7 s on this laptop)
freenect2_bridge_path = "freenect2_bridge/build/Release/freenect2_bridge.exe"
freenect2_pipeline = "opencl"                 # opencl | cuda | cpu

# ------------------------------------------------------------ view checks
projector_view_colour_by_depth = True    # projector's-eye view painted by distance (blue near .. red far) instead of camera colour
camera_view_layer = "colour"             # left view of the pointer test at start: colour | depth | beam (switchable from the panel)

# -------------------------------------------------------------- face mesh
face_mesh_start_mode = "texture"                        # texture | paint | wire (kinect_face_mesh.py)
face_paint_texture_path = "export/uv_layout.png"  # painted texture in the canonical MediaPipe UV layout
face_depth_smoothing_factor = 0.25       # per-frame fraction of the measured face distance adopted (lower = smoother)
face_depth_jump_metres = 0.10            # a distance change larger than this is followed only when it persists ...
face_depth_jump_frames = 5               # ... for this many frames (a bad depth patch never moves the mesh)
face_mesh_hold_seconds = 0.5             # keep projecting the last mesh this long when MediaPipe loses the face
face_mesh_depth_smoothing_passes = 2     # Laplacian passes on each vertex's relative depth over the canonical mesh
face_mesh_depth_smoothing_weight = 0.5   # how far a vertex moves toward its neighbours' mean per pass
# MediaPipe's landmark depth is only roughly to scale; the Kinect depth at the landmarks
# gives the true nose-to-cheek relief, and the mesh's relief is scaled to match it.
face_relief_scale_smoothing_factor = 0.1 # per-frame fraction of the newly fitted relief scale adopted
face_relief_scale_limits = (0.3, 2.0)    # the fitted scale is clamped to this range
face_relief_outlier_metres = 0.03        # a landmark whose Kinect depth is further than this from the fitted relief is dropped
face_relief_min_landmarks = 60           # fewer landmarks with usable depth than this: the last scale is kept

# --------------------------------------------------------------- projector
projector_background_fraction = 0.01     # projector output never darker than this fraction of white (1% = 3/255)
prediction_deadband_pixels_per_second = 40.0   # below this median landmark speed no prediction is applied (rest = no wobble)
prediction_full_pixels_per_second = 120.0      # above this the full lead is applied; ramps in between
prediction_deadband_metres_per_second = 0.03   # the nose-dot tool's dead band, in metres per second (about 40 px/s at 0.8 m)
prediction_full_metres_per_second = 0.09       # full lead from this speed on
