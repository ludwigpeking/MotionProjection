# Motion Projection

Projection mapping on moving objects: an image is projected onto a moving face and
stays in place. By Richard Qian Li 陈腐粉碎机, ITP, NYU.

A Kinect v2 depth camera watches the scene. MediaPipe finds 468 landmarks on the face
in the colour image, and the depth sensor turns each one into a point in the room, in
metres. The projector is modelled as a camera running backwards; a Gray-code scan of a
static scene calibrates its focal length, lens centre, distortion and pose. Every point
is then mapped to the projector pixel that lights it, and a texture painted in Blender
is warped onto the live mesh, triangle by triangle.

The explainer film in `explainer/` walks through the geometry (English and Chinese).

## Hardware

- Kinect v2 (Xbox One sensor) with its USB 3 adapter
- any projector connected as a second display in extend mode, keystone at zero
- Windows, Python 3.12

## Install

```
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The Kinect is read through libfreenect2 and a small bridge program; building it is
described under "Kinect backends" below.

## Use

```
.venv\Scripts\python.exe web_panel.py
```

opens the panel at http://127.0.0.1:8765. From there:

| button | what it does | scene |
|---|---|---|
| Full calibration | Gray-code scan, fits the complete projector model | static, objects near and far, nobody in it |
| Landing test | projects discs across the scene and measures where they land | static |
| Quick calibration | coded dots, pose only unless near dots exist | a person may sit in the beam |
| Start nose dot | a dot that follows the nose, with the landing error shown | person |
| Face mesh | the painted texture on the tracked face | person |

The projector window is fullscreen and borderless. To close it: double-click or
right-click it, or Alt+F4 on it, or the Stop button of the panel.

After any change to the rig, run Full calibration and then Landing test. On the
author's rig the landing test gives a median error of 2 mm.

## Main files

| file | purpose |
|---|---|
| `kinect_calibrate_dense.py`, `gray_code_scan.py` | full calibration from a static scene |
| `projector_calibration.py` | the projector model and its fits |
| `kinect_face_mesh.py` | face mesh projection |
| `kinect_red_dot.py`, `analyse_dot_offsets.py` | nose dot and its error analysis |
| `kinect_pointer.py` | landing test and click-to-aim |
| `freenect2_camera.py`, `freenect2_bridge/` | Kinect access through libfreenect2 |
| `web_panel.py`, `web_panel.html` | the panel |
| `config.py` | every setting, with comments |
| `explainer/` | the explainer film: script, scenes, narration builders, covers |

## What is not in the repository

Recordings, clips cut from them, generated narration, rendered films and music are
left out. The explainer's scripts rebuild everything from your own recordings; see
`explainer/README.md`.

## Earlier work

This project grew out of a webcam version that used a homography with a depth
correction (`main.py`). Its documentation follows, together with the notes on the
Kinect drivers.

---

# Feedback projection mapping — panda face

Projects a panda pattern onto a tracked face. The camera and projector are
never calibrated by hand: the mapping between them is measured live by
projecting dots and watching where they land.

## Run

```
.venv\Scripts\python.exe main.py                 # normal
.venv\Scripts\python.exe main.py --landmarks     # project the face landmarks as dots instead (alignment check)
.venv\Scripts\python.exe main.py --seconds 30    # quit by itself (unattended tests)
.venv\Scripts\python.exe test_bootstrap.py       # only the bootstrap, prints the fit and exits
.venv\Scripts\python.exe diagnose.py             # does the camera see the projector at all?
```

The first run downloads the MediaPipe face landmarker model (~3 MB).

Setup: projector on a **second display in extend mode**, auto-keystone off,
focused at the distance where the face will be. Stand 1–2 m from it.

## What happens

1. **Bootstrap (~5 s, hold still).** 180 coded dots are shown on the whole
   projector, then 8 frames in which each dot blinks according to one bit of
   its ID. Every dot the camera can see gets a position and an identity. Dots
   inside the MediaPipe face box (at face depth) are preferred for the fit.
   In a bright room only the dots that hit the person are visible, which is
   exactly what is wanted.
2. **Servo (continuous).** Every 1.5 s (4 s once converged) the projection
   blinks off for ~0.4 s and seven small dots are flashed at face landmarks.
   The dots that land are matched to the aimed pattern as a whole, giving
   true camera↔projector correspondences on the face.
3. **Depth model.** The mapping is a fixed homography plus a projector-space
   shift that is linear in the face's size in pixels (∝ 1/depth), fitted
   from the recent correspondences. Moving toward or away from the projector
   is compensated every frame, not only when a measurement happens.
4. If the servo finds no dots for 8 cycles in a row, the bootstrap runs again.

The mapping is saved to `homography.json` on exit and loaded on the next start.

## Keys

| key | action |
|---|---|
| `q` / Esc | quit (saves the mapping) |
| `b` | re-run the bootstrap |
| `s` | save the mapping now |
| `m` | toggle the projection |
| `l` | landmark mode: project the face landmarks as dots instead of the panda |
| `v` | toggle the servo |
| `a` | permanent dots on the seven servo landmarks |
| `r` | forget face correspondences |

## Debug window

Green rings: the seven servo landmarks. Orange rings: where the last
measurement expected them. Red dots + lines: where the flashed dots actually
landed. Inset top-right: last difference image. The status lines show fps,
the fit's inlier count and residual, and the face size driving the depth model.
With `bootstrap_save_debug_images = True` (default) the `diagnostics/`
folder gets the bootstrap frames, per-cycle difference images and a snapshot
of the debug view every 2 s.

## Measured on this rig (config.py)

- Projector→camera latency: a change is invisible in frames read before
  +0.14 s and complete after +0.27 s. Everything is scheduled by wall-clock
  time from these two numbers; re-measure them if the camera or projector
  changes (see the scratch `latency_test.py` from the session).
- Camera: Media Foundation backend, manual exposure −6 (DirectShow ignores
  exposure control on this webcam).

## If things go wrong

- **Bootstrap finds 0 blobs / "camera never saw the dots"**: run
  `diagnose.py`. If "fraction lit by full white" is 0, the projector is not
  showing the second screen or not pointing where the camera looks.
- **Servo sees 0 dots every cycle**: the person is probably moving during
  the 0.4 s measurement; hold still for a few cycles. Dots aimed at the side
  of the face turned away from the projector are normally invisible.
- **Mask visibly late**: `prediction_lead_seconds` (0.22) extrapolates the
  landmarks to cover loop + projector latency; raise it slightly if the mask
  trails, lower it if it overshoots on quick moves.
- **Too much flashing**: raise `servo_period_converged_seconds`, or press
  `v` once the residual is low.

## Web panel

```
.venv\Scripts\python.exe web_panel.py
```

Opens a local page (http://127.0.0.1:8765, or the next free port; the URL is
printed) with buttons for the calibration, the nose dot and the pointer test,
a Stop button, lead and dot-size adjustments, the live Kinect view, the log,
the current calibration and the diagnostics images. The tools run as normal
subprocesses with `--panel`: no OpenCV debug window, the view and the
commands go through files in `diagnostics/`. In the pointer test, click the
live image to aim. Closing the panel (Ctrl+C in its terminal) stops the
running tool.

## Kinect backends: Microsoft SDK or libfreenect2

`config.kinect_backend` selects how the sensor is read.

- `"sdk"`: the Microsoft Kinect SDK 2.0 through pykinect2 (`kinect_camera.py`).
  On this laptop the Microsoft driver restarts the sensor every ~7 s
  (a thermal-protection loop on a faulty temperature reading), which
  libfreenect2 does not do.
- `"freenect2"`: libfreenect2 through `freenect2_bridge.exe`
  (`freenect2_camera.py`). The bridge publishes colour + depth-on-colour
  ("bigdepth") frames in shared memory; 3D points are then in the colour
  camera's frame using the intrinsics the sensor reports, so the projector
  must be recalibrated after switching.

Building the bridge (once):

```
C:\Users\qli\tools\vcpkg\vcpkg install libusb libjpeg-turbo opencl --triplet x64-windows
cd C:\Users\qli\tools\libfreenect2 && cmake -B build -G "Visual Studio 17 2022" -A x64 ^
    -DCMAKE_TOOLCHAIN_FILE=C:\Users\qli\tools\vcpkg\scripts\buildsystems\vcpkg.cmake ^
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DENABLE_OPENGL=OFF -DENABLE_CUDA=OFF -DBUILD_OPENNI2_DRIVER=OFF ^
    -DCMAKE_INSTALL_PREFIX=C:\Users\qli\tools\libfreenect2\install
cmake --build build --config Release --target install
cd <this folder>\freenect2_bridge && cmake -B build -G "Visual Studio 17 2022" -A x64 ^
    -DCMAKE_TOOLCHAIN_FILE=C:\Users\qli\tools\vcpkg\scripts\buildsystems\vcpkg.cmake ^
    -Dfreenect2_DIR=C:\Users\qli\tools\libfreenect2\install\lib\cmake\freenect2
cmake --build build --config Release
```

USB driver: **libusbK 3.1** installed with Zadig on the sensor's composite
parent ("Xbox NUI Sensor (Composite Parent)", USB ID 045E 02C4), then the
Kinect's USB cable unplugged and plugged back in once (without that
re-enumeration the depth stream stayed empty). libusbK keeps no state: a
crashed or killed bridge does not affect the next start.

UsbDk (`C:\Users\qli\tools\usbdk`) is installed but **not used**: our
libfreenect2 build only enables it when `LIBFREENECT2_USBDK=1` is set. It
streamed too, but its device redirection sticks after any abnormal exit of
the bridge, degrades the next open to a trickle of frames, and eventually
refuses to open the sensor at all until a reboot. It can be uninstalled from
Windows' Apps list. To restore the Microsoft SDK path: Device Manager → the
Kinect composite device → uninstall the libusbK driver, then reinstall the
Kinect runtime.

Never kill `freenect2_bridge.exe`; tools and the panel stop it through its
named event `Local\freenect2_bridge_stop`.

## Kinect v2 path: measured depth instead of a depth model

With a Kinect v2 (Xbox One sensor + Kinect SDK 2.0 runtime) the webcam,
the homography and the servo are not used. The projector is fitted once as
a camera looking at the Kinect's metric 3D points, and every tracked
landmark is projected through that model at its measured depth.

```
.venv\Scripts\python.exe -m pip install comtypes pykinect2
.venv\Scripts\python.exe patch_pykinect2.py          # once: pykinect2 0.1.0 needs three fixes for Python 3.12
.venv\Scripts\python.exe kinect_calibrate.py         # ~15 s: latency flashes + coded-dot scan -> projector_calibration.json
.venv\Scripts\python.exe kinect_red_dot.py           # red dot on the nose tip, live
.venv\Scripts\python.exe kinect_red_dot.py --landmark 9 --seconds 30
```

- `kinect_camera.py`: colour (1920x1080) + depth (512x424) captured together
  from one multi-source frame, same interface as `camera.Camera`, plus
  `points_3d_at()` (colour pixel -> metres). The Kinect's colour exposure is
  automatic and cannot be locked.
- `kinect_calibrate.py`: the scene must be static for ~10 s and the beam
  must hit surfaces at different depths (wall, chair, desk, a person). It
  measures the projector -> Kinect latency first, then runs the dense scan
  with the Kinect as the camera, and reports the fit and a held-out error.
  `diagnostics/kinect_calibration.png` shows every dot with its residual.
- `kinect_red_dot.py`: the debug window shows the landmark (green ring), the
  red disc as the Kinect sees it (red cross) and their distance in camera
  pixels; while the head is still that distance is the geometric error.
  Snapshots go to `diagnostics/kinect_red_dot_*.png` every 2 s.
- `kinect_pointer.py`: static geometry test. Click a point in the live Kinect
  view and a disc is projected at its 3D position; the window shows where it
  landed and the error in mm. `--grid` sweeps points inside the beam
  automatically, nearest first (white disc, frame differencing, so dark
  surfaces count). Proven 2026-09-26: 24 of 26 points within 11 mm at
  1.1-2.9 m (the misses were targets on depth edges).
- `fit_calibration_points.py`: refit the raw scan points saved in
  `diagnostics/kinect_calibration_points.json` without touching the sensor
  (`--save` overwrites the calibration file).
- **Calibrate with static objects at several depths** (a chair at 1 m plus
  the wall). A scan of the wall alone is one plane and cannot fix the focal
  length; the fit then slides between two models that both fit the wall and
  disagree by 10 cm at 0.8 m. A person is a poor calibration target (moves,
  silhouette depth edges).
- **Frames stall for seconds at a time**: the Kinect service needs CPU
  headroom. Close Kinect Studio and anything encoding video (ffmpeg,
  screen recorders); at 100 % total CPU the stream drops to 0 for ~5 s
  bursts.
