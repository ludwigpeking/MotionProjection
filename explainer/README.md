# System explainer video

By Richard Qian Li 陈腐粉碎机, ITP, NYU.

A narrated, subtitled Manim (3Blue1Brown's animation library) video explaining the
face-projection pipeline: the equipment, the three pixel coordinate systems, how a
colour pixel and a distance become a point in the room, how a depth pixel finds its
colour pixel, the projector modelled as a camera, the coded-dot calibration and what
it solves, why the focal length matters, the face mesh, how the painted texture is
mapped onto it, where the error comes from, and the feedback loop around the moving face.

Output: the finished films are in `final/`, with their subtitles and covers.

## Folders

| folder | what is in it |
|---|---|
| `final/` | the finished films, English and Chinese, with their subtitles and covers. Nothing else. |
| `working/masters/` | the films without intro and music, which every new mix starts from |
| `working/leftovers/` | intermediate files of past edits; safe to delete |
| `covers/`, `images/`, `clips/` | pictures and clips the films are built from |
| `audio/`, `audio_zh/`, `name_readings/` | generated narration, and the takes to choose from |
| `media/` | Manim's render output |

## Notation

| image or space | coordinates |
|---|---|
| depth camera pixel | (x, y) with distance d |
| colour camera pixel | (x', y') |
| projector pixel | (x'', y'') |
| point in the room | (X, Y, Z) in metres, from the colour camera: X right, Y up, Z ahead |

## Files

- `narration.py`: the script, one entry per section, in playing order.
- `build_audio_doubao.py`: the narration in the author's cloned voice 陈腐粉碎机 (Volcano
  Engine / Doubao text-to-speech). It imports the speech client of the Beethoven
  documentary project, which holds the voice and the API key file; the key is never
  copied here. It writes `audio/<section>.mp3` and the subtitle lines with times taken
  from the spoken words (`audio/narration.json`). The narration text is sent to the
  Volcano Engine service. Only sections whose text changed are regenerated; `--force`
  redoes all.
- `build_audio.py`: the same with Microsoft Edge's neural voice (edge-tts), as a fallback.
- `subtitle_timing.py`: cuts the narration into subtitle lines.
- `prepare_clips.py`: cuts the clips in `clips/` from `../recordings`: the projected
  result in a lit and in a dark room, the Blender texture painting, the 3D scene from
  the panel capture, and a take with the MediaPipe landmarks drawn on it.
- `explainer_video.py`: the scenes. Each animation is tied to a phrase of the narration
  with `self.at("phrase")`, so picture and voice stay together when the script changes.
- `images/kinect.jpg`: Kinect for Xbox One photo by Evan-Amos, public domain
  (Wikimedia Commons, File:Xbox-One-Kinect.jpg).
- `images/projector.jpg`: optional. Put a photo of the projector here and it replaces
  the drawn projector in the equipment section.

## Rebuild

From this folder:

```
..\.venv\Scripts\python.exe prepare_clips.py
C:\Users\qli\Desktop\IT\2508_Beethoven\documentary\.venv-tts\Scripts\python.exe -X utf8 build_audio_doubao.py
..\.venv\Scripts\python.exe -m manim render -qm --disable_caching explainer_video.py SystemExplainer
copy media\videos\explainer_video\720p30\SystemExplainer.mp4 system_explainer.mp4
```

`-ql` renders a fast 480p preview; `-qh` renders 1080p60. The 720p render takes about
an hour.

To render only some sections while editing:

```
$env:EXPLAINER_SECTIONS = "error,loop"
```

After editing the narration, rerun `build_audio_doubao.py`. If a phrase used as a cue
was reworded, the render stops and names the cue that no longer exists.
