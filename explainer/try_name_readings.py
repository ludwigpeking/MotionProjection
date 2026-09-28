"""Tries several spellings of a name with the narration voice, in its English mode, and
saves each reading to listen to: name_readings/<number>_<spelling>.wav

    <speech environment python> -X utf8 try_name_readings.py Kinect "Kinnect" "Kin-ect" ...

The first argument is the name as it is written in the script; the others are the
spellings to try. Run check afterwards with the project's own environment:
    ..\\.venv\\Scripts\\python.exe -X utf8 try_name_readings.py --listen
"""
import asyncio
import json
import os
import re
import sys

OUTPUT_DIRECTORY = "name_readings"
SPEAKER_IDENTIFIER = "S_xwd8PIsf2"          # 李谦1


def file_name(number, spelling):
    return os.path.join(OUTPUT_DIRECTORY, f"{number:02d}_{re.sub(r'[^A-Za-z0-9]+', '_', spelling).strip('_')}.wav")


def speak(spellings):
    sys.path.insert(0, r"C:\Users\qli\Desktop\IT\2508_Beethoven\documentary")
    import speak_doubao
    headers = {"X-Api-Key": speak_doubao.API_KEY_PATH.read_text(encoding="utf-8").strip(),
               "X-Api-Resource-Id": speak_doubao.RESOURCE_ID}
    parameters = {"speaker": SPEAKER_IDENTIFIER, "additions": json.dumps({"explicit_language": "en"}),
                  "audio_params": {"format": "pcm", "sample_rate": speak_doubao.SAMPLE_RATE_HERTZ, "speech_rate": 0}}
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    for number, spelling in enumerate(spellings, start=1):
        audio = asyncio.run(speak_doubao.synthesize(spelling, parameters, headers))
        speak_doubao.write_wave(file_name(number, spelling), audio)
        print(f"{number:2d}  {spelling!r}: {len(audio) / 2 / speak_doubao.SAMPLE_RATE_HERTZ:.2f} s")
    with open(os.path.join(OUTPUT_DIRECTORY, "spellings.json"), "w", encoding="utf-8") as file:
        json.dump(spellings, file, ensure_ascii=False)


def listen():
    from faster_whisper import WhisperModel
    with open(os.path.join(OUTPUT_DIRECTORY, "spellings.json"), encoding="utf-8") as file:
        spellings = json.load(file)
    model = WhisperModel("medium", device="cpu", compute_type="int8")
    for number, spelling in enumerate(spellings, start=1):
        segments, _ = model.transcribe(file_name(number, spelling), language="en", beam_size=5)
        heard = " ".join(segment.text.strip() for segment in segments)
        print(f"{number:2d}  written {spelling!r:24s} heard {heard!r}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--listen":
        listen()
    else:
        speak(sys.argv[1:])
