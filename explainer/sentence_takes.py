"""Several whole takes of every clause of the Chinese script that holds an English name,
spoken in one breath by the narration voice, to choose from by ear.

    <speech environment python> -X utf8 sentence_takes.py                  # every clause
    <speech environment python> -X utf8 sentence_takes.py title_kinect     # one clause again
    ..\\.venv\\Scripts\\python.exe -X utf8 sentence_takes.py --listen        # what a recogniser hears

Writes name_readings/<identifier>_take_<number>.wav with the word timings beside it.
The chosen take of each clause goes into language_zh.CHOSEN_TAKES; build_audio_doubao.py
then assembles the narration from the chosen takes.

Half of the takes force the language to Chinese, half leave it to the service (mixed
reading). The voice does not read a name the same way twice, so every take is a new try.
"""
import asyncio
import json
import os
import sys

import language_zh

TAKES_DIRECTORY = "name_readings"
SPEAKER_IDENTIFIER = "S_xwd8PIsf2"          # 李谦1
TAKES_PER_CLAUSE = 6


def take_stem(identifier, take_number):
    return os.path.join(TAKES_DIRECTORY, f"{identifier}_take_{take_number:02d}")


def existing_takes(identifier):
    numbers = []
    for file_name in os.listdir(TAKES_DIRECTORY) if os.path.isdir(TAKES_DIRECTORY) else []:
        if file_name.startswith(identifier + "_take_") and file_name.endswith(".wav"):
            numbers.append(int(file_name[len(identifier) + len("_take_"):-len(".wav")]))
    return sorted(numbers)


def speak(identifiers, add_to_existing=False):
    import build_audio_doubao
    speak_doubao = build_audio_doubao.speak_doubao
    headers = {"X-Api-Key": speak_doubao.API_KEY_PATH.read_text(encoding="utf-8").strip(),
               "X-Api-Resource-Id": speak_doubao.RESOURCE_ID}
    os.makedirs(TAKES_DIRECTORY, exist_ok=True)
    for identifier, _, clause in language_zh.NAME_CLAUSES:
        if identifiers and identifier not in identifiers:
            continue
        print(f"{identifier}: {clause}")
        first_number = (max(existing_takes(identifier), default=0) + 1) if add_to_existing else 1
        for take_number in range(first_number, first_number + TAKES_PER_CLAUSE):
            language_setting = "chinese" if take_number % 2 else "detected"
            additions = {"explicit_language": "zh-cn"} if language_setting == "chinese" else {}
            parameters = {"speaker": SPEAKER_IDENTIFIER, "additions": json.dumps(additions),
                          "audio_params": {"format": "pcm", "sample_rate": speak_doubao.SAMPLE_RATE_HERTZ,
                                           "speech_rate": 0, "enable_subtitle": True}}
            audio = asyncio.run(speak_doubao.synthesize(clause, parameters, headers))
            speak_doubao.write_wave(take_stem(identifier, take_number) + ".wav", audio)
            with open(take_stem(identifier, take_number) + ".json", "w", encoding="utf-8") as file:
                json.dump(list(speak_doubao.LAST_WORD_TIMINGS), file, ensure_ascii=False)
            print(f"   take {take_number} ({language_setting}): {len(audio) / 2 / speak_doubao.SAMPLE_RATE_HERTZ:.1f} s", flush=True)


def listen():
    """What a recogniser hears in every take not yet listened to; kept in heard.json."""
    from faster_whisper import WhisperModel
    heard_path = os.path.join(TAKES_DIRECTORY, "heard.json")
    heard = {}
    if os.path.exists(heard_path):
        with open(heard_path, encoding="utf-8") as file:
            heard = json.load(file)
    model = None
    for identifier, _, clause in language_zh.NAME_CLAUSES:
        print(f"{identifier}: {clause}")
        for take_number in existing_takes(identifier):
            key = f"{identifier}_take_{take_number:02d}"
            if key not in heard:
                if model is None:
                    model = WhisperModel("medium", device="cpu", compute_type="int8")
                segments, _ = model.transcribe(take_stem(identifier, take_number) + ".wav", language="zh", beam_size=5,
                                               initial_prompt="以下是普通话的技术讲解。")
                heard[key] = "".join(segment.text for segment in segments)
                with open(heard_path, "w", encoding="utf-8") as file:
                    json.dump(heard, file, indent=1, ensure_ascii=False)
            print(f"   take {take_number}: {heard[key]}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--listen":
        listen()
    else:
        speak([argument for argument in sys.argv[1:] if not argument.startswith("--")],
              add_to_existing="--more" in sys.argv)
