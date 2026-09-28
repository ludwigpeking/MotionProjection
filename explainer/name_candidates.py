"""Several readings of one English name by the narration voice, each placed in a real
sentence of the script, to choose from by ear.

    <speech environment python> -X utf8 name_candidates.py Kinect

Writes name_readings/<name>_<number>.wav (the sentence with that reading in it) and
name_readings/<name>_<number>_alone.wav (the reading alone), and <name>_candidates.json
describing how each was made. The readings come from the same voice in its English
mode: different spellings, repeated takes (the voice does not say a word the same way
twice), and the name cut out of a whole English phrase, where the voice has context.
"""
import asyncio
import json
import os
import struct
import sys

import build_audio_doubao
import language_zh

speak_doubao = build_audio_doubao.speak_doubao
OUTPUT_DIRECTORY = "name_readings"
SPEAKER_IDENTIFIER = "S_xwd8PIsf2"          # 李谦1
SAMPLE_RATE = speak_doubao.SAMPLE_RATE_HERTZ

# how each candidate is made: ("alone", spelling) or ("phrase", English phrase holding the name)
RECIPES = {
    "Kinect": [("alone", "Kinekt"), ("alone", "Kinekt"), ("alone", "Kinekt"),
               ("phrase", "the Kinect sensor"), ("phrase", "a Kinect camera"), ("phrase", "my Kinect is here"),
               ("alone", "Kinect"), ("alone", "Kinect.")],
}
SENTENCES = {"Kinect": "一台 Kinect 深度相机负责看，一台小型的投影仪负责画。"}


def parameters(language_code, with_timings):
    audio = {"format": "pcm", "sample_rate": SAMPLE_RATE, "speech_rate": 0}
    if with_timings:
        audio["enable_subtitle"] = True
    return {"speaker": SPEAKER_IDENTIFIER, "additions": json.dumps({"explicit_language": language_code}),
            "audio_params": audio}


def main():
    name = sys.argv[1]
    headers = {"X-Api-Key": speak_doubao.API_KEY_PATH.read_text(encoding="utf-8").strip(),
               "X-Api-Resource-Id": speak_doubao.RESOURCE_ID}
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    sentence = SENTENCES[name]
    sentence_audio = asyncio.run(speak_doubao.synthesize(sentence, parameters("zh-cn", True), headers))
    sentence_samples = speak_doubao.samples_of(sentence_audio)
    sentence_timings = list(speak_doubao.LAST_WORD_TIMINGS)

    described = []
    for number, (kind, text) in enumerate(RECIPES[name], start=1):
        if kind == "alone":
            reading = speak_doubao.trimmed(speak_doubao.samples_of(
                asyncio.run(speak_doubao.synthesize(text, parameters("en", False), headers))))
            how = f"the spelling '{text}' spoken alone"
        else:
            phrase_audio = asyncio.run(speak_doubao.synthesize(text, parameters("en", True), headers))
            timings = list(speak_doubao.LAST_WORD_TIMINGS)
            index = next((i for i, word in enumerate(timings) if name.lower() in word.get("word", "").lower()), None)
            if index is None:
                print(f"{number}: the name was not found in the timings of '{text}'; skipped")
                continue
            start = float(timings[index - 1]["endTime"]) if index > 0 else float(timings[index]["startTime"])
            end = float(timings[index + 1]["startTime"]) if index + 1 < len(timings) else float(timings[index]["endTime"])
            reading = speak_doubao.trimmed(speak_doubao.samples_of(phrase_audio)[int(start * SAMPLE_RATE):int(end * SAMPLE_RATE)])
            how = f"cut out of the phrase '{text}'"
        if not reading:
            print(f"{number}: empty reading; skipped")
            continue
        alone_path = os.path.join(OUTPUT_DIRECTORY, f"{name}_{number:02d}_alone.wav")
        speak_doubao.write_wave(alone_path, struct.pack(f"<{len(reading)}h", *reading))
        samples, _, replaced = build_audio_doubao.with_english_names(
            sentence_samples, sentence_timings, [name], lambda _name, reading=reading: reading, SAMPLE_RATE)
        sentence_path = os.path.join(OUTPUT_DIRECTORY, f"{name}_{number:02d}.wav")
        speak_doubao.write_wave(sentence_path, struct.pack(f"<{len(samples)}h", *samples))
        described.append({"number": number, "how": how, "kind": kind, "text": text, "sentence": sentence_path,
                          "alone": alone_path, "seconds": len(reading) / float(SAMPLE_RATE), "placed": bool(replaced)})
        print(f"{number}: {how}, {len(reading) / float(SAMPLE_RATE):.2f} s{'' if replaced else '  (NOT placed in the sentence)'}")
    with open(os.path.join(OUTPUT_DIRECTORY, f"{name}_candidates.json"), "w", encoding="utf-8") as file:
        json.dump(described, file, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
