"""Whole-sentence takes of a sentence that holds an English name, spoken in one breath by
the narration voice: no splice, so no break around the name.

    <speech environment python> -X utf8 sentence_candidates.py Kinect

The takes differ in how the name is written for the voice and in whether the language is
forced to Chinese or left for the service to detect (mixed reading). Every take is
repeated, because the voice does not read a name the same way twice.
Writes name_readings/<name>_sentence_<number>.wav, the name cut out of each as
<name>_sentence_<number>_name.wav (for the listening check), and a description.
"""
import asyncio
import json
import os
import struct
import sys

import build_audio_doubao

speak_doubao = build_audio_doubao.speak_doubao
OUTPUT_DIRECTORY = "name_readings"
SPEAKER_IDENTIFIER = "S_xwd8PIsf2"          # 李谦1
SAMPLE_RATE = speak_doubao.SAMPLE_RATE_HERTZ
TAKES_PER_RECIPE = 3

SENTENCES = {"Kinect": "一台 {name} 深度相机负责看，一台小型的投影仪负责画。"}
# (how the name is written for the voice, language setting)
RECIPES = {"Kinect": [("Kinect", "chinese"), ("Kinect", "detected"), ("Kinekt", "chinese"), ("Kinekt", "detected")]}


def parameters(language_setting):
    additions = {"explicit_language": "zh-cn"} if language_setting == "chinese" else {}
    return {"speaker": SPEAKER_IDENTIFIER, "additions": json.dumps(additions),
            "audio_params": {"format": "pcm", "sample_rate": SAMPLE_RATE, "speech_rate": 0, "enable_subtitle": True}}


def main():
    name = sys.argv[1]
    headers = {"X-Api-Key": speak_doubao.API_KEY_PATH.read_text(encoding="utf-8").strip(),
               "X-Api-Resource-Id": speak_doubao.RESOURCE_ID}
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    described = []
    number = 0
    for written, language_setting in RECIPES[name]:
        for take in range(1, TAKES_PER_RECIPE + 1):
            number += 1
            text = SENTENCES[name].format(name=written)
            audio = asyncio.run(speak_doubao.synthesize(text, parameters(language_setting), headers))
            timings = list(speak_doubao.LAST_WORD_TIMINGS)
            samples = speak_doubao.samples_of(audio)
            sentence_path = os.path.join(OUTPUT_DIRECTORY, f"{name}_sentence_{number:02d}.wav")
            speak_doubao.write_wave(sentence_path, audio)
            index = next((i for i, word in enumerate(timings) if written.lower() in word.get("word", "").lower()), None)
            name_path = None
            confidence = None
            if index is not None:
                start = float(timings[index - 1]["endTime"]) if index > 0 else float(timings[index]["startTime"])
                end = float(timings[index + 1]["startTime"]) if index + 1 < len(timings) else float(timings[index]["endTime"])
                piece = samples[int(start * SAMPLE_RATE):int(end * SAMPLE_RATE)]
                name_path = os.path.join(OUTPUT_DIRECTORY, f"{name}_sentence_{number:02d}_name.wav")
                speak_doubao.write_wave(name_path, struct.pack(f"<{len(piece)}h", *piece))
                confidence = timings[index].get("confidence")
            described.append({"number": number, "written": written, "language": language_setting, "take": take,
                              "sentence": sentence_path, "name_alone": name_path, "confidence": confidence,
                              "seconds": len(samples) / float(SAMPLE_RATE)})
            print(f"{number:2d}: name written '{written}', language {language_setting}, take {take}, "
                  f"{len(samples) / float(SAMPLE_RATE):.1f} s, service confidence for the name "
                  f"{confidence if confidence is None else round(confidence, 2)}")
    with open(os.path.join(OUTPUT_DIRECTORY, f"{name}_sentence_candidates.json"), "w", encoding="utf-8") as file:
        json.dump(described, file, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
