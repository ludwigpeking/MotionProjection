"""Listens to the generated narration and reports where it differs from the script.

    ..\\.venv\\Scripts\\python.exe -X utf8 check_narration.py --language zh

A speech recogniser writes down what it hears in every clip. The script and the
transcript are both turned into pinyin with tones and compared: a polyphonic character
read the wrong way, or an English name spoken badly, comes out as different syllables.
The recogniser makes mistakes of its own, so every reported difference is a place to
listen to, not a proven error.
"""

import argparse
import difflib
import json
import os
import re

from faster_whisper import WhisperModel
from pypinyin import Style, lazy_pinyin

import language_zh
from narration import NARRATION as NARRATION_ENGLISH

VERSIONS = {"en": (NARRATION_ENGLISH, "audio", "en"), "zh": (language_zh.NARRATION, "audio_zh", "zh")}


def syllables(text, language):
    """The text as a list of comparable units: pinyin syllables with tone for Chinese
    characters, lower-case words for anything written in Latin letters, digits one by one."""
    units = []
    for piece in re.findall(r"[一-鿿]+|[A-Za-z]+|\d", text):
        if re.match(r"[一-鿿]", piece):
            units += lazy_pinyin(piece, style=Style.TONE3, neutral_tone_with_five=True) if language == "zh" else [piece]
        else:
            units.append(piece.lower())
    return units


def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--language", choices=sorted(VERSIONS), default="zh")
    argument_parser.add_argument("--model", type=str, default="medium")
    argument_parser.add_argument("--sections", type=str, default="")
    arguments = argument_parser.parse_args()
    narration, audio_directory, language_code = VERSIONS[arguments.language]
    wanted = [name for name in arguments.sections.split(",") if name]
    model = WhisperModel(arguments.model, device="cpu", compute_type="int8")
    report = {}
    for key, text in narration:
        if wanted and key not in wanted:
            continue
        segments, _ = model.transcribe(os.path.join(audio_directory, f"{key}.mp3"), language=language_code,
                                       beam_size=5, initial_prompt="以下是普通话的技术讲解。" if language_code == "zh" else None)
        heard = "".join(segment.text for segment in segments)
        written_units = syllables(text, arguments.language)
        heard_units = syllables(heard, arguments.language)
        matcher = difflib.SequenceMatcher(None, written_units, heard_units, autojunk=False)
        differences = []
        for operation, first, last, heard_first, heard_last in matcher.get_opcodes():
            if operation == "equal":
                continue
            context = " ".join(written_units[max(0, first - 3):first])
            differences.append(f"after [{context}]: written [{' '.join(written_units[first:last])}] "
                               f"heard [{' '.join(heard_units[heard_first:heard_last])}]")
        report[key] = {"heard": heard, "differences": differences}
        print(f"== {key}: {len(differences)} differences in {len(written_units)} syllables")
        print(f"   heard: {heard}")
        for difference in differences:
            print(f"   {difference}")
    with open(os.path.join(audio_directory, "narration_check.json"), "w", encoding="utf-8") as file:
        json.dump(report, file, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
