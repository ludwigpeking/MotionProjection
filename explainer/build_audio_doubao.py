"""Narration in the author's own cloned voice (Volcano Engine / Doubao text-to-speech).

The speech client, the voice and the API key live in the Beethoven documentary project
(speak_doubao.py and the key file next to it); this script imports that client and never
reads or prints the key itself. Run it with that project's speech environment:

    C:\\Users\\qli\\Desktop\\IT\\2508_Beethoven\\documentary\\.venv-tts\\Scripts\\python.exe build_audio_doubao.py
    ... build_audio_doubao.py --force        # regenerate every section

For every section this writes audio/<key>.mp3 and, in audio/narration.json, the clip's
length and its subtitle lines with start and end times taken from the spoken words.
The narration text is sent to the Volcano Engine speech service.
"""

import argparse
import asyncio
import hashlib
import json
import os
import re
import subprocess
import sys
import wave

import language_zh
from narration import NARRATION as NARRATION_ENGLISH
from subtitle_timing import subtitle_lines, subtitle_lines_chinese

SPEECH_CLIENT_DIRECTORY = r"C:\Users\qli\Desktop\IT\2508_Beethoven\documentary"
SPEECH_RATE = 0                      # -50 .. 100; 0 is the voice's natural pace
# per language: narration, folder, voice name, voice identifier in the Volcano Engine voice library, language code
VERSIONS = {
    "en": (NARRATION_ENGLISH, "audio", "陈腐粉碎机", "S_ZUg7PIsf2", "en"),
    "zh": (language_zh.NARRATION, "audio_zh", "李谦1", "S_xwd8PIsf2", "zh-cn"),
}

sys.path.insert(0, SPEECH_CLIENT_DIRECTORY)
import speak_doubao                  # noqa: E402  (needs the path above)


def letters_of(text):
    """Letters, digits and Chinese characters: what both the text and the timed words hold."""
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]", "", text.lower())


def time_lines_by_spoken_words(lines, spoken_lines, word_timings, clip_seconds):
    """Start and end of every subtitle line from the service's word timings. Lines and timed
    words are matched by their position in the run of letters, so differences in how the
    two sides split words (hyphens, numbers) do not matter."""
    word_spans = []
    letter_count = 0
    for word in word_timings:
        letters = len(letters_of(word.get("word", "")))
        if letters == 0:
            continue
        word_spans.append((letter_count, letter_count + letters, float(word["startTime"]), float(word["endTime"])))
        letter_count += letters
    line_letters = [len(letters_of(spoken)) for spoken in spoken_lines]
    if not word_spans or sum(line_letters) == 0:
        return None
    scale = letter_count / float(sum(line_letters))
    if not 0.9 < scale < 1.1:
        return None                                   # the two texts differ too much to match by position

    def word_at(letter_position):
        for start_letter, end_letter, start_seconds, end_seconds in word_spans:
            if letter_position < end_letter:
                return start_seconds, end_seconds
        return word_spans[-1][2], word_spans[-1][3]

    timed = []
    position = 0
    for line, letters in zip(lines, line_letters):
        start_seconds = word_at(position * scale)[0]
        end_seconds = word_at(max(position + letters - 1, position) * scale)[1]
        timed.append({"text": line, "start": start_seconds, "end": end_seconds})
        position += letters
    for index in range(len(timed) - 1):               # a line stays up until the next one starts
        timed[index]["end"] = timed[index + 1]["start"]
    timed[-1]["end"] = min(timed[-1]["end"] + 0.4, clip_seconds)
    return timed


def time_lines_by_length(lines, clip_seconds):
    total = float(sum(len(line) for line in lines))
    timed = []
    elapsed = 0
    for line in lines:
        start = clip_seconds * elapsed / total
        elapsed += len(line)
        timed.append({"text": line, "start": start, "end": clip_seconds * elapsed / total})
    return timed


def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--force", action="store_true")
    argument_parser.add_argument("--language", choices=sorted(VERSIONS), default="en")
    arguments = argument_parser.parse_args()
    narration, audio_directory, voice_name, speaker_identifier, language_code = VERSIONS[arguments.language]
    narration_data_path = os.path.join(audio_directory, "narration.json")
    chinese = arguments.language == "zh"
    os.makedirs(audio_directory, exist_ok=True)
    previous = {}
    if os.path.exists(narration_data_path):
        with open(narration_data_path, encoding="utf-8") as file:
            previous = json.load(file)
    headers = {"X-Api-Key": speak_doubao.API_KEY_PATH.read_text(encoding="utf-8").strip(),
               "X-Api-Resource-Id": speak_doubao.RESOURCE_ID}
    request_parameters = {"speaker": speaker_identifier,
                          "additions": json.dumps({"explicit_language": language_code}),
                          "audio_params": {"format": "pcm", "sample_rate": speak_doubao.SAMPLE_RATE_HERTZ,
                                           "speech_rate": SPEECH_RATE, "enable_subtitle": True}}
    sections = {}
    for key, text in narration:
        clip_path = os.path.join(audio_directory, f"{key}.mp3")
        text_hash = hashlib.sha256(f"doubao|{speaker_identifier}|{SPEECH_RATE}|{text}".encode()).hexdigest()
        kept = previous.get(key)
        if not arguments.force and kept and kept.get("text_hash") == text_hash and os.path.exists(clip_path):
            sections[key] = kept
        else:
            spoken_text = text if chinese else speak_doubao.english_reading(text)
            audio = asyncio.run(speak_doubao.synthesize(spoken_text, request_parameters, headers))
            word_timings = list(speak_doubao.LAST_WORD_TIMINGS)
            wave_path = os.path.join(audio_directory, f"{key}.wav")
            speak_doubao.write_wave(wave_path, audio)
            with wave.open(wave_path) as wave_file:
                clip_seconds = wave_file.getnframes() / float(wave_file.getframerate())
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", wave_path, "-codec:a", "libmp3lame", "-q:a", "2", clip_path],
                           check=True)
            os.remove(wave_path)
            lines = subtitle_lines_chinese(text) if chinese else subtitle_lines(text)
            spoken_lines = lines if chinese else [speak_doubao.english_reading(line) for line in lines]
            timed = time_lines_by_spoken_words(lines, spoken_lines, word_timings, clip_seconds)
            sections[key] = {"text_hash": text_hash, "seconds": clip_seconds, "voice": voice_name,
                             "subtitles": timed if timed is not None else time_lines_by_length(lines, clip_seconds),
                             "timed_by_words": timed is not None}
        print(f"{key:15s} {sections[key]['seconds']:5.1f} s, {len(sections[key]['subtitles']):2d} subtitle lines"
              f"{'' if sections[key]['timed_by_words'] else '  (timed by length: the word timings did not match)'}", flush=True)
        with open(narration_data_path, "w", encoding="utf-8") as file:
            json.dump(sections, file, indent=1, ensure_ascii=False)
    print(f"total {sum(section['seconds'] for section in sections.values()):.0f} s of narration")


if __name__ == "__main__":
    main()
