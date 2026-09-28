"""Generate the narration clips with Microsoft Edge's neural voices (edge-tts, needs the network).

For every section this writes audio/<key>.mp3 and, in audio/narration.json, the clip's
length and its subtitle lines with start and end times taken from the spoken words.

    ..\\.venv\\Scripts\\python.exe build_audio.py            # only sections whose text changed
    ..\\.venv\\Scripts\\python.exe build_audio.py --force    # regenerate everything
"""

import argparse
import asyncio
import hashlib
import json
import os
import re

import av
import edge_tts

from narration import NARRATION
from subtitle_timing import subtitle_lines

AUDIO_DIRECTORY = "audio"
NARRATION_DATA_PATH = os.path.join(AUDIO_DIRECTORY, "narration.json")
VOICE = "en-US-AndrewNeural"
SPEAKING_RATE = "-8%"
SUBTITLE_LINE_MAX_CHARACTERS = 64
TICKS_PER_SECOND = 10_000_000.0          # edge-tts reports times in 100 ns ticks


def clip_duration_seconds(path):
    with av.open(path) as container:
        return container.duration / av.time_base


async def synthesize(text, path, attempts=6):
    """Save the speech and return the spoken words as (start seconds, end seconds)."""
    for attempt in range(1, attempts + 1):
        words = []
        try:
            communicate = edge_tts.Communicate(text, voice=VOICE, rate=SPEAKING_RATE, boundary="WordBoundary")
            with open(path, "wb") as file:
                async for message in communicate.stream():
                    if message["type"] == "audio":
                        file.write(message["data"])
                    elif message["type"] == "WordBoundary":
                        start_seconds = message["offset"] / TICKS_PER_SECOND
                        words.append((start_seconds, start_seconds + message["duration"] / TICKS_PER_SECOND))
            return words
        except (OSError, ConnectionError) as error:          # the speech service drops connections now and then
            if os.path.exists(path):
                os.remove(path)
            if attempt == attempts:
                raise
            print(f"  attempt {attempt} failed ({type(error).__name__}), retrying")
            await asyncio.sleep(2.0 * attempt)


def time_subtitles(lines, words, clip_seconds):
    """Give every line the time span of the words it holds. The speech service reports one
    boundary per written word in most cases; when the counts differ (numbers, symbols) the
    lines are spread over the clip in proportion to their length instead."""
    word_counts = [len(line.split()) for line in lines]
    timed = []
    if sum(word_counts) == len(words):
        first_word = 0
        for line, word_count in zip(lines, word_counts):
            timed.append({"text": line, "start": words[first_word][0], "end": words[first_word + word_count - 1][1]})
            first_word += word_count
    else:
        total_characters = float(sum(len(line) for line in lines))
        speech_start = words[0][0] if words else 0.0
        speech_end = words[-1][1] if words else clip_seconds
        elapsed_characters = 0
        for line in lines:
            start = speech_start + (speech_end - speech_start) * elapsed_characters / total_characters
            elapsed_characters += len(line)
            end = speech_start + (speech_end - speech_start) * elapsed_characters / total_characters
            timed.append({"text": line, "start": start, "end": end})
    for index in range(len(timed) - 1):                      # a line stays up until the next one starts
        timed[index]["end"] = timed[index + 1]["start"]
    if timed:
        timed[-1]["end"] = min(timed[-1]["end"] + 0.4, clip_seconds)
    return timed


def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--force", action="store_true")
    arguments = argument_parser.parse_args()
    os.makedirs(AUDIO_DIRECTORY, exist_ok=True)
    previous = {}
    if os.path.exists(NARRATION_DATA_PATH):
        with open(NARRATION_DATA_PATH) as file:
            previous = json.load(file)
    sections = {}
    for key, text in NARRATION:
        path = os.path.join(AUDIO_DIRECTORY, f"{key}.mp3")
        text_hash = hashlib.sha256(f"{VOICE}|{SPEAKING_RATE}|{text}".encode()).hexdigest()
        kept = previous.get(key)
        if not arguments.force and kept and kept.get("text_hash") == text_hash and os.path.exists(path):
            sections[key] = kept
        else:
            words = asyncio.run(synthesize(text, path))
            clip_seconds = clip_duration_seconds(path)
            lines = subtitle_lines(text)
            matched = sum(len(line.split()) for line in lines) == len(words)
            sections[key] = {"text_hash": text_hash, "seconds": clip_seconds,
                             "subtitles": time_subtitles(lines, words, clip_seconds), "timed_by_words": matched}
        print(f"{key:15s} {sections[key]['seconds']:5.1f} s, {len(sections[key]['subtitles']):2d} subtitle lines"
              f"{'' if sections[key]['timed_by_words'] else '  (timed by length, word count differed)'}")
        with open(NARRATION_DATA_PATH, "w") as file:
            json.dump(sections, file, indent=1)
    print(f"total {sum(section['seconds'] for section in sections.values()):.0f} s of narration")


if __name__ == "__main__":
    main()
