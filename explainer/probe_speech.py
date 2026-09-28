"""What does the speech service report about how it read a sentence?

    <speech environment python> -X utf8 probe_speech.py "要朗读的句子"

Prints every event the service sends besides the audio, so that the reading of
polyphonic characters and of English words can be checked without listening.
"""
import asyncio
import json
import sys
import uuid

SPEECH_CLIENT_DIRECTORY = r"C:\Users\qli\Desktop\IT\2508_Beethoven\documentary"
sys.path.insert(0, SPEECH_CLIENT_DIRECTORY)
import speak_doubao          # noqa: E402
import websockets            # noqa: E402

SPEAKER_IDENTIFIER = "S_xwd8PIsf2"          # 李谦1


async def probe(text):
    headers = {"X-Api-Key": speak_doubao.API_KEY_PATH.read_text(encoding="utf-8").strip(),
               "X-Api-Resource-Id": speak_doubao.RESOURCE_ID, "X-Api-Connect-Id": str(uuid.uuid4())}
    parameters = {"speaker": SPEAKER_IDENTIFIER, "additions": json.dumps({"explicit_language": "zh-cn"}),
                  "audio_params": {"format": "pcm", "sample_rate": speak_doubao.SAMPLE_RATE_HERTZ, "speech_rate": 0,
                                   "enable_subtitle": True, "enable_timestamp": True}}
    async with websockets.connect(speak_doubao.SERVICE_URL, additional_headers=headers, max_size=16 * 1024 * 1024,
                                  ping_interval=None, open_timeout=30) as connection:
        await connection.send(speak_doubao.client_frame(speak_doubao.START_CONNECTION, {}))
        while (await speak_doubao.next_frame(connection))["event"] != speak_doubao.CONNECTION_STARTED:
            pass
        session_identifier = str(uuid.uuid4())
        await connection.send(speak_doubao.client_frame(
            speak_doubao.START_SESSION, {"event": speak_doubao.START_SESSION, "req_params": parameters}, session_identifier))
        while (await speak_doubao.next_frame(connection))["event"] != speak_doubao.SESSION_STARTED:
            pass
        await connection.send(speak_doubao.client_frame(
            speak_doubao.TASK_REQUEST, {"event": speak_doubao.TASK_REQUEST, "req_params": {**parameters, "text": text}},
            session_identifier))
        await connection.send(speak_doubao.client_frame(speak_doubao.FINISH_SESSION, {}, session_identifier))
        audio_bytes = 0
        while True:
            frame = await asyncio.wait_for(speak_doubao.next_frame(connection), timeout=60)
            if frame["type"] == speak_doubao.AUDIO_ONLY_SERVER:
                audio_bytes += len(frame["payload"])
                continue
            payload = frame["payload"].decode("utf-8", errors="replace")
            print(f"event {frame['event']}: {payload[:3000]}")
            if frame["event"] == speak_doubao.SESSION_FINISHED:
                break
        await connection.send(speak_doubao.client_frame(speak_doubao.FINISH_CONNECTION, {}))
    print(f"audio: {audio_bytes / 2 / speak_doubao.SAMPLE_RATE_HERTZ:.1f} s")


asyncio.run(probe(sys.argv[1]))
