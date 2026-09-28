"""A local page to listen to the takes of the name clauses and choose one of each.

    ..\\.venv\\Scripts\\python.exe -X utf8 takes_page.py          # then open http://127.0.0.1:8770

The choice is saved in name_readings/chosen_takes.json, which language_zh.py reads.
"More takes" asks the narration voice for six further takes of a clause (the clause's
text is sent to the Volcano Engine speech service).
"""

import http.server
import json
import os
import re
import subprocess
import sys
import threading

import language_zh

EXPLAINER_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
TAKES_DIRECTORY = os.path.join(EXPLAINER_DIRECTORY, "name_readings")
AUDIO_DIRECTORY = os.path.join(EXPLAINER_DIRECTORY, "audio_zh")
CHOICES_PATH = os.path.join(TAKES_DIRECTORY, "chosen_takes.json")
HEARD_PATH = os.path.join(TAKES_DIRECTORY, "heard.json")
SPEECH_PYTHON = r"C:\Users\qli\Desktop\IT\2508_Beethoven\documentary\.venv-tts\Scripts\python.exe"
PORT = 8770
busy_clauses = set()
busy_lock = threading.Lock()


def read_json(path, fallback):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as file:
            return json.load(file)
    return fallback


def takes_of(identifier):
    numbers = []
    for file_name in os.listdir(TAKES_DIRECTORY):
        match = re.fullmatch(re.escape(identifier) + r"_take_(\d+)\.wav", file_name)
        if match:
            numbers.append(int(match.group(1)))
    return sorted(numbers)


def state():
    choices = read_json(CHOICES_PATH, {})
    heard = read_json(HEARD_PATH, {})
    clauses = []
    for identifier, section, clause in language_zh.NAME_CLAUSES:
        clauses.append({"identifier": identifier, "section": section, "clause": clause,
                        "chosen": choices.get(identifier), "busy": identifier in busy_clauses,
                        "takes": [{"number": number, "file": f"{identifier}_take_{number:02d}.wav",
                                   "heard": heard.get(f"{identifier}_take_{number:02d}", "")}
                                  for number in takes_of(identifier)]})
    sections = sorted({section for _, section, _ in language_zh.NAME_CLAUSES},
                      key=[key for key, _ in language_zh.NARRATION].index)
    return {"clauses": clauses, "sections": [section for section in sections
                                             if os.path.exists(os.path.join(AUDIO_DIRECTORY, f"{section}.mp3"))]}


def render_more_takes(identifier):
    try:
        subprocess.run([SPEECH_PYTHON, "-X", "utf8", "sentence_takes.py", identifier, "--more"],
                       cwd=EXPLAINER_DIRECTORY, check=False)
    finally:
        with busy_lock:
            busy_clauses.discard(identifier)


PAGE = """<!doctype html>
<html lang="zh"><head><meta charset="utf-8"><title>Narration takes</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root { --ground: #101418; --panel: #182028; --line: #2c3a48; --text: #e8edf2; --soft: #9fb0c0; --accent: #ffd83d; --good: #50e8a0; }
body { margin: 0; background: var(--ground); color: var(--text); font-family: "Segoe UI", "Microsoft YaHei", sans-serif; }
main { max-width: 980px; margin: 0 auto; padding: 24px 16px 60px; }
h1 { font-size: 24px; margin: 0 0 6px; }
p.lead { color: var(--soft); margin: 0 0 22px; line-height: 1.5; }
section { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 16px 18px; margin-bottom: 18px; }
h2 { font-size: 19px; margin: 0 0 4px; font-weight: 600; }
.identifier { color: var(--soft); font-size: 13px; margin-bottom: 12px; }
.take { display: grid; grid-template-columns: 74px minmax(200px, 300px) 1fr 96px; gap: 12px; align-items: center;
        padding: 8px 10px; border-radius: 8px; border: 1px solid transparent; }
.take.chosen { border-color: var(--accent); background: #2a2a18; }
.take .number { font-weight: 600; }
.take .heard { color: var(--soft); font-size: 14px; }
audio { width: 100%; height: 34px; }
button { background: #2c3a48; color: var(--text); border: 1px solid #40556a; border-radius: 6px; padding: 7px 12px;
         font-size: 14px; cursor: pointer; }
button:hover { background: #3a4c5f; }
button.chosen { background: var(--accent); color: #1a1a1a; border-color: var(--accent); font-weight: 600; }
button:disabled { opacity: 0.5; cursor: default; }
.more { margin-top: 10px; display: flex; gap: 12px; align-items: center; color: var(--soft); font-size: 14px; }
.previews audio { max-width: 520px; }
.previews .row { display: grid; grid-template-columns: 120px 1fr; gap: 12px; align-items: center; margin: 6px 0; }
#summary { position: sticky; bottom: 0; background: #0c1014; border-top: 1px solid var(--line); padding: 12px 16px;
           font-size: 14px; color: var(--soft); }
#summary b { color: var(--good); }
@media (max-width: 700px) { .take { grid-template-columns: 60px 1fr; } .take .heard { grid-column: 1 / -1; } }
</style></head>
<body><main>
<h1>Narration takes: choose one per clause</h1>
<p class="lead">Every clause that holds an English name was spoken whole, several times, by the 李谦1 voice.
Listen, then press <b>Choose</b> on the take that reads the name right. The grey text is what a speech
recogniser heard: a first filter, not a verdict. Choices are saved at once.</p>
<div id="clauses"></div>
<section class="previews"><h2>Sections as last assembled</h2>
<div class="identifier">These use the choices in force when the narration was last built, not necessarily the ones shown above.</div>
<div id="previews"></div></section>
</main>
<div id="summary"></div>
<script>
var playing = null;

function request(method, path, body, done) {
  var call = new XMLHttpRequest();
  call.open(method, path);
  call.setRequestHeader("Content-Type", "application/json");
  call.onload = function () { done(JSON.parse(call.responseText)); };
  call.send(body ? JSON.stringify(body) : null);
}

function element(tag, className, text) {
  var node = document.createElement(tag);
  if (className) { node.className = className; }
  if (text !== undefined) { node.textContent = text; }
  return node;
}

function player(source) {
  var audio = element("audio");
  audio.controls = true;
  audio.preload = "none";
  audio.src = source;
  audio.addEventListener("play", function () {
    if (playing && playing !== audio) { playing.pause(); }
    playing = audio;
  });
  return audio;
}

function draw(state) {
  var holder = document.getElementById("clauses");
  var positions = {};
  var existing = holder.querySelectorAll("audio");
  for (var index = 0; index < existing.length; index += 1) {
    if (!existing[index].paused) { return; }          // do not redraw under a playing take
  }
  holder.innerHTML = "";
  var chosenCount = 0;
  state.clauses.forEach(function (clause) {
    var box = element("section");
    box.appendChild(element("h2", "", clause.clause));
    box.appendChild(element("div", "identifier", clause.identifier + "  ·  section " + clause.section));
    clause.takes.forEach(function (take) {
      var isChosen = clause.chosen === take.number;
      var row = element("div", "take" + (isChosen ? " chosen" : ""));
      row.appendChild(element("div", "number", "take " + take.number));
      row.appendChild(player("/takes/" + take.file));
      row.appendChild(element("div", "heard", take.heard));
      var button = element("button", isChosen ? "chosen" : "", isChosen ? "Chosen" : "Choose");
      button.onclick = function () {
        request("POST", "/api/choose", {identifier: clause.identifier, take: take.number}, draw);
      };
      row.appendChild(button);
      box.appendChild(row);
    });
    if (clause.chosen) { chosenCount += 1; }
    var more = element("div", "more");
    var moreButton = element("button", "", clause.busy ? "Rendering six more…" : "None is good: six more takes");
    moreButton.disabled = clause.busy;
    moreButton.onclick = function () {
      request("POST", "/api/more", {identifier: clause.identifier}, draw);
    };
    more.appendChild(moreButton);
    more.appendChild(element("span", "", clause.busy ? "takes about half a minute" : ""));
    box.appendChild(more);
    holder.appendChild(box);
  });
  var previews = document.getElementById("previews");
  if (!previews.hasChildNodes()) {
    state.sections.forEach(function (section) {
      var row = element("div", "row");
      row.appendChild(element("div", "", section));
      row.appendChild(player("/sections/" + section + ".mp3"));
      previews.appendChild(row);
    });
  }
  document.getElementById("summary").innerHTML = "Chosen: <b>" + chosenCount + " of " + state.clauses.length +
    "</b> clauses. When you are done, tell Claude and the film will be built from these choices.";
}

function refresh() { request("GET", "/api/state", null, draw); }
refresh();
setInterval(refresh, 4000);
</script></body></html>
"""


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format_text, *arguments):
        pass

    def send_json(self, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_audio(self, directory, file_name, content_type):
        if not re.fullmatch(r"[A-Za-z0-9_]+\.(wav|mp3)", file_name):
            self.send_error(404)
            return
        path = os.path.join(directory, file_name)
        if not os.path.exists(path):
            self.send_error(404)
            return
        with open(path, "rb") as file:
            data = file.read()
        start, end = 0, len(data) - 1
        requested = re.fullmatch(r"bytes=(\d*)-(\d*)", self.headers.get("Range", "") or "")
        if requested and (requested.group(1) or requested.group(2)):
            if requested.group(1):
                start = int(requested.group(1))
                if requested.group(2):
                    end = min(int(requested.group(2)), len(data) - 1)
            else:
                start = max(0, len(data) - int(requested.group(2)))
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(data)}")
        else:
            self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data[start:end + 1])

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/":
            body = PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif path == "/api/state":
            self.send_json(state())
        elif path.startswith("/takes/"):
            self.send_audio(TAKES_DIRECTORY, path[len("/takes/"):], "audio/wav")
        elif path.startswith("/sections/"):
            self.send_audio(AUDIO_DIRECTORY, path[len("/sections/"):], "audio/mpeg")
        else:
            self.send_error(404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0") or 0)
        try:
            request = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self.send_error(400)
            return
        identifier = request.get("identifier")
        if identifier not in [entry[0] for entry in language_zh.NAME_CLAUSES]:
            self.send_error(400)
            return
        if self.path == "/api/choose":
            take_number = int(request.get("take", 0))
            if take_number in takes_of(identifier):
                choices = read_json(CHOICES_PATH, {})
                choices[identifier] = take_number
                with open(CHOICES_PATH, "w", encoding="utf-8") as file:
                    json.dump(choices, file, indent=1)
            self.send_json(state())
        elif self.path == "/api/more":
            with busy_lock:
                already = identifier in busy_clauses
                busy_clauses.add(identifier)
            if not already:
                threading.Thread(target=render_more_takes, args=(identifier,), daemon=True).start()
            self.send_json(state())
        else:
            self.send_error(404)


def main():
    os.makedirs(TAKES_DIRECTORY, exist_ok=True)
    server = http.server.ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"[takes] http://127.0.0.1:{PORT}  (Ctrl+C stops it)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    sys.exit(main())
