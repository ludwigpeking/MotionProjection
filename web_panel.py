"""Local web panel for the Kinect projection tools.

    .venv\\Scripts\\python.exe web_panel.py

Opens http://127.0.0.1:8765 in the browser. From there: run the calibration,
start the nose dot, run the pointer test (click the live image to aim), stop
the running tool, adjust the lead and the dot radius, watch the live Kinect
view and the log, and browse the diagnostics images. One tool runs at a time;
it is a normal subprocess of the scripts in this folder started with --panel.
"""

import collections
import http.server
import json
import mimetypes
import os
import subprocess
import sys
import threading
import time
import urllib.parse
import webbrowser

import config
from panel_link import (CLOUD_FILE_NAME, COMMAND_FILE_NAME, LIVE_VIEW_FILE_NAME, PROJECTOR_VIEW_FILE_NAME,
                        STATE_FILE_NAME)

PROJECT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
DIAGNOSTICS_DIRECTORY = os.path.join(PROJECT_DIRECTORY, config.debug_image_directory)
HOST = "127.0.0.1"
PORT = 8765

JOBS = {
    "kinect_view": {"title": "Kinect view", "command": ["kinect_view.py", "--panel"]},
    "calibrate": {"title": "Calibration", "command": ["kinect_calibrate.py", "--panel"]},
    "calibrate_dense": {"title": "Full calibration (static scene)", "command": ["kinect_calibrate_dense.py", "--panel"]},
    "pointer_grid": {"title": "Landing test (static scene)", "command": ["kinect_pointer.py", "--panel", "--grid"]},
    "red_dot": {"title": "Nose dot", "command": ["kinect_red_dot.py", "--panel", "--radius", "30"]},
    "face_mesh": {"title": "Face mesh", "command": ["kinect_face_mesh.py", "--panel"]},
    "pointer": {"title": "Pointer test", "command": ["kinect_pointer.py", "--panel"]},
}
LOG_NOISE_PREFIXES = ("WARNING:", "I0000", "W0000", "INFO: Created TensorFlow")


def stop_bridge(timeout_seconds=8.0):
    """Stop a running freenect2_bridge.exe cleanly through its named stop event; kill it only
    if it is still alive after timeout_seconds. Returns a log line."""
    import ctypes
    from panel_link import COMMAND_FILE_NAME as _unused  # noqa: F401  (panel_link is already imported)
    from freenect2_camera import STOP_EVENT_NAME
    kernel32 = ctypes.windll.kernel32
    event = kernel32.CreateEventW(None, True, False, STOP_EVENT_NAME)
    kernel32.SetEvent(event)
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        alive = subprocess.run(["tasklist", "/FI", "IMAGENAME eq freenect2_bridge.exe"], capture_output=True, text=True)
        if "freenect2_bridge.exe" not in alive.stdout:
            kernel32.ResetEvent(event)
            kernel32.CloseHandle(event)
            return "[panel] the Kinect bridge stopped cleanly"
        time.sleep(0.3)
    subprocess.run(["taskkill", "/F", "/IM", "freenect2_bridge.exe"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    kernel32.ResetEvent(event)
    kernel32.CloseHandle(event)
    return "[panel] the Kinect bridge ignored the stop request and was killed; the sensor may need a moment"


class JobRunner:
    def __init__(self):
        self.lock = threading.Lock()
        self.process = None
        self.job_name = None
        self.started_at = None
        self.exit_code = None
        self.log_lines = collections.deque(maxlen=500)

    def is_running(self):
        return self.process is not None and self.process.poll() is None

    def start(self, job_name):
        with self.lock:
            if self.is_running():
                return False, f"{JOBS[self.job_name]['title']} is still running; stop it first"
            if job_name not in JOBS:
                return False, f"unknown job {job_name}"
            for file_name in (LIVE_VIEW_FILE_NAME, PROJECTOR_VIEW_FILE_NAME, CLOUD_FILE_NAME, STATE_FILE_NAME,
                              COMMAND_FILE_NAME):
                path = os.path.join(DIAGNOSTICS_DIRECTORY, file_name)
                if os.path.exists(path):
                    os.remove(path)
            command = [sys.executable, "-u"] + JOBS[job_name]["command"]
            # A child inherits the panel's initial window state (minimized, hidden) unless
            # told otherwise; the projector window must come up normal and fullscreen.
            startup_information = subprocess.STARTUPINFO()
            startup_information.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startup_information.wShowWindow = 1     # SW_SHOWNORMAL (subprocess only names SW_HIDE)
            self.process = subprocess.Popen(command, cwd=PROJECT_DIRECTORY, stdout=subprocess.PIPE,
                                            stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                                            startupinfo=startup_information)
            self.job_name = job_name
            self.started_at = time.time()
            self.exit_code = None
            self.log_lines.clear()
            self.log_lines.append(f"[panel] started {JOBS[job_name]['title']}: {' '.join(command[1:])}")
            reader = threading.Thread(target=self._read_output, args=(self.process,), daemon=True)
            reader.start()
            return True, f"{JOBS[job_name]['title']} started"

    def _read_output(self, process):
        for line in process.stdout:
            line = line.rstrip()
            if not line or line.startswith(LOG_NOISE_PREFIXES) or "absl" in line:
                continue
            self.log_lines.append(line)
        code = process.wait()
        self.exit_code = code
        self.log_lines.append(f"[panel] {JOBS[self.job_name]['title']} ended with exit code {code}")

    def send_command(self, text):
        if not self.is_running():
            return False, "nothing is running"
        os.makedirs(DIAGNOSTICS_DIRECTORY, exist_ok=True)
        with open(os.path.join(DIAGNOSTICS_DIRECTORY, COMMAND_FILE_NAME), "w") as file:
            file.write(text)
        return True, f"sent {text}"

    def stop(self):
        with self.lock:
            if not self.is_running():
                return False, "nothing is running"
            self.send_command("quit")
            deadline = time.time() + 4.0
            while time.time() < deadline and self.is_running():
                time.sleep(0.1)
            if self.is_running():
                self.process.terminate()
                self.log_lines.append("[panel] the tool did not quit by itself; terminated")
                deadline = time.time() + 3.0
                while time.time() < deadline and self.is_running():
                    time.sleep(0.1)
                if self.is_running():
                    self.process.kill()
                # A terminated tool cannot stop its bridge. Ask the bridge to stop through its
                # named event (a killed bridge leaves the sensor's streams open and the next
                # bridge then receives no frames); kill only if it ignores the request.
                self.log_lines.append(stop_bridge())
            return True, "stopped"

    def status(self):
        running = self.is_running()
        return {
            "running": running,
            "job": self.job_name,
            "title": JOBS[self.job_name]["title"] if self.job_name else None,
            "seconds": round(time.time() - self.started_at) if self.started_at and running else None,
            "exit_code": self.exit_code,
            "log": list(self.log_lines)[-80:],
            "live_view": os.path.exists(os.path.join(DIAGNOSTICS_DIRECTORY, LIVE_VIEW_FILE_NAME)),
        }


runner = JobRunner()


def calibration_summary():
    path = os.path.join(PROJECT_DIRECTORY, config.kinect_calibration_file_path)
    if not os.path.exists(path):
        return {"exists": False}
    with open(path) as file:
        data = json.load(file)
    camera_matrix = data.get("camera_matrix", [[0, 0, 0], [0, 0, 0], [0, 0, 1]])
    return {
        "exists": True,
        "calibrated_at": data.get("calibrated_at"),
        "rms_pixels": data.get("rms_pixels"),
        "holdout_rms_pixels": data.get("holdout_rms_pixels"),
        "inlier_count": data.get("inlier_count"),
        "point_count": data.get("point_count"),
        "focal_pixels": camera_matrix[0][0],
        "principal_point": [camera_matrix[0][2], camera_matrix[1][2]],
        "prediction_lead_seconds": data.get("prediction_lead_seconds"),
        "latency_full_after_seconds": data.get("latency_full_after_seconds"),
    }


def geometry_summary():
    """The projector's pose for the 3D view: rotation (world -> projector) as a 3x3, the
    projector's optical centre in the Kinect frame, and its image geometry."""
    path = os.path.join(PROJECT_DIRECTORY, config.kinect_calibration_file_path)
    if not os.path.exists(path):
        return {"exists": False}
    import cv2
    import numpy
    with open(path) as file:
        data = json.load(file)
    rotation, _ = cv2.Rodrigues(numpy.array(data["rotation_vector"], dtype=numpy.float64))
    translation = numpy.array(data["translation_vector"], dtype=numpy.float64).reshape(3)
    centre = -rotation.T @ translation
    camera_matrix = data["camera_matrix"]
    return {
        "exists": True,
        "rotation": rotation.tolist(),
        "translation": translation.tolist(),
        "projector_centre": centre.tolist(),
        "focal_pixels": camera_matrix[0][0],
        "principal_point": [camera_matrix[0][2], camera_matrix[1][2]],
        "projector_width": data["projector_width"],
        "projector_height": data["projector_height"],
    }


class PanelHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *arguments):
        return      # quiet: the panel's own log is in the page

    def _send_json(self, payload, status=200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path, content_type=None):
        if not os.path.isfile(path):
            self.send_response(404)
            self.end_headers()
            return
        body = None
        for attempt in range(5):
            try:
                with open(path, "rb") as file:
                    body = file.read()
                break
            except PermissionError:
                # The tool replaces the live files atomically; a read that lands on the swap fails once.
                time.sleep(0.02)
        if body is None:
            self.send_response(503)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type or mimetypes.guess_type(path)[0] or "application/octet-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            self._send_file(os.path.join(PROJECT_DIRECTORY, "web_panel.html"), "text/html; charset=utf-8")
        elif parsed.path == "/api/status":
            self._send_json(runner.status())
        elif parsed.path == "/api/calibration":
            self._send_json(calibration_summary())
        elif parsed.path == "/api/live.jpg":
            self._send_file(os.path.join(DIAGNOSTICS_DIRECTORY, LIVE_VIEW_FILE_NAME), "image/jpeg")
        elif parsed.path == "/api/projector_view.jpg":
            self._send_file(os.path.join(DIAGNOSTICS_DIRECTORY, PROJECTOR_VIEW_FILE_NAME), "image/jpeg")
        elif parsed.path == "/api/cloud.bin":
            self._send_file(os.path.join(DIAGNOSTICS_DIRECTORY, CLOUD_FILE_NAME), "application/octet-stream")
        elif parsed.path == "/api/geometry":
            self._send_json(geometry_summary())
        elif parsed.path == "/api/state":
            self._send_file(os.path.join(DIAGNOSTICS_DIRECTORY, STATE_FILE_NAME), "application/json")
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        query = urllib.parse.parse_qs(parsed.query)
        try:
            if parsed.path == "/api/run":
                ok, message = runner.start(query.get("job", [""])[0])
            elif parsed.path == "/api/stop":
                ok, message = runner.stop()
            elif parsed.path == "/api/command":
                ok, message = runner.send_command(query.get("text", [""])[0])
            else:
                self.send_response(404)
                self.end_headers()
                return
        except Exception as error:      # never drop the connection: the page must see the error
            ok, message = False, f"panel error: {error!r}"
            runner.log_lines.append(f"[panel] {message}")
        self._send_json({"ok": ok, "message": message})


class PanelServer(http.server.ThreadingHTTPServer):
    # Without this, Windows lets a second program bind a port that is already in
    # use and silently hands the connections to the first one.
    allow_reuse_address = False


def main():
    server = None
    port = PORT
    for port in range(PORT, PORT + 20):
        try:
            server = PanelServer((HOST, port), PanelHandler)
            break
        except OSError:
            print(f"[panel] port {port} is taken by another program; trying the next one")
    if server is None:
        raise SystemExit("[panel] no free port found")
    url = f"http://{HOST}:{port}/"
    print(f"[panel] serving {url}  (Ctrl+C to stop the panel; a running tool is stopped with it)")
    if "--no-browser" not in sys.argv:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if runner.is_running():
            runner.stop()
        server.server_close()


if __name__ == "__main__":
    main()
