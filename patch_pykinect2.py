"""Make the PyPI pykinect2 (0.1.0) import on 64-bit Python 3.12 with comtypes >= 1.2.

Run once after installing comtypes and pykinect2:

    .venv\\Scripts\\python.exe patch_pykinect2.py

Idempotent: already-patched files are left alone.
"""

import importlib.util
import os

PATCHES = {
    "PyKinectV2.py": [
        ("assert sizeof(tagSTATSTG) == 72, sizeof(tagSTATSTG)",
         "assert sizeof(tagSTATSTG) in (72, 80), sizeof(tagSTATSTG)   # 80 on 64-bit Python"),
        ("from comtypes import _check_version; _check_version('')",
         "# comtypes >= 1.2: _check_version('') always raises; the typelib is hand-written above, nothing to check"),
    ],
    "PyKinectRuntime.py": [
        ("time.clock()", "time.perf_counter()"),      # time.clock was removed in Python 3.8
    ],
}


def main():
    specification = importlib.util.find_spec("pykinect2")
    if specification is None or not specification.submodule_search_locations:
        raise SystemExit("pykinect2 is not installed in this interpreter")
    package_directory = list(specification.submodule_search_locations)[0]
    for file_name, replacements in PATCHES.items():
        path = os.path.join(package_directory, file_name)
        with open(path, encoding="utf-8") as file:
            source = file.read()
        applied = 0
        for old, new in replacements:
            if old in source:
                source = source.replace(old, new)
                applied += 1
        if applied:
            with open(path, "w", encoding="utf-8") as file:
                file.write(source)
        print(f"{file_name}: {applied} patch(es) applied, {len(replacements) - applied} already present")


if __name__ == "__main__":
    main()
