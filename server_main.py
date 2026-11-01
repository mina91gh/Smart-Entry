
import importlib.util
import multiprocessing
import runpy
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
import os

import uvicorn


# Server network settings
HOST = "0.0.0.0"
PORT = 8002

# Local health check address
HEALTH_HOST = "127.0.0.1"
HEALTH_URL = f"http://{HEALTH_HOST}:{PORT}/docs"


def get_project_root():
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parent


PROJECT_ROOT = get_project_root()
APP_DATA_DIR = (
    Path(os.environ.get("PROGRAMDATA", r"C:\ProgramData"))
    / "SmartEntry"
)

CONFIG_FILE = APP_DATA_DIR / "smart-entry.env"


def load_server_config():
    if not CONFIG_FILE.is_file():
        raise FileNotFoundError(
            f"Smart Entry configuration file not found: {CONFIG_FILE}"
        )

    os.environ["SMART_ENTRY_CONFIG_FILE"] = str(CONFIG_FILE)

    with CONFIG_FILE.open("r", encoding="utf-8") as config:
        for line in config:
            line = line.strip()

            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()

            if key and not os.environ.get(key):
                os.environ[key] = value

    print(
        f"Configuration loaded from: {CONFIG_FILE}",
        flush=True,
    )
WEB_UI_DIR = PROJECT_ROOT / "Web-UI"
FACE_RECOGNITION_DIR = PROJECT_ROOT / "Face-Recognition"

web_process = None
face_process = None


def run_web_server():
    sys.path.insert(0, str(FACE_RECOGNITION_DIR))
    sys.path.insert(0, str(WEB_UI_DIR))

    app_file = WEB_UI_DIR / "app.py"

    if not app_file.is_file():
        raise FileNotFoundError(f"Web app not found: {app_file}")

    spec = importlib.util.spec_from_file_location(
        "smart_entry_web_app", app_file
    )

    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load Web app: {app_file}")

    app_module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = app_module
    spec.loader.exec_module(app_module)

    uvicorn.run(
        app_module.app,
        host=HOST,
        port=PORT,
        reload=False,
    )


def run_face_recognition():
    sys.path.insert(0, str(FACE_RECOGNITION_DIR))

    script = FACE_RECOGNITION_DIR / "Face-Emotion-Recog.py"

    if not script.is_file():
        raise FileNotFoundError(
            f"Recognition script not found: {script}"
        )

    runpy.run_path(str(script), run_name="__main__")


def wait_for_web_server(timeout=60):
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:

        if web_process is None or not web_process.is_alive():
            raise RuntimeError("Web server stopped unexpectedly.")

        try:
            with urllib.request.urlopen(
                HEALTH_URL, timeout=2
            ) as response:

                if response.status == 200:
                    print(
                        "Smart Entry Web Server is ready.",
                        flush=True,
                    )
                    return

        except (
            urllib.error.URLError,
            TimeoutError,
            OSError,
        ):
            time.sleep(0.5)

    raise TimeoutError(
        f"Web server did not start at {HEALTH_URL}"
    )


def stop_process(process, name):
    if process is None or not process.is_alive():
        return

    print(f"Stopping {name}...", flush=True)

    process.terminate()
    process.join(timeout=10)

    if process.is_alive():
        process.kill()
        process.join(timeout=5)


def main():
    global web_process, face_process
    load_server_config()

    if not WEB_UI_DIR.is_dir():
        raise FileNotFoundError(
            f"Web-UI directory not found: {WEB_UI_DIR}"
        )

    if not FACE_RECOGNITION_DIR.is_dir():
        raise FileNotFoundError(
            f"Face-Recognition directory not found: {FACE_RECOGNITION_DIR}"
        )

    try:
        print("Starting Web Server...", flush=True)

        web_process = multiprocessing.Process(
            target=run_web_server
        )
        web_process.start()

        wait_for_web_server()

        print("Starting Face Recognition...", flush=True)

        face_process = multiprocessing.Process(
            target=run_face_recognition
        )
        face_process.start()

        print("Smart Entry Server is running.", flush=True)
        print(f"Local Admin URL: http://127.0.0.1:{PORT}", flush=True)
        print(
            f"Network Admin URL: http://<SERVER-IP>:{PORT}",
            flush=True,
        )
        print("Press Ctrl+C to stop the server.", flush=True)

        while True:

            if not web_process.is_alive():
                raise RuntimeError("Web server process exited.")

            if not face_process.is_alive():
                raise RuntimeError(
                    "Face recognition process exited."
                )

            time.sleep(2)

    except KeyboardInterrupt:
        print("Server shutdown requested.", flush=True)

    finally:
        stop_process(face_process, "Face Recognition")
        stop_process(web_process, "Web Server")

        print("Smart Entry Server stopped.", flush=True)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
