
import multiprocessing
import runpy
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import uvicorn
import webview


HOST = "127.0.0.1"
PORT = 8002
APP_URL = f"http://{HOST}:{PORT}"


def get_project_root():
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS)

    return Path(__file__).resolve().parent


PROJECT_ROOT = get_project_root()
WEB_UI_DIR = PROJECT_ROOT / "Web-UI"
FACE_RECOGNITION_DIR = PROJECT_ROOT / "Face-Recognition"

web_process = None
face_process = None



def run_web_server():
    import importlib.util

    sys.path.insert(0, str(FACE_RECOGNITION_DIR))
    sys.path.insert(0, str(WEB_UI_DIR))

    app_file = WEB_UI_DIR / "app.py"

    if not app_file.is_file():
        raise FileNotFoundError(f"Web-UI app.py not found: {app_file}")

    spec = importlib.util.spec_from_file_location(
        "smart_entry_web_app",
        app_file,
    )

    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load Web-UI app from: {app_file}")

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
    runpy.run_path(str(script), run_name="__main__")


def wait_for_web_server(timeout=45):
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        if web_process is not None and not web_process.is_alive():
            raise RuntimeError("The Web-UI server stopped unexpectedly.")

        try:
            with urllib.request.urlopen(APP_URL, timeout=2) as response:
                if response.status == 200:
                    print("Web-UI is ready.", flush=True)
                    return
        except (urllib.error.URLError, TimeoutError, OSError):
            time.sleep(0.5)

    raise TimeoutError(f"Web-UI did not become ready at {APP_URL}")


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

    if not WEB_UI_DIR.is_dir():
        raise FileNotFoundError(f"Web-UI directory not found: {WEB_UI_DIR}")

    if not FACE_RECOGNITION_DIR.is_dir():
        raise FileNotFoundError(
            f"Face-Recognition directory not found: {FACE_RECOGNITION_DIR}"
        )

    try:
        print("Starting Web-UI...", flush=True)
        web_process = multiprocessing.Process(target=run_web_server)
        web_process.start()

        wait_for_web_server()

        print("Starting face recognition...", flush=True)
        face_process = multiprocessing.Process(target=run_face_recognition)
        face_process.start()

        print("Opening Smart Entry desktop window...", flush=True)
        webview.create_window(
            "Smart Entry",
            APP_URL,
            width=1280,
            height=800,
            min_size=(960, 600),
        )
        webview.start()

    finally:
        stop_process(face_process, "Face Recognition")
        stop_process(web_process, "Web-UI")
        print("Smart Entry has stopped.", flush=True)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
