import os
import sys
from pathlib import Path

import uvicorn


def main():
    if getattr(sys, "frozen", False):
        project_root = Path(sys.executable).resolve().parent
    else:
        project_root = Path(__file__).resolve().parent

    web_ui_dir = project_root / "Web-UI"
    face_recognition_dir = project_root / "Face-Recognition"

    sys.path.insert(0, str(face_recognition_dir))
    sys.path.insert(0, str(web_ui_dir))

    os.chdir(web_ui_dir)

    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8002,
        reload=False,
    )


if __name__ == "__main__":
    main()

