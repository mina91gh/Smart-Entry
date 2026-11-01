
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit

import webview


def get_config_path():
    if os.environ.get("SMART_ENTRY_CLIENT_CONFIG_FILE"):
        return Path(
            os.environ["SMART_ENTRY_CLIENT_CONFIG_FILE"]
        ).expanduser()

    if getattr(sys, "frozen", False):
        program_data = Path(
            os.environ.get("PROGRAMDATA", r"C:\ProgramData")
        )
        return program_data / "SmartEntry" / "client.env"

    return Path(__file__).resolve().parent / "client.env"


def load_server_url():
    # Environment variable has priority.
    server_url = os.environ.get("SMART_ENTRY_SERVER_URL")

    if server_url:
        return server_url.strip().rstrip("/")

    config_file = get_config_path()

    if not config_file.is_file():
        raise FileNotFoundError(
            f"Client configuration file not found: {config_file}"
        )

    settings = {}

    with config_file.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            settings[key.strip()] = value.strip().strip('"').strip("'")

    server_url = settings.get("SMART_ENTRY_SERVER_URL")

    if not server_url:
        raise ValueError(
            "SMART_ENTRY_SERVER_URL is missing in client.env"
        )

    return server_url.rstrip("/")


def validate_server_url(server_url):
    parsed = urlsplit(server_url)

    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError(
            f"Invalid server URL: {server_url}"
        )


def is_server_running(server_url):
    health_url = f"{server_url}/docs"

    try:
        with urllib.request.urlopen(
            health_url,
            timeout=3,
        ) as response:
            return response.status == 200

    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def main():
    try:
        server_url = load_server_url()
        validate_server_url(server_url)

    except (FileNotFoundError, ValueError) as error:
        print(f"Configuration error: {error}")
        input("Press Enter to exit...")
        return

    app_url = f"{server_url}/"

    if not is_server_running(server_url):
        print(
            "Smart Entry Server is not running or is unreachable."
        )
        print(f"Server address: {server_url}")
        input("Press Enter to exit...")
        return

    print("Connected to Smart Entry Server.")

    webview.create_window(
        "Smart Entry Admin",
        app_url,
        width=1280,
        height=800,
        min_size=(960, 600),
    )

    webview.start()


if __name__ == "__main__":
    main()
