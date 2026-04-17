"""Скачивает тренировочные данные с data-сервера."""
import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

DATA_BASE_URL = os.environ.get("DATA_BASE_URL", "https://data.ai-business-spb.ru").rstrip("/")
API_KEY = os.environ.get("API_KEY")
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

FILES = [
    "mediascope/train.csv",
]


def download(remote_path: str) -> Path:
    if not API_KEY:
        print("ERROR: API_KEY not set (put it in .env)", file=sys.stderr)
        sys.exit(1)
    url = f"{DATA_BASE_URL}/data/{remote_path}"
    filename = remote_path.split("/")[-1]
    dest = DATA_DIR / filename
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"GET {url}")
    with requests.get(url, headers={"X-API-Key": API_KEY}, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 16):
                if chunk:
                    f.write(chunk)
    print(f"  -> {dest} ({dest.stat().st_size} bytes)")
    return dest


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for path in FILES:
        try:
            download(path)
        except requests.HTTPError as e:
            print(f"  ! failed: {e}", file=sys.stderr)
    print("done")


if __name__ == "__main__":
    main()
