import os
import sys
import time
import zipfile
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ.get("API_KEY")
APP_BASE_URL = os.environ.get("APP_BASE_URL", "https://app.ai-business-spb.ru").rstrip("/")
CASE = "mediascope"
ROOT = Path(__file__).resolve().parent.parent
BUNDLE = ROOT / "bundle.zip"

INCLUDE = [
    "solution.py",
    "rules.py",
    "yandex_llm.py",
    "router.py",
    "titles_dict.json",
]
INCLUDE_DIRS = ["models"]


def build_bundle() -> Path:
    if BUNDLE.exists():
        BUNDLE.unlink()
    with zipfile.ZipFile(BUNDLE, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in INCLUDE:
            path = ROOT / name
            if path.exists():
                zf.write(path, name)
        for d in INCLUDE_DIRS:
            for path in (ROOT / d).rglob("*"):
                if path.is_file():
                    zf.write(path, path.relative_to(ROOT).as_posix())
    print(f"built {BUNDLE} ({BUNDLE.stat().st_size} bytes)")
    return BUNDLE


def submit(bundle: Path) -> int:
    if not API_KEY:
        print("ERROR: API_KEY not set", file=sys.stderr)
        sys.exit(1)
    url = f"{APP_BASE_URL}/api/{CASE}/submissions"
    with open(bundle, "rb") as f:
        r = requests.post(
            url,
            headers={"X-API-Key": API_KEY},
            files={"file": (bundle.name, f, "application/zip")},
            timeout=300,
        )
    r.raise_for_status()
    sub = r.json()
    sub_id = sub.get("id")
    print(f"submitted: id={sub_id} status={sub.get('status')}")
    return int(sub_id)


def poll(sub_id: int) -> None:
    url = f"{APP_BASE_URL}/api/{CASE}/submissions/{sub_id}"
    while True:
        r = requests.get(url, headers={"X-API-Key": API_KEY}, timeout=60)
        r.raise_for_status()
        sub = r.json()
        status = sub.get("status")
        print(f"  status={status}")
        if status in ("completed", "failed", "timeout", "error"):
            print(f"score: {sub.get('score')}")
            print(f"details: {sub.get('score_details')}")
            if sub.get("error_log"):
                print(f"error_log: {sub.get('error_log')}")
            return
        time.sleep(10)


def main() -> None:
    bundle = build_bundle()
    sub_id = submit(bundle)
    poll(sub_id)


if __name__ == "__main__":
    main()
