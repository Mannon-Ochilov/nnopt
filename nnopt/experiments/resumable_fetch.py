"""Byte-range resumable downloader with verified completion.

Neither huggingface_hub nor BITS worked here: hf_hub exhausts its internal
retries and then returns SUCCESS with a partial file (which silently defeats
any retry loop built around its return value), and BITS fails with a COM
exception on the HF CDN redirect.

This does the one thing that has to be right: track the expected size, send
a Range header for whatever is missing, and keep going until the file on
disk matches Content-Length exactly. Failure is reported by size mismatch,
not by trusting a library's exit status.

Usage:  python resumable_fetch.py <url> <destination> [expected_bytes]
"""

import os
import sys
import time

import requests

CHUNK = 8 * 1024 * 1024
MAX_ATTEMPTS = 200
SLEEP_SECONDS = 8
TIMEOUT = (15, 120)   # (connect, read)


def remote_size(url):
    for _ in range(5):
        try:
            r = requests.head(url, allow_redirects=True, timeout=TIMEOUT)
            if r.status_code < 400 and "content-length" in r.headers:
                return int(r.headers["content-length"])
        except Exception:
            time.sleep(3)
    return None


def fetch(url, dst, expected=None):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if expected is None:
        expected = remote_size(url)
    if expected:
        print(f"kutilayotgan hajm: {expected/1024**3:.2f} GB")

    for attempt in range(1, MAX_ATTEMPTS + 1):
        have = os.path.getsize(dst) if os.path.exists(dst) else 0
        if expected and have >= expected:
            print(f"\nTAYYOR: {have/1024**3:.2f} GB")
            return 0

        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with requests.get(url, headers=headers, stream=True,
                              allow_redirects=True, timeout=TIMEOUT) as r:
                if r.status_code not in (200, 206):
                    raise RuntimeError(f"HTTP {r.status_code}")
                mode = "ab" if have else "wb"
                last_report = time.time()
                with open(dst, mode) as f:
                    for chunk in r.iter_content(CHUNK):
                        if not chunk:
                            break
                        f.write(chunk)
                        if time.time() - last_report > 30:
                            cur = os.path.getsize(dst)
                            pct = f" ({cur/expected:.1%})" if expected else ""
                            print(f"  {cur/1024**3:.2f} GB{pct}", flush=True)
                            last_report = time.time()
        except Exception as exc:
            cur = os.path.getsize(dst) if os.path.exists(dst) else 0
            print(f"[urinish {attempt}] uzildi {cur/1024**3:.2f} GB da: "
                  f"{type(exc).__name__}: {str(exc)[:90]}", flush=True)
            time.sleep(SLEEP_SECONDS)
            continue

        have = os.path.getsize(dst)
        if expected and have < expected:
            print(f"[urinish {attempt}] qisman: {have/1024**3:.2f} / "
                  f"{expected/1024**3:.2f} GB — davom etadi", flush=True)
            time.sleep(2)

    have = os.path.getsize(dst) if os.path.exists(dst) else 0
    print(f"TUGALLANMADI: {have/1024**3:.2f} / "
          f"{(expected or 0)/1024**3:.2f} GB")
    return 1


if __name__ == "__main__":
    sys.exit(fetch(sys.argv[1], sys.argv[2],
                   int(sys.argv[3]) if len(sys.argv) > 3 else None))
