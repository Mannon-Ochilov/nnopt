"""Resumable model download with retries.

The network on this machine drops DNS intermittently, and a 6.9 GB
single-file checkpoint will not survive that in one attempt. huggingface_hub
keeps partial files and resumes, so the fix is simply to keep re-entering
snapshot_download until it returns.

Usage:  python robust_download.py <repo_id> [cache_dir]
"""

import sys
import time

from huggingface_hub import snapshot_download

MAX_ATTEMPTS = 40
SLEEP_SECONDS = 15


def main():
    repo = sys.argv[1] if len(sys.argv) > 1 else "openlm-research/open_llama_3b_v2"
    cache_dir = sys.argv[2] if len(sys.argv) > 2 else "models/llama_cache"

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            path = snapshot_download(
                repo_id=repo,
                cache_dir=cache_dir,
                allow_patterns=["*.json", "*.model", "*.bin", "*.safetensors", "*.txt"],
                max_workers=2,
            )
            print(f"\nTAYYOR: {path}")
            return 0
        except Exception as exc:
            print(f"[urinish {attempt}/{MAX_ATTEMPTS}] uzildi: {type(exc).__name__}: "
                  f"{str(exc)[:120]}", flush=True)
            if attempt == MAX_ATTEMPTS:
                print("barcha urinishlar tugadi")
                return 1
            time.sleep(SLEEP_SECONDS)


if __name__ == "__main__":
    sys.exit(main())
