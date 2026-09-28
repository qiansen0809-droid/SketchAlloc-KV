#!/usr/bin/env bash
set -euo pipefail

OUT="${1:-/root/autodl-tmp/datasets/ruler_8192/test-00000-of-00001.parquet}"
mkdir -p "$(dirname "$OUT")"

export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-600}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"

python - "$OUT" <<'PY'
import sys
import time
from pathlib import Path

from huggingface_hub import hf_hub_download

out = Path(sys.argv[1])
out.parent.mkdir(parents=True, exist_ok=True)

for attempt in range(1, 21):
    try:
        print(f"[RULER] download attempt {attempt}/20")
        cached = hf_hub_download(
            repo_id="simonjegou/ruler",
            repo_type="dataset",
            filename="8192/test-00000-of-00001.parquet",
            revision="24adceac8a0e6532936e8d721cd9e9084d2e4686",
            local_dir=str(out.parent),
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        cached = Path(cached)
        if cached.resolve() != out.resolve():
            out.write_bytes(cached.read_bytes())
        print(f"[RULER] ready: {out}")
        print(f"[RULER] size: {out.stat().st_size} bytes")
        break
    except Exception as e:
        print(f"[RULER] attempt {attempt} failed: {type(e).__name__}: {e}")
        if attempt == 20:
            raise
        time.sleep(min(5 * attempt, 30))
else:
    raise RuntimeError("unreachable")
PY
