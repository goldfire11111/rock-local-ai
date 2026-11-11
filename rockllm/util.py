import json
import os
import sys
import time
import urllib.error
import urllib.request

UA = "Rock-LocalAI/1.0"


def log(msg):
    sys.stderr.write(time.strftime("[%H:%M:%S] ") + str(msg) + "\n")
    sys.stderr.flush()


def _headers(url):
    h = {"User-Agent": UA}
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token and "huggingface.co" in url:
        h["Authorization"] = f"Bearer {token}"
    return h


def http_json(url, timeout=30):
    req = urllib.request.Request(url, headers=_headers(url))
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def download(url, dest, progress=None, expected_size=None):
    """Download with resume support (writes to dest.part, renames when complete)."""
    dest = dest if hasattr(dest, "with_name") else __import__("pathlib").Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and (expected_size is None or dest.stat().st_size == expected_size):
        if progress:
            size = dest.stat().st_size
            progress(size, size)
        return dest
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    headers = _headers(url)
    if have:
        headers["Range"] = f"bytes={have}-"
    req = urllib.request.Request(url, headers=headers)
    try:
        r = urllib.request.urlopen(req, timeout=60)
    except urllib.error.HTTPError as e:
        if e.code == 416 and have:  # already complete
            part.replace(dest)
            return dest
        if e.code in (401, 403):
            raise RuntimeError(f"Access denied ({e.code}). This model may be gated: accept its license on "
                               "huggingface.co and set the HF_TOKEN environment variable.") from e
        raise
    with r:
        if have and r.status != 206:
            have = 0  # server ignored Range; start over
        total = r.headers.get("Content-Length")
        total = int(total) + have if total else (expected_size or 0)
        done = have
        last = 0
        with open(part, "ab" if have else "wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                now = time.time()
                if progress and now - last > 0.25:
                    progress(done, total)
                    last = now
    if total and done < total:
        raise RuntimeError(f"Download interrupted ({done}/{total} bytes). Run the same command again to resume.")
    part.replace(dest)
    if progress:
        progress(done, total or done)
    return dest


def human_size(n):
    n = float(n or 0)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit in ("B", "KB") else f"{n:.1f} {unit}"
        n /= 1024


def ago(ts):
    s = time.time() - ts
    for n, u in ((86400 * 30, "month"), (86400 * 7, "week"), (86400, "day"), (3600, "hour"), (60, "minute")):
        if s >= n:
            k = int(s // n)
            return f"{k} {u}{'s' if k > 1 else ''} ago"
    return "just now"
