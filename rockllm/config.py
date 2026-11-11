"""Paths and settings. Everything can be overridden with environment variables."""
import os
import sys
from pathlib import Path

APP_NAME = "Rock"
VERSION = "1.1.0"

FROZEN = getattr(sys, "frozen", False)  # running as the packaged Rock.exe
if FROZEN:
    ROOT = Path(sys.executable).resolve().parent          # folder containing Rock.exe
    RESOURCES = Path(getattr(sys, "_MEIPASS", ROOT))       # bundled files (web UI)
else:
    ROOT = Path(__file__).resolve().parent.parent
    RESOURCES = ROOT
WEB_DIR = RESOURCES / "rockllm" / "web"


def _writable(d):
    try:
        d.mkdir(parents=True, exist_ok=True)
        t = d / ".write-test"
        t.write_text("ok")
        t.unlink()
        return True
    except OSError:
        return False


def _default_home():
    # Running from source, or a "data" folder already sits next to Rock.exe (portable mode):
    # use it. Otherwise (e.g. Rock.exe run from Downloads) keep data in %LOCALAPPDATA%\Rock.
    here = ROOT / "data"
    if (not FROZEN or here.is_dir()) and _writable(here):
        return here
    base = os.environ.get("LOCALAPPDATA") or (Path.home() / ".local" / "share")
    return Path(base) / "Rock"


# Where engine + models live.
HOME = Path(os.environ["ROCK_HOME"]) if os.environ.get("ROCK_HOME") else _default_home()
ENGINE_DIR = HOME / "engine"
MODELS_DIR = Path(os.environ.get("ROCK_MODELS", HOME / "models"))
REGISTRY_FILE = HOME / "models.json"
LOG_DIR = HOME / "logs"

# API server address (Ollama uses 11434; we default to 11435 so both can run side by side).
_host = os.environ.get("ROCK_HOST", "127.0.0.1:11435")
if "://" in _host:
    _host = _host.split("://", 1)[1]
HOST, _, _port = _host.rpartition(":")
HOST = HOST or "127.0.0.1"
PORT = int(_port or 11435)

# Engine defaults
BACKEND = os.environ.get("ROCK_BACKEND", "auto")          # auto | cuda | vulkan | cpu
DEFAULT_CTX = int(os.environ.get("ROCK_NUM_CTX", "8192"))
KEEP_ALIVE = int(os.environ.get("ROCK_KEEP_ALIVE", "300"))  # seconds idle before unloading
ENGINE_PORT = int(os.environ.get("ROCK_ENGINE_PORT", "39281"))

for d in (HOME, ENGINE_DIR, MODELS_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)


def base_url():
    host = "127.0.0.1" if HOST in ("0.0.0.0", "") else HOST
    return f"http://{host}:{PORT}"
