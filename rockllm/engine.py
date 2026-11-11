"""The inference engine: downloads llama.cpp's official llama-server and runs one model at a time."""
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import urllib.request
import zipfile
from contextlib import contextmanager
from pathlib import Path

from . import config
from .util import download, http_json, log

RELEASES_API = "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=15"
ENGINE_INFO = config.ENGINE_DIR / "engine.json"
BIN_DIR = config.ENGINE_DIR / "bin"
IS_WIN = os.name == "nt"

install_status = {"state": "idle", "message": "", "completed": 0, "total": 0}
_install_lock = threading.Lock()


# --------------------------------------------------------------------------- install

def nvidia_driver_major():
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "--query-gpu=driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10,
                             creationflags=_no_window()).stdout.strip()
        return int(out.splitlines()[0].split(".")[0])
    except Exception:
        return None


def detect_backend():
    if config.BACKEND != "auto":
        return config.BACKEND
    if sys.platform == "darwin":
        return "metal"
    drv = nvidia_driver_major()
    if drv and drv >= 525:  # CUDA 12.x needs driver 525+
        return "cuda"
    if IS_WIN:
        # Vulkan works on AMD, Intel and older NVIDIA GPUs; needs the Vulkan runtime from the GPU driver.
        sysdir = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32"
        return "vulkan" if (sysdir / "vulkan-1.dll").exists() else "cpu"
    return "vulkan" if shutil.which("vulkaninfo") else "cpu"


def _platform_tag():
    machine = platform.machine().lower()
    arch = "arm64" if machine in ("arm64", "aarch64") else "x64"
    if IS_WIN:
        return "win", arch
    if sys.platform == "darwin":
        return "macos", arch
    return "ubuntu", arch


def _pick_assets(release, backend):
    """Return (main_asset, extra_assets) for this OS/backend, or None."""
    osname, arch = _platform_tag()
    names = {a["name"]: a for a in release.get("assets", [])}
    tag = release["tag_name"]
    pre = f"llama-{tag}-bin-{osname}-"
    if osname == "macos":
        a = names.get(f"{pre}{arch}.tar.gz")
        return (a, []) if a else None
    if backend == "cuda":
        driver = nvidia_driver_major() or 0
        cuda = [n for n in names if n.startswith(pre + "cuda-") and n.endswith(f"-{arch}.zip" if IS_WIN else f"-{arch}.tar.gz")]
        # CUDA 13 needs driver >= 580; otherwise use the newest 12.x build.
        def ver(n):
            m = re.search(r"cuda-(\d+)\.(\d+)", n)
            return (int(m.group(1)), int(m.group(2))) if m else (0, 0)
        ok = [n for n in cuda if ver(n)[0] < 13 or driver >= 580]
        if not ok:
            return None
        best = max(ok, key=ver)
        v = "%d.%d" % ver(best)
        extras = [a for n, a in names.items() if n.startswith("cudart-") and f"cuda-{v}-{arch}" in n and osname in n]
        return names[best], extras
    ext = "zip" if IS_WIN else "tar.gz"
    key = f"{pre}{backend}-{arch}.{ext}" if backend != "cpu" else (f"{pre}cpu-{arch}.{ext}" if IS_WIN else f"{pre}{arch}.{ext}")
    a = names.get(key)
    return (a, []) if a else None


def engine_info():
    try:
        return json.loads(ENGINE_INFO.read_text())
    except Exception:
        return None


def server_exe():
    exe_name = "llama-server.exe" if IS_WIN else "llama-server"
    for p in BIN_DIR.rglob(exe_name):
        return p
    return None


def is_installed():
    return server_exe() is not None


def install(backend=None, force=False, progress=None):
    """Download the newest llama.cpp build for this machine into data/engine."""
    with _install_lock:
        if is_installed() and not force:
            return engine_info()
        return _install(backend or detect_backend(), progress)


def _install(backend, progress):
        install_status.update(state="installing", message="Finding latest llama.cpp release...", completed=0, total=0)

        def report(msg, done=0, total=0):
            install_status.update(message=msg, completed=done, total=total)
            if progress:
                progress(msg, done, total)

        try:
            report("Finding latest llama.cpp release...")
            releases = http_json(RELEASES_API)
            chosen = None
            for rel in releases:
                if not rel["tag_name"].startswith("b"):
                    continue
                picked = _pick_assets(rel, backend)
                if picked:
                    chosen = (rel, picked)
                    break
            if not chosen and backend != "cpu":
                log(f"No {backend} build found, falling back to CPU")
                return _install("cpu", progress)
            if not chosen:
                raise RuntimeError("Could not find a llama.cpp build for this computer.")
            rel, (main, extras) = chosen
            tmp = config.ENGINE_DIR / "_download"
            tmp.mkdir(exist_ok=True)
            new_bin = config.ENGINE_DIR / "bin.new"
            if new_bin.exists():
                shutil.rmtree(new_bin)
            new_bin.mkdir()
            for asset in [main] + extras:
                dest = tmp / asset["name"]
                download(asset["browser_download_url"], dest,
                         lambda d, t, n=asset["name"]: report(f"Downloading {n}", d, t))
                report(f"Extracting {asset['name']}")
                _extract(dest, new_bin)
            if BIN_DIR.exists():
                shutil.rmtree(BIN_DIR)
            new_bin.rename(BIN_DIR)
            shutil.rmtree(tmp, ignore_errors=True)
            if not IS_WIN:
                for p in BIN_DIR.rglob("llama-*"):
                    p.chmod(0o755)
            info = {"tag": rel["tag_name"], "backend": backend, "asset": main["name"],
                    "installed": time.strftime("%Y-%m-%d %H:%M:%S")}
            ENGINE_INFO.write_text(json.dumps(info, indent=2))
            _help_cache.clear()
            install_status.update(state="ready", message=f"llama.cpp {rel['tag_name']} ({backend})")
            return info
        except Exception as e:
            install_status.update(state="error", message=str(e))
            raise


def _extract(archive, dest):
    if archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive) as z:
            z.extractall(dest)
    else:
        with tarfile.open(archive) as t:
            t.extractall(dest)


# --------------------------------------------------------------------------- running

def _no_window():
    return 0x08000000 if IS_WIN else 0  # CREATE_NO_WINDOW


_job = None


def _kill_with_parent(proc):
    """Windows: put the engine in a Job Object so it dies if Rock exits or crashes."""
    global _job
    if not IS_WIN:
        return
    try:
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.CreateJobObjectW.restype = wintypes.HANDLE
        k32.OpenProcess.restype = wintypes.HANDLE
        k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        if _job is None:
            class LIMIT(ctypes.Structure):
                _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                            ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                            ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                            ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                            ("SchedulingClass", wintypes.DWORD)]

            class IO(ctypes.Structure):
                _fields_ = [(n, ctypes.c_uint64) for n in ("r", "w", "o", "rb", "wb", "ob")]

            class EXT(ctypes.Structure):
                _fields_ = [("Basic", LIMIT), ("Io", IO), ("ProcessMemoryLimit", ctypes.c_size_t),
                            ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t),
                            ("PeakJobMemoryUsed", ctypes.c_size_t)]
            job = k32.CreateJobObjectW(None, None)
            info = EXT()
            info.Basic.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            k32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info))
            _job = job
        h = k32.OpenProcess(0x0101, False, proc.pid)  # PROCESS_SET_QUOTA | PROCESS_TERMINATE
        k32.AssignProcessToJobObject(_job, h)
        k32.CloseHandle(h)
    except Exception as e:
        log(f"(could not tie engine lifetime to Rock: {e})")


_help_cache = {}


def supported_flags():
    """Read `llama-server --help` once so we only pass flags this build understands."""
    if "flags" not in _help_cache:
        try:
            out = subprocess.run([str(server_exe()), "--help"], capture_output=True, text=True,
                                 timeout=30, creationflags=_no_window(), encoding="utf-8", errors="replace")
            text = out.stdout + out.stderr
        except Exception:
            text = ""
        _help_cache["flags"] = set(re.findall(r"(?<![\w-])(--?[a-zA-Z][\w-]*)", text))
    return _help_cache["flags"]


def parse_keep_alive(v):
    """Ollama-style keep_alive: seconds, '5m', '1h', 0 = unload now, negative = forever."""
    if v is None or v == "":
        return None
    if isinstance(v, (int, float)):
        return float(v)
    m = re.fullmatch(r"\s*(-?\d+(?:\.\d+)?)\s*(ms|s|m|h)?\s*", str(v))
    if not m:
        return None
    n = float(m.group(1))
    return n * {"ms": 0.001, "s": 1, "m": 60, "h": 3600, None: 1}[m.group(2)]


class Engine:
    """Keeps at most one llama-server process alive and swaps models on demand."""

    def __init__(self):
        self.proc = None
        self.model = None      # registry entry dict of loaded model
        self.key = None        # (path, ctx, ngl, mmproj)
        self.loaded_at = 0
        self.load_seconds = 0
        self.last_used = 0
        self.keep_alive = config.KEEP_ALIVE
        self.active = 0
        self.cond = threading.Condition()
        self.port = config.ENGINE_PORT
        self.logfile = config.LOG_DIR / "engine.log"
        threading.Thread(target=self._reaper, daemon=True).start()

    # ---- public
    @contextmanager
    def use(self, model, num_ctx=None, num_gpu=None, keep_alive=None):
        """Make sure `model` is loaded, hold it while the request runs."""
        ctx = int(num_ctx or model.get("params", {}).get("num_ctx") or config.DEFAULT_CTX)
        key = (model["path"], ctx, num_gpu, model.get("mmproj"))
        ka = parse_keep_alive(keep_alive)
        with self.cond:
            while self.active and self.key != key:
                self.cond.wait()
            fresh = False
            if self.key != key or not self._alive():
                self._stop_locked()
                self._start_locked(model, ctx, num_gpu)
                self.key = key
                fresh = True
            self.active += 1
            if ka is not None:
                self.keep_alive = ka
        try:
            yield {"port": self.port, "load_seconds": self.load_seconds if fresh else 0}
        finally:
            with self.cond:
                self.active -= 1
                self.last_used = time.time()
                if self.active == 0 and self.keep_alive == 0:
                    self._stop_locked()
                self.cond.notify_all()

    def unload(self, path=None):
        with self.cond:
            while self.active:
                self.cond.wait()
            if path is None or (self.model and self.model["path"] == path):
                self._stop_locked()

    def status(self):
        with self.cond:
            if not self._alive():
                return None
            exp = None if self.keep_alive < 0 else self.last_used + self.keep_alive
            return {"model": self.model, "ctx": self.key[1], "loaded_at": self.loaded_at,
                    "expires_at": exp, "busy": self.active > 0, "port": self.port}

    def shutdown(self):
        with self.cond:
            self._stop_locked()

    # ---- internals
    def _alive(self):
        return self.proc is not None and self.proc.poll() is None

    def _start_locked(self, model, ctx, num_gpu):
        if not is_installed():
            log("Engine not installed yet, installing llama.cpp...")
            install()
        exe = server_exe()
        flags = supported_flags()
        cmd = [str(exe), "-m", model["path"], "--host", "127.0.0.1", "--port", str(self.port),
               "-c", str(ctx), "--alias", model["name"]]
        if "--jinja" in flags:
            cmd.append("--jinja")
        if "--parallel" in flags:
            cmd += ["--parallel", "1"]
        if "--no-webui" in flags:
            cmd.append("--no-webui")
        if num_gpu is not None:
            cmd += ["-ngl", str(num_gpu)]
        elif "--fit" not in flags:
            cmd += ["-ngl", "999"]  # older builds: offload everything to the GPU
        if model.get("mmproj") and Path(model["mmproj"]).exists():
            cmd += ["--mmproj", model["mmproj"]]
        if model.get("chat_template_file") and Path(model["chat_template_file"]).exists():
            cmd += ["--chat-template-file", model["chat_template_file"]]
        extra = os.environ.get("ROCK_ENGINE_ARGS")
        if extra:
            cmd += extra.split()
        log(f"Loading {model['name']} (ctx {ctx})")
        lf = open(self.logfile, "w", encoding="utf-8", errors="replace")
        lf.write(" ".join(f'"{c}"' if " " in c else c for c in cmd) + "\n\n")
        lf.flush()
        t0 = time.time()
        self.proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                     cwd=str(exe.parent), creationflags=_no_window())
        _kill_with_parent(self.proc)
        self.model = model
        try:
            self._wait_ready()
        except Exception:
            self._stop_locked()
            raise
        self.load_seconds = time.time() - t0
        self.loaded_at = self.last_used = time.time()
        self.keep_alive = config.KEEP_ALIVE
        log(f"Loaded {model['name']} in {self.load_seconds:.1f}s")

    def _wait_ready(self):
        deadline = time.time() + 900
        url = f"http://127.0.0.1:{self.port}/health"
        while time.time() < deadline:
            if self.proc.poll() is not None:
                tail = self.log_tail()
                hint = ""
                if "wrong number of tensors" in tail or "unknown model architecture" in tail:
                    if str(self.model.get("source", "")).startswith("ollama:registry.ollama.ai"):
                        hint = ("\n\nThis model came from Ollama's own library, which packages some newer models in a "
                                "format only Ollama's engine reads. Download the standard GGUF instead, e.g. "
                                "`rock pull gemma4:e4b` (see `rock catalog`).")
                    else:
                        hint = "\n\nThis llama.cpp build doesn't support this model. Try `rock setup` to update the engine."
                elif "out of memory" in tail.lower() or "failed to allocate" in tail.lower():
                    hint = "\n\nNot enough memory. Try a smaller context (Settings) or a smaller quantization."
                raise RuntimeError("The engine failed to load this model:\n" + tail + hint)
            try:
                with urllib.request.urlopen(url, timeout=2) as r:
                    if r.status == 200:
                        return
            except Exception:
                pass
            time.sleep(0.25)
        raise RuntimeError("Timed out waiting for the model to load.")

    def log_tail(self, n=15):
        try:
            lines = self.logfile.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            return ""
        keep = [l for l in lines if re.search(r"error|fail|unknown|unsupported|out of memory|exception", l, re.I)]
        return "\n".join((keep or lines)[-n:])

    def _stop_locked(self):
        if self.proc is not None:
            if self.proc.poll() is None:
                log(f"Unloading {self.model['name'] if self.model else 'model'}")
                self.proc.terminate()
                try:
                    self.proc.wait(10)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
        self.proc = None
        self.model = None
        self.key = None

    def _reaper(self):
        while True:
            time.sleep(5)
            with self.cond:
                if (self._alive() and self.active == 0 and self.keep_alive >= 0
                        and time.time() - self.last_used > self.keep_alive):
                    self._stop_locked()
