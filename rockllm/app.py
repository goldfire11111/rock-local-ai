"""Desktop app mode: runs the server in this process, shows a tray icon and opens the chat window."""
import os
import shutil
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

from . import config


def message_box(text, title="Rock", error=True):
    if os.name == "nt":
        import ctypes
        ctypes.windll.user32.MessageBoxW(0, text, title, 0x10 if error else 0x40)
    else:
        print(text, file=sys.stderr)


def _find_app_browser():
    """Edge (always present on Windows 11) or Chrome can show a page as a standalone app window."""
    cands = [shutil.which("msedge"), shutil.which("chrome")]
    for env in ("ProgramFiles(x86)", "ProgramFiles", "LOCALAPPDATA"):
        base = os.environ.get(env)
        if base:
            cands += [os.path.join(base, "Microsoft", "Edge", "Application", "msedge.exe"),
                      os.path.join(base, "Google", "Chrome", "Application", "chrome.exe")]
    for c in cands:
        if c and os.path.exists(c):
            return c
    return None


def open_window():
    url = config.base_url() + "/"
    browser = _find_app_browser()
    if browser:
        import subprocess
        profile = config.HOME / "window"
        subprocess.Popen([browser, f"--app={url}", f"--user-data-dir={profile}", "--window-size=1180,820",
                          "--no-first-run", "--no-default-browser-check", "--disable-sync",
                          "--disable-features=msEdgeSidebarV2,msHubApps,msUndersideButton"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         creationflags=0x08000000 if os.name == "nt" else 0)
    else:
        webbrowser.open(url)


def make_icon_image(size=64):
    from PIL import Image, ImageDraw
    s = size / 32
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    pts = [(2, 26), (9, 8), (18, 4), (28, 12), (30, 26)]
    d.polygon([(x * s, y * s) for x, y in pts], fill=(212, 163, 115, 255))
    lw = max(1, int(1.6 * s))
    for a, b in (((9, 8), (14, 17)), ((14, 17), (18, 4)), ((14, 17), (28, 12)), ((14, 17), (2, 26)),
                 ((14, 17), (22, 26))):
        d.line([(a[0] * s, a[1] * s), (b[0] * s, b[1] * s)], fill=(60, 42, 30, 255), width=lw)
    return img


def _server_up():
    try:
        with urllib.request.urlopen(config.base_url() + "/api/version", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def _redirect_logs():
    """With no console (double-clicked app), write logs to data/logs/app.log."""
    if sys.stderr is None or config.FROZEN:
        f = open(config.LOG_DIR / "app.log", "a", encoding="utf-8", buffering=1)
        sys.stderr = f
        if sys.stdout is None or config.FROZEN:
            sys.stdout = f


def run(no_window=False):
    if _server_up():  # already running: just show the window
        if not no_window:
            open_window()
        return
    _redirect_logs()
    from . import server
    try:
        httpd = server.make_server()
    except SystemExit as e:
        # Another Rock instance may have started a moment earlier: hand over to it quietly.
        for _ in range(40):
            if _server_up():
                if not no_window:
                    open_window()
                return
            time.sleep(0.25)
        message_box(str(e) + "\n\nAnother program may be using this port. Set ROCK_HOST to use a different one.")
        return
    srv = threading.Thread(target=server.serve, args=(httpd,), daemon=True)
    srv.start()
    if not no_window:
        open_window()

    try:
        import pystray
    except ImportError:  # running from source without pystray: no tray, just keep serving
        srv.join()
        return

    def status_text(_=None):
        st = server.ENGINE.status()
        if server.engine.install_status.get("state") == "installing":
            return "Installing engine..."
        return f"Loaded: {st['model']['name']}" if st else "No model loaded"

    def on_open(icon, item):
        open_window()

    def on_unload(icon, item):
        threading.Thread(target=server.ENGINE.unload, daemon=True).start()

    def on_folder(path):
        return lambda icon, item: os.startfile(str(path)) if os.name == "nt" else None

    def on_quit(icon, item):
        icon.visible = False
        icon.stop()

    menu = pystray.Menu(
        pystray.MenuItem("Open Rock", on_open, default=True),
        pystray.MenuItem(status_text, None, enabled=False),
        pystray.MenuItem("Unload model", on_unload, enabled=lambda item: server.ENGINE.status() is not None),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Open models folder", on_folder(config.MODELS_DIR)),
        pystray.MenuItem("Open logs folder", on_folder(config.LOG_DIR)),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Quit Rock", on_quit),
    )
    icon = pystray.Icon("Rock", make_icon_image(64), f"Rock - {config.base_url()}", menu)
    server.QUIT_HOOK = lambda: icon.stop()

    def refresher():
        while True:
            time.sleep(3)
            try:
                icon.title = f"Rock - {status_text()}"
                icon.update_menu()
            except Exception:
                pass

    threading.Thread(target=refresher, daemon=True).start()
    try:
        icon.run()
    finally:
        server.ENGINE.shutdown()
        threading.Thread(target=httpd.shutdown, daemon=True).start()
        time.sleep(0.5)
