"""Command line interface - works like the `ollama` command."""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser

from . import config, engine, models
from .util import ago, human_size

BASE = config.base_url()


# --------------------------------------------------------------------------- client helpers

def api(path, body=None, method=None, timeout=30):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method or ("POST" if data else "GET"),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        try:
            msg = json.loads(e.read()).get("error")
        except Exception:
            msg = str(e)
        raise SystemExit(f"Error: {msg}")


def api_stream(path, body):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        r = urllib.request.urlopen(req, timeout=3600)
    except urllib.error.HTTPError as e:
        try:
            msg = json.loads(e.read()).get("error")
        except Exception:
            msg = str(e)
        raise SystemExit(f"Error: {msg}")
    with r:
        for line in r:
            line = line.strip()
            if not line:
                continue
            ev = json.loads(line)
            if ev.get("error"):
                raise SystemExit(f"\nError: {ev['error']}")
            yield ev


def server_up():
    try:
        with urllib.request.urlopen(BASE + "/api/version", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def self_command(*args):
    """Command line that re-runs this program (Rock.exe, or python rock.py)."""
    if config.FROZEN:
        return [sys.executable, *args]
    exe = sys.executable
    if os.name == "nt":
        pyw = exe.replace("python.exe", "pythonw.exe")
        exe = pyw if os.path.exists(pyw) else exe
    return [exe, str(config.ROOT / "rock.py"), *args]


def spawn_detached(*args):
    log = open(config.LOG_DIR / "server.log", "a", encoding="utf-8")
    kw = {}
    if os.name == "nt":
        # Hidden console (not DETACHED_PROCESS): the exe's internal child process then shares that
        # console instead of having Windows create a new one, which can fail with 0xc0000142.
        kw["creationflags"] = 0x00000200 | 0x08000000  # NEW_PROCESS_GROUP | NO_WINDOW
    else:
        kw["start_new_session"] = True
    env = dict(os.environ, PYINSTALLER_RESET_ENVIRONMENT="1")  # new exe instance, own temp folder
    subprocess.Popen(self_command(*args), stdout=log, stderr=log, stdin=subprocess.DEVNULL,
                     cwd=str(config.ROOT), env=env, **kw)


def ensure_server(quiet=False):
    """Start Rock in the background (tray icon, no window) if it isn't running - like Ollama."""
    if server_up():
        return
    if not quiet:
        print("Starting Rock in the background...", file=sys.stderr)
    spawn_detached("app", "--no-window")
    for _ in range(300):  # first launch can be slow while antivirus scans the exe
        if server_up():
            return
        time.sleep(0.2)
    raise SystemExit(f"Server did not start. See {config.LOG_DIR / 'server.log'}")


class Bar:
    def __init__(self):
        self.label = None

    def __call__(self, status, done=0, total=0):
        if status != self.label and self.label is not None:
            sys.stderr.write("\n")
        self.label = status
        if total:
            pct = done / total
            w = 30
            bar = "#" * int(pct * w) + "-" * (w - int(pct * w))
            sys.stderr.write(f"\r{status[:40]:<40} [{bar}] {pct*100:5.1f}%  {human_size(done)}/{human_size(total)}  ")
        else:
            sys.stderr.write(f"\r{status}")
        sys.stderr.flush()

    def end(self):
        sys.stderr.write("\n")


def table(rows, headers):
    widths = [max(len(str(x)) for x in col) for col in zip(headers, *rows)]
    for r in [headers] + rows:
        print("    ".join(str(c).ljust(w) for c, w in zip(r, widths)).rstrip())


# --------------------------------------------------------------------------- commands

def cmd_serve(a):
    from . import server
    server.serve()


def cmd_setup(a):
    bar = Bar()
    info = engine.install(a.backend, force=True, progress=bar)
    bar.end()
    print(f"Installed llama.cpp {info['tag']} ({info['backend']}) into {engine.BIN_DIR}")


def cmd_pull(a):
    bar = Bar()
    if server_up():
        for ev in api_stream("/api/pull", {"model": a.model}):
            bar(ev["status"], ev.get("completed", 0), ev.get("total", 0))
    else:
        models.pull(a.model, bar)
    bar.end()


def cmd_list(a):
    rows = []
    for m in models.list_models().values():
        rows.append([m["name"] + ("  (can't load: Ollama-only format)" if m.get("unsupported") else ""),
                     human_size(m["size"]), "yes" if m.get("mmproj") else "",
                     ago(m.get("added") or 0), m.get("source") or ""])
    if not rows:
        print("No models yet. Try:  rock pull qwen3:4b   or   rock import-ollama")
        return
    table(rows, ["NAME", "SIZE", "VISION", "MODIFIED", "SOURCE"])


def cmd_ps(a):
    if not server_up():
        print("Server is not running.")
        return
    rows = []
    for m in api("/api/ps")["models"]:
        exp = m.get("expires_at")
        rows.append([m["name"], human_size(m["size"]), m.get("context_length"),
                     "busy" if m.get("busy") else "idle", exp[:19].replace("T", " ") + " UTC" if exp else "forever"])
    table(rows, ["NAME", "SIZE", "CONTEXT", "STATE", "UNTIL"])


def cmd_rm(a):
    for name in a.models:
        if server_up():
            api("/api/delete", {"model": name}, method="DELETE")
        else:
            try:
                models.remove(name)
            except KeyError:
                raise SystemExit(f"Error: model '{name}' not found")
        print(f"deleted '{name}'")


def cmd_stop(a):
    if server_up():
        api("/api/unload", {"model": a.model} if a.model else {})
    print("stopped")


def cmd_show(a):
    m = models.resolve(a.model)
    if not m:
        raise SystemExit(f"Error: model '{a.model}' not found")
    d = models.describe(m)
    if a.template:
        print(d.get("chat_template") or "(no chat template)")
        return
    print("  Model")
    for k, label in (("architecture", "architecture"), ("parameters", "parameters"),
                     ("context_length", "context length"), ("embedding_length", "embedding length"),
                     ("quantization", "quantization")):
        if d.get(k):
            print(f"    {label:<20}{d[k]}")
    print(f"    {'size':<20}{human_size(d['size'])}")
    print(f"    {'vision':<20}{'yes' if d.get('mmproj') else 'no'}")
    print(f"    {'chat template':<20}{'yes' if d.get('has_chat_template') else 'no'}")
    print("\n  File")
    print(f"    {d['path']}")
    if d.get("source"):
        print(f"    source: {d['source']}")
    if d.get("params"):
        print("\n  Parameters")
        for k, v in d["params"].items():
            print(f"    {k:<20}{v}")
    if d.get("system"):
        print("\n  System\n    " + d["system"].strip().replace("\n", "\n    "))
    if d.get("license"):
        print(f"\n  License\n    {d['license']}")


def cmd_import(a):
    m = models.import_file(a.path, a.name)
    print(f"imported '{m['name']}' -> {m['path']}")


def cmd_import_ollama(a):
    found = models.import_ollama(a.path)
    if not found:
        print("No Ollama models found.")
    for name, new in found:
        print(f"{'imported' if new else 'already have'}  {name}")


def cmd_catalog(a):
    have = models.list_models()
    rows = [[k, v[1], "installed" if k in have else "", v[0]] for k, v in models.CATALOG.items()]
    table(rows, ["NAME", "DESCRIPTION", "", "HUGGING FACE"])
    print("\nAny GGUF repo on huggingface.co works too, e.g.  rock pull unsloth/Qwen3-4B-GGUF:Q4_K_M")


def cmd_ui(a):
    from . import app
    ensure_server()
    app.open_window()
    print(f"Chat UI: {BASE}/")


def cmd_app(a):
    from . import app
    app.run(no_window=a.no_window)


def cmd_quit(a):
    if not server_up():
        print("Rock is not running.")
        return
    try:
        api("/api/quit", {})
    except Exception:
        pass
    print("Rock stopped.")


def cmd_run(a):
    m = models.resolve(a.model)
    if not m:
        try:
            models.parse_spec(a.model)
        except ValueError as e:
            raise SystemExit(f"Error: {e}")
        print(f"pulling {a.model}...")
        cmd_pull(a)
        m = models.resolve(a.model) or models.resolve(models.parse_spec(a.model)["name"])
    ensure_server()
    name = m["name"]
    opts = {}
    if a.ctx:
        opts["num_ctx"] = a.ctx
    think = None if a.think is None else a.think == "on"

    if a.prompt:
        prompt = " ".join(a.prompt)
        if not sys.stdin.isatty():
            prompt = sys.stdin.read() + "\n\n" + prompt
        chat_once(name, [{"role": "user", "content": prompt}], opts, think, a.verbose)
        return

    sys.stderr.write("Loading model...")
    sys.stderr.flush()
    for _ in api_stream("/api/chat", {"model": name, "messages": [], "options": opts}):
        pass
    sys.stderr.write("\r" + " " * 20 + "\r")
    print(f">>> Chatting with {name}. Type /? for help, /bye to exit.")
    history, system = [], None
    while True:
        try:
            line = input(">>> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if line.startswith('"""'):
            buf = [line[3:]]
            while not buf[-1].endswith('"""'):
                try:
                    buf.append(input("... "))
                except (EOFError, KeyboardInterrupt):
                    break
            line = "\n".join(buf)[:-3] if buf[-1].endswith('"""') else "\n".join(buf)
        s = line.strip()
        if not s:
            continue
        if s in ("/bye", "/exit", "/quit"):
            break
        if s in ("/?", "/help"):
            print("  /set system <text>   set a system prompt\n  /set think on|off    toggle thinking\n"
                  "  /set ctx <n>         context length\n  /clear               forget the conversation\n"
                  "  /show                model info\n  /bye                 exit\n"
                  '  """                  start/end a multi-line message')
            continue
        if s == "/clear":
            history = []
            print("Cleared.")
            continue
        if s == "/show":
            a.template = False
            a.model = name
            cmd_show(a)
            continue
        if s.startswith("/set "):
            _, key, *rest = s.split(" ", 2)
            val = rest[0] if rest else ""
            if key == "system":
                system = val
                print("Set system prompt.")
            elif key == "think":
                think = val != "off"
                print(f"Thinking {'on' if think else 'off'}.")
            elif key in ("ctx", "num_ctx"):
                opts["num_ctx"] = int(val)
                print(f"Context set to {val}.")
            elif key == "verbose":
                a.verbose = True
            else:
                opts[key] = json.loads(val) if val else None
            continue
        history.append({"role": "user", "content": line})
        msgs = ([{"role": "system", "content": system}] if system else []) + history
        try:
            reply = chat_once(name, msgs, opts, think, a.verbose)
        except KeyboardInterrupt:
            print("\n(stopped)")
            reply = None
        if reply is None:
            history.pop()
        else:
            history.append({"role": "assistant", "content": reply})


def chat_once(name, msgs, opts, think, verbose):
    body = {"model": name, "messages": msgs, "options": opts, "stream": True}
    if think is not None:
        body["think"] = think
    out, in_think = [], False
    final = {}
    for ev in api_stream("/api/chat", body):
        msg = ev.get("message") or {}
        if msg.get("thinking"):
            if not in_think:
                sys.stdout.write("\x1b[2mThinking...\n")
                in_think = True
            sys.stdout.write(msg["thinking"])
        if msg.get("content"):
            if in_think:
                sys.stdout.write("\n...done thinking.\x1b[0m\n\n")
                in_think = False
            sys.stdout.write(msg["content"])
            out.append(msg["content"])
        sys.stdout.flush()
        if ev.get("done"):
            final = ev
    if in_think:
        sys.stdout.write("\x1b[0m")
    print("\n")
    if verbose and final:
        ec, ed = final.get("eval_count", 0), final.get("eval_duration", 0) / 1e9
        pc, pd = final.get("prompt_eval_count", 0), final.get("prompt_eval_duration", 0) / 1e9
        print(f"total duration:    {final.get('total_duration', 0)/1e9:.2f}s")
        print(f"load duration:     {final.get('load_duration', 0)/1e9:.2f}s")
        print(f"prompt eval:       {pc} tokens, {pc/pd if pd else 0:.1f} tokens/s")
        print(f"eval:              {ec} tokens, {ec/ed if ed else 0:.1f} tokens/s\n")
    return "".join(out)


def _enable_ansi():
    """Turn on ANSI colours in the Windows console (without spawning cmd.exe)."""
    try:
        import ctypes
        k32 = ctypes.windll.kernel32
        h = k32.GetStdHandle(-11)
        mode = ctypes.c_uint32()
        if k32.GetConsoleMode(h, ctypes.byref(mode)):
            k32.SetConsoleMode(h, mode.value | 0x0004)  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
    except Exception:
        pass


def main(argv=None):
    if os.name == "nt":
        _enable_ansi()
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    p = argparse.ArgumentParser(prog="rock", description="Rock - run GGUF AI models locally (powered by llama.cpp)")
    sub = p.add_subparsers(dest="cmd")

    s = sub.add_parser("serve", help="start the API server + web UI")
    s.set_defaults(fn=cmd_serve)
    s = sub.add_parser("run", help="chat with a model (pulls it if needed)")
    s.add_argument("model")
    s.add_argument("prompt", nargs="*")
    s.add_argument("--verbose", action="store_true", help="show token speed")
    s.add_argument("--ctx", type=int, help="context length")
    s.add_argument("--think", choices=["on", "off"], help="enable/disable thinking")
    s.set_defaults(fn=cmd_run)
    s = sub.add_parser("pull", help="download a model from Hugging Face")
    s.add_argument("model")
    s.set_defaults(fn=cmd_pull)
    for n in ("list", "ls"):
        sub.add_parser(n, help="list local models" if n == "list" else argparse.SUPPRESS).set_defaults(fn=cmd_list)
    sub.add_parser("ps", help="show the loaded model").set_defaults(fn=cmd_ps)
    s = sub.add_parser("rm", help="delete a model")
    s.add_argument("models", nargs="+")
    s.set_defaults(fn=cmd_rm)
    s = sub.add_parser("stop", help="unload the running model")
    s.add_argument("model", nargs="?")
    s.set_defaults(fn=cmd_stop)
    s = sub.add_parser("show", help="show model details")
    s.add_argument("model")
    s.add_argument("--template", action="store_true", help="print the chat template")
    s.set_defaults(fn=cmd_show)
    s = sub.add_parser("import", help="add a .gguf file from anywhere on disk")
    s.add_argument("path")
    s.add_argument("--name")
    s.set_defaults(fn=cmd_import)
    s = sub.add_parser("import-ollama", help="use models you already downloaded with Ollama")
    s.add_argument("--path", help="Ollama models folder (default ~/.ollama/models)")
    s.set_defaults(fn=cmd_import_ollama)
    sub.add_parser("catalog", help="list model shortcuts you can pull").set_defaults(fn=cmd_catalog)
    s = sub.add_parser("setup", help="download/update the llama.cpp engine")
    s.add_argument("--backend", choices=["cuda", "vulkan", "cpu"], help="force a backend")
    s.set_defaults(fn=cmd_setup)
    sub.add_parser("ui", help="open the chat window").set_defaults(fn=cmd_ui)
    s = sub.add_parser("app", help="run the desktop app (tray icon + chat window)")
    s.add_argument("--no-window", action="store_true", help="start in the tray without opening a window")
    s.set_defaults(fn=cmd_app)
    sub.add_parser("quit", help="stop Rock completely (server + tray icon)").set_defaults(fn=cmd_quit)

    if argv is None and len(sys.argv) == 1:
        # Double-clicked (or run with no arguments): launch the desktop app detached so no
        # console window hangs around, then exit.
        if server_up():
            from . import app
            app.open_window()
        else:
            spawn_detached("app")
        return
    a = p.parse_args(argv)
    if not getattr(a, "fn", None):
        p.print_help()
        return
    try:
        a.fn(a)
    except KeyboardInterrupt:
        print()
    except (ValueError, RuntimeError, FileNotFoundError) as e:
        raise SystemExit(f"Error: {e}")
