"""HTTP server: Ollama-compatible /api/*, OpenAI-compatible /v1/*, and the web chat UI."""
import http.client
import json
import os
import re
import threading
import time
import traceback
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import config, engine, models
from .util import log

ENGINE = engine.Engine()
WEB_DIR = config.WEB_DIR

OPTION_MAP = {  # Ollama option -> llama-server / OpenAI field
    "temperature": "temperature", "top_p": "top_p", "top_k": "top_k", "min_p": "min_p",
    "repeat_penalty": "repeat_penalty", "repeat_last_n": "repeat_last_n", "seed": "seed",
    "presence_penalty": "presence_penalty", "frequency_penalty": "frequency_penalty",
    "typical_p": "typical_p", "mirostat": "mirostat", "mirostat_tau": "mirostat_tau",
    "mirostat_eta": "mirostat_eta", "stop": "stop",
}
LOAD_OPTIONS = {"num_ctx", "num_gpu", "num_thread", "num_batch", "use_mmap", "main_gpu"}


def now_iso():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class ApiError(Exception):
    def __init__(self, status, msg):
        super().__init__(msg)
        self.status = status


def get_model(body):
    name = body.get("model") or body.get("name")
    if not name:
        raise ApiError(400, "model is required")
    m = models.resolve(name)
    if not m:
        raise ApiError(404, f"model '{name}' not found, try pulling it first")
    if m.get("unsupported"):
        raise ApiError(400, f"{m['name']}: {m['unsupported']}")
    return m


def merged_options(m, body):
    opts = dict(m.get("params") or {})
    opts.update(body.get("options") or {})
    return opts


def build_openai_body(m, body, messages, opts):
    out = {"model": m["name"], "messages": messages, "stream": True,
           "stream_options": {"include_usage": True}}
    for k, v in opts.items():
        if k in OPTION_MAP and v is not None:
            out[OPTION_MAP[k]] = v
    if opts.get("num_predict") not in (None, -1, -2):
        out["max_tokens"] = int(opts["num_predict"])
    fmt = body.get("format")
    if fmt == "json":
        out["response_format"] = {"type": "json_object"}
    elif isinstance(fmt, dict):
        out["response_format"] = {"type": "json_schema", "json_schema": {"name": "schema", "schema": fmt}}
    if body.get("tools"):
        out["tools"] = body["tools"]
    if "think" in body and body["think"] is not None:
        think = body["think"]
        out["chat_template_kwargs"] = {"enable_thinking": bool(think)}
        if isinstance(think, str):
            out["reasoning_effort"] = think
    return out


def convert_messages(m, msgs, system=None):
    out = []
    has_system = any(x.get("role") == "system" for x in msgs)
    default_system = system or m.get("system")
    if default_system and not has_system:
        out.append({"role": "system", "content": default_system})
    for x in msgs:
        role = x.get("role", "user")
        content = x.get("content") or ""
        if x.get("images"):
            parts = [{"type": "text", "text": content}] if content else []
            for img in x["images"]:
                url = img if str(img).startswith("data:") else f"data:image/png;base64,{img}"
                parts.append({"type": "image_url", "image_url": {"url": url}})
            content = parts
        msg = {"role": role, "content": content}
        if x.get("tool_calls"):
            calls = []
            for i, c in enumerate(x["tool_calls"]):
                fn = c.get("function", {})
                args = fn.get("arguments", {})
                calls.append({"id": c.get("id") or f"call_{i}", "type": "function",
                              "function": {"name": fn.get("name"),
                                           "arguments": args if isinstance(args, str) else json.dumps(args)}})
            msg["tool_calls"] = calls
        if role == "tool":
            msg["tool_call_id"] = x.get("tool_call_id") or x.get("tool_name") or "call_0"
        if x.get("thinking") and role == "assistant":
            msg["reasoning_content"] = x["thinking"]
        out.append(msg)
    return out


def stream_engine(port, path, payload):
    """POST to llama-server and yield parsed SSE JSON events."""
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=3600)
    conn.request("POST", path, body=json.dumps(payload), headers={"Content-Type": "application/json"})
    r = conn.getresponse()
    if r.status != 200:
        raw = r.read().decode("utf-8", "replace")
        try:
            msg = json.loads(raw).get("error", {})
            msg = msg.get("message") if isinstance(msg, dict) else msg
        except Exception:
            msg = raw
        conn.close()
        raise ApiError(r.status if r.status < 500 else 500, msg or f"engine error {r.status}")
    try:
        while True:
            line = r.readline()
            if not line:
                break
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                break
            try:
                yield json.loads(data)
            except ValueError:
                continue
    finally:
        conn.close()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "Rock/" + config.VERSION

    def log_message(self, fmt, *args):
        if "/api/ps" in self.path or "/api/status" in self.path:
            return
        log(f"{self.command} {self.path} {args[1] if len(args) > 1 else ''}")

    # ---- helpers
    def _origin_ok(self):
        """Like Ollama: browsers may only call the API from local pages (or ROCK_ORIGINS)."""
        origin = self.headers.get("Origin")
        if not origin:  # curl, Python, other apps: not a web page
            return True
        allowed = [o.strip() for o in os.environ.get("ROCK_ORIGINS", "").split(",") if o.strip()]
        if "*" in allowed or origin in allowed:
            return True
        return bool(re.fullmatch(r"(https?|app|tauri|vscode-webview|chrome-extension|moz-extension)://"
                                 r"(localhost|127\.0\.0\.1|\[::1\]|0\.0\.0\.0)(:\d+)?", origin)
                    or re.fullmatch(r"(chrome-extension|moz-extension|vscode-webview|tauri|app)://.*", origin))

    def _cors(self):
        origin = self.headers.get("Origin")
        self.send_header("Access-Control-Allow-Origin", origin if origin and self._origin_ok() else "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")

    def send_json(self, obj, status=200):
        data = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self._cors()
        self.end_headers()
        self.wfile.write(data)

    def send_error_json(self, status, msg):
        self.send_json({"error": msg}, status)

    def start_stream(self, ctype="application/x-ndjson"):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("Cache-Control", "no-cache")
        self._cors()
        self.end_headers()
        self._streaming = True

    def write_chunk(self, data):
        if isinstance(data, (dict, list)):
            data = json.dumps(data) + "\n"
        if isinstance(data, str):
            data = data.encode()
        self.wfile.write(f"{len(data):X}\r\n".encode() + data + b"\r\n")
        self.wfile.flush()

    def end_stream(self):
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()

    def body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            return {}
        try:
            return json.loads(self.rfile.read(n) or b"{}")
        except ValueError:
            raise ApiError(400, "invalid JSON body")

    # ---- dispatch
    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_HEAD(self):
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        self._route("GET")

    def do_POST(self):
        self._route("POST")

    def do_DELETE(self):
        self._route("DELETE")

    def _route(self, method):
        self._streaming = False
        path = self.path.split("?")[0].rstrip("/") or "/"
        if not self._origin_ok():
            return self.send_error_json(403, "origin not allowed (set ROCK_ORIGINS to allow it)")
        routes = {
            ("GET", "/"): self.ui, ("GET", "/index.html"): self.ui,
            ("GET", "/api/version"): lambda: self.send_json({"version": config.VERSION}),
            ("GET", "/api/tags"): self.tags, ("GET", "/api/ps"): self.ps,
            ("GET", "/api/status"): self.status, ("GET", "/api/catalog"): self.catalog,
            ("POST", "/api/show"): self.show, ("POST", "/api/pull"): self.pull,
            ("DELETE", "/api/delete"): self.delete, ("POST", "/api/delete"): self.delete,
            ("POST", "/api/chat"): self.chat, ("POST", "/api/generate"): self.generate,
            ("POST", "/api/unload"): self.unload, ("POST", "/api/import"): self.import_file,
            ("POST", "/api/import-ollama"): self.import_ollama, ("POST", "/api/setup"): self.engine_setup,
            ("POST", "/api/quit"): self.quit,
            ("POST", "/api/embed"): self.embed, ("POST", "/api/embeddings"): self.embed,
            ("GET", "/v1/models"): self.v1_models,
            ("POST", "/v1/chat/completions"): lambda: self.v1_proxy("/v1/chat/completions"),
            ("POST", "/v1/completions"): lambda: self.v1_proxy("/v1/completions"),
            ("POST", "/v1/embeddings"): lambda: self.v1_proxy("/v1/embeddings"),
        }
        fn = routes.get((method, path))
        if fn is None:
            if method == "GET" and path.startswith("/static/"):
                return self.static(path[len("/static/"):])
            return self.send_error_json(404, "not found")
        try:
            fn()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass
        except Exception as e:
            status = e.status if isinstance(e, ApiError) else 500
            if not isinstance(e, ApiError):
                log(traceback.format_exc())
            msg = str(e)
            try:
                if self._streaming:
                    self.write_chunk({"error": msg})
                    self.end_stream()
                else:
                    self.send_error_json(status, msg)
            except Exception:
                pass

    # ---- UI
    def ui(self):
        data = (WEB_DIR / "index.html").read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)

    def static(self, name):
        p = (WEB_DIR / name).resolve()
        if WEB_DIR.resolve() not in p.parents or not p.is_file():
            return self.send_error_json(404, "not found")
        types = {".js": "text/javascript", ".css": "text/css", ".svg": "image/svg+xml", ".png": "image/png"}
        data = p.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", types.get(p.suffix, "application/octet-stream"))
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    # ---- model management
    def _model_json(self, m):
        meta = {}
        try:
            md = models.read_gguf_metadata(m["path"])
            arch = md.get("general.architecture")
            meta = {"family": arch, "parameter_size": md.get("general.size_label") or "",
                    "quantization_level": str(models.FILE_TYPES.get(md.get("general.file_type"), ""))}
        except Exception:
            pass
        mod = datetime.fromtimestamp(m.get("added") or time.time(), timezone.utc).isoformat()
        return {"name": m["name"], "model": m["name"], "modified_at": mod, "size": m["size"],
                "digest": "", "details": {"format": "gguf", "family": meta.get("family", ""),
                                          "families": [meta["family"]] if meta.get("family") else None,
                                          "parameter_size": meta.get("parameter_size", ""),
                                          "quantization_level": meta.get("quantization_level", "")},
                "source": m.get("source"), "vision": bool(m.get("mmproj")),
                "unsupported": m.get("unsupported")}

    def tags(self):
        self.send_json({"models": [self._model_json(m) for m in models.list_models().values()]})

    def ps(self):
        st = ENGINE.status()
        out = []
        if st:
            m = st["model"]
            exp = st["expires_at"]
            out.append(dict(self._model_json(dict(m, size=models.model_size(m))),
                            expires_at=datetime.fromtimestamp(exp, timezone.utc).isoformat() if exp else None,
                            size_vram=0, context_length=st["ctx"], busy=st["busy"]))
        self.send_json({"models": out})

    def status(self):
        st = ENGINE.status()
        self.send_json({"version": config.VERSION, "engine": engine.engine_info(),
                        "engine_installed": engine.is_installed(), "install": engine.install_status,
                        "loaded": st["model"]["name"] if st else None, "home": str(config.HOME),
                        "models_dir": str(config.MODELS_DIR), "default_ctx": config.DEFAULT_CTX})

    def catalog(self):
        have = {(m.get("source") or "") for m in models.list_models().values()} | set(models.list_models())
        self.send_json({"models": [{"name": k, "repo": v[0], "description": v[1],
                                    "installed": k in have or f"hf.co/{v[0]}" in have}
                                   for k, v in models.CATALOG.items()]})

    def show(self):
        m = get_model(self.body())
        d = models.describe(m)
        params = "\n".join(f"{k} {v}" for k, v in (m.get("params") or {}).items())
        self.send_json({"modelfile": f"FROM {m['path']}", "parameters": params,
                        "template": d.get("chat_template") or "", "system": d.get("system") or "",
                        "details": self._model_json(m)["details"], "model_info": d.get("metadata", {}),
                        "capabilities": ["completion"] + (["vision"] if m.get("mmproj") else []),
                        "rock": {k: v for k, v in d.items() if k not in ("metadata", "chat_template")}})

    def pull(self):
        body = self.body()
        spec = body.get("model") or body.get("name")
        if not spec:
            raise ApiError(400, "model is required")
        stream = body.get("stream", True)
        if not stream:
            models.pull(spec)
            return self.send_json({"status": "success"})
        self.start_stream()
        last = [0.0]

        def progress(status, done, total):
            t = time.time()
            if status.startswith("pulling ") and total and done < total and t - last[0] < 0.3:
                return
            last[0] = t
            msg = {"status": status}
            if total:
                msg.update(total=total, completed=done)
            self.write_chunk(msg)

        models.pull(spec, progress)
        self.end_stream()

    def delete(self):
        body = self.body()
        name = body.get("model") or body.get("name")
        m = models.resolve(name or "")
        if not m:
            raise ApiError(404, f"model '{name}' not found")
        ENGINE.unload(m["path"])
        _, deleted = models.remove(m["name"])
        self.send_json({"status": "success", "files_deleted": deleted})

    def unload(self):
        body = self.body()
        name = body.get("model")
        m = models.resolve(name) if name else None
        ENGINE.unload(m["path"] if m else None)
        self.send_json({"status": "success"})

    def import_file(self):
        body = self.body()
        try:
            m = models.import_file(body.get("path", ""), body.get("name"))
        except FileNotFoundError as e:
            raise ApiError(400, str(e))
        self.send_json({"status": "success", "name": m["name"]})

    def import_ollama(self):
        found = models.import_ollama(self.body().get("path"))
        self.send_json({"status": "success", "imported": [n for n, new in found if new],
                        "already": [n for n, new in found if not new]})

    def quit(self):
        if self.client_address[0] not in ("127.0.0.1", "::1"):
            raise ApiError(403, "quit is only allowed from this computer")
        self.send_json({"status": "stopping"})
        self.quit_server()

    def engine_setup(self):
        body = self.body()
        threading.Thread(target=lambda: _safe(engine.install, body.get("backend"), bool(body.get("force"))),
                         daemon=True).start()
        self.send_json({"status": "started"})

    # ---- inference
    def chat(self):
        body = self.body()
        m = get_model(body)
        msgs = body.get("messages") or []
        self._run_chat(m, body, convert_messages(m, msgs), kind="chat", empty=not msgs)

    def generate(self):
        body = self.body()
        m = get_model(body)
        prompt = body.get("prompt") or ""
        if body.get("raw"):
            return self._raw_generate(m, body)
        msgs = [{"role": "user", "content": prompt, "images": body.get("images")}]
        if body.get("suffix"):
            msgs[0]["content"] = prompt
        self._run_chat(m, body, convert_messages(m, msgs, body.get("system")), kind="generate", empty=not prompt)

    def _run_chat(self, m, body, messages, kind, empty=False):
        opts = merged_options(m, body)
        stream = body.get("stream", True)
        t_start = time.time()
        with ENGINE.use(m, opts.get("num_ctx"), opts.get("num_gpu"), body.get("keep_alive")) as eng:
            if empty:  # Ollama semantics: empty request just loads/unloads the model
                res = {"model": m["name"], "created_at": now_iso(), "done": True, "done_reason": "load"}
                res["message" if kind == "chat" else "response"] = {"role": "assistant", "content": ""} if kind == "chat" else ""
                return self.send_json(res)
            payload = build_openai_body(m, body, messages, opts)
            events = stream_engine(eng["port"], "/v1/chat/completions", payload)
            content, thinking, tool_calls = [], [], {}
            usage, timings, reason = {}, {}, "stop"
            if stream:
                first = next(events, None)  # surface engine errors before sending headers
                self.start_stream()
                evs = [first] if first else []
            else:
                evs = []

            def all_events():
                yield from evs
                yield from events

            for ev in all_events():
                if ev.get("usage"):
                    usage = ev["usage"]
                if ev.get("timings"):
                    timings = ev["timings"]
                for ch in ev.get("choices") or []:
                    d = ch.get("delta") or {}
                    if ch.get("finish_reason"):
                        reason = ch["finish_reason"]
                    piece_c = d.get("content") or ""
                    piece_t = d.get("reasoning_content") or ""
                    for tc in d.get("tool_calls") or []:
                        slot = tool_calls.setdefault(tc.get("index", 0), {"name": "", "arguments": "", "id": None})
                        fn = tc.get("function") or {}
                        slot["name"] += fn.get("name") or ""
                        slot["arguments"] += fn.get("arguments") or ""
                        slot["id"] = slot["id"] or tc.get("id")
                    if piece_c:
                        content.append(piece_c)
                    if piece_t:
                        thinking.append(piece_t)
                    if stream and (piece_c or piece_t):
                        self.write_chunk(self._piece(m, kind, piece_c, piece_t))
            final = self._final(m, kind, reason, usage, timings, t_start, eng["load_seconds"])
            calls = []
            for i in sorted(tool_calls):
                tc = tool_calls[i]
                try:
                    args = json.loads(tc["arguments"] or "{}")
                except ValueError:
                    args = tc["arguments"]
                calls.append({"id": tc["id"], "function": {"index": i, "name": tc["name"], "arguments": args}})
            if stream:
                if calls:
                    piece = self._piece(m, kind, "", "")
                    piece["message"]["tool_calls"] = calls
                    self.write_chunk(piece)
                final.update(self._piece(m, kind, "", ""))
                final["done"] = True
                self.write_chunk(final)
                self.end_stream()
            else:
                res = self._piece(m, kind, "".join(content), "".join(thinking))
                if calls and kind == "chat":
                    res["message"]["tool_calls"] = calls
                res.update(final)
                res["done"] = True
                self.send_json(res)

    def _piece(self, m, kind, content, thinking):
        out = {"model": m["name"], "created_at": now_iso(), "done": False}
        if kind == "chat":
            out["message"] = {"role": "assistant", "content": content}
            if thinking:
                out["message"]["thinking"] = thinking
        else:
            out["response"] = content
            if thinking:
                out["thinking"] = thinking
        return out

    def _final(self, m, kind, reason, usage, timings, t_start, load_s):
        ns = 1_000_000
        res = {"done_reason": {"length": "length", "tool_calls": "stop"}.get(reason, "stop"),
               "total_duration": int((time.time() - t_start) * 1e9),
               "load_duration": int(load_s * 1e9),
               "prompt_eval_count": timings.get("prompt_n", usage.get("prompt_tokens", 0)),
               "prompt_eval_duration": int(timings.get("prompt_ms", 0) * ns),
               "eval_count": timings.get("predicted_n", usage.get("completion_tokens", 0)),
               "eval_duration": int(timings.get("predicted_ms", 0) * ns)}
        if kind == "generate":
            res["context"] = []
        return res

    def _raw_generate(self, m, body):
        opts = merged_options(m, body)
        stream = body.get("stream", True)
        t0 = time.time()
        with ENGINE.use(m, opts.get("num_ctx"), opts.get("num_gpu"), body.get("keep_alive")) as eng:
            payload = {"prompt": body.get("prompt", ""), "stream": True,
                       "n_predict": opts.get("num_predict", -1)}
            for k, v in opts.items():
                if k in OPTION_MAP:
                    payload[OPTION_MAP[k]] = v
            text, timings = [], {}
            if stream:
                self.start_stream()
            for ev in stream_engine(eng["port"], "/completion", payload):
                piece = ev.get("content", "")
                if ev.get("timings"):
                    timings = ev["timings"]
                if piece:
                    text.append(piece)
                    if stream:
                        self.write_chunk(self._piece(m, "generate", piece, ""))
            final = self._final(m, "generate", "stop", {}, timings, t0, eng["load_seconds"])
            res = self._piece(m, "generate", "" if stream else "".join(text), "")
            res.update(final, done=True)
            if stream:
                self.write_chunk(res)
                self.end_stream()
            else:
                self.send_json(res)

    def embed(self):
        body = self.body()
        m = get_model(body)
        inp = body.get("input", body.get("prompt", ""))
        with ENGINE.use(m, keep_alive=body.get("keep_alive")) as eng:
            conn = http.client.HTTPConnection("127.0.0.1", eng["port"], timeout=600)
            conn.request("POST", "/v1/embeddings", body=json.dumps({"input": inp, "model": m["name"]}),
                         headers={"Content-Type": "application/json"})
            r = conn.getresponse()
            data = json.loads(r.read() or b"{}")
            conn.close()
        if r.status != 200:
            raise ApiError(400, (data.get("error") or {}).get("message", "embedding failed. Start the model "
                                                                  "with ROCK_ENGINE_ARGS=--embeddings"))
        vecs = [d["embedding"] for d in data.get("data", [])]
        if self.path.startswith("/api/embeddings"):
            return self.send_json({"embedding": vecs[0] if vecs else []})
        self.send_json({"model": m["name"], "embeddings": vecs})

    # ---- OpenAI-compatible passthrough
    def v1_models(self):
        self.send_json({"object": "list", "data": [
            {"id": n, "object": "model", "created": int(m.get("added") or 0), "owned_by": "local"}
            for n, m in models.list_models().items()]})

    def v1_proxy(self, path):
        body = self.body()
        m = get_model(body)
        body["model"] = m["name"]
        with ENGINE.use(m) as eng:
            conn = http.client.HTTPConnection("127.0.0.1", eng["port"], timeout=3600)
            conn.request("POST", path, body=json.dumps(body), headers={"Content-Type": "application/json"})
            r = conn.getresponse()
            if body.get("stream"):
                self.send_response(r.status)
                self.send_header("Content-Type", r.getheader("Content-Type", "text/event-stream"))
                self.send_header("Transfer-Encoding", "chunked")
                self._cors()
                self.end_headers()
                self._streaming = True
                while True:
                    line = r.readline()
                    if not line:
                        break
                    self.write_chunk(line)
                self.end_stream()
            else:
                data = r.read()
                self.send_response(r.status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self._cors()
                self.end_headers()
                self.wfile.write(data)
            conn.close()


def _safe(fn, *a):
    try:
        fn(*a)
    except Exception as e:
        log(f"Error: {e}")


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False


QUIT_HOOK = None  # set by the desktop app so /api/quit also removes the tray icon


def make_server():
    try:
        return Server((config.HOST, config.PORT), Handler)
    except OSError as e:
        raise SystemExit(f"Could not listen on {config.HOST}:{config.PORT} ({e}). Is Rock already running?")


def serve(httpd=None):
    httpd = httpd or make_server()

    def quit_server():
        if QUIT_HOOK:
            QUIT_HOOK()
        threading.Thread(target=httpd.shutdown, daemon=True).start()
    Handler.quit_server = staticmethod(quit_server)
    log(f"Rock {config.VERSION} listening on {config.base_url()}  (web UI: {config.base_url()}/)")
    log(f"Models folder: {config.MODELS_DIR}")
    if not engine.is_installed():
        log("Engine not installed - downloading llama.cpp in the background...")
        threading.Thread(target=_safe, args=(engine.install,), daemon=True).start()
    else:
        info = engine.engine_info() or {}
        log(f"Engine: llama.cpp {info.get('tag', '?')} ({info.get('backend', '?')})")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        ENGINE.shutdown()
        httpd.server_close()
