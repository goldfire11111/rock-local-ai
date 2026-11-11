"""Model library: registry of local GGUF files, Hugging Face pulls, Ollama import, GGUF metadata."""
import json
import os
import re
import struct
import threading
import time
from pathlib import Path

from . import config
from .util import download, http_json

_lock = threading.RLock()

# Short names -> Hugging Face repo:quant (like `ollama pull llama3.2`)
CATALOG = {
    "qwen3.5:4b":       ("unsloth/Qwen3.5-4B-GGUF:Q4_K_M", "Qwen 3.5 4B - strong all-rounder"),
    "qwen3.5:9b":       ("unsloth/Qwen3.5-9B-GGUF:Q4_K_M", "Qwen 3.5 9B - smarter, ~5.4 GB"),
    "qwen3:0.6b":       ("unsloth/Qwen3-0.6B-GGUF:Q4_K_M", "Tiny & fast, good for testing"),
    "qwen3:1.7b":       ("unsloth/Qwen3-1.7B-GGUF:Q4_K_M", "Small thinking model"),
    "qwen3:4b":         ("unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_K_M", "Qwen3 4B Instruct 2507"),
    "qwen3:8b":         ("unsloth/Qwen3-8B-GGUF:Q4_K_M", "Qwen3 8B thinking model"),
    "qwen3-vl:4b":      ("unsloth/Qwen3-VL-4B-Instruct-GGUF:Q4_K_M", "Vision: understands images"),
    "gemma4:e2b":       ("unsloth/gemma-4-E2B-it-GGUF:Q4_K_M", "Google Gemma 4 E2B"),
    "gemma4:e4b":       ("unsloth/gemma-4-E4B-it-GGUF:Q4_K_M", "Google Gemma 4 E4B"),
    "gemma3:1b":        ("unsloth/gemma-3-1b-it-GGUF:Q4_K_M", "Google Gemma 3 1B"),
    "gemma3:4b":        ("unsloth/gemma-3-4b-it-GGUF:Q4_K_M", "Google Gemma 3 4B (vision)"),
    "gemma3:12b":       ("unsloth/gemma-3-12b-it-GGUF:Q4_K_M", "Google Gemma 3 12B, ~7 GB"),
    "llama3.2:1b":      ("bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_K_M", "Meta Llama 3.2 1B"),
    "llama3.2:3b":      ("bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M", "Meta Llama 3.2 3B"),
    "llama3.1:8b":      ("bartowski/Meta-Llama-3.1-8B-Instruct-GGUF:Q4_K_M", "Meta Llama 3.1 8B"),
    "mistral:7b":       ("bartowski/Mistral-7B-Instruct-v0.3-GGUF:Q4_K_M", "Mistral 7B v0.3"),
    "phi4-mini":        ("unsloth/Phi-4-mini-instruct-GGUF:Q4_K_M", "Microsoft Phi-4 mini"),
    "qwen2.5-coder:7b": ("bartowski/Qwen2.5-Coder-7B-Instruct-GGUF:Q4_K_M", "Coding assistant"),
    "deepseek-r1:8b":   ("unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF:Q4_K_M", "Reasoning model"),
    "smollm3:3b":       ("ggml-org/SmolLM3-3B-GGUF:Q4_K_M", "HF SmolLM3 3B"),
    "smollm2:360m":     ("bartowski/SmolLM2-360M-Instruct-GGUF:Q4_K_M", "Very tiny (258 MB)"),
    "gpt-oss:20b":      ("ggml-org/gpt-oss-20b-GGUF:MXFP4", "OpenAI gpt-oss 20B, ~11.5 GB"),
}

QUANT_PREFERENCE = ["Q4_K_M", "Q4_K_S", "Q4_K_XL", "Q5_K_M", "Q4_0", "MXFP4", "Q6_K", "Q8_0", "Q3_K_M", "IQ4_XS"]
SPLIT_RE = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$", re.I)


# --------------------------------------------------------------------------- registry

def _load():
    try:
        return json.loads(config.REGISTRY_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {"models": {}}


def _save(reg):
    tmp = config.REGISTRY_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(reg, indent=2), encoding="utf-8")
    tmp.replace(config.REGISTRY_FILE)


def normalize(name):
    name = name.strip()
    for p in ("hf.co/", "huggingface.co/", "https://huggingface.co/"):
        if name.lower().startswith(p):
            name = name[len(p):]
    return name


def _is_main_gguf(p):
    n = p.name.lower()
    if not n.endswith(".gguf") or "mmproj" in n:
        return False
    m = SPLIT_RE.search(n)
    return not m or int(m.group(1)) == 1


def list_models():
    """All usable models: registered ones + any .gguf dropped into the models folder."""
    with _lock:
        reg = _load()
        out = {}
        known_paths = set()
        for name, m in reg["models"].items():
            p = Path(m["path"])
            if p.exists():
                m = dict(m, name=name)
                out[name] = m
                known_paths.add(str(p.resolve()).lower())
        for p in config.MODELS_DIR.rglob("*.gguf"):
            if not _is_main_gguf(p) or str(p.resolve()).lower() in known_paths:
                continue
            name = SPLIT_RE.sub("", p.stem).lower()
            if name in out:
                continue
            mm = _find_mmproj(p.parent)
            out[name] = {"name": name, "path": str(p), "source": "folder", "added": p.stat().st_mtime,
                         "mmproj": str(mm) if mm else None}
        for m in out.values():
            m["size"] = model_size(m)
            m["unsupported"] = compat_problem(m["path"])
        return dict(sorted(out.items(), key=lambda kv: (bool(kv[1]["unsupported"]), kv[0])))


def model_size(m):
    p = Path(m["path"])
    total = 0
    m2 = SPLIT_RE.search(p.name)
    files = sorted(p.parent.glob(SPLIT_RE.sub("", p.name) + "-*-of-*.gguf")) if m2 else [p]
    for f in files:
        try:
            total += f.stat().st_size
        except OSError:
            pass
    return total


def _find_mmproj(folder):
    c = sorted(folder.glob("*mmproj*.gguf"))
    return c[0] if c else None


def resolve(name):
    """Find a model by name, tolerant of ':latest', case and unique prefixes."""
    if not name:
        return None
    models = list_models()
    n = normalize(name).lower()
    if n in models:
        return models[n]
    for k in (n.removesuffix(":latest"), n + ":latest"):
        if k in models:
            return models[k]
    lower = {k.lower(): v for k, v in models.items()}
    if n in lower:
        return lower[n]
    # catalog alias or hf spec that was pulled under a derived name
    for m in models.values():
        if (m.get("source") or "").lower() in (n, f"hf.co/{n}"):
            return m
    starts = [v for k, v in lower.items() if k.startswith(n)]
    return starts[0] if len(starts) == 1 else None


def register(name, path, **extra):
    with _lock:
        reg = _load()
        entry = {"path": str(path), "added": time.time()}
        entry.update({k: v for k, v in extra.items() if v is not None})
        reg["models"][name] = entry
        _save(reg)
        return dict(entry, name=name)


def update_params(name, params=None, system=None):
    with _lock:
        reg = _load()
        m = resolve(name)
        if not m:
            raise KeyError(name)
        if m["name"] not in reg["models"]:
            reg["models"][m["name"]] = {"path": m["path"], "added": time.time(), "source": "folder"}
        e = reg["models"][m["name"]]
        if params is not None:
            e["params"] = params
        if system is not None:
            e["system"] = system
        _save(reg)


def remove(name):
    """Delete a model. Files we downloaded are deleted; imported files are only unregistered."""
    with _lock:
        m = resolve(name)
        if not m:
            raise KeyError(name)
        reg = _load()
        reg["models"].pop(m["name"], None)
        _save(reg)
        p = Path(m["path"])
        inside = str(p.resolve()).lower().startswith(str(config.MODELS_DIR.resolve()).lower())
        if inside:
            m2 = SPLIT_RE.search(p.name)
            files = list(p.parent.glob(SPLIT_RE.sub("", p.name) + "-*-of-*.gguf")) if m2 else [p]
            for f in files:
                f.unlink(missing_ok=True)
            still_used = any(Path(x["path"]).parent == p.parent for x in list_models().values())
            if not still_used:
                for f in p.parent.glob("*mmproj*.gguf"):
                    f.unlink(missing_ok=True)
                try:
                    if p.parent != config.MODELS_DIR:
                        p.parent.rmdir()
                except OSError:
                    pass
        return m, inside


def import_file(path, name=None):
    p = Path(path).expanduser().resolve()
    if not p.exists() or p.suffix.lower() != ".gguf":
        raise FileNotFoundError(f"Not a .gguf file: {p}")
    name = (name or SPLIT_RE.sub("", p.stem)).lower()
    mm = _find_mmproj(p.parent)
    return register(name, p, source="import", mmproj=str(mm) if mm and mm != p else None)


def import_ollama(models_dir=None):
    """Register models already downloaded by Ollama (its blobs are plain GGUF files). No copying."""
    root = Path(models_dir or os.environ.get("OLLAMA_MODELS") or Path.home() / ".ollama" / "models")
    man_root = root / "manifests"
    found = []
    if not man_root.exists():
        return found
    existing = list_models()
    for mf in man_root.rglob("*"):
        if not mf.is_file():
            continue
        try:
            manifest = json.loads(mf.read_text(encoding="utf-8"))
        except Exception:
            continue
        parts = mf.relative_to(man_root).parts  # host/namespace/model/tag
        if len(parts) < 3:
            continue
        model, tag = parts[-2], parts[-1]
        name = f"{model}:{tag}".lower().removesuffix(":latest")
        info = {"mmproj": None, "params": None, "system": None}
        blob = None
        for layer in manifest.get("layers", []):
            bp = root / "blobs" / layer["digest"].replace(":", "-")
            mt = layer.get("mediaType", "")
            if mt.endswith(".model"):
                blob = bp
            elif mt.endswith(".projector"):
                info["mmproj"] = str(bp)
            elif mt.endswith(".params"):
                try:
                    info["params"] = json.loads(bp.read_text(encoding="utf-8"))
                except Exception:
                    pass
            elif mt.endswith(".system"):
                try:
                    info["system"] = bp.read_text(encoding="utf-8")
                except Exception:
                    pass
            elif mt.endswith(".template"):
                try:
                    info["ollama_template"] = bp.read_text(encoding="utf-8")
                except Exception:
                    pass
        if not blob or not blob.exists():
            continue
        tmpl = info.pop("ollama_template", None)
        if tmpl:
            try:
                has_tmpl = "tokenizer.chat_template" in read_gguf_metadata(blob)
            except Exception:
                has_tmpl = True
            jinja = None if has_tmpl else ollama_template_to_jinja(tmpl)
            if jinja:
                tdir = config.HOME / "templates"
                tdir.mkdir(exist_ok=True)
                tf = tdir / (re.sub(r"[^\w.-]", "_", name) + ".jinja")
                tf.write_text(jinja, encoding="utf-8")
                info["chat_template_file"] = str(tf)
        already = name in existing and existing[name]["path"] == str(blob)
        if name in existing and not already:
            name = "ollama-" + name  # don't clobber a different model with the same name
            already = name in existing
        prev = existing.get(name, {})
        register(name, blob, source="ollama:" + "/".join(parts), **info,
                 **{k: prev[k] for k in ("added",) if k in prev})
        found.append((name, not already))
    return found


def ollama_template_to_jinja(tmpl):
    """Convert a simple (legacy) Ollama Go template using .System/.Prompt/.Response into a Jinja
    chat template llama.cpp can use. Returns None for templates we can't translate (e.g. range loops)."""
    if re.search(r"\{\{-?\s*(range|with|else|\.Messages|\.Tools|template|block)", tmpl):
        return None
    tokens = re.split(r"(\{\{-?.*?-?\}\})", tmpl)
    out, depth, head_cut = [], 0, None
    for tok in tokens:
        m = re.fullmatch(r"\{\{(-?)\s*(.*?)\s*(-?)\}\}", tok, re.S)
        if not m:
            out.append(tok.replace("{", "{{ '{' }}") if "{" in tok else tok)
            continue
        lt, expr, rt = m.group(1), m.group(2), m.group(3)
        var = {".System": "system", ".Prompt": "prompt", ".Response": "response"}
        if expr in var:
            if expr == ".Response" and head_cut is None:
                head_cut = (len(out), depth)
            out.append("{{%s %s %s}}" % (lt, var[expr], rt))
        elif expr.startswith("if "):
            cond = expr[3:].strip()
            if cond not in var:
                return None
            depth += 1
            out.append("{%%%s if %s %s%%}" % (lt, var[cond], rt))
        elif expr == "end":
            depth -= 1
            out.append("{%%%s endif %s%%}" % (lt, rt))
        else:
            return None
    if depth != 0:
        return None
    if head_cut is None:  # Ollama appends the response after the template
        head_cut = (len(out), 0)
        out.append("{{ response }}")
    full = "".join(out)
    head = "".join(out[:head_cut[0]]) + "{% endif %}" * head_cut[1]
    return (
        "{%- set ns = namespace(sys='', first=true) -%}"
        "{%- for m in messages -%}{%- if m.role == 'system' -%}{%- set ns.sys = m.content -%}{%- endif -%}{%- endfor -%}"
        "{%- for m in messages -%}"
        "{%- if m.role == 'user' -%}"
        "{%- set system = ns.sys if ns.first else '' -%}{%- set ns.first = false -%}"
        "{%- set prompt = m.content -%}"
        "{%- if not loop.last and messages[loop.index0 + 1].role == 'assistant' -%}"
        "{%- set response = messages[loop.index0 + 1].content %}" + full +
        "{% else -%}{%- set response = '' %}" + head +
        "{% endif -%}"
        "{%- endif -%}"
        "{%- endfor -%}"
    )


# --------------------------------------------------------------------------- pulling

def parse_spec(spec):
    """'qwen3:4b' | 'user/repo' | 'user/repo:Q4_K_M' | 'hf.co/user/repo:quant' | https URL."""
    spec = spec.strip()
    if spec.lower() in CATALOG:
        repo_q = CATALOG[spec.lower()][0]
        repo, _, quant = repo_q.partition(":")
        return {"kind": "hf", "repo": repo, "quant": quant or None, "name": spec.lower()}
    if spec.startswith(("http://", "https://")) and spec.lower().split("?")[0].endswith(".gguf"):
        fname = spec.split("?")[0].rsplit("/", 1)[-1]
        return {"kind": "url", "url": spec, "name": SPLIT_RE.sub("", fname[:-5]).lower()}
    s = normalize(spec)
    if "/" not in s:
        raise ValueError(f"Unknown model '{spec}'. Use a catalog name (see `rock catalog`) "
                         "or a Hugging Face repo like 'unsloth/Qwen3-4B-GGUF:Q4_K_M'.")
    repo, _, quant = s.partition(":")
    if quant.lower() == "latest":
        quant = ""
    base = repo.split("/")[-1]
    base = re.sub(r"[-_.]?gguf$", "", base, flags=re.I).lower()
    name = f"{base}:{quant.lower()}" if quant else base
    return {"kind": "hf", "repo": repo, "quant": quant or None, "name": name}


def _choose_files(files, quant):
    ggufs = [f for f in files if f["path"].lower().endswith(".gguf")]
    mmproj = [f for f in ggufs if "mmproj" in f["path"].lower()]
    weights = [f for f in ggufs if "mmproj" not in f["path"].lower()]
    if not weights:
        raise RuntimeError("No .gguf files found in that repository.")
    groups = {}
    for f in weights:
        key = SPLIT_RE.sub("", f["path"])
        key = key[:-5] if key.lower().endswith(".gguf") else key
        groups.setdefault(key, []).append(f)

    def tokens(k):
        return set(re.split(r"[-_./]", k.upper()))

    def matches(k, q):
        q = q.upper()
        return q in tokens(k) or k.upper().endswith(q) or ("-" + q) in k.upper() or ("/" + q) in k.upper()

    chosen = None
    if quant:
        hits = [k for k in groups if matches(k, quant)]
        if not hits:
            avail = ", ".join(sorted({re.split(r"[-.]", k.split("/")[-1])[-1] for k in groups}))
            raise RuntimeError(f"Quantization '{quant}' not found. Available: {avail}")
        chosen = min(hits, key=len)
    else:
        for q in QUANT_PREFERENCE:
            hits = [k for k in groups if matches(k, q)]
            if hits:
                chosen = min(hits, key=len)
                break
        if not chosen:
            chosen = min(groups, key=lambda k: sum(f.get("size", 0) for f in groups[k]))
    parts = sorted(groups[chosen], key=lambda f: f["path"])
    proj = None
    if mmproj:
        pref = [f for f in mmproj if "f16" in f["path"].lower()] or mmproj
        proj = min(pref, key=lambda f: f.get("size", 0))
    return parts, proj


def pull(spec, progress=None):
    """Download a model. progress(status:str, completed:int, total:int)."""
    progress = progress or (lambda *a: None)
    info = parse_spec(spec)
    progress("pulling manifest", 0, 0)
    if info["kind"] == "url":
        fname = info["url"].split("?")[0].rsplit("/", 1)[-1]
        dest = config.MODELS_DIR / "downloads" / fname
        download(info["url"], dest, lambda d, t: progress(f"pulling {fname}", d, t))
        entry = register(info["name"], dest, source=info["url"])
        progress("success", 0, 0)
        return entry
    repo = info["repo"]
    try:
        files = http_json(f"https://huggingface.co/api/models/{repo}/tree/main?recursive=true")
    except Exception as e:
        raise RuntimeError(f"Could not find '{repo}' on Hugging Face ({e}).") from e
    parts, proj = _choose_files(files, info["quant"])
    folder = config.MODELS_DIR / repo.replace("/", "--")
    total = sum(f.get("size", 0) for f in parts) + (proj.get("size", 0) if proj else 0)
    done_before = 0
    paths = []
    for f in parts + ([proj] if proj else []):
        url = f"https://huggingface.co/{repo}/resolve/main/{f['path']}?download=true"
        dest = folder / Path(f["path"]).name
        label = Path(f["path"]).name
        base = done_before
        download(url, dest, lambda d, t, b=base, l=label: progress(f"pulling {l}", b + d, total or t),
                 expected_size=f.get("size"))
        done_before += f.get("size", 0)
        paths.append(dest)
    progress("verifying", total, total)
    main = paths[0]
    entry = register(info["name"], main, source=f"hf.co/{repo}" + (f":{info['quant']}" if info["quant"] else ""),
                     mmproj=str(paths[-1]) if proj else None)
    progress("success", total, total)
    return entry


# --------------------------------------------------------------------------- GGUF metadata

_GGUF_TYPES = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
FILE_TYPES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1", 10: "Q2_K",
              11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S", 15: "Q4_K_M", 16: "Q5_K_S",
              17: "Q5_K_M", 18: "Q6_K", 19: "IQ2_XXS", 20: "IQ2_XS", 21: "Q2_K_S", 22: "IQ3_XS",
              23: "IQ3_XXS", 24: "IQ1_S", 25: "IQ4_NL", 26: "IQ3_S", 27: "IQ3_M", 28: "IQ2_S",
              29: "IQ2_M", 30: "IQ4_XS", 31: "IQ1_M", 32: "BF16", 36: "TQ1_0", 37: "TQ2_0", 38: "MXFP4"}


_compat_cache = {}


def compat_problem(path):
    """Return a reason string if llama.cpp can't load this file, else None.
    Ollama's own library packs vision/audio encoders (tensors 'v.*', 'a.*', 'mm.*') into the main
    model file; llama.cpp expects those in a separate mmproj file and refuses to load it."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    key = (str(path), st.st_size, st.st_mtime)
    if key not in _compat_cache:
        reason = None
        try:
            names = read_gguf_metadata(path, tensor_names=True)["_tensor_names"]
            if any(n.startswith(("v.", "a.", "mm.")) for n in names):
                reason = ("Packaged in Ollama's own format (vision/audio built in), which llama.cpp can't load. "
                          "Download the standard GGUF instead (see the catalog).")
        except Exception:
            pass
        _compat_cache[key] = reason
    return _compat_cache[key]


def read_gguf_metadata(path, keep_arrays=False, tensor_names=False):
    """Parse the key/value header of a GGUF file (skips big arrays like the vocabulary)."""
    meta = {}
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError("Not a GGUF file")
        version, = struct.unpack("<I", f.read(4))
        if version == 1:
            raise ValueError("GGUF v1 is not supported")
        n_tensors, n_kv = struct.unpack("<QQ", f.read(16))

        def rstr():
            n, = struct.unpack("<Q", f.read(8))
            return f.read(n).decode("utf-8", "replace")

        def rval(t):
            if t == 8:
                return rstr()
            if t == 9:
                et, n = struct.unpack("<IQ", f.read(12))
                if et == 8:
                    if keep_arrays or n <= 64:
                        return [rstr() for _ in range(n)]
                    for _ in range(n):
                        k, = struct.unpack("<Q", f.read(8))
                        f.seek(k, 1)
                    return f"<{n} strings>"
                fmt = _GGUF_TYPES[et]
                size = struct.calcsize(fmt)
                if n <= 64:
                    return list(struct.unpack("<" + fmt * n, f.read(size * n)))
                f.seek(size * n, 1)
                return f"<{n} values>"
            fmt = _GGUF_TYPES[t]
            return struct.unpack("<" + fmt, f.read(struct.calcsize(fmt)))[0]

        for _ in range(n_kv):
            key = rstr()
            t, = struct.unpack("<I", f.read(4))
            meta[key] = rval(t)
        if tensor_names:
            names = []
            for _ in range(n_tensors):
                names.append(rstr())
                nd, = struct.unpack("<I", f.read(4))
                f.seek(8 * nd + 4 + 8, 1)  # dims, type, offset
            meta["_tensor_names"] = names
    meta["_tensor_count"] = n_tensors
    return meta


def describe(m):
    """Human-friendly details for `show`."""
    out = {"name": m["name"], "path": m["path"], "size": m.get("size") or model_size(m),
           "source": m.get("source"), "mmproj": m.get("mmproj"), "params": m.get("params") or {},
           "system": m.get("system")}
    try:
        meta = read_gguf_metadata(m["path"])
    except Exception as e:
        out["error"] = str(e)
        return out
    arch = meta.get("general.architecture", "?")
    out.update({
        "architecture": arch,
        "model_name": meta.get("general.name"),
        "parameters": meta.get("general.size_label") or _guess_params(out["size"], meta),
        "context_length": meta.get(f"{arch}.context_length"),
        "embedding_length": meta.get(f"{arch}.embedding_length"),
        "quantization": FILE_TYPES.get(meta.get("general.file_type"), meta.get("general.file_type")),
        "has_chat_template": "tokenizer.chat_template" in meta,
        "license": meta.get("general.license"),
        "chat_template": meta.get("tokenizer.chat_template"),
        "metadata": {k: v for k, v in meta.items() if k != "tokenizer.chat_template"},
    })
    return out


def _guess_params(size, meta):
    return None
