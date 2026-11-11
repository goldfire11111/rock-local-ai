# Rock: run local AI models (GGUF) on your laptop

Rock works like Ollama. It downloads and runs GGUF models on your own machine, with its own chat window, a system-tray icon, an `ollama`-style command line, and an Ollama/OpenAI-compatible API. Under the hood it uses **llama.cpp**, the same engine Ollama is built on, and installs the newest official build for your GPU automatically.

Everything ships as one file: **`Rock.exe`** (about 16 MB, no Python needed).

## Download

**[⬇ Download Rock.exe](https://github.com/goldfire11111/rock-local-ai/releases/latest/download/Rock.exe)** (Windows 10/11, 64-bit). All versions are on the [Releases page](https://github.com/goldfire11111/rock-local-ai/releases).

1. Download `Rock.exe` and double-click it. Nothing to install.
2. Windows may show **"Windows protected your PC"** because the app isn't code-signed. Click **More info → Run anyway**.
3. On first launch Rock downloads the AI engine (llama.cpp) for your hardware. Then open **Models** and download a model. `qwen3:4b` or `gemma4:e2b` are good first picks for most laptops.

**What you need:** Windows 10 or 11 (64-bit), an internet connection for the first setup and model downloads, and at least 8 GB of RAM (16 GB recommended). Any GPU helps but isn't required:

| Your hardware | Engine Rock installs |
|---|---|
| NVIDIA GPU | CUDA (fastest) |
| AMD or Intel GPU | Vulkan |
| No GPU | CPU (works, slower; stick to small models) |

**Choosing a model size:** a model needs roughly its download size in free memory. With 8 GB of RAM, stay under about 3 GB (`qwen3:1.7b`, `llama3.2:3b`, `gemma4:e2b`). With 16 GB, models up to about 6–8 GB are fine.

## Quick start

**Double-click `Rock.exe`.**

- The chat opens in its own app window. It uses Microsoft Edge's app mode, so there are no browser tabs or address bar.
- A **rock icon appears in the system tray**, next to the clock (it may be under the ^ arrow). Rock keeps running there after you close the window, like Ollama. Right-click the icon to **Open Rock**, see which model is loaded, **Unload model**, open the models or logs folder, or **Quit Rock**.
- On first launch on a new computer, Rock downloads the llama.cpp engine in the background (about 500 MB for NVIDIA GPUs). The status line in the window shows the progress.

In the window:

- **Models → Download**: pick something from the catalog, or paste any Hugging Face GGUF repo.
- **Models → Import my Ollama models**: reuses the models you already downloaded with Ollama, without copying them. Models Rock can't load are flagged "can't load".
- **Settings**: system prompt, temperature, context length, thinking on/off, keep-alive.

### Where things are stored

The engine, models, logs and settings go in **`%LOCALAPPDATA%\Rock`**, so you can keep `Rock.exe` anywhere, even in Downloads. Models can be large; to keep them on another drive, set the `ROCK_HOME` or `ROCK_MODELS` environment variable.

**Portable mode:** if a folder named `data` sits next to `Rock.exe`, Rock uses that folder instead. This is handy on a USB stick.

## Command line

The same `Rock.exe` is also the command-line tool. Run it from a terminal in this folder (`.\Rock.exe list`), or add the folder to your PATH and just type `rock ...`. Commands that need a model start Rock in the tray automatically if it isn't running.

```
rock run qwen3:4b                 chat in the terminal (downloads the model if needed)
rock run qwen3:4b "hi" --verbose  one-shot answer, with tokens/sec
rock pull llama3.2:3b             download a catalog model
rock pull unsloth/Qwen3-8B-GGUF:Q5_K_M   any Hugging Face GGUF repo (optional :QUANT)
rock pull https://.../model.gguf  direct URL
rock list                         installed models
rock ps                           the model currently loaded in memory
rock stop                         unload it and free GPU memory
rock rm <model>                   delete a model
rock show <model> [--template]    architecture, context size, quantization, ...
rock import D:\path\model.gguf    add a GGUF file from anywhere (not copied)
rock import-ollama                register your Ollama models
rock catalog                      list model shortcuts
rock ui                           open the chat window
rock app [--no-window]            start the desktop app (tray icon); what double-clicking does
rock quit                         stop Rock completely (unloads models, removes the tray icon)
rock serve                        run the server in the foreground, with logs, and no tray icon
rock setup [--backend cuda|vulkan|cpu]   install or update the llama.cpp engine
```

In the terminal chat you can use `/set system ...`, `/set think on|off`, `/set ctx 16384`, `/clear`, `/show` and `/bye`. Wrap text in `"""` to send a multi-line message.

You can also drop `.gguf` files into `data\models\`. Rock finds them automatically, along with any `*mmproj*.gguf` file in the same folder, which enables image input.

## API (for other apps)

The server runs at `http://127.0.0.1:11435`:

| Ollama-compatible | OpenAI-compatible |
|---|---|
| `POST /api/chat`, `/api/generate`, `/api/pull`, `/api/show`, `/api/embed` | `POST /v1/chat/completions`, `/v1/completions`, `/v1/embeddings` |
| `GET /api/tags`, `/api/ps`, `/api/version`, `DELETE /api/delete` | `GET /v1/models` |

To point an Ollama client at Rock, set `OLLAMA_HOST=127.0.0.1:11435`. For OpenAI SDKs, use `base_url="http://127.0.0.1:11435/v1"` with any API key.

Like Ollama, Rock only accepts requests from **web pages** served from this computer (localhost), so a random website can't use or delete your models. Programs such as scripts, apps and curl aren't affected. To allow a specific site, set `ROCK_ORIGINS=https://example.com`.

## How it works

- `rockllm/app.py`: the desktop app: tray icon (pystray), the app window (Edge or Chrome `--app` mode), and single-instance handling.
- `rockllm/engine.py`: downloads llama.cpp's `llama-server` from the official GitHub releases (CUDA for NVIDIA, Vulkan for other GPUs, otherwise CPU). It runs one model at a time and swaps models on demand, the way Ollama does. Idle models are unloaded after 5 minutes. llama.cpp's `--fit` automatically splits models that are too big for VRAM between GPU and RAM. The engine process is tied to Rock, so it always closes when Rock does.
- `rockllm/models.py`: the model registry, Hugging Face downloads (resumable), Ollama import (including converting simple Ollama chat templates), and a GGUF metadata reader.
- `rockllm/server.py`: the HTTP API that translates between Ollama/OpenAI requests and llama.cpp. It also serves the web UI.
- `rockllm/web/index.html`: the chat UI. Chats are saved in the app window's own profile (`data\window`).
- `data/`: the engine, models, logs (`data/logs/engine.log` is useful when a model fails to load) and `models.json`.

## Settings (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `ROCK_HOST` | `127.0.0.1:11435` | Server address. Use `0.0.0.0:11435` to allow other devices on your network. |
| `ROCK_HOME` | `.\data` | Where the engine, models and logs live |
| `ROCK_MODELS` | `ROCK_HOME\models` | Models folder |
| `ROCK_NUM_CTX` | `8192` | Default context length |
| `ROCK_KEEP_ALIVE` | `300` | Seconds of idle time before a model is unloaded |
| `ROCK_BACKEND` | `auto` | `cuda`, `vulkan` or `cpu` |
| `ROCK_ENGINE_ARGS` | | Extra `llama-server` flags, e.g. `--embeddings` or `-ngl 20` |
| `ROCK_ORIGINS` | | Extra web origins allowed to call the API (comma-separated, `*` = any) |
| `HF_TOKEN` | | For gated Hugging Face models |

## Notes

- Some models from **Ollama's own library** (for example `gemma4` from ollama.com) are packaged in a format only Ollama's engine can read. Rock detects these, marks them "can't load", and won't select them. Pull the normal GGUF instead, e.g. `rock pull gemma4:e4b`. Models Ollama pulled from Hugging Face (`hf.co/...`) and older library models work fine.
- The first launch of a newly built `Rock.exe` can take a few seconds while Windows Defender scans it. Antivirus programs sometimes wrongly flag single-file Python apps; if that happens, build it yourself (below) or add an exclusion.

## Building Rock.exe from source

```
powershell -ExecutionPolicy Bypass -File build.ps1
```

This creates `build\venv`, installs PyInstaller, pystray and Pillow into it, draws the icon, and writes `Rock.exe` into this folder. You can also run from source without building: `python rock.py ...` uses only the standard library. The tray icon needs `pip install pystray pillow`; without them Rock runs with no tray icon.
