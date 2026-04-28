# AI File Brain

A local-first desktop app that watches a folder, indexes `.txt` and `.pdf` files into a local vector store, and answers natural-language questions about them — all without leaving your machine.

## Stack

- Python 3.12
- PySide6 (Qt 6) for the desktop UI
- ChromaDB `PersistentClient` for the vector store (in-process, no server)
- Ollama for embeddings (`nomic-embed-text`) and chat (`llama3.2`)
- watchdog for file events
- PyInstaller for the single-folder build

## Prerequisites

1. **Python 3.12** on PATH.
2. **[Ollama](https://ollama.com)** running locally. Pull the models once:
   ```
   ollama pull nomic-embed-text
   ollama pull llama3.2
   ```
3. **[uv](https://docs.astral.sh/uv/)** (recommended) or pip.

## Set up

```bash
uv venv
uv pip install -e ".[dev]"
```

## Configure

Edit `settings.toml` (or set `AFB_*` env vars). The default watch folder is `C:/Users/ASUS/Documents/AIFileBrainTest` — create it or change it.

## Run

```bash
uv run ai-file-brain
# or
uv run python -m ai_file_brain.app.main
```

The app opens a chat window and parks itself in the system tray. Closing the window hides it; quit from the tray menu.

## Test

```bash
uv run pytest
```

## Build a standalone folder (single .exe + deps)

```bash
uv pip install -e ".[build]"
uv run pyinstaller pyinstaller.spec
```

Output lands in `dist/ai-file-brain/` — ship the whole folder.
