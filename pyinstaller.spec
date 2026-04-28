# -*- mode: python ; coding: utf-8 -*-
# Run: `pyinstaller pyinstaller.spec`
#
# Output: dist/ai-file-brain/  (folder; ai-file-brain.exe is the entry point)

import os

from PyInstaller.utils.hooks import collect_submodules, collect_data_files

block_cipher = None
ICON_PATH = "src/ai_file_brain/app/assets/tray_icon.ico"
ICON_ARG = ICON_PATH if os.path.exists(ICON_PATH) else None

hidden_imports = []
hidden_imports += collect_submodules("chromadb")
hidden_imports += collect_submodules("chromadb.telemetry")
hidden_imports += collect_submodules("chromadb.api")
hidden_imports += collect_submodules("chromadb.db")
hidden_imports += collect_submodules("chromadb.segment")
hidden_imports += collect_submodules("chromadb.utils")
hidden_imports += collect_submodules("PySide6")
hidden_imports += [
    "qasync",
    "ollama",
    "pypdf",
    "watchdog.observers",
    "watchdog.events",
    "pydantic_settings",
    "pydantic_settings.sources",
]

datas = []
datas += collect_data_files("chromadb")
datas += [("settings.toml", "."),
          ("src/ai_file_brain/app/assets", "ai_file_brain/app/assets")]

excludes = [
    # Chroma's optional default embedder; we provide our own via Ollama.
    "onnxruntime",
    # Test-only deps shouldn't ship.
    "pytest",
    "pytest_asyncio",
    "pytest_qt",
]


a = Analysis(
    ["src/ai_file_brain/app/main.py"],
    pathex=["src"],
    binaries=[],
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="ai-file-brain",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,            # GUI app — no console window
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=ICON_ARG,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="ai-file-brain",
)
