from __future__ import annotations

from PySide6.QtCore import QObject, Signal


class StatusBarViewModel(QObject):
    changed = Signal()

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._watch_folder = ""
        self._chunk_count = 0
        self._ollama_healthy = False
        self._chroma_healthy = False

    @property
    def watch_folder(self) -> str:
        return self._watch_folder

    @watch_folder.setter
    def watch_folder(self, v: str) -> None:
        if v != self._watch_folder:
            self._watch_folder = v
            self.changed.emit()

    @property
    def chunk_count(self) -> int:
        return self._chunk_count

    @chunk_count.setter
    def chunk_count(self, v: int) -> None:
        if v != self._chunk_count:
            self._chunk_count = v
            self.changed.emit()

    @property
    def ollama_healthy(self) -> bool:
        return self._ollama_healthy

    @ollama_healthy.setter
    def ollama_healthy(self, v: bool) -> None:
        if v != self._ollama_healthy:
            self._ollama_healthy = v
            self.changed.emit()

    @property
    def chroma_healthy(self) -> bool:
        return self._chroma_healthy

    @chroma_healthy.setter
    def chroma_healthy(self, v: bool) -> None:
        if v != self._chroma_healthy:
            self._chroma_healthy = v
            self.changed.emit()

    def render(self) -> str:
        ollama = "✓" if self._ollama_healthy else "✗"
        chroma = "✓" if self._chroma_healthy else "✗"
        folder = self._watch_folder or "(none)"
        return (
            f"Watching {folder} · {self._chunk_count} chunks · "
            f"Ollama {ollama} · Chroma {chroma}"
        )
