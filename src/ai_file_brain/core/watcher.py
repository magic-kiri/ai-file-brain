from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from watchdog.events import (
    FileCreatedEvent,
    FileDeletedEvent,
    FileModifiedEvent,
    FileMovedEvent,
    FileSystemEventHandler,
)
from watchdog.observers import Observer

from ai_file_brain.config import AiFileBrainSettings
from ai_file_brain.core.chunking import ChunkingService
from ai_file_brain.core.embedding import EmbeddingService
from ai_file_brain.core.extraction import get_extractor, is_supported
from ai_file_brain.core.models import FileChunk
from ai_file_brain.core.storage import VectorRepository

logger = logging.getLogger(__name__)

DEBOUNCE_SECONDS = 0.5
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = (1.0, 2.0, 4.0)


ProgressCallback = Callable[["IndexingProgress"], None]


@dataclass(frozen=True, slots=True)
class IndexingProgress:
    file_path: str
    state: str  # "indexing" | "indexed" | "deleted" | "error"
    detail: str = ""


class IndexingPipeline:
    def __init__(
        self,
        chunker: ChunkingService,
        embedder: EmbeddingService,
        repo: VectorRepository,
    ) -> None:
        self._chunker = chunker
        self._embedder = embedder
        self._repo = repo

    async def index_file(self, file_path: str) -> int:
        if not is_supported(file_path):
            return 0
        for attempt, backoff in enumerate(RETRY_BACKOFF_SECONDS, start=1):
            try:
                return await self._index_file_once(file_path)
            except FileNotFoundError:
                logger.info("File disappeared before indexing: %s", file_path)
                return 0
            except (PermissionError, OSError) as ex:
                if attempt >= MAX_RETRIES:
                    logger.warning(
                        "Giving up indexing %s after %d attempts: %s",
                        file_path,
                        attempt,
                        ex,
                    )
                    return 0
                logger.debug(
                    "Indexing %s failed (attempt %d/%d): %s; retrying in %.1fs",
                    file_path,
                    attempt,
                    MAX_RETRIES,
                    ex,
                    backoff,
                )
                await asyncio.sleep(backoff)
        return 0

    async def _index_file_once(self, file_path: str) -> int:
        extractor = get_extractor(file_path)
        text = await extractor.extract(file_path)
        if not text.strip():
            logger.info("No extractable text in %s; clearing prior chunks", file_path)
            await self._repo.delete_by_path(file_path)
            return 0

        text_chunks = self._chunker.chunk(text)
        if not text_chunks:
            await self._repo.delete_by_path(file_path)
            return 0

        stat = os.stat(file_path)
        modified = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
        created = datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc)
        file_name = os.path.basename(file_path)

        await self._repo.delete_by_path(file_path)

        chunks: list[FileChunk] = []
        embeddings: list[list[float]] = []
        for tc in text_chunks:
            chunk = FileChunk(
                id=FileChunk.make_id(file_path, tc.chunk_index),
                file_path=file_path,
                file_name=file_name,
                chunk_index=tc.chunk_index,
                text=tc.text,
                created_at=created,
                modified_at=modified,
            )
            embedding = await self._embedder.embed(tc.text)
            if not embedding:
                continue
            chunks.append(chunk)
            embeddings.append(embedding)

        if chunks:
            await self._repo.upsert_batch(chunks, embeddings)
        return len(chunks)


class FileWatcherService:
    def __init__(
        self,
        settings: AiFileBrainSettings,
        pipeline: IndexingPipeline,
        repo: VectorRepository,
        progress: ProgressCallback | None = None,
    ) -> None:
        self._settings = settings
        self._pipeline = pipeline
        self._repo = repo
        self._progress = progress
        self._loop: asyncio.AbstractEventLoop | None = None
        self._observer: Observer | None = None
        self._handler: _Handler | None = None
        self._debouncers: dict[str, asyncio.TimerHandle] = {}
        self._tasks: set[asyncio.Task] = set()
        self._running = False

    async def start(self) -> None:
        if self._running:
            return
        self._loop = asyncio.get_running_loop()
        Path(self._settings.watch_folder).mkdir(parents=True, exist_ok=True)
        self._observer = Observer()
        self._handler = _Handler(self)
        self._observer.schedule(self._handler, self._settings.watch_folder, recursive=True)
        self._observer.start()
        self._running = True
        logger.info("Watching %s", self._settings.watch_folder)
        await self._initial_scan()

    async def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        for handle in list(self._debouncers.values()):
            handle.cancel()
        self._debouncers.clear()
        for task in list(self._tasks):
            task.cancel()
        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=2.0)
            self._observer = None

    # called from watchdog thread
    def _on_event(self, kind: str, src: str, dst: str | None = None) -> None:
        loop = self._loop
        if not loop or not self._running:
            return
        loop.call_soon_threadsafe(self._handle_event, kind, src, dst)

    def _handle_event(self, kind: str, src: str, dst: str | None) -> None:
        if kind == "deleted":
            if is_supported(src):
                self._schedule_task(self._handle_delete(src))
            return
        if kind == "moved":
            if dst and is_supported(dst):
                self._schedule_task(self._handle_rename(src, dst))
            elif is_supported(src):
                self._schedule_task(self._handle_delete(src))
            return
        # created / modified
        if is_supported(src):
            self._debounce_index(src)

    def _debounce_index(self, file_path: str) -> None:
        loop = self._loop
        if not loop:
            return
        existing = self._debouncers.pop(file_path, None)
        if existing:
            existing.cancel()
        handle = loop.call_later(
            DEBOUNCE_SECONDS,
            lambda p=file_path: self._fire_debounced(p),
        )
        self._debouncers[file_path] = handle

    def _fire_debounced(self, file_path: str) -> None:
        self._debouncers.pop(file_path, None)
        self._schedule_task(self._index(file_path))

    def _schedule_task(self, coro: Awaitable[None]) -> None:
        loop = self._loop
        if not loop:
            return
        task = loop.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _index(self, file_path: str) -> None:
        self._notify(IndexingProgress(file_path, "indexing"))
        try:
            count = await self._pipeline.index_file(file_path)
            self._notify(IndexingProgress(file_path, "indexed", f"{count} chunks"))
        except Exception as ex:
            logger.exception("Indexing failed for %s", file_path)
            self._notify(IndexingProgress(file_path, "error", str(ex)))

    async def _handle_delete(self, file_path: str) -> None:
        try:
            await self._repo.delete_by_path(file_path)
            self._notify(IndexingProgress(file_path, "deleted"))
        except Exception as ex:
            logger.warning("Delete-from-index failed for %s: %s", file_path, ex)

    async def _handle_rename(self, old_path: str, new_path: str) -> None:
        try:
            await self._repo.delete_by_path(old_path)
        except Exception as ex:
            logger.warning("Delete-on-rename failed for %s: %s", old_path, ex)
        await self._index(new_path)

    async def _initial_scan(self) -> None:
        folder = Path(self._settings.watch_folder)
        for path in folder.rglob("*"):
            if not path.is_file():
                continue
            file_path = str(path)
            if not is_supported(file_path):
                continue
            try:
                already = await self._repo.has_path(file_path)
            except Exception as ex:
                logger.warning("has_path check failed for %s: %s", file_path, ex)
                already = False
            if not already:
                self._schedule_task(self._index(file_path))

    def _notify(self, progress: IndexingProgress) -> None:
        if self._progress is None:
            return
        try:
            self._progress(progress)
        except Exception:
            logger.exception("Progress callback raised")


class _Handler(FileSystemEventHandler):
    def __init__(self, parent: FileWatcherService) -> None:
        super().__init__()
        self._parent = parent

    def on_created(self, event):
        if isinstance(event, FileCreatedEvent):
            self._parent._on_event("created", event.src_path)

    def on_modified(self, event):
        if isinstance(event, FileModifiedEvent):
            self._parent._on_event("modified", event.src_path)

    def on_deleted(self, event):
        if isinstance(event, FileDeletedEvent):
            self._parent._on_event("deleted", event.src_path)

    def on_moved(self, event):
        if isinstance(event, FileMovedEvent):
            self._parent._on_event("moved", event.src_path, event.dest_path)
