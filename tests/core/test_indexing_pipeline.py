from __future__ import annotations

from pathlib import Path

import pytest

from ai_file_brain.config import AiFileBrainSettings
from ai_file_brain.core.chunking import ChunkingService
from ai_file_brain.core.models import FileChunk
from ai_file_brain.core.watcher import IndexingPipeline


class _FakeEmbedder:
    async def embed(self, text: str) -> list[float]:
        return [0.1] * 8


class _FakeRepo:
    def __init__(self) -> None:
        self.upserted: list[FileChunk] = []
        self.deletions: list[str] = []

    async def initialize(self) -> None: ...
    async def upsert(self, *a, **kw) -> None: ...
    async def upsert_batch(self, chunks, embeddings) -> None:
        self.upserted.extend(chunks)
    async def delete_by_path(self, path: str) -> None:
        self.deletions.append(path)
    async def has_path(self, path: str) -> bool:
        return False
    async def query(self, *a, **kw):
        return []
    async def count(self) -> int:
        return 0
    async def heartbeat(self) -> bool:
        return True


def _build_pipeline(settings: AiFileBrainSettings) -> tuple[IndexingPipeline, _FakeRepo]:
    repo = _FakeRepo()
    pipeline = IndexingPipeline(
        chunker=ChunkingService(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap),
        embedder=_FakeEmbedder(),
        repo=repo,
        settings=settings,
    )
    return pipeline, repo


@pytest.mark.asyncio
async def test_pipeline_indexes_small_text_file(tmp_path: Path):
    settings = AiFileBrainSettings()
    pipeline, repo = _build_pipeline(settings)

    f = tmp_path / "note.txt"
    f.write_text("a short note about ai file brain", encoding="utf-8")

    count = await pipeline.index_file(str(f))
    assert count == 1
    assert len(repo.upserted) == 1
    assert repo.upserted[0].extraction_source == "native"


@pytest.mark.asyncio
async def test_pipeline_skips_files_over_size_cap(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("AFB_MAX_FILE_SIZE_BYTES", "100")
    settings = AiFileBrainSettings()
    assert settings.max_file_size_bytes == 100

    pipeline, repo = _build_pipeline(settings)

    big = tmp_path / "huge.txt"
    big.write_bytes(b"x" * 500)  # 500 bytes > 100-byte cap

    count = await pipeline.index_file(str(big))
    assert count == 0
    assert repo.upserted == []
    # Stale chunks for that path are cleared.
    assert str(big) in repo.deletions


@pytest.mark.asyncio
async def test_pipeline_indexes_unsupported_extension_as_filename_only(tmp_path: Path):
    """Files we can't extract text from still get a tiny filename-only chunk so
    "do I have files about X" can still find them. The chunk is excluded from
    semantic search via the extraction_source metadata filter in the repo."""
    settings = AiFileBrainSettings()
    pipeline, repo = _build_pipeline(settings)

    f = tmp_path / "mcfcoreinstaller.zip"
    f.write_bytes(b"PK\x03\x04")  # would-be ZIP header

    count = await pipeline.index_file(str(f))
    assert count == 1
    assert len(repo.upserted) == 1
    chunk = repo.upserted[0]
    assert chunk.extraction_source == "filename_only"
    assert chunk.file_name == "mcfcoreinstaller.zip"
    # The chunk text is the filename itself so it has *something* to embed and
    # so the chunk-level document is non-empty.
    assert chunk.text == "mcfcoreinstaller.zip"
