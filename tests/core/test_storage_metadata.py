from __future__ import annotations

from datetime import UTC, datetime

import pytest

from ai_file_brain.config import AiFileBrainSettings
from ai_file_brain.core.models import FileChunk
from ai_file_brain.core.storage import ChromaVectorRepository


class _StubCollection:
    def __init__(self) -> None:
        self.last_metadatas: list[dict] | None = None
        self.last_query_where: dict | None = None

    def upsert(self, ids, embeddings, documents, metadatas):
        self.last_metadatas = list(metadatas)

    def query(self, query_embeddings, n_results, where=None):
        self.last_query_where = where
        return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}


@pytest.mark.asyncio
async def test_extraction_source_round_trips_through_metadata():
    repo = ChromaVectorRepository(AiFileBrainSettings())
    stub = _StubCollection()
    repo._collection = stub  # bypass real chromadb init

    now = datetime.now(UTC)
    chunk_native = FileChunk(
        id="a",
        file_path="/p/native.txt",
        file_name="native.txt",
        chunk_index=0,
        text="native body",
        created_at=now,
        modified_at=now,
        extraction_source="native",
    )
    chunk_ocr = FileChunk(
        id="b",
        file_path="/p/scan.png",
        file_name="scan.png",
        chunk_index=0,
        text="ocr body",
        created_at=now,
        modified_at=now,
        extraction_source="ocr",
    )

    await repo.upsert_batch([chunk_native, chunk_ocr], [[0.1, 0.2], [0.3, 0.4]])

    assert stub.last_metadatas is not None
    assert stub.last_metadatas[0]["extraction_source"] == "native"
    assert stub.last_metadatas[1]["extraction_source"] == "ocr"


@pytest.mark.asyncio
async def test_query_excludes_filename_only_chunks():
    """Filename-only stubs (.zip, .exe, …) must not pollute semantic results."""
    repo = ChromaVectorRepository(AiFileBrainSettings())
    stub = _StubCollection()
    repo._collection = stub

    await repo.query([0.1, 0.2, 0.3], top_k=5)

    assert stub.last_query_where == {"extraction_source": {"$ne": "filename_only"}}


@pytest.mark.asyncio
async def test_query_with_time_window_still_excludes_filename_only():
    repo = ChromaVectorRepository(AiFileBrainSettings())
    stub = _StubCollection()
    repo._collection = stub

    start = datetime(2026, 1, 1, tzinfo=UTC)
    end = datetime(2026, 2, 1, tzinfo=UTC)
    await repo.query([0.1, 0.2, 0.3], top_k=5, modified_at_range=(start, end))

    where = stub.last_query_where
    assert where is not None and "$and" in where
    clauses = where["$and"]
    assert {"extraction_source": {"$ne": "filename_only"}} in clauses


@pytest.mark.asyncio
async def test_filechunk_default_extraction_source_is_native():
    now = datetime.now(UTC)
    chunk = FileChunk(
        id="a",
        file_path="/p/x.txt",
        file_name="x.txt",
        chunk_index=0,
        text="x",
        created_at=now,
        modified_at=now,
    )
    assert chunk.extraction_source == "native"
