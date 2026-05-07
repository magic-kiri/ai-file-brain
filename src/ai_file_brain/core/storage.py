from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Protocol, runtime_checkable

from ai_file_brain.core.models import FileChunk, QueryHit
from ai_file_brain.config import AiFileBrainSettings

logger = logging.getLogger(__name__)

COLLECTION_NAME = "ai-file-brain"


@runtime_checkable
class VectorRepository(Protocol):
    async def initialize(self) -> None: ...
    async def upsert(self, chunk: FileChunk, embedding: list[float]) -> None: ...
    async def upsert_batch(
        self, chunks: list[FileChunk], embeddings: list[list[float]]
    ) -> None: ...
    async def delete_by_path(self, file_path: str) -> None: ...
    async def query(
        self,
        embedding: list[float],
        top_k: int,
        modified_at_range: tuple[datetime, datetime] | None = None,
    ) -> list[QueryHit]: ...
    async def has_path(self, file_path: str) -> bool: ...
    async def count(self) -> int: ...
    async def heartbeat(self) -> bool: ...


class ChromaVectorRepository:
    def __init__(self, settings: AiFileBrainSettings) -> None:
        self._settings = settings
        self._client = None
        self._collection = None

    async def initialize(self) -> None:
        await asyncio.to_thread(self._init_sync)

    def _init_sync(self) -> None:
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        path = self._settings.chroma_path_resolved()
        path.mkdir(parents=True, exist_ok=True)
        logger.info("Opening ChromaDB at %s", path)

        self._client = chromadb.PersistentClient(
            path=str(path),
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("ChromaDB collection '%s' ready", COLLECTION_NAME)

    def _require(self):
        if self._collection is None:
            raise RuntimeError("ChromaVectorRepository.initialize() was not called")
        return self._collection

    async def upsert(self, chunk: FileChunk, embedding: list[float]) -> None:
        await self.upsert_batch([chunk], [embedding])

    async def upsert_batch(
        self, chunks: list[FileChunk], embeddings: list[list[float]]
    ) -> None:
        if not chunks:
            return
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must be same length")
        col = self._require()

        ids = [c.id for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [
            {
                "file_path": c.file_path,
                "file_name": c.file_name,
                "chunk_index": c.chunk_index,
                "created_at": c.created_at.isoformat(),
                "modified_at": c.modified_at.isoformat(),
                "extraction_source": c.extraction_source,
            }
            for c in chunks
        ]

        await asyncio.to_thread(
            col.upsert,
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    async def delete_by_path(self, file_path: str) -> None:
        col = self._require()
        await asyncio.to_thread(col.delete, where={"file_path": file_path})

    async def query(
        self,
        embedding: list[float],
        top_k: int,
        modified_at_range: tuple[datetime, datetime] | None = None,
    ) -> list[QueryHit]:
        col = self._require()
        kwargs: dict = {
            "query_embeddings": [embedding],
            "n_results": top_k,
        }
        if modified_at_range is not None:
            start, end = modified_at_range
            # ISO 8601 strings sort lexicographically by date, so $gte/$lte
            # work directly on the stored "modified_at" string metadata.
            kwargs["where"] = {
                "$and": [
                    {"modified_at": {"$gte": start.isoformat()}},
                    {"modified_at": {"$lt": end.isoformat()}},
                ]
            }
        result = await asyncio.to_thread(col.query, **kwargs)
        return _result_to_hits(result)

    async def has_path(self, file_path: str) -> bool:
        col = self._require()
        result = await asyncio.to_thread(
            col.get,
            where={"file_path": file_path},
            limit=1,
            include=[],
        )
        ids = result.get("ids") or []
        return bool(ids)

    async def count(self) -> int:
        col = self._require()
        return await asyncio.to_thread(col.count)

    async def heartbeat(self) -> bool:
        if self._client is None:
            return False
        try:
            await asyncio.to_thread(self._client.heartbeat)
            return True
        except Exception as ex:
            logger.debug("Chroma heartbeat failed: %s", ex)
            return False


def _result_to_hits(result: dict) -> list[QueryHit]:
    ids_outer = result.get("ids") or []
    if not ids_outer:
        return []
    ids = ids_outer[0] or []
    distances = (result.get("distances") or [[]])[0] or [0.0] * len(ids)
    documents = (result.get("documents") or [[]])[0] or [""] * len(ids)
    metadatas = (result.get("metadatas") or [[]])[0] or [{}] * len(ids)

    hits: list[QueryHit] = []
    for i, chunk_id in enumerate(ids):
        meta = metadatas[i] or {}
        modified_iso = meta.get("modified_at")
        modified_at = None
        if isinstance(modified_iso, str):
            try:
                modified_at = datetime.fromisoformat(modified_iso)
            except ValueError:
                modified_at = None
        hits.append(
            QueryHit(
                chunk_id=chunk_id,
                file_path=str(meta.get("file_path", "")),
                file_name=str(meta.get("file_name", "")),
                chunk_index=int(meta.get("chunk_index", 0) or 0),
                text=documents[i] or "",
                distance=float(distances[i] or 0.0),
                modified_at=modified_at,
            )
        )
    return hits
