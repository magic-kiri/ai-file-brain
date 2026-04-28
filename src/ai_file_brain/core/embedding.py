from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

from ollama import AsyncClient

logger = logging.getLogger(__name__)


@runtime_checkable
class EmbeddingService(Protocol):
    async def embed(self, text: str) -> list[float]: ...


class OllamaEmbeddingService:
    def __init__(self, client: AsyncClient, model: str) -> None:
        self._client = client
        self._model = model

    async def embed(self, text: str) -> list[float]:
        if not text:
            return []
        resp = await self._client.embeddings(model=self._model, prompt=text)
        return list(resp["embedding"])
