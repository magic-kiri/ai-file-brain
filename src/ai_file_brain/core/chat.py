from __future__ import annotations

import logging
from collections.abc import AsyncIterator

from ollama import AsyncClient

from ai_file_brain.config import AiFileBrainSettings
from ai_file_brain.core.embedding import EmbeddingService
from ai_file_brain.core.models import (
    ChatResult,
    ChatStreamChunk,
    QueryHit,
    SourcesChunk,
    TokenChunk,
)
from ai_file_brain.core.storage import VectorRepository
from ai_file_brain.core.time_intent import TimeWindow, parse_time_intent

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions based on the user's local files.\n"
    "Use ONLY the following file excerpts to answer. "
    "If the answer is not in the excerpts, say so."
)


class ChatService:
    def __init__(
        self,
        settings: AiFileBrainSettings,
        embedder: EmbeddingService,
        vector_repo: VectorRepository,
        ollama: AsyncClient,
    ) -> None:
        self._settings = settings
        self._embedder = embedder
        self._vector_repo = vector_repo
        self._ollama = ollama

    async def ask(self, question: str) -> ChatResult:
        answer_parts: list[str] = []
        sources: tuple[str, ...] = ()
        async for chunk in self.ask_stream(question):
            if isinstance(chunk, TokenChunk):
                answer_parts.append(chunk.text)
            elif isinstance(chunk, SourcesChunk):
                sources = chunk.paths
        return ChatResult(answer="".join(answer_parts), sources=sources)

    async def ask_stream(self, question: str) -> AsyncIterator[ChatStreamChunk]:
        question = (question or "").strip()
        if not question:
            yield SourcesChunk(paths=())
            return

        window = parse_time_intent(question)

        embedding = await self._embedder.embed(question)
        hits = await self._vector_repo.query(
            embedding,
            self._settings.top_k,
            modified_at_range=(window.start, window.end) if window else None,
        )

        if not hits:
            if window is not None:
                yield TokenChunk(
                    text=f"I couldn't find any files modified during {window.label}."
                )
            else:
                yield TokenChunk(text="I couldn't find any relevant content in your files.")
            yield SourcesChunk(paths=())
            return

        user_message = _build_user_message(question, hits, window)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]

        try:
            stream = await self._ollama.chat(
                model=self._settings.chat_model,
                messages=messages,
                stream=True,
            )
            async for part in stream:
                token = (part.get("message") or {}).get("content", "")
                if token:
                    yield TokenChunk(text=token)
        except Exception as ex:
            logger.exception("Ollama chat stream failed")
            yield TokenChunk(text=f"\n[error: {ex}]")

        seen: list[str] = []
        for hit in hits:
            if hit.file_path and hit.file_path not in seen:
                seen.append(hit.file_path)
        yield SourcesChunk(paths=tuple(seen))


def _build_user_message(
    question: str, hits: list[QueryHit], window: TimeWindow | None = None
) -> str:
    blocks: list[str] = []
    if window is not None:
        blocks.append(
            f"[Filtered to files modified during {window.label} "
            f"({window.start.isoformat()} to {window.end.isoformat()})]"
        )
    for hit in hits:
        modified = hit.modified_at.isoformat() if hit.modified_at else "unknown"
        blocks.append(
            f"--- File: {hit.file_name} (modified: {modified}) ---\n{hit.text}"
        )
    blocks.append("")
    blocks.append(f"Question: {question}")
    return "\n\n".join(blocks)
