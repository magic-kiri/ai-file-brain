from datetime import UTC, datetime
from typing import Any

import pytest

from ai_file_brain.config import AiFileBrainSettings
from ai_file_brain.core.chat import ChatService
from ai_file_brain.core.models import (
    QueryHit,
    SourcesChunk,
    TokenChunk,
)


class FakeEmbedder:
    async def embed(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]


class FakeRepo:
    def __init__(self, hits: list[QueryHit]) -> None:
        self.hits = hits
        self.last_modified_at_range: tuple | None = None

    async def initialize(self): ...
    async def upsert(self, *a, **kw): ...
    async def upsert_batch(self, *a, **kw): ...
    async def delete_by_path(self, *a, **kw): ...
    async def has_path(self, *a, **kw):
        return False
    async def count(self):
        return 0
    async def heartbeat(self):
        return True

    async def query(self, embedding, top_k, modified_at_range=None):
        self.last_modified_at_range = modified_at_range
        return self.hits[:top_k]


class FakeOllama:
    def __init__(self, tokens: list[str]) -> None:
        self.tokens = tokens

    async def chat(self, **kwargs: Any):
        async def gen():
            for tok in self.tokens:
                yield {"message": {"content": tok}}

        return gen()


def _settings() -> AiFileBrainSettings:
    return AiFileBrainSettings(
        watch_folder=".",
        ollama_url="http://x",
        chroma_path="./tmp",
        embedding_model="m",
        chat_model="m",
        chunk_size=100,
        chunk_overlap=10,
        top_k=3,
    )


@pytest.mark.asyncio
async def test_ask_stream_yields_tokens_then_sources():
    hits = [
        QueryHit(
            chunk_id="1",
            file_path="/a.txt",
            file_name="a.txt",
            chunk_index=0,
            text="content from a",
            distance=0.1,
            modified_at=datetime.now(UTC),
        ),
        QueryHit(
            chunk_id="2",
            file_path="/b.txt",
            file_name="b.txt",
            chunk_index=0,
            text="content from b",
            distance=0.2,
            modified_at=None,
        ),
        QueryHit(
            chunk_id="3",
            file_path="/a.txt",  # duplicate path -> dedup
            file_name="a.txt",
            chunk_index=1,
            text="more from a",
            distance=0.3,
            modified_at=None,
        ),
    ]
    chat = ChatService(_settings(), FakeEmbedder(), FakeRepo(hits), FakeOllama(["Hel", "lo"]))

    chunks = []
    async for c in chat.ask_stream("what?"):
        chunks.append(c)

    tokens = [c.text for c in chunks if isinstance(c, TokenChunk)]
    sources_chunks = [c for c in chunks if isinstance(c, SourcesChunk)]
    assert tokens == ["Hel", "lo"]
    assert len(sources_chunks) == 1
    assert sources_chunks[0].paths == ("/a.txt", "/b.txt")


@pytest.mark.asyncio
async def test_ask_stream_handles_no_hits():
    chat = ChatService(_settings(), FakeEmbedder(), FakeRepo([]), FakeOllama([]))
    chunks = [c async for c in chat.ask_stream("hi")]
    tokens = [c for c in chunks if isinstance(c, TokenChunk)]
    sources = [c for c in chunks if isinstance(c, SourcesChunk)]
    assert any("couldn't find" in t.text for t in tokens)
    assert sources and sources[0].paths == ()


@pytest.mark.asyncio
async def test_ask_aggregates_into_chat_result():
    hits = [
        QueryHit("1", "/a.txt", "a.txt", 0, "stuff", 0.1, None),
    ]
    chat = ChatService(_settings(), FakeEmbedder(), FakeRepo(hits), FakeOllama(["abc", "def"]))
    result = await chat.ask("?")
    assert result.answer == "abcdef"
    assert result.sources == ("/a.txt",)


@pytest.mark.asyncio
async def test_temporal_question_passes_time_window_to_repo():
    hits = [QueryHit("1", "/a.txt", "a.txt", 0, "stuff", 0.1, None)]
    repo = FakeRepo(hits)
    chat = ChatService(_settings(), FakeEmbedder(), repo, FakeOllama(["ok"]))
    await chat.ask("what was I working on yesterday?")
    assert repo.last_modified_at_range is not None
    start, end = repo.last_modified_at_range
    assert (end - start).total_seconds() == 86400  # exactly one day


@pytest.mark.asyncio
async def test_non_temporal_question_passes_no_window():
    hits = [QueryHit("1", "/a.txt", "a.txt", 0, "stuff", 0.1, None)]
    repo = FakeRepo(hits)
    chat = ChatService(_settings(), FakeEmbedder(), repo, FakeOllama(["ok"]))
    await chat.ask("explain how ranking works")
    assert repo.last_modified_at_range is None


@pytest.mark.asyncio
async def test_temporal_no_hits_uses_window_specific_message():
    repo = FakeRepo([])
    chat = ChatService(_settings(), FakeEmbedder(), repo, FakeOllama([]))
    chunks = [c async for c in chat.ask_stream("what changed last week?")]
    tokens = [c for c in chunks if isinstance(c, TokenChunk)]
    assert any("last week" in t.text for t in tokens)
