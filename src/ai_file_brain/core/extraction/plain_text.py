from __future__ import annotations

import aiofiles


class PlainTextExtractor:
    async def extract(self, file_path: str) -> str:
        async with aiofiles.open(file_path, mode="r", encoding="utf-8", errors="replace") as f:
            return await f.read()
