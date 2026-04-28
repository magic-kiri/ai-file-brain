from __future__ import annotations

import asyncio
import logging

from pypdf import PdfReader

logger = logging.getLogger(__name__)


class PdfExtractor:
    async def extract(self, file_path: str) -> str:
        return await asyncio.to_thread(self._read_sync, file_path)

    @staticmethod
    def _read_sync(file_path: str) -> str:
        try:
            reader = PdfReader(file_path)
        except Exception as ex:
            logger.warning("Failed to open PDF %s: %s", file_path, ex)
            return ""
        pages: list[str] = []
        for i, page in enumerate(reader.pages):
            try:
                pages.append(page.extract_text() or "")
            except Exception as ex:
                logger.warning("Failed to extract page %d of %s: %s", i, file_path, ex)
        return "\n".join(pages)
