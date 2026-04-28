from pathlib import Path

import pytest

from ai_file_brain.core.extraction import (
    UnsupportedFileTypeError,
    get_extractor,
    is_supported,
)


def test_is_supported():
    assert is_supported("a.TXT")
    assert is_supported("b.pdf")
    assert not is_supported("c.docx")


def test_unsupported_raises():
    with pytest.raises(UnsupportedFileTypeError):
        get_extractor("/whatever/file.docx")


@pytest.mark.asyncio
async def test_plain_text_extractor_reads_file(tmp_path: Path):
    file = tmp_path / "hello.txt"
    file.write_text("greetings, earth", encoding="utf-8")
    extractor = get_extractor(str(file))
    text = await extractor.extract(str(file))
    assert text == "greetings, earth"


@pytest.mark.asyncio
async def test_pdf_extractor_returns_empty_for_missing_file(tmp_path: Path):
    fake = tmp_path / "missing.pdf"
    extractor = get_extractor(str(fake))
    text = await extractor.extract(str(fake))
    assert text == ""
