from collections.abc import AsyncIterator

import pytest

from ai_file_brain.app.view_models.main_window_vm import MainWindowViewModel
from ai_file_brain.core.models import ChatStreamChunk, SourcesChunk, TokenChunk


class ScriptedChat:
    def __init__(self, scripts: dict[str, list[ChatStreamChunk]]) -> None:
        self.scripts = scripts

    async def ask_stream(self, question: str) -> AsyncIterator[ChatStreamChunk]:
        for chunk in self.scripts.get(question, []):
            yield chunk


@pytest.fixture
def qapp(qtbot):
    return qtbot  # ensures QApplication exists


@pytest.mark.asyncio
async def test_send_appends_turn_and_streams(qapp):
    chat = ScriptedChat({"hi": [TokenChunk("Hel"), TokenChunk("lo"), SourcesChunk(("/a.txt",))]})
    vm = MainWindowViewModel(chat)
    appended = []
    vm.turn_appended.connect(appended.append)

    vm.input_text = "hi"
    vm.send()
    assert vm.input_text == ""
    assert len(appended) == 1
    turn = appended[0]
    assert turn.question == "hi"

    # let the streaming task complete
    await vm._send_task

    assert turn.answer == "Hello"
    assert turn.sources == ("/a.txt",)


@pytest.mark.asyncio
async def test_cannot_send_empty(qapp):
    vm = MainWindowViewModel(ScriptedChat({}))
    vm.input_text = "   "
    vm.send()
    assert vm.turns == []


@pytest.mark.asyncio
async def test_is_sending_toggles(qapp):
    chat = ScriptedChat({"q": [TokenChunk("a")]})
    vm = MainWindowViewModel(chat)
    states: list[bool] = []
    vm.is_sending_changed.connect(states.append)
    vm.input_text = "q"
    vm.send()
    await vm._send_task
    assert True in states and False in states
    assert vm.is_sending is False
