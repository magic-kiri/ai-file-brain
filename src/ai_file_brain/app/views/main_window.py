from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import Qt
from PySide6.QtGui import QCloseEvent, QFont, QKeyEvent
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ai_file_brain.app.models.chat_turn import ChatTurn
from ai_file_brain.app.view_models.main_window_vm import MainWindowViewModel
from ai_file_brain.app.view_models.status_bar_vm import StatusBarViewModel


class _ChatTurnWidget(QFrame):
    def __init__(self, turn: ChatTurn, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self._turn = turn

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        question_font = QFont()
        question_font.setBold(True)
        self._question_label = QLabel(f"You: {turn.question}")
        self._question_label.setWordWrap(True)
        self._question_label.setFont(question_font)
        layout.addWidget(self._question_label)

        self._answer_label = QLabel("")
        self._answer_label.setWordWrap(True)
        self._answer_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self._answer_label)

        self._sources_label = QLabel("")
        self._sources_label.setWordWrap(True)
        self._sources_label.setStyleSheet("color: #888; font-size: 10px;")
        self._sources_label.setVisible(False)
        layout.addWidget(self._sources_label)

        self._error_label = QLabel("")
        self._error_label.setWordWrap(True)
        self._error_label.setStyleSheet("color: #c0392b;")
        self._error_label.setVisible(False)
        layout.addWidget(self._error_label)

        turn.answer_changed.connect(self._refresh_answer)
        turn.sources_changed.connect(self._refresh_sources)
        turn.error_changed.connect(self._refresh_error)

    def _refresh_answer(self) -> None:
        self._answer_label.setText(self._turn.answer)

    def _refresh_sources(self) -> None:
        if not self._turn.sources:
            self._sources_label.setVisible(False)
            return
        rendered = "\n".join(f"  • {p}" for p in self._turn.sources)
        self._sources_label.setText(f"Sources:\n{rendered}")
        self._sources_label.setVisible(True)

    def _refresh_error(self) -> None:
        msg = self._turn.error
        if not msg:
            self._error_label.setVisible(False)
            return
        self._error_label.setText(f"Error: {msg}")
        self._error_label.setVisible(True)


class MainWindow(QWidget):
    def __init__(
        self,
        vm: MainWindowViewModel,
        status_vm: StatusBarViewModel,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._vm = vm
        self._status_vm = status_vm
        self._is_quitting = False
        self._change_folder_handler: Callable[[], None] | None = None

        self.setWindowTitle("AI File Brain")
        self.setMinimumSize(600, 500)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._transcript_host = QWidget()
        self._transcript_layout = QVBoxLayout(self._transcript_host)
        self._transcript_layout.setContentsMargins(0, 0, 0, 0)
        self._transcript_layout.setSpacing(0)
        self._transcript_layout.addStretch(1)
        self._scroll.setWidget(self._transcript_host)
        outer.addWidget(self._scroll, 1)

        input_row = QHBoxLayout()
        input_row.setContentsMargins(8, 8, 8, 8)
        self._input = _EnterToSendTextEdit(self._on_enter_pressed)
        self._input.setPlaceholderText("Ask a question about your files…  (Shift+Enter for newline)")
        self._input.setMaximumHeight(80)
        input_row.addWidget(self._input, 1)

        self._send_button = QPushButton("Send")
        self._send_button.clicked.connect(self._on_send_clicked)
        input_row.addWidget(self._send_button)

        outer.addLayout(input_row)

        status_strip = QWidget()
        status_strip.setStyleSheet("background-color: #f0f0f0;")
        status_layout = QHBoxLayout(status_strip)
        status_layout.setContentsMargins(8, 4, 8, 4)
        status_layout.setSpacing(8)

        self._status_label = QLabel(self._status_vm.render())
        self._status_label.setStyleSheet("color: #333; font-size: 11px;")
        self._status_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        status_layout.addWidget(self._status_label, 1)

        self._change_folder_button = QToolButton()
        self._change_folder_button.setText("Change folder…")
        self._change_folder_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._change_folder_button.setStyleSheet(
            "QToolButton { color: #2c5282; font-size: 11px; padding: 0 6px;"
            " background: transparent; border: none; }"
            "QToolButton:hover { color: #1a365d; text-decoration: underline; }"
        )
        self._change_folder_button.clicked.connect(self._on_change_folder_clicked)
        status_layout.addWidget(self._change_folder_button)

        outer.addWidget(status_strip)

        vm.turn_appended.connect(self._on_turn_appended)
        vm.is_sending_changed.connect(self._on_sending_changed)
        vm.input_text_changed.connect(self._on_input_text_changed)
        status_vm.changed.connect(self._refresh_status)

        self._input.textChanged.connect(self._sync_input_to_vm)

    # ---- public API used by tray ----

    def show_and_raise(self) -> None:
        self.show()
        self.raise_()
        self.activateWindow()

    def toggle_visibility(self) -> None:
        if self.isVisible():
            self.hide()
        else:
            self.show_and_raise()

    def mark_quitting(self) -> None:
        self._is_quitting = True

    def set_change_folder_handler(self, handler: Callable[[], None]) -> None:
        self._change_folder_handler = handler

    # ---- event handlers ----

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._is_quitting:
            event.accept()
            return
        event.ignore()
        self.hide()

    def _on_enter_pressed(self) -> None:
        self._on_send_clicked()

    def _on_send_clicked(self) -> None:
        if self._vm.is_sending:
            self._vm.stop()
        else:
            self._vm.send()

    def _on_change_folder_clicked(self) -> None:
        if self._change_folder_handler is not None:
            self._change_folder_handler()

    def _on_turn_appended(self, turn: ChatTurn) -> None:
        widget = _ChatTurnWidget(turn)
        # insert before the trailing stretch
        self._transcript_layout.insertWidget(self._transcript_layout.count() - 1, widget)
        # auto-scroll to bottom on next event-loop tick
        self._scroll.verticalScrollBar().rangeChanged.connect(self._scroll_to_bottom)

    def _scroll_to_bottom(self) -> None:
        bar = self._scroll.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _on_sending_changed(self, sending: bool) -> None:
        self._send_button.setText("Stop" if sending else "Send")

    def _on_input_text_changed(self, text: str) -> None:
        if self._input.toPlainText() != text:
            self._input.blockSignals(True)
            self._input.setPlainText(text)
            self._input.blockSignals(False)

    def _sync_input_to_vm(self) -> None:
        self._vm.input_text = self._input.toPlainText()

    def _refresh_status(self) -> None:
        self._status_label.setText(self._status_vm.render())


class _EnterToSendTextEdit(QPlainTextEdit):
    def __init__(self, on_submit) -> None:
        super().__init__()
        self._on_submit = on_submit

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                super().keyPressEvent(event)
                return
            self._on_submit()
            return
        super().keyPressEvent(event)
