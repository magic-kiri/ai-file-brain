from ai_file_brain.app.view_models.status_bar_vm import StatusBarViewModel


def test_render_shows_all_fields(qtbot):
    vm = StatusBarViewModel()
    vm.watch_folder = "C:/notes"
    vm.chunk_count = 42
    vm.ollama_healthy = True
    vm.chroma_healthy = False
    text = vm.render()
    assert "C:/notes" in text
    assert "42 chunks" in text
    assert "Ollama ✓" in text
    assert "Chroma ✗" in text


def test_changed_signal_fires_once_per_update(qtbot):
    vm = StatusBarViewModel()
    fires = []
    vm.changed.connect(lambda: fires.append(1))
    vm.watch_folder = "C:/x"
    vm.watch_folder = "C:/x"  # no change
    vm.chunk_count = 1
    assert len(fires) == 2
