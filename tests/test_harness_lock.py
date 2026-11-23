import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _tool():
    spec = importlib.util.spec_from_file_location("harness_lock", ROOT / "scripts" / "harness_lock.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_locked_files_match_the_manifest() -> None:
    assert _tool().check() == []


def test_manifest_covers_the_kernel_and_the_config() -> None:
    files = json.loads((ROOT / "HARNESS_LOCK.json").read_text())["files"]
    for required in (
        "src/wrfcomp/harness/metrics.py",
        "src/wrfcomp/harness/evaluate.py",
        "src/wrfcomp/harness/truth.py",
        "configs/splits.yaml",
        "configs/stations_conus.csv",
    ):
        assert required in files


def test_a_changed_locked_file_is_detected(tmp_path, monkeypatch) -> None:
    tool = _tool()
    (tmp_path / "src/wrfcomp/harness").mkdir(parents=True)
    (tmp_path / "configs").mkdir()
    (tmp_path / "src/wrfcomp/harness/metrics.py").write_text("a = 1\n")
    (tmp_path / "configs/splits.yaml").write_text("x: 1\n")
    monkeypatch.setattr(tool, "ROOT", tmp_path)
    monkeypatch.setattr(tool, "MANIFEST", tmp_path / "HARNESS_LOCK.json")
    tool.update("test")
    assert tool.check() == []
    (tmp_path / "configs/splits.yaml").write_text("x: 2\n")
    assert tool.check() == ["changed: configs/splits.yaml"]
    (tmp_path / "src/wrfcomp/harness/extra.py").write_text("b = 2\n")
    assert "added: src/wrfcomp/harness/extra.py" in tool.check()
    (tmp_path / "src/wrfcomp/harness/metrics.py").unlink()
    assert "removed: src/wrfcomp/harness/metrics.py" in tool.check()
