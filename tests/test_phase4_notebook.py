from pathlib import Path
from shutil import copyfile

import pytest


def test_phase4_notebook_executes_with_only_public_aggregates(tmp_path, monkeypatch):
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    pytest.importorskip("ipykernel")
    pytest.importorskip("matplotlib")
    root = Path.cwd()
    (tmp_path / "reports/phase3").mkdir(parents=True)
    (tmp_path / "reports/phase4").mkdir(parents=True)
    copyfile(root / "reports/phase3/metrics_summary.csv", tmp_path / "reports/phase3/metrics_summary.csv")
    monkeypatch.setenv("IPYTHONDIR", str(tmp_path / "ipython"))
    notebook = nbformat.read(root / "notebooks/04_weather_mlp_analysis.ipynb", as_version=4)
    executed = nbclient.NotebookClient(
        notebook, timeout=120, kernel_name="python3",
        resources={"metadata": {"path": str(tmp_path)}},
    ).execute()
    cells = [cell for cell in executed.cells if cell.cell_type == "code"]
    assert len(cells) == 4
    assert all(cell.execution_count for cell in cells)
    assert not any(output.output_type == "error" for cell in cells for output in cell.outputs)
    assert any("G1_G2_G3" in str(output) for output in cells[0].outputs)

