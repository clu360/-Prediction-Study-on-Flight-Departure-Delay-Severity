"""Execute the analysis notebook in a checkout containing aggregates only."""

from pathlib import Path
import os
import asyncio
import shutil
import nbformat
from nbclient import NotebookClient


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "data/model_outputs/phase3/notebook_checks"
    checkout = output / "aggregate_only_checkout"
    (checkout / "notebooks").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(root / "pyproject.toml", checkout / "pyproject.toml")
    if (root / "reports/phase3").exists():
        shutil.copytree(
            root / "reports/phase3", checkout / "reports/phase3", dirs_exist_ok=True
        )
    notebook = nbformat.read(
        root / "notebooks/03_classical_models_analysis.ipynb", as_version=4
    )
    os.environ["JUPYTER_PATH"] = str(root / ".venv/share/jupyter")
    os.environ["PYTHONPYCACHEPREFIX"] = str(root / ".venv/pycache")
    for variable, name in [
        ("IPYTHONDIR", "ipython"),
        ("MPLCONFIGDIR", "matplotlib"),
        ("JUPYTER_RUNTIME_DIR", "jupyter_runtime"),
    ]:
        directory = output / name
        directory.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(directory)
    if os.name == "nt":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    NotebookClient(
        notebook,
        timeout=120,
        kernel_name="flight-delay-phase3",
        resources={"metadata": {"path": str(checkout / "notebooks")}},
    ).execute()
    nbformat.write(notebook, output / "aggregate_only.executed.ipynb")
    assert all(
        not cell.get("outputs")
        or all(item.get("output_type") != "error" for item in cell["outputs"])
        for cell in notebook.cells
    )
    print(f"Aggregate-only notebook execution passed: {len(notebook.cells)} cells")


if __name__ == "__main__":
    main()
