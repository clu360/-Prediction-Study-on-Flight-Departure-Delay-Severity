"""Execute the analysis notebook in a checkout containing aggregates only."""

from pathlib import Path
import argparse
import os
import asyncio
import re
import shutil
import nbformat
from nbclient import NotebookClient


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-dir", type=Path, default=root / "reports/phase3")
    parser.add_argument("--check-name", default="aggregate_only")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.check_name):
        raise ValueError("Check name must be a simple directory-safe label")
    output = root / "data/model_outputs/phase3/notebook_checks"
    checkout = output / f"{args.check_name}_checkout"
    (checkout / "notebooks").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(root / "pyproject.toml", checkout / "pyproject.toml")
    if args.reports_dir.exists():
        shutil.copytree(
            args.reports_dir, checkout / "reports/phase3", dirs_exist_ok=True
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
    nbformat.write(notebook, output / f"{args.check_name}.executed.ipynb")
    assert all(
        not cell.get("outputs")
        or all(item.get("output_type") != "error" for item in cell["outputs"])
        for cell in notebook.cells
    )
    print(f"Aggregate-only notebook execution passed: {len(notebook.cells)} cells")


if __name__ == "__main__":
    main()
