"""Execute the Phase III/IV notebooks with public aggregates only."""

import argparse
import asyncio
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def count_rows(relative):
    with (ROOT / relative).open(encoding="utf-8", newline="") as stream:
        return len(list(csv.DictReader(stream)))


def validate(kernel="flight-delay-phase3"):
    expected_rows = {
        "reports/phase3/metrics_summary.csv": 780,
        "reports/phase4/metrics_summary.csv": 546,
        "reports/phase4/mlp_tuning_summary.csv": 26,
        "reports/mlp_source/metrics_summary.csv": 24,
        "reports/mlp_source/metrics_by_seed.csv": 72,
        "reports/mlp_source/tuning_G1_G2_G3.csv": 26,
        "reports/mlp_source/tuning_G1_G2_G3_G4.csv": 26,
    }
    for relative, expected in expected_rows.items():
        if count_rows(relative) != expected:
            raise ValueError(f"Incomplete published result coverage: {relative}")

    def read(relative):
        return json.loads((ROOT / relative).read_text(encoding="utf-8"))

    phase3_execution = read("reports/phase3/execution_summary.json")
    phase4_manifest = read("reports/phase4/experiment_manifest.json")
    source_manifest = read("reports/mlp_source/manifest.json")
    source_details = read("reports/mlp_source/metric_details.json")
    source_audit = read("reports/mlp_source/detail_export_verification.json")
    if (
        phase3_execution["fit_counts"]
        != {"development": 367, "frozen_final": 13, "total": 380}
        or len(phase4_manifest["fits"]) != 14
        or source_manifest["completed"] != 24
        or source_manifest["expected"] != 24
        or len(source_details) != 72
        or source_audit["status"] != "passed"
        or source_audit["details_sha256"]
        != sha(ROOT / "reports/mlp_source/metric_details.json")
    ):
        raise ValueError("Incomplete or stale experiment/detail manifests")
    output = ROOT / "data/model_outputs/analysis_notebook_checks"
    output.mkdir(parents=True, exist_ok=True)
    checkout = Path(tempfile.mkdtemp(prefix="aggregate_only_", dir=output))
    (checkout / "notebooks").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / "pyproject.toml", checkout / "pyproject.toml")
    for name in ["phase3", "phase4", "mlp_source"]:
        shutil.copytree(
            ROOT / "reports" / name, checkout / "reports" / name, dirs_exist_ok=True
        )
    assert not (checkout / "data").exists() and not (checkout / "src").exists()
    for variable, name in [
        ("IPYTHONDIR", "ipython"),
        ("MPLCONFIGDIR", "matplotlib"),
        ("JUPYTER_RUNTIME_DIR", "runtime"),
    ]:
        directory = output / name
        directory.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(directory)
    os.environ["JUPYTER_PATH"] = str(ROOT / ".venv/share/jupyter")
    os.environ["POLARS_FMT_MAX_ROWS"] = "-1"
    os.environ["POLARS_FMT_MAX_COLS"] = "-1"
    os.environ["POLARS_FMT_STR_LEN"] = "1024"
    (Path(os.environ["MPLCONFIGDIR"]) / "matplotlibrc").write_text(
        "figure.autolayout: True\n"
    )
    if os.name == "nt":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    inputs = {
        p.relative_to(ROOT).as_posix(): sha(p)
        for name in ["phase3", "phase4", "mlp_source"]
        for p in sorted((ROOT / "reports" / name).iterdir())
        if p.suffix in {".csv", ".json"} and p.name not in {"notebook_validation.json"}
    }
    records = []
    for name in ["03_classical_models_analysis.ipynb", "04_weather_mlp_analysis.ipynb"]:
        path = ROOT / "notebooks" / name
        source = nbformat.read(path, as_version=4)
        code = [cell for cell in source.cells if cell.cell_type == "code"]
        assert all(not cell.outputs and cell.execution_count is None for cell in code)
        assert all(
            ".fit(" not in cell.source and "fit_transform" not in cell.source
            for cell in code
        )
        executed = NotebookClient(
            source,
            timeout=180,
            kernel_name=kernel,
            resources={"metadata": {"path": str(checkout / "notebooks")}},
        ).execute()
        errors = [
            item
            for cell in executed.cells
            for item in cell.get("outputs", [])
            if item.output_type == "error"
        ]
        assert not errors
        target = output / name.replace(".ipynb", ".executed.ipynb")
        target.write_bytes(nbformat.writes(executed).encode("utf-8"))
        images = sum(
            "image/png" in item.get("data", {})
            for cell in executed.cells
            for item in cell.get("outputs", [])
        )
        records.append(
            dict(
                notebook=path.relative_to(ROOT).as_posix(),
                source_sha256=sha(path),
                executed_sha256=sha(target),
                code_cells=len(code),
                figures=images,
                tracebacks=0,
            )
        )
        print(
            f"Executed {name}: {len(code)} code cells, {images} figures, no errors",
            flush=True,
        )
    coverage = dict(
        status="passed",
        contains_private_input_tree=False,
        experiments=dict(
            phase3_development_fits=367,
            phase3_final_fits=13,
            original_phase4_final_fits=14,
            source_input_final_fits=24,
        ),
        aggregate_rows=expected_rows,
        notebooks=records,
        aggregate_input_sha256=inputs,
        markdown_input_sha256_lf={
            p.relative_to(ROOT).as_posix(): hashlib.sha256(
                p.read_bytes().replace(b"\r\n", b"\n")
            ).hexdigest()
            for name in ["phase3", "phase4", "mlp_source"]
            for p in sorted((ROOT / "reports" / name).glob("*.md"))
        },
        scope="All completed experiment families; source outputs cleared; full aggregate tables and saved fit-level/class tables",
    )
    target = ROOT / "reports/analysis_notebook_coverage.json"
    target.write_bytes((json.dumps(coverage, indent=2, sort_keys=True) + "\n").encode())
    phase3 = ROOT / "reports/phase3/notebook_validation.json"
    evidence = json.loads(phase3.read_text(encoding="utf-8"))
    evidence.update(
        final_source_sha256=records[0]["source_sha256"],
        executed_notebook_sha256=records[0]["executed_sha256"],
        cross_phase_notebook_coverage="reports/analysis_notebook_coverage.json",
    )
    phase3.write_bytes((json.dumps(evidence, indent=2) + "\n").encode())
    phase4 = ROOT / "reports/phase4/notebook_validation.json"
    previous = json.loads(phase4.read_text(encoding="utf-8"))
    if "source_commit" in previous:
        previous["historical_source_commit"] = previous.pop("source_commit")
    previous.update(
        status="passed",
        input_notebook_sha256=records[1]["source_sha256"],
        executed_notebook_sha256=records[1]["executed_sha256"],
        executed_code_cells=records[1]["code_cells"],
        rendered_figures=records[1]["figures"],
        tracebacks=0,
        cross_phase_notebook_coverage="reports/analysis_notebook_coverage.json",
        includes_source_input_followup=True,
    )
    phase4.write_bytes((json.dumps(previous, indent=2, sort_keys=True) + "\n").encode())
    return coverage


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel-name", default="flight-delay-phase3")
    args = parser.parse_args()
    validate(args.kernel_name)
