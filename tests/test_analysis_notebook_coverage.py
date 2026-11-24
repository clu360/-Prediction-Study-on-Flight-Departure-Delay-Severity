import importlib.util
import json
from pathlib import Path


def test_phase4_saved_tables_cover_all_completed_fit_and_class_results():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "phase4_builder", root / "scripts/phase4_build_notebook.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    notebook = module.build()
    tracked = json.loads(
        (root / "notebooks/04_weather_mlp_analysis.ipynb").read_text(encoding="utf-8")
    )
    assert ["".join(cell["source"]) for cell in tracked["cells"]] == [
        cell.source for cell in notebook.cells
    ]
    saved = notebook.cells[0].source
    assert [cell.id for cell in notebook.cells] == [f"phase4-{i:02}" for i in range(8)]
    assert len([cell for cell in notebook.cells if cell.cell_type == "code"]) == 4
    assert "post-holdout" in saved or "previously examined" in saved
    for relative, columns in [
        (
            "reports/phase4/metrics_summary.csv",
            [
                "arm",
                "seed",
                "month",
                "n",
                "accuracy",
                "macro_f1",
                "log_loss",
                "severe_precision",
                "severe_recall",
                "severe_pr_auc_ap",
            ],
        ),
        (
            "reports/mlp_source/metrics_by_seed.csv",
            [
                "regime",
                "arm",
                "seed",
                "month",
                "n",
                "accuracy",
                "macro_f1",
                "log_loss",
                "severe_precision",
                "severe_recall",
                "severe_pr_auc_ap",
                "delta_macro_f1",
            ],
        ),
    ]:
        rows = module.csv_records(relative)
        if relative.startswith("reports/phase4"):
            rows = [row for row in rows if row["subgroup"] == "overall"]
            assert len(rows) == 42
        else:
            assert len(rows) == 72
        for row in rows:
            expected = module.saved_table([row], columns).splitlines()[-1]
            assert expected in saved
    for relative, source, expected in [
        ("reports/phase4/metric_details.json", False, 126),
        ("reports/mlp_source/metric_details.json", True, 216),
    ]:
        rows = module.class_records(
            json.loads((root / relative).read_text(encoding="utf-8")), source
        )
        assert len(rows) == expected
        columns = [
            "regime",
            "arm",
            "seed",
            "month",
            "severity",
            "precision",
            "recall",
            "f1",
            "support",
        ]
        for row in rows:
            assert module.saved_table([row], columns).splitlines()[-1] in saved
    assert module.build() == notebook
