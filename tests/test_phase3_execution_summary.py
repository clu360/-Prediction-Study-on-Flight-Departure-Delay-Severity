"""Aggregate reporting counts unique fits and exposes every recorded warning."""

from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "execution_summary",
    Path(__file__).resolve().parents[1] / "scripts/summarize_phase3_execution.py",
)
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


def record(model="logistic", candidate="a", warnings=None, transformer="train-only"):
    return dict(
        model=model,
        feature_view="G1",
        scored_month="2025-04",
        selected_config_id=candidate,
        transformer_hash=transformer,
        model_fit_id=model + candidate,
        model_hash="learned",
        train_ids_hash="digest-only",
        warnings=warnings or [],
        resources=dict(fit_seconds=2.0, peak_rss_bytes=100),
    )


def test_summary_counts_all_candidates_but_final_fit_only_once():
    warning = "ConvergenceWarning: max_iter reached"
    development = [
        record(warnings=[warning]),
        record(candidate="b"),
        record(model="random_forest"),
    ]
    final = [
        record(warnings=[warning], transformer="final-train"),
        record(model="random_forest", transformer="final-train"),
    ]
    for row in final:
        row.pop("scored_month")
        row.pop("selected_config_id")
    result = summary.summarize_records(development, final)
    assert result["fit_counts"] == dict(development=3, frozen_final=2, total=5)
    assert result["total_unique_fit_seconds"] == 10
    assert result["fits_with_warnings"] == 2
    assert result["warning_counts"] == {warning: 2}
    assert len(result["preprocessing_hashes"]) == 2
    assert len(result["frozen_fit_hashes"]) == 2
    assert all("target_record_id" not in row for row in result["fit_warnings"])


def test_summary_rejects_duplicate_units_and_inconsistent_preprocessing():
    row = record()
    with pytest.raises(ValueError, match="Duplicate fitted unit"):
        summary.summarize_records([row, deepcopy(row)], [])
    with pytest.raises(ValueError, match="Preprocessing states disagree"):
        summary.summarize_records(
            [row, record(candidate="b", transformer="different")], []
        )
