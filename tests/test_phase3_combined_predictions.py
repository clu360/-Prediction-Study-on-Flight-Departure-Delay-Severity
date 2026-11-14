"""The consolidated private artifact must preserve every monthly prediction value."""

import importlib.util
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest


def verifier():
    spec = importlib.util.spec_from_file_location(
        "verify_phase3_results",
        Path(__file__).resolve().parents[1] / "scripts/verify_phase3_results.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("damage", [None, "value", "rows", "schema", "extra_source"])
def test_combined_predictions_match_all_monthly_values_across_batches(tmp_path, damage):
    directory = tmp_path / "final/predictions"
    directory.mkdir(parents=True)
    first = pa.table({"target_record_id": range(17), "p_severe": [0.2] * 17})
    second = pa.table({"target_record_id": range(17, 50024), "p_severe": [0.3] * 50007})
    sources = [directory / "a.parquet", directory / "b.parquet"]
    for path, table in zip(sources, [first, second]):
        pq.write_table(table, path)
    combined = pa.concat_tables([first, second])
    if damage == "value":
        probabilities = combined["p_severe"].to_pylist()
        probabilities[-1] = 0.9
        combined = combined.set_column(1, "p_severe", pa.array(probabilities))
    elif damage == "rows":
        combined = combined.slice(0, combined.num_rows - 1)
    elif damage == "schema":
        combined = combined.rename_columns(["target_record_id", "wrong_probability"])
    elif damage == "extra_source":
        pq.write_table(first, directory / "unexpected.parquet")
    # Deliberately use different row-group and file boundaries from the originals.
    pq.write_table(
        combined, tmp_path / "final/predictions.parquet", row_group_size=20003
    )
    check = verifier().check_combined_predictions
    if damage:
        with pytest.raises(ValueError, match="Combined prediction"):
            check(tmp_path, sources, 50024)
    else:
        evidence = check(tmp_path, sources, 50024)
        assert evidence["rows"] == 50024 and evidence["monthly_files"] == 2
        assert len(evidence["sha256"]) == 64
