from pathlib import Path

import polars as pl
import pytest
import yaml

from flight_delay.experiments.phase4_inputs import BASE, ID, WEATHER, Inputs
from flight_delay.experiments.results import write_frame
from flight_delay.experiments.synthetic import make_fixture


@pytest.fixture
def phase4_inputs(tmp_path):
    root = Path.cwd()
    make_fixture(tmp_path, rows_per_month=9)
    config = {
        "features_dir": str(tmp_path / "features"),
        "processed_dir": str(tmp_path / "processed"),
        "weather_dir": str(tmp_path / "weather"),
        "output_dir": str(tmp_path / "outputs"),
        "reports_dir": str(tmp_path / "reports"),
        "weather": str(root / "configs/g4_weather.yaml"),
        "mlp": str(root / "configs/phase4_mlp.yaml"),
    }
    path = tmp_path / "phase4.yaml"
    path.write_text(yaml.safe_dump(config))
    return Inputs(path)


def test_weather_join_retains_all_null_rows_and_rejects_missing_id(phase4_inputs):
    inputs = phase4_inputs
    base = inputs.rows(11, "test", BASE)
    path = inputs.path("weather_dir") / "g4_features/year=2025/month=11/part-test.parquet"
    weather = base.select(ID).reverse().with_columns([
        pl.lit(None, dtype=pl.Float32).alias(column) for column in inputs.weather["features"]
    ])
    write_frame(path, weather)
    paired = inputs.rows(11, "test", WEATHER)
    assert paired.select(base.columns).equals(base)
    assert paired.select(inputs.weather["features"]).null_count().row(0) == (base.height,) * 5
    write_frame(path, weather.head(weather.height - 1))
    with pytest.raises(ValueError, match="population"):
        inputs.rows(11, "test", WEATHER)

