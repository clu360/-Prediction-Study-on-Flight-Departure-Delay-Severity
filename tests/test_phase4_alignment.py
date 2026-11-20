import polars as pl
import pytest

from flight_delay.experiments.phase4_inputs import BASE, ID, WEATHER
from flight_delay.experiments.results import write_frame



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
