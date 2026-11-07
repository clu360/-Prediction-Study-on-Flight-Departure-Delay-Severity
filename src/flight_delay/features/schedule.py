"""Schedule information available before prediction."""
import math
import polars as pl
from .views import G1, G1_MAP


def schedule_features(cohort: pl.DataFrame) -> pl.DataFrame:
    hour = pl.col("target_crs_dep_local_hour") * (2 * math.pi / 24)
    return cohort.select("target_record_id", *[pl.col(source).alias(name) for name, source in G1_MAP.items()],
        hour.sin().alias("g1_dep_hour_sin"), hour.cos().alias("g1_dep_hour_cos")).select("target_record_id", *G1)
