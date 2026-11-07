"""Reuse Phase I incoming-aircraft values without any predecessor search."""
import polars as pl
from .views import G2_MAP


def incoming_features(cohort: pl.DataFrame) -> pl.DataFrame:
    return cohort.select("target_record_id", *[pl.col(source).alias(name) for name, source in G2_MAP.items()])
