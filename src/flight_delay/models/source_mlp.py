"""Minimal numeric adaptation and learned embeddings for the causal source columns."""

from dataclasses import dataclass
from functools import partial
import os

import numpy as np
import polars as pl
import torch
from torch import nn

from ..features.views import CATEGORICAL

# Fixed unit conversions chosen before development scoring, shared by all folds.
DIVISORS = {
    "g1_dep_hour_sin": 1, "g1_dep_hour_cos": 1,
    "g1_crs_elapsed_min": 60, "g1_distance_miles": 1000,
    "g2_prior_dep_delay_min": 60, "g2_prior_crs_elapsed_min": 60,
    "g2_scheduled_turnaround_min": 60, "g2_minutes_until_crs_dep_at_prediction": 60,
    "g3_airport_moderate_rate_28d": 1, "g3_airport_severe_rate_28d": 1,
    "g3_airport_mean_dep_delay_28d": 60, "g3_airport_scheduled_volume_7d": 1000,
    "g3_airline_airport_severe_rate_28d": 1, "g3_airline_airport_count_28d": 1000,
    "g4_origin_worst_category_180m": 1, "g4_origin_frozen_precip_180m": 1,
    "g4_origin_thunderstorm_60m": 1, "g4_origin_wind_max_kt": 25,
    "g4_dest_worst_category_180m": 1,
}
EMBEDDING_WIDTH = 8


@dataclass(frozen=True)
class SourceAdapter:
    numeric: tuple
    categorical: tuple
    vocabularies: tuple

    @classmethod
    def fit(cls, columns, train):
        numeric = tuple(name for name in columns if name not in CATEGORICAL)
        categorical = tuple(name for name in columns if name in CATEGORICAL)
        if set(numeric) - DIVISORS.keys() or not train.height:
            raise ValueError("Missing numeric unit conversion or empty training rows")
        vocabularies = tuple(
            tuple(train[name].cast(pl.String).drop_nulls().unique().sort())
            for name in categorical
        )
        return cls(numeric, categorical, vocabularies)

    @property
    def cardinality(self):
        return 1 + sum(1 + len(vocabulary) for vocabulary in self.vocabularies)

    @property
    def input_width(self):
        return 2 * len(self.numeric) + len(self.categorical)

    def transform(self, frame):
        matrix = np.empty((frame.height, self.input_width), dtype=np.float32)
        for i, name in enumerate(self.numeric):
            values = frame[name].cast(pl.Float32).to_numpy()
            missing = frame[name].is_null().to_numpy()
            if not np.isfinite(values[~missing]).all():
                raise ValueError(f"Nonfinite source value in {name}")
            matrix[:, i] = np.where(missing, 0, values / DIVISORS[name])
            matrix[:, i + len(self.numeric)] = missing
        offset = 1
        for i, (name, vocabulary) in enumerate(zip(self.categorical, self.vocabularies)):
            mapping = {value: offset + index + 1 for index, value in enumerate(vocabulary)}
            values = pl.col(name).cast(pl.String)
            known = values.replace_strict(mapping, default=0, return_dtype=pl.Int32) if mapping else pl.lit(0)
            matrix[:, 2 * len(self.numeric) + i] = frame.select(
                pl.when(values.is_null()).then(offset).otherwise(known)
            ).to_series().to_numpy()
            offset += 1 + len(vocabulary)
        return matrix

    def factory(self):
        return partial(create, cardinality=self.cardinality, categories=len(self.categorical))


class EmbeddedInput(nn.Module):
    def __init__(self, cardinality, numeric_width):
        super().__init__()
        self.numeric_width = numeric_width
        self.embedding = nn.Embedding(cardinality, EMBEDDING_WIDTH, padding_idx=0)

    def forward(self, packed):
        numeric = packed[:, :self.numeric_width]
        embeds = self.embedding(packed[:, self.numeric_width:].long()).flatten(1)
        return torch.cat((numeric, embeds), dim=1)


def create(input_width, candidate, seed, threads, device, *, cardinality, categories):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    numeric_width = input_width - categories
    first, second = candidate["widths"]
    return nn.Sequential(
        EmbeddedInput(cardinality, numeric_width),
        nn.Linear(numeric_width + categories * EMBEDDING_WIDTH, first),
        nn.ReLU(), nn.Dropout(candidate["dropout"]),
        nn.Linear(first, second), nn.ReLU(), nn.Linear(second, 3),
    ).to(device)
