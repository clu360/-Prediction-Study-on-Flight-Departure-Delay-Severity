"""Causal range queries on sorted departure events, with strict upper bounds.

Prefix arrays are a vectorized equivalent of chronologically updated rolling
counts. searchsorted(side='left') at both boundaries implements [t-window,t).
No historical row is revisited for each target, and no month-wide rate is used.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import polars as pl
from .config import FeatureConfig
from .views import G3

DAY = 86_400_000_000


@dataclass
class PrefixHistory:
    times: np.ndarray
    moderate: np.ndarray
    severe: np.ndarray
    delay: np.ndarray | None = None

    @classmethod
    def from_events(cls, events: pl.DataFrame, with_delay: bool = True) -> "PrefixHistory":
        ordered = events.filter(pl.col("valid_observation")).sort("observed_departure_time_utc")
        times = ordered["observed_departure_time_utc"].dt.timestamp("us").to_numpy()
        severity = ordered["departure_severity"].to_numpy()
        prefix = lambda x: np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(x, dtype=np.int64)])
        delay = np.concatenate([np.zeros(1), np.cumsum(ordered["dep_delay"].to_numpy(), dtype=np.float64)]) if with_delay else None
        return cls(times, prefix(severity == 1), prefix(severity == 2), delay)

    def window(self, times: np.ndarray, days: int) -> dict[str, np.ndarray]:
        lo = np.searchsorted(self.times, times-days*DAY, side="left")
        hi = np.searchsorted(self.times, times, side="left")
        return dict(n=hi-lo, mod=self.moderate[hi]-self.moderate[lo], sev=self.severe[hi]-self.severe[lo],
            total=self.delay[hi]-self.delay[lo] if self.delay is not None else np.zeros(len(times)), hi=hi)


def history_features(queries: pl.DataFrame, events: pl.DataFrame, schedule: pl.DataFrame,
                     global_history: PrefixHistory, cfg: FeatureConfig) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Generate G3 plus separate audit metadata; support multiple query airports."""
    q = queries.sort(["prediction_time_utc", "target_record_id"]).with_row_index("_row")
    t = q["prediction_time_utc"].dt.timestamp("us").to_numpy()
    global_stats = global_history.window(t, cfg.history_days)
    n = q.height
    airport_n, cell_n, airport_m, airport_s, cell_s = [np.zeros(n, dtype=np.int64) for _ in range(5)]
    delay_sum = np.zeros(n)
    volume = np.zeros(n, dtype=np.int64)
    valid_events = events.filter(pl.col("valid_observation"))
    # Explicitly remove a target's own observed outcome if a retained Phase I
    # anomaly places it before prediction. This preserves the sample, not its label in X.
    own = q.select("target_record_id").join(valid_events.select(
        pl.col("record_id").alias("target_record_id"),
        pl.col("observed_departure_time_utc").alias("_own_time"),
        pl.col("dep_delay").alias("_own_delay"), pl.col("departure_severity").alias("_own_class")),
        on="target_record_id", how="left", validate="1:1", maintain_order="left")
    own_time = own["_own_time"].dt.timestamp("us").fill_null(np.iinfo(np.int64).max).to_numpy()
    own_in = (own_time < t) & (own_time >= t-cfg.history_days*DAY)
    own_class = own["_own_class"].fill_null(-1).to_numpy()
    own_delay = own["_own_delay"].fill_null(0).to_numpy()
    for key, group in q.partition_by("target_origin", as_dict=True).items():
        airport = key[0]
        positions = group["_row"].to_numpy()
        ev = valid_events.filter(pl.col("origin") == airport)
        stats = PrefixHistory.from_events(ev).window(t[positions], cfg.history_days)
        airport_n[positions], airport_m[positions], airport_s[positions], delay_sum[positions] = (
            stats["n"], stats["mod"], stats["sev"], stats["total"])
        scheduled = schedule.filter((pl.col("origin") == airport) & pl.col("scheduled_departure_utc").is_not_null())
        st = scheduled["scheduled_departure_utc"].dt.timestamp("us").sort().to_numpy()
        volume[positions] = np.searchsorted(st, t[positions], side="left") - np.searchsorted(
            st, t[positions]-cfg.schedule_days*DAY, side="left")
        for carrier_key, carrier_queries in group.partition_by("target_carrier", as_dict=True).items():
            pos = carrier_queries["_row"].to_numpy()
            cell = PrefixHistory.from_events(ev.filter(pl.col("reporting_airline") == carrier_key[0]), with_delay=False)
            counts = cell.window(t[pos], cfg.history_days)
            cell_n[pos], cell_s[pos] = counts["n"], counts["sev"]
    global_stats["n"] -= own_in
    global_stats["mod"] -= own_in & (own_class == 1)
    global_stats["sev"] -= own_in & (own_class == 2)
    airport_n -= own_in
    airport_m -= own_in & (own_class == 1)
    airport_s -= own_in & (own_class == 2)
    cell_n -= own_in
    cell_s -= own_in & (own_class == 2)
    delay_sum -= np.where(own_in, own_delay, 0)
    divide = lambda a, b: np.divide(a, b, out=np.full(n, np.nan), where=b > 0)
    global_mod = divide(global_stats["mod"], global_stats["n"])
    global_sev = divide(global_stats["sev"], global_stats["n"])
    airport_mod = (airport_m + cfg.alpha*global_mod)/(airport_n+cfg.alpha)
    airport_sev = (airport_s + cfg.alpha*global_sev)/(airport_n+cfg.alpha)
    airline_sev = (cell_s + cfg.alpha*airport_sev)/(cell_n+cfg.alpha)
    features = pl.DataFrame({"target_record_id": q["target_record_id"],
        G3[0]: airport_mod, G3[1]: airport_sev, G3[2]: divide(delay_sum, airport_n),
        G3[3]: volume, G3[4]: airline_sev, G3[5]: cell_n}).with_columns(pl.col(pl.Float64).fill_nan(None))
    max_times = np.full(n, np.iinfo(np.int64).min, dtype=np.int64)
    has_global = global_stats["n"] > 0
    if len(global_history.times):
        max_times[has_global] = global_history.times[global_stats["hi"][has_global]-1]
    coverage = np.minimum(cfg.history_days, np.maximum(0, (t-global_history.times[0])/DAY)) if len(global_history.times) else np.zeros(n)
    metadata = pl.DataFrame({"target_record_id": q["target_record_id"],
        "global_history_count_28d": global_stats["n"], "airport_history_count_28d": airport_n,
        "history_available_days": coverage, "no_global_history": ~has_global,
        "no_airport_history": airport_n == 0, "no_airline_airport_history": cell_n == 0,
        "own_outcome_excluded": own_in,
        "history_max_event_utc": pl.Series(max_times, dtype=pl.Int64).cast(pl.Datetime("us", "UTC"))})
    metadata = metadata.with_columns(pl.when(pl.col("no_global_history")).then(None)
        .otherwise(pl.col("history_max_event_utc")).alias("history_max_event_utc"))
    return features, metadata
