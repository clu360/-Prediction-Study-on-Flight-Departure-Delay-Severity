"""Checked, ordered feature contracts. Never infer X by dropping label columns."""
from pathlib import Path
import yaml

G1_MAP = {
    "g1_carrier": "target_carrier", "g1_origin": "target_origin", "g1_dest": "target_dest",
    "g1_day_of_week": "target_day_of_week", "g1_month": "target_month",
    "g1_crs_elapsed_min": "target_crs_elapsed_min", "g1_distance_miles": "target_distance_miles",
}
G1 = ["g1_carrier", "g1_origin", "g1_dest", "g1_dep_hour_sin", "g1_dep_hour_cos",
      "g1_day_of_week", "g1_month", "g1_crs_elapsed_min", "g1_distance_miles"]
G2_MAP = {
    "g2_prior_origin": "prior_origin", "g2_prior_dep_delay_min": "prior_dep_delay_min",
    "g2_prior_crs_elapsed_min": "prior_crs_elapsed_min",
    "g2_scheduled_turnaround_min": "scheduled_turnaround_min",
    "g2_minutes_until_crs_dep_at_prediction": "minutes_until_target_crs_dep_at_prediction",
}
G2 = list(G2_MAP)
G3 = ["g3_airport_moderate_rate_28d", "g3_airport_severe_rate_28d", "g3_airport_mean_dep_delay_28d",
      "g3_airport_scheduled_volume_7d", "g3_airline_airport_severe_rate_28d", "g3_airline_airport_count_28d"]
ALL = G1 + G2 + G3
VIEWS = {"G1": G1, "G1_G2": G1+G2, "G1_G3": G1+G3, "G1_G2_G3": ALL}
CATEGORICAL = ["g1_carrier", "g1_origin", "g1_dest", "g1_day_of_week", "g1_month", "g2_prior_origin"]


def load_views(path: Path | None = None) -> dict[str, list[str]]:
    views = yaml.safe_load(path.read_text(encoding="utf-8")) if path else VIEWS
    if views != VIEWS or len(ALL) != len(set(ALL)) or len(ALL) != 20:
        raise ValueError("Feature manifest must contain exactly the locked four ordered views")
    return {k: list(v) for k, v in views.items()}
