"""Deterministic tiny fixtures. NEVER a source of study performance claims."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np
import polars as pl
from ..features.views import ALL
from .config import digest
from .results import write_json, write_frame


def make_fixture(root, rows_per_month=18):
    root = Path(root)
    rng = np.random.default_rng(6740)
    memberships = {m: [] for m in range(1, 13)}
    for month in range(1, 13):
        rows = []
        labels = []
        meta = []
        for i in range(rows_per_month):
            target_id = f"SYNTHETIC-2025-{month:02}-{i:05}"
            label = i % 3
            origin = ["ATL", "DFW", "DEN", "ORD", "LAX"][i % 5]
            time = datetime(2025, month, 2, tzinfo=timezone.utc) + timedelta(
                minutes=10 * i
            )
            row = dict(
                g1_carrier=["AA", "BB"][i % 2],
                g1_origin=origin,
                g1_dest=["DFW", "ATL"][i % 2],
                g1_dep_hour_sin=float(np.sin(i)),
                g1_dep_hour_cos=float(np.cos(i)),
                g1_day_of_week=(i % 7) + 1,
                g1_month=month,
                g1_crs_elapsed_min=float(60 + i),
                g1_distance_miles=float(300 + i * 3),
                g2_prior_origin=["ORD", "ATL"][i % 2],
                g2_prior_dep_delay_min=float(label * 20 + rng.normal()),
                g2_prior_crs_elapsed_min=float(50 + i),
                g2_scheduled_turnaround_min=float(30 + i),
                g2_minutes_until_crs_dep_at_prediction=float(90 + i),
                g3_airport_moderate_rate_28d=0.1,
                g3_airport_severe_rate_28d=0.05,
                g3_airport_mean_dep_delay_28d=None if i == 0 else float(i),
                g3_airport_scheduled_volume_7d=100 + i,
                g3_airline_airport_severe_rate_28d=0.06,
                g3_airline_airport_count_28d=20 + i,
            )
            rows.append(dict(target_record_id=target_id, **{c: row[c] for c in ALL}))
            labels.append(
                dict(
                    target_record_id=target_id,
                    target_class=label,
                    target_dep_delay_min=[-5.0, 25.0, 75.0][label],
                )
            )
            meta.append(
                dict(
                    target_record_id=target_id,
                    target_origin=origin,
                    prediction_time_utc=time,
                    history_max_event_utc=time - timedelta(minutes=1),
                    flight_date=time.date(),
                    source_month=month,
                )
            )
            for test_month in range(4, 13):
                if test_month - 3 <= month < test_month or month == test_month:
                    memberships[month].append(
                        dict(
                            target_record_id=target_id,
                            source_month=month,
                            fold_id=f"2025-{test_month:02}",
                            role="test" if month == test_month else "train",
                            test_month=test_month,
                            is_final_holdout=test_month >= 11,
                        )
                    )
        for table, data in [
            ("full_features", rows),
            ("targets", labels),
            ("metadata", meta),
        ]:
            write_frame(
                root
                / "features"
                / table
                / "year=2025"
                / f"month={month:02}"
                / "part-synthetic.parquet",
                pl.DataFrame(data),
            )
    for month, rows in memberships.items():
        write_frame(
            root
            / "processed/folds/year=2025"
            / f"month={month:02}"
            / "part-synthetic.parquet",
            pl.DataFrame(rows),
        )
    return digest(
        {"kind": "synthetic_not_study", "seed": 6740, "rows_per_month": rows_per_month}
    )


def fixture_config(cfg, relative_root, rows_per_month=18):
    relative_root = Path(relative_root)
    values = dict(
        cfg.values,
        version="phase3-synthetic-smoke-not-study",
        processed_dir=str(relative_root / "processed"),
        features_dir=str(relative_root / "features"),
        output_dir=str(relative_root / "outputs"),
        reports_dir=str(relative_root / "aggregate_reports"),
        max_wall_seconds=1200,
        min_free_disk_gib=0.05,
    )
    grids = {
        name: [dict(params)]
        for name, (params) in [(name, grid[0]) for name, grid in cfg.grids.items()]
    }
    grids["logistic"][0].update(max_iter=100)
    grids["random_forest"][0].update(n_estimators=3, max_depth=2, min_samples_leaf=2)
    grids["xgboost"][0].update(n_estimators=3, max_depth=2)
    configured = replace(cfg, values=values, grids=grids)
    fingerprint = make_fixture(cfg.root / relative_root, rows_per_month)
    gate = dict(passed=True, input_digest=fingerprint, data_kind="synthetic_not_study")
    return configured, gate


def smoke(cfg, resume=False):
    from .runner import tune, frozen_final, context
    from .results import ResultStore
    from .config import code_fingerprints

    relative = (
        Path(cfg["output_dir"])
        / "_synthetic_smoke"
        / digest(code_fingerprints(cfg.root))[:16]
    )
    configured, gate = fixture_config(cfg, relative, cfg["smoke_rows_per_month"])
    store = ResultStore(
        configured.path("output_dir"), context(configured, gate), resume
    )
    if not (resume and (store.root / "final/selection_lock.json").exists()):
        store = tune(configured, resume, gate=gate)
    store = frozen_final(
        configured, store.root / "development/selected_configs.json", resume, gate=gate
    )
    result = dict(
        status="synthetic_smoke_complete",
        study_results=False,
        run_id=store.run_id,
        models=cfg["models"],
        views=cfg["views"],
        train_months=[8, 9, 10],
        scored_months=[11, 12],
    )
    write_json(cfg.path("output_dir") / "synthetic_smoke.json", result)
    return result
