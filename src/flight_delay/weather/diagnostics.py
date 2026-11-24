"""Persist per-target weather coverage and age outside the predictor table."""

from time import sleep

import numpy as np
import polars as pl

from .features import monthly_queries
from .fetch import Archive, bounds
from .metar import decode
from .stations import load_map
from ..experiments.config import file_hash
from ..experiments.phase4_inputs import ID
from ..experiments.results import exclusive, read_json, write_frame, write_json

MINUTE = np.timedelta64(1, "m")


def audit_windows(prediction, available):
    right = np.searchsorted(available, prediction, side="left")
    counts = tuple(
        right - np.searchsorted(available, prediction - window * MINUTE, side="left")
        for window in (180, 60)
    )
    age = np.full(len(prediction), np.nan, dtype=np.float64)
    observed = right > 0
    age[observed] = (
        (prediction[observed] - available[right[observed] - 1]) / MINUTE + 10
    )
    return counts, age


def read_queries(inputs, month):
    for attempt in range(4):
        try:
            return monthly_queries(inputs, month)
        except OSError as error:
            sharing = isinstance(error, PermissionError) or "os error 32" in str(error)
            if not sharing or attempt == 3:
                raise
            print(f"Weather diagnostic month {month:02}: retrying a source read lock", flush=True)
            sleep(3 * (attempt + 1))


def coverage_diagnostics(inputs, version):
    weather_root = inputs.path("weather_dir")
    verified = read_json(weather_root / "audit" / "verified.json")
    root = weather_root / "audit" / "target_windows"
    manifest = root / "verified.json"
    with exclusive(root):
        if manifest.exists():
            saved = read_json(manifest)
            if saved["g4_signature"] != verified["signature"] or any(
                file_hash(root / name) != checksum for name, checksum in saved["files"].items()
            ):
                raise ValueError("Stale per-target weather diagnostic checkpoint")
            return saved
        archive = Archive(inputs.path("raw_weather_dir"), *bounds(inputs.weather), version)
        codes = load_map(inputs.path("station_map"))
        stations = {row["iata"]: row["station"] for row in codes}
        observation_times = {}
        for code, station in stations.items():
            path, _ = archive.files(station)
            if file_hash(path) != verified["context"]["raw"][code]:
                raise ValueError(f"Source changed after verified G4 build: {code}")
            reports, _ = decode(path.read_bytes(), station, archive.start, archive.end,
                                routine_and_special_only=True)
            observation_times[code] = np.array(
                [report[0].replace(tzinfo=None) for report in reports], dtype="datetime64[us]"
            )
        files, aggregate = {}, {}
        for month in range(1, 13):
            query = read_queries(inputs, month)
            times = query["prediction_time_utc"].to_numpy()
            columns = []
            for source, label in (("g1_origin", "origin"), ("g1_dest", "destination")):
                locations = query[source].to_numpy()
                order = np.argsort(locations, kind="stable")
                ordered = locations[order]
                n180, n60 = np.zeros(query.height, dtype=np.uint16), np.zeros(query.height, dtype=np.uint16)
                age = np.full(query.height, np.nan)
                for indices in np.split(order, np.flatnonzero(ordered[1:] != ordered[:-1]) + 1):
                    (n180[indices], n60[indices]), age[indices] = audit_windows(
                        times[indices], observation_times[locations[indices[0]]]
                    )
                if np.any(age[np.isfinite(age)] <= 10):
                    raise ValueError("Diagnostic age violates strict availability")
                columns.extend([
                    pl.col(source).replace_strict(stations).alias(f"{label}_station"),
                    pl.Series(f"{label}_reports_180m", n180),
                    pl.Series(f"{label}_reports_60m", n60),
                    pl.Series(f"{label}_latest_observation_age_min", age, nan_to_null=True),
                ])
            part = f"year=2025/month={month:02}/part-00.parquet"
            feature_path = f"g4_features/{part}"
            if file_hash(weather_root / feature_path) != verified["files"][feature_path]:
                raise ValueError("Weather feature bytes changed after validation")
            features = pl.read_parquet(weather_root / feature_path)
            if not query[ID].equals(features[ID]):
                raise ValueError("Diagnostic rows differ from verified feature IDs")
            diagnostic = query.select(ID, *columns).with_columns([
                features[name].is_null().alias(f"missing_{name}") for name in inputs.weather["features"]
            ])
            for field in ("origin", "destination"):
                if diagnostic.filter(pl.col(f"{field}_reports_60m") > pl.col(f"{field}_reports_180m")).height:
                    raise ValueError("Diagnostic weather window nesting failed")
            write_frame(root / part, diagnostic)
            files[part] = file_hash(root / part)
            aggregate[str(month)] = dict(
                rows=diagnostic.height,
                origin_no_report_180m=diagnostic.filter(pl.col("origin_reports_180m") == 0).height,
                destination_no_report_180m=diagnostic.filter(pl.col("destination_reports_180m") == 0).height,
            )
            print(f"Weather diagnostic month {month:02}: {diagnostic.height} IDs verified", flush=True)
        result = dict(g4_signature=verified["signature"], months=aggregate, files=files,
                      schema={name: str(dtype) for name, dtype in diagnostic.schema.items()},
                      code_sha256=file_hash(__file__))
        write_json(manifest, result)
        return result
