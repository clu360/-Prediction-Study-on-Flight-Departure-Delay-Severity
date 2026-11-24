"""Read and fingerprint locked Phase III populations for Phase IV."""

from collections import Counter
import hashlib
import json
from pathlib import Path

import polars as pl
import yaml

from .config import digest, file_hash
from .inputs import checked_rows, memberships, table_scan
from .results import read_json, write_json
from ..features.transforms import FoldTransformer
from ..features.views import VIEWS

ID = "target_record_id"
BASE = "G1_G2_G3"
WEATHER = "G1_G2_G3_G4"
DEV = range(4, 11)
FINAL_TRAIN = [8, 9, 10]
HOLDOUT = [11, 12]


class Inputs:
    def __init__(self, config):
        self.config_path = Path(config)
        self.settings = yaml.safe_load(self.config_path.read_text())
        self.mlp = yaml.safe_load(self.path("mlp").read_text())
        self.weather = yaml.safe_load(self.path("weather").read_text())
        self.columns = {BASE: VIEWS[BASE], WEATHER: VIEWS[BASE] + self.weather["features"]}
        if len(set(self.columns[WEATHER])) != 25:
            raise ValueError("G4 allowlist must extend the locked 20-column view by five")

    def path(self, key):
        return Path(self.settings[key])

    def rows(self, fold_month, role, view=BASE, months=None):
        fold = f"2025-{fold_month:02}"
        if role == "train" and fold_month not in (*DEV, 11):
            raise ValueError("This fold cannot enter training")
        expected = list(range(fold_month - 3, fold_month)) if role == "train" else [fold_month]
        months = months or expected
        if not set(months).issubset(expected):
            raise ValueError("Requested months outside locked fold role")
        ids = memberships(self.path("processed_dir"), fold, role)
        if months != expected:
            ids = ids.join(
                table_scan(self.path("processed_dir"), "folds", months)
                .filter((pl.col("fold_id") == fold) & (pl.col("role") == role))
                .select(ID).collect(),
                on=ID, how="semi",
            )
        frame = checked_rows(self, ids, months, BASE)
        if view == BASE:
            return frame
        if view != WEATHER:
            raise ValueError(f"Invalid Phase IV view {view}")
        weather = table_scan(self.path("weather_dir"), "g4_features", months).collect()
        if weather.columns != [ID, *self.weather["features"]]:
            raise ValueError("Unexpected G4 fields")
        if weather.height != frame.height or weather[ID].n_unique() != frame.height:
            raise ValueError("Unexpected G4 population")
        if frame.select(ID).join(weather.select(ID), on=ID, how="anti").height:
            raise ValueError("Missing G4 IDs")
        return frame.join(weather, on=ID, how="left", validate="1:1", maintain_order="left")

    def transformer(self, view, train):
        return FoldTransformer(self.columns[view], scale=True).fit(train)

    def verify_source(self, scope):
        gate = read_json(self.path("output_dir") / f"upstream_{scope}.json")
        if not gate["passed"] or digest(gate["file_hashes"]) != gate["input_sha256"]:
            raise ValueError("Invalid upstream verification record")
        for name, expected in gate["file_hashes"].items():
            root, relative = name.split("/", 1)
            if file_hash(self.path(root) / relative) != expected:
                raise ValueError(f"Upstream bytes changed: {name}")
        return gate["input_sha256"]


def upstream_gate(inputs, scope):
    if scope not in ("development", "all"):
        raise ValueError("Unknown audit scope")
    months = range(1, 11 if scope == "development" else 13)
    features, processed = inputs.path("features_dir"), inputs.path("processed_dir")
    locked = read_json(Path("data/features/audit/source_contract.json"))
    published = read_json(Path("reports/phase2/source_contract.json"))
    if any(locked[key] != published[key] for key in ("input_sha256", "phase1_signature", "tables")):
        raise ValueError("Local Phase I source contract differs from the published audit")
    materialized = read_json(Path("data/phase3_inputs/materialization.json"))["files"]
    expected_logical = read_json(Path("data/features/audit/logical_hashes.json"))
    reproduction = read_json(Path("reports/phase2/reproducibility.json"))
    if hashlib.sha256(json.dumps(expected_logical, sort_keys=True).encode()).hexdigest() != reproduction["combined_logical_sha256"]:
        raise ValueError("Canonical feature logical manifest differs from Phase II report")
    from ..pipeline import logical_hash

    hashes, counts, monthly_counts = {}, Counter(), {}
    for month in months:
        suffix = f"year=2025/month={month:02}"
        for root, folder, tables in (
            ("features_dir", features, ("full_features", "targets", "metadata")),
            ("processed_dir", processed, ("folds",)),
        ):
            for table in tables:
                parts = sorted((folder / table / suffix).glob("*.parquet"))
                expected_parts = expected_logical if root == "features_dir" else locked["files"]
                prefix = f"{table}/{suffix}/"
                if {p.relative_to(folder).as_posix() for p in parts} != {
                    name for name in expected_parts if name.startswith(prefix)
                } or not parts:
                    raise ValueError(f"Missing or unexpected {table} partition for {month}")
                for path in parts:
                    relative = path.relative_to(folder).as_posix()
                    hashed = file_hash(path)
                    hashes[f"{root}/{relative}"] = hashed
                    if root == "processed_dir":
                        if hashed != locked["files"][relative]:
                            raise ValueError(f"Phase I fold checksum mismatch: {relative}")
                        continue
                    known = materialized.get(f"features/{relative}", {}).get("sha256")
                    if hashed != known or table == "full_features":
                        frame = pl.read_parquet(path)
                        if logical_hash(frame, [ID]) != expected_logical[relative]:
                            raise ValueError(f"Phase II logical checksum mismatch: {relative}")
        ids = [table_scan(features, table, [month]).select(ID).collect()[ID].sort()
               for table in ("full_features", "targets", "metadata")]
        if ids[0].null_count() or ids[0].n_unique() != len(ids[0]) or not all(
            other.equals(ids[0]) for other in ids[1:]
        ):
            raise ValueError("Phase II populations are not identically aligned")
        counts.update({"full_features": len(ids[0]), "targets": len(ids[1]), "metadata": len(ids[2])})
        from .runner import ids_hash
        monthly_counts[str(month)] = {"rows": len(ids[0]), "ids_sha256": ids_hash(ids[0])}
        print(f"Verified Phase II checksums and IDs for month {month:02}", flush=True)
        if month == 10:
            write_json(inputs.path("output_dir") / "upstream_development.json",
                       dict(passed=True, input_sha256=digest(hashes), file_hashes=hashes,
                            counts=dict(counts), monthly_populations=monthly_counts))
    if scope == "all" and counts["full_features"] != locked["tables"]["modeling_cohort"]["rows"]:
        raise ValueError("Phase II cohort count differs from Phase I")
    result = dict(passed=True, input_sha256=digest(hashes), file_hashes=hashes,
                  counts=dict(counts), monthly_populations=monthly_counts)
    write_json(inputs.path("output_dir") / f"upstream_{scope}.json", result)
    return result

