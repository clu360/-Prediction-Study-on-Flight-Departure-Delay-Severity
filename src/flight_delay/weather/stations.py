"""Match Phase II airport codes against pinned airportsdata and IEM metadata."""

import csv
import json
import math

import airportsdata
import polars as pl

from ..experiments.inputs import table_scan
from ..experiments.results import atomic


def airport_codes(inputs):
    features = table_scan(inputs.path("features_dir"), "full_features", list(range(1, 13)))
    columns = ["g1_origin", "g1_dest"]
    return sorted(set(features.select(pl.col(columns[0]).unique()).collect().to_series())
                  | set(features.select(pl.col(columns[1]).unique()).collect().to_series()))


def distance(airport, site):
    lon, lat = site["geometry"]["coordinates"]
    average = math.radians((lat + airport["lat"]) / 2)
    return 111.195 * math.hypot(lat - airport["lat"], (lon - airport["lon"]) * math.cos(average))


def build_map(codes, metadata_paths, destination, verified_overrides=None, *, allow_partial=False):
    airports = airportsdata.load("IATA")
    sites = [
        (path, site) for path in sorted(metadata_paths)
        for site in json.loads(path.read_text(encoding="utf-8"))["features"]
    ]
    overrides = verified_overrides or {}
    rows, unresolved = [], []
    for code in codes:
        airport = airports.get(code)
        if airport is None:
            unresolved.append((code, "missing pinned IATA entry"))
            continue
        icao = airport["icao"]
        explicit = overrides.get(code)
        expected_ids = {icao, icao[1:]} if icao.startswith("K") else {icao, code}
        if explicit:
            expected_ids = {explicit["station"]}
        matches = [
            (distance(airport, site), path, site) for path, site in sites
            if site["id"] in expected_ids and site["geometry"] is not None
            and distance(airport, site) < 10
            and (site["properties"]["archive_end"] or "9999") >= "2025-01-01"
        ]
        if len(matches) != 1:
            unresolved.append((code, f"{len(matches)} geographically and temporally matching station records"))
            continue
        km, path, site = matches[0]
        metar_identity = f"{icao}|{icao[1:]}" if icao.startswith("K") else icao
        props = site["properties"]
        rows.append(dict(iata=code, icao=icao, station=site["id"], network=props["network"],
                         metar_identity=explicit.get("metar_identity", metar_identity) if explicit else metar_identity,
                         distance_km=round(km, 3), archive_begin=props["archive_begin"],
                         archive_end=props["archive_end"],
                         evidence=explicit["evidence"] if explicit else (
                             f"https://mesonet.agron.iastate.edu/geojson/network/{path.name}"
                         )))
    if unresolved:
        if not allow_partial:
            raise ValueError(f"Station identities require evidence: {unresolved}")
        print(f"Provisional station map. Station identities require evidence: {unresolved}", flush=True)
    if len({row["station"] for row in rows}) != len(rows):
        raise ValueError("Multiple airports resolved to one archive station")
    atomic(destination, lambda p: pl.DataFrame(rows).write_csv(p))
    return rows


def load_map(path):
    with path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if len({row["iata"] for row in rows}) != len(rows) or any(not row["evidence"] for row in rows):
        raise ValueError("Incomplete or duplicate station map evidence")
    return rows

