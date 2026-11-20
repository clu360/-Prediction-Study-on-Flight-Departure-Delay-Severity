"""Publish aggregate proof of the archived request and station coverage."""

from collections import Counter

import polars as pl

from .fetch import Archive, bounds, request_parameters
from .stations import load_map
from ..experiments.config import digest, file_hash
from ..experiments.results import read_json, write_frame, write_json


def audit_archive(inputs, version):
    raw = inputs.path("raw_weather_dir")
    archive = Archive(raw, *bounds(inputs.weather), version=version)
    station_map = inputs.path("station_map")
    mapping = load_map(station_map)
    output = inputs.path("reports_dir")
    inventory, totals = [], Counter()
    for row in mapping:
        code, station = row["iata"], row["station"]
        csv_path, sidecar = archive.files(station)
        record = read_json(sidecar)
        checksum = file_hash(csv_path)
        expected_params = [list(pair) for pair in request_parameters(station, archive.start, archive.end)]
        if (checksum != record["sha256"] or record["bytes"] != csv_path.stat().st_size
                or record["parameters"] != expected_params or record["station"] != station):
            raise ValueError(f"Archived request mismatch for {station}")
        qc = record["audit"]
        counts = {key: value for key, value in qc.items() if isinstance(value, int)}
        inventory.append(dict(airport=code, station=station, sha256=checksum, bytes=record["bytes"],
                              retrieved_utc=record["retrieved_utc"], archive_days=len(qc["days"]),
                              report_identities="|".join(qc["metar_identities"]), **counts))
        totals.update(counts)
    table = output / "weather_source_inventory.csv"
    write_frame(table, pl.DataFrame(inventory, infer_schema_length=None).sort("airport"))
    result = dict(
        stations=len(inventory), raw_request_period=f"{archive.start.date()} through {archive.end.date()} exclusive",
        source="IEM ASOS archive report types 3 and 4", version=version, totals=dict(totals),
        stations_without_rows=[row["station"] for row in inventory if not row["rows"]],
        request_inventory_sha256=file_hash(table), mapping_sha256=file_hash(station_map),
        archive_digest=digest({row["station"]: row["sha256"] for row in inventory}),
        station_metadata_sha256={path.name: file_hash(path)
                                 for path in sorted((raw / "stations").glob("*.geojson"))},
    )
    write_json(output / "weather_source_verification.json", result)
    return result

