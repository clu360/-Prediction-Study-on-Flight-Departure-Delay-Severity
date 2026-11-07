"""Inspect every input before normalization; reject missing/duplicate months."""
from __future__ import annotations
from contextlib import contextmanager
import calendar
import hashlib
import io
import json
from pathlib import Path
import zipfile
import polars as pl

ALIASES = {
    "YEAR": "year", "MONTH": "month", "FLDATE": "flight_date", "FLIGHTDATE": "flight_date",
    "OPUNIQUECARRIER": "reporting_airline", "REPORTINGAIRLINE": "reporting_airline",
    "OPCARRIERFLNUM": "flight_number", "FLIGHTNUMBERREPORTINGAIRLINE": "flight_number",
    "TAILNUM": "tail_number", "TAILNUMBER": "tail_number", "ORIGIN": "origin", "DEST": "dest",
    "CRSDEPTIME": "crs_dep_time", "CRSARRTIME": "crs_arr_time", "CRSELAPSEDTIME": "crs_elapsed_time",
    "DISTANCE": "distance", "DEPDELAY": "dep_delay", "CANCELLED": "cancelled", "DIVERTED": "diverted",
    "DEPTIME": "dep_time", "TAXIOUT": "taxi_out", "WHEELSOFF": "wheels_off",
    "ARRTIME": "arr_time", "ARRDELAY": "arr_delay", "WHEELSON": "wheels_on",
    "ACTUALELAPSEDTIME": "actual_elapsed_time", "DAYOFWEEK": "day_of_week",
    "ORIGINAIRPORTID": "origin_airport_id", "DESTAIRPORTID": "dest_airport_id",
    "OPCARRIERAIRLINEID": "dot_airline_id", "DOTIDREPORTINGAIRLINE": "dot_airline_id",
}
REQUIRED = {"flight_date", "reporting_airline", "tail_number", "origin", "dest", "crs_dep_time",
            "crs_arr_time", "crs_elapsed_time", "distance", "dep_delay", "cancelled", "diverted"}
OPTIONAL = {"year", "month", "flight_number", "dep_time", "taxi_out", "wheels_off", "arr_time",
            "arr_delay", "wheels_on", "actual_elapsed_time", "day_of_week", "dot_airline_id",
            "origin_airport_id", "dest_airport_id"}


def canonical(name: str) -> str | None:
    return ALIASES.get("".join(c for c in name.upper() if c.isalnum()))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


@contextmanager
def source_stream(path: Path, member: str | None = None):
    if member:
        with zipfile.ZipFile(path) as z, z.open(member) as f:
            yield f
    else:
        with path.open("rb") as f:
            yield f


def read_source(path: Path, member: str | None = None, selected: bool = False) -> pl.DataFrame:
    with source_stream(path, member) as f:
        # Explicit string dtypes preserve identifiers and numeric null tokens.
        df = pl.read_csv(f, infer_schema=False, null_values=["", "NA", "N/A", "NULL", "NaN", "nan"],
                         try_parse_dates=False, raise_if_empty=True)
    mapping = {c: canonical(c) for c in df.columns if canonical(c)}
    if len(set(mapping.values())) != len(mapping):
        raise ValueError(f"Colliding source aliases: {path}")
    if selected:
        return df.select(list(mapping)).rename(mapping)
    return df


def parse_date() -> pl.Expr:
    s = pl.col("flight_date").str.strip_chars()
    return pl.coalesce(s.str.strptime(pl.Date, "%m/%d/%Y %I:%M:%S %p", strict=False),
                       s.str.strptime(pl.Date, "%Y-%m-%d", strict=False),
                       s.str.strptime(pl.Date, "%Y-%m-%d %H:%M:%S", strict=False),
                       s.str.strptime(pl.Date, "%Y%m%d", strict=False))


def discover(raw_dir: Path, year: int, manifest_path: Path | None = None) -> list[dict]:
    files = sorted(p for p in raw_dir.rglob("*") if p.suffix.lower() in {".csv", ".zip"})
    entries: list[dict] = []
    errors = []
    for path in files:
        try:
            members = [None] if path.suffix.lower() == ".csv" else sorted(
                n for n in zipfile.ZipFile(path).namelist() if n.lower().endswith(".csv"))
            if not members:
                raise ValueError("ZIP contains no CSV")
            file_hash = sha256(path)
            for member in members:
                print(f"Discovering {path.name} {member or ''}", flush=True)
                raw = read_source(path, member)
                header = raw.columns
                names = {c: canonical(c) for c in header if canonical(c)}
                missing = sorted(REQUIRED - set(names.values()))
                if "wheels_off" not in names.values() and "taxi_out" not in names.values():
                    missing.append("wheels_off OR taxi_out")
                if missing:
                    raise ValueError(f"Source-column mismatch: {missing}")
                selected = raw.select(list(names)).rename(names)
                dates = selected.select(parse_date().alias("date"))["date"]
                periods = dates.drop_nulls().dt.strftime("%Y-%m").unique().sort().to_list()
                if dates.null_count() or len(periods) != 1 or not periods[0].startswith(f"{year}-"):
                    raise ValueError(f"Invalid dates or mixed/wrong month: {periods}, null={dates.null_count()}")
                month = int(periods[0][-2:])
                for field, expected in [("year", year), ("month", month)]:
                    if field in selected and selected.select((pl.col(field).cast(pl.Float64, strict=False)
                            .ne(expected).fill_null(True)).sum()).item():
                        raise ValueError(f"Inconsistent {field} versus FlightDate")
                tokens = (path.stem + " " + (member or "")).lower()
                named = [i for i in range(1, 13) if calendar.month_name[i].lower() in tokens]
                if named and month not in named:
                    raise ValueError(f"Filename/date mismatch: {named} vs {month}")
                # Compare complete raw records, including unknown audit-only columns.
                duplicates = raw.height - raw.n_unique()
                entry = dict(path=str(path.resolve()), filename=path.name, zip_member=member,
                    size_bytes=path.stat().st_size, sha256=file_hash, month=month, year=year,
                    row_count=raw.height, date_min=str(dates.min()), date_max=str(dates.max()),
                    headers=header, canonical_headers=list(names.values()), exact_duplicate_rows=duplicates,
                    missing_required_columns=[], absent_optional_columns=sorted(OPTIONAL-set(names.values())),
                    null_counts=dict(zip(header, raw.null_count().row(0))), read_error=None)
                entries.append(entry)
        except Exception as exc:
            errors.append(f"{path.name}: {exc}")
            entries.append(dict(path=str(path.resolve()), read_error=str(exc)))
    months = [e["month"] for e in entries if not e["read_error"]]
    if sorted(months) != list(range(1, 13)):
        errors.append(f"Require exactly one authoritative input per month 1..12; got {sorted(months)}")
    payload = {"year": year, "inputs": entries, "errors": errors}
    if manifest_path:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if errors:
        raise ValueError("Input discovery failed: " + "; ".join(errors))
    return sorted(entries, key=lambda e: e["month"])
