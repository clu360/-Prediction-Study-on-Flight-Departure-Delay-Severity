"""Parse the IEM CSV contract and exclude unavailable correction versions."""

import csv
from datetime import datetime, timedelta, timezone
from io import StringIO
import math
import re

FIELDS = {"station", "valid", "vsby", "sknt", "gust", "wxcodes", "metar"} | {
    f"{prefix}{layer}" for prefix in ("skyc", "skyl") for layer in range(1, 5)
}
LAG = timedelta(minutes=10)


def numeric(text):
    if text in ("", "M", None):
        return None
    value = float(text)
    return value if math.isfinite(value) and value >= 0 else None


def category(row):
    ceiling = math.inf
    for layer in range(1, 5):
        if row[f"skyc{layer}"] in ("BKN", "OVC", "VV"):
            height = numeric(row[f"skyl{layer}"])
            if height is None:
                return None
            ceiling = min(ceiling, height)
    visibility = numeric(row["vsby"])
    if visibility is None:
        return None
    ceiling_code = 0 if ceiling > 3000 else 1 if ceiling >= 1000 else 2 if ceiling >= 500 else 3
    visibility_code = 0 if visibility > 5 else 1 if visibility >= 3 else 2 if visibility >= 1 else 3
    return max(ceiling_code, visibility_code)


def flags(codes):
    tokens = [token.lstrip("+-") for token in codes.split()]
    thunder = any(token.startswith(("TS", "VCTS")) for token in tokens)
    frozen = any(
        re.fullmatch(r"(?:VC|MI|PR|BC|DR|BL|SH|TS)?(?:DZ|RA|SN|SG|IC|PL|GR|GS|UP)*(?:SN|SG|PL|GS)(?:DZ|RA|SN|SG|IC|PL|GR|GS|UP)*", token)
        or token in ("FZRA", "FZDZ", "FZRASN", "FZDZSN")
        for token in tokens
    )
    return int(frozen), int(thunder)


def decode(raw, station, start, end, *, routine_and_special_only=False):
    reader = csv.DictReader(StringIO(raw.decode("utf-8-sig"), newline=""))
    if not FIELDS.issubset(reader.fieldnames or []):
        raise ValueError("IEM response lacks required CSV fields")
    reports, seen, versions = [], set(), {}
    audit = {"rows": 0, "corrections_dropped": 0, "duplicates": 0,
             "distinct_same_observation": 0, "unknown_category": 0,
             "unknown_wind": 0, "invalid_numeric_fields": 0}
    days, identities = set(), set()
    for row in reader:
        if None in row or any(row[field] is None for field in FIELDS) or row["station"] != station:
            raise ValueError("Malformed or foreign station report")
        valid = datetime.fromisoformat(row["valid"]).replace(tzinfo=timezone.utc)
        if not start <= valid < end:
            raise ValueError("Report outside the requested UTC date range")
        audit["rows"] += 1
        days.add(valid.date().isoformat())
        metar = row["metar"].split()
        identity = re.match(r"^(?:(?:METAR|SPECI|COR) )*([A-Z0-9]{3,4}) [0-9]{6}Z\b", row["metar"])
        if identity:
            identities.add(identity[1])
        if routine_and_special_only and "MADISHF" in metar:
            audit["out_of_scope_high_frequency"] = audit.get("out_of_scope_high_frequency", 0) + 1
            continue
        if "COR" in metar[:5]:
            audit["corrections_dropped"] += 1
            continue
        if not identity:
            raise ValueError("Cannot verify original report identity and correction status")
        key = (valid, tuple(row[name] for name in sorted(FIELDS)))
        if key in seen:
            audit["duplicates"] += 1
            continue
        seen.add(key)
        audit["invalid_numeric_fields"] += sum(
            row[name] not in ("", "M") and numeric(row[name]) is None
            for name in ["vsby", "sknt", "gust", *(f"skyl{i}" for i in range(1, 5))]
        )
        sustained, gust = numeric(row["sknt"]), numeric(row["gust"])
        wind = max(sustained, gust or 0) if sustained is not None else None
        result = (valid + LAG, category(row), *flags(row["wxcodes"]), wind)
        audit["unknown_category"] += result[1] is None
        audit["unknown_wind"] += wind is None
        if valid in versions:
            audit["distinct_same_observation"] += 1
        versions[valid] = result
        reports.append(result)
    audit["days"] = sorted(days)
    audit["metar_identities"] = sorted(identities)
    return sorted(reports, key=lambda report: report[0]), audit

