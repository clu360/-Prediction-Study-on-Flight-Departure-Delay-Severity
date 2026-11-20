from datetime import datetime, timedelta, timezone
import csv
from io import StringIO

import numpy as np
import pytest

from flight_delay.weather.features import direct_oracle, station_features
from flight_delay.weather.metar import FIELDS, category, decode, flags


@pytest.mark.parametrize(("ceiling", "visibility", "expected"), [
    ("3000", "6", 1), ("3001", "6", 0), ("1000", "6", 1), ("999", "6", 2),
    ("500", "6", 2), ("499", "6", 3), ("5000", "5", 1), ("5000", "5.01", 0),
    ("5000", "3", 1), ("5000", "2.99", 2), ("5000", "1", 2), ("5000", "0.99", 3),
    ("", "6", None), ("3001", "", None),
])
def test_category(ceiling, visibility, expected):
    report = {name: "" for name in FIELDS}
    report.update(skyc2="OVC", skyl2=ceiling, vsby=visibility)
    assert category(report) == expected
    report["skyc2"] = "VV"
    assert category(report) == expected
    report["skyc2"] = "BKN"
    assert category(report) == expected


@pytest.mark.parametrize(("codes", "expected"), [
    ("TSRA", (0, 1)), ("VCTS", (0, 1)), ("+TSRA", (0, 1)),
    ("BLSN", (1, 0)), ("-FZRA", (1, 0)), ("FZDZ", (1, 0)),
    ("FZFG", (0, 0)), ("TSRASN", (1, 1)), ("", (0, 0)),
])
def test_flags(codes, expected):
    assert flags(codes) == expected


def test_strict_windows_stale_wind_and_future_injection():
    t = datetime(2025, 11, 2, 6, 30, tzinfo=timezone.utc)
    times = np.array([t.replace(tzinfo=None)], dtype="datetime64[us]")
    reports = [(t - timedelta(minutes=180), 3, 1, 0, 38),
               (t - timedelta(minutes=60), 0, 0, 1, None)]
    assert tuple(value[0] for value in station_features(times, reports))[:3] == (3, 1, 1)
    a = station_features(times, reports)
    b = station_features(times, reports + [(t, 3, 1, 1, 90)])
    for left, right in zip(a, b):
        np.testing.assert_equal(left, right)
    assert np.isnan(a[3][0])
    assert np.isnan(station_features(times, reports[:1])[3][0])
    assert direct_oracle(t, reports, []) == (3, 1, 1, None, None)
    assert direct_oracle(t, [], reports) == (None, None, None, None, 3)


def test_corrections_blank_gust_and_unknown_sustained_wind():
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    end = start + timedelta(days=1)
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=sorted(FIELDS))
    writer.writeheader()
    base = dict.fromkeys(FIELDS, "") | {"station": "ATL", "valid": "2025-01-01 00:00",
                                       "metar": "KATL 010000Z", "vsby": "10", "sknt": "6"}
    writer.writerow(base | {"metar": "KATL 010000Z COR", "sknt": "50"})
    writer.writerow(base)
    writer.writerow(base)
    writer.writerow(base | {"valid": "2025-01-01 00:20", "metar": "KATL 010020Z", "sknt": "", "gust": "22"})
    reports, audit = decode(buffer.getvalue().encode(), "ATL", start, end)
    assert reports == [(start + timedelta(minutes=10), 0, 0, 0, 6),
                       (start + timedelta(minutes=30), 0, 0, 0, None)]
    assert audit["corrections_dropped"] == audit["duplicates"] == 1

