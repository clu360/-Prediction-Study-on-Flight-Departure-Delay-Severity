from __future__ import annotations
import polars as pl
import pytest
from flight_delay.config import Config
from flight_delay.normalize import normalize


def leg(**changes) -> dict:
    row = dict(flight_date="2025-01-01", year="2025", month="1", reporting_airline="AA", flight_number="1",
        tail_number="N1", origin="ATL", dest="DFW", crs_dep_time="0800", crs_arr_time="0900",
        crs_elapsed_time="120", distance="700", dep_delay="0", cancelled="0", diverted="0",
        dep_time="0800", wheels_off="0815", actual_elapsed_time="120")
    row.update({k: str(v) if v is not None else None for k, v in changes.items()})
    return row


def target(**changes) -> dict:
    defaults = dict(flight_number="2", origin="DFW", dest="ATL", crs_dep_time="1000", crs_arr_time="1300",
                    dep_time="1000", wheels_off="1015")
    defaults.update(changes)
    return leg(**defaults)


@pytest.fixture
def normalized():
    def factory(rows):
        raw = pl.DataFrame(rows, infer_schema_length=None).cast(pl.String)
        return normalize(raw, dict(month=1,year=2025,filename="synthetic.csv",sha256="0"*64),
                         Config(), {"ATL": "America/New_York", "DFW": "America/Chicago"})[0]
    return factory
