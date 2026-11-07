import hashlib
from pathlib import Path
import zipfile
import polars as pl
import pytest
from flight_delay.discovery import discover
from conftest import leg, target


def make_year(path):
    for month in range(1,13):
        rows=[leg(flight_date=f"2025-{month:02}-01",month=month),target(flight_date=f"2025-{month:02}-01",month=month)]
        # Official mixed-case aliases must be accepted, not invented semantics.
        df=pl.DataFrame(rows).rename({"flight_date":"FlightDate","reporting_airline":"Reporting_Airline",
            "tail_number":"Tail_Number","flight_number":"Flight_Number_Reporting_Airline"})
        df.write_csv(path/f"input_{month:02}.csv")


def test_discovery_complete_and_immutable(tmp_path):
    make_year(tmp_path)
    initial={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.glob("*.csv")}
    result=discover(tmp_path,2025)
    assert [r["month"] for r in result]==list(range(1,13))
    assert all(r["row_count"]==2 for r in result)
    assert initial=={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.glob("*.csv")}


def test_missing_duplicate_and_wrong_month(tmp_path):
    make_year(tmp_path)
    (tmp_path/"input_12.csv").unlink()
    with pytest.raises(ValueError,match="exactly one"):
        discover(tmp_path,2025)
    make_year(tmp_path)
    (tmp_path/"copy.csv").write_bytes((tmp_path/"input_01.csv").read_bytes())
    with pytest.raises(ValueError,match="exactly one"):
        discover(tmp_path,2025)


def test_zip_internal_names_and_schema_failure(tmp_path):
    make_year(tmp_path)
    p=tmp_path/"input_01.csv"
    with zipfile.ZipFile(tmp_path/"bundle.zip","w") as z:
        z.write(p,arcname="unexpected.csv")
    p.unlink()
    assert discover(tmp_path,2025)[0]["zip_member"]=="unexpected.csv"
    (tmp_path/"input_02.csv").write_text("FlightDate\n2025-02-01\n")
    with pytest.raises(ValueError,match="Source-column mismatch"):
        discover(tmp_path,2025)
