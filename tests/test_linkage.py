import polars as pl
import pytest
from flight_delay.config import Config
from flight_delay.linkage import folds, link_events
from flight_delay import outputs
from conftest import leg, target


def reason(normalized, rows):
    return link_events(normalized(rows), Config()).sort("raw_row_number")["link_reason"].to_list()


def test_immediate_predecessor(normalized):
    assert reason(normalized,[leg(),target()]) == ["no_previous_record","valid"]


@pytest.mark.parametrize("changes,expected", [
    ({"tail_number":"N2"},"no_previous_record"),
    ({"tail_number":None},"missing_tail"),
    ({"origin":"ATL"},"airport_mismatch"),
    ({"crs_dep_time":"0816"},"negative_scheduled_turnaround"),
    ({"crs_dep_time":"0716"},"late_prior_takeoff"),
    ({"origin":"ZZZ"},"invalid_timezone"),
])
def test_target_rejections(normalized,changes,expected):
    assert reason(normalized,[leg(),target(**changes)])[-1] == expected


@pytest.mark.parametrize("flag",["cancelled","diverted"])
def test_intervening_nonoperation_not_skipped(normalized,flag):
    rows = [leg(),target(flight_number="9",crs_dep_time="0930",**{flag:1}),target()]
    assert reason(normalized,rows)[-1] == "intervening_cancelled_or_diverted"


def test_intervening_moved_leg(normalized):
    rows=[leg(),target(flight_number="9",crs_dep_time="0930",dep_time="0930",wheels_off="0945"),target()]
    assert reason(normalized,rows)[-1] == "airport_mismatch"


def test_missing_takeoff(normalized):
    assert reason(normalized,[leg(wheels_off=None),target()])[-1] == "predecessor_missing_takeoff"


def test_turnaround_boundaries(normalized):
    assert reason(normalized,[leg(),target(crs_dep_time="0900")])[-1] == "valid" # zero turn
    assert reason(normalized,[leg(),target(flight_date="2025-01-02",crs_dep_time="0900")])[-1] == "valid"
    assert reason(normalized,[leg(),target(flight_date="2025-01-02",crs_dep_time="0901")])[-1] == "gap_over_24h"


def test_ambiguous_tied_schedule(normalized):
    assert reason(normalized,[leg(),leg(flight_number="99"),target()]) == ["ambiguous_rotation"]*3


def test_leakage_invariance(normalized):
    rows = [leg(arr_delay="1",arr_time="0900",wheels_on="0850"),target()]
    first = link_events(normalized(rows),Config()).select("record_id","prior_record_id","link_reason","prediction_time_utc")
    rows[0].update(arr_delay="9999",arr_time="0001",wheels_on="2333",actual_elapsed_time="9999")
    rows[1].update(dep_delay="-200",dep_time="0010",arr_delay="-9999")
    second = link_events(normalized(rows),Config()).select(first.columns)
    assert first.equals(second)
    for row in rows:
        for c in ["arr_delay","arr_time","wheels_on","actual_elapsed_time"]:
            row.pop(c,None)
    third = link_events(normalized(rows),Config()).select(first.columns)
    assert first.equals(third)


def test_folds_and_allowlist():
    df=pl.DataFrame({"target_record_id":[f"id{i}" for i in range(1,13)],"source_month":list(range(1,13)),
                     "target_month":list(range(1,13))})
    f=folds(df,2025)
    assert f.height==36
    assert f.filter((pl.col("fold_id")=="2025-04") & (pl.col("role")=="train"))["source_month"].to_list()==[1,2,3]
    assert f.filter(pl.col("is_final_holdout"))["test_month"].unique().sort().to_list()==[11,12]
    assert f.select("target_record_id","fold_id","role").n_unique()==f.height
    assert not set(outputs.G1_ALLOWED+outputs.G2_ALLOWED)&set(outputs.LABEL_COLUMNS)
