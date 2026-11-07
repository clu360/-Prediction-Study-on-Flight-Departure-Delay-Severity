import polars as pl
import pytest
from flight_delay.config import Config
from flight_delay.linkage import link_events
from flight_delay.pipeline import logical_hash, select_review_examples
from flight_delay.outputs import history
from conftest import leg, target


def test_month_boundary_predecessor(normalized):
    rows=[leg(flight_date="2025-01-31",crs_dep_time="2330",crs_arr_time="0030",dep_time="2330",wheels_off="2345"),
          target(flight_date="2025-02-01",crs_dep_time="0130",crs_arr_time="0430",dep_time="0130",wheels_off="0145")]
    df=link_events(normalized(rows),Config()).sort("raw_row_number")
    assert df["link_reason"].to_list()==["no_previous_record","valid"]
    assert df["scheduled_turnaround_min"][-1]==60


def test_missing_distance_retains_label(normalized):
    df=link_events(normalized([leg(),target(distance=None,dep_delay="60")]),Config())
    assert df.height==2
    assert df.filter(pl.col("flight_number")=="2")["target_class"].item()==2
    assert df.filter(pl.col("flight_number")=="2")["link_reason"].item()=="valid"


def test_invalid_flags_retain_intervening_guard(normalized):
    rows=[leg(),target(flight_number="3",crs_dep_time="0930",cancelled="bad"),target()]
    df=link_events(normalized(rows),Config())
    assert df.height==2
    assert df.sort("raw_row_number")["link_reason"][-1]=="intervening_cancelled_or_diverted"


def test_future_delay_causes_have_no_effect(normalized):
    rows=[leg(),target(CarrierDelay="999",WeatherDelay="100")]
    a=link_events(normalized(rows),Config()).select("record_id","prior_record_id","link_status","prediction_time_utc")
    rows[1].update(CarrierDelay="0",WeatherDelay="9999")
    b=link_events(normalized(rows),Config()).select(a.columns)
    assert a.equals(b)


def test_unknown_schedule_does_not_skip_anywhere_on_tail(normalized):
    rows=[leg(),target(flight_number="3",crs_dep_time="bad"),target()]
    df=link_events(normalized(rows),Config())
    assert set(df["link_reason"].to_list())=={"invalid_timezone"}


def test_logical_hash_ignores_order_but_detects_changes():
    a=pl.DataFrame({"record_id":["b","a"],"value":[None,2.]})
    assert logical_hash(a,["record_id"])==logical_hash(a.reverse(),["record_id"])
    b=a.with_columns(pl.col("value").fill_null(0))
    assert logical_hash(a,["record_id"])!=logical_hash(b,["record_id"])


def test_configuration_contract():
    with pytest.raises(ValueError,match="locked forecast"):
        Config(prediction_offset_minutes=5)
    with pytest.raises(ValueError,match="turnaround"):
        Config(max_turnaround_minutes=5)


def test_review_examples_cover_all_months_and_strata():
    data=[]
    for month in range(1,13):
        for stratum in ["cross_zone","ordinary","day_crossing","unusual_taxi"]:
            for reason in ["valid","no_previous_record","airport_mismatch"]:
                for i in range(5):
                    data.append(dict(record_id=f"{month:02}-{stratum}-{reason}-{i}",source_month=month,
                                     example_stratum=stratum,link_reason=reason))
    sample=select_review_examples(pl.DataFrame(data))
    assert sample.height==100
    assert set(sample["source_month"])==set(range(1,13))
    assert sample["example_stratum"].n_unique()==4


def test_dst_crossing_inconsistent_gate_not_a_history_observation(normalized):
    events=normalized([leg(flight_date="2025-11-01",crs_dep_time="2055",crs_arr_time="2155",
                           dep_delay="662",dep_time="0757",wheels_off="0828")])
    assert events["eligible_target"].item()
    assert not events["gate_departure_clock_ok"].item()
    assert events["actual_wheels_off_utc"].item() is None
    assert history(events).height==0


def test_utc_order_can_reverse_local_clock_order(normalized):
    rows=[leg(crs_dep_time="0900",dep_time="0900",wheels_off="0910",crs_elapsed_time="45",crs_arr_time="0845"),
          target(crs_dep_time="0855",dep_time="0855",wheels_off="0905",crs_arr_time="1140",crs_elapsed_time="105")]
    linked=link_events(normalized(rows),Config()).sort("raw_row_number")
    assert linked["link_reason"].to_list()==["no_previous_record","valid"]
