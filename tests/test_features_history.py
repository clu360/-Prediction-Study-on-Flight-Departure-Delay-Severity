from datetime import datetime, timedelta, timezone
import numpy as np
import polars as pl
import pytest
from flight_delay.features.config import FeatureConfig
from flight_delay.features.history import PrefixHistory, history_features
from flight_delay.features.views import G3

T = datetime(2025, 2, 1, 12, tzinfo=timezone.utc)
EVENT_SCHEMA = {"record_id": pl.String, "origin": pl.String, "reporting_airline": pl.String,
    "observed_departure_time_utc": pl.Datetime("us", "UTC"), "dep_delay": pl.Float64,
    "departure_severity": pl.Int8, "valid_observation": pl.Boolean}
SCHEDULE_SCHEMA = {"origin": pl.String, "scheduled_departure_utc": pl.Datetime("us", "UTC")}


def event(name, time, delay, airport="ATL", carrier="AA", valid=True):
    return dict(record_id=name, origin=airport, reporting_airline=carrier,
        observed_departure_time_utc=time, dep_delay=float(delay), departure_severity=0 if delay < 15 else 1 if delay < 60 else 2,
        valid_observation=valid)


def queries(times=None, airport="ATL", carrier="AA"):
    return pl.DataFrame([dict(target_record_id=f"q{i}", target_origin=airport, target_carrier=carrier,
                             prediction_time_utc=t) for i, t in enumerate(times or [T])])


def compute(rows, q=None, scheduled=None):
    events = pl.DataFrame(rows, schema=EVENT_SCHEMA)
    schedule = pl.DataFrame(scheduled or [], schema=SCHEDULE_SCHEMA)
    return history_features(q if q is not None else queries(), events, schedule,
                            PrefixHistory.from_events(events, with_delay=False), FeatureConfig())


def test_manual_smoothing_and_nested_fallback():
    rows = [event("a",T-timedelta(days=28),15), event("b",T-timedelta(hours=1),60,carrier="BB"),
            event("c",T-timedelta(hours=2),-5,airport="DFW")]
    f, m = compute(rows)
    airport_severe=(1+20/3)/22
    assert f[G3[0]].item()==pytest.approx((1+20/3)/22)
    assert f[G3[1]].item()==pytest.approx(airport_severe)
    assert f[G3[2]].item()==37.5
    assert f[G3[4]].item()==pytest.approx(20*airport_severe/21)
    assert f[G3[5]].item()==1
    assert m["airport_history_count_28d"].item()==2
    f,_=compute(rows,queries(carrier="NEW"))
    assert f[G3[4]].item()==f[G3[1]].item()
    f,_=compute(rows,queries(airport="DEN"))
    assert f[G3[0]].item()==pytest.approx(1/3)
    assert f[G3[1]].item()==pytest.approx(1/3)
    assert f[G3[2]].item() is None


def test_inclusive_left_and_strict_right_boundaries():
    rows=[event("left",T-timedelta(days=28),15),event("too_old",T-timedelta(days=28,microseconds=1),60),
          event("equal",T,60),event("future",T+timedelta(microseconds=1),60)]
    schedule=[dict(origin="ATL",scheduled_departure_utc=t) for t in
              [T-timedelta(days=7),T-timedelta(days=7,microseconds=1),T,T+timedelta(seconds=1)]]
    f,m=compute(rows,scheduled=schedule)
    assert m["global_history_count_28d"].item()==1
    assert f[G3[0]].item()==1
    assert f[G3[1]].item()==0
    assert f[G3[3]].item()==1


def test_future_events_and_own_label_do_not_change_features():
    baseline=[event("old",T-timedelta(days=1),-5)]
    a,_=compute(baseline)
    b,_=compute(baseline+[event("future",T+timedelta(seconds=1),1000)])
    assert a.equals(b)
    a,m=compute(baseline+[event("q0",T-timedelta(minutes=5),0)])
    b,_=compute(baseline+[event("q0",T-timedelta(minutes=5),9999)])
    assert a.equals(b)
    assert m["own_outcome_excluded"].item()


def test_same_day_update_and_equal_time_determinism():
    rows=[event("a",T,60)]
    q=queries([T,T+timedelta(minutes=1),T])
    f,m=compute(rows,q)
    assert f.filter(pl.col("target_record_id")=="q0")[G3[1]].item() is None
    assert f.filter(pl.col("target_record_id")=="q1")[G3[1]].item()==1
    assert compute(rows,q.reverse())[0].equals(f)
    assert m["no_global_history"].sum()==2


def test_empty_history_retains_rows_and_nulls():
    f,m=compute([])
    assert f.height==1 and f[G3[0]].item() is None and f[G3[2]].item() is None
    assert f[G3[5]].item()==0 and m["no_global_history"].item()


def test_invalid_observations_ignored_and_signed_mean():
    f,m=compute([event("a",T-timedelta(hours=1),-10),event("bad",T-timedelta(hours=1),999,valid=False)])
    assert f[G3[2]].item()==-10 and m["airport_history_count_28d"].item()==1


def test_scheduled_volume_includes_future_completion_status():
    schedule=pl.DataFrame([dict(origin="ATL",scheduled_departure_utc=T-timedelta(hours=1),cancelled=1,diverted=0),
                          dict(origin="ATL",scheduled_departure_utc=T-timedelta(hours=2),cancelled=0,diverted=1)])
    empty=pl.DataFrame(schema=EVENT_SCHEMA)
    f,_=history_features(queries(),empty,schedule,PrefixHistory.from_events(empty,False),FeatureConfig())
    assert f[G3[3]].item()==2


def test_diversion_outcome_is_not_an_eligibility_filter():
    events=pl.DataFrame([event("diverted",T-timedelta(minutes=1),60)],schema=EVENT_SCHEMA).with_columns(pl.lit(1).alias("diverted"))
    f,_=history_features(queries(),events,pl.DataFrame(schema=SCHEDULE_SCHEMA),PrefixHistory.from_events(events,False),FeatureConfig())
    assert f[G3[1]].item()==1


def test_airport_and_airline_grouping_do_not_mix_cells():
    rows=[event("a",T-timedelta(hours=1),15),event("b",T-timedelta(hours=1),60,carrier="BB"),
          event("c",T-timedelta(hours=1),-5,airport="DFW")]
    q=pl.DataFrame([dict(target_record_id=name,target_origin=airport,target_carrier=carrier,prediction_time_utc=T)
        for name,airport,carrier in [("qA","ATL","AA"),("qB","ATL","BB"),("qC","DFW","AA")]])
    f,m=compute(rows,q)
    assert f[G3[5]].to_list()==[1,1,1]
    assert f[G3[2]].to_list()==[37.5,37.5,-5]
    assert f[G3[4]][1]>f[G3[4]][0]
    assert m["airport_history_count_28d"].to_list()==[2,2,1]
