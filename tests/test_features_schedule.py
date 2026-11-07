import numpy as np
import polars as pl
from datetime import date
from flight_delay.features.schedule import schedule_features
from flight_delay.features.incoming import incoming_features
from flight_delay.features.views import G1, G1_MAP, G2, G2_MAP


def test_cyclic_hours_and_source_values():
    frame=pl.DataFrame({"target_record_id":["a","b","c","d"],"target_crs_dep_local_hour":[0,6,12,23],
        **{source:[1,2,3,4] for source in G1_MAP.values()}})
    features=schedule_features(frame)
    assert features.columns==["target_record_id"]+G1
    assert np.allclose(features["g1_dep_hour_sin"][:3].to_numpy(),[0,1,0],atol=1e-14)
    assert np.allclose(features["g1_dep_hour_cos"][:3].to_numpy(),[1,0,-1],atol=1e-14)
    assert features["g1_dep_hour_cos"][-1]>.9
    for target,source in G1_MAP.items():
        assert features[target].to_list()==frame[source].to_list()


def test_incoming_reuses_signed_values_and_excludes_outcomes():
    frame=pl.DataFrame({"target_record_id":["a"],**{source:[-5.] for source in G2_MAP.values()},
                        "arr_delay":[999.],"tail_number":["N1"],"target_class":[2]})
    features=incoming_features(frame)
    assert features.columns==["target_record_id"]+G2
    assert features["g2_prior_dep_delay_min"].item()==-5


def test_phase1_polars_weekday_convention_is_iso():
    days=pl.Series([date(2025,1,6),date(2025,1,5)])
    assert days.dt.weekday().to_list()==[1,7]
