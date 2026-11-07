from pathlib import Path
import pytest
from flight_delay.features.views import ALL, VIEWS, load_views
from flight_delay.features.config import FeatureConfig


def test_exact_checked_views():
    assert [len(v) for v in VIEWS.values()]==[9,14,15,20]
    assert len(set(ALL))==20
    assert load_views(Path("configs/feature_groups.yaml"))==VIEWS
    assert not any("minimal" in c for c in ALL)
    forbidden={"target_class","target_dep_delay_min","target_record_id","prediction_time_utc","source_month","prior_record_id"}
    assert not forbidden.intersection(ALL)


def test_locked_history_configuration():
    with pytest.raises(ValueError):
        FeatureConfig(alpha=1)
