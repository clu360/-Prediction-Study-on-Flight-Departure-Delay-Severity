import json
import subprocess
import sys
from pathlib import Path
import polars as pl
from flight_delay.config import Config
from flight_delay.pipeline import run
from flight_delay.features.builder import build
from flight_delay.features.config import FeatureConfig
from flight_delay.features.transforms import build_fold
from test_discovery import make_year


def test_feature_build_reproducibility_and_all_fold_views(tmp_path):
    raw=tmp_path/"raw"; raw.mkdir(); make_year(raw)
    processed=tmp_path/"processed"
    run(raw,processed,2025,Config(evaluation_airports=("DFW",)))
    out=tmp_path/"features"
    cfg=FeatureConfig(evaluation_airports=("DFW",))
    first=build(processed,out,cfg)
    assert first["acceptance_passed"] and first["counts"]["full_features"]==12
    hashes=(out/"audit/logical_hashes.json").read_text()
    second=build(processed,out,cfg)
    assert second["counts"]==first["counts"] and hashes==(out/"audit/logical_hashes.json").read_text()
    copied=tmp_path/"features_reused"
    reused=build(processed,copied,cfg,reuse_cache=out/"cache")
    assert reused["counts"]==first["counts"]
    assert hashes==(copied/"audit/logical_hashes.json").read_text()
    expected=None
    for view in ["G1","G1_G2","G1_G3","G1_G2_G3"]:
        fold=build_fold("2025-04",view,features_dir=out,processed_dir=processed)
        assert fold.X_train.shape[0]==3 and fold.X_test.shape[0]==1
        assert len(fold.train_ids)==len(fold.y_train)
        if expected is not None:
            assert fold.train_ids.tolist()==expected
        expected=fold.train_ids.tolist()
    holdout=build_fold("2025-12","G1",features_dir=out,processed_dir=processed)
    assert holdout.fitted_transformer.is_final_holdout_
    for command in [["features-audit","--features-dir",str(out)],
        ["feature-fold","--fold","2025-04","--feature-set","G1_G2_G3","--encoding","onehot",
         "--features-dir",str(out),"--processed-dir",str(processed)]]:
        result=subprocess.run([sys.executable,"-m","flight_delay.cli",*command],capture_output=True,text=True)
        assert result.returncode==0,result.stderr
