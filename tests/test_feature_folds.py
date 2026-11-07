import numpy as np
import polars as pl
from scipy import sparse
from flight_delay.features.transforms import FoldTransformer


def test_train_only_vocab_imputation_and_scaler_frozen():
    columns=["g1_carrier","g1_month","g1_distance_miles"]
    train=pl.DataFrame({"g1_carrier":["AA","BB",None],"g1_month":[1,2,3],"g1_distance_miles":[0.,10.,None]})
    test=pl.DataFrame({"g1_carrier":["NEW",None],"g1_month":[4,4],"g1_distance_miles":[1e6,None]})
    transformer=FoldTransformer(columns,batch_rows=1).fit(train)
    mean=transformer.scaler.mean_.copy()
    median=transformer.imputer.statistics_.copy()
    output=transformer.transform(test)
    assert sparse.isspmatrix_csr(output) and output.dtype==np.float32
    assert np.isfinite(output.data).all()
    assert transformer.unknown_counts(test)=={"g1_carrier":1,"g1_month":2}
    assert "NEW" not in transformer.encoder.categories_[0]
    assert np.array_equal(mean,transformer.scaler.mean_)
    assert np.array_equal(median,transformer.imputer.statistics_)
    assert median[0]==5 and mean[0]==5


def test_all_missing_training_numeric_keeps_feature():
    train=pl.DataFrame({"g1_carrier":["AA"],"g3_airport_mean_dep_delay_28d":[None]})
    t=FoldTransformer(train.columns).fit(train)
    assert t.transform(train).shape==(1,2)
    assert np.isfinite(t.transform(train).data).all()
