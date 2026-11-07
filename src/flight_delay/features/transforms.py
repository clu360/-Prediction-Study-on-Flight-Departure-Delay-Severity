"""Train-only fold transforms returning sparse CSR; no models are trained."""
from __future__ import annotations
import hashlib
from pathlib import Path
from typing import NamedTuple
import numpy as np
import polars as pl
from scipy import sparse
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from .contracts import scan as source_scan
from .validation import scan
from .views import CATEGORICAL, load_views


class FoldMatrices(NamedTuple):
    X_train: sparse.csr_matrix
    X_test: sparse.csr_matrix
    y_train: np.ndarray
    y_test: np.ndarray
    train_ids: np.ndarray
    test_ids: np.ndarray
    fitted_transformer: "FoldTransformer"


class FoldTransformer:
    """Fit one view on train only; transform later rows in bounded batches."""
    def __init__(self, columns: list[str], scale: bool = True, batch_rows: int = 50000):
        self.columns = list(columns)
        self.categorical = [c for c in columns if c in CATEGORICAL]
        self.numeric = [c for c in columns if c not in CATEGORICAL]
        self.scale = scale
        self.batch_rows = batch_rows

    def _categories(self, frame: pl.DataFrame) -> pl.DataFrame:
        return frame.select([pl.col(c).cast(pl.String).fill_null("__MISSING__") for c in self.categorical])

    def fit(self, train: pl.DataFrame) -> "FoldTransformer":
        if not train.height:
            raise ValueError("Cannot fit an empty training fold")
        if not set(self.columns).issubset(train.columns):
            raise ValueError("Training data missing allowlisted features")
        categorical = self._categories(train)
        vocabularies = [categorical[c].unique().sort().to_list() for c in self.categorical]
        self.encoder = OneHotEncoder(categories=vocabularies, handle_unknown="ignore", sparse_output=True, dtype=np.float32)
        self.encoder.fit(np.array([[v[0] for v in vocabularies]], dtype=object))
        self.imputer = SimpleImputer(strategy="median", keep_empty_features=True, copy=False)
        numbers = train.select([pl.col(c).cast(pl.Float32) for c in self.numeric]).to_numpy(order="c", writable=True)
        numbers = self.imputer.fit_transform(numbers)
        self.scaler = StandardScaler(copy=False).fit(numbers) if self.scale else None
        self.training_rows_ = train.height
        self.feature_names_out_ = list(self.encoder.get_feature_names_out(self.categorical))+self.numeric
        return self

    def transform(self, frame: pl.DataFrame) -> sparse.csr_matrix:
        chunks = []
        for batch in frame.iter_slices(self.batch_rows):
            categories = self.encoder.transform(self._categories(batch).to_numpy())
            numbers = batch.select([pl.col(c).cast(pl.Float32) for c in self.numeric]).to_numpy(order="c", writable=True)
            numbers = self.imputer.transform(numbers)
            if self.scaler is not None:
                numbers = self.scaler.transform(numbers)
            chunks.append(sparse.hstack([categories, sparse.csr_matrix(numbers)], format="csr", dtype=np.float32))
        result = sparse.vstack(chunks, format="csr") if chunks else sparse.csr_matrix((0, len(self.feature_names_out_)), dtype=np.float32)
        if not np.isfinite(result.data).all():
            raise ValueError("Transformed matrix contains nonfinite values")
        return result

    def unknown_counts(self, frame: pl.DataFrame) -> dict[str, int]:
        categories = self._categories(frame)
        return {c: categories.filter(~pl.col(c).is_in(self.encoder.categories_[i].tolist())).height
                for i, c in enumerate(self.categorical)}


def build_fold(fold_id: str, feature_set: str, encoding: str = "onehot", *,
               features_dir: Path = Path("data/features"), processed_dir: Path = Path("data/processed"),
               scale: bool = True, batch_rows: int = 50000) -> FoldMatrices:
    if encoding != "onehot":
        raise ValueError("Phase II supports onehot CSR; embedding/model training is not implemented")
    views = load_views()
    if feature_set not in views:
        raise ValueError(f"Unknown feature view: {feature_set}")
    columns = views[feature_set]
    memberships = source_scan(processed_dir, "folds").filter(pl.col("fold_id") == fold_id).select(
        "target_record_id", "role", "source_month", "test_month", "is_final_holdout").collect(engine="streaming")
    if not memberships.height or memberships["target_record_id"].n_unique() != memberships.height:
        raise ValueError("Unknown fold or duplicate fold membership")
    test_month = int(fold_id.split("-")[1])
    if not 4 <= test_month <= 12:
        raise ValueError("Only the nine existing rolling folds are supported")
    invalid = memberships.filter(((pl.col("role") == "train") &
        ~pl.col("source_month").is_in(list(range(test_month-3, test_month)))) |
        ((pl.col("role") == "test") & (pl.col("source_month") != test_month)) |
        (pl.col("is_final_holdout") != (test_month >= 11)))
    if invalid.height:
        raise ValueError("Existing fold calendar contract violated")
    # Membership comes from Phase I, never a fresh random/date-derived split.
    feature_rows = scan(features_dir, "full_features").select("target_record_id", *columns).with_columns(
        pl.lit(True).alias("_feature_row_present"))
    labels = scan(features_dir, "targets").select("target_record_id", "target_class")
    roles = []
    for role in ["train", "test"]:
        ids = memberships.filter(pl.col("role") == role).select("target_record_id")
        frame = ids.lazy().join(feature_rows, on="target_record_id", how="left", validate="1:1").join(
            labels, on="target_record_id", how="left", validate="1:1").sort("target_record_id").collect(engine="streaming")
        if frame.height != ids.height or frame["target_class"].null_count() or frame["_feature_row_present"].null_count():
            raise ValueError("Fold ID/label alignment failed")
        roles.append(frame)
    train, test = roles
    transformer = FoldTransformer(columns, scale, batch_rows).fit(train)
    transformer.fold_id_ = fold_id
    transformer.is_final_holdout_ = test_month >= 11
    transformer.test_unknown_counts_ = transformer.unknown_counts(test)
    return FoldMatrices(transformer.transform(train), transformer.transform(test),
        train["target_class"].to_numpy(), test["target_class"].to_numpy(),
        train["target_record_id"].to_numpy(), test["target_record_id"].to_numpy(), transformer)


def matrix_summary(bundle: FoldMatrices, view: str) -> dict:
    sizes = lambda x: x.data.nbytes+x.indices.nbytes+x.indptr.nbytes
    digest = lambda ids: hashlib.sha256("\n".join(ids).encode()).hexdigest()
    return dict(feature_set=view, X_train_shape=list(bundle.X_train.shape), X_test_shape=list(bundle.X_test.shape),
        train_rows=len(bundle.y_train), test_rows=len(bundle.y_test),
        train_ids_sha256=digest(bundle.train_ids), test_ids_sha256=digest(bundle.test_ids),
        train_class_counts=np.bincount(bundle.y_train, minlength=3).tolist(),
        test_class_counts=np.bincount(bundle.y_test, minlength=3).tolist(),
        train_csr_bytes=sizes(bundle.X_train), test_csr_bytes=sizes(bundle.X_test), dtype=str(bundle.X_train.dtype),
        source_column_count=len(bundle.fitted_transformer.columns), train_only_fit=True,
        unknown_categories_test=bundle.fitted_transformer.test_unknown_counts_,
        class_order=[0, 1, 2], is_final_holdout=bundle.fitted_transformer.is_final_holdout_)
