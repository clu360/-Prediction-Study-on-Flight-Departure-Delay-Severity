"""One sequential runner; separate development and frozen two-month holdout paths."""

from contextlib import contextmanager
from copy import deepcopy
import gc
import importlib.metadata
import os
import shutil
import subprocess
import threading
import time
import warnings
import joblib
import numpy as np
import polars as pl
import psutil
from threadpoolctl import threadpool_limits
from ..features.views import VIEWS
from ..models.registry import create, check_training_labels, ordered_probabilities
from .config import (
    DEV_FOLDS,
    code_fingerprints,
    digest,
    file_hash,
    analysis_fingerprint,
)
from .evaluation import evaluate, airport_metrics
from .inputs import (
    gate_path,
    verified_gate,
    development_fold,
    final_holdout_loader,
    holdout_month,
    origins_for,
)
from .results import (
    ResultStore,
    exclusive,
    write_json,
    read_json,
    write_frame,
    write_model,
    prediction_frame,
    public_tables,
    PROB_COLUMNS,
)
from .tuning import aggregate, rank, config_id


def ids_hash(ids):
    return digest(list(ids))


def probabilities_hash(p):
    import hashlib

    return hashlib.sha256(np.asarray(p, dtype=np.float32).tobytes()).hexdigest()


class Budget:
    """Cooperative limits at operation boundaries, with sampled process peak RSS."""

    def __init__(self, cfg):
        self.cfg, self.started = cfg, time.monotonic()
        self.peak = psutil.Process().memory_info().rss

    def check(self):
        self.peak = max(self.peak, psutil.Process().memory_info().rss)
        if time.monotonic() - self.started > self.cfg["max_wall_seconds"]:
            raise ValueError("Wall-time budget exhausted; resume the same signed run")
        if self.peak > self.cfg["max_rss_gib"] * 1024**3:
            raise ValueError(
                "Process RSS budget exceeded; no official population downsampling is permitted"
            )
        if (
            shutil.disk_usage(self.cfg.root).free
            < self.cfg["min_free_disk_gib"] * 1024**3
        ):
            raise ValueError(
                "Free disk reserve reached; preserve checkpoints and free disk before resuming"
            )

    @contextmanager
    def measure(self):
        self.check()
        stop = threading.Event()

        def sample():
            while not stop.wait(0.05):
                self.peak = max(self.peak, psutil.Process().memory_info().rss)

        thread = threading.Thread(target=sample, daemon=True)
        thread.start()
        try:
            yield
        finally:
            stop.set()
            thread.join()
        self.check()


def context(cfg, gate):
    versions = {
        n: importlib.metadata.version(n)
        for n in [
            "numpy",
            "scipy",
            "scikit-learn",
            "polars",
            "pyarrow",
            "xgboost",
            "joblib",
            "threadpoolctl",
        ]
    }
    return dict(
        config_hash=cfg.config_hash,
        resolved_config=cfg.values,
        grids=cfg.grids,
        code=code_fingerprints(cfg.root),
        input_digest=gate["input_digest"],
        data_kind=gate["data_kind"],
        versions=versions,
        class_order=[0, 1, 2],
        history_policy=cfg["history_policy"],
    )


def _unit_key(model, view, params, fold):
    return f"development/units/{model}_{view}_{config_id(params)}_{fold}"


def _matrix_info(X):
    return dict(
        shape=list(X.shape),
        dtype=str(X.dtype),
        format=X.format,
        csr_bytes=int(X.data.nbytes + X.indices.nbytes + X.indptr.nbytes),
    )


def transform_diagnostics(transformer, unknown, rows):
    return dict(
        unknown_counts=unknown,
        unknown_rates={k: v / rows for k, v in unknown.items()},
        numeric_imputation="Training median; entirely-null training columns use zero",
        imputation_statistics={
            c: float(v)
            for c, v in zip(transformer.numeric, transformer.imputer.statistics_)
        },
        category_cardinalities={
            c: len(v)
            for c, v in zip(transformer.categorical, transformer.encoder.categories_)
        },
        train_rows=transformer.training_rows_,
        encoded_columns=len(transformer.feature_names_out_),
        source_features=transformer.columns,
        scaling="Train-fitted StandardScaler on numerical columns",
        unknown_month_policy="all-zero one-hot block; existing Phase II encoding retained",
    )


def fit_estimator(cfg, name, params, X, y, budget):
    check_training_labels(name, y)
    estimator = create(name, params, cfg["seed"], cfg["threads"])
    start = time.monotonic()
    with (
        budget.measure(),
        threadpool_limits(limits=cfg["threads"]),
        warnings.catch_warnings(record=True) as caught,
    ):
        warnings.simplefilter("always")
        estimator.fit(X, y)
    messages = sorted({f"{type(w.message).__name__}: {w.message}" for w in caught})
    return estimator, time.monotonic() - start, messages


def predict_frozen(model, transformer, X):
    before = (joblib.hash(model), joblib.hash(transformer))
    start = time.monotonic()
    p = ordered_probabilities(model, X)
    if before != (joblib.hash(model), joblib.hash(transformer)):
        raise ValueError("Inference mutated the frozen estimator or transformer")
    return p, time.monotonic() - start


def evaluate_development(
    cfg,
    store,
    bundle,
    model,
    view,
    params,
    fold,
    budget,
    preprocessing_seconds,
    origins,
    save_predictions=False,
    candidate=False,
):
    transformer = deepcopy(
        bundle.fitted_transformer
    )  # candidate-local object; matrices reuse only this fold/view's fitted preprocessing
    estimator, seconds, messages = fit_estimator(
        cfg, model, params, bundle.X_train, bundle.y_train, budget
    )
    fit_id = digest(
        dict(
            model=model,
            view=view,
            params=params,
            train_ids=ids_hash(bundle.train_ids),
            model_hash=joblib.hash(estimator),
            transformer_hash=joblib.hash(transformer),
        )
    )
    p, inference = predict_frozen(estimator, transformer, bundle.X_test)
    record = dict(
        run_id=store.run_id,
        model=model,
        feature_view=view,
        params=params,
        evaluation_stage="development",
        scored_month=fold,
        selected_config_id=config_id(params),
        model_fit_id=fit_id,
        provenance_hash=store.signature,
        train_ids_hash=ids_hash(bundle.train_ids),
        test_ids_hash=ids_hash(bundle.test_ids),
        probability_hash=probabilities_hash(p),
        transformer_hash=joblib.hash(transformer),
        model_hash=joblib.hash(estimator),
        metrics=evaluate(bundle.y_test, p),
        airport_metrics=airport_metrics(
            bundle.y_test, p, origins, cfg["evaluation_airports"]
        ),
        warnings=messages,
        resources=dict(
            fit_seconds=seconds,
            preprocessing_seconds=preprocessing_seconds,
            inference_seconds=inference,
            peak_rss_bytes=budget.peak,
            train_matrix=_matrix_info(bundle.X_train),
            test_matrix=_matrix_info(bundle.X_test),
        ),
        transform=transform_diagnostics(
            transformer, transformer.test_unknown_counts_, len(bundle.y_test)
        ),
    )
    if save_predictions:
        path = (
            store.root
            / f"development/candidate_predictions/{model}_{view}_{config_id(params)}_{fold}.parquet"
            if candidate
            else store.root
            / f"development/selected_predictions/{model}_{view}_{fold}.parquet"
        )
        write_frame(
            path,
            prediction_frame(
                store.run_id,
                model,
                view,
                "development",
                fold,
                bundle.test_ids,
                bundle.y_test,
                p,
                fit_id,
            ),
        )
        return record, path
    return record


def _baseline_bundle(cfg, fold):
    """Feature-independent priors: no vocabulary fitting or four duplicate baselines."""
    from scipy.sparse import csr_matrix

    a, b = [baseline_rows(cfg, fold, role) for role in ["train", "test"]]
    return (
        a,
        b,
        csr_matrix((a.height, 0), dtype=np.float32),
        csr_matrix((b.height, 0), dtype=np.float32),
    )


def baseline_rows(cfg, fold, role):
    from .inputs import memberships, table_scan

    month = int(fold[-2:])
    months = list(range(month - 3, month)) if role == "train" else [month]
    ids = memberships(cfg.path("processed_dir"), fold, role)
    labels = (
        table_scan(cfg.path("features_dir"), "targets", months)
        .select("target_record_id", "target_class")
        .collect(engine="streaming")
    )
    if (
        labels.height != ids.height
        or labels["target_record_id"].n_unique() != labels.height
    ):
        raise ValueError("Baseline label/manifest population mismatch")
    rows = ids.join(labels, on="target_record_id", how="left", validate="1:1").sort(
        "target_record_id"
    )
    if rows["target_class"].null_count():
        raise ValueError("Baseline missing labels")
    return rows


def baseline_development(cfg, store, fold, budget, save_predictions=False):
    start = time.monotonic()
    a, b, X, Y = _baseline_bundle(cfg, fold)
    preprocessing = time.monotonic() - start
    estimator, seconds, messages = fit_estimator(
        cfg, "baseline", {}, X, a["target_class"].to_numpy(), budget
    )
    start = time.monotonic()
    p = ordered_probabilities(estimator, Y)
    inference = time.monotonic() - start
    fit_id = digest(
        dict(
            model="baseline",
            train_ids=ids_hash(a["target_record_id"].to_list()),
            priors=estimator.priors_.tolist(),
        )
    )
    ids, y = b["target_record_id"].to_numpy(), b["target_class"].to_numpy()
    record = dict(
        run_id=store.run_id,
        model="baseline",
        feature_view="independent",
        params={},
        evaluation_stage="development",
        scored_month=fold,
        selected_config_id=config_id({}),
        model_fit_id=fit_id,
        provenance_hash=store.signature,
        train_ids_hash=ids_hash(a["target_record_id"].to_list()),
        test_ids_hash=ids_hash(ids),
        probability_hash=probabilities_hash(p),
        metrics=evaluate(y, p),
        airport_metrics=airport_metrics(
            y, p, origins_for(cfg, ids, int(fold[-2:])), cfg["evaluation_airports"]
        ),
        resources=dict(
            fit_seconds=seconds,
            preprocessing_seconds=preprocessing,
            inference_seconds=inference,
            peak_rss_bytes=budget.peak,
            train_matrix=_matrix_info(X),
            test_matrix=_matrix_info(Y),
        ),
        warnings=messages,
    )
    if save_predictions:
        path = (
            store.root
            / f"development/selected_predictions/baseline_independent_{fold}.parquet"
        )
        write_frame(
            path,
            prediction_frame(
                store.run_id,
                "baseline",
                "independent",
                "development",
                fold,
                ids,
                y,
                p,
                fit_id,
            ),
        )
        return record, path
    return record


def tune(cfg, resume=False, *, gate=None):
    gate = gate or verified_gate(cfg, "development")
    store = ResultStore(cfg.path("output_dir"), context(cfg, gate), resume)
    budget = Budget(cfg)
    with exclusive(cfg.path("output_dir")), exclusive(store.root):
        if (cfg.path("output_dir") / "PRIMARY_HOLDOUT_LOCK.json").exists() or (
            store.root / "final/selection_lock.json"
        ).exists():
            raise ValueError(
                "Holdout execution has begun: development choices are permanently locked"
            )
        if list((store.root / "development/units").glob("*.json")) and not resume:
            raise ValueError("Development checkpoints exist; explicitly pass --resume")
        selected = {}
        learned = [name for name in cfg["models"] if name != "baseline"]

        def run_fold(view, fold, requested):
            missing = [
                (name, params)
                for name, parameters in requested.items()
                for params in parameters
                if store.load(_unit_key(name, view, params, fold)) is None
            ]
            if not missing:
                return
            budget.check()
            print(f"Loading one shared sparse matrix for {view}/{fold}", flush=True)
            start = time.monotonic()
            with budget.measure():
                bundle = development_fold(cfg, fold, view, store.root)
            preprocessing = time.monotonic() - start
            origins = origins_for(cfg, bundle.test_ids, int(fold[-2:]))
            for name, params in missing:
                print(
                    f"Fitting {name}/{view}/{fold}/{config_id(params)} on {len(bundle.y_train):,} rows",
                    flush=True,
                )
                record, path = evaluate_development(
                    cfg,
                    store,
                    bundle,
                    name,
                    view,
                    params,
                    fold,
                    budget,
                    preprocessing,
                    origins,
                    save_predictions=True,
                    candidate=True,
                )
                store.save(_unit_key(name, view, params, fold), record, [path])
                print(
                    f"{name}/{view}/{fold}/{config_id(params)} macro-F1={record['metrics']['macro_f1']:.5f}",
                    flush=True,
                )
                gc.collect()
            del bundle
            gc.collect()

        if "baseline" in cfg["models"]:
            print("Fitting seven feature-independent development baselines", flush=True)
            for fold in cfg["development_folds"]:
                key = _unit_key("baseline", "independent", {}, fold)
                if store.load(key) is None:
                    budget.check()
                    record, path = baseline_development(cfg, store, fold, budget, True)
                    store.save(key, record, [path])
            evidence = [
                store.load(_unit_key("baseline", "independent", {}, fold))
                for fold in cfg["development_folds"]
            ]
            chosen = aggregate(evidence, cfg["development_folds"])
            selected["baseline/independent"] = dict(
                chosen,
                model="baseline",
                feature_view="independent",
                screening_ranking=[chosen],
                finalist_ranking=[chosen],
                tie_rule=cfg["tie_breakers"],
                folds=cfg["development_folds"],
            )

        # One sparse fold/view bundle is reused across every model/candidate.
        print("Stage B: full-population April/July/October screening", flush=True)
        for view in cfg["views"]:
            for fold in cfg["screening_folds"]:
                run_fold(view, fold, {name: cfg.grids[name] for name in learned})
        screening_rankings = {}
        shortlists = {}
        for name in learned:
            for view in cfg["views"]:
                ranked = rank(
                    [
                        aggregate(
                            [
                                store.load(_unit_key(name, view, p, fold))
                                for fold in cfg["screening_folds"]
                            ],
                            cfg["screening_folds"],
                        )
                        for p in cfg.grids[name]
                    ],
                    cfg["tie_tolerance"],
                )
                screening_rankings[(name, view)] = ranked
                shortlists[(name, view)] = ranked[: cfg["shortlist_size"]]
        print(
            "Stage C: complete the seven development folds for shortlisted candidates",
            flush=True,
        )
        for view in cfg["views"]:
            for fold in cfg["development_folds"]:
                run_fold(
                    view,
                    fold,
                    {
                        name: [x["params"] for x in shortlists[(name, view)]]
                        for name in learned
                    },
                )
        for name in learned:
            for view in cfg["views"]:
                finalists = rank(
                    [
                        aggregate(
                            [
                                store.load(_unit_key(name, view, p["params"], fold))
                                for fold in cfg["development_folds"]
                            ],
                            cfg["development_folds"],
                        )
                        for p in shortlists[(name, view)]
                    ],
                    cfg["tie_tolerance"],
                )
                selected[f"{name}/{view}"] = dict(
                    finalists[0],
                    model=name,
                    feature_view=view,
                    screening_ranking=screening_rankings[(name, view)],
                    finalist_ranking=finalists,
                    tie_rule=cfg["tie_breakers"],
                    folds=cfg["development_folds"],
                )

        # Reuse the chosen candidate's verified probabilities; never refit simply
        # to publish predictions. Hardlinks do not duplicate cached data bytes.
        for choice in selected.values():
            name, view = choice["model"], choice["feature_view"]
            for fold in cfg["development_folds"]:
                key = f"development/selected/{name}_{view}_{fold}"
                if store.load(key):
                    continue
                record = store.load(_unit_key(name, view, choice["params"], fold))
                source = store.root / next(iter(record["artifacts"]))
                target = (
                    store.root
                    / f"development/selected_predictions/{name}_{view}_{fold}.parquet"
                )
                target.parent.mkdir(parents=True, exist_ok=True)
                if source != target and not target.exists():
                    try:
                        os.link(source, target)
                    except OSError:
                        shutil.copyfile(source, target)
                if file_hash(source) != file_hash(target):
                    raise ValueError(
                        "Selected prediction copy differs from verified candidate cache"
                    )
                frame = pl.read_parquet(
                    target, columns=["target_record_id", *PROB_COLUMNS]
                )
                if (
                    ids_hash(frame["target_record_id"].to_list())
                    != record["test_ids_hash"]
                    or probabilities_hash(frame.select(PROB_COLUMNS).to_numpy())
                    != record["probability_hash"]
                ):
                    raise ValueError(
                        "Selected candidate probability/ID checksum mismatch"
                    )
                record["resources"]["selected_prediction_replay"] = False
                store.save(key, record, [target])
        locked = dict(
            complete=True,
            run_id=store.run_id,
            run_signature=store.signature,
            selection_stage="development_only",
            class_order=[0, 1, 2],
            selection_rule="7 equal-month macro-F1 mean; tolerance=1e-6; then log loss, population std, median fit seconds, lexical parameters",
            selected=selected,
        )
        locked["selection_hash"] = digest(locked)
        write_json(store.root / "development/selected_configs.json", locked)
        write_json(cfg.path("reports_dir") / "selected_configs.json", locked)
        candidates_rows = []
        selected_rows = []
        for path in sorted((store.root / "development/units").glob("*.json")):
            r = store.load(
                str(path.relative_to(store.root).with_suffix("")).replace("\\", "/")
            )
            candidates_rows.append(
                dict(
                    model=r["model"],
                    feature_view=r["feature_view"],
                    config_id=config_id(r["params"]),
                    scored_month=r["scored_month"],
                    macro_f1=r["metrics"]["macro_f1"],
                    log_loss=r["metrics"]["log_loss"],
                    fit_seconds=r["resources"]["fit_seconds"],
                )
            )
        for path in sorted((store.root / "development/selected").glob("*.json")):
            r = store.load(
                str(path.relative_to(store.root).with_suffix("")).replace("\\", "/")
            )
            selected_rows.append(
                dict(
                    model=r["model"],
                    feature_view=r["feature_view"],
                    scored_month=r["scored_month"],
                    macro_f1=r["metrics"]["macro_f1"],
                )
            )
        write_frame(
            store.root / "development/candidate_metrics.parquet",
            pl.DataFrame(candidates_rows),
        )
        write_frame(
            store.root / "development/selected_fold_metrics.parquet",
            pl.DataFrame(selected_rows),
        )
    report(cfg, store, status="development_complete")
    return store


def validate_selection(cfg, store, selection):
    value = deepcopy(selection)
    expected = value.pop("selection_hash", None)
    if (
        digest(value) != expected
        or not value.get("complete")
        or value.get("run_signature") != store.signature
        or value.get("selection_stage") != "development_only"
    ):
        raise ValueError(
            "Selected configurations are incomplete, modified or from a different run"
        )
    expected_keys = {"baseline/independent"} | {
        f"{m}/{v}" for m in cfg["models"] if m != "baseline" for v in cfg["views"]
    }
    if set(value["selected"]) != expected_keys:
        raise ValueError(
            "All model/view development choices must finish before any holdout scoring"
        )
    for item in value["selected"].values():
        if (
            set(item["fold_scores"]) != set(DEV_FOLDS)
            or item["folds"] != DEV_FOLDS
            or item["params"] not in cfg.grids[item["model"]]
        ):
            raise ValueError(
                "Selection is not backed by seven development months / declared candidates"
            )
        evidence = [
            store.load(
                _unit_key(item["model"], item["feature_view"], item["params"], f)
            )
            for f in DEV_FOLDS
        ]
        if (
            any(r is None for r in evidence)
            or aggregate(evidence, DEV_FOLDS)["evidence_hash"] != item["evidence_hash"]
        ):
            raise ValueError(
                "Selection development checkpoint evidence is missing or changed"
            )


def feature_importance(estimator, transformer):
    names = transformer.feature_names_out_
    values = (
        np.abs(estimator.coef_).sum(axis=0)
        if hasattr(estimator, "coef_")
        else np.asarray(estimator.feature_importances_)
    )
    if len(values) != len(names):
        raise ValueError("Encoded importance dimensions mismatch")
    source = []
    for name in names:
        if name in transformer.numeric:
            source.append(name)
        else:
            candidates = [
                c for c in transformer.categorical if name.startswith(c + "_")
            ]
            if len(candidates) != 1:
                raise ValueError("Encoded importance source mapping is ambiguous")
            source.append(candidates[0])
    return [
        dict(
            encoded_feature=n,
            source_feature=s,
            group=s.split("_")[0].upper(),
            importance=float(v),
        )
        for n, s, v in zip(names, source, values)
    ]


def frozen_final(cfg, selected_path, resume=False, *, gate=None):
    gate = gate or verified_gate(cfg, "frozen_holdout")
    store = ResultStore(cfg.path("output_dir"), context(cfg, gate), resume)
    selection = read_json(selected_path)
    validate_selection(cfg, store, selection)
    budget = Budget(cfg)
    with exclusive(cfg.path("output_dir")), exclusive(store.root):
        lock = store.root / "final/selection_lock.json"
        lock_value = dict(
            selection_hash=selection["selection_hash"],
            training_months=[8, 9, 10],
            scored_months=[11, 12],
            run_signature=store.signature,
            analysis_code_hash=analysis_fingerprint(cfg.root),
        )
        global_lock = cfg.path("output_dir") / "PRIMARY_HOLDOUT_LOCK.json"
        if global_lock.exists() and read_json(global_lock) != lock_value:
            raise ValueError(
                "Primary holdout was already opened under different choices; a new run cannot retune it"
            )
        if lock.exists() and read_json(lock) != lock_value:
            raise ValueError(
                "Holdout selection lock differs; no tuning or replacement allowed"
            )
        write_json(global_lock, lock_value)
        write_json(
            lock, lock_value
        )  # immutable choices BEFORE any holdout targets are opened
        reference_train = None
        reference_test = {}
        for item in selection["selected"].values():
            name, view, params = item["model"], item["feature_view"], item["params"]
            key = f"final/fits/{name}_{view}"
            fitted = store.load(key)
            if fitted:
                if not resume:
                    raise ValueError(
                        "Final fit already exists; use --resume; never refit after scoring"
                    )
                state = joblib.load(
                    store.root / f"final/frozen_model_{name}_{view}.joblib"
                )
                estimator, transformer = state["estimator"], state["transformer"]
            else:
                start = time.monotonic()
                if name == "baseline":
                    from scipy.sparse import csr_matrix

                    a = baseline_rows(cfg, "2025-11", "train")
                    transformer = None
                    X = csr_matrix((a.height, 0), dtype=np.float32)
                    y, train_ids = (
                        a["target_class"].to_numpy(),
                        a["target_record_id"].to_numpy(),
                    )
                else:
                    with budget.measure():
                        transformer, X, y, train_ids = final_holdout_loader(cfg, view)
                preprocessing = time.monotonic() - start
                estimator, seconds, messages = fit_estimator(
                    cfg, name, params, X, y, budget
                )
                train_digest = ids_hash(train_ids)
                fitted = dict(
                    model=name,
                    feature_view=view,
                    params=params,
                    train_ids_hash=train_digest,
                    train_months=[8, 9, 10],
                    training_cutoff="2025-10-31 target-local date",
                    model_hash=joblib.hash(estimator),
                    transformer_hash=joblib.hash(transformer),
                    warnings=messages,
                    resources=dict(
                        preprocessing_seconds=preprocessing,
                        fit_seconds=seconds,
                        peak_rss_bytes=budget.peak,
                        train_matrix=_matrix_info(X),
                    ),
                )
                fitted["model_fit_id"] = digest(fitted)
                model_path = store.root / f"final/frozen_model_{name}_{view}.joblib"
                transformer_path = (
                    store.root / f"final/frozen_transformer_{view}.joblib"
                )
                write_model(
                    model_path, dict(estimator=estimator, transformer=transformer)
                )
                if transformer_path.exists() and joblib.hash(
                    joblib.load(transformer_path)
                ) != joblib.hash(transformer):
                    raise ValueError(
                        "Final transformer disagrees across models on the same view"
                    )
                if not transformer_path.exists():
                    write_model(transformer_path, transformer)
                if name != "baseline":
                    write_json(
                        store.root / f"final/importance_{name}_{view}.json",
                        feature_importance(estimator, transformer),
                    )
                store.save(key, fitted, [model_path, transformer_path])
                del X, y, train_ids
                gc.collect()
            if reference_train is None:
                reference_train = fitted["train_ids_hash"]
            if (
                fitted["train_ids_hash"] != reference_train
                or fitted["params"] != params
            ):
                raise ValueError(
                    "Final fits do not share exact training IDs/selected parameters"
                )
            month_records = []
            for month in [11, 12]:
                record_key = f"final/months/{name}_{view}_2025-{month:02}"
                record = store.load(record_key)
                path = (
                    store.root
                    / f"final/predictions/{name}_{view}_2025-{month:02}.parquet"
                )
                if record is None:
                    budget.check()
                    start = time.monotonic()
                    if name == "baseline":
                        from scipy.sparse import csr_matrix

                        b = baseline_rows(cfg, f"2025-{month:02}", "test")
                        X = csr_matrix((b.height, 0), dtype=np.float32)
                        y, ids, unknown = (
                            b["target_class"].to_numpy(),
                            b["target_record_id"].to_numpy(),
                            {},
                        )
                    else:
                        X, y, ids, unknown = holdout_month(
                            cfg, month, view, transformer
                        )
                    transform_seconds = time.monotonic() - start
                    with budget.measure():
                        p, inference = predict_frozen(estimator, transformer, X)
                    if (
                        joblib.hash(estimator) != fitted["model_hash"]
                        or joblib.hash(transformer) != fitted["transformer_hash"]
                    ):
                        raise ValueError("Final fit checksum changed between months")
                    record = dict(
                        run_id=store.run_id,
                        model=name,
                        feature_view=view,
                        evaluation_stage="frozen_holdout",
                        scored_month=f"2025-{month:02}",
                        model_fit_id=fitted["model_fit_id"],
                        selected_config_id=config_id(params),
                        provenance_hash=store.signature,
                        train_ids_hash=fitted["train_ids_hash"],
                        test_ids_hash=ids_hash(ids),
                        model_hash=joblib.hash(estimator),
                        transformer_hash=joblib.hash(transformer),
                        metrics=evaluate(y, p),
                        airport_metrics=airport_metrics(
                            y,
                            p,
                            origins_for(cfg, ids, month),
                            cfg["evaluation_airports"],
                        ),
                        warnings=fitted["warnings"],
                        transform=transform_diagnostics(transformer, unknown, len(y))
                        if transformer is not None
                        else dict(feature_independent=True),
                        resources=dict(
                            fitted["resources"],
                            transform_seconds=transform_seconds,
                            inference_seconds=inference,
                            test_matrix=_matrix_info(X),
                        ),
                    )
                    write_frame(
                        path,
                        prediction_frame(
                            store.run_id,
                            name,
                            view,
                            "frozen_holdout",
                            f"2025-{month:02}",
                            ids,
                            y,
                            p,
                            fitted["model_fit_id"],
                        ),
                    )
                    store.save(record_key, record, [path])
                    del X, y, p, ids
                    gc.collect()
                if (
                    record["model_fit_id"] != fitted["model_fit_id"]
                    or record["model_hash"] != fitted["model_hash"]
                    or record["transformer_hash"] != fitted["transformer_hash"]
                ):
                    raise ValueError(
                        "Resumed holdout result belongs to a different fitted state"
                    )
                if (
                    month in reference_test
                    and reference_test[month] != record["test_ids_hash"]
                ):
                    raise ValueError(
                        "Holdout target IDs differ across compared models/views"
                    )
                reference_test[month] = record["test_ids_hash"]
                month_records.append((record, path))
            pooled = pl.concat([pl.read_parquet(p) for _, p in month_records])
            if pooled["target_record_id"].n_unique() != pooled.height:
                raise ValueError("Duplicate targets in pooled holdout")
            origins = np.concatenate(
                [
                    origins_for(
                        cfg,
                        pl.read_parquet(path, columns=["target_record_id"])[
                            "target_record_id"
                        ].to_numpy(),
                        m,
                    )
                    for m, (_, path) in zip([11, 12], month_records)
                ]
            )
            r = dict(
                month_records[0][0],
                scored_month="2025-11_12_pooled",
                metrics=evaluate(
                    pooled["y_true"].to_numpy(), pooled.select(PROB_COLUMNS).to_numpy()
                ),
                airport_metrics=airport_metrics(
                    pooled["y_true"].to_numpy(),
                    pooled.select(PROB_COLUMNS).to_numpy(),
                    origins,
                    cfg["evaluation_airports"],
                ),
            )
            r["test_ids_hash"] = ids_hash(pooled["target_record_id"].to_list())
            store.save(f"final/pooled/{name}_{view}", r)
            del estimator, transformer, pooled
            gc.collect()
        _combine_predictions(store)
    report(cfg, store, status="complete")
    return store


def _combine_predictions(store):
    import pyarrow.parquet as pq
    from .results import atomic

    files = sorted((store.root / "final/predictions").glob("*.parquet"))

    def write(path):
        writer = None
        try:
            for source in files:
                for batch in pq.ParquetFile(source).iter_batches(batch_size=50000):
                    if writer is None:
                        writer = pq.ParquetWriter(path, batch.schema)
                    writer.write_batch(batch)
        finally:
            if writer:
                writer.close()

    atomic(store.root / "final/predictions.parquet", write)


def report(cfg, store=None, status=None, error=None):
    records = []
    candidate_summaries = []
    if store:
        for prefix in ["development/selected", "final/months", "final/pooled"]:
            for path in sorted((store.root / prefix).glob("*.json")):
                records.append(
                    store.load(
                        str(path.relative_to(store.root).with_suffix("")).replace(
                            "\\", "/"
                        )
                    )
                )
        for path in sorted((store.root / "development/units").glob("*.json")):
            r = store.load(
                str(path.relative_to(store.root).with_suffix("")).replace("\\", "/")
            )
            candidate_summaries.append(
                dict(
                    model=r["model"],
                    feature_view=r["feature_view"],
                    config_id=r["selected_config_id"],
                    scored_month=r["scored_month"],
                    params=r["params"],
                    macro_f1=r["metrics"]["macro_f1"],
                    log_loss=r["metrics"]["log_loss"],
                    fit_seconds=r["resources"]["fit_seconds"],
                )
            )
    public_tables(cfg.path("reports_dir"), records)
    write_json(cfg.path("reports_dir") / "candidate_scores.json", candidate_summaries)
    if store and records:
        rows = [
            dict(
                model=r["model"],
                feature_view=r["feature_view"],
                scored_month=r["scored_month"],
                **r["metrics"],
            )
            for r in records
            if r["evaluation_stage"] == "frozen_holdout"
        ]
        if rows:
            write_frame(
                store.root / "final/holdout_metrics.parquet",
                pl.DataFrame(rows, infer_schema_length=None),
            )
    selected = (
        read_json(store.root / "development/selected_configs.json")
        if store and (store.root / "development/selected_configs.json").exists()
        else {
            "complete": False,
            "selected": {},
            "reason": "Development has not completed; no holdout evaluation authorized",
        }
    )
    if status is None and store:
        expected = 3 * (1 + (len(cfg["models"]) - 1) * len(cfg["views"]))
        final_count = sum(r["evaluation_stage"] == "frozen_holdout" for r in records)
        status = (
            "complete"
            if final_count == expected
            else ("development_complete" if selected["complete"] else "partial")
        )
    write_json(cfg.path("reports_dir") / "selected_configs.json", selected)
    importance = []
    if store:
        for path in sorted((store.root / "final").glob("importance_*.json")):
            name = path.stem.removeprefix("importance_")
            importance.extend([dict(model_view=name, **r) for r in read_json(path)])
    write_json(cfg.path("reports_dir") / "feature_importance.json", importance)
    git = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=cfg.root, capture_output=True, text=True
    )
    git_status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=cfg.root, capture_output=True, text=True
    )
    manifest = dict(
        version=cfg["version"],
        completion_status=status or ("partial" if store else "not_started"),
        run_id=store.run_id if store else None,
        git_commit=git.stdout.strip(),
        dirty_worktree=bool(git_status.stdout.strip()),
        resolved_config=cfg.values,
        config_hash=cfg.config_hash,
        candidate_grids=cfg.grids,
        source_code_hashes=code_fingerprints(cfg.root),
        feature_views=VIEWS,
        class_order=[0, 1, 2],
        development_folds=DEV_FOLDS,
        final_training_months=[8, 9, 10],
        final_scored_months=[11, 12],
        baseline_feature_independent=True,
        as_of_history_policy=cfg["history_policy"],
        selected_configs_hash=selected.get("selection_hash"),
        run_context=store.context if store else None,
        completed_candidate_fits=len(candidate_summaries),
        completed_selected_metric_rows=len(records),
        error=error,
        development_results="Selected-config backtests have selection optimism; not independent tests",
        analysis_code_hash=analysis_fingerprint(cfg.root),
        limits=[
            "No IID uncertainty interval: correlated flights",
            "One-hot month unseen in every following-month score",
            "Native fits have cooperative budget checks; no hard preemption inside estimator.fit",
            "Candidate probabilities are cached privately; selected outputs reuse verified files without refitting",
            "Tree impurity/gain importance is biased and associative; permutation analysis disabled",
        ],
    )
    if gate_path(cfg).exists():
        gate = read_json(gate_path(cfg))
        manifest["upstream_validation"] = {
            k: gate.get(k)
            for k in ["passed", "cohort_count", "input_digest", "checks", "errors"]
        }
    write_json(cfg.path("reports_dir") / "experiment_manifest.json", manifest)
    return manifest
