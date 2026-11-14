"""Live upstream gate and immutable month-scoped views of Phase I/II files."""

from pathlib import Path
import os
import re
import subprocess
import sys
import numpy as np
import polars as pl
from ..models.state import state_hash
import psutil
from ..features.transforms import FoldTransformer, build_fold
from ..features.views import VIEWS
from .config import DEV_FOLDS, digest, file_hash, code_fingerprints
from .results import write_json, read_json

ID = "target_record_id"
TABLES = ["full_features", "targets", "metadata"]
PHASE1 = [
    "flights_clean",
    "flight_connections",
    "modeling_cohort",
    "history_departure_events",
    "folds",
]


def gate_path(cfg):
    return cfg.path("output_dir") / "validation" / f"{cfg.config_hash}.json"


def local_file(path):
    if not Path(path).is_file():
        raise ValueError(f"Required local artifact missing: {Path(path).name}")
    if getattr(Path(path).stat(), "st_file_attributes", 0) & 0x1000:
        raise ValueError(
            f"OneDrive online-only artifact: {Path(path).name}; make upstream data available locally"
        )


def bounded_python(cfg, arguments, timeout):
    env = dict(os.environ, PYTHONPYCACHEPREFIX=str(cfg.root / ".venv/pycache"))
    command = [sys.executable, "-u", *arguments]
    process = subprocess.Popen(
        command,
        cwd=cfg.root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        # Windows venv launchers spawn a base-Python child; kill the owned tree,
        # not just the launcher, so a cloud-file stall cannot leak processes.
        try:
            children = psutil.Process(process.pid).children(recursive=True)
        except psutil.NoSuchProcess:
            children = []
        for child in reversed(children):
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        process.kill()
        process.communicate()
        raise
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


def validate_inputs(cfg):
    """No training. Existing audits run in a bounded subprocess, not a stalled CLI."""
    result = {
        "config_hash": cfg.config_hash,
        "passed": False,
        "data_kind": "real",
        "checks": {},
        "code": code_fingerprints(cfg.root),
        "input_files": {},
        "errors": [],
    }
    gate_path(cfg).parent.mkdir(parents=True, exist_ok=True)
    operation = "local artifact inventory"
    timeout = cfg["metadata_read_timeout_seconds"]
    try:
        files = []
        for key, tables in [("processed_dir", PHASE1), ("features_dir", TABLES)]:
            root = cfg.path(key)
            for table in tables:
                parts = sorted((root / table).glob("year=2025/month=*/*.parquet"))
                if not parts or {p.parent.name for p in parts} != {
                    f"month={m:02}" for m in range(1, 13)
                }:
                    raise ValueError(f"Missing twelve-month table: {table}")
                files += parts
            files += [
                root / "audit" / name
                for name in [
                    "SUCCESS.json",
                    "run_signature.json",
                    "logical_hashes.json",
                ]
            ]
        files += [
            cfg.path("processed_dir") / "audit" / name
            for name in ["input_manifest.json", "execution.json", "schema.json"]
        ]
        files += [
            cfg.path("features_dir") / "audit" / name
            for name in ["source_contract.json", "feature_schema.json"]
        ]
        for path in files:
            local_file(path)
        operation = "essential upstream audit metadata read"
        metadata = [p for p in files if p.suffix == ".json"]
        probe_script = "from pathlib import Path; import sys\nfor p in sys.argv[1:]:\n print(p,flush=True)\n Path(p).read_bytes()\nprint('METADATA_READABLE')"
        probe = bounded_python(
            cfg, ["-c", probe_script, *[str(p) for p in metadata]], timeout
        )
        (gate_path(cfg).parent / "metadata_probe.log").write_text(
            probe.stdout + probe.stderr, encoding="utf-8"
        )
        if probe.returncode:
            raise ValueError(
                "Essential upstream audit metadata is unreadable; no training is permitted"
            )
        # Master schedule and history caches are essential for the independent audits.
        signature = read_json(cfg.path("processed_dir") / "audit/run_signature.json")
        master = cfg.path("processed_dir").parent / "interim" / signature["run_key"]
        master_files = sorted(master.glob("bucket-*/*.parquet"))
        cache_files = sorted((cfg.path("features_dir") / "cache").rglob("*.parquet"))
        if not master_files or not cache_files:
            raise ValueError("Full master schedule/history cache unavailable")
        files += master_files + cache_files
        for path in files:
            local_file(path)
        script = (
            "from pathlib import Path; import sys; from flight_delay.audit import audit; "
            "from flight_delay.features.validation import audit_features; "
            "a=audit(Path(sys.argv[1])); b=audit_features(Path(sys.argv[2]),processed=Path(sys.argv[1])); "
            "assert a['acceptance_passed'] and b['acceptance_passed']; "
            "assert len(b['acceptance_checks'])==19; print('UPSTREAM_AUDITS_PASSED')"
        )
        operation = "independent full-data upstream audits"
        timeout = cfg["input_read_timeout_seconds"]
        process = bounded_python(
            cfg,
            [
                "-c",
                script,
                str(cfg.path("processed_dir")),
                str(cfg.path("features_dir")),
            ],
            timeout,
        )
        (gate_path(cfg).parent / "upstream_audit.log").write_text(
            process.stdout + process.stderr, encoding="utf-8"
        )
        if process.returncode or "UPSTREAM_AUDITS_PASSED" not in process.stdout:
            raise ValueError(
                "Independent upstream audits failed; inspect private upstream_audit.log"
            )
        result["checks"]["fresh_phase1_and_19_phase2_checks"] = True
        for path in files:
            result["input_files"][
                str(path.relative_to(cfg.root)).replace("\\", "/")
            ] = file_hash(path)
        summary = read_json(cfg.path("features_dir") / "audit/phase2_summary.json")
        counts = summary["counts"]
        if len(set(counts.values())) != 1 or not counts["full_features"]:
            raise ValueError("Feature/label/metadata population counts disagree")
        result.update(
            passed=True,
            cohort_count=counts["full_features"],
            input_digest=digest(result["input_files"]),
        )
    except subprocess.TimeoutExpired:
        result["errors"] = [
            f"{operation} exceeded {timeout} seconds; upstream validation remains incomplete"
        ]
    except (OSError, ValueError) as exc:
        result["errors"] = [public_error(exc, cfg)]
    write_json(gate_path(cfg), result)
    return result


def public_error(exc, cfg):
    text = str(exc)
    for root in [str(cfg.root), str(cfg.root).removeprefix("\\\\?\\")]:
        text = text.replace(root, "<repository>").replace(
            repr(root)[1:-1], "<repository>"
        )
    if re.search(r"[A-Za-z]:[\\/]", text):
        return f"{type(exc).__name__}: file operation failed; inspect private execution logs"
    return text


def verified_gate(cfg, stage):
    if stage not in ["development", "frozen_holdout"]:
        raise ValueError("Unknown execution stage")
    gate = read_json(gate_path(cfg))
    if (
        not gate["passed"]
        or gate["data_kind"] != "real"
        or gate["config_hash"] != cfg.config_hash
        or gate["code"] != code_fingerprints(cfg.root)
    ):
        raise ValueError(
            "Run phase3-validate successfully with the current code/config before real experiments"
        )
    for relative, expected in gate["input_files"].items():
        # Never open Nov/Dec feature/label rows in the development execution path.
        if stage == "development" and re.search(r"/month=(11|12)/", relative):
            continue
        local_file(cfg.root / relative)
        if file_hash(cfg.root / relative) != expected:
            raise ValueError(
                "Upstream content changed after validation; rerun the gate"
            )
    return gate


def table_scan(root, table, months):
    if not months or not set(months).issubset(range(1, 13)):
        raise ValueError("Invalid source months")
    paths = []
    for month in sorted(months):
        parts = sorted(
            (Path(root) / table / "year=2025" / f"month={month:02}").glob("*.parquet")
        )
        if not parts:
            raise ValueError(f"Missing partition {table}/{month}")
        paths += parts
    return pl.scan_parquet(paths, hive_partitioning=False)


def memberships(processed, fold, role):
    month = int(fold[-2:])
    months = list(range(month - 3, month)) if role == "train" else [month]
    rows = (
        table_scan(processed, "folds", months)
        .filter((pl.col("fold_id") == fold) & (pl.col("role") == role))
        .collect(engine="streaming")
    )
    if (
        not rows.height
        or rows[ID].null_count()
        or rows[ID].n_unique() != rows.height
        or set(rows["source_month"].to_list()) != set(months)
    ):
        raise ValueError("Invalid exact Phase I membership IDs/months")
    if rows.filter(
        (pl.col("test_month") != month) | (pl.col("is_final_holdout") != (month >= 11))
    ).height:
        raise ValueError("Phase I fold metadata mismatch")
    return rows.select(ID)


def checked_rows(cfg, ids, months, view, labels=True):
    if view not in VIEWS or ids[ID].null_count() or ids[ID].n_unique() != ids.height:
        raise ValueError("Unknown view or duplicate/null stable IDs")
    features = (
        table_scan(cfg.path("features_dir"), "full_features", months)
        .select(ID, *VIEWS[view])
        .collect(engine="streaming")
    )
    if (
        features.height != ids.height
        or features[ID].n_unique() != features.height
        or features[ID].null_count()
    ):
        raise ValueError("Feature population differs from manifest")
    if (
        features.select(ID).join(ids, on=ID, how="anti").height
        or ids.join(features.select(ID), on=ID, how="anti").height
    ):
        raise ValueError("Missing/extra feature IDs")
    frame = ids.join(features, on=ID, validate="1:1")
    if labels:
        targets = (
            table_scan(cfg.path("features_dir"), "targets", months)
            .select(ID, "target_class")
            .collect(engine="streaming")
        )
        if targets.height != ids.height or targets[ID].n_unique() != targets.height:
            raise ValueError("Target population differs from manifest")
        frame = frame.join(targets, on=ID, how="left", validate="1:1")
        if (
            frame["target_class"].null_count()
            or not frame["target_class"].is_in([0, 1, 2]).all()
        ):
            raise ValueError("Missing/invalid target labels")
    return frame.sort(ID)


def _scoped_inputs(cfg, fold, private_root):
    """Read-only hardlink projection lets the UNCHANGED Phase II loader read dev months only.

    No bytes are copied or rewritten; each fold has a separate facade. Other months
    do not exist in the facade, so the original glob scans cannot open holdout data.
    """
    if fold not in DEV_FOLDS:
        raise ValueError("Development loader cannot open a holdout fold")
    month = int(fold[-2:])
    scope = Path(private_root) / "input_scopes" / fold
    expected_files = set()
    for root_key, label, tables in [
        ("features_dir", "features", ["full_features", "targets"]),
        ("processed_dir", "processed", ["folds"]),
    ]:
        for table in tables:
            for m in range(month - 3, month + 1):
                parts = sorted(
                    (cfg.path(root_key) / table / "year=2025" / f"month={m:02}").glob(
                        "*.parquet"
                    )
                )
                if not parts:
                    raise ValueError("Missing development partition")
                for source in parts:
                    target = scope / label / source.relative_to(cfg.path(root_key))
                    expected_files.add(target.resolve())
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if target.exists():
                        if not os.path.samefile(source, target):
                            raise ValueError(
                                "Stale input scope: hardlink identity mismatch"
                            )
                    else:
                        os.link(source, target)
    if {p.resolve() for p in scope.rglob("*.parquet")} != expected_files:
        raise ValueError(
            "Unexpected/stale partitions in development input scope; do not scan them"
        )
    return scope


def development_fold(cfg, fold, view, private_root):
    scope = _scoped_inputs(cfg, fold, private_root)
    # Check exact population before the existing loader's inner join can hide extras.
    month = int(fold[-2:])
    for role, months in [("train", list(range(month - 3, month))), ("test", [month])]:
        ids = memberships(scope / "processed", fold, role)
        for table in ["full_features", "targets"]:
            source_ids = (
                table_scan(scope / "features", table, months)
                .select(ID)
                .collect(engine="streaming")
            )
            if (
                source_ids.height != ids.height
                or source_ids[ID].n_unique() != source_ids.height
                or ids.join(source_ids, on=ID, how="anti").height
            ):
                raise ValueError(
                    "Development feature/target IDs differ from exact manifest"
                )
    bundle = build_fold(
        fold,
        view,
        encoding=cfg["encoding"],
        features_dir=scope / "features",
        processed_dir=scope / "processed",
        scale=cfg["scale"],
        batch_rows=cfg["matrix_batch_rows"],
    )
    check_bundle(bundle)
    return bundle


def check_bundle(bundle):
    from scipy.sparse import isspmatrix_csr

    for X, y, ids in [
        (bundle.X_train, bundle.y_train, bundle.train_ids),
        (bundle.X_test, bundle.y_test, bundle.test_ids),
    ]:
        if (
            not isspmatrix_csr(X)
            or X.dtype != np.float32
            or len(y) != len(ids)
            or X.shape[0] != len(y)
            or len(set(ids)) != len(ids)
            or not np.isfinite(X.data).all()
        ):
            raise ValueError("Invalid CSR float32 matrix/label/ID alignment")
    if (
        set(bundle.train_ids) & set(bundle.test_ids)
        or bundle.X_train.shape[1] != bundle.X_test.shape[1]
    ):
        raise ValueError("Overlapping populations or incompatible encodings")


def final_holdout_loader(cfg, view):
    """Fit only Aug-Oct. This never calls build_fold, especially not 2025-12."""
    ids = memberships(cfg.path("processed_dir"), "2025-11", "train")
    frame = checked_rows(cfg, ids, [8, 9, 10], view)
    transformer = FoldTransformer(
        VIEWS[view], scale=cfg["scale"], batch_rows=cfg["matrix_batch_rows"]
    ).fit(frame)
    return (
        transformer,
        transformer.transform(frame),
        frame["target_class"].to_numpy(),
        frame[ID].to_numpy(),
    )


def holdout_month(cfg, month, view, transformer):
    if month not in [11, 12] or transformer.columns != VIEWS[view]:
        raise ValueError("Only the two frozen holdout months are permitted")
    before = state_hash(transformer)
    ids = memberships(cfg.path("processed_dir"), f"2025-{month:02}", "test")
    frame = checked_rows(cfg, ids, [month], view)
    X = transformer.transform(frame)
    unknown = transformer.unknown_counts(frame)
    if state_hash(transformer) != before:
        raise ValueError("Holdout transform mutated learned state")
    return X, frame["target_class"].to_numpy(), frame[ID].to_numpy(), unknown


def origins_for(cfg, ids, month):
    meta = (
        table_scan(cfg.path("features_dir"), "metadata", [month])
        .select(ID, "target_origin", "prediction_time_utc", "history_max_event_utc")
        .collect(engine="streaming")
    )
    if (
        meta.height != len(ids)
        or meta[ID].n_unique() != meta.height
        or meta.filter(
            pl.col("history_max_event_utc") >= pl.col("prediction_time_utc")
        ).height
    ):
        raise ValueError("Metadata population or strict causal history cutoff violated")
    aligned = pl.DataFrame({ID: ids}).join(meta, on=ID, how="left", validate="1:1")
    if aligned["target_origin"].null_count():
        raise ValueError("Origin metadata missing")
    return aligned["target_origin"].to_numpy()
