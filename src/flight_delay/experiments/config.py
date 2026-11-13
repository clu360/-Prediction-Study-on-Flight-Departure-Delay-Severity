"""Strict, fully resolved Phase III protocol configuration."""

from dataclasses import dataclass
from itertools import product
from pathlib import Path
import hashlib
import json
import os
import yaml
from ..features.views import VIEWS, load_views
from ..models.registry import validate_params

FIELDS = set(
    "version year processed_dir features_dir output_dir reports_dir development_folds screening_folds final_train_months final_eval_months views class_order models model_configs primary_metric tie_breakers tie_tolerance shortlist_size seed encoding scale matrix_batch_rows calibration calibration_bins evaluation_airports outer_parallelism threads max_wall_seconds max_rss_gib input_read_timeout_seconds metadata_read_timeout_seconds candidate_budget_reason smoke_rows_per_month history_policy permutation_importance bootstrap_repetitions candidate_prediction_policy min_free_disk_gib".split()
)
DEV_FOLDS = [f"2025-{m:02}" for m in range(4, 11)]
MODELS = ["baseline", "logistic", "random_forest", "xgboost"]


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def candidates(spec):
    if (
        set(spec) != {"name", "fixed", "grid"}
        or not isinstance(spec["fixed"], dict)
        or not isinstance(spec["grid"], dict)
    ):
        raise ValueError("Model YAML requires exactly name, fixed, grid")
    if set(spec["fixed"]) & set(spec["grid"]):
        raise ValueError("Fixed and candidate hyperparameters overlap")
    keys = sorted(spec["grid"])
    if any(not isinstance(spec["grid"][k], list) or not spec["grid"][k] for k in keys):
        raise ValueError("Every candidate axis requires a nonempty list")
    result = [
        dict(spec["fixed"], **dict(zip(keys, values)))
        for values in product(*(spec["grid"][k] for k in keys))
    ]
    if len(result) > 8 or len({canonical(p) for p in result}) != len(result):
        raise ValueError("Candidate grids must be unique and bounded to at most 8")
    for params in result:
        validate_params(spec["name"], params)
    return sorted(result, key=canonical)


@dataclass(frozen=True)
class Phase3Config:
    root: Path
    values: dict
    grids: dict

    def __getitem__(self, key):
        return self.values[key]

    def path(self, key):
        return (self.root / self.values[key]).resolve()

    @property
    def config_hash(self):
        return digest({"resolved": self.values, "candidate_grids": self.grids})

    @classmethod
    def load(cls, path=Path("configs/phase3.yaml")):
        path = Path(path).resolve()
        root = next((p for p in path.parents if (p / "pyproject.toml").exists()), None)
        if root is None:
            raise ValueError("Configuration must reside inside the repository")
        if os.name == "nt" and not str(root).startswith("\\\\?\\"):
            root = Path("\\\\?\\" + str(root))
        values = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(values, dict) or set(values) != FIELDS:
            raise ValueError(
                f"Phase III YAML keys mismatch: {sorted(FIELDS ^ set(values or {}))}"
            )
        locked = dict(
            year=2025,
            development_folds=DEV_FOLDS,
            screening_folds=["2025-04", "2025-07", "2025-10"],
            final_train_months=[8, 9, 10],
            final_eval_months=[11, 12],
            views=list(VIEWS),
            class_order=[0, 1, 2],
            models=MODELS,
            primary_metric="macro_f1",
            tie_breakers=["log_loss", "macro_f1_std", "median_runtime", "lexical_key"],
            tie_tolerance=1e-6,
            seed=6740,
            encoding="onehot",
            scale=True,
            calibration="none",
            calibration_bins=10,
            evaluation_airports=["ATL", "DFW", "DEN", "ORD", "LAX"],
            outer_parallelism=1,
            history_policy="causal_online_inputs_only_frozen_learned_state",
            permutation_importance=False,
            bootstrap_repetitions=0,
            candidate_prediction_policy="persist_all_private",
        )
        for key, expected in locked.items():
            if canonical(values[key]) != canonical(expected):
                raise ValueError(f"Locked Phase III field {key}: expected {expected}")
        if not isinstance(values["version"], str) or not values["version"]:
            raise ValueError("Version tag required")
        for key in [
            "threads",
            "matrix_batch_rows",
            "max_wall_seconds",
            "max_rss_gib",
            "input_read_timeout_seconds",
            "metadata_read_timeout_seconds",
            "smoke_rows_per_month",
            "min_free_disk_gib",
        ]:
            if (
                isinstance(values[key], bool)
                or not isinstance(values[key], (int, float))
                or values[key] <= 0
            ):
                raise ValueError(f"Positive compute budget required: {key}")
        for key in [
            "threads",
            "matrix_batch_rows",
            "smoke_rows_per_month",
            "shortlist_size",
        ]:
            if type(values[key]) is not int:
                raise ValueError(f"Integer required: {key}")
        if values["threads"] > 4 or values["shortlist_size"] not in [1, 2]:
            raise ValueError("At most 4 threads and 2 shortlisted candidates")
        for key in ["processed_dir", "features_dir", "output_dir", "reports_dir"]:
            if (
                not isinstance(values[key], str)
                or Path(values[key]).is_absolute()
                or ".." in Path(values[key]).parts
            ):
                raise ValueError("Use repository-relative dataset and output paths")
        if not (root / values["output_dir"]).is_relative_to(root / "data") or not (
            root / values["reports_dir"]
        ).is_relative_to(root / "reports"):
            raise ValueError(
                "Private outputs must be under data; aggregate reports under reports"
            )
        if set(values["model_configs"]) != set(MODELS):
            raise ValueError("Exactly four model YAMLs required")
        grids = {}
        for name in MODELS:
            model_path = root / values["model_configs"][name]
            if not model_path.resolve().is_relative_to(root):
                raise ValueError("Model YAML outside repository")
            spec = yaml.safe_load(model_path.read_text(encoding="utf-8"))
            if spec["name"] != name:
                raise ValueError("Model YAML name mismatch")
            grids[name] = candidates(spec)
        if {name: len(grid) for name, grid in grids.items()} != {
            "baseline": 1,
            "logistic": 6,
            "random_forest": 8,
            "xgboost": 8,
        } and not values["candidate_budget_reason"]:
            raise ValueError(
                "Reducing official candidate grids requires an explicit candidate_budget_reason"
            )
        load_views(root / "configs/feature_groups.yaml")
        return cls(root, values, grids)


def code_fingerprints(root):
    paths = sorted((root / "src/flight_delay").rglob("*.py"))
    paths += [
        root / "configs/phase3.yaml",
        root / "configs/feature_groups.yaml",
        root / "pyproject.toml",
    ]
    paths += sorted((root / "configs/models").glob("*.yaml"))
    paths = [p for p in paths if p.is_file()]
    for path in paths:
        if getattr(path.stat(), "st_file_attributes", 0) & 0x1000:
            raise ValueError(f"Source/configuration is online-only: {path.name}")
    return {str(p.relative_to(root)).replace("\\", "/"): file_hash(p) for p in paths}


def analysis_fingerprint(root):
    """Freeze executable analysis choices, ignoring Markdown and saved outputs."""
    notebook = root / "notebooks/03_classical_models_analysis.ipynb"
    if not notebook.exists():
        return None
    content = json.loads(notebook.read_text(encoding="utf-8"))
    return digest(
        [
            "".join(cell["source"])
            for cell in content["cells"]
            if cell["cell_type"] == "code"
        ]
    )
