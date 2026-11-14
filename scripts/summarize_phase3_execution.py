"""Publish aggregate warning, compute and fitted-state provenance after verification."""

from collections import defaultdict
from pathlib import Path
import argparse

from flight_delay.experiments.config import (
    Phase3Config,
    file_hash,
    digest,
    code_fingerprints,
)
from flight_delay.experiments.inputs import gate_path
from flight_delay.experiments.results import ResultStore, read_json, write_json


def summarize_records(development, final_fits):
    groups, preprocessing, final_states, warned = {}, {}, {}, []
    keys = set()
    for stage, records in [
        ("development", development),
        ("frozen_final_fit", final_fits),
    ]:
        for record in records:
            model, view = record["model"], record["feature_view"]
            month = record.get("scored_month", "August-October fit")
            key = (stage, model, view, month, record.get("selected_config_id"))
            if key in keys:
                raise ValueError("Duplicate fitted unit in execution summary")
            keys.add(key)
            group = groups.setdefault(
                (stage, model, view),
                dict(fits=0, fit_seconds=0.0, max_sampled_process_rss_bytes=0),
            )
            resource = record["resources"]
            group["fits"] += 1
            group["fit_seconds"] += resource["fit_seconds"]
            group["max_sampled_process_rss_bytes"] = max(
                group["max_sampled_process_rss_bytes"], resource["peak_rss_bytes"]
            )
            if record.get("warnings"):
                warned.append(
                    dict(
                        stage=stage,
                        model=model,
                        feature_view=view,
                        period=month,
                        config_id=record.get("selected_config_id"),
                        warnings=record["warnings"],
                    )
                )
            if model != "baseline":
                prep_key = f"{stage}/{view}/{month}"
                if (
                    preprocessing.setdefault(prep_key, record["transformer_hash"])
                    != record["transformer_hash"]
                ):
                    raise ValueError(
                        "Preprocessing states disagree within the same training window/view"
                    )
            if stage == "frozen_final_fit":
                final_states[f"{model}/{view}"] = {
                    key: record[key]
                    for key in [
                        "model_fit_id",
                        "model_hash",
                        "transformer_hash",
                        "train_ids_hash",
                    ]
                }
    counts = defaultdict(int)
    for row in warned:
        for message in row["warnings"]:
            counts[message] += 1
    return dict(
        fit_counts=dict(
            development=len(development),
            frozen_final=len(final_fits),
            total=len(development) + len(final_fits),
        ),
        compute=[
            dict(stage=stage, model=model, feature_view=view, **value)
            for (stage, model, view), value in sorted(groups.items())
        ],
        total_unique_fit_seconds=sum(row["fit_seconds"] for row in groups.values()),
        fits_with_warnings=len(warned),
        warning_counts=dict(sorted(counts.items())),
        fit_warnings=warned,
        preprocessing_hashes=dict(sorted(preprocessing.items())),
        frozen_fit_hashes=dict(sorted(final_states.items())),
        scope="Official unique development candidate/baseline fits and final fits only; excludes selected replays and independent reference fits",
        resource_interpretation="Fit seconds summed once per unique fit; shared preprocessing and duplicated monthly final-fit resources are not summed. RSS is the sampled cumulative process peak, not exclusive memory per fit. Fit seconds are not billed CPU seconds or end-to-end elapsed time.",
    )


def publish(cfg):
    reports = cfg.path("reports_dir")
    manifest = read_json(reports / "experiment_manifest.json")
    verification = read_json(reports / "result_verification.json")
    if (
        manifest["completion_status"] != "complete"
        or not verification["passed"]
        or verification["run_id"] != manifest["run_id"]
    ):
        raise ValueError(
            "Complete independently verified results are required for final execution reporting"
        )
    store = ResultStore(cfg.path("output_dir"), manifest["run_context"], True)
    if (
        store.run_id != manifest["run_id"]
        or cfg.config_hash != store.context["config_hash"]
    ):
        raise ValueError(
            "Execution reporting configuration does not match the verified run"
        )
    if code_fingerprints(cfg.root) != store.context["code"]:
        raise ValueError("Scientific source changed before final execution reporting")
    selection = read_json(reports / "selected_configs.json")
    unsigned = {
        key: value for key, value in selection.items() if key != "selection_hash"
    }
    if (
        digest(unsigned) != selection["selection_hash"]
        or selection["selection_hash"] != manifest["selected_configs_hash"]
    ):
        raise ValueError(
            "Published selection evidence changed before execution reporting"
        )
    records = []
    for prefix in ["development/units", "final/fits"]:
        records.append(
            [
                store.load(f"{prefix}/{path.stem}")
                for path in sorted((store.root / prefix).glob("*.json"))
            ]
        )
    learned = [model for model in cfg["models"] if model != "baseline"]
    expected_development = len(cfg["development_folds"]) + len(cfg["views"]) * sum(
        len(cfg["screening_folds"]) * len(cfg.grids[model])
        + (len(cfg["development_folds"]) - len(cfg["screening_folds"]))
        * min(cfg["shortlist_size"], len(cfg.grids[model]))
        for model in learned
    )
    if len(records[0]) != expected_development or len(records[1]) != 1 + len(
        learned
    ) * len(cfg["views"]):
        raise ValueError("Execution summary does not contain every planned unique fit")
    summary = dict(
        run_id=store.run_id,
        data_kind=store.context["data_kind"],
        **summarize_records(*records),
    )
    gate = read_json(gate_path(cfg))
    if not gate["passed"] or gate["input_digest"] != store.context["input_digest"]:
        raise ValueError("Execution summary requires the run's validated input gate")
    schemas = {}
    for relative, expected in gate["input_files"].items():
        if Path(relative).suffix == ".json" and (
            "schema" in Path(relative).name
            or Path(relative).name == "source_contract.json"
        ):
            if file_hash(cfg.root / relative) != expected:
                raise ValueError("Validated input schema changed before reporting")
            schemas[relative] = expected
    summary["input_schema_hashes"] = schemas
    write_json(reports / "execution_summary.json", summary)
    manifest.update(
        warnings=list(summary["warning_counts"]),
        fits_with_warnings=summary["fits_with_warnings"],
        preprocessing_hashes=summary["preprocessing_hashes"],
        frozen_fit_hashes=summary["frozen_fit_hashes"],
        input_schema_hashes=schemas,
        chosen_parameters={
            name: item["params"] for name, item in selection["selected"].items()
        },
        selection_rule=dict(
            primary_metric=cfg["primary_metric"],
            monthly_weighting="equal",
            tie_breakers=cfg["tie_breakers"],
            tie_tolerance=cfg["tie_tolerance"],
        ),
        execution_summary_sha256=file_hash(reports / "execution_summary.json"),
        reporting_code_sha256=file_hash(Path(__file__)),
    )
    write_json(reports / "experiment_manifest.json", manifest)
    print(
        f"Verified execution summary: {summary['fit_counts']}, {summary['fits_with_warnings']} fits with warnings",
        flush=True,
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/phase3.yaml"))
    publish(Phase3Config.load(parser.parse_args().config))
