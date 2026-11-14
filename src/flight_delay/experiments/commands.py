"""Small explicit CLI dispatcher with truthful partial-run reporting."""

import json
from .config import Phase3Config
from .inputs import validate_inputs, gate_path, public_error
from .results import read_json, ResultStore
from .runner import tune, frozen_final, report, context


def dispatch(args):
    cfg = Phase3Config.load(args.config)
    if args.command == "phase3-validate":
        gate = validate_inputs(cfg)
        report(
            cfg,
            status="inputs_validated"
            if gate["passed"]
            else "blocked_upstream_unavailable",
            error="; ".join(gate["errors"]) if gate["errors"] else None,
        )
        print(
            json.dumps(
                {
                    "passed": gate["passed"],
                    "cohort_count": gate.get("cohort_count"),
                    "errors": gate["errors"],
                },
                indent=2,
            )
        )
        return 0 if gate["passed"] else 1
    if args.command == "phase3-smoke":
        from .synthetic import smoke

        print(json.dumps(smoke(cfg, args.resume), indent=2))
        return 0
    if args.command == "phase3-report":
        gate = read_json(gate_path(cfg)) if gate_path(cfg).exists() else None
        store = None
        if gate and gate["passed"]:
            store = ResultStore(cfg.path("output_dir"), context(cfg, gate), True)
        result = report(
            cfg, store, status=None if store else "blocked_upstream_unavailable"
        )
        print(
            json.dumps(
                {
                    "status": result["completion_status"],
                    "completed_candidate_fits": result["completed_candidate_fits"],
                }
            )
        )
        return 0
    store = None
    try:
        if args.command == "phase3-tune":
            store = tune(cfg, args.resume)
        elif args.command == "phase3-final":
            store = frozen_final(cfg, args.selected, args.resume)
    except (ValueError, OSError) as exc:
        gate = read_json(gate_path(cfg)) if gate_path(cfg).exists() else None
        if gate and gate.get("passed"):
            store = ResultStore(cfg.path("output_dir"), context(cfg, gate), True)
        report(cfg, store, status="incomplete", error=public_error(exc, cfg))
        raise
    print(json.dumps({"run_id": store.run_id, "status": "completed_requested_stage"}))
    return 0
