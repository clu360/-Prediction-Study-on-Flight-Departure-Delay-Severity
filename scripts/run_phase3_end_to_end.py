"""Sequential full-data execution, bounded resumption, verification and analysis."""

from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import time

from flight_delay.experiments.config import Phase3Config, code_fingerprints
from flight_delay.experiments.results import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/phase3.yaml"))
    parser.add_argument(
        "--resume-validated",
        action="store_true",
        help="Reuse a passed full audit only after rechecking its source and input fingerprints",
    )
    args = parser.parse_args()
    cfg = Phase3Config.load(args.config)
    frozen_source = code_fingerprints(cfg.root)
    output = cfg.path("output_dir") / "execution"
    output.mkdir(parents=True, exist_ok=True)
    env = dict(
        os.environ,
        PYTHONPYCACHEPREFIX=str(cfg.root / ".venv/pycache"),
        POLARS_MAX_THREADS=str(cfg["threads"]),
        PYTHONIOENCODING="utf-8",
    )
    steps = [
        (
            "validate",
            ["-m", "flight_delay.cli", "phase3-validate", "--config", str(args.config)],
        ),
        (
            "development",
            [
                "-m",
                "flight_delay.cli",
                "phase3-tune",
                "--config",
                str(args.config),
                "--resume",
            ],
        ),
        (
            "frozen_holdout",
            [
                "-m",
                "flight_delay.cli",
                "phase3-final",
                "--config",
                str(args.config),
                "--selected",
                str(cfg.path("reports_dir") / "selected_configs.json"),
                "--resume",
            ],
        ),
        (
            "verify_results",
            ["scripts/verify_phase3_results.py", "--config", str(args.config)],
        ),
        (
            "report",
            ["-m", "flight_delay.cli", "phase3-report", "--config", str(args.config)],
        ),
        (
            "execution_summary",
            ["scripts/summarize_phase3_execution.py", "--config", str(args.config)],
        ),
        ("notebook", ["scripts/execute_phase3_notebook.py"]),
    ]
    completed = []
    if args.resume_validated:
        from flight_delay.experiments.inputs import verified_gate

        gate = verified_gate(cfg, "development")
        steps = steps[1:]
        completed.append(
            dict(
                stage="validate",
                reused_verified_gate=True,
                input_digest=gate["input_digest"],
            )
        )
        print(
            "Reusing passed full audit after source/input fingerprint checks",
            flush=True,
        )
    for stage, arguments in steps:
        attempts = 0
        while True:
            if code_fingerprints(cfg.root) != frozen_source:
                raise ValueError(
                    "Source/config changed during execution; refusing mixed experiment provenance"
                )
            attempts += 1
            write_json(
                output / "status.json",
                dict(
                    status="running", stage=stage, attempt=attempts, completed=completed
                ),
            )
            print(f"START {stage}, attempt {attempts}", flush=True)
            lines = []
            start = time.monotonic()
            with (output / f"{stage}-{attempts}.log").open(
                "w", encoding="utf-8"
            ) as log:
                process = subprocess.Popen(
                    [sys.executable, "-u", *arguments],
                    cwd=cfg.root,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                )
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    print(line, end="", flush=True)
                    lines.append(line)
                code = process.wait()
            if code == 0:
                completed.append(
                    dict(
                        stage=stage, seconds=time.monotonic() - start, attempts=attempts
                    )
                )
                break
            if (
                stage in {"development", "frozen_holdout"}
                and attempts < 8
                and any("Wall-time budget exhausted" in line for line in lines)
            ):
                print(
                    "Resuming the same verified checkpoints after command wall budget",
                    flush=True,
                )
                continue
            write_json(
                output / "status.json",
                dict(
                    status="incomplete",
                    stage=stage,
                    exit_code=code,
                    completed=completed,
                ),
            )
            raise SystemExit(code)
    write_json(output / "status.json", dict(status="complete", completed=completed))
    print(
        json.dumps({"status": "full_experiment_and_verification_complete"}), flush=True
    )


if __name__ == "__main__":
    main()
