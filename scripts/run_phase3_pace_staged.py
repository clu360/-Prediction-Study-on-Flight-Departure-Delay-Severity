"""Run development batches against exact local copies with shared durable outputs."""

from pathlib import Path
import argparse
import importlib.util
import os
import time

from flight_delay.experiments.config import Phase3Config
from validate_phase3_pace import staged_repository


def run(cfg, command, stage, task_id=None, attempt=None):
    started = time.monotonic()
    with staged_repository(cfg, cfg.root.parent) as local_cfg:
        # cfg.path resolves symlinks: all existing locks, manifests, attempt
        # stores and atomic checkpoints keep their original shared locations.
        shared = cfg.path("output_dir")
        shared.mkdir(parents=True, exist_ok=True)
        local_output = local_cfg.root / cfg["output_dir"]
        if local_output.exists() or local_output.is_symlink():
            raise ValueError("Source/input archives must not contain fitted outputs")
        local_output.parent.mkdir(parents=True, exist_ok=True)
        local_output.symlink_to(shared, target_is_directory=True)
        if local_cfg.path("output_dir") != shared:
            raise ValueError("Staged batch outputs do not resolve to the shared store")
        spec = importlib.util.spec_from_file_location(
            "staged_batch", local_cfg.root / "scripts/run_phase3_batch.py"
        )
        batch = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(batch)
        print(
            f"Exact local staging completed in {time.monotonic() - started:.2f}s; "
            "verifying the full development gate before execution",
            flush=True,
        )
        if command == "prepare":
            manifest = batch.prepare(local_cfg, stage)
            print(
                f"Prepared {len(manifest['tasks'])} tasks / {manifest['fit_count']} fits"
            )
            return manifest
        if command == "worker" and task_id is not None:
            return batch.worker(local_cfg, stage, task_id, attempt)
        if command == "merge":
            return batch.merge(local_cfg, stage)
        raise ValueError("Expected prepare, merge, or worker with an explicit task ID")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "worker", "merge"])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=["B", "C"], required=True)
    parser.add_argument("--task-id", type=int)
    args = parser.parse_args()
    attempt = (
        os.environ.get("SLURM_JOB_ID", "local")
        + "-"
        + os.environ.get("SLURM_RESTART_COUNT", "0")
    )
    run(Phase3Config.load(args.config), args.command, args.stage, args.task_id, attempt)
