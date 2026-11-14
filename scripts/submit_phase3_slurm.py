"""Submit staged Phase III jobs on PACE; this login-node launcher uses only stdlib."""

from pathlib import Path
import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess


def write_metadata(path, value):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def submit(root, config, account, partition, qos, concurrency, dry_run=False):
    for label in [account, partition, qos]:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", label):
            raise ValueError("Expected simple Slurm account/partition/QoS labels")
    if not 1 <= concurrency <= 16:
        raise ValueError("Array concurrency must be within 1..16")
    directory = root / "data/model_outputs/phase3_pace/slurm"
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "workflow.json").exists():
        raise ValueError(
            "Workflow already exists; inspect recorded job IDs before resubmitting"
        )
    config_arg = shlex.quote(str(config))
    python = '"$phase3_root/.venv/bin/python"'
    common = (
        "#!/bin/bash -l\nset -euo pipefail\n"
        f'phase3_root={shlex.quote(str(root))}\ncd "$phase3_root"\n'
        "module load python/3.12.5\n"
        "export POLARS_MAX_THREADS=4 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4\n"
    )
    runtime = 'export PYTHONPYCACHEPREFIX="$phase3_root/.venv/pycache" PYTHONIOENCODING=utf-8\n'
    batch = f"{python} -u scripts/run_phase3_batch.py"
    staged_batch = f"{python} -u scripts/run_phase3_pace_staged.py"
    cli = f"{python} -u -m flight_delay.cli"
    stages = [
        (
            "environment",
            None,
            None,
            "01:00:00",
            common
            + "python3 -m venv .venv\n"
            + runtime
            + f"{python} -m pip install --disable-pip-version-check -e '.[dev,modeling,notebook]'\n"
            f"{python} -m pip check\n"
            f"{python} -m ipykernel install --prefix .venv --name flight-delay-phase3 --display-name 'Flight Delay (Phase III)'\n"
            f"{python} -m pytest -q\n",
        ),
        (
            "audit_plan_B",
            "environment",
            None,
            "02:00:00",
            common
            + runtime
            + f"{python} -u scripts/validate_phase3_pace.py --config {config_arg}\n"
            f"{staged_batch} prepare --config {config_arg} --stage B\n",
        ),
        (
            "screen_B",
            "audit_plan_B",
            f"0-11%{min(concurrency, 12)}",
            "08:00:00",
            common
            + runtime
            + f'{staged_batch} worker --config {config_arg} --stage B --task-id "$SLURM_ARRAY_TASK_ID"\n',
        ),
        (
            "merge_B_plan_C",
            "screen_B",
            None,
            "01:00:00",
            common + runtime + f"{staged_batch} merge --config {config_arg} --stage B\n"
            f"{staged_batch} prepare --config {config_arg} --stage C\n",
        ),
        (
            "finalists_C",
            "merge_B_plan_C",
            f"0-15%{concurrency}",
            "08:00:00",
            common
            + runtime
            + f'{staged_batch} worker --config {config_arg} --stage C --task-id "$SLURM_ARRAY_TASK_ID"\n',
        ),
        (
            "select",
            "finalists_C",
            None,
            "01:00:00",
            common + runtime + f"{batch} select --config {config_arg}\n",
        ),
        (
            "frozen_final_verify",
            "select",
            None,
            "08:00:00",
            common
            + runtime
            + f"{cli} phase3-final --config {config_arg} --selected reports/phase3/selected_configs.json --resume\n"
            f"{python} scripts/verify_phase3_results.py --config {config_arg}\n"
            f"{cli} phase3-report --config {config_arg}\n"
            f"{python} scripts/summarize_phase3_execution.py --config {config_arg}\n"
            f"{python} scripts/execute_phase3_notebook.py\n",
        ),
    ]
    workflow = dict(
        account=account,
        partition=partition,
        qos=qos,
        concurrency=concurrency,
        cpus_per_task=4,
        memory_per_task_gib=16,
        config=str(config),
        stages=[],
    )
    ids = {}
    for index, (name, parent, array, wall, text) in enumerate(stages):
        path = directory / f"{index:02}_{name}.sbatch"
        path.write_text(text, encoding="utf-8", newline="\n")
        command = [
            "sbatch",
            "--parsable",
            f"--job-name=FD3-{name}",
            f"--account={account}",
            f"--partition={partition}",
            f"--qos={qos}",
            "--nodes=1",
            "--ntasks=1",
            "--cpus-per-task=4",
            "--mem=16G",
            f"--time={wall}",
            f"--chdir={root}",
            f"--output={directory}/{name}-%A_%a.log",
        ]
        if parent:
            command.append(f"--dependency=afterok:{ids[parent]}")
        if array:
            command.extend([f"--array={array}", "--requeue"])
        command.append(str(path))
        if dry_run:
            job_id = str(900000 + index)
        else:
            output = subprocess.check_output(command, text=True).strip()
            job_id = output.split(";")[0]
            if not job_id.isdecimal():
                raise ValueError("Slurm did not return a valid job ID")
        ids[name] = job_id
        workflow["stages"].append(
            dict(
                stage=name,
                job_id=job_id,
                parent=parent,
                array=array,
                walltime=wall,
                script=path.name,
            )
        )
        write_metadata(directory / "workflow.json", workflow)
        public = dict(
            backend="PACE Slurm",
            concurrency=concurrency,
            cpus_per_task=4,
            memory_per_task_gib=16,
            config=str(config),
            stages=workflow["stages"],
            worker_code_sha256=hashlib.sha256(
                (root / "scripts/run_phase3_batch.py").read_bytes()
            ).hexdigest()
            if (root / "scripts/run_phase3_batch.py").exists()
            else None,
            scope="Independent development fold/view jobs; original tuner selects; original frozen final runner fits each chosen model once",
        )
        (root / "reports/phase3").mkdir(parents=True, exist_ok=True)
        write_metadata(root / "reports/phase3/batch_execution.json", public)
        print(f"{'Prepared' if dry_run else 'Submitted'} {name}: {job_id}", flush=True)
    return workflow


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/phase3_pace.yaml"))
    parser.add_argument("--account", required=True)
    parser.add_argument("--partition", default="cpu-gnr")
    parser.add_argument("--qos", default="inferno")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    submit(
        Path(__file__).resolve().parents[1],
        args.config,
        args.account,
        args.partition,
        args.qos,
        args.concurrency,
        args.dry_run,
    )
