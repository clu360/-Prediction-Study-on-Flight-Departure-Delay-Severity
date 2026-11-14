"""Run unchanged upstream audits on exact node-local copies, then publish their gate."""

from dataclasses import replace
from contextlib import contextmanager
from pathlib import Path
import argparse
import shutil
import subprocess
import tempfile
import time

from flight_delay.experiments.config import Phase3Config, code_fingerprints, file_hash
from flight_delay.experiments.inputs import validate_inputs, gate_path
from flight_delay.experiments.results import read_json, write_json
from flight_delay.experiments.runner import report


@contextmanager
def staged_repository(cfg, archive_root, temporary_parent=Path("/tmp")):
    """Yield an exact disposable source/input copy; output sharing is caller-owned."""
    archive_root, temporary_parent = archive_root.resolve(), temporary_parent.resolve()
    for kind in ["source", "inputs"]:
        metadata = read_json(archive_root / f"{kind}_archive.json")
        if file_hash(archive_root / f"{kind}.tar") != metadata["archive_sha256"]:
            raise ValueError(f"{kind} archive no longer matches its transfer checksum")
    with tempfile.TemporaryDirectory(
        prefix="flight-delay-audit-", dir=temporary_parent
    ) as name:
        local_root = Path(name).resolve()
        assert local_root.is_relative_to(temporary_parent)
        for kind in ["source", "inputs"]:
            subprocess.run(
                [
                    "tar",
                    "-xf",
                    str(archive_root / f"{kind}.tar"),
                    "-C",
                    str(local_root),
                    "--no-same-owner",
                    "--no-same-permissions",
                ],
                check=True,
            )
        current_config = cfg.root / "configs/phase3_pace.yaml"
        if current_config.exists():
            shutil.copyfile(current_config, local_root / "configs/phase3_pace.yaml")
        local_cfg = replace(cfg, root=local_root)
        if local_cfg.config_hash != cfg.config_hash or code_fingerprints(
            local_root
        ) != code_fingerprints(cfg.root):
            raise ValueError(
                "Node-local staging changed source/configuration provenance"
            )
        yield local_cfg


def validate_staged(cfg, archive_root, temporary_parent=Path("/tmp")):
    start = time.monotonic()
    with staged_repository(cfg, archive_root, temporary_parent) as local_cfg:
        print(
            "Node-local input staging complete; running unchanged full upstream audits",
            flush=True,
        )
        staging_seconds = time.monotonic() - start
        gate = validate_inputs(local_cfg)
        if (
            gate["code"] != code_fingerprints(cfg.root)
            or gate["config_hash"] != cfg.config_hash
        ):
            raise ValueError("Audit gate does not match shared source/configuration")
        gate["checks"]["exact_node_local_staging"] = True
        gate["node_local_staging_seconds"] = staging_seconds
        gate["total_validation_seconds"] = time.monotonic() - start
        gate_path(cfg).parent.mkdir(parents=True, exist_ok=True)
        for path in gate_path(local_cfg).parent.glob("*.log"):
            shutil.copyfile(path, gate_path(cfg).parent / path.name)
        write_json(gate_path(cfg), gate)
        report(
            cfg,
            status="inputs_validated" if gate["passed"] else "incomplete",
            error="; ".join(gate["errors"]) if gate["errors"] else None,
        )
        print(
            {
                "passed": gate["passed"],
                "cohort_count": gate.get("cohort_count"),
                "errors": gate["errors"],
            },
            flush=True,
        )
        return gate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/phase3_pace.yaml"))
    parser.add_argument("--archive-root", type=Path)
    args = parser.parse_args()
    cfg = Phase3Config.load(args.config)
    result = validate_staged(cfg, args.archive_root or cfg.root.parent)
    raise SystemExit(0 if result["passed"] else 1)
