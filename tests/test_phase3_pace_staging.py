"""Node-local audit staging preserves provenance and never promotes a failed gate."""

import importlib.util
from pathlib import Path
import tarfile
from contextlib import contextmanager
from dataclasses import replace
import shutil
import tempfile
import os
import errno

import pytest

from flight_delay.experiments.config import code_fingerprints, file_hash
from flight_delay.experiments.inputs import gate_path
from flight_delay.experiments.results import read_json, write_json

spec = importlib.util.spec_from_file_location(
    "pace_validation",
    Path(__file__).resolve().parents[1] / "scripts/validate_phase3_pace.py",
)
staging = importlib.util.module_from_spec(spec)
spec.loader.exec_module(staging)


@pytest.mark.parametrize("passed", [True, False])
def test_exact_staging_preserves_audit_outcome(
    phase3_fixture, tmp_path, monkeypatch, passed
):
    cfg, _ = phase3_fixture
    archives = tmp_path / "archives"
    archives.mkdir()
    payload = cfg.root / "input-proof.bin"
    payload.write_bytes(b"exact original bytes\x00\xff")
    for kind in ["source", "inputs"]:
        archive = archives / f"{kind}.tar"
        with tarfile.open(archive, "w") as stream:
            if kind == "inputs":
                stream.add(payload, arcname=payload.name)
        write_json(
            archives / f"{kind}_archive.json", {"archive_sha256": file_hash(archive)}
        )

    def extract(command, **kwargs):
        with tarfile.open(command[2]) as stream:
            stream.extractall(command[4], filter="data")

    visited = []

    def audit(local_cfg):
        visited.append(local_cfg.root)
        assert local_cfg.root != cfg.root
        assert (local_cfg.root / payload.name).read_bytes() == payload.read_bytes()
        assert local_cfg.config_hash == cfg.config_hash
        return dict(
            passed=passed,
            config_hash=cfg.config_hash,
            code=code_fingerprints(cfg.root),
            checks={},
            errors=[] if passed else ["independent audit failed"],
        )

    reports = []
    monkeypatch.setattr(staging.subprocess, "run", extract)
    monkeypatch.setattr(staging, "validate_inputs", audit)
    monkeypatch.setattr(staging, "report", lambda cfg, **kw: reports.append(kw))
    result = staging.validate_staged(cfg, archives, temporary_parent=tmp_path)
    assert result["passed"] is passed
    assert read_json(gate_path(cfg))["passed"] is passed
    assert reports[0]["status"] == ("inputs_validated" if passed else "incomplete")
    assert not visited[0].exists()
    assert payload.read_bytes() == b"exact original bytes\x00\xff"
    (archives / "inputs.tar").write_bytes(b"corrupted transfer")
    with pytest.raises(ValueError, match="transfer checksum"):
        staging.validate_staged(cfg, archives, temporary_parent=tmp_path)
    assert len(visited) == 1


def test_staged_batches_share_checkpoints_and_global_holdout_lock(
    phase3_fixture, tmp_path, monkeypatch
):
    from flight_delay.experiments import inputs
    from flight_delay.experiments.runner import context
    from flight_delay.experiments.results import ResultStore

    cfg, gate = phase3_fixture
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    staged_roots = []
    original_link = os.link

    def same_filesystem_only(source, target, **kwargs):
        for root in staged_roots:
            if Path(source).is_relative_to(root) and not Path(target).is_relative_to(
                root
            ):
                raise OSError(
                    errno.EXDEV, "Simulated node-local/shared filesystem boundary"
                )
        return original_link(source, target, **kwargs)

    monkeypatch.setattr(os, "link", same_filesystem_only)
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location(
        "pace_staged", scripts / "run_phase3_pace_staged.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Some Windows hosts lack symlink privileges; production Linux executes this test.
    probe = tmp_path / "symlink_probe"
    try:
        probe.symlink_to(scripts, target_is_directory=True)
    except OSError:
        pytest.skip("Host does not permit directory symlinks")
    probe.unlink()

    @contextmanager
    def staged(original, archive_root):
        with tempfile.TemporaryDirectory(dir=tmp_path) as name:
            local_cfg = replace(original, root=Path(name))
            staged_roots.append(local_cfg.root)
            for key in ["processed_dir", "features_dir"]:
                shutil.copytree(original.path(key), local_cfg.path(key))
            target = local_cfg.root / "scripts/run_phase3_batch.py"
            target.parent.mkdir()
            shutil.copyfile(scripts / target.name, target)
            yield local_cfg

    def verified(local_cfg, stage):
        assert stage == "development" and local_cfg.root != cfg.root
        assert local_cfg.path("output_dir") == cfg.path("output_dir")
        assert local_cfg.path("features_dir") != cfg.path("features_dir")
        return gate

    monkeypatch.setattr(module, "staged_repository", staged)
    monkeypatch.setattr(inputs, "verified_gate", verified)
    plan = module.run(cfg, "prepare", "B")
    store = ResultStore(cfg.path("output_dir"), context(cfg, gate), True)
    assert plan["run_id"] == store.run_id
    assert len(list((store.root / "development/units").glob("*.json"))) == 7
    worker = module.run(cfg, "worker", "B", 0, "staged-test")
    assert worker.root.is_relative_to(cfg.path("output_dir"))
    assert len(list((worker.root / "development/units").glob("*.json"))) == 3
    (cfg.path("output_dir") / "PRIMARY_HOLDOUT_LOCK.json").write_text("{}")
    with pytest.raises(ValueError, match="Holdout execution has begun"):
        module.run(cfg, "worker", "B", 0, "locked-test")
