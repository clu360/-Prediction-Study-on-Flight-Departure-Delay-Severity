"""Node-local audit staging preserves provenance and never promotes a failed gate."""

import importlib.util
from pathlib import Path
import tarfile

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
