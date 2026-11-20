import pytest

from flight_delay.experiments.results import ResultStore, exclusive


def test_phase4_resume_rejects_corrupt_artifacts_and_concurrent_attempt(tmp_path):
    store = ResultStore(tmp_path, {"phase": "phase4", "seed": 6740}, resume=True)
    assert store.load("fit") is None
    with exclusive(store.root):
        with pytest.raises(ValueError, match="writer"):
            with exclusive(store.root):
                pass
        artifact = store.root / "weights.bin"
        artifact.write_bytes(b"synthetic state")
        store.save("fit", {"status": "complete"}, (artifact,))
    assert store.load("fit")["status"] == "complete"
    artifact.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="corrupted"):
        store.load("fit")

