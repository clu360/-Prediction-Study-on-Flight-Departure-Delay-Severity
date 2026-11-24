"""Report saved Phase IV attempts and verified checkpoints without submitting work."""

import argparse
from pathlib import Path

from flight_delay.experiments.config import digest, file_hash
from flight_delay.experiments.results import read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/model_outputs/phase4", type=Path)
    args = parser.parse_args()
    root = args.root
    for hint in ("upstream_development.json", "upstream_all.json", "mlp_selection.json"):
        path = root / hint
        if path.exists():
            record = read_json(path)
            print(hint, record.get("input_sha256", record.get("sha256")), record.get("counts", ""))
    for path in sorted(root.rglob("status.json")):
        status = read_json(path)
        print(path.parent.name, status.get("arm", status.get("candidate")), status["state"],
              file_hash(path))
    for path in sorted(root.rglob("month-*.json")):
        record = read_json(path)
        if "record_hash" not in record:
            continue
        signature = record.pop("record_hash")
        state = "record valid" if digest(record) == signature else "record checksum mismatch"
        print(record.get("run_signature"), record.get("month"), state, signature)


if __name__ == "__main__":
    main()

