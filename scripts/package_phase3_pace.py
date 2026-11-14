"""Package private PACE inputs or source, preserving and checking original bytes."""

from pathlib import Path
import argparse
import hashlib
import io
import subprocess
import tarfile
import time

from flight_delay.experiments.config import Phase3Config, code_fingerprints, file_hash
from flight_delay.experiments.inputs import gate_path
from flight_delay.experiments.results import read_json, write_json


def package(cfg, kind):
    gate = read_json(gate_path(cfg))
    if not gate["passed"] or gate["code"] != code_fingerprints(cfg.root):
        raise ValueError("Package only the current independently validated workspace")
    root = cfg.root.resolve()
    output = root / "data/model_outputs/phase3/pace_transfer"
    output.mkdir(parents=True, exist_ok=True)
    if kind == "inputs":
        paths = [
            path
            for path in (root / "data/phase3_inputs").rglob("*")
            if path.is_file() and path.suffix != ".copying"
        ]
    else:
        names = (
            subprocess.check_output(["git", "ls-files", "-z"], cwd=root)
            .decode("utf-8")
            .split("\0")
        )
        paths = [root / name for name in names if name]
        paths += [
            path for path in (root / "reports/phase3").rglob("*") if path.is_file()
        ]
    paths = sorted(set(paths))
    expected = dict(gate["input_files"], **gate["code"])
    target = output / f"{kind}.tar"
    started, verified, retries, total = time.monotonic(), 0, 0, 0
    with tarfile.open(target, "w", dereference=True) as archive:
        for index, path in enumerate(paths):
            if not path.resolve().is_relative_to(root):
                raise ValueError("Archive source escapes the workspace")
            relative = path.relative_to(root).as_posix()
            # Read before writing a tar header, so a transient Windows sharing
            # violation cannot leave a half-written member in the archive.
            for attempt in range(6):
                try:
                    payload = path.read_bytes()
                    break
                except PermissionError:
                    if attempt == 5:
                        raise
                    retries += 1
                    time.sleep(0.5)
            if relative in expected:
                if hashlib.sha256(payload).hexdigest() != expected[relative]:
                    raise ValueError(
                        f"Validated input/source bytes changed: {relative}"
                    )
                verified += 1
            info = archive.gettarinfo(str(path), arcname=relative)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
            total += len(payload)
            if index % 5000 == 0:
                print(f"Packaged {index + 1:,}/{len(paths):,} {kind} files", flush=True)
    result = dict(
        kind=kind,
        files=len(paths),
        checked_original_hashes=verified,
        input_bytes=total,
        archive_bytes=target.stat().st_size,
        archive_sha256=file_hash(target),
        transient_read_retries=retries,
        seconds=time.monotonic() - started,
    )
    write_json(output / f"{kind}_archive.json", result)
    print(result, flush=True)
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=["inputs", "source"])
    parser.add_argument("--config", type=Path, default=Path("configs/phase3.yaml"))
    args = parser.parse_args()
    package(Phase3Config.load(args.config), args.kind)
