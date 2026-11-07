"""Reproducible command-line entry points."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
from .config import Config


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 1 BTS preprocessing")
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ["discover", "preprocess"]:
        sub = commands.add_parser(command)
        sub.add_argument("--raw-dir", type=Path, required=True)
        sub.add_argument("--year", type=int, default=2025)
        if command == "discover":
            sub.add_argument("--manifest", type=Path, default=Path("data/processed/audit/input_manifest.json"))
        else:
            sub.add_argument("--out-dir", type=Path, default=Path("data/processed"))
            sub.add_argument("--config", type=Path, default=Path("configs/preprocessing.yaml"))
    sub = commands.add_parser("audit")
    sub.add_argument("--processed-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "discover":
            from .discovery import discover
            entries = discover(args.raw_dir, args.year, args.manifest)
            print(json.dumps({e["filename"]: e["row_count"] for e in entries}, indent=2))
        elif args.command == "preprocess":
            from .pipeline import run
            run(args.raw_dir, args.out_dir, args.year, Config.load(args.config))
        else:
            from .audit import audit
            result = audit(args.processed_dir)
            print(json.dumps(result["acceptance_checks"], indent=2))
            return 0 if result["acceptance_passed"] else 1
    except (ValueError, OSError, pl_error_types()) as exc:
        print(json.dumps({"error": type(exc).__name__, "message": str(exc)}), file=sys.stderr)
        return 1
    return 0


def pl_error_types():
    import polars as pl
    return pl.exceptions.PolarsError


if __name__ == "__main__":
    sys.exit(main())
