"""Reproducible command-line entry points."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
from .config import Config


def main() -> int:
    parser = argparse.ArgumentParser(description="BTS Phase I preprocessing and Phase II causal features")
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
    sub = commands.add_parser("features")
    sub.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    sub.add_argument("--out-dir", type=Path, default=Path("data/features"))
    sub.add_argument("--config", type=Path, default=Path("configs/phase2.yaml"))
    sub.add_argument("--reuse-cache", type=Path)
    sub = commands.add_parser("features-audit")
    sub.add_argument("--features-dir", type=Path, default=Path("data/features"))
    sub = commands.add_parser("feature-fold")
    sub.add_argument("--fold", required=True)
    sub.add_argument("--feature-set", required=True)
    sub.add_argument("--encoding", default="onehot")
    sub.add_argument("--features-dir", type=Path, default=Path("data/features"))
    sub.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    sub.add_argument("--no-scale", action="store_true")
    sub.add_argument("--summary", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "discover":
            from .discovery import discover
            entries = discover(args.raw_dir, args.year, args.manifest)
            print(json.dumps({e["filename"]: e["row_count"] for e in entries}, indent=2))
        elif args.command == "preprocess":
            from .pipeline import run
            run(args.raw_dir, args.out_dir, args.year, Config.load(args.config))
        elif args.command == "audit":
            from .audit import audit
            result = audit(args.processed_dir)
            print(json.dumps(result["acceptance_checks"], indent=2))
            return 0 if result["acceptance_passed"] else 1
        elif args.command == "features":
            from .features.builder import build
            from .features.config import FeatureConfig
            result = build(args.processed_dir, args.out_dir, FeatureConfig.load(args.config), reuse_cache=args.reuse_cache)
            print(json.dumps(result["counts"], indent=2))
        elif args.command == "features-audit":
            from .features.validation import audit_features
            result = audit_features(args.features_dir)
            print(json.dumps(result["acceptance_checks"], indent=2))
            return 0 if result["acceptance_passed"] else 1
        else:
            from .features.transforms import build_fold, matrix_summary
            from .pipeline import write_json
            result = matrix_summary(build_fold(args.fold, args.feature_set, args.encoding,
                features_dir=args.features_dir, processed_dir=args.processed_dir, scale=not args.no_scale), args.feature_set)
            if args.summary:
                write_json(args.summary, result)
            print(json.dumps(result, indent=2))
    except (ValueError, OSError, pl_error_types()) as exc:
        print(json.dumps({"error": type(exc).__name__, "message": str(exc)}), file=sys.stderr)
        return 1
    return 0


def pl_error_types():
    import polars as pl
    return pl.exceptions.PolarsError


if __name__ == "__main__":
    sys.exit(main())
