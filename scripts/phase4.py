"""Run independent Phase IV gates, weather steps, experiments, and verification."""

import argparse

from flight_delay.experiments.phase4_inputs import Inputs, upstream_gate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phase4.yaml")
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit")
    audit.add_argument("scope", choices=["development", "all"])
    commands.add_parser("stations")
    for name in ("fetch", "audit-weather", "build-weather", "audit-g4-details"):
        step = commands.add_parser(name)
        step.add_argument("--version", default="v1")
    tune = commands.add_parser("tune")
    tune.add_argument("--device", default="cpu")
    final = commands.add_parser("fit")
    final.add_argument("arm", choices=["XGB-base", "XGB-weather", "MLP-A", "MLP-B", "MLP-C", "MLP-D"])
    final.add_argument("--seed", type=int, default=6740)
    final.add_argument("--device", default="cpu")
    commands.add_parser("publish")
    commands.add_parser("replay")
    args = parser.parse_args()
    inputs = Inputs(args.config)
    if args.command == "audit":
        result = upstream_gate(inputs, args.scope)
        print(result["counts"], result["input_sha256"])
    elif args.command == "stations":
        from flight_delay.experiments.results import read_json
        from flight_delay.weather.stations import airport_codes, build_map

        directory = inputs.path("raw_weather_dir") / "stations"
        path = inputs.path("station_map").with_name("g4_station_overrides.json")
        overrides = read_json(path) if path.exists() else {}
        rows = build_map(airport_codes(inputs), directory.glob("*.geojson"),
                         inputs.path("station_map"), overrides)
        print(f"Mapped {len(rows)} airports")
    elif args.command == "fetch":
        from flight_delay.weather.fetch import Archive, bounds
        from flight_delay.weather.stations import load_map

        mapping = load_map(inputs.path("station_map"))
        archive = Archive(inputs.path("raw_weather_dir"), *bounds(inputs.weather), args.version)
        archive.fetch([row["station"] for row in mapping])
    elif args.command == "audit-weather":
        from flight_delay.weather.validation import audit_archive

        print(audit_archive(inputs, args.version)["archive_digest"])
    elif args.command == "audit-g4-details":
        from flight_delay.weather.diagnostics import coverage_diagnostics

        print(coverage_diagnostics(inputs, args.version)["g4_signature"])
    elif args.command == "build-weather":
        from flight_delay.weather.features import build

        print(build(inputs, args.version)["signature"])
    elif args.command == "tune":
        from flight_delay.experiments.phase4_mlp import tune

        print(tune(inputs, args.device))
    elif args.command == "fit":
        from flight_delay.experiments.phase4_final import execute_arm

        print(execute_arm(inputs, args.arm, args.seed, args.device).signature)
    elif args.command == "replay":
        from flight_delay.experiments.phase4_replay import replay

        print(f"Verified {len(replay(inputs))} frozen prediction slices")
    elif args.command == "publish":
        from flight_delay.experiments.phase4_results import publish

        print(publish(inputs)["status"])


if __name__ == "__main__":
    main()
