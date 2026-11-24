"""Run or resume the source-input MLP study."""

import argparse

from flight_delay.experiments.phase4_inputs import Inputs
from flight_delay.experiments.source_mlp import ARMS, REGIMES, VIEWS, context, final_fit, publish, tune


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["all", "tune", "fit", "publish"])
    parser.add_argument("--view", choices=VIEWS)
    parser.add_argument("--regime", choices=REGIMES)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    inputs = Inputs("configs/phase4.yaml")
    views = [args.view] if args.view else VIEWS
    if args.stage in ("all", "tune"):
        for view in views:
            selected = tune(inputs, view, args.device)
            print(f"{view} locked candidate {selected['candidate_id']}, epochs={selected['epochs']}", flush=True)
    if args.stage in ("all", "fit"):
        for view in views:
            provenance = context(inputs, view, "all", args.device)
            for regime in ([args.regime] if args.regime else REGIMES):
                for arm in ARMS[view]:
                    for seed in inputs.mlp["final_seeds"]:
                        final_fit(inputs, view, regime, arm, seed, args.device, provenance)
    if args.stage in ("all", "publish"):
        print(publish(inputs)["status"])


if __name__ == "__main__":
    main()
