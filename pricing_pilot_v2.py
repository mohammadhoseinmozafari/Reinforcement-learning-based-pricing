"""Enumerate or run one of the nine predeclared v2 anchor pilots."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from env.pricing_contracts import AgentArchitecture
from universal_pricing_v2.pilot import load_v2_pilot_config
from universal_pricing_v2.protocol import V2ExperimentRunId
from universal_pricing_v2.trainer import HierarchicalPricingTrainer


DEFAULT_PILOT = (
    Path(__file__).resolve().parent
    / "config"
    / "universal_pricing_v2"
    / "pilot.yaml"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-config", type=Path, default=DEFAULT_PILOT)
    parser.add_argument("--enumerate", action="store_true")
    parser.add_argument(
        "--agent", choices=[item.value for item in AgentArchitecture]
    )
    parser.add_argument("--anchor-index", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    if args.enumerate:
        if args.agent is not None or args.anchor_index is not None:
            parser.error("--enumerate rejects pilot selectors")
    elif args.agent is None or args.anchor_index is None:
        parser.error("pilot run requires --agent and --anchor-index")
    if args.anchor_index is not None and not 0 <= args.anchor_index <= 2:
        parser.error("--anchor-index must be from 0 through 2")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    pilot = load_v2_pilot_config(args.pilot_config)
    protocol = pilot.resolved_protocol()
    if args.enumerate:
        for coordinate in pilot.coordinates():
            print(V2ExperimentRunId.from_coordinate(coordinate))
        return
    coordinate = next(
        item
        for item in pilot.coordinates()
        if item.agent_architecture.value == args.agent
        and item.distribution_combination
        == pilot.anchors[args.anchor_index].distribution_combination
    )
    manifest = HierarchicalPricingTrainer(
        protocol,
        coordinate,
        device=args.device,
        resume=args.resume,
        verbose=not args.quiet,
    ).train()
    print(f"Pilot finished with status: {manifest.status.value}")


if __name__ == "__main__":
    main()
