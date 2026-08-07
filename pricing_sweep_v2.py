"""Enumerate and register the two five-seed universal-pricing v2 waves."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from universal_pricing_v2.protocol import load_universal_pricing_v2_protocol
from universal_pricing_v2.pilot import (
    V2PilotReadinessGate,
    load_v2_pilot_config,
)
from universal_pricing_v2.sweep import (
    ProductionSeedWave,
    V2ProductionSweep,
    V2SweepRegistry,
)


DEFAULT_PROTOCOL = (
    Path(__file__).resolve().parent
    / "config"
    / "protocols"
    / "universal_pricing_v2.yaml"
)
DEFAULT_PILOT = (
    Path(__file__).resolve().parent
    / "config"
    / "universal_pricing_v2"
    / "pilot.yaml"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--pilot-config", type=Path, default=DEFAULT_PILOT)
    parser.add_argument("--wave", type=int, choices=(1, 2))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--python-executable", default="python")
    parser.add_argument(
        "--register",
        type=Path,
        help="Atomically write the selected jobs as JSON",
    )
    parser.add_argument(
        "--commands",
        action="store_true",
        help="Print shell-escaped training commands instead of run IDs",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    protocol = load_universal_pricing_v2_protocol(args.protocol)
    sweep = V2ProductionSweep(
        protocol,
        protocol_path=args.protocol,
        python_executable=args.python_executable,
        device=args.device,
    )
    wave = None if args.wave is None else ProductionSeedWave(args.wave)
    jobs = sweep.jobs(wave)
    if args.commands or args.register is not None:
        report = V2PilotReadinessGate().evaluate(
            load_v2_pilot_config(args.pilot_config)
        )
        if not report.ready:
            raise SystemExit(
                "Production launch is blocked until all nine pilots pass; "
                f"completed={report.completed_run_count}/"
                f"{report.required_run_count}, failures="
                + "; ".join(report.failures[:5])
            )
    for job in jobs:
        if args.commands:
            if job.launchable:
                print(job.shell_command)
        else:
            print(job.run_id)
    if args.register is not None:
        path = V2SweepRegistry.write(args.register, jobs)
        print(f"Registered {len(jobs)} jobs in {path}")
    print(f"Status counts: {V2SweepRegistry.status_counts(jobs)}")


if __name__ == "__main__":
    main()
