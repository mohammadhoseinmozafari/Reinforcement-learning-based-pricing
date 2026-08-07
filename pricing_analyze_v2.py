"""Collect completed v2 evaluations and generate paired research tables."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from universal_pricing_v2.analysis import (
    V2AnalysisRepository,
    V2PairedStatisticalAnalyzer,
    V2ResultCollector,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-root", type=Path, default=Path("experiments/upv2")
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("experiments/upv2_analysis"),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    results = V2ResultCollector().collect(args.artifact_root)
    comparisons = V2PairedStatisticalAnalyzer().architecture_comparisons(
        results.episodes
    )
    economic_summaries = (
        V2PairedStatisticalAnalyzer.economic_summaries(
            results.episodes
        )
    )
    paths = V2AnalysisRepository().write(
        args.output_directory,
        results=results,
        comparisons=comparisons,
        economic_summaries=economic_summaries,
    )
    print(
        f"Collected {results.run_count} completed runs and "
        f"{len(results.episodes)} final episodes."
    )
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
