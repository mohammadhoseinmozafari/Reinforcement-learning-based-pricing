"""Analysis-ready collection and paired statistics for v2 experiments."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.stats import ttest_rel

from env.pricing_contracts import AgentArchitecture
from train.universal_pricing_protocol import RunStatus
from universal_pricing_v2.protocol import V2ManifestRepository


def interquartile_mean(values: Sequence[float]) -> float:
    """Return the central 50% mean using fractional endpoint weights."""

    data = np.sort(np.asarray(values, dtype=np.float64))
    if data.size == 0 or not np.all(np.isfinite(data)):
        raise ValueError("IQM requires non-empty finite values")
    lower = 0.25 * data.size
    upper = 0.75 * data.size
    indices = np.arange(data.size, dtype=np.float64)
    weights = np.maximum(
        0.0,
        np.minimum(indices + 1.0, upper) - np.maximum(indices, lower),
    )
    return float(np.sum(data * weights) / np.sum(weights))


def paired_bootstrap_interval(
    differences: Sequence[float],
    *,
    confidence: float = 0.95,
    repetitions: int = 10_000,
    seed: int = 20260805,
) -> tuple[float, float]:
    """Return a reproducible percentile interval over paired differences."""

    values = np.asarray(differences, dtype=np.float64)
    if (
        values.ndim != 1
        or values.size < 2
        or not np.all(np.isfinite(values))
    ):
        raise ValueError("Bootstrap requires at least two finite differences")
    if not 0.0 < confidence < 1.0 or repetitions <= 0:
        raise ValueError("Invalid bootstrap configuration")
    generator = np.random.default_rng(seed)
    indices = generator.integers(
        0, values.size, size=(int(repetitions), values.size)
    )
    means = values[indices].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    return (
        float(np.quantile(means, tail)),
        float(np.quantile(means, 1.0 - tail)),
    )


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Return Holm family-wise adjusted p-values in original order."""

    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1 or not np.all(
        np.isfinite(values) & (values >= 0.0) & (values <= 1.0)
    ):
        raise ValueError("p-values must be finite and in [0, 1]")
    count = values.size
    order = np.argsort(values)
    adjusted = np.empty(count, dtype=np.float64)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (count - rank) * values[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


@dataclass(frozen=True)
class CollectedV2Results:
    episodes: tuple[Mapping[str, Any], ...]
    transfer_cells: tuple[Mapping[str, Any], ...]
    run_count: int


class V2ResultCollector:
    """Load only completed v2 manifests and their immutable evaluation data."""

    def collect(self, artifact_root: str | Path) -> CollectedV2Results:
        root = Path(artifact_root)
        episodes: list[dict[str, Any]] = []
        transfers: list[dict[str, Any]] = []
        run_count = 0
        repository = V2ManifestRepository()
        for manifest_path in sorted(root.glob("*/*/s*/manifest.json")):
            manifest = repository.read(manifest_path)
            if manifest.status is not RunStatus.COMPLETED:
                continue
            run_count += 1
            identity = {
                "run_id": manifest.run_id.value,
                "architecture": (
                    manifest.coordinate.agent_architecture.value
                ),
                "training_seed_index": (
                    manifest.coordinate.training_seed_index
                ),
                "training_distribution": (
                    manifest.coordinate.distribution_combination.identifier
                ),
            }
            episode_path = (
                manifest_path.parent / "evaluation" / "final_episodes.jsonl"
            )
            if episode_path.is_file():
                with episode_path.open("r", encoding="utf-8") as stream:
                    for line_number, line in enumerate(stream, start=1):
                        try:
                            value = json.loads(line)
                        except json.JSONDecodeError as exc:
                            raise ValueError(
                                f"Malformed {episode_path}:{line_number}"
                            ) from exc
                        episodes.append({**identity, **value})
            transfer_path = (
                manifest_path.parent
                / "evaluation"
                / "transfer_matrix.json"
            )
            if transfer_path.is_file():
                with transfer_path.open("r", encoding="utf-8") as stream:
                    values = json.load(stream)
                if not isinstance(values, list):
                    raise ValueError(
                        f"Transfer matrix must be a list: {transfer_path}"
                    )
                transfers.extend({**identity, **item} for item in values)
        return CollectedV2Results(
            episodes=tuple(episodes),
            transfer_cells=tuple(transfers),
            run_count=run_count,
        )


class V2PairedStatisticalAnalyzer:
    """Produce architecture comparisons on exactly matched episode keys."""

    METRIC = "net_agent_profit_total"

    @staticmethod
    def economic_summaries(
        rows: Iterable[Mapping[str, Any]]
    ) -> list[dict[str, Any]]:
        grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
        for row in rows:
            grouped.setdefault(
                (str(row["architecture"]), str(row["regime_mode"])), []
            ).append(row)
        summaries: list[dict[str, Any]] = []
        metrics = (
            "gross_agent_profit_total",
            "agent_bbp_operating_cost_total",
            "net_agent_profit_total",
            "net_profit_advantage_total",
            "bbp_period_fraction",
        )
        for (architecture, regime_mode), selected in sorted(
            grouped.items()
        ):
            summary: dict[str, Any] = {
                "architecture": architecture,
                "regime_mode": regime_mode,
                "episode_count": len(selected),
            }
            for metric in metrics:
                values = np.asarray(
                    [float(item[metric]) for item in selected],
                    dtype=np.float64,
                )
                if not np.all(np.isfinite(values)):
                    raise ValueError(
                        f"Non-finite economic metric: {metric}"
                    )
                summary[f"mean_{metric}"] = float(np.mean(values))
                summary[f"standard_error_{metric}"] = float(
                    np.std(values, ddof=1) / np.sqrt(values.size)
                ) if values.size > 1 else 0.0
            summaries.append(summary)
        return summaries

    @staticmethod
    def _pair_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
        return (
            row["training_distribution"],
            int(row["training_seed_index"]),
            row["regime_mode"],
            int(row["evaluation_seed"]),
            row["opponent_policy_name"],
            row["location_distribution"],
            row["strategicness_distribution"],
            row["exclusivity_distribution"],
        )

    def architecture_comparisons(
        self, rows: Iterable[Mapping[str, Any]]
    ) -> list[dict[str, Any]]:
        by_architecture: dict[str, dict[tuple[Any, ...], float]] = {}
        for row in rows:
            value = float(row[self.METRIC])
            if not np.isfinite(value):
                raise ValueError("Analysis input contains a non-finite value")
            by_architecture.setdefault(
                str(row["architecture"]), {}
            )[self._pair_key(row)] = value
        results: list[dict[str, Any]] = []
        p_values: list[float] = []
        architectures = [
            item.value
            for item in AgentArchitecture
            if item.value in by_architecture
        ]
        for left, right in combinations(architectures, 2):
            shared = sorted(
                set(by_architecture[left]) & set(by_architecture[right])
            )
            if len(shared) < 2:
                continue
            left_values = np.asarray(
                [by_architecture[left][key] for key in shared]
            )
            right_values = np.asarray(
                [by_architecture[right][key] for key in shared]
            )
            differences = left_values - right_values
            interval = paired_bootstrap_interval(differences)
            p_value = float(ttest_rel(left_values, right_values).pvalue)
            if not np.isfinite(p_value):
                p_value = 1.0
            p_values.append(p_value)
            results.append(
                {
                    "architecture_a": left,
                    "architecture_b": right,
                    "pair_count": len(shared),
                    "mean_difference": float(np.mean(differences)),
                    "median_difference": float(np.median(differences)),
                    "iqm_difference": interquartile_mean(differences),
                    "probability_a_better": float(
                        np.mean(differences > 0.0)
                        + 0.5 * np.mean(differences == 0.0)
                    ),
                    "bootstrap_95_low": interval[0],
                    "bootstrap_95_high": interval[1],
                    "paired_t_p_value": p_value,
                }
            )
        for row, adjusted in zip(results, holm_adjust(p_values)):
            row["holm_adjusted_p_value"] = adjusted
        return results


class V2AnalysisRepository:
    """Atomically write analysis tables as JSON and CSV."""

    @staticmethod
    def _atomic_text(path: Path, writer: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=".a-", suffix=".tmp", dir=path.parent
        )
        try:
            with os.fdopen(
                descriptor, "w", encoding="utf-8", newline=""
            ) as stream:
                writer(stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass

    def write(
        self,
        output_directory: str | Path,
        *,
        results: CollectedV2Results,
        comparisons: Sequence[Mapping[str, Any]],
        economic_summaries: Sequence[Mapping[str, Any]],
    ) -> tuple[Path, Path, Path, Path, Path]:
        output = Path(output_directory)
        summary_path = output / "analysis_summary.json"
        episode_path = output / "final_episode_table.csv"
        comparison_path = output / "architecture_comparisons.csv"
        transfer_path = output / "transfer_matrix_table.csv"
        economics_path = output / "economic_cost_summary.csv"
        payload = {
            "run_count": results.run_count,
            "episode_count": len(results.episodes),
            "transfer_cell_count": len(results.transfer_cells),
            "architecture_comparisons": list(comparisons),
            "economic_cost_summaries": list(economic_summaries),
        }

        def write_summary(stream: Any) -> None:
            json.dump(
                payload,
                stream,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            stream.write("\n")

        self._atomic_text(summary_path, write_summary)
        self._write_csv(episode_path, results.episodes)
        self._write_csv(comparison_path, comparisons)
        self._write_csv(transfer_path, results.transfer_cells)
        self._write_csv(economics_path, economic_summaries)
        return (
            summary_path,
            episode_path,
            comparison_path,
            transfer_path,
            economics_path,
        )

    def _write_csv(
        self,
        path: Path,
        rows: Sequence[Mapping[str, Any]],
    ) -> None:
        fields = sorted({name for row in rows for name in row})

        def writer(stream: Any) -> None:
            output = csv.DictWriter(stream, fieldnames=fields)
            output.writeheader()
            output.writerows(rows)

        self._atomic_text(path, writer)
