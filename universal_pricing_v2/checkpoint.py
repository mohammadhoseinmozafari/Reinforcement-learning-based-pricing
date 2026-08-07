"""Windows-safe atomic continuation snapshots for v2."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from universal_pricing_v2.curriculum import (
    HierarchicalCurriculumCoordinator,
    HierarchicalCurriculumState,
)


@dataclass(frozen=True)
class HierarchicalPricingTrainingSnapshot:
    """Everything required to continue at an episode boundary."""

    schema_version: int
    agent_state: Mapping[str, Any]
    replay_state: Mapping[str, Any]
    curriculum_state: Mapping[str, Any]
    next_global_episode_index: int
    warmup_rng_state: Mapping[str, Any]
    metric_records: tuple[Mapping[str, Any], ...]
    elapsed_wall_seconds: float
    timing_totals: Mapping[str, float]


class V2TrainingSnapshotRepository:
    """Use short temporary names and atomic replacement on Linux/Windows."""

    SCHEMA_VERSION = 1

    def write(
        self,
        path: str | Path,
        *,
        agent: Any,
        curriculum: HierarchicalCurriculumCoordinator,
        next_global_episode_index: int,
        warmup_rng_state: Mapping[str, Any],
        metric_records: Sequence[Mapping[str, Any]],
        elapsed_wall_seconds: float,
        timing_totals: Mapping[str, float],
    ) -> None:
        snapshot_path = Path(path)
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        payload = HierarchicalPricingTrainingSnapshot(
            schema_version=self.SCHEMA_VERSION,
            agent_state=agent.state_dict(),
            replay_state=agent.replay.state_dict(),
            curriculum_state=curriculum.state.to_dict(),
            next_global_episode_index=int(next_global_episode_index),
            warmup_rng_state=dict(warmup_rng_state),
            metric_records=tuple(dict(item) for item in metric_records),
            elapsed_wall_seconds=float(elapsed_wall_seconds),
            timing_totals={
                name: float(value) for name, value in timing_totals.items()
            },
        )
        descriptor, temporary = tempfile.mkstemp(
            prefix=".s-", suffix=".pt", dir=snapshot_path.parent
        )
        os.close(descriptor)
        try:
            torch.save(payload, temporary)
            os.replace(temporary, snapshot_path)
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass

    def restore(
        self,
        path: str | Path,
        *,
        agent: Any,
        protocol: Any,
    ) -> tuple[
        HierarchicalPricingTrainingSnapshot,
        HierarchicalCurriculumCoordinator,
    ]:
        snapshot = torch.load(
            Path(path), map_location="cpu", weights_only=False
        )
        if (
            not isinstance(snapshot, HierarchicalPricingTrainingSnapshot)
            or snapshot.schema_version != self.SCHEMA_VERSION
        ):
            raise ValueError("Malformed or incompatible v2 training snapshot")
        agent.load_state_dict(snapshot.agent_state)
        agent.replay.load_state_dict(snapshot.replay_state)
        coordinator = HierarchicalCurriculumCoordinator(
            protocol,
            state=HierarchicalCurriculumState.from_dict(
                snapshot.curriculum_state
            ),
        )
        return snapshot, coordinator
