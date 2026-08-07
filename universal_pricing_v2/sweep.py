"""Deterministic registration and launch commands for the 810-run v2 sweep."""

from __future__ import annotations

import json
import os
import shlex
import tempfile
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any, Iterable, Mapping

from universal_pricing_v2.protocol import (
    UniversalPricingV2ProtocolConfig,
    V2ArtifactLayout,
    V2ExperimentCoordinate,
    V2ExperimentMatrix,
    V2ExperimentRunId,
    V2ManifestRepository,
)


class ProductionSeedWave(IntEnum):
    """Two predeclared five-seed production waves."""

    FIRST = 1
    SECOND = 2

    @property
    def seed_indices(self) -> range:
        return range(0, 5) if self is self.FIRST else range(5, 10)


@dataclass(frozen=True)
class V2SweepJob:
    """One immutable production command and its current artifact status."""

    run_id: str
    wave: int
    coordinate: Mapping[str, Any]
    run_directory: str
    command: tuple[str, ...]
    status: str
    launchable: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "wave": self.wave,
            "coordinate": dict(self.coordinate),
            "run_directory": self.run_directory,
            "command": list(self.command),
            "status": self.status,
            "launchable": self.launchable,
        }

    @property
    def shell_command(self) -> str:
        return shlex.join(self.command)


class V2ProductionSweep:
    """Enumerate production work without starting processes implicitly."""

    def __init__(
        self,
        protocol: UniversalPricingV2ProtocolConfig,
        *,
        protocol_path: str | Path,
        python_executable: str = "python",
        device: str = "cuda",
    ) -> None:
        self.protocol = protocol
        self.protocol_path = Path(protocol_path)
        self.python_executable = python_executable
        self.device = device
        self.layout = V2ArtifactLayout(protocol.artifact_root)
        self.manifests = V2ManifestRepository()

    def _status(self, coordinate: V2ExperimentCoordinate) -> str:
        path = self.layout.manifest_path(coordinate)
        if not path.is_file():
            return "unregistered"
        try:
            return self.manifests.read(path).status.value
        except (OSError, TypeError, ValueError):
            return "invalid_manifest"

    def _command(
        self,
        coordinate: V2ExperimentCoordinate,
        status: str,
    ) -> tuple[str, ...]:
        combination = coordinate.distribution_combination
        command = (
            self.python_executable,
            "pricing_train_v2.py",
            "--protocol",
            str(self.protocol_path),
            "--agent",
            coordinate.agent_architecture.value,
            "--location-distribution",
            combination.location.value,
            "--strategicness-distribution",
            combination.strategicness.value,
            "--exclusivity-distribution",
            combination.exclusivity.value,
            "--seed-index",
            str(coordinate.training_seed_index),
            "--device",
            self.device,
        )
        snapshot_exists = self.layout.latest_snapshot_path(
            coordinate
        ).is_file()
        if status in {"running", "failed", "interrupted"}:
            return (
                (*command, "--resume") if snapshot_exists else ()
            )
        if status in {"completed", "invalid_manifest"}:
            return ()
        return command

    def jobs(
        self, wave: ProductionSeedWave | int | None = None
    ) -> tuple[V2SweepJob, ...]:
        selected_wave = (
            None if wave is None else ProductionSeedWave(int(wave))
        )
        jobs: list[V2SweepJob] = []
        for coordinate in V2ExperimentMatrix(self.protocol).coordinates():
            job_wave = (
                ProductionSeedWave.FIRST
                if coordinate.training_seed_index < 5
                else ProductionSeedWave.SECOND
            )
            if selected_wave is not None and selected_wave is not job_wave:
                continue
            status = self._status(coordinate)
            command = self._command(coordinate, status)
            jobs.append(
                V2SweepJob(
                    run_id=str(V2ExperimentRunId.from_coordinate(coordinate)),
                    wave=int(job_wave),
                    coordinate=coordinate.to_dict(),
                    run_directory=str(
                        self.layout.run_directory(coordinate)
                    ),
                    command=command,
                    status=status,
                    launchable=bool(command),
                )
            )
        return tuple(jobs)


class V2SweepRegistry:
    """Atomically record a launch-ready, reviewable sweep manifest."""

    @staticmethod
    def write(path: str | Path, jobs: Iterable[V2SweepJob]) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "jobs": [job.to_dict() for job in jobs],
        }
        descriptor, temporary = tempfile.mkstemp(
            prefix=".w-", suffix=".tmp", dir=output.parent
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, sort_keys=True, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, output)
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
        return output

    @staticmethod
    def status_counts(jobs: Iterable[V2SweepJob]) -> dict[str, int]:
        result: dict[str, int] = {}
        for job in jobs:
            result[job.status] = result.get(job.status, 0) + 1
        return dict(sorted(result.items()))
