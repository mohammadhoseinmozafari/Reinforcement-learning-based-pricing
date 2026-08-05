"""Shared fixed-step trainer and exact resume snapshots for universal pricing."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math
import os
from pathlib import Path
import platform
import signal
import subprocess
import tempfile
import time
from typing import Any, Mapping

import numpy as np
import torch

from env.pricing_contracts import PricingAction, PricingActionCodec, PricingRegime
from env.universal_pricing_factory import UniversalPricingEnvironmentFactory
from evaluation.universal_pricing_evaluator import UniversalPricingEvaluator
from models.universal_pricing_agents import (
    UniversalPricingAgentComponents,
    UniversalPricingAgentFactory,
)
from models.universal_pricing_replay import UniversalPricingTransition
from models.universal_pricing_sequence_replay import (
    UniversalPricingEpisodeBuilder,
)
from train.universal_pricing_protocol import (
    ArtifactLayout,
    ExperimentCoordinate,
    ExperimentRunId,
    ExperimentRunManifest,
    ManifestRepository,
    RunStatus,
    TrainingBudgetConfig,
    UniversalPricingProtocolConfig,
    stable_configuration_hash,
)


@dataclass(frozen=True)
class UniversalPricingTrainingSnapshot:
    """Complete episode-boundary state required for exact continuation."""

    environment_steps: int
    next_episode_index: int
    agent_checkpoint: bytes
    replay_state: dict[str, Any]
    warmup_rng_state: dict[str, Any]
    metric_records: tuple[dict[str, Any], ...]
    policy_wall_seconds: float
    environment_wall_seconds: float
    update_wall_seconds: float


class UniversalPricingTrainingSnapshotRepository:
    """Atomically persist a runner snapshot containing agent and replay state."""

    def write(
        self,
        path: Path,
        *,
        agent: Any,
        replay_buffer: Any,
        environment_steps: int,
        next_episode_index: int,
        warmup_rng_state: dict[str, Any],
        metric_records: list[dict[str, Any]],
        policy_wall_seconds: float,
        environment_wall_seconds: float,
        update_wall_seconds: float,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        agent_path = path.with_name(f".{path.name}.agent")
        agent.save(agent_path)
        agent_bytes = agent_path.read_bytes()
        agent_path.unlink()
        payload = UniversalPricingTrainingSnapshot(
            environment_steps=environment_steps,
            next_episode_index=next_episode_index,
            agent_checkpoint=agent_bytes,
            replay_state=replay_buffer.state_dict(),
            warmup_rng_state=warmup_rng_state,
            metric_records=tuple(metric_records),
            policy_wall_seconds=policy_wall_seconds,
            environment_wall_seconds=environment_wall_seconds,
            update_wall_seconds=update_wall_seconds,
        )
        temporary = path.with_name(f".{path.name}.temporary")
        try:
            torch.save(payload, temporary)
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()

    def restore(
        self,
        path: Path,
        *,
        agent: Any,
        replay_buffer: Any,
    ) -> UniversalPricingTrainingSnapshot:
        snapshot = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(snapshot, UniversalPricingTrainingSnapshot):
            raise ValueError("Malformed universal training snapshot")
        temporary_agent = path.with_name(f".{path.name}.restore-agent")
        try:
            temporary_agent.write_bytes(snapshot.agent_checkpoint)
            agent.load(temporary_agent)
        finally:
            if temporary_agent.exists():
                temporary_agent.unlink()
        replay_buffer.load_state_dict(snapshot.replay_state)
        return snapshot


class UniversalPricingTrainer:
    """Train every universal architecture through one fixed-step workflow."""

    def __init__(
        self,
        protocol: UniversalPricingProtocolConfig,
        coordinate: ExperimentCoordinate,
        *,
        device: str = "cpu",
        resume: bool = False,
        budget: TrainingBudgetConfig | None = None,
        run_validation: bool = True,
    ) -> None:
        self.protocol = protocol
        self.coordinate = coordinate
        self.device = device
        self.resume = resume
        self.budget = budget or protocol.training_budget
        self.run_validation = run_validation
        self.layout = ArtifactLayout(protocol.artifact_root)
        self.run_directory = self.layout.run_directory(coordinate)
        self.manifest_path = self.layout.manifest_path(coordinate)
        self.metrics_path = self.layout.metrics_path(coordinate)
        self.latest_snapshot_path = (
            self.layout.checkpoint_directory(coordinate)
            / "latest_training_state.pt"
        )
        self.final_checkpoint_path = (
            self.layout.checkpoint_directory(coordinate) / "final.pt"
        )
        self.seed_bundle = protocol.run_seed_bundle(
            coordinate.training_seed_index
        )
        self.components: UniversalPricingAgentComponents = (
            UniversalPricingAgentFactory.create(
                protocol.agent_profiles[coordinate.agent_architecture],
                self.seed_bundle,
                device=device,
            )
        )
        self.environment = UniversalPricingEnvironmentFactory(
            protocol
        ).create_environment(coordinate)
        self._warmup_rng = np.random.default_rng(
            np.random.SeedSequence(
                [self.seed_bundle.exploration_seed, 0]
            )
        )
        self.environment_steps = 0
        self.next_episode_index = 0
        self.metric_records: list[dict[str, Any]] = []
        self.stop_requested = False
        self.environment_wall_seconds = 0.0
        self.policy_wall_seconds = 0.0
        self.update_wall_seconds = 0.0
        self._manifest_repository = ManifestRepository()
        self._snapshot_repository = (
            UniversalPricingTrainingSnapshotRepository()
        )

    @staticmethod
    def _git_commit() -> str:
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return "unknown-source"

    def _hardware_metadata(self) -> dict[str, Any]:
        return {
            "device": str(self.device),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "parameter_count": int(
                self.components.agent.policy_diagnostics().get(
                    "parameter_count", 0
                )
            ),
        }

    def _new_manifest(self) -> ExperimentRunManifest:
        resolved = self.protocol.to_dict()
        return ExperimentRunManifest(
            protocol_version=self.protocol.protocol_version,
            run_id=ExperimentRunId.from_coordinate(self.coordinate),
            coordinate=self.coordinate,
            run_seed_bundle=self.seed_bundle,
            resolved_protocol=resolved,
            configuration_hash=stable_configuration_hash(resolved),
            git_commit=self._git_commit(),
            hardware_metadata=self._hardware_metadata(),
            status=RunStatus.REGISTERED,
            artifact_references={},
        )

    def _prepare(self) -> ExperimentRunManifest:
        if self.resume:
            manifest = self._manifest_repository.read(self.manifest_path)
            if manifest.coordinate != self.coordinate:
                raise ValueError("Resume manifest coordinate mismatch")
            if manifest.status is RunStatus.COMPLETED:
                raise ValueError("Completed runs cannot be resumed")
            if not self.latest_snapshot_path.is_file():
                raise ValueError("Resume snapshot does not exist")
            snapshot = self._snapshot_repository.restore(
                self.latest_snapshot_path,
                agent=self.components.agent,
                replay_buffer=self.components.replay_buffer,
            )
            self.environment_steps = snapshot.environment_steps
            self.next_episode_index = snapshot.next_episode_index
            self.metric_records = list(snapshot.metric_records)
            self.policy_wall_seconds = snapshot.policy_wall_seconds
            self.environment_wall_seconds = snapshot.environment_wall_seconds
            self.update_wall_seconds = snapshot.update_wall_seconds
            self._warmup_rng.bit_generator.state = dict(
                snapshot.warmup_rng_state
            )
            self._write_metrics()
            manifest = replace(manifest, status=RunStatus.RUNNING)
            self._manifest_repository.write(self.manifest_path, manifest)
            return manifest
        if self.run_directory.exists():
            raise ValueError(
                "Run directory already exists; use --resume for continuation"
            )
        manifest = self._new_manifest()
        self._manifest_repository.write(self.manifest_path, manifest)
        manifest = replace(manifest, status=RunStatus.RUNNING)
        self._manifest_repository.write(self.manifest_path, manifest)
        return manifest

    def _random_action(self) -> PricingAction:
        controls = self._warmup_rng.uniform(-1.0, 1.0, size=3)
        return PricingAction(
            PricingRegime(int(self._warmup_rng.integers(0, 2))),
            *controls,
        )

    def _ready_to_update(self) -> bool:
        if self.environment_steps < self.budget.warmup_steps:
            return False
        if self.components.is_recurrent:
            minimum_episodes = math.ceil(
                self.budget.warmup_steps / self.environment.episode_length
            )
            return len(self.components.replay_buffer) >= minimum_episodes
        return (
            len(self.components.replay_buffer)
            >= self.components.batch_size
        )

    def _write_metrics(self) -> None:
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.metrics_path.with_name(
            f".{self.metrics_path.name}.temporary"
        )
        with temporary.open("w", encoding="utf-8") as stream:
            for record in self.metric_records:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, self.metrics_path)

    def _save_snapshot(self) -> None:
        self.components.agent.reset_recurrent_state()
        self._snapshot_repository.write(
            self.latest_snapshot_path,
            agent=self.components.agent,
            replay_buffer=self.components.replay_buffer,
            environment_steps=self.environment_steps,
            next_episode_index=self.next_episode_index,
            warmup_rng_state=self._warmup_rng.bit_generator.state,
            metric_records=self.metric_records,
            policy_wall_seconds=self.policy_wall_seconds,
            environment_wall_seconds=self.environment_wall_seconds,
            update_wall_seconds=self.update_wall_seconds,
        )

    def _periodic_checkpoint(self) -> Path:
        checkpoint = self.layout.checkpoint_directory(
            self.coordinate
        ) / f"step-{self.environment_steps:09d}.pt"
        self.components.agent.save(checkpoint)
        self._save_snapshot()
        return checkpoint

    def _run_validation(self, checkpoint: Path) -> None:
        _, summary = UniversalPricingEvaluator(
            self.protocol, self.coordinate, device=self.device
        ).evaluate_checkpoint(
            checkpoint,
            self.protocol.seed_manifest.validation_environment_seeds,
            suite="validation",
        )
        self.metric_records.append(
            {
                "phase": "validation",
                "environment_steps": self.environment_steps,
                **summary,
            }
        )
        self._write_metrics()
        # Periodic validation is part of the resumable research record. Persist
        # it after evaluation so a resume from this step cannot silently lose it.
        self._save_snapshot()

    def request_stop(self) -> None:
        self.stop_requested = True

    def train(self) -> ExperimentRunManifest:
        manifest = self._prepare()
        training_started = time.perf_counter()
        previous_handlers: dict[int, Any] = {}

        def request_stop(_signum, _frame):
            self.stop_requested = True

        for signal_name in ("SIGINT", "SIGTERM"):
            signal_value = getattr(signal, signal_name, None)
            if signal_value is not None:
                previous_handlers[signal_value] = signal.getsignal(signal_value)
                signal.signal(signal_value, request_stop)
        try:
            while self.environment_steps < self.budget.environment_steps:
                episode_index = self.next_episode_index
                observation, reset_info = self.environment.reset(
                    options={"episode_index": episode_index}
                )
                self.components.agent.reset_recurrent_state()
                builder = (
                    UniversalPricingEpisodeBuilder(
                        episode_index=episode_index,
                        consumer_seed=reset_info["consumer_seed"],
                        opponent_seed=reset_info["opponent_seed"],
                        opponent_family=reset_info["opponent_family"],
                        opponent_policy_name=reset_info[
                            "opponent_policy_name"
                        ],
                    )
                    if self.components.is_recurrent
                    else None
                )
                rewards: list[float] = []
                profits: list[float] = []
                opponent_profits: list[float] = []
                update_metrics: list[Mapping[str, float]] = []
                episode_started = time.perf_counter()
                while self.environment_steps < self.budget.environment_steps:
                    policy_started = time.perf_counter()
                    action = (
                        self._random_action()
                        if self.environment_steps < self.budget.warmup_steps
                        else self.components.agent.select_action(observation)
                    )
                    self.policy_wall_seconds += (
                        time.perf_counter() - policy_started
                    )
                    environment_started = time.perf_counter()
                    (
                        next_observation,
                        reward,
                        terminated,
                        truncated,
                        info,
                    ) = self.environment.step(PricingActionCodec.to_gym(action))
                    self.environment_wall_seconds += (
                        time.perf_counter() - environment_started
                    )
                    transition = UniversalPricingTransition.from_environment_step(
                        observation=observation,
                        reward=reward,
                        next_observation=next_observation,
                        terminated=terminated,
                        truncated=truncated,
                        info=info,
                    )
                    self.components.agent.observe_transition(transition)
                    if builder is None:
                        self.components.replay_buffer.push(transition)
                    else:
                        builder.append(transition)
                    self.environment_steps += 1
                    rewards.append(float(reward))
                    profits.append(float(info["raw_agent_profit"]))
                    opponent_profits.append(float(info["raw_opponent_profit"]))
                    if truncated and builder is not None:
                        self.components.replay_buffer.push_episode(
                            builder.build()
                        )
                    if self._ready_to_update():
                        update_started = time.perf_counter()
                        for _ in range(self.budget.updates_per_step):
                            batch = self.components.replay_buffer.sample(
                                self.components.batch_size
                            )
                            update_metrics.append(
                                self.components.agent.update(batch)
                            )
                        self.update_wall_seconds += (
                            time.perf_counter() - update_started
                        )
                    observation = next_observation
                    if terminated or truncated:
                        break
                self.next_episode_index += 1
                record: dict[str, Any] = {
                    "phase": "training",
                    "episode_index": episode_index,
                    "environment_steps": self.environment_steps,
                    "opponent_family": reset_info["opponent_family"],
                    "opponent_policy_name": reset_info["opponent_policy_name"],
                    "normalized_reward_total": float(np.sum(rewards)),
                    "raw_agent_profit_total": float(np.sum(profits)),
                    "raw_opponent_profit_total": float(
                        np.sum(opponent_profits)
                    ),
                    "profit_advantage_total": float(
                        np.sum(profits) - np.sum(opponent_profits)
                    ),
                    "episode_wall_seconds": time.perf_counter()
                    - episode_started,
                    "cumulative_policy_wall_seconds": (
                        self.policy_wall_seconds
                    ),
                    "cumulative_environment_wall_seconds": (
                        self.environment_wall_seconds
                    ),
                    "cumulative_update_wall_seconds": (
                        self.update_wall_seconds
                    ),
                }
                if update_metrics:
                    for name in update_metrics[0]:
                        record[f"mean_{name}"] = float(
                            np.mean([metrics[name] for metrics in update_metrics])
                        )
                self.metric_records.append(record)
                self._write_metrics()
                checkpoint_due = (
                    self.environment_steps
                    % self.budget.checkpoint_interval_steps
                    == 0
                )
                evaluation_due = (
                    self.run_validation
                    and self.environment_steps
                    % self.budget.evaluation_interval_steps
                    == 0
                )
                checkpoint = None
                if checkpoint_due:
                    checkpoint = self._periodic_checkpoint()
                if evaluation_due:
                    if checkpoint is None:
                        checkpoint = self.layout.checkpoint_directory(
                            self.coordinate
                        ) / (
                            f"evaluation-step-{self.environment_steps:09d}.pt"
                        )
                        self.components.agent.save(checkpoint)
                    self._run_validation(checkpoint)
                if self.stop_requested:
                    self._save_snapshot()
                    manifest = replace(
                        manifest, status=RunStatus.INTERRUPTED
                    )
                    self._manifest_repository.write(
                        self.manifest_path, manifest
                    )
                    return manifest

            self.components.agent.reset_recurrent_state()
            self.components.agent.save(self.final_checkpoint_path)
            peak_memory = (
                int(torch.cuda.max_memory_allocated())
                if self.device.startswith("cuda") and torch.cuda.is_available()
                else 0
            )
            self.metric_records.append(
                {
                    "phase": "efficiency",
                    "environment_steps": self.environment_steps,
                    "total_wall_seconds": (
                        time.perf_counter() - training_started
                    ),
                    "policy_wall_seconds": self.policy_wall_seconds,
                    "environment_wall_seconds": (
                        self.environment_wall_seconds
                    ),
                    "update_wall_seconds": self.update_wall_seconds,
                    "peak_cuda_memory_bytes": peak_memory,
                    "final_checkpoint_bytes": (
                        self.final_checkpoint_path.stat().st_size
                    ),
                    "parameter_count": self.components.agent.policy_diagnostics().get(
                        "parameter_count", 0.0
                    ),
                }
            )
            self._write_metrics()
            self._save_snapshot()
            references = {
                "final_checkpoint": str(self.final_checkpoint_path),
                "latest_training_snapshot": str(
                    self.latest_snapshot_path
                ),
                "metrics": str(self.metrics_path),
            }
            manifest = replace(
                manifest,
                status=RunStatus.COMPLETED,
                artifact_references=references,
            )
            self._manifest_repository.write(self.manifest_path, manifest)
            return manifest
        except BaseException:
            failed = replace(manifest, status=RunStatus.FAILED)
            self._manifest_repository.write(self.manifest_path, failed)
            raise
        finally:
            for signal_value, handler in previous_handlers.items():
                signal.signal(signal_value, handler)
            self.environment.close()
