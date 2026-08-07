"""Phase-aware hierarchical v2 trainer with exact episode-boundary resume."""

from __future__ import annotations

import copy
import platform
import signal
import subprocess
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from env.pricing_contracts import (
    AgentArchitecture,
    PricingAction,
    PricingActionCodec,
    PricingRegime,
)
from train.universal_pricing_protocol import (
    RunStatus,
    stable_configuration_hash,
)
from universal_pricing_v2.agents import (
    BaseHierarchicalPricingAgent,
    HierarchicalPricingAgentFactory,
)
from universal_pricing_v2.checkpoint import (
    V2TrainingSnapshotRepository,
)
from universal_pricing_v2.curriculum import (
    HierarchicalCurriculumCoordinator,
    MasteryResult,
    OracleNormalizedMasteryEvaluator,
)
from universal_pricing_v2.environment import (
    HierarchicalPricingEnvironmentFactoryV2,
)
from universal_pricing_v2.evaluation import (
    ConstantPriceOracleEvaluator,
    EvaluationRegimeMode,
    RandomPriceBaselineEvaluator,
    StrategyMasteryScenarioEvaluator,
    UniversalPricingV2Evaluator,
)
from universal_pricing_v2.logging import (
    HierarchicalPricingEpisodeMetrics,
    HierarchicalPricingMetricsAdapter,
)
from universal_pricing_v2.protocol import (
    AgentRegimeMode,
    HierarchicalTrainingPhase,
    PricingSkill,
    UniversalPricingV2ProtocolConfig,
    UniversalPricingV2RunManifest,
    V2ArtifactLayout,
    V2ExperimentCoordinate,
    V2ExperimentRunId,
    V2ManifestRepository,
)
from universal_pricing_v2.replay import (
    PricingSkillEpisode,
    PricingSkillTransition,
    StrategyEpisode,
    StrategyTransition,
)


class HierarchicalPricingTrainer:
    """Train uniform skill, BBP skill, strategy, then joint consolidation."""

    def __init__(
        self,
        protocol: UniversalPricingV2ProtocolConfig,
        coordinate: V2ExperimentCoordinate,
        *,
        device: str = "cpu",
        resume: bool = False,
        verbose: bool = True,
        enable_mastery_evaluation: bool = True,
        maximum_environment_steps: int | None = None,
        episode_length: int | None = None,
        logger: HierarchicalPricingMetricsAdapter | None = None,
    ) -> None:
        self.protocol = protocol
        self.coordinate = coordinate
        self.device = device
        self.resume = resume
        self.enable_mastery_evaluation = enable_mastery_evaluation
        self.maximum_environment_steps = maximum_environment_steps
        self.layout = V2ArtifactLayout(protocol.artifact_root)
        self.run_directory = self.layout.run_directory(coordinate)
        self.manifest_path = self.layout.manifest_path(coordinate)
        self.snapshot_path = self.layout.latest_snapshot_path(coordinate)
        self.final_checkpoint_path = (
            self.layout.final_checkpoint_path(coordinate)
        )
        self.seed_bundle = protocol.run_seed_bundle(
            coordinate.training_seed_index
        )
        self.agent: BaseHierarchicalPricingAgent = (
            HierarchicalPricingAgentFactory.create(
                protocol.agent_profiles[coordinate.agent_architecture],
                self.seed_bundle,
                device=device,
            )
        )
        self.parameter_counts = self.agent.parameter_counts()
        if self.agent.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.agent.device)
        factory_values = (
            {} if episode_length is None else {
                "episode_length": episode_length
            }
        )
        self.environment_factory = HierarchicalPricingEnvironmentFactoryV2(
            protocol, **factory_values
        )
        self.environment = self.environment_factory.create_environment(
            coordinate
        )
        self.context_factory = self.environment.context_factory
        self.curriculum = HierarchicalCurriculumCoordinator(protocol)
        self.mastery_evaluator = OracleNormalizedMasteryEvaluator(
            protocol.mastery_gate
        )
        self._baseline_cache: dict[
            tuple[str, str, str], tuple[float, float]
        ] = {}
        self._warmup_rng = np.random.default_rng(
            self.seed_bundle.exploration_seed
        )
        self.global_episode_index = 0
        self.metric_records: list[dict[str, Any]] = []
        self.stop_requested = False
        self.started_at = 0.0
        self.timing_totals = {
            "environment_seconds": 0.0,
            "inference_seconds": 0.0,
            "update_seconds": 0.0,
            "validation_seconds": 0.0,
            "checkpoint_seconds": 0.0,
        }
        self.manifest_repository = V2ManifestRepository()
        self.snapshot_repository = V2TrainingSnapshotRepository()
        self.logger = logger or HierarchicalPricingMetricsAdapter(
            self.layout.metrics_path(coordinate), verbose=verbose
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
        result = {
            "device": str(self.device),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            **self.agent.parameter_counts(),
        }
        if torch.cuda.is_available():
            result.update(
                {
                    "cuda": torch.version.cuda,
                    "gpu_name": torch.cuda.get_device_name(0),
                }
            )
        return result

    def _manifest(
        self,
        status: RunStatus,
        *,
        artifact_references: Mapping[str, str] | None = None,
    ) -> UniversalPricingV2RunManifest:
        resolved = self.protocol.to_dict()
        return UniversalPricingV2RunManifest(
            protocol_version=self.protocol.protocol_version,
            run_id=V2ExperimentRunId.from_coordinate(self.coordinate),
            coordinate=self.coordinate,
            run_seed_bundle=self.seed_bundle,
            resolved_protocol=resolved,
            configuration_hash=stable_configuration_hash(resolved),
            git_commit=self._git_commit(),
            hardware_metadata=self._hardware_metadata(),
            status=status,
            curriculum_state=self.curriculum.state.to_dict(),
            stage_outcomes=tuple(
                self.curriculum.state.stage_outcomes
            ),
            artifact_references=artifact_references or {},
        )

    def _prepare(self) -> UniversalPricingV2RunManifest:
        if self.resume:
            manifest = self.manifest_repository.read(self.manifest_path)
            if manifest.coordinate != self.coordinate:
                raise ValueError("Resume coordinate mismatch")
            if manifest.status is RunStatus.COMPLETED:
                raise ValueError("Completed v2 runs cannot be resumed")
            if not self.snapshot_path.is_file():
                raise ValueError("V2 resume snapshot does not exist")
            snapshot, coordinator = self.snapshot_repository.restore(
                self.snapshot_path,
                agent=self.agent,
                protocol=self.protocol,
            )
            self.curriculum = coordinator
            self.global_episode_index = (
                snapshot.next_global_episode_index
            )
            self._warmup_rng.bit_generator.state = dict(
                snapshot.warmup_rng_state
            )
            self.metric_records = [
                dict(item) for item in snapshot.metric_records
            ]
            self.timing_totals = dict(snapshot.timing_totals)
            self.started_at = (
                time.perf_counter() - snapshot.elapsed_wall_seconds
            )
            running = replace(manifest, status=RunStatus.RUNNING)
        else:
            if self.manifest_path.exists():
                raise ValueError(
                    "Run manifest already exists; use --resume or a new coordinate"
                )
            self.run_directory.mkdir(parents=True, exist_ok=True)
            self.started_at = time.perf_counter()
            running = self._manifest(RunStatus.RUNNING)
        self.manifest_repository.write(self.manifest_path, running)
        self.logger.log_run_start(
            run_id=running.run_id.value,
            architecture=self.coordinate.agent_architecture.value,
            distributions=(
                self.coordinate.distribution_combination.identifier
            ),
            environment_steps=(
                self.protocol.training_budget.environment_steps
            ),
            warmup_steps=self.protocol.training_budget.warmup_steps,
            device=self.device,
            parameter_count=self.agent.parameter_counts()[
                "total_parameters"
            ],
            run_directory=self.run_directory,
            resumed=self.resume,
        )
        return running

    def _request_stop(self, signum: int, frame: Any) -> None:
        del signum, frame
        self.stop_requested = True

    def _random_action(
        self,
        observation: Mapping[str, np.ndarray],
        mode: AgentRegimeMode,
    ) -> PricingAction:
        # Advance recurrent actor state even while replay is warming up.
        self.agent.prepare_observation(observation)
        _ = self.agent.select_action(
            observation, regime_mode=mode, deterministic=False
        )
        if mode is AgentRegimeMode.FORCED_UNIFORM:
            regime = PricingRegime.UNIFORM
        elif mode is AgentRegimeMode.FORCED_BBP:
            regime = PricingRegime.BBP
        elif observation["pricing"][17] > 0:
            regime = PricingRegime(
                int(self._warmup_rng.integers(0, 2))
            )
        else:
            regime = (
                PricingRegime.BBP
                if observation["pricing"][11] > 0
                else PricingRegime.UNIFORM
            )
        controls = self._warmup_rng.uniform(-1.0, 1.0, size=3)
        return PricingAction(
            regime=regime,
            uniform_control=float(controls[0]),
            bbp_new_control=float(controls[1]),
            bbp_premium_control=float(controls[2]),
        )

    def _select_action(
        self,
        observation: Mapping[str, np.ndarray],
        mode: AgentRegimeMode,
    ) -> PricingAction:
        if (
            self.curriculum.state.total_environment_steps
            < self.protocol.training_budget.warmup_steps
        ):
            return self._random_action(observation, mode)
        self.agent.prepare_observation(observation)
        return self.agent.select_action(
            observation, regime_mode=mode, deterministic=False
        )

    def _close_macro_transition(
        self,
        pending: dict[str, Any],
        *,
        next_observation: Mapping[str, np.ndarray],
        done: bool,
        stage_key: str,
    ) -> StrategyTransition:
        return StrategyTransition(
            observation=pending["observation"],
            regime_action=pending["regime_action"],
            macro_reward=pending["macro_reward"],
            next_observation=next_observation["strategy"],
            done=done,
            duration=pending["duration"],
            stage_key=stage_key,
            opponent_embedding=pending["opponent_embedding"],
            next_opponent_embedding=(
                self.agent.current_opponent_embedding()
            ),
        )

    def _run_episode(
        self,
    ) -> tuple[dict[str, Any], int]:
        phase = self.curriculum.state.phase
        mode = self.curriculum.regime_mode
        stage = self.curriculum.current_stage
        stage_key = self.curriculum.stage_key
        local_episode_index = (
            self.curriculum.state.stage_local_episode_index
        )
        context = self.context_factory.create(
            phase=phase,
            stage_index=self.curriculum.state.stage_index,
            local_episode_index=local_episode_index,
            agent_regime_mode=mode,
            opponent_policy_name=(
                stage.opponent_policy_name if stage is not None else None
            ),
            stage_key=stage_key,
        )
        observation, reset_info = self.environment.reset(
            options={"episode_context": context}
        )
        self.agent.set_training_phase(phase)
        self.agent.reset_recurrent_state()
        episode_metrics = HierarchicalPricingEpisodeMetrics()
        timing_before = dict(self.timing_totals)
        uniform_transitions: list[PricingSkillTransition] = []
        bbp_transitions: list[PricingSkillTransition] = []
        strategy_transitions: list[StrategyTransition] = []
        pending_macro: dict[str, Any] | None = None
        episode_started = time.perf_counter()
        episode_step_count = 0
        update_metric_values: list[dict[str, float]] = []

        while True:
            self.agent.prepare_observation(observation)
            boundary = bool(observation["pricing"][17] > 0.0)
            if (
                mode is AgentRegimeMode.LEARNED
                and boundary
                and pending_macro is not None
            ):
                transition = self._close_macro_transition(
                    pending_macro,
                    next_observation=observation,
                    done=False,
                    stage_key=stage_key,
                )
                self.agent.observe_strategy_transition(transition)
                strategy_transitions.append(transition)
                if self.coordinate.agent_architecture is AgentArchitecture.SAC:
                    self.agent.record_strategy_transition(transition)
                pending_macro = None

            inference_started = time.perf_counter()
            action = self._select_action(observation, mode)
            self.timing_totals["inference_seconds"] += (
                time.perf_counter() - inference_started
            )
            if mode is AgentRegimeMode.LEARNED and boundary:
                pending_macro = {
                    "observation": observation["strategy"].copy(),
                    "regime_action": int(action.regime),
                    "macro_reward": 0.0,
                    "duration": 0,
                    "opponent_embedding": (
                        self.agent.current_opponent_embedding()
                    ),
                }

            environment_started = time.perf_counter()
            (
                next_observation,
                reward,
                terminated,
                truncated,
                info,
            ) = self.environment.step(PricingActionCodec.to_gym(action))
            self.timing_totals["environment_seconds"] += (
                time.perf_counter() - environment_started
            )
            episode_step_count += 1
            if pending_macro is not None:
                pending_macro["macro_reward"] += float(reward)
                pending_macro["duration"] += 1

            uniform_transition = (
                PricingSkillTransition.from_environment_step(
                    pricing_skill=PricingSkill.UNIFORM,
                    observation=observation,
                    reward=reward,
                    next_observation=next_observation,
                    terminated=terminated,
                    truncated=truncated,
                    info=info,
                    stage_key=stage_key,
                )
            )
            bbp_transition = PricingSkillTransition.from_environment_step(
                pricing_skill=PricingSkill.BBP,
                observation=observation,
                reward=reward,
                next_observation=next_observation,
                terminated=terminated,
                truncated=truncated,
                info=info,
                stage_key=stage_key,
            )
            uniform_transitions.append(uniform_transition)
            bbp_transitions.append(bbp_transition)
            self.agent.observe_pricing_transition(uniform_transition)
            if self.coordinate.agent_architecture is AgentArchitecture.SAC:
                self.agent.record_pricing_transition(uniform_transition)
                self.agent.record_pricing_transition(bbp_transition)

            episode_metrics.record_step(
                reward=reward,
                info=info,
                policy_diagnostics=self.agent.policy_diagnostics(),
            )
            update_started = time.perf_counter()
            if (
                self.curriculum.state.total_environment_steps
                + episode_step_count
                >= self.protocol.training_budget.warmup_steps
            ):
                for _ in range(
                    self.protocol.training_budget.updates_per_step
                ):
                    values = self.agent.update_for_phase(
                        phase, stage_key=stage_key
                    )
                    if values:
                        episode_metrics.record_update(values)
                        update_metric_values.append(values)
            self.timing_totals["update_seconds"] += (
                time.perf_counter() - update_started
            )
            observation = next_observation
            if terminated or truncated:
                if pending_macro is not None:
                    transition = self._close_macro_transition(
                        pending_macro,
                        next_observation=next_observation,
                        done=True,
                        stage_key=stage_key,
                    )
                    self.agent.observe_strategy_transition(transition)
                    strategy_transitions.append(transition)
                    if (
                        self.coordinate.agent_architecture
                        is AgentArchitecture.SAC
                    ):
                        self.agent.record_strategy_transition(transition)
                break

        if self.coordinate.agent_architecture is not AgentArchitecture.SAC:
            self.agent.record_pricing_episode(
                PricingSkillEpisode(
                    pricing_skill=PricingSkill.UNIFORM,
                    transitions=tuple(uniform_transitions),
                    stage_key=stage_key,
                )
            )
            self.agent.record_pricing_episode(
                PricingSkillEpisode(
                    pricing_skill=PricingSkill.BBP,
                    transitions=tuple(bbp_transitions),
                    stage_key=stage_key,
                )
            )
            if strategy_transitions:
                self.agent.record_strategy_episode(
                    StrategyEpisode(tuple(strategy_transitions))
                )

        record: dict[str, Any] = {
            "record_type": "training_episode",
            "episode_index": self.global_episode_index,
            "environment_steps": (
                self.curriculum.state.total_environment_steps
                + episode_step_count
            ),
            "phase_at_episode_start": phase.value,
            "stage_key": stage_key,
            "opponent_family": reset_info["opponent_family"],
            "opponent_policy_name": reset_info["opponent_policy_name"],
            "consumer_seed": reset_info["consumer_seed"],
            "opponent_seed": reset_info["opponent_seed"],
            "episode_wall_seconds": (
                time.perf_counter() - episode_started
            ),
            **episode_metrics.summary(),
            **self.curriculum.diagnostics(),
            **self.agent.replay.diagnostics(),
            **self.parameter_counts,
        }
        episode_wall_seconds = float(record["episode_wall_seconds"])
        record.update(
            {
                "episode_environment_seconds": (
                    self.timing_totals["environment_seconds"]
                    - timing_before["environment_seconds"]
                ),
                "episode_inference_seconds": (
                    self.timing_totals["inference_seconds"]
                    - timing_before["inference_seconds"]
                ),
                "episode_update_seconds": (
                    self.timing_totals["update_seconds"]
                    - timing_before["update_seconds"]
                ),
                "cumulative_validation_seconds": self.timing_totals[
                    "validation_seconds"
                ],
                "cumulative_checkpoint_seconds": self.timing_totals[
                    "checkpoint_seconds"
                ],
                "wall_clock_seconds": (
                    time.perf_counter() - self.started_at
                ),
                "environment_steps_per_second": (
                    episode_step_count / max(episode_wall_seconds, 1e-12)
                ),
                "peak_gpu_memory_bytes": (
                    float(
                        torch.cuda.max_memory_allocated(self.agent.device)
                    )
                    if self.agent.device.type == "cuda"
                    else 0.0
                ),
                "strategy_macro_transition_count": float(
                    len(strategy_transitions)
                ),
                "strategy_macro_return_total": float(
                    np.sum(
                        [
                            item.macro_reward
                            for item in strategy_transitions
                        ]
                    )
                ),
                "strategy_mean_macro_return": float(
                    np.mean(
                        [
                            item.macro_reward
                            for item in strategy_transitions
                        ]
                    )
                )
                if strategy_transitions
                else 0.0,
                "strategy_mean_macro_duration": float(
                    np.mean(
                        [item.duration for item in strategy_transitions]
                    )
                )
                if strategy_transitions
                else 0.0,
            }
        )
        active_prefix = (
            "uniform_replay"
            if phase is HierarchicalTrainingPhase.UNIFORM_PRICING
            else "bbp_replay"
            if phase is HierarchicalTrainingPhase.BBP_PRICING
            else "strategy_replay"
        )
        record["active_replay_size"] = record.get(
            f"{active_prefix}_transition_count", 0.0
        )
        record["active_replay_unit"] = (
            "episodes"
            if self.coordinate.agent_architecture
            is not AgentArchitecture.SAC
            else "transitions"
        )
        record["active_replay_bbp_fraction"] = record[
            "agent_bbp_period_fraction"
        ]
        return record, episode_step_count

    def _pricing_mastery_result(self) -> MasteryResult:
        stage = self.curriculum.current_stage
        if stage is None:
            raise RuntimeError("Pricing mastery requires an active stage")
        phase = self.curriculum.state.phase
        regime = (
            PricingRegime.UNIFORM
            if phase is HierarchicalTrainingPhase.UNIFORM_PRICING
            else PricingRegime.BBP
        )
        mode = (
            EvaluationRegimeMode.FORCED_UNIFORM
            if regime is PricingRegime.UNIFORM
            else EvaluationRegimeMode.FORCED_BBP
        )
        seeds = self.protocol.seed_manifest.validation_environment_seeds[
            : self.protocol.mastery_gate.validation_seed_count
        ]
        evaluator = UniversalPricingV2Evaluator(
            self.protocol,
            self.coordinate,
            device=self.device,
            episode_length=self.environment.episode_length,
        )
        _, summary = evaluator.evaluate_agent(
            self.agent,
            seeds,
            regime_mode=mode,
            opponent_policy_names=[stage.opponent_policy_name],
            suite="mastery",
        )
        agent_profit = float(summary["mean_net_agent_profit_total"])
        cache_key = (
            self.coordinate.distribution_combination.identifier,
            regime.name,
            stage.opponent_policy_name,
        )
        if cache_key not in self._baseline_cache:
            random_profit = RandomPriceBaselineEvaluator(
                self.protocol,
                self.coordinate,
                episode_length=self.environment.episode_length,
            ).evaluate(
                regime=regime,
                opponent_policy_name=stage.opponent_policy_name,
                environment_seeds=seeds,
            )
            oracle_profit = ConstantPriceOracleEvaluator(
                self.protocol,
                self.coordinate,
                episode_length=self.environment.episode_length,
            ).optimize(
                regime=regime,
                opponent_policy_name=stage.opponent_policy_name,
                environment_seeds=seeds,
            ).mean_net_profit
            self._baseline_cache[cache_key] = (
                random_profit,
                oracle_profit,
            )
        random_profit, oracle_profit = self._baseline_cache[cache_key]
        return self.mastery_evaluator.pricing_result(
            agent_net_profit=agent_profit,
            random_net_profit=random_profit,
            oracle_net_profit=oracle_profit,
            validation_episode_count=len(seeds),
        )

    def _strategy_mastery_result(self) -> MasteryResult:
        seeds = self.protocol.seed_manifest.validation_environment_seeds[
            : self.protocol.mastery_gate.validation_seed_count
        ]
        evaluator = UniversalPricingV2Evaluator(
            self.protocol,
            self.coordinate,
            device=self.device,
            episode_length=self.environment.episode_length,
        )
        results: dict[EvaluationRegimeMode, list[dict[str, Any]]] = {}
        for mode in EvaluationRegimeMode:
            episodes, _ = evaluator.evaluate_agent(
                self.agent,
                seeds,
                regime_mode=mode,
                suite="strategy_mastery",
            )
            results[mode] = episodes

        def key(item: Mapping[str, Any]) -> tuple[int, str]:
            return (
                int(item["evaluation_seed_index"]),
                str(item["opponent_policy_name"]),
            )

        learned = {key(item): item for item in results[EvaluationRegimeMode.LEARNED]}
        random = {
            key(item): item
            for item in results[EvaluationRegimeMode.RANDOM_REGIME]
        }
        uniform = {
            key(item): item
            for item in results[EvaluationRegimeMode.FORCED_UNIFORM]
        }
        bbp = {
            key(item): item
            for item in results[EvaluationRegimeMode.FORCED_BBP]
        }
        shared = sorted(set(learned) & set(random) & set(uniform) & set(bbp))
        learned_profit = float(
            np.mean(
                [
                    learned[item]["net_agent_profit_total"]
                    for item in shared
                ]
            )
        )
        random_profit = float(
            np.mean(
                [
                    random[item]["net_agent_profit_total"]
                    for item in shared
                ]
            )
        )
        best_forced = float(
            np.mean(
                [
                    max(
                        uniform[item]["net_agent_profit_total"],
                        bbp[item]["net_agent_profit_total"],
                    )
                    for item in shared
                ]
            )
        )
        uniform_accuracy, bbp_accuracy = (
            StrategyMasteryScenarioEvaluator().selection_accuracies(
                self.agent, seeds
            )
        )
        return self.mastery_evaluator.strategy_result(
            learned_net_profit=learned_profit,
            random_regime_net_profit=random_profit,
            best_forced_net_profit=best_forced,
            uniform_scenario_accuracy=uniform_accuracy,
            bbp_scenario_accuracy=bbp_accuracy,
            validation_episode_count=len(shared),
        )

    def _run_mastery_if_due(self) -> None:
        if not self.curriculum.should_validate():
            return
        if not self.enable_mastery_evaluation:
            self.curriculum.advance_if_capped()
            return
        started = time.perf_counter()
        if self.curriculum.current_stage is not None:
            result = self._pricing_mastery_result()
        else:
            result = self._strategy_mastery_result()
        self.timing_totals["validation_seconds"] += (
            time.perf_counter() - started
        )
        stage_name = (
            self.curriculum.current_stage.name
            if self.curriculum.current_stage is not None
            else "strategy"
        )
        self.curriculum.record_mastery_result(result)
        self.logger.log_mastery(
            phase=self.curriculum.state.phase.value,
            stage=stage_name,
            score=result.skill_score,
            passed=result.passed,
            consecutive_passes=(
                self.curriculum.state.consecutive_mastery_passes
            ),
        )
        self.metric_records.append(
            {
                "record_type": "mastery",
                "environment_steps": (
                    self.curriculum.state.total_environment_steps
                ),
                "phase": self.curriculum.state.phase.value,
                "stage": stage_name,
                **result.to_dict(),
            }
        )
        self.curriculum.advance_if_capped()

    def _write_snapshot(self) -> None:
        started = time.perf_counter()
        self.snapshot_repository.write(
            self.snapshot_path,
            agent=self.agent,
            curriculum=self.curriculum,
            next_global_episode_index=self.global_episode_index,
            warmup_rng_state=self._warmup_rng.bit_generator.state,
            metric_records=self.metric_records,
            elapsed_wall_seconds=time.perf_counter() - self.started_at,
            timing_totals=self.timing_totals,
        )
        self.timing_totals["checkpoint_seconds"] += (
            time.perf_counter() - started
        )
        self.logger.log_checkpoint(
            self.snapshot_path,
            self.curriculum.state.total_environment_steps,
        )

    def train(self) -> UniversalPricingV2RunManifest:
        running = self._prepare()
        previous_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._request_stop)
        last_checkpoint = self.curriculum.state.total_environment_steps
        try:
            while not self.curriculum.is_completed:
                if self.stop_requested:
                    break
                if (
                    self.maximum_environment_steps is not None
                    and self.curriculum.state.total_environment_steps
                    >= self.maximum_environment_steps
                ):
                    self.stop_requested = True
                    break
                record, step_count = self._run_episode()
                self.curriculum.register_episode()
                self.curriculum.record_environment_steps(step_count)
                self.global_episode_index += 1
                record.update(self.curriculum.diagnostics())
                self.metric_records.append(record)
                self.logger.write_metric_records(self.metric_records)
                self.logger.log_episode(
                    record,
                    budget_steps=(
                        self.protocol.training_budget.environment_steps
                    ),
                )
                self._run_mastery_if_due()
                current_steps = (
                    self.curriculum.state.total_environment_steps
                )
                if (
                    current_steps - last_checkpoint
                    >= self.protocol.training_budget.checkpoint_interval_steps
                ):
                    self._write_snapshot()
                    last_checkpoint = current_steps
                    running = replace(
                        running,
                        curriculum_state=self.curriculum.state.to_dict(),
                        stage_outcomes=tuple(
                            self.curriculum.state.stage_outcomes
                        ),
                    )
                    self.manifest_repository.write(
                        self.manifest_path, running
                    )

            if self.stop_requested:
                self._write_snapshot()
                interrupted = replace(
                    running,
                    status=RunStatus.INTERRUPTED,
                    curriculum_state=self.curriculum.state.to_dict(),
                    stage_outcomes=tuple(
                        self.curriculum.state.stage_outcomes
                    ),
                    artifact_references={
                        "latest_snapshot": str(self.snapshot_path)
                    },
                )
                self.manifest_repository.write(
                    self.manifest_path, interrupted
                )
                self.logger.log_terminal(
                    "interrupted",
                    environment_steps=(
                        self.curriculum.state.total_environment_steps
                    ),
                )
                return interrupted

            self.agent.save(self.final_checkpoint_path)
            completed = replace(
                running,
                status=RunStatus.COMPLETED,
                curriculum_state=self.curriculum.state.to_dict(),
                stage_outcomes=tuple(
                    self.curriculum.state.stage_outcomes
                ),
                artifact_references={
                    "final_checkpoint": str(self.final_checkpoint_path),
                    "metrics": str(
                        self.layout.metrics_path(self.coordinate)
                    ),
                },
            )
            self.manifest_repository.write(
                self.manifest_path, completed
            )
            self.logger.log_terminal(
                "completed",
                environment_steps=(
                    self.curriculum.state.total_environment_steps
                ),
            )
            return completed
        except BaseException as exc:
            failed = replace(
                running,
                status=RunStatus.FAILED,
                curriculum_state=self.curriculum.state.to_dict(),
                stage_outcomes=tuple(
                    self.curriculum.state.stage_outcomes
                ),
                artifact_references={
                    "latest_snapshot": str(self.snapshot_path)
                    if self.snapshot_path.exists()
                    else ""
                },
            )
            self.manifest_repository.write(self.manifest_path, failed)
            self.logger.log_terminal(
                "failed",
                environment_steps=(
                    self.curriculum.state.total_environment_steps
                ),
                message=str(exc),
            )
            raise
        finally:
            signal.signal(signal.SIGINT, previous_handler)
            self.environment.close()
