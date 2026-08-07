"""Composite hierarchical SAC, RSAC, and opponent-embedding RSAC agents."""

from __future__ import annotations

import copy
import math
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from env.pricing_contracts import (
    AgentArchitecture,
    PricingAction,
    PricingObservationCodec,
    PricingRegime,
)
from universal_pricing_v2.controllers import (
    FeedForwardContinuousSACController,
    FeedForwardDiscreteSACController,
    RecurrentContinuousSACController,
    RecurrentDiscreteSACController,
)
from universal_pricing_v2.networks import (
    OpponentControlPredictor,
    SharedOpponentHistoryEncoder,
)
from universal_pricing_v2.observations import StrategyObservationCodec
from universal_pricing_v2.protocol import (
    AgentRegimeMode,
    HierarchicalAgentProfileConfig,
    HierarchicalSeedDeriver,
    HierarchicalTrainingPhase,
    PricingSkill,
    RunSeedBundle,
    V2SeedNamespace,
)
from universal_pricing_v2.replay import (
    CurriculumReplayRepository,
    FeedForwardPricingReplayBuffer,
    PricingSkillEpisode,
    PricingSkillTransition,
    RecurrentPricingEpisodeReplay,
    RecurrentStrategyEpisodeReplay,
    StrategyEpisode,
    StrategyReplayBuffer,
    StrategyTransition,
)


V2_AGENT_CHECKPOINT_SCHEMA = 1


@contextmanager
def _isolated_torch_initialization(seed: int):
    """Initialize modules without perturbing caller Torch RNG state."""

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        yield


def _device_generator(device: torch.device, seed: int) -> torch.Generator:
    generator = torch.Generator(device=device.type)
    generator.manual_seed(int(seed))
    return generator


def _soft_update_module(
    source: nn.Module, target: nn.Module, tau: float
) -> None:
    with torch.no_grad():
        for source_parameter, target_parameter in zip(
            source.parameters(), target.parameters()
        ):
            target_parameter.mul_(1.0 - tau)
            target_parameter.add_(source_parameter, alpha=tau)


class BaseHierarchicalPricingAgent:
    """Common runtime, replay, phase, and checkpoint behavior."""

    architecture: AgentArchitecture

    def __init__(
        self,
        profile: HierarchicalAgentProfileConfig,
        run_seed_bundle: RunSeedBundle,
        *,
        device: str | torch.device = "cpu",
    ) -> None:
        self.profile = profile
        self.run_seed_bundle = run_seed_bundle
        self.device = torch.device(device)
        if (
            self.device.type == "cuda"
            and not torch.cuda.is_available()
        ):
            raise ValueError("CUDA requested but unavailable")
        self.architecture = profile.architecture
        self._exploration_generator = _device_generator(
            self.device,
            HierarchicalSeedDeriver.derive(
                run_seed_bundle.run_seed, V2SeedNamespace.EXPLORATION
            ),
        )
        self._training_generator = _device_generator(
            self.device,
            HierarchicalSeedDeriver.derive(
                run_seed_bundle.run_seed, V2SeedNamespace.TRAINING
            ),
        )
        embedding_dimension = (
            int(profile.opponent_embedding_dimension)
            if profile.architecture is AgentArchitecture.OE_RSAC
            else 0
        )
        self.replay = CurriculumReplayRepository(
            architecture=profile.architecture,
            pricing_capacity=profile.pricing_replay_capacity,
            strategy_capacity=profile.strategy_replay_capacity,
            episode_capacity=profile.episode_replay_capacity,
            price_sequence_length=profile.price_sequence_length,
            price_burn_in_length=profile.price_burn_in_length,
            strategy_sequence_length=profile.strategy_sequence_length,
            strategy_burn_in_length=profile.strategy_burn_in_length,
            uniform_seed=HierarchicalSeedDeriver.derive(
                run_seed_bundle.run_seed, V2SeedNamespace.UNIFORM_REPLAY
            ),
            bbp_seed=HierarchicalSeedDeriver.derive(
                run_seed_bundle.run_seed, V2SeedNamespace.BBP_REPLAY
            ),
            strategy_seed=HierarchicalSeedDeriver.derive(
                run_seed_bundle.run_seed, V2SeedNamespace.STRATEGY_REPLAY
            ),
            opponent_embedding_dimension=embedding_dimension,
        )
        self.current_phase = HierarchicalTrainingPhase.UNIFORM_PRICING
        self.environment_steps = 0
        self.update_steps = 0
        self._last_diagnostics: dict[str, float] = {}

    def set_training_phase(
        self, phase: HierarchicalTrainingPhase
    ) -> None:
        phase = HierarchicalTrainingPhase(phase)
        self.current_phase = phase
        if phase is HierarchicalTrainingPhase.JOINT_CONSOLIDATION:
            self.uniform_controller.set_actor_learning_rate(
                self.profile.joint_price_learning_rate
            )
            self.bbp_controller.set_actor_learning_rate(
                self.profile.joint_price_learning_rate
            )
        else:
            self.uniform_controller.set_actor_learning_rate(
                self.profile.actor_learning_rate
            )
            self.bbp_controller.set_actor_learning_rate(
                self.profile.actor_learning_rate
            )

    @staticmethod
    def _validate_observation(
        observation: Mapping[str, Any],
    ) -> tuple[np.ndarray, np.ndarray]:
        if not isinstance(observation, Mapping) or set(observation) != {
            "pricing",
            "strategy",
        }:
            raise ValueError(
                "V2 observation must contain pricing and strategy"
            )
        return (
            PricingObservationCodec.validate_vector(
                observation["pricing"]
            ),
            StrategyObservationCodec.validate_vector(
                observation["strategy"]
            ),
        )

    @staticmethod
    def _current_regime(pricing_observation: np.ndarray) -> PricingRegime:
        return (
            PricingRegime.BBP
            if pricing_observation[11] > 0.0
            else PricingRegime.UNIFORM
        )

    @staticmethod
    def _decision_allowed(pricing_observation: np.ndarray) -> bool:
        return bool(pricing_observation[17] > 0.0)

    def _resolve_mode_regime(
        self,
        pricing_observation: np.ndarray,
        regime_mode: AgentRegimeMode,
        proposed: PricingRegime | None,
    ) -> PricingRegime:
        mode = AgentRegimeMode(regime_mode)
        if mode is AgentRegimeMode.FORCED_UNIFORM:
            return PricingRegime.UNIFORM
        if mode is AgentRegimeMode.FORCED_BBP:
            return PricingRegime.BBP
        return proposed or self._current_regime(pricing_observation)

    def record_pricing_transition(
        self, transition: PricingSkillTransition
    ) -> None:
        replay = (
            self.replay.uniform_pricing
            if transition.pricing_skill is PricingSkill.UNIFORM
            else self.replay.bbp_pricing
        )
        if not isinstance(replay, FeedForwardPricingReplayBuffer):
            raise TypeError(
                "Recurrent agents record complete pricing episodes"
            )
        replay.push(transition)

    def record_pricing_episode(self, episode: PricingSkillEpisode) -> None:
        replay = (
            self.replay.uniform_pricing
            if episode.pricing_skill is PricingSkill.UNIFORM
            else self.replay.bbp_pricing
        )
        if not isinstance(replay, RecurrentPricingEpisodeReplay):
            raise TypeError(
                "Feed-forward agents record individual transitions"
            )
        replay.push_episode(episode)

    def record_strategy_transition(
        self, transition: StrategyTransition
    ) -> None:
        if not isinstance(self.replay.strategy, StrategyReplayBuffer):
            raise TypeError(
                "Recurrent agents record complete strategy episodes"
            )
        self.replay.strategy.push(transition)

    def record_strategy_episode(self, episode: StrategyEpisode) -> None:
        if not isinstance(
            self.replay.strategy, RecurrentStrategyEpisodeReplay
        ):
            raise TypeError(
                "Feed-forward agents record strategy transitions"
            )
        self.replay.strategy.push_episode(episode)

    def current_opponent_embedding(self) -> np.ndarray | None:
        return None

    def prepare_observation(self, observation: Mapping[str, Any]) -> None:
        """Prepare optional opponent context before a macro decision."""

        self._validate_observation(observation)

    def observe_pricing_transition(
        self, transition: PricingSkillTransition
    ) -> None:
        self.environment_steps += 1

    def observe_strategy_transition(
        self, transition: StrategyTransition
    ) -> None:
        return None

    def reset_recurrent_state(self) -> None:
        return None

    def _controller_state_dict(self) -> dict[str, Any]:
        return {
            "uniform": self.uniform_controller.state_dict(),
            "bbp": self.bbp_controller.state_dict(),
            "strategy": self.strategy_controller.state_dict(),
        }

    def _load_controller_state_dict(
        self, values: Mapping[str, Any]
    ) -> None:
        self.uniform_controller.load_state_dict(values["uniform"])
        self.bbp_controller.load_state_dict(values["bbp"])
        self.strategy_controller.load_state_dict(values["strategy"])

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": V2_AGENT_CHECKPOINT_SCHEMA,
            "protocol_version": "universal_pricing_v2",
            "architecture": self.architecture.value,
            "profile": self.profile.to_dict(),
            "controllers": self._controller_state_dict(),
            "environment_steps": self.environment_steps,
            "update_steps": self.update_steps,
            "current_phase": self.current_phase.value,
            "exploration_generator_state": (
                self._exploration_generator.get_state().cpu()
            ),
            "training_generator_state": (
                self._training_generator.get_state().cpu()
            ),
            "online_state": self._online_state_dict(),
        }

    def load_state_dict(self, values: Mapping[str, Any]) -> None:
        if (
            int(values["schema_version"]) != V2_AGENT_CHECKPOINT_SCHEMA
            or values["protocol_version"] != "universal_pricing_v2"
            or values["architecture"] != self.architecture.value
            or values["profile"] != self.profile.to_dict()
        ):
            raise ValueError("Incompatible v2 agent checkpoint")
        self._load_controller_state_dict(values["controllers"])
        self.environment_steps = int(values["environment_steps"])
        self.update_steps = int(values["update_steps"])
        self.current_phase = HierarchicalTrainingPhase(
            values["current_phase"]
        )
        self._exploration_generator.set_state(
            values["exploration_generator_state"].cpu()
        )
        self._training_generator.set_state(
            values["training_generator_state"].cpu()
        )
        self._load_online_state_dict(values["online_state"])

    def _online_state_dict(self) -> dict[str, Any]:
        return {}

    def _load_online_state_dict(self, values: Mapping[str, Any]) -> None:
        if values:
            raise ValueError("Feed-forward checkpoint has recurrent state")

    def save(self, path: str | Path) -> None:
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=".a-", suffix=".pt", dir=checkpoint_path.parent
        )
        os.close(descriptor)
        try:
            torch.save(self.state_dict(), temporary)
            os.replace(temporary, checkpoint_path)
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass

    def load(self, path: str | Path) -> None:
        values = torch.load(
            Path(path), map_location=self.device, weights_only=False
        )
        if not isinstance(values, Mapping):
            raise ValueError("Malformed v2 agent checkpoint")
        self.load_state_dict(values)

    def parameter_counts(self) -> dict[str, int]:
        def count_controller(controller: Any) -> int:
            modules = (
                controller.actor,
                controller.critic_1,
                controller.critic_2,
            )
            return int(
                sum(
                    parameter.numel()
                    for module in modules
                    for parameter in module.parameters()
                )
                + controller.log_temperature.numel()
            )

        result = {
            "uniform_controller_parameters": count_controller(
                self.uniform_controller
            ),
            "bbp_controller_parameters": count_controller(
                self.bbp_controller
            ),
            "strategy_controller_parameters": count_controller(
                self.strategy_controller
            ),
        }
        result["total_parameters"] = sum(result.values())
        return result

    def policy_diagnostics(self) -> dict[str, float]:
        result = dict(self._last_diagnostics)
        result.update(
            {
                name: float(value)
                for name, value in self.parameter_counts().items()
            }
        )
        result.update(self.replay.diagnostics())
        result["environment_steps"] = float(self.environment_steps)
        result["update_steps"] = float(self.update_steps)
        return result


class HierarchicalSACPricingAgent(BaseHierarchicalPricingAgent):
    """Feed-forward SAC with three completely separate controller groups."""

    architecture = AgentArchitecture.SAC

    def __init__(
        self,
        profile: HierarchicalAgentProfileConfig,
        run_seed_bundle: RunSeedBundle,
        *,
        device: str | torch.device = "cpu",
    ) -> None:
        if profile.architecture is not AgentArchitecture.SAC:
            raise ValueError("Hierarchical SAC requires sac profile")
        super().__init__(profile, run_seed_bundle, device=device)
        with _isolated_torch_initialization(
            run_seed_bundle.network_initialization_seed
        ):
            common = {
                "actor_learning_rate": profile.actor_learning_rate,
                "critic_learning_rate": profile.critic_learning_rate,
                "entropy_learning_rate": profile.entropy_learning_rate,
                "initial_temperature": profile.initial_temperature,
                "gamma": profile.gamma_price,
                "tau": profile.tau,
                "gradient_clip_norm": profile.gradient_clip_norm,
                "device": self.device,
            }
            self.uniform_controller = FeedForwardContinuousSACController(
                observation_dimension=18,
                action_dimension=1,
                actor_hidden_dimensions=profile.pricing_hidden_dimensions,
                critic_hidden_dimensions=profile.pricing_hidden_dimensions,
                **common,
            )
            self.bbp_controller = FeedForwardContinuousSACController(
                observation_dimension=18,
                action_dimension=2,
                actor_hidden_dimensions=profile.pricing_hidden_dimensions,
                critic_hidden_dimensions=profile.pricing_hidden_dimensions,
                **common,
            )
            self.strategy_controller = FeedForwardDiscreteSACController(
                observation_dimension=19,
                hidden_dimensions=profile.strategy_hidden_dimensions,
                actor_learning_rate=profile.actor_learning_rate,
                critic_learning_rate=profile.critic_learning_rate,
                entropy_learning_rate=profile.entropy_learning_rate,
                initial_temperature=profile.initial_temperature,
                gamma_price=profile.gamma_strategy,
                tau=profile.tau,
                gradient_clip_norm=profile.gradient_clip_norm,
                device=self.device,
            )

    def select_action(
        self,
        observation: Mapping[str, Any],
        *,
        regime_mode: AgentRegimeMode = AgentRegimeMode.LEARNED,
        deterministic: bool = False,
    ) -> PricingAction:
        pricing, strategy = self._validate_observation(observation)
        pricing_tensor = torch.as_tensor(
            pricing, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        strategy_tensor = torch.as_tensor(
            strategy, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        proposed: PricingRegime | None = None
        strategy_probabilities = np.asarray([1.0, 0.0])
        with torch.no_grad():
            if (
                AgentRegimeMode(regime_mode) is AgentRegimeMode.LEARNED
                and self._decision_allowed(pricing)
            ):
                actions, _, probabilities = self.strategy_controller.select(
                    strategy_tensor,
                    generator=self._exploration_generator,
                    deterministic=deterministic,
                )
                proposed = PricingRegime(int(actions[0, 0].cpu()))
                strategy_probabilities = (
                    probabilities[0].detach().cpu().numpy()
                )
            regime = self._resolve_mode_regime(
                pricing, regime_mode, proposed
            )
            uniform_controls, _ = self.uniform_controller.select(
                pricing_tensor,
                generator=self._exploration_generator,
                deterministic=deterministic,
            )
            bbp_controls, _ = self.bbp_controller.select(
                pricing_tensor,
                generator=self._exploration_generator,
                deterministic=deterministic,
            )
        self._last_diagnostics = {
            "uniform_regime_probability": float(
                strategy_probabilities[0]
            ),
            "bbp_regime_probability": float(strategy_probabilities[1]),
            "selected_regime": float(regime),
            "uniform_control": float(uniform_controls[0, 0].cpu()),
            "bbp_new_control": float(bbp_controls[0, 0].cpu()),
            "bbp_premium_control": float(bbp_controls[0, 1].cpu()),
        }
        return PricingAction(
            regime=regime,
            uniform_control=float(uniform_controls[0, 0].cpu()),
            bbp_new_control=float(bbp_controls[0, 0].cpu()),
            bbp_premium_control=float(bbp_controls[0, 1].cpu()),
        )

    def update_for_phase(
        self,
        phase: HierarchicalTrainingPhase,
        *,
        stage_key: str,
    ) -> dict[str, float]:
        phase = HierarchicalTrainingPhase(phase)
        metrics: dict[str, float] = {}
        batch_size = self.profile.batch_size
        if phase is HierarchicalTrainingPhase.UNIFORM_PRICING:
            if len(self.replay.uniform_pricing) >= batch_size:
                batch = self.replay.uniform_pricing.sample(
                    batch_size, current_stage_key=stage_key
                )
                metrics.update(
                    self.uniform_controller.update(
                        batch.as_mapping(),
                        generator=self._training_generator,
                    ).prefixed("uniform")
                )
        elif phase is HierarchicalTrainingPhase.BBP_PRICING:
            if len(self.replay.bbp_pricing) >= batch_size:
                batch = self.replay.bbp_pricing.sample(
                    batch_size, current_stage_key=stage_key
                )
                metrics.update(
                    self.bbp_controller.update(
                        batch.as_mapping(),
                        generator=self._training_generator,
                    ).prefixed("bbp")
                )
        else:
            if len(self.replay.strategy) >= batch_size:
                strategy_batch = self.replay.strategy.sample(batch_size)
                metrics.update(
                    self.strategy_controller.update(
                        strategy_batch.as_mapping()
                    ).prefixed("strategy")
                )
            if phase is HierarchicalTrainingPhase.JOINT_CONSOLIDATION:
                for prefix, replay, controller in (
                    (
                        "uniform",
                        self.replay.uniform_pricing,
                        self.uniform_controller,
                    ),
                    ("bbp", self.replay.bbp_pricing, self.bbp_controller),
                ):
                    if len(replay) >= batch_size:
                        batch = replay.sample(
                            batch_size, current_stage_key=stage_key
                        )
                        if np.sum(
                            batch.active_controller_masks
                        ) > 0:
                            metrics.update(
                                controller.update(
                                    batch.as_mapping(),
                                    generator=self._training_generator,
                                ).prefixed(prefix)
                            )
        if metrics:
            self.update_steps += 1
        return metrics


class HierarchicalRecurrentSACPricingAgent(BaseHierarchicalPricingAgent):
    """Plain recurrent hierarchical SAC with no opponent encoder."""

    architecture = AgentArchitecture.RSAC

    def __init__(
        self,
        profile: HierarchicalAgentProfileConfig,
        run_seed_bundle: RunSeedBundle,
        *,
        device: str | torch.device = "cpu",
    ) -> None:
        if profile.architecture not in {
            AgentArchitecture.RSAC,
            AgentArchitecture.OE_RSAC,
        }:
            raise ValueError("Recurrent hierarchical agent needs recurrent profile")
        super().__init__(profile, run_seed_bundle, device=device)
        embedding_dimension = self._embedding_dimension
        hidden = int(profile.recurrent_hidden_dimension)
        with _isolated_torch_initialization(
            run_seed_bundle.network_initialization_seed
        ):
            common = {
                "hidden_dimension": hidden,
                "actor_learning_rate": profile.actor_learning_rate,
                "critic_learning_rate": profile.critic_learning_rate,
                "entropy_learning_rate": profile.entropy_learning_rate,
                "initial_temperature": profile.initial_temperature,
                "gamma": profile.gamma_price,
                "tau": profile.tau,
                "gradient_clip_norm": profile.gradient_clip_norm,
                "device": self.device,
            }
            self.uniform_controller = RecurrentContinuousSACController(
                recurrent_input_dimension=21 + embedding_dimension,
                action_dimension=1,
                **common,
            )
            self.bbp_controller = RecurrentContinuousSACController(
                recurrent_input_dimension=22 + embedding_dimension,
                action_dimension=2,
                **common,
            )
            self.strategy_controller = RecurrentDiscreteSACController(
                recurrent_input_dimension=22 + embedding_dimension,
                hidden_dimension=hidden,
                actor_learning_rate=profile.actor_learning_rate,
                critic_learning_rate=profile.critic_learning_rate,
                entropy_learning_rate=profile.entropy_learning_rate,
                initial_temperature=profile.initial_temperature,
                gamma_price=profile.gamma_strategy,
                tau=profile.tau,
                gradient_clip_norm=profile.gradient_clip_norm,
                device=self.device,
            )
        self.reset_recurrent_state()

    @property
    def _embedding_dimension(self) -> int:
        return 0

    def _online_embedding(
        self, pricing: np.ndarray
    ) -> torch.Tensor | None:
        return None

    def reset_recurrent_state(self) -> None:
        self._uniform_actor_hidden: torch.Tensor | None = None
        self._bbp_actor_hidden: torch.Tensor | None = None
        self._strategy_actor_hidden: torch.Tensor | None = None
        self._previous_uniform_action = np.zeros(1, dtype=np.float32)
        self._previous_bbp_action = np.zeros(2, dtype=np.float32)
        self._previous_reward = 0.0
        self._previous_uniform_active = 0.0
        self._previous_bbp_active = 0.0
        self._previous_effective_action = np.zeros(5, dtype=np.float32)
        self._previous_opponent_controls = np.zeros(3, dtype=np.float32)
        self._previous_strategy_one_hot = np.zeros(2, dtype=np.float32)
        self._previous_macro_reward = 0.0
        self._current_embedding: torch.Tensor | None = None
        self._prepared_pricing_observation: bytes | None = None
        self._prepared_embedding: torch.Tensor | None = None

    def prepare_observation(self, observation: Mapping[str, Any]) -> None:
        pricing, _ = self._validate_observation(observation)
        key = pricing.tobytes()
        if self._prepared_pricing_observation == key:
            return
        self._prepared_embedding = self._online_embedding(pricing)
        self._prepared_pricing_observation = key

    def _price_input(
        self,
        pricing: np.ndarray,
        *,
        previous_action: np.ndarray,
        previous_active: float,
        embedding: torch.Tensor | None,
    ) -> torch.Tensor:
        parts = [
            torch.as_tensor(
                pricing, dtype=torch.float32, device=self.device
            ),
            torch.as_tensor(
                previous_action, dtype=torch.float32, device=self.device
            ),
            torch.tensor(
                [self._previous_reward],
                dtype=torch.float32,
                device=self.device,
            ),
            torch.tensor(
                [previous_active],
                dtype=torch.float32,
                device=self.device,
            ),
        ]
        if embedding is not None:
            parts.append(embedding.reshape(-1))
        return torch.cat(parts).reshape(1, 1, -1)

    def select_action(
        self,
        observation: Mapping[str, Any],
        *,
        regime_mode: AgentRegimeMode = AgentRegimeMode.LEARNED,
        deterministic: bool = False,
    ) -> PricingAction:
        pricing, strategy = self._validate_observation(observation)
        self.prepare_observation(observation)
        embedding = self._prepared_embedding
        uniform_input = self._price_input(
            pricing,
            previous_action=self._previous_uniform_action,
            previous_active=self._previous_uniform_active,
            embedding=embedding,
        )
        bbp_input = self._price_input(
            pricing,
            previous_action=self._previous_bbp_action,
            previous_active=self._previous_bbp_active,
            embedding=embedding,
        )
        proposed: PricingRegime | None = None
        strategy_probabilities = np.asarray([1.0, 0.0])
        with torch.no_grad():
            uniform_action, _, self._uniform_actor_hidden = (
                self.uniform_controller.actor.sample(
                    uniform_input,
                    self._uniform_actor_hidden,
                    generator=self._exploration_generator,
                    deterministic=deterministic,
                )
            )
            bbp_action, _, self._bbp_actor_hidden = (
                self.bbp_controller.actor.sample(
                    bbp_input,
                    self._bbp_actor_hidden,
                    generator=self._exploration_generator,
                    deterministic=deterministic,
                )
            )
            if (
                AgentRegimeMode(regime_mode) is AgentRegimeMode.LEARNED
                and self._decision_allowed(pricing)
            ):
                strategy_parts = [
                    torch.as_tensor(
                        strategy,
                        dtype=torch.float32,
                        device=self.device,
                    ),
                    torch.as_tensor(
                        self._previous_strategy_one_hot,
                        dtype=torch.float32,
                        device=self.device,
                    ),
                    torch.tensor(
                        [self._previous_macro_reward],
                        dtype=torch.float32,
                        device=self.device,
                    ),
                ]
                if embedding is not None:
                    strategy_parts.append(embedding.reshape(-1))
                strategy_input = torch.cat(strategy_parts).reshape(
                    1, 1, -1
                )
                (
                    strategy_action,
                    _,
                    probabilities,
                    self._strategy_actor_hidden,
                ) = self.strategy_controller.actor.sample(
                    strategy_input,
                    self._strategy_actor_hidden,
                    generator=self._exploration_generator,
                    deterministic=deterministic,
                )
                proposed = PricingRegime(
                    int(strategy_action[0, 0, 0].cpu())
                )
                strategy_probabilities = (
                    probabilities[0, 0].detach().cpu().numpy()
                )
        regime = self._resolve_mode_regime(pricing, regime_mode, proposed)
        self._last_diagnostics = {
            "uniform_regime_probability": float(
                strategy_probabilities[0]
            ),
            "bbp_regime_probability": float(strategy_probabilities[1]),
            "selected_regime": float(regime),
            "uniform_control": float(uniform_action[0, 0, 0].cpu()),
            "bbp_new_control": float(bbp_action[0, 0, 0].cpu()),
            "bbp_premium_control": float(bbp_action[0, 0, 1].cpu()),
            "uniform_actor_hidden_norm": float(
                self._uniform_actor_hidden.norm().cpu()
            ),
            "bbp_actor_hidden_norm": float(
                self._bbp_actor_hidden.norm().cpu()
            ),
        }
        return PricingAction(
            regime=regime,
            uniform_control=float(uniform_action[0, 0, 0].cpu()),
            bbp_new_control=float(bbp_action[0, 0, 0].cpu()),
            bbp_premium_control=float(bbp_action[0, 0, 1].cpu()),
        )

    def observe_pricing_transition(
        self, transition: PricingSkillTransition
    ) -> None:
        super().observe_pricing_transition(transition)
        effective = transition.effective_action
        self._previous_uniform_action = effective[2:3].copy()
        self._previous_bbp_action = effective[3:5].copy()
        self._previous_reward = transition.reward
        self._previous_uniform_active = float(effective[0])
        self._previous_bbp_active = float(effective[1])
        self._previous_effective_action = effective.copy()
        self._previous_opponent_controls = (
            transition.opponent_price_controls.copy()
        )
        self._prepared_pricing_observation = None
        self._prepared_embedding = None

    def observe_strategy_transition(
        self, transition: StrategyTransition
    ) -> None:
        self._previous_strategy_one_hot = np.zeros(2, dtype=np.float32)
        self._previous_strategy_one_hot[transition.regime_action] = 1.0
        self._previous_macro_reward = transition.macro_reward

    def _online_state_dict(self) -> dict[str, Any]:
        def cpu(value: torch.Tensor | None) -> torch.Tensor | None:
            return None if value is None else value.detach().cpu()

        return {
            "uniform_actor_hidden": cpu(self._uniform_actor_hidden),
            "bbp_actor_hidden": cpu(self._bbp_actor_hidden),
            "strategy_actor_hidden": cpu(self._strategy_actor_hidden),
            "previous_uniform_action": self._previous_uniform_action,
            "previous_bbp_action": self._previous_bbp_action,
            "previous_reward": self._previous_reward,
            "previous_uniform_active": self._previous_uniform_active,
            "previous_bbp_active": self._previous_bbp_active,
            "previous_effective_action": self._previous_effective_action,
            "previous_opponent_controls": self._previous_opponent_controls,
            "previous_strategy_one_hot": self._previous_strategy_one_hot,
            "previous_macro_reward": self._previous_macro_reward,
            "prepared_pricing_observation": (
                self._prepared_pricing_observation
            ),
            "prepared_embedding": cpu(self._prepared_embedding),
        }

    def _load_online_state_dict(self, values: Mapping[str, Any]) -> None:
        for name in (
            "uniform_actor_hidden",
            "bbp_actor_hidden",
            "strategy_actor_hidden",
        ):
            value = values[name]
            setattr(
                self,
                f"_{name}",
                None if value is None else value.to(self.device),
            )
        for name in (
            "previous_uniform_action",
            "previous_bbp_action",
            "previous_effective_action",
            "previous_opponent_controls",
            "previous_strategy_one_hot",
        ):
            setattr(
                self,
                f"_{name}",
                np.asarray(values[name], dtype=np.float32).copy(),
            )
        for name in (
            "previous_reward",
            "previous_uniform_active",
            "previous_bbp_active",
            "previous_macro_reward",
        ):
            setattr(self, f"_{name}", float(values[name]))
        self._prepared_pricing_observation = values.get(
            "prepared_pricing_observation"
        )
        prepared_embedding = values.get("prepared_embedding")
        self._prepared_embedding = (
            None
            if prepared_embedding is None
            else prepared_embedding.to(self.device)
        )

    def _price_embeddings(
        self,
        batch: Mapping[str, Any],
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        return None, None

    def _after_price_update(
        self,
        batch: Mapping[str, Any],
        metrics: dict[str, float],
    ) -> None:
        return None

    def _strategy_embeddings(
        self, batch: Mapping[str, Any]
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        return None, None

    def update_for_phase(
        self,
        phase: HierarchicalTrainingPhase,
        *,
        stage_key: str,
    ) -> dict[str, float]:
        phase = HierarchicalTrainingPhase(phase)
        metrics: dict[str, float] = {}
        batch_size = self.profile.batch_size

        def update_price(
            prefix: str,
            replay: RecurrentPricingEpisodeReplay,
            controller: RecurrentContinuousSACController,
        ) -> None:
            if not replay:
                return
            batch = replay.sample(
                batch_size, current_stage_key=stage_key
            ).as_mapping()
            active_loss = (
                batch["active_controller_masks"] * batch["loss_masks"]
            )
            if np.sum(active_loss) <= 0:
                return
            valid_count = float(np.sum(batch["valid_masks"]))
            loss_count = float(np.sum(batch["loss_masks"]))
            metrics.update(
                {
                    f"{prefix}_recurrent_padding_fraction": (
                        1.0
                        - valid_count
                        / float(np.size(batch["valid_masks"]))
                    ),
                    f"{prefix}_recurrent_burn_in_fraction": (
                        (valid_count - loss_count) / max(valid_count, 1.0)
                    ),
                }
            )
            embeddings, next_embeddings = self._price_embeddings(batch)
            result = controller.update(
                batch,
                generator=self._training_generator,
                embeddings=embeddings,
                next_embeddings=next_embeddings,
            )
            metrics.update(result.prefixed(prefix))
            self._after_price_update(batch, metrics)

        if phase is HierarchicalTrainingPhase.UNIFORM_PRICING:
            update_price(
                "uniform",
                self.replay.uniform_pricing,
                self.uniform_controller,
            )
        elif phase is HierarchicalTrainingPhase.BBP_PRICING:
            update_price(
                "bbp", self.replay.bbp_pricing, self.bbp_controller
            )
        else:
            if self.replay.strategy:
                batch = self.replay.strategy.sample(
                    batch_size
                ).as_mapping()
                valid_count = float(np.sum(batch["valid_masks"]))
                loss_count = float(np.sum(batch["loss_masks"]))
                metrics.update(
                    {
                        "strategy_recurrent_padding_fraction": (
                            1.0
                            - valid_count
                            / float(np.size(batch["valid_masks"]))
                        ),
                        "strategy_recurrent_burn_in_fraction": (
                            (valid_count - loss_count)
                            / max(valid_count, 1.0)
                        ),
                    }
                )
                embeddings, next_embeddings = (
                    self._strategy_embeddings(batch)
                )
                metrics.update(
                    self.strategy_controller.update(
                        batch,
                        embeddings=embeddings,
                        next_embeddings=next_embeddings,
                    ).prefixed("strategy")
                )
            if phase is HierarchicalTrainingPhase.JOINT_CONSOLIDATION:
                update_price(
                    "uniform",
                    self.replay.uniform_pricing,
                    self.uniform_controller,
                )
                update_price(
                    "bbp",
                    self.replay.bbp_pricing,
                    self.bbp_controller,
                )
        if metrics:
            self.update_steps += 1
        return metrics


class HierarchicalOpponentEmbeddingRecurrentSACPricingAgent(
    HierarchicalRecurrentSACPricingAgent
):
    """Hierarchical RSAC with one shared period-level opponent encoder."""

    architecture = AgentArchitecture.OE_RSAC

    @property
    def _embedding_dimension(self) -> int:
        return int(self.profile.opponent_embedding_dimension)

    def __init__(
        self,
        profile: HierarchicalAgentProfileConfig,
        run_seed_bundle: RunSeedBundle,
        *,
        device: str | torch.device = "cpu",
    ) -> None:
        if profile.architecture is not AgentArchitecture.OE_RSAC:
            raise ValueError("Opponent-embedding agent needs oe_rsac profile")
        # Parent constructs controllers using the overridden embedding dimension.
        super().__init__(profile, run_seed_bundle, device=device)
        with _isolated_torch_initialization(
            run_seed_bundle.network_initialization_seed + 17
        ):
            self.opponent_encoder = SharedOpponentHistoryEncoder(
                input_dimension=27,
                hidden_dimension=int(profile.encoder_hidden_dimension),
                embedding_dimension=int(
                    profile.opponent_embedding_dimension
                ),
            ).to(self.device)
            self.target_opponent_encoder = copy.deepcopy(
                self.opponent_encoder
            ).to(self.device)
            self.opponent_predictor = OpponentControlPredictor(
                embedding_dimension=int(
                    profile.opponent_embedding_dimension
                ),
                hidden_dimension=int(profile.encoder_hidden_dimension),
            ).to(self.device)
        self.encoder_optimizer = torch.optim.Adam(
            list(self.opponent_encoder.parameters())
            + list(self.opponent_predictor.parameters()),
            lr=float(profile.encoder_learning_rate),
        )
        self._encoder_hidden: torch.Tensor | None = None

    def reset_recurrent_state(self) -> None:
        super().reset_recurrent_state()
        self._encoder_hidden = None

    def _online_embedding(self, pricing: np.ndarray) -> torch.Tensor:
        encoder_input = np.concatenate(
            [
                pricing,
                self._previous_effective_action,
                self._previous_opponent_controls,
                np.asarray([self._previous_reward], dtype=np.float32),
            ]
        )
        tensor = torch.as_tensor(
            encoder_input,
            dtype=torch.float32,
            device=self.device,
        ).reshape(1, 1, 27)
        with torch.no_grad():
            embedding, self._encoder_hidden = self.opponent_encoder(
                tensor, self._encoder_hidden
            )
        self._current_embedding = embedding[0, 0].detach()
        return self._current_embedding

    def current_opponent_embedding(self) -> np.ndarray | None:
        if self._current_embedding is None:
            return np.zeros(
                self._embedding_dimension, dtype=np.float32
            )
        return self._current_embedding.detach().cpu().numpy().copy()

    def _encoder_inputs(
        self,
        batch: Mapping[str, Any],
        *,
        next_inputs: bool,
    ) -> torch.Tensor:
        if next_inputs:
            values = (
                batch["next_observations"],
                batch["effective_actions"],
                batch["opponent_price_controls"],
                batch["rewards"],
            )
        else:
            values = (
                batch["observations"],
                batch["previous_effective_actions"],
                batch["previous_opponent_price_controls"],
                batch["previous_rewards"],
            )
        return torch.cat(
            [
                torch.as_tensor(
                    value, dtype=torch.float32, device=self.device
                )
                for value in values
            ],
            dim=-1,
        )

    def _price_embeddings(
        self,
        batch: Mapping[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.encoder_optimizer.zero_grad(set_to_none=True)
        embeddings, _ = self.opponent_encoder(
            self._encoder_inputs(batch, next_inputs=False)
        )
        with torch.no_grad():
            next_embeddings, _ = self.target_opponent_encoder(
                self._encoder_inputs(batch, next_inputs=True)
            )
        self._pending_encoder_embeddings = embeddings
        return embeddings, next_embeddings

    def _after_price_update(
        self,
        batch: Mapping[str, Any],
        metrics: dict[str, float],
    ) -> None:
        # Recompute to keep the auxiliary graph independent of critic backward.
        embeddings, _ = self.opponent_encoder(
            self._encoder_inputs(batch, next_inputs=False)
        )
        effective_actions = torch.as_tensor(
            batch["effective_actions"],
            dtype=torch.float32,
            device=self.device,
        )
        targets = torch.as_tensor(
            batch["opponent_price_controls"],
            dtype=torch.float32,
            device=self.device,
        )
        masks = torch.as_tensor(
            batch["loss_masks"],
            dtype=torch.float32,
            device=self.device,
        )
        predictions = self.opponent_predictor(
            embeddings, effective_actions
        )
        element_loss = F.smooth_l1_loss(
            predictions, targets, reduction="none"
        ).mean(dim=-1, keepdim=True)
        auxiliary_loss = (
            element_loss * masks
        ).sum() / masks.sum().clamp_min(1.0)
        (
            float(self.profile.auxiliary_loss_weight) * auxiliary_loss
        ).backward()
        encoder_gradient = nn.utils.clip_grad_norm_(
            list(self.opponent_encoder.parameters())
            + list(self.opponent_predictor.parameters()),
            self.profile.gradient_clip_norm,
        )
        self.encoder_optimizer.step()
        _soft_update_module(
            self.opponent_encoder,
            self.target_opponent_encoder,
            self.profile.tau,
        )
        absolute_error = (
            (predictions.detach() - targets).abs() * masks
        )
        denominator = masks.sum().clamp_min(1.0)
        metrics.update(
            {
                "opponent_auxiliary_loss": float(
                    auxiliary_loss.detach().cpu()
                ),
                "opponent_encoder_gradient_norm": float(
                    encoder_gradient.detach().cpu()
                ),
                "opponent_control_mae_uniform": float(
                    (absolute_error[..., 0:1].sum() / denominator)
                    .detach()
                    .cpu()
                ),
                "opponent_control_mae_bbp_new": float(
                    (absolute_error[..., 1:2].sum() / denominator)
                    .detach()
                    .cpu()
                ),
                "opponent_control_mae_bbp_premium": float(
                    (absolute_error[..., 2:3].sum() / denominator)
                    .detach()
                    .cpu()
                ),
            }
        )

    def _strategy_embeddings(
        self, batch: Mapping[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.as_tensor(
                batch["opponent_embeddings"],
                dtype=torch.float32,
                device=self.device,
            ),
            torch.as_tensor(
                batch["next_opponent_embeddings"],
                dtype=torch.float32,
                device=self.device,
            ),
        )

    def _controller_state_dict(self) -> dict[str, Any]:
        values = super()._controller_state_dict()
        values["opponent_encoder"] = self.opponent_encoder.state_dict()
        values["target_opponent_encoder"] = (
            self.target_opponent_encoder.state_dict()
        )
        values["opponent_predictor"] = self.opponent_predictor.state_dict()
        values["encoder_optimizer"] = self.encoder_optimizer.state_dict()
        return values

    def _load_controller_state_dict(
        self, values: Mapping[str, Any]
    ) -> None:
        super()._load_controller_state_dict(values)
        self.opponent_encoder.load_state_dict(values["opponent_encoder"])
        self.target_opponent_encoder.load_state_dict(
            values["target_opponent_encoder"]
        )
        self.opponent_predictor.load_state_dict(
            values["opponent_predictor"]
        )
        self.encoder_optimizer.load_state_dict(
            values["encoder_optimizer"]
        )

    def _online_state_dict(self) -> dict[str, Any]:
        values = super()._online_state_dict()
        values["encoder_hidden"] = (
            None
            if self._encoder_hidden is None
            else self._encoder_hidden.detach().cpu()
        )
        values["current_embedding"] = (
            None
            if self._current_embedding is None
            else self._current_embedding.detach().cpu()
        )
        return values

    def _load_online_state_dict(self, values: Mapping[str, Any]) -> None:
        super()._load_online_state_dict(values)
        self._encoder_hidden = (
            None
            if values["encoder_hidden"] is None
            else values["encoder_hidden"].to(self.device)
        )
        self._current_embedding = (
            None
            if values["current_embedding"] is None
            else values["current_embedding"].to(self.device)
        )

    def parameter_counts(self) -> dict[str, int]:
        result = super().parameter_counts()
        result["opponent_encoder_parameters"] = int(
            sum(
                parameter.numel()
                for module in (
                    self.opponent_encoder,
                    self.opponent_predictor,
                )
                for parameter in module.parameters()
            )
        )
        result["total_parameters"] = sum(
            value
            for name, value in result.items()
            if name != "total_parameters"
        )
        return result


class HierarchicalPricingAgentFactory:
    """Construct the architecture-specific hierarchical v2 agent."""

    @staticmethod
    def create(
        profile: HierarchicalAgentProfileConfig,
        run_seed_bundle: RunSeedBundle,
        *,
        device: str | torch.device = "cpu",
    ) -> BaseHierarchicalPricingAgent:
        if profile.architecture is AgentArchitecture.SAC:
            return HierarchicalSACPricingAgent(
                profile, run_seed_bundle, device=device
            )
        if profile.architecture is AgentArchitecture.RSAC:
            return HierarchicalRecurrentSACPricingAgent(
                profile, run_seed_bundle, device=device
            )
        return HierarchicalOpponentEmbeddingRecurrentSACPricingAgent(
            profile, run_seed_bundle, device=device
        )
