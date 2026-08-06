"""Hybrid Soft Actor-Critic for the universal pricing action contract.

This module is intentionally separate from :mod:`models.sac`.  The latter is
the legacy three-continuous-action implementation and remains checkpoint
compatible with the original experiments.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from env.pricing_contracts import (
    AgentArchitecture,
    PricingAction,
    PricingObservationCodec,
    PricingObservationFeature,
    PricingRegime,
)
from models.universal_pricing_replay import (
    UniversalPricingReplayBatch,
    UniversalPricingReplayBuffer,
)
from models.hybrid_sac_objective import HybridSACObjective


SAC_PRICING_CHECKPOINT_SCHEMA_VERSION = "sac_pricing_checkpoint_v1"
SAC_PRICING_ACTION_CONTRACT_VERSION = "pricing_action_v1"
SAC_PRICING_OBSERVATION_CONTRACT_VERSION = "pricing_observation_v1"


@dataclass(frozen=True)
class SACPricingAgentConfig:
    """Validated hyperparameters for the feed-forward hybrid SAC agent."""

    actor_hidden_dimensions: tuple[int, ...] = (256, 256)
    critic_hidden_dimensions: tuple[int, ...] = (256, 256)
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    entropy_learning_rate: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    initial_regime_temperature: float = 0.2
    initial_uniform_price_temperature: float = 0.2
    initial_bbp_price_temperature: float = 0.2
    regime_target_entropy_ratio: float = 0.98
    uniform_price_target_entropy: float = -1.0
    bbp_price_target_entropy: float = -2.0
    log_std_min: float = -5.0
    log_std_max: float = 2.0
    gradient_clip_norm: float = 10.0
    replay_capacity: int = 300_000
    batch_size: int = 256

    def __post_init__(self) -> None:
        for field_name in (
            "actor_hidden_dimensions",
            "critic_hidden_dimensions",
        ):
            raw_dimensions = getattr(self, field_name)
            if not isinstance(raw_dimensions, Sequence) or isinstance(
                raw_dimensions, (str, bytes)
            ):
                raise ValueError(f"{field_name} must be a sequence")
            dimensions = tuple(raw_dimensions)
            if not dimensions or any(
                not isinstance(value, int)
                or isinstance(value, bool)
                or value <= 0
                for value in dimensions
            ):
                raise ValueError(
                    f"{field_name} must contain positive integers"
                )
            object.__setattr__(self, field_name, dimensions)

        positive_fields = (
            "actor_learning_rate",
            "critic_learning_rate",
            "entropy_learning_rate",
            "initial_regime_temperature",
            "initial_uniform_price_temperature",
            "initial_bbp_price_temperature",
            "gradient_clip_norm",
        )
        for field_name in positive_fields:
            value = float(getattr(self, field_name))
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{field_name} must be finite and positive")
            object.__setattr__(self, field_name, value)

        for field_name in ("gamma", "tau", "regime_target_entropy_ratio"):
            value = float(getattr(self, field_name))
            if not np.isfinite(value) or not 0.0 < value <= 1.0:
                raise ValueError(f"{field_name} must be in (0, 1]")
            object.__setattr__(self, field_name, value)

        for field_name in (
            "uniform_price_target_entropy",
            "bbp_price_target_entropy",
            "log_std_min",
            "log_std_max",
        ):
            value = float(getattr(self, field_name))
            if not np.isfinite(value):
                raise ValueError(f"{field_name} must be finite")
            object.__setattr__(self, field_name, value)
        if self.uniform_price_target_entropy >= 0:
            raise ValueError("uniform_price_target_entropy must be negative")
        if self.bbp_price_target_entropy >= 0:
            raise ValueError("bbp_price_target_entropy must be negative")
        if self.log_std_min >= self.log_std_max:
            raise ValueError("log_std_min must be below log_std_max")

        for field_name in ("replay_capacity", "batch_size"):
            value = getattr(self, field_name)
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value <= 0
            ):
                raise ValueError(f"{field_name} must be a positive integer")
        if self.batch_size > self.replay_capacity:
            raise ValueError("batch_size cannot exceed replay_capacity")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["actor_hidden_dimensions"] = list(
            self.actor_hidden_dimensions
        )
        result["critic_hidden_dimensions"] = list(
            self.critic_hidden_dimensions
        )
        return result


class HybridPricingActionTensorCodec:
    """Canonicalize hybrid price actions before critic evaluation."""

    ACTION_DIMENSION = 5

    @staticmethod
    def uniform_actions(uniform_controls: torch.Tensor) -> torch.Tensor:
        if uniform_controls.ndim != 2 or uniform_controls.shape[-1] != 1:
            raise ValueError("uniform_controls must have shape (batch, 1)")
        zeros = torch.zeros_like(uniform_controls)
        ones = torch.ones_like(uniform_controls)
        return torch.cat([ones, zeros, uniform_controls, zeros, zeros], dim=-1)

    @staticmethod
    def bbp_actions(bbp_controls: torch.Tensor) -> torch.Tensor:
        if bbp_controls.ndim != 2 or bbp_controls.shape[-1] != 2:
            raise ValueError("bbp_controls must have shape (batch, 2)")
        zeros = torch.zeros_like(bbp_controls[:, :1])
        ones = torch.ones_like(zeros)
        return torch.cat([zeros, ones, zeros, bbp_controls], dim=-1)

    @classmethod
    def canonicalize_replay_actions(
        cls,
        replay_actions: torch.Tensor,
    ) -> torch.Tensor:
        if (
            replay_actions.ndim != 2
            or replay_actions.shape[-1] != cls.ACTION_DIMENSION
        ):
            raise ValueError("replay_actions must have shape (batch, 5)")
        regime = replay_actions[:, :2]
        if not torch.allclose(
            regime.sum(dim=-1),
            torch.ones(
                regime.shape[0],
                device=regime.device,
                dtype=regime.dtype,
            ),
        ) or torch.any((regime != 0) & (regime != 1)):
            raise ValueError("replay_actions must contain one-hot regimes")
        controls = replay_actions[:, 2:]
        if not torch.isfinite(replay_actions).all() or torch.any(
            (controls < -1) | (controls > 1)
        ):
            raise ValueError("replay_actions contain invalid controls")
        uniform = cls.uniform_actions(controls[:, :1])
        bbp = cls.bbp_actions(controls[:, 1:])
        return regime[:, :1] * uniform + regime[:, 1:] * bbp


@dataclass(frozen=True)
class HybridPricingPolicyOutput:
    """All actor heads in their stable, named representation."""

    regime_logits: torch.Tensor
    uniform_mean: torch.Tensor
    uniform_log_std: torch.Tensor
    bbp_mean: torch.Tensor
    bbp_log_std: torch.Tensor


def _orthogonal_initialize(module: nn.Module) -> None:
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            gain = (
                math.sqrt(2.0)
                if layer.out_features not in (1, 2)
                else 0.01
            )
            nn.init.orthogonal_(layer.weight, gain=gain)
            nn.init.zeros_(layer.bias)


def _build_mlp(
    input_dimension: int,
    hidden_dimensions: Sequence[int],
) -> tuple[nn.Sequential, int]:
    layers: list[nn.Module] = []
    previous_dimension = input_dimension
    for hidden_dimension in hidden_dimensions:
        layers.extend(
            [
                nn.Linear(previous_dimension, hidden_dimension),
                nn.ReLU(),
            ]
        )
        previous_dimension = hidden_dimension
    return nn.Sequential(*layers), previous_dimension


class SACPricingActor(nn.Module):
    """Categorical regime policy with conditional tanh-Gaussian price heads."""

    def __init__(
        self,
        observation_dimension: int,
        hidden_dimensions: Sequence[int],
        log_std_min: float,
        log_std_max: float,
    ) -> None:
        super().__init__()
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        self.trunk, output_dimension = _build_mlp(
            observation_dimension,
            hidden_dimensions,
        )
        self.regime_logits = nn.Linear(output_dimension, 2)
        self.uniform_mean = nn.Linear(output_dimension, 1)
        self.uniform_log_std = nn.Linear(output_dimension, 1)
        self.bbp_mean = nn.Linear(output_dimension, 2)
        self.bbp_log_std = nn.Linear(output_dimension, 2)
        _orthogonal_initialize(self)

    def forward(self, observations: torch.Tensor) -> HybridPricingPolicyOutput:
        features = self.trunk(observations)
        return HybridPricingPolicyOutput(
            regime_logits=self.regime_logits(features),
            uniform_mean=self.uniform_mean(features),
            uniform_log_std=torch.clamp(
                self.uniform_log_std(features),
                self.log_std_min,
                self.log_std_max,
            ),
            bbp_mean=self.bbp_mean(features),
            bbp_log_std=torch.clamp(
                self.bbp_log_std(features),
                self.log_std_min,
                self.log_std_max,
            ),
        )

    @staticmethod
    def sample_price_controls(
        mean: torch.Tensor,
        log_std: torch.Tensor,
        *,
        generator: torch.Generator,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if deterministic:
            pre_tanh = mean
        else:
            noise = torch.randn(
                mean.shape,
                dtype=mean.dtype,
                device=mean.device,
                generator=generator,
            )
            pre_tanh = mean + log_std.exp() * noise
        controls = torch.tanh(pre_tanh)
        normal_log_probability = (
            -0.5
            * (
                ((pre_tanh - mean) / log_std.exp()).pow(2)
                + 2.0 * log_std
                + math.log(2.0 * math.pi)
            )
        )
        correction = torch.log(
            torch.clamp(1.0 - controls.pow(2), min=1e-6)
        )
        log_probability = (
            normal_log_probability - correction
        ).sum(dim=-1, keepdim=True)
        return controls, log_probability


class SACPricingCritic(nn.Module):
    """One scalar Q network over observations and canonical hybrid actions."""

    def __init__(
        self,
        observation_dimension: int,
        hidden_dimensions: Sequence[int],
    ) -> None:
        super().__init__()
        trunk, output_dimension = _build_mlp(
            observation_dimension
            + HybridPricingActionTensorCodec.ACTION_DIMENSION,
            hidden_dimensions,
        )
        self.network = nn.Sequential(trunk, nn.Linear(output_dimension, 1))
        _orthogonal_initialize(self)

    def forward(
        self,
        observations: torch.Tensor,
        canonical_actions: torch.Tensor,
    ) -> torch.Tensor:
        return self.network(torch.cat([observations, canonical_actions], dim=-1))


@dataclass(frozen=True)
class SACPricingUpdateMetrics:
    """Scalar diagnostics produced by one optimization step."""

    critic_loss: float
    actor_loss: float
    regime_temperature_loss: float
    uniform_temperature_loss: float
    bbp_temperature_loss: float
    regime_temperature: float
    uniform_price_temperature: float
    bbp_price_temperature: float
    target_q_mean: float
    q1_mean: float
    q2_mean: float
    decision_fraction: float
    critic_gradient_norm: float
    actor_gradient_norm: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


class SACPricingAgent:
    """Feed-forward hybrid SAC implementation for ``universal_pricing_v1``."""

    OBSERVATION_DIMENSION = PricingObservationCodec.FEATURE_COUNT
    OWN_REGIME_INDEX = PricingObservationFeature.OWN_REGIME.index
    DECISION_ALLOWED_INDEX = (
        PricingObservationFeature.REGIME_DECISION_ALLOWED.index
    )

    def __init__(
        self,
        config: SACPricingAgentConfig,
        *,
        network_initialization_seed: int,
        exploration_seed: int,
        torch_cpu_seed: int,
        torch_cuda_seed: int,
        device: str | torch.device = "cpu",
    ) -> None:
        if not isinstance(config, SACPricingAgentConfig):
            raise TypeError("config must be SACPricingAgentConfig")
        self.config = config
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available")

        # Module constructors normally consume global Torch randomness.  The
        # fork restores that state after deterministic local construction.
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(
                int(network_initialization_seed)
            )
            self.actor = SACPricingActor(
                self.OBSERVATION_DIMENSION,
                config.actor_hidden_dimensions,
                config.log_std_min,
                config.log_std_max,
            )
            self.critic_1 = SACPricingCritic(
                self.OBSERVATION_DIMENSION,
                config.critic_hidden_dimensions,
            )
            self.critic_2 = SACPricingCritic(
                self.OBSERVATION_DIMENSION,
                config.critic_hidden_dimensions,
            )
            self.target_critic_1 = SACPricingCritic(
                self.OBSERVATION_DIMENSION,
                config.critic_hidden_dimensions,
            )
            self.target_critic_2 = SACPricingCritic(
                self.OBSERVATION_DIMENSION,
                config.critic_hidden_dimensions,
            )

        self.actor.to(self.device)
        self.critic_1.to(self.device)
        self.critic_2.to(self.device)
        self.target_critic_1.to(self.device)
        self.target_critic_2.to(self.device)
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        self.target_critic_1.requires_grad_(False)
        self.target_critic_2.requires_grad_(False)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=config.actor_learning_rate,
        )
        critic_parameters = list(self.critic_1.parameters()) + list(
            self.critic_2.parameters()
        )
        self.critic_optimizer = torch.optim.Adam(
            critic_parameters,
            lr=config.critic_learning_rate,
        )

        self.log_regime_temperature = nn.Parameter(
            torch.tensor(
                math.log(config.initial_regime_temperature),
                dtype=torch.float32,
                device=self.device,
            )
        )
        self.log_uniform_price_temperature = nn.Parameter(
            torch.tensor(
                math.log(config.initial_uniform_price_temperature),
                dtype=torch.float32,
                device=self.device,
            )
        )
        self.log_bbp_price_temperature = nn.Parameter(
            torch.tensor(
                math.log(config.initial_bbp_price_temperature),
                dtype=torch.float32,
                device=self.device,
            )
        )
        self.regime_temperature_optimizer = torch.optim.Adam(
            [self.log_regime_temperature],
            lr=config.entropy_learning_rate,
        )
        self.uniform_temperature_optimizer = torch.optim.Adam(
            [self.log_uniform_price_temperature],
            lr=config.entropy_learning_rate,
        )
        self.bbp_temperature_optimizer = torch.optim.Adam(
            [self.log_bbp_price_temperature],
            lr=config.entropy_learning_rate,
        )

        generator_device = self.device.type
        self._exploration_generator = torch.Generator(device=generator_device)
        self._training_generator = torch.Generator(device=generator_device)
        self._exploration_generator.manual_seed(int(exploration_seed))
        training_seed = (
            torch_cuda_seed
            if self.device.type == "cuda"
            else torch_cpu_seed
        )
        self._training_generator.manual_seed(int(training_seed))
        self.environment_steps = 0
        self.update_steps = 0
        self._last_policy_diagnostics: dict[str, float] = {}

    @property
    def regime_temperature(self) -> torch.Tensor:
        return self.log_regime_temperature.exp()

    @property
    def uniform_price_temperature(self) -> torch.Tensor:
        return self.log_uniform_price_temperature.exp()

    @property
    def bbp_price_temperature(self) -> torch.Tensor:
        return self.log_bbp_price_temperature.exp()

    @classmethod
    def from_run_seed_bundle(
        cls,
        config: SACPricingAgentConfig,
        run_seed_bundle: Any,
        *,
        device: str | torch.device = "cpu",
    ) -> "SACPricingAgent":
        required_fields = (
            "network_initialization_seed",
            "exploration_seed",
            "torch_cpu_seed",
            "torch_cuda_seed",
        )
        missing = [
            field_name
            for field_name in required_fields
            if not hasattr(run_seed_bundle, field_name)
        ]
        if missing:
            raise TypeError(
                "run_seed_bundle is missing: " + ", ".join(missing)
            )
        return cls(
            config,
            network_initialization_seed=(
                run_seed_bundle.network_initialization_seed
            ),
            exploration_seed=run_seed_bundle.exploration_seed,
            torch_cpu_seed=run_seed_bundle.torch_cpu_seed,
            torch_cuda_seed=run_seed_bundle.torch_cuda_seed,
            device=device,
        )

    @staticmethod
    def _decision_allowed(observations: torch.Tensor) -> torch.Tensor:
        return (
            observations[
                :, SACPricingAgent.DECISION_ALLOWED_INDEX : (
                    SACPricingAgent.DECISION_ALLOWED_INDEX + 1
                )
            ]
            > 0.0
        ).to(observations.dtype)

    @staticmethod
    def _current_regime_one_hot(
        observations: torch.Tensor,
    ) -> torch.Tensor:
        bbp = (
            observations[
                :, SACPricingAgent.OWN_REGIME_INDEX : (
                    SACPricingAgent.OWN_REGIME_INDEX + 1
                )
            ]
            > 0.0
        ).to(observations.dtype)
        return torch.cat([1.0 - bbp, bbp], dim=-1)

    def select_action(
        self,
        observation: np.ndarray,
        *,
        deterministic: bool = False,
    ) -> PricingAction:
        validated = PricingObservationCodec.validate_vector(observation)
        observations = torch.as_tensor(
            validated,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        with torch.no_grad():
            output = self.actor(observations)
            regime_probabilities = F.softmax(output.regime_logits, dim=-1)
            decision_allowed = bool(
                self._decision_allowed(observations).item()
            )
            if decision_allowed:
                if deterministic:
                    regime_index = int(
                        torch.argmax(regime_probabilities, dim=-1).item()
                    )
                else:
                    regime_index = int(
                        torch.multinomial(
                            regime_probabilities,
                            num_samples=1,
                            generator=self._exploration_generator,
                        ).item()
                    )
            else:
                regime_index = int(
                    torch.argmax(
                        self._current_regime_one_hot(observations),
                        dim=-1,
                    ).item()
                )

            if regime_index == int(PricingRegime.UNIFORM):
                controls, _ = self.actor.sample_price_controls(
                    output.uniform_mean,
                    output.uniform_log_std,
                    generator=self._exploration_generator,
                    deterministic=deterministic,
                )
                uniform_control = float(controls[0, 0].cpu())
                bbp_new_control = 0.0
                bbp_premium_control = 0.0
            else:
                controls, _ = self.actor.sample_price_controls(
                    output.bbp_mean,
                    output.bbp_log_std,
                    generator=self._exploration_generator,
                    deterministic=deterministic,
                )
                uniform_control = 0.0
                bbp_new_control = float(controls[0, 0].cpu())
                bbp_premium_control = float(controls[0, 1].cpu())

        self.environment_steps += 1
        self._last_policy_diagnostics = {
            "decision_allowed": float(decision_allowed),
            "selected_regime": float(regime_index),
            "uniform_regime_probability": float(
                regime_probabilities[0, 0].cpu()
            ),
            "bbp_regime_probability": float(
                regime_probabilities[0, 1].cpu()
            ),
            "uniform_mean": float(output.uniform_mean[0, 0].cpu()),
            "uniform_log_std": float(
                output.uniform_log_std[0, 0].cpu()
            ),
            "bbp_new_mean": float(output.bbp_mean[0, 0].cpu()),
            "bbp_premium_mean": float(output.bbp_mean[0, 1].cpu()),
            "regime_temperature": float(
                self.regime_temperature.detach().cpu()
            ),
            "uniform_price_temperature": float(
                self.uniform_price_temperature.detach().cpu()
            ),
            "bbp_price_temperature": float(
                self.bbp_price_temperature.detach().cpu()
            ),
        }
        return PricingAction(
            regime=PricingRegime(regime_index),
            uniform_control=uniform_control,
            bbp_new_control=bbp_new_control,
            bbp_premium_control=bbp_premium_control,
        )

    def _sample_policy_branches(
        self,
        observations: torch.Tensor,
    ) -> tuple[
        HybridPricingPolicyOutput,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        output = self.actor(observations)
        uniform_controls, uniform_log_probability = (
            self.actor.sample_price_controls(
                output.uniform_mean,
                output.uniform_log_std,
                generator=self._training_generator,
            )
        )
        bbp_controls, bbp_log_probability = (
            self.actor.sample_price_controls(
                output.bbp_mean,
                output.bbp_log_std,
                generator=self._training_generator,
            )
        )
        return (
            output,
            HybridPricingActionTensorCodec.uniform_actions(
                uniform_controls
            ),
            uniform_log_probability,
            HybridPricingActionTensorCodec.bbp_actions(bbp_controls),
            bbp_log_probability,
        )

    def _hybrid_soft_value(
        self,
        observations: torch.Tensor,
        critic_1: SACPricingCritic,
        critic_2: SACPricingCritic,
    ) -> torch.Tensor:
        (
            output,
            uniform_actions,
            uniform_log_probability,
            bbp_actions,
            bbp_log_probability,
        ) = self._sample_policy_branches(observations)
        probabilities = F.softmax(output.regime_logits, dim=-1)
        log_probabilities = F.log_softmax(output.regime_logits, dim=-1)
        uniform_q = torch.minimum(
            critic_1(observations, uniform_actions),
            critic_2(observations, uniform_actions),
        )
        bbp_q = torch.minimum(
            critic_1(observations, bbp_actions),
            critic_2(observations, bbp_actions),
        )
        uniform_soft_q = (
            uniform_q
            - self.uniform_price_temperature.detach()
            * uniform_log_probability
        )
        bbp_soft_q = (
            bbp_q
            - self.bbp_price_temperature.detach() * bbp_log_probability
        )
        return HybridSACObjective.soft_value(
            regime_probabilities=probabilities,
            regime_log_probabilities=log_probabilities,
            uniform_soft_q=uniform_soft_q,
            bbp_soft_q=bbp_soft_q,
            current_regime_one_hot=self._current_regime_one_hot(
                observations
            ),
            regime_decision_masks=self._decision_allowed(observations),
            regime_temperature=self.regime_temperature.detach(),
        )

    def compute_critic_target(
        self,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        """Compute detached Bellman targets using exact regime marginalization."""

        with torch.no_grad():
            next_value = self._hybrid_soft_value(
                next_observations,
                self.target_critic_1,
                self.target_critic_2,
            )
            return rewards + self.config.gamma * (1.0 - dones) * next_value

    def compute_actor_loss(
        self,
        observations: torch.Tensor,
        regime_decision_masks: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the masked hybrid policy objective."""

        (
            output,
            uniform_actions,
            uniform_log_probability,
            bbp_actions,
            bbp_log_probability,
        ) = self._sample_policy_branches(observations)
        probabilities = F.softmax(output.regime_logits, dim=-1)
        log_probabilities = F.log_softmax(output.regime_logits, dim=-1)
        uniform_q = torch.minimum(
            self.critic_1(observations, uniform_actions),
            self.critic_2(observations, uniform_actions),
        )
        bbp_q = torch.minimum(
            self.critic_1(observations, bbp_actions),
            self.critic_2(observations, bbp_actions),
        )
        uniform_objective = (
            self.uniform_price_temperature.detach()
            * uniform_log_probability
            - uniform_q
        )
        bbp_objective = (
            self.bbp_price_temperature.detach() * bbp_log_probability
            - bbp_q
        )
        return HybridSACObjective.actor_objective(
            regime_probabilities=probabilities,
            regime_log_probabilities=log_probabilities,
            uniform_branch_objective=uniform_objective,
            bbp_branch_objective=bbp_objective,
            current_regime_one_hot=self._current_regime_one_hot(
                observations
            ),
            regime_decision_masks=regime_decision_masks,
            regime_temperature=self.regime_temperature.detach(),
        ).mean()

    def _temperature_losses(
        self,
        observations: torch.Tensor,
        regime_decision_masks: torch.Tensor,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        with torch.no_grad():
            output = self.actor(observations)
            probabilities = F.softmax(output.regime_logits, dim=-1)
            log_probabilities = F.log_softmax(output.regime_logits, dim=-1)
            _, uniform_log_probability = self.actor.sample_price_controls(
                output.uniform_mean,
                output.uniform_log_std,
                generator=self._training_generator,
            )
            _, bbp_log_probability = self.actor.sample_price_controls(
                output.bbp_mean,
                output.bbp_log_std,
                generator=self._training_generator,
            )
            current_regime = self._current_regime_one_hot(observations)
            eligible = regime_decision_masks
            uniform_weights = (
                eligible * probabilities[:, :1]
                + (1.0 - eligible) * current_regime[:, :1]
            )
            bbp_weights = (
                eligible * probabilities[:, 1:]
                + (1.0 - eligible) * current_regime[:, 1:]
            )
            expected_regime_log_probability = (
                probabilities * log_probabilities
            ).sum(dim=-1, keepdim=True)

        regime_weight = eligible.sum()
        regime_loss = None
        if float(regime_weight) > 0.0:
            target = -(
                self.config.regime_target_entropy_ratio * math.log(2.0)
            )
            error = target - expected_regime_log_probability.detach()
            regime_loss = (
                self.log_regime_temperature
                * (eligible * error).sum()
                / regime_weight
            )

        uniform_weight = uniform_weights.sum()
        uniform_loss = None
        if float(uniform_weight) > 0.0:
            error = (
                self.config.uniform_price_target_entropy
                - uniform_log_probability.detach()
            )
            uniform_loss = (
                self.log_uniform_price_temperature
                * (uniform_weights * error).sum()
                / uniform_weight
            )

        bbp_weight = bbp_weights.sum()
        bbp_loss = None
        if float(bbp_weight) > 0.0:
            error = (
                self.config.bbp_price_target_entropy
                - bbp_log_probability.detach()
            )
            bbp_loss = (
                self.log_bbp_price_temperature
                * (bbp_weights * error).sum()
                / bbp_weight
            )
        return regime_loss, uniform_loss, bbp_loss

    @staticmethod
    def _mapping_to_tensors(
        replay_batch: Mapping[str, Any] | UniversalPricingReplayBatch,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        values = (
            replay_batch.as_mapping()
            if isinstance(replay_batch, UniversalPricingReplayBatch)
            else dict(replay_batch)
        )
        required = {
            "observations",
            "effective_actions",
            "rewards",
            "next_observations",
            "dones",
            "regime_decision_masks",
            "opponent_price_controls",
        }
        missing = required - set(values)
        if missing:
            raise ValueError(
                "Replay batch is missing: " + ", ".join(sorted(missing))
            )
        tensors = {
            name: torch.as_tensor(value, dtype=torch.float32, device=device)
            for name, value in values.items()
            if name in required
        }
        batch_size = tensors["observations"].shape[0]
        expected_shapes = {
            "observations": (batch_size, 18),
            "effective_actions": (batch_size, 5),
            "rewards": (batch_size, 1),
            "next_observations": (batch_size, 18),
            "dones": (batch_size, 1),
            "regime_decision_masks": (batch_size, 1),
            "opponent_price_controls": (batch_size, 3),
        }
        for name, expected_shape in expected_shapes.items():
            if tuple(tensors[name].shape) != expected_shape:
                raise ValueError(f"{name} must have shape {expected_shape}")
            if not torch.isfinite(tensors[name]).all():
                raise ValueError(f"{name} must contain only finite values")
        for name in ("dones", "regime_decision_masks"):
            if torch.any(
                (tensors[name] != 0.0) & (tensors[name] != 1.0)
            ):
                raise ValueError(f"{name} must contain only 0 and 1")
        return tensors

    def update(
        self,
        replay_batch: Mapping[str, Any] | UniversalPricingReplayBatch,
    ) -> Mapping[str, float]:
        tensors = self._mapping_to_tensors(replay_batch, self.device)
        observations = tensors["observations"]
        canonical_actions = (
            HybridPricingActionTensorCodec.canonicalize_replay_actions(
                tensors["effective_actions"]
            )
        )
        targets = self.compute_critic_target(
            tensors["rewards"],
            tensors["next_observations"],
            tensors["dones"],
        )

        q1 = self.critic_1(observations, canonical_actions)
        q2 = self.critic_2(observations, canonical_actions)
        critic_loss = F.mse_loss(q1, targets) + F.mse_loss(q2, targets)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_gradient_norm = nn.utils.clip_grad_norm_(
            list(self.critic_1.parameters())
            + list(self.critic_2.parameters()),
            self.config.gradient_clip_norm,
        )
        self.critic_optimizer.step()

        self.critic_1.requires_grad_(False)
        self.critic_2.requires_grad_(False)
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss = self.compute_actor_loss(
            observations,
            tensors["regime_decision_masks"],
        )
        actor_loss.backward()
        actor_gradient_norm = nn.utils.clip_grad_norm_(
            self.actor.parameters(),
            self.config.gradient_clip_norm,
        )
        self.actor_optimizer.step()
        self.critic_1.requires_grad_(True)
        self.critic_2.requires_grad_(True)

        temperature_losses = self._temperature_losses(
            observations,
            tensors["regime_decision_masks"],
        )
        temperature_optimizers = (
            self.regime_temperature_optimizer,
            self.uniform_temperature_optimizer,
            self.bbp_temperature_optimizer,
        )
        numeric_temperature_losses: list[float] = []
        for loss, optimizer in zip(
            temperature_losses,
            temperature_optimizers,
        ):
            if loss is None:
                numeric_temperature_losses.append(0.0)
                continue
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            numeric_temperature_losses.append(float(loss.detach().cpu()))

        self._soft_update_targets()
        self.update_steps += 1
        metrics = SACPricingUpdateMetrics(
            critic_loss=float(critic_loss.detach().cpu()),
            actor_loss=float(actor_loss.detach().cpu()),
            regime_temperature_loss=numeric_temperature_losses[0],
            uniform_temperature_loss=numeric_temperature_losses[1],
            bbp_temperature_loss=numeric_temperature_losses[2],
            regime_temperature=float(
                self.regime_temperature.detach().cpu()
            ),
            uniform_price_temperature=float(
                self.uniform_price_temperature.detach().cpu()
            ),
            bbp_price_temperature=float(
                self.bbp_price_temperature.detach().cpu()
            ),
            target_q_mean=float(targets.mean().detach().cpu()),
            q1_mean=float(q1.mean().detach().cpu()),
            q2_mean=float(q2.mean().detach().cpu()),
            decision_fraction=float(
                tensors["regime_decision_masks"].mean().detach().cpu()
            ),
            critic_gradient_norm=float(
                critic_gradient_norm.detach().cpu()
            ),
            actor_gradient_norm=float(
                actor_gradient_norm.detach().cpu()
            ),
        )
        return metrics.to_dict()

    def _soft_update_targets(self) -> None:
        with torch.no_grad():
            for source, target in (
                (self.critic_1, self.target_critic_1),
                (self.critic_2, self.target_critic_2),
            ):
                for source_parameter, target_parameter in zip(
                    source.parameters(),
                    target.parameters(),
                ):
                    target_parameter.mul_(1.0 - self.config.tau)
                    target_parameter.add_(
                        source_parameter,
                        alpha=self.config.tau,
                    )

    def reset_recurrent_state(self) -> None:
        """Feed-forward SAC has no recurrent state."""

    def observe_transition(self, transition: Any) -> None:
        """Feed-forward SAC does not maintain online transition context."""

    def policy_diagnostics(self) -> Mapping[str, float]:
        diagnostics = dict(self._last_policy_diagnostics)
        diagnostics.update(
            {
                "parameter_count": float(
                    sum(
                        parameter.numel()
                        for module in (
                            self.actor,
                            self.critic_1,
                            self.critic_2,
                        )
                        for parameter in module.parameters()
                    )
                    + self.log_regime_temperature.numel()
                    + self.log_uniform_price_temperature.numel()
                    + self.log_bbp_price_temperature.numel()
                ),
                "environment_steps": float(self.environment_steps),
                "update_steps": float(self.update_steps),
            }
        )
        return diagnostics

    def _checkpoint_payload(self) -> dict[str, Any]:
        return {
            "schema_version": SAC_PRICING_CHECKPOINT_SCHEMA_VERSION,
            "protocol_version": "universal_pricing_v1",
            "action_contract_version": SAC_PRICING_ACTION_CONTRACT_VERSION,
            "observation_contract_version": (
                SAC_PRICING_OBSERVATION_CONTRACT_VERSION
            ),
            "architecture": AgentArchitecture.SAC.value,
            "config": self.config.to_dict(),
            "actor": self.actor.state_dict(),
            "critic_1": self.critic_1.state_dict(),
            "critic_2": self.critic_2.state_dict(),
            "target_critic_1": self.target_critic_1.state_dict(),
            "target_critic_2": self.target_critic_2.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "log_regime_temperature": (
                self.log_regime_temperature.detach().cpu()
            ),
            "log_uniform_price_temperature": (
                self.log_uniform_price_temperature.detach().cpu()
            ),
            "log_bbp_price_temperature": (
                self.log_bbp_price_temperature.detach().cpu()
            ),
            "regime_temperature_optimizer": (
                self.regime_temperature_optimizer.state_dict()
            ),
            "uniform_temperature_optimizer": (
                self.uniform_temperature_optimizer.state_dict()
            ),
            "bbp_temperature_optimizer": (
                self.bbp_temperature_optimizer.state_dict()
            ),
            "environment_steps": self.environment_steps,
            "update_steps": self.update_steps,
            "exploration_generator_state": (
                self._exploration_generator.get_state().cpu()
            ),
            "training_generator_state": (
                self._training_generator.get_state().cpu()
            ),
        }

    def save(self, checkpoint_path: str | Path) -> None:
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_name(f".{path.name}.temporary")
        try:
            torch.save(self._checkpoint_payload(), temporary_path)
            os.replace(temporary_path, path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()

    def load(self, checkpoint_path: str | Path) -> None:
        path = Path(checkpoint_path)
        try:
            payload = torch.load(
                path,
                map_location=self.device,
                weights_only=False,
            )
        except TypeError:
            payload = torch.load(path, map_location=self.device)
        expected_metadata = {
            "schema_version": SAC_PRICING_CHECKPOINT_SCHEMA_VERSION,
            "protocol_version": "universal_pricing_v1",
            "action_contract_version": SAC_PRICING_ACTION_CONTRACT_VERSION,
            "observation_contract_version": (
                SAC_PRICING_OBSERVATION_CONTRACT_VERSION
            ),
            "architecture": AgentArchitecture.SAC.value,
            "config": self.config.to_dict(),
        }
        for field_name, expected_value in expected_metadata.items():
            if payload.get(field_name) != expected_value:
                raise ValueError(
                    f"Incompatible SAC pricing checkpoint {field_name}"
                )

        self.actor.load_state_dict(payload["actor"])
        self.critic_1.load_state_dict(payload["critic_1"])
        self.critic_2.load_state_dict(payload["critic_2"])
        self.target_critic_1.load_state_dict(payload["target_critic_1"])
        self.target_critic_2.load_state_dict(payload["target_critic_2"])
        self.actor_optimizer.load_state_dict(payload["actor_optimizer"])
        self.critic_optimizer.load_state_dict(payload["critic_optimizer"])
        with torch.no_grad():
            self.log_regime_temperature.copy_(
                payload["log_regime_temperature"]
            )
            self.log_uniform_price_temperature.copy_(
                payload["log_uniform_price_temperature"]
            )
            self.log_bbp_price_temperature.copy_(
                payload["log_bbp_price_temperature"]
            )
        self.regime_temperature_optimizer.load_state_dict(
            payload["regime_temperature_optimizer"]
        )
        self.uniform_temperature_optimizer.load_state_dict(
            payload["uniform_temperature_optimizer"]
        )
        self.bbp_temperature_optimizer.load_state_dict(
            payload["bbp_temperature_optimizer"]
        )
        self.environment_steps = int(payload["environment_steps"])
        self.update_steps = int(payload["update_steps"])
        self._exploration_generator.set_state(
            payload["exploration_generator_state"].cpu()
        )
        self._training_generator.set_state(
            payload["training_generator_state"].cpu()
        )


class SACPricingAgentFactory:
    """Construct SAC and its replay buffer from one committed run seed bundle."""

    @staticmethod
    def create_agent(
        config: SACPricingAgentConfig,
        run_seed_bundle: Any,
        *,
        device: str | torch.device = "cpu",
    ) -> SACPricingAgent:
        return SACPricingAgent.from_run_seed_bundle(
            config,
            run_seed_bundle,
            device=device,
        )

    @staticmethod
    def create_replay_buffer(
        config: SACPricingAgentConfig,
        run_seed_bundle: Any,
    ) -> UniversalPricingReplayBuffer:
        if not hasattr(run_seed_bundle, "replay_sampling_seed"):
            raise TypeError(
                "run_seed_bundle is missing replay_sampling_seed"
            )
        return UniversalPricingReplayBuffer(
            capacity=config.replay_capacity,
            replay_sampling_seed=run_seed_bundle.replay_sampling_seed,
        )
