"""Universal recurrent hybrid SAC agents, with and without opponent embedding."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import os
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
    PricingObservationFeature,
    PricingRegime,
)
from models.sac_pricing import HybridPricingPolicyOutput, SACPricingActor
from models.hybrid_sac_objective import HybridSACObjective
from models.universal_pricing_replay import UniversalPricingTransition
from models.universal_pricing_sequence_replay import UniversalPricingSequenceBatch


RECURRENT_PRICING_CHECKPOINT_VERSION = "recurrent_pricing_checkpoint_v1"


@dataclass(frozen=True)
class RecurrentSACPricingAgentConfig:
    """Validated shared configuration for plain and opponent-embedding RSAC."""

    architecture: AgentArchitecture
    learning_sequence_length: int = 16
    burn_in_length: int = 16
    episode_replay_capacity: int = 1000
    batch_size: int = 64
    actor_hidden_dimension: int = 128
    critic_hidden_dimension: int = 128
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
    opponent_embedding_dimension: int | None = None
    encoder_hidden_dimension: int | None = None
    encoder_learning_rate: float | None = None
    auxiliary_loss_weight: float | None = None

    def __post_init__(self) -> None:
        architecture = AgentArchitecture(self.architecture)
        if architecture not in {AgentArchitecture.RSAC, AgentArchitecture.OE_RSAC}:
            raise ValueError("Recurrent config architecture must be rsac or oe_rsac")
        object.__setattr__(self, "architecture", architecture)
        for name in (
            "learning_sequence_length",
            "burn_in_length",
            "episode_replay_capacity",
            "batch_size",
            "actor_hidden_dimension",
            "critic_hidden_dimension",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "actor_learning_rate",
            "critic_learning_rate",
            "entropy_learning_rate",
            "initial_regime_temperature",
            "initial_uniform_price_temperature",
            "initial_bbp_price_temperature",
            "gradient_clip_norm",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)
        for name in ("gamma", "tau", "regime_target_entropy_ratio"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"{name} must be in (0, 1]")
            object.__setattr__(self, name, value)
        if not self.log_std_min < self.log_std_max:
            raise ValueError("log_std_min must be below log_std_max")
        if self.uniform_price_target_entropy >= 0 or self.bbp_price_target_entropy >= 0:
            raise ValueError("Continuous target entropies must be negative")

        encoder_names = (
            "opponent_embedding_dimension",
            "encoder_hidden_dimension",
            "encoder_learning_rate",
            "auxiliary_loss_weight",
        )
        if architecture is AgentArchitecture.RSAC:
            if any(getattr(self, name) is not None for name in encoder_names):
                raise ValueError("Plain rsac rejects opponent encoder fields")
        else:
            for name in encoder_names[:2]:
                value = getattr(self, name)
                if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                    raise ValueError(f"oe_rsac requires positive {name}")
            if self.encoder_learning_rate is None or self.encoder_learning_rate <= 0:
                raise ValueError("oe_rsac requires positive encoder_learning_rate")
            if self.auxiliary_loss_weight is None or self.auxiliary_loss_weight < 0:
                raise ValueError("oe_rsac requires nonnegative auxiliary_loss_weight")

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["architecture"] = self.architecture.value
        return {name: value for name, value in values.items() if value is not None}


@dataclass(frozen=True)
class RecurrentPricingPolicyOutput:
    """Hybrid policy heads and recurrent hidden state."""

    policy: HybridPricingPolicyOutput
    next_hidden: torch.Tensor


def _initialize(module: nn.Module) -> None:
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            nn.init.orthogonal_(layer.weight, gain=0.01)
            nn.init.zeros_(layer.bias)
        elif isinstance(layer, nn.GRU):
            for name, parameter in layer.named_parameters():
                if "weight" in name:
                    nn.init.orthogonal_(parameter)
                else:
                    nn.init.zeros_(parameter)


class RecurrentPricingActor(nn.Module):
    """GRU policy over market observation and previous own interaction."""

    def __init__(
        self,
        input_dimension: int,
        hidden_dimension: int,
        log_std_min: float,
        log_std_max: float,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(input_dimension, hidden_dimension, batch_first=True)
        self.regime_logits = nn.Linear(hidden_dimension, 2)
        self.uniform_mean = nn.Linear(hidden_dimension, 1)
        self.uniform_log_std = nn.Linear(hidden_dimension, 1)
        self.bbp_mean = nn.Linear(hidden_dimension, 2)
        self.bbp_log_std = nn.Linear(hidden_dimension, 2)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        _initialize(self)

    def forward(
        self,
        inputs: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> RecurrentPricingPolicyOutput:
        output, next_hidden = self.gru(inputs, hidden)
        return RecurrentPricingPolicyOutput(
            policy=HybridPricingPolicyOutput(
                regime_logits=self.regime_logits(output),
                uniform_mean=self.uniform_mean(output),
                uniform_log_std=self.uniform_log_std(output).clamp(
                    self.log_std_min, self.log_std_max
                ),
                bbp_mean=self.bbp_mean(output),
                bbp_log_std=self.bbp_log_std(output).clamp(
                    self.log_std_min, self.log_std_max
                ),
            ),
            next_hidden=next_hidden,
        )


class RecurrentPricingCritic(nn.Module):
    """Action-independent recurrent belief followed by a candidate-action Q head."""

    def __init__(self, input_dimension: int, hidden_dimension: int) -> None:
        super().__init__()
        self.gru = nn.GRU(input_dimension, hidden_dimension, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dimension + 5, hidden_dimension),
            nn.ReLU(),
            nn.Linear(hidden_dimension, 1),
        )
        _initialize(self)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.gru(inputs)[0]

    def evaluate(
        self,
        encoded_history: torch.Tensor,
        canonical_actions: torch.Tensor,
    ) -> torch.Tensor:
        return self.head(torch.cat([encoded_history, canonical_actions], dim=-1))


class OpponentHistoryEncoder(nn.Module):
    """Encode pre-decision opponent history and predict current controls."""

    def __init__(
        self,
        input_dimension: int,
        hidden_dimension: int,
        embedding_dimension: int,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(input_dimension, hidden_dimension, batch_first=True)
        self.embedding_head = nn.Linear(hidden_dimension, embedding_dimension)
        self.prediction_head = nn.Sequential(
            nn.Linear(embedding_dimension + 5, hidden_dimension),
            nn.ReLU(),
            nn.Linear(hidden_dimension, 3),
            nn.Tanh(),
        )
        _initialize(self)

    def forward(
        self,
        inputs: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output, next_hidden = self.gru(inputs, hidden)
        return self.embedding_head(output), next_hidden

    def predict(
        self,
        embeddings: torch.Tensor,
        effective_actions: torch.Tensor,
    ) -> torch.Tensor:
        return self.prediction_head(
            torch.cat([embeddings, effective_actions], dim=-1)
        )


def _canonical_actions(
    uniform_controls: torch.Tensor,
    bbp_controls: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    zeros_u = torch.zeros_like(uniform_controls)
    ones_u = torch.ones_like(uniform_controls)
    uniform = torch.cat(
        [ones_u, zeros_u, uniform_controls, zeros_u, zeros_u], dim=-1
    )
    zeros_b = torch.zeros_like(bbp_controls[..., :1])
    ones_b = torch.ones_like(zeros_b)
    bbp = torch.cat([zeros_b, ones_b, zeros_b, bbp_controls], dim=-1)
    return uniform, bbp


def _canonicalize_replay(actions: torch.Tensor) -> torch.Tensor:
    uniform, bbp = _canonical_actions(actions[..., 2:3], actions[..., 3:5])
    return actions[..., :1] * uniform + actions[..., 1:2] * bbp


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


class _UniversalRecurrentSACPricingAgent:
    """Shared implementation; concrete public subclasses freeze architecture."""

    OWN_REGIME_INDEX = PricingObservationFeature.OWN_REGIME.index
    DECISION_INDEX = PricingObservationFeature.REGIME_DECISION_ALLOWED.index

    def __init__(
        self,
        config: RecurrentSACPricingAgentConfig,
        *,
        network_initialization_seed: int,
        exploration_seed: int,
        torch_cpu_seed: int,
        torch_cuda_seed: int,
        device: str | torch.device = "cpu",
    ) -> None:
        self.config = config
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable")
        self.uses_opponent_encoder = (
            config.architecture is AgentArchitecture.OE_RSAC
        )
        embedding_dimension = (
            int(config.opponent_embedding_dimension)
            if self.uses_opponent_encoder
            else 0
        )
        actor_input = 18 + 5 + 1 + embedding_dimension
        critic_input = actor_input
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(int(network_initialization_seed))
            self.actor = RecurrentPricingActor(
                actor_input,
                config.actor_hidden_dimension,
                config.log_std_min,
                config.log_std_max,
            )
            self.critic_1 = RecurrentPricingCritic(
                critic_input, config.critic_hidden_dimension
            )
            self.critic_2 = RecurrentPricingCritic(
                critic_input, config.critic_hidden_dimension
            )
            self.target_critic_1 = RecurrentPricingCritic(
                critic_input, config.critic_hidden_dimension
            )
            self.target_critic_2 = RecurrentPricingCritic(
                critic_input, config.critic_hidden_dimension
            )
            if self.uses_opponent_encoder:
                self.opponent_encoder = OpponentHistoryEncoder(
                    18 + 5 + 1 + 3,
                    int(config.encoder_hidden_dimension),
                    embedding_dimension,
                )
                self.target_opponent_encoder = OpponentHistoryEncoder(
                    18 + 5 + 1 + 3,
                    int(config.encoder_hidden_dimension),
                    embedding_dimension,
                )
            else:
                self.opponent_encoder = None
                self.target_opponent_encoder = None

        for module in (
            self.actor,
            self.critic_1,
            self.critic_2,
            self.target_critic_1,
            self.target_critic_2,
            self.opponent_encoder,
            self.target_opponent_encoder,
        ):
            if module is not None:
                module.to(self.device)
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        self.target_critic_1.requires_grad_(False)
        self.target_critic_2.requires_grad_(False)
        if self.opponent_encoder is not None:
            self.target_opponent_encoder.load_state_dict(
                self.opponent_encoder.state_dict()
            )
            self.target_opponent_encoder.requires_grad_(False)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=config.actor_learning_rate
        )
        critic_parameters = list(self.critic_1.parameters()) + list(
            self.critic_2.parameters()
        )
        self.critic_optimizer = torch.optim.Adam(
            critic_parameters, lr=config.critic_learning_rate
        )
        self.encoder_optimizer = (
            torch.optim.Adam(
                self.opponent_encoder.parameters(),
                lr=float(config.encoder_learning_rate),
            )
            if self.opponent_encoder is not None
            else None
        )
        self.log_regime_temperature = nn.Parameter(
            torch.tensor(math.log(config.initial_regime_temperature), device=self.device)
        )
        self.log_uniform_temperature = nn.Parameter(
            torch.tensor(
                math.log(config.initial_uniform_price_temperature),
                device=self.device,
            )
        )
        self.log_bbp_temperature = nn.Parameter(
            torch.tensor(math.log(config.initial_bbp_price_temperature), device=self.device)
        )
        self.temperature_optimizers = (
            torch.optim.Adam(
                [self.log_regime_temperature], lr=config.entropy_learning_rate
            ),
            torch.optim.Adam(
                [self.log_uniform_temperature], lr=config.entropy_learning_rate
            ),
            torch.optim.Adam(
                [self.log_bbp_temperature], lr=config.entropy_learning_rate
            ),
        )
        generator_device = self.device.type
        self._exploration_generator = torch.Generator(device=generator_device)
        self._training_generator = torch.Generator(device=generator_device)
        self._exploration_generator.manual_seed(int(exploration_seed))
        self._training_generator.manual_seed(
            int(torch_cuda_seed if self.device.type == "cuda" else torch_cpu_seed)
        )
        self.environment_steps = 0
        self.update_steps = 0
        self._last_diagnostics: dict[str, float] = {}
        self.reset_recurrent_state()

    @property
    def temperatures(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.log_regime_temperature.exp(),
            self.log_uniform_temperature.exp(),
            self.log_bbp_temperature.exp(),
        )

    def reset_recurrent_state(self) -> None:
        self._actor_hidden: torch.Tensor | None = None
        self._encoder_hidden: torch.Tensor | None = None
        self._previous_action = np.zeros(5, dtype=np.float32)
        self._previous_reward = 0.0
        self._previous_opponent_controls = np.zeros(3, dtype=np.float32)

    def observe_transition(self, transition: UniversalPricingTransition) -> None:
        if not isinstance(transition, UniversalPricingTransition):
            raise TypeError("transition must be UniversalPricingTransition")
        self._previous_action = transition.effective_action.copy()
        self._previous_reward = transition.reward
        if self.uses_opponent_encoder:
            self._previous_opponent_controls = (
                transition.opponent_price_controls.copy()
            )

    def _online_tensor(self, value: np.ndarray | list[float]) -> torch.Tensor:
        return torch.as_tensor(
            value, dtype=torch.float32, device=self.device
        ).view(1, 1, -1)

    def select_action(
        self,
        observation: np.ndarray,
        *,
        deterministic: bool = False,
    ) -> PricingAction:
        observation = PricingObservationCodec.validate_vector(observation)
        obs = self._online_tensor(observation)
        action = self._online_tensor(self._previous_action)
        reward = self._online_tensor([self._previous_reward])
        base = torch.cat([obs, action, reward], dim=-1)
        with torch.no_grad():
            if self.opponent_encoder is not None:
                encoder_input = torch.cat(
                    [
                        base,
                        self._online_tensor(self._previous_opponent_controls),
                    ],
                    dim=-1,
                )
                embedding, self._encoder_hidden = self.opponent_encoder(
                    encoder_input, self._encoder_hidden
                )
                base = torch.cat([base, embedding], dim=-1)
            output = self.actor(base, self._actor_hidden)
            self._actor_hidden = output.next_hidden
            policy = output.policy
            probabilities = F.softmax(policy.regime_logits, dim=-1)
            allowed = observation[self.DECISION_INDEX] > 0
            if allowed:
                regime = int(
                    torch.argmax(probabilities, dim=-1).item()
                    if deterministic
                    else torch.multinomial(
                        probabilities.view(1, 2),
                        1,
                        generator=self._exploration_generator,
                    ).item()
                )
            else:
                regime = int(observation[self.OWN_REGIME_INDEX] > 0)
            if regime == 0:
                controls, _ = SACPricingActor.sample_price_controls(
                    policy.uniform_mean,
                    policy.uniform_log_std,
                    generator=self._exploration_generator,
                    deterministic=deterministic,
                )
                values = (float(controls.item()), 0.0, 0.0)
            else:
                controls, _ = SACPricingActor.sample_price_controls(
                    policy.bbp_mean,
                    policy.bbp_log_std,
                    generator=self._exploration_generator,
                    deterministic=deterministic,
                )
                values = (0.0, float(controls[0, 0, 0]), float(controls[0, 0, 1]))
        self.environment_steps += 1
        self._last_diagnostics = {
            "uniform_regime_probability": float(probabilities[0, 0, 0]),
            "bbp_regime_probability": float(probabilities[0, 0, 1]),
            "selected_regime": float(regime),
            "actor_hidden_norm": float(self._actor_hidden.norm()),
        }
        return PricingAction(PricingRegime(regime), *values)

    @staticmethod
    def _decision(observations: torch.Tensor) -> torch.Tensor:
        return (observations[..., 17:18] > 0).float()

    @staticmethod
    def _regime_one_hot(observations: torch.Tensor) -> torch.Tensor:
        bbp = (observations[..., 11:12] > 0).float()
        return torch.cat([1 - bbp, bbp], dim=-1)

    def _sample_branches(
        self, policy: HybridPricingPolicyOutput
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        uniform_controls, uniform_log_prob = SACPricingActor.sample_price_controls(
            policy.uniform_mean,
            policy.uniform_log_std,
            generator=self._training_generator,
        )
        bbp_controls, bbp_log_prob = SACPricingActor.sample_price_controls(
            policy.bbp_mean,
            policy.bbp_log_std,
            generator=self._training_generator,
        )
        uniform_actions, bbp_actions = _canonical_actions(
            uniform_controls, bbp_controls
        )
        return uniform_actions, uniform_log_prob, bbp_actions, bbp_log_prob

    def _prepare_batch(
        self, batch: Mapping[str, Any] | UniversalPricingSequenceBatch
    ) -> dict[str, torch.Tensor]:
        values = batch.as_mapping() if isinstance(
            batch, UniversalPricingSequenceBatch
        ) else dict(batch)
        expected = {
            "observations": 18,
            "previous_effective_actions": 5,
            "previous_rewards": 1,
            "previous_opponent_price_controls": 3,
            "effective_actions": 5,
            "rewards": 1,
            "next_observations": 18,
            "dones": 1,
            "regime_decision_masks": 1,
            "opponent_price_controls": 3,
            "valid_masks": 1,
            "loss_masks": 1,
        }
        tensors: dict[str, torch.Tensor] = {}
        prefix = None
        for name, dimension in expected.items():
            if name not in values:
                raise ValueError(f"Sequence batch is missing {name}")
            tensor = torch.as_tensor(
                values[name], dtype=torch.float32, device=self.device
            )
            if tensor.ndim != 3 or tensor.shape[-1] != dimension:
                raise ValueError(f"{name} must have shape (B, T, {dimension})")
            if not torch.isfinite(tensor).all():
                raise ValueError(f"{name} must be finite")
            prefix = tensor.shape[:2] if prefix is None else prefix
            if tensor.shape[:2] != prefix:
                raise ValueError("Sequence fields must share (B, T)")
            tensors[name] = tensor

        for name in ("valid_masks", "loss_masks"):
            tensor = tensors[name]
            if not torch.all((tensor == 0.0) | (tensor == 1.0)):
                raise ValueError(f"{name} must contain only zero or one")
        if torch.any(tensors["loss_masks"] > tensors["valid_masks"]):
            raise ValueError("loss_masks cannot include padded transitions")

        valid_rows = tensors["valid_masks"].squeeze(-1) == 1.0
        for name in ("dones", "regime_decision_masks"):
            values_on_valid_rows = tensors[name].squeeze(-1)[valid_rows]
            if not torch.all(
                (values_on_valid_rows == 0.0) | (values_on_valid_rows == 1.0)
            ):
                raise ValueError(
                    f"{name} must contain only zero or one on valid transitions"
                )
        for name in ("observations", "next_observations"):
            if torch.any(torch.abs(tensors[name][valid_rows]) > 1.0):
                raise ValueError(f"{name} must be normalized to [-1, 1]")
        for name in ("previous_effective_actions", "effective_actions"):
            actions = tensors[name][valid_rows]
            if torch.any(actions[..., :2] < 0.0) or torch.any(
                actions[..., :2] > 1.0
            ):
                raise ValueError(f"{name} regime indicators must be in [0, 1]")
            if torch.any(torch.abs(actions[..., 2:]) > 1.0):
                raise ValueError(f"{name} price controls must be in [-1, 1]")
        actions = tensors["effective_actions"][valid_rows]
        if torch.any(
            torch.abs(actions[..., :2].sum(dim=-1) - 1.0) > 1e-6
        ):
            raise ValueError(
                "effective_actions must use one-hot regimes on valid transitions"
            )
        return tensors

    def update(
        self,
        replay_batch: Mapping[str, Any] | UniversalPricingSequenceBatch,
    ) -> Mapping[str, float]:
        b = self._prepare_batch(replay_batch)
        obs, next_obs = b["observations"], b["next_observations"]
        current_base = torch.cat(
            [obs, b["previous_effective_actions"], b["previous_rewards"]],
            dim=-1,
        )
        next_base = torch.cat(
            [next_obs, b["effective_actions"], b["rewards"]], dim=-1
        )
        augmented_base = torch.cat([current_base[:, :1], next_base], dim=1)
        loss_mask = b["loss_masks"]
        regime_mask = b["regime_decision_masks"]
        prediction_loss = torch.zeros((), device=self.device)

        if self.opponent_encoder is not None:
            current_encoder_input = torch.cat(
                [current_base, b["previous_opponent_price_controls"]], dim=-1
            )
            next_encoder_input = torch.cat(
                [next_base, b["opponent_price_controls"]], dim=-1
            )
            augmented_encoder_input = torch.cat(
                [current_encoder_input[:, :1], next_encoder_input], dim=1
            )
            embeddings, _ = self.opponent_encoder(augmented_encoder_input)
            current_embedding = embeddings[:, :-1]
            predictions = self.opponent_encoder.predict(
                current_embedding, _canonicalize_replay(b["effective_actions"])
            )
            prediction_loss = _masked_mean(
                (predictions - b["opponent_price_controls"]).pow(2).mean(
                    dim=-1, keepdim=True
                ),
                loss_mask,
            )
            with torch.no_grad():
                target_embeddings, _ = self.target_opponent_encoder(
                    augmented_encoder_input
                )
            target_embedding = target_embeddings[:, 1:]
        else:
            current_embedding = obs.new_zeros((*obs.shape[:2], 0))
            target_embedding = obs.new_zeros((*obs.shape[:2], 0))

        current_history = torch.cat(
            [current_base, current_embedding], dim=-1
        )
        with torch.no_grad():
            actor_augmented_input = torch.cat(
                [augmented_base, (
                    target_embeddings
                    if self.opponent_encoder is not None
                    else augmented_base.new_zeros((*augmented_base.shape[:2], 0))
                )],
                dim=-1,
            )
            next_policy = self.actor(actor_augmented_input).policy
            next_policy = HybridPricingPolicyOutput(
                regime_logits=next_policy.regime_logits[:, 1:],
                uniform_mean=next_policy.uniform_mean[:, 1:],
                uniform_log_std=next_policy.uniform_log_std[:, 1:],
                bbp_mean=next_policy.bbp_mean[:, 1:],
                bbp_log_std=next_policy.bbp_log_std[:, 1:],
            )
            ua, ulp, ba, blp = self._sample_branches(next_policy)
            target_history = torch.cat(
                [
                    augmented_base,
                    (
                        target_embeddings
                        if self.opponent_encoder is not None
                        else augmented_base.new_zeros(
                            (*augmented_base.shape[:2], 0)
                        )
                    ),
                ],
                dim=-1,
            )
            th1 = self.target_critic_1.encode(target_history)[:, 1:]
            th2 = self.target_critic_2.encode(target_history)[:, 1:]
            uq = torch.minimum(
                self.target_critic_1.evaluate(th1, ua),
                self.target_critic_2.evaluate(th2, ua),
            )
            bq = torch.minimum(
                self.target_critic_1.evaluate(th1, ba),
                self.target_critic_2.evaluate(th2, ba),
            )
            ar, au, ab = (value.detach() for value in self.temperatures)
            probs = F.softmax(next_policy.regime_logits, dim=-1)
            log_probs = F.log_softmax(next_policy.regime_logits, dim=-1)
            next_value = HybridSACObjective.soft_value(
                regime_probabilities=probs,
                regime_log_probabilities=log_probs,
                uniform_soft_q=uq - au * ulp,
                bbp_soft_q=bq - ab * blp,
                current_regime_one_hot=self._regime_one_hot(next_obs),
                regime_decision_masks=self._decision(next_obs),
                regime_temperature=ar,
            )
            target = b["rewards"] + self.config.gamma * (
                1 - b["dones"]
            ) * next_value

        canonical = _canonicalize_replay(b["effective_actions"])
        h1 = self.critic_1.encode(current_history)
        h2 = self.critic_2.encode(current_history)
        q1 = self.critic_1.evaluate(h1, canonical)
        q2 = self.critic_2.evaluate(h2, canonical)
        critic_loss = _masked_mean((q1 - target).pow(2), loss_mask) + (
            _masked_mean((q2 - target).pow(2), loss_mask)
        )
        combined_critic_loss = critic_loss + (
            float(self.config.auxiliary_loss_weight or 0.0) * prediction_loss
        )
        self.critic_optimizer.zero_grad(set_to_none=True)
        if self.encoder_optimizer is not None:
            self.encoder_optimizer.zero_grad(set_to_none=True)
        combined_critic_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.critic_1.parameters()) + list(self.critic_2.parameters()),
            self.config.gradient_clip_norm,
        )
        self.critic_optimizer.step()
        if self.encoder_optimizer is not None:
            nn.utils.clip_grad_norm_(
                self.opponent_encoder.parameters(),
                self.config.gradient_clip_norm,
            )
            self.encoder_optimizer.step()

        self.critic_1.requires_grad_(False)
        self.critic_2.requires_grad_(False)
        actor_embedding = current_embedding.detach()
        actor_policy = self.actor(
            torch.cat([current_base, actor_embedding], dim=-1)
        ).policy
        ua, ulp, ba, blp = self._sample_branches(actor_policy)
        actor_h1 = self.critic_1.encode(
            torch.cat([current_base, actor_embedding], dim=-1)
        )
        actor_h2 = self.critic_2.encode(
            torch.cat([current_base, actor_embedding], dim=-1)
        )
        uq = torch.minimum(
            self.critic_1.evaluate(actor_h1, ua),
            self.critic_2.evaluate(actor_h2, ua),
        )
        bq = torch.minimum(
            self.critic_1.evaluate(actor_h1, ba),
            self.critic_2.evaluate(actor_h2, ba),
        )
        ar, au, ab = (value.detach() for value in self.temperatures)
        probabilities = F.softmax(actor_policy.regime_logits, dim=-1)
        categorical_logs = F.log_softmax(actor_policy.regime_logits, dim=-1)
        actor_loss = _masked_mean(
            HybridSACObjective.actor_objective(
                regime_probabilities=probabilities,
                regime_log_probabilities=categorical_logs,
                uniform_branch_objective=au * ulp - uq,
                bbp_branch_objective=ab * blp - bq,
                current_regime_one_hot=self._regime_one_hot(obs),
                regime_decision_masks=regime_mask,
                regime_temperature=ar,
            ),
            loss_mask,
        )
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(
            self.actor.parameters(), self.config.gradient_clip_norm
        )
        self.actor_optimizer.step()
        self.critic_1.requires_grad_(True)
        self.critic_2.requires_grad_(True)

        temperature_losses = self._update_temperatures(
            actor_policy, ulp, blp, obs, regime_mask, loss_mask
        )
        self._soft_update()
        self.update_steps += 1
        return {
            "critic_loss": float(critic_loss.detach()),
            "actor_loss": float(actor_loss.detach()),
            "opponent_prediction_loss": float(prediction_loss.detach()),
            "regime_temperature_loss": temperature_losses[0],
            "uniform_temperature_loss": temperature_losses[1],
            "bbp_temperature_loss": temperature_losses[2],
            "regime_temperature": float(self.temperatures[0].detach()),
            "uniform_price_temperature": float(self.temperatures[1].detach()),
            "bbp_price_temperature": float(self.temperatures[2].detach()),
            "decision_fraction": float(_masked_mean(regime_mask, loss_mask)),
        }

    def _update_temperatures(
        self,
        policy: HybridPricingPolicyOutput,
        uniform_log_prob: torch.Tensor,
        bbp_log_prob: torch.Tensor,
        observations: torch.Tensor,
        regime_masks: torch.Tensor,
        loss_masks: torch.Tensor,
    ) -> tuple[float, float, float]:
        with torch.no_grad():
            probabilities = F.softmax(policy.regime_logits, dim=-1)
            categorical_logs = F.log_softmax(policy.regime_logits, dim=-1)
            regimes = self._regime_one_hot(observations)
            uniform_weights = loss_masks * (
                regime_masks * probabilities[..., :1]
                + (1 - regime_masks) * regimes[..., :1]
            )
            bbp_weights = loss_masks * (
                regime_masks * probabilities[..., 1:]
                + (1 - regime_masks) * regimes[..., 1:]
            )
            categorical_expected = (
                probabilities * categorical_logs
            ).sum(dim=-1, keepdim=True)
        specifications = (
            (
                self.log_regime_temperature,
                loss_masks * regime_masks,
                categorical_expected,
                -self.config.regime_target_entropy_ratio * math.log(2),
            ),
            (
                self.log_uniform_temperature,
                uniform_weights,
                uniform_log_prob,
                self.config.uniform_price_target_entropy,
            ),
            (
                self.log_bbp_temperature,
                bbp_weights,
                bbp_log_prob,
                self.config.bbp_price_target_entropy,
            ),
        )
        values: list[float] = []
        for (log_temperature, weights, log_probability, target), optimizer in zip(
            specifications, self.temperature_optimizers
        ):
            weight = weights.sum()
            if float(weight) == 0:
                values.append(0.0)
                continue
            loss = log_temperature * (
                weights * (target - log_probability.detach())
            ).sum() / weight
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            values.append(float(loss.detach()))
        return tuple(values)  # type: ignore[return-value]

    def _soft_update(self) -> None:
        with torch.no_grad():
            pairs = [
                (self.critic_1, self.target_critic_1),
                (self.critic_2, self.target_critic_2),
            ]
            if self.opponent_encoder is not None:
                pairs.append(
                    (self.opponent_encoder, self.target_opponent_encoder)
                )
            for source, target in pairs:
                for source_parameter, target_parameter in zip(
                    source.parameters(), target.parameters()
                ):
                    target_parameter.mul_(1 - self.config.tau).add_(
                        source_parameter, alpha=self.config.tau
                    )

    def policy_diagnostics(self) -> Mapping[str, float]:
        values = dict(self._last_diagnostics)
        values.update(
            {
                "parameter_count": float(
                    sum(p.numel() for p in self._trainable_parameters())
                ),
                "environment_steps": float(self.environment_steps),
                "update_steps": float(self.update_steps),
            }
        )
        return values

    def _trainable_parameters(self):
        modules = [self.actor, self.critic_1, self.critic_2]
        if self.opponent_encoder is not None:
            modules.append(self.opponent_encoder)
        return [parameter for module in modules for parameter in module.parameters()]

    def _payload(self) -> dict[str, Any]:
        payload = {
            "schema_version": RECURRENT_PRICING_CHECKPOINT_VERSION,
            "architecture": self.config.architecture.value,
            "config": self.config.to_dict(),
            "actor": self.actor.state_dict(),
            "critic_1": self.critic_1.state_dict(),
            "critic_2": self.critic_2.state_dict(),
            "target_critic_1": self.target_critic_1.state_dict(),
            "target_critic_2": self.target_critic_2.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "log_temperatures": [
                value.detach().cpu()
                for value in (
                    self.log_regime_temperature,
                    self.log_uniform_temperature,
                    self.log_bbp_temperature,
                )
            ],
            "temperature_optimizers": [
                optimizer.state_dict() for optimizer in self.temperature_optimizers
            ],
            "exploration_rng": self._exploration_generator.get_state().cpu(),
            "training_rng": self._training_generator.get_state().cpu(),
            "environment_steps": self.environment_steps,
            "update_steps": self.update_steps,
            "actor_hidden": (
                None if self._actor_hidden is None else self._actor_hidden.cpu()
            ),
            "encoder_hidden": (
                None if self._encoder_hidden is None else self._encoder_hidden.cpu()
            ),
            "previous_action": self._previous_action,
            "previous_reward": self._previous_reward,
            "previous_opponent_controls": self._previous_opponent_controls,
        }
        if self.opponent_encoder is not None:
            payload.update(
                {
                    "opponent_encoder": self.opponent_encoder.state_dict(),
                    "target_opponent_encoder": (
                        self.target_opponent_encoder.state_dict()
                    ),
                    "encoder_optimizer": self.encoder_optimizer.state_dict(),
                }
            )
        return payload

    def save(self, checkpoint_path: str | Path) -> None:
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.temporary")
        try:
            torch.save(self._payload(), temporary)
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()

    def load(self, checkpoint_path: str | Path) -> None:
        payload = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )
        if (
            payload.get("schema_version") != RECURRENT_PRICING_CHECKPOINT_VERSION
            or payload.get("architecture") != self.config.architecture.value
            or payload.get("config") != self.config.to_dict()
        ):
            raise ValueError("Incompatible recurrent pricing checkpoint")
        for name in (
            "actor",
            "critic_1",
            "critic_2",
            "target_critic_1",
            "target_critic_2",
        ):
            getattr(self, name).load_state_dict(payload[name])
        self.actor_optimizer.load_state_dict(payload["actor_optimizer"])
        self.critic_optimizer.load_state_dict(payload["critic_optimizer"])
        for target, value in zip(
            (
                self.log_regime_temperature,
                self.log_uniform_temperature,
                self.log_bbp_temperature,
            ),
            payload["log_temperatures"],
        ):
            with torch.no_grad():
                target.copy_(value.to(self.device))
        for optimizer, state in zip(
            self.temperature_optimizers, payload["temperature_optimizers"]
        ):
            optimizer.load_state_dict(state)
        if self.opponent_encoder is not None:
            self.opponent_encoder.load_state_dict(payload["opponent_encoder"])
            self.target_opponent_encoder.load_state_dict(
                payload["target_opponent_encoder"]
            )
            self.encoder_optimizer.load_state_dict(payload["encoder_optimizer"])
        self._exploration_generator.set_state(payload["exploration_rng"].cpu())
        self._training_generator.set_state(payload["training_rng"].cpu())
        self.environment_steps = int(payload["environment_steps"])
        self.update_steps = int(payload["update_steps"])
        self._actor_hidden = (
            None
            if payload["actor_hidden"] is None
            else payload["actor_hidden"].to(self.device)
        )
        self._encoder_hidden = (
            None
            if payload["encoder_hidden"] is None
            else payload["encoder_hidden"].to(self.device)
        )
        self._previous_action = np.asarray(
            payload["previous_action"], dtype=np.float32
        )
        self._previous_reward = float(payload["previous_reward"])
        self._previous_opponent_controls = np.asarray(
            payload["previous_opponent_controls"], dtype=np.float32
        )


class RecurrentSACPricingAgent(_UniversalRecurrentSACPricingAgent):
    """Plain recurrent SAC with no opponent encoder."""

    def __init__(self, config: RecurrentSACPricingAgentConfig, **kwargs) -> None:
        if config.architecture is not AgentArchitecture.RSAC:
            raise ValueError("RecurrentSACPricingAgent requires rsac config")
        super().__init__(config, **kwargs)


class OpponentEmbeddingRecurrentSACPricingAgent(
    _UniversalRecurrentSACPricingAgent
):
    """Recurrent SAC with a dedicated opponent-history encoder."""

    def __init__(self, config: RecurrentSACPricingAgentConfig, **kwargs) -> None:
        if config.architecture is not AgentArchitecture.OE_RSAC:
            raise ValueError(
                "OpponentEmbeddingRecurrentSACPricingAgent requires oe_rsac config"
            )
        super().__init__(config, **kwargs)
