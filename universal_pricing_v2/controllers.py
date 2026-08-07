"""Independent SAC objectives for v2 price and regime tasks."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from universal_pricing_v2.networks import (
    FeedForwardPriceActor,
    FeedForwardPriceCritic,
    FeedForwardStrategyActor,
    FeedForwardStrategyCritic,
    RecurrentPriceActor,
    RecurrentPriceCritic,
    RecurrentStrategyActor,
    RecurrentStrategyCritic,
)


def _tensor(
    value: Any,
    device: torch.device,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    result = torch.as_tensor(value, dtype=dtype, device=device)
    if dtype.is_floating_point and not torch.isfinite(result).all():
        raise ValueError("Controller batch contains non-finite values")
    return result


def _masked_mean(values: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    denominator = masks.sum().clamp_min(1.0)
    return (values * masks).sum() / denominator


def _soft_update(
    sources: Sequence[nn.Module],
    targets: Sequence[nn.Module],
    tau: float,
) -> None:
    with torch.no_grad():
        for source, target in zip(sources, targets):
            for source_parameter, target_parameter in zip(
                source.parameters(), target.parameters()
            ):
                target_parameter.mul_(1.0 - tau)
                target_parameter.add_(source_parameter, alpha=tau)


@dataclass(frozen=True)
class ControllerUpdateMetrics:
    critic_loss: float
    actor_loss: float
    temperature_loss: float
    temperature: float
    target_q_mean: float
    q1_mean: float
    q2_mean: float
    critic_gradient_norm: float
    actor_gradient_norm: float
    active_fraction: float

    def prefixed(self, prefix: str) -> dict[str, float]:
        return {
            f"{prefix}_{name}": float(getattr(self, name))
            for name in self.__dataclass_fields__
        }


class FeedForwardContinuousSACController:
    """Independent continuous SAC controller for uniform or BBP pricing."""

    def __init__(
        self,
        *,
        observation_dimension: int,
        action_dimension: int,
        actor_hidden_dimensions: Sequence[int],
        critic_hidden_dimensions: Sequence[int],
        actor_learning_rate: float,
        critic_learning_rate: float,
        entropy_learning_rate: float,
        initial_temperature: float,
        gamma: float,
        tau: float,
        gradient_clip_norm: float,
        device: torch.device,
    ) -> None:
        self.device = device
        self.action_dimension = action_dimension
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.gradient_clip_norm = float(gradient_clip_norm)
        self.actor = FeedForwardPriceActor(
            observation_dimension,
            actor_hidden_dimensions,
            action_dimension,
        ).to(device)
        self.critic_1 = FeedForwardPriceCritic(
            observation_dimension,
            action_dimension,
            critic_hidden_dimensions,
        ).to(device)
        self.critic_2 = FeedForwardPriceCritic(
            observation_dimension,
            action_dimension,
            critic_hidden_dimensions,
        ).to(device)
        self.target_critic_1 = copy.deepcopy(self.critic_1).to(device)
        self.target_critic_2 = copy.deepcopy(self.critic_2).to(device)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_learning_rate
        )
        self.critic_optimizer = torch.optim.Adam(
            list(self.critic_1.parameters())
            + list(self.critic_2.parameters()),
            lr=critic_learning_rate,
        )
        self.log_temperature = torch.tensor(
            math.log(initial_temperature),
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )
        self.temperature_optimizer = torch.optim.Adam(
            [self.log_temperature], lr=entropy_learning_rate
        )
        self.target_entropy = -float(action_dimension)
        self.update_steps = 0

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp()

    def set_actor_learning_rate(self, learning_rate: float) -> None:
        for group in self.actor_optimizer.param_groups:
            group["lr"] = float(learning_rate)

    def select(
        self,
        observation: torch.Tensor,
        *,
        generator: torch.Generator | None,
        deterministic: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.actor.sample(
            observation,
            generator=generator,
            deterministic=deterministic,
        )

    def update(
        self,
        batch: Mapping[str, Any],
        *,
        generator: torch.Generator | None,
    ) -> ControllerUpdateMetrics:
        observations = _tensor(batch["observations"], self.device)
        actions = _tensor(batch["price_actions"], self.device)
        rewards = _tensor(batch["rewards"], self.device)
        next_observations = _tensor(
            batch["next_observations"], self.device
        )
        dones = _tensor(batch["dones"], self.device)
        masks = _tensor(batch["active_controller_masks"], self.device)
        active_fraction = float(masks.mean().detach().cpu())
        if float(masks.sum()) <= 0.0:
            raise ValueError("Price controller batch has no active transitions")

        with torch.no_grad():
            next_actions, next_log_probability = self.actor.sample(
                next_observations, generator=generator
            )
            next_q = torch.minimum(
                self.target_critic_1(next_observations, next_actions),
                self.target_critic_2(next_observations, next_actions),
            )
            target_q = rewards + self.gamma * (1.0 - dones) * (
                next_q
                - self.temperature.detach() * next_log_probability
            )

        q1 = self.critic_1(observations, actions)
        q2 = self.critic_2(observations, actions)
        critic_loss = _masked_mean(
            (q1 - target_q).pow(2) + (q2 - target_q).pow(2),
            masks,
        )
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_gradient = nn.utils.clip_grad_norm_(
            list(self.critic_1.parameters())
            + list(self.critic_2.parameters()),
            self.gradient_clip_norm,
        )
        self.critic_optimizer.step()

        self.critic_1.requires_grad_(False)
        self.critic_2.requires_grad_(False)
        sampled_actions, log_probability = self.actor.sample(
            observations, generator=generator
        )
        sampled_q = torch.minimum(
            self.critic_1(observations, sampled_actions),
            self.critic_2(observations, sampled_actions),
        )
        actor_loss = _masked_mean(
            self.temperature.detach() * log_probability - sampled_q,
            masks,
        )
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_gradient = nn.utils.clip_grad_norm_(
            self.actor.parameters(), self.gradient_clip_norm
        )
        self.actor_optimizer.step()
        self.critic_1.requires_grad_(True)
        self.critic_2.requires_grad_(True)

        entropy = -log_probability.detach()
        temperature_loss = _masked_mean(
            self.log_temperature * (entropy - (-self.target_entropy)),
            masks,
        )
        self.temperature_optimizer.zero_grad(set_to_none=True)
        temperature_loss.backward()
        self.temperature_optimizer.step()
        _soft_update(
            (self.critic_1, self.critic_2),
            (self.target_critic_1, self.target_critic_2),
            self.tau,
        )
        self.update_steps += 1
        return ControllerUpdateMetrics(
            critic_loss=float(critic_loss.detach().cpu()),
            actor_loss=float(actor_loss.detach().cpu()),
            temperature_loss=float(temperature_loss.detach().cpu()),
            temperature=float(self.temperature.detach().cpu()),
            target_q_mean=float(
                _masked_mean(target_q, masks).detach().cpu()
            ),
            q1_mean=float(_masked_mean(q1, masks).detach().cpu()),
            q2_mean=float(_masked_mean(q2, masks).detach().cpu()),
            critic_gradient_norm=float(critic_gradient.detach().cpu()),
            actor_gradient_norm=float(actor_gradient.detach().cpu()),
            active_fraction=active_fraction,
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "critic_1": self.critic_1.state_dict(),
            "critic_2": self.critic_2.state_dict(),
            "target_critic_1": self.target_critic_1.state_dict(),
            "target_critic_2": self.target_critic_2.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "log_temperature": self.log_temperature.detach().cpu(),
            "temperature_optimizer": self.temperature_optimizer.state_dict(),
            "update_steps": self.update_steps,
        }

    def load_state_dict(self, values: Mapping[str, Any]) -> None:
        for name in (
            "actor",
            "critic_1",
            "critic_2",
            "target_critic_1",
            "target_critic_2",
        ):
            getattr(self, name).load_state_dict(values[name])
        self.actor_optimizer.load_state_dict(values["actor_optimizer"])
        self.critic_optimizer.load_state_dict(values["critic_optimizer"])
        self.log_temperature.data.copy_(
            values["log_temperature"].to(self.device)
        )
        self.temperature_optimizer.load_state_dict(
            values["temperature_optimizer"]
        )
        self.update_steps = int(values["update_steps"])


class FeedForwardDiscreteSACController:
    """Exact two-action categorical SAC strategy controller."""

    def __init__(
        self,
        *,
        observation_dimension: int,
        hidden_dimensions: Sequence[int],
        actor_learning_rate: float,
        critic_learning_rate: float,
        entropy_learning_rate: float,
        initial_temperature: float,
        gamma_price: float,
        tau: float,
        gradient_clip_norm: float,
        device: torch.device,
    ) -> None:
        self.device = device
        self.gamma_price = float(gamma_price)
        self.tau = float(tau)
        self.gradient_clip_norm = float(gradient_clip_norm)
        self.actor = FeedForwardStrategyActor(
            observation_dimension, hidden_dimensions
        ).to(device)
        self.critic_1 = FeedForwardStrategyCritic(
            observation_dimension, hidden_dimensions
        ).to(device)
        self.critic_2 = FeedForwardStrategyCritic(
            observation_dimension, hidden_dimensions
        ).to(device)
        self.target_critic_1 = copy.deepcopy(self.critic_1).to(device)
        self.target_critic_2 = copy.deepcopy(self.critic_2).to(device)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_learning_rate
        )
        self.critic_optimizer = torch.optim.Adam(
            list(self.critic_1.parameters())
            + list(self.critic_2.parameters()),
            lr=critic_learning_rate,
        )
        self.log_temperature = torch.tensor(
            math.log(initial_temperature),
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )
        self.temperature_optimizer = torch.optim.Adam(
            [self.log_temperature], lr=entropy_learning_rate
        )
        self.target_entropy = 0.98 * math.log(2.0)
        self.update_steps = 0

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp()

    def select(
        self,
        observation: torch.Tensor,
        *,
        generator: torch.Generator | None,
        deterministic: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.actor.sample(
            observation,
            generator=generator,
            deterministic=deterministic,
        )

    def update(
        self,
        batch: Mapping[str, Any],
    ) -> ControllerUpdateMetrics:
        observations = _tensor(batch["observations"], self.device)
        actions = _tensor(
            batch["regime_actions"], self.device, dtype=torch.long
        )
        rewards = _tensor(batch["macro_rewards"], self.device)
        next_observations = _tensor(
            batch["next_observations"], self.device
        )
        dones = _tensor(batch["dones"], self.device)
        durations = _tensor(batch["durations"], self.device)
        masks = torch.ones_like(rewards)

        with torch.no_grad():
            next_probabilities, next_log_probabilities = (
                self.actor.probabilities(next_observations)
            )
            next_q = torch.minimum(
                self.target_critic_1(next_observations),
                self.target_critic_2(next_observations),
            )
            next_value = (
                next_probabilities
                * (
                    next_q
                    - self.temperature.detach()
                    * next_log_probabilities
                )
            ).sum(dim=-1, keepdim=True)
            # ``gamma_price`` is the frozen ten-period macro discount
            # (0.99 ** 10). Fractional final windows therefore bootstrap with
            # (0.99 ** 10) ** (k / 10) == 0.99 ** k.
            discount = torch.pow(
                torch.full_like(durations, self.gamma_price),
                durations / 10.0,
            )
            target_q = rewards + discount * (1.0 - dones) * next_value

        all_q1 = self.critic_1(observations)
        all_q2 = self.critic_2(observations)
        q1 = all_q1.gather(1, actions)
        q2 = all_q2.gather(1, actions)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_gradient = nn.utils.clip_grad_norm_(
            list(self.critic_1.parameters())
            + list(self.critic_2.parameters()),
            self.gradient_clip_norm,
        )
        self.critic_optimizer.step()

        self.critic_1.requires_grad_(False)
        self.critic_2.requires_grad_(False)
        probabilities, log_probabilities = self.actor.probabilities(
            observations
        )
        minimum_q = torch.minimum(
            self.critic_1(observations),
            self.critic_2(observations),
        )
        actor_loss = (
            probabilities
            * (
                self.temperature.detach() * log_probabilities
                - minimum_q
            )
        ).sum(dim=-1).mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_gradient = nn.utils.clip_grad_norm_(
            self.actor.parameters(), self.gradient_clip_norm
        )
        self.actor_optimizer.step()
        self.critic_1.requires_grad_(True)
        self.critic_2.requires_grad_(True)

        entropy = -(
            probabilities.detach() * log_probabilities.detach()
        ).sum(dim=-1, keepdim=True)
        temperature_loss = (
            self.log_temperature * (entropy - self.target_entropy)
        ).mean()
        self.temperature_optimizer.zero_grad(set_to_none=True)
        temperature_loss.backward()
        self.temperature_optimizer.step()
        _soft_update(
            (self.critic_1, self.critic_2),
            (self.target_critic_1, self.target_critic_2),
            self.tau,
        )
        self.update_steps += 1
        return ControllerUpdateMetrics(
            critic_loss=float(critic_loss.detach().cpu()),
            actor_loss=float(actor_loss.detach().cpu()),
            temperature_loss=float(temperature_loss.detach().cpu()),
            temperature=float(self.temperature.detach().cpu()),
            target_q_mean=float(target_q.mean().detach().cpu()),
            q1_mean=float(q1.mean().detach().cpu()),
            q2_mean=float(q2.mean().detach().cpu()),
            critic_gradient_norm=float(critic_gradient.detach().cpu()),
            actor_gradient_norm=float(actor_gradient.detach().cpu()),
            active_fraction=1.0,
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "critic_1": self.critic_1.state_dict(),
            "critic_2": self.critic_2.state_dict(),
            "target_critic_1": self.target_critic_1.state_dict(),
            "target_critic_2": self.target_critic_2.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "log_temperature": self.log_temperature.detach().cpu(),
            "temperature_optimizer": self.temperature_optimizer.state_dict(),
            "update_steps": self.update_steps,
        }

    def load_state_dict(self, values: Mapping[str, Any]) -> None:
        for name in (
            "actor",
            "critic_1",
            "critic_2",
            "target_critic_1",
            "target_critic_2",
        ):
            getattr(self, name).load_state_dict(values[name])
        self.actor_optimizer.load_state_dict(values["actor_optimizer"])
        self.critic_optimizer.load_state_dict(values["critic_optimizer"])
        self.log_temperature.data.copy_(
            values["log_temperature"].to(self.device)
        )
        self.temperature_optimizer.load_state_dict(
            values["temperature_optimizer"]
        )
        self.update_steps = int(values["update_steps"])


class RecurrentContinuousSACController:
    """Sequence-masked recurrent SAC for one price skill."""

    def __init__(
        self,
        *,
        recurrent_input_dimension: int,
        action_dimension: int,
        hidden_dimension: int,
        actor_learning_rate: float,
        critic_learning_rate: float,
        entropy_learning_rate: float,
        initial_temperature: float,
        gamma: float,
        tau: float,
        gradient_clip_norm: float,
        device: torch.device,
    ) -> None:
        self.device = device
        self.recurrent_input_dimension = recurrent_input_dimension
        self.action_dimension = action_dimension
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.gradient_clip_norm = float(gradient_clip_norm)
        self.actor = RecurrentPriceActor(
            recurrent_input_dimension, hidden_dimension, action_dimension
        ).to(device)
        self.critic_1 = RecurrentPriceCritic(
            recurrent_input_dimension, hidden_dimension, action_dimension
        ).to(device)
        self.critic_2 = RecurrentPriceCritic(
            recurrent_input_dimension, hidden_dimension, action_dimension
        ).to(device)
        self.target_critic_1 = copy.deepcopy(self.critic_1).to(device)
        self.target_critic_2 = copy.deepcopy(self.critic_2).to(device)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_learning_rate
        )
        self.critic_optimizer = torch.optim.Adam(
            list(self.critic_1.parameters())
            + list(self.critic_2.parameters()),
            lr=critic_learning_rate,
        )
        self.log_temperature = torch.tensor(
            math.log(initial_temperature),
            device=device,
            requires_grad=True,
        )
        self.temperature_optimizer = torch.optim.Adam(
            [self.log_temperature], lr=entropy_learning_rate
        )
        self.target_entropy = -float(action_dimension)
        self.update_steps = 0

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp()

    def set_actor_learning_rate(self, learning_rate: float) -> None:
        for group in self.actor_optimizer.param_groups:
            group["lr"] = float(learning_rate)

    @staticmethod
    def recurrent_inputs(
        observations: torch.Tensor,
        previous_actions: torch.Tensor,
        previous_rewards: torch.Tensor,
        previous_active_masks: torch.Tensor,
        embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        values = [
            observations,
            previous_actions,
            previous_rewards,
            previous_active_masks,
        ]
        if embeddings is not None:
            values.append(embeddings)
        return torch.cat(values, dim=-1)

    def update(
        self,
        batch: Mapping[str, Any],
        *,
        generator: torch.Generator | None,
        embeddings: torch.Tensor | None = None,
        next_embeddings: torch.Tensor | None = None,
    ) -> ControllerUpdateMetrics:
        observations = _tensor(batch["observations"], self.device)
        previous_actions = _tensor(
            batch["previous_price_actions"], self.device
        )
        previous_rewards = _tensor(
            batch["previous_rewards"], self.device
        )
        previous_masks = _tensor(
            batch["previous_active_masks"], self.device
        )
        actions = _tensor(batch["price_actions"], self.device)
        rewards = _tensor(batch["rewards"], self.device)
        next_observations = _tensor(
            batch["next_observations"], self.device
        )
        dones = _tensor(batch["dones"], self.device)
        active_masks = _tensor(
            batch["active_controller_masks"], self.device
        )
        loss_masks = _tensor(batch["loss_masks"], self.device)
        masks = active_masks * loss_masks
        if float(masks.sum()) <= 0:
            raise ValueError("Recurrent price batch has no active loss steps")
        current_inputs = self.recurrent_inputs(
            observations,
            previous_actions,
            previous_rewards,
            previous_masks,
            embeddings,
        )
        next_inputs = self.recurrent_inputs(
            next_observations,
            actions,
            rewards,
            active_masks,
            next_embeddings,
        )

        with torch.no_grad():
            next_actions, next_log_probability, _ = self.actor.sample(
                next_inputs, generator=generator
            )
            next_q1, _ = self.target_critic_1(
                next_inputs, next_actions
            )
            next_q2, _ = self.target_critic_2(
                next_inputs, next_actions
            )
            next_q = torch.minimum(next_q1, next_q2)
            target_q = rewards + self.gamma * (1.0 - dones) * (
                next_q
                - self.temperature.detach() * next_log_probability
            )

        q1, _ = self.critic_1(current_inputs, actions)
        q2, _ = self.critic_2(current_inputs, actions)
        critic_loss = _masked_mean(
            (q1 - target_q).pow(2) + (q2 - target_q).pow(2),
            masks,
        )
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_gradient = nn.utils.clip_grad_norm_(
            list(self.critic_1.parameters())
            + list(self.critic_2.parameters()),
            self.gradient_clip_norm,
        )
        self.critic_optimizer.step()

        detached_inputs = (
            current_inputs.detach()
            if embeddings is not None
            else current_inputs
        )
        self.critic_1.requires_grad_(False)
        self.critic_2.requires_grad_(False)
        sampled_actions, log_probability, _ = self.actor.sample(
            detached_inputs, generator=generator
        )
        actor_q1, _ = self.critic_1(detached_inputs, sampled_actions)
        actor_q2, _ = self.critic_2(detached_inputs, sampled_actions)
        actor_loss = _masked_mean(
            self.temperature.detach() * log_probability
            - torch.minimum(actor_q1, actor_q2),
            masks,
        )
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_gradient = nn.utils.clip_grad_norm_(
            self.actor.parameters(), self.gradient_clip_norm
        )
        self.actor_optimizer.step()
        self.critic_1.requires_grad_(True)
        self.critic_2.requires_grad_(True)

        entropy = -log_probability.detach()
        temperature_loss = _masked_mean(
            self.log_temperature * (entropy - (-self.target_entropy)),
            masks,
        )
        self.temperature_optimizer.zero_grad(set_to_none=True)
        temperature_loss.backward()
        self.temperature_optimizer.step()
        _soft_update(
            (self.critic_1, self.critic_2),
            (self.target_critic_1, self.target_critic_2),
            self.tau,
        )
        self.update_steps += 1
        return ControllerUpdateMetrics(
            critic_loss=float(critic_loss.detach().cpu()),
            actor_loss=float(actor_loss.detach().cpu()),
            temperature_loss=float(temperature_loss.detach().cpu()),
            temperature=float(self.temperature.detach().cpu()),
            target_q_mean=float(
                _masked_mean(target_q, masks).detach().cpu()
            ),
            q1_mean=float(_masked_mean(q1, masks).detach().cpu()),
            q2_mean=float(_masked_mean(q2, masks).detach().cpu()),
            critic_gradient_norm=float(critic_gradient.detach().cpu()),
            actor_gradient_norm=float(actor_gradient.detach().cpu()),
            active_fraction=float(active_masks.mean().detach().cpu()),
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "critic_1": self.critic_1.state_dict(),
            "critic_2": self.critic_2.state_dict(),
            "target_critic_1": self.target_critic_1.state_dict(),
            "target_critic_2": self.target_critic_2.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "log_temperature": self.log_temperature.detach().cpu(),
            "temperature_optimizer": self.temperature_optimizer.state_dict(),
            "update_steps": self.update_steps,
        }

    def load_state_dict(self, values: Mapping[str, Any]) -> None:
        for name in (
            "actor",
            "critic_1",
            "critic_2",
            "target_critic_1",
            "target_critic_2",
        ):
            getattr(self, name).load_state_dict(values[name])
        self.actor_optimizer.load_state_dict(values["actor_optimizer"])
        self.critic_optimizer.load_state_dict(values["critic_optimizer"])
        self.log_temperature.data.copy_(
            values["log_temperature"].to(self.device)
        )
        self.temperature_optimizer.load_state_dict(
            values["temperature_optimizer"]
        )
        self.update_steps = int(values["update_steps"])


class RecurrentDiscreteSACController:
    """Sequence-masked categorical SAC macro controller."""

    def __init__(
        self,
        *,
        recurrent_input_dimension: int,
        hidden_dimension: int,
        actor_learning_rate: float,
        critic_learning_rate: float,
        entropy_learning_rate: float,
        initial_temperature: float,
        gamma_price: float,
        tau: float,
        gradient_clip_norm: float,
        device: torch.device,
    ) -> None:
        self.device = device
        self.gamma_price = float(gamma_price)
        self.tau = float(tau)
        self.gradient_clip_norm = float(gradient_clip_norm)
        self.actor = RecurrentStrategyActor(
            recurrent_input_dimension, hidden_dimension
        ).to(device)
        self.critic_1 = RecurrentStrategyCritic(
            recurrent_input_dimension, hidden_dimension
        ).to(device)
        self.critic_2 = RecurrentStrategyCritic(
            recurrent_input_dimension, hidden_dimension
        ).to(device)
        self.target_critic_1 = copy.deepcopy(self.critic_1).to(device)
        self.target_critic_2 = copy.deepcopy(self.critic_2).to(device)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_learning_rate
        )
        self.critic_optimizer = torch.optim.Adam(
            list(self.critic_1.parameters())
            + list(self.critic_2.parameters()),
            lr=critic_learning_rate,
        )
        self.log_temperature = torch.tensor(
            math.log(initial_temperature),
            device=device,
            requires_grad=True,
        )
        self.temperature_optimizer = torch.optim.Adam(
            [self.log_temperature], lr=entropy_learning_rate
        )
        self.target_entropy = 0.98 * math.log(2.0)
        self.update_steps = 0

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp()

    @staticmethod
    def recurrent_inputs(
        observations: torch.Tensor,
        previous_regime_one_hot: torch.Tensor,
        previous_macro_rewards: torch.Tensor,
        embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        values = [
            observations,
            previous_regime_one_hot,
            previous_macro_rewards,
        ]
        if embeddings is not None:
            values.append(embeddings)
        return torch.cat(values, dim=-1)

    def update(
        self,
        batch: Mapping[str, Any],
        *,
        embeddings: torch.Tensor | None = None,
        next_embeddings: torch.Tensor | None = None,
    ) -> ControllerUpdateMetrics:
        observations = _tensor(batch["observations"], self.device)
        previous_one_hot = _tensor(
            batch["previous_regime_one_hot"], self.device
        )
        previous_rewards = _tensor(
            batch["previous_macro_rewards"], self.device
        )
        actions = _tensor(
            batch["regime_actions"], self.device, dtype=torch.long
        )
        rewards = _tensor(batch["macro_rewards"], self.device)
        next_observations = _tensor(
            batch["next_observations"], self.device
        )
        dones = _tensor(batch["dones"], self.device)
        durations = _tensor(batch["durations"], self.device)
        loss_masks = _tensor(batch["loss_masks"], self.device)
        current_inputs = self.recurrent_inputs(
            observations,
            previous_one_hot,
            previous_rewards,
            embeddings,
        )
        current_action_one_hot = F.one_hot(
            actions.squeeze(-1), num_classes=2
        ).float()
        next_inputs = self.recurrent_inputs(
            next_observations,
            current_action_one_hot,
            rewards,
            next_embeddings,
        )

        with torch.no_grad():
            next_probabilities, next_log_probabilities, _ = (
                self.actor.probabilities(next_inputs)
            )
            next_q1, _ = self.target_critic_1(next_inputs)
            next_q2, _ = self.target_critic_2(next_inputs)
            next_value = (
                next_probabilities
                * (
                    torch.minimum(next_q1, next_q2)
                    - self.temperature.detach()
                    * next_log_probabilities
                )
            ).sum(dim=-1, keepdim=True)
            discount = torch.pow(
                torch.full_like(durations, self.gamma_price),
                durations / 10.0,
            )
            target_q = rewards + discount * (1.0 - dones) * next_value

        all_q1, _ = self.critic_1(current_inputs)
        all_q2, _ = self.critic_2(current_inputs)
        q1 = all_q1.gather(-1, actions)
        q2 = all_q2.gather(-1, actions)
        critic_loss = _masked_mean(
            (q1 - target_q).pow(2) + (q2 - target_q).pow(2),
            loss_masks,
        )
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_gradient = nn.utils.clip_grad_norm_(
            list(self.critic_1.parameters())
            + list(self.critic_2.parameters()),
            self.gradient_clip_norm,
        )
        self.critic_optimizer.step()

        actor_inputs = (
            current_inputs.detach()
            if embeddings is not None
            else current_inputs
        )
        self.critic_1.requires_grad_(False)
        self.critic_2.requires_grad_(False)
        probabilities, log_probabilities, _ = self.actor.probabilities(
            actor_inputs
        )
        actor_q1, _ = self.critic_1(actor_inputs)
        actor_q2, _ = self.critic_2(actor_inputs)
        actor_loss_values = (
            probabilities
            * (
                self.temperature.detach() * log_probabilities
                - torch.minimum(actor_q1, actor_q2)
            )
        ).sum(dim=-1, keepdim=True)
        actor_loss = _masked_mean(actor_loss_values, loss_masks)
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_gradient = nn.utils.clip_grad_norm_(
            self.actor.parameters(), self.gradient_clip_norm
        )
        self.actor_optimizer.step()
        self.critic_1.requires_grad_(True)
        self.critic_2.requires_grad_(True)

        entropy = -(
            probabilities.detach() * log_probabilities.detach()
        ).sum(dim=-1, keepdim=True)
        temperature_loss = _masked_mean(
            self.log_temperature * (entropy - self.target_entropy),
            loss_masks,
        )
        self.temperature_optimizer.zero_grad(set_to_none=True)
        temperature_loss.backward()
        self.temperature_optimizer.step()
        _soft_update(
            (self.critic_1, self.critic_2),
            (self.target_critic_1, self.target_critic_2),
            self.tau,
        )
        self.update_steps += 1
        return ControllerUpdateMetrics(
            critic_loss=float(critic_loss.detach().cpu()),
            actor_loss=float(actor_loss.detach().cpu()),
            temperature_loss=float(temperature_loss.detach().cpu()),
            temperature=float(self.temperature.detach().cpu()),
            target_q_mean=float(
                _masked_mean(target_q, loss_masks).detach().cpu()
            ),
            q1_mean=float(_masked_mean(q1, loss_masks).detach().cpu()),
            q2_mean=float(_masked_mean(q2, loss_masks).detach().cpu()),
            critic_gradient_norm=float(critic_gradient.detach().cpu()),
            actor_gradient_norm=float(actor_gradient.detach().cpu()),
            active_fraction=1.0,
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "critic_1": self.critic_1.state_dict(),
            "critic_2": self.critic_2.state_dict(),
            "target_critic_1": self.target_critic_1.state_dict(),
            "target_critic_2": self.target_critic_2.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "log_temperature": self.log_temperature.detach().cpu(),
            "temperature_optimizer": self.temperature_optimizer.state_dict(),
            "update_steps": self.update_steps,
        }

    def load_state_dict(self, values: Mapping[str, Any]) -> None:
        for name in (
            "actor",
            "critic_1",
            "critic_2",
            "target_critic_1",
            "target_critic_2",
        ):
            getattr(self, name).load_state_dict(values[name])
        self.actor_optimizer.load_state_dict(values["actor_optimizer"])
        self.critic_optimizer.load_state_dict(values["critic_optimizer"])
        self.log_temperature.data.copy_(
            values["log_temperature"].to(self.device)
        )
        self.temperature_optimizer.load_state_dict(
            values["temperature_optimizer"]
        )
        self.update_steps = int(values["update_steps"])
