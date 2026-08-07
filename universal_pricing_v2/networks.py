"""Neural components for the three independent v2 controller tasks."""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F


LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


def multilayer_perceptron(
    input_dimension: int,
    hidden_dimensions: Sequence[int],
    output_dimension: int,
) -> nn.Sequential:
    """Create the protocol's ReLU MLP with explicit dimensions."""

    dimensions = [input_dimension, *hidden_dimensions, output_dimension]
    layers: list[nn.Module] = []
    for index, (source, target) in enumerate(
        zip(dimensions, dimensions[1:])
    ):
        layers.append(nn.Linear(source, target))
        if index < len(dimensions) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def _sample_tanh_gaussian(
    mean: torch.Tensor,
    log_standard_deviation: torch.Tensor,
    *,
    generator: torch.Generator | None,
    deterministic: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    log_standard_deviation = torch.clamp(
        log_standard_deviation, LOG_STD_MIN, LOG_STD_MAX
    )
    if deterministic:
        pre_tanh = mean
    else:
        noise = torch.randn(
            mean.shape,
            dtype=mean.dtype,
            device=mean.device,
            generator=generator,
        )
        pre_tanh = mean + log_standard_deviation.exp() * noise
    action = torch.tanh(pre_tanh)
    variance_term = (
        (pre_tanh - mean) / (log_standard_deviation.exp() + 1e-8)
    ).pow(2)
    normal_log_probability = -0.5 * (
        variance_term
        + 2.0 * log_standard_deviation
        + math.log(2.0 * math.pi)
    )
    correction = torch.log(1.0 - action.pow(2) + 1e-6)
    log_probability = (
        normal_log_probability - correction
    ).sum(dim=-1, keepdim=True)
    return action, log_probability


class FeedForwardPriceActor(nn.Module):
    """Continuous tanh-Gaussian actor for one pricing skill."""

    def __init__(
        self,
        observation_dimension: int,
        hidden_dimensions: Sequence[int],
        action_dimension: int,
    ) -> None:
        super().__init__()
        self.action_dimension = action_dimension
        self.trunk = nn.Sequential(
            multilayer_perceptron(
                observation_dimension,
                hidden_dimensions[:-1],
                hidden_dimensions[-1],
            ),
            nn.ReLU(),
        )
        self.mean = nn.Linear(hidden_dimensions[-1], action_dimension)
        self.log_standard_deviation = nn.Linear(
            hidden_dimensions[-1], action_dimension
        )

    def forward(
        self, observations: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.trunk(observations)
        return (
            self.mean(features),
            torch.clamp(
                self.log_standard_deviation(features),
                LOG_STD_MIN,
                LOG_STD_MAX,
            ),
        )

    def sample(
        self,
        observations: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self(observations)
        return _sample_tanh_gaussian(
            mean,
            log_std,
            generator=generator,
            deterministic=deterministic,
        )


class FeedForwardPriceCritic(nn.Module):
    """Scalar Q critic for one continuous pricing action."""

    def __init__(
        self,
        observation_dimension: int,
        action_dimension: int,
        hidden_dimensions: Sequence[int],
    ) -> None:
        super().__init__()
        self.network = multilayer_perceptron(
            observation_dimension + action_dimension,
            hidden_dimensions,
            1,
        )

    def forward(
        self, observations: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        return self.network(torch.cat([observations, actions], dim=-1))


class FeedForwardStrategyActor(nn.Module):
    """Categorical macro-regime actor."""

    def __init__(
        self,
        observation_dimension: int,
        hidden_dimensions: Sequence[int],
    ) -> None:
        super().__init__()
        self.network = multilayer_perceptron(
            observation_dimension, hidden_dimensions, 2
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.network(observations)

    def probabilities(
        self, observations: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self(observations)
        return F.softmax(logits, dim=-1), F.log_softmax(logits, dim=-1)

    def sample(
        self,
        observations: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        probabilities, log_probabilities = self.probabilities(observations)
        if deterministic:
            actions = probabilities.argmax(dim=-1, keepdim=True)
        else:
            actions = torch.multinomial(
                probabilities,
                num_samples=1,
                replacement=True,
                generator=generator,
            )
        selected_log_probability = log_probabilities.gather(1, actions)
        return actions, selected_log_probability, probabilities


class FeedForwardStrategyCritic(nn.Module):
    """Return Q-values for both categorical regime choices."""

    def __init__(
        self,
        observation_dimension: int,
        hidden_dimensions: Sequence[int],
    ) -> None:
        super().__init__()
        self.network = multilayer_perceptron(
            observation_dimension, hidden_dimensions, 2
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.network(observations)


class RecurrentPriceActor(nn.Module):
    """GRU price actor consuming controller-specific temporal context."""

    def __init__(
        self,
        recurrent_input_dimension: int,
        hidden_dimension: int,
        action_dimension: int,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            recurrent_input_dimension,
            hidden_dimension,
            batch_first=True,
        )
        self.mean = nn.Linear(hidden_dimension, action_dimension)
        self.log_standard_deviation = nn.Linear(
            hidden_dimension, action_dimension
        )

    def forward(
        self,
        recurrent_inputs: torch.Tensor,
        hidden_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features, hidden = self.gru(recurrent_inputs, hidden_state)
        return (
            self.mean(features),
            torch.clamp(
                self.log_standard_deviation(features),
                LOG_STD_MIN,
                LOG_STD_MAX,
            ),
            hidden,
        )

    def sample(
        self,
        recurrent_inputs: torch.Tensor,
        hidden_state: torch.Tensor | None = None,
        *,
        generator: torch.Generator | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std, hidden = self(recurrent_inputs, hidden_state)
        actions, log_probability = _sample_tanh_gaussian(
            mean,
            log_std,
            generator=generator,
            deterministic=deterministic,
        )
        return actions, log_probability, hidden


class RecurrentPriceCritic(nn.Module):
    """GRU history encoder followed by current-action scalar Q."""

    def __init__(
        self,
        recurrent_input_dimension: int,
        hidden_dimension: int,
        action_dimension: int,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            recurrent_input_dimension,
            hidden_dimension,
            batch_first=True,
        )
        self.q_head = multilayer_perceptron(
            hidden_dimension + action_dimension,
            (hidden_dimension,),
            1,
        )

    def forward(
        self,
        recurrent_inputs: torch.Tensor,
        actions: torch.Tensor,
        hidden_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features, hidden = self.gru(recurrent_inputs, hidden_state)
        return self.q_head(torch.cat([features, actions], dim=-1)), hidden


class RecurrentStrategyActor(nn.Module):
    """GRU categorical actor operating only at macro boundaries."""

    def __init__(
        self, recurrent_input_dimension: int, hidden_dimension: int
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            recurrent_input_dimension,
            hidden_dimension,
            batch_first=True,
        )
        self.logits = nn.Linear(hidden_dimension, 2)

    def forward(
        self,
        recurrent_inputs: torch.Tensor,
        hidden_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features, hidden = self.gru(recurrent_inputs, hidden_state)
        return self.logits(features), hidden

    def probabilities(
        self,
        recurrent_inputs: torch.Tensor,
        hidden_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, hidden = self(recurrent_inputs, hidden_state)
        return (
            F.softmax(logits, dim=-1),
            F.log_softmax(logits, dim=-1),
            hidden,
        )

    def sample(
        self,
        recurrent_inputs: torch.Tensor,
        hidden_state: torch.Tensor | None = None,
        *,
        generator: torch.Generator | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        probabilities, log_probabilities, hidden = self.probabilities(
            recurrent_inputs, hidden_state
        )
        flat_probabilities = probabilities.reshape(-1, 2)
        if deterministic:
            flat_actions = flat_probabilities.argmax(
                dim=-1, keepdim=True
            )
        else:
            flat_actions = torch.multinomial(
                flat_probabilities,
                num_samples=1,
                replacement=True,
                generator=generator,
            )
        actions = flat_actions.reshape(*probabilities.shape[:-1], 1)
        selected = log_probabilities.gather(-1, actions)
        return actions, selected, probabilities, hidden


class RecurrentStrategyCritic(nn.Module):
    """GRU critic returning both categorical macro Q-values."""

    def __init__(
        self, recurrent_input_dimension: int, hidden_dimension: int
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            recurrent_input_dimension,
            hidden_dimension,
            batch_first=True,
        )
        self.q_head = nn.Linear(hidden_dimension, 2)

    def forward(
        self,
        recurrent_inputs: torch.Tensor,
        hidden_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features, hidden = self.gru(recurrent_inputs, hidden_state)
        return self.q_head(features), hidden


class SharedOpponentHistoryEncoder(nn.Module):
    """One period-level opponent encoder shared by all OE-RSAC tasks."""

    def __init__(
        self,
        input_dimension: int = 27,
        hidden_dimension: int = 128,
        embedding_dimension: int = 32,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_dimension, hidden_dimension, batch_first=True
        )
        self.embedding_head = nn.Linear(
            hidden_dimension, embedding_dimension
        )

    def forward(
        self,
        inputs: torch.Tensor,
        hidden_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features, hidden = self.gru(inputs, hidden_state)
        return torch.tanh(self.embedding_head(features)), hidden


class OpponentControlPredictor(nn.Module):
    """Auxiliary next-opponent-control prediction head."""

    def __init__(
        self,
        embedding_dimension: int = 32,
        action_dimension: int = 5,
        hidden_dimension: int = 128,
    ) -> None:
        super().__init__()
        self.network = multilayer_perceptron(
            embedding_dimension + action_dimension,
            (hidden_dimension,),
            3,
        )

    def forward(
        self,
        embeddings: torch.Tensor,
        effective_actions: torch.Tensor,
    ) -> torch.Tensor:
        return torch.tanh(
            self.network(torch.cat([embeddings, effective_actions], dim=-1))
        )
