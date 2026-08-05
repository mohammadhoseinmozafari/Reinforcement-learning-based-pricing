"""Shared construction of all universal-pricing agent architectures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from env.pricing_contracts import AgentArchitecture, PricingAgent
from models.recurrent_sac_pricing import (
    OpponentEmbeddingRecurrentSACPricingAgent,
    RecurrentSACPricingAgent,
)
from models.sac_pricing import SACPricingAgent
from models.universal_pricing_replay import UniversalPricingReplayBuffer
from models.universal_pricing_sequence_replay import (
    UniversalPricingSequenceReplayBuffer,
)


@dataclass(frozen=True)
class UniversalPricingAgentComponents:
    """One universal agent paired with its architecture-appropriate replay."""

    agent: PricingAgent
    replay_buffer: Any
    batch_size: int
    is_recurrent: bool


class UniversalPricingAgentFactory:
    """Build SAC, RSAC, or OE-RSAC from a resolved profile and seed bundle."""

    @staticmethod
    def create(
        profile: Any,
        run_seed_bundle: Any,
        *,
        device: str | torch.device = "cpu",
    ) -> UniversalPricingAgentComponents:
        architecture = AgentArchitecture(profile.architecture)
        seed_arguments = {
            "network_initialization_seed": (
                run_seed_bundle.network_initialization_seed
            ),
            "exploration_seed": run_seed_bundle.exploration_seed,
            "torch_cpu_seed": run_seed_bundle.torch_cpu_seed,
            "torch_cuda_seed": run_seed_bundle.torch_cuda_seed,
            "device": device,
        }
        if architecture is AgentArchitecture.SAC:
            config = profile.sac_pricing_config
            agent = SACPricingAgent(config, **seed_arguments)
            replay = UniversalPricingReplayBuffer(
                config.replay_capacity,
                run_seed_bundle.replay_sampling_seed,
            )
            return UniversalPricingAgentComponents(
                agent, replay, config.batch_size, False
            )

        config = profile.recurrent_pricing_config
        agent_class = (
            RecurrentSACPricingAgent
            if architecture is AgentArchitecture.RSAC
            else OpponentEmbeddingRecurrentSACPricingAgent
        )
        agent = agent_class(config, **seed_arguments)
        replay = UniversalPricingSequenceReplayBuffer(
            capacity_episodes=config.episode_replay_capacity,
            learning_sequence_length=config.learning_sequence_length,
            burn_in_length=config.burn_in_length,
            batch_size=config.batch_size,
            replay_sampling_seed=run_seed_bundle.replay_sampling_seed,
        )
        return UniversalPricingAgentComponents(
            agent, replay, config.batch_size, True
        )
