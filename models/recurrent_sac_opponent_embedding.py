"""Recurrent Soft Actor-Critic with a learned opponent embedding.

The agent consumes padded episode sequences produced by ``EpisodeReplayBuffer``
or ``CurriculumSequenceReplayBuffer``. Every temporal loss is mask-aware, so
padding never affects optimization.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


Tensor = torch.Tensor


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    """Average values over valid timesteps only."""
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


class OpponentEncoder(nn.Module):
    """Encode interaction history and predict the opponent's next action."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        opponent_action_dim: int,
        hidden_dim: int = 128,
        opponent_embedding_dim: int = 32,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.opponent_action_dim = opponent_action_dim
        self.opponent_embedding_dim = opponent_embedding_dim

        input_dim = obs_dim + action_dim + 1 + opponent_action_dim
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.embedding_head = nn.Linear(hidden_dim, opponent_embedding_dim)
        self.prediction_head = nn.Linear(opponent_embedding_dim, opponent_action_dim)

    def forward(
        self,
        interaction_sequence: Tensor,
        hidden: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return timestep embeddings, final embedding, predictions, and hidden state."""
        recurrent_output, next_hidden = self.gru(interaction_sequence, hidden)
        z_seq = self.embedding_head(recurrent_output)
        opponent_prediction = self.prediction_head(z_seq)

        if mask is None:
            z_last = z_seq[:, -1]
        else:
            lengths = mask.squeeze(-1).sum(dim=1).long().clamp_min(1)
            batch_indices = torch.arange(z_seq.shape[0], device=z_seq.device)
            z_last = z_seq[batch_indices, lengths - 1]
        return z_seq, z_last, opponent_prediction, next_hidden

    @staticmethod
    def prediction_loss(
        predictions: Tensor,
        opponent_actions: Tensor,
        mask: Tensor,
    ) -> Tensor:
        """Predict opponent action at ``t`` from pre-decision history at ``t``."""
        squared_error = (predictions - opponent_actions).pow(2)
        return (squared_error * mask).sum() / (
            mask.sum().clamp_min(1.0) * predictions.shape[-1]
        )


class RecurrentActor(nn.Module):
    """GRU Gaussian policy supporting sequence training and online hidden state."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        opponent_embedding_dim: int,
        hidden_dim: int = 128,
        log_std_min: float = -10.0,
        log_std_max: float = 1.0,

    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.gru = nn.GRU(obs_dim + opponent_embedding_dim, hidden_dim, batch_first=True)
        # self.head = nn.Sequential(
        #     nn.LayerNorm(hidden_dim),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.SiLU(),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.SiLU(),
        # )
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)


  

    def forward(
        self,
        obs_sequence: Tensor,
        opponent_embeddings: Tensor,
        hidden: Optional[Tensor] = None,
        deterministic: bool = False,
        epsilon: float = 1e-6,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        inputs = torch.cat([obs_sequence, opponent_embeddings], dim=-1)

        recurrent_output, next_hidden = self.gru(inputs, hidden)
        
        # recurrent_output = self.head(recurrent_output)
        
        mean = self.mean_head(recurrent_output)
        raw_log_std = self.log_std_head(recurrent_output)
        log_std = raw_log_std.clamp(self.log_std_min, self.log_std_max)
        self.last_mean = mean.detach()
        self.last_raw_log_std = raw_log_std.detach()
        self.last_log_std = log_std.detach()
        self.last_std = log_std.exp().detach()

        distribution = Normal(mean, log_std.exp())
        pre_tanh = mean if deterministic else distribution.rsample()
        
        squashed = torch.tanh(pre_tanh)
        action = squashed 
        mean_action = torch.tanh(mean) 

        log_prob = distribution.log_prob(pre_tanh)
        correction = torch.log(
            (1.0 - squashed.pow(2)) + epsilon
        )
        log_prob = (log_prob - correction).sum(dim=-1, keepdim=True)
        return action, log_prob, mean_action, next_hidden


class _RecurrentQNetwork(nn.Module):
    """Encode belief history separately, then evaluate a candidate action."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        opponent_embedding_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        history_input_dim = obs_dim + action_dim + opponent_embedding_dim
        head_input_dim = hidden_dim + action_dim
        head_hidden_dim = max(hidden_dim // 2, 1)
        self.gru = nn.GRU(history_input_dim, hidden_dim, batch_first=True)
        self.head = nn.Sequential(
            nn.LayerNorm(head_input_dim),
            nn.Linear(head_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, head_hidden_dim),
            nn.ReLU(),
            nn.Linear(head_hidden_dim, 1),
        )

    def forward(
        self,
        obs_sequence: Tensor,
        prev_action_sequence: Tensor,
        action_sequence: Tensor,
        opponent_embeddings: Tensor,
        hidden: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        history_inputs = torch.cat(
            [obs_sequence, prev_action_sequence, opponent_embeddings], dim=-1
        )
        recurrent_output, next_hidden = self.gru(history_inputs, hidden)
        q_inputs = torch.cat([recurrent_output, action_sequence], dim=-1)
        return self.head(q_inputs), next_hidden


class RecurrentCritic(nn.Module):
    """Twin Q-functions with action-independent recurrent belief states."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        opponent_embedding_dim: int,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.q1_network = _RecurrentQNetwork(
            obs_dim, action_dim, opponent_embedding_dim, hidden_dim
        )
        self.q2_network = _RecurrentQNetwork(
            obs_dim, action_dim, opponent_embedding_dim, hidden_dim
        )

    def forward(
        self,
        obs_sequence: Tensor,
        prev_action_sequence: Tensor,
        action_sequence: Tensor,
        opponent_embeddings: Tensor,
        hidden_states: Optional[
            Tuple[Optional[Tensor], Optional[Tensor]]
        ] = None,
        return_hidden: bool = False,
    ):
        """Evaluate current actions from history encoded with previous actions."""
        q1_hidden, q2_hidden = hidden_states or (None, None)
        q1, next_q1_hidden = self.q1_network(
            obs_sequence,
            prev_action_sequence,
            action_sequence,
            opponent_embeddings,
            q1_hidden,
        )
        q2, next_q2_hidden = self.q2_network(
            obs_sequence,
            prev_action_sequence,
            action_sequence,
            opponent_embeddings,
            q2_hidden,
        )
        if return_hidden:
            return q1, q2, (next_q1_hidden, next_q2_hidden)
        return q1, q2


class RecurrentSACOpponentEmbeddingAgent:
    """Recurrent SAC agent trained from masked episode sequence batches."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        opponent_action_dim: int,
        opponent_embedding_dim: int = 32,
        encoder_hidden_dim: int = 128,
        actor_hidden_dim: int = 128,
        critic_hidden_dim: int = 128,
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        lr_encoder: float = 3e-4,
        lr_alpha: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        auto_alpha: bool = True,
        target_entropy: Optional[float] = None,
        opponent_aux_loss_weight: float = 1.0,
        log_std_min: float = -10.0,
        log_std_max: float = 1.0,
        action_low: float | Sequence[float] = -1.0,
        action_high: float | Sequence[float] = 1.0,
        grad_clip_norm: Optional[float] = 10.0,
        device: Optional[str] = None,
        replay_buffer=None,
        min_episodes_before_update: Optional[int] = None,
    ) -> None:
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.opponent_action_dim = opponent_action_dim
        self.gamma = gamma
        self.tau = tau
        self.auto_alpha = auto_alpha
        self.target_entropy = -float(action_dim) if target_entropy is None else target_entropy
        self.opponent_aux_loss_weight = opponent_aux_loss_weight
        self.grad_clip_norm = grad_clip_norm
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.replay_buffer = replay_buffer
        self._minimum_episodes_was_defaulted = min_episodes_before_update is None
        default_minimum = getattr(replay_buffer, "batch_size", 1)
        self.min_episodes_before_update = (
            default_minimum
            if min_episodes_before_update is None
            else min_episodes_before_update
        )
        if self.min_episodes_before_update <= 0:
            raise ValueError("min_episodes_before_update must be positive")

        self.opponent_encoder = OpponentEncoder(
            obs_dim, action_dim, opponent_action_dim,
            encoder_hidden_dim, opponent_embedding_dim,
        ).to(self.device)
        self.actor = RecurrentActor(
            obs_dim, action_dim, opponent_embedding_dim, actor_hidden_dim,
            log_std_min, log_std_max,
        ).to(self.device)
        self.critic = RecurrentCritic(
            obs_dim, action_dim, opponent_embedding_dim, critic_hidden_dim
        ).to(self.device)
        self.target_critic = RecurrentCritic(
            obs_dim, action_dim, opponent_embedding_dim, critic_hidden_dim
        ).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_critic.requires_grad_(False)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.encoder_optimizer = torch.optim.Adam(
            self.opponent_encoder.parameters(), lr=lr_encoder
        )

        initial_log_alpha = float(np.log(alpha))
        self.log_alpha = torch.tensor(
            [initial_log_alpha], device=self.device, requires_grad=auto_alpha
        )
        self.alpha_optimizer = (
            torch.optim.Adam([self.log_alpha], lr=lr_alpha) if auto_alpha else None
        )
        self._fixed_alpha = float(alpha)
        self.encoder_hidden: Optional[Tensor] = None
        self.actor_hidden: Optional[Tensor] = None
        self._last_policy_stats: Optional[Dict[str, np.ndarray]] = None
        self.total_updates = 0

    @property
    def alpha(self) -> float:
        return float(self.log_alpha.exp().detach().item()) if self.auto_alpha else self._fixed_alpha

    def reset_hidden(self) -> None:
        """Reset online recurrent state at the beginning of an episode."""
        self.encoder_hidden = None
        self.actor_hidden = None

    def select_action(
        self,
        obs,
        prev_action=None,
        prev_reward=None,
        opponent_action=None,
        deterministic: bool = False,
    ) -> np.ndarray:
        """Select one online action and advance encoder/actor hidden states."""
        obs_tensor = self._online_vector(obs, self.obs_dim, "obs")
        action_tensor = self._online_vector(
            np.zeros(self.action_dim) if prev_action is None else prev_action,
            self.action_dim,
            "prev_action",
        )
        opponent_tensor = self._online_vector(
            np.zeros(self.opponent_action_dim) if opponent_action is None else opponent_action,
            self.opponent_action_dim,
            "opponent_action",
        )
        reward_value = 0.0 if prev_reward is None else float(prev_reward)
        reward_tensor = torch.tensor(
            [[[reward_value]]], dtype=torch.float32, device=self.device
        )
        interaction = torch.cat(
            [obs_tensor, action_tensor, reward_tensor, opponent_tensor], dim=-1
        )

        with torch.no_grad():
            z_seq, _, _, self.encoder_hidden = self.opponent_encoder(
                interaction, self.encoder_hidden
            )
            action, _, mean_action, self.actor_hidden = self.actor(
                obs_tensor, z_seq, self.actor_hidden, deterministic=deterministic
            )
        selected = mean_action if deterministic else action
        self._last_policy_stats = {
            "mean": self.actor.last_mean[0, -1].cpu().numpy(),
            "std": self.actor.last_std[0, -1].cpu().numpy(),
            "raw_log_std": self.actor.last_raw_log_std[0, -1].cpu().numpy(),
            "log_std": self.actor.last_log_std[0, -1].cpu().numpy(),
            "action": selected[0, 0].cpu().numpy(),
        }
        return selected[0, 0].cpu().numpy()

    def _online_vector(self, value, expected_dim: int, name: str) -> Tensor:
        array = np.asarray(value, dtype=np.float32).reshape(-1)
        if array.size != expected_dim:
            raise ValueError(f"{name} must contain {expected_dim} values")
        return torch.as_tensor(array, device=self.device).view(1, 1, expected_dim)

    def update(self) -> Optional[Dict[str, float]]:
        """Sample sequence replay internally once enough episodes are available."""
        if self.replay_buffer is None:
            raise RuntimeError("update() requires an injected replay_buffer")
        if len(self.replay_buffer) < self.min_episodes_before_update:
            return None
        return self.update_from_batch(self.replay_buffer.sample())

    def attach_replay_buffer(self, replay_buffer) -> None:
        """Attach sequence replay and adopt its batch-size readiness default."""
        self.replay_buffer = replay_buffer
        if self._minimum_episodes_was_defaulted:
            self.min_episodes_before_update = replay_buffer.batch_size

    def update_from_batch(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Perform one masked recurrent SAC update from a sequence batch."""
        tensors = self._prepare_batch(batch)
        obs = tensors["obs"]
        actions = tensors["actions"]
        rewards = tensors["rewards"]
        next_obs = tensors["next_obs"]
        dones = tensors["dones"]
        opponent_actions = tensors["opponent_actions"]
        mask = tensors["mask"]

        # At state t the policy may only use information available before
        # action_t. The post-transition context is reserved for state t+1.
        prev_actions = self._shift_sequence(actions)
        prev_rewards = self._shift_sequence(rewards)
        prev_opponent_actions = self._shift_sequence(opponent_actions)
        current_interaction = torch.cat(
            [obs, prev_actions, prev_rewards, prev_opponent_actions], dim=-1
        )
        next_interaction = torch.cat(
            [next_obs, actions, rewards, opponent_actions], dim=-1
        )

        z_current, _, predictions, _ = self.opponent_encoder(
            current_interaction, mask=mask
        )
        prediction_loss = self.opponent_encoder.prediction_loss(
            predictions, opponent_actions, mask
        )

        with torch.no_grad():
            z_next, _, _, _ = self.opponent_encoder(next_interaction, mask=mask)
            z_next = z_next.detach()
            next_actions, next_log_prob, _, _ = self.actor(next_obs, z_next)
            # At next_obs[t], replay action[t] is the previous action.
            target_q1, target_q2 = self.target_critic(
                next_obs, actions, next_actions, z_next
            )
            target_q = torch.minimum(target_q1, target_q2)
            alpha_tensor = self._alpha_tensor().detach()
            target = rewards + (1.0 - dones) * self.gamma * (
                target_q - alpha_tensor * next_log_prob
            )

        q1, q2 = self.critic(obs, prev_actions, actions, z_current)
        critic_loss = _masked_mean((q1 - target).pow(2), mask) + _masked_mean(
            (q2 - target).pow(2), mask
        )
        encoder_critic_loss = critic_loss + (
            self.opponent_aux_loss_weight * prediction_loss
        )

        self.critic_optimizer.zero_grad(set_to_none=True)
        self.encoder_optimizer.zero_grad(set_to_none=True)
        encoder_critic_loss.backward()
        self._clip_gradients(self.critic.parameters())
        self._clip_gradients(self.opponent_encoder.parameters())
        self.critic_optimizer.step()
        self.encoder_optimizer.step()

        with torch.no_grad():
            actor_z, _, _, _ = self.opponent_encoder(
                current_interaction, mask=mask
            )
        self.critic.requires_grad_(False)
        new_actions, log_prob, _, _ = self.actor(obs, actor_z)
        actor_q1, actor_q2 = self.critic(
            obs, prev_actions, new_actions, actor_z
        )
        q_min = torch.minimum(actor_q1, actor_q2)
        actor_loss = _masked_mean(
            self._alpha_tensor().detach() * log_prob - q_min, mask
        )
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self._clip_gradients(self.actor.parameters())
        self.actor_optimizer.step()
        self.critic.requires_grad_(True)

        alpha_loss = torch.zeros((), device=self.device)
        if self.auto_alpha and self.alpha_optimizer is not None:
            alpha_loss = -_masked_mean(
                self.log_alpha * (log_prob.detach() + self.target_entropy), mask
            )
            self.alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_optimizer.step()

        self.soft_update()
        self.total_updates += 1
        total_loss = (
            critic_loss.detach()
            + actor_loss.detach()
            + self.opponent_aux_loss_weight * prediction_loss.detach()
            + alpha_loss.detach()
        )
        return {
            "critic_loss": float(critic_loss.detach().item()),
            "actor_loss": float(actor_loss.detach().item()),
            "opponent_prediction_loss": float(prediction_loss.detach().item()),
            "alpha_loss": float(alpha_loss.detach().item()),
            "alpha": self.alpha,
            "total_loss": float(total_loss.item()),
        }

    def get_policy_stats(self, obs=None) -> Dict[str, np.ndarray]:
        """Return distribution statistics from the latest online action."""
        if self._last_policy_stats is None:
            zeros = np.zeros(self.action_dim, dtype=np.float32)
            return {
                "mean": zeros.copy(),
                "std": zeros.copy(),
                "raw_log_std": zeros.copy(),
                "log_std": zeros.copy(),
                "action": zeros.copy(),
            }
        return {
            name: values.copy()
            for name, values in self._last_policy_stats.items()
        }

    def get_current_lrs(self) -> Dict[str, float]:
        """Expose optimizer rates expected by the shared curriculum logger."""
        rates = {
            "actor_lr": self.actor_optimizer.param_groups[0]["lr"],
            "critic_lr": self.critic_optimizer.param_groups[0]["lr"],
            "encoder_lr": self.encoder_optimizer.param_groups[0]["lr"],
        }
        if self.alpha_optimizer is not None:
            rates["alpha_lr"] = self.alpha_optimizer.param_groups[0]["lr"]
        return rates

    def get_info(self) -> Dict[str, Any]:
        """Return concise architecture and optimization configuration."""
        return {
            "gamma": self.gamma,
            "tau": self.tau,
            "alpha": self.alpha,
            "auto_alpha": self.auto_alpha,
            "target_entropy": self.target_entropy,
            "opponent_action_dim": self.opponent_action_dim,
            "opponent_aux_loss_weight": self.opponent_aux_loss_weight,
            "min_episodes_before_update": self.min_episodes_before_update,
        }

    def _prepare_batch(self, batch: Dict[str, Any]) -> Dict[str, Tensor]:
        required = {
            "obs": self.obs_dim,
            "actions": self.action_dim,
            "rewards": 1,
            "next_obs": self.obs_dim,
            "dones": 1,
            "opponent_actions": self.opponent_action_dim,
            "mask": 1,
        }
        tensors: Dict[str, Tensor] = {}
        shape_prefix = None
        for name, feature_dim in required.items():
            if name not in batch:
                raise KeyError(f"Missing sequence batch field: {name}")
            tensor = torch.as_tensor(
                batch[name], dtype=torch.float32, device=self.device
            )
            if tensor.ndim != 3 or tensor.shape[-1] != feature_dim:
                raise ValueError(
                    f"batch[{name!r}] must have shape (B, T, {feature_dim})"
                )
            if shape_prefix is None:
                shape_prefix = tensor.shape[:2]
            elif tensor.shape[:2] != shape_prefix:
                raise ValueError("All sequence batch fields must share (B, T)")
            tensors[name] = tensor
        tensors["mask"] = (tensors["mask"] > 0).float()
        return tensors

    @staticmethod
    def _shift_sequence(values: Tensor) -> Tensor:
        """Build a pre-decision sequence by shifting right and zero-prefixing."""
        previous_values = torch.zeros_like(values)
        previous_values[:, 1:] = values[:, :-1]
        return previous_values

    def _alpha_tensor(self) -> Tensor:
        if self.auto_alpha:
            return self.log_alpha.exp()
        return torch.tensor(self._fixed_alpha, device=self.device)

    def _clip_gradients(self, parameters) -> None:
        if self.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(parameters, self.grad_clip_norm)

    def soft_update(self) -> None:
        """Polyak-average critic parameters into the target critic."""
        with torch.no_grad():
            for target_parameter, parameter in zip(
                self.target_critic.parameters(), self.critic.parameters()
            ):
                target_parameter.mul_(1.0 - self.tau).add_(parameter, alpha=self.tau)

    def save(self, path: str | Path) -> None:
        """Persist model, optimizer, entropy, and update state."""
        torch.save({
            "opponent_encoder": self.opponent_encoder.state_dict(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "encoder_optimizer": self.encoder_optimizer.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "alpha_optimizer": (
                self.alpha_optimizer.state_dict() if self.alpha_optimizer else None
            ),
            "log_alpha": self.log_alpha.detach(),
            "fixed_alpha": self._fixed_alpha,
            "total_updates": self.total_updates,
        }, str(path))

    def load(self, path: str | Path) -> None:
        """Restore a checkpoint into an identically configured agent."""
        checkpoint = torch.load(str(path), map_location=self.device)
        self.opponent_encoder.load_state_dict(checkpoint["opponent_encoder"])
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.target_critic.load_state_dict(checkpoint["target_critic"])
        self.encoder_optimizer.load_state_dict(checkpoint["encoder_optimizer"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        if self.alpha_optimizer is not None and checkpoint["alpha_optimizer"] is not None:
            self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer"])
        with torch.no_grad():
            self.log_alpha.copy_(checkpoint["log_alpha"].to(self.device))
        self._fixed_alpha = float(checkpoint.get("fixed_alpha", self._fixed_alpha))
        self.total_updates = int(checkpoint.get("total_updates", 0))
        self.reset_hidden()
