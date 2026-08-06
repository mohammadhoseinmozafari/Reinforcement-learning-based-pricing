"""Shape-agnostic hybrid SAC objectives shared by all universal agents."""

from __future__ import annotations

import torch


class HybridSACObjective:
    """Combine regime probabilities and conditional branch values exactly."""

    @staticmethod
    def soft_value(
        *,
        regime_probabilities: torch.Tensor,
        regime_log_probabilities: torch.Tensor,
        uniform_soft_q: torch.Tensor,
        bbp_soft_q: torch.Tensor,
        current_regime_one_hot: torch.Tensor,
        regime_decision_masks: torch.Tensor,
        regime_temperature: torch.Tensor,
    ) -> torch.Tensor:
        branches = torch.cat([uniform_soft_q, bbp_soft_q], dim=-1)
        eligible = (
            regime_probabilities
            * (
                branches
                - regime_temperature * regime_log_probabilities
            )
        ).sum(dim=-1, keepdim=True)
        locked = (current_regime_one_hot * branches).sum(
            dim=-1, keepdim=True
        )
        return (
            regime_decision_masks * eligible
            + (1.0 - regime_decision_masks) * locked
        )

    @staticmethod
    def actor_objective(
        *,
        regime_probabilities: torch.Tensor,
        regime_log_probabilities: torch.Tensor,
        uniform_branch_objective: torch.Tensor,
        bbp_branch_objective: torch.Tensor,
        current_regime_one_hot: torch.Tensor,
        regime_decision_masks: torch.Tensor,
        regime_temperature: torch.Tensor,
    ) -> torch.Tensor:
        branches = torch.cat(
            [uniform_branch_objective, bbp_branch_objective], dim=-1
        )
        eligible = (
            regime_probabilities
            * (
                branches
                + regime_temperature * regime_log_probabilities
            )
        ).sum(dim=-1, keepdim=True)
        locked = (current_regime_one_hot * branches).sum(
            dim=-1, keepdim=True
        )
        return (
            regime_decision_masks * eligible
            + (1.0 - regime_decision_masks) * locked
        )
