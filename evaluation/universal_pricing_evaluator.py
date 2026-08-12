"""Locked deterministic evaluation for universal-pricing checkpoints."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Sequence

import numpy as np

from env.pricing_contracts import PricingActionCodec, PricingRegime
from env.pricing_factory import UniversalPricingEnvironmentFactory
from models.universal_pricing_agents import UniversalPricingAgentFactory
from models.universal_pricing_replay import UniversalPricingTransition
from train.universal_pricing_protocol import (
    ExperimentCoordinate,
    SeedDeriver,
    UniversalPricingProtocolConfig,
)


class UniversalPricingEvaluationRepository:
    """Atomically write raw episodes and aggregate evaluation summaries."""

    @staticmethod
    def _atomic_json(path: Path, value: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(value, stream, sort_keys=True, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        except BaseException:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise

    def write(
        self,
        output_directory: Path,
        suite: str,
        episodes: list[dict[str, Any]],
        summary: dict[str, Any],
    ) -> tuple[Path, Path]:
        output_directory.mkdir(parents=True, exist_ok=True)
        episodes_path = output_directory / f"{suite}_episodes.jsonl"
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{episodes_path.name}.",
            suffix=".tmp",
            dir=output_directory,
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                for episode in episodes:
                    stream.write(json.dumps(episode, sort_keys=True) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, episodes_path)
        except BaseException:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise
        summary_path = output_directory / f"{suite}_summary.json"
        self._atomic_json(summary_path, summary)
        return episodes_path, summary_path


class UniversalPricingEvaluator:
    """Evaluate a final checkpoint on balanced pairs from committed seeds."""

    def __init__(
        self,
        protocol: UniversalPricingProtocolConfig,
        coordinate: ExperimentCoordinate,
        *,
        device: str = "cpu",
    ) -> None:
        self.protocol = protocol
        self.coordinate = coordinate
        self.device = device

    def evaluate_checkpoint(
        self,
        checkpoint_path: str | Path,
        environment_seeds: Sequence[int],
        *,
        suite: str,
        output_directory: str | Path | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        profile = self.protocol.agent_profiles[
            self.coordinate.agent_architecture
        ]
        components = UniversalPricingAgentFactory.create(
            profile,
            self.protocol.run_seed_bundle(
                self.coordinate.training_seed_index
            ),
            device=self.device,
        )
        components.agent.load(checkpoint_path)
        factory = UniversalPricingEnvironmentFactory(self.protocol)
        episodes: list[dict[str, Any]] = []
        inference_times: list[float] = []
        for seed_index, environment_seed in enumerate(environment_seeds):
            bundle = SeedDeriver.derive_run_bundle(int(environment_seed))
            environment = factory.create_environment_with_run_seed(
                self.coordinate, bundle
            )
            for pair_episode_index in (0, 1):
                observation, reset_info = environment.reset(
                    options={"episode_index": pair_episode_index}
                )
                components.agent.reset_recurrent_state()
                rewards: list[float] = []
                profits: list[float] = []
                opponent_profits: list[float] = []
                regimes: list[int] = []
                regime_changes = 0
                uniform_prices: list[float] = []
                new_prices: list[float] = []
                old_prices: list[float] = []
                market_shares: list[float] = []
                retention_rates: list[float] = []
                for _ in range(environment.episode_length):
                    started = time.perf_counter()
                    action = components.agent.select_action(
                        observation, deterministic=True
                    )
                    inference_times.append(time.perf_counter() - started)
                    (
                        next_observation,
                        reward,
                        terminated,
                        truncated,
                        info,
                    ) = environment.step(PricingActionCodec.to_gym(action))
                    transition = UniversalPricingTransition.from_environment_step(
                        observation=observation,
                        reward=reward,
                        next_observation=next_observation,
                        terminated=terminated,
                        truncated=truncated,
                        info=info,
                    )
                    components.agent.observe_transition(transition)
                    rewards.append(float(reward))
                    profits.append(float(info["raw_agent_profit"]))
                    opponent_profits.append(float(info["raw_opponent_profit"]))
                    regimes.append(int(info["agent_regime"]))
                    regime_changes += int(info["regime_changed"])
                    firm = environment.market.firms[0]
                    uniform_prices.append(float(firm.uniform_price))
                    new_prices.append(float(firm.price_new))
                    old_prices.append(float(firm.price_old))
                    market_shares.append(float(firm.market_share))
                    retention_rates.append(float(firm.retention_rate))
                    observation = next_observation
                    if terminated or truncated:
                        break
                episodes.append(
                    {
                        "suite": suite,
                        "evaluation_seed_index": seed_index,
                        "evaluation_seed": int(environment_seed),
                        "pair_episode_index": pair_episode_index,
                        "opponent_family": reset_info["opponent_family"],
                        "opponent_policy_name": reset_info[
                            "opponent_policy_name"
                        ],
                        "normalized_reward_total": float(np.sum(rewards)),
                        "raw_agent_profit_total": float(np.sum(profits)),
                        "raw_opponent_profit_total": float(
                            np.sum(opponent_profits)
                        ),
                        "profit_advantage_total": float(
                            np.sum(profits) - np.sum(opponent_profits)
                        ),
                        "bbp_period_fraction": float(
                            np.mean(
                                np.asarray(regimes)
                                == int(PricingRegime.BBP)
                            )
                        ),
                        "regime_change_count": regime_changes,
                        "mean_uniform_price": float(
                            np.mean(uniform_prices)
                        ),
                        "mean_bbp_new_price": float(np.mean(new_prices)),
                        "mean_bbp_old_price": float(np.mean(old_prices)),
                        "mean_bbp_price_spread": float(
                            np.mean(np.asarray(old_prices) - new_prices)
                        ),
                        "mean_market_share": float(
                            np.mean(market_shares)
                        ),
                        "mean_retention_rate": float(
                            np.mean(retention_rates)
                        ),
                        "checkpoint_sha256": checkpoint_hash,
                    }
                )
            environment.close()
        summary = {
            "suite": suite,
            "architecture": self.coordinate.agent_architecture.value,
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_hash,
            "seed_count": len(environment_seeds),
            "episode_count": len(episodes),
            "mean_normalized_reward_total": float(
                np.mean(
                    [item["normalized_reward_total"] for item in episodes]
                )
            ),
            "mean_raw_agent_profit_total": float(
                np.mean([item["raw_agent_profit_total"] for item in episodes])
            ),
            "mean_raw_opponent_profit_total": float(
                np.mean(
                    [item["raw_opponent_profit_total"] for item in episodes]
                )
            ),
            "mean_profit_advantage_total": float(
                np.mean([item["profit_advantage_total"] for item in episodes])
            ),
            "mean_bbp_period_fraction": float(
                np.mean([item["bbp_period_fraction"] for item in episodes])
            ),
            "mean_regime_change_count": float(
                np.mean([item["regime_change_count"] for item in episodes])
            ),
            "mean_uniform_price": float(
                np.mean([item["mean_uniform_price"] for item in episodes])
            ),
            "mean_bbp_new_price": float(
                np.mean([item["mean_bbp_new_price"] for item in episodes])
            ),
            "mean_bbp_old_price": float(
                np.mean([item["mean_bbp_old_price"] for item in episodes])
            ),
            "mean_bbp_price_spread": float(
                np.mean(
                    [item["mean_bbp_price_spread"] for item in episodes]
                )
            ),
            "mean_market_share": float(
                np.mean([item["mean_market_share"] for item in episodes])
            ),
            "mean_retention_rate": float(
                np.mean([item["mean_retention_rate"] for item in episodes])
            ),
            "mean_inference_seconds": float(np.mean(inference_times)),
        }
        summary["by_opponent_family"] = {
            family: {
                "episode_count": len(family_episodes),
                "mean_raw_agent_profit_total": float(
                    np.mean(
                        [
                            item["raw_agent_profit_total"]
                            for item in family_episodes
                        ]
                    )
                ),
                "mean_profit_advantage_total": float(
                    np.mean(
                        [
                            item["profit_advantage_total"]
                            for item in family_episodes
                        ]
                    )
                ),
                "mean_bbp_period_fraction": float(
                    np.mean(
                        [
                            item["bbp_period_fraction"]
                            for item in family_episodes
                        ]
                    )
                ),
                "mean_market_share": float(
                    np.mean(
                        [item["mean_market_share"] for item in family_episodes]
                    )
                ),
            }
            for family in ("uniform", "bbp")
            for family_episodes in [
                [
                    item
                    for item in episodes
                    if item["opponent_family"] == family
                ]
            ]
        }
        if output_directory is not None:
            UniversalPricingEvaluationRepository().write(
                Path(output_directory), suite, episodes, summary
            )
        return episodes, summary
