"""Stateful human-versus-agent game logic for the simulation dashboard."""

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Dict, Tuple

from env.opponent_policies import OpponentObservation, OpponentPolicy
from env.pricing_env import PricingEnv
from env.type import EnvironmentType
from models.buffer import ReplayBuffer
from models.sac import SAC


STRATEGY_TO_ENVIRONMENT = {
    "uniform": EnvironmentType.UNIFORM_PRICING,
    "bbp": EnvironmentType.BBP_PRICING,
}


class HumanPricingPolicy(OpponentPolicy):
    """Opponent policy whose posted prices are supplied by dashboard input."""

    def __init__(self, strategy: str, seed: int = 42) -> None:
        if strategy not in STRATEGY_TO_ENVIRONMENT:
            raise ValueError(f"Unknown human strategy: {strategy}")
        super().__init__(regime=0 if strategy == "uniform" else 1, seed=seed)
        self._uniform_price = 2.5
        self._price_new = 2.0
        self._price_old = 3.0

    def set_prices(self, uniform_price: float, price_new: float, price_old: float) -> None:
        """Store the human's next bounded market action."""
        self._uniform_price = self.bounds.clip_uniform(uniform_price)
        self._price_new = self.bounds.clip_bbp_new(price_new)
        self._price_old = self.bounds.clip_bbp_old(max(price_old, self._price_new))

    def get_uniform_price(self, observation: OpponentObservation) -> float:
        return self._uniform_price

    def get_bbp_prices(self, observation: OpponentObservation) -> Tuple[float, float]:
        return self._price_new, self._price_old

    @property
    def prices(self) -> Dict[str, float]:
        return {
            "uniform_price": float(self._uniform_price),
            "price_new": float(self._price_new),
            "price_old": float(self._price_old),
        }


def resolve_run_directory(
    project_root: Path,
    agent_strategy: str,
    human_strategy: str,
    run: int = 1,
) -> Path:
    """Resolve the trained run matching the selected strategy pairing."""
    if agent_strategy not in STRATEGY_TO_ENVIRONMENT:
        raise ValueError(f"Unknown agent strategy: {agent_strategy}")
    if human_strategy not in STRATEGY_TO_ENVIRONMENT:
        raise ValueError(f"Unknown human strategy: {human_strategy}")
    return project_root / "experiments" / f"{agent_strategy}_vs_{human_strategy}" / "runs" / str(run)


def _load_agent(run_directory: Path, config: dict) -> SAC:
    checkpoint = run_directory / "sac_uniform_final.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Trained checkpoint not found: {checkpoint}")

    agent = SAC(
        state_dim=13,
        action_dim=3,
        replay_buffer=ReplayBuffer(max(int(config.get("batch_size", 256)), 1)),
        hidden_dim=int(config.get("hidden_dim", 32)),
        lr_actor=float(config.get("lr_actor", 5e-4)),
        lr_critic=float(config.get("lr_critic", 5e-4)),
        lr_alpha=float(config.get("lr_alpha", 1e-4)),
        gamma=float(config.get("gamma", 0.9)),
        tau=float(config.get("tau", 0.005)),
        alpha=float(config.get("alpha", 1.0)),
        auto_alpha=bool(config.get("auto_alpha", True)),
        target_entropy=float(config.get("target_entropy", -0.5)),
        log_std_min=float(config.get("log_std_min", -10.0)),
        log_std_max=float(config.get("log_std_max", 0.1)),
        batch_size=int(config.get("batch_size", 256)),
        device=config.get("device"),
    )
    agent.load(str(checkpoint))
    return agent


@dataclass
class SimulationGame:
    """One interactive episode between a human and a trained SAC agent."""

    agent_strategy: str
    human_strategy: str
    run_directory: Path
    config: dict
    env: PricingEnv
    agent: SAC
    human_policy: HumanPricingPolicy
    state: object
    history: list[dict] = field(default_factory=list)
    finished: bool = False

    @property
    def episode_length(self) -> int:
        return int(self.config["episode_length"])

    @property
    def step_number(self) -> int:
        return len(self.history)

    def step(self, uniform_price: float, price_new: float, price_old: float) -> dict:
        """Post the human price, let the agent respond, and clear one period."""
        if self.finished:
            raise RuntimeError("The episode is already complete")

        self.human_policy.set_prices(uniform_price, price_new, price_old)
        human_prices = self.human_policy.prices

        # PricingEnv observations include the opponent's currently posted prices.
        # Publish the human action before asking the trained agent to respond.
        self.env.market.firms[1].set_prices(**human_prices)
        agent_regime = 0 if self.agent_strategy == "uniform" else 1
        self.env.market.set_regimes(agent_regime, self.human_policy.regime)
        visible_state = self.env._get_observation()
        agent_action = self.agent.select_action(visible_state, deterministic=True)

        next_state, _, terminated, truncated, info = self.env.step(agent_action)
        self.state = next_state
        self.finished = bool(terminated or truncated)

        agent_firm = self.env.market.firms[0]
        human_firm = self.env.market.firms[1]
        record = {
            "step": self.step_number + 1,
            "agent_profit": float(agent_firm.last_period_profit),
            "user_profit": float(human_firm.last_period_profit),
            "agent_market_share": float(agent_firm.market_share),
            "user_market_share": float(human_firm.market_share),
            "agent_uniform_price": float(info["uniform_price"]),
            "agent_new_price": float(info["bbp_price_new"]),
            "agent_old_price": float(info["bbp_price_old"]),
            "user_uniform_price": float(info["opponent_price_uniform"]),
            "user_new_price": float(info["opponent_price_new"]),
            "user_old_price": float(info["opponent_price_old"]),
        }
        previous_agent = self.history[-1]["agent_cumulative_profit"] if self.history else 0.0
        previous_user = self.history[-1]["user_cumulative_profit"] if self.history else 0.0
        record["agent_cumulative_profit"] = previous_agent + record["agent_profit"]
        record["user_cumulative_profit"] = previous_user + record["user_profit"]
        self.history.append(record)
        return record


def create_game(
    project_root: Path,
    agent_strategy: str,
    human_strategy: str,
    run: int = 1,
) -> SimulationGame:
    """Load the matching checkpoint and initialize a fresh market episode."""
    run_directory = resolve_run_directory(project_root, agent_strategy, human_strategy, run)
    config_path = run_directory / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Training configuration not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as config_file:
        config = json.load(config_file)

    expected_environment = STRATEGY_TO_ENVIRONMENT[agent_strategy]
    if config.get("environment_type") != expected_environment.value:
        raise ValueError(
            f"Checkpoint strategy mismatch: expected {expected_environment.value}, "
            f"found {config.get('environment_type')}"
        )

    seed = int(config.get("seed", 42))
    human_policy = HumanPricingPolicy(human_strategy, seed=seed)
    env = PricingEnv(
        environment_type=expected_environment,
        opponent_policy=human_policy,
        num_consumers=int(config.get("num_consumers", 50)),
        episode_length=int(config.get("episode_length", 100)),
        seed=seed,
    )
    state, _ = env.reset(seed=seed)
    agent = _load_agent(run_directory, config)
    return SimulationGame(
        agent_strategy=agent_strategy,
        human_strategy=human_strategy,
        run_directory=run_directory,
        config=config,
        env=env,
        agent=agent,
        human_policy=human_policy,
        state=state,
    )
