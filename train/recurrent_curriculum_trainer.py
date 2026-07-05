"""Curriculum trainer for recurrent SAC with complete-episode replay."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Tuple

import numpy as np

from config.constants import (
    PRICE_BBP_NEW_MAX,
    PRICE_BBP_NEW_MIN,
    PRICE_BBP_OLD_MAX,
    PRICE_BBP_OLD_MIN,
    PRICE_UNIFORM_MAX,
    PRICE_UNIFORM_MIN,
)
from models.buffer import CurriculumSequenceReplayBuffer
from models.recurrent_sac_opponent_embedding import (
    RecurrentSACOpponentEmbeddingAgent,
)
from train.config import TrainingConfig
from train.curriculum import CurriculumConfig, OpponentCurriculumScheduler
from train.logger import CurriculumTrainingLogger
from train.metrics import TrainingMetrics

if TYPE_CHECKING:
    from env.factory import EnvironmentFactory


class RecurrentCurriculumTrainer:
    """Train a recurrent agent from episode sequences across curriculum stages."""

    def __init__(
        self,
        config: TrainingConfig,
        curriculum_config: CurriculumConfig,
        env_factory: EnvironmentFactory,
        base_env,
        env,
        replay_buffer: CurriculumSequenceReplayBuffer,
        agent: RecurrentSACOpponentEmbeddingAgent,
    ) -> None:
        self.config = config
        self.curriculum_config = curriculum_config
        self.env_factory = env_factory
        self.base_env = base_env
        self.env = env
        self.replay_buffer = replay_buffer
        self.agent = agent

        if agent.replay_buffer is None:
            agent.attach_replay_buffer(replay_buffer)
        elif agent.replay_buffer is not replay_buffer:
            raise ValueError("agent and trainer must use the same replay buffer")

    @staticmethod
    def _normalize(value: float, minimum: float, maximum: float) -> float:
        normalized = 2.0 * (float(value) - minimum) / (maximum - minimum) - 1.0
        return float(np.clip(normalized, -1.0, 1.0))

    def _extract_opponent_action(self, info: Dict) -> np.ndarray:
        """Extract opponent prices and normalize them to the agent action scale."""
        if "opponent_action" in info:
            actions = np.asarray(
                info["opponent_action"], dtype=np.float32
            ).reshape(-1)
            if actions.size == 1 and self.agent.opponent_action_dim > 1:
                actions = np.repeat(actions, self.agent.opponent_action_dim)
            if actions.size != self.agent.opponent_action_dim:
                raise ValueError(
                    "opponent_action must contain "
                    f"{self.agent.opponent_action_dim} values"
                )
            return actions

        price_keys = (
            "opponent_price_uniform",
            "opponent_price_new",
            "opponent_price_old",
        )
        if all(key in info for key in price_keys):
            return np.asarray([
                self._normalize(
                    info[price_keys[0]], PRICE_UNIFORM_MIN, PRICE_UNIFORM_MAX
                ),
                self._normalize(
                    info[price_keys[1]], PRICE_BBP_NEW_MIN, PRICE_BBP_NEW_MAX
                ),
                self._normalize(
                    info[price_keys[2]], PRICE_BBP_OLD_MIN, PRICE_BBP_OLD_MAX
                ),
            ], dtype=np.float32)

        for key in ("opponent_price", "competitor_price"):
            if key not in info:
                continue
            prices = np.asarray(info[key], dtype=np.float32).reshape(-1)
            if prices.size == 1:
                uniform = self._normalize(
                    prices[0], PRICE_UNIFORM_MIN, PRICE_UNIFORM_MAX
                )
                return np.full(3, uniform, dtype=np.float32)
            if prices.size == 3:
                return np.asarray([
                    self._normalize(prices[0], PRICE_UNIFORM_MIN, PRICE_UNIFORM_MAX),
                    self._normalize(prices[1], PRICE_BBP_NEW_MIN, PRICE_BBP_NEW_MAX),
                    self._normalize(prices[2], PRICE_BBP_OLD_MIN, PRICE_BBP_OLD_MAX),
                ], dtype=np.float32)
            raise ValueError(
                f"{key} must be a scalar or contain three price heads"
            )

        supported = (
            "opponent_action, opponent_price, competitor_price, or "
            "opponent_price_uniform/new/old"
        )
        raise KeyError(f"Cannot extract opponent action; expected {supported}")

    def warmup(
        self,
        env,
        replay_buffer: CurriculumSequenceReplayBuffer,
        steps: int,
        seed: int,
    ) -> None:
        """Collect exactly ``steps`` random transitions as episodic replay."""
        if steps <= 0:
            return

        collected = 0
        next_seed = seed
        while collected < steps:
            state, _ = env.reset(seed=next_seed)
            next_seed += 1
            builder = replay_buffer.create_episode_builder()

            while collected < steps and len(builder) < self.config.episode_length:
                action = env.action_space.sample()
                next_state, reward, terminated, truncated, info = env.step(action)
                done = bool(terminated or truncated)
                builder.append(
                    state,
                    action,
                    reward,
                    next_state,
                    done,
                    self._extract_opponent_action(info),
                )
                collected += 1
                state = next_state
                if done:
                    break

            if len(builder):
                replay_buffer.push(builder.build())

        self.agent.reset_hidden()

    def run_episode(
        self,
        env,
        agent: RecurrentSACOpponentEmbeddingAgent,
        metrics: TrainingMetrics,
    ) -> Tuple[float, List[float], List[float]]:
        """Collect one episode, update from prior episodes, then store it."""
        state, _ = env.reset()
        metrics.reset_episode()
        agent.reset_hidden()
        builder = self.replay_buffer.create_episode_builder()

        episode_reward = 0.0
        critic_losses: List[float] = []
        actor_losses: List[float] = []
        previous_action = np.zeros(agent.action_dim, dtype=np.float32)
        previous_reward = 0.0
        previous_opponent_action = np.zeros(
            agent.opponent_action_dim, dtype=np.float32
        )

        try:
            for _ in range(self.config.episode_length):
                action = agent.select_action(
                    state,
                    prev_action=previous_action,
                    prev_reward=previous_reward,
                    opponent_action=previous_opponent_action,
                )
                next_state, reward, terminated, truncated, info = env.step(action)
                done = bool(terminated or truncated)
                opponent_action = self._extract_opponent_action(info)
                builder.append(
                    state, action, reward, next_state, done, opponent_action
                )

                for _ in range(self.config.updates_per_step):
                    update_metrics = agent.update()
                    if update_metrics is not None:
                        critic_losses.append(update_metrics["critic_loss"])
                        actor_losses.append(update_metrics["actor_loss"])

                metrics.record_step(info)
                episode_reward += float(reward)
                state = next_state
                previous_action = np.asarray(action, dtype=np.float32)
                previous_reward = float(reward)
                previous_opponent_action = opponent_action
                if done:
                    break
        finally:
            agent.reset_hidden()

        if len(builder):
            self.replay_buffer.push(builder.build())

        return episode_reward, critic_losses, actor_losses

    @staticmethod
    def _empty_policy_stats(action_dim: int = 3) -> Dict[str, Dict[str, float]]:
        heads = ("uniform", "new", "old")
        return {
            head: {
                "mean": 0.0,
                "std": 0.0,
                "raw_log_std": 0.0,
                "log_std": 0.0,
                "action": 0.0,
            }
            for head in heads[:action_dim]
        }

    def evaluate_recurrent_agent(
        self,
        env,
        agent: RecurrentSACOpponentEmbeddingAgent,
        num_episodes: int,
        max_steps: int,
    ) -> Tuple[float, Dict[str, Dict[str, float]]]:
        """Evaluate deterministically with hidden state isolated per episode."""
        if num_episodes <= 0:
            raise ValueError("num_episodes must be positive")

        total_reward = 0.0
        stat_names = ("mean", "std", "raw_log_std", "log_std", "action")
        samples = {name: [] for name in stat_names}

        for _ in range(num_episodes):
            state, _ = env.reset()
            agent.reset_hidden()
            previous_action = np.zeros(agent.action_dim, dtype=np.float32)
            previous_reward = 0.0
            previous_opponent_action = np.zeros(
                agent.opponent_action_dim, dtype=np.float32
            )
            episode_reward = 0.0
            try:
                for _ in range(max_steps):
                    action = agent.select_action(
                        state,
                        prev_action=previous_action,
                        prev_reward=previous_reward,
                        opponent_action=previous_opponent_action,
                        deterministic=True,
                    )
                    get_policy_stats = getattr(agent, "get_policy_stats", None)
                    if callable(get_policy_stats):
                        try:
                            policy_stats = get_policy_stats()
                        except (AttributeError, KeyError, TypeError, ValueError):
                            policy_stats = {}
                        if all(name in policy_stats for name in stat_names):
                            values = {
                                name: np.asarray(
                                    policy_stats[name], dtype=float
                                ).reshape(-1)
                                for name in stat_names
                            }
                            if all(value.size == 3 for value in values.values()):
                                for name, value in values.items():
                                    samples[name].append(value)

                    next_state, reward, terminated, truncated, info = env.step(action)
                    opponent_action = self._extract_opponent_action(info)
                    episode_reward += float(reward)
                    state = next_state
                    previous_action = np.asarray(action, dtype=np.float32)
                    previous_reward = float(reward)
                    previous_opponent_action = opponent_action
                    if terminated or truncated:
                        break
            finally:
                agent.reset_hidden()
            total_reward += episode_reward

        if not samples["action"]:
            return total_reward / num_episodes, self._empty_policy_stats()

        averages = {
            name: np.mean(np.stack(values), axis=0)
            for name, values in samples.items()
        }
        heads = ("uniform", "new", "old")
        policy_stats = {
            head: {
                name: float(averages[name][index]) for name in stat_names
            }
            for index, head in enumerate(heads)
        }
        return total_reward / num_episodes, policy_stats

    def train(self):
        """Run recurrent episodic training with the feed-forward curriculum flow."""
        from train.utils import save_checkpoint

        np.random.seed(self.config.seed)
        curriculum = OpponentCurriculumScheduler(self.curriculum_config)
        logger = CurriculumTrainingLogger(
            self.curriculum_config, verbose=self.config.verbose
        )
        env_factory = self.env_factory
        base_env = self.base_env
        env = self.env
        agent = self.agent

        logger.print_training_header()
        logger.log_replay_buffer(self.replay_buffer)
        logger.log_agent_config(agent)

        metrics = TrainingMetrics()
        logger.log_warmup_start(self.config.warmup_steps)
        self.warmup(
            env, self.replay_buffer, self.config.warmup_steps, self.config.seed
        )
        logger.log_start_training()

        for episode in range(self.config.num_episodes):
            current_stage = curriculum.current_stage
            if current_stage.opponent_type == "mixed":
                assert current_stage.opponent_types
                opponent_type = str(np.random.choice(current_stage.opponent_types))
                base_env.close()
                base_env, env = env_factory.create_environment(
                    config=self.config, opponent_type=opponent_type
                )
                self.replay_buffer.set_stage(opponent_type)
                agent.reset_hidden()

            episode_reward, critic_losses, actor_losses = self.run_episode(
                env, agent, metrics
            )
            metrics.end_episode(episode_reward)
            has_updates = bool(critic_losses)
            avg_critic = float(np.mean(critic_losses)) if has_updates else 0.0
            avg_actor = float(np.mean(actor_losses)) if actor_losses else 0.0

            if has_updates:
                metrics.critic_losses.append(avg_critic)
                metrics.actor_losses.append(avg_actor)
                metrics.alphas.append(agent.alpha)
                # Replay-fill episodes intentionally do not advance the scheduler.
                curriculum.step(avg_critic, avg_actor, agent.alpha)

            if (episode + 1) % self.config.eval_freq == 0:
                eval_reward, policy_stats = self.evaluate_recurrent_agent(
                    base_env,
                    agent,
                    self.config.eval_episodes,
                    self.config.episode_length,
                )
                metrics.eval_rewards.append(eval_reward)
                logger.log_episode_progress(
                    episode,
                    metrics,
                    agent,
                    eval_reward,
                    curriculum,
                    policy_stats,
                    self.config,
                )

                new_opponent = curriculum.advance() if has_updates else None
                if new_opponent is not None:
                    logger.log_stage_transition(new_opponent)
                    base_env.close()
                    if new_opponent.opponent_type == "mixed":
                        assert new_opponent.opponent_types
                        logger.log_mixed_stage_entry(new_opponent.opponent_types)
                    else:
                        opponent_type = new_opponent.opponent_type
                        base_env, env = env_factory.create_environment(
                            config=self.config, opponent_type=opponent_type
                        )
                        self.replay_buffer.set_stage(opponent_type)
                        agent.reset_hidden()
                        logger.log_replay_buffer_stage_change(
                            self.replay_buffer.current_stage
                        )
                        logger.log_replay_buffer(self.replay_buffer)
                        logger.log_warmup_new_opponent(opponent_type)
                        self.warmup(
                            env,
                            self.replay_buffer,
                            self.config.warmup_steps,
                            self.config.seed,
                        )

                if (episode + 1) % self.config.save_freq == 0:
                    save_checkpoint(agent, metrics, self.config, episode + 1)

        save_checkpoint(
            agent, metrics, self.config, self.config.num_episodes, final=True
        )
        env.close()
        return agent, metrics
