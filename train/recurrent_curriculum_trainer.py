"""Curriculum trainer for recurrent SAC with complete-episode replay."""

from __future__ import annotations

import logging


import hashlib
import random

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

        self.rng = np.random.default_rng(self.config.seed)


        
        if agent.replay_buffer is None:
            agent.attach_replay_buffer(replay_buffer)
        elif agent.replay_buffer is not replay_buffer:
            raise ValueError("agent and trainer must use the same replay buffer")



    @staticmethod
    def _normalize(value: float, minimum: float, maximum: float) -> float:
        normalized = (2.0 * (float(value) - minimum) / (maximum - minimum)) - 1.0
        return float(normalized)

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

    def _sample_seed(self) -> int:
        """Draw a reproducible, fresh seed from the trainer RNG."""
        return int(self.rng.integers(0, 2**31 - 1))

    @staticmethod
    def _stable_int_hash(text: str) -> int:
        digest = hashlib.md5(text.encode("utf-8")).hexdigest()
        return int(digest[:8], 16)

    def _get_eval_seeds(self, opponent_type: str, n: int) -> List[int]:
        configured_base = getattr(self.config, "eval_seed", None)
        base = self.config.seed + 10_000 if configured_base is None else configured_base
        opponent_offset = self._stable_int_hash(str(opponent_type)) % 1_000_000
        return [int(base + opponent_offset + index) for index in range(n)]

    @staticmethod
    def _create_episode_builder(
        replay_buffer,
        episode_seed=None,
        opponent_type=None,
        stage_id=None,
    ):
        """Create a builder while remaining compatible with simpler buffers."""
        try:
            return replay_buffer.create_episode_builder(
                episode_seed=episode_seed,
                opponent_type=opponent_type,
                stage_id=stage_id,
            )
        except TypeError:
            try:
                return replay_buffer.create_episode_builder(
                    episode_seed=episode_seed,
                )
            except TypeError:
                return replay_buffer.create_episode_builder()

    def warmup(
        self,
        env,
        replay_buffer: CurriculumSequenceReplayBuffer,
        steps: int,
        seed: int,
        agent=None,
        random_action_prob: float = 1.0,
    ) -> None:
        """Collect reproducibly randomized warmup episodes."""
        if steps <= 0:
            return
        if not 0.0 <= random_action_prob <= 1.0:
            raise ValueError("random_action_prob must be in [0, 1]")

        rng = np.random.default_rng(seed)
        collected = 0
        while collected < steps:
            episode_seed = int(rng.integers(0, 2**31 - 1))
            state, _ = env.reset(seed=episode_seed)
            if hasattr(env.action_space, "seed"):
                env.action_space.seed(episode_seed)

            opponent_type = getattr(replay_buffer, "current_stage", None)
            stage_id = getattr(replay_buffer, "current_stage_id", None)
            builder = self._create_episode_builder(
                replay_buffer,
                episode_seed=episode_seed,
                opponent_type=opponent_type,
                stage_id=stage_id,
            )

            previous_action = np.zeros(self.agent.action_dim, dtype=np.float32)
            previous_reward = 0.0
            previous_opponent_action = np.zeros(
                self.agent.opponent_action_dim, dtype=np.float32
            )
            if agent is not None:
                agent.reset_hidden()

            try:
                while collected < steps and len(builder) < self.config.episode_length:
                    policy_action = None
                    if agent is not None:
                        policy_action = agent.select_action(
                            state,
                            prev_action=previous_action,
                            prev_reward=previous_reward,
                            opponent_action=previous_opponent_action,
                        )
                    use_random = agent is None or rng.random() < random_action_prob
                    action = env.action_space.sample() if use_random else policy_action
                    next_state, reward, terminated, truncated, info = env.step(action)
                    done = bool(terminated or truncated)
                    opponent_action = self._extract_opponent_action(info)
                    builder.append(
                        state, action, reward, next_state, done, opponent_action
                    )
                    collected += 1
                    state = next_state
                    previous_action = np.asarray(action, dtype=np.float32)
                    previous_reward = float(reward)
                    previous_opponent_action = opponent_action
                    if done:
                        break
            finally:
                if agent is not None:
                    agent.reset_hidden()

            if len(builder):
                replay_buffer.push_episode(builder.build())

        if agent is not None:
            agent.reset_hidden()

    def run_episode(
        self,
        env,
        agent: RecurrentSACOpponentEmbeddingAgent,
        metrics: TrainingMetrics,
    ) -> Tuple[float, List[float], List[float]]:
        """Collect one episode, update from prior episodes, then store it."""
        episode_seed = self._sample_seed()
        print(f"episode seed: {episode_seed} ")
        state, _ = env.reset(seed=episode_seed)

        metrics.reset_episode()

        agent.reset_hidden()

        builder = self._create_episode_builder(
            self.replay_buffer,
            episode_seed=episode_seed,
            opponent_type=getattr(self.replay_buffer, "current_stage", None),
            stage_id=getattr(self.replay_buffer, "current_stage_id", None),
        )

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
            self.replay_buffer.push_episode(builder.build())

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
        opponent_type: str = "unknown",
    ) -> Tuple[float, Dict[str, Dict[str, float]]]:
        """Evaluate deterministically with hidden state isolated per episode."""
        if num_episodes <= 0:
            raise ValueError("num_episodes must be positive")

        total_reward = 0.0
        stat_names = ("mean", "std", "raw_log_std", "log_std", "action")
        samples = {name: [] for name in stat_names}

        eval_seeds = self._get_eval_seeds(opponent_type, num_episodes)
        for episode_seed in eval_seeds:
            state, _ = env.reset(seed=episode_seed)
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
        random.seed(self.config.seed)
        try:
            import torch

            torch.manual_seed(self.config.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.config.seed)
        except ImportError:
            pass

        curriculum = OpponentCurriculumScheduler(self.curriculum_config)
        logger = CurriculumTrainingLogger(
            self.curriculum_config, verbose=self.config.verbose
        )
        env_factory = self.env_factory
        base_env = self.base_env
        env = self.env
        agent = self.agent
        
        logger.log_environment_config(base_env)
        logger.print_training_header()
        logger.log_replay_buffer(self.replay_buffer)
        logger.log_agent_config(agent)

        metrics = TrainingMetrics()
        logger.log_warmup_start(self.config.warmup_steps)
        initial_warmup_seed = self._sample_seed()
        self.warmup(
            env=env,
            replay_buffer=self.replay_buffer,
            steps=self.config.warmup_steps,
            seed=initial_warmup_seed,
            random_action_prob=1.0,
        )
        logger.log_start_training()

        opponent_type = curriculum.current_stage.opponent_type
        completed_episodes = 0
        for episode in range(self.config.num_episodes):
            current_stage = curriculum.current_stage
            if current_stage.opponent_type == "mixed":
                assert current_stage.opponent_types
                opponent_type = str(self.rng.choice(current_stage.opponent_types))
                base_env.close()
                base_env, env = env_factory.create_environment(
                    config=self.config, opponent_type=opponent_type
                )
                self.replay_buffer.set_stage(opponent_type)
                agent.reset_hidden()

            episode_reward, critic_losses, actor_losses = self.run_episode(
                env, agent, metrics
            )
            completed_episodes = episode + 1
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

            if completed_episodes % self.config.eval_freq == 0:
                eval_episode_count = (
                    getattr(self.config, "eval_seed_count", None)
                    or self.config.eval_episodes
                )
                eval_reward, policy_stats = self.evaluate_recurrent_agent(
                    base_env,
                    agent,
                    eval_episode_count,
                    self.config.episode_length,
                    opponent_type=opponent_type,
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

            # Replay-fill episodes do not count, but every episode that
            # produced recurrent updates checks the stage exit condition.
            new_opponent = curriculum.advance() if has_updates else None
            if new_opponent is not None:
                logger.log_stage_transition(new_opponent)
                base_env.close()
                if new_opponent.opponent_type == "mixed":
                    assert new_opponent.opponent_types
                    logger.log_mixed_stage_entry(new_opponent.opponent_types)
                    opponent_type = str(
                        self.rng.choice(new_opponent.opponent_types)
                    )
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
                stage_seed = self._sample_seed()
                self.warmup(
                    env=env,
                    replay_buffer=agent.replay_buffer,
                    steps=self.config.warmup_steps,
                    seed=stage_seed,
                    agent=agent,
                    random_action_prob=getattr(
                        self.config,
                        "stage_warmup_random_prob",
                        0.3,
                    ),
                )

            if completed_episodes % self.config.save_freq == 0:
                save_checkpoint(
                    agent, metrics, self.config, completed_episodes
                )

            if curriculum.is_complete:
                break

        save_checkpoint(
            agent, metrics, self.config, completed_episodes, final=True
        )
        env.close()
        return agent, metrics
