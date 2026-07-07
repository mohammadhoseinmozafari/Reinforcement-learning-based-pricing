from typing import List, Tuple
import numpy as np
from train.utils import evaluate_agent, save_checkpoint

from env.factory import EnvironmentFactory
from models.buffer import BaseReplayBuffer
from models.sac import SAC
from train.curriculum import CurriculumConfig , OpponentCurriculumScheduler
from train.logger import CurriculumTrainingLogger
from train.config import TrainingConfig
from train.metrics import TrainingMetrics

class CurriculumTrainer:


    def __init__(
        self,
        config: TrainingConfig,
        curriculum_config: CurriculumConfig,
        env_factory: EnvironmentFactory,
        base_env,
        env,
        replay_buffer: BaseReplayBuffer,
        agent: SAC,
    ) -> None:
        self.config = config
        self.curriculum_config = curriculum_config
        self.env_factory = env_factory
        self.base_env = base_env
        self.env = env
        self.replay_buffer = replay_buffer
        self.agent = agent
    

    def warmup (self, env , replay_buffer: BaseReplayBuffer, steps:int , seed: int) -> None:
        
        state, _ = env.reset(seed = seed)
    
        for _ in range (steps):

            action = env.action_space.sample()
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            replay_buffer.push(state, action, reward, next_state, done)

            state = next_state if not done else env.reset()[0]

    def run_episode(self, env, agent, metrics: TrainingMetrics) -> Tuple[float , List, List]:
        """Run a single episode, return (episode_reward, critic_losses, actor_losses)."""
        state, _ = env.reset()
        metrics.reset_episode()

        episode_reward = 0.0
        episode_critic_loss: List = []
        episode_actor_loss: List = []

        for _ in range(self.config.episode_length):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.replay_buffer.push(state, action, reward, next_state, done)

            for _ in range(self.config.updates_per_step):
                update_metrics = agent.update()
                if update_metrics is not None:
                    episode_critic_loss.append(update_metrics['critic_loss'])
                    episode_actor_loss.append(update_metrics['actor_loss'])

            metrics.record_step(info)
            episode_reward += float(reward)
            state = next_state

            if done:
                break

        return episode_reward, episode_critic_loss, episode_actor_loss

    def train(self):
        np.random.seed(self.config.seed)

        curriculum = OpponentCurriculumScheduler(self.curriculum_config)
        logger = CurriculumTrainingLogger(
            self.curriculum_config, verbose=self.config.verbose
        )
        env_factory = self.env_factory
        base_env = self.base_env
        env = self.env
        replay_buffer = self.replay_buffer
        agent = self.agent

        logger.print_training_header()
        logger.log_replay_buffer(replay_buffer)
        logger.log_agent_config(agent)

        metrics = TrainingMetrics()
        logger.log_warmup_start(self.config.warmup_steps)
        self.warmup(
            env, agent.replay_buffer, self.config.warmup_steps, self.config.seed
        )
        logger.log_start_training()

        completed_episodes = 0
        for episode in range(self.config.num_episodes):
            current_stage = curriculum.current_stage

            if current_stage.opponent_type == "mixed":
                assert current_stage.opponent_types
                opponent_type = np.random.choice(current_stage.opponent_types)
                base_env.close()
                base_env, env = env_factory.create_environment(
                    config=self.config, opponent_type=opponent_type
                )
                agent.replay_buffer.set_stage(opponent_type)

            episode_reward, critic_losses, actor_losses = self.run_episode(
                env, agent, metrics
            )
            completed_episodes = episode + 1
            metrics.end_episode(episode_reward)
            avg_critic = float(np.mean(critic_losses) if critic_losses else 0)
            avg_actor = float(np.mean(actor_losses) if actor_losses else 0)
            current_alpha = agent.alpha

            if critic_losses:
                metrics.critic_losses.append(avg_critic)
                metrics.actor_losses.append(avg_actor)
                metrics.alphas.append(current_alpha)

            curriculum.step(avg_critic, avg_actor, current_alpha)

            if completed_episodes % self.config.eval_freq == 0:
                eval_reward, policy_stats = evaluate_agent(
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

            # Stage limits are episode-based and therefore independent of
            # evaluation frequency.
            new_opponent = curriculum.advance()
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
                    agent.replay_buffer.set_stage(opponent_type)
                    logger.log_replay_buffer_stage_change(
                        agent.replay_buffer.current_stage
                    )
                    logger.log_replay_buffer(agent.replay_buffer)
                    logger.log_warmup_new_opponent(opponent_type)
                    self.warmup(
                        env=env,
                        replay_buffer=agent.replay_buffer,
                        steps=self.config.warmup_steps,
                        seed=self.config.seed,
                    )

            if completed_episodes % self.config.save_freq == 0:
                save_checkpoint(
                    agent, metrics, self.config, completed_episodes
                )

            if curriculum.is_complete:
                break

        save_checkpoint(
            agent,
            metrics,
            self.config,
            completed_episodes,
            final=True,
        )
        env.close()
        return agent, metrics
