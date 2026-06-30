from typing import List, Tuple
import numpy as np
from train.uniform_training.uniform_training import evaluate_agent, save_checkpoint

from env import EnvironmentFactory
from models import (
    BaseReplayBuffer,
    SAC,
    CurriculumReplayBuffer,
    FixedRewardNormalizer
    )
from train import CurriculumConfig, CurriculumTrainingLogger, OpponentCurriculumScheduler
from train.config import TrainingConfig
from train.metrics import TrainingMetrics

class CurriculumTrainer :


    def __init__(self,  config : TrainingConfig, curriculum_config: CurriculumConfig) -> None:
        self.config = config
        self.curriculum_config = curriculum_config
        


    def create_replay_buffer (self)  -> CurriculumReplayBuffer:
        
        return CurriculumReplayBuffer(
            capacity= self.config.buffer_size,
            batch_size=self.config.batch_size,
            curriculum= self.curriculum_config.curriculum
        )
    
    def create_agent (self , env , replay_buffer : BaseReplayBuffer) -> SAC:
        
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        agent = SAC(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=self.config.hidden_dim,
            lr_actor=self.config.lr_actor,
            lr_critic=self.config.lr_critic,
            lr_alpha=self.config.lr_alpha,
            gamma=self.config.gamma,
            tau=self.config.tau,
            alpha= self.config.alpha,
            auto_alpha=self.config.auto_alpha,
            replay_buffer= replay_buffer,
            target_entropy=self.config.target_entropy,
            log_std_min=self.config.log_std_min,
            log_std_max=self.config.log_std_max
        )

        return agent
    

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
            logger = CurriculumTrainingLogger(self.curriculum_config, verbose= self.config.verbose)
            env_factory = EnvironmentFactory(
                env_type= self.config.environment_type,
                reward_normalizer= FixedRewardNormalizer
            )



            current_opponent = curriculum.current_opponent
            

            base_env , env = env_factory.create_environment(
                config = self.config,
                opponent_type=current_opponent,
            )



            logger.print_training_header()
            
            # create replay buffer
            replay_buffer = self.create_replay_buffer ()
            logger.log_replay_buffer(replay_buffer)
                

            # Create SAC agent
            agent = self.create_agent(
                env= env,
                replay_buffer= replay_buffer
            )
            logger.log_agent_config(agent)

            metrics = TrainingMetrics()
            # =========================================
            # WARMUP: Random exploration
            # =========================================
            logger.log_warmup_start(self.config.warmup_steps)
            self.warmup(env, agent.replay_buffer, self.config.warmup_steps, self.config.seed)


            # =========================================
            # TRAINING LOOP
            # =========================================
            logger.log_start_training()

            num_episodes = self.config.num_episodes
            for episode in range(num_episodes):
                current_stage = curriculum.current_stage
                
                if current_stage.opponent_type == 'mixed':
                    assert current_stage.opponent_types
                    opponent_type = np.random.choice(current_stage.opponent_types)
                    base_env.close()
                    
                
                    base_env, env = env_factory.create_environment(
                        config= self.config,
                        opponent_type= opponent_type
                    )

                    agent.replay_buffer.set_stage(opponent_type)
                
                episode_reward, episode_critic_loss , episode_actor_loss = self.run_episode(
                    env,
                    agent,
                    metrics,
                )
                # End episode
                metrics.end_episode(episode_reward)
                avg_critic = float(np.mean(episode_critic_loss) if episode_critic_loss else 0)
                avg_actor = float(np.mean(episode_actor_loss) if episode_actor_loss else 0)
                current_alpha = agent.alpha
                
                if episode_critic_loss:
                    metrics.critic_losses.append(avg_critic)
                    metrics.actor_losses.append(avg_actor)
                    metrics.alphas.append(current_alpha)
                
                curriculum.step(
                    critic_loss=avg_critic,
                    actor_loss=avg_actor,
                    alpha= current_alpha
                )
                # =========================================
                # EVALUATION
                # =========================================
                if (episode + 1) % self.config.eval_freq == 0:
                    eval_reward, policy_stats = evaluate_agent(base_env, agent, self.config.eval_episodes, self.config.episode_length)
                    metrics.eval_rewards.append(eval_reward)

                    logger.log_episode_progress(episode, metrics, agent, eval_reward, curriculum, policy_stats , self.config)

                    new_opponent = curriculum.advance()
                    if new_opponent is not None:
                        logger.log_stage_transition(new_opponent)
                        
                        base_env.close()
                        if new_opponent.opponent_type =="mixed":
                            assert new_opponent.opponent_types
                            logger.log_mixed_stage_entry(new_opponent.opponent_types)
                        else:
                            opponent_type = new_opponent.opponent_type

                            base_env , env = env_factory.create_environment (
                                config= self.config,
                                opponent_type= opponent_type
                            )
                            agent.replay_buffer.set_stage(
                                opponent_type
                            )
                            
                            logger.log_replay_buffer_stage_change(agent.replay_buffer.current_stage)
                            logger.log_replay_buffer(agent.replay_buffer)
                            
                            
                        if new_opponent.opponent_type != 'mixed':
                            

                            logger.log_warmup_new_opponent(new_opponent.opponent_type)
                            self.warmup (env= env, replay_buffer= agent.replay_buffer, steps = self.config.warmup_steps, seed= self.config.seed)
                        
                    
                    if (episode + 1) % self.config.save_freq == 0:
                        save_checkpoint(agent, metrics, self.config, episode + 1)
                # Final save
            save_checkpoint(agent, metrics, self.config, self.config.num_episodes, final=True)
        
            env.close()
            return agent, metrics


