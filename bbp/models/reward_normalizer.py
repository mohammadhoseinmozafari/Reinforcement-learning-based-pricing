from typing import SupportsFloat

import gymnasium as gym
import numpy as np
from collections import deque

import gymnasium as gym
import numpy as np


class FixedRewardNormalizer(gym.RewardWrapper):
    """
    Fixed reward normalization for Hotelling pricing environment.

    Raw profit:
        reward = price * quantity

    Theoretical maximum:
        max_price * num_consumers

    Normalized reward:
        reward / max_step_profit
    """

    def __init__(
        self,
        env: gym.Env,
        max_price: float = 5.0,
        num_consumers: int = 100,
        clip_reward: bool = False,
    ):
        super().__init__(env)

        self.max_price = max_price
        self.num_consumers = num_consumers

        # Maximum possible profit in one step
        self.max_step_profit = max_price * num_consumers

        self.clip_reward = clip_reward

    def reward(self, reward):
        normalized = float(reward) / self.max_step_profit

        if self.clip_reward:
            normalized = np.clip(normalized, -1.0, 1.0)

        return float(normalized)

class EpisodeRewardNormalizer(gym.Wrapper):
    """
    Normalize rewards within each episode using episode-level statistics.
    
    This is what production RL systems use BEFORE value normalization.
    It ensures that the agent sees consistent reward scales regardless of
    opponent configuration.
    
    Two strategies are provided:
    1. Running statistics (adapts within episode)
    2. Symmetric log transform (handles any scale gracefully)
    """
    
    def __init__(self, env, strategy='running', epsilon=1e-8, clip_range=10.0):
        """
        Args:
            env: Gym environment
            strategy: 'running' or 'log' or 'both'
            epsilon: Small constant to avoid division by zero
            clip_range: Clip normalized rewards to [-clip_range, clip_range]
        """
        super().__init__(env)
        self.strategy = strategy
        self.epsilon = epsilon
        self.clip_range = clip_range
        
        # For running statistics within episode
        
        self.episode_rewards = []
        self.reward_mean :float = 0.0
        self.reward_std  :float = 1.0
        self.step_count : float = 0
        
        # For log-based normalization
        self.log_scale  = 1.0
        
    def reset(self, **kwargs):
        """Reset episode-level statistics."""
        obs, info = self.env.reset(**kwargs)
        
        # Reset episode statistics
        self.episode_rewards = []
        self.reward_mean : float = 0.0
        self.reward_std = 1.0
        self.step_count = 0
        self.log_scale = 1.0
        
        return obs, info
    
    def step(self, action):
        """Execute step and normalize reward."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Store raw reward for diagnostics
        raw_reward = reward
        
        # Normalize based on chosen strategy
        if self.strategy == 'running':
            normalized_reward = self._normalize_running(reward)
        elif self.strategy == 'log':
            normalized_reward = self._normalize_log(reward)
        elif self.strategy == 'both':
            normalized_reward = self._normalize_both(reward)
        else:
            normalized_reward = reward
        
        # Add raw reward to info for monitoring
        if 'raw_reward' not in info:
            info['raw_reward'] = raw_reward
        info['normalized_reward'] = normalized_reward
        
        return obs, normalized_reward, terminated, truncated, info
    
    def _normalize_running(self, reward : SupportsFloat) -> SupportsFloat:
        """
        Normalize using running statistics within the episode.
        
        This is like batch normalization but within a single episode.
        Early in the episode, normalization is noisy but improves over time.
        """
        reward = float(reward)
        self.step_count += 1
        self.episode_rewards.append(reward)
        
        # Welford's online algorithm for running mean and variance
        if self.step_count == 1:
            self.reward_mean = reward
            self.reward_std = 1.0  # Can't compute std with one sample
        else:
            old_mean = self.reward_mean
            self.reward_mean += (reward - old_mean) / self.step_count
            
            # Update variance estimate
            if self.step_count == 2:
                self.reward_std = np.sqrt(
                    ((reward - old_mean)**2 + (self.episode_rewards[0] - self.reward_mean)**2) / 2
                )
            else:
                # Use exponential moving average for variance (more stable)
                diff = (reward - old_mean) * (reward - self.reward_mean)
                self.reward_std = np.sqrt(
                    ((self.step_count - 1) * self.reward_std**2 + diff) / self.step_count
                )
        
        # Normalize
        if self.reward_std > self.epsilon:
            normalized = (reward - self.reward_mean) / (self.reward_std + self.epsilon)
        else:
            normalized = reward - self.reward_mean
        
        # Clip to prevent extreme values
        normalized  = float(np.clip(normalized, -self.clip_range, self.clip_range))
        
        return normalized
    
    def _normalize_log(self, reward: SupportsFloat) -> SupportsFloat:
        """
        Normalize using symmetric log transform.
        
        This handles ANY reward scale gracefully without needing statistics.
        f(x) = sign(x) * log(1 + |x|)
        
        For x = 50: f(50) = log(51) ≈ 3.93
        For x = 5000: f(5000) = log(5001) ≈ 8.52
        For x = 50000: f(50000) = log(50001) ≈ 10.82
        
        It's a soft compression that works for any scale.
        """
        # Symmetric log transform
        reward = float(reward)
        sign = np.sign(reward)
        magnitude = np.log1p(np.abs(reward))  # log(1 + |x|)
        
        # Scale to maintain differentiation
        normalized = sign * magnitude
        
        # Clip just in case
        normalized = float(np.clip(normalized, -self.clip_range, self.clip_range))
        
        return normalized
    
    def _normalize_both(self, reward: SupportsFloat) -> SupportsFloat:
        """
        Combine log transform with running normalization.
        
        First compress with log, then normalize to zero mean unit variance.
        This is the most robust approach used in state-of-the-art systems.
        """
        reward = float(reward)
        # Step 1: Log compress to handle extreme scales
        log_reward = float(self._normalize_log(reward))
        
        # Step 2: Running normalization on log-compressed rewards
        self.step_count += 1
        self.episode_rewards.append(log_reward)
        
        # Update running statistics on log-compressed rewards
        old_mean = self.reward_mean
        self.reward_mean += (log_reward - old_mean) / self.step_count
        
        if self.step_count > 1:
            diff = (log_reward - old_mean) * (log_reward - self.reward_mean)
            self.reward_std = np.sqrt(
                ((self.step_count - 1) * self.reward_std**2 + diff) / self.step_count
            )
        
        # Normalize log-compressed reward
        if self.reward_std > self.epsilon:
            normalized = (log_reward - self.reward_mean) / (self.reward_std + self.epsilon)
        else:
            normalized = log_reward - self.reward_mean
        
        normalized = float(np.clip(normalized, -self.clip_range, self.clip_range))
        
        return normalized




