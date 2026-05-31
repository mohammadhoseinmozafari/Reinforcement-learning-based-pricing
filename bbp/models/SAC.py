"""
Soft Actor-Critic (SAC) Implementation
Based on "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor"
by Haarnoja et al. (2018)
"""
import gymnasium as gym

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
from collections import deque
import random


class ReplayBuffer:
    """Experience Replay Buffer for off-policy learning."""
    
    def __init__(self, capacity: int, recent_bias: float = 0.3):
        self.buffer = deque(maxlen=capacity)
        self.insertion_order = deque(maxlen=capacity)  # Track insertion time
        self.total_insertions = 0
        self.recent_bias = recent_bias  
    
    def push(self, state, action, reward, next_state, done):
        """Store a transition in the buffer."""     
        self.buffer.append((state, action, reward, next_state, done))
        self.insertion_order.append(self.total_insertions)
        self.total_insertions+=1
    
    def sample(self, batch_size: int):
        """Sample a batch of transitions."""
        if len(self.buffer)< batch_size:
            return self._sample_uniform(batch_size)
        
        insertion_times = np.array(self.insertion_order)
        max_time = insertion_times.max()
        time_diffs = max_time- insertion_times
        weights = np.exp(-self.recent_bias*time_diffs / len(self.buffer))
        weights = weights/weights.sum()

        indices = np.random.choice(len(self.buffer),
                                   size = batch_size,
                                   p = weights,
                                   replace=False)
        
        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32)
        )
    
    def _sample_uniform(self, batch_size) :
            batch = random.sample(self.buffer, batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)
        
            return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32)
        )

    def __len__(self):
        return len(self.buffer)


class Actor(nn.Module):
    """
    Stochastic Actor Network (Policy).
    Outputs mean and log_std for a Gaussian distribution over actions.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        log_std_min: float = -15,
        log_std_max: float = 0.0
    ):
        super(Actor, self).__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 128)
        self.mean = nn.Linear(128, action_dim)
        self.log_std = nn.Linear(128, action_dim)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for layer in [self.fc1, self.fc2]:
            nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
            nn.init.zeros_(layer.bias)
        
        # Output layers with smaller initialization
        nn.init.orthogonal_(self.mean.weight, gain=0.01)
        nn.init.zeros_(self.mean.bias)
        nn.init.orthogonal_(self.log_std.weight, gain=0.01)
        nn.init.zeros_(self.log_std.bias)
    
    def forward(self, state):
        """Forward pass to get mean and log_std."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def sample(self, state, epsilon=1e-6):
        """
        Sample action using reparameterization trick.
        Returns action, log_prob, and mean.
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        # Reparameterization trick
        normal = Normal(mean, std)
        z = normal.rsample()
        
        # Apply tanh squashing
        action = torch.tanh(z)
        
        # Compute log probability with correction for tanh squashing
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        return action, log_prob, mean
    
    def get_action(self, state, deterministic=False):
        """Get action for inference."""
        mean, log_std = self.forward(state)
        
        if deterministic:
            return torch.tanh(mean)
        
        std = log_std.exp()
        normal = Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)
        
        return action


class Critic(nn.Module):
    """
    Twin Q-Network (Critic).
    Uses two Q-networks to mitigate overestimation bias.
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(Critic, self).__init__()
        
        # Q1 network
        self.q1_fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.q1_fc2 = nn.Linear(hidden_dim, 128)
        self.q1_fc3 = nn.Linear(128, 64)
        self.q1_out = nn.Linear(64, 1)
        
        # Q2 network
        self.q2_fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.q2_fc2 = nn.Linear(hidden_dim, 128)
        self.q2_fc3 = nn.Linear(128, 64)
        self.q2_out = nn.Linear(64, 1)
        
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for layer in [self.q1_fc1, self.q1_fc2,self.q1_fc3, self.q2_fc1,  self.q2_fc2, self.q2_fc3]:
            nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
            nn.init.zeros_(layer.bias)
        
        for layer in [self.q1_out, self.q2_out]:
            nn.init.orthogonal_(layer.weight, gain=1.0)
            nn.init.zeros_(layer.bias)
    
    def forward(self, state, action):
        """Forward pass for both Q-networks."""
        x = torch.cat([state, action], dim=-1)
        
        # Q1
        q1 = F.relu(self.q1_fc1(x))
        q1 = F.relu(self.q1_fc2(q1))
        q1 = F.relu(self.q1_fc3(q1))
        q1 = self.q1_out(q1)
        
        # Q2
        q2 = F.relu(self.q2_fc1(x))
        q2 = F.relu(self.q2_fc2(q2))
        q2= F.relu(self.q2_fc3(q2))
        q2 = self.q2_out(q2)
        
        return q1, q2
    
    def q1(self, state, action):
        """Forward pass for Q1 only."""
        x = torch.cat([state, action], dim=-1)
        q1 = F.relu(self.q1_fc1(x))
        q1 = F.relu(self.q1_fc2(q1))
        q1= F.relu(self.q1_fc3(q1))
        q1 = self.q1_out(q1)
        return q1


class SAC:
    """
    Soft Actor-Critic Agent.
    
    Args:
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        action_scale: Scale factor for actions (max action value)
        hidden_dim: Hidden layer dimension
        lr_actor: Learning rate for actor
        lr_critic: Learning rate for critic
        lr_alpha: Learning rate for entropy coefficient
        gamma: Discount factor
        tau: Soft update coefficient for target networks
        alpha: Initial entropy coefficient (if auto_alpha is False)
        auto_alpha: Whether to automatically tune alpha
        target_entropy: Target entropy for automatic alpha tuning
        buffer_size: Replay buffer capacity
        batch_size: Training batch size
        device: Device to run on (cuda/cpu)
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_scale: float = 1.0,
        hidden_dim: int = 256,

        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        lr_alpha: float = 3e-4,
        gamma: float = 0.95,
        tau: float = 0.005,
        alpha: float = 0.2,
        auto_alpha: bool = True,
        target_entropy: Optional[float] = None,
        buffer_size: int = 1000000,
        grad_clip_norm : float = 1.0,
        batch_size: int = 256,
        lr_scheduler : Optional[str] = None,
        lr_scheduler_kwargs: Optional[Dict] = None,
        device: Optional[str] = None
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_scale = action_scale
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.auto_alpha = auto_alpha
        self.grad_clip_norm = grad_clip_norm
        self.lr_scheduler_type = lr_scheduler
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Initialize networks
        self.actor = Actor(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic = Critic(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim, hidden_dim).to(self.device)
        
        # Copy weights to target network
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        


        # Entropy coefficient (alpha)
        if auto_alpha:
            # Target entropy is -dim(A) by default
            self.target_entropy = target_entropy if target_entropy else -action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp().item()
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr_alpha)
        else:
            self.alpha = alpha
            self.log_alpha = None
            self.alpha_optimizer = None

        if lr_scheduler:
        # Learning rate schedulers
            print("Initializing Schedulers")
            self.actor_scheduler = self._create_scheduler(
                self.actor_optimizer, lr_scheduler, lr_scheduler_kwargs
            )
            self.critic_scheduler = self._create_scheduler(
                self.critic_optimizer, lr_scheduler, lr_scheduler_kwargs
            )
            self.alpha_scheduler = self._create_scheduler(
                self.alpha_optimizer, lr_scheduler, lr_scheduler_kwargs
            ) if self.alpha_optimizer is not None else None
            

        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Training info
        self.total_steps = 0

    def _create_scheduler(
        self, 
        optimizer: Optional[optim.Optimizer], 
        scheduler_type: Optional[str],
        kwargs: Optional[Dict]
    ):
        """Create a learning rate scheduler."""
        if optimizer is None or scheduler_type is None:
            return None
        
        kwargs = kwargs or {}
        
        if scheduler_type == "cosine":
            # Cosine annealing: smoothly decreases LR following cosine curve
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=kwargs.get("T_max", 5000),      # Number of steps for one cycle
                eta_min=kwargs.get("eta_min", 1e-6)      # Minimum LR
            )
        
        elif scheduler_type == "step":
            # Step decay: reduces LR by gamma every step_size steps
            return optim.lr_scheduler.StepLR(
                optimizer,
                step_size=kwargs.get("step_size", 20000),
                gamma=kwargs.get("step_gamma", 0.5)
            )
        
        elif scheduler_type == "exponential":
            # Exponential decay: multiplies LR by gamma every step
            return optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=kwargs.get("exp_gamma", 0.9999)
            )
  
        return None
    
    def select_action(self, state, deterministic=False):
        """
        Select action given state.
        
        Args:
            state: Current state
            deterministic: If True, use mean action (no sampling)
        
        Returns:
            action: Selected action scaled to action space
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action = self.actor.get_action(state, deterministic)
        
        action = action.cpu().numpy()[0]
        return action * self.action_scale
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer."""
        # Normalize action back to [-1, 1]
        action = action / self.action_scale
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def update(self):
        """
        Perform one update step.
        
        Returns:
            dict: Training metrics
        """
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # Update critic
        with torch.no_grad():
            next_actions, next_log_probs, _ = self.actor.sample(next_states)
            q1_target, q2_target = self.critic_target(next_states, next_actions)
            q_target = torch.min(q1_target, q2_target)
            target_value = rewards + (1 - dones) * self.gamma * (q_target - self.alpha * next_log_probs)
        
        q1, q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(q1, target_value) + F.mse_loss(q2, target_value)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip_norm)
        self.critic_optimizer.step()
        
        # Update actor
        new_actions, log_probs, _ = self.actor.sample(states)
        q1_new, q2_new = self.critic(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        
        actor_loss = (self.alpha * log_probs - q_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip_norm)
        self.actor_optimizer.step()
        
        # Update alpha (entropy coefficient)
        alpha_loss = None
        if self.auto_alpha and self.log_alpha is not None and self.alpha_optimizer is not None:
            alpha_loss = -(self.log_alpha * (log_probs.detach() + self.target_entropy)).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            # torch.nn.utils.clip_grad_norm_([self.log_alpha], self.grad_clip_norm)
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp().item()
        
        # Soft update target networks
        self._step_schedulers()
        self._soft_update()
        
        self.total_steps += 1
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha_loss': alpha_loss.item() if alpha_loss is not None else 0.0,
            'alpha': self.alpha,
            'q_value': q_new.mean().item()
        }
    
    def _step_schedulers(self):
        """Step all schedulers (except ReduceLROnPlateau which needs a metric)."""
        if self.actor_scheduler is not None :
            self.actor_scheduler.step()
        
        if self.critic_scheduler is not None :
            self.critic_scheduler.step()
        
        if self.alpha_scheduler is not None :
            self.alpha_scheduler.step()
    
    def _soft_update(self):
        """Soft update target networks."""
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def get_current_lrs(self) -> Dict[str, float]:
        """Get current learning rates."""
        lrs = {
            'actor_lr': self.actor_optimizer.param_groups[0]['lr'],
            'critic_lr': self.critic_optimizer.param_groups[0]['lr']
        }
        if self.alpha_optimizer is not None:
            lrs['alpha_lr'] = self.alpha_optimizer.param_groups[0]['lr']
        return lrs
    
    def save(self, path: str):
        """Save model checkpoint."""
       
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'log_alpha': self.log_alpha if self.auto_alpha else None,
            'alpha_optimizer_state_dict': self.alpha_optimizer.state_dict() if self.auto_alpha and self.alpha_optimizer is not None else None,
            'alpha': self.alpha,
            'total_steps': self.total_steps
        }, path)
        print(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        
        if self.auto_alpha and checkpoint['log_alpha'] is not None:
            self.log_alpha = checkpoint['log_alpha']
            if self.alpha_optimizer is not None and checkpoint['alpha_optimizer_state_dict'] is not None:
                self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
        
        self.alpha = checkpoint['alpha']
        self.total_steps = checkpoint['total_steps']
        
        print(f"Model loaded from {path}")
    
    def get_info(self):
        return {
            'gammma': self.gamma,
            'tau': self.tau,
            'alpha': self.alpha,
            'auto_alpha': self.auto_alpha,
            'target_entropy': self.target_entropy,

        }


def train_sac(
    env,
    agent: SAC,
    num_episodes: int = 500,
    max_steps: int = 200,
    warmup_steps: int = 1000,
    updates_per_step: int = 1,
    eval_freq: int = 10,
    eval_episodes: int = 5,
    verbose: bool = True
):
    """
    Train SAC agent on environment.
    
    Args:
        env: Gymnasium/Gym environment
        agent: SAC agent
        num_episodes: Number of training episodes
        max_steps: Maximum steps per episode
        warmup_steps: Random exploration steps before training
        updates_per_step: Number of gradient updates per environment step
        eval_freq: Evaluation frequency (episodes)
        eval_episodes: Number of evaluation episodes
        verbose: Print training progress
    
    Returns:
        dict: Training history
    """
    history = {
        'episode_rewards': [],
        'eval_rewards': [],
        'critic_losses': [],
        'actor_losses': [],
        'alphas': []
    }
    
    total_steps = 0
    
    # Warmup: random exploration
    if verbose:
        print(f"Warming up with {warmup_steps} random steps...")
    
    state, _ = env.reset()
    for _ in range(warmup_steps):
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        agent.store_transition(state, action, reward, next_state, done)
        
        if done:
            state, _ = env.reset()
        else:
            state = next_state
    
    if verbose:
        print("Starting training...")
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_critic_loss = []
        episode_actor_loss = []
        
        for step in range(max_steps):
            # Select action
            action = agent.select_action(state)
            
            # Environment step
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store transition
            agent.store_transition(state, action, reward, next_state, done)
            
            # Update agent
            for _ in range(updates_per_step):
                metrics = agent.update()
                if metrics is not None:
                    episode_critic_loss.append(metrics['critic_loss'])
                    episode_actor_loss.append(metrics['actor_loss'])
            
            episode_reward += reward
            total_steps += 1
            state = next_state
            
            if done:
                break
        
        history['episode_rewards'].append(episode_reward)
        
        if episode_critic_loss:
            history['critic_losses'].append(np.mean(episode_critic_loss))
            history['actor_losses'].append(np.mean(episode_actor_loss))
            history['alphas'].append(agent.alpha)
        
        # Evaluation
        if (episode + 1) % eval_freq == 0:
            eval_reward = evaluate(env, agent, eval_episodes, max_steps)
            history['eval_rewards'].append(eval_reward)
            
            if verbose:
                avg_reward = np.mean(history['episode_rewards'][-eval_freq:])
                print(f"Episode {episode + 1}/{num_episodes} | "
                      f"Avg Reward: {avg_reward:.2f} | "
                      f"Eval Reward: {eval_reward:.2f} | "
                      f"Alpha: {agent.alpha:.4f}")
    
    return history


def evaluate(env, agent: SAC, num_episodes: int = 5, max_steps: int = 200):
    """
    Evaluate agent performance.
    
    Args:
        env: Gymnasium/Gym environment
        agent: SAC agent
        num_episodes: Number of evaluation episodes
        max_steps: Maximum steps per episode
    
    Returns:
        float: Average reward over episodes
    """
    total_reward = 0
    
    for _ in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        
        for _ in range(max_steps):
            action = agent.select_action(state, deterministic=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        total_reward += episode_reward
    
    return total_reward / num_episodes

 