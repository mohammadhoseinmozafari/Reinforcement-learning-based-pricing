from abc import ABC, abstractmethod
import numpy as np
from collections import deque
import random
from typing import Dict, Optional

from train.curriculum import Curriculum
class BaseReplayBuffer(ABC):
    @abstractmethod
    def push(self , state, action , reward, next_state, done):
        raise NotImplementedError
    @abstractmethod
    def sample(self, batch_size):
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        raise NotImplementedError


class ReplayBuffer(BaseReplayBuffer):
    def __init__(self, capacity: int) -> None:
        self.buffer = deque(maxlen=capacity)
  
    
    def push(self, state, action, reward, next_state, done) -> None:
        """Store a transition in the buffer."""     
        self.buffer.append((state, action, reward, next_state, done))

    
    def sample(self, batch_size: int):
        """Sample a batch of transitions."""
        
        return self._sample_uniform(batch_size)
        
        
    
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

    def clear(self, retain_fraction: Optional[float] = 0.0) -> None :
        if not retain_fraction or retain_fraction <= 0.0:
            self.buffer.clear()
            return
        keep_count = int(len(self.buffer)*retain_fraction)
        retained = list (self.buffer)[-keep_count:]
        self.buffer.clear()
        self.buffer.extend(retained)
        
    def __len__(self):
        return len(self.buffer)


class CurriculumReplayBuffer(BaseReplayBuffer):
    """
    Maintains one replay buffer per curriculum stage and
    performs mixed sampling to prevent catastrophic forgetting.
    """

    def __init__(self, capacity: int, batch_size: int, curriculum : Curriculum):

        self.batch_size = batch_size
        self.capacity = capacity
        self.buffers = self._create_buffers(curriculum)
        self.sampling_weights = self._create_sampling_weights(curriculum)
       
        self.current_stage = curriculum.get_sequence()[0].opponent_type

    
    def _create_buffers(self, curriculum: Curriculum)-> Dict[str, ReplayBuffer]:
        stages = curriculum.get_sequence()
        buffers = {}
        for stage in stages : 
                buffers[stage.opponent_type] = ReplayBuffer(self.capacity)

        return buffers

    def _create_sampling_weights(self, curriculum: Curriculum, primal_weight : float = 0.75)-> Dict[str,Dict[str, float]]:
        
        assert 0.0 < primal_weight <=1.0 , 'primal_weight must be in (0,1]'

        stages = curriculum.get_sequence()
        
        weights = {}
        
        for current_idx , current_stage in enumerate(stages):
            current_name = current_stage.opponent_type

            if current_idx == 0 :
                weights[current_name] = {
                    current_name: 1.0
                } 
                continue
            current_weights = {}

            current_weights[current_name] = primal_weight
            prev_weight = (1.0 - primal_weight) / current_idx
            for prev_idx in range(current_idx):
                prev_name = stages[prev_idx].opponent_type
                current_weights[prev_name]= prev_weight
            
            weights[current_name] = current_weights
        
        return weights



    def set_stage(self, stage_name: str):
        self.current_stage = stage_name

    def push(self, state, action, reward, next_state, done):
        self.buffers[self.current_stage].push(
            state,
            action,
            reward,
            next_state,
            done
        )

    def __len__(self):
        return sum(len(buf) for buf in self.buffers.values())

    def stage_size(self, stage_name):
        return len(self.buffers[stage_name])

    def sample(self, batch_size=None):

        if batch_size is None:
            batch_size = self.batch_size

        weights = self.sampling_weights[self.current_stage]

        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        collected = 0

        for stage_name, weight in weights.items():

            buffer = self.buffers[stage_name]

            if len(buffer) == 0:
                continue

            n_samples = int(batch_size * weight)

            n_samples = min(
                n_samples,
                len(buffer)
            )

            if n_samples == 0:
                continue

            batch = buffer.sample(n_samples)

            states.extend(batch[0])
            actions.extend(batch[1])
            rewards.extend(batch[2])
            next_states.extend(batch[3])
            dones.extend(batch[4])

            collected += n_samples

        # Fill remainder from current stage
        remaining = batch_size - collected

        if remaining > 0:

            current_buffer = self.buffers[self.current_stage]

            if len(current_buffer) >= remaining:

                batch = current_buffer.sample(remaining)

                states.extend(batch[0])
                actions.extend(batch[1])
                rewards.extend(batch[2])
                next_states.extend(batch[3])
                dones.extend(batch[4])

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
        )
    
    def get_info(self):
        lens = {}
        for stage, buffer in self.buffers.items():
            lens[stage] = len(buffer)
        return lens
    


class RecencyBiasReplayBuffer(BaseReplayBuffer):
    """Experience Replay Buffer which cares more about the recent experiences."""
    
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


class EpisodeBuilder:
    """Accumulate transitions and materialize one replay-ready episode."""

    FIELDS = (
        "obs", "actions", "rewards", "next_obs", "dones", "opponent_actions"
    )

    def __init__(self) -> None:
        self._values = {field: [] for field in self.FIELDS}

    def append(
        self,
        obs,
        action,
        reward,
        next_obs,
        done,
        opponent_action,
    ) -> None:
        """Append one transition using stable float32 replay shapes."""
        self._values["obs"].append(np.asarray(obs, dtype=np.float32).copy())
        self._values["actions"].append(np.asarray(action, dtype=np.float32).copy())
        self._values["rewards"].append(
            np.asarray([float(reward)], dtype=np.float32)
        )
        self._values["next_obs"].append(
            np.asarray(next_obs, dtype=np.float32).copy()
        )
        self._values["dones"].append(
            np.asarray([float(done)], dtype=np.float32)
        )
        self._values["opponent_actions"].append(
            np.asarray(opponent_action, dtype=np.float32).reshape(-1).copy()
        )

    def build(self):
        """Return a complete episode dictionary of stacked numpy arrays."""
        return {
            field: np.asarray(values, dtype=np.float32)
            for field, values in self._values.items()
        }

    def __len__(self) -> int:
        return len(self._values["obs"])


class EpisodeReplayBuffer:
    def __init__(self, capacity_episodes: int, sequence_length: int) -> None:
        self.episodes = deque(maxlen=capacity_episodes)
        self.sequence_length = sequence_length

    def push(self, episode) -> None:
        """Store an episode in the buffer."""
        if len(episode["obs"]) == 0:
            return
        self.episodes.append(episode)

    def create_episode_builder(self) -> EpisodeBuilder:
        """Create a builder compatible with this buffer's episode schema."""
        return EpisodeBuilder()

    def sample(self):
        """Samples an episodes, masks the episodes which their length is below the sequence length"""

        episode = random.choice(self.episodes)
        episode_len = len(episode["obs"])
        T = self.sequence_length

        if episode_len >= T:
            start = random.randint(0, episode_len - T)
            end = start + T
            mask = np.ones((T, 1), dtype=np.float32)

            return {
                "obs": episode["obs"][start:end],
                "actions": episode["actions"][start:end],
                "rewards": episode["rewards"][start:end],
                "next_obs": episode["next_obs"][start:end],
                "dones": episode["dones"][start:end],
                "opponent_actions": episode["opponent_actions"][start:end],
                "mask": mask,
                "opponent_type": episode["opponent_type"],
                "stage_id": episode["stage_id"],
            }

        pad = T - episode_len
        mask = np.concatenate(
            [
                np.ones((episode_len, 1), dtype=np.float32),
                np.zeros((pad, 1), dtype=np.float32),
            ],
            axis=0,
        )

        return {
            "obs": self._pad(episode["obs"], T),
            "actions": self._pad(episode["actions"], T),
            "rewards": self._pad(episode["rewards"], T),
            "next_obs": self._pad(episode["next_obs"], T),
            "dones": self._pad(episode["dones"], T),
            "opponent_actions": self._pad(episode["opponent_actions"], T),
            "mask": mask,
            "opponent_type": episode["opponent_type"],
            "stage_id": episode["stage_id"],
        }

    def _pad(self, arr, target_len):
        current_len = len(arr)
        if current_len >= target_len:
            return arr[:target_len]

        pad_shape = (target_len - current_len,) + arr.shape[1:]
        padding = np.zeros(pad_shape, dtype=np.float32)
        return np.concatenate([arr, padding], axis=0)

    def __len__(self):
        return len(self.episodes)

class CurriculumSequenceReplayBuffer:
    """Curriculum-aware replay for fixed-length sequences sampled from episodes.

    Its stage-management interface mirrors :class:`CurriculumReplayBuffer`,
    while each per-stage buffer remains an :class:`EpisodeReplayBuffer`.
    ``push`` therefore accepts a complete episode rather than one transition.
    The active stage is authoritative: opponent and stage metadata are attached
    automatically before the episode is stored.
    """

    def __init__(
        self,
        capacity: int,
        batch_size: int,
        curriculum: Curriculum,
        sequence_length: int,
        current_stage_weight: float = 0.75,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        if not 0.0 < current_stage_weight <= 1.0:
            raise ValueError("current_stage_weight must be in (0, 1]")

        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.capacity = capacity
        self.current_stage_weight = current_stage_weight

        stages = curriculum.get_sequence()
        if not stages:
            raise ValueError("curriculum must contain at least one stage")

        self.stage_ids = {
            stage.opponent_type: index for index, stage in enumerate(stages)
        }
        if len(self.stage_ids) != len(stages):
            raise ValueError("curriculum opponent types must be unique")

        self.buffers = self._create_buffers(curriculum)
        self.sampling_weights = self._create_sampling_weights(curriculum)
        self.current_stage = stages[0].opponent_type
        self.current_stage_id = 0

    def _create_buffers(self, curriculum: Curriculum) -> Dict[str, EpisodeReplayBuffer]:
        stages = curriculum.get_sequence()
        buffers = {}
        for stage in stages:
            buffers[stage.opponent_type] = EpisodeReplayBuffer(
                self.capacity,
                self.sequence_length,
            )
        return buffers

    def _create_sampling_weights(
        self,
        curriculum: Curriculum,
    ) -> Dict[str, Dict[str, float]]:
        """Match the current/prior-stage weighting of CurriculumReplayBuffer."""
        stages = curriculum.get_sequence()
        weights: Dict[str, Dict[str, float]] = {}
        for current_index, current_stage in enumerate(stages):
            current_name = current_stage.opponent_type
            if current_index == 0:
                weights[current_name] = {current_name: 1.0}
                continue

            stage_weights = {current_name: self.current_stage_weight}
            previous_weight = (1.0 - self.current_stage_weight) / current_index
            for previous_stage in stages[:current_index]:
                stage_weights[previous_stage.opponent_type] = previous_weight
            weights[current_name] = stage_weights
        return weights

    def set_stage(self, stage_name: str) -> None:
        """Select the stage used for insertion and curriculum-weighted sampling."""
        if stage_name not in self.buffers:
            raise KeyError(f"Unknown curriculum stage: {stage_name}")
        self.current_stage = stage_name
        self.current_stage_id = self.stage_ids[stage_name]

    def push(self, episode) -> None:
        """Store a complete episode in the active stage buffer.

        A shallow copy prevents curriculum metadata from mutating the caller's
        episode dictionary. Array payloads are intentionally not copied.
        """
        staged_episode = dict(episode)
        staged_episode["opponent_type"] = self.current_stage
        staged_episode["stage_id"] = self.current_stage_id
        self.buffers[self.current_stage].push(staged_episode)

    def create_episode_builder(self) -> EpisodeBuilder:
        """Create a builder; curriculum metadata is attached by ``push``."""
        return EpisodeBuilder()

    def sample(self, batch_size=None):
        """Sample and stack a batch of padded sequences across eligible stages."""
        if batch_size is None:
            batch_size = self.batch_size
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        selected_opponents = self._sample_opponent_types(batch_size)
        sequences = [
            self.buffers[opponent_type].sample()
            for opponent_type in selected_opponents
        ]

        return self._stack_sequences(sequences)

    def _sample_opponent_types(self, batch_size):
        configured_weights = self.sampling_weights[self.current_stage]
        available = {
            opponent_type: weight
            for opponent_type, weight in configured_weights.items()
            if len(self.buffers[opponent_type]) > 0 and weight > 0
        }
        if not available:
            raise ValueError("No episodes available in replay buffer.")

        opponents = list(available)
        weights = list(available.values())
        return random.choices(opponents, weights=weights, k=batch_size)

    def _stack_sequences(self, sequences):
        return {
            "obs": np.stack([s["obs"] for s in sequences], axis=0),
            "actions": np.stack([s["actions"] for s in sequences], axis=0),
            "rewards": np.stack([s["rewards"] for s in sequences], axis=0),
            "next_obs": np.stack([s["next_obs"] for s in sequences], axis=0),
            "dones": np.stack([s["dones"] for s in sequences], axis=0),
            "opponent_actions": np.stack(
                [s["opponent_actions"] for s in sequences],
                axis=0,
            ),
            "mask": np.stack([s["mask"] for s in sequences], axis=0),
            "opponent_types": [s["opponent_type"] for s in sequences],
            "stage_ids": np.asarray(
                [s["stage_id"] for s in sequences],
                dtype=np.int64,
            ),
        }

    def __len__(self):
        return sum(len(buffer) for buffer in self.buffers.values())

    def stage_size(self, stage_name):
        """Return the number of complete episodes stored for one stage."""
        return len(self.buffers[stage_name])

    def get_info(self):
        return {
            opponent_type: len(buffer)
            for opponent_type, buffer in self.buffers.items()
        }
