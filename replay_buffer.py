"""
replay_buffer.py - Experience replay buffer for off-policy RL.

Stores (state, action, reward, next_state, done) transitions in a fixed-size
circular NumPy buffer and supports uniform random mini-batch sampling.
"""

import numpy as np

from config import ACTION_DIM, STATE_DIM


class ReplayBuffer:
    """Fixed-size circular replay buffer backed by NumPy arrays."""

    def __init__(self, capacity: int):
        """Create a replay buffer with pre-allocated storage."""
        self.capacity = capacity
        self.ptr = 0
        self.size = 0

        self.states = np.zeros((capacity, STATE_DIM), dtype=np.float32)
        self.actions = np.zeros((capacity, ACTION_DIM), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, STATE_DIM), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

    def add(self, state, action, reward, next_state, done):
        """Store a single transition in the buffer."""
        idx = self.ptr
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = float(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        """Sample a random mini-batch of transitions."""
        indices = np.random.randint(0, self.size, size=batch_size)
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
        )

    def __len__(self):
        return self.size

    def is_ready(self, batch_size: int) -> bool:
        """Return True if the buffer contains enough samples for a batch."""
        return self.size >= batch_size
