"""
replay_buffer.py – Experience replay buffer for off-policy RL.

Stores (state, action, reward, next_state, done) transitions in a fixed-size
circular NumPy buffer and supports uniform random mini-batch sampling.
"""

import numpy as np
from config import STATE_DIM, ACTION_DIM


class ReplayBuffer:
    """Fixed-size circular replay buffer backed by NumPy arrays."""

    def __init__(self, capacity: int):
        """
        Args:
            capacity: Maximum number of transitions to store.
        """
        self.capacity = capacity
        self.ptr = 0          # next write position
        self.size = 0         # current number of stored transitions

        # Pre-allocate contiguous arrays for speed
        self.states = np.zeros((capacity, STATE_DIM), dtype=np.float32)
        self.actions = np.zeros((capacity, ACTION_DIM), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, STATE_DIM), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------
    def add(self, state, action, reward, next_state, done):
        """Store a single transition in the buffer.

        Args:
            state:      np.array of shape (STATE_DIM,)
            action:     np.array of shape (ACTION_DIM,)
            reward:     float
            next_state: np.array of shape (STATE_DIM,)
            done:       bool or float (1.0 if terminal, else 0.0)
        """
        idx = self.ptr
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = float(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    def sample(self, batch_size: int):
        """Sample a random mini-batch of transitions.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            Tuple of NumPy arrays:
            (states, actions, rewards, next_states, dones)
        """
        indices = np.random.randint(0, self.size, size=batch_size)

        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
        )

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def __len__(self):
        return self.size

    def is_ready(self, batch_size: int) -> bool:
        """Return True if the buffer contains enough samples for a batch."""
        return self.size >= batch_size
