"""
ddpg_agent.py - Deep Deterministic Policy Gradient (DDPG).

Implements the actor, single critic network, and DDPG agent with:
  - deterministic policy gradients
  - single Q critic
  - target networks
  - soft target updates
  - gradient clipping for stability
"""

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    ACTOR_LR,
    ACTION_DIM,
    CRITIC_LR,
    GAMMA,
    GRADIENT_CLIP_MAX_NORM,
    HIDDEN_DIM_1,
    HIDDEN_DIM_2,
    STATE_DIM,
    TAU,
)


class Actor(nn.Module):
    """Deterministic policy network mapping state to continuous actions."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden1: int = HIDDEN_DIM_1,
        hidden2: int = HIDDEN_DIM_2,
    ):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, action_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))


class Critic(nn.Module):
    """Single Q-network that estimates Q(s, a)."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden1: int = HIDDEN_DIM_1,
        hidden2: int = HIDDEN_DIM_2,
    ):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        sa = torch.cat([state, action], dim=1)
        q = F.relu(self.fc1(sa))
        q = F.relu(self.fc2(q))
        return self.fc3(q)


class DDPGAgent:
    """Deep Deterministic Policy Gradient (DDPG) agent for continuous control."""

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)

        self.actor = Actor(STATE_DIM, ACTION_DIM).to(self.device)
        self.critic = Critic(STATE_DIM, ACTION_DIM).to(self.device)

        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=ACTOR_LR)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=CRITIC_LR)

    def select_action(self, state: np.ndarray, add_noise: bool = True, noise_scale: float = 0.1) -> np.ndarray:
        """Choose an action for the current state with optional exploration noise."""
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(state_t).cpu().numpy().flatten()

        if add_noise:
            noise = np.random.normal(0, noise_scale, size=ACTION_DIM)
            action = action + noise

        return np.clip(action, -1.0, 1.0)

    def train(self, replay_buffer, batch_size: int):
        """Perform one DDPG training step with soft target updates."""
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

        state = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        action = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        reward = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        next_state = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        done = torch.as_tensor(dones, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            next_action = self.actor_target(next_state)
            target_q = self.critic_target(next_state, next_action)
            target_q = reward + (1.0 - done) * GAMMA * target_q

        current_q = self.critic(state, action)
        critic_loss = F.mse_loss(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=GRADIENT_CLIP_MAX_NORM)
        self.critic_optimizer.step()

        actor_loss = -self.critic(state, self.actor(state)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=GRADIENT_CLIP_MAX_NORM)
        self.actor_optimizer.step()

        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)

    @staticmethod
    def _soft_update(source: nn.Module, target: nn.Module):
        """Polyak-average update for a target network."""
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(TAU * param.data + (1.0 - TAU) * target_param.data)

    def save(self, filepath: str):
        """Persist actor and critic weights to disk."""
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "actor_target": self.actor_target.state_dict(),
                "critic_target": self.critic_target.state_dict(),
            },
            filepath,
        )
        print(f"[DDPG] Model saved -> {filepath}")

    def load(self, filepath: str):
        """Load actor and critic weights from disk."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.actor_target.load_state_dict(checkpoint["actor_target"])
        self.critic_target.load_state_dict(checkpoint["critic_target"])
        print(f"[DDPG] Model loaded <- {filepath}")
