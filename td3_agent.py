"""
td3_agent.py - Twin Delayed Deep Deterministic Policy Gradient (TD3).

Implements the actor, critic networks, and the TD3 agent with:
  - clipped double-Q learning
  - delayed policy updates
  - target policy smoothing
  - soft target updates
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
    EXPLORATION_NOISE,
    GAMMA,
    HIDDEN_DIM_1,
    HIDDEN_DIM_2,
    NOISE_CLIP,
    POLICY_DELAY,
    POLICY_NOISE,
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
    """Twin Q-networks that estimate Q(s, a)."""

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

        self.fc4 = nn.Linear(state_dim + action_dim, hidden1)
        self.fc5 = nn.Linear(hidden1, hidden2)
        self.fc6 = nn.Linear(hidden2, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        sa = torch.cat([state, action], dim=1)

        q1 = F.relu(self.fc1(sa))
        q1 = F.relu(self.fc2(q1))
        q1 = self.fc3(q1)

        q2 = F.relu(self.fc4(sa))
        q2 = F.relu(self.fc5(q2))
        q2 = self.fc6(q2)

        return q1, q2

    def q1_forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Return only Q1, which is used for the actor loss."""
        sa = torch.cat([state, action], dim=1)
        q1 = F.relu(self.fc1(sa))
        q1 = F.relu(self.fc2(q1))
        return self.fc3(q1)


class TD3Agent:
    """TD3 reinforcement-learning agent."""

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)

        self.actor = Actor(STATE_DIM, ACTION_DIM).to(self.device)
        self.critic = Critic(STATE_DIM, ACTION_DIM).to(self.device)

        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=ACTOR_LR)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=CRITIC_LR)

        self.total_it = 0

    def select_action(self, state: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """Choose an action for the current state."""
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.actor(state_t).cpu().numpy().flatten()

        if add_noise:
            noise = np.random.normal(0, EXPLORATION_NOISE, size=ACTION_DIM)
            action = action + noise

        return np.clip(action, -1.0, 1.0)

    def train(self, replay_buffer, batch_size: int):
        """Perform one TD3 training step."""
        self.total_it += 1

        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

        state = torch.FloatTensor(states).to(self.device)
        action = torch.FloatTensor(actions).to(self.device)
        reward = torch.FloatTensor(rewards).to(self.device)
        next_state = torch.FloatTensor(next_states).to(self.device)
        done = torch.FloatTensor(dones).to(self.device)

        with torch.no_grad():
            noise = (torch.randn_like(action) * POLICY_NOISE).clamp(-NOISE_CLIP, NOISE_CLIP)
            next_action = (self.actor_target(next_state) + noise).clamp(-1.0, 1.0)

            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + (1.0 - done) * GAMMA * target_q

        current_q1, current_q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if self.total_it % POLICY_DELAY == 0:
            actor_loss = -self.critic.q1_forward(state, self.actor(state)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
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
        print(f"[TD3] Model saved -> {filepath}")

    def load(self, filepath: str):
        """Load actor and critic weights from disk."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.actor_target.load_state_dict(checkpoint["actor_target"])
        self.critic_target.load_state_dict(checkpoint["critic_target"])
        print(f"[TD3] Model loaded <- {filepath}")
