# import torch
# import torch.optim as optim
# import torch.nn as nn
# import random
# from networks import QNetwork
# from replay_buffer import ReplayBuffer


# class DQNAgent:
#     """
#     Deep Q-Learning Agent with soft target network updates.
#     Updated for 20-dimensional state space.
#     """
    
#     def __init__(self, state_dim=20, num_actions=4, lr=0.001,
#                  gamma=0.95, buffer_capacity=100000, batch_size=128, tau=0.005):
#         """
#         Args:
#             state_dim: State dimension (20 for GridWorld)
#             num_actions: Number of available actions (4 for GridWorld)
#             lr: Learning rate
#             gamma: Discount factor
#             buffer_capacity: Maximum replay buffer size
#             batch_size: Minibatch size for training
#             tau: Soft update parameter for target network
#         """
#         self.state_dim = state_dim
#         self.num_actions = num_actions
#         self.gamma = gamma
#         self.batch_size = batch_size
#         self.tau = tau

#         # Create Q-network and target network
#         self.q_net = QNetwork(input_dim=state_dim, output_dim=num_actions)
#         self.target_net = QNetwork(input_dim=state_dim, output_dim=num_actions)
#         self.target_net.load_state_dict(self.q_net.state_dict())
#         self.target_net.eval()

#         # Replay memory
#         self.memory = ReplayBuffer(capacity=buffer_capacity)

#         # Optimizer
#         self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
#         self.last_loss = 0.0

#     def select_action(self, state, epsilon):
#         """Epsilon-greedy action selection."""
#         if random.random() < epsilon:
#             return random.randrange(self.num_actions)

#         state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
#         with torch.no_grad():
#             q_values = self.q_net(state_tensor)
#         return torch.argmax(q_values, dim=1).item()

#     def store_experience(self, state, action, reward, next_state, done):
#         """Store transition in replay buffer."""
#         self.memory.push(state, action, reward, next_state, done)

#     def train_step(self):
#         """Sample batch from memory and perform one gradient update."""
#         if len(self.memory) < self.batch_size:
#             return None

#         # Sample minibatch
#         states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

#         # Convert to tensors
#         states = torch.tensor(states, dtype=torch.float32)
#         actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
#         rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
#         next_states = torch.tensor(next_states, dtype=torch.float32)
#         dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

#         # Compute current Q-values
#         q_values = self.q_net(states).gather(1, actions)

#         # Compute target Q-values (Q-Learning with target network)
#         with torch.no_grad():
#             next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
#             target_q = rewards + (1 - dones) * self.gamma * next_q

#         # Compute loss
#         loss = nn.functional.mse_loss(q_values, target_q)

#         # Backpropagation
#         self.optimizer.zero_grad()
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
#         self.optimizer.step()

#         # Soft update of target network
#         with torch.no_grad():
#             for target_param, param in zip(self.target_net.parameters(), self.q_net.parameters()):
#                 target_param.data.mul_(1.0 - self.tau)
#                 target_param.data.add_(self.tau * param.data)

#         self.last_loss = loss.item()
#         return loss.item()

#     def save(self, path):
#         """Save model weights."""
#         torch.save(self.q_net.state_dict(), path)
#         print(f"Model saved to {path}")

#     def load(self, path):
#         """Load model weights."""
#         self.q_net.load_state_dict(torch.load(path, weights_only=True))
#         self.target_net.load_state_dict(self.q_net.state_dict())
#         self.q_net.eval()
#         print(f"Model loaded from {path}")

# ========================================
# FILE: dqn_agent.py (FIXED - Compatible)
# ========================================

import torch
import torch.optim as optim
import torch.nn as nn
import random
from networks import QNetwork
from replay_buffer import ReplayBuffer


class DQNAgent:
    """
    Deep Q-Learning Agent with soft target network updates.
    Updated for 20-dimensional state space.
    """
    
    def __init__(self, state_dim=20, num_actions=4, lr=0.001,
                 gamma=0.95, buffer_capacity=100000, batch_size=128, tau=0.005, hidden_dim=256):
        """
        Args:
            state_dim: State dimension (20 for GridWorld)
            num_actions: Number of available actions (4 for GridWorld)
            lr: Learning rate
            gamma: Discount factor
            buffer_capacity: Maximum replay buffer size
            batch_size: Minibatch size for training
            tau: Soft update parameter for target network
            hidden_dim: Hidden layer dimension for Q-network
        """
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau

        # Create Q-network and target network
        self.q_net = QNetwork(input_dim=state_dim, output_dim=num_actions, hidden_dim=hidden_dim)
        self.target_net = QNetwork(input_dim=state_dim, output_dim=num_actions, hidden_dim=hidden_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        # Replay memory
        self.memory = ReplayBuffer(capacity=buffer_capacity)

        # Optimizer
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.last_loss = 0.0

    def select_action(self, state, epsilon):
        """Epsilon-greedy action selection."""
        if random.random() < epsilon:
            return random.randrange(self.num_actions)

        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        return torch.argmax(q_values, dim=1).item()

    def store_experience(self, state, action, reward, next_state, done):
        """Store transition in replay buffer."""
        self.memory.push(state, action, reward, next_state, done)

    def train_step(self):
        """Sample batch from memory and perform one gradient update."""
        if len(self.memory) < self.batch_size:
            return None

        # Sample minibatch
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        # Convert to tensors
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

        # Compute current Q-values
        q_values = self.q_net(states).gather(1, actions)

        # Compute target Q-values (Q-Learning with target network)
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + (1 - dones) * self.gamma * next_q

        # Compute loss
        loss = nn.functional.mse_loss(q_values, target_q)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        # Soft update of target network
        self._soft_update_target_network()

        self.last_loss = loss.item()
        return loss.item()

    def _soft_update_target_network(self):
        """Soft update of target network using tau parameter."""
        with torch.no_grad():
            for target_param, param in zip(self.target_net.parameters(), self.q_net.parameters()):
                target_param.data.mul_(1.0 - self.tau)
                target_param.data.add_(self.tau * param.data)

    def save(self, path):
        """Save model weights."""
        torch.save(self.q_net.state_dict(), path)
        print(f"Model saved to {path}")

    def load(self, path):
        """Load model weights."""
        self.q_net.load_state_dict(torch.load(path, weights_only=True))
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.q_net.eval()
        print(f"Model loaded from {path}")