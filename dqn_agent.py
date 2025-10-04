import numpy as np
import random
from collections import deque
import pickle
import os
import logging


class DQNAgent:
    """Deep Q-Learning Agent for Go game"""

    def __init__(self, board_size=9):
        self.board_size = board_size
        self.state_size = board_size * board_size * 2  # 2 channels: own stones and opponent stones
        self.action_size = board_size * board_size

        # Hyperparameters
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 32

        # Experience replay memory
        self.memory = deque(maxlen=2000)

        # Neural network weights (simple linear model for demonstration)
        # For production, use a proper neural network (e.g., with TensorFlow/PyTorch)
        self.weights = np.random.randn(self.state_size, self.action_size) * 0.01
        self.bias = np.zeros(self.action_size)

        logging.info("DQN Agent initialized")
        logging.info(f"State size: {self.state_size}, Action size: {self.action_size}")
        logging.info(f"Hyperparameters - Gamma: {self.gamma}, Epsilon: {self.epsilon}, LR: {self.learning_rate}")

    def get_state(self, board, player):
        """
        Convert board to state representation
        Returns flattened array with 2 channels:
        - Channel 0: Own stones (1 where player has stones)
        - Channel 1: Opponent stones (1 where opponent has stones)
        """
        state = np.zeros((self.board_size, self.board_size, 2))

        for i in range(self.board_size):
            for j in range(self.board_size):
                if board[i][j] == player:
                    state[i][j][0] = 1  # Own stones
                elif board[i][j] is not None:
                    state[i][j][1] = 1  # Opponent stones

        return state.flatten()

    def predict(self, state):
        """Predict Q-values for all actions using current weights"""
        return np.dot(state, self.weights) + self.bias

    def get_valid_actions(self, board):
        """Get list of valid (empty) positions on the board"""
        valid_actions = []
        for i in range(self.board_size):
            for j in range(self.board_size):
                if board[i][j] is None:
                    valid_actions.append(i * self.board_size + j)
        return valid_actions

    def act(self, state, valid_actions):
        """
        Choose action using epsilon-greedy policy
        - With probability epsilon: random action (exploration)
        - With probability 1-epsilon: best action based on Q-values (exploitation)
        """
        if len(valid_actions) == 0:
            return None

        if np.random.rand() <= self.epsilon:
            # Exploration: random valid action
            action = random.choice(valid_actions)
            logging.debug(f"Agent exploring: random action {action}")
        else:
            # Exploitation: best valid action based on Q-values
            q_values = self.predict(state)
            # Mask invalid actions with -infinity
            masked_q = np.full(self.action_size, -np.inf)
            masked_q[valid_actions] = q_values[valid_actions]
            action = np.argmax(masked_q)
            logging.debug(f"Agent exploiting: best action {action} with Q-value {q_values[action]:.3f}")

        return action

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        """
        Train on batch of experiences from memory
        Uses Q-learning update rule: Q(s,a) = r + gamma * max(Q(s',a'))
        """
        if len(self.memory) < self.batch_size:
            return

        # Sample random minibatch from memory
        minibatch = random.sample(self.memory, self.batch_size)

        total_loss = 0
        for state, action, reward, next_state, done in minibatch:
            # Calculate target Q-value
            target = reward
            if not done:
                target = reward + self.gamma * np.max(self.predict(next_state))

            # Get current Q-values
            q_values = self.predict(state)

            # Calculate error
            error = target - q_values[action]
            total_loss += error ** 2

            # Gradient descent update
            self.weights[:, action] += self.learning_rate * state * error
            self.bias[action] += self.learning_rate * error

        # Decay epsilon (reduce exploration over time)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        avg_loss = total_loss / self.batch_size
        logging.debug(f"Training completed - Avg Loss: {avg_loss:.4f}, Epsilon: {self.epsilon:.4f}")

    def save(self, filename="dqn_agent.pkl"):
        """Save agent weights and memory to file"""
        data = {
            'weights': self.weights,
            'bias': self.bias,
            'epsilon': self.epsilon,
            'memory': list(self.memory)
        }
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        logging.info(f"Agent saved to {filename}")

    def load(self, filename="dqn_agent.pkl"):
        """Load agent weights and memory from file"""
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                data = pickle.load(f)
            self.weights = data['weights']
            self.bias = data['bias']
            self.epsilon = data['epsilon']
            self.memory = deque(data['memory'], maxlen=2000)
            logging.info(f"Agent loaded from {filename} (Epsilon: {self.epsilon:.4f})")
            return True
        else:
            logging.warning(f"Agent file {filename} not found, starting with fresh agent")
            return False
