import math
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import tkinter as tk
import random
import torch.optim as optim
import os

MODEL_PATH = 'dqn_TicTacToe3x3.pth'

class TicTacToe3x3:
    def __init__(self):
        self.size = 3
        self.reset()

    def reset(self):
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.current_player = 1
        self.done = False
        self.winner = None
        return self.get_state()

    def get_state(self):
        return np.append(self.board.flatten(), self.current_player)

    def get_available_actions(self):
        return [(i, j) for i in range(self.size) for j in range(self.size) if self.board[i, j] == 0]

    def step(self, action):
        i, j = action
        if self.board[i, j] != 0 or self.done:
            return self.get_state(), -10, True, {}
        self.board[i, j] = self.current_player
        winner = self.check_winner()
        if winner is not None:
            self.done = True
            self.winner = winner
            if winner == 0:
                return self.get_state(), 0, True, {}
            else:
                return self.get_state(), 1 if winner == 1 else -1, True, {}
        elif not self.get_available_actions():
            self.done = True
            self.winner = 0
            return self.get_state(), 0, True, {}
        else:
            self.current_player *= -1
            return self.get_state(), 0, False, {}

    def check_winner(self):
            for i in range(self.size):
                for j in range(self.size - 2):
                    # Rows
                    s = np.sum(self.board[i, j:j + 3])
                    if abs(s) == 3 and len(set(self.board[i, j:j + 3])) == 1 and self.board[i, j] != 0:
                        return np.sign(s)
                    # Columns
                    s = np.sum(self.board[j:j + 3, i])
                    if abs(s) == 3 and len(set(self.board[j:j + 3, i])) == 1 and self.board[j, i] != 0:
                        return np.sign(s)
            # Diagonals
            for i in range(self.size - 2):
                for j in range(self.size - 2):
                    diag1 = [self.board[i + k, j + k] for k in range(3)]
                    diag2 = [self.board[i + 2 - k, j + k] for k in range(3)]
                    if abs(sum(diag1)) == 3 and len(set(diag1)) == 1 and diag1[0] != 0:
                        return np.sign(sum(diag1))
                    if abs(sum(diag2)) == 3 and len(set(diag2)) == 1 and diag2[0] != 0:
                        return np.sign(sum(diag2))
            return None

class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQN_Agent:
    def __init__(self, env, epsilon=0.9, alpha=0.001, gamma=0.99,
                 batch_size=64, memory_size=5000):
        self.env = env
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.batch_size = batch_size
        self.state_size = env.size * env.size + 1
        self.action_size = env.size * env.size
        self.model = DQN(self.state_size, self.action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=alpha)
        self.criterion = nn.MSELoss()
        self.memory = deque(maxlen=memory_size)

    def state_to_tensor(self, state):
        return torch.FloatTensor(state).unsqueeze(0)

    def choose_action(self, available_actions, state, greedy=False):
        if not greedy and np.random.random() < self.epsilon:
            return random.choice(available_actions)
        else:
            state_tensor = self.state_to_tensor(state)
            with torch.no_grad():
                q_values = self.model(state_tensor).cpu().numpy().flatten()
            mask = np.full(self.action_size, -np.inf)
            for (i, j) in available_actions:
                mask[i * self.env.size + j] = q_values[i * self.env.size + j]
            idx = np.argmax(mask)
            return (idx // self.env.size, idx % self.env.size)

    def remember(self, state, action, reward, next_state, done):
        idx = action[0] * self.env.size + action[1]
        self.memory.append((state, idx, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        for state, action_idx, reward, next_state, done in batch:
            state_tensor = self.state_to_tensor(state)
            next_state_tensor = self.state_to_tensor(next_state)
            q_values = self.model(state_tensor)
            if done:
                target_q = reward
            else:
                with torch.no_grad():
                    next_q_values = self.model(next_state_tensor)
                    target_q = reward + self.gamma * torch.max(next_q_values)
            target = q_values.clone()
            target[0][action_idx] = target_q
            self.optimizer.zero_grad()
            loss = self.criterion(q_values, target)
            loss.backward()
            self.optimizer.step()

# --- Minimax opponent (depth-limited for speed) ---
def minimax(env, player, depth=0, max_depth=2):
    winner = env.check_winner()
    if winner == 1:
        return 1
    elif winner == -1:
        return -1
    elif not env.get_available_actions():
        return 0
    if depth >= max_depth:
        return 0  # Limit depth for speed

    if player == -1:
        best = math.inf
        for action in env.get_available_actions():
            i, j = action
            env.board[i, j] = player
            score = minimax(env, -player, depth+1, max_depth)
            env.board[i, j] = 0
            best = min(best, score)
        return best
    else:
        best = -math.inf
        for action in env.get_available_actions():
            i, j = action
            env.board[i, j] = player
            score = minimax(env, -player, depth+1, max_depth)
            env.board[i, j] = 0
            best = max(best, score)
        return best

def minimax_action(env, max_depth=2):
    best_score = math.inf
    best_action = None
    for action in env.get_available_actions():
        i, j = action
        env.board[i, j] = -1
        score = minimax(env, 1, 0, max_depth)
        env.board[i, j] = 0
        if score < best_score:
            best_score = score
            best_action = action
    return best_action

# --- Enhanced training loop ---
def play_dqn_vs_minimax(env, agent, trials=100000, max_steps_per_episode=100, epsilon_decay=0.9995, min_epsilon=0.01, max_depth=2):
    reward_per_episode = []
    for trial in range(trials):
        state = env.reset()
        cumulative_reward = 0
        step = 0
        done = False
        agent.epsilon = max(agent.epsilon * epsilon_decay, min_epsilon)
        while not done and step < max_steps_per_episode:
            if env.current_player == 1:
                available_actions = env.get_available_actions()
                action = agent.choose_action(available_actions, state)
            else:
                action = minimax_action(env, max_depth=max_depth)
            next_state, reward, done, _ = env.step(action)
            # Reward shaping
            if done:
                if reward == 1:
                    shaped_reward = 1      # Win
                elif reward == -1:
                    shaped_reward = -1     # Lose
                else:
                    shaped_reward = 0      # Draw
            else:
                shaped_reward = 0.1        # Small reward for a valid move
            shaped_reward -= 0.01          # Small penalty per move
            if env.current_player == -1:   # Only train agent on its own moves
                agent.remember(state, action, shaped_reward, next_state, done)
                agent.replay()
            state = next_state
            cumulative_reward += shaped_reward
            step += 1
        if (trial + 1) % 100 == 0 or trial == 0:
            print(f"Episode {trial + 1}/{trials} - Reward: {cumulative_reward:.2f} - Epsilon: {agent.epsilon:.4f}")
        reward_per_episode.append(cumulative_reward)
    return reward_per_episode

class TicTacToeVisualizer(tk.Tk):
    def __init__(self, env, agent):
        super().__init__()
        self.env = env
        self.agent = agent
        self.title("TicTacToe 3x3 DQN Agent Visualization")
        self.cell_size = 60
        self.canvas = tk.Canvas(self, width=env.size*self.cell_size, height=env.size*self.cell_size)
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.on_click)
        self.draw_board()
        self.status = tk.Label(self, text="X's turn", font=("TkDefaultFont", 14))
        self.status.pack()

    def draw_board(self):
        self.canvas.delete("all")
        for i in range(self.env.size):
            for j in range(self.env.size):
                x0 = j * self.cell_size
                y0 = i * self.cell_size
                x1 = x0 + self.cell_size
                y1 = y0 + self.cell_size
                self.canvas.create_rectangle(x0, y0, x1, y1, fill="white", outline="black")
                if self.env.board[i, j] == 1:
                    self.canvas.create_text(x0 + self.cell_size//2, y0 + self.cell_size//2, text="X", font=("TkDefaultFont", 24), fill="blue")
                elif self.env.board[i, j] == -1:
                    self.canvas.create_text(x0 + self.cell_size//2, y0 + self.cell_size//2, text="O", font=("TkDefaultFont", 24), fill="red")
        self.update()

    def on_click(self, event):
        if self.env.done:
            return
        col = event.x // self.cell_size
        row = event.y // self.cell_size
        if (row, col) in self.env.get_available_actions():
            _, _, done, _ = self.env.step((row, col))
            self.draw_board()
            if done:
                self.show_result()
            else:
                self.status.config(text="O's turn" if self.env.current_player == -1 else "X's turn")
                self.after(300, self.agent_move)

    def agent_move(self):
        if self.env.done:
            return
        state = self.env.get_state()
        available_actions = self.env.get_available_actions()
        action = self.agent.choose_action(available_actions, state, greedy=True)
        _, _, done, _ = self.env.step(action)
        self.draw_board()
        if done:
            self.show_result()
        else:
            self.status.config(text="O's turn" if self.env.current_player == -1 else "X's turn")

    def show_result(self):
        if self.env.winner == 1:
            self.status.config(text="X wins!")
        elif self.env.winner == -1:
            self.status.config(text="O wins!")
        else:
            self.status.config(text="Draw!")

if __name__ == "__main__":
    env = TicTacToe3x3()
    agent = DQN_Agent(env, epsilon=0.9)
    if os.path.exists(MODEL_PATH):
        agent.model.load_state_dict(torch.load(MODEL_PATH))
        print("Loaded trained model.")
    else:
        print("Training agent...")
        play_dqn_vs_minimax(env, agent, trials=5000, max_steps_per_episode=100, epsilon_decay=0.9995, min_epsilon=0.01, max_depth=2)
        torch.save(agent.model.state_dict(), MODEL_PATH)
        print("Training complete and model saved.")
    app = TicTacToeVisualizer(env, agent)
    app.mainloop()