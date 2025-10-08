import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque  # kept in case you want logs; not required
import random
import tkinter as tk
import tkinter.messagebox
import os

# Configuration
GRID_SIZE = 10
MODEL_PATH = 'improved_D_SARSA_snake_10x10.pth'
CELL_SIZE = 40

# =====================
# GLOBAL TRAINING HYPERPARAMETERS
# =====================
STATE_SIZE = 16
ACTION_SIZE = 3
LEARNING_RATE = 0.0003       # SARSA: slightly higher LR is often fine 0.0005 0.0003
GAMMA = 0.95
EPSILON_START = 1.0
EPSILON_DECAY = 0.9998     # slower decay keeps exploration longer for on-policy 0.9998 0.9998995 
EPSILON_MIN = 0.01
USE_TARGET = False           # gentle target net for stability (still on-policy)
TAU = 0.005                 # soft update rate
USE_REPLAY_BUFFER = False
BATCH_SIZE = 64
TRAIN_EPISODES = 25000      # default training episodes
MOVING_AVG_WINDOW = 100    # window for average score logging

# =====================
# OPTIONAL REPLAY BUFFER
# =====================
class ReplayBuffer:
    """Simple replay buffer for optional off-policy learning."""
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def store(self, state, action, reward, next_state, next_action, done):
        self.buffer.append((state, action, reward, next_state, next_action, done))

    def sample(self, batch_size=64):
        batch = random.sample(self.buffer, min(len(self.buffer), batch_size))
        s, a, r, ns, na, d = zip(*batch)
        return (np.array(s), np.array(a), np.array(r, dtype=np.float32),
                np.array(ns), np.array(na), np.array(d, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)

class SnakeGame:
    def __init__(self, grid_size=GRID_SIZE):
        self.grid_size = grid_size
        self.reset()

    def reset(self):
        # Snake starts in center
        center = self.grid_size // 2
        self.snake = [(center, center), (center, center - 1), (center, center - 2)]
        self.direction = (0, 1)  # Moving right
        self.food = self._place_food()
        self.score = 0
        self.steps_without_food = 0
        self.done = False
        self.prev_distance = self._get_distance_to_food()
        return self.get_state()

    def _place_food(self, manual_position=None):
        if manual_position is not None:
            if manual_position not in self.snake:
                return manual_position
        while True:
            food = (random.randint(0, self.grid_size - 1),
                    random.randint(0, self.grid_size - 1))
            if food not in self.snake:
                return food

    def place_food_at(self, position):
        """Manually place food at a specific position"""
        if position not in self.snake:
            self.food = position
            return True
        return False

    def _get_distance_to_food(self):
        """Manhattan distance to food"""
        head = self.snake[0]
        return abs(head[0] - self.food[0]) + abs(head[1] - self.food[1])

    def get_state(self):
        """
        Enhanced state representation (16 features):
        - Danger in 8 directions (8)
        - Current direction (4: up, down, left, right)
        - Food location relative (4: up, down, left, right)
        """
        head = self.snake[0]

        # Check danger in 8 directions
        danger_up = self._is_collision((head[0] - 1, head[1]))
        danger_down = self._is_collision((head[0] + 1, head[1]))
        danger_left = self._is_collision((head[0], head[1] - 1))
        danger_right = self._is_collision((head[0], head[1] + 1))
        danger_up_left = self._is_collision((head[0] - 1, head[1] - 1))
        danger_up_right = self._is_collision((head[0] - 1, head[1] + 1))
        danger_down_left = self._is_collision((head[0] + 1, head[1] - 1))
        danger_down_right = self._is_collision((head[0] + 1, head[1] + 1))

        # Current direction
        dir_up = (self.direction == (-1, 0))
        dir_down = (self.direction == (1, 0))
        dir_left = (self.direction == (0, -1))
        dir_right = (self.direction == (0, 1))

        # Food location (normalized)
        food_up = (self.food[0] < head[0])
        food_down = (self.food[0] > head[0])
        food_left = (self.food[1] < head[1])
        food_right = (self.food[1] > head[1])

        state = [
            danger_up, danger_down, danger_left, danger_right,
            danger_up_left, danger_up_right, danger_down_left, danger_down_right,
            dir_up, dir_down, dir_left, dir_right,
            food_up, food_down, food_left, food_right
        ]

        return np.array(state, dtype=np.float32)

    def _turn_right(self, direction):
        """Rotate direction 90° clockwise"""
        return (direction[1], -direction[0])

    def _turn_left(self, direction):
        """Rotate direction 90° counter-clockwise"""
        return (-direction[1], direction[0])

    def _is_collision(self, pos):
        """Check if position is collision"""
        row, col = pos
        if row < 0 or row >= self.grid_size or col < 0 or col >= self.grid_size:
            return True
        if pos in self.snake:
            return True
        return False

    def step(self, action):
        if action == 1:
            self.direction = self._turn_right(self.direction)
        elif action == 2:
            self.direction = self._turn_left(self.direction)

        head = self.snake[0]
        new_head = (head[0] + self.direction[0], head[1] + self.direction[1])

        if self._is_collision(new_head):
            self.done = True
            return self.get_state(), -10, True

        self.snake.insert(0, new_head)
        current_distance = self._get_distance_to_food()

        reward = 0
        if new_head == self.food:
            self.score += 1
            reward = 12
            self.food = self._place_food()
            self.steps_without_food = 0
            self.prev_distance = self._get_distance_to_food()
        else:
            self.snake.pop()
            if current_distance < self.prev_distance:
                reward = 2
            else:
                reward = -1
            self.prev_distance = current_distance
            self.steps_without_food += 1

        # Small survival reward (encourage not dying)
        reward += 0.05

        if self.steps_without_food > 100 * self.grid_size:
            self.done = True
            reward = -10

        return self.get_state(), reward, self.done


class D_SARSA(nn.Module):
    def __init__(self, input_size=STATE_SIZE, hidden_size=512, output_size=ACTION_SIZE):
        super(D_SARSA, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc4 = nn.Linear(hidden_size // 2, output_size)
        self.dropout = nn.Dropout(0.05)  # small dropout for regularization

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        return self.fc4(x)


# =========================
# Deep SARSA Agent (on-policy)
# =========================
class DeepSarsaAgent:
    def __init__(self,
                 state_size=STATE_SIZE,
                 action_size=ACTION_SIZE,
                 learning_rate=LEARNING_RATE,
                 gamma=GAMMA,
                 epsilon=EPSILON_START,
                 epsilon_decay=EPSILON_DECAY,
                 epsilon_min=EPSILON_MIN,
                 use_target=USE_TARGET,
                 tau=TAU,
                 batch_size=BATCH_SIZE,
                 replay_buffer=USE_REPLAY_BUFFER):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.model = D_SARSA(state_size, 512, action_size)
        self.target_model = D_SARSA(state_size, 512, action_size)
        self.target_model.load_state_dict(self.model.state_dict())
        self.use_target = use_target
        self.tau = tau

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.SmoothL1Loss()
        self.use_replay = replay_buffer
        if self.use_replay:
            self.replay_buffer = ReplayBuffer(100000)
            self.batch_size = batch_size

    def _soft_update(self):
        if not self.use_target:
            return
        with torch.no_grad():
            for tp, p in zip(self.target_model.parameters(), self.model.parameters()):
                tp.data.mul_(1.0 - self.tau).add_(self.tau * p.data)

    def act(self, state, training=True):
        if training and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return torch.argmax(q_values, dim=1).item()

    def learn(self, state, action, reward, next_state, next_action, done):
        s = torch.FloatTensor(state).unsqueeze(0)
        q_sa = self.model(s)[0, action]
        with torch.no_grad():
            ns = torch.FloatTensor(next_state).unsqueeze(0)
            if done:
                target_val = torch.tensor(reward, dtype=torch.float32)
            else:
                if self.use_target:
                    q_next = self.target_model(ns)[0, next_action]
                else:
                    q_next = self.model(ns)[0, next_action]
                target_val = torch.tensor(reward, dtype=torch.float32) + self.gamma * q_next

        loss = self.criterion(q_sa, target_val)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        # only apply soft update if target network is enabled
        if self.use_target:
            self._soft_update()

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def remember(self, state, action, reward, next_state, next_action, done):
        """Store transition in replay buffer if enabled."""
        if self.use_replay:
            self.replay_buffer.store(state, action, reward, next_state, next_action, done)

    def replay(self):
        """Sample from replay buffer and learn from mini-batches."""
        if not self.use_replay or len(self.replay_buffer) < self.batch_size:
            return
        s_batch, a_batch, r_batch, ns_batch, na_batch, d_batch = self.replay_buffer.sample(self.batch_size)
        for s, a, r, ns, na, d in zip(s_batch, a_batch, r_batch, ns_batch, na_batch, d_batch):
            self.learn(s, a, r, ns, na, d)


def train_agent(training_env, training_agent, episodes=TRAIN_EPISODES):
    scores = []
    rewards = []
    epsilons = []
    best_score = 0
    moving_avg_window = MOVING_AVG_WINDOW

    # show mode summary
    mode = []
    if training_agent.use_replay:
        mode.append("Replay Buffer")
    if training_agent.use_target:
        mode.append("Target Network")
    mode_name = " + ".join(mode) if mode else "Pure Deep SARSA"
    print(f"Training ({mode_name}) for {episodes} episodes...")

    for episode in range(episodes):
        state = training_env.reset()
        action = training_agent.act(state, training=True)
        total_reward = 0

        while not training_env.done:
            next_state, reward, done = training_env.step(action)
            next_action = training_agent.act(next_state, training=True) if not done else 0
            # training_agent.learn(state, action, reward, next_state, next_action, done)

            # Modular replay / on-policy switch
            if training_agent.use_replay:
                training_agent.remember(state, action, reward, next_state, next_action, done)
                training_agent.replay()
            else:
                training_agent.learn(state, action, reward, next_state, next_action, done)

            state, action = next_state, next_action
            total_reward += reward

        training_agent.decay_epsilon()
        scores.append(training_env.score)
        rewards.append(total_reward)
        epsilons.append(training_agent.epsilon)

        if training_env.score > best_score:
            best_score = training_env.score

        if (episode + 1) % 100 == 0:
            avg_score = np.mean(scores[-moving_avg_window:])
            print(f"Episode {episode + 1}/{episodes} | "
                  f"Score: {training_env.score} | "
                  f"Avg({moving_avg_window}): {avg_score:.2f} | "
                  f"Best: {best_score} | "
                  f"ε: {training_agent.epsilon:.3f}")

    print(f"\nTraining complete! Best score: {best_score}")
    final_avg = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
    print(f"Final 100-episode average: {final_avg:.2f}")
    return scores, rewards, epsilons


class SnakeVisualizer(tk.Tk):
    def __init__(self, env, agent):
        super().__init__()
        self.env = env
        self.agent = agent
        self.title("Snake Game - D_SARSA")
        self.manual_food_mode = False

        # Score limit dropdown
        self.score_limit_var = tk.IntVar(value=10)
        score_limits = [i for i in range(5, 51, 5)]
        score_limit_frame = tk.Frame(self)
        score_limit_frame.pack(pady=5)
        tk.Label(score_limit_frame, text="Winning Score Limit:").pack(side=tk.LEFT)
        tk.OptionMenu(score_limit_frame, self.score_limit_var, *score_limits).pack(side=tk.LEFT)

        self.canvas = tk.Canvas(self, width=env.grid_size * CELL_SIZE,
                                height=env.grid_size * CELL_SIZE, bg='black')
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        self.info_frame = tk.Frame(self)
        self.info_frame.pack(pady=10)

        self.score_label = tk.Label(self.info_frame, text="Score: 0",
                                    font=("Arial", 14, "bold"))
        self.score_label.pack(side=tk.LEFT, padx=10)

        self.high_score = 0
        self.high_score_label = tk.Label(self.info_frame, text="Best: 0",
                                         font=("Arial", 14))
        self.high_score_label.pack(side=tk.LEFT, padx=10)

        self.epsilon_label = tk.Label(self.info_frame, text=f"ε: {agent.epsilon:.3f}",
                                      font=("Arial", 12))
        self.epsilon_label.pack(side=tk.LEFT, padx=10)

        btn_frame = tk.Frame(self)
        btn_frame.pack()

        self.play_btn = tk.Button(btn_frame, text="▶ Watch AI Play",
                                  command=self.auto_play, font=("Arial", 11))
        self.play_btn.pack(side=tk.LEFT, padx=5)

        self.reset_btn = tk.Button(btn_frame, text="🔄 Reset",
                                   command=self.reset_game, font=("Arial", 11))
        self.reset_btn.pack(side=tk.LEFT, padx=5)

        self.food_mode_btn = tk.Button(btn_frame, text="🍎 Manual Food: OFF",
                                       command=self.toggle_food_mode, font=("Arial", 11),
                                       bg="lightgray")
        self.food_mode_btn.pack(side=tk.LEFT, padx=5)

        self.retrain_btn = tk.Button(btn_frame, text="🎓 Train More",
                                     command=self.retrain, font=("Arial", 11))
        self.retrain_btn.pack(side=tk.LEFT, padx=5)

        self.speed_var = tk.IntVar(value=100)
        speed_frame = tk.Frame(self)
        speed_frame.pack()
        tk.Label(speed_frame, text="Speed:").pack(side=tk.LEFT)
        tk.Scale(speed_frame, from_=10, to=500, orient=tk.HORIZONTAL,
                 variable=self.speed_var, length=200).pack(side=tk.LEFT)

        self.mode_label = tk.Label(self, text="Click grid to place food (Manual Food mode)",
                                   font=("Arial", 10), fg="gray")
        self.mode_label.pack()
        self.mode_label.pack_forget()  # Hide initially

        self.playing = False
        self.draw_game()

    def toggle_food_mode(self):
        self.manual_food_mode = not self.manual_food_mode
        if self.manual_food_mode:
            self.food_mode_btn.config(text="🍎 Manual Food: ON", bg="lightgreen")
            self.mode_label.pack()
        else:
            self.food_mode_btn.config(text="🍎 Manual Food: OFF", bg="lightgray")
            self.mode_label.pack_forget()

    def on_canvas_click(self, event):
        col = event.x // CELL_SIZE
        row = event.y // CELL_SIZE

        if row >= self.env.grid_size or col >= self.env.grid_size or row < 0 or col < 0:
            return

        if self.manual_food_mode:
            # Place food at clicked location
            position = (row, col)
            if self.env.place_food_at(position):
                self.draw_game()
                print(f"Food placed at: ({row}, {col})")
            else:
                print(f"Cannot place food at ({row}, {col}) - snake is there!")
        else:
            # Ignore clicks when not in manual food mode
            pass

    def draw_game(self):
        self.canvas.delete("all")

        # Draw grid
        for i in range(self.env.grid_size + 1):
            self.canvas.create_line(0, i * CELL_SIZE,
                                    self.env.grid_size * CELL_SIZE, i * CELL_SIZE,
                                    fill='#222222')
            self.canvas.create_line(i * CELL_SIZE, 0,
                                    i * CELL_SIZE, self.env.grid_size * CELL_SIZE,
                                    fill='#222222')

        # Draw snake
        for i, (row, col) in enumerate(self.env.snake):
            x1 = col * CELL_SIZE
            y1 = row * CELL_SIZE
            x2 = x1 + CELL_SIZE
            y2 = y1 + CELL_SIZE

            if i == 0:
                # Head - brighter with eyes
                self.canvas.create_rectangle(x1, y1, x2, y2, fill="#00ff00", outline="#00cc00", width=2)
                # Eyes
                eye_size = 4
                self.canvas.create_oval(x1 + 10, y1 + 10, x1 + 10 + eye_size, y1 + 10 + eye_size, fill='black')
                self.canvas.create_oval(x2 - 14, y1 + 10, x2 - 14 + eye_size, y1 + 10 + eye_size, fill='black')
            else:
                # Body
                self.canvas.create_rectangle(x1 + 2, y1 + 2, x2 - 2, y2 - 2, fill="#00cc00", outline="#008800")

        # Draw food
        food_row, food_col = self.env.food
        x1 = food_col * CELL_SIZE
        y1 = food_row * CELL_SIZE
        x2 = x1 + CELL_SIZE
        y2 = y1 + CELL_SIZE
        self.canvas.create_oval(x1 + 5, y1 + 5, x2 - 5, y2 - 5, fill="red", outline="darkred", width=2)

        self.score_label.config(text=f"Score: {self.env.score}")
        self.epsilon_label.config(text=f"ε: {self.agent.epsilon:.3f}")
        self.update()

    def auto_play(self):
        if self.playing:
            self.playing = False
            self.play_btn.config(text="▶ Watch AI Play")
            return

        self.playing = True
        self.play_btn.config(text="⏸ Pause")
        self.env.reset()
        self.run_episode()

    def run_episode(self):
        if not self.playing or self.env.done:
            self.playing = False
            self.play_btn.config(text="▶ Watch AI Play")
            if self.env.score > self.high_score:
                self.high_score = self.env.score
                self.high_score_label.config(text=f"Best: {self.high_score}")
            return

        # Check for winning score limit
        if self.env.score >= self.score_limit_var.get():
            self.env.done = True
            self.draw_game()
            tk.messagebox.showinfo("Game Over", f"Game finished (Score: {self.env.score})")
            self.playing = False
            self.play_btn.config(text="▶ Watch AI Play")
            if self.env.score > self.high_score:
                self.high_score = self.env.score
                self.high_score_label.config(text=f"Best: {self.high_score}")
            return

        state = self.env.get_state()
        action = self.agent.act(state, training=False)
        self.env.step(action)
        self.draw_game()

        delay = max(10, 510 - self.speed_var.get())
        self.after(delay, self.run_episode)

    def reset_game(self):
        self.playing = False
        self.play_btn.config(text="▶ Watch AI Play")
        self.env.reset()
        self.draw_game()

    def retrain(self):
        self.playing = False
        self.play_btn.config(text="Training...")
        self.play_btn.config(state='disabled')
        self.update()

        train_agent(self.env, self.agent, episodes=TRAIN_EPISODES)
        torch.save(self.agent.model.state_dict(), MODEL_PATH)

        self.play_btn.config(text="▶ Watch AI Play", state='normal')
        self.reset_game()
        print("✓ Model saved!")


if __name__ == "__main__":
    env = SnakeGame(grid_size=GRID_SIZE)
    agent = DeepSarsaAgent(
        use_target=USE_TARGET,
        replay_buffer=USE_REPLAY_BUFFER
    )

    if os.path.exists(MODEL_PATH):
        agent.model.load_state_dict(torch.load(MODEL_PATH))
        # keep target synced if using it
        agent.target_model.load_state_dict(agent.model.state_dict())
        agent.epsilon = 0.02  # Low exploration when playing
        print(f"✓ Loaded model from {MODEL_PATH}")
    else:
        print("No model found. Training from scratch (Deep SARSA)...")
        train_agent(env, agent, episodes=TRAIN_EPISODES)
        torch.save(agent.model.state_dict(), MODEL_PATH)
        torch.save(agent.model, "./neutron/D_SARSA_snake_full_model.pth")
        print(f"✓ Model saved to {MODEL_PATH}")
        print("✓ Full model saved to D_SARSA_snake_full_model.pth")

    # Export ONNX model (same IO: state -> Q_values for all actions)
    dummy_input = torch.randn(1, 16)
    torch.onnx.export(agent.model, dummy_input, "./neutron/D_SARSA_snake.onnx",
                      input_names=["state"], output_names=["Q_values"],
                      opset_version=12)
    print("✓ Model exported to D_SARSA_snake.onnx for Netron visualization")

    app = SnakeVisualizer(env, agent)
    app.mainloop()
