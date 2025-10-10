import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import tkinter as tk
from tkinter import messagebox
import os

# =====================
# CONFIGURATION
# =====================
GRID_SIZE = 10
MODEL_PATH = 'deep_sarsa_snake_optimized_25000_10x10.pth'
CELL_SIZE = 40


# =====================
# ENVIRONMENT
# =====================
# class SnakeGame:
#     def __init__(self, grid_size=GRID_SIZE):
#         self.grid_size = grid_size
#         self.reset()

#     def reset(self):
#         """Reset game to starting state."""
#         center = self.grid_size // 2
#         self.snake = [(center, center), (center, center - 1), (center, center - 2)]
#         self.direction = (0, 1)  # start moving right
#         self.food = self._place_food()
#         self.score = 0
#         self.steps_without_food = 0
#         self.done = False
#         self.prev_distance = self._get_distance_to_food()
#         return self.get_state()

#     def _place_food(self):
#         """Place food randomly on empty cell."""
#         while True:
#             food = (random.randint(0, self.grid_size - 1),
#                     random.randint(0, self.grid_size - 1))
#             if food not in self.snake:
#                 return food

#     def _get_distance_to_food(self):
#         """Manhattan distance between snake head and food."""
#         head = self.snake[0]
#         return abs(head[0] - self.food[0]) + abs(head[1] - self.food[1])

#     def get_state(self):
#         """Return 16-dimensional state representation."""
#         head = self.snake[0]
#         # Danger detection in 8 directions
#         danger_up = self._is_collision((head[0] - 1, head[1]))
#         danger_down = self._is_collision((head[0] + 1, head[1]))
#         danger_left = self._is_collision((head[0], head[1] - 1))
#         danger_right = self._is_collision((head[0], head[1] + 1))
#         danger_up_left = self._is_collision((head[0] - 1, head[1] - 1))
#         danger_up_right = self._is_collision((head[0] - 1, head[1] + 1))
#         danger_down_left = self._is_collision((head[0] + 1, head[1] - 1))
#         danger_down_right = self._is_collision((head[0] + 1, head[1] + 1))

#         # Current direction
#         dir_up = (self.direction == (-1, 0))
#         dir_down = (self.direction == (1, 0))
#         dir_left = (self.direction == (0, -1))
#         dir_right = (self.direction == (0, 1))

#         # Food position relative to head
#         food_up = (self.food[0] < head[0])
#         food_down = (self.food[0] > head[0])
#         food_left = (self.food[1] < head[1])
#         food_right = (self.food[1] > head[1])

#         state = [
#             danger_up, danger_down, danger_left, danger_right,
#             danger_up_left, danger_up_right, danger_down_left, danger_down_right,
#             dir_up, dir_down, dir_left, dir_right,
#             food_up, food_down, food_left, food_right
#         ]

#         return np.array(state, dtype=np.float32)

#     def _turn_right(self, direction):
#         return (direction[1], -direction[0])

#     def _turn_left(self, direction):
#         return (-direction[1], direction[0])

#     def _is_collision(self, pos):
#         """Detect wall or self-collision."""
#         row, col = pos
#         if row < 0 or row >= self.grid_size or col < 0 or col >= self.grid_size:
#             return True
#         if pos in self.snake:
#             return True
#         return False

#     def step(self, action):
#         """Perform one action step and return (next_state, reward, done)."""
#         # 0=straight, 1=right, 2=left
#         if action == 1:
#             self.direction = self._turn_right(self.direction)
#         elif action == 2:
#             self.direction = self._turn_left(self.direction)

#         head = self.snake[0]
#         new_head = (head[0] + self.direction[0], head[1] + self.direction[1])

#         # If hit wall or self → big negative reward
#         if self._is_collision(new_head):
#             self.done = True
#             return self.get_state(), -10, True

#         # Move snake
#         self.snake.insert(0, new_head)
#         current_distance = self._get_distance_to_food()

#         # Initialize reward
#         reward = 0.0

#         # Reward shaping: stronger move-toward-food reward
#         if new_head == self.food:
#             self.score += 1
#             reward = 12  # higher reward for food
#             self.food = self._place_food()
#             self.steps_without_food = 0
#             self.prev_distance = self._get_distance_to_food()
#         else:
#             self.snake.pop()  # remove tail segment
#             if current_distance < self.prev_distance:
#                 reward = 2  # move closer
#             else:
#                 reward = -1  # move away
#             self.prev_distance = current_distance
#             self.steps_without_food += 1

#         # Small survival reward (encourage not dying)
#         reward += 0.05

#         # End episode if snake takes too long
#         if self.steps_without_food > 100 * self.grid_size:
#             self.done = True
#             reward = -10

#         return self.get_state(), reward, self.done

# =====================
# ENVIRONMENT
# =====================
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
        """Random food, or a manual position if provided and valid."""
        if manual_position is not None:
            if manual_position not in self.snake:
                return manual_position
        while True:
            food = (random.randint(0, self.grid_size - 1),
                    random.randint(0, self.grid_size - 1))
            if food not in self.snake:
                return food

    def place_food_at(self, position):
        """Manually place food at a specific position."""
        if position not in self.snake:
            self.food = position
            return True
        return False

    def _get_distance_to_food(self):
        """Manhattan distance to food."""
        head = self.snake[0]
        return abs(head[0] - self.food[0]) + abs(head[1] - self.food[1])

    def get_state(self):
        """
        16 features:
        - Danger in 8 directions (8)
        - Current direction (4)
        - Food relative location (4)
        """
        head = self.snake[0]

        # Danger in 8 directions
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

        # Food relative position
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
        """Rotate direction 90° clockwise."""
        return (direction[1], -direction[0])

    def _turn_left(self, direction):
        """Rotate direction 90° counter-clockwise."""
        return (-direction[1], direction[0])

    def _is_collision(self, pos):
        """Check wall or self collision."""
        row, col = pos
        if row < 0 or row >= self.grid_size or col < 0 or col >= self.grid_size:
            return True
        if pos in self.snake:
            return True
        return False

    def step(self, action):
        """
        Actions: 0=straight, 1=turn right, 2=turn left
        """
        # Update direction from action
        if action == 1:
            self.direction = self._turn_right(self.direction)
        elif action == 2:
            self.direction = self._turn_left(self.direction)

        # Move
        head = self.snake[0]
        new_head = (head[0] + self.direction[0], head[1] + self.direction[1])

        # Collision check (death)
        if self._is_collision(new_head):
            self.done = True
            return self.get_state(), -10, True

        # Advance snake
        self.snake.insert(0, new_head)

        # Distance-based shaping
        current_distance = self._get_distance_to_food()
        reward = 0

        # Eat food?
        if new_head == self.food:
            self.score += 1
            reward = 12  # DQL value
            self.food = self._place_food()
            self.steps_without_food = 0
            self.prev_distance = self._get_distance_to_food()
        else:
            # Move tail
            self.snake.pop()

            # Shaping: closer = +2, away = -1 (DQL)
            if current_distance < self.prev_distance:
                reward = 2
            else:
                reward = -1

            self.prev_distance = current_distance
            self.steps_without_food += 1

        # Small survival reward (encourage not dying)
        reward += 0.05

        # Timeout -> end (DQL)
        if self.steps_without_food > 100 * self.grid_size:
            self.done = True
            reward = -10

        return self.get_state(), reward, self.done



# =====================
# Q-NETWORK (Bigger capacity for better learning)
# =====================
class DQN(nn.Module):
    def __init__(self, input_size=16, hidden1=512, hidden2=512, hidden3=256, output_size=3):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, hidden3)
        self.fc4 = nn.Linear(hidden3, output_size)
        self.dropout = nn.Dropout(0.05)  # slightly less dropout to retain info

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        return self.fc4(x)


# =====================
# SARSA AGENT
# =====================
class SARSAAgent:
    def __init__(self, state_size=16, action_size=3, learning_rate=0.0003,
                 gamma=0.95, epsilon=1.0, epsilon_decay=0.9998, epsilon_min=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.model = DQN(state_size, 512, 512, 256, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.SmoothL1Loss()

    def act(self, state, training=True):
        """Choose action using ε-greedy policy."""
        if training and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return torch.argmax(q_values).item()

    def learn(self, state, action, reward, next_state, next_action, done):
        """Update Q-values using SARSA update rule."""
        state_t = torch.FloatTensor(state).unsqueeze(0)
        next_state_t = torch.FloatTensor(next_state).unsqueeze(0)

        q_pred = self.model(state_t)[0, action]

        with torch.no_grad():
            q_next = self.model(next_state_t)[0, next_action] if not done else torch.tensor(0.0)
            target = reward + self.gamma * q_next

        loss = self.criterion(q_pred, target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return loss

    def decay_epsilon(self):
        """Gradually reduce ε for less exploration over time."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# =====================
# TRAINING LOOP
# =====================
def train_agent_sarsa(env, agent, episodes=25000):
    scores = []
    rewards = []
    epsilons = []
    losses = []
    # episode_losses = []
    best_score = 0
    moving_avg_window = 100

    print(f"Training Deep SARSA for {episodes} episodes...")

    for episode in range(episodes):
        state = env.reset()
        action = agent.act(state, training=True)
        total_reward = 0

        while not env.done:
            next_state, reward, done = env.step(action)
            next_action = agent.act(next_state, training=True) if not done else 0
            loss = agent.learn(state, action, reward, next_state, next_action, done)
            # episode_losses.append(loss.item())
            losses.append(loss.item())
            state, action = next_state, next_action
            total_reward += reward

        agent.decay_epsilon()
        scores.append(env.score)
        # losses.append(np.mean(episode_losses))
        rewards.append(total_reward)
        epsilons.append(agent.epsilon)


        if env.score > best_score:
            best_score = env.score

        if (episode + 1) % 100 == 0:
            avg_score = np.mean(scores[-moving_avg_window:])
            avg_loss = np.mean(losses[-moving_avg_window:]) if len(losses) >= moving_avg_window else np.mean(losses)
            print(f"Episode {episode + 1}/{episodes} | "
                  f"Score: {env.score} | Avg(100): {avg_score:.2f} | "
                  f"Best: {best_score} | ε: {agent.epsilon:.3f} | "
                  f"Avg Loss: {avg_loss:.4f}")

    print(f"\n✅ Training complete! Best score: {best_score}")
    final_avg = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
    print(f"Final 100-episode average: {final_avg:.2f}")
    return scores, rewards, epsilons, losses

class SnakeVisualizer(tk.Tk):
    def __init__(self, env, agent):
        super().__init__()
        self.env = env
        self.agent = agent
        self.title("Snake Game - Deep SARSA (DQL Style)")
        self.manual_food_mode = False

        # ---- Winning Score Limit Dropdown ----
        self.score_limit_var = tk.IntVar(value=10)
        score_limits = [i for i in range(5, 51, 5)]
        score_limit_frame = tk.Frame(self)
        score_limit_frame.pack(pady=5)
        tk.Label(score_limit_frame, text="Winning Score Limit:").pack(side=tk.LEFT)
        tk.OptionMenu(score_limit_frame, self.score_limit_var, *score_limits).pack(side=tk.LEFT)

        # ---- Canvas ----
        self.canvas = tk.Canvas(self, width=env.grid_size * CELL_SIZE,
                                height=env.grid_size * CELL_SIZE, bg='black')
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        # ---- Info bar ----
        self.info_frame = tk.Frame(self)
        self.info_frame.pack(pady=10)
        self.score_label = tk.Label(self.info_frame, text="Score: 0", font=("Arial", 14, "bold"))
        self.score_label.pack(side=tk.LEFT, padx=10)

        self.high_score = 0
        self.high_score_label = tk.Label(self.info_frame, text="Best: 0", font=("Arial", 14))
        self.high_score_label.pack(side=tk.LEFT, padx=10)

        self.epsilon_label = tk.Label(self.info_frame, text=f"ε: {agent.epsilon:.3f}", font=("Arial", 12))
        self.epsilon_label.pack(side=tk.LEFT, padx=10)

        # ---- Buttons ----
        btn_frame = tk.Frame(self)
        btn_frame.pack()
        self.play_btn = tk.Button(btn_frame, text="▶ Watch AI Play", command=self.auto_play, font=("Arial", 11))
        self.play_btn.pack(side=tk.LEFT, padx=5)

        self.reset_btn = tk.Button(btn_frame, text="🔄 Reset", command=self.reset_game, font=("Arial", 11))
        self.reset_btn.pack(side=tk.LEFT, padx=5)

        self.food_mode_btn = tk.Button(btn_frame, text="🍎 Manual Food: OFF",
                                       command=self.toggle_food_mode, font=("Arial", 11),
                                       bg="lightgray")
        self.food_mode_btn.pack(side=tk.LEFT, padx=5)

        self.retrain_btn = tk.Button(btn_frame, text="🎓 Train More", command=self.retrain, font=("Arial", 11))
        self.retrain_btn.pack(side=tk.LEFT, padx=5)

        # ---- Speed Control ----
        self.speed_var = tk.IntVar(value=100)
        speed_frame = tk.Frame(self)
        speed_frame.pack()
        tk.Label(speed_frame, text="Speed:").pack(side=tk.LEFT)
        tk.Scale(speed_frame, from_=10, to=500, orient=tk.HORIZONTAL,
                 variable=self.speed_var, length=200).pack(side=tk.LEFT)

        # ---- Mode label ----
        self.mode_label = tk.Label(self, text="Click grid to place food (Manual Food mode)",
                                   font=("Arial", 10), fg="gray")
        self.mode_label.pack()
        self.mode_label.pack_forget()  # hide initially

        self.playing = False
        self.draw_game()

    # ----------------------------------------
    # Manual Food Mode controls
    # ----------------------------------------
    def toggle_food_mode(self):
        self.manual_food_mode = not self.manual_food_mode
        if self.manual_food_mode:
            self.food_mode_btn.config(text="🍎 Manual Food: ON", bg="lightgreen")
            self.mode_label.pack()
        else:
            self.food_mode_btn.config(text="🍎 Manual Food: OFF", bg="lightgray")
            self.mode_label.pack_forget()

    def on_canvas_click(self, event):
        """Handle manual food placement."""
        col = event.x // CELL_SIZE
        row = event.y // CELL_SIZE
        if row >= self.env.grid_size or col >= self.env.grid_size or row < 0 or col < 0:
            return

        if self.manual_food_mode:
            position = (row, col)
            if self.env.place_food_at(position):
                self.draw_game()
                print(f"Food placed at: ({row}, {col})")
            else:
                print(f"Cannot place food at ({row}, {col}) - snake is there!")

    # ----------------------------------------
    # Rendering logic
    # ----------------------------------------
    def draw_game(self):
        """Draw grid, snake, and food."""
        self.canvas.delete("all")

        # Grid
        for i in range(self.env.grid_size + 1):
            self.canvas.create_line(0, i * CELL_SIZE,
                                    self.env.grid_size * CELL_SIZE, i * CELL_SIZE,
                                    fill='#222222')
            self.canvas.create_line(i * CELL_SIZE, 0,
                                    i * CELL_SIZE, self.env.grid_size * CELL_SIZE,
                                    fill='#222222')

        # Snake
        for i, (row, col) in enumerate(self.env.snake):
            x1, y1 = col * CELL_SIZE, row * CELL_SIZE
            x2, y2 = x1 + CELL_SIZE, y1 + CELL_SIZE
            if i == 0:
                self.canvas.create_rectangle(x1, y1, x2, y2, fill="#00ff00", outline="#00cc00", width=2)
                eye_size = 4
                self.canvas.create_oval(x1 + 10, y1 + 10, x1 + 10 + eye_size, y1 + 10 + eye_size, fill='black')
                self.canvas.create_oval(x2 - 14, y1 + 10, x2 - 14 + eye_size, y1 + 10 + eye_size, fill='black')
            else:
                self.canvas.create_rectangle(x1 + 2, y1 + 2, x2 - 2, y2 - 2, fill="#00cc00", outline="#008800")

        # Food
        food_row, food_col = self.env.food
        x1, y1 = food_col * CELL_SIZE, food_row * CELL_SIZE
        x2, y2 = x1 + CELL_SIZE, y1 + CELL_SIZE
        self.canvas.create_oval(x1 + 5, y1 + 5, x2 - 5, y2 - 5, fill="red", outline="darkred", width=2)

        self.score_label.config(text=f"Score: {self.env.score}")
        self.epsilon_label.config(text=f"ε: {self.agent.epsilon:.3f}")
        self.update()

    # ----------------------------------------
    # Control Buttons
    # ----------------------------------------
    def auto_play(self):
        """AI play toggle."""
        if self.playing:
            self.playing = False
            self.play_btn.config(text="▶ Watch AI Play")
            return

        self.playing = True
        self.play_btn.config(text="⏸ Pause")
        self.env.reset()
        self.run_episode()

    def run_episode(self):
        """AI step loop."""
        if not self.playing or self.env.done:
            self.playing = False
            self.play_btn.config(text="▶ Watch AI Play")
            if self.env.score > self.high_score:
                self.high_score = self.env.score
                self.high_score_label.config(text=f"Best: {self.high_score}")
            return

        # Stop if winning limit reached
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

        # Agent move
        state = self.env.get_state()
        action = self.agent.act(state, training=False)
        self.env.step(action)
        self.draw_game()

        delay = max(10, 510 - self.speed_var.get())
        self.after(delay, self.run_episode)

    def reset_game(self):
        """Reset board."""
        self.playing = False
        self.play_btn.config(text="▶ Watch AI Play")
        self.env.reset()
        self.draw_game()

    def retrain(self):
        """Continue training."""
        self.playing = False
        self.play_btn.config(text="Training...")
        self.play_btn.config(state='disabled')
        self.update()

        train_agent_sarsa(self.env, self.agent, episodes=5000)
        torch.save(self.agent.model.state_dict(), MODEL_PATH)

        self.play_btn.config(text="▶ Watch AI Play", state='normal')
        self.reset_game()
        print("✓ Model retrained and saved!")

# =====================
# MAIN
# =====================
if __name__ == "__main__":
    env = SnakeGame(grid_size=GRID_SIZE)
    agent = SARSAAgent()

    if os.path.exists(MODEL_PATH):
        agent.model.load_state_dict(torch.load(MODEL_PATH))
        agent.epsilon = 0.01
        print(f"✓ Loaded trained Deep SARSA model from {MODEL_PATH}")
    else:
        print("🚀 No model found — training Deep SARSA from scratch...")
        train_agent_sarsa(env, agent, episodes=25000)
        torch.save(agent.model.state_dict(), MODEL_PATH)
        print(f"✓ Model saved to {MODEL_PATH}")

    app = SnakeVisualizer(env, agent)
    app.mainloop()

