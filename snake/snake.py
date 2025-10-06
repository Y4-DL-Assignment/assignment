import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import tkinter as tk
import os

# Configuration
GRID_SIZE = 10
MODEL_PATH = 'improved_dqn_snake_10x10.pth'
CELL_SIZE = 40


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

    def _place_food(self):
        while True:
            food = (random.randint(0, self.grid_size - 1),
                    random.randint(0, self.grid_size - 1))
            if food not in self.snake:
                return food

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

        return np.array(state, dtype=float)

    def _turn_right(self, direction):
        """Rotate direction 90° clockwise"""
        return (direction[1], -direction[0])

    def _turn_left(self, direction):
        """Rotate direction 90° counter-clockwise"""
        return (-direction[1], direction[0])

    def _is_collision(self, pos):
        """Check if position is collision"""
        row, col = pos
        # Wall collision
        if row < 0 or row >= self.grid_size or col < 0 or col >= self.grid_size:
            return True
        # Self collision
        if pos in self.snake:
            return True
        return False

    def step(self, action):
        """
        Actions: 0=straight, 1=turn right, 2=turn left
        """
        # Update direction based on action
        if action == 1:  # Turn right
            self.direction = self._turn_right(self.direction)
        elif action == 2:  # Turn left
            self.direction = self._turn_left(self.direction)

        # Move snake
        head = self.snake[0]
        new_head = (head[0] + self.direction[0], head[1] + self.direction[1])

        # Check collision
        if self._is_collision(new_head):
            self.done = True
            return self.get_state(), -10, True  # Large penalty for dying

        self.snake.insert(0, new_head)

        # Calculate distance-based reward
        current_distance = self._get_distance_to_food()

        # Check if food eaten
        reward = 0
        if new_head == self.food:
            self.score += 1
            reward = 10  # Large reward for eating food
            self.food = self._place_food()
            self.steps_without_food = 0
            self.prev_distance = self._get_distance_to_food()
        else:
            self.snake.pop()  # Remove tail

            # Reward shaping: encourage moving toward food
            if current_distance < self.prev_distance:
                reward = 1  # Moving closer to food
            else:
                reward = -1  # Moving away from food

            self.prev_distance = current_distance
            self.steps_without_food += 1

        # Penalty for taking too long (prevent infinite loops)
        if self.steps_without_food > 100 * self.grid_size:
            self.done = True
            reward = -10

        return self.get_state(), reward, self.done


class DQN(nn.Module):
    def __init__(self, input_size=16, hidden_size=256, output_size=3):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc4 = nn.Linear(hidden_size // 2, output_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        return self.fc4(x)


class DQNAgent:
    def __init__(self, state_size=16, action_size=3, learning_rate=0.0005,
                 gamma=0.95, epsilon=1.0, epsilon_decay=0.9995, epsilon_min=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=100000)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Main and target networks
        self.model = DQN(state_size, 256, action_size)
        self.target_model = DQN(state_size, 256, action_size)
        self.update_target_network()

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.SmoothL1Loss()  # Huber loss - more stable
        self.update_counter = 0

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, training=True):
        if training and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return torch.argmax(q_values).item()

    def replay(self, batch_size=128):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)

        states = torch.FloatTensor([x[0] for x in minibatch])
        actions = torch.LongTensor([x[1] for x in minibatch])
        rewards = torch.FloatTensor([x[2] for x in minibatch])
        next_states = torch.FloatTensor([x[3] for x in minibatch])
        dones = torch.FloatTensor([x[4] for x in minibatch])

        # Current Q values
        current_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze()

        # Double DQN: use main network to select action, target network to evaluate
        with torch.no_grad():
            next_actions = self.model(next_states).max(1)[1]
            next_q = self.target_model(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
            target_q = rewards + (1 - dones) * self.gamma * next_q

        loss = self.criterion(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        # Update target network periodically
        self.update_counter += 1
        if self.update_counter % 500 == 0:
            self.update_target_network()

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def train_agent(env, agent, episodes=50000):
    scores = []
    best_score = 0
    moving_avg_window = 100

    print(f"Training for {episodes} episodes with improved rewards...")

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0

        while not env.done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.replay()

            state = next_state
            total_reward += reward

        agent.decay_epsilon()
        scores.append(env.score)

        if env.score > best_score:
            best_score = env.score

        if (episode + 1) % 100 == 0:
            avg_score = np.mean(scores[-moving_avg_window:])
            print(f"Episode {episode + 1}/{episodes} | "
                  f"Score: {env.score} | "
                  f"Avg({moving_avg_window}): {avg_score:.2f} | "
                  f"Best: {best_score} | "
                  f"ε: {agent.epsilon:.3f}")

    print(f"\nTraining complete! Best score: {best_score}")
    final_avg = np.mean(scores[-100:])
    print(f"Final 100-episode average: {final_avg:.2f}")
    return scores


class SnakeVisualizer(tk.Tk):
    def __init__(self, env, agent):
        super().__init__()
        self.env = env
        self.agent = agent
        self.title("Snake Game - Improved DQN")

        self.canvas = tk.Canvas(self, width=env.grid_size * CELL_SIZE,
                                height=env.grid_size * CELL_SIZE, bg='black')
        self.canvas.pack()

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

        self.retrain_btn = tk.Button(btn_frame, text="🎓 Train More",
                                     command=self.retrain, font=("Arial", 11))
        self.retrain_btn.pack(side=tk.LEFT, padx=5)

        self.speed_var = tk.IntVar(value=100)
        speed_frame = tk.Frame(self)
        speed_frame.pack()
        tk.Label(speed_frame, text="Speed:").pack(side=tk.LEFT)
        tk.Scale(speed_frame, from_=10, to=500, orient=tk.HORIZONTAL,
                 variable=self.speed_var, length=200).pack(side=tk.LEFT)

        self.playing = False
        self.draw_game()

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

        train_agent(self.env, self.agent, episodes=5000)
        torch.save(self.agent.model.state_dict(), MODEL_PATH)

        self.play_btn.config(text="▶ Watch AI Play", state='normal')
        self.reset_game()
        print("✓ Model saved!")


if __name__ == "__main__":
    env = SnakeGame(grid_size=GRID_SIZE)
    agent = DQNAgent()

    if os.path.exists(MODEL_PATH):
        agent.model.load_state_dict(torch.load(MODEL_PATH))
        agent.update_target_network()
        agent.epsilon = 0.01  # Low exploration when playing
        print(f"✓ Loaded model from {MODEL_PATH}")
    else:
        print("No model found. Training from scratch...")
        train_agent(env, agent, episodes=50000)
        torch.save(agent.model.state_dict(), MODEL_PATH)
        print(f"✓ Model saved to {MODEL_PATH}")

    app = SnakeVisualizer(env, agent)
    app.mainloop()