# 🧠 Reinforcement Learning — Snake Game 🐍

This repository contains **two deep reinforcement learning projects** that teach an AI to play the classic **Snake game** using:

* **Deep Q-Learning (DQN)** — in the `deep_ql` folder
* **Deep SARSA** — in the `snake-sarsa` folder

Each project includes:

* A **training loop** with PyTorch
* A **Tkinter GUI** to watch the trained agent play in real-time
* Model saving/loading for quick experimentation
* Reward shaping to guide learning

---

## 📁 Project Structure

```
ASSIGNMENT/
├── deep_ql/                  # Project A — DQN Snake
│   ├── dql_snake.py          # main file (training + GUI)
│   ├── improved_dqn_snake_10x10.pth
│   ├── neutron/              # exported ONNX / full models
│   └── *.ipynb               # training notebooks
│
├── snake-sarsa/              # Project B — Deep SARSA Snake
│   ├── snake_deep_sarsa.py   # main file (training + GUI)
│   ├── deep_sarsa_snake_optimized_*.pth
│   └── *.ipynb
│
├── notebooks/                # Shared experiment notebooks
├── .venv/                    # Optional virtual environmen
└── README.md                 # 👈 this file
```

---

## ⚙️ Requirements

Both projects use **Python** and **PyTorch**.

* Python ≥ 3.9
* [PyTorch](https://pytorch.org/get-started/locally/) (CPU or CUDA)
* NumPy
* Tkinter (comes with most Python installations — install separately on Linux)

Install them (recommended inside a virtual environment):

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install numpy
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

Linux users may need:

```bash
sudo apt-get install -y python3-tk
```

---

## 🧠 Project A — Deep Q-Learning (DQN)

**Location:** `deep_ql/dql_snake.py`

### 📌 What it does

* Implements a **DQN agent** with **Double DQN** updates and a **target network**.
* Uses experience replay (buffer size = 100,000) and mini-batches to stabilize training.
* Supports ONNX export for visualization in [Netron](https://netron.app).

### 🏋️ Train the Agent

From the root folder:

```bash
cd deep_ql
python dql_snake.py
```

If no trained model is found, it trains for `20,000` episodes by default, then saves:

* `improved_dqn_snake_10x10.pth` — weights only
* `neutron/dqn_snake_full_model.pth` — full model
* `neutron/dqn_snake.onnx` — ONNX export

### 🎮 Watch the Agent

After training, a **Tkinter GUI** opens where you can:

* ▶ **Watch AI Play**
* 🔄 **Reset**
* 🍎 **Toggle manual food mode** (click grid to place food)
* 🎓 **Retrain** (extra 5,000 episodes)
* 🐢 Speed slider (10–500 ms delay)

---

## 🧠 Project B — Deep SARSA

**Location:** `snake-sarsa/snake_deep_sarsa.py`

### 📌 What it does

* Implements a **SARSA agent** with a **deep neural network** instead of a Q-table.
* Trains with **on-policy updates** (Q(s,a) → Q(s′,a′)), no replay buffer or target net.
* Uses richer state features (danger in 8 directions, food direction, movement direction).
* GUI similar to the DQN project.

### 🏋️ Train the Agent

From the root folder:

```bash
cd snake-sarsa
python snake_deep_sarsa.py
```

If no trained model is found, it trains for `25,000` episodes and saves weights automatically.

### 🎮 Watch the Agent

Same GUI controls as the DQN project:

* ▶ Watch AI Play
* 🔄 Reset
* 🍎 Manual Food Mode
* 🎓 Train More
* 🐢 Speed control

---

## 📊 State & Action Representation

| Feature                 | Count | Description                                    |
| ----------------------- | ----- | ---------------------------------------------- |
| Danger in 8 directions  | 8     | Up, Down, Left, Right, and diagonals           |
| Current direction flags | 4     | Up / Down / Left / Right                       |
| Food location relative  | 4     | Whether food is above/below/left/right of head |
| **Total**               | 16    | Input size to the neural network               |

**Actions (3):**

* `0`: Go straight
* `1`: Turn right
* `2`: Turn left

---

## 🏅 Reward Structure (Both)

| Event               | Reward                  |
| ------------------- | ----------------------- |
| Eat food            | +10 (DQN) / +12 (SARSA) |
| Move closer to food | +1 (DQN) / +2 (SARSA)   |
| Move away from food | −1                      |
| Timeout             | −10                     |
| Die                 | −10                     |
| Small living bonus  | +0.05 (SARSA only)      |

---

## 🧪 Common Troubleshooting

| Issue                                          | Solution                                                                               |
| ---------------------------------------------- | -------------------------------------------------------------------------------------- |
| `ModuleNotFoundError: No module named 'numpy'` | Install `numpy` in your active venv                                                    |
| `No module named 'torch'`                      | Install PyTorch with correct index-url for your platform                               |
| `Tkinter not found` (Linux)                    | `sudo apt-get install python3-tk`                                                      |
| Model file not found                           | Run training once to generate the `.pth` file                                          |
| GUI freezes during training                    | Training happens before GUI launches, so wait until training finishes (first run only) |

---

## 📡 Export (DQN only)

The DQN project automatically exports to ONNX:

```
neutron/dqn_snake.onnx
```

You can open this file in [https://netron.app](https://netron.app) to explore the network structure visually.

---

## 📝 License

Specify your license here (MIT, Apache-2.0, etc.).

---

## 🙌 Credits

* **Deep Q-Learning Implementation** — based on standard Double DQN with replay buffer
* **Deep SARSA Implementation** — on-policy deep RL adapted for Snake
* GUI: Tkinter
* Code: Python + PyTorch
