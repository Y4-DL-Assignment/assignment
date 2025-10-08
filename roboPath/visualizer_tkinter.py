# import tkinter as tk
# from tkinter import ttk, messagebox
# import threading
# import os
# import torch
# import numpy as np

# # Assuming all classes are imported
# from gridworld_env import GridWorldEnv
# from dqn_agent import DQNAgent
# from train_dqn import train_dqn

# # Colors
# COLORS = {
#     'bg': '#f0f0f0',
#     'grid_bg': '#ffffff',
#     'agent': '#0066ff',
#     'goal': '#00ff00',
#     'bomb': '#ff0000',
#     'obstacle': '#333333',
#     'wall': '#cccccc',
#     'empty': '#ffffff',
#     'button': '#4CAF50',
#     'button_hover': '#45a049',
#     'danger': '#ff6b6b'
# }

# GRID_SIZE = 10
# CELL_SIZE = 50
# MARGIN = 2

# class GridWorldGUI:
#     def __init__(self, root):
#         self.eval_speed = 200   # default speed in milliseconds
#         self.manual_goal_mode = False
#         self.root = root
#         self.root.title("GridWorld DQN - Snake-Style Training")
#         self.root.configure(bg=COLORS['bg'])
        
#         # Initialize environment and agent
#         self.env = GridWorldEnv(grid_size=GRID_SIZE, max_steps=1000, 
#                                 use_obstacles=True, fixed_start=True)
#         self.agent = DQNAgent(state_dim=20, num_actions=4, lr=0.001, 
#                              gamma=0.95, batch_size=128, tau=0.005)
        
#         # Training state
#         self.training = False
#         self.evaluating = False
#         self.stop_requested = False
#         self.training_thread = None
#         self.training_data = None
        
#         # Create UI
#         self.create_widgets()
#         self.draw_grid()
        
#     def create_widgets(self):
#         """Create all UI widgets."""
#         # Main container
#         main_frame = tk.Frame(self.root, bg=COLORS['bg'])
#         main_frame.pack(padx=20, pady=20)
        
#         # Top info panel
#         info_frame = tk.Frame(main_frame, bg=COLORS['bg'])
#         info_frame.pack(pady=(0, 10))
        
#         self.info_label = tk.Label(info_frame, text="Ready to train or evaluate", 
#                                    font=('Arial', 12), bg=COLORS['bg'])
#         self.info_label.pack()
        
#         self.stats_label = tk.Label(info_frame, 
#                                     text="Score: 0 | Steps: 0 | Episode: 0", 
#                                     font=('Arial', 10), bg=COLORS['bg'])
#         self.stats_label.pack()
        
#         # Canvas for grid
#         canvas_frame = tk.Frame(main_frame, bg=COLORS['bg'])
#         canvas_frame.pack()
        
#         canvas_size = GRID_SIZE * (CELL_SIZE + MARGIN) + MARGIN
#         self.canvas = tk.Canvas(canvas_frame, width=canvas_size, height=canvas_size,
#                                bg=COLORS['grid_bg'], highlightthickness=1,
#                                highlightbackground='#999999')
#         self.canvas.pack()

#         # Bind mouse click for manual goal placement
#         self.canvas.bind("<Button-1>", self.on_canvas_click)
        
#         # Button panel
#         button_frame = tk.Frame(main_frame, bg=COLORS['bg'])
#         button_frame.pack(pady=(10, 0))
        
#         self.train_btn = tk.Button(button_frame, text="🎓 Train (20k episodes)",
#                                    command=self.start_training, font=('Arial', 11),
#                                    bg=COLORS['button'], fg='white', padx=15, pady=8,
#                                    relief=tk.RAISED, cursor='hand2')
#         self.train_btn.grid(row=0, column=0, padx=5)
        
#         self.eval_btn = tk.Button(button_frame, text="▶️ Evaluate",
#                                   command=self.start_evaluation, font=('Arial', 11),
#                                   bg='#2196F3', fg='white', padx=15, pady=8,
#                                   relief=tk.RAISED, cursor='hand2')
#         self.eval_btn.grid(row=0, column=1, padx=5)
        
#         self.stop_btn = tk.Button(button_frame, text="⏹️ Stop",
#                                   command=self.stop_action, font=('Arial', 11),
#                                   bg=COLORS['danger'], fg='white', padx=15, pady=8,
#                                   relief=tk.RAISED, cursor='hand2', state=tk.DISABLED)
#         self.stop_btn.grid(row=0, column=2, padx=5)
        
#         self.reset_btn = tk.Button(button_frame, text="🔄 Reset",
#                                    command=self.reset_env, font=('Arial', 11),
#                                    bg='#FF9800', fg='white', padx=15, pady=8,
#                                    relief=tk.RAISED, cursor='hand2')
#         self.reset_btn.grid(row=0, column=3, padx=5)

#         # Manual goal placement toggle
#         self.manual_goal_btn = tk.Button(button_frame, text="📍 Manual Goal: OFF",
#                                          command=self.toggle_manual_goal_mode,
#                                          font=('Arial', 11), bg='#9C27B0', fg='white',
#                                          padx=15, pady=8, relief=tk.RAISED, cursor='hand2')
#         self.manual_goal_btn.grid(row=0, column=4, padx=5)

#         # Speed slider
#         speed_frame = tk.Frame(main_frame, bg=COLORS['bg'])
#         speed_frame.pack(pady=(10, 0))
        
#         tk.Label(speed_frame, text="Evaluation Speed:", bg=COLORS['bg'], font=('Arial', 10)).pack(side=tk.LEFT, padx=5)
#         self.speed_slider = ttk.Scale(speed_frame, from_=50, to=1000, orient="horizontal",
#                                       command=self.update_speed)
#         self.speed_slider.set(self.eval_speed)  # default
#         self.speed_slider.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
    
        
#         # Progress bar
#         self.progress_frame = tk.Frame(main_frame, bg=COLORS['bg'])
#         self.progress_frame.pack(pady=(10, 0), fill=tk.X)
        
#         self.progress_label = tk.Label(self.progress_frame, text="Training Progress:",
#                                        font=('Arial', 10), bg=COLORS['bg'])
#         self.progress_label.pack()
        
#         self.progress_bar = ttk.Progressbar(self.progress_frame, length=400, 
#                                            mode='determinate')
#         self.progress_bar.pack(pady=5)
#         self.progress_bar.pack_forget()  # Hide initially
        
#         # Legend
#         legend_frame = tk.Frame(main_frame, bg=COLORS['bg'])
#         legend_frame.pack(pady=(10, 0))
        
#         legend_items = [
#             ("🔵 Agent", COLORS['agent']),
#             ("🟢 Goal", COLORS['goal']),
#             ("🔴 Bomb", COLORS['bomb']),
#             ("⬛ Obstacle", COLORS['obstacle'])
#         ]
        
#         for i, (text, color) in enumerate(legend_items):
#             lbl = tk.Label(legend_frame, text=text, font=('Arial', 9),
#                           bg=COLORS['bg'])
#             lbl.grid(row=0, column=i, padx=10)

#     def toggle_manual_goal_mode(self):
#         """Enable/disable manual goal placement mode."""
#         self.manual_goal_mode = not self.manual_goal_mode
#         if self.manual_goal_mode:
#             self.manual_goal_btn.config(text="📍 Manual Goal: ON", bg="#673AB7")
#         else:
#             self.manual_goal_btn.config(text="📍 Manual Goal: OFF", bg="#9C27B0")


#     def on_canvas_click(self, event):
#         """Handle canvas clicks for manual goal placement."""
#         if not self.manual_goal_mode:
#             return
#         # Determine clicked cell
#         col = event.x // (CELL_SIZE + MARGIN)
#         row = event.y // (CELL_SIZE + MARGIN)
#         if 0 <= row < self.env.n and 0 <= col < self.env.n:
#             self.env.set_goal(row, col)
#             self.draw_grid()
#             self.info_label.config(text=f"Goal manually placed at ({row}, {col})")

#     def update_speed(self, value):
#         """Update evaluation speed based on slider."""
#         self.eval_speed = int(float(value))
    
#     def draw_grid(self):
#         """Draw the grid with all entities."""
#         self.canvas.delete("all")
        
#         # Draw grid cells
#         for i in range(self.env.n):
#             for j in range(self.env.n):
#                 x1 = j * (CELL_SIZE + MARGIN) + MARGIN
#                 y1 = i * (CELL_SIZE + MARGIN) + MARGIN
#                 x2 = x1 + CELL_SIZE
#                 y2 = y1 + CELL_SIZE
                
#                 # Determine cell color
#                 pos = (i, j)
#                 if pos == self.env.agent:
#                     color = COLORS['agent']
#                 elif pos == self.env.goal:
#                     color = COLORS['goal']
#                 elif pos == self.env.bomb:
#                     color = COLORS['bomb']
#                 elif pos in self.env.obstacles:
#                     color = COLORS['obstacle']
#                 else:
#                     color = COLORS['empty']
                
#                 # Draw cell
#                 self.canvas.create_rectangle(x1, y1, x2, y2, fill=color,
#                                             outline='#cccccc', width=1)
                
#                 # Add labels
#                 cx = (x1 + x2) / 2
#                 cy = (y1 + y2) / 2
#                 if pos == self.env.agent:
#                     self.canvas.create_text(cx, cy, text="A", font=('Arial', 16, 'bold'),
#                                           fill='white')
#                 elif pos == self.env.goal:
#                     self.canvas.create_text(cx, cy, text="G", font=('Arial', 16, 'bold'),
#                                           fill='white')
#                 elif pos == self.env.bomb:
#                     self.canvas.create_text(cx, cy, text="B", font=('Arial', 16, 'bold'),
#                                           fill='white')
#                 elif pos in self.env.obstacles:
#                     self.canvas.create_text(cx, cy, text="W", font=('Arial', 16, 'bold'),
#                                           fill='white')
        
#         self.root.update()
    
#     def update_stats(self):
#         """Update statistics display."""
#         self.stats_label.config(
#             text=f"Score: {self.env.score} | Steps: {self.env.t} | "
#                  f"Steps w/o goal: {self.env.steps_without_goal}"
#         )
    
#     def reset_env(self):
#         """Reset environment and redraw."""
#         self.env.reset()
#         self.draw_grid()
#         self.update_stats()
#         self.info_label.config(text="Environment reset")
    
#     def start_training(self):
#         """Start training in background thread."""
#         if self.training or self.evaluating:
#             messagebox.showwarning("Busy", "Already training or evaluating!")
#             return
        
#         self.training = True
#         self.stop_requested = False
        
#         # Update UI
#         self.train_btn.config(state=tk.DISABLED)
#         self.eval_btn.config(state=tk.DISABLED)
#         self.stop_btn.config(state=tk.NORMAL)
#         self.info_label.config(text="Training in progress...")
#         self.progress_label.pack()
#         self.progress_bar.pack(pady=5)
#         self.progress_bar['value'] = 0
        
#         # Start training thread
#         self.training_thread = threading.Thread(target=self.train_worker, daemon=True)
#         self.training_thread.start()
        
#         # Monitor training
#         self.monitor_training()
    
#     def train_worker(self):
#         """Training worker (runs in background thread)."""
#         try:
#             # Create new environment and agent for training
#             train_env = GridWorldEnv(grid_size=GRID_SIZE, max_steps=1000, 
#                                     use_obstacles=True, fixed_start=True)
#             train_agent = DQNAgent(state_dim=20, num_actions=4, lr=0.001, 
#                                   gamma=0.95, batch_size=128, tau=0.005)
            
#             # Training with progress callback
#             episodes = 5000
#             epsilon = 1.0
#             end_epsilon = 0.01
#             epsilon_decay = 0.9995
            
#             all_rewards = []
#             all_scores = []
#             success_window = []
#             goal_count = 0
            
#             for ep in range(episodes):
#                 if self.stop_requested:
#                     print("\nTraining stopped by user")
#                     break
                
#                 state = train_env.reset()
#                 total_reward = 0
#                 done = False
#                 episode_goals = 0
                
#                 while not done:
#                     if self.stop_requested:
#                         break
                    
#                     action = train_agent.select_action(state, epsilon)
#                     result = train_env.step(action)
                    
#                     train_agent.store_experience(state, action, result.reward, 
#                                                 result.next_state, result.done)
#                     train_agent.train_step()
                    
#                     state = result.next_state
#                     total_reward += result.reward
#                     done = result.done
                    
#                     if result.info.get("reason") == "goal":
#                         goal_count += 1
#                         episode_goals += 1
                
#                 if self.stop_requested:
#                     break
                
#                 success_window.append(1 if episode_goals > 0 else 0)
#                 if len(success_window) > 100:
#                     success_window.pop(0)
                
#                 epsilon = max(end_epsilon, epsilon * epsilon_decay)
#                 all_rewards.append(total_reward)
#                 all_scores.append(episode_goals)
                
#                 # Update progress
#                 progress = (ep + 1) / episodes * 100
#                 self.progress_bar['value'] = progress
                
#                 if (ep + 1) % 500 == 0:
#                     avg_score = np.mean(all_scores[-100:]) if len(all_scores) >= 100 else np.mean(all_scores)
#                     success_rate = np.mean(success_window) * 100 if success_window else 0
#                     print(f"\n[Episode {ep+1:>5}/{episodes}] "
#                           f"Avg Score: {avg_score:.2f} | "
#                           f"Success: {success_rate:.1f}% | "
#                           f"Goals: {goal_count} | "
#                           f"ε: {epsilon:.4f}")
            
#             if not self.stop_requested:
#                 # Save model
#                 train_agent.save('trained_gridworld_dqn.pth')
                
#                 # Load into main agent
#                 self.agent.load('trained_gridworld_dqn.pth')
                
#                 final_score = np.mean(all_scores[-100:]) if len(all_scores) >= 100 else np.mean(all_scores)
#                 final_success = np.mean(success_window) * 100 if success_window else 0
                
#                 print(f"\n{'='*60}")
#                 print(f"Training Complete!")
#                 print(f"Final Avg Score: {final_score:.2f} goals/episode")
#                 print(f"Final Success Rate: {final_success:.1f}%")
#                 print(f"Total Goals: {goal_count}")
#                 print(f"{'='*60}\n")
        
#         except Exception as e:
#             print(f"Training error: {e}")
#             import traceback
#             traceback.print_exc()
        
#         finally:
#             self.training = False
    
#     def monitor_training(self):
#         """Monitor training progress (runs in main thread)."""
#         if self.training and self.training_thread and self.training_thread.is_alive():
#             # Still training, check again in 500ms
#             self.root.after(500, self.monitor_training)
#         else:
#             # Training finished
#             self.training = False
#             self.train_btn.config(state=tk.NORMAL)
#             self.eval_btn.config(state=tk.NORMAL)
#             self.stop_btn.config(state=tk.DISABLED)
            
#             if self.stop_requested:
#                 self.info_label.config(text="Training stopped by user")
#             else:
#                 self.info_label.config(text="Training complete! Model saved.")
            
#             self.progress_bar.pack_forget()
#             self.progress_label.pack_forget()
    
#     def start_evaluation(self):
#         """Start evaluation mode."""
#         if self.training or self.evaluating:
#             messagebox.showwarning("Busy", "Already training or evaluating!")
#             return
        
#         # Check if model exists
#         if not os.path.exists('trained_gridworld_dqn.pth'):
#             messagebox.showerror("Error", "No trained model found! Please train first.")
#             return
        
#         # Load model
#         try:
#             self.agent.load('trained_gridworld_dqn.pth')
#         except Exception as e:
#             messagebox.showerror("Error", f"Failed to load model: {e}")
#             return
        
#         self.evaluating = True
#         self.stop_requested = False
        
#         # Update UI
#         self.train_btn.config(state=tk.DISABLED)
#         self.eval_btn.config(state=tk.DISABLED)
#         self.stop_btn.config(state=tk.NORMAL)
#         self.info_label.config(text="Evaluation in progress...")
        
#         # Reset environment
#         self.env.reset()
#         self.draw_grid()
#         self.update_stats()
        
#         # Start evaluation
#         self.evaluate_step()
    
#     def evaluate_step(self):
#         """Single evaluation step (called repeatedly)."""
#         if self.stop_requested or self.env.done:
#             self.evaluating = False
#             self.train_btn.config(state=tk.NORMAL)
#             self.eval_btn.config(state=tk.NORMAL)
#             self.stop_btn.config(state=tk.DISABLED)
            
#             if self.env.done:
#                 reason = "Goal reached!" if self.env.score > 0 else "Failed"
#                 self.info_label.config(
#                     text=f"Evaluation complete! Score: {self.env.score} | {reason}"
#                 )
#             else:
#                 self.info_label.config(text="Evaluation stopped")
#             return
        
#         # Get action from agent
#         state = self.env._get_obs()
#         state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
#         with torch.no_grad():
#             q_values = self.agent.q_net(state_tensor)
#         action = torch.argmax(q_values).item()
        
#         # Execute action
#         result = self.env.step(action)
        
#         # Update display
#         self.draw_grid()
#         self.update_stats()
        
#         # Continue evaluation after delay
#         if not self.env.done and not self.stop_requested:
#             self.root.after(self.eval_speed, self.evaluate_step)
#         else:
#             self.evaluating = False
#             self.train_btn.config(state=tk.NORMAL)
#             self.eval_btn.config(state=tk.NORMAL)
#             self.stop_btn.config(state=tk.DISABLED)
            
#             if self.env.done:
#                 self.info_label.config(
#                     text=f"Evaluation complete! Score: {self.env.score} goals"
#                 )
    
#     def stop_action(self):
#         """Stop current training or evaluation."""
#         self.stop_requested = True
#         self.stop_btn.config(state=tk.DISABLED)
#         self.info_label.config(text="Stopping...")


"""
Tkinter GUI for GridWorld DQN
Now properly modular and uses predefined functions
All parameters from config.py
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import os
import numpy as np

from gridworld_env import GridWorldEnv
from dqn_agent import DQNAgent
from train_dqn import train_dqn
from config import (
    ENV_CONFIG, AGENT_CONFIG, NETWORK_CONFIG, TRAINING_CONFIG,
    GUI_CONFIG, COLORS
)

# Extract GUI parameters from config
GRID_SIZE = ENV_CONFIG['grid_size']
CELL_SIZE = GUI_CONFIG['cell_size']
MARGIN = GUI_CONFIG['margin']


class GridWorldGUI:
    def __init__(self, root):
        self.eval_speed = GUI_CONFIG['default_eval_speed']
        self.manual_goal_mode = False
        self.root = root
        self.root.title("GridWorld DQN - Snake-Style Training")
        self.root.configure(bg=COLORS['bg'])
        
        # Initialize environment and agent using config
        self.env = GridWorldEnv(**ENV_CONFIG)
        self.agent = DQNAgent(**AGENT_CONFIG, hidden_dim=NETWORK_CONFIG['hidden_dim'])
        
        # Training state
        self.training = False
        self.evaluating = False
        self.stop_requested = False
        self.training_thread = None
        self.training_data = None
        
        # Create UI
        self.create_widgets()
        self.draw_grid()
        
    def create_widgets(self):
        """Create all UI widgets."""
        # Main container
        main_frame = tk.Frame(self.root, bg=COLORS['bg'])
        main_frame.pack(padx=20, pady=20)
        
        # Top info panel
        info_frame = tk.Frame(main_frame, bg=COLORS['bg'])
        info_frame.pack(pady=(0, 10))
        
        self.info_label = tk.Label(info_frame, text="Ready to train or evaluate", 
                                   font=('Arial', 12), bg=COLORS['bg'])
        self.info_label.pack()
        
        self.stats_label = tk.Label(info_frame, 
                                    text="Score: 0 | Steps: 0 | Episode: 0", 
                                    font=('Arial', 10), bg=COLORS['bg'])
        self.stats_label.pack()
        
        # Canvas for grid
        canvas_frame = tk.Frame(main_frame, bg=COLORS['bg'])
        canvas_frame.pack()
        
        canvas_size = GRID_SIZE * (CELL_SIZE + MARGIN) + MARGIN
        self.canvas = tk.Canvas(canvas_frame, width=canvas_size, height=canvas_size,
                               bg=COLORS['grid_bg'], highlightthickness=1,
                               highlightbackground='#999999')
        self.canvas.pack()

        # Bind mouse click for manual goal placement
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        
        # Button panel
        button_frame = tk.Frame(main_frame, bg=COLORS['bg'])
        button_frame.pack(pady=(10, 0))
        
        episodes_text = f"{TRAINING_CONFIG['episodes']} episodes"
        self.train_btn = tk.Button(button_frame, text=f"🎓 Train ({episodes_text})",
                                   command=self.start_training, font=('Arial', 11),
                                   bg=COLORS['button'], fg='white', padx=15, pady=8,
                                   relief=tk.RAISED, cursor='hand2')
        self.train_btn.grid(row=0, column=0, padx=5)
        
        self.eval_btn = tk.Button(button_frame, text="▶️ Evaluate",
                                  command=self.start_evaluation, font=('Arial', 11),
                                  bg='#2196F3', fg='white', padx=15, pady=8,
                                  relief=tk.RAISED, cursor='hand2')
        self.eval_btn.grid(row=0, column=1, padx=5)
        
        self.stop_btn = tk.Button(button_frame, text="⏹️ Stop",
                                  command=self.stop_action, font=('Arial', 11),
                                  bg=COLORS['danger'], fg='white', padx=15, pady=8,
                                  relief=tk.RAISED, cursor='hand2', state=tk.DISABLED)
        self.stop_btn.grid(row=0, column=2, padx=5)
        
        self.reset_btn = tk.Button(button_frame, text="🔄 Reset",
                                   command=self.reset_env, font=('Arial', 11),
                                   bg='#FF9800', fg='white', padx=15, pady=8,
                                   relief=tk.RAISED, cursor='hand2')
        self.reset_btn.grid(row=0, column=3, padx=5)

        # Manual goal placement toggle
        self.manual_goal_btn = tk.Button(button_frame, text="📍 Manual Goal: OFF",
                                         command=self.toggle_manual_goal_mode,
                                         font=('Arial', 11), bg='#9C27B0', fg='white',
                                         padx=15, pady=8, relief=tk.RAISED, cursor='hand2')
        self.manual_goal_btn.grid(row=0, column=4, padx=5)

        # Speed slider
        speed_frame = tk.Frame(main_frame, bg=COLORS['bg'])
        speed_frame.pack(pady=(10, 0))
        
        tk.Label(speed_frame, text="Evaluation Speed:", bg=COLORS['bg'], font=('Arial', 10)).pack(side=tk.LEFT, padx=5)
        self.speed_slider = ttk.Scale(speed_frame, from_=50, to=1000, orient="horizontal",
                                      command=self.update_speed)
        self.speed_slider.set(self.eval_speed)
        self.speed_slider.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # Progress bar
        self.progress_frame = tk.Frame(main_frame, bg=COLORS['bg'])
        self.progress_frame.pack(pady=(10, 0), fill=tk.X)
        
        self.progress_label = tk.Label(self.progress_frame, text="Training Progress:",
                                       font=('Arial', 10), bg=COLORS['bg'])
        self.progress_label.pack()
        
        self.progress_bar = ttk.Progressbar(self.progress_frame, length=400, 
                                           mode='determinate')
        self.progress_bar.pack(pady=5)
        self.progress_bar.pack_forget()  # Hide initially
        
        # Legend
        legend_frame = tk.Frame(main_frame, bg=COLORS['bg'])
        legend_frame.pack(pady=(10, 0))
        
        legend_items = [
            ("🔵 Agent", COLORS['agent']),
            ("🟢 Goal", COLORS['goal']),
            ("🔴 Bomb", COLORS['bomb']),
            ("⬛ Obstacle", COLORS['obstacle'])
        ]
        
        for i, (text, color) in enumerate(legend_items):
            lbl = tk.Label(legend_frame, text=text, font=('Arial', 9),
                          bg=COLORS['bg'])
            lbl.grid(row=0, column=i, padx=10)

    def toggle_manual_goal_mode(self):
        """Enable/disable manual goal placement mode."""
        self.manual_goal_mode = not self.manual_goal_mode
        if self.manual_goal_mode:
            self.manual_goal_btn.config(text="📍 Manual Goal: ON", bg="#673AB7")
        else:
            self.manual_goal_btn.config(text="📍 Manual Goal: OFF", bg="#9C27B0")

    def on_canvas_click(self, event):
        """Handle canvas clicks for manual goal placement."""
        if not self.manual_goal_mode:
            return
        # Determine clicked cell
        col = event.x // (CELL_SIZE + MARGIN)
        row = event.y // (CELL_SIZE + MARGIN)
        if 0 <= row < self.env.n and 0 <= col < self.env.n:
            if self.env.set_goal(row, col):
                self.draw_grid()
                self.info_label.config(text=f"Goal manually placed at ({row}, {col})")

    def update_speed(self, value):
        """Update evaluation speed based on slider."""
        self.eval_speed = int(float(value))
    
    def draw_grid(self):
        """Draw the grid with all entities."""
        self.canvas.delete("all")
        
        # Draw grid cells
        for i in range(self.env.n):
            for j in range(self.env.n):
                x1 = j * (CELL_SIZE + MARGIN) + MARGIN
                y1 = i * (CELL_SIZE + MARGIN) + MARGIN
                x2 = x1 + CELL_SIZE
                y2 = y1 + CELL_SIZE
                
                # Determine cell color
                pos = (i, j)
                if pos == self.env.agent:
                    color = COLORS['agent']
                elif pos == self.env.goal:
                    color = COLORS['goal']
                elif pos == self.env.bomb:
                    color = COLORS['bomb']
                elif pos in self.env.obstacles:
                    color = COLORS['obstacle']
                else:
                    color = COLORS['empty']
                
                # Draw cell
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color,
                                            outline='#cccccc', width=1)
                
                # Add labels
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                if pos == self.env.agent:
                    self.canvas.create_text(cx, cy, text="A", font=('Arial', 16, 'bold'),
                                          fill='white')
                elif pos == self.env.goal:
                    self.canvas.create_text(cx, cy, text="G", font=('Arial', 16, 'bold'),
                                          fill='white')
                elif pos == self.env.bomb:
                    self.canvas.create_text(cx, cy, text="B", font=('Arial', 16, 'bold'),
                                          fill='white')
                elif pos in self.env.obstacles:
                    self.canvas.create_text(cx, cy, text="W", font=('Arial', 16, 'bold'),
                                          fill='white')
        
        self.root.update()
    
    def update_stats(self):
        """Update statistics display."""
        self.stats_label.config(
            text=f"Score: {self.env.score} | Steps: {self.env.t} | "
                 f"Steps w/o goal: {self.env.steps_without_goal}"
        )
    
    def reset_env(self):
        """Reset environment and redraw."""
        self.env.reset()
        self.draw_grid()
        self.update_stats()
        self.info_label.config(text="Environment reset")
    
    def start_training(self):
        """Start training in background thread."""
        if self.training or self.evaluating:
            messagebox.showwarning("Busy", "Already training or evaluating!")
            return
        
        self.training = True
        self.stop_requested = False
        
        # Update UI
        self.train_btn.config(state=tk.DISABLED)
        self.eval_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.info_label.config(text="Training in progress...")
        self.progress_label.pack()
        self.progress_bar.pack(pady=5)
        self.progress_bar['value'] = 0
        
        # Start training thread
        self.training_thread = threading.Thread(target=self.train_worker, daemon=True)
        self.training_thread.start()
        
        # Monitor training
        self.monitor_training()
    
    def train_worker(self):
        """
        Training worker (runs in background thread).
        NOW USES train_dqn() function instead of duplicating logic!
        """
        try:
            # Create new environment and agent for training using config
            train_env = GridWorldEnv(**ENV_CONFIG)
            train_agent = DQNAgent(**AGENT_CONFIG, hidden_dim=NETWORK_CONFIG['hidden_dim'])
            
            # Progress callback for updating UI
            def progress_callback(episode, progress_percent):
                self.progress_bar['value'] = progress_percent
            
            # Stop check function
            def stop_check():
                return self.stop_requested
            
            # Use the train_dqn function with config parameters
            self.training_data = train_dqn(
                env=train_env,
                agent=train_agent,
                episodes=TRAINING_CONFIG['episodes'],
                start_epsilon=TRAINING_CONFIG['start_epsilon'],
                end_epsilon=TRAINING_CONFIG['end_epsilon'],
                epsilon_decay=TRAINING_CONFIG['epsilon_decay'],
                print_freq=TRAINING_CONFIG['print_freq'],
                save_path=TRAINING_CONFIG['save_path'],
                progress_callback=progress_callback,
                stop_check=stop_check
            )
            
            if not self.stop_requested:
                # Load trained model into main agent
                self.agent.load(TRAINING_CONFIG['save_path'])
        
        except Exception as e:
            print(f"Training error: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            self.training = False
    
    def monitor_training(self):
        """Monitor training progress (runs in main thread)."""
        if self.training and self.training_thread and self.training_thread.is_alive():
            # Still training, check again in 500ms
            self.root.after(500, self.monitor_training)
        else:
            # Training finished
            self.training = False
            self.train_btn.config(state=tk.NORMAL)
            self.eval_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            
            if self.stop_requested:
                self.info_label.config(text="Training stopped by user")
            else:
                self.info_label.config(text="Training complete! Model saved.")
            
            self.progress_bar.pack_forget()
            self.progress_label.pack_forget()
    
    def start_evaluation(self):
        """Start evaluation mode."""
        if self.training or self.evaluating:
            messagebox.showwarning("Busy", "Already training or evaluating!")
            return
        
        # Check if model exists
        if not os.path.exists(TRAINING_CONFIG['save_path']):
            messagebox.showerror("Error", "No trained model found! Please train first.")
            return
        
        # Load model
        try:
            self.agent.load(TRAINING_CONFIG['save_path'])
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {e}")
            return
        
        self.evaluating = True
        self.stop_requested = False
        
        # Update UI
        self.train_btn.config(state=tk.DISABLED)
        self.eval_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.info_label.config(text="Evaluation in progress...")
        
        # Reset environment
        self.env.reset()
        self.draw_grid()
        self.update_stats()
        
        # Start evaluation
        self.evaluate_step()
    
    def evaluate_step(self):
        """Single evaluation step (called repeatedly)."""
        if self.stop_requested or self.env.done:
            self.evaluating = False
            self.train_btn.config(state=tk.NORMAL)
            self.eval_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            
            if self.env.done:
                reason = "Goal reached!" if self.env.score > 0 else "Failed"
                self.info_label.config(
                    text=f"Evaluation complete! Score: {self.env.score} | {reason}"
                )
            else:
                self.info_label.config(text="Evaluation stopped")
            return
        
        # Get action from agent using select_action with epsilon=0 (greedy)
        state = self.env._get_obs()
        action = self.agent.select_action(state, epsilon=0.0)
        
        # Execute action
        result = self.env.step(action)
        
        # Update display
        self.draw_grid()
        self.update_stats()
        
        # Continue evaluation after delay
        if not self.env.done and not self.stop_requested:
            self.root.after(self.eval_speed, self.evaluate_step)
        else:
            self.evaluating = False
            self.train_btn.config(state=tk.NORMAL)
            self.eval_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            
            if self.env.done:
                self.info_label.config(
                    text=f"Evaluation complete! Score: {self.env.score} goals"
                )
    
    def stop_action(self):
        """Stop current training or evaluation."""
        self.stop_requested = True
        self.stop_btn.config(state=tk.DISABLED)
        self.info_label.config(text="Stopping...")