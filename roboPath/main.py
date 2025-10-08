# import pygame
# import sys
# import os
# import threading
# import torch
# import numpy as np
# from gridworld_env import GridWorldEnv
# from dqn_agent import DQNAgent
# from train_dqn import train_dqn
# import tkinter as tk
# from tkinter import messagebox
# from visualizer_tkinter import GridWorldGUI


# def main():
#     """Main entry point for the application."""
#     print("\n" + "="*70)
#     print("GridWorld DQN - Snake-Style Training")
#     print("="*70)
#     print("Grid Size: 10×10")
#     print("State Dimension: 20 binary features")
#     print("Actions: 4 (UP, DOWN, LEFT, RIGHT)")
#     print("Algorithm: Deep Q-Learning (DQN) with experience replay")
#     print("="*70 + "\n")
    
#     root = tk.Tk()
#     app = GridWorldGUI(root)
#     root.mainloop()


# if __name__ == "__main__":
#     # If running as standalone script, you can train from command line
#     import sys
    
#     if len(sys.argv) > 1 and sys.argv[1] == '--train-cli':
#         # Command-line training mode
#         print("Starting command-line training...")
#         env = GridWorldEnv(grid_size=10, max_steps=1000, use_obstacles=True, fixed_start=True)
#         agent = DQNAgent(state_dim=20, num_actions=4, lr=0.001, gamma=0.95, batch_size=128, tau=0.005)
        
#         training_data = train_dqn(
#             env=env,
#             agent=agent,
#             episodes=20000,
#             start_epsilon=1.0,
#             end_epsilon=0.01,
#             epsilon_decay=0.9995,
#             print_freq=500,
#             save_path='trained_gridworld_dqn.pth'
#         )
        
#         print("\nTraining data saved. You can now run the GUI to visualize.")
#     else:
#         # GUI mode
#         main()




"""
Main entry point for GridWorld DQN application
Supports both GUI and command-line training modes
All parameters configured in config.py
"""

import sys
import tkinter as tk
from gridworld_env import GridWorldEnv
from dqn_agent import DQNAgent
from train_dqn import train_dqn
from visualizer_tkinter import GridWorldGUI
from config import ENV_CONFIG, AGENT_CONFIG, NETWORK_CONFIG, TRAINING_CONFIG


def main():
    """Main entry point for GUI application."""
    print("\n" + "="*70)
    print("GridWorld DQN - Snake-Style Training")
    print("="*70)
    print(f"Grid Size: {ENV_CONFIG['grid_size']}×{ENV_CONFIG['grid_size']}")
    print(f"State Dimension: {AGENT_CONFIG['state_dim']} binary features")
    print(f"Actions: {AGENT_CONFIG['num_actions']} (UP, DOWN, LEFT, RIGHT)")
    print("Algorithm: Deep Q-Learning (DQN) with experience replay")
    print("="*70 + "\n")
    
    root = tk.Tk()
    app = GridWorldGUI(root)
    root.mainloop()


def train_cli():
    """Command-line training mode using config parameters."""
    print("\n" + "="*70)
    print("GridWorld DQN - Command-Line Training")
    print("="*70)
    print("\nCurrent Configuration:")
    print(f"  Environment: {ENV_CONFIG}")
    print(f"  Agent: {AGENT_CONFIG}")
    print(f"  Network: {NETWORK_CONFIG}")
    print(f"  Training: {TRAINING_CONFIG}")
    print("="*70 + "\n")
    
    print("Starting command-line training...")
    
    # Create environment using config
    env = GridWorldEnv(**ENV_CONFIG)
    
    # Create agent using config
    agent = DQNAgent(**AGENT_CONFIG, hidden_dim=NETWORK_CONFIG['hidden_dim'])
    
    # Train agent using config
    training_data = train_dqn(
        env=env,
        agent=agent,
        episodes=TRAINING_CONFIG['episodes'],
        start_epsilon=TRAINING_CONFIG['start_epsilon'],
        end_epsilon=TRAINING_CONFIG['end_epsilon'],
        epsilon_decay=TRAINING_CONFIG['epsilon_decay'],
        print_freq=TRAINING_CONFIG['print_freq'],
        save_path=TRAINING_CONFIG['save_path']
    )
    
    print("\nTraining complete! Model saved.")
    print("You can now run the GUI to visualize the trained agent.")
    
    return training_data


if __name__ == "__main__":
    # Check for command-line arguments
    if len(sys.argv) > 1 and sys.argv[1] == '--train-cli':
        # Command-line training mode
        train_cli()
    else:
        # GUI mode (default)
        main()