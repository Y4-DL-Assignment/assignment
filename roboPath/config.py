"""
Configuration file for GridWorld DQN
All parameters can be modified here before running
"""

# Environment Parameters
ENV_CONFIG = {
    'grid_size': 10,
    'max_steps': 1000,
    'use_obstacles': True,
    'fixed_start': True,
    'obstacle_move_freq': 3 #TRY WITH 5
}

# Agent Parameters
AGENT_CONFIG = {
    'state_dim': 20,
    'num_actions': 4,
    'lr': 0.001,
    'gamma': 0.95,
    'buffer_capacity': 100000,
    'batch_size': 128, 
    'tau': 0.005
}

# Network Parameters
NETWORK_CONFIG = {
    'input_dim': 20,
    'output_dim': 4,
    'hidden_dim': 256 #TRY WITH 512
}

# Training Parameters
TRAINING_CONFIG = {
    'episodes': 9500,
    'start_epsilon': 1.0,
    'end_epsilon': 0.01,
    'epsilon_decay': 0.9995, #0.9997
    'print_freq': 100,
    'save_path': 'trained_gridworld_dqn.pth'
}

# GUI Parameters
GUI_CONFIG = {
    'cell_size': 50,
    'margin': 2,
    'default_eval_speed': 200  # milliseconds
}

# Colors
COLORS = {
    'bg': '#f0f0f0',
    'grid_bg': '#ffffff',
    'agent': '#0066ff',
    'goal': '#00ff00',
    'bomb': '#ff0000',
    'obstacle': '#333333',
    'wall': '#cccccc',
    'empty': '#ffffff',
    'button': '#4CAF50',
    'button_hover': '#45a049',
    'danger': '#ff6b6b'
}