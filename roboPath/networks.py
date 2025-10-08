import torch
import torch.nn as nn

class QNetwork(nn.Module):
    """
    Deep Q-Network for GridWorld.
    Input: 20 binary features
    Output: 4 Q-values (one per action)
    """
    def __init__(self, input_dim=20, output_dim=4, hidden_dim=256):
        super(QNetwork, self).__init__()
        
        # Save architecture in case you want to print/debug later
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        # Define the network layers
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        # Allow passing in single state (1D) or batch (2D)
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        return self.network(x)

