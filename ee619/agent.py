"""Agent for DMControl Walker-Run task."""
from os.path import abspath, dirname, realpath, join
from typing import Dict, Tuple

from dm_env import TimeStep
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import MultivariateNormal


ROOT = dirname(abspath(realpath(__file__)))  # path to the ee619 directory
STATE_DIM = 24
ACTION_DIM = 6
NUM_HIDDEN_LAYER = 64
device = "cuda:0" if torch.cuda.is_available() else "cpu"

def flatten_and_concat(dmc_observation: Dict[str, np.ndarray]) -> np.ndarray:
    """Convert a DMControl observation (OrderedDict of NumPy arrays)
    into a single NumPy array.

    """
    return np.concatenate([[obs] if np.isscalar(obs) else obs.ravel()
                           for obs in dmc_observation.values()])

def to_tensor(array: np.ndarray) -> torch.Tensor:
    """Convert NumPy array to a PyTorch Tensor of data type torch.float32"""
    return torch.as_tensor(array, dtype=torch.float32).to(device)

class Agent:
    """Agent for a Walker2DBullet environment."""
    def __init__(self) -> None:
        # Create class variables here if you need to.
        # Example:
        #     self.policy = torch.nn.Sequential(...)
        self.policy = Policy(STATE_DIM, ACTION_DIM)
        self.path = join(ROOT, 'model.pth')

    def act(self, time_step: TimeStep) -> np.ndarray:
        """Returns the action to take for the current time-step.

        Args:
            time_step: a namedtuple with four fields step_type, reward,
                discount, and observation.
        """
        observation = flatten_and_concat(time_step.observation)
        action, _ = self.policy.act(observation)
        return action

    def load(self):
        """Loads network parameters if there are any."""
        self.policy.load_state_dict(torch.load(self.path))

class Net(nn.Module):
    """3 layer neural network
    """
    def __init__(self, in_features: int, out_features: int, hidden_layer):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_layer),
            nn.Tanh(),
            nn.Linear(hidden_layer, hidden_layer),
            nn.Tanh(),
            nn.Linear(hidden_layer, out_features)
        )
        self.to(device)
    
    def forward(self, x: torch.Tensor):
        x = x.to(device)
        return self.net(x)

class Policy(Net):
    def __init__(self, in_features: int, out_features: int):
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            hidden_layer=NUM_HIDDEN_LAYER
        )
        self.cov_var = torch.full(size=(out_features,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var).to(device)

    def act(self, observation: np.ndarray) -> Tuple[np.ndarray, float]:
        mean = self(to_tensor(observation).unsqueeze(0))
        dist = MultivariateNormal(mean, self.cov_mat)
        action = dist.sample()
        return action.squeeze(0).cpu().numpy()
    
    def act_with_prob(self, observation):
        mean = self(to_tensor(observation).unsqueeze(0))
        dist = MultivariateNormal(mean, self.cov_mat)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.squeeze(0).cpu().numpy(), log_prob.item()

class Critic(Net):
    def __init__(self, in_features: int):
        super().__init__(
            in_features=in_features,
            out_features=1,
            hidden_layer=NUM_HIDDEN_LAYER
        )