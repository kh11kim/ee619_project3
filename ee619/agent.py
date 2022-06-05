"""Agent for DMControl Walker-Run task."""
from os.path import abspath, dirname, realpath
from typing import Dict

from dm_env import TimeStep
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import MultivariateNormal

ROOT = dirname(abspath(realpath(__file__)))  # path to the ee619 directory


def flatten_and_concat(dmc_observation: Dict[str, np.ndarray]) -> np.ndarray:
    """Convert a DMControl observation (OrderedDict of NumPy arrays)
    into a single NumPy array.

    """
    return np.concatenate([[obs] if np.isscalar(obs) else obs.ravel()
                           for obs in dmc_observation.values()])

def to_tensor(array: np.ndarray) -> torch.Tensor:
    """Convert NumPy array to a PyTorch Tensor of data type torch.float32"""
    return torch.as_tensor(array, dtype=torch.float32)

class Agent:
    """Agent for a Walker2DBullet environment."""
    def __init__(self) -> None:
        # Create class variables here if you need to.
        # Example:
        #     self.policy = torch.nn.Sequential(...)
        pass

    def act(self, time_step: TimeStep) -> np.ndarray:
        """Returns the action to take for the current time-step.

        Args:
            time_step: a namedtuple with four fields step_type, reward,
                discount, and observation.
        """
        # You can access each member of time_step by time_step.[name], a
        # for example, time_step.reward or time_step.observation.
        # You may also check if the current time-step is the last one
        # of the episode, by calling the method time_step.last().
        # The return value will be True if it is the last time-step,
        # and False otherwise.
        # Note that the observation is given as an OrderedDict of
        # NumPy arrays, so you would need to convert it into a
        # single NumPy array before you feed it into your network.
        # It can be done by using the `flatten_and_concat` function, e.g.,
        #   observation = flatten_and_concat(time_step.observation)
        #   logits = self.policy(torch.as_tensor(observation))
        return np.ones(6)

    def load(self):
        """Loads network parameters if there are any."""
        # Example:
        #   path = join(ROOT, 'model.pth')
        #   self.policy.load_state_dict(torch.load(path))

class Net(nn.Module):
    """3 layer neural network
    """
    def __init__(self, in_features: int, out_features: int, hidden_layer):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_layer),
            nn.ReLU(),
            nn.Linear(hidden_layer, hidden_layer),
            nn.ReLU(),
            nn.Linear(hidden_layer, out_features)
        )
    
    def forward(self, x):
        return self.net(x)


class Policy(Net):
    def __init__(self, in_features: int, out_features: int):
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            hidden_layer=64
        )
        self.cov_var = torch.full(size=(out_features,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)

    def act(self, observation):
        mean = self(to_tensor(observation).unsqueeze(0))
        dist = MultivariateNormal(mean, self.cov_mat)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.squeeze(0).numpy(), log_prob.item()

class Critic(Net):
    def __init__(self, in_features: int):
        super().__init__(
            in_features=in_features,
            out_features=1,
            hidden_layer=64
        )