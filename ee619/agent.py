"""Agent for DMControl Walker-Run task."""
from os.path import abspath, dirname, realpath, join
from typing import Dict, Tuple, Optional

from dm_env import TimeStep
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import MultivariateNormal
from torch.utils.tensorboard.writer import SummaryWriter
from datetime import datetime
import os


ROOT = dirname(abspath(realpath(__file__)))  # path to the ee619 directory
STATE_DIM = 24
ACTION_DIM = 6
NUM_HIDDEN_LAYER = 128
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
            nn.ReLU(),
            nn.Linear(hidden_layer, hidden_layer),
            nn.ReLU(),
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

class PPO:
    def __init__(
        self, 
        observation_shape: int, 
        action_shape: int,
        lr: float,
        num_epochs: int,
        clip: float,
        save_path: str,
        writer: Optional[SummaryWriter]=None,
    ):
        self.policy = Policy(observation_shape, action_shape)
        self.critic = Critic(observation_shape)
        self.policy_optim = optim.Adam(
            self.policy.parameters(), lr=lr/5)
        self.critic_optim = optim.Adam(
            self.critic.parameters(), lr=lr)
        self.num_epochs = num_epochs
        self.clip = clip
        self.writer = writer
        timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
        self.save_folder = f"{save_path}/{timestamp}_PPO"
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)
    
    def train_mode(self):
        self.policy.train()
        self.critic.train()
    
    def eval_mode(self):
        self.policy.eval()
        self.critic.eval()
    
    def get_value_log_probs(
        self,
        batch_obs, 
        batch_actions,
    ):
        mean = self.policy(batch_obs)
        dist = MultivariateNormal(mean, self.policy.cov_mat)
        log_probs = dist.log_prob(batch_actions)
        V = self.critic(batch_obs)
        return V, log_probs
    
    def update(
        self, 
        batch_obs,
        batch_actions,
        batch_returns,
        batch_log_probs,
        episode
    ):
        batch_obs = torch.concat(batch_obs)
        batch_actions = torch.concat(batch_actions)
        batch_returns = torch.concat(batch_returns)
        batch_log_probs = torch.concat(batch_log_probs)
        #normalize returns
        batch_returns_norm = (batch_returns - batch_returns.mean())/(batch_returns.std()+1e-10)
    
        #epoch
        for _ in range(self.num_epochs):
            V, log_probs_curr = self.get_value_log_probs(batch_obs, batch_actions)
            V = V.squeeze()

            ratios = torch.exp(log_probs_curr - batch_log_probs)
            A = batch_returns_norm - V.detach()
            surr1 = ratios * A
            surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A
            actor_loss = (-torch.min(surr1, surr2)).mean()

            #policy update
            self.policy_optim.zero_grad()
            actor_loss.backward(retain_graph=True)
            self.policy_optim.step()
            
            #critic update
            critic_loss = nn.MSELoss()(V, batch_returns_norm)

            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()
        
        if self.writer is not None:
            train_return = batch_returns[0].item()
            self.writer.add_scalar('train/return', train_return, episode)
            self.writer.add_scalar('train/policy_loss', actor_loss.item(), episode)
            self.writer.add_scalar('train/critic_loss', critic_loss.item(), episode)
    
    def save(self, filename):
        save_path = self.save_folder + "/" + filename
        torch.save(self.policy.state_dict(), save_path + "_policy.pth")
        torch.save(self.critic.state_dict(), save_path +"_critic.pth")
        torch.save(self.policy_optim.state_dict(), save_path +"_policy_optim.pth")
        torch.save(self.critic_optim.state_dict(), save_path +"_critic_optim.pth")
    
    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic.pth"))
        self.critic_optim.load_state_dict(torch.load(filename + "_critic_optim.pth"))
        self.policy.load_state_dict(torch.load(filename + "_policy.pth"))
        self.policy_optim.load_state_dict(torch.load(filename + "_policy_optim.pth"))