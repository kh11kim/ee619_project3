from argparse import ArgumentParser
import numpy as np
import operator
from tqdm import trange
from functools import reduce
from typing import Callable, Iterable, List, NamedTuple, Optional

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import MultivariateNormal
from torch.utils.tensorboard.writer import SummaryWriter
from dm_control import suite
from dm_control.rl.control import Environment

from agent import flatten_and_concat, Policy, to_tensor, Critic, PPO


class Trajectory(NamedTuple):
    """Class to manage trajectory samples."""
    observations: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    returns: torch.Tensor

def product(iterable: Iterable[int]) -> int:
    """Return the product of every element in iterable."""
    return reduce(operator.mul, iterable, 1)


def main(domain: str,
         gamma: float,
         learning_rate: float,
         log: bool,
         num_episodes: int,
         save_path: str,
         seed: int,
         task: str,
         test_every: int,
         save_every: int,
         test_num: int,
         load_path: Optional[str]=None
    ) -> None:
    """Create a policy and train it using the REINFORCE algorithm.

    Args:
        domain: DMControl domain to train the policy
        gamma: discount factor
        learning_rate: learning rate for the Adam optimizer
        log: if set to True, use tensorboard to log intermediate results
        num_episodes: number of episodes to collect
        save_path: the path to save the trained policy
        seed: random seed for determinism
        task: DMControl task to train the policy
        test_every: test period
        test_num: number of episodes to collect during test
    """
    torch.manual_seed(seed)
    writer = SummaryWriter() if log else None
    env: Environment = suite.load(domain, task, task_kwargs={'random': seed})
    observation_spec = env.observation_spec()
    observation_shape = sum(product(value.shape)
                            for value in observation_spec.values())
    action_spec = env.action_spec()
    action_shape = product(action_spec.shape)
    max_action = action_spec.maximum
    min_action = action_spec.minimum
    loc_action = (max_action + min_action) / 2
    scale_action = (max_action - min_action) / 2

    def map_action(input_: np.ndarray) -> np.ndarray:
        return np.tanh(input_) * scale_action + loc_action

    agent = PPO(
        observation_shape, 
        action_shape,
        lr=learning_rate,
        num_epochs=10,
        clip=0.2,
        writer=writer,
        save_path=save_path
    )
    if load_path is not None:
        agent.load(load_path)

    agent.train_mode()

    def test_function(episode: int):
        test(domain=domain,
             episode=episode,
             map_action=map_action,
             policy=agent.policy,
             repeat=test_num,
             seed=seed,
             task=task,
             writer=writer)

    train(env=env,
          gamma=gamma,
          map_action=map_action,
          num_episodes=num_episodes,
          num_episodes_per_update=4,
          agent=agent,
          test_every=test_every,
          save_every=save_every,
          test_function=test_function,
    )
    # if log:
    #     torch.save(policy.state_dict(), save_path+'/model.pth')

@torch.no_grad()
def test(domain: str,
         episode: int,
         map_action: Callable[[np.ndarray], np.ndarray],
         policy: Policy,
         repeat: int,
         seed: int,
         task: str,
         writer: Optional[SummaryWriter]) -> None:
    """Evaluates the policy on a DMControl suite

    Args:
        domain: DMControl domain to evaluate the policy
        episode: the index of current episode (used for logging)
        map_action: function that maps the output of the network to a
            viable action
        policy: the policy to evaluate
        repeat: number of episodes to evaluate
        seed: random seed for determinism
        task: DMControl task to evaluate the policy
        writer: tensorboard logger. does not log when set to None
    """
    returns: List[float] = []
    for seed_ in trange(seed, seed + repeat, desc='Testing', leave=False):
        env = suite.load(domain, task, task_kwargs={'random': seed_})
        time_step = env.reset()
        rewards: List[float] = []
        while not time_step.last():
            observation = flatten_and_concat(time_step.observation)
            action = map_action(policy.act(observation))
            time_step = env.step(action)
            rewards.append(time_step.reward)
        returns.append(sum(rewards))
    if writer is not None:
        writer.add_scalar('test/mean', np.mean(returns), episode)
        writer.add_scalar('test/stddev', np.std(returns), episode)

def train(
    env: Environment,
    gamma: float,
    map_action: Callable[[np.ndarray], np.ndarray],
    num_episodes: int,
    num_episodes_per_update: int,
    agent: PPO,
    test_every: int,
    save_every: int,
    test_function: Callable[[int], None],
) -> None:
    """Train the policy using the PPO algorithm on the given environment.

    Args:
        env: environment to train the policy
        gamma: discount factor
        map_action: function that maps the output of the network to a
            viable action
        num_episodes: number of episodes to collect
        optimizer: optimizer used for maximizing the expected return
        policy: policy to train
        test_every: test period
        test_function: a function that evaluates the current policy.
            it takes the current episode number as its argument.
        writer: tensorboard logger. if set to None, it does not log
    """
    batch_obs = []
    batch_actions = []
    batch_returns = []
    batch_log_probs = []
    for episode in trange(num_episodes):
        traj = rollout(
            env=env, gamma=gamma,
            map_action=map_action, 
            policy=agent.policy
        )
        batch_obs.append(traj.observations)
        batch_actions.append(traj.actions)
        batch_returns.append(traj.returns)
        batch_log_probs.append(traj.log_probs)

        if episode % num_episodes_per_update == 0:
            agent.update(
                batch_obs,
                batch_actions,
                batch_returns,
                batch_log_probs,
                episode
            )
            #init
            batch_obs = []
            batch_actions = []
            batch_returns = []
            batch_log_probs = []

        if episode % test_every == test_every - 1:
            agent.eval_mode()
            test_function(episode)
            agent.train_mode()
        
        if episode % save_every == save_every - 1:
            agent.save(f"ep{episode+1}")

# def evaluate(batch_obs, batch_actions, policy:Policy, critic:Critic):
#     mean = policy(batch_obs)
#     dist = MultivariateNormal(mean, policy.cov_mat)
#     log_probs = dist.log_prob(batch_actions)
#     V = critic(batch_obs)
#     return V, log_probs

def rollout(env: Environment,
            gamma: float,
            map_action: Callable[[np.ndarray], np.ndarray],
            policy: Policy) -> Trajectory:
    """Collect a trajectory of an episode.

    Args:
        env: Environment to rollout an episode
        gamma: discount factor
        map_action: function that maps unbounded action
        map_action: function that maps the output of the network to a
            viable action
        policy: policy used for the rollout

    Returns:
        A namedtuple instance that has three fields: observations, actions,
        and returns.
    """
    time_step = env.reset()
    observations: List[np.ndarray] = []
    actions: List[np.ndarray] = []
    log_probs: List[float] = []
    rewards: List[float] = []
    while not time_step.last():
        observation = flatten_and_concat(time_step.observation)
        observations.append(observation.copy())
        with torch.no_grad():
            action, log_prob = policy.act_with_prob(observation)
        actions.append(action.copy())
        log_probs.append(log_prob)
        time_step = env.step(map_action(action))
        rewards.append(time_step.reward)
    returns = []
    return_ = 0
    for reward in rewards[::-1]:
        return_ = return_ * gamma + reward
        returns.append(return_)
    observations_batch = to_tensor(np.asarray(observations))
    actions_batch = to_tensor(np.asarray(actions))
    log_prob_batch = to_tensor(np.asarray(log_probs))
    returns_batch = to_tensor(np.asarray(returns[::-1]))
    return Trajectory(
        observations_batch, 
        actions_batch, 
        log_prob_batch,
        returns_batch
    )

def build_argument_parser() -> ArgumentParser:
    """Returns an argument parser for main."""
    parser = ArgumentParser()
    parser.add_argument('save_path')
    parser.add_argument('-q', action='store_false', dest='log')
    parser.add_argument('--domain', default='walker')
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--num-episodes', type=int, default=int(1e4))
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--task', default='run')
    parser.add_argument('--test-every', type=int, default=1000)
    parser.add_argument('--save-every', type=int, default=10)
    parser.add_argument('--test-num', type=int, default=10)
    parser.add_argument('--load-path', type=str, default="tmp/run/220616_115101_PPO/ep20")
    return parser

if __name__ == '__main__':
    main(**vars(build_argument_parser().parse_args()))    