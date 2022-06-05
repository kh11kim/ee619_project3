"""Evaluates an agent on a DMControl environment.

DO NOT MODIFY THIS FILE!!!
"""
from argparse import ArgumentParser, Namespace
from math import fsum
from pickle import dump
from typing import List, Optional

from dm_control import suite
from dm_control.rl.control import Environment

from ee619.agent import Agent


def parse_args() -> Namespace:
    """Parses arguments for evaluate()"""
    parser = ArgumentParser(
        description='Evaluates an agent on a Walker-Run environment.')
    parser.add_argument('-l', dest='label', default=None,
                        help='if unspecified, the mean episodic return will be'
                             ' printed to stdout. otherwise, it will be dumped'
                             ' to a pickle file of the given path.')
    parser.add_argument('-n', type=int, dest='repeat', default=1,
                        help='number of trials.')
    parser.add_argument('-s', type=int, dest='seed', default=0,
                        help='passed to the environment for determinism.')
    parser.add_argument('--domain', default='walker',
                        help='DMControl domain to evaluate the agent')
    parser.add_argument('--task', default='run',
                        help='DMControl task of interest')
    return parser.parse_args()


def evaluate(agent: Agent,
             domain: str,
             label: Optional[str],
             repeat: int,
             seed: int,
             task: str):
    """Computes the mean episodic return of the agent.

    Args:
        agent_class: The agent class to evaluate.
        domain: DMControl domain to evaluate the agent.
        label: If None, the mean episodic return will be printed to stdout.
            Otherwise, it will be dumped to a pickle file of the given name
            under the "data" directory.
        repeat: Number of trials.
        seed: Passed to the environment for determinism.
        task: DMControl task to evaluate the agent.
    """
    rewards: List[float] = []
    for seed_ in range(seed, seed + repeat):
        env: Environment = suite.load(domain_name=domain,
                                      task_name=task,
                                      task_kwargs={'random': seed_})
        agent.load()
        env.physics.render()#
        time_step = env.reset()
        while not time_step.last():
            action = agent.act(time_step)
            time_step = env.step(action)
            env.physics.render()#
            rewards.append(time_step.reward)
    mean_episode_return = fsum(rewards) / repeat
    if label is None:
        print(mean_episode_return)
    else:
        if not label.endswith('.pkl'):
            label += '.pkl'
        with open(label, 'wb') as file:
            dump(mean_episode_return, file)
    return mean_episode_return


if __name__ == '__main__':
    evaluate(agent=Agent(), **vars(parse_args()))
