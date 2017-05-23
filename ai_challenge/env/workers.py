# Village People, 2017

import time
from ai_challenge.pig_chase.environment import PigChaseEnvironment
from ai_challenge.pig_chase.environment import PigChaseSymbolicStateBuilder
from ai_challenge.pig_chase.environment import PigChaseTopDownStateBuilder
from ai_challenge.pig_chase.agent import PigChaseChallengeAgent
from ai_challenge.pig_chase.common import ENV_ACTIONS

from numpy.random import randint

from utils import parse_clients_args, visualize_training

AGENTS = {
    "empathetic": EmpatheticAgent,
    "dqn": DQNAgent,
    "random": RandomAgent
}

class RandomAgent(object):
    def __init__(self, name, action_space):
        self.name = name
        self.action_space = action_space
        self.action_space_len = len(action_space)

    def act(self, obs, reward, agent_done, is_training):
        return randint(self.action_space_len)

    def inject_summaries(self, step):
        pass


def get_agent(name):
    return AGENTS[name]

def train_our_agent(name, endpoints, role, visualizer, cmdl):
    # builder = PigChaseSymbolicStateBuilder()
    builder = PigChaseTopDownStateBuilder()
    env = PigChaseEnvironment(parse_clients_args(endpoints), builder,
                              role=role, randomize_positions=True)
    agent = get_agent(cmdl.type)(name, ENV_ACTIONS)

    obs = env.reset()
    reward = 0
    agent_done = False
    viz_rewards = []
    ep_cnt = 0

    start_time = time.time()

    max_training_steps = cmdl.epoch_steps * cmdl.epochs
    for step in range(1, max_training_steps + 1):

        # check if env needs reset
        if env.done:

            if visualizer:
                visualize_training(visualizer, step, viz_rewards)

            obs = env.reset()
            ep_cnt += 1
            print("Ep: %d | Rw: %d" % (ep_cnt, sum(viz_rewards)))
            viz_rewards = []

        # select an action
        action = agent.act(obs, reward, agent_done, is_training=True)
        # take a step
        obs, reward, agent_done = env.do(action)

        viz_rewards.append(reward)

        agent.inject_summaries(step)

    elapsed_time = time.time() - start_time
    print("Finished in %.2f seconds at %.2ffps." % (
        elapsed_time, max_training_steps / elapsed_time))


def run_alien_agent(name, endpoints, role):
    assert len(endpoints) >= 2, 'Not enough clients (need at least 2)'

    builder = PigChaseSymbolicStateBuilder()
    # builder = PigChaseTopDownStateBuilder()
    env = PigChaseEnvironment(parse_clients_args(endpoints), builder,
                              role=role, randomize_positions=True)

    agent = PigChaseChallengeAgent(name)
    agent_type = PigChaseEnvironment.AGENT_TYPE_1

    obs = env.reset(agent_type)
    reward = 0
    agent_done = False

    while True:

        # select an action
        action = agent.act(obs, reward, agent_done, is_training=True)

        # reset if needed
        if env.done:
            obs = env.reset(agent_type)

        # take a step
        obs, reward, agent_done = env.do(action)
