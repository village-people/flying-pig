# Village People, 2017

import time
from termcolor import colored as clr
from pig_chase.environment import PigChaseEnvironment
from pig_chase.environment import PigChaseSymbolicStateBuilder
from pig_chase.environment import PigChaseTopDownStateBuilder
from pig_chase.agent import PigChaseChallengeAgent
from pig_chase.common import ENV_ACTIONS

from agents import get_agent
from utils.utils import parse_clients_args

def train_agent(shared_objects, cfg):

    env = PigChaseEnvironment(
            parse_clients_args(cfg.envs.minecraft.ports),
            PigChaseTopDownStateBuilder(),
            role=cfg.agent.role,
            randomize_positions=cfg.envs.minecraft.randomize_positions)
    agent = get_agent(cfg.agent.type)(cfg.agent.name, ENV_ACTIONS)

    print(clr("[ %s ] type=%s, role=%d. Agent started." %
          (cfg.agent.name, cfg.agent.type, cfg.agent.role), 'cyan'))

    obs = env.reset()
    reward = 0
    is_terminal = False
    viz_rewards = []
    ep_cnt = 0

    start_time = time.time()

    print("No of epochs: %d. Max no of steps/epoch: %d" %
          (cfg.training.episodes_no, cfg.training.max_step_no))

    training_steps = cfg.training.episodes_no * cfg.training.max_step_no
    for step in range(1, training_steps + 1):
        # check if env needs reset
        if env.done:
            obs = env.reset()
            ep_cnt += 1
            if ep_cnt % cfg.general.report_freq == 0:
                print("[DQN] Ep: %d | Rw: %d" % (
                    ep_cnt, sum(viz_rewards) / cfg.general.report_freq))
                viz_rewards.clear()

        # select an action
        action = agent.act(obs, reward, is_terminal, is_training=True)

        # take a step
        obs, reward, is_terminal = env.do(action)
        viz_rewards.append(reward)

    elapsed_time = time.time() - start_time
    print("Finished in %.2f seconds at %.2ffps." % (
        elapsed_time, training_steps / elapsed_time))


def run_alien(shared_objects, cfg):
    assert len(cfg.envs.minecraft.ports) >= 2, \
        clr('Not enough clients (need at least 2)', 'white', 'on_red')

    # builder = PigChaseTopDownStateBuilder()
    env = PigChaseEnvironment(
            parse_clients_args(cfg.envs.minecraft.ports),
            PigChaseSymbolicStateBuilder(),
            role=cfg.alien.role,
            randomize_positions=cfg.envs.minecraft.randomize_positions)

    agent = PigChaseChallengeAgent(cfg.alien.name)
    agent_type = PigChaseEnvironment.AGENT_TYPE_1

    print(clr("[ %s ] type=%s, role=%d. Agent started." %
          (cfg.alien.name, agent_type, cfg.alien.role), 'blue'))

    obs = env.reset(agent_type)
    reward = 0
    ep_cnt = 0
    viz_rewards = []
    is_terminal = False

    while True:

        # select an action
        action = agent.act(obs, reward, is_terminal, is_training=True)

        # reset if needed
        if env.done:
            obs = env.reset(agent_type)
            ep_cnt += 1
            if ep_cnt % cfg.general.report_freq == 0:
                print("[A*?] Ep: %d | Rw: %d" % (
                    ep_cnt, sum(viz_rewards) / cfg.general.report_freq))
                viz_rewards.clear()

        # take a step
        obs, reward, is_terminal = env.do(action)
        viz_rewards.append(reward)
