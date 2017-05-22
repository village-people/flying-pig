# Village People, 2017

import torch

import time
from termcolor import colored as clr

from models import get_model
from utils import read_config
from utils import AtomicStatistics
from utils import AverageMeter

from agents import get_agent

from env.artificial_malmo import ArificialMalmo
from env.village_env_agents import  Binary18BatchAgentWrapper, MalmoAgentWrapper

from pig_chase.common import ENV_ACTIONS
from pig_chase.agent import PigChaseChallengeAgent

# Useful functions
def print_info(message):
    print(clr("[MAIN] ", "yellow") + message)


def mock_function():
    print("Target function not implemented.")

def train_agent_simulated(shared_objects, cfg):

    batch_size = cfg.general.batch_size

    #INIT Simulated env
    env = ArificialMalmo(cfg.envs.simulated)

    #Init agent2
    agent_class = shared_objects["agent"]
    agent_runner = Binary18BatchAgentWrapper(agent_class, cfg.agent.name,
                                             cfg.general)
    #Init alien
    alien = MalmoAgentWrapper(PigChaseChallengeAgent, "Agent_1", cfg.general)

    agents = [alien, agent_runner]
    agent_idx = 1

    env_agents_data = [env.agent0, env.agent1]

    print(clr("[ %s ] type=%s, role=%d. Agent started." %
          (cfg.agent.name, cfg.agent.type, cfg.agent.role), 'cyan'))

    dtype = torch.LongTensor(0)
    if cfg.general.use_cuda:
        dtype = dtype.cuda()

    def restartGame():
        obs = env.reset()
        reward = torch.zeros(batch_size).type_as(dtype)
        done = torch.zeros(batch_size).type_as(dtype)

        for agent in agents:
            agent.reset()
        return obs, reward, done

    obs, reward, done = restartGame()
    ep_cnt = 0
    crt_agent = 0
    viz_rewards = torch.LongTensor(batch_size).type_as(dtype)
    viz_rewards.fill_(0)
    start_time = time.time()
    episode_time = AverageMeter()
    report_freq = cfg.general.report_freq

    print("No of epochs: %d. Max no of steps/epoch: %d" %
          (cfg.training.episodes_no, cfg.training.max_step_no))

    training_steps = cfg.training.episodes_no * cfg.training.max_step_no * 2

    start_episode_time = time.time()
    for step in range(1, training_steps + 1):
        # check if env needs reset
        if env.done.all():
            episode_time.update(time.time() - start_episode_time)
            start_episode_time = time.time()

            obs, reward, done = restartGame()
            ep_cnt += 1
            crt_agent = 0

            if ep_cnt % report_freq == 0:
                batch_mean_reward = torch.sum(viz_rewards) / report_freq
                game_mean_reward = batch_mean_reward / batch_size

                print("[DQN] Ep: %d | batch_avg_R: %.4f | game_avg_R: %.4f "
                      "| (Ep_avg_time: %.4f)" % (ep_cnt, batch_mean_reward,
                                                 game_mean_reward,
                                                 episode_time.avg))
                viz_rewards.fill_(0)

        # select an action
        agent_act = agents[crt_agent].act(obs, reward, done,
                                    (1 - env_agents_data[crt_agent].got_done))

        # take a step
        obs, reward, done = env.do(agent_act)
        crt_agent = (crt_agent + 1) % 2

        if crt_agent == agent_idx:
            viz_rewards.add_(reward)

    elapsed_time = time.time() - start_time
    print("Finished in %.2f seconds at %.2ffps." % (
        elapsed_time, training_steps / elapsed_time))

def main():

    print_info("Starting...")

    # -- Read configuration
    config = read_config()

    # -- Configure Torch
    if config.general.seed != 0:
        torch.manual_seed(config.general.seed)
        if config.general.use_cuda:
            torch.cuda.manual_seed_all(config.general.seed)

    # Configure model
    shared_model = get_model(config.model.name)
    if config.general.use_cuda:
        shared_model.cuda()
    shared_model.share_memory()

    # Get agent
    agent = get_agent(config.agent.type)(config.agent.name, ENV_ACTIONS,
                                         shared_model, config)

    # Shared statistics
    shared_stats = AtomicStatistics()

    # Shared objects
    shared_objects = {
        "agent": agent,
        "stats_leRMS": shared_stats
    }

    start_time = time.time()

    #Train Agent
    train_agent_simulated(shared_objects, config)

    total_time = time.time() - start_time
    print_info("Everything done in {:.2f}!".format(total_time))


if __name__ == "__main__":
    main()
