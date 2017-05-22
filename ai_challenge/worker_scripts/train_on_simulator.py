# Village People, 2017

import torch

from env.artificial_malmo import ArificialMalmo
from env.artificial_malmo import VillagePeopleEnvAgent
from env.village_env_agents import Binary18BatchAgentWrapper
from env.village_env_agents import MalmoAgentWrapper
from pig_chase.common import ENV_ACTIONS
from pig_chase.agent import PigChaseChallengeAgent
from env.PigChaseChallengeAgent_Replica import PigChaseChallengeAgent_V
from utils import AverageMeter
from env.village_env_agents import VillagePeopleEnvChallengeAgent

from agents import get_agent

import time
from termcolor import colored as clr


def print_info(message):
    print(clr("[ARTF_TRAIN] ", "magenta") + message)


# Train on our environment
def train_on_simulator(shared_objects, cfg):
    batch_size = cfg.general.batch_size
    stats = shared_objects["stats_leRMS"]

    # -- Initialize simulated environment
    env = ArificialMalmo(cfg.envs.simulated)
    print_info("Environment initialized (batch_size:={:d}).".format(batch_size))

    # -- Initialize agent and wrap it in a Binary18BatchAgentWrapper :)
    Agent = get_agent(cfg.agent.type)
    agent = Agent(cfg.agent.name, ENV_ACTIONS, cfg, shared_objects)
    agent_runner = Binary18BatchAgentWrapper(agent, cfg.agent.name, cfg)

    print_info("{:s}<{:s}> agent is up and waiting to learn. |role={:d}".format(
        cfg.agent.name, cfg.agent.type, cfg.agent.role
    ))

    # -- Initialize alien
    alien = VillagePeopleEnvChallengeAgent(PigChaseChallengeAgent_V,
                                           cfg.alien.name,
                                           env._board_one_hot,
                                           cfg)
    print_info("Alien is up.")

    # -- Start training
    agents = [alien, agent_runner]
    agent_idx = 1

    env_agents_data = [env.agent0, env.agent1]

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
    viz_steps = torch.LongTensor(batch_size).type_as(dtype)

    # Batch of agents used for evaluation during training.
    eval_agents_count = batch_size
    if cfg.evaluation.during_training.truncate:
        eval_agents_count = int(batch_size * cfg.agent.exploration[0][1])

    viz_rewards = torch.LongTensor(eval_agents_count).type_as(dtype)

    viz_rewards.fill_(0)
    viz_steps.fill_(0)

    start_time = time.time()
    episode_time = AverageMeter()
    report_freq = cfg.general.report_freq

    print_info("No of epochs: {:d}. Max no of steps/epoch: {:d}".format(
        cfg.training.episodes_no, cfg.training.max_step_no
    ))

    training_steps = cfg.training.episodes_no * cfg.training.max_step_no * 2

    start_episode_time = time.time()
    start_report_time = time.time()

    max_freq_r = -100
    max_freq_r_ep = -1

    for step in range(1, training_steps + 1):
        # check if env needs reset
        if env.done.all():
            episode_time.update(time.time() - start_episode_time)
            start_episode_time = time.time()

            obs, reward, done = restartGame()
            ep_cnt += 1
            stats.inc_episodes(batch_size)
            crt_agent = 0

            if ep_cnt % report_freq == 0:
                batch_mean_reward = torch.sum(viz_rewards) / report_freq
                game_mean_reward = batch_mean_reward / eval_agents_count
                last_report_time = time.time() - start_report_time
                start_report_time = time.time()
                r_step = torch.mean(viz_rewards.float() / viz_steps.float())

                if game_mean_reward > max_freq_r:
                    max_freq_r = game_mean_reward
                    max_freq_r_ep = ep_cnt
                    agent.model_utils.save_model(r_step, game_mean_reward,
                                                 ep_cnt, save_only_min=False)

                print_info("Ep: %d | batch_avg_R: %.4f | game_avg_R: %.4f "
                           "| R_step: %.4f | (Max_R: %.4f at ep %d)" %
                           (ep_cnt, batch_mean_reward, game_mean_reward, r_step,
                            max_freq_r, max_freq_r_ep))
                print_info(
                    "Ep: %d | (Ep_avg_time: %.4f) | (Last_report: %.4f)" %
                    (ep_cnt, episode_time.avg, last_report_time))
                viz_rewards.fill_(0)
                viz_steps.fill_(0)

        # select an action
        agent_act = agents[crt_agent].act(
            obs, reward, done, (1 - env_agents_data[crt_agent].got_done))
        stats.inc_frames((1 - env.done.long()).sum())

        # take a step
        obs, reward, done = env.do(agent_act)
        crt_agent = (crt_agent + 1) % 2

        if crt_agent == agent_idx:
            viz_steps.add_(1 - env.done.long())
            viz_rewards.add_(reward[:eval_agents_count])

    elapsed_time = time.time() - start_time
    print("Finished in %.2f seconds at %.2ffps." % (
        elapsed_time, training_steps / elapsed_time))
