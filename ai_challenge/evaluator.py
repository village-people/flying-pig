# Village People, 2017

import torch
import torch.multiprocessing as mp
from models import get_model
from termcolor import colored as clr
from time import sleep

class Evaluator(mp.Process):
    def __init__(self, shared_objects, config):
        super(Evaluator, self).__init__()
        self.shared_objects = shared_objects
        self.config = config

    def run(self):

        config = self.config
        torch.set_num_threads(config.num_threads)

        self.stats = stats = self.shared_objects["stats_leRMS"]
        self.shared_model = shared_model = self.shared_objects["model"]
        self.model = model = get_model(config.model.name)(config.model)
        if config.use_cuda:
            model.cuda()

        start_at, stop_at = config.start_at, config.stop_at
        eval_every = config.frequency
        last_eval_at = -eval_every

        crt_episode = stats.get_episodes()
        while crt_episode < stop_at:
            if crt_episode > start_at and crt_episode > last_eval + eval_every:
                last_eval = crt_episode
                model.load_state_dict(shared_model.state_dict())
                self.start_evaluation(crt_episode)
            else:
                print("not yet")
                sleep(1)
            crt_episode = stats.get_episodes()

        self.info("Evaluation done")

    def start_evaluation(self, crt_episode):
        config = self.config
        eval_episodes = config.episodes
        clients = config.clients

        self.info("Evaluation started at {:d}".format(crt_episode))

        env = PigChaseEnvironment(clients,
                                  self.state_builder,
                                  role=1,
                                  randomize_positions=True)
        alien = mp.Process(run_alien, args=(clients, eval_episodes))
        alien.start()
        sleep(5)
        self.metrics[crt_episode] = metrics = []
        agent_loop(self.model, env, eval_episodes, metrics)
        alien.join()

        from numpy import mean, var
        m, v = mean(metircs), var(metrics)

        if self.best_m is None or m > self.best_m:
            m_str = clr("{:2.4f}".format(m), "white", "on_magenta")
            self.best_m = m
        else:
            m_str = clr("{:2.4f}".format(m), "red")

        self.info("Rewards: Mean={:s}, Var={:2.4f}".format(m_str, v))

    def info(self, message):
        print(clr("[EVAL] ", "red") + message)


def run_alien(clients, eval_episodes):
    builder = PigChaseSymbolicStateBuilder()
    env = PigChaseEnvironment(clients, builder, role=0,
                              randomize_positions=True)
    agent = PigChaseChallengeAgent("Alien")
    agent_loop(agent, env, eval_episodes)


def agent_loop(agent, env, eval_episodes, metrics_acc=None):
    """Adapted from pig_chase/evaluation.py"""

    agent_done = False
    reward = 0
    episode = 0
    obs = env.reset()

    it = 0
    while episode < eval_episodes:
        if env.done:
            obs = env.reset()
            while obs is None:
                obs = env.reset()
            episode += 1
        it += 1
        action = agent.act(obs, reward, agent_done, is_training=False)
        obs, reward, agent_done = env.do(action)

        if metrics_acc is not None:
            metrics_acc.append(reward)
