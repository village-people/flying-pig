# Village People, 2017

from env.artificial_malmo import BUILDER_18BINARY, BUILDER_SIMPLE
from env.artificial_malmo import ArificialMalmo

from env.village_env_agents import VillagePeopleEnvChallengeAgent

from env.PigChaseChallengeAgent_Replica import PigChaseChallengeAgent_V
from malmopy.agent.gui import ARROW_KEYS_MAPPING

from malmopy.visualization import ConsoleVisualizer
from pig_chase.agent import PigChaseHumanAgent
from models import get_model
from utils import read_config
from agents import get_agent
import torch
from utils import AtomicStatistics
from pig_chase.common import ENV_ACTIONS, ENV_AGENT_NAMES
from env.village_env_agents import Binary18BatchAgentWrapper

if __name__ == '__main__':
    import time

    config = read_config()

    dtype = torch.zeros(0).long()
    if config.general.use_cuda:
        dtype = dtype.cuda()
    #
    config.envs.simulated.agent0_builder = BUILDER_SIMPLE

    EVAL_EPISODES = 100

    env = ArificialMalmo(config.envs.simulated)
    env.selected_state_builders[BUILDER_18BINARY] = True

    ag0 = VillagePeopleEnvChallengeAgent(PigChaseChallengeAgent_V,
                                         config.alien.name,
                                         env._board_one_hot,
                                         config)

    shared_model = get_model(config.model.name)(config.model)

    shared_model.eval()
    if isinstance(config.model.load, str):
        checkpoint = torch.load(config.model.load)
        iteration = checkpoint['iteration']
        reward = checkpoint['reward']
        print("LOADING MODEL: {} ---> MAX R: {}".format(config.model.load,
                                                        reward))
        shared_model.load_state_dict(checkpoint['state_dict'])
    if config.general.use_cuda:
        shared_model.cuda()
    shared_objects = {
        "model": shared_model,
        "stats_leRMS": AtomicStatistics()
    }

    agent_actor = get_agent(config.agent.type)(config.agent.name,
                                               ENV_ACTIONS, config,
                                               shared_objects)
    agent_role = 1
    ag1 = Binary18BatchAgentWrapper(agent_actor, config.agent.name, config,
                                    is_training=False)

    # ag1 = VillagePeopleEnvChallengeAgent(PigChaseChallengeAgent_V, "Agent_2",
    #                                      env._board_one_hot, config)
    # ag1 = MalmoAgentWrapper(PigChaseChallengeAgent, "Agent_1", config)
    # ag1 = VillagePeopleEnvRandomAgent("Agent_2", config)

    agents = [ag0, ag1]
    env_agents = [env.agent0, env.agent1]
    start = time.time()


    def restartGame():
        obs = env.reset()
        reward = torch.zeros(config.general.batch_size).type_as(dtype)
        done = torch.zeros(config.general.batch_size).type_as(dtype)

        for agent in agents:
            agent.reset()

        return obs, reward, done

    obs, reward, done = restartGame()

    if config.envs.visualize:
        visualizer = ConsoleVisualizer(prefix='Agent %d' % 0)
        ag2 = PigChaseHumanAgent("Agent_2", env,
                                 list(ARROW_KEYS_MAPPING.keys()),
                                 10, 25, visualizer, quit)
        ag2.show()

    crt_agent = 0
    it = [0, 0]
    episode = 0

    # Action batch
    rew_1 = 0
    done_1 = 0
    all_rewards = 0

    while episode < EVAL_EPISODES:
        # check if env needs reset
        if env.done.all():
            print("Time: {}".format(time.time() - start))

            print('Episode %d (%.2f)%%' % (
            episode, (episode / EVAL_EPISODES) * 100.))
            start = time.time()

            obs, _, done = restartGame()

            episode += 1
            it = [0, 0]
            crt_agent = 0
            print(all_rewards / (float(episode) * config.general.batch_size))

        it[crt_agent] += 1

        if crt_agent == 1:
            agent_act = agents[crt_agent].act(obs, reward, done,
                                              (1 - env_agents[
                                                  crt_agent].got_done),
                                              bn=True)
        else:
            agent_act = agents[crt_agent].act(obs, reward, done,
                                              (1 - env_agents[
                                                  crt_agent].got_done))

        obs, reward, done = env.do(agent_act)

        if crt_agent == 1:
            all_rewards += reward.sum()

        crt_agent = (crt_agent + 1) % 2
