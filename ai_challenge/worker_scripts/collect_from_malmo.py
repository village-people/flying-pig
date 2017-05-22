# Village People, 2017

from pig_chase.common import ENV_ACTIONS
from pig_chase.agent import PigChaseChallengeAgent
from pig_chase.common import ENV_AGENT_NAMES
from pig_chase.environment import PigChaseEnvironment, \
    PigChaseSymbolicStateBuilder

from env.VillagePeopleBuilders import PigChaseVillagePeopleBuilder18Binary
import torch

from time import sleep

from agents import get_agent
import torch.multiprocessing as mp

from termcolor import colored as clr


def print_info(message):
    print(clr("[QUEUE_COLLECT] ", "magenta") + message)


def start_minecraft():
    raise NotImplementedError


def restart_minecraft():
    raise NotImplementedError


def collect_from_malmo(id_, shared_objects, cfg):
    clients = cfg.envs.minecraft.ports
    my_id = id_
    reset = shared_objects["reset"]
    session_id = shared_objects["session"]
    queue = shared_objects["queue"]

    if "predict_queue" in shared_objects:
        use_predict_queue = True
        predict_queue = shared_objects["predict_queue"]
        answer_queue = shared_objects["answer_pipe"][my_id]
    else:
        use_predict_queue = False

    # ----------------------- Run Challenge agent ------------------------------
    challenger_stopped = mp.Value("i", 0)
    shared_obj_ = {"stopped": challenger_stopped}
    p = mp.Process(target=run_challenge_agent, args=(id_, clients, shared_obj_))
    p.start()
    sleep(5)

    # ----------------------- Run VillageP Agent -------------------------------
    # --- Start agent

    agent_role = 1
    cfg.general.use_cuda = True

    if not use_predict_queue:
        agent_actor = get_agent(cfg.agent.type)(cfg.agent.name, ENV_ACTIONS,
                                                cfg,
                                                shared_objects)
        # SET Not max predictor
        agent_actor.predict_max = False

    # agent = PigChaseVillagePopleAgent(ENV_AGENT_NAMES[agent_role], ENV_ACTIONS,
    #                                   agent_actor,
    #                                   use_cuda=cfg.general.use_cuda)

    state_builder = PigChaseVillagePeopleBuilder18Binary(agent_role)
    print("A3C: ", clients)
    env = PigChaseEnvironment(clients, state_builder,
                              role=1, randomize_positions=True)

    agent_done = False
    reward = 0
    episode = 0
    step = 0
    obs = env.reset()
    received_none = 0

    while obs is None:
        # this can happen if the episode ended with the first
        # action of the other agent
        # print('Warning: received obs == None.')
        received_none += 1
        if received_none == 10:
            print("[[{}]] Panic !!! > Received {} None in a row"
                  .format(id_, received_none))

        if received_none == 100:
            print("[[{}]] Panic! Challenger stopped."
                  " Received {} None in a row"
                  .format(id_, received_none))
            return -1

    print("[[{}]] Born an playing!".format(id_))

    ep_states = []
    crt_session_id = session_id.value
    while True:
        step += 1
        # check if env needs reset
        # print("AGENT123123")

        if env.done or agent_done:
            # print("[[{}]] Done ep {}.".format(id_, episode))

            if challenger_stopped.value < 0:
                print("[[{}]] Child process ended!!".format(id_))
                pass

            if reset.value == 1:
                # --- Master is training network

                # ---- Restart ----------------------
                # TODO restart MInecraft process

                while reset.value == 1:
                    sleep(0.1)
                ep_states.clear()

            if session_id.value != crt_session_id:
                ep_states.clear()
                crt_session_id = session_id.value

            if len(ep_states) > 0:
                # --- Will be restarted
                state_ = torch.LongTensor(obs).unsqueeze(0)
                done_ = torch.LongTensor([int(agent_done)])
                reward_ = torch.FloatTensor([reward])
                if use_predict_queue:
                    predict_queue.put(
                        (my_id, state_.cpu().numpy(), done_.cpu().numpy(), 23))
                    (act, _) = answer_queue.recv()
                    act = torch.LongTensor([act])
                else:
                    act = agent_actor.act(state_.cuda(), reward_.cuda(),
                                          done_.cuda(), False)
                    # act = agent_actor.act(state_, reward_, done_, False)
                ep_states.append((state_.cpu().numpy(),
                                  reward_.cpu().numpy(),
                                  done_.cpu().numpy(),
                                  act.cpu().numpy()))

                queue.put(ep_states)
                ep_states = []

            obs = env.reset()
            received_none = 0
            while obs is None:
                # this can happen if the episode ended with the first
                # action of the other agent
                # print('Warning: received obs == None.')
                received_none += 1
                if received_none == 10:
                    print("[[{}]] Panic !!! > Received {} None in a row"
                          .format(id_, received_none))

                if received_none == 10000:
                    print("[[{}]] Panic! Challenger stopped."
                          " Received {} None in a row"
                          .format(id_, received_none))
                    sleep(5)
                obs = env.reset()

            episode += 1

        state_ = torch.LongTensor(obs).unsqueeze(0)
        reward_ = torch.FloatTensor([reward])
        done_ = torch.LongTensor([int(agent_done)])

        if not agent_done:
            if use_predict_queue:
                predict_queue.put((my_id, state_.cpu().numpy(),
                                   done_.cpu().numpy(), 23))
                (act, _) = answer_queue.recv()
                act = torch.LongTensor([act])
            else:
                act = agent_actor.act(state_.cuda(), reward_.cuda(),
                                      done_.cuda(), False)
        else:
            reward_[0] = 0
            done_[0] = 0
            if use_predict_queue:
                predict_queue.put((my_id, state_.cpu().numpy(),
                                   done_.cpu().numpy(), 23))
                (act, _) = answer_queue.recv()
                act = torch.LongTensor([act])
            else:
                act = agent_actor.act(state_.cuda(), reward_.cuda(),
                                      done_.cuda(), False)
                # act = agent_actor.act(state_, reward_, done_, False)

        ep_states.append((state_.cpu().numpy(),
                          reward_.cpu().numpy(),
                          done_.cpu().numpy(),
                          act.cpu().numpy()))

        obs, reward, agent_done = env.do(act[0])


def run_challenge_agent(id_, clients, shared_objects):
    # print("AGENT2")
    builder = PigChaseSymbolicStateBuilder()
    print("Challanger: ", clients)
    env = PigChaseEnvironment(clients, builder, role=0,
                              randomize_positions=True)
    agent = PigChaseChallengeAgent(ENV_AGENT_NAMES[0])
    # agent = RandomAgent(ENV_AGENT_NAMES[0])

    agent_done = False
    reward = 0
    episode = 0
    obs = env.reset()

    stop_ = shared_objects["stopped"]
    # print("AGENT22")
    while True:
        # check if env needs reset
        if env.done:

            obs = env.reset()
            received_none = 0
            while obs is None:
                # this can happen if the episode ended with the first
                # action of the other agent
                # print('Warning: received obs == None.')
                received_none += 1
                if received_none == 10:
                    print("[[{}]] Panic !!! > Received {} None in a row"
                          .format(id_, received_none))

                if received_none == 30:
                    print("[[{}]] Panic! Challenger stopped."
                          " Received {} None in a row"
                          .format(id_, received_none))
                    stop_.value = -1
                    return -1

                obs = env.reset()

            episode += 1

        # select an action
        action = agent.act(obs, reward, agent_done, is_training=False)
        # take a step
        obs, reward, agent_done = env.do(action)
