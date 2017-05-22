# Village People, 2017

import torch
from torch.multiprocessing import Queue
import time
from termcolor import colored as clr

from pig_chase.common import ENV_ACTIONS
from agents import get_agent
import numpy as np

import queue as Queue

def print_info(message):
    print(clr("[QUEUE_TRAIN] ", "magenta") + message)

# Train on our environment
def train_from_malmo(shared_objects, cfg):
    batch_size = cfg.general.batch_size
    queue = shared_objects["queue"]
    session = shared_objects["session"]
    reset = shared_objects["reset"]

    # -- Initialize agent and wrap it in a Binary18BatchAgentWrapper :)
    Agent = get_agent(cfg.agent.type)
    agent = Agent(cfg.agent.name, ENV_ACTIONS, cfg, shared_objects)

    print_info("{:s}<{:s}> learns from queue. |role={:d}".format(
        cfg.agent.name, cfg.agent.type, cfg.agent.role
    ))

    dtype = torch.LongTensor(0)
    if cfg.general.use_cuda:
        dtype = dtype.cuda()

    episodes_no = cfg.training.episodes_no
    best_r_ep = None
    best_r_frame = None

    frame_rewards = []
    episode_rewards = []

    for episode in range(1, episodes_no + 1):
        # 1. Checks the queue. If less than 32.. check again, else goto 2.
        # 2. Inform others to drop future experiences and wait for new params.
        # 3. Collect transitions
        # 4. Train agent and update parameters
        # 5. Drop any shit from queue
        # 6. Inform others that they should take the new params. Go to 1.

        # -- 1.
        while queue.qsize() < batch_size:
            time.sleep(.1)

        # -- 2.
        reset.value = 1

        # -- 3.

        transitions = []
        while len(transitions) < batch_size:
            try:
                t = queue.get()
                transitions.append(t)
            except Queue.Empty:
                print("futere")
                break
        while not queue.empty():
            try:
                t = queue.get()
                transitions.append(t)
            except Queue.Empty:
                print("futere")
                break

        # -- 4.
        # 4.a. Create batch from transitions
        # !! This is incomplete

        print(transitions)
        (s, r, d, a) = transitions[0][0]
        (s, r, d, a) = torch.LongTensor(s), torch.FloatTensor(
            r), torch.LongTensor(d), torch.LongTensor(a)
        _s = s.new().resize_(torch.Size([0]) + s.size()[1:])
        _a = a.new().resize_(torch.Size([0]) + a.size()[1:])
        _r = r.new().resize_(torch.Size([0]) + r.size()[1:])
        _d = d.new().resize_(torch.Size([0]) + d.size()[1:])

        # -- Apply padding on short games
        n = len(transitions)
        max_len = max([len(game) for game in transitions])
        avg_len = np.mean([len(game) for game in transitions])

        fake = [(_s, _r, _d, _a)]
        transitions = [t + fake * (max_len - len(t)) for t in transitions]
        transitions = list(map(list, zip(*transitions)))

        all_r = .0
        total_r = .0

        a = time.time()
        rewards_no = 0
        for step, all_t in enumerate(transitions):
            states = torch.cat(
                list(map(lambda t: torch.LongTensor(t[0]), all_t)), 0)
            rewards = torch.cat(
                list(map(lambda t: torch.FloatTensor(t[1]), all_t)), 0)
            done = torch.cat(list(map(lambda t: torch.LongTensor(t[2]), all_t)),
                             0)
            actions = torch.cat(
                list(map(lambda t: torch.LongTensor(t[3]), all_t)), 0)

            _alive_no = states.size(0)
            print("Alive: {:d}, but {:d} are dead!".format(_alive_no,
                                                           done.nonzero().nelement()))

            assert actions.size(0) == _alive_no
            assert rewards.size(0) == _alive_no
            assert done.size(0) == _alive_no

            if cfg.general.use_cuda:
                states = states.cuda()
                rewards = rewards.cuda()
                done = done.cuda()
                actions = actions.cuda()

            # print("---------Step {} ==========".format(step))
            # print("Some transition:")
            # one_hot = states
            # print("rewards: :", rewards)
            # print("done: ", done)
            # print("action: ", actions)
            # print(
            #     one_hot[0, 4] + one_hot[0, 5] * 2 + one_hot[0, 6] * 3 +
            #     one_hot[
            #         0, 7] * 4
            #     + one_hot[0, 8] * 7
            #     + one_hot[0, 13] * 11)

            agent.act(states, rewards, done, True, actions=actions)
            all_r += rewards.sum()
            rewards_no += _alive_no
            total_r += rewards.sum()

        b = time.time()

        agent.reset()

        # -- 5.

        session.value = session.value + 1

        while not queue.empty():
            try:
                queue.get_nowait()
            except Queue.Empty:
                break

        print_info("Go again!")
        reset.value = 0

        all_r /= rewards_no
        total_r /= n

        do_save = False

        if best_r_frame is None or best_r_frame < all_r:
            do_save = True
            best_r_frame = all_r
            r_str = clr("{:.6f}".format(best_r_frame), "white", "on_magenta")
            # salveaza ceva
            save_model(self, best_r_frame, episode, save_only_min=False)
            # agent.save_model()
        else:
            r_str = clr("{:.6f}".format(all_r), "magenta")

        if best_r_ep is None or best_r_ep < total_r:
            do_save = True
            best_r_ep = total_r
            r2_str = clr("{:.6f}".format(best_r_ep), "white", "on_magenta")
            # salveaza ceva
            # agent.save_model()

        else:
            r2_str = clr("{:.6f}".format(total_r), "magenta")

        print_info(
            "Episode: " + clr("{:d}".format(episode), "blue") + clr(" | ",
                                                                    "yellow") +
            "Rewards per episode: " + r2_str + clr(" | ", "yellow") +
            "Rewards per frame: " + r_str + clr(" | ", "yellow") +
            "Batch size: " + clr("{:d}".format(n), "blue") + clr(" | ",
                                                                 "yellow") +
            "Avg length: " + clr("{:.2f}".format(avg_len), "blue") + clr(" | ",
                                                                         "yellow") +
            "Back time: " + clr("{:.2f}".format(b - a), "blue")
        )

        if do_save:
            agent.model_utils.save_model(all_r, total_r, episode,
                                         save_only_min=False)

        frame_rewards.append(all_r)
        episode_rewards.append(total_r)

        print("-----------------")
        print("Last ten:")
        print("Last ten step rewards: ", frame_rewards[-10:])
        print("Last ten epis rewards: ", episode_rewards[-10:])
        print("-----------------")

        torch.save(torch.stack([torch.FloatTensor(frame_rewards),
                                torch.FloatTensor(frame_rewards)]),
                   "results/rewards.torch")
