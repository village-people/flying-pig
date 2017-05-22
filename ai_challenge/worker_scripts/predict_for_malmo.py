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
    print(clr("[QUEUE_PREDICT] ", "magenta") + message)


# Train on our environment
def predict_for_malmo(shared_objects, cfg):
    predict_queue = shared_objects["predict_queue"]
    send_back_queues = shared_objects["send_back_queues"]

    N = len(send_back_queues)

    # -- Initialize agent and wrap it in a Binary18BatchAgentWrapper :)
    Agent = get_agent(cfg.agent.type)
    agent = Agent(cfg.agent.name, ENV_ACTIONS, cfg, shared_objects)

    print_info("{:s}<{:s}> learns from queue. |role={:d}".format(
        cfg.agent.name, cfg.agent.type, cfg.agent.role
    ))

    dtype = torch.LongTensor(0)
    if cfg.general.use_cuda:
        dtype = dtype.cuda()

    while True:

        # -- 1.

        tasks = []
        while True:
            try:
                t = predict_queue.get(True, 0.1)
                tasks.append(t)
            except Queue.Empty:
                if len(tasks) > 1:
                    break

        # print("Let's predict for {:d} losers.".format(len(tasks)))

        # -- 4.
        # 4.a. Create batch from transitions
        # !! This is incomplete

        ids = torch.LongTensor([_id for (_id, _, _, _) in tasks])
        state = torch.cat([torch.LongTensor(s) for (_, s, _, _) in tasks],
                          0).float()
        done = torch.cat([torch.LongTensor(d) for (_, _, d, _) in tasks], 0)
        tokens = torch.LongTensor([token for (_, _, _, token) in tasks])

        if cfg.general.use_cuda:
            ids = ids.cuda()
            state = state.cuda()
            done = done.cuda()

        actions = agent.batch_predict(ids, state, done)
        # print(actions.unsqueeze(0))
        for _id, action, token in zip(ids, actions.tolist(), tokens):
            send_back_queues[_id].send((action, token))
            # print("Sent to {:d} action {:d}".format(_id, action))
        # print("Done! Waiting again...")
        tasks.clear()
