# Village People, 2017
#
# !! ATTENTION: IPS are written below and they not taken from some
# !! Fancy yaml file !!

import torch
import torch.multiprocessing as mp
from multiprocessing import Queue
from copy import deepcopy
from termcolor import colored as clr

from utils import read_config
from models import get_model

from worker_scripts import collect_from_malmo, train_from_malmo
from worker_scripts import predict_for_malmo

USE_PREDICTOR = False

LAN_IPS = [

    # Doar patru

    # ("172.19.3.173", [(10000, 10001)]),
    # ("172.19.3.236", [(10000, 10001)]),
    # ("172.19.3.232", [(10000, 10001)]),
    # ("172.19.3.234", [(10000, 10001)]),


    # Doar opt

    # ("172.19.3.173", [(10000, 10001)]),
    # ("172.19.3.236", [(10000, 10001)]),
    # ("172.19.3.232", [(10000, 10001)]),
    # ("172.19.3.234", [(10000, 10001)]),
    # ("172.19.3.234", [(10000, 10001)]),
    # ("172.19.3.230", [(10000, 10001)]),
    # ("172.19.3.229", [(10000, 10001)]),
    # ("172.19.3.240", [(10000, 10001)]),
    # ("172.19.3.240", [(10000, 10001)]),

    # doispe

    # ("172.19.3.173", [(10000, 10001)]),
    # ("172.19.3.236", [(10000, 10001)]),
    # ("172.19.3.232", [(10000, 10001)]),
    # ("172.19.3.234", [(10000, 10001)]),
    # ("172.19.3.234", [(10000, 10001)]),
    # ("172.19.3.230", [(10000, 10001)]),
    # ("172.19.3.229", [(10000, 10001)]),
    # ("172.19.3.240", [(10000, 10001)]),
    # ("172.19.3.240", [(10000, 10001)]),
    #
    # ("172.19.3.196", [(10000, 10001), (10002, 10003)]),
    # ("172.19.3.201", [(10000, 10001), (10002, 10003)]),

    # optspe

    # ("172.19.3.173", [(10000, 10001), (10002, 10003), (10004, 10005)]),
    # ("172.19.3.236", [(10000, 10001), (10002, 10003), (10004, 10005)]),
    # ("172.19.3.235", [(10000, 10001), (10002, 10003), (10004, 10005)]),
    # ("172.19.3.232", [(10000, 10001), (10002, 10003), (10004, 10005)]),
    # ("172.19.3.234", [(10000, 10001), (10002, 10003), (10004, 10005)])
    # ("172.19.3.173", [(10000, 10001), (10002, 10003), (10004, 10005)])

    # Toate

    # Local host
    ("192.168.0.100", [(10000, 10001), (10002, 10003), (10004, 10005)]),
    ("192.168.0.102", [(10000, 10001), (10002, 10003), (10004, 10005)]),
    ("192.168.0.101", [(10000, 10001), (10002, 10003), (10004, 10005)]),
    #("172.19.3.209", [(10000, 10001), (10002, 10003), (10004, 10005)]),
    #("172.19.3.240", [(10000, 10001), (10002, 10003), (10004, 10005)])

    # ("172.19.3.173", [(10000, 10001), (10002, 10003), (10004, 10005)]),
    # ("172.19.3.236", [(10000, 10001), (10002, 10003), (10004, 10005)]),
    # ("172.19.3.232", [(10000, 10001), (10002, 10003), (10004, 10005)]),
    # ("172.19.3.234", [(10000, 10001), (10002, 10003), (10004, 10005)]),
    # ("172.19.3.230", [(10000, 10001), (10002, 10003), (10004, 10005)]),
    # ("172.19.3.229", [(10000, 10001), (10002, 10003), (10004, 10005)]),
    # #
    # ("172.19.3.240", [(10000, 10001), (10002, 10003), (10004, 10005)]),
    # ("172.19.3.196", [(10000, 10001), (10002, 10003), (10004, 10005)]),
    # ("172.19.3.201", [(10000, 10001), (10002, 10003), (10004, 10005)]),
    # # ("172.19.3.189", [(10000, 10001)]),
    # ("172.19.3.208", [(10000, 10001)]),
    # ("172.19.3.190", [(10000, 10001)])
]


HOSTS = []
for ip, pairs in LAN_IPS:
    HOSTS.extend([[(ip, p1), (ip, p2)] for (p1, p2) in pairs])

def print_info(message):
    print(clr("[COLLECT-MAIN] ", "yellow") + message)


def immortal_collector(_id, shared_objects, cfg):
    while True:
        shared_objects["model"].share_memory()
        collector2 = mp.Process(target=collect_from_malmo,
                           args=(_id, shared_objects, cfg))
        collector2.start()
        collector2.join()


if __name__ == "__main__":
    print_info("Booting...")
    cfg = read_config()

        # -- Configure Torch

    if cfg.general.seed > 0:
        torch.manual_seed(cfg.general.seed)
        if cfg.general.use_cuda:
            torch.cuda.manual_seed_all(cfg.general.seed)

    if cfg.general.use_cuda:
        print_info("Using CUDA.")
    else:
        print_info("No GPU for you, Sir!")
    mp.set_start_method("spawn")
    print_info("Torch setup finished.")

    # -- Configure model

    shared_model = get_model(cfg.model.name)(cfg.model)
    if cfg.general.use_cuda:
        shared_model.cuda()
    shared_model.share_memory()
    print_info("Shared model {:s} initalized.".format(
        clr(cfg.model.name, "red"))
    )

    if isinstance(cfg.model.load, str):
        checkpoint = torch.load(cfg.model.load)
        iteration = checkpoint['iteration']
        reward = checkpoint['reward']
        print("LOADING MODEL: {} ---> MAX R: {}".format(cfg.model.load, reward))
        shared_model.load_state_dict(checkpoint['state_dict'])
    #

    # -- Shared objects
    shared_objects = {
        "model": shared_model,
        "queue": mp.Queue(),
        "reset": mp.Value("i", 0),
        "session": mp.Value("i", 0)
    }

    # -- Create predictor

    if USE_PREDICTOR:
        recv_queues, send_queues = {}, {}
        for i in range(len(HOSTS)):
            recv_queues[i], send_queues[i] = mp.Pipe()
        shared_objects["send_back_queues"] = send_queues
        shared_objects["predict_queue"] = mp.Queue()
        predictor = mp.Process(target=predict_for_malmo,
                               args=(shared_objects, deepcopy(cfg)))

    # -- Create players

    collectors = []
    for _id, hosts in enumerate(HOSTS):
        cfg.envs.minecraft.ports = hosts
        cfg.agent.mode = "collect"
        if USE_PREDICTOR:
            shared_objects["answer_pipe"] = recv_queues
        collector = mp.Process(target=collect_from_malmo,
                               args=(_id, shared_objects, deepcopy(cfg)))
        collectors.append(collector)

    # -- Create trainer
    cfg.agent.mode = "train_from_queue"
    trainer = mp.Process(target=train_from_malmo, args=(shared_objects, cfg))


    # -- Start all
    if USE_PREDICTOR:
        predictor.start()
    for c in collectors:
        c.start()
    trainer.start()

    # -- Finished

    trainer.join()
    for c in collectors:
        c.join()

    print_info("Done")
