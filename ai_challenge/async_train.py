# Village People, 2017

# This script trains an agent on Artificial Pig Chase environment and
# evaluates it both on Artificial Pig Chase, and Malmo environments.

import torch
import torch.multiprocessing as mp

import time
from termcolor import colored as clr

# from agents import get_agent
from models import get_model
from worker_scripts import train_on_simulator
from evaluator import Evaluator
from utils import read_config
from utils import AtomicStatistics


# from utils import AverageMeter


def print_info(message):
    print(clr("[MAIN] ", "yellow") + message)


def startup_message(cfg):
    print_info("Training {:s} versus {:s}!".format(
        clr("{:s}<{:s}>".format(cfg.agent.type, cfg.model.name),
            "white", "on_magenta"),
        clr("others", "white", "on_cyan")  # <TODO:tudor:put alien's name>
    ))


def main():
    print_info("Booting...")
    cfg = read_config()
    startup_message(cfg)

    # -- Print important conf info
    print()
    print_info("CONFIG >")
    print_info(clr("P_FOCUS_simulated: {}".format(cfg.envs.simulated.p_focus),
                   "red"))
    print_info(clr("P_FOCUS_minecraft: {}".format(cfg.envs.minecraft.p_focus),
                   "red"))
    print_info(clr("Pig_max_moves: {}".format(cfg.envs.simulated.pig_max_moves),
                   "red"))
    print()

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

    if isinstance(cfg.model.load, str):
        checkpoint = torch.load(cfg.model.load)
        iteration = checkpoint['iteration']
        if "min_r_ep" in checkpoint:
            reward = checkpoint['min_r_ep']
            reward_r_frame = checkpoint['min_r_frame']
        else:
            reward = checkpoint['reward']

        print("Loading Model: {} Mean reward/ episode: {}"
              "".format(cfg.model.load, reward))
        shared_model.load_state_dict(checkpoint['state_dict'])
    #

    print_info("Shared model {:s} initalized.".format(
        clr(cfg.model.name, "red"))
    )

    # -- Shared objects
    shared_objects = {
        "model": shared_model,
        "stats_leRMS": AtomicStatistics()
    }

    # -- Start processes
    procs = []
    sleep_between_processes = False

    if cfg.envs.minecraft.use:
        """ Has been separated to different processing pipeline"""
        raise NotImplementedError  # TODO
        minecraft_procs = train_on_malmo(shared_objects, cfg)
        procs.extend(minecraft_procs)
        sleep_between_processes = 1

    if cfg.envs.simulated.use:
        simulator_proc = mp.Process(target=train_on_simulator,
                                    args=(shared_objects, cfg))
        procs.append(simulator_proc)
        # simulator_proc = mp.Process(target=train_on_simulator,
        #                             args=(shared_objects, cfg))
        # procs.append(simulator_proc)

    if cfg.evaluation.malmo.use:
        evaluator_proc = Evaluator(shared_objects, cfg.evaluation.malmo)
        procs.append(evaluator_proc)

    if cfg.evaluation.artificial.use:
        raise NotImplementedError
        artif_evaluator_proc = ArtificialEnvEvaluator(
            shared_objects, cfg.evaluation.artificial)
        procs.append(artif_evaluator_proc)

    print_info("Training starts now!")
    start_time = time.time()

    for p in procs:
        p.start()
        if sleep_between_processes:
            time.sleep(sleep_between_processes)

    for p in procs:
        p.join()

    total_time = time.time() - start_time
    print_info("Training and evluation done in {:.2f}!".format(total_time))


if __name__ == "__main__":
    main()
