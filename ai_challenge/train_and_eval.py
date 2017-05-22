# Village People, 2017

import torch
import torch.multiprocessing as mp

import time
from termcolor import colored as clr

from models import get_model
from utils import read_config
from utils import AtomicStatistics

from evaluator import Evaluator
from workers import train_agent, run_alien

# Useful functions
def print_info(message):
    print(clr("[MAIN] ", "yellow") + message)


def mock_function():
    print("Target function not implemented.")


# Train on Minecraft
def train_on_malmo(shared_objects, config):
    # alien first
    return [mp.Process(target=run_alien, args=(shared_objects, config)),
            mp.Process(target=train_agent, args=(shared_objects, config))]


# Train on our environment
def train_on_simulator(shared_objects, config):
    return mp.Process(target=mock_function)


def main():

    print_info("Starting...")

    # -- Read configuration
    config = read_config()

    # -- Configure Torch
    torch.manual_seed(config.general.seed)
    if config.general.use_cuda:
        torch.cuda.manual_seed_all(config.general.seed)
    mp.set_start_method("spawn")

    # Configure model
    shared_model = get_model(config.model.name)(config.model)
    if config.general.use_cuda:
        shared_model.cuda()
    shared_model.share_memory()

    # Shared statistics
    shared_stats = AtomicStatistics()

    # Shared objects
    shared_objects = {
        "model": shared_model,
        "stats_leRMS": shared_stats
    }

    procs = []
    if config.envs.minecraft.use:
        minecraft_procs = train_on_malmo(shared_objects, config)
        procs.extend(minecraft_procs)

    if config.envs.simulated.use:
        simulator_proc = train_on_simulator(shared_objects, config)
        procs.append(simulator_proc)

    evaluator_proc = Evaluator(shared_objects, config.evaluation)  # noqa
    # procs.append(evaluator_proc)

    start_time = time.time()

    for p in procs:
        p.start()
        time.sleep(1)

    for p in procs:
        p.join()

    total_time = time.time() - start_time
    print_info("Everything done in {:.2f}!".format(total_time))

if __name__ == "__main__":
    main()
