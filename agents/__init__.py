from .empathetic_agent import EmpatheticAgent
from .dqn_agent import DQNAgent
from .random_agent import RandomAgent

AGENTS = {
    "empathetic": EmpatheticAgent,
    "dqn": DQNAgent,
    "random": RandomAgent
}


def get_agent(name):
    return AGENTS[name]
