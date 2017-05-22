# Village People, 2017

from .agent import ReportingAgent
from .batch_dqn_agent import BetaDQNBatchAgent
from .dqn_agent import DQNAgent
from .empathetic_agent import EmpatheticAgent
from .episodic_rdqn_agent import EpisodicRDQNAgent
from .random_agent import RandomAgent
from .actor_critic import Village_ActorCritic
from .actor_critic_aux import Village_ActorCritic_Aux
from .meta_actor_critic_aux import Meta_ActorCritic_Aux
from .feudal_agent import FeudalAgent

AGENTS = {
    "empathetic": EmpatheticAgent,
    "dqn": DQNAgent,
    "random": RandomAgent,
    "betaDQN": BetaDQNBatchAgent,
    "EpisodicRDQNAgent": EpisodicRDQNAgent,
    "ActorCritic": Village_ActorCritic,
    "ActorCriticAux": Village_ActorCritic_Aux,
    "MetaActorCriticAux": Meta_ActorCritic_Aux,
    "ActorCritic_Aux": Village_ActorCritic_Aux,
    "feudal": FeudalAgent
}

def get_agent(name):
    return AGENTS[name]
