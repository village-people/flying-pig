# Village People, 2017

from models.test_model import TestModel
from models.pig_chase import TopDown
from models.pig_chase_beta import BetaBinaryBoardModel
from models.feudal import FuN
from models.recurrent_models import RecurrentQEstimator
from models.utils import conv_out_dim
from models.ActorCriticNet import Policy

ALL_MODELS = {
    "top_down": TopDown,
    "test": TestModel,
    "beta_binary18": BetaBinaryBoardModel,
    "recurrent_q": RecurrentQEstimator,
    "ActorCriticNet": Policy,
    "feudal": FuN
}

def get_model(name):
    return ALL_MODELS[name]
