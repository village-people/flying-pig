# Bitdefender, 2107

from estimators.atari_net import AtariNet
from estimators.mini_net import MiniNet
from estimators.catch_net import CatchNet

ESTIMATORS = {
    "atari": AtariNet,
    "mini": MiniNet,
    "catch": CatchNet
}


def get_estimator(estimator_name):
    return ESTIMATORS[estimator_name]
