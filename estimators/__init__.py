# Bitdefender, 2107

from estimators.pig_chase import TopDown
from estimators.mini_net import MiniNet

ESTIMATORS = {
    "mini": MiniNet,
    "top_down": TopDown
}


def get_estimator(estimator_name):
    return ESTIMATORS[estimator_name]
