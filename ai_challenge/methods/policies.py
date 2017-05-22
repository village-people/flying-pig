# Bitdefender, 2017

import torch
from torch.autograd import Variable

class DeterministicPolicy(object):
    def __init__(self, estimator):
        """Assumes estimator returns an autograd.Variable"""

        self.name = "DP"
        self.estimator = estimator

    def get_action(self, state_batch):
        """ Takes best action based on estimated state-action values."""
        # ret  = self.estimator(Variable(state_batch, volatile=True)).data
        state_ = Variable(state_batch.float(), volatile=True)

        return self.estimator(state_).data.max(1)


class StochasticPolicy(object):
    def __init__(self, estimator):
        self.name = "SP"
        self.estimator = estimator

    def get_action(self, state_batch):
        state_ = Variable(state_batch.float(), volatile=True)

        p = self.estimator(state_).data
        return torch.multinomial(p)
