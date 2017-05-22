# 2017, Andrei N., Tudor B.
from sphinx.addnodes import centered

from ._ignore_Agent import Agent
from ._ignore_Agent import Transition
import matplotlib.pyplot as plt

from random import choice
import logging
import os

import numpy as np
import math
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
import torchvision.transforms as T

from torch.autograd import Variable


class BetaDQNBatchAgent(Agent):
    """
    Baseline Agent - Q-Learning with CNN
    """

    def __init__(self, name, action_space, model, cfg):
        super(BetaDQNBatchAgent, self).__init__(name, cfg)

        self.logger.info("On duty...")

        self.eps_start = float(0.9)
        self.eps_end = float(0.05)
        self.eps_decay = float(200)

        self.gameMoves = 0
        self.gameLoss = 0

        self._lastLoss = 0
        self._losses = []
        self.model_class = model

        self.cfg = cfg

        super().__post_init__()

    def _act(self, observation, reward, done, is_training):
        """Class code here"""
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
                                       math.exp(-1. * self._crtStep /
                                                self.eps_decay)
        if sample > eps_threshold:
            q = self._modelClass._model(Variable(observation, volatile=True))
            action = q.data.max(1)
        else:
            action = torch.LongTensor([[self.action_space.sample()]])

        return action

    def _restart(self):
        pass

    def _epochFinished(self):
        pass

    def _report(self):
        self._losses.append(self._lastLoss)
        self.logger.info("Loss:: {}".format(self._lastLoss))
        self._lastLoss = 0

    def _saveModel(self, *args, **kwargs):
        pass

    def _createLearningArchitecture(self):
        model = self.model_class(self.cfg)
        optimizer = optim.RMSprop(model.parameters())
        criterion = F.smooth_l1_loss
        self._modelClass.loadModel(model, optimizer, criterion)

    def _optimizeModel(self):

        transition = self._memory.last()

        BATCH_SIZE = len(transition)

        if BATCH_SIZE <= 0:
            return

        batch = Transition(*zip(*transition))

        state_batch = Variable(torch.cat(batch.state), volatile=True)
        action_batch = Variable(torch.cat(batch.action), volatile=True)
        reward_batch = Variable(torch.cat(batch.reward), volatile=True)
        next_state_values = Variable(torch.zeros(BATCH_SIZE), volatile=True)

        non_final_mask = torch.ByteTensor(batch.done)
        if non_final_mask.any():
            non_final_next_states_t = torch.cat(
                tuple(s for s in batch.next_state
                      if s is not batch.done)) \
                .type(self.dtype)
            non_final_next_states = Variable(non_final_next_states_t,
                                             volatile=True)
            next_state_values[non_final_mask] = self._modelClass._model(
                non_final_next_states).max(1)[0].cpu()

        if self._useCUDA:
            action_batch = action_batch.cuda()

        expected_state_action_values = (
                                       next_state_values * self.discount) + reward_batch
        state_action_values = self._modelClass._model(state_batch). \
            gather(1, action_batch).cpu()

        loss = self._modelClass._criterion(state_action_values,
                                           expected_state_action_values)

        self._lastLoss += loss.data[0]
        self._modelClass._optimizer.zero_grad()
        loss.backward()
        for param in self._modelClass._model.parameters():
            param.grad.data.clamp_(-1, 1)
        self._modelClass._optimizer.step()
