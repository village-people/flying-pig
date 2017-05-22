# Village People, 2017

import torch
import numpy as np

from models import get_model
from methods import DeterministicPolicy, DQNPolicyImprovement
from methods import get_schedule
from data_structures.transition import Transition
from utils.torch_types import TorchTypes
# from termcolor import colored as clr


class DQNAgent(object):
    def __init__(self, name, action_space):
        self.name = name
        self.pidx = 0
        self.actions_no = len(action_space)
        self.target_update_freq = 100
        self.update_freq = 4
        self.batch_size = 64
        self.hist_len = 1
        self.cuda = False

        Model = get_model("top_down")
        self.net = Model((1, 18, self.hist_len), self.actions_no, 4)
        self.target = Model((1, 18, self.hist_len), self.actions_no, 4)

        self.behaviour = DeterministicPolicy(self.net)
        self.exploration_strategy = get_schedule("linear", 1, .1, 10000)
        self.algorithm = DQNPolicyImprovement(self.net, self.target)
        self.batch = []

        self.dtype = TorchTypes(self.cuda)
        self.step_cnt = 0
        self._o = None
        self._a = None

    def act(self, obs, reward, done, is_training):
        print("sigut")

        # I can't figure out why but sometimes I get a None observation.
        if obs is None:
            return np.random.randint(self.actions_no)

        # self.epsilon = next(self.exploration_strategy)
        self.epsilon = 0

        action = None
        if self.epsilon < np.random.uniform():
            state = self._frame2torch(obs)
            _, action = self.behaviour.get_action(state)
            action = action[0, 0]
        else:
            action = np.random.randint(self.actions_no)

        if self._o is not None and is_training:
            self._improve_policy(self._o, self._a, reward, obs, done)

        if not done:
            self._o, self._a = obs, action
        else:
            self._o, self._a = None, None

        self.step_cnt += 1
        return action

    def _improve_policy(self, _s, _a, r, s, done):

        self.batch.append((self.pidx, _s, _a, r, s, done))

        if (len(self.batch) == self.update_freq) or done:
            batch = self._batch2torch(self.batch)
            self.algorithm.accumulate_gradient(*batch)
            self.algorithm.update_model()
            # print(self.step_cnt, ":", len(self.batch), done)
            self.batch.clear()

        if self.step_cnt % self.target_update_freq == 0:
            self.algorithm.update_target_net()

    def _frame2torch(self, obs):
        return torch.from_numpy(obs).unsqueeze(0).unsqueeze(0)

    def _batch2torch(self, batch, batch_sz=None):
        """ List of Transitions to List of torch states, actions, rewards.

            From a batch of transitions (id, s0, a0, s1, r0, d)
            get a batch of the form state=(s0,s1...), action=(a1,a2...),
            state(s1,s2...), reward(s1, s2...)
            Inefficient. Adds 1.5s~2s for 20,000 steps with 32 agents.
        """

        batch_sz = len(batch) if batch_sz is None else batch_sz
        batch = Transition(*zip(*batch))
        # print("[%s] Batch len=%d" % (self.name, batch_sz))

        states = [torch.from_numpy(s).unsqueeze(0) for s in batch.state]
        states_ = [torch.from_numpy(s).unsqueeze(0) for s in batch.state_]

        state_batch = torch.stack(states).type(self.dtype.FloatTensor)
        action_batch = self.dtype.LongTensor(batch.action)
        reward_batch = self.dtype.FloatTensor(batch.reward)
        next_state_batch = torch.stack(states_).type(self.dtype.FloatTensor)

        # Compute a mask for terminal next states
        # [True, False, False] -> [1, 0, 0]::ByteTensor
        mask = 1 - self.dtype.ByteTensor(batch.done)

        return [batch_sz, state_batch, action_batch, reward_batch,
                next_state_batch, mask]
