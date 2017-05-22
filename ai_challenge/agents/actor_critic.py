# Village People, 2017

import torch
import numpy as np

import torch.nn as nn
import torch.nn.functional as F

from models.utils import conv_out_dim

from agents import ReportingAgent
from models import get_model
from methods import DeterministicPolicy, DQNPolicyImprovement
from methods import get_batch_schedule
from data_structures.transition import Transition
from utils.torch_types import TorchTypes
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

from termcolor import colored as clr

from collections import namedtuple

# from termcolor import colored as clr
ENV_CAUGHT_REWARD = 25

class Policy(nn.Module):
    """ PigChase Model for the 18BinaryView batch x 18 x 9 x 9.

    Args:
        state_dim (tuple): input dims: (channels, width, history length)
        action_no (int): no of actions
        hidden_size (int): size of the hidden linear layer
    """

    def __init__(self, config):
        state_dim = (18, 9, 1)
        action_no = 3
        hidden_size = 128

        super(Policy, self).__init__()

        self.in_channels, self.in_width, self.hist_len = state_dim
        self.action_no = action_no
        self.hidden_size = hidden_size
        in_depth = self.hist_len * self.in_channels

        self.conv1 = nn.Conv2d(in_depth, 16, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(32)

        map_width1 = conv_out_dim(self.in_width, self.conv1)
        map_width2 = conv_out_dim(map_width1, self.conv2)
        map_width3 = conv_out_dim(map_width2, self.conv3)

        self.lin1 = nn.Linear(32 * map_width3**2, self.hidden_size)

        self.action_head = nn.Linear(self.hidden_size, action_no)
        self.value_head = nn.Linear(self.hidden_size, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.lin1(x))

        action_scores = self.action_head(x)
        state_values = self.value_head(x)

        return F.softmax(action_scores), state_values

    def get_attributes(self):
        return (self.input_channels, self.hist_len, self.action_no,
                self.hidden_size)


SavedTransition = namedtuple('Transition',
                             ['obs', 'action', 'value', 'reward',
                              'next_obs', 'playing_idx', 'live_idx'])

class Village_ActorCritic(ReportingAgent):
    def __init__(self, name, action_space, cfg, shared_objects={}):
        super(Village_ActorCritic, self).__init__()

        self.name = name
        self.pidx = 0
        self.actions_no = len(action_space)
        self.target_update_freq = 400
        self.batch_size = batch_size = cfg.general.batch_size
        self.update_freq = batch_size
        self.hist_len = 1
        self.cuda = cfg.general.use_cuda
        self.torch_obs = True
        self.dtype = TorchTypes(self.cuda)

        #TODO move to
        self.net = net = Policy(cfg)
        if self.cuda:
            net.cuda()

        self.target = get_model(cfg.model.name)(cfg.model)
        if self.cuda:
            self.net.cuda()
            self.target.cuda()

        self.exploration_strategy = get_batch_schedule(cfg.agent.exploration,
                                                       batch_size)
        self.batch_games = 0
        self.step_cnt = 0

        self.saved_transitions = []

        self._o, self._a, self._v = None, None, None

        self.live_idx = torch.linspace(0, batch_size-1, batch_size) \
                             .type(self.dtype.LongTensor)

        self.lr = .001
        self.gamma = cfg.agent.gamma

        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        self.optimizer.zero_grad()


    def get_models(self):
        return [self.net, self.target]

    def act(self, obs, reward, done, is_training):
        self.epsilon = next(self.exploration_strategy)
        # self.epsilon = 0
        not_done = (1 - done)
        have_more_games = not_done.byte().any()
        if have_more_games:
            not_done_idx = not_done.nonzero().view(-1)
            non_done_states = obs.index_select(0, not_done_idx)
        else:
            not_done_idx = torch.LongTensor(0).type_as(done)
            non_done_states = None

        action = None
        if have_more_games:

            state = non_done_states.float()

            # select Action
            probs, state_value = self.net(Variable(state))
            action = probs.multinomial()

        if self._o is not None and is_training:
            self.saved_transitions.append(SavedTransition(
                self._o, self._a, self._v, reward.float(), obs, not_done_idx, self.live_idx))

        if have_more_games:
            self.live_idx = live_idx = \
                        self.live_idx.index_select(0, not_done_idx)
            self._o = state
            self._a = action
            self._v = state_value
        else:
            self.finish_episode()
            # input("Press Enter to continue...")

            self._o, self._a, self._v = None, None, None
            batch_size = self.batch_size
            self.live_idx = torch.linspace(0, batch_size-1, batch_size) \
                                 .type(self.dtype.LongTensor)



        self.step_cnt += 1

        #MUST RETURN SAME SIZE AS OBS SIZE
        action_holder = torch.LongTensor(reward.size(0))\
            .type(self.dtype.LongTensor)
        action_holder.fill_(3)
        if action is not None:
            action_holder.scatter_(0, not_done_idx, action.data.view(-1).long())
        return action_holder


    def finish_episode(self):

        saved_transitions = self.saved_transitions
        value_loss = 0
        rewards = []

        for i in range(len(saved_transitions))[::-1]:
            r = saved_transitions[i].reward
            playing_idx = saved_transitions[i].playing_idx
            if i < len(saved_transitions)-1:
                r.index_add_(0, playing_idx, self.gamma * rewards[0])

            rewards.insert(0, r)

        all_r = torch.cat(rewards, 0)
        mean_r = all_r.mean()
        std_r = all_r.std() + np.finfo(np.float32).eps

        for i in range(len(rewards)):
            rewards[i].sub_(mean_r).div_(std_r)

        all_actions = []
        for i in range(len(saved_transitions)):
            tr = saved_transitions[i]
            r = rewards[i]
            reward = r - tr.value.data[0, 0]
            tr.action.reinforce(reward.unsqueeze(1))
            value_loss += F.smooth_l1_loss(tr.value,
                                           Variable(r))
            all_actions.append(tr.action)

        self.optimizer.zero_grad()
        final_nodes = [value_loss] + all_actions
        # print(value_loss)
        gradients = [torch.ones(1).type_as(r)] + [None] * len(saved_transitions)
        torch.autograd.backward(final_nodes, gradients)
        self.optimizer.step()

        self.step_cnt = 0
        self.batch_games = 0
        del self.saved_transitions[:]

    def _frame2torch(self, obs):
        return torch.from_numpy(obs).unsqueeze(0).unsqueeze(0)

    def _batch2torch(self, batch, batch_sz=None):
        """
        List of Transitions of batches to tensors batch states, actions,
        rewards.
        """

        batch = Transition(*zip(*batch))
        # print("[%s] Batch len=%d" % (self.name, batch_sz))

        state_batch = torch.cat(batch.state, 0).float()

        #ONLY NON Terminal next states
        next_state_batch = [x for x in batch.state_ if x is not None]
        next_state_batch = torch.cat(next_state_batch, 0).float()

        action_batch = torch.cat(batch.action, 0)
        reward_batch = torch.cat(batch.reward, 0).float()

        mask = torch.cat(batch.done, 0)
        mask = (1 - mask).nonzero().view(-1)

        batch_sz = action_batch.size(0)

        return [batch_sz, state_batch, action_batch, reward_batch,
                next_state_batch, mask]
