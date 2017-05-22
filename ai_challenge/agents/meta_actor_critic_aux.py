# Village People, 2017

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils import conv_out_dim

from agents import ReportingAgent
from models import get_model
from methods import get_batch_schedule
from data_structures.transition import Transition
from utils.torch_types import TorchTypes
from torch.autograd import Variable
import torch.optim as optim

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

    def __init__(self, config, hidden_size):
        super(Policy, self).__init__()

        state_dim = (15, 9, 1)
        action_no = 3
        hidden_size = hidden_size

        self.rnn_type = rnn_type = "LSTMCell"
        self.rnn_layers = 2
        self.rnn_nhid = rnn_nhid = hidden_size

        self.activation = nn.ReLU()

        self.in_channels, self.in_width, self.hist_len = state_dim
        self.action_no = action_no
        self.hidden_size = hidden_size
        in_depth = self.hist_len * self.in_channels

        self.conv1 = nn.Conv2d(in_depth, 256, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)

        map_width1 = conv_out_dim(self.in_width, self.conv1)
        map_width2 = conv_out_dim(map_width1, self.conv2)
        map_width3 = conv_out_dim(map_width2, self.conv3)

        lin_size = 64 * map_width3**2

        self.lin1 = nn.Linear(lin_size, rnn_nhid)
        self.bnLin1 = nn.BatchNorm1d(rnn_nhid)

        self.rnn1 = getattr(nn, rnn_type)(lin_size, rnn_nhid)
        self.rnn2 = getattr(nn, rnn_type)(rnn_nhid, rnn_nhid)

        self.bnRnn1 = nn.BatchNorm1d(rnn_nhid)
        self.bnRnn2 = nn.BatchNorm1d(rnn_nhid * self.rnn_layers)

        self.lin2 = nn.Linear(rnn_nhid * self.rnn_layers, hidden_size)
        self.bnLin2 = nn.BatchNorm1d(hidden_size)

        self.action_head = nn.Linear(self.hidden_size, action_no)
        self.value_head = nn.Linear(self.hidden_size, 1)
        self.bn_value_head = nn.BatchNorm1d(1)

    def forward(self, x, hidden_states):
        act = self.activation
        x = act(self.bn1(self.conv1(x)))
        x = act(self.bn2(self.conv2(x)))
        x = act(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)

        hidden0 = self.rnn1(x, hidden_states[0])
        if self.rnn_type == 'LSTMCell':
            x = hidden0[0]
        else:
            x = hidden0

        hx = [x]

        hidden1 = self.rnn2(x, hidden_states[1])
        if self.rnn_type == 'LSTMCell':
            x = hidden1[0]
        else:
            x = hidden1

        all_hidden = [hidden0, hidden1]
        hx.append(x)
        x = torch.cat(hx, 1)
        x = act(self.bnLin2(self.lin2(x)))

        action_scores = self.action_head(x)
        state_values = self.value_head(x)

        return F.softmax(action_scores), state_values, all_hidden

    def init_hidden(self, bsz):
        hidden_states = []
        weight = next(self.parameters()).data
        # Hidden 0
        if self.rnn_type == 'LSTMCell':
            hidden_states.append((Variable(weight.new(bsz, self.rnn_nhid)
                                           .zero_()),
                                  Variable(weight.new(bsz, self.rnn_nhid)
                                           .zero_())))
        else:
            hidden_states.append(Variable(weight.new(bsz, self.rnn_nhid)
                                          .zero_()))
        # Hidden 0
        if self.rnn_type == 'LSTMCell':
            hidden_states.append((Variable(weight.new(bsz, self.rnn_nhid)
                                           .zero_()),
                                  Variable(weight.new(bsz, self.rnn_nhid)
                                           .zero_())))
        else:
            hidden_states.append(Variable(weight.new(bsz, self.rnn_nhid)
                                 .zero_()))

        return hidden_states

    def slice_hidden(self, hidden_state, not_done_idx):
        hidden_states = []

        i = 0
        if self.rnn_type == 'LSTMCell':
            hidden_states.append((hidden_state[i][0]
                                  .index_select(0, Variable(not_done_idx)),
                                 hidden_state[i][1]
                                 .index_select(0, Variable(not_done_idx))))
        else:
            hidden_states.append(hidden_state[i]
                                 .index_select(0, Variable(not_done_idx)))
        i = 1
        if self.rnn_type == 'LSTMCell':
            hidden_states.append((hidden_state[i][0]
                                  .index_select(0, Variable(not_done_idx)),
                                  hidden_state[i][1]
                                  .index_select(0, Variable(not_done_idx))))
        else:
            hidden_states.append(hidden_state[i]
                                 .index_select(0, Variable(not_done_idx)))
        return hidden_states

    def get_attributes(self):
        return (self.input_channels, self.hist_len, self.action_no,
                self.hidden_size)


SavedTransition = namedtuple('Transition',
                             ['obs', 'action', 'value', 'other_task',
                              'reward', 'next_obs', 'playing_idx', 'live_idx'])


class Meta_ActorCritic_Aux(ReportingAgent):
    def __init__(self, name, action_space, cfg, shared_objects={}):
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
        self.clip_grad = 0.25

        self.hidden_size = 256
        self.max_reward = 0
        self.net = net = Policy(cfg, self.hidden_size)

        if self.cuda:
            net.cuda()

        self.target = get_model(cfg.model.name)(cfg.model)
        if self.cuda:
            self.net.cuda()
            self.target.cuda()

        self.exploration_strategy = get_batch_schedule(
                cfg.agent.exploration, batch_size)
        self.batch_games = 0
        self.step_cnt = 0

        self.saved_transitions = []

        self._o, self._a, self._v = None, None, None

        self.live_idx = torch.linspace(0, batch_size-1, batch_size) \
                             .type(self.dtype.LongTensor)

        self.lr = .001
        self.gamma = .99

        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        self.optimizer.zero_grad()

        self.hidden_state = self.net.init_hidden(self.batch_size)

        self.obs_select = torch.LongTensor([0, 1, 2,
                                            4, 5, 6, 7,
                                            9, 10, 11, 12,
                                            14, 15, 16, 17]).type(
                                                    self.dtype.LongTensor)

    def get_models(self):
        return [self.net, self.target]

    def act(self, obs, reward, done, is_training):
        obs = obs.index_select(1, self.obs_select)
        self.epsilon = next(self.exploration_strategy)
        not_done = (1 - done)
        have_more_games = not_done.byte().any()

        hidden_state = self.hidden_state

        if have_more_games:
            not_done_idx = not_done.nonzero().view(-1)
            non_done_states = obs.index_select(0, not_done_idx)
            hidden_state = self.net.slice_hidden(hidden_state, not_done_idx)
        else:
            not_done_idx = torch.LongTensor(0).type_as(done)
            non_done_states = None
        action = None
        if have_more_games:

            state = non_done_states.float()

            # select Action
            probs, state_value, new_hidden_state = \
                self.net(Variable(state), hidden_state)

            action = probs.multinomial()

        if self._o is not None and is_training:
            self.saved_transitions.append(SavedTransition(
                self._o, self._a, self._v, None, reward.float(),
                obs.float(), not_done_idx, self.live_idx))

        if have_more_games:
            # self.live_idx = live_idx = \
                        # self.live_idx.index_select(0, not_done_idx)
            self._o = state
            self._a = action
            self._v = state_value
            self.hidden_state = new_hidden_state
        else:
            self.finish_episode()

            self.hidden_state = self.net.init_hidden(self.batch_size)

            self._o, self._a, self._v = None, None, None
            batch_size = self.batch_size
            self.live_idx = torch.linspace(0, batch_size-1, batch_size) \
                                 .type(self.dtype.LongTensor)

        self.step_cnt += 1

        # MUST RETURN SAME SIZE AS OBS SIZE
        action_holder = torch.LongTensor(reward.size(0))\
            .type(self.dtype.LongTensor)
        action_holder.fill_(3)
        if action is not None:
            action_holder.scatter_(0, not_done_idx,
                                   action.data.view(-1).long())
        return action_holder

    def finish_episode(self):
        saved_transitions = self.saved_transitions
        value_loss = 0
        rewards = []

        for i in range(len(saved_transitions))[::-1]:
            r = saved_transitions[i].reward
            playing_idx = saved_transitions[i].playing_idx
            if i < len(saved_transitions)-1:
                r.index_add_(0, playing_idx, rewards[0])

            rewards.insert(0, r)

        for i in range(len(rewards)):
            rewards[i].div_(24)

        all_actions = []

        for i in range(len(saved_transitions)):
            tr = saved_transitions[i]
            r = rewards[i]

            reward = r - tr.value.data[:, 0].mean()
            tr.action.reinforce(reward.unsqueeze(1))
            value_loss += nn.MSELoss()(tr.value, Variable(r))
            all_actions.append(tr.action)

        self.optimizer.zero_grad()
        final_nodes = [value_loss] + all_actions

        # print("Total_loss: {0:.4}".format(value_loss.data[0]))

        gradients = [torch.ones(1).type_as(r)] + [None] * len(
                saved_transitions)
        torch.autograd.backward(final_nodes, gradients)

        self.optimizer.step()

        self.step_cnt = 0
        self.batch_games = 0
        del self.saved_transitions[:]

    def _frame2torch(self, obs):
        return torch.from_numpy(obs).unsqueeze(0).unsqueeze(0)

    def _batch2torch(self, batch, batch_sz=None):
        """ List of Transitions of batches to tensors batch states, actions,
            rewards.
        """

        batch = Transition(*zip(*batch))
        state_batch = torch.cat(batch.state, 0).float()

        # ONLY NON Terminal next states
        next_state_batch = [x for x in batch.state_ if x is not None]
        next_state_batch = torch.cat(next_state_batch, 0).float()

        action_batch = torch.cat(batch.action, 0)
        reward_batch = torch.cat(batch.reward, 0).float()

        mask = torch.cat(batch.done, 0)
        mask = (1 - mask).nonzero().view(-1)

        batch_sz = action_batch.size(0)

        return [batch_sz, state_batch, action_batch, reward_batch,
                next_state_batch, mask]
