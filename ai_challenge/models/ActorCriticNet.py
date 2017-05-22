# Village People, 2017

# This is the model we got our best results with.
#
# The model receives the current state, and predicts the policy, the value
# of the state and various other outputs for the auxiliary task.
#
# The model uses BatchNormalization in the first stages of training

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from models.utils import conv_out_dim


ENV_CAUGHT_REWARD = 25

class NextRewardPredictor(nn.Module):
    """This model is used for the prediction of the next reward."""
    def __init__(self, in_size):
        super(NextRewardPredictor, self).__init__()
        self.predictor = nn.Linear(in_size, 1)

    def forward(self, x):
        x = nn.Tanh()(self.predictor(x))
        return x


class NextStateDepthPrediction(nn.Module):
    """This model """
    def __init__(self, in_size, out_size):
        super(NextStateDepthPrediction, self).__init__()

        self.act = nn.ReLU()

        inter = int(in_size / 2)
        self.predictor1 = nn.Linear(in_size, inter)
        self.bn1 = nn.BatchNorm1d(inter)
        self.predictor2 = nn.Linear(inter, out_size)

    def forward(self, x):
        act = self.act
        x = act(self.predictor1(x))
        x = nn.Sigmoid()(self.predictor2(x))
        return x


class PredictNextState(nn.Module):
    def __init__(self, in_size):
        super(PredictNextState, self).__init__()

        self.act = nn.ReLU(True)
        self.in_size = in_size
        ngf = 64
        nc = 15
        self.first_cond_w = first_cond_w = 3
        first_conv_depth = int(in_size // (first_cond_w ** 2))
        cn_size = first_conv_depth * first_cond_w * first_cond_w
        self.lin1 = nn.Linear(in_size, cn_size)
        self.bnLin = nn.BatchNorm1d(cn_size)

        self.cn2 = nn.ConvTranspose2d(first_conv_depth, 64, kernel_size=3,
                                      stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.cn3 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1,
                                      bias=False)
        self.bn3 = nn.BatchNorm2d(32)
        self.cn4 = nn.ConvTranspose2d(32, nc, kernel_size=3, stride=1,
                                      bias=False)
        self.bn4 = nn.BatchNorm2d(nc)

    def forward(self, x):
        act = self.act
        x = act(self.bnLin(self.lin1(x)))
        x = x.view(x.size(0), -1, self.first_cond_w, self.first_cond_w)
        x = act(self.bn2(self.cn2(x)))
        x = act(self.bn3(self.cn3(x)))
        x = self.cn4(x)
        x = nn.Sigmoid()(x)

        return x.view(-1, 15, 9, 9)


def sampler(input_, tau=100):
    noise = Variable(torch.randn(input_.size(0), input_.size(1))
                     .type_as(input_.data)) / tau

    x = input_ + noise
    return x


class Policy(nn.Module):
    """ PigChase Model for the 18BinaryView batch x 18 x 9 x 9.

    Args:
        state_dim (tuple): input dims: (channels, width, history length)
        action_no (int): no of actions
        hidden_size (int): size of the hidden linear layer
    """

    def __init__(self, config):
        super(Policy, self).__init__()

        state_dim = (18, 9, 1)
        action_no = 3
        hidden_size = hidden_size = 256
        dropout = 0.1

        self.rnn_type = rnn_type = "GRUCell"
        self.rnn_layers = rnn_layers = 2
        self.rnn_nhid = rnn_nhid = hidden_size

        self.activation = nn.ReLU()

        self.drop = nn.Dropout(dropout)
        self.drop2d = nn.Dropout2d(p=dropout)

        self.in_channels, self.in_width, self.hist_len = state_dim
        self.action_no = action_no
        self.hidden_size = hidden_size
        in_depth = self.hist_len * self.in_channels

        self.conv0 = nn.Conv2d(in_depth, 64, kernel_size=1, stride=1)
        self.bn0 = nn.BatchNorm2d(64)
        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)

        map_width1 = conv_out_dim(self.in_width, self.conv1)
        map_width2 = conv_out_dim(map_width1, self.conv2)
        map_width3 = conv_out_dim(map_width2, self.conv3)
        # map_width4 = conv_out_dim(map_width3, self.conv4)

        lin_size = 64 * map_width3 ** 2

        # self.lin1 = nn.Linear(lin_size, lin_size)
        # self.bnLin1 = nn.BatchNorm1d(lin_size)

        self.rnn1 = getattr(nn, rnn_type)(lin_size, rnn_nhid)
        self.rnn2 = getattr(nn, rnn_type)(rnn_nhid, rnn_nhid)

        self.bnRnn1 = nn.BatchNorm1d(rnn_nhid)
        self.bnRnn2 = nn.BatchNorm1d(rnn_nhid * self.rnn_layers)

        self.lin2 = nn.Linear(rnn_nhid * self.rnn_layers, hidden_size)
        self.bnLin2 = nn.BatchNorm1d(hidden_size)

        lin_size_3 = hidden_size
        self.lin3 = nn.Linear(hidden_size, lin_size_3)
        self.bnLin3 = nn.BatchNorm1d(lin_size_3)

        self.action_head = nn.Linear(lin_size_3, action_no)
        self.value_head = nn.Linear(lin_size_3, 1)
        self.bn_value_head = nn.BatchNorm1d(1)

        # ---- Aux tasks

        self.aux_predictors = []

        if "noise" in [t[0] for t in config.auxiliary_tasks]:
            self.action_noise = 10000
        #
        if "next_reward" in [t[0] for t in config.auxiliary_tasks]:
            _input_size = rnn_nhid * self.rnn_layers
            self.next_reward = NextRewardPredictor(_input_size)
            self.aux_predictors.append(("next_reward", self.next_reward))

        if "next_state_depth" in [t[0] for t in config.auxiliary_tasks]:
            _input_size = rnn_nhid * self.rnn_layers + state_dim[0]
            self.next_state_depth = NextStateDepthPrediction(_input_size,
                                                             state_dim[1] ** 2)
            self.aux_predictors.append(("next_state_depth",
                                        self.next_state_depth))

    def forward(self, x, hidden_states, bn=True, aux_input={}):
        act = self.activation

        # if bn:

        x = act(self.bn0(self.conv0(x)))
        x = act(self.bn1(self.conv1(x)))
        x = act(self.bn2(self.conv2(x)))
        x = act(self.bn3(self.conv3(x)))

        out_conv = x.view(x.size(0), -1)

        hidden0 = self.rnn1(out_conv, hidden_states[0])
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

        aux_predictions = {}
        if self.training:
            for (task, predictor) in self.aux_predictors:
                if task == "next_reward":
                    aux_predictions[task] = predictor(x)
                if task == "next_state_depth" and "next_state_depth" in aux_input:
                    in_next = torch.cat([aux_input["next_state_depth"], x], 1)
                    aux_predictions[task] = predictor(in_next)

        x = act(self.lin2(x))
        x = act(self.lin3(x))

        action_scores = self.action_head(x)
        state_values = self.value_head(x)

        action_prob = sampler(F.softmax(action_scores), tau=self.action_noise)
        return action_prob, state_values, all_hidden, aux_predictions

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
                                  hidden_state[i][1].index_select(0, Variable(
                                      not_done_idx))))
        else:
            hidden_states.append(hidden_state[i]
                                 .index_select(0, Variable(not_done_idx)))
        i = 1
        if self.rnn_type == 'LSTMCell':
            hidden_states.append((hidden_state[i][0]
                                  .index_select(0, Variable(not_done_idx)),
                                  hidden_state[i][1].index_select(0, Variable(
                                      not_done_idx))))
        else:
            hidden_states.append(hidden_state[i]
                                 .index_select(0, Variable(not_done_idx)))
        return hidden_states

    def get_attributes(self):
        return (self.input_channels, self.hist_len, self.action_no,
                self.hidden_size)
