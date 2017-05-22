# Village People, 2017

# This file contains the following classes:
#
# RecurrentQEstimator   - used to estimate the Q values for all valid actions
# TimeStepPredictor     - used to estimate the current time step in the episode
# NextStatePredictor    - used to estimate the next state
#
# All models assume that in the first step you send None for the hidden state.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from models.utils import conv_out_dim


class TimeStepPredictor(nn.Module):

    def __init__(self, in_size, time_steps):
        super(TimeStepPredictor, self).__init__()
        self.predictor = nn.Linear(in_size, time_steps)
        self.time_steps = time_steps

    def forward(self, x, prev_state=None):
        return self.predictor(x), [] #[h, c]

class GameEndsPredictor(nn.Module):

    def __init__(self, in_size):
        super(GameEndsPredictor, self).__init__()
        self.predictor = nn.Linear(in_size, 2)

    def forward(self, x, prev_state=None):
        return self.predictor(x), []

class HowManyStepsLeft(nn.Module):

    def __init__(self, in_size):
        super(HowManyStepsLeft, self).__init__()
        self.predictor = nn.Linear(in_size, 1)

    def forward(self, x, prev_state=None):
        return self.predictor(x), [] #[h, c]


class RecurrentQEstimator(nn.Module):
    """ Recurrent model Model for the 18BinaryView batch x 18 x 9 x 9.
    """

    def __init__(self, config):
        super(RecurrentQEstimator, self).__init__()
        state_dim = (18, 9, 1)
        action_no = 3
        hidden_size = 128

        self.in_channels, self.in_width, self.hist_len = state_dim
        self.action_no = action_no
        self.hidden_size = hidden_size
        in_depth = self.hist_len * self.in_channels

        self.conv1 = nn.Conv2d(in_depth, 64, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(3)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)

        # Other shit:
        # self.bn1 = nn.BatchNorm2d(32)
        # self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1)
        # self.bn2 = nn.BatchNorm2d(32)
        # self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1)
        # self.bn3 = nn.BatchNorm2d(32)

        # map_width1 = conv_out_dim(self.in_width, self.conv1)
        # map_width2 = conv_out_dim(map_width1, self.conv2)
        # map_width3 = conv_out_dim(map_width2, self.conv3)

        self.rnn1 = nn.LSTMCell(128 * 9, self.hidden_size)
        self.rnn2 = nn.LSTMCell(hidden_size, self.hidden_size)

        self.head = nn.Linear(self.hidden_size, action_no)

        self.last_batch_size = None

        self.aux_predictors = []
        if "time_step" in [t[0] for t in config.auxiliary_tasks]:
            _input_size = self.hidden_size  * 2
            self.time_step_predictor = TimeStepPredictor(_input_size, 25)
            self.aux_predictors.append(("time_step", self.time_step_predictor))

        if "game_ends" in [t[0] for t in config.auxiliary_tasks]:
            _input_size = self.hidden_size  * 2
            self.game_ends_predictor = GameEndsPredictor(_input_size)
            self.aux_predictors.append(("game_ends", self.game_ends_predictor))


    def get_zero_state(self, batch_size, t):
        if self.last_batch_size is None or self.last_batch_size != batch_size:
            self.last_batch_size = batch_size
            self.zero_h1 = torch.zeros(batch_size, self.hidden_size).type_as(t)
            self.zero_c1 = torch.zeros(batch_size, self.hidden_size).type_as(t)
            self.zero_h2 = torch.zeros(batch_size, self.hidden_size).type_as(t)
            self.zero_c2 = torch.zeros(batch_size, self.hidden_size).type_as(t)
        return [self.zero_h1, self.zero_c1, self.zero_h2, self.zero_c2]

    def forward(self, x, full_prev_state={}):

        batch_size = x.size(0)

        # -- Get previous state

        prev_state = full_prev_state.get("main", None)
        if prev_state is None:
            prev_state = self.get_zero_state(batch_size, x.data)
            prev_state = [Variable(s) for s in prev_state]

        rnn1_prev_state = (prev_state[0], prev_state[1])
        rnn2_prev_state = (prev_state[2], prev_state[3])

        crt_state = {}

        # -- Forward

        x = F.relu(self.conv1(x)) # x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x) # x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.conv2(x)) # x = F.relu(self.bn3(self.conv3(x)))

        h1, c1 = self.rnn1(x.view(batch_size, -1), rnn1_prev_state)
        h2, c2 = self.rnn2(h1, rnn2_prev_state)

        crt_state["main"] = [h1, c1, h2, c2]

        h = torch.cat([h1, h2], 1)

        # -- Auxiliary tasks

        aux_predictions = {}
        if self.training:
            for (task, predictor) in self.aux_predictors:
                prev_state = full_prev_state.get(task, None)
                y, new_state = predictor(h, prev_state)
                crt_state[task] = new_state
                aux_predictions[task] = y

        return self.head(F.relu(h2)), aux_predictions, crt_state
