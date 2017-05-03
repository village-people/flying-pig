""" Asynchronous Reinforcement Learning Framework.

    Neural Network architecture for games in PyGame Learning Environment.
"""

import torch.nn as nn
import torch.nn.functional as F


class MiniNet(nn.Module):
    """
    conv -> pool -> fc -> out
    """

    def __init__(self, input_channels, hist_len, action_no, hidden_size=16):
        super(MiniNet, self).__init__()
        self.input_channels = input_channels
        self.hist_len = hist_len
        self.action_no = action_no
        self.hidden_size = hidden_size
        self.input_depth = hist_len * input_channels

        self.conv = nn.Conv2d(self.input_depth, 4, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(3, 3)
        self.fc   = nn.Linear(256, self.hidden_size)
        self.out  = nn.Linear(self.hidden_size, action_no)

    def forward(self, x):
        x = F.relu(self.pool(self.conv(x)))
        return self.out(F.relu(self.fc(x.view(x.size(0), -1))))

    def get_attributes(self):
        return (self.input_channels, self.hist_len, self.action_no,
                self.hidden_size)
