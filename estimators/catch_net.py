""" Neural Network architecture for low-dimensional games.
"""

import torch.nn as nn
import torch.nn.functional as F


class CatchNet(nn.Module):
    def __init__(self, input_channels, hist_len, action_no, hidden_size=32):
        super(CatchNet, self).__init__()
        self.input_channels = input_channels
        self.hist_len = hist_len
        self.action_no = action_no
        self.hidden_size = hidden_size
        self.input_depth = hist_len * input_channels

        self.conv1 = nn.Conv2d(self.input_depth, 32, kernel_size=5,
                               stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.lin1 = nn.Linear(512, self.hidden_size)
        self.head = nn.Linear(self.hidden_size, action_no)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.lin1(x.view(x.size(0), -1)))
        return self.head(x.view(x.size(0), -1))

    def get_attributes(self):
        return (self.input_channels, self.hist_len, self.action_no,
                self.hidden_size)
