""" Neural Network architecture for low-dimensional games.
"""
import torch.nn as nn
import torch.nn.functional as F

from models.utils import conv_out_dim

class BetaBinaryBoardModel(nn.Module):
    """ PigChase Model for the 18BinaryView batch x 18 x 9 x 9.

    Args:
        state_dim (tuple): input dims: (channels, width, history length)
        action_no (int): no of actions
        hidden_size (int): size of the hidden linear layer
    """

    def __init__(self, config):
        state_dim = (18, 9, 2)
        action_no = 3
        hidden_size = 64

        super(BetaBinaryBoardModel, self).__init__()

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
        self.head = nn.Linear(self.hidden_size, action_no)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.lin1(x.view(x.size(0), -1)))
        return self.head(x.view(x.size(0), -1))

    def get_attributes(self):
        return (self.input_channels, self.hist_len, self.action_no,
                self.hidden_size)


if __name__ == "__main__":
    import torch
    from torch.autograd import Variable

    # model = BinaryBoardModel(state_dim=(18, 9, 1), action_no=3)
    # print(model)
    # print(model(Variable(torch.rand(1, 18*1, 9, 9))))
    #
    # model = BinaryBoardModel(state_dim=(18, 9, 4), action_no=3)
    # print(model)
    # print(model(Variable(torch.rand(1, 18*4, 9, 9))))
    #
    # model = BinaryBoardModel(state_dim=(18, 9, 4), action_no=3, hidden_size=256)
    # print(model)
    # print(model(Variable(torch.rand(1, 18*4, 9, 9))))
