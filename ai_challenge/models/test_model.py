# Village People, 2017

import torch
import torch.nn as nn
import torch.nn.functional as F

class TestModel(nn.Module):

    def __init__(self, config):
        super(TestModel, self).__init__()
        self.c = nn.Conv2d(18, 1, (9,9))


    def forward(self, inputs):
        batch_size = inputs.size(0)
        inputs.view(batch_size, -1)

        return F.softmax(self.c(inputs))
