import torch
import torch.nn as nn
from torch.autograd import Variable


ACTIONS_NO = 10
BATCH_SIZE = 13

Q = Variable(torch.randn(BATCH_SIZE, ACTIONS_NO), requires_grad=True)
A = Variable(torch.LongTensor(BATCH_SIZE).random_(ACTIONS_NO))

T = Variable(torch.randn(BATCH_SIZE))

_Q = Q.gather(1, A.unsqueeze(1))
