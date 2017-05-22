import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable

_x = torch.randn(10, 5)
f = nn.Sequential(nn.Linear(5, 5), nn.Softmax())

f.zero_grad()
y1 = f(Variable(_x.clone()))

a = torch.multinomial(y1)

a.reinforce(torch.FloatTensor(10, 1).fill_(-2.3))
autograd.backward([a], [None])

for p in f.parameters():
    print(p.grad.data)

f.zero_grad()
y2 = f(Variable(_x.clone()))

grad = torch.FloatTensor(10, 5).fill_(0)
grad.scatter_(1, a.data, -2.3)
grad /= -y2.data
y2.backward(grad)

for p in f.parameters():
    print(p.grad.data)

f.zero_grad()
y3 = f(Variable(_x.clone()))
z = -torch.log(y3)

grad = torch.FloatTensor(10, 5).fill_(0)
grad.scatter_(1, a.data, -2.3)
z.backward(grad)

for p in f.parameters():
    print(p.grad.data)
