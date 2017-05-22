import torch
from torch.autograd import Variable

reward = Variable(torch.Tensor(10).random_(0, 3))
print(reward)

done = torch.LongTensor(10).random_(0, 2).byte()
idx = Variable(done.nonzero().view(-1))

print(idx)

q_next = Variable(torch.randn(idx.size(0))).clone()
q_next.detach_()
print(q_next)
print(reward.index_add(0, idx, q_next))
