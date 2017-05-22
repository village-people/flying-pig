import torch

B = 10

action_holder = torch.LongTensor(B)

not_done_idx = torch.LongTensor(B).random_(0,2).nonzero().view(-1)
print(not_done_idx)

action_holder.fill_(3)

actions = torch.LongTensor(not_done_idx.size(0)).random_(0, 3)

action_holder.index_fill_(0, not_done_idx, actions)

print(action_holder)
