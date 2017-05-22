import torch
from methods import get_batch_schedule

partition = [
    ["constant", 0.2, [.5]],
    ["constant", 0.2, [0.1]],
    ["linear",   0.6, [1.0, 0.005, 100]]
]

sch = get_batch_schedule(partition, 20)

for x in sch:
    print(x)
    print(torch.bernoulli(x).long().nonzero().squeeze(1))
    input("Push!")
