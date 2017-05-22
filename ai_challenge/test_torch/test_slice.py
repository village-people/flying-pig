import torch
import time
from torch.autograd import Variable
from torch.nn import Parameter

B = 10024
x = torch.randn(B, 18, 9, 9).cuda()
done = torch.LongTensor(B).random_(0, 2).byte().cuda()

print(x)
print(done)

t1, t2, t3 = .0, .0, .0

for j in range(20):
    x1 = Parameter(x.clone())
    x2 = Parameter(x.clone())
    x3 = Parameter(x.clone())

    torch.cuda.synchronize()

    a = time.time()

    _non_zero = done.nonzero()
    done2 = Variable(_non_zero.unsqueeze(2).unsqueeze(2) \
                     .expand(_non_zero.size(0), 18, 9, 9))
    y2 = x2.gather(0, done2)
    y2.backward(y2.data.clone().fill_(-1))

    torch.cuda.synchronize()
    b = time.time()

    done1 = Variable(done.unsqueeze(1).unsqueeze(1).unsqueeze(1) \
                     .expand_as(x1))
    y1 = x1.masked_select(done1).view(-1, 18, 9, 9)
    y1.backward(y1.data.clone().fill_(-1))

    torch.cuda.synchronize()
    c = time.time()

    _non_zero_3 = Variable(done.nonzero()).view(-1)
    y3 = x3.index_select(0, _non_zero_3)
    y3.backward(y3.data.clone().fill_(-1))

    torch.cuda.synchronize()
    d = time.time()

    n1 = (y1-y2).le(0.001).long().sum().data[0]
    n2 = (y2-y3).le(0.001).long().sum().data[0]
    n3 = (y3-y1).le(0.001).long().sum().data[0]

    g1, g2, g3 = x1.grad.data, x2.grad.data, x3.grad.data

    m1 = (g1-g2).le(0.001).long().sum()
    m2 = (g2-g3).le(0.001).long().sum()
    m3 = (g3-g1).le(0.001).long().sum()


    print(n1, n2, n3, m1, m2, m3)

    t1 += b-a
    t2 += c-b
    t3 += d-c


print(t1, t2, t3)

"""
print((b-a) / (c-b))
print("Done!")
print((y1-y2).le(0.001).long().sum().data[0])
print(_non_zero.nelement() * 18 * 9 * 9)
"""
