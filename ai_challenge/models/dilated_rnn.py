import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Parameter
from torch.autograd import Variable

class DilatedLSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size, clock=10, bias=True):
        super(DilatedLSTMCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        self.weight_ih, self.weight_hh = {}, {}
        if bias:
            self.bias_ih, self.bias_hh = {}, {}

        q, r = hidden_size // clock, hidden_size % clock
        hidden_sizes = [q + (1 if j < r else 0) for j in range(clock)]
        self.hidden_sizes = hidden_sizes
        self.slice_idx = slice_idx = []
        n = 0
        for j, hidden_size_j in enumerate(hidden_sizes):
            slice_idx.append((n, n + hidden_size_j))
            n += hidden_size_j

            input_size_j = input_size + hidden_size - hidden_size_j

            w_ih_j = Parameter(torch.Tensor(4 * hidden_size_j, input_size_j))
            w_hh_j = Parameter(torch.Tensor(4 * hidden_size_j, hidden_size_j))
            setattr(self, 'weight_ih_{:d}'.format(j), w_ih_j)
            setattr(self, 'weight_hh_{:d}'.format(j), w_hh_j)
            self.weight_ih[j] = w_ih_j
            self.weight_hh[j] = w_hh_j

            if bias:
                bias_ih_j = Parameter(torch.Tensor(4 * hidden_size_j))
                bias_hh_j = Parameter(torch.Tensor(4 * hidden_size_j))
                setattr(self, 'bias_ih_{:d}'.format(j), bias_ih_j)
                setattr(self, 'bias_hh_{:d}'.format(j), bias_hh_j)
                self.bias_ih[j] = bias_ih_j
                self.bias_hh[j] = bias_hh_j

        self.reset_parameters()
        self.clock = clock

    def reset_parameters(self):
        import math
        stdv = 1.0 / math.sqrt(self.hidden_sizes[0])
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x, hx, clock):
        j = clock % self.clock
        (start, end) = self.slice_idx[j]
        hidden_size = self.hidden_size

        (hx, cx) = hx
        h_j = hx.narrow(1, start, end-start)
        c_j = cx.narrow(1, start, end-start)

        inputs = [x]
        if start > 0:
            inputs.append(hx.narrow(1, 0, start))
        if end < hidden_size:
            inputs.append(hx.narrow(1, end, hidden_size - end))
        input_j = torch.cat(inputs, 1)

        h_j, c_j = self._backend.LSTMCell(
            input_j, (h_j, c_j),
            self.weight_ih[j], self.weight_hh[j],
            self.bias_ih[j], self.bias_hh[j],
        )

        hs, cs = [], []
        if start > 0:
            hs.append(hx.narrow(1, 0, start))
            cs.append(cx.narrow(1, 0, start))
        hs.append(h_j)
        cs.append(c_j)
        if end < hidden_size:
            hs.append(hx.narrow(1, end, hidden_size - end))
            cs.append(cx.narrow(1, end, hidden_size - end))


        return torch.cat(hs, 1), torch.cat(cs, 1)

if __name__ == "__main__":

    dlstm = DilatedLSTMCell(16, 32, 8)

    h, c = Variable(torch.zeros(10, 32)), Variable(torch.zeros(10, 32))
    for clock in range(100):
        x = Variable(torch.randn(10, 16))
        h, c = dlstm(x, (h, c), clock)
