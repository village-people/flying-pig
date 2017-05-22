import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

from .dilated_rnn import DilatedLSTMCell
from .utils import conv_out_dim

class Percept(nn.Module):
    """Extracts features from the observation into a shared space"""

    def __init__(self, cfg):
        super(Percept, self).__init__()
        state_dim = (18, 9, cfg.hist_len)
        action_no = 3
        hidden_size = 64

        super(Percept, self).__init__()

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

        self.linear1 = nn.Linear(32 * map_width3**2, self.hidden_size)
        self.linear2 = nn.Linear(64, cfg.percept_size)

    def forward(self, x_t):
        batch_size = x_t.size(0)
        x_t = F.relu(self.bn1(self.conv1(x_t)))
        x_t = F.relu(self.bn2(self.conv2(x_t)))
        x_t = F.relu(self.bn3(self.conv3(x_t)))
        x_t = F.relu(self.linear1(x_t.view(batch_size, -1)))
        z_t = F.relu(self.linear2(x_t))
        return z_t

class Mspace(nn.Module):
    """This module projects the perception into Manager's latent space"""

    def __init__(self, cfg):
        super(Mspace, self).__init__()
        self.percept_size = percept_size = cfg.percept_size
        self.latent_size = latent_size = cfg.latent_size
        self.transform = nn.Linear(percept_size, latent_size)

    def forward(self, z_t):
        s_t = F.relu(self.transform(z_t))
        return s_t

class Mrnn(nn.Module):
    """Manager's recurrent module (a Dilated LSTM)"""

    def __init__(self, cfg):
        super(Mrnn, self).__init__()
        self.latent_size = latent_size = cfg.latent_size
        self.goal_size = goal_size = cfg.goal_size
        self.transform = DilatedLSTMCell(latent_size, goal_size, cfg.c)
        self.clock = 0

    def forward(self, s_t, hM_t_1=None, clock=None):
        batch_size = s_t.size(0)
        clock = self.clock if clock is None else clock
        self.clock = clock + 1

        if hM_t_1 is None:
            goal_size = self.goal_size
            h = Variable(torch.zeros(batch_size, goal_size).type_as(s_t.data))
            c = Variable(torch.zeros(batch_size, goal_size).type_as(s_t.data))
            hM_t_1 = (h, c)

        g_hat_t, c_t = self.transform(s_t, hM_t_1, clock)
        hM_t = (g_hat_t, c_t)
        return g_hat_t, hM_t

class Wrnn(nn.Module):
    def __init__(self, cfg):
        super(Wrnn, self).__init__()
        self.percept_size = percept_size = cfg.percept_size
        self.actions_no = actions_no = cfg.actions_no
        self.emb_size = emb_size = cfg.emb_size
        self.embedding = nn.LSTMCell(percept_size, actions_no * emb_size)

    def forward(self, z_t, hW_t_1=None):
        batch_size = z_t.size(0)
        actions_no, emb_size = self.actions_no, self.emb_size
        if hW_t_1 is None:
            out_size = actions_no * emb_size
            h = Variable(torch.zeros(batch_size, out_size).type_as(z_t.data))
            c = Variable(torch.zeros(batch_size, out_size).type_as(z_t.data))
            hW_t_1 = (h, c)
        U_t, c_t = self.embedding(z_t, hW_t_1)
        hW_t = (U_t, c_t)
        U_t = U_t.view(batch_size, actions_no, emb_size)
        return U_t, hW_t

class Phi(nn.Module):
    def __init__(self, cfg):
        super(Phi, self).__init__()
        self.goal_size = goal_size = cfg.goal_size
        self.emb_size = emb_size = cfg.emb_size
        self.transform = nn.Linear(goal_size, emb_size, bias=False)

    def forward(self, goals):
        return self.transform(goals)

class MValue(nn.Module):
    def __init__(self, cfg):
        super(MValue, self).__init__()
        self.latent_size = latent_size = cfg.latent_size
        self.V = nn.Linear(latent_size, 1)

    def forward(self, s_t):
        return self.V(s_t)

class WValue(nn.Module):
    def __init__(self, cfg):
        super(WValue, self).__init__()
        self.actions_no = actions_no = cfg.actions_no
        self.emb_size = emb_size = cfg.emb_size
        self.percept_size = percept_size = cfg.percept_size
        self.V = nn.Linear(actions_no * emb_size + percept_size, 1)

    def forward(self, z_t, U_t):
        batch_size = z_t.size(0)
        full_emb_size = self.actions_no * self.emb_size
        return self.V(torch.cat([z_t, U_t.view(batch_size, full_emb_size)], 1))


class FuN(nn.Module):
    def __init__(self, cfg):
        super(FuN, self).__init__()
        self.c = cfg.c
        self.actions_no = actions_no = cfg.actions_no = 4
        self.emb_size = emb_size = cfg.emb_size
        self.shortcut = shortcut = cfg.shortcut

        self.percept = Percept(cfg)
        self.m_space = Mspace(cfg)
        self.m_rnn = Mrnn(cfg)
        self.w_rnn = Wrnn(cfg)
        if shortcut:
            self.direct_link = nn.Linear(actions_no * emb_size, actions_no)
        else:
            self.phi = Phi(cfg)
        self.m_value = MValue(cfg)
        self.w_value = WValue(cfg)




    def forward(self, x_t, clock, goals=None, prev_state=None):

        if prev_state is None:
            hW_t_1 = None
            hM_t_1 = None
        else:
            hM_t_1 = (prev_state[0], prev_state[1])
            hW_t_1 = (prev_state[2], prev_state[3])

        goals = [] if goals is None else goals

        # -- Manager

        z_t = self.percept(x_t)
        s_t = self.m_space(z_t)
        g_hat_t, hM_t = self.m_rnn(s_t, clock=clock, hM_t_1=hM_t_1)
        g_t = g_hat_t #/ torch.norm(g_hat_t, 1, 1, True).expand_as(g_hat_t)
        goals.append(g_t)

        U_t, hW_t = self.w_rnn(z_t, hW_t_1=hW_t_1)
        if self.shortcut:
            batch_size = U_t.size(0)
            prod_t = self.direct_link(U_t.view(batch_size, -1))
        else:
            w_t = self.phi(sum(goals[-self.c:]))
            prod_t = torch.bmm(U_t, w_t.unsqueeze(2)).squeeze(2)

        pi_t = F.softmax(prod_t)

        Vm_t = self.m_value(s_t)
        Vw_t = self.w_value(z_t, U_t)

        return pi_t, s_t, goals, Vm_t, Vw_t, list(hM_t) + list(hW_t)
