# Village People, 2017

# This class is used to train agents with recurrent models.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from agents import ReportingAgent
from models import get_model
from methods import get_batch_schedule
from utils.torch_types import TorchTypes


class EpisodicRDQNAgent(ReportingAgent):
    def __init__(self, name, action_space, cfg, shared_objects={}):
        super(EpisodicRDQNAgent, self).__init__()

        self.name = name
        self.actions_no = len(action_space)
        self.batch_size = batch_size = cfg.general.batch_size
        self.cuda = cfg.general.use_cuda
        self.dtype = TorchTypes(self.cuda)

        if "model" in shared_objects:
            self.net = net = shared_objects["model"]
            self.print_info("Training some shared model.")
        else:
            self.net = net = get_model(cfg.model.name)(cfg.model)
            if self.cuda:
                net.cuda()

        Optimizer = getattr(optim, cfg.training.algorithm)
        optim_args = vars(cfg.training.algorithm_args)
        self.optimizer = Optimizer(net.parameters(), **optim_args)

        self.losses = []

        self.exploration_strategy = get_batch_schedule(cfg.agent.exploration,
                                                       batch_size)

        self.last_q = None
        self.last_a = None
        self.last_aux = None

        self.live_idx = torch.linspace(0, batch_size-1, batch_size) \
                             .type(self.dtype.LongTensor)
        self.prev_state = {}
        self.crt_step = -1

        self.loss_coeff = {k: v for [k,v] in cfg.model.auxiliary_tasks}


    def get_model(self):
        return self.net

    def predict(self, obs, done):
        if done.all():
            self.prev_state = {}
        else:
            not_done = (1 - done)
            not_done_idx = not_done.nonzero().view(-1)
            not_done_idx_var = Variable(not_done_idx)
            prev_state = self.prev_state
            prev_state = {
                task: [s.index_select(0, not_done_idx_var) for s in ss] \
                for task, ss in prev_state.items()
            }


    def act(self, obs, reward, done, is_training):

        long_type = self.dtype.LongTensor
        batch_size = self.batch_size

        not_done = (1 - done)
        not_all_done = not_done.byte().any()
        not_done_idx = not_done.nonzero().view(-1)
        not_done_idx_var = Variable(not_done_idx)

        # -- Prepare previous state
        prev_state = self.prev_state
        if prev_state is not None and not_all_done:
            prev_state = \
                    {task: [s.index_select(0, not_done_idx_var) for s in ss]
                     for task, ss in prev_state.items()}
        else:
            prev_state = {}
        live_obs = obs.index_select(0, not_done_idx) if not_all_done else None

        if not_all_done:
            self.live_idx = live_idx = \
                        self.live_idx.index_select(0, not_done_idx)
            self.crt_step += 1

            # TODO: de scos float-ul daca se trimite direct din mediu asa
            qs, aux, h = self.net(Variable(live_obs.float()), prev_state)
            self.prev_state = h

            _, actions = qs.data.max(1)
            actions.squeeze_(1)

            epsilon = next(self.exploration_strategy)
            rand_idx = torch.bernoulli(torch.FloatTensor(epsilon)) \
                            .type(self.dtype.LongTensor) \
                            .index_select(0, live_idx) \
                            .nonzero() \
                            .squeeze(1)
            n = rand_idx.nelement()
            if n > 0 and is_training:
                rand_actions = torch.LongTensor(n).random_(0, self.actions_no)
                rand_idx = rand_idx.type(long_type)
                rand_actions = rand_actions.type(long_type)
                actions.index_fill_(0, rand_idx, 0)
                actions.index_add_(0, rand_idx, rand_actions)
            actions = actions.type(long_type)
        else:
            qs = None
            actions = None
            aux = None
            self.crt_step = -1
            self.live_idx = torch.linspace(0, batch_size-1, batch_size) \
                                 .type(long_type)
            self.prev_state = {}

        if self.last_q is not None and is_training:
            reward = reward.clone()
            last_q, last_a, last_aux = self.last_q, self.last_a, self.last_aux
            self._improve_policy(last_q, last_aux,
                                 last_a, reward,
                                 qs, aux,
                                 not_done_idx, done)

        self.last_q = qs if not_all_done else None
        self.last_aux = aux if not_all_done else None
        self.last_a = actions

        full_size_actions = torch.LongTensor(reward.size(0)).type(long_type)
        full_size_actions.fill_(3) # TODO: some variable

        if actions is not None:
            full_size_actions.scatter_(0, not_done_idx, actions)
        return full_size_actions


    def _improve_policy(self, q, aux, a, r, next_q, next_aux,
                        not_done_idx, done):

        optimizer = self.optimizer
        losses = self.losses

        reward = Variable(r.float(), requires_grad=False)

        if not_done_idx.nelement() > 0:
            q_target, _ = next_q.max(1)
            q_target = q_target.squeeze(1)
            q_target.detach_()
            target = reward.index_add(0, Variable(not_done_idx), q_target)
        else:
            target = reward
        target.detach_()

        _q = q.gather(1, Variable(a.unsqueeze(1)))
        losses.append(F.smooth_l1_loss(_q, target))

        loss_coeff = self.loss_coeff

        if aux is not None and "time_step" in aux and self.crt_step >= 0:
            pred_crt_step = aux["time_step"]
            alive_no = pred_crt_step.size(0)
            losses.append(nn.CrossEntropyLoss()(
                pred_crt_step,
                Variable(torch.LongTensor(alive_no).fill_(self.crt_step-1) \
                         .type(self.dtype.LongTensor))
            ) * loss_coeff["time_step"])

        if aux is not None and "game_ends" in aux:
            losses.append(nn.CrossEntropyLoss()(
                aux["game_ends"],
                Variable(done)
            ) * loss_coeff["game_ends"])

        if not_done_idx.nelement() == 0:
            # Backward
            total_loss = sum(losses)
            total_loss.data.clamp_(-1.0, 1.0)
            print("losss=", total_loss.data[0])
            total_loss.backward()

            #for param in self.net.parameters():
            #    param.grad.data.clamp_(-1, 1)

            self.get_model_stats()

            # Step in optimizer
            optimizer.step()

            # Reset losses
            optimizer.zero_grad()
            losses.clear()

    def get_model_stats(self):
        param_abs_mean = 0
        grad_abs_mean = 0
        grad_abs_max = None
        n_params = 0
        for p in self.net.parameters():
            param_abs_mean += p.data.abs().sum()
            if p.grad:
                grad_abs_mean  += p.grad.data.abs().sum()
                if grad_abs_max is None:
                    grad_abs_max = p.grad.data.abs().max()
                grad_abs_max = max(grad_abs_max, p.grad.data.abs().max())
                n_params += p.data.nelement()

        self.print_info("W_avg: %.9f | G_avg: %.9f | G_max: %.9f" % (
            param_abs_mean / n_params,
            grad_abs_mean / n_params,
            grad_abs_max)
        )
