# Village People, 2017

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np

from .agent import ReportingAgent
from models import get_model
from utils.torch_types import TorchTypes
from collections import namedtuple
import time

# from termcolor import colored as clr
ENV_CAUGHT_REWARD = 25

SavedTransition = namedtuple('Transition',
                             ['obs', 'action', 'value', 'other_task',
                              'reward', 'next_obs', 'playing_idx', 'live_idx',
                              'probs'])


def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad


class Village_ActorCritic_Aux(ReportingAgent):
    def __init__(self, name, action_space, cfg, shared_objects={}):
        # super(Village_ActorCritic_Aux, self).__init__()
        ReportingAgent.__init__(self, cfg)

        self.name = name
        self.pidx = 0
        self.actions_no = len(action_space)
        self.batch_size = batch_size = cfg.general.batch_size
        self.update_freq = batch_size
        self.hist_len = 1
        self.cuda = cfg.general.use_cuda
        self.torch_obs = True
        self.dtype = TorchTypes(self.cuda)
        self.clip_grad = 1
        self.lr = cfg.training.algorithm_args.lr
        self.gamma = cfg.training.gamma

        self.state_dim = (18, 9, 1)
        self.max_reward = 0

        self.id_ = np.random.randint(2)
        self.ep_training = 0

        if "model" in shared_objects:
            self.shared_model = shared_objects["model"]
            self.net = net = shared_objects["model"]
            self.print_info("Training some shared model.")
        else:
            self.net = net = get_model(cfg.model.name)(cfg.model)
            if self.cuda:
                net.cuda()

        self.model_utils.set_model(self.net)

        self.batch_games = 0
        self.step_cnt = 0

        self.saved_transitions = []

        self._o, self._a, self._v, self._probs = None, None, None, None

        self.live_idx = None

        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        self.optimizer.zero_grad()

        self.hidden_state = None
        self.last_time = time.time()
        self.full_hidden_state = self.net.init_hidden(128)

        self.predict_max = True
        self.clock = 0

        self.aux_tasks = [t[0] for t in cfg.model.auxiliary_tasks]
        self.reset()

    def get_models(self):
        return [self.net, self.target]

    def reset(self):
        self.predict_depth = None

    def batch_predict(self, ids, states, done):
        # self.net.eval()
        self.net.train()
        full_hidden_state = self.full_hidden_state

        # -- Put to zero previously finished games
        done_idx = done.nonzero()
        if done_idx.nelement() > 0:
            done_ids = ids.index_select(0, done_idx.squeeze(1))
            for ss in full_hidden_state:
                for s in list(ss):
                    s.data.index_fill_(0, done_ids, 0)
        # -- Select the relevant hidden states
        hidden_state = self.net.slice_hidden(full_hidden_state, ids)

        # -- Predict
        p, _, h, _ = self.net(Variable(states, volatile=True), hidden_state)

        # -- Put new hidden states in full hidden state
        for ss, ss_new in zip(full_hidden_state, h):
            if isinstance(ss, Variable):
                ss = [ss]
                ss_new = [ss_new]
            for s, s_new in zip(list(ss), list(ss_new)):
                s.data.index_fill_(0, ids, 0)
                s.data.index_add_(0, ids, s_new.data)
                s.detach_()

        # Sample actions
        actions = p.multinomial().data.squeeze(1)
        return actions

    def act(self, obs, reward, done, is_training, actions=None, **kwargs):
        if not is_training:
            self.net.eval()
            return self.predict(obs, reward, done, **kwargs)
        self.net.train()
        not_done = (1 - done)
        have_more_games = not_done.byte().any()
        hidden_state = self.hidden_state

        if self.hidden_state is None:
            assert self.clock == 0
            __BS = obs.size(0)
            hidden_state = self.hidden_state = self.net.init_hidden(__BS)
            self.live_idx = live_idx = torch.linspace(0, __BS - 1, __BS) \
                .type(self.dtype.LongTensor)
            if "next_state_depth" in self.aux_tasks:
                self.predict_depth = torch.zeros(self.state_dim[0]) \
                    .type(self.dtype.FloatTensor)
                predict_depth_idx = (torch.rand(1) * self.state_dim[0]).int()[0]
                self.predict_depth[predict_depth_idx] = 1
                self.predict_depth_idx = torch.LongTensor([predict_depth_idx]) \
                    .type(self.dtype.LongTensor)
                self.predict_depth = self.predict_depth.unsqueeze(0) \
                    .expand(__BS, self.state_dim[0])

        if have_more_games:
            not_done_idx = not_done.nonzero().view(-1)
            non_done_states = obs.index_select(0, not_done_idx)
            hidden_state = self.net.slice_hidden(hidden_state, not_done_idx)
        else:
            not_done_idx = torch.LongTensor(0).type_as(done)
            non_done_states = None

        action = None
        if have_more_games:
            state = non_done_states.float()

            # select Action
            aux_in = {}
            if "next_state_depth" in self.aux_tasks:
                self.predict_depth = self.predict_depth. \
                    index_select(0, not_done_idx)
                aux_in["next_state_depth"] = Variable(self.predict_depth)

            probs, state_value, new_hidden_state, other_task = \
                self.net(Variable(state), hidden_state,
                         aux_input=aux_in, **kwargs)

            if actions is None:
                action = probs.multinomial().data
            else:
                action = actions.unsqueeze(1).index_select(0, not_done_idx)

        if self._o is not None and is_training:
            self.saved_transitions.append(SavedTransition(
                self._o, self._a, self._v, self._other_task, reward.float(),
                obs.float(), not_done_idx, self.live_idx, self._probs))

        if have_more_games:
            self.live_idx = live_idx = \
                self.live_idx.index_select(0, not_done_idx)
            self._o = state
            self._a = action
            self._v = state_value
            self._probs = probs
            self._other_task = other_task
            self.hidden_state = new_hidden_state
            self.clock = self.clock + 1
        else:
            self.finish_episode()
            self.clock = 0
            self.hidden_state = None
            self.predict_depth = None

            self._o, self._a, self._v, self._probs = None, None, None, None
            batch_size = self.batch_size
            self.live_idx = torch.linspace(0, batch_size - 1, batch_size) \
                .type(self.dtype.LongTensor)

        self.step_cnt += 1

        # MUST RETURN SAME SIZE AS OBS SIZE
        action_holder = torch.LongTensor(reward.size(0)) \
            .type(self.dtype.LongTensor)
        action_holder.fill_(3)
        if action is not None:
            action_holder.scatter_(0, not_done_idx, action.view(-1).long())
        return action_holder

    def predict(self, obs, reward, done, **kwargs):
        not_done = (1 - done)
        have_more_games = not_done.byte().any()

        hidden_state = self.hidden_state

        if self.hidden_state is None:
            assert self.clock == 0
            __BS = obs.size(0)
            hidden_state = self.hidden_state = self.net.init_hidden(__BS)
            self.live_idx = live_idx = torch.linspace(0, __BS - 1, __BS) \
                .type(self.dtype.LongTensor)
            if "next_state_depth" in self.aux_tasks:
                self.predict_depth = torch.zeros(self.state_dim[0]) \
                    .type(self.dtype.FloatTensor)
                predict_depth_idx = (torch.rand(1) * self.state_dim[0]).int()[0]
                self.predict_depth[predict_depth_idx] = 1
                self.predict_depth_idx = torch.LongTensor([predict_depth_idx]) \
                    .type(self.dtype.LongTensor)
                self.predict_depth = self.predict_depth.unsqueeze(0) \
                    .expand(__BS, self.state_dim[0])

        if have_more_games:
            not_done_idx = not_done.nonzero().view(-1)
            non_done_states = obs.index_select(0, not_done_idx)
            hidden_state = self.net.slice_hidden(hidden_state, not_done_idx)
        else:
            not_done_idx = torch.LongTensor(0).type_as(done)
            non_done_states = None

        action = None
        if have_more_games:
            state = non_done_states.float()

            # select Action
            probs, state_value, new_hidden_state, other_task = \
                self.net(Variable(state, requires_grad=False), hidden_state,
                         aux_input={}, **kwargs)
            # print(probs)
            if self.predict_max:
                _, action = probs.max(1)
            else:
                action = probs.multinomial()

        if have_more_games:
            self.live_idx = live_idx = \
                self.live_idx.index_select(0, not_done_idx)
            self.hidden_state = new_hidden_state
        else:
            self.hidden_state = self.net.init_hidden(self.batch_size)

            batch_size = self.batch_size
            self.live_idx = torch.linspace(0, batch_size - 1, batch_size) \
                .type(self.dtype.LongTensor)

        self.step_cnt += 1

        # MUST RETURN SAME SIZE AS OBS SIZE
        action_holder = torch.LongTensor(reward.size(0)) \
            .type(self.dtype.LongTensor)
        action_holder.fill_(3)
        if action is not None:
            action_holder.scatter_(
                0, not_done_idx, action.data.view(-1).long())
        return action_holder

    def finish_episode(self):
        new_time = time.time()
        print("-------------> {:f} seconds".format(new_time - self.last_time))
        self.last_time = new_time

        saved_transitions = self.saved_transitions
        value_loss = 0
        rewards = []
        rewards_norm = []
        rewards_true_norm = []

        for i in range(len(saved_transitions))[::-1]:
            r = saved_transitions[i].reward.clone()
            playing_idx = saved_transitions[i].playing_idx
            if i < len(saved_transitions) - 1:
                r.index_add_(0, playing_idx, rewards[0] * self.gamma)

            rewards.insert(0, r)

        all_r = torch.cat(rewards, 0)
        mean_r = all_r.mean()
        std_r = all_r.std() + np.finfo(np.float32).eps
        rewards_mean_step = []
        backprop_reward = []
        for i in range(len(rewards)):
            mean_r = rewards[i].mean()
            max_r = rewards[i].max()

            rewards_norm.append((rewards[i] - mean_r) / std_r)
            rewards_true_norm.append(rewards[i] / ENV_CAUGHT_REWARD)
            rewards_mean_step.append(rewards_true_norm[i].mean())

            # Normalize action advantage in 1.0
            reward_adv = rewards_true_norm[i] - saved_transitions[i].value.data[
                                                :, 0]

            backprop_reward.append(reward_adv)

        # -------- Solve other tasks; Predict next state --------------------
        loss_aux = {}
        if "next_state_depth" in self.aux_tasks:
            loss_ = 0
            for i in range(len(saved_transitions)):
                pred_depth = saved_transitions[i].other_task["next_state_depth"]
                next_state_D = saved_transitions[i].next_obs \
                    .index_select(1, self.predict_depth_idx)

                next_state_D = Variable(next_state_D.view(-1, 81))
                loss_ += nn.BCELoss()(pred_depth, next_state_D)
            loss_aux["next_state_depth"] = loss_

        if "next_reward" in self.aux_tasks:
            loss_ = 0
            for i in range(len(saved_transitions)):
                pred_next_r = saved_transitions[i].other_task["next_reward"]
                loss_ += nn.MSELoss()(pred_next_r,
                                      Variable(saved_transitions[i].reward /
                                               ENV_CAUGHT_REWARD))
            loss_aux["next_reward"] = loss_

        # ----------------------------------------------------------------------
        actor_loss = None
        for i in range(len(saved_transitions)):
            tr = saved_transitions[i]
            true_env_r = rewards_true_norm[i]
            probs = tr.probs

            reward = backprop_reward[i]
            grad = probs.data.new().resize_as_(probs.data).fill_(0)

            grad.scatter_(1, tr.action, reward.unsqueeze(1))
            grad /= -probs.data

            l = torch.dot(Variable(grad), probs)
            actor_loss = l if actor_loss is None else (actor_loss + l)

            # TODO put if
            # tr.action.reinforce(reward.unsqueeze(1))
            value_loss += nn.MSELoss()(tr.value[:, 0],
                                       Variable(rewards_true_norm[i]))

        self.optimizer.zero_grad()

        full_loss = value_loss + actor_loss + \
                    loss_aux["next_reward"] * 0.1 + \
                    loss_aux["next_state_depth"] * 0.1
        full_loss.backward()

        print(
            "Total_loss: {:.5f}. value_loss: {:.5f} actor_loss: {:.5f} "
            "next_reward: {:.5f} "
            "next_state_depth: {:.5f}".format(full_loss.data[0],
                                              value_loss.data[0],
                                              actor_loss.data[0],
                                              loss_aux["next_reward"].data[0],
                                              loss_aux["next_state_depth"].data[
                                                  0]))

        # `clip_grad_norm` helps prevent the exploding gradient problem
        torch.nn.utils.clip_grad_norm(self.net.parameters(), self.clip_grad)

        self.optimizer.step()

        self.step_cnt = 0
        self.batch_games = 0
        self.ep_training += 1
        del self.saved_transitions[:]
        self.reset()
