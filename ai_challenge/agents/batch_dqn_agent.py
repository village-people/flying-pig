# Village People, 2017

import torch
from collections import namedtuple
from agents import ReportingAgent
from models import get_model
from methods import DeterministicPolicy, DQNPolicyImprovement
from methods import get_batch_schedule
from utils.torch_types import TorchTypes
# from termcolor import colored as clr
ENV_CAUGHT_REWARD = 25

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'state_', 'done', 'a_'))

class BetaDQNBatchAgent(ReportingAgent):
    def __init__(self, name, action_space, cfg, shared_objects={}):
        super(BetaDQNBatchAgent, self).__init__()

        self.name = name
        self.pidx = 0
        self.actions_no = len(action_space)
        self.target_update_freq = cfg.training.target_update
        self.batch_size = batch_size = cfg.general.batch_size
        self.update_freq = batch_size
        self.hist_len = cfg.model.hist_len
        self.cuda = cfg.general.use_cuda
        self.torch_obs = True
        self.dtype = TorchTypes(self.cuda)

        if "model" in shared_objects:
            self.net = net = shared_objects["model"]
            self.print_info("Training some shared model.")
        else:
            self.net = net = get_model(cfg.model.name)(cfg.model)
            if self.cuda:
                net.cuda()

        self.target = get_model(cfg.model.name)(cfg.model)
        if self.cuda:
            self.net.cuda()
            self.target.cuda()

        self.behaviour = DeterministicPolicy(self.net)
        self.exploration_strategy = get_batch_schedule(cfg.agent.exploration,
                                                       batch_size)
        self.ddqn = cfg.agent.ddqn
        self.algorithm = DQNPolicyImprovement(self.net, self.target, cfg,
                                              self.ddqn)
        self.batch = []
        self.history = []

        self.batch_games = 0
        self.step_cnt = 0
        self._o = None
        self._a = None
        self.live_idx = torch.linspace(0, batch_size-1, batch_size) \
                             .type(self.dtype.LongTensor)

    def get_models(self):
        return [self.net, self.target]

    def act(self, obs, reward, done, is_training):
        self.epsilon = next(self.exploration_strategy)

        if len(self.history) == 0:
            self.history = [obs] * (self.hist_len - 1)
        self.history = history = self.history[(1 - self.hist_len):] + [obs]

        not_done = (1 - done)
        have_more_games = not_done.byte().any()
        if have_more_games:
            not_done_idx = not_done.nonzero().view(-1)
            self.history = [s.index_select(0, not_done_idx) for s in history]
            non_done_states = torch.cat(self.history, 1)
            self.live_idx = live_idx = \
                self.live_idx.index_select(0, not_done_idx)
        else:
            non_done_states = None
            self.history = []

        action = None
        if non_done_states is not None:

            epsilon = next(self.exploration_strategy)
            mask = torch.bernoulli(torch.FloatTensor(epsilon)) \
                        .type(self.dtype.LongTensor) \
                        .index_select(0, live_idx)
            n = mask.nonzero().nelement()
            state = non_done_states
            _, action = self.behaviour.get_action(state)
            action.squeeze_(1)
            if n > 0:
                rand_idx = mask.nonzero().squeeze(1)
                r_action = torch.LongTensor(rand_idx.size(0))
                r_action.random_(self.actions_no)
                r_action = r_action.type(self.dtype.LongTensor)
                action.index_fill_(0, rand_idx, 0)
                action.index_add_(0, rand_idx, r_action)

        # TODO not sure if necessary
        done = done.clone()
        reward = reward.clone()

        if self._o is not None and is_training:
            if self.ddqn:
                self._improve_policy(self._o, self._a, reward,
                                     non_done_states, done, action)
            else:
                self._improve_policy(self._o, self._a, reward,
                                     non_done_states, done)

        if have_more_games:
            self._o = non_done_states
            self._a = action
        else:
            self._o, self._a = None, None
            batch_size = self.batch_size
            self.live_idx = torch.linspace(0, batch_size-1, batch_size) \
                                 .type(self.dtype.LongTensor)

        self.step_cnt += 1

        # MUST RETURN SAME SIZE AS OBS SIZE
        action_holder = torch.LongTensor(reward.size(0))\
            .type(self.dtype.LongTensor)
        action_holder.fill_(3)
        if action is not None:
            action_holder.scatter_(0, not_done_idx, action)
        return action_holder

    def _improve_policy(self, _s, _a, r, s, done, a_=None):

        self.batch.append((_s, _a, r, s, done, a_))
        self.batch_games += _s.size(0)

        # TODO Training will be lost on the last batch of running games
        if (self.batch_games > self.update_freq):
            batch = self._batch2torch(self.batch)
            self.algorithm.accumulate_gradient(*batch)
            self.algorithm.update_model()
            # print(self.step_cnt, ":", len(self.batch), done)
            self.batch_games = 0
            self.batch.clear()

        if self.step_cnt % self.target_update_freq == 0:
            print("------Updated target--------")
            # print("EPS: {}".format(self.epsilon))
            self.algorithm.update_target_net()

    def _batch2torch(self, batch, batch_sz=None):
        """
        List of Transitions of batches to tensors batch states, actions,
        rewards.
        """

        batch = Transition(*zip(*batch))
        # print("[%s] Batch len=%d" % (self.name, batch_sz))

        state_batch = torch.cat(batch.state, 0).float()

        # ONLY NON Terminal next states
        next_state_batch = [x for x in batch.state_ if x is not None]
        next_state_batch = torch.cat(next_state_batch, 0).float()

        action_batch = torch.cat(batch.action, 0)
        reward_batch = torch.cat(batch.reward, 0).float()

        mask = torch.cat(batch.done, 0)
        mask = (1 - mask).nonzero().view(-1)

        batch_sz = action_batch.size(0)

        if self.ddqn:
            # ONLY NON Terminal actions
            next_action_batch = [x for x in batch.a_ if x is not None]
            next_action_batch = torch.cat(next_action_batch, 0)

            return [batch_sz, state_batch, action_batch, reward_batch,
                    next_state_batch, mask, next_action_batch]
        else:
            return [batch_sz, state_batch, action_batch, reward_batch,
                    next_state_batch, mask]
