# Village People, 2017

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from models import FuN

from collections import namedtuple
from termcolor import colored as clr

_fields = 'state'
_fields += ' policy action goals m_value w_value latents'
_fields += ' reward new_state'
_fields += ' back_idx alive_no'
Transition = namedtuple('Transition', _fields)


class FeudalAgent():
    def __init__(self, name, action_space, cfg, shared_objects={}):
        super(FeudalAgent, self).__init__()

        self.name = name
        self.actions_no = len(action_space)

        self.batch_size = batch_size = cfg.general.batch_size
        m_gamma = cfg.agent.m_gamma
        w_gamma = cfg.agent.w_gamma

        cfg.model.actons_no = self.actions_no

        self.fun = fun = FuN(cfg.model)
        if cfg.general.use_cuda:
            fun.cuda()

        self.best_reward = None

        print(clr("Training starts.", "green"))

        Optimizer = getattr(optim, cfg.training.algorithm)
        self.optimizer = Optimizer(fun.parameters(),
                                   **vars(cfg.training.algorithm_args))

        self.coeff = cfg.training.feudal_coeff

        self._reset()

    def _reset(self):
        self.last_ten = []
        self.last_done = None
        self.transitions = []

        self.clock = 0
        self.prev_done = prev_done = torch.ByteTensor(self.batch_size) \
            .fill_(0)  # .cuda()
        self.prev_not_done = prev_not_done = 1 - prev_done
        self.prev_alive_idx = prev_not_done.nonzero().squeeze(1)
        self.back_idx = None
        self.latents = []
        self.back_links = []
        self.alive_no = [self.batch_size]
        self.goals = None
        self.model_state = None
        self.total_reward = .0

    def act(self, states, rewards, done, is_training):
        batch_size = self.batch_size
        states = states.float()

        done = done.byte()

        assert states.size(0) == batch_size
        assert rewards.size(0) == batch_size
        assert done.size(0) == batch_size

        if self.clock > 0:
            prev_done = self.prev_done
            prev_alive_idx = self.prev_alive_idx

            if prev_done.any():
                new_states = states.index_select(0, prev_alive_idx)
                rewards = rewards.index_select(0, prev_alive_idx)
            else:
                new_states = states

            self.total_reward = self.total_reward + rewards.sum()

            transitions.append(
                Transition(
                    state=self.prev_states,
                    policy=self.policy,
                    action=self.actions,
                    goals=self.goals,
                    m_value=self.m_value,
                    w_value=self.w_value,
                    latents=self.latents,
                    reward=rewards,
                    new_state=self.new_states,
                    back_idx=self.back_idx,
                    alive_no=self.alive_no[-1]
                )
            )

        done = done | self.prev_done

        if done.all():
            self._improve_policy()
            self._reset()

        not_done = 1 - done
        self.alive_no.append(not_done.nonzero().nelement())
        alive_no = self.alive_no

        prev_done = self.prev_done
        prev_alive_idx = self.prev_alive_idx

        goals = self.goals
        model_state = self.model_state

        if done.any():
            # -- If there are some dead games: slice the states
            self.alive_idx = alive_idx = not_done.nonzero().squeeze(1)
            states = states.index_select(0, alive_idx)
            if (done - prev_done).nonzero().nelement() > 0:
                self.back_idx = not_done.index_select(0, prev_alive_idx) \
                    .nonzero().squeeze(1)
                _idx = Variable(self.back_idx)
                if self.clock > 0:
                    model_state = [s.index_select(0, _idx) for s in model_state]
                    self.goals = goals = [g.index_select(0, _idx) for g in
                                          goals]
                    self.latents = [l.index_select(0, _idx) for l in
                                    self.latents]
            else:
                back_idx = None

        assert states.size(0) == alive_no[-1]

        policy, latent, goals, self.m_value, self.w_value, self.model_state = \
            self.fun(Variable(states), self.clock, goals, model_state)
        actions = torch.multinomial(policy)

        assert actions.size(0) == alive_no[-1]

        full_actions = actions.data.new().resize_(batch_size).fill_(0)
        full_actions.scatter_(0, self.alive_idx, actions.data.squeeze(1))

        self.latents.append(latent)
        self.goals = goals
        self.policy = policy
        self.actions = actions

        prev_states, prev_done, prev_alive_idx = states, done, alive_idx

        not_done = 1 - done
        alive_no.append(not_done.nonzero().nelement())
        self.clock += 1

    def _improve_policy(self):

        total_reward = self.total_reward
        last_te = self.last_ten
        transitions = self.transitions
        m_gamma = self.m_gamma
        w_gamma = self.w_gamma

        # -- Loop ends here

        last_ten.append(total_reward)

        # -- Improve policy

        for tr in transitions:
            assert tr.state.size(0) == tr.alive_no
            assert tr.action.size(0) == tr.alive_no
            for l in tr.latents:
                assert l.size(0) == tr.alive_no
            for g in tr.goals:
                assert g.size(0) == tr.alive_no
            assert tr.m_value.size(0) == tr.alive_no
            assert tr.w_value.size(0) == tr.alive_no
            assert tr.reward.size(0) == tr.alive_no
            assert tr.new_state.size(0) == tr.alive_no
            assert (tr.back_idx is None) or (tr.back_idx.size(0) == tr.alive_no)

        m_return = None
        T = len(transitions)

        tpgs, m_critics, w_actors, w_critics = [], [], [], []

        for (t, tr) in reversed(
                [(t, tr) for (t, tr) in enumerate(transitions)]):
            # -- Compute manager's reward
            if m_return is None:
                m_return = tr.reward
            elif fwd_idx is None:
                m_return = tr.reward + m_gamma * m_return
            else:
                m_return = tr.reward.clone().fill_(0) \
                    .scatter_(0, fwd_idx, m_return)
                m_return = tr.reward + m_gamma * m_return

            # -- Transition policy gradients

            c = cfg.model.c
            if t + c < T:
                m_adv = m_return - tr.m_value.data
                s_t = tr.latents[-1].data
                g_t = tr.goals[-1]
                for i in range(t + 1, t + c + 1):
                    tr_i = transitions[i]
                    if tr_i.back_idx is not None:
                        g_t = g_t.index_select(0, Variable(tr_i.back_idx))
                        s_t = s_t.index_select(0, tr_i.back_idx)
                        m_adv = m_adv.index_select(0, tr_i.back_idx)
                    s_t_c = tr_i.latents[-1].data

                cos = F.cosine_similarity(Variable(s_t_c - s_t), g_t, 1)
                tpgs.append(torch.dot(cos, Variable(m_adv)))

                m_critics.append(
                    F.smooth_l1_loss(tr.m_value, Variable(m_return))
                )

            # -- Compute worker's intrinsic reward
            c = min(cfg.model.c, len(tr.latents) - 1)
            if c > 0:
                last_l = tr.latents[-1].data
                intr_r = last_l.new().resize_(tr.alive_no).fill_(0)
                for l, g in zip(tr.latents[-(c + 1):-1], tr.goals[-(c + 1):-1]):
                    intr_r += F.cosine_similarity(last_l - l.data, g.data, 1)
                intr_r /= c

            if cfg.shortcut:
                w_return = m_return
            else:
                w_return = m_return + cfg.model.alpha * intr_r

            w_adv = w_return - tr.w_value.data

            grad = w_adv.new().resize_(tr.policy.size()).fill_(0)
            grad.scatter_(1, tr.action.data, w_adv.unsqueeze(1))
            grad /= -tr.policy.data
            grad /= tr.policy.size(0)

            w_actors.append(torch.dot(Variable(grad), tr.policy))
            w_critics.append(F.smooth_l1_loss(tr.w_value, Variable(w_return)))

            fwd_idx = tr.back_idx

        t_loss, mc_loss = sum(tpgs), sum(m_critics)
        wa_loss, wc_loss = sum(w_actors), sum(w_critics)

        if len(last_ten) == 10:

            print("// Episode " + clr("{:d}".format(ep + 1), "yellow") + ".")

            mean_r = sum(last_ten) / 10.0
            if best_reward is None or best_reward < mean_r:
                print(clr("Reward : ") + \
                      clr("{:.2f}".format(mean_r), "white", "on_magenta")
                      )
                best_reward = mean_r
            else:
                print(clr("Reward : ") + clr("{:.2f}".format(mean_r), "red"))
            last_ten = []
            sep = clr(" | ", "green")
            print("TPG: " + \
                  clr("{:f}".format(t_loss.data[0]), "yellow") + sep + \
                  "M critic: " + \
                  clr("{:f}".format(mc_loss.data[0]), "yellow") + sep + \
                  "W actor: " + \
                  clr("{:f}".format(wa_loss.data[0]), "yellow") + sep + \
                  "W critic: " + \
                  clr("{:f}".format(wc_loss.data[0]), "yellow"))

        if self.shortcut:
            (coeff["WACTOR"] * wa_loss + \
             coeff["WCRITIC"] * wc_loss
             ).backward()
        else:
            (coeff["TPG"] * t_loss + \
             coeff["MCRITIC"] * mc_loss + \
             coeff["WACTOR"] * wa_loss + \
             coeff["WCRITIC"] * wc_loss
             ).backward()

        self.optimizer.step()
        self.optimizer.zero_grad()
