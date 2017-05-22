""" Deep Q-Learning policy evaluation and improvement
"""
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

from termcolor import colored as clr


class DQNPolicyImprovement(object):
    """ Deep Q-Learning training method. """

    def __init__(self, policy, target_policy, cfg, ddqn=False):
        self.name = "DQN-PI"
        self.policy = policy
        self.target_policy = target_policy

        self.lr = cfg.training.lr
        self.gamma = cfg.agent.gamma

        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
        self.optimizer.zero_grad()

        self.decouple_grad = False
        self.grads_decoupled = False
        self.ddqn = ddqn

    def accumulate_gradient(self, batch_sz, states, actions, rewards,
                            next_states, mask, next_actions=None):
        """ Compute the temporal difference error.
            td_error = (r + gamma * max Q(s_,a)) - Q(s,a)
        """
        states = Variable(states)
        actions = Variable(actions)
        rewards = Variable(rewards)
        next_states = Variable(next_states, volatile=True)
        if self.ddqn:
            next_actions = Variable(next_actions)

        # Compute Q(s, a)
        policy_val = self.policy(states)
        q_values = policy_val.gather(1, actions.unsqueeze(1))

        # Compute Q(s_, a)
        q_target_values = None
        if next_states.is_cuda:
            q_target_values = Variable(torch.zeros(batch_sz).cuda())
        else:
            q_target_values = Variable(torch.zeros(batch_sz))

        # Bootstrap for non-terminal states
        real_mask = Variable(mask)
        target_q_values = self.target_policy(next_states)

        if self.ddqn:
            policy_next_state = target_q_values.gather(
                    1, next_actions.unsqueeze(1)).squeeze()
        else:
            policy_next_state = target_q_values.max(1)[0].view(-1)

        q_target_values.scatter_(0, real_mask, policy_next_state)

        q_target_values.volatile = False      # So we don't mess the huber loss
        expected_q_values = (q_target_values * self.gamma) + rewards

        # Compute Huber loss
        loss = F.smooth_l1_loss(q_values, expected_q_values)
        loss.data.clamp_(-1, 1)
        # print("Loss: {:.6f}".format(loss.data[0]))
        # Accumulate gradients
        loss.backward()

    def update_model(self):
        for param in self.policy.parameters():
            param.grad.data.clamp_(-1, 1)
            if not self.grads_decoupled and self.decouple_grad:
                param.grad.data = param.grad.data.clone()
                self.grads_decoupled = True

        self.optimizer.step()
        self.optimizer.zero_grad()

    def update_target_net(self):
        """ Update the target net with the parameters in the online model."""
        self.target_policy.load_state_dict(self.policy.state_dict())

    def get_model_stats(self):
        if self.grads_decoupled or not self.decouple_grad:
            param_abs_mean = 0
            grad_abs_mean = 0
            t_param_abs_mean = 0
            n_params = 0
            for p in self.policy.parameters():
                param_abs_mean += p.data.abs().sum()
                grad_abs_mean += p.grad.data.abs().sum()
                n_params += p.data.nelement()
            for t in self.target_policy.parameters():
                t_param_abs_mean += t.data.abs().sum()

            print("Wm: %.9f | Gm: %.9f | Tm: %.9f" % (
                param_abs_mean / n_params,
                grad_abs_mean / n_params,
                t_param_abs_mean / n_params))

    def _debug_transitions(self, mask, reward_batch):
        if mask[0] == 0:
            r = reward_batch[0, 0]
            if r == 1.0:
                print(r)

    def _debug_states(self, state_batch, next_state_batch, mask):
        for i in range(24):
            for j in range(24):
                px = state_batch[0, 0, i, j]
                if px < 0.90:
                    print(clr("%.2f  " % px, 'magenta'), end="")
                else:
                    print(("%.2f  " % px), end="")
            print()
        for i in range(24):
            for j in range(24):
                px = next_state_batch[0, 0, i, j]
                if px < 0.90:
                    print(clr("%.2f  " % px, 'magenta'), end="")
                else:
                    print(clr("%.2f  " % px, 'white'), end="")
            print()
        if mask[0] == 0:
            print(clr("Done batch ............", 'magenta'))
        else:
            print(".......................")
