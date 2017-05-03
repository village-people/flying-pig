from numpy.random import randint


class DQNAgent(object):
    def __init__(self, name, action_space):
        self.name = name
        self.action_space = action_space
        self.action_space_len = len(action_space)

    def act(self, obs, reward, agent_done, is_training):
        return randint(self.action_space_len)

    def inject_summaries(self, step):
        pass
