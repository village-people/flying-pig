from __future__ import division

from collections import namedtuple

import numpy as np
from pig_chase.common import ENV_TARGET_NAMES, ENV_ACTIONS
from .artificial_malmo import BLOCK_TYPE
from six.moves import range

from malmopy.agent import AStarAgent
from malmopy.agent import BaseAgent


class RandomAgent_V(BaseAgent):
    """
    An agent that selects actions uniformly at random
    """

    def __init__(self, name, nb_actions, visualizer=None):
        self._type = 0
        super(RandomAgent_V, self).__init__(name, nb_actions, visualizer)

    def act(self, new_state, reward, done, is_training=False):
        return np.random.randint(0, self.nb_actions)


class PigChaseChallengeAgent_V(BaseAgent):
    """
    Pig Chase challenge agent - behaves focused or random.
    This is a version of the PigChaseChallengeAgent from Malmo, only that it
    accepts as input other format.
    """

    def __init__(self, name, visualizer=None, p_focus=0.7):
        self.p_focused = p_focus

        nb_actions = len(ENV_ACTIONS)
        super(PigChaseChallengeAgent_V, self).__init__(name, nb_actions,
                                                       visualizer=visualizer)

        self._agents = []
        self._agents.append(FocusedAgent_V(name, ENV_TARGET_NAMES[0],
                                           visualizer=visualizer))
        self._agents.append(RandomAgent_V(name, nb_actions,
                                          visualizer=visualizer))
        self.current_agent = self._select_agent(self.p_focused)

    def _select_agent(self, p_focused):
        return self._agents[np.random.choice(range(len(self._agents)),
                                             p=[p_focused, 1. - p_focused])]

    def act(self, new_state, reward, done, is_training=False):
        if done:
            self.current_agent = self._select_agent(self.p_focused)
        return self.current_agent.act(new_state, reward, done, is_training)

    def save(self, out_dir):
        self.current_agent.save(out_dir)

    def load(self, out_dir):
        self.current_agent(out_dir)

    def inject_summaries(self, idx):
        self.current_agent.inject_summaries(idx)


class FocusedAgent_V(AStarAgent):
    # -- HardCoded Should be from common.ENV_ACTIONS
    ACTIONS = ENV_ACTIONS
    ACTION_COMMANDS = list(map(lambda x: int(x.split(' ')[1]), ACTIONS))
    sand_type = BLOCK_TYPE.index("sand")

    Neighbour = namedtuple('Neighbour',
                           ['cost', 'x', 'z', 'direction', 'action'])

    def __init__(self, name, target, visualizer=None):
        super(FocusedAgent_V, self).__init__(name, len(FocusedAgent_V.ACTIONS),
                                             visualizer=visualizer)
        self._type = 1
        self._target = str(target)
        self._previous_target_pos = None
        self._action_list = []

    def act(self, obs, reward, done, is_training=False):
        """
        :param state: ([(me_row, me_col, me_direction),
                    (opponent_row, opponent_col, opponent_direction),
                    (pig_row, pig_col, pig_direction)],
                    map)
            map is of type numpy filled with index(artificial_malmo.BLOCK_TYPE)
        """
        if done:
            self._action_list = []
            self._previous_target_pos = None

        (player, opponent, pig), state = obs

        # -- Coordinates should me saved as (col, row)
        me = [(player[1], player[0])]
        direction = int(player[2])  # 0=north, 1=east etc.
        target = [(pig[1], pig[0])]

        # Get agent and target nodes
        me = FocusedAgent_V.Neighbour(1, me[0][0], me[0][1], direction, "")
        target = FocusedAgent_V.Neighbour(1, target[0][0], target[0][1], 0, "")

        # If distance to the pig is one, just turn and wait
        if self.heuristic(me, target) == 1:
            # substitutes for a no-op command
            return FocusedAgent_V.ACTIONS.index("turn 1")

        if not self._previous_target_pos == target:
            # Target has moved, or this is the first action of a new mission
            #  - calculate a new action list
            self._previous_target_pos = target

            path, costs = self._find_shortest_path(me, target, state=state)
            self._action_list = []
            for point in path:
                self._action_list.append(point.action)

        if self._action_list is not None and len(self._action_list) > 0:
            action = self._action_list.pop(0)
            return FocusedAgent_V.ACTIONS.index(action)

        # reached end of action list - turn on the spot
        # substitutes for a no-op command
        return FocusedAgent_V.ACTIONS.index("turn 1")

    def neighbors(self, pos, state=None):
        state_width = state.shape[1]
        state_height = state.shape[0]
        dir_north, dir_east, dir_south, dir_west = range(4)
        neighbors = []

        inc_x = lambda x, dir, delta: x + delta \
            if dir == dir_east else x - delta if dir == dir_west else x
        inc_z = lambda z, dir, delta: z + delta \
            if dir == dir_south else z - delta if dir == dir_north else z

        # add a neighbour for each potential action;
        # - prune out the disallowed states afterwards
        action_cmd = FocusedAgent_V.ACTION_COMMANDS
        for ix, action in enumerate(FocusedAgent_V.ACTIONS):
            if action.startswith("turn"):
                neighbors.append(
                    FocusedAgent_V.Neighbour(1, pos.x, pos.z,
                                             (pos.direction + action_cmd[ix])
                                             % 4, action))
            # note the space to distinguish from movemnorth etc
            elif action.startswith("move "):
                sign = action_cmd[ix]
                weight = 1 if sign == 1 else 1.5
                neighbors.append(
                    FocusedAgent_V.Neighbour(weight,
                                             inc_x(pos.x, pos.direction, sign),
                                             inc_z(pos.z, pos.direction, sign),
                                             pos.direction, action))
            else:
                print("-- ERROR -- action starts with {}".format(action))
                exit(0)

        # now prune:
        sand_type = self.sand_type
        valid_neighbours = [n for n in neighbors if
                            n.x >= 0 and n.x < state_width and n.z >= 0
                            and n.z < state_height
                            and state[n.z, n.x] != sand_type]
        return valid_neighbours

    def heuristic(self, a, b, state=None):
        (x1, y1) = (a.x, a.z)
        (x2, y2) = (b.x, b.z)
        return abs(x1 - x2) + abs(y1 - y2)

    def matches(self, a, b):
        # don't worry about dir and action
        return a.x == b.x and a.z == b.z
