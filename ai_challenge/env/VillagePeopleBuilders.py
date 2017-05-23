# Village People, 2017

from __future__ import absolute_import

import numpy as np

from pig_chase.common import ENV_BOARD, ENV_ENTITIES, \
    ENV_BOARD_SHAPE, ENV_AGENT_NAMES, ENV_TARGET_NAMES

from malmopy.environment.malmo import MalmoStateBuilder

from .artificial_malmo import BLOCK_TYPE
from .artificial_malmo import yaw_to_angle

from malmopy.agent import BaseAgent
from pig_chase.environment import PigChaseEnvironment

import torch


def transform_angle(agent):
    agent['yaw'] = (((((int(agent['yaw']) - 45) % 360) // 90) - 1) % 4) + 2
    return agent['yaw']


def getAngle(agent):
    return (((((int(agent['yaw']) - 45) % 360) // 90) - 1) % 4) + 2


class PigChaseVillagePopleAgent(BaseAgent):
    """
    Village Pople Official Agent
    -- is_training -> False !
    -- Assumes observations got from the PigChaseVillagePeopleBuilder18Binary
    """

    def __init__(self, name, nb_actions, agent, use_cuda=False,
                 visualizer=None):
        super(PigChaseVillagePopleAgent, self).__init__(name, nb_actions,
                                                        visualizer)

        self.dtype = torch.LongTensor([])
        if use_cuda:
            self.dtype = self.dtype.cuda()

        self.agent = agent

    def act(self, new_state, reward, done, is_training=False):
        dtype = self.dtype

        state_ = torch.LongTensor(new_state).type_as(dtype).unsqueeze(0)
        reward_ = torch.FloatTensor([reward]).type_as(dtype)
        done_ = torch.LongTensor([int(done)]).type_as(dtype)

        if not done:
            act = self.agent.act(state_, reward_, done_, False)
        else:
            state_empty = state_.clone()
            state_empty.fill_(0)
            _ = self.agent.act(state_empty, reward_, done_, False)
            reward_[0] = 0
            done_[0] = 0
            act = self.agent.act(state_, reward_, done_, False)

        return act[0]

    def save(self, out_dir):
        pass

    def load(self, out_dir):
        pass

    def inject_summaries(self, idx):
        pass

    def reset(self):
        pass


class PigChaseVillagePeopleBuilder18Binary(MalmoStateBuilder):
    """
    This class build a symbolic representation of the current environment.
    GENERATES A MAP OF 9x9x18 binary maps
    x3 - block type ["grass", "sand", "lapis_block"]
    x5 - Player agent - 1x location + x4 (Direction type) [N, E, S, V]
    x5 - Opponent agent -  - ... -
    x5 - Pig -  - ... -
    """

    def __init__(self, player_idx):
        """
        :param player_idx: must specify the player for whom the map is built
        Order is important
        """
        self.player = ENV_AGENT_NAMES[player_idx]
        self.opponent = ENV_AGENT_NAMES[(player_idx + 1) % len(ENV_AGENT_NAMES)]
        self.pig = ENV_TARGET_NAMES[0]

        self.block_board_size = (len(BLOCK_TYPE),) + ENV_BOARD_SHAPE
        self.entity_board_size = (5,) + ENV_BOARD_SHAPE
        self.block_type = BLOCK_TYPE

        self.last_binary_view_0 = self.last_binary_view_1 = \
            self.last_binary_view_2 = np.zeros(self.entity_board_size,
                                               dtype=int)

    def binary_view_entity(self, entity):
        """
        Receives a dictionary with entity data:
        row, col, direction
        """
        i, j, direction = entity["row"], entity["col"], entity["direction"]

        board_binary = np.zeros(self.entity_board_size, dtype=int)
        board_binary[0][i][j] = 1
        board_binary[direction + 1][i][j] = 1
        return board_binary

    def build(self, environment):
        """
        Return a symbolic view of the board

        :param environment Reference to the pig chase environment
        :return (board, entities) where board is an array of shape (9, 9)
        """
        assert isinstance(environment,
                          PigChaseEnvironment), \
            'environment is not a Pig Chase Environment instance'

        block_board_size = self.block_board_size
        block_type = self.block_type

        world_obs = environment.world_observations

        if world_obs is None or ENV_BOARD not in world_obs:
            return None

        # Generate symbolic view
        board = np.array(world_obs[ENV_BOARD], dtype=object).reshape(
            ENV_BOARD_SHAPE)

        board_binary = np.zeros(block_board_size, dtype=int)
        for ix, block_type in enumerate(block_type):
            board_binary[ix] = board == block_type

        entities = world_obs[ENV_ENTITIES]
        entities_dict = dict()
        for entity in entities:
            entities_dict[entity["name"]] = dict(
                {"row": int(entity['z'] + 1),
                 "col": int(entity['x']),
                 "direction": yaw_to_angle(entity['yaw'])})

        if self.player in entities_dict:
            binary_view_0 = self.binary_view_entity(entities_dict[self.player])
        else:
            binary_view_0 = np.zeros(self.entity_board_size, dtype=int)

        if self.opponent in entities_dict:
            binary_view_1 = self.binary_view_entity(
                entities_dict[self.opponent])
        else:
            binary_view_1 = np.zeros(self.entity_board_size, dtype=int)

        if self.pig in entities_dict:
            binary_view_2 = self.binary_view_entity(entities_dict[self.pig])
        else:
            binary_view_2 = np.zeros(self.entity_board_size, dtype=int)

        self.last_binary_view_0 = binary_view_0
        self.last_binary_view_1 = binary_view_1
        self.last_binary_view_2 = binary_view_2

        game_view = np.concatenate([board_binary, binary_view_0,
                                    binary_view_1, binary_view_2])

        return game_view
