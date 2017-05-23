# Village People, 2017

import torch
import numpy as np
import sys
from copy import deepcopy
from malmopy.environment import VideoCapableEnvironment
import re

from pig_chase.environment import PigChaseSymbolicStateBuilder

sys.path.append("../ai_challenge/pig_chase/")

np.set_printoptions(linewidth=400)

BUILDER_18BINARY = "VillagePeople18Binary"
BUILDER_MALMO = "Malmo"
BUILDER_SYMBOLIC = "Symbolic"
BUILDER_SIMPLE = "Simple"

STATE_BUILDERS = {
    BUILDER_18BINARY: False,
    BUILDER_MALMO: False,
    BUILDER_SYMBOLIC: False,
    BUILDER_SIMPLE: False
}

BUILDER_OBSERVATION_DATA_TYPES = {
    BUILDER_18BINARY: torch.LongTensor(),
    BUILDER_MALMO: list(),
    BUILDER_SYMBOLIC: list(),
    BUILDER_SIMPLE: torch.LongTensor()
}
GRAY_PALETTE = {
    'sand': 255,
    'grass': 200,
    'lapis_block': 150,
    'Agent_1': 100,
    'Agent_2': 50,
    'Pig': 0
}

BLOCK_TYPE = ["grass", "sand", "lapis_block"]

ENV_AGENT_NAMES = ['Agent_1', 'Agent_2']
ENV_TARGET_NAMES = ['Pig']
ENV_ENTITIES_NAME = ENV_AGENT_NAMES + ENV_TARGET_NAMES
ENV_ACTIONS = ["move 1", "turn -1", "turn 1", "nothing"]
ENV_ENTITIES = 'entities'
ENV_BOARD = 'board'
ENV_BOARD_SHAPE = (9, 9)
DIRECTIONS = ["north", "east", "south", "west"]
DIRECTIONS_YAW = [180, 270, 0, 90]
DIRECTIONS_MOVE = [(-1, 0), (0, 1), (1, 0), (0, -1)]
REWARDS = [-1, +5, +25]  # move, exit, catch
REWARD_MOVE = -1
REWARD_EXIT = 5
REWARD_CATCH = 25

ENV_NULL_ACTION_IDX = 3

ENV_MAX_MOVES = 25

WORLD_OBSERVATION = {'board': ['grass', 'grass', 'grass', 'grass', 'grass',
                               'grass', 'grass', 'grass', 'grass', 'grass',
                               'sand', 'sand', 'sand', 'sand', 'sand', 'sand',
                               'sand', 'grass', 'grass', 'sand', 'grass',
                               'grass', 'grass', 'grass', 'grass', 'sand',
                               'grass', 'sand', 'sand', 'grass', 'sand',
                               'grass', 'sand', 'grass', 'sand', 'sand', 'sand',
                               'lapis_block', 'grass', 'grass', 'grass',
                               'grass', 'grass', 'lapis_block', 'sand', 'sand',
                               'sand', 'grass', 'sand', 'grass', 'sand',
                               'grass', 'sand', 'sand', 'grass', 'sand',
                               'grass', 'grass', 'grass', 'grass', 'grass',
                               'sand', 'grass', 'grass', 'sand', 'sand', 'sand',
                               'sand', 'sand', 'sand', 'sand', 'grass', 'grass',
                               'grass', 'grass', 'grass', 'grass', 'grass',
                               'grass', 'grass', 'grass'],
                     'entities':
                         [{'yaw': 0, 'x': 6.5, 'z': 3.5, 'name': 'Agent_1',
                           'y': 4.0, 'pitch': 30.0},
                          {'yaw': 0, 'x': 6.5, 'z': 3.5, 'name': 'Agent_2',
                           'y': 4.0, 'pitch': 30.0},
                          {'yaw': 0, 'x': 6.5, 'z': 3.5, 'name': 'Pig',
                           'y': 4.0, 'pitch': 30.0}]
                     }

BOARD = np.array(
    [['grass', 'grass', 'grass', 'grass', 'grass', 'grass', 'grass',
      'grass', 'grass'],
     ['grass', 'sand', 'sand', 'sand', 'sand', 'sand', 'sand', 'sand',
      'grass'],
     ['grass', 'sand', 'grass', 'grass', 'grass', 'grass', 'grass',
      'sand', 'grass'],
     ['sand', 'sand', 'grass', 'sand', 'grass', 'sand', 'grass', 'sand',
      'sand'],
     ['sand', 'lapis_block', 'grass', 'grass', 'grass', 'grass', 'grass',
      'lapis_block', 'sand'],
     ['sand', 'sand', 'grass', 'sand', 'grass', 'sand', 'grass', 'sand',
      'sand'],
     ['grass', 'sand', 'grass', 'grass', 'grass', 'grass', 'grass',
      'sand', 'grass'],
     ['grass', 'sand', 'sand', 'sand', 'sand', 'sand', 'sand', 'sand',
      'grass'],
     ['grass', 'grass', 'grass', 'grass', 'grass', 'grass', 'grass',
      'grass', 'grass']], dtype=object)

VALID_POSITIONS = [
    (2, 2), (2, 3), (2, 4), (2, 5), (2, 6),
    (3, 2), (3, 4), (3, 6),
    (4, 2), (4, 3), (4, 4), (4, 5), (4, 6),
    (5, 2), (5, 4), (5, 6),
    (6, 2), (6, 3), (6, 4), (6, 5), (6, 6)
]

EXIT_POINTS = [
    (4, 1), (4, 7)
]
ALL_POS = VALID_POSITIONS + EXIT_POINTS

PIG_DANGER_POS = [
    1, 1, 0, 1, 1,
    1, 1, 1,
    0, 1, 0, 1, 0,
    1, 0, 1,
    1, 1, 0, 1, 1,
    1, 1
]

ARROW_KEYS_MAPPING = {'Up': 'move 1', 'Left': 'turn -1', 'Right': 'turn 1'}


# from directions apply (turn -1, turn 1) -> new direction index
def apply_turn(direction_idx, turn):
    d_len = len(DIRECTIONS)
    turn_ = turn
    new_direction = direction_idx + turn_
    return (direction_idx + turn_ + d_len) % d_len


def add_tuple(x, y):
    return (x[0] + y[0], x[1] + y[1])


def distance_tuple(x, y):
    return abs(x[0] - y[0]) + abs(x[1] - y[1])


def cost_rotate(x, y):
    min_v, max_v = min(x, y), max(x, y)
    return min(abs(x - y), abs((min_v + len(DIRECTIONS_MOVE)) - max_v))


def yaw_to_angle(x):
    ans = ((((int(x) - 45) % 360) // 90) - 1) % 4
    return ans


def change_entity(entity, x, z, yaw):
    entity["x"] = x
    entity["z"] = z
    entity["yaw"] = yaw


class VillagePeopleEnvAgent():
    def __init__(self, name, env, agent_builder, malmo_data, cfg):
        self.name = name
        self.batch_size = batch_size = cfg.batch_size
        self.use_cuda = use_cuda = cfg.use_cuda
        self.move_final_state = move_final_state = cfg.move_final_state

        self.agent_builder = agent_builder

        self.valid_states_no = env.valid_states_no
        self.action_no = env.action_no
        self._coord_hash = env._coord_hash
        self._moves_hash = env._moves_hash
        if malmo_data is not None:
            self.malmo_boards = malmo_data[0]
            self.malmo_state = malmo_data[1]
            self.board_name = "/" + self.name

        self.yaw = env.yaw
        self.directions_no = env.directions_no
        dtype = self.dtype = env.dtype

        self.coord_size = coord_size = torch.Size((batch_size, 2))
        self.states_size = torch.Size((batch_size, self.action_no))
        self.agent_onehot_size = torch.Size((batch_size, 5, 9, 9))
        self.agent_onehot_size_line = torch.Size((batch_size, 5 * 9 * 9))
        self.agent_onehot_slice_size = 9 * 9

        self.agent_state = torch.LongTensor(batch_size).type_as(dtype)
        self.agent_coord = torch.LongTensor(batch_size).type_as(dtype)

        if not move_final_state:
            self.agent_pre_state = torch.LongTensor(batch_size).type_as(dtype)
            self.agent_pre_coord = torch.LongTensor(coord_size).type_as(dtype)

        self.prev_ag_catch_pig = torch.LongTensor(batch_size).type_as(dtype)
        self.prev_ag_exit_map = torch.LongTensor(batch_size).type_as(dtype)
        self.prev_ag_act = torch.LongTensor(batch_size).type_as(dtype)
        self.sent_done = torch.LongTensor(batch_size).type_as(dtype)
        self.got_done = torch.LongTensor(batch_size).type_as(dtype)

        self.ag_step_no = -1

        self.agent_onehot = torch.zeros(self.agent_onehot_size_line).long() \
            .type_as(dtype)

        # Auxiliary structures for actions in space
        self.random_action_storage = torch.LongTensor(batch_size) \
            .type_as(dtype)
        self.random_action_cpu = torch.LongTensor(batch_size)

    def random_batch(self, max):
        self.random_action_cpu.random_(max)
        return self.random_action_cpu

    def move_agent(self, actions):
        self.ag_step_no += 1

        agent_choice = self._moves_hash.gather(0, self.agent_state.unsqueeze(1)
                                               .expand(self.states_size))
        self.agent_state = agent_choice.gather(1, actions
                                               .unsqueeze(1)).view(-1)
        self.agent_coord = self._coord_hash.gather(0, self.agent_state
                                                   .unsqueeze(1)
                                                   .expand(self.coord_size))

    def set_state(self, state):
        self.ag_step_no += 1
        self.agent_state = state
        self.agent_coord = self._coord_hash.gather(0, self.agent_state
                                                   .unsqueeze(1)
                                                   .expand(self.coord_size))

    def move_pre_final_state(self, game_finished):
        not_finished = 1 - game_finished
        self.agent_pre_state.mul_(game_finished).add_(self.agent_state *
                                                      not_finished)
        self.agent_pre_coord.mul_(game_finished).add_(self.agent_coord *
                                                      not_finished)

    def finished_move(self, acted, cought_pig, exit_map):
        self.prev_ag_act.copy_(acted)
        self.prev_ag_catch_pig.copy_(cought_pig)
        self.prev_ag_exit_map.copy_(exit_map)

    def sent_info(self, done):
        self.sent_done = (self.sent_done.byte()
                          | done.byte()).long()

    def got_info(self):
        self.got_done.mul_(0).add_(self.sent_done)

    def check_if_exit_action(self, actions):
        states_choice = self._moves_hash.gather(0, self.agent_state.unsqueeze(1)
                                                .expand(self.states_size))

        states = states_choice.gather(1, actions
                                      .unsqueeze(1)).view(-1)
        return (states >= self.valid_states_no).long()

    def update_agent_one_hot(self):
        agent_onehot_slice_size = self.agent_onehot_slice_size
        holder = self.agent_onehot

        if self.move_final_state:
            states = self.agent_state
            coords = self.agent_coord
        else:
            states = self.agent_pre_state
            coords = self.agent_pre_coord
        holder.fill_(0)
        coord_ = (coords[:, 0] * 9 + coords[:, 1])
        holder.scatter_(1, coord_.unsqueeze(1), 1)
        direction = ((states % self.directions_no) + 1) * \
                    agent_onehot_slice_size + coord_
        holder.scatter_(1, direction.unsqueeze(1), 1)

        self.agent_onehot = holder

    def update_malmo_state(self):
        yaws = self.yaw
        ag_cd = self.agent_coord
        ag_st = self.agent_state
        ml_state = self.malmo_state

        x_list = ag_cd[:, 1].float().cpu() + 0.5
        z_list = ag_cd[:, 0].float().cpu() - 0.5
        yaw = yaws.gather(1, (ag_st % 4).unsqueeze(1)).view(-1)
        yaw = yaw.cpu()

        iterator_ = zip(ml_state, x_list, z_list, yaw)
        for m, x, z, yaw in iterator_:
            change_entity(m, x, z, yaw)

    def update_symbolic_state(self):
        board_name = self.board_name
        ml_state = self.malmo_state
        ml_board = self.malmo_boards

        for ix in range(len(ml_state)):
            i = int(ml_state[ix]['z'] + 1)
            j = int(ml_state[ix]['x'])
            ml_board[ix][i, j] = re.sub(board_name, "", ml_board[ix][i, j])

        self.update_malmo_state()

        for ix in range(len(ml_state)):
            i = int(ml_state[ix]['z'] + 1)
            j = int(ml_state[ix]['x'])
            ml_board[ix][i, j] += board_name

    def reset(self):
        self.ag_step_no = 0
        self.agent_state.copy_(self.random_batch(self.valid_states_no))
        self.agent_coord = self._coord_hash.gather(0,
                                                   self.agent_state.unsqueeze(1)
                                                   .expand(self.coord_size))
        if not self.move_final_state:
            self.agent_pre_state.copy_(self.agent_state)
            self.agent_pre_coord.copy_(self.agent_coord)

        self.prev_ag_catch_pig.fill_(0)
        self.prev_ag_exit_map.fill_(0)
        self.prev_ag_act.fill_(0)
        self.sent_done.fill_(0)
        self.got_done.fill_(0)


class VillagePeoplePig():
    direction_no = len(DIRECTIONS_MOVE)

    # --- IMPORTANT ADD !!! CONSTANT TO ADD FOR DISTANCE FOR IMPOSSIBLE MOVE
    no_move_prob = 100
    # --- 0 - > 8

    # prob_moves_distance = [22, 220, 297, 300, 300, 300, 300, 300]
    prob_moves_distance = [-200, 200, 200, 200, 200, 200, 200, 200]

    # --- Box to box distance 23 x 23
    pos_distance = []
    for ix, pos in enumerate(ALL_POS):
        dist_ix = []
        for iy, pos in enumerate(ALL_POS):
            dist_ix.append(distance_tuple(ALL_POS[ix], ALL_POS[iy]))
        pos_distance.append(dist_ix)

    # - Calculate possible boxes to move at ALL_POS x ALL_POS x 4*move_dir+stay
    coord_distance = []
    for ix, pos in enumerate(ALL_POS):
        to_ = []
        crt_coord = ALL_POS[ix]
        for iy, pos in enumerate(ALL_POS):
            move_d = []
            to_coord = ALL_POS[iy]

            for move in DIRECTIONS_MOVE:
                move_coord = add_tuple(crt_coord, move)
                if move_coord in ALL_POS:
                    move_d.append(distance_tuple(move_coord, to_coord))
                else:
                    move_d.append(-1)

            # crt distance
            move_d.append(distance_tuple(crt_coord, to_coord))
            to_.append(move_d)

        coord_distance.append(to_)

    # -- Calculate new pos
    move_state = []
    for ix, pos in enumerate(ALL_POS):
        to_ = []
        crt_coord = ALL_POS[ix]
        for move in DIRECTIONS_MOVE:
            move_coord = add_tuple(crt_coord, move)
            if move_coord in ALL_POS:
                to_.append(ALL_POS.index(move_coord))
            else:
                to_.append(ix)
        to_.append(ix)
        move_state.append(to_)
    move_state = np.array(move_state)

    # -- Calculate prob to move according to distance
    prob_move = np.array(coord_distance)
    prob_move_orig = np.array(coord_distance)

    prob_move[prob_move_orig == -1] = no_move_prob
    for i in range(len(prob_moves_distance)):
        prob_move[prob_move_orig == i] = prob_moves_distance[i]

    def __init__(self, batch_size, use_cuda):
        self.batch_size = batch_size

        self.dtype = torch.zeros(1).long()
        if use_cuda:
            self.dtype = self.dtype.cuda()

        dtype = self.dtype

        self.prob_move = torch.from_numpy(self.prob_move).type_as(dtype)
        self.move_state = torch.from_numpy(self.move_state).type_as(dtype)

        self.random_action_cpu = torch.zeros(batch_size)

    def random_batch(self):
        self.random_action_cpu.random_(self.direction_no)
        return self.random_action_cpu

    def move_pig(self, pig_state, agent0_state, agent1_state):
        direction_no = self.direction_no
        prob_pos = direction_no + 1
        batch_size = self.batch_size

        pig_pos = (pig_state / direction_no).long()
        agent0_pos = (agent0_state / direction_no).long()
        agent1_pos = (agent1_state / direction_no).long()

        pig_move_prob = self.prob_move.index_select(0, pig_pos)
        pig_move_agent0 = pig_move_prob.gather(1, agent0_pos.unsqueeze(1)
                                               .unsqueeze(1)
                                               .expand(batch_size, 1, prob_pos))
        pig_move_agent0 = pig_move_agent0.view(batch_size, prob_pos)
        pig_move_agent1 = pig_move_prob.gather(1, agent1_pos.unsqueeze(1)
                                               .unsqueeze(1)
                                               .expand(batch_size, 1, prob_pos)
                                               .contiguous())
        pig_move_agent1 = pig_move_agent1.view(batch_size, prob_pos)

        pig_move_prob = (pig_move_agent0 + pig_move_agent1).float() \
            .multinomial(1)

        possible_moves = self.move_state.index_select(0, pig_pos)

        pig_new_pos = possible_moves.gather(1, pig_move_prob)
        pig_new_pos.mul_(self.direction_no).add_(self.random_batch().type_as(
            self.dtype
        ))
        return pig_new_pos


class ArificialMalmo(VideoCapableEnvironment):
    assert len(DIRECTIONS) == len(DIRECTIONS_MOVE), "Direction & move problems"

    # GENERATE LOCAL MAPS
    map = torch.ones(ENV_BOARD_SHAPE)
    for empty_pos in VALID_POSITIONS:
        map[empty_pos] = 0
    for exit_pos in EXIT_POINTS:
        map[exit_pos] = -1

    directions_no = len(DIRECTIONS)
    valid_pos_no = len(VALID_POSITIONS)
    exit_pos_no = len(EXIT_POINTS)
    all_pos_no = len(ALL_POS)
    action_no = len(ENV_ACTIONS)
    null_move = action_no - 1
    max_moves = ENV_MAX_MOVES
    reward_move = REWARD_MOVE
    reward_exit = REWARD_EXIT
    reward_catch = REWARD_CATCH

    # GENERATE MOVE HASH TABLE
    valid_states_no = valid_pos_no * directions_no
    exit_states_no = exit_pos_no * directions_no
    exit_state_first_idx = valid_states_no
    states_no = valid_states_no + exit_states_no

    _pig_danger_pos = torch.LongTensor(PIG_DANGER_POS)

    _moves_hash = torch.zeros(states_no, action_no).long()
    _coord_hash = torch.zeros(states_no, 2).long()
    for state_idx in range(states_no):
        valid_pos = state_idx // directions_no
        direction = state_idx % directions_no

        _coord_hash[state_idx] = torch.LongTensor(ALL_POS[valid_pos])

        # Calculate move
        if state_idx < valid_states_no:
            new_pos = add_tuple(ALL_POS[valid_pos],
                                DIRECTIONS_MOVE[direction])
            if new_pos in EXIT_POINTS:
                new_pos_idx = EXIT_POINTS.index(new_pos) + valid_pos_no
                _moves_hash[state_idx, 0] = new_pos_idx * \
                                            directions_no + direction
            elif new_pos in VALID_POSITIONS:
                new_pos_idx = VALID_POSITIONS.index(new_pos)
                _moves_hash[state_idx, 0] = new_pos_idx * \
                                            directions_no + direction
            else:
                _moves_hash[state_idx, 0] = state_idx
        else:
            _moves_hash[state_idx, 0] = state_idx

        # Calculate rotation turn -1
        new_direction = apply_turn(direction, -1)
        _moves_hash[state_idx, 1] = valid_pos * directions_no + new_direction

        new_direction = apply_turn(direction, 1)
        _moves_hash[state_idx, 2] = valid_pos * directions_no + new_direction
        _moves_hash[state_idx, 3] = state_idx

    # BUILD board
    board_str = np.array(WORLD_OBSERVATION["board"])
    _board_one_hot = torch.zeros(3, ENV_BOARD_SHAPE[0],
                                 ENV_BOARD_SHAPE[0]).byte()
    _board_one_hot[0] = torch.Tensor((board_str == BLOCK_TYPE[0])
                                     .astype(float)).byte().view(
        ENV_BOARD_SHAPE)
    _board_one_hot[1] = torch.Tensor((board_str == BLOCK_TYPE[1])
                                     .astype(float)).byte().view(
        ENV_BOARD_SHAPE)
    _board_one_hot[2] = torch.Tensor((board_str == BLOCK_TYPE[2])
                                     .astype(float)).byte().view(
        ENV_BOARD_SHAPE)

    def __init__(self, cfg):
        self.batch_size = batch_size = cfg.batch_size
        self.use_cuda = use_cuda = cfg.use_cuda

        self.agent0_builder_type = agent0_builder_type = cfg.agent0_builder
        self.agent1_builder_type = agent1_builder_type = cfg.agent1_builder

        self.move_final_state = move_final_state = cfg.move_final_state

        self.dtype = torch.zeros(1).long()
        if use_cuda:
            self.dtype = self.dtype.cuda()

        dtype = self.dtype

        # Extend super class (CONFIGURE ENV FOR VISUALIZATION)
        self._visualize = cfg.visualize

        if (self._visualize):
            print("Vizualization ON ---!! FORCING 1 BATCH !!---")
            self.batch_size = batch_size = 1
            self.agent0_builder_type = agent0_builder_type \
                = BUILDER_MALMO
            self.agent1_builder_type = agent1_builder_type \
                = BUILDER_MALMO

        self._recording = False
        self._internal_symbolic_builder = PigChaseSymbolicStateBuilder()

        self.done_max = False

        self.states_size = torch.Size((batch_size, self.action_no))
        self.coord_size = coord_size = torch.Size((batch_size, 2))
        self.agent_onehot_size = torch.Size((batch_size, 5, 9, 9))
        self.agent_onehot_size_line = torch.Size((batch_size, 5 * 9 * 9))
        self.agent_onehot_slice_size = 9 * 9

        # Necessary data for actions in environment
        self._moves_hash = self._moves_hash.type_as(dtype)
        self._coord_hash = self._coord_hash.type_as(dtype)
        self._pig_danger_pos = self._pig_danger_pos.unsqueeze(0) \
            .expand(batch_size, self.all_pos_no).type_as(dtype)

        self.yaw = yaw = torch.LongTensor(DIRECTIONS_YAW).type_as(dtype)
        self.yaw = yaw.unsqueeze(0).expand(batch_size, len(DIRECTIONS))

        # Game finish state variables
        self.game_finished = torch.LongTensor(batch_size).type_as(dtype)

        self.sent_done = torch.LongTensor(batch_size).type_as(dtype)

        # Auxiliary structures for actions in space
        self.random_action_storage = torch.LongTensor(batch_size) \
            .type_as(dtype)
        self.random_action_cpu = torch.LongTensor(batch_size)
        self.reward_aux = torch.LongTensor(batch_size).type_as(dtype)
        self.pig_moved = torch.LongTensor(batch_size).type_as(dtype)
        self.pig_act = torch.LongTensor(batch_size).type_as(dtype)

        # Environment variables
        self.max_pig_moves = cfg.pig_max_moves
        self.pig_ai = VillagePeoplePig(batch_size, use_cuda)

        self.step_moves_no = 0
        self.last_step_agent_idx = -1
        self.current_agent_idx = -1

        """Data for representations"""
        self.selected_state_builders = dict(STATE_BUILDERS)
        self.selected_state_builders[agent0_builder_type] = True
        self.selected_state_builders[agent1_builder_type] = True

        state_builders = {
            BUILDER_18BINARY: self._state_18binary,
            BUILDER_MALMO: self._state_malmo,
            BUILDER_SYMBOLIC: self._state_symbolic,
            BUILDER_SIMPLE: self._state_simple
        }

        self.agent0_state_builder = state_builders[agent0_builder_type]
        self.agent1_state_builder = state_builders[agent1_builder_type]

        # Malmo default map representation data
        if self.selected_state_builders[BUILDER_MALMO] or \
                self.selected_state_builders[BUILDER_SYMBOLIC]:
            self.malmo_states = malmo_states = list()
            self.malmo_boards = malmo_boards = list()
            self.malmo_st_agent0 = malmo_st_agent0 = list()
            self.malmo_st_agent1 = malmo_st_agent1 = list()
            self.malmo_st_pig = malmo_st_pig = list()

            if self.selected_state_builders[BUILDER_MALMO]:
                for i in range(batch_size):
                    malmo_states.append(deepcopy(WORLD_OBSERVATION))

                for i in malmo_states:
                    malmo_boards.append(i["board"])
                    malmo_st_agent0.append(i["entities"][0])
                    malmo_st_agent1.append(i["entities"][1])
                    malmo_st_pig.append(i["entities"][2])
            else:
                for i in range(batch_size):
                    malmo_states.append((
                        np.copy(BOARD),
                        deepcopy(WORLD_OBSERVATION["entities"])
                    ))

                for i in malmo_states:
                    malmo_boards.append(i[0])
                    malmo_st_agent0.append(i[1][0])
                    malmo_st_agent1.append(i[1][1])
                    malmo_st_pig.append(i[1][2])

            malmo_st_ag0_data = (malmo_boards, malmo_st_agent0)
            malmo_st_ag1_data = (malmo_boards, malmo_st_agent1)
            malmo_st_pig_data = (malmo_boards, malmo_st_pig)

            if self.selected_state_builders[BUILDER_MALMO]:
                # Observations for parent class (Important for
                # using malmo vizualization)
                self.world_observations = self.malmo_states[0]
                self._world_obs = self.malmo_states[0]
                self._world_obs["Yaw"] = self.malmo_st_agent1[0]["yaw"]

        else:
            malmo_st_ag0_data = None
            malmo_st_ag1_data = None
            malmo_st_pig_data = None

        # Current Environment representation data (One hot)
        self.board_onehot = self._board_one_hot.unsqueeze(0) \
            .expand(batch_size, 3, 9, 9).long().type_as(dtype)

        # Init Agents
        self.agent0 = VillagePeopleEnvAgent(ENV_AGENT_NAMES[0], self,
                                            self.agent0_state_builder,
                                            malmo_st_ag0_data, cfg)
        self.agent1 = VillagePeopleEnvAgent(ENV_AGENT_NAMES[1],
                                            self, self.agent1_state_builder,
                                            malmo_st_ag1_data, cfg)
        self.pig = VillagePeopleEnvAgent(ENV_TARGET_NAMES[0],
                                         self, None, malmo_st_pig_data, cfg)
        self.agents_list = [self.agent0, self.agent1]

    @property
    def available_actions(self):
        """
        Returns the number of actions available in this environment
        :return: Integer > 0
        """
        return self.action_no

    @property
    def frame(self):
        """
        Return the most recent frame from the environment
        :return: PIL Image representing the current environment
        """
        return None

    @property
    def state(self):
        next_ag_idx = self.next_agent_idx()
        state_representation = self.agents_list[next_ag_idx].agent_builder

        return state_representation(next_ag_idx)

    @property
    def done(self):
        """
        Done if we have caught the pig or exit map
        Return done if next last env act hast to be forward tot the agent
        """
        done = self.sent_done >= 2

        return done.byte()

    def reset(self, agent_type=None, agent_positions=None):
        """ Overrides reset() to allow changes in agent appearance between
        missions."""
        self.step_moves_no = 0
        self.last_step_agent_idx = -1
        self.current_agent_idx = -1

        reset_ans = self._reset(agent_type=None, agent_positions=None)

        return reset_ans

    def do(self, action_player, agent_idx=None):
        """
        Do the action
        If agent_idx is not specified
        we consider doing first action for Agent 0 -> the agent ->2
        """

        if agent_idx is None:
            agent_idx = (self.last_step_agent_idx + 1) % 2

        self.current_agent_idx = agent_idx

        if isinstance(action_player, int):
            action_player_ = torch.zeros(self.batch_size).long().cuda()
            action_player_[0] = action_player
            action_player = action_player_
            print(action_player)
            print(agent_idx)

        state, reward, done = self._do(action_player, agent_idx=agent_idx)

        if self._visualize:
            reward = reward[0]
            done = done[0]

        self.last_step_agent_idx = agent_idx
        self.current_agent_idx = -1
        return state, reward, done

    def is_valid(self, world_state):
        """ Pig Chase Environment is valid if the the board and entities
        are present """
        pass

    def next_agent_idx(self):
        return (self.current_agent_idx + 1) % 2

    def dist_states(self, coord_x, coord_y):
        dist = torch.abs(coord_x - coord_y).sum(1)
        return dist.view(-1)

    def random_action(self, include_no_act=True):
        do_nothing = 0 if include_no_act else 1
        random_action_storage = self.random_action_storage
        random_action_storage.copy_(self.random_batch(self.action_no -
                                                      do_nothing))
        return random_action_storage

    def random_batch(self, max):
        self.random_action_cpu.random_(max)
        return self.random_action_cpu

    def _reset(self, agent_type=None, agent_positions=None):
        self.it_no = 0

        self.done_max = False
        self.game_finished.fill_(0)

        final_state = True
        while final_state:
            self.agent0.reset()
            self.agent1.reset()
            self.pig.reset()

            # TODO Must find a faster way to start env in a non end state
            # crt_catch_pig = self.check_catch(self.agent0.agent_coord,
            #                                  self.agent1.agent_coord,
            #                                  self.pig.agent_coord,
            #                                  self.pig.agent_state)
            final_state = False  # crt_catch_pig.byte().any()

        self.update_observations(self.agent0)
        self.update_observations(self.agent1)
        self.update_observations(self.pig)

        self.sent_done.fill_(0)

        return self.state

    def update_observations(self, agent):
        # Update agent observations
        if self.selected_state_builders[BUILDER_18BINARY]:
            agent.update_agent_one_hot()
        if self.selected_state_builders[BUILDER_MALMO]:
            agent.update_malmo_state()
        if self.selected_state_builders[BUILDER_SYMBOLIC]:
            agent.update_symbolic_state()

    def _state_18binary(self, for_idx):
        onehot_size = self.agent_onehot_size

        # SHOULD concat board_onehot and agent states
        if for_idx == 0:
            first_board = self.agent0.agent_onehot
            second_board = self.agent1.agent_onehot
        else:
            first_board = self.agent1.agent_onehot
            second_board = self.agent0.agent_onehot

        pig_board = self.pig.agent_onehot

        state = torch.cat([self.board_onehot, first_board.view(onehot_size),
                           second_board.view(onehot_size),
                           pig_board.view(onehot_size)], 1)
        return state

    def _state_malmo(self, for_idx):
        state = self.malmo_states

        if self._visualize:
            state = state[0]
            self.world_observations = self.malmo_states[0]
            self._world_obs = self.malmo_states[0]
            self._world_obs["Yaw"] = self.malmo_st_agent1[0]["yaw"]

        return state

    def _state_symbolic(self, for_idx):
        return self.malmo_states

    def _state_simple(self, for_idx):
        direction_no = self.directions_no

        if for_idx == 0:
            first_agent = self.agent0
            second_agent = self.agent1
        else:
            first_agent = self.agent1
            second_agent = self.agent0

        pig = self.pig

        first_agent_data = torch.cat([first_agent.agent_state.unsqueeze(1),
                                      first_agent.agent_coord],
                                     1)
        second_agent_data = torch.cat([second_agent.agent_state.unsqueeze(1),
                                       second_agent.agent_coord],
                                      1)
        pig_data = torch.cat([pig.agent_state.unsqueeze(1),
                              pig.agent_coord],
                             1)

        return ([first_agent_data, second_agent_data, pig_data],
                self._board_one_hot)

    def check_catch(self, coord_agent0, coord_agent1, coord_pig, pig_state):
        directions_no = self.directions_no
        agent0_pig_dist = self.dist_states(coord_agent0, coord_pig)
        agent1_pig_dist = self.dist_states(coord_agent1, coord_pig)
        pig_danger = ((pig_state - pig_state % directions_no) / directions_no) \
            .unsqueeze(1)

        pig_danger = self._pig_danger_pos.gather(1, pig_danger).byte().view(-1)

        agent0_catch = (agent0_pig_dist == 1)
        agent1_catch = (agent1_pig_dist == 1)

        pig_exit = self.check_exit(pig_state)
        pig_out_catch = pig_exit.byte() & (agent0_catch | agent1_catch)

        true_catch = ((agent0_catch) & (agent1_catch) & pig_danger &
                      ((coord_agent0 == coord_agent1).sum(1) < 2))
        return (true_catch | pig_out_catch).long()

    def check_exit(self, agent_state):
        """Calculating reward will modify states of exit
        (from -pos -> neutral)"""
        return (agent_state >= self.exit_state_first_idx).long()

    def _do(self, action_player, agent_idx=None):
        """
        Do the action
        #RETURNS finished states in this move (not previously fininshed states)
        """
        action_player = action_player.clone()

        state = done = None

        agent0 = self.agent0
        agent0 = self.agent0
        agent1 = self.agent1
        pig = self.pig

        reward_catch = self.reward_catch
        reward = self.reward_aux
        reward.fill_(0)

        null_move = self.null_move

        prev_finish = self.game_finished

        # NULL ACTIONS FOR FINISHED GAMES
        action_player.mul_(1 - prev_finish).add_(null_move * prev_finish)

        # act agent_0 (Player)
        if agent_idx == 0:
            agent0.move_agent(action_player)
            acting_agent = agent0
            next_agent = agent1

        elif agent_idx == 1:
            agent1.move_agent(action_player)
            acting_agent = agent1
            next_agent = agent0

        # UPDATE DONE states received
        acting_agent.got_info()

        self.it_no = min(agent0.ag_step_no, agent1.ag_step_no)

        crt_catch_pig = self.check_catch(agent0.agent_coord, agent1.agent_coord,
                                         pig.agent_coord, pig.agent_state)
        crt_agent_exit = self.check_exit(acting_agent.agent_state)
        crt_agent_exit.sub_(prev_finish).clamp_(0, 1)

        game_finished = (prev_finish.byte() | crt_catch_pig.byte()
                         | crt_agent_exit.byte()).long()

        # Act pig - AI
        if self.max_pig_moves > torch.rand(1)[0]:
            new_pig_state = self.pig_ai.move_pig(pig.agent_state,
                                                 agent0.agent_state,
                                                 agent1.agent_state)
            new_pig_state.mul_(1 - game_finished)
            pig.agent_state.mul_(game_finished).add_(new_pig_state)

            pig.set_state(pig.agent_state)

            crt_catch_pig = self.check_catch(agent0.agent_coord,
                                             agent1.agent_coord,
                                             pig.agent_coord,
                                             pig.agent_state)

            game_finished = (game_finished.byte() |
                             crt_catch_pig.byte()).long()

        crt_catch_pig.sub_(prev_finish).clamp_(0, 1)

        ### Calculate reward for catching pig
        reward.add_(next_agent.prev_ag_act * self.reward_move)

        reward.add_(next_agent.prev_ag_catch_pig * reward_catch)
        reward.add_(next_agent.prev_ag_exit_map * self.reward_exit)

        done = next_agent.prev_ag_catch_pig.byte() \
               | next_agent.prev_ag_exit_map.byte()

        # CURRENT AGENT ACTIONS IN ENV
        if True:
            reward.add_(crt_catch_pig * reward_catch)
            done = done | crt_agent_exit.byte() | crt_catch_pig.byte()
            # COPY STATE BEFOR MODIFICATIONS

        if next_agent.ag_step_no == self.max_moves:
            done = 1 - next_agent.sent_done

        if not self.move_final_state:
            acting_agent.move_pre_final_state(game_finished)
            pig.move_pre_final_state(game_finished)

        # Update agent observations
        self.update_observations(acting_agent)
        self.update_observations(pig)

        if True:
            state = next_agent.agent_builder((agent_idx + 1) % 2)

        acting_agent.finished_move((1 - prev_finish), crt_catch_pig,
                                   crt_agent_exit)
        self.sent_done.mul_(0).add_(agent0.sent_done).add_(agent1.sent_done)

        next_agent.sent_info(done)
        self.game_finished = game_finished

        return state, reward, done.long()
