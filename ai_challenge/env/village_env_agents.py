# Village People, 2017

""""
This wrappers help communication between agents and different environments.
They can handle either batches of games or individual games.
"""

import torch
from .artificial_malmo import ALL_POS, VALID_POSITIONS, DIRECTIONS_MOVE, \
    EXIT_POINTS, ENV_ACTIONS
from .artificial_malmo import add_tuple, cost_rotate
from .artificial_malmo import ENV_NULL_ACTION_IDX, BUILDER_18BINARY, \
    BUILDER_OBSERVATION_DATA_TYPES, BUILDER_SYMBOLIC, BUILDER_SIMPLE
import numpy as np


class VillagePeopleEnvPlayerParent(object):
    """ Should be extended and implemented for each agent """

    def __init__(self, batch_size, builder_type,
                 use_cuda=True, batch_processing=True,
                 slice_finished_games=True):

        self.batch_size = batch_size
        self.batch_processing = batch_processing
        self.slice_finished_games = slice_finished_games
        self.builder_type = builder_type

        self.null_action = ENV_NULL_ACTION_IDX

        self.dtype = torch.zeros(0).long()
        if use_cuda:
            self.dtype = self.dtype.cuda()
        dtype = self.dtype

        self.action_holder = torch.LongTensor(batch_size).type_as(dtype)

        self.obs_type_list = False
        if isinstance(BUILDER_OBSERVATION_DATA_TYPES[builder_type], list):
            self.obs_type_list = True

    def act(self, obs, reward, done, ignore_games, **kwargs):
        actions = None

        if self.batch_processing:
            if self.slice_finished_games:
                ignore_games_idx = ignore_games.nonzero().view(-1)
                obs_ = obs.index_select(0, ignore_games_idx)
                reward_ = reward.index_select(0, ignore_games_idx)
                done_ = done.index_select(0, ignore_games_idx)

                action_holder = self.action_holder
                action_holder.fill_(self.null_action)

                actions = self._act(obs_, reward_, done_, **kwargs)

                action_holder.scatter_(0, ignore_games_idx, actions)
                actions = action_holder
            else:
                actions = self._act(obs, reward, done, ignore_games, **kwargs)
        else:
            # Sliceing implied when not sending obs in batch
            action_holder = self.action_holder
            action_holder.fill_(self.null_action)
            for ix in range(len(obs)):
                if ignore_games[ix]:
                    action = self._act(obs[ix], reward[ix], done[ix], game=ix)
                    action_holder[ix] = action

            actions = action_holder

        return actions

    def reset(self):
        raise NotImplementedError('Subclasses must override reset')

    def _act(self, obs, reward, done, game=None, **kwargs):
        """game -> specifies game index. If None -> entire batch is sent"""
        raise NotImplementedError('Subclasses must override reset')


class VillagePeopleEnvRandomAgent(VillagePeopleEnvPlayerParent):
    def __init__(self, name, config):
        self.name = name
        use_cuda = config.general.use_cuda

        self.dtype = torch.zeros(0).long()
        if use_cuda:
            self.dtype = self.dtype.cuda()

        super(VillagePeopleEnvRandomAgent, self) \
            .__init__(config.general.batch_size, BUILDER_18BINARY,
                      use_cuda=config.general.use_cuda,
                      batch_processing=True, slice_finished_games=False)

    def _act(self, obs, reward, done, game=None, **kwargs):
        actions = torch.LongTensor(reward.size(0))
        actions.random_(3)
        return actions.type_as(self.dtype)

    def reset(self):
        pass


class MalmoAgentWrapper(VillagePeopleEnvPlayerParent):
    """Used for symbolic representation"""

    def __init__(self, agent_class, agent_name, config):
        self.agent_class = agent_class
        self.agent_p_focus = config.envs.simulated.p_focus
        self.batch_size = batch_size = config.general.batch_size
        self.agent_name = agent_name
        self.agents = []

        super(MalmoAgentWrapper, self) \
            .__init__(batch_size, BUILDER_SYMBOLIC,
                      use_cuda=config.general.use_cuda,
                      batch_processing=False)

    def _act(self, obs, reward, done, game=None, **kwargs):
        assert game is not None, "Must send game index"
        agent = self.agents[game]
        action = agent.act(obs, reward, done, **kwargs)
        return action

    def reset(self):
        agent_name = self.agent_name
        agent_class = self.agent_class
        batch_size = self.batch_size
        self.agents = [agent_class(agent_name, p_focused_new=self.agent_p_focus)
                       for i in range(batch_size)]


class Binary18BatchAgentWrapper(VillagePeopleEnvPlayerParent):
    """Used for Binary maps representation"""

    def __init__(self, agent, agent_name, config, is_training=True):
        self.agent = agent
        self.batch_size = batch_size = config.general.batch_size
        self.agent_name = agent_name
        self.is_training = is_training
        use_cuda = config.general.use_cuda

        self.dtype = torch.zeros(0).long()
        if use_cuda:
            self.dtype = self.dtype.cuda()

        super(Binary18BatchAgentWrapper, self) \
            .__init__(batch_size, BUILDER_18BINARY,
                      use_cuda=config.general.use_cuda,
                      batch_processing=True, slice_finished_games=True)

    def _act(self, obs, reward, done, game=None, **kwargs):
        action = self.agent.act(obs.type_as(self.dtype),
                                reward.type_as(self.dtype),
                                done.type_as(self.dtype),
                                self.is_training, **kwargs)
        return action

    def set_is_training(self, value):
        assert value, bool
        self.is_training = value

    def reset(self):
        self.agent.reset()


class ChallengeAgentState():
    def __init__(self, obs):
        self.state = obs.narrow(1, 0, 1)
        self.coord = obs.narrow(1, 1, 2)
        self.direction = self.state % len(DIRECTIONS_MOVE)

    def to_numpy(self):
        self.state = self.state.cpu().numpy()
        self.coord = self.coord.cpu().numpy()
        self.direction = self.direction.cpu().numpy()


class VillagePeopleEnvChallengeAgent(VillagePeopleEnvPlayerParent):
    def __init__(self, agent_class, name, map, config):
        """ Challenge agent with strategy Catch/ exit/ random"""

        self.agent_class = agent_class
        self.agent_p_focus = config.envs.simulated.p_focus
        self.batch_size = batch_size = config.general.batch_size
        self.agent_name = name
        self.agents = []

        use_cuda = config.general.use_cuda

        self.dtype = torch.zeros(0).long()
        if use_cuda:
            self.dtype = self.dtype.cuda()

        self.agents_type = torch.zeros(self.batch_size).type_as(self.dtype)
        config.agents_type = self.agents_type

        super(VillagePeopleEnvChallengeAgent, self) \
            .__init__(config.general.batch_size, BUILDER_SIMPLE,
                      use_cuda=config.general.use_cuda,
                      batch_processing=True, slice_finished_games=False)

        map = map[1] + map[2] * 2

        self.orig_map = map.cpu().numpy()

    def _act(self, obs, reward, done, ignore_games, **kwargs):
        # Get CPU
        (agent0_obs, agent1_obs, pig_obs), map = obs
        map = self.orig_map

        action_holder = self.action_holder
        action_holder.fill_(self.null_action)

        # Slice games in torch
        play_games_idx = ignore_games.nonzero().view(-1)

        if play_games_idx.numel() <= 0:
            return action_holder

        agent0_obs_ = agent0_obs.index_select(0, play_games_idx)
        agent1_obs_ = agent1_obs.index_select(0, play_games_idx)
        pig_obs_ = pig_obs.index_select(0, play_games_idx)

        agent0 = ChallengeAgentState(agent0_obs_)
        agent1 = ChallengeAgentState(agent1_obs_)
        pig = ChallengeAgentState(pig_obs_)

        # Transform to numpy
        agent0.to_numpy()
        agent1.to_numpy()
        pig.to_numpy()

        for ix in range(play_games_idx.size(0)):
            game_idx = play_games_idx[ix]
            agent = self.agents[game_idx]
            obs_ag0 = (agent0.coord[ix, 0], agent0.coord[ix, 1],
                       agent0.direction[ix])
            obs_ag1 = (agent1.coord[ix, 0], agent1.coord[ix, 1],
                       agent1.direction[ix])
            obs_pig = (pig.coord[ix, 0], pig.coord[ix, 1],
                       pig.direction[ix])
            obs_agent = ([obs_ag0, obs_ag1, obs_pig], map)

            action = agent.act(obs_agent, reward[game_idx], done[game_idx])
            action_holder[game_idx] = action

        return action_holder

    def reset(self):
        agent_name = self.agent_name
        agent_class = self.agent_class
        batch_size = self.batch_size
        self.agents = [agent_class(agent_name, p_focus=self.agent_p_focus)
                       for i in range(batch_size)]
        for i in range(len(self.agents)):
            self.agents_type[i] = self.agents[i].current_agent._type


class VillagePeopleEnvSmartAgent(VillagePeopleEnvPlayerParent):
    """
    Heuristic challenge agent
    Takes into account other agents & pig coordinates. Distance to exit
    and smart distances to 'attack' the pig
    """

    directions_no = len(DIRECTIONS_MOVE)
    valid_pos_no = len(VALID_POSITIONS)
    exit_pos_no = len(EXIT_POINTS)
    all_pos_no = len(ALL_POS)
    action_no = len(ENV_ACTIONS)
    valid_states_no = valid_pos_no * directions_no
    exit_states_no = exit_pos_no * directions_no
    exit_state_first_idx = valid_states_no
    states_no = valid_states_no + exit_states_no

    _cost_move = torch.zeros(states_no, directions_no).long()
    _coord_move = torch.zeros(states_no, directions_no * 2).long()
    _cost_move.fill_(3333)
    _coord_move.fill_(-1)
    exit0 = torch.LongTensor(EXIT_POINTS[0])
    exit1 = torch.LongTensor(EXIT_POINTS[1])
    _neighbours = []

    for state_idx in range(states_no):
        valid_pos = state_idx // directions_no
        direction = state_idx % directions_no

        for im, pos_move in enumerate(DIRECTIONS_MOVE):
            new_pos = add_tuple(ALL_POS[valid_pos], pos_move)

            _coord_move[state_idx, im * 2] = new_pos[0]
            _coord_move[state_idx, im * 2 + 1] = new_pos[1]

            # Calculate cost of move + rotation
            if new_pos in ALL_POS:
                cost = 1 + cost_rotate(direction, im)
                _cost_move[state_idx, im] = cost

    for ix in range(len(ALL_POS)):
        neigh = list()
        for im, pos_move in enumerate(DIRECTIONS_MOVE):
            new_pos = add_tuple(ALL_POS[ix], pos_move)
            if new_pos in ALL_POS:
                neigh.append(np.array(new_pos))

        _neighbours.append(neigh)

    def __init__(self, name, config):
        """Challenge agent with strategy Catch/ exit/ random"""

        self.name = name
        use_cuda = config.general.use_cuda

        self.dtype = torch.zeros(0).long()
        if use_cuda:
            self.dtype = self.dtype.cuda()
            self._cost_move = self._cost_move.cuda()
            self._coord_move = self._coord_move.cuda()
            self.exit0 = self.exit0.cuda()
            self.exit1 = self.exit1.cuda()

        super(VillagePeopleEnvSmartAgent, self) \
            .__init__(config.general.batch_size, BUILDER_SIMPLE,
                      use_cuda=config.general.use_cuda,
                      batch_processing=True, slice_finished_games=False)

    def dist_coord(self, coord_x, coord_y):
        dist = torch.abs(coord_x - coord_y).sum(1)
        return dist.view(-1)

    def cost_move_batch(self, agent, coord):
        batch_size = coord.size(0)
        possible_coord = self.directions_no * 2

        coord_move = self._coord_move.gather(0, agent.state.
                                             expand(batch_size, possible_coord))
        cost_turn = self._cost_move.gather(0, agent.state.expand(batch_size,
                                                                 self.directions_no))

        coord_move = coord_move.view(-1, 2)
        coord_target = coord.unsqueeze(1).expand(batch_size, 4, 2) \
            .contiguous().view(-1, 2)
        dist = self.dist_coord(coord_move, coord_target).view(batch_size, 4)

        all_cost = cost_turn + dist

        return all_cost

    def dist_coord_np(self, x, y):
        return np.abs(x - y).sum()

    def get_random_min(self, ls):
        min_v = np.inf
        min_ix = -1
        for ix, val in enumerate(ls):
            if val < min_v:
                min_v, min_ix = val, ix
            elif val == min_v:
                if np.random.rand() < 0.5:
                    min_v, min_ix = val, ix
        return (min_v, min_ix)

    def min_cost_move(self, state, coord):
        cost = []

        for ix in range(self.directions_no):
            cost_turn = self._cost_move[state, ix]
            cost_move = self.dist_coord_np(
                self._coord_move[state, ix * 2:ix * 2 + 2].cpu().numpy(),
                coord
            )
            cost.append(cost_move + cost_turn)

        return self.get_random_min(cost)

    def _act(self, obs, reward, done, ignore_game, **kwargs):
        # Get CPU
        (agent0_obs, agent1_obs, pig_obs), map = obs
        agent0 = ChallengeAgentState(agent0_obs)
        agent1 = ChallengeAgentState(agent1_obs)
        pig = ChallengeAgentState(pig_obs)

        cost_exit0 = self.cost_move_batch(agent0, self.exit0.unsqueeze(0)
                                          .expand_as(agent0.coord))
        cost_exit1 = self.cost_move_batch(agent0, self.exit1.unsqueeze(0)
                                          .expand_as(agent0.coord))
        actions = torch.LongTensor(reward.size(0))

        for g in range(reward.size(0)):
            # Calculate Ag1 cost to neighbours
            pig_coord_idx = pig.state[g][0] // self.directions_no
            pig_neigh = self._neighbours[pig_coord_idx]
            ag0_st = agent0.state[g][0]
            ag1_st = agent1.state[g][0]

            # Calculate min cost to catch pig considering a rational agent
            size_ = len(pig_neigh)
            ag0_min_dist = np.zeros(size_)
            ag0_min_dist_arg = [-1] * size_
            ag1_min_dist = np.zeros(size_)
            ag1_min_dist_arg = [-1] * size_

            for i, pig_neigh_coord in enumerate(pig_neigh):
                ag0_min_dist[i], ag0_min_dist_arg[i] = self.min_cost_move(
                    ag0_st, pig_neigh_coord)
                ag1_min_dist[i], ag1_min_dist_arg[i] = self.min_cost_move(
                    ag1_st, pig_neigh_coord)

            # -- Get min cost combination to surround THE PIG
            sorted_ag1_idx = sorted(range(len(ag1_min_dist)),
                                    key=lambda k: ag1_min_dist[k])
            best_move_idx = -1
            min_catch_cost = np.inf
            for i in range(len(pig_neigh)):
                if i == sorted_ag1_idx[0]:
                    # Same minimum pos for ag1
                    min_ag1_dist = ag1_min_dist[sorted_ag1_idx[1]]
                else:
                    min_ag1_dist = ag1_min_dist[sorted_ag1_idx[0]]

                cost_ = ag0_min_dist[i] + min_ag1_dist

                if cost_ < min_catch_cost:
                    min_catch_cost, min_ag0_move = cost_, i

            # -- Min cost to catch pig
            catch_min_cost = ag0_min_dist[best_move_idx]
            catch_min_cost_dir = ag0_min_dist_arg[best_move_idx]

            # -- Opponent move
            if best_move_idx == sorted_ag1_idx[0]:
                # Same minimum pos for ag1
                best_move_ag1_idx = sorted_ag1_idx[1]
            else:
                best_move_ag1_idx = sorted_ag1_idx[0]

        actions = torch.LongTensor(reward.size(0))
        # actions.random_(3)
        return actions.type_as(self.dtype)

    def reset(self):
        self.step_no = 0
