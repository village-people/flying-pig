# Copyright (c) 2017 Microsoft Corporation.
#
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
# BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# =====================================================================

from pig_chase.common import ENV_ACTIONS, ENV_AGENT_NAMES
from pig_chase.evaluation import PigChaseEvaluator

from env.VillagePeopleBuilders import PigChaseVillagePeopleBuilder18Binary, \
    PigChaseVillagePopleAgent

from models import get_model
from utils import read_config
from agents import get_agent
import torch
from utils import AtomicStatistics

if __name__ == '__main__':

    # -- Read configuration & Load agent FORCE BATCH == 1
    config = read_config()
    config.general.batch_size = 1
    config.general.use_cuda = False

    shared_model = get_model(config.model.name)(config.model)
    shared_model.eval()
    print(shared_model)

    if isinstance(config.model.load, str):
        checkpoint = torch.load(config.model.load)
        iteration = checkpoint['iteration']
        if "min_r_ep" in checkpoint:
            reward = checkpoint['min_r_ep']
            reward_r_frame = checkpoint['min_r_frame']
        else:
            reward = checkpoint['reward']

        print("Loading Model: {} Mean reward/ episode: {}"
              "".format(config.model.load, reward))
        shared_model.load_state_dict(checkpoint['state_dict'])
    shared_model.eval()
    shared_objects = {
        "model": shared_model,
        "stats_leRMS": AtomicStatistics()
    }

    agent_actor = get_agent(config.agent.type)(config.agent.name,
                                               ENV_ACTIONS, config,
                                               shared_objects)

    agent_role = 1
    agent = PigChaseVillagePopleAgent(ENV_AGENT_NAMES[agent_role],
                                      len(ENV_ACTIONS),
                                      agent_actor,
                                      use_cuda=config.general.use_cuda)

    clients = [('127.0.0.1', 10000), ('127.0.0.1', 10001)]

    eval = PigChaseEvaluator(clients, agent, agent,
                             PigChaseVillagePeopleBuilder18Binary(agent_role))
    eval.run()

    eval.save('My Exp 1', 'pig_chase_results.json')
