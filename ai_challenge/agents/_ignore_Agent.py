import torch
from argparse import ArgumentParser, Namespace
import logging
import os
from collections import namedtuple
import random
import torch

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))

class AgentMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.reset()

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity


    def pushNewObservation(self, observation, reward, done, info, action):
        if self._lastState is None:
            self._lastState = observation
        else:
            next_state = observation

            self.push(self._lastState, action, next_state, reward, done)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def last(self):
        return [] if len(self.memory) <= 0 else [self.memory[-1]]

    def all(self):
        return self.memory[:]

    def reset(self):
        self.memory = []
        self.position = 0
        self._lastState = None

    def __len__(self):
        return len(self.memory)


class AgentModel:
    _model = None
    _optimizer = None
    _criterion = None
    _loaded = False

    def __init__(self, saveFolder, modelName, logger=None, modelSaveSuffix=""):
        self._model = None
        self._optimizer = None
        self.modelName = modelName
        self.modelPath = os.path.join(saveFolder, modelName + modelSaveSuffix +
                                      ".tar")
        self.bestModelPath = os.path.join(saveFolder, modelName +
                                          modelSaveSuffix + "_best.tar")
        self._maxMeanReward = -1

        if logger is None:
            self.logger = logging.getLogger(modelName + "_" + modelSaveSuffix)
        else:
            self.logger = logger

    def loadModel(self, model, optimizer, criterion):
        self._model = model
        self._optimizer = optimizer
        self._criterion = criterion
        self._loaded = True

    def loaded(self):
        return self._loaded

    def saveModel(self, epoch, meanReward):
        if self._model is None:
            self.logger.info("No model to save")
            return

        if meanReward > self._maxMeanReward:
            self._maxMeanReward= meanReward
            torch.save({
                'epoch': epoch + 1,
                'arch': self.modelName,
                'state_dict': self._model.state_dict(),
                'meanReward': meanReward,
                'bestMeanReward': self._maxMeanReward
            }, self.bestModelPath)

        torch.save({
            'epoch': epoch + 1,
            'arch': self.modelName,
            'state_dict': self._model.state_dict(),
            'mean_reward': meanReward,
            'best_mean_reward': self._maxMeanReward
        }, self.modelPath)

    def loadModelFromFile(self, path):
        if os.path.isfile(path):
            self.logger.info("=> loading checkpoint '{}'".format(path))
            checkpoint = torch.load(path)
            self.start_epoch = checkpoint['epoch']
            self._maxMeanReward = checkpoint['best_mean_reward']
            self._model.load_state_dict(checkpoint['state_dict'])
            self.logger.info("=> loaded checkpoint '{}' (epoch {})"
                  .format(path, checkpoint['epoch']))
        else:
            self.logger.debug("=> no checkpoint found at '{}'".format(path))

    def modelReport(self):
        pass

class Agent:

    def __init__(self, name, cfg):
        self.modelDataType = cfg.general.use_cuda
        self.saveFolder = cfg.general.save_folder
        self.logger = logging.getLogger(name + "_" + str(self.agentID))
        self.sharedModel = cfg.model.shared
        self.transitionMemory = cfg.model.transition_memory

        self._modelClass = None
        self._memory = AgentMemory(self.transitionMemory)
        self._crtStep = 0
        self._crtEpoch = 0

        self._useCUDA = False
        if self.modelDataType:
            self.dtype = torch.cuda.FloatTensor
            self._useCUDA = True
        else:
            self.dtype = torch.FloatTensor


        #Instantiate Agents model
        if self.sharedModel:
            if hasattr(self.env, "_agentsModel"):
                self._modelClass = self.env._agentsModel
            else:
                self._modelClass = AgentModel(self.saveFolder, self.name,
                                              logger=self.logger)
                self.env._agentsModel = self._modelClass
        else:
            self._modelClass = AgentModel(self.saveFolder, self.name,
                                          logger=self.logger,
                                          modelSaveSuffix=str(self.agentID))

    """
    Baseline methods.
    Should not be overridden when extending
    """
    def __post_init__(self):
        if not (self.sharedModel and self._modelClass.loaded()):
            self._createLearningArchitecture()
            self._modelClass._model.type(self.dtype)
            self.logger.info("Loaded architecture")


    def act(self, observation, reward, done, is_training):
        self._crtStep += 1
        observation = observation
        reward = reward
        action = self._act(observation, reward, done, is_training)

        self._postAction()

        self._memory.pushNewObservation(observation, reward, done, None, action)
        self._optimizeModel()

        return action.view(-1)

    def restart(self):
        """
        Called when game restarts
        """
        self._lastState = None
        self._restart()


    def epochFinished(self):
        """
        Called after end of training epoch
        """
        self._crtEpoch += 1
        self._epochFinished()
        pass


    def report(self):
        """
        Should log internal information
        """
        self._modelClass.modelReport()

        self._report()


    def saveModel(self, epoch, meanReward):
        """
        save model information
        """
        self._modelClass.saveModel(epoch, meanReward)

        self._saveModel(epoch, meanReward)


    """
    Classes extending this class should override only methods starting "_"
    to keep base class methods
    """
    def _act(self, observation, reward, done, info):
        pass

    def _reset(self):
        pass

    def _epochFinished(self):
        pass

    def _report(self):
        pass

    def _saveModel(self, epoch, meanReward):
        pass

    def _postAction(self):
        pass

    def _createLearningArchitecture(self):
        """
        Should create learning architecture
        #!!! Instantiate self._modelClass._model (sibling of nn.Module)
        #Instantiate other learning models
        """
        self._modelClass._model = None

    def _optimizeModel(self):
        """
        Is called after registering each new transition.
        """
        pass



