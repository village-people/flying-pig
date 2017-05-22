# Village People, 2017

from termcolor import colored as clr
from utils.utils import ModelUtil

class ReportingAgent(object):

    def __init__(self, cfg):
        self.name = "Agent"
        self.model_utils = ModelUtil(cfg.model.saving)

    def print_info(self, message):
        print(clr("[{:s}] ".format(self.name), "green") + message)

    def _save_model():
        pass
