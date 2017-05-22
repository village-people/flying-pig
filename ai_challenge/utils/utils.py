# Village People, 2017

import torch
import os

def read_config():
    import yaml
    import argparse

    def to_namespace(d):
        n = argparse.Namespace()
        for k, v in d.items():
            setattr(n, k, to_namespace(v) if isinstance(v, dict) else v)
        return n

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-cf" "--config_file",
                            default="basic",
                            dest="config_file",
                            help="Configuration file.")
    args = arg_parser.parse_args()
    with open("./configs/{:s}.yaml".format(args.config_file)) as f:
        config_data = yaml.load(f, Loader=yaml.SafeLoader)

    return to_namespace(config_data)

def parse_clients_args(endpoints):
    """ Return an array of tuples (ip, port) extracted from ip:port string
        :param args_clients:
        :return:
    """
    return [str.split(str(client), ':') for client in endpoints]


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ModelUtil(object):
    def __init__(self, cfg):
        self.save_path = cfg.save_folder + "/" + cfg.save_prefix
        self.arch_name = cfg.name
        self.max_r_frame = -25
        self.max_r_ep = -25

        if not os.path.exists(cfg.save_folder):
            print("Creating checkpoints folder...")
            os.makedirs(cfg.save_folder)

    def save_model(self, r_frame, r_ep, iteration, save_only_min=False):
        if not save_only_min:
            torch.save({
                'iteration': iteration,
                'arch': self.arch_name,
                'state_dict': self.model.state_dict(),
                'r_frame': r_frame,
                'r_ep': r_ep,
                'min_r_ep': self.max_r_ep,
                'min_r_frame': self.max_r_frame
            }, (self.save_path + "_{}".format(iteration)))

        if r_ep > self.max_r_ep:
            self.max_r_ep
        if r_frame > self.max_r_frame:
            self.max_r_frame

    def loadModelFromFile(self, path):
        model = self.model

        if os.path.isfile(path):
            print("=> loading checkpoint '{}'".format(path))
            checkpoint = torch.load(path)
            iteration = checkpoint['iteration']
            reward = checkpoint['reward']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (iteration {} -- reward: {})"
                  .format(path, iteration, reward))
        else:
            print("=> no checkpoint found at '{}'".format(path))

    def set_model(self, model):
        self.model = model
