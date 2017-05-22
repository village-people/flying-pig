# Village People, 2017

import torch.multiprocessing as mp

class AtomicStatistics(object):
    def __init__(self):
        self.episodes = mp.Value('i', 0)
        self.frames = mp.Value('i', 0)
        self.lock = mp.Lock()

    def inc_episodes(self, value=1):
        with self.lock:
            self.episodes.value += value
            new_value = self.episodes.value
        return new_value

    def get_episodes(self):
        return self.episodes.value

    def inc_frames(self, value=1):
        with self.lock:
            self.frames.value += value
            new_value = self.frames.value
        return new_value

    def get_frames(self):
        return self.frames.value
