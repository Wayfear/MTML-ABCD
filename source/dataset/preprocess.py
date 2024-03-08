import numpy as np
from omegaconf import DictConfig


class StandardScaler:
    """
    Standard the input
    """

    global mean, std

    def transform(self, data: np.array):
        return (data - self.mean) / self.std

    def inverse_transform(self, data: np.array):
        return (data * self.std) + self.mean
    
    def to_cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()


def reduce_sample_size(config: DictConfig, *args):
    sz = args[0].shape[0]
    used_sz = int(sz * config.datasz.percentage)
    return [d[:used_sz] for d in args]
