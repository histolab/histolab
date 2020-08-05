import numpy as np

from histolab.tile import Tile


class RandomScorer(object):
    def __call__(self, tile: Tile) -> float:
        return np.random.random()
