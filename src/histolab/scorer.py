from abc import abstractmethod

import numpy as np

from histolab.tile import Tile

try:
    from typing import Protocol, runtime_checkable
except ImportError:
    from typing_extensions import Protocol, runtime_checkable


@runtime_checkable
class Scorer(Protocol):
    @abstractmethod
    def __call__(self, tile: Tile) -> float:
        raise NotImplementedError


class RandomScorer(Scorer):
    def __call__(self, tile: Tile) -> float:
        return np.random.random()
