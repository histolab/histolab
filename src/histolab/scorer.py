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
    """Implement a Scorer that returns a random float score between 0 and 1."""

    def __call__(self, tile: Tile) -> float:
        """Return the random score associated with the tile.

        Parameters
        ----------
        tile : Tile
            The tile to calculate the score from.

        Returns
        -------
        float
            Random score ranging between 0 and 1.
        """
        return np.random.random()
