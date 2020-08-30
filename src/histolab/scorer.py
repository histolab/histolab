from abc import abstractmethod

import numpy as np

from .filters import image_filters as imf
from .filters import morphological_filters as mof
from .filters.util import mask_difference
from .tile import Tile

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


class NucleiScorer(Scorer):
    """Implement a Scorer that estimates the presence of nuclei in the tile.

    An higher presence of nuclei is associated with an higher scorer.
    """

    def __call__(self, tile: Tile, alpha: float = 0.6) -> float:
        tissue_ratio = tile.tissue_ratio

        filters_raw_nuclei = imf.Compose([imf.HematoxylinChannel(), imf.YenThreshold()])
        filters_nuclei_cleaner = imf.Compose(
            [imf.HematoxylinChannel(), imf.YenThreshold(), mof.WhiteTopHat()]
        )

        mask_raw_nuclei = np.array(tile.apply_filters(filters_raw_nuclei).image)
        mask_nuclei_clean = np.array(tile.apply_filters(filters_nuclei_cleaner).image)

        mask_nuclei = mask_difference(mask_raw_nuclei, mask_nuclei_clean)
        nuclei_ratio = np.count_nonzero(mask_nuclei) / (tissue_ratio * mask_nuclei.size)

        return nuclei_ratio * alpha + tissue_ratio * (1 - alpha)
