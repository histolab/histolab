# encoding: utf-8

# ------------------------------------------------------------------------
# Copyright 2022 All Histolab Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------

import operator
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
    """General scorer object

    .. automethod:: __call__
    """

    @abstractmethod
    def __call__(self, tile: Tile) -> float:
        pass  # pragma: no cover


class RandomScorer(Scorer):
    """Implement a Scorer that returns a random float score between 0 and 1.

    .. automethod:: __call__
    """

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


class CellularityScorer(Scorer):
    """Implement a Scorer that estimates the cellularity in an H&E-stained tile.

    This class deconvolves the hematoxylin channel and uses the fraction of tile
    occupied by hematoxylin as the cellularity score.

    Notice that this scorer is useful when tiles are extracted at a very low resolution
    with no artifacts; in this case,  using the``NucleiScorer()`` instead would not
    work well as nuclei are no discernible at low magnification.

    .. automethod:: __call__

    Parameters
    ----------
    consider_tissue : bool, optional
        Whether the detected tissue on the tile should be considered to compute the
        cellularity score. Default is True

    Notes
    -----
    If the tile presents artifacts (e.g., tissue folds, markers), the scorer cannot be
    fully trusted.
    """

    def __init__(self, consider_tissue: bool = True) -> None:
        self.consider_tissue = consider_tissue

    def __call__(self, tile: Tile) -> float:
        """Return the tile cellularity score.

        Parameters
        ----------
        tile : Tile
            The tile to calculate the score from.
        consider_tissue : bool
            Whether the cellularity score should be computed by considering the tissue
            on the tile. Default is True

        Returns
        -------
        float
            Cellularity score
        """

        filters_cellularity = imf.Compose(
            [
                imf.HematoxylinChannel(),
                imf.RgbToGrayscale(),
                imf.YenThreshold(operator.lt),
            ]
        )

        mask_nuclei = np.array(tile.apply_filters(filters_cellularity).image)

        return (
            np.count_nonzero(mask_nuclei) / np.count_nonzero(tile.tissue_mask)
            if self.consider_tissue
            else np.count_nonzero(mask_nuclei) / mask_nuclei.size
        )


class NucleiScorer(Scorer):
    r"""Implement a Scorer that estimates the presence of nuclei in an H&E-stained tile.

    This class implements an hybrid algorithm that combines thresholding and
    morphological operations to segment nuclei on H&E-stained histological images.

    The NucleiScorer class defines the score of a given tile t as:

    .. math::

        s_t = N_t\cdot \mathrm{tanh}(T_t) \mathrm{, } \; 0\le s_t<1

    where :math:`N_t` is the nuclei ratio on t, computed as number of white pixels on
    the segmented mask over the tile size, and :math:`T_t` the fraction of tissue in t.

    Notice that we introduced the hyperbolic tangent to bound the weight of the tissue
    ratio over the nuclei ratio.

    Notes
    -----
    If the tile presents artifacts (e.g., tissue folds, markers), the scorer cannot be
    fully trusted.

    .. automethod:: __call__
    """

    def __call__(self, tile: Tile) -> float:
        """Return the nuclei score associated with the tile.

        Parameters
        ----------
        tile : Tile
            The tile to calculate the score from.

        Returns
        -------
        float
            Nuclei score
        """

        filters_raw_nuclei = imf.Compose(
            [
                imf.HematoxylinChannel(),
                imf.RgbToGrayscale(),
                imf.YenThreshold(operator.lt),
            ]
        )
        filters_nuclei_cleaner = imf.Compose(
            [
                imf.HematoxylinChannel(),
                imf.RgbToGrayscale(),
                imf.YenThreshold(operator.lt),
                mof.WhiteTopHat(),
            ]
        )

        mask_raw_nuclei = np.array(tile.apply_filters(filters_raw_nuclei).image)
        mask_nuclei_clean = np.array(tile.apply_filters(filters_nuclei_cleaner).image)

        mask_nuclei = mask_difference(mask_raw_nuclei, mask_nuclei_clean)
        nuclei_ratio = np.count_nonzero(mask_nuclei) / mask_nuclei.size

        return nuclei_ratio * np.tanh(tile.tissue_ratio)
