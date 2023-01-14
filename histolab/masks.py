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
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Iterable, List, Union

import numpy as np

from .filters.compositions import FiltersComposition
from .filters.image_filters import Compose, Filter
from .slide import Slide
from .tile import Tile
from .types import Region
from .util import (
    lazyproperty,
    method_dispatch,
    rectangle_to_mask,
    region_coordinates,
    regions_from_binary_mask,
)


class BinaryMask(ABC):
    """Generic object for binary masks.

    This object can be used to create a custom binary mask object.

    Example:
        >>> from histolab.slide import Slide
        >>> class MyCustomMask(BinaryMask):
        ...     def _mask(self, slide):
        ...         my_mask = np.array([0,1])
        ...         return my_mask
        >>> binary_mask = MyCustomMask()
        >>> slide = Slide("path/to/slide") # doctest: +SKIP
        >>> binary_mask(slide) # doctest: +SKIP
    """

    def __call__(self, slide):
        return self._mask(slide)

    @lazyproperty
    @abstractmethod
    def _mask(self, slide):  # pragma: no cover
        # This property will be supplied by the inheriting classes individually
        pass  # pragma: no cover


class BiggestTissueBoxMask(BinaryMask):
    r"""Object that represents the box containing the largest contiguous tissue area.

    .. figure:: https://user-images.githubusercontent.com/31658006/116549379-b14d0200-a8f5-11eb-85b1-46abc14c73bf.jpeg

    """  # noqa

    def __init__(self, *filters: Iterable[Filter]) -> None:
        """Create a new tissue mask and then retain the largest connected component.

        If custom image filters are specified, those are used instead the default ones.
        By default, the tissue within the slide or tile is automatically detected
        through a predefined chain of filters.

        Parameters
        ----------
        *filters : Iterable[Filter]
            Custom filters to derive a BiggestTissueBoxMask which overwrite the default
            pipeline.
        """
        self.custom_filters = filters

    @lru_cache(maxsize=100)
    def _mask(self, slide) -> np.ndarray:
        """Return the tissue box mask containing the largest contiguous tissue area.

        Parameters
        ----------
        slide : Slide
            The Slide from which to compute the extraction mask

        Returns
        -------
        mask: np.ndarray
            Binary mask of the box containing the largest contiguous tissue area.
            The dimensions are defined as the maximum between the 1/32th of the slide
            dimensions and the dimensions of the slide's thumbnail.
        """
        thumbnail = slide.thumbnail
        thumbnail_size = thumbnail.size[0] * thumbnail.size[1]
        scaled_image = slide.scaled_image()
        scaled_image_size = scaled_image.size[0] * scaled_image.size[1]
        scaled_slide = thumbnail if thumbnail_size > scaled_image_size else scaled_image

        if len(self.custom_filters) == 0:
            composition = FiltersComposition(Slide)
        else:
            composition = FiltersComposition(Compose, *self.custom_filters)

        scaled_mask = composition.tissue_mask_filters(scaled_slide)

        regions = regions_from_binary_mask(scaled_mask)
        biggest_region = self._regions(regions, n=1)[0]
        biggest_region_coordinates = region_coordinates(biggest_region)
        scaled_bbox_mask = rectangle_to_mask(
            scaled_slide.size[::-1], biggest_region_coordinates
        )
        return scaled_bbox_mask

    @staticmethod
    def _regions(regions: List[Region], n: int = 1) -> List[Region]:
        """Return the biggest ``n`` regions.

        Parameters
        ----------
        regions : List[Region]
            List of regions
        n : int, optional
            Number of regions to return, by default 1

        Returns
        -------
        List[Region]
            List of ``n`` biggest regions

        Raises
        ------
        ValueError
            If ``n`` is not between 1 and the number of elements of ``regions``
        """
        if n < 1:
            raise ValueError(f"Number of regions must be greater than 0, got {n}.")
        if n > len(regions):
            raise ValueError(
                f"n should be smaller than the number of regions [{len(regions)}], "
                f"got {n}"
            )

        sorted_regions = sorted(regions, key=lambda r: r.area, reverse=True)
        return sorted_regions[:n]


class TissueMask(BinaryMask):
    """Object that represent the whole tissue area mask."""

    def __init__(self, *filters: Iterable[Filter]) -> None:
        """
        Create a new tissue mask.

        If custom image filters are specified, those are used instead the default ones.
        By default, the tissue within the slide or tile is automatically detected
        through a predefined chain of filters.

        Parameters
        ----------
        *filters : Iterable[Filter]
            Custom filters to derive a TissueMask which overwrite the default pipeline.
        """
        self.custom_filters = filters

    def __call__(self, obj: Union[Slide, Tile]) -> np.ndarray:
        """Apply a predefined chain of filters to calculate the tissue area mask.

        The applied filters will be different based on the type of ``obj``, please see

        `filters.compositions.FiltersComposition <filters/compositions.html#histolab.filters.compositions.FiltersComposition>`_

        Parameters
        ----------
        obj : Union[Slide, Tile]
            ``Slide`` or ``Tile`` from which to compute the extraction mask.

        Returns
        -------
        np.ndarray
            Binary mask of the tissue area. If ``obj`` is a ``Slide``, the dimensions
            are defined as the maximum between the 1/32th of the slide dimensions and
            the dimensions of the slide's thumbnail; otherwise they are the same as the tile.

        See Also
        --------
        `filters.compositions.FiltersComposition <filters/compositions.html#histolab.filters.compositions.FiltersComposition>`_
        """  # noqa: E501

        return self._mask(obj)

    @lru_cache(maxsize=100)
    @method_dispatch
    def _mask(self, slide) -> np.ndarray:
        """Return a scaled version of the binary mask of the tissue area.

        Parameters
        ----------
        slide : Slide
            The Slide from which to compute the extraction mask

        Returns
        -------
        mask: np.ndarray
            Binary mask of the tissue area. The dimensions are defined as the maximum
            between the 1/32th of the slide dimensions and the dimensions of the slide's
            thumbnail.
        """
        thumbnail = slide.thumbnail
        thumbnail_size = thumbnail.size[0] * thumbnail.size[1]
        scaled_image = slide.scaled_image()
        scaled_image_size = scaled_image.size[0] * scaled_image.size[1]
        scaled_slide = thumbnail if thumbnail_size > scaled_image_size else scaled_image

        # Generate appropriate composition of filter
        if len(self.custom_filters) == 0:
            composition = FiltersComposition(Slide)
        else:
            composition = FiltersComposition(Compose, *self.custom_filters)

        scaled_mask = composition.tissue_mask_filters(scaled_slide)
        return scaled_mask

    @lru_cache(maxsize=100)
    @_mask.register(Tile)
    def _(self, tile: Tile) -> np.ndarray:
        """Return a scaled version of the binary mask of the tissue area.

        Parameters
        ----------
        tile : Tile
            The Tile from which to compute the extraction mask

        Returns
        -------
        mask: np.ndarray
            Binary mask of the tissue area. The dimensions are the ones of the tile.
        """

        # Check if calculating a customized mask is required.
        # Otherwise, fall back to the default one.
        if len(self.custom_filters) > 0:
            custom_filters = FiltersComposition(Compose, *self.custom_filters)
            return tile.calculate_tissue_mask(custom_filters)
        return tile.tissue_mask
