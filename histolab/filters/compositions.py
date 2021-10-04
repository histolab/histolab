# encoding: utf-8

# ------------------------------------------------------------------------
# Copyright 2020 All Histolab Contributors
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

import numpy as np

from ..exceptions import FilterCompositionError
from ..util import lazyproperty
from . import image_filters as imf
from . import morphological_filters as mof


class FiltersComposition:
    """Provide appropriate filters compositions based on the ``cls_`` parameter.

    Arguments
    ---------
    cls_ : type, {Tile, Slide}
        The class to get the appropriate filters composition for


    Example:
        >>> from histolab.filters.compositions import FiltersComposition
        >>> from histolab.slide import Slide
        >>> from histolab.tile import Tile
        >>> filters_slide = FiltersComposition(Slide).tissue_mask_filters
        >>> filters_tile = FiltersComposition(Tile).tissue_mask_filters
    """

    def __new__(cls: type, cls_):
        if not cls_:
            raise FilterCompositionError("cls_ parameter cannot be None")
        FiltersSubCls = {
            "Tile": _TileFiltersComposition,
            "Slide": _SlideFiltersComposition,
        }.get(cls_.__name__)
        if FiltersSubCls:
            instance = super(FiltersComposition, FiltersSubCls).__new__(FiltersSubCls)
            return instance
        raise FilterCompositionError(
            f"Filters composition for the class {cls_.__name__} is not available"
        )

    @lazyproperty
    def tissue_mask_filters(self) -> imf.Compose:
        """Return filters composition based on the ``cls_`` parameter.


        Returns
        -------
        imf.Compose

            If the ``cls_`` parameter is the class ``Slide`` the returned filters chain
            is composed of:

            - `image_filters.RgbToGrayscale()
              <image_filters.html#histolab.filters.image_filters.RgbToGrayscale>`_

            - `image_filters.OtsuThreshold()
              <image_filters.html#histolab.filters.image_filters.OtsuThreshold>`_

            - `morphological_filters.BinaryDilation()
              <morphological_filters.html#histolab.filters.morphological_filters.BinaryDilation>`_

            - `morphological_filters.RemoveSmallHoles()
              <morphological_filters.html#histolab.filters.morphological_filters.RemoveSmallHoles>`_

            - `morphological_filters.RemoveSmallObjects()
              <morphological_filters.html#histolab.filters.morphological_filters.RemoveSmallObjects>`_

            If the ``cls_`` parameter is the class ``Tile`` the returned filters chain
            is composed of:

            - `image_filters.RgbToGrayscale()
              <image_filters.html#histolab.filters.image_filters.RgbToGrayscale>`_

            - `image_filters.OtsuThreshold()
              <image_filters.html#histolab.filters.image_filters.OtsuThreshold>`_

            - `morphological_filters.BinaryDilation()
              <morphological_filters.html#histolab.filters.morphological_filters.BinaryDilation>`_

            - `morphological_filters.BinaryFillHoles(structure=np.ones((5, 5)))
              <morphological_filters.html#histolab.filters.morphological_filters.BinaryFillHoles>`_
        """
        raise NotImplementedError("Must be implemented by each subclass")


class _SlideFiltersComposition(FiltersComposition):
    @lazyproperty
    def tissue_mask_filters(self) -> imf.Compose:
        """Filters composition for slide's tissue estimation.

        Returns
        -------
        imf.Compose
            Filters composition
        """
        return imf.Compose(
            [
                imf.RgbToGrayscale(),
                imf.OtsuThreshold(),
                mof.BinaryDilation(),
                mof.RemoveSmallHoles(),
                mof.RemoveSmallObjects(),
            ]
        )


class _TileFiltersComposition(FiltersComposition):
    @lazyproperty
    def tissue_mask_filters(self) -> imf.Compose:
        """Filters composition for tile's tissue estimation.

        Returns
        -------
        imf.Compose
            Filters composition
        """
        return imf.Compose(
            [
                imf.RgbToGrayscale(),
                imf.OtsuThreshold(),
                mof.BinaryDilation(),
                mof.BinaryFillHoles(structure=np.ones((5, 5))),
            ]
        )
