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
        else:
            raise FilterCompositionError(
                f"Filters composition for the class {cls_.__name__} is not available"
            )


class _SlideFiltersComposition(FiltersComposition):
    @lazyproperty
    def tissue_mask_filters(self) -> imf.Compose:
        """Return a filters composition to get a binary mask to estimate tissue in a slide.

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
        """Return a filters composition to get a binary mask to estimate tissue in a tile.

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
