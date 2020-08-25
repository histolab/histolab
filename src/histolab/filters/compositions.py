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

from . import image_filters as imf
from . import morphological_filters as mof


def tile_tissue_mask_filters() -> imf.Compose:
    """Return a filters composition to get a binary mask to estimate tissue in a tile.

    Returns
    -------
    imf.Compose
        Filters composition
    """
    filters = imf.Compose(
        [
            imf.RgbToGrayscale(),
            imf.OtsuThreshold(),
            mof.BinaryDilation(),
            mof.BinaryFillHoles(structure=np.ones((5, 5))),
        ]
    )
    return filters
