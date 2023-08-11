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

import numpy as np
import pytest

from histolab.tile import Tile

from ..fixtures import TILES
from ..util import load_expectation


class Describe_Tile:
    @pytest.mark.parametrize(
        "tile_img, expected_result",
        (
            (TILES.ALMOST_WHITE_1, True),
            (TILES.ALMOST_WHITE_2, True),
            (TILES.ALMOST_WHITE_IHC, True),
            (TILES.TISSUE_LEVEL0_4302_10273_4814_10785, False),
            (TILES.TISSUE_LEVEL0_7352_11762_7864_12274, False),
            (TILES.TISSUE_LEVEL2_1784_6289_5880_10386, False),
            (TILES.TISSUE_LEVEL2_3000_7666_7096_11763, False),
            (TILES.TISSUE_LEVEL2_4640_4649_8736_8746, False),
            (TILES.TISSUE_LEVEL2_4760_5241_8856_9338, False),
            (TILES.KIDNEY_IHC_LEVEL0_310_500_360_530, False),
            (TILES.KIDNEY_IHC_LEVEL0_340_520_390_550, False),
        ),
    )
    def it_knows_if_is_is_almost_white(self, tile_img, expected_result):
        tile = Tile(tile_img, None)

        is_almost_white = tile._is_almost_white

        assert is_almost_white == expected_result

    @pytest.mark.parametrize(
        "tile_image, expected_value",
        (
            (TILES.LIVER_LEVEl2_10907_7808_11707_8608, False),  # all tissue
            (TILES.LIVER_LEVEl2_20914_13715_21714_14515, True),  # some tissue
            (TILES.LIVER_LEVEl2_57138_8209_57938_9009, True),  # some tissue
            (TILES.LIVER_LEVEl2_38626_13514_39426_14315, False),  # no tissue
            (TILES.KIDNEY_IHC_LEVEL0_340_520_390_550, True),  # some tissue
            (TILES.KIDNEY_IHC_LEVEL0_350_530_360_540, True),  # all tissue
            (TILES.ALMOST_WHITE_IHC, False),  # no tissue
        ),
    )
    def it_knows_if_it_has_only_some_tissue(self, tile_image, expected_value):
        tile = Tile(tile_image, None)

        assert tile._has_only_some_tissue() == expected_value

    @pytest.mark.parametrize(
        "tile_img, expected_array",
        (
            (
                TILES.HIGH_NUCLEI_SCORE_LEVEL2,  # lot of background - big
                "mask-arrays/tile-tissue-mask-high-nuclei-score-level2",
            ),
            (
                TILES.TISSUE_LEVEL2_3000_7666_7096_11763,  # little background - big
                "mask-arrays/tile-tissue-mask-tissue-level2-3000-7666-7096-11763",
            ),
            (
                TILES.LIVER_LEVEl2_10907_7808_11707_8608,  # all tissue - very small
                "mask-arrays/tile-tissue-mask-liver-level2-10907-7808-11707-8608",
            ),
            (
                TILES.TISSUE_LEVEL0_4302_10273_4814_10785,  # all tissue - big
                "mask-arrays/tile-tissue-mask-tissue-level0-4302-10275-4814-10785",
            ),
            (
                TILES.HIGH_NUCLEI_SCORE_LEVEL0,  # all tissue - very big
                "mask-arrays/tile-tissue-mask-high-nuclei-score-level0",
            ),
            (
                TILES.VERY_LOW_NUCLEI_SCORE_RED_PEN_LEVEL1,  # all tissue - big
                "mask-arrays/very-low-nuclei-score-red-pen-level1",
            ),
        ),
    )
    def it_knows_its_tissue_mask(self, tile_img, expected_array):
        expected_value = load_expectation(expected_array, type_="npy")
        tile = Tile(tile_img, None)

        tissue_mask = tile.tissue_mask

        np.testing.assert_array_equal(tissue_mask, expected_value)
        assert isinstance(tissue_mask, np.ndarray) is True
