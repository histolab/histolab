import pytest

from histolab.tile import Tile
from histolab.types import CP

from ..fixtures import TILES


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
        coords = CP(0, 512, 0, 512)
        tile = Tile(tile_img, coords)

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
        tile = Tile(tile_image, CP(5, 5, 5, 5))

        assert tile._has_only_some_tissue() == expected_value
