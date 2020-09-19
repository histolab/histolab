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
            (TILES.TISSUE_LEVEL0_4302_10273_4814_10785, False),
            (TILES.TISSUE_LEVEL0_7352_11762_7864_12274, False),
            (TILES.TISSUE_LEVEL2_1784_6289_5880_10386, False),
            (TILES.TISSUE_LEVEL2_3000_7666_7096_11763, False),
            (TILES.TISSUE_LEVEL2_4640_4649_8736_8746, False),
            (TILES.TISSUE_LEVEL2_4760_5241_8856_9338, False),
        ),
    )
    def it_knows_if_is_is_almost_white(self, tile_img, expected_result):
        coords = CP(0, 512, 0, 512)
        tile = Tile(tile_img, coords)

        is_almost_white = tile._is_almost_white

        assert is_almost_white == expected_result
