import pytest

from histolab import scorer
from histolab.tile import Tile

from ..fixtures import TILES


class Describe_Scorers(object):
    @pytest.mark.parametrize(
        "tile_img, expected_score",
        (
            # level 0
            (TILES.VERY_LOW_NUCLEI_SCORE_LEVEL0, 0.39668202921529044),
            (TILES.LOW_NUCLEI_SCORE_LEVEL0, 0.34299989754219434),
            (TILES.MEDIUM_NUCLEI_SCORE_LEVEL0, 0.3832490771628042),
            (TILES.HIGH_NUCLEI_SCORE_LEVEL0, 0.7147274900858304),
            # level 1
            (TILES.VERY_LOW_NUCLEI_SCORE_RED_PEN_LEVEL1, 0.16439756135044395),
            (TILES.LOW_NUCLEI_SCORE_LEVEL1, 0.2412938986183465),
            (TILES.MEDIUM_NUCLEI_SCORE_LEVEL1, 0.3171736698917567),
            (TILES.MEDIUM_NUCLEI_SCORE_LEVEL1_2, 0.5193736600810265),
            (TILES.MEDIUM_NUCLEI_SCORE_GREEN_PEN_LEVEL1, 0.7200658493037305),
            (TILES.HIGH_NUCLEI_SCORE_RED_PEN_LEVEL1, 0.5549940151965075),
            # level 2
            (TILES.MEDIUM_NUCLEI_SCORE_LEVEL2, 0.5226490280587772),
            (TILES.HIGH_NUCLEI_SCORE_LEVEL2, 0.5941818245460105),
            # no tissue
            (TILES.NO_TISSUE, 0.015577231690474633),
            (TILES.NO_TISSUE2, 0.03227306029240194),
            (TILES.NO_TISSUE_LINE, 0.2140289904514813),
            (TILES.NO_TISSUE_RED_PEN, 0.6531987758882148),
            (TILES.NO_TISSUE_GREEN_PEN, 0.6926820791810822),
        ),
    )
    def it_knows_nuclei_score(self, tile_img, expected_score):
        tile = Tile(tile_img, None)
        nuclei_scorer = scorer.NucleiScorer()

        score = nuclei_scorer(tile, 0.6)

        assert score == expected_score
