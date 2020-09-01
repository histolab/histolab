import pytest

from histolab import scorer
from histolab.tile import Tile

from ..fixtures import TILES


class Describe_Scorers(object):
    @pytest.mark.parametrize(
        "tile_img, expected_score",
        (
            (TILES.VERY_LOW_NUCLEI_SCORE_LEVEL0, 0.39668202921529044),
            (TILES.LOW_NUCLEI_SCORE_LEVEL0, 0.34299989754219434),
            (TILES.MEDIUM_NUCLEI_SCORE_LEVEL0, 0.3832490771628042),
            (TILES.HIGH_NUCLEI_SCORE_LEVEL0, 0.7147274900858304),
            (TILES.MEDIUM_NUCLEI_SCORE_LEVEL2, 0.5226490280587772),
            (TILES.HIGH_NUCLEI_SCORE_LEVEL2, 0.5941818245460105),
            (TILES.NO_TISSUE, 0.015577231690474633),
        ),
    )
    def it_knows_nuclei_score(self, tile_img, expected_score):
        tile = Tile(tile_img, None)
        nuclei_scorer = scorer.NucleiScorer()

        score = nuclei_scorer(tile, 0.6)

        assert score == expected_score
