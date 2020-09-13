import pytest

from histolab import scorer
from histolab.tile import Tile

from ..fixtures import TILES


class Describe_Scorers:
    @pytest.mark.parametrize(
        "tile_img, expected_score",
        (
            # level 0
            (TILES.VERY_LOW_NUCLEI_SCORE_LEVEL0, 4.95387907194613e-05),
            (TILES.LOW_NUCLEI_SCORE_LEVEL0, 0.011112025501054716),
            (TILES.MEDIUM_NUCLEI_SCORE_LEVEL0, 0.018651677436394662),
            (TILES.HIGH_NUCLEI_SCORE_LEVEL0, 0.39901978131493154),
            # level 1
            (
                TILES.VERY_LOW_NUCLEI_SCORE_RED_PEN_LEVEL1,
                0.0017590279896743531,
            ),  # breast - red pen
            (TILES.LOW_NUCLEI_SCORE_LEVEL1, 0.019689596845556157),  # breast - green pen
            (TILES.MEDIUM_NUCLEI_SCORE_LEVEL1, 0.009512701556682022),  # aorta
            (
                TILES.MEDIUM_NUCLEI_SCORE_LEVEL1_2,
                0.1519627864167197,
            ),  # breast - green pen
            (
                TILES.MEDIUM_NUCLEI_SCORE_GREEN_PEN_LEVEL1,
                0.35696740368342295,
            ),  # breast - green pen
            (
                TILES.HIGH_NUCLEI_SCORE_RED_PEN_LEVEL1,
                0.19867406253648537,
            ),  # breast - red pen
            # level 2
            (TILES.MEDIUM_NUCLEI_SCORE_LEVEL2, 0.17959041877765877),  # prostate
            (TILES.HIGH_NUCLEI_SCORE_LEVEL2, 0.02165773137397028),  # prostate
            # no tissue
            (TILES.NO_TISSUE, 6.677516254309149e-07),
            (TILES.NO_TISSUE2, 7.505964521573081e-06),
            (TILES.NO_TISSUE_LINE, 0.00028575083246431935),
            (TILES.NO_TISSUE_RED_PEN, 0.2051245888881937),
            (TILES.NO_TISSUE_GREEN_PEN, 0.28993597176882124),
        ),
    )
    def it_knows_nuclei_score(self, tile_img, expected_score):
        tile = Tile(tile_img, None)
        nuclei_scorer = scorer.NucleiScorer()

        score = nuclei_scorer(tile)

        assert round(score, 5) == round(expected_score, 5)
