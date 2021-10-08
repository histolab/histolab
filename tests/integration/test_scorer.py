import pytest
from histolab.tile import Tile

from histolab import scorer

from ..fixtures import TILES


class Describe_Scorers:
    @pytest.mark.parametrize(
        "tile_img, expected_score",
        (
            # level 0
            (TILES.VERY_LOW_NUCLEI_SCORE_LEVEL0, 0.0042),
            (TILES.LOW_NUCLEI_SCORE_LEVEL0, 0.03001),
            (TILES.MEDIUM_NUCLEI_SCORE_LEVEL0, 0.266),
            (TILES.HIGH_NUCLEI_SCORE_LEVEL0, 0.008),
            # level 1
            (
                TILES.VERY_LOW_NUCLEI_SCORE_RED_PEN_LEVEL1,
                0.10605,
            ),  # breast - red pen
            (TILES.LOW_NUCLEI_SCORE_LEVEL1, 0.00704),  # breast - green pen
            (TILES.MEDIUM_NUCLEI_SCORE_LEVEL1, 0.08904),  # aorta
            (TILES.MEDIUM_NUCLEI_SCORE_LEVEL1_2, 0.09247),  # breast - green pen
            (TILES.MEDIUM_NUCLEI_SCORE_GREEN_PEN_LEVEL1, 0.08088),
            # breast - green pen
            (TILES.HIGH_NUCLEI_SCORE_RED_PEN_LEVEL1, 0.10297),  # breast - red pen
            # level 2
            (TILES.MEDIUM_NUCLEI_SCORE_LEVEL2, 0.00337),  # prostate
            (TILES.HIGH_NUCLEI_SCORE_LEVEL2, 0.00217),  # prostate
            # no tissue
            (TILES.NO_TISSUE, 0.0),
            (TILES.NO_TISSUE2, 0.0),
            (TILES.NO_TISSUE_LINE, 0.00026),
            (TILES.NO_TISSUE_RED_PEN, 0.20831),
            (TILES.NO_TISSUE_GREEN_PEN, 0.23808),
        ),
    )
    def it_knows_nuclei_score(self, tile_img, expected_score):
        tile = Tile(tile_img, None)
        nuclei_scorer = scorer.NucleiScorer()
        expected_warning_regex = (
            r"Input image must be RGB. NOTE: the image will be converted to RGB before"
            r" HED conversion."
        )

        with pytest.warns(UserWarning, match=expected_warning_regex):
            score = nuclei_scorer(tile)

        assert round(score, 5) == round(expected_score, 5)

    @pytest.mark.parametrize(
        "tile_img, tissue, expected_score",
        (
            # level 0
            (TILES.VERY_LOW_NUCLEI_SCORE_LEVEL0, True, 0.01494),
            (TILES.LOW_NUCLEI_SCORE_LEVEL0, True, 0.13681),
            (TILES.MEDIUM_NUCLEI_SCORE_LEVEL0, True, 0.48335),
            (TILES.HIGH_NUCLEI_SCORE_LEVEL0, True, 0.01808),
            # level 1
            (
                TILES.VERY_LOW_NUCLEI_SCORE_RED_PEN_LEVEL1,
                True,
                0.76041,
            ),  # breast - red pen
            (
                TILES.LOW_NUCLEI_SCORE_LEVEL1,
                True,
                0.07103,
            ),  # breast - green pen
            (TILES.MEDIUM_NUCLEI_SCORE_LEVEL1, True, 0.39569),  # aorta
            (
                TILES.MEDIUM_NUCLEI_SCORE_LEVEL1_2,
                True,
                0.15244,
            ),  # breast - green pen
            (
                TILES.MEDIUM_NUCLEI_SCORE_GREEN_PEN_LEVEL1,
                True,
                0.15774,
            ),  # breast - green pen
            (
                TILES.HIGH_NUCLEI_SCORE_RED_PEN_LEVEL1,
                True,
                0.21587,
            ),  # breast - red pen
            # level 2
            (TILES.MEDIUM_NUCLEI_SCORE_LEVEL2, True, 0.02112),  # prostate
            (TILES.HIGH_NUCLEI_SCORE_LEVEL2, True, 0.18165),  # prostate
            # no tissue
            (TILES.NO_TISSUE, True, 0.05021),
            (TILES.NO_TISSUE2, True, 0.00203),
            (TILES.NO_TISSUE_LINE, True, 0.31477),
            (TILES.NO_TISSUE_RED_PEN, True, 0.72669),
            (TILES.NO_TISSUE_GREEN_PEN, True, 0.56731),
            (TILES.VERY_LOW_NUCLEI_SCORE_LEVEL0, False, 0.01482),
            (TILES.LOW_NUCLEI_SCORE_LEVEL0, False, 0.11326),
            (TILES.MEDIUM_NUCLEI_SCORE_LEVEL0, False, 0.44272),
            (TILES.HIGH_NUCLEI_SCORE_LEVEL0, False, 0.01801),
            # level 1
            (
                TILES.VERY_LOW_NUCLEI_SCORE_RED_PEN_LEVEL1,
                False,
                0.29887,
            ),  # breast - red pen
            (
                TILES.LOW_NUCLEI_SCORE_LEVEL1,
                False,
                0.02445,
            ),  # breast - green pen
            (TILES.MEDIUM_NUCLEI_SCORE_LEVEL1, False, 0.30227),
            (
                TILES.MEDIUM_NUCLEI_SCORE_LEVEL1_2,
                False,
                0.15219,
            ),  # breast - green pen
            (
                TILES.MEDIUM_NUCLEI_SCORE_GREEN_PEN_LEVEL1,
                False,
                0.12733,
            ),  # breast - green pen
            (
                TILES.HIGH_NUCLEI_SCORE_RED_PEN_LEVEL1,
                False,
                0.21373,
            ),  # breast - red pen
            # level 2
            (TILES.MEDIUM_NUCLEI_SCORE_LEVEL2, False, 0.01728),  # prostate
            (TILES.HIGH_NUCLEI_SCORE_LEVEL2, False, 0.02852),  # prostate
            # no tissue
            (TILES.NO_TISSUE, False, 0.00027),
            (TILES.NO_TISSUE2, False, 0.00016),
            (TILES.NO_TISSUE_LINE, False, 0.00916),
            (TILES.NO_TISSUE_RED_PEN, False, 0.40948),
            (TILES.NO_TISSUE_GREEN_PEN, False, 0.39577),
        ),
    )
    def it_knows_cellularity_score(self, tile_img, tissue, expected_score):
        tile = Tile(tile_img, None)
        cell_scorer = scorer.CellularityScorer(consider_tissue=tissue)
        expected_warning_regex = (
            r"Input image must be RGB. NOTE: the image will be converted to RGB before"
            r" HED conversion."
        )

        with pytest.warns(UserWarning, match=expected_warning_regex):
            score = cell_scorer(tile)

        assert round(score, 5) == round(expected_score, 5)
