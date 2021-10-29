import pytest

from histolab import scorer
from histolab.tile import Tile

from ..fixtures import TILES


class Describe_Scorers:
    @pytest.mark.parametrize(
        "tile_img, expected_score",
        (  # IMPORTANT: with artifacts, the NucleiScorer cannot be fully trusted
            # level 0
            (TILES.VERY_LOW_NUCLEI_SCORE_LEVEL0, 5e-05),
            (TILES.LOW_NUCLEI_SCORE_LEVEL0, 0.01066),
            (TILES.MEDIUM_NUCLEI_SCORE_LEVEL0, 0.01673),
            (TILES.HIGH_NUCLEI_SCORE_LEVEL0, 0.37285),
            # level 1
            (
                TILES.VERY_LOW_NUCLEI_SCORE_RED_PEN_LEVEL1,
                0.00177,
            ),  # breast - red pen
            (TILES.LOW_NUCLEI_SCORE_LEVEL1, 0.01641),  # breast - green pen
            (TILES.MEDIUM_NUCLEI_SCORE_LEVEL1, 0.00698),  # aorta
            (TILES.MEDIUM_NUCLEI_SCORE_LEVEL1_2, 0.14523),  # breast - green pen
            (TILES.MEDIUM_NUCLEI_SCORE_GREEN_PEN_LEVEL1, 0.35342),
            # breast - green pen
            (
                TILES.HIGH_NUCLEI_SCORE_RED_PEN_LEVEL1,
                0.03176,
            ),  # breast - red pen + tissue fold
            # level 2
            (TILES.MEDIUM_NUCLEI_SCORE_LEVEL2, 0.03463),  # prostate
            (TILES.HIGH_NUCLEI_SCORE_LEVEL2, 0.00326),  # prostate
            # no tissue
            (TILES.NO_TISSUE, 0.0),
            (TILES.NO_TISSUE2, 1e-05),
            (TILES.NO_TISSUE_LINE, 0.00028),
            (TILES.NO_TISSUE_RED_PEN, 0.18849),
            (TILES.NO_TISSUE_GREEN_PEN, 0.30601),
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
        (  # IMPORTANT: with artifacts, the CellularityScorer cannot be fully trusted
            # level 0
            (TILES.VERY_LOW_NUCLEI_SCORE_LEVEL0, True, 8e-05),
            (TILES.LOW_NUCLEI_SCORE_LEVEL0, True, 0.02759),
            (TILES.MEDIUM_NUCLEI_SCORE_LEVEL0, True, 0.02709),
            (TILES.HIGH_NUCLEI_SCORE_LEVEL0, True, 0.51171),
            # level 1
            (
                TILES.VERY_LOW_NUCLEI_SCORE_RED_PEN_LEVEL1,
                True,
                0.01606,
            ),  # breast - red pen
            (
                TILES.LOW_NUCLEI_SCORE_LEVEL1,
                True,
                0.158,
            ),  # breast - green pen
            (TILES.MEDIUM_NUCLEI_SCORE_LEVEL1, True, 0.04082),  # aorta
            (
                TILES.MEDIUM_NUCLEI_SCORE_LEVEL1_2,
                True,
                0.22551,
            ),  # breast - green pen
            (
                TILES.MEDIUM_NUCLEI_SCORE_GREEN_PEN_LEVEL1,
                True,
                0.66435,
            ),  # breast - green pen
            (
                TILES.HIGH_NUCLEI_SCORE_RED_PEN_LEVEL1,
                True,
                0.07081,
            ),  # breast - red pen
            # level 2
            (TILES.MEDIUM_NUCLEI_SCORE_LEVEL2, True, 0.13723),  # prostate
            (TILES.HIGH_NUCLEI_SCORE_LEVEL2, True, 0.23546),  # prostate
            # no tissue
            (TILES.NO_TISSUE, True, 0.06485),
            (TILES.NO_TISSUE2, True, 0.00958),
            (TILES.NO_TISSUE_LINE, True, 0.34281),
            (TILES.NO_TISSUE_RED_PEN, True, 0.65595),
            (TILES.NO_TISSUE_GREEN_PEN, True, 0.72955),
            (TILES.VERY_LOW_NUCLEI_SCORE_LEVEL0, False, 8e-05),
            (TILES.LOW_NUCLEI_SCORE_LEVEL0, False, 0.02284),
            (TILES.MEDIUM_NUCLEI_SCORE_LEVEL0, False, 0.02481),
            (TILES.HIGH_NUCLEI_SCORE_LEVEL0, False, 0.50961),
            # level 1
            (
                TILES.VERY_LOW_NUCLEI_SCORE_RED_PEN_LEVEL1,
                False,
                0.00631,
            ),  # breast - red pen
            (
                TILES.LOW_NUCLEI_SCORE_LEVEL1,
                False,
                0.05439,
            ),  # breast - green pen
            (TILES.MEDIUM_NUCLEI_SCORE_LEVEL1, False, 0.03118),
            (
                TILES.MEDIUM_NUCLEI_SCORE_LEVEL1_2,
                False,
                0.22515,
            ),  # breast - green pen
            (
                TILES.MEDIUM_NUCLEI_SCORE_GREEN_PEN_LEVEL1,
                False,
                0.53626,
            ),  # breast - green pen
            (
                TILES.HIGH_NUCLEI_SCORE_RED_PEN_LEVEL1,
                False,
                0.07011,
            ),  # breast - red pen
            # level 2
            (TILES.MEDIUM_NUCLEI_SCORE_LEVEL2, False, 0.11224),  # prostate
            (TILES.HIGH_NUCLEI_SCORE_LEVEL2, False, 0.03697),  # prostate
            # no tissue
            (TILES.NO_TISSUE, False, 0.00035),
            (TILES.NO_TISSUE2, False, 0.00076),
            (TILES.NO_TISSUE_LINE, False, 0.00998),
            (TILES.NO_TISSUE_RED_PEN, False, 0.36961),
            (TILES.NO_TISSUE_GREEN_PEN, False, 0.50896),
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
