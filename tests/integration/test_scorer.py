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
            (TILES.VERY_LOW_NUCLEI_SCORE_LEVEL0, True, 8.237973509211956e-05),
            (TILES.LOW_NUCLEI_SCORE_LEVEL0, True, 0.029559623811739984),
            (TILES.MEDIUM_NUCLEI_SCORE_LEVEL0, True, 0.030890383076685523),
            (TILES.HIGH_NUCLEI_SCORE_LEVEL0, True, 0.5452530357003211),
            # level 1
            (
                TILES.VERY_LOW_NUCLEI_SCORE_RED_PEN_LEVEL1,
                True,
                0.01610148106450298,
            ),  # breast - red pen
            (
                TILES.LOW_NUCLEI_SCORE_LEVEL1,
                True,
                0.1887327689375471,
            ),  # breast - green pen
            (TILES.MEDIUM_NUCLEI_SCORE_LEVEL1, True, 0.04807442622295907),  # aorta
            (
                TILES.MEDIUM_NUCLEI_SCORE_LEVEL1_2,
                True,
                0.23493428091089713,
            ),  # breast - green pen
            (
                TILES.MEDIUM_NUCLEI_SCORE_GREEN_PEN_LEVEL1,
                True,
                0.6710679905294349,
            ),  # breast - green pen
            (
                TILES.HIGH_NUCLEI_SCORE_RED_PEN_LEVEL1,
                True,
                0.30155274716806657,
            ),  # breast - red pen
            # level 2
            (TILES.MEDIUM_NUCLEI_SCORE_LEVEL2, True, 0.40466129363258146),  # prostate
            (TILES.HIGH_NUCLEI_SCORE_LEVEL2, True, 0.906540012633011),  # prostate
            # no tissue
            (TILES.NO_TISSUE, True, 0.05299860529986053),
            (TILES.NO_TISSUE2, True, 0.003337363966142684),
            (TILES.NO_TISSUE_LINE, True, 0.3471366793342943),
            (TILES.NO_TISSUE_RED_PEN, True, 0.7135526324697217),
            (TILES.NO_TISSUE_GREEN_PEN, True, 0.6917688745017198),
            (TILES.VERY_LOW_NUCLEI_SCORE_LEVEL0, False, 8.168825750149552e-05),
            (TILES.LOW_NUCLEI_SCORE_LEVEL0, False, 0.024471282958984375),
            (TILES.MEDIUM_NUCLEI_SCORE_LEVEL0, False, 0.028293456864885436),
            (TILES.HIGH_NUCLEI_SCORE_LEVEL0, False, 0.5430199644432031),
            # level 1
            (
                TILES.VERY_LOW_NUCLEI_SCORE_RED_PEN_LEVEL1,
                False,
                0.006328582763671875,
            ),  # breast - red pen
            (
                TILES.LOW_NUCLEI_SCORE_LEVEL1,
                False,
                0.064971923828125,
            ),  # breast - green pen
            (TILES.MEDIUM_NUCLEI_SCORE_LEVEL1, False, 0.036724090576171875),
            (
                TILES.MEDIUM_NUCLEI_SCORE_LEVEL1_2,
                False,
                0.23455429077148438,
            ),  # breast - green pen
            (
                TILES.MEDIUM_NUCLEI_SCORE_GREEN_PEN_LEVEL1,
                False,
                0.5416870117187,
            ),  # breast - green pen
            (
                TILES.HIGH_NUCLEI_SCORE_RED_PEN_LEVEL1,
                False,
                0.2985572814941406,
            ),  # breast - red pen
            # level 2
            (TILES.MEDIUM_NUCLEI_SCORE_LEVEL2, False, 0.3309669494628906),  # prostate
            (TILES.HIGH_NUCLEI_SCORE_LEVEL2, False, 0.14234542846679688),  # prostate
            # no tissue
            (TILES.NO_TISSUE, False, 0.0002899169921875),
            (TILES.NO_TISSUE2, False, 0.000263214111328125),
            (TILES.NO_TISSUE_LINE, False, 0.010105133056640625),
            (TILES.NO_TISSUE_RED_PEN, False, 0.4020729064941406),
            (TILES.NO_TISSUE_GREEN_PEN, False, 0.48259735107421875),
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
