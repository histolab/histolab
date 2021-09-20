import numpy as np
import pytest
from PIL import ImageChops

from histolab.stain_normalizer import MacenkoStainNormalizer

from ..fixtures import TILES
from ..util import load_expectation


class Describe_MacenkoStainNormalizer:
    @pytest.mark.parametrize(
        "img, expected_stain_matrix",
        [
            (
                TILES.TISSUE_LEVEL0_7352_11762_7864_12274,
                np.array(
                    [
                        [0.52069671, 0.20735896, 0.12419761],
                        [0.76067927, 0.84840022, -0.51667523],
                        [0.38761062, 0.48705166, 0.84712553],
                    ]
                ),
            ),
            (
                TILES.MEDIUM_NUCLEI_SCORE_LEVEL1,
                np.array(
                    [
                        [0.53941357, 0.12592985, 0.17475281],
                        [0.73491653, 0.81801475, -0.57519156],
                        [0.41101179, 0.56124285, 0.79913461],
                    ]
                ),
            ),
            (
                TILES.LOW_NUCLEI_SCORE_LEVEL0,
                np.array(
                    [
                        [0.65048231, 0.26120907, 0.63583375],
                        [0.72242689, 0.62082336, -0.68680445],
                        [0.23446141, 0.73915369, 0.35215775],
                    ]
                ),
            ),
        ],
    )
    def it_knows_its_stain_matrix(self, img, expected_stain_matrix):
        normalizer = MacenkoStainNormalizer()

        stain_matrix = normalizer.stain_matrix(img)

        np.testing.assert_almost_equal(stain_matrix, expected_stain_matrix)

    @pytest.mark.parametrize(
        "img, expected_stain_matrix_target, expected_max_concentrations_target",
        [
            (
                TILES.TISSUE_LEVEL0_7352_11762_7864_12274,
                np.array(
                    [
                        [0.52069671, 0.20735896, 0.12419761],
                        [0.76067927, 0.84840022, -0.51667523],
                        [0.38761062, 0.48705166, 0.84712553],
                    ]
                ),
                np.array([2.17012019, 1.2563706, 0.05241914]),
            ),
            (
                TILES.MEDIUM_NUCLEI_SCORE_LEVEL1,
                np.array(
                    [
                        [0.53941357, 0.12592985, 0.17475281],
                        [0.73491653, 0.81801475, -0.57519156],
                        [0.41101179, 0.56124285, 0.79913461],
                    ]
                ),
                np.array([1.12761155, 1.34362017, 0.14565865]),
            ),
            (
                TILES.LOW_NUCLEI_SCORE_LEVEL0,
                np.array(
                    [
                        [0.65048231, 0.26120907, 0.63583375],
                        [0.72242689, 0.62082336, -0.68680445],
                        [0.23446141, 0.73915369, 0.35215775],
                    ]
                ),
                np.array([0.74900284, 0.52391914, 0.09494156]),
            ),
        ],
    )
    def it_knows_how_to_fit(
        self, img, expected_stain_matrix_target, expected_max_concentrations_target
    ):
        normalizer = MacenkoStainNormalizer()
        normalizer.fit(img)

        stain_matrix_target = normalizer.stain_matrix_target
        max_concentrations_target = normalizer.max_concentrations_target

        np.testing.assert_almost_equal(
            stain_matrix_target, expected_stain_matrix_target
        )
        np.testing.assert_almost_equal(
            max_concentrations_target, expected_max_concentrations_target
        )

    @pytest.mark.parametrize(
        "img_to_fit, img_to_transform, expected_img_normalized_path",
        [
            (
                TILES.TISSUE_LEVEL0_7352_11762_7864_12274,
                TILES.MEDIUM_NUCLEI_SCORE_LEVEL1,
                "pil-images-rgb/tissue-level0-7352-11762-7864-12274"
                "--medium-nuclei-score-level1--macenko",
            ),
            (
                TILES.TISSUE_LEVEL0_7352_11762_7864_12274,
                TILES.LOW_NUCLEI_SCORE_LEVEL0,
                "pil-images-rgb/tissue-level0-7352-11762-7864-12274"
                "--low-nuclei-score-level0--macenko",
            ),
        ],
    )
    def it_knows_how_to_fit_and_transform(
        self, img_to_fit, img_to_transform, expected_img_normalized_path
    ):
        expected_img_normalized = load_expectation(
            expected_img_normalized_path, type_="png"
        )
        normalizer = MacenkoStainNormalizer()

        normalizer.fit(img_to_fit)
        img_normalized = normalizer.transform(img_to_transform)

        assert (
            np.unique(
                np.array(ImageChops.difference(img_normalized, expected_img_normalized))
            )[0]
            == 0
        )
