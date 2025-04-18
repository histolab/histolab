import numpy as np
import pytest

from histolab.stain_normalizer import MacenkoStainNormalizer, ReinhardStainNormalizer

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
                        [0.520699, 0.2073588, 0.1242305],
                        [0.7606757, 0.8483905, -0.5166961],
                        [0.3876146, 0.4870686, 0.847108],
                    ]
                ),
            ),
            (
                TILES.MEDIUM_NUCLEI_SCORE_LEVEL1,
                np.array(
                    [
                        [0.5388396, 0.1261264, 0.1762086],
                        [0.7349083, 0.8172188, -0.5763255],
                        [0.4117787, 0.5623572, 0.7979971],
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
                        [0.520699, 0.2073588, 0.1242305],
                        [0.7606757, 0.8483905, -0.5166961],
                        [0.3876146, 0.4870686, 0.847108],
                    ]
                ),
                np.array([2.1701043, 1.2563618, 0.0523954]),
            ),
            (
                TILES.MEDIUM_NUCLEI_SCORE_LEVEL1,
                np.array(
                    [
                        [0.5388396, 0.1261264, 0.1762086],
                        [0.7349083, 0.8172188, -0.5763255],
                        [0.4117787, 0.5623572, 0.7979971],
                    ]
                ),
                np.array([1.1295585, 1.3438867, 0.1440942]),
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

        np.testing.assert_array_almost_equal(
            np.array(img_normalized), np.array(expected_img_normalized)
        )


class Describe_ReinhardStainNormalizer:
    @pytest.mark.parametrize(
        "img, expected_mean, expected_std",
        [
            (
                TILES.TISSUE_LEVEL0_7352_11762_7864_12274,
                np.array([57.77235476, 30.6667066, -12.3231239]),
                np.array([15.37156544, 10.19152132, 5.33366456]),
            ),
            (
                TILES.MEDIUM_NUCLEI_SCORE_LEVEL1,
                np.array([63.60525855, 34.10860319, -3.26439523]),
                np.array([13.75089992, 13.89511025, 6.91485312]),
            ),
            (
                TILES.LOW_NUCLEI_SCORE_LEVEL0,
                np.array([82.67172754, 8.5236446, -4.30401803]),
                np.array([12.48770309, 8.40158118, 10.3241243]),
            ),
        ],
    )
    def it_knows_its_summary_statistics(self, img, expected_mean, expected_std):
        normalizer = ReinhardStainNormalizer()

        mean, std = normalizer._summary_statistics(img)

        np.testing.assert_almost_equal(mean, expected_mean)
        np.testing.assert_almost_equal(std, expected_std)

    @pytest.mark.parametrize(
        "img, expected_mean, expected_std",
        [
            (
                TILES.TISSUE_LEVEL0_7352_11762_7864_12274,
                np.array([57.77235476, 30.6667066, -12.3231239]),
                np.array([15.37156544, 10.19152132, 5.33366456]),
            ),
            (
                TILES.MEDIUM_NUCLEI_SCORE_LEVEL1,
                np.array([63.60525855, 34.10860319, -3.26439523]),
                np.array([13.75089992, 13.89511025, 6.91485312]),
            ),
            (
                TILES.LOW_NUCLEI_SCORE_LEVEL0,
                np.array([82.67172754, 8.5236446, -4.30401803]),
                np.array([12.48770309, 8.40158118, 10.3241243]),
            ),
        ],
    )
    def it_knows_how_to_fit(self, img, expected_mean, expected_std):
        normalizer = ReinhardStainNormalizer()

        assert normalizer.target_means is None
        assert normalizer.target_stds is None

        normalizer.fit(img)

        target_means = normalizer.target_means
        target_stds = normalizer.target_stds

        np.testing.assert_almost_equal(target_means, expected_mean)
        np.testing.assert_almost_equal(target_stds, expected_std)

    @pytest.mark.parametrize(
        "img_to_fit, img_to_transform, expected_img_normalized_path",
        [
            (
                TILES.TISSUE_LEVEL0_7352_11762_7864_12274,
                TILES.MEDIUM_NUCLEI_SCORE_LEVEL1,
                "pil-images-rgb/tissue-level0-7352-11762-7864-12274"
                "--medium-nuclei-score-level1--reinhard",
            ),
            (
                TILES.TISSUE_LEVEL0_7352_11762_7864_12274,
                TILES.LOW_NUCLEI_SCORE_LEVEL0,
                "pil-images-rgb/tissue-level0-7352-11762-7864-12274"
                "--low-nuclei-score-level0--reinhard",
            ),
        ],
    )
    def it_knows_how_to_fit_and_transform_Re(
        self, img_to_fit, img_to_transform, expected_img_normalized_path
    ):
        expected_img_normalized = load_expectation(
            expected_img_normalized_path, type_="png"
        )
        normalizer = ReinhardStainNormalizer()

        normalizer.fit(img_to_fit)
        img_normalized = normalizer.transform(img_to_transform)

        np.testing.assert_almost_equal(
            np.array(img_normalized), np.array(expected_img_normalized)
        )
