import numpy as np
import pytest

from histolab.masks import TissueMask
from histolab.mixins import LinalgMixin
from histolab.stain_normalizer import MacenkoStainNormalizer

from ..unitutil import PILIMG, NpArrayMock, method_mock


class Describe_MacenkoStainNormalizer:
    def it_knows_how_to_complement_stain_matrix(self):
        stain_matrix = np.array(
            [
                [0.17030583, 0.5657144],
                [0.79976781, 0.72100778],
                [0.57564518, 0.40014373],
            ]
        )
        expected_complemented_stain_matrix = np.array(
            [
                [0.17030583, 0.5657144, -0.22151917],
                [0.79976781, 0.72100778, 0.60030006],
                [0.57564518, 0.40014373, -0.76848494],
            ]
        )

        complemented_stain_matrix = MacenkoStainNormalizer._complement_stain_matrix(
            stain_matrix
        )

        assert isinstance(complemented_stain_matrix, np.ndarray)
        np.testing.assert_array_almost_equal(
            complemented_stain_matrix, expected_complemented_stain_matrix
        )

    @pytest.mark.parametrize(
        "reference, expected_stain_index",
        [
            ([0.65, 0.7, 0.29], 1),
            ([0.07, 0.99, 0.11], 1),
        ],
    )
    def it_knows_how_to_find_stain_index_issue(self, reference, expected_stain_index):
        """With some stain vectors, changing the order of the stains doesn't change the
        stain index, documented issue:
        https://github.com/DigitalSlideArchive/HistomicsTK/issues/556
        """
        stain_vector = np.array(
            [
                [-0.38326727, 0.55173352, -0.19168126],
                [0.68482841, 0.72174826, 0.59742868],
                [0.61977113, 0.41793489, -0.77867662],
            ]
        )

        stain_index = MacenkoStainNormalizer._find_stain_index(reference, stain_vector)

        assert type(stain_index) == np.int64
        assert stain_index == expected_stain_index

    @pytest.mark.parametrize(
        "reference, expected_stain_index",
        [
            ([0.65, 0.7, 0.29], 0),
            ([0.07, 0.99, 0.11], 1),
        ],
    )
    def it_knows_how_to_find_stain_index_correct(self, reference, expected_stain_index):
        stain_vector = np.array(
            [
                [0.52069671, 0.20735896, 0.12419761],
                [0.76067927, 0.84840022, -0.51667523],
                [0.38761062, 0.48705166, 0.84712553],
            ]
        )

        stain_index = MacenkoStainNormalizer._find_stain_index(reference, stain_vector)

        assert type(stain_index) == np.int64
        assert stain_index == expected_stain_index

    @pytest.mark.parametrize(
        "stains, first_stain_index, first_stain_reference, "
        "expected_reordered_stain_matrix",
        [
            (
                ["hematoxylin", "eosin"],
                0,
                [0.65, 0.70, 0.29],
                np.array(
                    [
                        [0.52069671, 0.20735896, 0.12419761],
                        [0.76067927, 0.84840022, -0.51667523],
                        [0.38761062, 0.48705166, 0.84712553],
                    ]
                ),
            ),
            (
                ["eosin", "hematoxylin"],
                1,
                [0.07, 0.99, 0.11],
                np.array(
                    [
                        [0.207359, 0.520697, 0.124198],
                        [0.8484, 0.760679, -0.516675],
                        [0.487052, 0.387611, 0.847126],
                    ]
                ),
            ),
        ],
    )
    def it_knows_how_to_reorder_stains(
        self,
        stains,
        first_stain_index,
        first_stain_reference,
        expected_reordered_stain_matrix,
        request,
    ):
        stain_matrix = np.array(
            [
                [0.52069671, 0.20735896, 0.12419761],
                [0.76067927, 0.84840022, -0.51667523],
                [0.38761062, 0.48705166, 0.84712553],
            ]
        )
        _find_stain_index_ = method_mock(
            request, MacenkoStainNormalizer, "_find_stain_index", autospec=False
        )
        _find_stain_index_.return_value = first_stain_index

        reordered_stain_matrix = MacenkoStainNormalizer._reorder_stains(
            stain_matrix, stains
        )

        assert isinstance(reordered_stain_matrix, np.ndarray)
        np.testing.assert_array_almost_equal(
            reordered_stain_matrix, expected_reordered_stain_matrix
        )
        _find_stain_index_.assert_called_once_with(first_stain_reference, stain_matrix)

    @pytest.mark.parametrize(
        "img",
        [
            PILIMG.RGB_RANDOM_COLOR_500X500,
            PILIMG.RGBA_RANDOM_COLOR_500X500,
        ],
    )
    def it_knows_its_stain_matrix(self, request, img):
        _tissue_mask_call = method_mock(request, TissueMask, "__call__")
        _tissue_mask_call.return_value = NpArrayMock.ONES_500X500_BOOL
        _two_principal_components = method_mock(
            request, LinalgMixin, "two_principal_components", autospec=False
        )
        _two_principal_components.return_value = np.array(
            [
                [-0.64316707, -0.38117699],
                [0.29357369, -0.92376891],
                [-0.70721327, -0.03681175],
            ]
        )
        _normalize_columns = method_mock(request, LinalgMixin, "normalize_columns")
        _normalize_columns.return_value = np.array(
            [
                [0.71250099, 0.28600942],
                [-0.07793929, 0.95608979],
                [0.69732905, -0.06396029],
            ]
        )
        _complement_stain_matrix = method_mock(
            request, MacenkoStainNormalizer, "_complement_stain_matrix", autospec=False
        )
        _complement_stain_matrix.return_value = np.array(
            [
                [0.71250099, 0.28600942, -0.66410859],
                [-0.07793929, 0.95608979, 0.24589732],
                [0.69732905, -0.06396029, 0.70604128],
            ]
        )
        _reorder_stains = method_mock(
            request, MacenkoStainNormalizer, "_reorder_stains", autospec=False
        )
        _reorder_stains.return_value = np.array(
            [
                [0.28600942, 0.71250099, -0.66410859],
                [0.95608979, -0.07793929, 0.24589732],
                [-0.06396029, 0.69732905, 0.70604128],
            ]
        )
        expected_stain_matrix = np.array(
            [
                [0.286009, 0.712501, -0.664109],
                [0.95609, -0.077939, 0.245897],
                [-0.06396, 0.697329, 0.706041],
            ]
        )
        macenko_stain_normalizer = MacenkoStainNormalizer()

        stain_matrix = macenko_stain_normalizer.stain_matrix(img)

        assert isinstance(stain_matrix, np.ndarray)
        np.testing.assert_array_almost_equal(stain_matrix, expected_stain_matrix)
        _tissue_mask_call.assert_called_once()
        _two_principal_components.assert_called_once()
        _normalize_columns.assert_called_once()
        _complement_stain_matrix.assert_called_once()
        _reorder_stains.assert_called_once()

    def but_it_raises_value_error_if_img_not_rgb_or_rgba(self):
        macenko_stain_normalizer = MacenkoStainNormalizer()
        img = PILIMG.GRAY_RANDOM_10X10

        with pytest.raises(ValueError) as err:
            macenko_stain_normalizer.stain_matrix(img)

        assert str(err.value) == "Input image must be RGB or RGBA"

    def but_it_raises_valueerror_if_incorrect_stains(self):
        macenko_stain_normalizer = MacenkoStainNormalizer()
        img = PILIMG.RGB_RANDOM_COLOR_500X500

        with pytest.raises(ValueError) as err:
            macenko_stain_normalizer.stain_matrix(img, stains=["one", "two", "three"])

        assert str(err.value) == "Only two-stains lists are currently supported."
