import numpy as np
import pytest

import histolab.filters.morphological_filters_functional as mof

from ..fixtures import MASKNPY
from ..util import load_expectation


@pytest.mark.parametrize(
    "mask_array, min_size, avoid_overmask, overmask_thresh, expected_array",
    (
        (
            MASKNPY.DIAGNOSTIC_SLIDE_THUMB_RGB1_HYSTERESIS_THRESHOLD_MASK,
            3000,
            True,
            95,
            "mask-arrays/diagnostic-slide-thumb-rgb1-hysteresis-threshold-remove-small-"
            "objects-mask",
        ),
        (
            MASKNPY.DIAGNOSTIC_SLIDE_THUMB_HSV_OTSU_THRESHOLD_MASK,
            3000,
            True,
            95,
            "mask-arrays/diagnostic-slide-thumb-hsv-otsu-threshold-remove-small-objects"
            "-mask",
        ),
        (
            MASKNPY.DIAGNOSTIC_SLIDE_THUMB_HSV_OTSU_THRESHOLD_MASK,
            3000,
            False,
            95,
            "mask-arrays/diagnostic-slide-thumb-hsv-otsu-threshold-remove-small-objects"
            "2-mask",
        ),
        (
            MASKNPY.DIAGNOSTIC_SLIDE_THUMB_RGB1_HYSTERESIS_THRESHOLD_MASK,
            1200,
            True,
            25,
            "mask-arrays/diagnostic-slide-thumb-rgb1-hysteresis-threshold-remove-small-"
            "objects2-mask",
        ),
    ),
)
def test_remove_small_objects_filter(
    mask_array, min_size, avoid_overmask, overmask_thresh, expected_array
):
    expected_value = load_expectation(expected_array, type_="npy")
    mask_no_small_object = mof.remove_small_objects(
        mask_array, min_size, avoid_overmask, overmask_thresh
    )

    np.testing.assert_array_equal(mask_no_small_object, expected_value)
    assert type(mask_no_small_object) == np.ndarray


@pytest.mark.parametrize(
    "mask_array, region_shape, expected_array",
    (
        (MASKNPY.YTMA1, 6, "mask-arrays/ytma1-watershed-segmentation-region6"),
        (MASKNPY.YTMA2, 6, "mask-arrays/ytma2-watershed-segmentation-region6"),
        (MASKNPY.YTMA1, 3, "mask-arrays/ytma1-watershed-segmentation-region3"),
        (MASKNPY.YTMA2, 3, "mask-arrays/ytma2-watershed-segmentation-region3"),
    ),
)
def test_watershed_segmentation_filter(mask_array, region_shape, expected_array):
    expected_value = load_expectation(expected_array, type_="npy")

    mask_watershed = mof.watershed_segmentation(mask_array, region_shape)

    np.testing.assert_array_equal(mask_watershed, expected_value)
    assert type(mask_watershed) == np.ndarray
