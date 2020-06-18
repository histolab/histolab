import pytest
import numpy as np

import histolab.filters.morphological_filters_functional as mof

from ..fixtures import NPY
from ..util import load_expectation


@pytest.mark.parametrize(
    "mask_array, min_size, avoid_overmask, overmask_thresh, expected_array",
    (
        (
            NPY.DIAGNOSTIC_SLIDE_THUMB_RGB1_HYSTERESIS_THRESHOLD_MASK,
            3000,
            True,
            95,
            "mask-arrays/diagnostic-slide-thumb-rgb1-hysteresis-threshold-remove-small-objects-mask",
        ),
        (
            NPY.DIAGNOSTIC_SLIDE_THUMB_HSV_OTSU_THRESHOLD_MASK,
            3000,
            True,
            95,
            "mask-arrays/diagnostic-slide-thumb-hsv-otsu-threshold-remove-small-objects-mask",
        ),
        (
            NPY.DIAGNOSTIC_SLIDE_THUMB_HSV_OTSU_THRESHOLD_MASK,
            3000,
            False,
            95,
            "mask-arrays/diagnostic-slide-thumb-hsv-otsu-threshold-remove-small-objects2-mask",
        ),
        (
            NPY.DIAGNOSTIC_SLIDE_THUMB_RGB1_HYSTERESIS_THRESHOLD_MASK,
            1200,
            True,
            25,
            "mask-arrays/diagnostic-slide-thumb-rgb1-hysteresis-threshold-remove-small-objects2-mask",
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
