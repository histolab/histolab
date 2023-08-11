# encoding: utf-8

# ------------------------------------------------------------------------
# Copyright 2022 All Histolab Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------

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
    assert isinstance(mask_no_small_object, np.ndarray) is True


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
    assert isinstance(mask_watershed, np.ndarray) is True
