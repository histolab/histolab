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

from histolab.filters.util import mask_difference

from ...base import (
    BASE_MASK,
    BASE_MASK2,
    BASE_MASK3,
    BASE_MASK4,
    COMPLEX_MASK,
    COMPLEX_MASK2,
    COMPLEX_MASK3,
)


@pytest.mark.parametrize(
    "array1, array2, expected_mask_difference",
    (
        (
            BASE_MASK,
            BASE_MASK2,
            np.array([[False, True, False, False], [False, False, False, False]]),
        ),
        (BASE_MASK3, BASE_MASK4, np.array([True, False, False, False])),
        (COMPLEX_MASK, COMPLEX_MASK2, COMPLEX_MASK3),
    ),
)
def test_mask_difference(array1, array2, expected_mask_difference):
    mask_difference_ = mask_difference(array1, array2)

    assert isinstance(mask_difference_, np.ndarray) is True
    assert mask_difference_.dtype == "bool"
    np.testing.assert_equal(mask_difference_, expected_mask_difference)
