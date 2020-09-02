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
    print(mask_difference_)

    assert type(mask_difference_) == np.ndarray
    assert mask_difference_.dtype == "bool"
    np.testing.assert_equal(mask_difference_, expected_mask_difference)
