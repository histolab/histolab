import numpy as np

from histolab.filters.util import mask_difference

from ...base import BASE_MASK, BASE_MASK2


def test_mask_difference():
    expected_mask_difference = np.array(
        [[False, True, False, False], [False, False, False, False]]
    )

    mask_difference_ = mask_difference(BASE_MASK, BASE_MASK2)

    assert type(mask_difference_) == np.ndarray
    assert mask_difference_.dtype == "bool"
    np.testing.assert_equal(mask_difference_, expected_mask_difference)
