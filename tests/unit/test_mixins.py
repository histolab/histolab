import numpy as np

from histolab.mixins import LinalgMixin


class Describe_LinalgMixin:
    def it_knows_how_to_normalize_columns(self):
        arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        expected_arr_normalized = np.array(
            [
                [0.19611614, 0.31622777, 0.3939193, 0.4472136],
                [0.98058068, 0.9486833, 0.91914503, 0.89442719],
            ]
        )

        matrix_normalized = LinalgMixin.normalize_columns(arr)

        assert isinstance(matrix_normalized, np.ndarray)
        np.testing.assert_array_almost_equal(matrix_normalized, expected_arr_normalized)
