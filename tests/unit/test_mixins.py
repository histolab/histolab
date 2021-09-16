import numpy as np

from histolab.mixins import LinalgMixin

from ..unitutil import function_mock


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

    def it_knows_how_to_calculate_two_principal_components(self, request):
        arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        np_cov_ = function_mock(request, "numpy.cov")
        np_cov_return_value = np.array(
            [
                [8.0, 8.0, 8.0, 8.0],
                [8.0, 8.0, 8.0, 8.0],
                [8.0, 8.0, 8.0, 8.0],
                [8.0, 8.0, 8.0, 8.0],
            ]
        )
        np_cov_.return_value = np_cov_return_value
        np_linalg_eigh_ = function_mock(request, "numpy.linalg.eigh")
        np_linalg_eigh_.return_value = (
            np.array(
                [-7.91853334e-15, -2.73960770e-15, -9.86076132e-32, 3.20000000e01]
            ),
            np.array(
                [
                    [-7.95140162e-01, -3.43150291e-01, -7.41676339e-18, 5.00000000e-01],
                    [5.88571918e-01, -6.35281904e-01, -2.66963658e-16, 5.00000000e-01],
                    [1.03284122e-01, 4.89216098e-01, -7.07106781e-01, 5.00000000e-01],
                    [1.03284122e-01, 4.89216098e-01, 7.07106781e-01, 5.00000000e-01],
                ]
            ),
        )
        expected_principal_components = np.array(
            [
                [-3.43150291e-01, -7.41676339e-18],
                [-6.35281904e-01, -2.66963658e-16],
                [4.89216098e-01, -7.07106781e-01],
                [4.89216098e-01, 7.07106781e-01],
            ]
        )

        two_principal_components = LinalgMixin.two_principal_components(arr)

        assert isinstance(two_principal_components, np.ndarray)
        np.testing.assert_array_almost_equal(
            two_principal_components, expected_principal_components
        )
        np_cov_.assert_called_once_with(arr, rowvar=False)
        np_linalg_eigh_.assert_called_once_with(np_cov_return_value)
