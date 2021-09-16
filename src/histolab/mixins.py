import numpy as np


class LinalgMixin:
    @staticmethod
    def normalize_columns(arr: np.ndarray) -> np.ndarray:
        """Normalize each column vector in an array"""
        return arr / np.linalg.norm(arr, axis=0)

    @staticmethod
    def two_principal_components(arr: np.ndarray) -> np.ndarray:
        """
        Return the two principal components of the covariance matrix of ``arr``.

        Parameters
        ----------
        arr : np.ndarray
            Input array

        Returns
        -------
        np.ndarray
            Two principal components.
        """
        _, V = np.linalg.eigh(np.cov(arr, rowvar=False))
        return V[:, 1:3]
