import numpy as np


class LinalgMixin:
    @staticmethod
    def normalize_columns(arr: np.ndarray) -> np.ndarray:
        """Normalize each column vector in an array"""
        return arr / np.linalg.norm(arr, axis=0)
