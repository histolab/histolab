# encoding: utf-8

# ------------------------------------------------------------------------
# Copyright 2021 All Histolab Contributors
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
