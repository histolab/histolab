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


# Code adapted from StainTools https://github.com/Peter554/StainTools,
# HistomicsTK https://github.com/DigitalSlideArchive/HistomicsTK
# and torchstain https://github.com/EIDOSlab/torchstain


from typing import List

import numpy as np
import PIL

from .filters.image_filters import RgbToOd
from .masks import TissueMask
from .mixins import LinalgMixin
from .tile import Tile


class MacenkoStainNormalizer(LinalgMixin):

    stain_color_map = {
        "hematoxylin": [0.65, 0.70, 0.29],
        "eosin": [0.07, 0.99, 0.11],
        "dab": [0.27, 0.57, 0.78],
        "null": [0.0, 0.0, 0.0],
    }

    def stain_matrix(
        self,
        img_rgb: PIL.Image.Image,
        alpha: int = 1,
        beta: float = 0.15,
        background_intensity: int = 240,
        stains: List[str] = None,
    ) -> np.ndarray:
        """Stain matrix estimation via method of M. Macenko et al. [1]_

        Parameters
        ----------
        img_rgb : PIL.Image.Image
            Input image RGB or RGBA
        alpha : int, optional
            Minimum angular percentile. Default is 1.
        background_intensity : int, optional
            Background transmitted light intensity. Default is 240.
        beta : float, optional
            Threshold value for which only the OD values above are kept. Default is 0.15
        stains : list, optional
            List of stain names (order is important). Default is
            ``["hematoxylin", "eosin"]``

        Returns
        -------
        np.ndarray
            Calculated stain matrix

        Raises
        ------
        ValueError
            if ``stain`` is not a two-stains list
        ValueError
            if the input image is not RGB or RGBA

        References
        ----------
        .. [1] Macenko, Marc, et al. "A method for normalizing histology slides for
            quantitative analysis." 2009 IEEE International Symposium on Biomedical
            Imaging: From Nano to Macro. IEEE, 2009.
        """

        if stains is not None and len(stains) != 2:
            raise ValueError("Only two-stains lists are currently supported.")

        stains = ["hematoxylin", "eosin"] if stains is None else stains

        if img_rgb.mode not in ["RGB", "RGBA"]:
            raise ValueError("Input image must be RGB or RGBA")

        # Convert to OD and ignore background
        tile = Tile(img_rgb, None, None)
        tissue_mask = TissueMask()
        mask = tissue_mask(tile)

        OD = RgbToOd(background_intensity)(img_rgb)
        OD = OD[mask].reshape(-1, 3)

        # Remove data with OD intensity less than β
        ODhat = OD[~np.any(OD < beta, axis=1)]

        # Calculate principal components and project input
        V = self.two_principal_components(ODhat)
        proj = np.dot(ODhat, V)

        # Angular coordinates with repect to the principle, orthogonal eigenvectors
        phi = np.arctan2(proj[:, 1], proj[:, 0])

        # Min and max angles
        min_phi = np.percentile(phi, alpha)
        max_phi = np.percentile(phi, 100 - alpha)

        # The two principle stains
        min_v = V.dot(np.array([(np.cos(min_phi), np.sin(min_phi))]).T)
        max_v = V.dot(np.array([(np.cos(max_phi), np.sin(max_phi))]).T)

        # Fill out empty columns in the stain matrix and reoder it
        unordered_stain_matrix = self._complement_stain_matrix(
            self.normalize_columns(np.hstack([min_v, max_v]))
        )
        ordered_stain_matrix = self._reorder_stains(
            unordered_stain_matrix, stains=stains
        )
        return ordered_stain_matrix

    @staticmethod
    def _complement_stain_matrix(stain_matrix: np.ndarray) -> np.ndarray:
        """Generates a complemented stain matrix.

        It is used to fill out empty columns of a stain matrix. Replaces right-most
        column with normalized cross-product of first two columns.

        Parameters
        ----------
        stain_matrix : np.ndarray
            A 3x3 stain calibration matrix with stain color vectors in columns.

        Returns
        -------
        np.ndarray
            A 3x3 complemented stain calibration matrix with a third orthogonal column.
        """

        stain0 = stain_matrix[:, 0]
        stain1 = stain_matrix[:, 1]
        stain2 = np.cross(stain0, stain1)

        # Normalize new vector to have unit norm
        return np.array([stain0, stain1, stain2 / np.linalg.norm(stain2)]).T

    @staticmethod
    def _find_stain_index(reference: np.ndarray, stain_matrix: np.ndarray) -> int:
        """Return index in ``stain_vector`` corresponding to the ``reference`` vector.

        Useful in connection with adaptive deconvolution routines in order to find the
        column corresponding with a certain expected stain.

        Parameters
        ----------
        reference : np.ndarray
            1D array that is the stain vector to find
        stain_matrix : np.ndarray
            2D array of columns the same size as reference.
            The columns should be normalized.

        Returns
        -------
        i : int
            Column of ``stain_matrix`` corresponding to ``reference``

        Notes
        -----
        The index of the vector with the smallest angular distance is returned.
        """
        dot_products = np.dot(reference, stain_matrix)
        return np.argmax(np.abs(dot_products))

    @staticmethod
    def _reorder_stains(stain_matrix: np.ndarray, stains: List[str]) -> np.ndarray:
        """Reorder stains in the ``stain_matrix`` to a specific order.

        This is particularly relevant in Macenko where the order of stains is not
        preserved during stain unmixing.

        Parameters
        ------------
        stain_matrix : np.ndarray
            A 3x3 matrix of stain column vectors.
        stains : list, optional
            List of stain names (order is important).

        Returns
        -------
        np.ndarray
            A re-ordered 3x3 matrix of stain column vectors.
        """

        def _get_channel_order(stain_matrix: np.ndarray) -> List[int]:
            first = MacenkoStainNormalizer._find_stain_index(
                MacenkoStainNormalizer.stain_color_map[stains[0]], stain_matrix
            )
            second = 1 - first
            # If 2 stains, third "stain" is cross product of 1st 2 channels
            # calculated using self._complement_stain_matrix()
            third = 2
            return first, second, third

        def _ordered_stack(stain_matrix: np.ndarray, order: List[int]) -> np.ndarray:
            return np.stack([stain_matrix[..., j] for j in order], -1)

        return _ordered_stack(stain_matrix, _get_channel_order(stain_matrix))
