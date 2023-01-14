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


from typing import List, Tuple

import numpy as np
import PIL

from .filters.image_filters import LabToRgb, RgbToLab, RgbToOd
from .masks import TissueMask
from .mixins import LinalgMixin
from .tile import Tile
from .util import np_to_pil


class TransformerStainMatrixMixin:
    """Mixin class implementing ``fit`` and ``transform`` methods for stain normalizers.

    It assumes that the subclass implements a method returning a 3x3 array of stain
    vectors with signature
    ``stain_matrix(self, img_rgb: PIL.Image.Image, background_intensity: int)``.
    """

    def fit(self, target_rgb: PIL.Image.Image, background_intensity: int = 240) -> None:
        """Fit stain normalizer using ``target_img``.

        Parameters
        ----------
        target_rgb : PIL.Image.Image
            Target image for stain normalization. Can be either RGB or RGBA.
        background_intensity : int, optional
            Background transmitted light intensity. Default is 240.
        """
        self.stain_matrix_target = self.stain_matrix(
            target_rgb, background_intensity=background_intensity
        )

        target_concentrations = self._find_concentrations(
            target_rgb, self.stain_matrix_target, background_intensity
        )

        self.max_concentrations_target = np.percentile(
            target_concentrations, 99, axis=1
        )

    def transform(
        self, img_rgb: PIL.Image.Image, background_intensity: int = 240
    ) -> PIL.Image.Image:
        """Normalize staining of ``img_rgb``.

        Parameters
        ----------
        img_rgb : PIL.Image.Image
            Image to normalize.
        background_intensity : int, optional
            Background transmitted light intensity. Default is 240.

        Returns
        -------
        PIL.Image.Image
            Image with normalized stain
        """
        stain_matrix_source = self.stain_matrix(
            img_rgb, background_intensity=background_intensity
        )

        source_concentrations = self._find_concentrations(
            img_rgb, stain_matrix_source, background_intensity
        )

        max_concentrations_source = np.percentile(source_concentrations, 99, axis=1)
        max_concentrations_source = np.divide(
            max_concentrations_source, self.max_concentrations_target
        )
        conc_tmp = np.divide(
            source_concentrations, max_concentrations_source[:, np.newaxis]
        )

        img_norm = np.multiply(
            background_intensity, np.exp(-self.stain_matrix_target.dot(conc_tmp))
        )
        img_norm = np.clip(img_norm, a_min=None, a_max=255)
        img_norm = np.reshape(img_norm.T, (*img_rgb.size[::-1], 3))
        return np_to_pil(img_norm)

    @staticmethod
    def _find_concentrations(
        img_rgb: PIL.Image.Image,
        stain_matrix: np.ndarray,
        background_intensity: int = 240,
    ) -> np.ndarray:
        """Return concentrations of the individual stains in ``img_rgb``.

        Parameters
        ----------
        img_rgb : PIL.Image.Image
            Input image.
        stain_matrix : np.ndarray
            Stain matrix of ``img_rgb``.
        background_intensity : int, optional
            Background transmitted light intensity. Default is 240.

        Returns
        -------
        np.ndarray
            Concentrations of the individual stains in ``img_rgb``.
        """
        od = RgbToOd(background_intensity)(img_rgb)
        # rows correspond to channels (RGB), columns to OD values
        od = np.reshape(od, (-1, 3)).T

        # determine concentrations of the individual stains
        return np.linalg.lstsq(stain_matrix, od, rcond=None)[0]


class MacenkoStainNormalizer(LinalgMixin, TransformerStainMatrixMixin):
    """Stain normalizer using the method of M. Macenko et al. [1]_

    References
    ----------
    .. [1] Macenko, Marc, et al. "A method for normalizing histology slides for
        quantitative analysis." 2009 IEEE International Symposium on Biomedical
        Imaging: From Nano to Macro. IEEE, 2009.
    """

    # normalized OD matrix M for hematoxylin, eosin and DAB stains
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
        """Return stain matrix estimation for color deconvolution with Macenko's method.

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
            if ``stains`` is not a two-stains list
        ValueError
            if the input image is not RGB or RGBA
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

        od = RgbToOd(background_intensity)(img_rgb)
        od = od[mask].reshape(-1, 3)

        # Remove data with OD intensity less than β
        od_hat = od[~np.any(od < beta, axis=1)]

        # Calculate principal components and project input
        V = self.principal_components(od_hat, n_components=2)
        proj = np.dot(od_hat, V)

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


class ReinhardStainNormalizer:
    """Stain normalizer using the method of E. Reinhard et al. [1]_

    References
    ----------
    .. [1] Reinhard, Erik, et al. "Color transfer between images." IEEE Computer
        graphics and applications 21.5 (2001): 34-41.

    """

    def __init__(self):
        self.target_means = None
        self.target_stds = None

    def fit(self, target_rgb: PIL.Image.Image) -> None:
        """Fit stain normalizer using ``target_img``.

        Parameters
        ----------
        target_rgb : PIL.Image.Image
            Target image for stain normalization. Can be either RGB or RGBA.
        """
        means, stds = self._summary_statistics(target_rgb)
        self.target_means = means
        self.target_stds = stds

    def transform(self, img_rgb: PIL.Image.Image) -> PIL.Image.Image:
        """Normalize ``img_rgb`` staining.

        Parameters
        ----------
        img_rgb : PIL.Image.Image
            Image to normalize. Can be either RGB or RGBA.

        Returns
        -------
        PIL.Image.Image
            Image with normalized stain.
        """
        means, stds = self._summary_statistics(img_rgb)
        img_lab = RgbToLab()(img_rgb)

        mask = self._tissue_mask(img_rgb)
        mask = np.dstack((mask, mask, mask))

        masked_img_lab = np.ma.masked_array(img_lab, ~mask)

        norm_lab = (
            ((masked_img_lab - means) * (self.target_stds / stds)) + self.target_means
        ).data

        for i in range(3):
            original = img_lab[:, :, i].copy()
            new = norm_lab[:, :, i].copy()
            original[np.not_equal(~mask[:, :, 0], True)] = 0
            new[~mask[:, :, 0]] = 0
            norm_lab[:, :, i] = new + original

        norm_rgb = LabToRgb()(norm_lab)
        return norm_rgb

    def _summary_statistics(
        self, img_rgb: PIL.Image.Image
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return mean and standard deviation of each channel in LAB color space.

        The statistics are calculated after tissue masking.

        Parameters
        ----------
        img_rgb : PIL.Image.Image
            Input image.

        Returns
        -------
        np.ndarray
            Mean of each channel in LAB color space.
        np.ndarray
            Standard deviation of each channel in LAB color space.
        """
        mask = self._tissue_mask(img_rgb)
        mask = np.dstack((mask, mask, mask))

        img_lab = RgbToLab()(img_rgb)
        mean_per_channel = img_lab.mean(axis=(0, 1), where=mask)
        std_per_channel = img_lab.std(axis=(0, 1), where=mask)
        return mean_per_channel, std_per_channel

    @staticmethod
    def _tissue_mask(img_rgb: PIL.Image.Image) -> np.ndarray:
        """Return a binary mask of the tissue in ``img_rgb``.

        Parameters
        ----------
        img_rgb : PIL.Image.Image
            Input image. Can be either RGB or RGBA.

        Returns
        -------
        np.ndarray
            binary tissue mask.
        """
        tile = Tile(img_rgb, None)
        mask = tile.tissue_mask
        return mask
