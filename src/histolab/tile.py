import os
from pathlib import Path

import numpy as np
from PIL import Image

from .filters import image_filters as imf
from .filters import morphological_filters as mof
from .types import CoordinatePair


class Tile:
    def __init__(self, image: Image.Image, coords: CoordinatePair, level: int = 0):
        self._image = image
        self._level = level
        self._coords = coords

    @property
    def image(self) -> Image.Image:
        return self._image

    @property
    def level(self) -> int:
        return self._level

    @property
    def coords(self) -> CoordinatePair:
        return self._coords

    def has_enough_tissue(
        self, threshold: float = 0.8, near_zero_var_threshold: float = 0.1
    ) -> bool:
        """Check if the tile has enough tissue.

        Parameters
        ----------
        threshold : float, optional
            Number between 0.0 and 1.0 representing the minimum required proportion of
            tissue over the total area of the image, default is 0.8
        near_zero_var_threshold : float, optional
            Minimum image variance after morphological operations (dilation, fill
            holes), default is 0.1

        Returns
        -------
        enough_tissue : bool
            Whether the image has enough tissue, i.e. if the proportion of tissue
            over the total area of the image is more than ``threshold`` and the image
            variance after morphological operations is more than
            ``near_zero_var_threshold``.
        """

        if self._is_almost_white:
            return False

        filters = self._enough_tissue_mask_filters

        image_filtered = filters(self._image)
        image_filtered_arr = np.array(image_filtered)

        # Near-zero variance threshold
        # This also includes cases in which there is ALL TISSUE (too clear) or
        # NO TISSUE (zeros)
        if np.var(image_filtered_arr) < near_zero_var_threshold:
            return False

        return np.mean(image_filtered_arr) > threshold

    def save(self, path):
        """Save tile at given path.

        The format to use is determined from the filename extension (to be compatible to
        PIL.Image formats). If no extension is provided, the image will be saved in png
        format.

        Parameters
        ---------
        path: str or pathlib.Path
            Path to which the tile is saved.

        """
        ext = os.path.splitext(path)[1]

        if not ext:
            path = f"{path}.png"

        Path(path).parent.mkdir(parents=True, exist_ok=True)

        self._image.save(path)

    @property
    def _enough_tissue_mask_filters(self) -> imf.Compose:
        """Return a filters composition to get a binary mask to estimate tissue.

        Returns
        -------
        imf.Compose
            Filters composition
        """
        filters = imf.Compose(
            [
                imf.RgbToGrayscale(),
                imf.OtsuThreshold(),
                mof.BinaryDilation(),
                mof.BinaryFillHoles(structure=np.ones((5, 5))),
            ]
        )
        return filters

    @property
    def _is_almost_white(self) -> bool:
        """Check if the image is almost white.

        Returns
        -------
        bool
            True if the image is almost white, False otherwise
        """
        rgb2grey = imf.RgbToGrayscale()
        image_gray = rgb2grey(self._image)
        image_gray_arr = np.array(image_gray)

        return (
            np.mean(image_gray_arr.ravel()) < 0.9
            and np.std(image_gray_arr.ravel()) > 0.09
        )
