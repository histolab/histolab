import os
from pathlib import Path

import numpy as np
import skimage.morphology as morph
from PIL import Image
from scipy import ndimage
from skimage import color
from skimage.filters import threshold_otsu

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
        image_arr = np.array(self._image)
        image_gray = color.rgb2gray(image_arr)
        # Check if image is FULL-WHITE
        if (
            np.mean(image_gray.ravel()) > 0.9 and np.std(image_gray.ravel()) < 0.09
        ):  # full or almost white
            return False
        # Calculate the threshold of pixel-values corresponding to FOREGROUND
        # using Threshold-Otsu Method
        thresh = threshold_otsu(image_gray)
        # Filter out the Background
        image_bw = image_gray < thresh
        # Generate a Disk shaped filter of radius=5
        strel = morph.disk(5)
        # Generate Morphological Dilation, i.e. enlarge dark regions, shrinks dark
        # regions
        image_bw_dilated = morph.dilation(image_bw, strel)
        # Fill holes in brightness based on a (minimum) reference structure to look for
        image_bw_filled = ndimage.binary_fill_holes(
            image_bw_dilated, structure=np.ones((5, 5))
        ).astype(np.uint8)

        # Near-zero variance threshold
        # This also includes cases in which there is ALL TISSUE (too clear) or
        # NO TISSUE (zeros)
        if np.var(image_bw_filled) < near_zero_var_threshold:
            return False

        return np.mean(image_bw_filled) > threshold

    def save(self, path):
        """Save tile at given path. The format to use is determined from the filename
        extension (to be compatible to PIL.Image formats). If no extension is provided,
        the image will be saved in png format.

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
