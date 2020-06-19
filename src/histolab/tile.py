import os
from pathlib import Path
from typing import Union

import numpy as np
import PIL

from .filters import image_filters as imf
from .filters import morphological_filters as mof
from .types import CoordinatePair
from .util import lazyproperty


class Tile:
    """Provide Tile object representing a tile generated from a Slide object.

    Arguments
    ---------
    image : PIL.Image.Image
        Image describing the tile
    coords : CoordinatePair
        Level 0 Coordinates of the Slide from which the tile was extracted
    level : int, optional
        Level of tile extraction, by default 0
    """

    def __init__(self, image: PIL.Image.Image, coords: CoordinatePair, level: int = 0):
        self._image = image
        self._level = level
        self._coords = coords

    @lazyproperty
    def image(self) -> PIL.Image.Image:
        return self._image

    @lazyproperty
    def level(self) -> int:
        return self._level

    @lazyproperty
    def coords(self) -> CoordinatePair:
        return self._coords

    def has_enough_tissue(
        self, tissue_percent: float = 80.0, near_zero_var_threshold: float = 0.1
    ) -> bool:
        """Check if the tile has enough tissue.

        Parameters
        ----------
        tissue_percent : float, optional
            Number between 0.0 and 100.0 representing the minimum required percentage of
            tissue over the total area of the image, default is 80.0
        near_zero_var_threshold : float, optional
            Minimum image variance after morphological operations (dilation, fill
            holes), default is 0.1

        Returns
        -------
        enough_tissue : bool
            Whether the image has enough tissue, i.e. if the proportion of tissue
            over the total area of the image is more than ``tissue_percent`` and the
            image variance after morphological operations is more than
            ``near_zero_var_threshold``.
        """

        if self._is_almost_white:
            return False

        if not self._has_only_some_tissue(near_zero_var_threshold):
            return False

        if not self._has_tissue_more_than_percent(tissue_percent):
            return False

        return True

    def save(self, path: Union[str, bytes, os.PathLike]) -> None:
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

    @lazyproperty
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

    def _has_only_some_tissue(self, near_zero_var_threshold: float = 0.1) -> np.bool_:
        """Check if the tile is composed by only some tissue.

        Parameters
        ----------
        near_zero_var_threshold : float, optional
            Minimum image variance after morphological operations (dilation, fill holes)
            to consider the image to be composed by only some tissue, default is 0.1

        Returns
        -------
        bool
            True if the image is composed by only some tissue. False if the tile is
            composed by all tissue or by no tissue at all.
        """
        filters = self._enough_tissue_mask_filters
        tissue_mask = filters(self._image)

        return np.var(tissue_mask) > near_zero_var_threshold

    @lazyproperty
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
        image_gray_arr = image_gray_arr / 255

        return (
            np.mean(image_gray_arr.ravel()) > 0.9
            and np.std(image_gray_arr.ravel()) < 0.09
        )

    def _has_tissue_more_than_percent(self, tissue_percent: float = 80.0) -> bool:
        """Check if tissue represent more than ``tissue_percent`` % of the image.

        Parameters
        ----------
        tissue_percent : float, optional
            Number between 0.0 and 100.0 representing the minimum required percentage of
            tissue over the total area of the image, default is 80.0

        Returns
        -------
        bool
            True if tissue represent more than ``tissue_percent`` % of the image, False
            otherwise.
        """
        filters = self._enough_tissue_mask_filters
        tissue_mask = filters(self._image)

        return np.mean(tissue_mask) * 100 > tissue_percent
