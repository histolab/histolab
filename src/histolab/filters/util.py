import math
import multiprocessing
import os

import numpy as np
import PIL
import scipy.ndimage.morphology as sc_morph
import skimage.color as sk_color
import skimage.exposure as sk_exposure
import skimage.feature as sk_feature
import skimage.filters as sk_filters
import skimage.future as sk_future
import skimage.morphology as sk_morphology
import skimage.segmentation as sk_segmentation
from PIL import Image, ImageOps
from ..util import np_to_pil

def mask_percent(mask: np.ndarray) -> float:
    """Compute mask percentage of pixels different from zero.

    Parameters
    ----------
    mask : np.ndarray
        Input mask as Numpy array

    Returns
    -------
    float
        Percentage of image masked
    """

    mask_percentage = 100 - np.count_nonzero(mask) / mask.size * 100
    return mask_percentage


def tissue_percent(img: PIL.Image.Image) -> float:
    """Compute percentage of tissue in an image.

        Parameters
        ----------
        img : PIL.Image.Image
            Input image

        Returns
        -------
        float
            Percentage of image that is not masked
        """
    return 100 - mask_percent(img)
