import numpy as np
import PIL


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
