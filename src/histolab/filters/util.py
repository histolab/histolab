import numpy as np


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
