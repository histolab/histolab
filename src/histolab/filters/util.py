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


def mask_difference(minuend: np.ndarray, subtrahend: np.ndarray) -> np.ndarray:
    """Return the element-wise difference between two binary masks.

    Parameters
    ----------
    minuend : np.ndarray
        The mask from which another is subtracted
    subtrahend : np.ndarray
        The mask that is to be subtracted

    Returns
    -------
    np.ndarray
        The element-wise difference between the two binary masks
    """
    difference = minuend.astype("int8") - subtrahend.astype("int8")
    difference[difference == -1] = 0
    return difference.astype("bool")
