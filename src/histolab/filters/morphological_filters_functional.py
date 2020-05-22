import numpy as np
import skimage.morphology as sk_morphology

from .util import mask_percent


def remove_small_objects(
    np_img: np.ndarray,
    min_size: int = 3000,
    avoid_overmask: bool = True,
    overmask_thresh: int = 95,
) -> np.ndarray:
    """Remove connected components which size is less than min_size.

    is True, this function can recursively call itself with progressively
    to avoid removing too many objects in the mask.

    Parameters
    ----------
    np_img : np.ndarray (arbitrary shape, int or bool type)
        Input mask
    min_size : int, optional
        Minimum size of small object to remove. Default is 3000
    avoid_overmask : bool, optional (default is True)
        If True, avoid masking above the overmask_thresh percentage.
    overmask_thresh : int, optional (default is 95)
        If avoid_overmask is True, avoid masking above this threshold percentage value.

    Returns
    -------
    np.ndarray
        Mask with small objects filtered out
    """
    mask_no_small_object = sk_morphology.remove_small_objects(np_img, min_size)
    if (
        avoid_overmask
        and mask_percent(mask_no_small_object) >= overmask_thresh
        and min_size >= 1
    ):
        new_min_size = min_size // 2
        mask_no_small_object = remove_small_objects(
            np_img, new_min_size, avoid_overmask, overmask_thresh
        )
    return mask_no_small_object
