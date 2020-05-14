import scipy.ndimage.morphology
import skimage.morphology
import numpy as np

from . import morphological_filters_functional as F


class RemoveSmallObjects(object):
    """Remove objects smaller than the specified size
    """

    def __call__(
        self,
        np_img,
        min_size: int = 3000,
        avoid_overmask: bool = True,
        overmask_thresh: int = 95,
    ) -> np.ndarray:
        """Remove connected components which size is less than min_size. If avoid_overmask
        is True, this function can recursively call itself with progressively halved minimum size objects
        to avoid removing too many objects in the mask.

        Parameters
        ----------
        np_img : np.ndarray (arbitrary shape, int or bool type)
            Input mask
        min_size : int, optional
            Minimum size of small object to remove. Default is 3000
        avoid_overmask : bool, optional
            If True, avoid masking above the overmask_thresh percentage. Default is True
        overmask_thresh : int, optional
            If avoid_overmask is True, avoid masking above this threshold percentage value.
            Default is 95

        Returns
        -------
        np.ndarray
                Mask with small objects filtered out
        """
        return F.remove_small_objects(np_img, min_size, avoid_overmask, overmask_thresh)

    def __repr__(self):
        return self.__class__.__name__ + "()"


class RemoveSmallHoles(object):
    """Remove holes smaller than the specified size.
    """

    def __call__(self, np_img, area_threshold: int = 3000) -> np.ndarray:
        """Remove holes that have size less than min_size.

        Parameters
        ----------
        np_img : np.ndarray (arbitrary shape, int or bool type)
            Input mask
        area_threshold: int, optional (default is 3000)
            Remove small holes below this size.

        Returns
        -------
        np.ndarray
            Mask with small holes filtered out
        """
        return skimage.morphology.remove_small_holes(
            np_img, area_threshold=area_threshold
        )

    def __repr__(self):
        return self.__class__.__name__ + "()"


class BinaryErosion(object):
    """Erode a binary mask.
    """

    def __call__(
        self, np_img: np.ndarray, disk_size: int = 5, iterations: int = 1
    ) -> np.ndarray:
        """Erode a binary mask.

        Parameters
        ----------
        np_img : np.ndarray (arbitrary shape, int or bool type)
            Numpy array of the binary mask
        disk_size : int, optional (default is 5)
            Radius of the disk structuring element used for erosion.
        iterations: int, optional (default is 1)
            How many times to repeat the erosion.

    Returns
    -------
    np.ndarray
        Mask after the erosion
    """
        return scipy.ndimage.morphology.binary_erosion(
            np_img, skimage.morphology.disk(disk_size), iterations=iterations
        )

    def __repr__(self):
        return self.__class__.__name__ + "()"


class BinaryDilation(object):
    """Dilate a binary mask.
    """

    def __call__(
        self, np_img: np.ndarray, disk_size: int = 5, iterations: int = 1
    ) -> np.ndarray:
        """
        Dilate a binary mask

        Parameters
        ----------
        np_img : np.ndarray (arbitrary shape, int or bool type)
            Numpy array of the binary mask
        disk_size : int, optional (default is 5)
            Radius of the disk structuring element used for dilation.
        iterations: int, optional (default is 1)
            How many times to repeat the dilation.

        Returns
        -------
        np.ndarray
            Mask after the dilation
        """
        return scipy.ndimage.morphology.binary_dilation(
            np_img, skimage.morphology.disk(disk_size), iterations=iterations
        )

    def __repr__(self):
        return self.__class__.__name__ + "()"


class BinaryOpening(object):
    """Open a binary mask.
    """

    def __call__(self, np_img, disk_size: int = 3, iterations: int = 1) -> np.ndarray:
        """Open a binary object (bool, float, or uint8). Opening is an erosion followed by a dilation.
        Opening can be used to remove small objects.

        Parameters
        ----------
        np_img : np.ndarray (arbitrary shape, int or bool type)
            Numpy array of the binary mask
        disk_size : int, optional (default is 5)
            Radius of the disk structuring element used for opening.
        iterations: int, optional (default is 1)
            How many times to repeat the opening.

        Returns
        -------
        np.ndarray
            Mask after the opening
        """
        return scipy.ndimage.morphology.binary_opening(
            np_img, skimage.morphology.disk(disk_size), iterations=iterations
        )

    def __repr__(self):
        return self.__class__.__name__ + "()"


class BinaryClosing(object):
    """Close a binary mask.
    """

    def __call__(self, np_img, disk_size: int = 3, iterations: int = 1) -> np.ndarray:
        """Close a binary mask. Closing is a dilation followed by an erosion.
        Closing can be used to remove small holes.

        Parameters
        ----------
        np_img : np.ndarray (arbitrary shape, int or bool type)
            Numpy array of the binary mask
        disk_size : int, optional (default is 5)
            Radius of the disk structuring element used for closing.
        iterations: int, optional (default is 1)
            How many times to repeat the closing.

        Returns
        -------
        np.ndarray
            Mask after the closing
        """
        return scipy.ndimage.morphology.binary_closing(
            np_img, skimage.morphology.disk(disk_size), iterations=iterations
        )

    def __repr__(self):
        return self.__class__.__name__ + "()"
