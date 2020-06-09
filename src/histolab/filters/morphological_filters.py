# encoding: utf-8

# ------------------------------------------------------------------------
# Copyright 2020 All Histolab Contributors
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

import numpy as np
import scipy.ndimage.morphology
import skimage.morphology

from . import morphological_filters_functional as F


class RemoveSmallObjects(object):
    """Remove objects smaller than the specified size.

    If avoid_overmask is True, this function can recursively call itself with
    progressively halved minimum size objects to avoid removing too many
    objects in the mask.

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

    def __init__(
        self,
        min_size: int = 3000,
        avoid_overmask: bool = True,
        overmask_thresh: int = 95,
    ):
        self.min_size = min_size
        self.avoid_overmask = avoid_overmask
        self.overmask_thresh = overmask_thresh

    def __call__(self, np_img) -> np.ndarray:
        return F.remove_small_objects(
            np_img, self.min_size, self.avoid_overmask, self.overmask_thresh
        )

    def __repr__(self):
        return self.__class__.__name__ + "()"


class RemoveSmallHoles(object):
    """Remove holes smaller than a specified size.

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

    def __init__(self, area_threshold: int = 3000):
        self.area_threshold = area_threshold

    def __call__(self, np_mask) -> np.ndarray:
        return skimage.morphology.remove_small_holes(np_mask, self.area_threshold)

    def __repr__(self):
        return self.__class__.__name__ + "()"


class BinaryErosion(object):
    """Erode a binary mask.

    Parameters
    ----------
    np_mask : np.ndarray (arbitrary shape, int or bool type)
        Numpy array of the binary mask
    disk_size : int, optional (default is 5)
        Radius of the disk structuring element used for erosion.
    iterations : int, optional (default is 1)
        How many times to repeat the erosion.

    Returns
    -------
    np.ndarray
        Mask after the erosion
    """

    def __init__(self, disk_size: int = 5, iterations: int = 1):
        self.disk_size = disk_size
        self.iterations = iterations

    def __call__(self, np_mask: np.ndarray) -> np.ndarray:
        if not np.array_equal(np_mask, np_mask.astype(bool)):
            raise ValueError("Mask must be binary")

        return scipy.ndimage.morphology.binary_erosion(
            np_mask, skimage.morphology.disk(self.disk_size), self.iterations
        )

    def __repr__(self):
        return self.__class__.__name__ + "()"


class BinaryDilation(object):
    """Dilate a binary mask.

    Parameters
    ----------
    np_mask : np.ndarray (arbitrary shape, int or bool type)
        Numpy array of the binary mask
    disk_size : int, optional (default is 5)
        Radius of the disk structuring element used for dilation.
    iterations : int, optional (default is 1)
        How many times to repeat the dilation.

    Returns
    -------
    np.ndarray
        Mask after the dilation
    """

    def __init__(self, disk_size: int = 5, iterations: int = 1):
        self.disk_size = disk_size
        self.iterations = iterations

    def __call__(self, np_mask: np.ndarray) -> np.ndarray:
        if not np.array_equal(np_mask, np_mask.astype(bool)):
            raise ValueError("Mask must be binary")
        return scipy.ndimage.morphology.binary_dilation(
            np_mask, skimage.morphology.disk(self.disk_size), self.iterations
        )

    def __repr__(self):
        return self.__class__.__name__ + "()"


class BinaryFillHoles(object):
    """Fill the holes in binary objects.

    Parameters
    ----------
    np_img : np.ndarray (arbitrary shape, int or bool type)
        Numpy array of the binary mask
    structure: np.ndarray, optional
        Structuring element used in the computation; The default element yields the
        intuitive result where all holes in the input have been filled.

    Returns
    -------
    np.ndarray
        Transformation of the initial image input where holes have been filled.
    """

    def __init__(self, structure: np.ndarray = None):
        self.structure = structure

    def __call__(self, np_img: np.ndarray) -> np.ndarray:
        return scipy.ndimage.morphology.binary_fill_holes(np_img, self.structure)


class BinaryOpening(object):
    """Open a binary mask.

    Opening is an erosion followed by a dilation. Opening can be used to remove
    small objects.

    Parameters
    ----------
    np_mask : np.ndarray (arbitrary shape, int or bool type)
        Numpy array of the binary mask
    disk_size : int, optional (default is 3)
        Radius of the disk structuring element used for opening.
    iterations : int, optional (default is 1)
            How many times to repeat the opening.

    Returns
    -------
    np.ndarray
        Mask after the opening
    """

    def __init__(self, disk_size: int = 3, iterations: int = 1):
        self.disk_size = disk_size
        self.iterations = iterations

    def __call__(self, np_mask) -> np.ndarray:
        if not np.array_equal(np_mask, np_mask.astype(bool)):
            raise ValueError("Mask must be binary")
        return scipy.ndimage.morphology.binary_opening(
            np_mask, skimage.morphology.disk(self.disk_size), self.iterations
        )

    def __repr__(self):
        return self.__class__.__name__ + "()"


class BinaryClosing(object):
    """Close a binary mask.

    Closing is a dilation followed by an erosion. Closing can be used to remove
    small holes.

    Parameters
    ----------
    np_mask : np.ndarray (arbitrary shape, int or bool type)
        Numpy array of the binary mask
    disk_size : int, optional (default is 3)
        Radius of the disk structuring element used for closing.
    iterations : int, optional (default is 1)
        How many times to repeat the closing.

    Returns
    -------
    np.ndarray
        Mask after the closing
    """

    def __init__(self, disk_size: int = 3, iterations: int = 1):
        self.disk_size = disk_size
        self.iterations = iterations

    def __call__(self, np_mask) -> np.ndarray:
        if not np.array_equal(np_mask, np_mask.astype(bool)):
            raise ValueError("Mask must be binary")
        return scipy.ndimage.morphology.binary_closing(
            np_mask, skimage.morphology.disk(self.disk_size), self.iterations
        )

    def __repr__(self):
        return self.__class__.__name__ + "()"
