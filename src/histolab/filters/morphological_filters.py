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
"""
Notice that both the input and the output of morphological filters are binary masks.
"""


from abc import abstractmethod

import numpy as np
import scipy.ndimage.morphology
import skimage.morphology

from . import morphological_filters_functional as F
from .image_filters import Filter

try:
    from typing import Protocol, runtime_checkable
except ImportError:
    from typing_extensions import Protocol, runtime_checkable


@runtime_checkable
class MorphologicalFilter(Filter, Protocol):
    """Morphological filter base class"""

    @abstractmethod
    def __call__(self, np_mask: np.ndarray) -> np.ndarray:
        pass  # pragma: no cover


class RemoveSmallObjects(MorphologicalFilter):
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



    Example:
        >>> from PIL import Image
        >>> from histolab.filters.image_filters import RgbToGrayscale, OtsuThreshold
        >>> from histolab.filters.morphological_filters import RemoveSmallObjects
        >>> import os
        >>> img_dir = os.path.join(os.path.abspath(os.getcwd()), "tests/fixtures/pil-images-rgb")
        >>> image_rgb = Image.open(os.path.join(img_dir,"tcga-lung-rgb.png")) # Read an RGB image
        >>> rgb_to_grayscale = RgbToGrayscale() # Instantiate the grayscale filter
        >>> otsu_threshold = OtsuThreshold() # Instantiate the Otsu filter
        >>> remove_small_objects = RemoveSmallObjects() # Instantiate the remove small objects filter
        >>> image_gray = rgb_to_grayscale(image_rgb) # Apply the grayscale filter
        >>> binary_image = otsu_threshold(image_gray) # Apply the Otsu filter to the grayscale image
        >>> image_no_small_objects = remove_small_objects(binary_image) # Remove small objects from the image
    """  # noqa

    def __init__(
        self,
        min_size: int = 3000,
        avoid_overmask: bool = True,
        overmask_thresh: int = 95,
    ):
        self.min_size = min_size
        self.avoid_overmask = avoid_overmask
        self.overmask_thresh = overmask_thresh

    def __call__(self, np_mask: np.ndarray) -> np.ndarray:
        return F.remove_small_objects(
            np_mask, self.min_size, self.avoid_overmask, self.overmask_thresh
        )


class RemoveSmallHoles(MorphologicalFilter):
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


    Example:
        >>> from PIL import Image
        >>> from histolab.filters.image_filters import RgbToGrayscale, OtsuThreshold
        >>> from histolab.filters.morphological_filters import RemoveSmallHoles
        >>> import os
        >>> img_dir = os.path.join(os.path.abspath(os.getcwd()), "tests/fixtures/pil-images-rgb")
        >>> image_rgb = Image.open(os.path.join(img_dir,"tcga-lung-rgb.png")) # Read an RGB image
        >>> rgb_to_grayscale = RgbToGrayscale() # Instantiate the grayscale filter
        >>> otsu_threshold = OtsuThreshold() # Instantiate the Otsu filter
        >>> remove_small_holes = RemoveSmallHoles() # Instantiate the remove small holes filter
        >>> image_gray = rgb_to_grayscale(image_rgb) # Apply the grayscale filter
        >>> binary_image = otsu_threshold(image_gray) # Apply the Otsu filter to the grayscale image
        >>> image_no_small_holes = remove_small_holes(binary_image) # Remove small holes from the image
    """  # noqa

    def __init__(self, area_threshold: int = 3000):
        self.area_threshold = area_threshold

    def __call__(self, np_mask: np.ndarray) -> np.ndarray:
        return skimage.morphology.remove_small_holes(np_mask, self.area_threshold)


class BinaryErosion(MorphologicalFilter):
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


    Example:
        >>> from PIL import Image
        >>> from histolab.filters.image_filters import RgbToGrayscale, OtsuThreshold
        >>> from histolab.filters.morphological_filters import BinaryErosion
        >>> import os
        >>> img_dir = os.path.join(os.path.abspath(os.getcwd()), "tests/fixtures/pil-images-rgb")
        >>> image_rgb = Image.open(os.path.join(img_dir,"tcga-lung-rgb.png")) # Read an RGB image
        >>> rgb_to_grayscale = RgbToGrayscale() # Instantiate the grayscale filter
        >>> otsu_threshold = OtsuThreshold() # Instantiate the Otsu filter
        >>> binary_erosion = BinaryErosion(disk_size=6) # Instantiate the binary erosion filter
        >>> image_gray = rgb_to_grayscale(image_rgb) # Apply the grayscale filter
        >>> binary_image = otsu_threshold(image_gray) # Apply the Otsu filter to the grayscale image
        >>> image_eroded = binary_erosion(binary_image) # Apply the binary erosion filter
    """  # noqa

    def __init__(self, disk_size: int = 5, iterations: int = 1):
        self.disk_size = disk_size
        self.iterations = iterations

    def __call__(self, np_mask: np.ndarray) -> np.ndarray:
        if not np.array_equal(np_mask, np_mask.astype(bool)):
            raise ValueError("Mask must be binary")

        return scipy.ndimage.morphology.binary_erosion(
            np_mask, skimage.morphology.disk(self.disk_size), self.iterations
        )


class BinaryDilation(MorphologicalFilter):
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


    Example:
        >>> from PIL import Image
        >>> from histolab.filters.image_filters import RgbToGrayscale, OtsuThreshold
        >>> from histolab.filters.morphological_filters import BinaryDilation
        >>> import os
        >>> img_dir = os.path.join(os.path.abspath(os.getcwd()), "tests/fixtures/pil-images-rgb")
        >>> image_rgb = Image.open(os.path.join(img_dir,"tcga-lung-rgb.png")) # Read an RGB image
        >>> rgb_to_grayscale = RgbToGrayscale() # Instantiate the grayscale filter
        >>> otsu_threshold = OtsuThreshold() # Instantiate the Otsu filter
        >>> binary_dilation = BinaryDilation() # Instantiate the binary dilation filter
        >>> image_gray = rgb_to_grayscale(image_rgb) # Apply the grayscale filter
        >>> binary_image = otsu_threshold(image_gray) # Apply the Otsu filter to the grayscale image
        >>> image_dilated = binary_dilation(binary_image) # Apply the binary dilation filter
    """  # noqa

    def __init__(self, disk_size: int = 5, iterations: int = 1):
        self.disk_size = disk_size
        self.iterations = iterations

    def __call__(self, np_mask: np.ndarray) -> np.ndarray:
        if not np.array_equal(np_mask, np_mask.astype(bool)):
            raise ValueError("Mask must be binary")
        return scipy.ndimage.morphology.binary_dilation(
            np_mask, skimage.morphology.disk(self.disk_size), self.iterations
        )


class BinaryFillHoles(MorphologicalFilter):
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


class BinaryOpening(MorphologicalFilter):
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


    Example:
        >>> from PIL import Image
        >>> from histolab.filters.image_filters import RgbToGrayscale, OtsuThreshold
        >>> from histolab.filters.morphological_filters import BinaryOpening
        >>> import os
        >>> img_dir = os.path.join(os.path.abspath(os.getcwd()), "tests/fixtures/pil-images-rgb")
        >>> image_rgb = Image.open(os.path.join(img_dir,"tcga-lung-rgb.png")) # Read an RGB image
        >>> rgb_to_grayscale = RgbToGrayscale() # Instantiate the grayscale filter
        >>> otsu_threshold = OtsuThreshold() # Instantiate the Otsu filter
        >>> binary_opening = BinaryOpening() # Instantiate the binary opening filter
        >>> image_gray = rgb_to_grayscale(image_rgb) # Apply the grayscale filter
        >>> binary_image = otsu_threshold(image_gray) # Apply the Otsu filter to the grayscale image
        >>> image_opened = binary_opening(binary_image) # Apply the binary opening filter
    """  # noqa

    def __init__(self, disk_size: int = 3, iterations: int = 1):
        self.disk_size = disk_size
        self.iterations = iterations

    def __call__(self, np_mask: np.ndarray) -> np.ndarray:
        if not np.array_equal(np_mask, np_mask.astype(bool)):
            raise ValueError("Mask must be binary")
        return scipy.ndimage.morphology.binary_opening(
            np_mask, skimage.morphology.disk(self.disk_size), self.iterations
        )


class BinaryClosing(MorphologicalFilter):
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


    Example:
        >>> from PIL import Image
        >>> from histolab.filters.image_filters import RgbToGrayscale, OtsuThreshold
        >>> from histolab.filters.morphological_filters import BinaryClosing
        >>> import os
        >>> img_dir = os.path.join(os.path.abspath(os.getcwd()), "tests/fixtures/pil-images-rgb")
        >>> image_rgb = Image.open(os.path.join(img_dir,"tcga-lung-rgb.png")) # Read an RGB image
        >>> rgb_to_grayscale = RgbToGrayscale() # Instantiate the grayscale filter
        >>> otsu_threshold = OtsuThreshold() # Instantiate the Otsu filter
        >>> binary_closing = BinaryClosing() # Instantiate the binary closing filter
        >>> image_gray = rgb_to_grayscale(image_rgb) # Apply the grayscale filter
        >>> binary_image = otsu_threshold(image_gray) # Apply the Otsu filter to the grayscale image
        >>> image_closed = binary_closing(binary_image) # Apply the binary closing filter
    """  # noqa

    def __init__(self, disk_size: int = 3, iterations: int = 1):
        self.disk_size = disk_size
        self.iterations = iterations

    def __call__(self, np_mask: np.ndarray) -> np.ndarray:
        if not np.array_equal(np_mask, np_mask.astype(bool)):
            raise ValueError("Mask must be binary")
        return scipy.ndimage.morphology.binary_closing(
            np_mask, skimage.morphology.disk(self.disk_size), self.iterations
        )


class WatershedSegmentation(MorphologicalFilter):
    """Segment and label an binary mask with Watershed segmentation [1]_

    The watershed algorithm treats pixels values as a local topography (elevation).

    Parameters
    ----------
    np_mask : np.ndarray
        Input mask
    region_shape : int, optional
        The local region within which to search for image peaks is defined as a squared
        area region_shape x region_shape. Default is 6.

    Returns
    -------
    np.ndarray
        Labelled segmentation mask

    References
    --------
    .. [1] Watershed segmentation.
       https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_watershed.html


    Example:
        >>> import numpy as np
        >>> from histolab.filters.morphological_filters import WatershedSegmentation
        >>> mask = np.array([[0,1],[1,0]]) # or np.load("/path/my_array_mask.npy")
        >>> watershed_segmentation = WatershedSegmentation() # Instantiate the Watershed filter
        >>> mask_segmented = watershed_segmentation(mask) # Apply the Watershed filter to the mask
    """  # noqa

    def __init__(self, region_shape: int = 6) -> None:
        self.region_shape = region_shape

    def __call__(self, np_mask: np.ndarray) -> np.ndarray:
        return F.watershed_segmentation(np_mask, self.region_shape)


class WhiteTopHat(MorphologicalFilter):
    """Return white top hat of an image.

    The white top hat of an image is defined as the image minus its morphological
    opening with respect to a structuring element. This operation returns the bright
    spots of the image that are smaller than the structuring element.

    Parameters
    ----------
    np_mask : np.ndarray (arbitrary shape, int or bool type)
        Numpy array of the binary mask
    structure : np.ndarray, optional
        The neighborhood expressed as an array of 1 and 0. If None, use cross-shaped
        structuring element (connectivity=1).

    """

    def __init__(self, structure: np.ndarray = None):
        self.structure = structure

    def __call__(self, np_mask: np.ndarray) -> np.ndarray:
        return skimage.morphology.white_tophat(np_mask, self.structure)
