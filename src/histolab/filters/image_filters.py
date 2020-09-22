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

import operator
from typing import Any, Callable, List, Union

import numpy as np
import PIL

from .. import util as U
from . import image_filters_functional as F


class Compose:
    """Composes several filters together.

    Parameters
    ----------
    filters : list of Filters
        List of filters to compose
    """

    def __init__(self, filters: List[Any]) -> None:
        self.filters = filters

    def __call__(self, img: PIL.Image.Image) -> Union[PIL.Image.Image, np.ndarray]:
        for filter_ in self.filters:
            img = filter_(img)
        return img


class Lambda:
    """Apply a user-defined lambda as a filter.

    Inspired from:
    https://pytorch.org/docs/stable/_modules/torchvision/transforms/transforms.html#Lambda

    Parameters
    ----------
    lambd : callable
        Lambda/function to be used as a filter.

    Returns
    -------
    PIL.Image.Image
        The image with the function applied.
    """

    def __init__(self, lambd: Callable[[PIL.Image.Image], PIL.Image.Image]) -> None:
        assert callable(lambd), repr(type(lambd).__name__) + " object is not callable"
        self.lambd = lambd

    def __call__(self, img: PIL.Image.Image) -> Union[PIL.Image.Image, np.ndarray]:
        return self.lambd(img)

    def __repr__(self) -> str:
        return self.__class__.__name__ + "()"


class ToPILImage:
    """Convert a ndarray to a PIL Image, while preserving the value range.

    Parameters
    ----------
    np_img : np.ndarray
        The image represented as a NumPy array.

    Returns
    -------
    PIL.Image.Image
        The image represented as PIL Image
    """

    def __call__(self, np_img: np.ndarray) -> PIL.Image.Image:
        return U.np_to_pil(np_img)

    def __repr__(self) -> str:
        return self.__class__.__name__ + "()"


class ApplyMaskImage:
    """Mask image with the provided binary mask.

    Parameters
    ----------
    img : PIL.Image.Image
        Input image
    mask : np.ndarray
        Binary mask

    Returns
    -------
    PIL.Image.Image
        Image with the mask applied
    """

    def __init__(self, img: PIL.Image.Image) -> None:
        self.img = img

    def __call__(self, mask: np.ndarray) -> PIL.Image.Image:
        return U.apply_mask_image(self.img, mask)

    def __repr__(self) -> str:
        return self.__class__.__name__ + "()"


class Invert:
    """Invert an image, i.e. take the complement of the correspondent array.

    Parameters
    ----------
    img : PIL.Image.Image
        Input image

    Returns
    -------
    PIL.Image.Image
        Inverted image
    """

    def __call__(self, img: PIL.Image.Image) -> PIL.Image.Image:
        return F.invert(img)

    def __repr__(self) -> str:
        return self.__class__.__name__ + "()"


class RgbToGrayscale:
    """Convert an RGB image to a grayscale image.

    Parameters
    ----------
    img : PIL.Image.Image
        Input image

    Returns
    -------
    PIL.Image.Image
        Grayscale image
    """

    def __call__(self, img: PIL.Image.Image) -> PIL.Image.Image:
        return PIL.ImageOps.grayscale(img)

    def __repr__(self) -> str:
        return self.__class__.__name__ + "()"


class RgbToHed:
    """Convert RGB channels to HED channels.

    image color space (RGB) is converted to Hematoxylin-Eosin-Diaminobenzidine space.

    Parameters
    ----------
    img : PIL.Image.Image
        Input image

    Returns
    -------
    PIL.Image.Image
        Image in HED space
    """

    def __call__(self, img: PIL.Image.Image) -> PIL.Image.Image:
        hed = F.rgb_to_hed(img)
        return hed

    def __repr__(self) -> str:
        return self.__class__.__name__ + "()"


class RgbToLab:
    """Convert from the sRGB color space to the CIE Lab colorspace.

    sRGB color space reference: IEC 61966-2-1:1999

    Parameters
    ----------
    img : PIL.Image.Image
        Input image
    illuminant : {“A”, “D50”, “D55”, “D65”, “D75”, “E”}, optional
        The name of the illuminant (the function is NOT case sensitive).
    observer : {“2”, “10”}, optional
        The aperture angle of the observer.

    Returns
    -------
    PIL.Image.Image
        Image in LAB space

    Raises
    ------
    Exception
        If the image mode is not RGB
    """

    def __init__(self, illuminant: str = "D65", observer: int = "2") -> None:
        self.illuminant = illuminant
        self.observer = observer

    def __call__(self, img: PIL.Image.Image) -> PIL.Image.Image:
        lab = F.rgb_to_lab(img)
        return lab

    def __repr__(self) -> str:
        return self.__class__.__name__ + "()"


class HematoxylinChannel:
    """Obtain Hematoxylin channel from RGB image.

    Input image is first converted into HED space and the hematoxylin channel is
    rescaled for increased contrast.

    Parameters
    ----------
    img : Image.Image
        Input RGB image

    Returns
    -------
    Image.Image
        Grayscale image corresponding to input image with Hematoxylin channel enhanced.
    """

    def __call__(self, img):
        hematoxylin = F.hematoxylin_channel(img)
        return hematoxylin

    def __repr__(self):
        return self.__class__.__name__ + "()"


class EosinChannel:
    """Obtain Eosin channel from RGB image.

    Input image is first converted into HED space and the Eosin channel is
    rescaled for increased contrast.

    Parameters
    ----------
    img : Image.Image
        Input RGB image

    Returns
    -------
    Image.Image
        Grayscale image corresponding to input image with Eosin channel enhanced.
    """

    def __call__(self, img):
        eosin = F.eosin_channel(img)
        return eosin

    def __repr__(self):
        return self.__class__.__name__ + "()"


class RgbToHsv:
    """Convert RGB channels to HSV channels.

    image color space (RGB) is converted to Hue - Saturation - Value (HSV) space.

    Parameters
    ----------
    img : PIL.Image.Image
        Input image

    Returns
    -------
    PIL.Image.Image
        Image in HED space
    """

    def __call__(self, img: PIL.Image.Image) -> PIL.Image.Image:
        hsv = F.rgb_to_hsv(img)
        return hsv

    def __repr__(self) -> str:
        return self.__class__.__name__ + "()"


class StretchContrast:
    """Increase image contrast.

    Th contrast in image is increased based on intensities in a specified range

    Parameters
    ----------
    img: PIL.Image.Image
        Input image
    low: int, optional
        Range low value (0 to 255). Default is 40.
    high: int, optional
        Range high value (0 to 255). Default is 60

    Returns
    -------
    PIL.Image.Image
        Image with contrast enhanced.
    """

    def __init__(self, low: int = 40, high: int = 60) -> None:
        self.low = low
        self.high = high

    def __call__(self, img: PIL.Image.Image) -> PIL.Image.Image:
        stretch_contrast = F.stretch_contrast(img, self.low, self.high)
        return stretch_contrast

    def __repr__(self) -> str:
        return self.__class__.__name__ + "()"


class HistogramEqualization:
    """Increase image contrast using histogram equalization.

    The input image (gray or RGB) is filterd using histogram equalization to increase
    contrast.

    Parameters
    ----------
    img : PIL.Image.Image
        Input image.
    n_bins : int. optional
        Number of histogram bins. Default is 256.

    Returns
    -------
    PIL.Image.Image
        Image with contrast enhanced by histogram equalization.
    """

    def __init__(self, n_bins: int = 256) -> None:
        self.n_bins = n_bins

    def __call__(self, img: PIL.Image.Image) -> PIL.Image.Image:
        hist_equ = F.histogram_equalization(img, self.n_bins)
        return hist_equ

    def __repr__(self) -> str:
        return self.__class__.__name__ + "()"


class AdaptiveEqualization:
    """Increase image contrast using adaptive equalization.

    Contrast in local region of input image (gray or RGB) is increased using
    adaptive equalization

    Parameters
    ----------
    img : PIL.Image.Image
        Input image (gray or RGB)
    nbins : int, optional
        Number of histogram bins. Default is 256.
    clip_limit : float, optional
        Clipping limit where higher value increases contrast. Default is 0.01.

    Returns
    -------
    PIL.Image.Image
        Image with contrast enhanced by adaptive equalization.
    """

    def __init__(self, n_bins: int = 256, clip_limit: float = 0.01) -> None:
        self.n_bins = n_bins
        self.clip_limit = clip_limit

    def __call__(self, img: PIL.Image.Image) -> PIL.Image.Image:
        adaptive_equ = F.adaptive_equalization(img, self.n_bins, self.clip_limit)
        return adaptive_equ

    def __repr__(self) -> str:
        return self.__class__.__name__ + "()"


# ------- LocalEqualization input must be 2D (grayscale)


class LocalEqualization:
    """Filter gray image using local equalization.

    Local equalization method uses local histograms based on a disk structuring element.

    Parameters
    ---------
    img: PIL.Image.Image
        Input image. Notice that it must be 2D
    disk_size: int, optional
        Radius of the disk structuring element used for the local histograms.
        Default is 50

    Returns
    -------
    PIL.Image.Image
        2D image with contrast enhanced using local equalization.
    """

    def __init__(self, disk_size: int = 50) -> None:
        self.disk_size = disk_size

    def __call__(self, img: PIL.Image.Image) -> PIL.Image.Image:
        local_equ = F.local_equalization(img, self.disk_size)
        return local_equ

    def __repr__(self) -> str:
        return self.__class__.__name__ + "()"


class KmeansSegmentation:
    """Segment an RGB image with K-means segmentation

    By using K-means segmentation (color/space proximity) each segment is colored based
    on the average color for that segment.

    Parameters
    ---------
    img : PIL.Image.Image
        Input image
    n_segments : int, optional
        The number of segments. Default is 800.
    compactness : float, optional
        Color proximity versus space proximity factor. Default is 10.0.

    Returns
    -------
    PIL.Image.Image
        Image where each segment has been colored based on the average
        color for that segment.
    """

    def __init__(self, n_segments: int = 800, compactness: float = 10.0) -> None:
        self.n_segments = n_segments
        self.compactness = compactness

    def __call__(self, img: PIL.Image.Image) -> PIL.Image.Image:
        kmeans_segmentation = F.kmeans_segmentation(
            img, self.n_segments, self.compactness
        )
        return kmeans_segmentation

    def __repr__(self) -> str:
        return self.__class__.__name__ + "()"


class RagThreshold:
    """Combine similar K-means segmented regions based on threshold value.

    Segment an image with K-means, build region adjacency graph based on
    the segments, combine similar regions based on threshold value,
    and then output these resulting region segments.

    Parameters
    ----------
    img : PIL.Image.Image
        Input image
    n_segments : int, optional
        The number of segments. Default is 800.
    compactness : float, optional
        Color proximity versus space proximity factor. Default is 10.0
    threshold : int, optional
        Threshold value for combining regions. Default is 9.

    Returns
    -------
    PIL.Image.Image
        Each segment has been colored based on the average
        color for that segment (and similar segments have been combined).
    """

    def __init__(
        self, n_segments: int = 800, compactness: float = 10.0, threshold: int = 9
    ) -> PIL.Image.Image:
        self.n_segments = n_segments
        self.compactness = compactness
        self.threshold = threshold

    def __call__(self, img: PIL.Image.Image) -> PIL.Image.Image:
        return F.rag_threshold(img, self.n_segments, self.compactness, self.threshold)

    def __repr__(self) -> str:
        return self.__class__.__name__ + "()"


class HysteresisThreshold:
    """Apply two-level (hysteresis) threshold to an image.

    Parameters
    ----------
    img : PIL.Image.Image
        Input image
    low : int, optional
        low threshold. Default is 50.
    high : int, optional
        high threshold. Default is 100

    Returns
    -------
    PIL.Image.Image
        Image with the hysteresis threshold applied
    """

    def __init__(self, low: int = 50, high: int = 100) -> None:
        self.low = low
        self.high = high

    def __call__(self, img: PIL.Image.Image) -> PIL.Image.Image:
        return F.hysteresis_threshold(img, self.low, self.high)

    def __repr__(self) -> str:
        return self.__class__.__name__ + "()"


# ----------- Branching functions (grayscale/invert input)-------------------

# invert --> grayscale ..> hysteresis
class HysteresisThresholdMask:
    """Mask an image using hysteresis threshold

    Compute the Hysteresis threshold on the complement of a grayscale image,
    and return boolean mask based on pixels above this threshold.

    Parameters
    ----------
    img : PIL.Image.Image
        Input image.
    low : int, optional
        low threshold. Default is 50.
    high : int, optional
        high threshold. Default is 100.

    Returns
    -------
    np.ndarray
        Boolean NumPy array where True represents a pixel above hysteresis threshold.
    """

    def __init__(self, low: int = 50, high: int = 100) -> None:
        self.low = low
        self.high = high

    def __call__(self, img: PIL.Image.Image) -> np.ndarray:
        hysteresis_threshold_mask = F.hysteresis_threshold_mask(
            img, self.low, self.high
        )
        return hysteresis_threshold_mask

    def __repr__(self) -> str:
        return self.__class__.__name__ + "()"


class OtsuThreshold:
    """Mask image based on pixel above Otsu threshold.

    Compute Otsu threshold on image as a NumPy array and return boolean mask
    based on pixels above this threshold.

    Note that Otsu threshold is expected to work correctly only for grayscale images

    Parameters
    ----------
    img : PIL.Image.Image
        Input image.

    Returns
    -------
    np.ndarray
        Boolean NumPy array where True represents a pixel above Otsu threshold.
    """

    def __call__(self, img: PIL.Image.Image) -> np.ndarray:
        return F.otsu_threshold(img)

    def __repr__(self) -> str:
        return self.__class__.__name__ + "()"


class LocalOtsuThreshold:
    """Mask image based on local Otsu threshold.

    Compute local Otsu threshold for each pixel and return boolean mask
    based on pixels being less than the local Otsu threshold.

    Note that the input image must be 2D.

    Parameters
    ----------
    img: PIL.Image.Image
        Input 2-dimensional image
    disk_size : float, optional
        Radius of the disk structuring element used to compute the Otsu threshold for
        each pixel. Default is 3.0

    Returns
    -------
    np.ndarray
        NumPy boolean array representing the mask based on local Otsu threshold
    """

    def __init__(self, disk_size: float = 3.0) -> None:
        self.disk_size = disk_size

    def __call__(self, img: PIL.Image.Image) -> np.ndarray:
        return F.local_otsu_threshold(img, self.disk_size)

    def __repr__(self) -> str:
        return self.__class__.__name__ + "()"


class FilterEntropy:
    """Filter image based on entropy (complexity).

    The area of the image included in the local neighborhood is defined by a square
    neighborhood x neighborhood

    Note that input must be 2D.

    Parameters
    ----------
    img : PIL.Image.Image
        input 2-dimensional image
    neighborhood : int, optional
        Neighborhood size (defines height and width of 2D array of 1's). Default is 9.
    threshold : float, optional
        Threshold value. Default is 5.0

    Returns
    -------
    np.ndarray
        NumPy boolean array where True represent a measure of complexity.
    """

    def __init__(self, neighborhood: int = 9, threshold: float = 5.0) -> None:
        self.neighborhood = neighborhood
        self.threshold = threshold

    def __call__(self, img: PIL.Image.Image) -> np.ndarray:
        return F.filter_entropy(img, self.neighborhood, self.threshold)

    def __repr__(self) -> str:
        return self.__class__.__name__ + "()"


class CannyEdges:
    """Filter image based on Canny edge algorithm.

    Note that input image must be 2D

    Parameters
    ----------
    img : PIL.Image.Image
        Input 2-dimensional image
    sigma : float, optional
        Width (std dev) of Gaussian. Default is 1.0.
    low_threshold : float, optional
        Low hysteresis threshold value. Default is 0.0.
    high_threshold : float, optional
        High hysteresis threshold value. Default is 25.0.

    Returns
    -------
    np.ndarray
        Boolean NumPy array representing Canny edge map.
    """

    def __init__(
        self,
        sigma: float = 1.0,
        low_threshold: float = 0.0,
        high_threshold: float = 25.0,
    ) -> None:
        self.sigma = sigma
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def __call__(self, img: PIL.Image.Image) -> np.ndarray:
        return F.canny_edges(img, self.sigma, self.low_threshold, self.high_threshold)

    def __repr__(self) -> str:
        return self.__class__.__name__ + "()"


class Grays:
    """Filter out gray pixels in RGB image.

    Gray pixels are those pixels where the red, green, and blue channel values
    are similar, i.e. under a specified tolerance.

    Parameters
    ----------
    img : PIL.Image.Image
        Input image
    tolerance : int, optional
        if difference between values is below this threshold, values are considered
        similar and thus filtered out. Default is 15.

    Returns
    -------
    PIL.Image.Image
        Mask image where the grays values are masked out
    """

    def __init__(self, tolerance: int = 15) -> None:
        self.tolerance = tolerance

    def __call__(self, img: PIL.Image.Image) -> PIL.Image.Image:
        return F.grays(img, self.tolerance)

    def __repr__(self) -> str:
        return self.__class__.__name__ + "()"


class GreenChannelFilter:
    """Mask pixels in an RGB image with G-channel greater than a specified threshold.

    Create a mask to filter out pixels with a green channel value greater than
    a particular threshold, since hematoxylin and eosin are purplish and pinkish,
    which do not have much green to them.

    Parameters
    ----------
    img : PIL.Image.Image
        Input RGB image
    green_thresh : int, optional
        Green channel threshold value (0 to 255). Default is 200.
        If value is greater than green_thresh, mask out pixel.
    avoid_overmask : bool, optional
        If True, avoid masking above the overmask_thresh percentage. Default is True.
    overmask_thresh : float, optional
        If avoid_overmask is True, avoid masking above this percentage value. Default
        is 90.

    Returns
    -------
    np.ndarray
        Boolean mask where pixels above a particular green channel
        threshold have been masked out.
    """

    def __init__(
        self,
        green_thresh: int = 200,
        avoid_overmask: bool = True,
        overmask_thresh: float = 90.0,
    ) -> None:
        self.green_thresh = green_thresh
        self.avoid_overmask = avoid_overmask
        self.overmask_thresh = overmask_thresh

    def __call__(self, img: PIL.Image.Image) -> np.ndarray:
        return F.green_channel_filter(
            img, self.green_thresh, self.avoid_overmask, self.overmask_thresh
        )

    def __repr__(self) -> str:
        return self.__class__.__name__ + "()"


class RedFilter:
    """Mask reddish colors in an RGB image.

    Create a mask to filter out reddish colors, where the mask is based on a pixel
    being above a red channel threshold value, below a green channel threshold value,
    and below a blue channel threshold value.

    Parameters
    ----------
    img : PIl.Image.Image
        Input RGB image
    red_lower_thresh : int
        Red channel lower threshold value.
    green_upper_thresh : int
        Green channel upper threshold value.
    blue_upper_thresh : int
        Blue channel upper threshold value.

    Returns
    -------
    np.ndarray
        Boolean NumPy array representing the mask.
    """

    def __init__(self, red_thresh: int, green_thresh: int, blue_thresh: int) -> None:
        self.red_thresh = red_thresh
        self.green_thresh = green_thresh
        self.blue_thresh = blue_thresh

    def __call__(self, img: PIL.Image.Image) -> np.ndarray:
        return F.red_filter(img, self.red_thresh, self.green_thresh, self.blue_thresh)

    def __repr__(self) -> str:
        return self.__class__.__name__ + "()"


class RedPenFilter:
    """Filter out red pen marks on diagnostic slides.

    The resulting mask is a composition of red filters with different thresholds
    for the RGB channels.

    Parameters
    ----------
    img : PIL.Image.Image
        Input RGB image.

    Returns
    -------
        Boolean NumPy array representing the mask with the pen marks filtered out.
    """

    def __call__(self, img: PIL.Image.Image):
        return F.red_pen_filter(img)

    def __repr__(self):
        return self.__class__.__name__ + "()"


class GreenFilter:
    """Filter out greenish colors in an RGB image.
    The mask is based on a pixel being above a red channel threshold value, below a
    green channel threshold value, and below a blue channel threshold value.

    Note that for the green ink, the green and blue channels tend to track together, so
    for blue channel we use a lower threshold rather than an upper threshold value.

    Parameters
    ----------
    img : PIL.image.Image
        RGB input image.
    red_thresh : int
        Red channel upper threshold value.
    green_thresh : int
        Green channel lower threshold value.
    blue_thresh : int
        Blue channel lower threshold value.

    Returns
    -------
    np.ndarray
        Boolean  NumPy array representing the mask.
    """

    def __init__(self, red_thresh, green_thresh, blue_thresh):
        self.red_thresh = red_thresh
        self.green_thresh = green_thresh
        self.blue_thresh = blue_thresh

    def __call__(self, img):
        return F.green_filter(img, self.red_thresh, self.green_thresh, self.blue_thresh)

    def __repr__(self):
        return self.__class__.__name__ + "()"


class GreenPenFilter:
    """Filter out green pen marks from a diagnostic slide.

    The resulting mask is a composition of green filters with different thresholds
    for the RGB channels.

    Parameters
    ---------
    img : PIL.Image.Image
        Input RGB image

    Returns
    -------
    PIL.Image.Image
        Image the green pen marks filtered out.
    """

    def __call__(self, img: PIL.Image.Image) -> PIL.Image.Image:
        return F.green_pen_filter(img)

    def __repr__(self) -> str:
        return self.__class__.__name__ + "()"


class BlueFilter:
    """Filter out blueish colors in an RGB image.

    Create a mask to filter out blueish colors, where the mask is based on a pixel
    being above a red channel threshold value, above a green channel threshold value,
    and below a blue channel threshold value.

    Parameters
    ----------
    img : PIl.Image.Image
        Input RGB image
    red_thresh : int
        Red channel lower threshold value.
    green_thresh : int
        Green channel lower threshold value.
    blue_thresh : int
        Blue channel upper threshold value.

    Returns
    -------
    np.ndarray
        Boolean NumPy array representing the mask.
    """

    def __init__(self, red_thresh: int, green_thresh: int, blue_thresh: int):
        self.red_thresh = red_thresh
        self.green_thresh = green_thresh
        self.blue_thresh = blue_thresh

    def __call__(self, img: PIL.Image.Image) -> np.ndarray:
        return F.blue_filter(img, self.red_thresh, self.green_thresh, self.blue_thresh)

    def __repr__(self) -> str:
        return self.__class__.__name__ + "()"


class BluePenFilter:
    """Filter out blue pen marks from a diagnostic slide.

    The resulting mask is a composition of green filters with different thresholds
    for the RGB channels.

    Parameters
    ---------
    img : PIL.Image.Image
        Input RGB image

    Returns
    -------
    np.ndarray
        NumPy array representing the mask with the blue pen marks filtered out.
    """

    def __call__(self, img: PIL.Image.Image) -> PIL.Image.Image:
        return F.blue_pen_filter(img)

    def __repr__(self) -> str:
        return self.__class__.__name__ + "()"


class PenMarks:
    """Filter out pen marks from a diagnostic slide.

    Pen marks are removed by applying Otsu threshold on the H channel of the image
    converted to the HSV space.

    Parameters
    ---------
    img : PIL.Image.Image
        Input RGB image

    Returns
    -------
    np.ndarray
        Boolean NumPy array representing the mask with the pen marks filtered out.
    """

    def __call__(self, img: PIL.Image.Image) -> PIL.Image.Image:
        return F.pen_marks(img)

    def __repr__(self) -> str:
        return self.__class__.__name__ + "()"


class YenThreshold:
    """Mask image based on pixel above Yen threshold.

    Compute Yen threshold on image and return boolean mask based on pixels below this
    threshold.

    Parameters
    ----------
    img : PIL.Image.Image
        Input image.
    relate : operator, optional
        Operator to be used to compute the mask from the threshold. Default is
        operator.lt

    Returns
    -------
    np.ndarray
        Boolean NumPy array where True represents a pixel below Yen's threshold.
    """

    def __init__(self, relate: Callable[..., bool] = operator.lt):
        self.relate = relate

    def __call__(self, img: PIL.Image.Image) -> np.ndarray:
        return F.yen_threshold(img, self.relate)

    def __repr__(self) -> str:
        return self.__class__.__name__ + "()"
