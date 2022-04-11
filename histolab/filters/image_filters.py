# encoding: utf-8

# ------------------------------------------------------------------------
# Copyright 2022 All Histolab Contributors
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
"""All filters implemented in the image filters submodule take as input a Pillow Image
object. Additionally, some of the image filters in histolab leverage functions and
utilities by scikit-image. Image filters are divided into sub-categories, depending on
their behaviour and output type.
 """

import operator
from abc import abstractmethod
from typing import Any, Callable, List, Union

import numpy as np
import PIL

from .. import util
from . import image_filters_functional as F

try:
    from typing import Protocol, runtime_checkable
except ImportError:
    from typing_extensions import Protocol, runtime_checkable


@runtime_checkable
class Filter(Protocol):
    """Filter protocol"""

    @abstractmethod
    def __call__(
        self, img: Union[PIL.Image.Image, np.ndarray]
    ) -> Union[PIL.Image.Image, np.ndarray]:
        pass  # pragma: no cover

    def __repr__(self) -> str:
        return self.__class__.__name__ + "()"


@runtime_checkable
class ImageFilter(Filter, Protocol):
    """Image filter protocol"""

    @abstractmethod
    def __call__(self, img: PIL.Image.Image) -> Union[PIL.Image.Image, np.ndarray]:
        pass  # pragma: no cover


class Compose(ImageFilter):
    """Composes several filters together.

    Parameters
    ----------
    filters : list of Filters
        List of filters to compose
    """

    def __init__(self, filters: List[Filter]) -> None:
        self.filters = filters

    def __call__(self, img: PIL.Image.Image) -> Union[PIL.Image.Image, np.ndarray]:
        for filter_ in self.filters:
            img = filter_(img)
        return img


class Lambda(ImageFilter):
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
    """  # noqa

    def __init__(self, lambd: Callable[[PIL.Image.Image], PIL.Image.Image]) -> None:
        assert callable(lambd), repr(type(lambd).__name__) + " object is not callable"
        self.lambd = lambd

    def __call__(self, img: PIL.Image.Image) -> Union[PIL.Image.Image, np.ndarray]:
        return self.lambd(img)


class ToPILImage(Filter):
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
        return util.np_to_pil(np_img)


class ApplyMaskImage(Filter):
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
        return util.apply_mask_image(self.img, mask)


class Invert(ImageFilter):
    r"""Invert an image, i.e. take the complement of the correspondent array.

    For binary images, the inversion flips True and False values. For RGB images, each
    pixel value p is replaced with :math:`\hat{p}-p` where :math:`\hat{p}` is the
    maximum value of pixels of the data type (i.e. 255).
    Usually, the tissue in a WSI is surrounded by a white light background (values close
    to 255). Therefore, inverting its values could ease the removal of non-tissue
    regions (values close or equal to 0).

    .. figure:: https://user-images.githubusercontent.com/31658006/116548383-6aaad800-a8f4-11eb-8ebd-46c873046447.png

    Parameters
    ----------
    img : PIL.Image.Image
        Input image

    Returns
    -------
    PIL.Image.Image
        Inverted image


    Example:
        >>> from PIL import Image
        >>> from histolab.filters.image_filters import Invert, RgbToGrayscale
        >>> image_rgb = Image.open("tests/fixtures/pil-images-rgb/tcga-lung-rgb.png")
        >>> rgb_to_grayscale = RgbToGrayscale()
        >>> invert = Invert()
        >>> image_gray = rgb_to_grayscale(image_rgb)
        >>> image_inv_rgb = invert(image_rgb)
        >>> image_inv_gray = invert(image_gray)
    """  # noqa

    def __call__(self, img: PIL.Image.Image) -> PIL.Image.Image:
        return F.invert(img)


class RgbToGrayscale(ImageFilter):
    """Convert an RGB image to a grayscale image.

    Parameters
    ----------
    img : PIL.Image.Image
        Input image

    Returns
    -------
    PIL.Image.Image
        Grayscale image


    Example:
        >>> from PIL import Image
        >>> from histolab.filters.image_filters import RgbToGrayscale
        >>> image_rgb = Image.open("tests/fixtures/pil-images-rgb/tcga-lung-rgb.png")
        >>> rgb_to_grayscale = RgbToGrayscale()
        >>> image_gray = rgb_to_grayscale(image_rgb)
    """  # noqa

    def __call__(self, img: PIL.Image.Image) -> PIL.Image.Image:
        return PIL.ImageOps.grayscale(img)


class RgbToHed(ImageFilter):
    """Convert RGB channels to HED channels.

    image color space (RGB) is converted to Hematoxylin-Eosin-Diaminobenzidine space.

    Parameters
    ----------
    img : PIL.Image.Image
        Input image

    Returns
    -------
    np.ndarray
        Array representation of the image in HED space


    Example:
        >>> from PIL import Image
        >>> from histolab.filters.image_filters import RgbToHed
        >>> image_rgb = Image.open("tests/fixtures/pil-images-rgb/tcga-lung-rgb.png")
        >>> rgb_to_hed = RgbToHed()
        >>> image_hed = rgb_to_hed(image_rgb)
    """  # noqa

    def __call__(self, img: PIL.Image.Image) -> np.ndarray:
        hed = F.rgb_to_hed(img)
        return hed


class RgbToLab(ImageFilter):
    """Convert from the sRGB color space to the CIE Lab colorspace.

    sRGB color space reference: IEC 61966-2-1:1999

    Parameters
    ----------
    img : PIL.Image.Image
        Input image
    illuminant : {"A", "B", "C", "D50", "D55", "D65", "D75", "E"}, optional
        The name of the illuminant (the function is NOT case sensitive).
    observer : {"2", "10", "R"}, optional
        The aperture angle of the observer.

    Returns
    -------
    np.ndarray
        Array representation of the image in LAB space

    Raises
    ------
    Exception
        If the ``img`` mode is not RGB


    Example:
        >>> from PIL import Image
        >>> from histolab.filters.image_filters import RgbToLab
        >>> image_rgb = Image.open("tests/fixtures/pil-images-rgb/tcga-lung-rgb.png")
        >>> rgb_to_lab = RgbToLab()
        >>> image_lab = rgb_to_lab(image_rgb)
    """  # noqa

    def __init__(self, illuminant: str = "D65", observer: int = "2") -> None:
        self.illuminant = illuminant
        self.observer = observer

    def __call__(self, img: PIL.Image.Image) -> np.ndarray:
        lab = F.rgb_to_lab(img, self.illuminant, self.observer)
        return lab


class RgbToOd(ImageFilter):
    """Convert from RGB to optical density (OD_RGB) space.

    Parameters
    ----------
    img : PIL.Image.Image
        Input image

    Returns
    -------
    np.ndarray
        Array representation of the image in OD space


    Example:
        >>> from PIL import Image
        >>> from histolab.filters.image_filters import RgbToOd
        >>> image_rgb = Image.open("tests/fixtures/pil-images-rgb/tcga-lung-rgb.png")
        >>> rgb_to_od = RgbToOd()
        >>> image_od = rgb_to_od(image_rgb)
    """  # noqa

    def __call__(self, img: PIL.Image.Image) -> np.ndarray:
        od = F.rgb_to_od(img)
        return od


class HedToRgb(ImageFilter):
    """Convert HED channels to RGB channels.

    Parameters
    ----------
    img_arr : np.ndarray
        Array representation of the image in HED color space

    Returns
    -------
    PIL.Image.Image
        Image in RGB space


    Example:
        >>> import numpy as np
        >>> from histolab.filters.image_filters import HedToRgb
        >>> hed_arr = np.load("tests/fixtures/arrays/diagnostic-slide-thumb-hed.npy")
        >>> hed_to_rgb = HedToRgb()
        >>> rgb = hed_to_rgb(hed_arr)
    """  # noqa

    def __call__(self, img_arr: np.array) -> PIL.Image.Image:
        rgb = F.hed_to_rgb(img_arr)
        return rgb


class HematoxylinChannel(ImageFilter):
    """Obtain Hematoxylin channel from RGB image.

    Input image is first converted into HED space and the hematoxylin channel is
    extracted via color deconvolution.

    Parameters
    ----------
    img : PIL.Image.Image
        Input RGB image

    Returns
    -------
    PIL.Image.Image
        RGB image with Hematoxylin staining separated.


    Example:
        >>> from PIL import Image
        >>> from histolab.filters.image_filters import HematoxylinChannel
        >>> image_rgb = Image.open("tests/fixtures/pil-images-rgb/tcga-lung-rgb.png")
        >>> hematoxylin_channel = HematoxylinChannel()
        >>> image_h = hematoxylin_channel(image_rgb)
    """  # noqa

    def __call__(self, img: PIL.Image.Image) -> PIL.Image.Image:
        hematoxylin = F.hematoxylin_channel(img)
        return hematoxylin


class EosinChannel(ImageFilter):
    """Obtain Eosin channel from RGB image.

    Input image is first converted into HED space and the Eosin channel is
    extracted via color deconvolution.

    Parameters
    ----------
    img : PIL.Image.Image
        Input RGB image

    Returns
    -------
    PIL.Image.Image
        RGB image with Eosin staining separated.


    Example:
        >>> from PIL import Image
        >>> from histolab.filters.image_filters import EosinChannel
        >>> image_rgb = Image.open("tests/fixtures/pil-images-rgb/tcga-lung-rgb.png")
        >>> eosin_channel = EosinChannel()
        >>> image_e = eosin_channel(image_rgb)
    """  # noqa

    def __call__(self, img: PIL.Image.Image) -> PIL.Image.Image:
        eosin = F.eosin_channel(img)
        return eosin


class DABChannel(ImageFilter):
    """Obtain DAB channel from RGB image.

    Input image is first converted into HED space and the DAB channel is
    extracted via color deconvolution.

    Parameters
    ----------
    img : PIL.Image.Image
        Input RGB image

    Returns
    -------
    PIL.Image.Image
        RGB image with Eosin staining separated.


    Example:
        >>> from PIL import Image
        >>> from histolab.filters.image_filters import DABChannel
        >>> image_rgb = Image.open("tests/fixtures/pil-images-rgb/tcga-lung-rgb.png")
        >>> dab_channel = DABChannel()
        >>> image_d = dab_channel(image_rgb)
    """  # noqa

    def __call__(self, img: PIL.Image.Image) -> PIL.Image.Image:
        dab = F.dab_channel(img)
        return dab


class RgbToHsv(ImageFilter):
    """Convert RGB channels to HSV channels.

    image color space (RGB) is converted to Hue - Saturation - Value (HSV) space.

    Parameters
    ----------
    img : PIL.Image.Image
        Input image

    Returns
    -------
    np.ndarray
        Array representation of the image in HSV space


    Example:
        >>> from PIL import Image
        >>> from histolab.filters.image_filters import RgbToHsv
        >>> image_rgb = Image.open("tests/fixtures/pil-images-rgb/tcga-lung-rgb.png")
        >>> rgb_to_hsv = RgbToHsv()
        >>> image_hsv = rgb_to_hsv(image_rgb)
    """  # noqa

    def __call__(self, img: PIL.Image.Image) -> np.ndarray:
        hsv = F.rgb_to_hsv(img)
        return hsv


class StretchContrast(ImageFilter):
    r"""Increase image contrast.

    A simple way to enhance the contrast in an image is to linearly rescale the
    intensity values within a desired range :math:`[v_{o,l}, v_{o,h}]`. In particular,
    if the lowest and highest pixel values of the input image are, respectively,
    :math:`v_{i,l}` and :math:`v_{i,h}`, an input pixel :math:`p_i` is remapped to
    the output pixel value:

    .. math::

        p_o = (p_i - v_{i,l})\left(\frac{v_{o,h}- v_{o,l}}{v_{i,h}- v_{i,l}}\right)+v_{o,l}

    The Stretch Contrast filter stretches the intensity values in an image, with
    :math:`v_{o,l}=40` and :math:`v_{o,l}=60` as default values.
    This filter is useful to highlight details in the input image.

    .. figure:: https://user-images.githubusercontent.com/31658006/116539805-9f656200-a8e9-11eb-913b-864c0a9d8baf.png

    Parameters
    ----------
    img : PIL.Image.Image
        Input image
    low : int, optional
        Range low value (0 to 255). Default is 40.
    high : int, optional
        Range high value (0 to 255). Default is 60

    Returns
    -------
    PIL.Image.Image
        Image with contrast enhanced.


    Example:
        >>> from PIL import Image
        >>> from histolab.filters.image_filters import RgbToGrayscale, StretchContrast
        >>> image_rgb = Image.open("tests/fixtures/pil-images-rgb/tcga-lung-rgb.png")
        >>> rgb_to_grayscale = RgbToGrayscale()
        >>> stretch_contrast = StretchContrast()
        >>> image_gray = rgb_to_grayscale(image_rgb)
        >>> image_stretched = stretch_contrast(image_gray)
    """  # noqa

    def __init__(self, low: int = 40, high: int = 60) -> None:
        self.low = low
        self.high = high

    def __call__(self, img: PIL.Image.Image) -> PIL.Image.Image:
        stretch_contrast = F.stretch_contrast(img, self.low, self.high)
        return stretch_contrast


class HistogramEqualization(ImageFilter):
    r"""Increase image contrast using histogram equalization.

    The input image (gray or RGB) is filterd using histogram equalization to increase
    contrast. In particular, this filter expands the range of intensity values in low
    contrast images. It first computes the normalized histogram H of an image: H(k)
    counts pixels with intensity values k, divided by the total number of pixels in
    the image. Then, it computes the cumulative sum of the histogram values as

    .. math::

        C[i] = \sum_{k=0}^{i} H[k]

    for i =0...255. Finally, for each pixel P, the algorithm computes  a new value

    .. math::

        p\prime = 255 \cdot C[p].

    The resulting image will have a uniform intensity distribution.
    The algorithm described is also called non-adaptive uniform histogram
    equalization, as it works uniformly on the whole image and the transformation of one
    pixel is independent from the transformation used on the neighboring pixels [4]_.

    .. figure:: https://user-images.githubusercontent.com/20052362/139671873-fa61fc20-a6b8-4302-8596-b6542b5b3aaf.png

    Notice that the histogram equalization method can be used for RGB images
    by applying the same algorithm on the R, G, and B channels separately [5]_;
    nonetheless, the high correlation of the three channels may distort the image and
    the color balance can change drastically.

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


    Example:
        >>> from PIL import Image
        >>> from histolab.filters.image_filters import HistogramEqualization, RgbToGrayscale
        >>> image_rgb = Image.open("tests/fixtures/pil-images-rgb/tcga-lung-rgb.png")
        >>> rgb_to_grayscale = RgbToGrayscale()
        >>> histogram_equalization = HistogramEqualization()
        >>> image_gray = rgb_to_grayscale(image_rgb)
        >>> image_he = histogram_equalization(image_gray)

    References
    --------
    .. [4] T Strothotte and S Schlechtweg. “Non-photorealistic computer graphics:
        modeling, rendering, and animation”. Morgan Kaufmann (2002)
    .. [5] Z Rong and et al. “Study of color heritage image enhancement algorithms based
        on histogram equalization”. Optik 126.24 (2015)
    """  # noqa

    def __init__(self, n_bins: int = 256) -> None:
        self.n_bins = n_bins

    def __call__(self, img: PIL.Image.Image) -> PIL.Image.Image:
        hist_equ = F.histogram_equalization(img, self.n_bins)
        return hist_equ


class AdaptiveEqualization(ImageFilter):
    """Increase image contrast using adaptive equalization.

    Rather than considering the global contrast in the image, the adaptive histogram
    equalization method applies the histogram equalization to smaller regions, or tiles,
    of the image; the tiles are then combined together using bilinear interpolation.
    This local approach is preferred when the image presents significantly darker or
    lighter regions that may be poorly enhanced by the global histogram equalization
    transformation.

    .. figure:: https://user-images.githubusercontent.com/20052362/139672710-e75d1f0e-e5f3-4365-8f83-b0d2316bfb10.png

    The Adaptive Equalization filter is based on the scikit-image implementation of
    the contrast limited adaptive histogram equalization (CLAHE) [1]_.

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


    Example:
        >>> from PIL import Image
        >>> from histolab.filters.image_filters import AdaptiveEqualization, RgbToGrayscale
        >>> image_rgb = Image.open("tests/fixtures/pil-images-rgb/tcga-lung-rgb.png")
        >>> rgb_to_grayscale = RgbToGrayscale()
        >>> adaptive_equalization = AdaptiveEqualization()
        >>> image_gray = rgb_to_grayscale(image_rgb)
        >>> image_clahe = adaptive_equalization(image_gray)

    References
    --------
    .. [1] S.M. Pizer andet al. "Adaptive histogram equalization and its variations”,
        Comput Vis Graph Image Process 39.3 (1987).
    """  # noqa

    def __init__(self, n_bins: int = 256, clip_limit: float = 0.01) -> None:
        self.n_bins = n_bins
        self.clip_limit = clip_limit

    def __call__(self, img: PIL.Image.Image) -> PIL.Image.Image:
        adaptive_equ = F.adaptive_equalization(img, self.n_bins, self.clip_limit)
        return adaptive_equ


class LabToRgb(ImageFilter):
    """Lab to RGB color space conversion.

    Parameters
    ----------
    img : np.array
        Input image in Lab space.
    illuminant : {"A", "B", "C", "D50", "D55", "D65", "D75", "E"}, optional
        The name of the illuminant (the function is NOT case sensitive). Default is
        "D65".
    observer : {"2", "10", "R"}, optional
        The aperture angle of the observer. Default is "2".

    Returns
    -------
    PIL.Image.Image
        Image in RGB space.

    Example:
        >>> import numpy as np
        >>> from histolab.filters.image_filters import LabToRgb
        >>> arr_lab = np.load("tests/fixtures/arrays/diagnostic-slide-thumb-lab.npy")
        >>> lab_to_rgb = LabToRgb()
        >>> image_rgb = lab_to_rgb(arr_lab)
    """

    def __init__(self, illuminant: str = "D65", observer: int = "2") -> None:
        self.illuminant = illuminant
        self.observer = observer

    def __call__(self, np_arr: np.ndarray) -> PIL.Image.Image:
        lab = F.lab_to_rgb(np_arr, self.illuminant, self.observer)
        return lab


class LocalEqualization(ImageFilter):
    """Filter gray image using local equalization.

    Local equalization method uses local histograms based on a disk structuring element.

    Parameters
    ---------
    img : PIL.Image.Image
        Grayscale input image
    disk_size : int, optional
        Radius of the disk structuring element used for the local histograms.
        Default is 50

    Returns
    -------
    PIL.Image.Image
        Grayscale image with contrast enhanced using local equalization.


    Example:
        >>> from PIL import Image
        >>> from histolab.filters.image_filters import LocalEqualization
        >>> image_rgb = Image.open("tests/fixtures/pil-images-gs/diagnostic-slide-thumb-gs.png")
        >>> local_equ = LocalEqualization()
        >>> local_equ_image = local_equ(image_rgb)
    """  # noqa

    def __init__(self, disk_size: int = 50) -> None:
        self.disk_size = disk_size

    def __call__(self, img: PIL.Image.Image) -> PIL.Image.Image:
        local_equ = F.local_equalization(img, self.disk_size)
        return local_equ


class KmeansSegmentation(ImageFilter):
    """Segment an RGB image with K-means segmentation

    By using K-means segmentation (color/space proximity) each segment is colored based
    on the average color for that segment.

    .. figure:: https://user-images.githubusercontent.com/20052362/144853490-741deb95-cd84-47b0-8227-38e6091753d8.png
        :figwidth: 60 %
        :align: center

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

    Raises
    ------
    ValueError
        If ``img`` mode is RGBA.


    Example:
        >>> from PIL import Image
        >>> from histolab.filters.image_filters import KmeansSegmentation
        >>> image_rgb = Image.open("tests/fixtures/pil-images-rgb/tcga-lung-rgb.png")
        >>> kmeans_segmentation = KmeansSegmentation()
        >>> kmeans_segmented_image = kmeans_segmentation(image_rgb)
    """  # noqa

    def __init__(self, n_segments: int = 800, compactness: float = 10.0) -> None:
        self.n_segments = n_segments
        self.compactness = compactness

    def __call__(self, img: PIL.Image.Image) -> PIL.Image.Image:
        kmeans_segmentation = F.kmeans_segmentation(
            img, self.n_segments, self.compactness
        )
        return kmeans_segmentation


class RagThreshold(ImageFilter):
    """Combine similar K-means segmented regions based on threshold value.

    Segment an image with K-means, build region adjacency graph based on
    the segments, combine similar regions based on threshold value,
    and then output these resulting region segments.

    .. figure:: https://user-images.githubusercontent.com/20052362/144852689-6098a415-1714-4abf-a361-485761647349.png
        :figwidth: 60 %
        :align: center

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
    return_labels : bool, optional
        If True, returns a labeled array where the value denotes segment
        membership. Otherwise, returns a PIL image where each segment is colored
        by the average color in it. Default is False.

    Returns
    -------
    PIL.Image.Image, if not ``return_labels``
        Each segment has been colored based on the average
        color for that segment (and similar segments have been combined).
    np.ndarray, if ``return_labels``
        Value denotes segment membership.

    Raises
    ------
    ValueError
        If ``img`` mode is RGBA.


    Example:
        >>> from PIL import Image
        >>> from histolab.filters.image_filters import RagThreshold
        >>> image_rgb = Image.open("tests/fixtures/pil-images-rgb/tcga-lung-rgb.png")
        >>> rag_threshold = RagThreshold()
        >>> rag_thresholded_array = rag_threshold(image_rgb)
    """  # noqa

    def __init__(
        self,
        n_segments: int = 800,
        compactness: float = 10.0,
        threshold: int = 9,
        return_labels: bool = False,
    ) -> None:
        self.n_segments = n_segments
        self.compactness = compactness
        self.threshold = threshold
        self.return_labels = return_labels

    def __call__(
        self,
        img: PIL.Image.Image,
        mask: np.ndarray = None,
    ) -> Union[PIL.Image.Image, np.ndarray]:
        return F.rag_threshold(
            img,
            n_segments=self.n_segments,
            compactness=self.compactness,
            threshold=self.threshold,
            mask=mask,
            return_labels=self.return_labels,
        )


class HysteresisThreshold(ImageFilter):
    r"""Apply two-level (hysteresis) threshold to an image.
    The hysteresis thresholding is a two-threshold method used to detect objects on an
    image, based on the assumption that points connected to an object are most likely
    objects themselves. In particular, pixels above a specified high threshold
    :math:`t_h`are considered as strong objects, pixels below a specified low threshold
    :math:`t_l` are labelled as non-objects, and pixels :math:`o\in[t_l, t_h]` are
    defined as weak objects; all the non-objects are removed, while the weak objects are
    kept only if connected to a strong one. The hysteresis thresholding can be applied
    to detect edges in an image.

    .. figure:: https://user-images.githubusercontent.com/31658006/116542328-d5f0ac00-a8ec-11eb-9f05-696ca0598fd4.png

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


    Example:
        >>> from PIL import Image
        >>> from histolab.filters.image_filters import HysteresisThreshold, RgbToGrayscale
        >>> image_rgb = Image.open("tests/fixtures/pil-images-rgb/tcga-lung-rgb.png")
        >>> rgb_to_grayscale = RgbToGrayscale()
        >>> hyst_threshold = HysteresisThreshold(low=200, high=250)
        >>> image_gray = rgb_to_grayscale(image_rgb)
        >>> image_thresholded = hyst_threshold(image_gray)
    """  # noqa

    def __init__(self, low: int = 50, high: int = 100) -> None:
        self.low = low
        self.high = high

    def __call__(self, img: PIL.Image.Image) -> PIL.Image.Image:
        return F.hysteresis_threshold(img, self.low, self.high)


class LocalOtsuThreshold(ImageFilter):
    """Mask image based on local Otsu threshold.

    Compute Otsu threshold for each pixel and return the image thresholded locally.

    Note that the input image must be 2D.

    Parameters
    ----------
    img : PIL.Image.Image
        Input 2-dimensional image
    disk_size : float, optional
        Radius of the disk structuring element used to compute the Otsu threshold for
        each pixel. Default is 3.0

    Returns
    -------
    PIL.Image.Image
        Image thresholded with the Otsu algorithm computed locally


    Example:
        >>> from PIL import Image
        >>> from histolab.filters.image_filters import LocalOtsuThreshold, RgbToGrayscale
        >>> image_rgb = Image.open("tests/fixtures/pil-images-rgb/tcga-lung-rgb.png")
        >>> rgb_to_grayscale = RgbToGrayscale()
        >>> local_otsu = LocalOtsuThreshold()
        >>> image_gray = rgb_to_grayscale(image_rgb)
        >>> image_thresholded_locally = local_otsu(image_gray)
    """  # noqa

    def __init__(self, disk_size: float = 3.0) -> None:
        self.disk_size = disk_size

    def __call__(self, img: PIL.Image.Image) -> np.ndarray:
        return F.local_otsu_threshold(img, self.disk_size)


# ----------- Branching functions (grayscale/invert input)-------------------

# invert --> grayscale ..> hysteresis
class HysteresisThresholdMask(ImageFilter):
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


    Example:
        >>> from PIL import Image
        >>> from histolab.filters.image_filters import HysteresisThresholdMask, RgbToGrayscale
        >>> image_rgb = Image.open("tests/fixtures/pil-images-rgb/tcga-lung-rgb.png")
        >>> rgb_to_grayscale = RgbToGrayscale()
        >>> hyst_threshold_mask = HysteresisThresholdMask()
        >>> image_gray = rgb_to_grayscale(image_rgb)
        >>> image_thresholded_array = hyst_threshold_mask(image_gray)
    """  # noqa

    def __init__(self, low: int = 50, high: int = 100) -> None:
        self.low = low
        self.high = high

    def __call__(self, img: PIL.Image.Image) -> np.ndarray:
        hysteresis_threshold_mask = F.hysteresis_threshold_mask(
            img, self.low, self.high
        )
        return hysteresis_threshold_mask


class OtsuThreshold(ImageFilter):
    """Mask image based on pixel above Otsu threshold.

    Compute Otsu threshold on image as a NumPy array and return boolean mask
    based on pixels above this threshold. The Otsu algorithm is a standard method to
    automatically compute the optimal threshold value to separate image background from
    the foreground [7]_. In this filter, the pixels below the
    Otsu threshold are considered as foreground.

    Note that Otsu threshold is expected to work correctly only for grayscale images.

    .. figure:: https://user-images.githubusercontent.com/31658006/116542034-76929c00-a8ec-11eb-98ca-9e0d283cdcbb.png

    Parameters
    ----------
    img : PIL.Image.Image
        Input image.

    Returns
    -------
    np.ndarray
        Boolean NumPy array where True represents a pixel above Otsu threshold.


    Example:
        >>> from PIL import Image
        >>> from histolab.filters.image_filters import OtsuThreshold, RgbToGrayscale
        >>> image_rgb = Image.open("tests/fixtures/pil-images-rgb/tcga-lung-rgb.png")
        >>> rgb_to_grayscale = RgbToGrayscale()
        >>> otsu_threshold = OtsuThreshold()
        >>> image_gray = rgb_to_grayscale(image_rgb)
        >>> image_thresholded_array = otsu_threshold(image_gray)


    Reference
    ---------
    .. [7] N Otsu. “A threshold selection method from gray-level histograms”. IEEE
        Trans SystMan Cybern Syst 9.1 (1979)
    """  # noqa

    def __call__(self, img: PIL.Image.Image) -> np.ndarray:
        return F.otsu_threshold(img)


class FilterEntropy(ImageFilter):
    """Filter image based on entropy (complexity).

    Entropy measures complexity in an image: the greater the entropy the more
    heterogeneous structures are found is the image, while slide backgrounds are
    usually less complex. This method filters out pixels of grayscale images based
    on the local entropy. In details: (i) the entropy is computed on a neighborhood
    defined by a squared all-ones matrix of size n (by default n=9); (ii) pixels with
    entropy greater than a specified threshold t (by default t=5) are replaced with 1,
    0 otherwise. This entropy filter can be used to detect highly hematoxylin-stained
    regions, which represent dense accumulation of nuclei (complex structures).

    Note that input must be 2D.

    .. figure:: https://user-images.githubusercontent.com/31658006/116543013-a1312480-a8ed-11eb-8f75-b25164286994.png

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


    Example:
        >>> from PIL import Image
        >>> from histolab.filters.image_filters import FilterEntropy, RgbToGrayscale
        >>> image_rgb = Image.open("tests/fixtures/pil-images-rgb/tcga-lung-rgb.png")
        >>> rgb_to_grayscale = RgbToGrayscale()
        >>> entropy_filter = FilterEntropy()
        >>> image_gray = rgb_to_grayscale(image_rgb)
        >>> image_thresholded_array = entropy_filter(image_gray)
    """  # noqa

    def __init__(self, neighborhood: int = 9, threshold: float = 5.0) -> None:
        self.neighborhood = neighborhood
        self.threshold = threshold

    def __call__(self, img: PIL.Image.Image) -> np.ndarray:
        return F.filter_entropy(img, self.neighborhood, self.threshold)


class CannyEdges(ImageFilter):
    r"""Filter image based on Canny edge algorithm.

    The Canny edge detector has been used to generate a version of the image that
    highlights edges within tissue fragments by detecting changes in pixel intensity
    [2]_ [3]_.
    The algorithm includes five steps: (i) smoothing the image (i.e. remove the noise);
    (ii) computing the gradient's magnitude :math:`M_\nabla` and direction
    :math:`\theta_\nabla`; (iii) keeping the direction :math:`\theta_\nabla` with
    greatest intensity :math:`M_\nabla` for each pixel; (iv) thinning the edges by
    suppressing non-maximal pixels; (v) applying the hysteresis thresholding algorithm
    for the final edge detection.

    Note that input image must be 2D.

    .. figure:: https://user-images.githubusercontent.com/20052362/144851746-3332560c-bd61-4b5a-baa3-6a8a528c0db0.png

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


    Example:
        >>> from PIL import Image
        >>> from histolab.filters.image_filters import CannyEdges, RgbToGrayscale
        >>> image_rgb = Image.open("tests/fixtures/pil-images-rgb/tcga-lung-rgb.png")
        >>> rgb_to_grayscale = RgbToGrayscale()
        >>> canny_edges_detection = CannyEdges()
        >>> image_gray = rgb_to_grayscale(image_rgb)
        >>> image_thresholded_array = canny_edges_detection(image_gray)

    References
    ----------
    .. [2] A Kumar and M Prateek. “Localization of Nuclei in Breast Cancer Using Whole
        SlideImaging System Supported by Morphological Features and Shape Formulas”.
        CancerManag Res 12 (2020)
    .. [3] M Mũnoz-Aguirre and et al. “PyHIST: A Histological Image Segmentation Tool”.
        PLOS Comput Biol 16.10 (2020)
    """  # noqa

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


class Grays(ImageFilter):
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


    Example:
        >>> from PIL import Image
        >>> from histolab.filters.image_filters import Grays
        >>> image_rgb = Image.open("tests/fixtures/pil-images-rgb/tcga-lung-rgb.png")
        >>> grays_filter = Grays(tolerance=5)
        >>> filtered_mask = grays_filter(image_rgb)
    """  # noqa

    def __init__(self, tolerance: int = 15) -> None:
        self.tolerance = tolerance

    def __call__(self, img: PIL.Image.Image) -> PIL.Image.Image:
        return F.grays(img, self.tolerance)


class GreenChannelFilter(ImageFilter):
    """Mask pixels in an RGB image with G-channel greater than a specified threshold.

    Create a binary mask where pixels with the green channel value above a specified
    threshold (by default 200) are set to 0. This filtering method can be used to
    detect tissue in H&E-stained images, considering that the green dye is poorly used
    in the tissue-related stains, i.e. eosin (pink) and hematoxylin (purple). To avoid
    over-masking the image, the overmask_thresh parameter defines the maximum percentage
    of tissue that can be masked by the green channel filter (by default 90%).

    This method alone may be sufficient to segment tissue on H&E-stained images.

    .. figure:: https://user-images.githubusercontent.com/31658006/116541610-f53b0980-a8eb-11eb-939d-944ebc7c87ee.png

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


    Example:
        >>> from PIL import Image
        >>> from histolab.filters.image_filters import GreenChannelFilter
        >>> image_rgb = Image.open("tests/fixtures/pil-images-rgb/tcga-lung-rgb.png")
        >>> g_channel_filter = GreenChannelFilter(avoid_overmask=True, overmask_thresh=90)
        >>> image_thresholded_array = g_channel_filter(image_rgb)
    """  # noqa

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


class RedFilter(ImageFilter):
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


    Example:
        >>> from PIL import Image
        >>> from histolab.filters.image_filters import RedFilter
        >>> image_rgb = Image.open("tests/fixtures/pil-images-rgb/tcga-red-pen.png")
        >>> red_filter = RedFilter(10, 30, 25)
        >>> mask_filtered = red_filter(image_rgb)
    """

    def __init__(self, red_thresh: int, green_thresh: int, blue_thresh: int) -> None:
        self.red_thresh = red_thresh
        self.green_thresh = green_thresh
        self.blue_thresh = blue_thresh

    def __call__(self, img: PIL.Image.Image) -> np.ndarray:
        return F.red_filter(img, self.red_thresh, self.green_thresh, self.blue_thresh)


class RedPenFilter(ImageFilter):
    """Filter out red pen marks on diagnostic slides.

    The resulting mask is a composition of red filters with different thresholds
    for the RGB channels.

    Parameters
    ----------
    img : PIL.Image.Image
        Input RGB image.

    Returns
    -------
    PIL.Image.Image
        Image the green red marks filtered out.


    Example:
        >>> from PIL import Image
        >>> from histolab.filters.image_filters import RedPenFilter
        >>> image_rgb = Image.open("tests/fixtures/pil-images-rgb/tcga-red-pen.png")
        >>> red_pen_filter = RedPenFilter()
        >>> image_no_red = red_pen_filter(image_rgb)
    """

    def __call__(self, img: PIL.Image.Image):
        return F.red_pen_filter(img)


class GreenFilter(ImageFilter):
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
        Boolean NumPy array representing the mask.


    Example:
        >>> from PIL import Image
        >>> from histolab.filters.image_filters import GreenFilter
        >>> image_rgb = Image.open("tests/fixtures/pil-images-rgb/tcga-red-pen.png")
        >>> green_filter = GreenFilter(230, 10, 105)
        >>> mask_filtered = green_filter(image_rgb)
    """

    def __init__(self, red_thresh, green_thresh, blue_thresh):
        self.red_thresh = red_thresh
        self.green_thresh = green_thresh
        self.blue_thresh = blue_thresh

    def __call__(self, img):
        return F.green_filter(img, self.red_thresh, self.green_thresh, self.blue_thresh)


class GreenPenFilter(ImageFilter):
    """Filter out green pen marks from a diagnostic slide.

    The resulting mask is a composition of green filters with different thresholds
    for the RGB channels.

    .. figure:: https://user-images.githubusercontent.com/31658006/116548722-f290e200-a8f4-11eb-9780-0ce5844295dd.png

    Parameters
    ---------
    img : PIL.Image.Image
        Input RGB image

    Returns
    -------
    PIL.Image.Image
        Image the green pen marks filtered out.


    Example:
        >>> from PIL import Image
        >>> from histolab.filters.image_filters import GreenPenFilter
        >>> image_rgb = Image.open("tests/fixtures/pil-images-rgb/tcga-green-pen.png")
        >>> green_pen_filter = GreenPenFilter()
        >>> image_no_green = green_pen_filter(image_rgb)
    """  # noqa

    def __call__(self, img: PIL.Image.Image) -> PIL.Image.Image:
        return F.green_pen_filter(img)


class BlueFilter(ImageFilter):
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


    Example:
        >>> from PIL import Image
        >>> from histolab.filters.image_filters import BlueFilter
        >>> image_rgb = Image.open("tests/fixtures/pil-images-rgb/wsi-blue-pen.png")
        >>> blue_filter = BlueFilter(30, 20, 105)
        >>> mask_filtered = blue_filter(image_rgb)
    """  # noqa

    def __init__(self, red_thresh: int, green_thresh: int, blue_thresh: int):
        self.red_thresh = red_thresh
        self.green_thresh = green_thresh
        self.blue_thresh = blue_thresh

    def __call__(self, img: PIL.Image.Image) -> np.ndarray:
        return F.blue_filter(img, self.red_thresh, self.green_thresh, self.blue_thresh)


class BluePenFilter(ImageFilter):
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


    Example:
        >>> from PIL import Image
        >>> from histolab.filters.image_filters import BluePenFilter
        >>> image_rgb = Image.open("tests/fixtures/pil-images-rgb/wsi-blue-pen.png")
        >>> blue_pen_filter = BluePenFilter()
        >>> image_no_blue = blue_pen_filter(image_rgb)
    """  # noqa

    def __call__(self, img: PIL.Image.Image) -> PIL.Image.Image:
        return F.blue_pen_filter(img)


class YenThreshold(ImageFilter):
    r"""Mask image based on pixel above Yen threshold.

    Compute Yen threshold on image and return boolean mask based on pixels below
    this threshold. The Yen method [8]_ is a multi-level image thresholding approach
    to separate objects from the background. It automatically computes the
    threshold that maximize the entropic correlation EC for a given gray level s
    defined as:

    .. math::

        EC(s) = -\ln{(G(s)\cdot G'(s))} + 2\ln(P(s)\cdot (1-P(s))

    where :math:`\displaystyle{G(s)=\sum_{i=0}^{s-1}p_i^2}`,
    :math:`\displaystyle{G'(s)=\sum_{i=s}^{m-1}p_i^2}`, m is the number of gray
    levels in the image, :math:`p_i` is the probability of the gray level i and
    :math:`\displaystyle{P(s)=\sum_{i=0}^{s-1}p_i}` is the total probability up to
    gray level (s-1). In this filter, pixels below the computed threshold are
    considered as foreground.

    .. figure:: https://user-images.githubusercontent.com/31658006/116542194-ab065800-a8ec-11eb-9fea-24dd97de8226.png

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


    Example:
        >>> from PIL import Image
        >>> from histolab.filters.image_filters import RgbToGrayscale, YenThreshold
        >>> image_rgb = Image.open("tests/fixtures/pil-images-rgb/tcga-lung-rgb.png")
        >>> rgb_to_grayscale = RgbToGrayscale()
        >>> yen_threshold = YenThreshold()
        >>> image_gray = rgb_to_grayscale(image_rgb)
        >>> image_thresholded_array = yen_threshold(image_gray)

    References
    ----------
    .. [8] J.C. Yen and et al.“A new criterion for automatic multilevel
        thresholding”. IEEE Trans Image Process 4.3 (1995)
    """  # noqa

    def __init__(self, relate: Callable[..., Any] = operator.lt):
        self.relate = relate

    def __call__(self, img: PIL.Image.Image) -> np.ndarray:
        return F.yen_threshold(img, self.relate)
