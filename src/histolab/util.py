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

import functools
import warnings
from typing import Callable, List, Tuple, Any

import numpy as np
import PIL
import PIL.ImageDraw
from skimage.measure import label, regionprops

from .types import CoordinatePair, Region

warn = functools.partial(warnings.warn, stacklevel=2)


def apply_mask_image(img: PIL.Image.Image, mask: np.ndarray) -> PIL.Image.Image:
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
    img_arr = np.array(img)

    if mask.ndim == 2 and img_arr.ndim != 2:
        masked_image = np.zeros(img_arr.shape, "uint8")
        n_channels = img_arr.shape[2]
        for channel_i in range(n_channels):
            masked_image[:, :, channel_i] = img_arr[:, :, channel_i] * mask
    else:
        masked_image = img_arr * mask
    return np_to_pil(masked_image)


def np_to_pil(np_img: np.ndarray) -> PIL.Image.Image:
    """Convert a NumPy array to a PIL Image.

    Parameters
    ----------
    np_img : np.ndarray
        The image represented as a NumPy array.

    Returns
    -------
    PIL.Image.Image
        The image represented as PIL Image
    """

    def _transform_bool(image_array):
        return image_array.astype(np.uint8) * 255

    def _transform_float(image_array):
        return (image_array * 255).astype(np.uint8)

    types_factory = {
        "bool": _transform_bool(np_img),
        "float64": _transform_float(np_img),
    }
    image_array = types_factory.get(str(np_img.dtype), np_img.astype(np.uint8))
    return PIL.Image.fromarray(image_array)


def polygon_to_mask_array(dims: tuple, vertices: CoordinatePair) -> np.ndarray:
    """Draw a white polygon of vertices on a black image of specified dimensions.

    Parameters
    ----------
    dims : tuple
        (w,h) of the black image
    vertices : CoordinatePair
        CoordinatePair representing the upper left and bottom right vertices of the
        polygon

    Returns
    -------
    np.ndarray
        NumPy array corresponding to the image with the polygon
    """

    poly_vertices = [
        (vertices.x_ul, vertices.y_ul),
        (vertices.x_ul, vertices.y_br),
        (vertices.x_br, vertices.y_br),
        (vertices.x_br, vertices.y_ul),
    ]

    img = PIL.Image.new("L", dims, 0)
    PIL.ImageDraw.Draw(img).polygon(poly_vertices, outline=1, fill=1)
    return np.array(img).astype(bool)


def regions_from_binary_mask(binary_mask: np.ndarray) -> List[Region]:
    """Calculate regions properties from a binary mask.

    Parameters
    ----------
    binary_mask : np.ndarray
        Binary mask from which to extract the regions

    Returns
    -------
    List[Region]
        Properties for all the regions present in the binary mask
    """

    thumb_labeled_regions = label(binary_mask)
    regions = [
        Region(index=i, area=rp.area, bbox=rp.bbox, center=rp.centroid)
        for i, rp in enumerate(regionprops(thumb_labeled_regions))
    ]
    return regions


def region_coordinates(region: Region) -> CoordinatePair:
    """Extract bbox coordinates from the region.

    Parameters
    ----------
    region : Region
        Region from which to extract the coordinates of the bbox

    Returns
    -------
    CoordinatePair
        Coordinates of the bbox
    """
    y_ul, x_ul, y_br, x_br = region.bbox
    return CoordinatePair(x_ul, y_ul, x_br, y_br)


def scale_coordinates(
    reference_coords: CoordinatePair,
    reference_size: Tuple[int, int],
    target_size: Tuple[int, int],
) -> CoordinatePair:
    """Compute the coordinates corresponding to a scaled version of the image.

    Parameters
    ----------
    reference_coords: CoordinatePair
        Coordinates referring to the upper left and lower right corners
        respectively.
    reference_size: tuple of int
        Reference (width, height) size to which input coordinates refer to
    target_size: tuple of int
        Target (width, height) size of the resulting scaled image

    Returns
    -------
    coords: CoordinatesPair
        Coordinates in the scaled image
    """
    reference_coords = np.asarray(reference_coords).ravel()
    reference_size = np.tile(reference_size, 2)
    target_size = np.tile(target_size, 2)
    return CoordinatePair(
        *np.floor((reference_coords * target_size) / reference_size).astype("int64")
    )


def threshold_to_mask(
    img: PIL.Image.Image, threshold: float, relate: Callable[..., bool]
) -> np.ndarray:
    """Mask image with pixel according to the threshold value.

    Parameters
    ----------
    img: PIL.Image.Image
        Input image
    threshold: float
        The threshold value to exceed.
    relate: callable operator
        Comparison operator between img pixel values and threshold

    Returns
    -------
    np.ndarray
        Boolean NumPy array representing a mask where a pixel has a value True
        if the corresponding input array pixel exceeds the threshold value.
    """
    img_arr = np.array(img)
    return relate(img_arr, threshold)


def lazyproperty(f: Callable[..., Any]):
    """Decorator like @property, but evaluated only on first access.

    Like @property, this can only be used to decorate methods having only
    a `self` parameter, and is accessed like an attribute on an instance,
    i.e. trailing parentheses are not used. Unlike @property, the decorated
    method is only evaluated on first access; the resulting value is cached
    and that same value returned on second and later access without
    re-evaluation of the method.

    Like @property, this class produces a *data descriptor* object, which is
    stored in the __dict__ of the *class* under the name of the decorated
    method ('fget' nominally). The cached value is stored in the __dict__ of
    the *instance* under that same name.

    Because it is a data descriptor (as opposed to a *non-data descriptor*),
    its `__get__()` method is executed on each access of the decorated
    attribute; the __dict__ item of the same name is "shadowed" by the
    descriptor.

    While this may represent a performance improvement over a property, its
    greater benefit may be its other characteristics. One common use is to
    construct collaborator objects, removing that "real work" from the
    constructor, while still only executing once. It also de-couples client
    code from any sequencing considerations; if it's accessed from more than
    one location, it's assured it will be ready whenever needed.

    A lazyproperty is read-only. There is no counterpart to the optional
    "setter" (or deleter) behavior of an @property. This is critically
    important to maintaining its immutability and idempotence guarantees.
    Attempting to assign to a lazyproperty raises AttributeError
    unconditionally.
    The parameter names in the methods below correspond to this usage
    example::

        class Obj(object):

            @lazyproperty
            def fget(self):
                return 'some result'

        obj = Obj()

    Not suitable for wrapping a function (as opposed to a method) because it
    is not callable.
    """
    # pylint: disable=unused-variable
    return property(functools.lru_cache(maxsize=100)(f))
