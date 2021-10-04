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
import scipy.ndimage as sc_ndimage
import skimage.feature as sk_feature
import skimage.morphology as sk_morphology
import skimage.segmentation as sk_segmentation

from .util import mask_percent


def remove_small_objects(
    np_mask: np.ndarray,
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
    mask_no_small_object = sk_morphology.remove_small_objects(np_mask, min_size)
    if (
        avoid_overmask
        and mask_percent(mask_no_small_object) >= overmask_thresh
        and min_size >= 1
    ):
        new_min_size = min_size // 2
        mask_no_small_object = remove_small_objects(
            np_mask, new_min_size, avoid_overmask, overmask_thresh
        )
    return mask_no_small_object


def watershed_segmentation(np_mask: np.ndarray, region_shape: int = 6) -> np.ndarray:
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
    """
    distance = sc_ndimage.distance_transform_edt(np_mask)
    local_maxi = sk_feature.peak_local_max(
        distance,
        indices=False,
        footprint=np.ones((region_shape, region_shape)),
        labels=np_mask,
    )
    markers = sc_ndimage.label(local_maxi)[0]
    labels = sk_segmentation.watershed(-distance, markers, mask=np_mask)
    return labels
