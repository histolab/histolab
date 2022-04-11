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
