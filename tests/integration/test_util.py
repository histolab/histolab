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
import pytest

from histolab.masks import BiggestTissueBoxMask, TissueMask
from histolab.slide import Slide
from histolab.util import random_choice_true_mask2d

from ..fixtures import SVS, TIFF


@pytest.mark.parametrize(
    "fixture_slide, binary_mask",
    [
        (TIFF.KIDNEY_48_5, BiggestTissueBoxMask()),
        (TIFF.KIDNEY_48_5, TissueMask()),
        (SVS.TCGA_CR_7395_01A_01_TS1, BiggestTissueBoxMask()),
        (SVS.TCGA_CR_7395_01A_01_TS1, TissueMask()),
    ],
)
def test_random_choice_true_mask2d_find_right_coordinates(fixture_slide, binary_mask):
    np.random.seed(0)
    slide = Slide(fixture_slide, "")
    bbox = binary_mask(slide)

    col, row = random_choice_true_mask2d(bbox)
    assert bbox[row, col]
