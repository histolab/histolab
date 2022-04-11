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

from histolab.filters.util import mask_difference, mask_percent
from histolab.util import apply_mask_image, np_to_pil

from ..fixtures import NPY
from ..unitutil import PILIMG, NpArrayMock


class TestDescribeBenchmarksFilterUtil:
    def test_mask_difference(self, benchmark):
        minuend = np.random.choice(a=[False, True], size=(5000, 4000))
        subtrahend = np.random.choice(a=[False, True], size=(5000, 4000))
        benchmark.pedantic(mask_difference, args=[minuend, subtrahend], rounds=100)

    def test_mask_percent(self, benchmark):
        mask_array = np.random.choice(a=[False, True], size=(10000, 4000))
        benchmark.pedantic(mask_percent, args=[mask_array], iterations=100, rounds=50)


class TestDescribeBenchmarksUtil:
    def test_apply_mask_image(self, benchmark):
        image = PILIMG.RGBA_COLOR_500X500_155_249_240
        mask = NpArrayMock.ONES_500X500X4_BOOL
        benchmark.pedantic(
            apply_mask_image, args=[image, mask], iterations=100, rounds=50
        )

    @pytest.mark.parametrize(
        "image", (NPY.NP_TO_PIL_RGBA, NPY.NP_TO_PIL_L, NPY.NP_TO_PIL_LA)
    )
    def test_np_to_pil(self, image, benchmark):
        benchmark.pedantic(np_to_pil, args=[image], iterations=500, rounds=250)
