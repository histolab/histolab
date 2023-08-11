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
import PIL

from histolab import scorer
from histolab.tile import Tile

from ..base import COMPLEX_MASK
from ..unitutil import PILIMG, function_mock, instance_mock, method_mock, property_mock


class DescribeScorers:
    def it_can_construct_randomscorer(self, request):
        tile = instance_mock(request, Tile)
        random_scorer = scorer.RandomScorer()

        score = random_scorer(tile)

        assert isinstance(random_scorer, scorer.RandomScorer)
        assert isinstance(score, float) is True

    def it_can_construct_nuclei_scorer(self, request):
        image = PILIMG.RGB_RANDOM_COLOR_10X10
        tissue_ratio_ = property_mock(request, Tile, "tissue_ratio")
        tissue_ratio_.return_value = 0.7
        apply_filters_ = method_mock(request, Tile, "apply_filters")
        apply_filters_.return_value = Tile(PIL.Image.fromarray(COMPLEX_MASK), None, 0)
        mask_difference_ = function_mock(request, "histolab.scorer.mask_difference")
        mask_difference_.return_value = np.zeros_like(COMPLEX_MASK).astype("bool")
        tile = Tile(image, None, 0)
        nuclei_scorer = scorer.NucleiScorer()

        score = nuclei_scorer(tile)

        tissue_ratio_.assert_called_once()
        assert len(apply_filters_.call_args_list) == 2
        # not possible to test filters compositions instances used in the call
        assert apply_filters_.call_args_list[0][0][0] == tile
        assert apply_filters_.call_args_list[1][0][0] == tile
        np.testing.assert_array_equal(
            mask_difference_.call_args_list[0][0][0], COMPLEX_MASK
        )
        np.testing.assert_array_equal(
            mask_difference_.call_args_list[0][0][1], COMPLEX_MASK
        )
        assert isinstance(nuclei_scorer, scorer.NucleiScorer)
        assert isinstance(score, np.float64) is True
        assert score == 0  # to avoid float representation issues

    def it_can_construct_cellurarity_scorer_considering_tissue(self, request):
        image = PILIMG.RGB_RANDOM_COLOR_10X10
        apply_filters_ = method_mock(request, Tile, "apply_filters")
        apply_filters_.return_value = Tile(PIL.Image.fromarray(COMPLEX_MASK), None, 0)
        tissue_mask_ = property_mock(request, Tile, "tissue_mask")
        tissue_mask_.return_value = COMPLEX_MASK
        tile = Tile(image, None, 0)
        cellularity_scorer = scorer.CellularityScorer(consider_tissue=True)

        score = cellularity_scorer(tile)

        tissue_mask_.assert_called_once()
        assert len(apply_filters_.call_args_list) == 1
        # not possible to test filters compositions instances used in the call

        assert apply_filters_.call_args_list[0][0][0] == tile
        assert isinstance(cellularity_scorer, scorer.CellularityScorer)
        assert isinstance(score, float) is True
        assert score == 1  # to avoid float representation issues

    def it_can_construct_cellurarity_scorer_without_considering_tissue(self, request):
        image = PILIMG.RGB_RANDOM_COLOR_10X10
        apply_filters_ = method_mock(request, Tile, "apply_filters")
        apply_filters_.return_value = Tile(PIL.Image.fromarray(COMPLEX_MASK), None, 0)
        tissue_mask_ = property_mock(request, Tile, "tissue_mask")
        tissue_mask_.return_value = COMPLEX_MASK
        tile = Tile(image, None, 0)
        cellularity_scorer = scorer.CellularityScorer(consider_tissue=False)

        score = cellularity_scorer(tile)

        assert len(apply_filters_.call_args_list) == 1
        # not possible to test filters compositions instances used in the call
        assert apply_filters_.call_args_list[0][0][0] == tile

        assert isinstance(cellularity_scorer, scorer.CellularityScorer)
        assert isinstance(score, float) is True
        assert score == 0.61  # to avoid float representation issues
