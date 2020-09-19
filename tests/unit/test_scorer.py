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
        assert type(score) == float

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
        assert type(score) == np.float64
        assert score == 0  # to avoid float representation issues
