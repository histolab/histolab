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
    slide = Slide(fixture_slide, "")
    bbox = binary_mask(slide)

    x, y = random_choice_true_mask2d(bbox)
    assert bbox[x, y]
