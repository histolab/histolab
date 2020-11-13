# encoding: utf-8

import os

import ntpath
import numpy as np
import PIL
import pytest

from histolab.slide import Slide

from ..fixtures import SVS
from ..util import load_expectation


class Describe_Slide:
    def it_knows_its_name(self):
        slide = Slide(
            SVS.CMU_1_SMALL_REGION, os.path.join(SVS.CMU_1_SMALL_REGION, "processed")
        )

        name = slide.name

        assert name == ntpath.basename(SVS.CMU_1_SMALL_REGION).split(".")[0]

    def it_calculate_resampled_nparray_from_small_region_svs_image(self):
        slide = Slide(
            SVS.CMU_1_SMALL_REGION, os.path.join(SVS.CMU_1_SMALL_REGION, "processed")
        )

        resampled_array = slide.resampled_array(scale_factor=32)

        expected_value = load_expectation(
            "svs-images/small-region-svs-resampled-array", type_="npy"
        )
        np.testing.assert_almost_equal(resampled_array, expected_value)

    def it_knows_the_right_slide_dimension(self):
        slide = Slide(
            SVS.CMU_1_SMALL_REGION, os.path.join(SVS.CMU_1_SMALL_REGION, "processed")
        )
        image = PIL.Image.open(SVS.CMU_1_SMALL_REGION)

        dimensions = slide.dimensions

        assert image.size == dimensions
        assert slide.dimensions == (2220, 2967)
        assert image.size == (2220, 2967)

    def it_raises_openslideerror_with_broken_wsi(self):
        slide = Slide(SVS.BROKEN, os.path.join(SVS.BROKEN, "processed"))

        with pytest.raises(PIL.UnidentifiedImageError) as err:
            slide._wsi

        assert isinstance(err.value, PIL.UnidentifiedImageError)
        assert (
            str(err.value) == "Your wsi has something broken inside, a doctor is needed"
        )
