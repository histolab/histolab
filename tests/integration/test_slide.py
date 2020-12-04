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

    @pytest.mark.parametrize(
        "slide_fixture, tissue_mask, save_scaled_image, expectation",
        [
            (
                SVS.CMU_1_SMALL_REGION,
                True,
                True,
                "cmu-1-small-region-bbox-location-tissue-mask-true",
            ),
            (
                SVS.CMU_1_SMALL_REGION,
                False,
                True,
                "cmu-1-small-region-bbox-location-tissue-mask-false",
            ),
            (
                SVS.TCGA_CR_7395_01A_01_TS1,
                True,
                False,
                "tcga-cr-7395-01a-01-ts1-bbox-location-tissue-mask-true",
            ),
            (
                SVS.TCGA_CR_7395_01A_01_TS1,
                False,
                False,
                "tcga-cr-7395-01a-01-ts1-bbox-location-tissue-mask-false",
            ),
        ],
    )
    def it_locates_the_biggest_bbox(
        self, tmpdir, slide_fixture, tissue_mask, save_scaled_image, expectation
    ):
        slide = Slide(slide_fixture, os.path.join(tmpdir, "processed"))
        if save_scaled_image:
            slide.save_scaled_image(3)
        expected_img = load_expectation(
            os.path.join("bbox-location-images", expectation),
            type_="png",
        )
        bbox_location_img = slide.locate_biggest_tissue_box(
            tissue_mask=tissue_mask, scale_factor=3
        )

        np.testing.assert_array_almost_equal(
            np.asarray(bbox_location_img), expected_img
        )
