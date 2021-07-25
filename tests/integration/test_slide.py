# encoding: utf-8

import ntpath
import os

import numpy as np
import PIL
import pytest

from histolab.exceptions import LevelError, SlidePropertyError
from histolab.masks import BiggestTissueBoxMask, TissueMask
from histolab.slide import Slide
from histolab.util import _check_largeimage

from ..fixtures import EXTERNAL_SVS, SVS
from ..unitutil import on_ci
from ..util import load_expectation, load_python_expression

LARGEIMAGE_IS_INSTALLED, LARGEIMAGE_INSTALL_PROMPT = _check_largeimage()


class Describe_Slide:
    def it_knows_its_name(self):
        slide = Slide(
            SVS.CMU_1_SMALL_REGION, os.path.join(SVS.CMU_1_SMALL_REGION, "processed")
        )

        name = slide.name

        assert name == ntpath.basename(SVS.CMU_1_SMALL_REGION).split(".")[0]

    @pytest.mark.parametrize(
        "use_largeimage, fake_props",
        [
            (True, None),
            (False, None),
            (False, {"aperio.MPP": 0.499}),
            (
                False,
                {
                    "tiff.XResolution": 20040.080160320642,
                    "tiff.ResolutionUnit": "centimeter",
                },
            ),
        ],
    )
    def it_knows_its_base_mpp(self, use_largeimage, fake_props):

        if use_largeimage and (not LARGEIMAGE_IS_INSTALLED):
            return

        slide = Slide(
            SVS.CMU_1_SMALL_REGION,
            os.path.join(SVS.CMU_1_SMALL_REGION, "processed"),
            use_largeimage=use_largeimage,
        )
        if fake_props:
            del slide.properties["openslide.mpp-x"]
            del slide.properties["aperio.MPP"]
            slide.properties.update(fake_props)

        mpp = slide.base_mpp

        np.testing.assert_almost_equal(mpp, 0.499)

    def it_calculate_resampled_nparray_from_small_region_svs_image(self):
        slide = Slide(
            SVS.CMU_1_SMALL_REGION, os.path.join(SVS.CMU_1_SMALL_REGION, "processed")
        )

        resampled_array = slide.resampled_array(scale_factor=32)

        expected_value = load_expectation(
            "svs-images/small-region-svs-resampled-array", type_="npy"
        )
        np.testing.assert_almost_equal(resampled_array, expected_value)

    @pytest.mark.parametrize(
        "use_largeimage",
        [
            (False,),
            (True,),
        ],
    )
    def it_knows_the_right_slide_dimension(self, use_largeimage):

        if use_largeimage and (not LARGEIMAGE_IS_INSTALLED):
            return

        slide = Slide(
            SVS.CMU_1_SMALL_REGION,
            os.path.join(SVS.CMU_1_SMALL_REGION, "processed"),
            use_largeimage=use_largeimage,
        )
        image = PIL.Image.open(SVS.CMU_1_SMALL_REGION)

        dimensions = slide.dimensions

        assert image.size == dimensions
        assert slide.dimensions == (2220, 2967)
        assert image.size == (2220, 2967)

    @pytest.mark.parametrize(
        "fixture_slide, level, expected_value",
        (
            pytest.param(
                EXTERNAL_SVS.LIVER_1,
                0,
                "20.0X",
                marks=pytest.mark.skipif(not on_ci(), reason="To run only on CI"),
            ),
            pytest.param(
                EXTERNAL_SVS.LIVER_1,
                1,
                "5.0X",
                marks=pytest.mark.skipif(not on_ci(), reason="To run only on CI"),
            ),
            pytest.param(
                EXTERNAL_SVS.LIVER_1,
                2,
                "1.25X",
                marks=pytest.mark.skipif(not on_ci(), reason="To run only on CI"),
            ),
            (SVS.CMU_1_SMALL_REGION, 0, "20.0X"),
        ),
    )
    def it_knows_its_magnification_factors(self, fixture_slide, level, expected_value):
        slide = Slide(fixture_slide, "")

        magnification_factor = slide.level_magnification_factor(level=level)

        assert magnification_factor == expected_value

    @pytest.mark.parametrize(
        "fixture_slide, level",
        (
            (SVS.TCGA_CR_7395_01A_01_TS1, 0),
            (SVS.TCGA_CR_7395_01A_01_TS1, 1),
        ),
    )
    def but_it_raises_exception_when_magnification_cannot_be_computed(
        self, fixture_slide, level
    ):
        slide = Slide(fixture_slide, "")

        with pytest.raises(SlidePropertyError) as err:
            slide.level_magnification_factor(level=level)

        assert isinstance(err.value, SlidePropertyError)
        assert (
            str(err.value)
            == f"Native magnification not available. Available slide properties: "
            f"{list(slide.properties.keys())}"
        )

    @pytest.mark.parametrize(
        "fixture_slide, level",
        [
            pytest.param(
                EXTERNAL_SVS.LIVER_1,
                4,
                marks=pytest.mark.skipif(not on_ci(), reason="To run only on CI"),
            ),
        ],
    )
    def and_when_level_is_incorrect(self, fixture_slide, level):
        slide = Slide(fixture_slide, "")

        with pytest.raises(LevelError) as err:
            slide.level_magnification_factor(level=level)

        assert isinstance(err.value, LevelError)
        assert (
            str(err.value)
            == f"Level {level} not available. Number of available levels: 3"
        )

    def it_raises_openslideerror_with_broken_wsi(self):
        slide = Slide(SVS.BROKEN, os.path.join(SVS.BROKEN, "processed"))

        with pytest.raises(PIL.UnidentifiedImageError) as err:
            slide._wsi

        assert isinstance(err.value, PIL.UnidentifiedImageError)
        broken_err = "Your wsi has something broken inside, a doctor is needed. "
        broken_err += LARGEIMAGE_INSTALL_PROMPT
        assert str(err.value) == broken_err

    @pytest.mark.parametrize(
        "slide_fixture, tissue_mask, binary_mask, expectation",
        [
            (
                SVS.CMU_1_SMALL_REGION,
                True,
                BiggestTissueBoxMask(),
                "cmu-1-small-region-bbox-location-tissue-mask-true",
            ),
            (
                SVS.CMU_1_SMALL_REGION,
                False,
                BiggestTissueBoxMask(),
                "cmu-1-small-region-bbox-location-tissue-mask-false",
            ),
            (
                SVS.TCGA_CR_7395_01A_01_TS1,
                True,
                BiggestTissueBoxMask(),
                "tcga-cr-7395-01a-01-ts1-bbox-location-tissue-mask-true",
            ),
            (
                SVS.TCGA_CR_7395_01A_01_TS1,
                False,
                BiggestTissueBoxMask(),
                "tcga-cr-7395-01a-01-ts1-bbox-location-tissue-mask-false",
            ),
            (
                SVS.CMU_1_SMALL_REGION,
                True,
                TissueMask(),
                "cmu-1-small-region-tissue-location-tissue-mask-true",
            ),
            (
                SVS.CMU_1_SMALL_REGION,
                False,
                TissueMask(),
                "cmu-1-small-region-tissue-location-tissue-mask-false",
            ),
            (
                SVS.TCGA_CR_7395_01A_01_TS1,
                True,
                TissueMask(),
                "tcga-cr-7395-01a-01-ts1-tissue-location-tissue-mask-true",
            ),
            (
                SVS.TCGA_CR_7395_01A_01_TS1,
                False,
                TissueMask(),
                "tcga-cr-7395-01a-01-ts1-tissue-location-tissue-mask-false",
            ),
        ],
    )
    def it_locates_the_mask(
        self, tmpdir, slide_fixture, tissue_mask, binary_mask, expectation
    ):
        slide = Slide(slide_fixture, os.path.join(tmpdir, "processed"))
        expected_img = load_expectation(
            os.path.join("mask-location-images", expectation),
            type_="png",
        )

        mask_location_img = slide.locate_mask(
            binary_mask, tissue_mask=tissue_mask, scale_factor=3
        )

        np.testing.assert_array_almost_equal(
            np.asarray(mask_location_img), expected_img
        )

    def it_knows_its_properties(self):
        slide = Slide(SVS.CMU_1_SMALL_REGION, "processed")

        properties = slide.properties

        assert isinstance(properties, dict)
        assert properties == load_python_expression("python-expr/slide_properties_dict")
