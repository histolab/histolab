# encoding: utf-8

import os

import numpy as np
import pytest

from histolab.masks import BiggestTissueBoxMask, TissueMask
from histolab.slide import Slide

from ..fixtures import SVS, TIFF
from ..util import load_expectation


class DescribeBiggestTissueBoxMask:
    @pytest.mark.parametrize(
        "wsi, expected_array",
        (
            (
                SVS.CMU_1_SMALL_REGION,
                "mask-arrays/biggest-tissue-box-cmu-1-small-region",
            ),
            (
                SVS.TCGA_CR_7395_01A_01_TS1,
                "mask-arrays/biggest-tissue-box-tcga-cr-7395",
            ),
            (
                TIFF.KIDNEY_48_5,
                "mask-arrays/biggest-tissue-box-kidney-48-5",
            ),
        ),
    )
    def it_can_construct_big_tissue_box_mask(self, wsi, expected_array):
        slide = Slide(wsi, "")
        expected_array = load_expectation(expected_array, type_="npy")
        biggest_tissue_box = BiggestTissueBoxMask()

        np.testing.assert_array_almost_equal(biggest_tissue_box(slide), expected_array)


class DescribeTissueMask:
    @pytest.mark.parametrize(
        "wsi, expected_array",
        (
            (
                SVS.CMU_1_SMALL_REGION,
                "mask-arrays/tissue-mask-cmu-1-small-region",
            ),
            (
                SVS.TCGA_CR_7395_01A_01_TS1,
                "mask-arrays/tissue-mask-tcga-cr-7395",
            ),
            (
                TIFF.KIDNEY_48_5,
                "mask-arrays/tissue-mask-kidney-48-5",
            ),
        ),
    )
    def it_can_construct_tissue_mask(self, wsi, expected_array):
        slide = Slide(wsi, os.path.join(wsi, "processed"))
        expected_array = load_expectation(expected_array, type_="npy")
        tissue_mask = TissueMask()
        np.testing.assert_array_almost_equal(tissue_mask(slide), expected_array)
