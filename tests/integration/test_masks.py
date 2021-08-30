# encoding: utf-8

import os

import numpy as np
import pytest

from histolab.masks import BiggestTissueBoxMask, TissueMask
from histolab.slide import Slide
from histolab.tile import Tile

from ..fixtures import RGB, SVS, TIFF, TILES
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
        mask = biggest_tissue_box(slide)

        np.testing.assert_array_almost_equal(mask, expected_array)


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
    def it_can_construct_tissue_mask_slide(self, wsi, expected_array):
        slide = Slide(wsi, os.path.join(wsi, "processed"))
        expected_array = load_expectation(expected_array, type_="npy")
        tissue_mask = TissueMask()

        slide_mask = tissue_mask(slide)

        np.testing.assert_array_almost_equal(slide_mask, expected_array)

    @pytest.mark.parametrize(
        "tile, expected_array",
        (
            (
                RGB.DIAGNOSTIC_SLIDE_THUMB_RGB,  # pen marks get considered as tissue
                "mask-arrays/tissue-mask-diagnostic-slide-thumb-rgb",
            ),
            (
                RGB.TCGA_LUNG_RGB,
                "mask-arrays/tissue-mask-tcga-lung-rgb",
            ),
            (
                TILES.TISSUE_LEVEL2_3000_7666_7096_11763,
                "mask-arrays/tissue-mask-tissue-level2-3000-7666-7096-11763",
            ),
        ),
    )
    def it_can_construct_tissue_mask_tile(self, tile, expected_array):
        tile = Tile(tile, None, None)
        expected_array = load_expectation(expected_array, type_="npy")
        tissue_mask = TissueMask()

        tile_mask = tissue_mask(tile)

        np.testing.assert_array_almost_equal(tile_mask, expected_array)
