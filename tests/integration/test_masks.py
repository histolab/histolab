# encoding: utf-8

import os

import numpy as np

from histolab.masks import BiggestTissueBoxMask
from histolab.slide import Slide

from ..fixtures import SVS
from ..util import load_expectation


class DescribeBiggestTissueBoxMask:
    def it_can_construct_big_tissue_box_mask(self):
        wsi = SVS.CMU_1_SMALL_REGION
        slide = Slide(wsi, os.path.join(wsi, "processed"))
        expected_array = load_expectation(
            "mask-arrays/biggest-tissue-box-cmu-1-small-region", type_="npy"
        )

        biggest_tissue_box = BiggestTissueBoxMask()

        np.testing.assert_array_almost_equal(biggest_tissue_box(slide), expected_array)
