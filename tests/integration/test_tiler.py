# encoding: utf-8

import os

import numpy as np

from histolab.slide import Slide
from histolab.tiler import GridTiler, RandomTiler, ScoreTiler
from histolab.scorer import NucleiScorer

from ..fixtures import SVS
from ..util import load_expectation


class DescribeRandomTiler:
    def it_locates_tiles_on_the_slide(self, tmpdir):
        slide = Slide(SVS.CMU_1_SMALL_REGION, os.path.join(tmpdir, "processed"))
        slide.save_thumbnail()
        random_tiles_extractor = RandomTiler(
            tile_size=(512, 512), n_tiles=2, level=0, seed=42, check_tissue=False
        )
        expectation = load_expectation(
            "tiles-location-images/cmu-1-small-region-tiles-location-random",
            type_="png",
        )
        tiles_location_img = random_tiles_extractor.locate_tiles(slide)

        np.testing.assert_array_almost_equal(
            np.asarray(tiles_location_img), expectation
        )


class DescribeGridTiler:
    def it_locates_tiles_on_the_slide(self, tmpdir):
        slide = Slide(SVS.CMU_1_SMALL_REGION, os.path.join(tmpdir, "processed"))
        grid_tiles_extractor = GridTiler(
            tile_size=(512, 512),
            level=0,
            check_tissue=False,
        )
        expectation = load_expectation(
            "tiles-location-images/cmu-1-small-region-tiles-location-grid", type_="png"
        )
        tiles_location_img = grid_tiles_extractor.locate_tiles(slide)

        np.testing.assert_array_almost_equal(
            np.asarray(tiles_location_img), expectation
        )


class DescribeScoreTiler:
    def it_locates_tiles_on_the_slide(self, tmpdir):
        slide = Slide(SVS.CMU_1_SMALL_REGION, os.path.join(tmpdir, "processed"))
        scored_tiles_extractor = ScoreTiler(
            scorer=NucleiScorer(),
            tile_size=(512, 512),
            n_tiles=100,
            level=0,
            check_tissue=False,
        )
        expectation = load_expectation(
            "tiles-location-images/cmu-1-small-region-tiles-location-scored",
            type_="png",
        )
        scored_location_img = scored_tiles_extractor.locate_tiles(slide)

        np.testing.assert_array_almost_equal(
            np.asarray(scored_location_img), expectation
        )
