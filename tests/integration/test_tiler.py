# encoding: utf-8

import os

import numpy as np
import pytest

from histolab.scorer import NucleiScorer
from histolab.slide import Slide
from histolab.tiler import GridTiler, RandomTiler, ScoreTiler

from ..fixtures import SVS
from ..util import load_expectation, expand_tests_report


class DescribeRandomTiler:
    @pytest.mark.parametrize(
        "fixture_slide, expectation",
        [
            (
                SVS.CMU_1_SMALL_REGION,
                "tiles-location-images/cmu-1-small-region-tiles-location-random",
            ),
            (
                SVS.TCGA_CR_7395_01A_01_TS1,
                "tiles-location-images/tcga-cr-7395-01a-01-ts1-tiles-location-random",
            ),
        ],
    )
    def it_locates_tiles_on_the_slide(
        self, request, fixture_slide, expectation, tmpdir
    ):
        slide = Slide(fixture_slide, os.path.join(tmpdir, "processed"))
        slide.save_scaled_image(10)
        random_tiles_extractor = RandomTiler(
            tile_size=(512, 512), n_tiles=2, level=0, seed=42, check_tissue=False
        )
        expected_img = load_expectation(
            expectation,
            type_="png",
        )
        tiles_location_img = random_tiles_extractor.locate_tiles(slide, scale_factor=10)
        # --- Expanding test report with actual and expected images ---
        expand_tests_report(request, actual=tiles_location_img, expected=expected_img)

        np.testing.assert_array_almost_equal(
            np.asarray(tiles_location_img), expected_img
        )


class DescribeGridTiler:
    @pytest.mark.parametrize(
        "fixture_slide, expectation",
        [
            (
                SVS.CMU_1_SMALL_REGION,
                "tiles-location-images/cmu-1-small-region-tiles-location-grid",
            ),
            (
                SVS.TCGA_CR_7395_01A_01_TS1,
                "tiles-location-images/tcga-cr-7395-01a-01-ts1-tiles-location-grid",
            ),
        ],
    )
    def it_locates_tiles_on_the_slide(
        self, request, fixture_slide, expectation, tmpdir
    ):
        slide = Slide(fixture_slide, os.path.join(tmpdir, "processed"))
        grid_tiles_extractor = GridTiler(
            tile_size=(512, 512),
            level=0,
            check_tissue=False,
        )
        expected_img = load_expectation(expectation, type_="png")
        tiles_location_img = grid_tiles_extractor.locate_tiles(slide, scale_factor=10)
        # --- Expanding test report with actual and expected images ---
        expand_tests_report(request, expected=expected_img, actual=tiles_location_img)

        np.testing.assert_array_almost_equal(
            np.asarray(tiles_location_img), expected_img
        )


class DescribeScoreTiler:
    @pytest.mark.parametrize(
        "fixture_slide, expectation",
        [
            (
                SVS.CMU_1_SMALL_REGION,
                "tiles-location-images/cmu-1-small-region-tiles-location-scored",
            ),
            (
                SVS.TCGA_CR_7395_01A_01_TS1,
                "tiles-location-images/tcga-cr-7395-01a-01-ts1-tiles-location-scored",
            ),
        ],
    )
    def it_locates_tiles_on_the_slide(
        self, request, fixture_slide, expectation, tmpdir
    ):
        slide = Slide(fixture_slide, os.path.join(tmpdir, "processed"))
        scored_tiles_extractor = ScoreTiler(
            scorer=NucleiScorer(),
            tile_size=(512, 512),
            n_tiles=2,
            level=0,
            check_tissue=True,
        )
        expected_img = load_expectation(
            expectation,
            type_="png",
        )
        tiles_location_img = scored_tiles_extractor.locate_tiles(slide, scale_factor=10)
        # --- Expanding test report with actual and expected images ---
        expand_tests_report(request, expected=expected_img, actual=tiles_location_img)

        np.testing.assert_array_almost_equal(
            np.asarray(tiles_location_img), expected_img
        )
