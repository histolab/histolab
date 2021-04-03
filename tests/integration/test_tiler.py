# encoding: utf-8

import os

import numpy as np
import pytest
from PIL import Image

from histolab.masks import BiggestTissueBoxMask
from histolab.scorer import NucleiScorer
from histolab.slide import Slide
from histolab.tiler import GridTiler, RandomTiler, ScoreTiler

from ..fixtures import SVS
from ..util import expand_tests_report, load_expectation


class DescribeRandomTiler:
    @pytest.mark.parametrize(
        "fixture_slide, level, check_tissue, expectation",
        [
            (
                SVS.CMU_1_SMALL_REGION,
                0,
                False,
                "tiles-location-images/cmu-1-small-region-tl-random-false",
            ),
            (
                SVS.TCGA_CR_7395_01A_01_TS1,
                -2,
                False,
                "tiles-location-images/tcga-cr-7395-01a-01-ts1-tl-random-false",
            ),
            (
                SVS.TCGA_CR_7395_01A_01_TS1,
                0,
                False,
                "tiles-location-images/tcga-cr-7395-01a-01-ts1-tl-random-false",
            ),
            (
                SVS.CMU_1_SMALL_REGION,
                0,
                True,
                "tiles-location-images/cmu-1-small-region-tl-random-true",
            ),
            (
                SVS.TCGA_CR_7395_01A_01_TS1,
                0,
                True,
                "tiles-location-images/tcga-cr-7395-01a-01-ts1-tl-random-true",
            ),
        ],
    )
    def it_locates_tiles_on_the_slide(
        self, request, fixture_slide, level, check_tissue, expectation, tmpdir
    ):
        slide = Slide(fixture_slide, os.path.join(tmpdir, "processed"))
        random_tiles_extractor = RandomTiler(
            tile_size=(512, 512),
            n_tiles=2,
            level=level,
            seed=42,
            check_tissue=check_tissue,
        )
        expected_img = load_expectation(
            expectation,
            type_="png",
        )
        binary_mask = BiggestTissueBoxMask()

        tiles_location_img = random_tiles_extractor.locate_tiles(
            slide, binary_mask, scale_factor=10
        )
        # --- Expanding test report with actual and expected images ---
        expand_tests_report(request, actual=tiles_location_img, expected=expected_img)

        np.testing.assert_array_almost_equal(tiles_location_img, expected_img)

    @pytest.mark.parametrize(
        "tile_size, level, seed, n_tiles",
        (
            # Squared tile size
            ((128, 128), 1, 42, 20),
            ((128, 128), 0, 42, 10),
            ((128, 128), 1, 2, 20),
            ((128, 128), 0, 2, 10),
            ((128, 128), 1, 20, 20),
            ((128, 128), 0, 20, 10),
            # Not squared tile size
            ((135, 128), 1, 42, 20),
            ((135, 128), 0, 42, 10),
            ((135, 128), 1, 2, 20),
            ((135, 128), 0, 2, 10),
            ((135, 128), 1, 20, 20),
            ((135, 128), 0, 20, 10),
        ),
    )
    def test_extract_tiles_respecting_the_given_tile_size(
        self, tmpdir, tile_size, level, seed, n_tiles
    ):
        processed_path = os.path.join(tmpdir, "processed")
        slide = Slide(SVS.TCGA_CR_7395_01A_01_TS1, processed_path)
        random_tiles_extractor = RandomTiler(
            tile_size=tile_size,
            n_tiles=n_tiles,
            level=level,
            seed=seed,
            check_tissue=True,
        )
        binary_mask = BiggestTissueBoxMask()

        random_tiles_extractor.extract(slide, binary_mask)

        for tile in os.listdir(processed_path):
            assert Image.open(os.path.join(processed_path, tile)).size == tile_size


class DescribeGridTiler:
    @pytest.mark.parametrize(
        "fixture_slide, level,check_tissue, expectation",
        [
            (
                SVS.CMU_1_SMALL_REGION,
                0,
                False,
                "tiles-location-images/cmu-1-small-region-tl-grid-false",
            ),
            (
                SVS.TCGA_CR_7395_01A_01_TS1,
                -2,
                False,
                "tiles-location-images/tcga-cr-7395-01a-01-ts1-tl-grid-false",
            ),
            (
                SVS.TCGA_CR_7395_01A_01_TS1,
                0,
                False,
                "tiles-location-images/tcga-cr-7395-01a-01-ts1-tl-grid-false",
            ),
            (
                SVS.CMU_1_SMALL_REGION,
                0,
                True,
                "tiles-location-images/cmu-1-small-region-tl-grid-true",
            ),
            (
                SVS.TCGA_CR_7395_01A_01_TS1,
                0,
                True,
                "tiles-location-images/tcga-cr-7395-01a-01-ts1-tl-grid-true",
            ),
        ],
    )
    def it_locates_tiles_on_the_slide(
        self, request, fixture_slide, level, check_tissue, expectation, tmpdir
    ):
        slide = Slide(fixture_slide, os.path.join(tmpdir, "processed"))
        grid_tiles_extractor = GridTiler(
            tile_size=(512, 512),
            level=level,
            check_tissue=check_tissue,
        )
        expected_img = load_expectation(expectation, type_="png")
        mask = BiggestTissueBoxMask()

        tiles_location_img = grid_tiles_extractor.locate_tiles(
            slide, mask, scale_factor=10
        )
        # --- Expanding test report with actual and expected images ---
        expand_tests_report(request, expected=expected_img, actual=tiles_location_img)

        np.testing.assert_array_almost_equal(tiles_location_img, expected_img)


class DescribeScoreTiler:
    @pytest.mark.parametrize(
        "fixture_slide, level, check_tissue, expectation",
        [
            (
                SVS.CMU_1_SMALL_REGION,
                0,
                False,
                "tiles-location-images/cmu-1-small-region-tl-scored-false",
            ),
            (
                SVS.TCGA_CR_7395_01A_01_TS1,
                -2,
                False,
                "tiles-location-images/tcga-cr-7395-01a-01-ts1-tl-scored-false",
            ),
            (
                SVS.TCGA_CR_7395_01A_01_TS1,
                0,
                False,
                "tiles-location-images/tcga-cr-7395-01a-01-ts1-tl-scored-false",
            ),
            (
                SVS.TCGA_CR_7395_01A_01_TS1,
                1,
                False,
                "tiles-location-images/tcga-cr-7395-01a-01-ts1-tl-scored-false-1",
            ),
            (
                SVS.CMU_1_SMALL_REGION,
                0,
                True,
                "tiles-location-images/cmu-1-small-region-tl-scored-true",
            ),
            (
                SVS.TCGA_CR_7395_01A_01_TS1,
                0,
                True,
                "tiles-location-images/tcga-cr-7395-01a-01-ts1-tl-scored-true",
            ),
        ],
    )
    def it_locates_tiles_on_the_slide(
        self,
        request,
        fixture_slide,
        level,
        check_tissue,
        expectation,
        tmpdir,
    ):
        slide = Slide(fixture_slide, os.path.join(tmpdir, "processed"))
        scored_tiles_extractor = ScoreTiler(
            scorer=NucleiScorer(),
            tile_size=(512, 512),
            n_tiles=2,
            level=level,
            check_tissue=check_tissue,
        )
        expected_img = load_expectation(
            expectation,
            type_="png",
        )
        expected_warning_regex = (
            r"Input image must be RGB. NOTE: the image will be converted to RGB before"
            r" HED conversion."
        )

        with pytest.warns(UserWarning, match=expected_warning_regex):
            # no binary mask object passed to locate_tiles
            # default value = BiggestTissueBoxMask
            tiles_location_img = scored_tiles_extractor.locate_tiles(
                slide, scale_factor=10
            )
        # --- Expanding test report with actual and expected images ---
        expand_tests_report(request, expected=expected_img, actual=tiles_location_img)

        np.testing.assert_array_almost_equal(tiles_location_img, expected_img)
