# encoding: utf-8

import os

import numpy as np
import pytest
from PIL import Image

from histolab.masks import BiggestTissueBoxMask, TissueMask
from histolab.scorer import NucleiScorer
from histolab.slide import Slide
from histolab.tiler import GridTiler, RandomTiler, ScoreTiler

from ..fixtures import SVS, TIFF
from ..util import expand_tests_report, load_expectation


class DescribeRandomTiler:
    @pytest.mark.parametrize(
        "fixture_slide, binary_mask, tile_size, level, check_tissue, expectation",
        [
            (
                SVS.CMU_1_SMALL_REGION,
                BiggestTissueBoxMask(),
                (512, 512),
                0,
                False,
                "tiles-location-images/cmu-1-small-region-tl-random-BTB-false-512x512",
            ),
            (
                SVS.TCGA_CR_7395_01A_01_TS1,
                TissueMask(),
                (512, 512),
                -2,
                False,
                "tiles-location-images/tcga-cr-7395-01a-01-ts1-tl-random-TM-f-512x512",
            ),
            (
                SVS.TCGA_CR_7395_01A_01_TS1,
                BiggestTissueBoxMask(),
                (512, 530),
                0,
                False,
                "tiles-location-images/tcga-cr-7395-01a-01-ts1-tl-random-BTB-f-512x530",
            ),
            (
                SVS.CMU_1_SMALL_REGION,
                TissueMask(),
                (512, 530),
                0,
                True,
                "tiles-location-images/cmu-1-small-region-tl-random-TM-true-512x530",
            ),
            (
                SVS.TCGA_CR_7395_01A_01_TS1,
                TissueMask(),
                (128, 128),
                0,
                True,
                "tiles-location-images/tcga-cr-7395-01a-01-ts1-tl-random-TM-t-128x128",
            ),
            (
                TIFF.KIDNEY_48_5,
                TissueMask(),
                (10, 10),
                0,
                True,
                "tiles-location-images/kidney-48-5-random-TM-true-10x10",
            ),
            (
                TIFF.KIDNEY_48_5,
                BiggestTissueBoxMask(),
                (20, 20),
                0,
                False,
                "tiles-location-images/kidney-48-5-random-TM-false-20x20",
            ),
        ],
    )
    def it_locates_tiles_on_the_slide(
        self,
        request,
        fixture_slide,
        binary_mask,
        tile_size,
        level,
        check_tissue,
        expectation,
    ):
        slide = Slide(fixture_slide, "")
        random_tiles_extractor = RandomTiler(
            tile_size=tile_size,
            n_tiles=2,
            level=level,
            seed=42,
            check_tissue=check_tissue,
        )
        expected_img = load_expectation(
            expectation,
            type_="png",
        )

        tiles_location_img = random_tiles_extractor.locate_tiles(
            slide, binary_mask, scale_factor=10
        )

        # --- Expanding test report with actual and expected images ---
        expand_tests_report(request, actual=tiles_location_img, expected=expected_img)

        np.testing.assert_array_almost_equal(tiles_location_img, expected_img)

    @pytest.mark.parametrize(
        "fixture_slide, tile_size, level, seed, n_tiles",
        (
            # Squared tile size
            (SVS.TCGA_CR_7395_01A_01_TS1, (128, 128), 1, 42, 20),
            (SVS.TCGA_CR_7395_01A_01_TS1, (128, 128), 0, 42, 10),
            (SVS.TCGA_CR_7395_01A_01_TS1, (128, 128), 1, 2, 20),
            (SVS.TCGA_CR_7395_01A_01_TS1, (128, 128), 0, 2, 10),
            (TIFF.KIDNEY_48_5, (10, 10), 0, 20, 20),
            (TIFF.KIDNEY_48_5, (20, 20), 0, 20, 10),
            # Not squared tile size
            (SVS.TCGA_CR_7395_01A_01_TS1, (135, 128), 1, 42, 20),
            (SVS.TCGA_CR_7395_01A_01_TS1, (135, 128), 0, 42, 10),
            (SVS.TCGA_CR_7395_01A_01_TS1, (135, 128), 1, 2, 20),
            (TIFF.KIDNEY_48_5, (10, 20), 0, 2, 10),
            (TIFF.KIDNEY_48_5, (20, 10), 0, 20, 20),
            (TIFF.KIDNEY_48_5, (10, 15), 0, 20, 10),
        ),
    )
    def test_extract_tiles_respecting_the_given_tile_size(
        self, tmpdir, fixture_slide, tile_size, level, seed, n_tiles
    ):
        processed_path = os.path.join(tmpdir, "processed")
        slide = Slide(fixture_slide, processed_path)
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
        "fixture_slide, binary_mask, tile_size, level,check_tissue, expectation",
        [
            (
                SVS.CMU_1_SMALL_REGION,
                BiggestTissueBoxMask(),
                (512, 512),
                0,
                False,
                "tiles-location-images/cmu-1-small-region-tl-grid-BTB-false-512x512",
            ),
            (
                SVS.TCGA_CR_7395_01A_01_TS1,
                TissueMask(),
                (512, 550),
                -2,
                False,
                "tiles-location-images/tcga-cr-7395-01a-01-ts1-tl-grid-TM-f-512x550",
            ),
            (
                SVS.TCGA_CR_7395_01A_01_TS1,
                BiggestTissueBoxMask(),
                (512, 512),
                0,
                False,
                "tiles-location-images/tcga-cr-7395-01a-01-ts1-tl-grid-BTB-f-512x512",
            ),
            (
                SVS.CMU_1_SMALL_REGION,
                TissueMask(),
                (128, 120),
                0,
                True,
                "tiles-location-images/cmu-1-small-region-tl-grid-TM-true-128x120",
            ),
            (
                SVS.TCGA_CR_7395_01A_01_TS1,
                TissueMask(),
                (128, 128),
                0,
                True,
                "tiles-location-images/tcga-cr-7395-01a-01-ts1-tl-grid-TM-true-128x128",
            ),
            (
                TIFF.KIDNEY_48_5,
                TissueMask(),
                (15, 10),
                0,
                True,
                "tiles-location-images/kidney-48-5-grid-TM-true-15x10",
            ),
            (
                TIFF.KIDNEY_48_5,
                BiggestTissueBoxMask(),
                (20, 20),
                0,
                False,
                "tiles-location-images/kidney-48-5-grid-TM-false-20x20",
            ),
        ],
    )
    def it_locates_tiles_on_the_slide(
        self,
        request,
        fixture_slide,
        binary_mask,
        tile_size,
        level,
        check_tissue,
        expectation,
    ):
        slide = Slide(fixture_slide, "")
        grid_tiles_extractor = GridTiler(
            tile_size=tile_size,
            level=level,
            check_tissue=check_tissue,
        )
        expected_img = load_expectation(expectation, type_="png")

        tiles_location_img = grid_tiles_extractor.locate_tiles(
            slide, binary_mask, scale_factor=10
        )

        # --- Expanding test report with actual and expected images ---
        expand_tests_report(request, expected=expected_img, actual=tiles_location_img)

        np.testing.assert_array_almost_equal(tiles_location_img, expected_img)


class DescribeScoreTiler:
    @pytest.mark.parametrize(
        "fixture_slide,tile_size, level,check_tissue, expectation",
        [
            (
                SVS.CMU_1_SMALL_REGION,
                (512, 512),
                0,
                False,
                "tiles-location-images/cmu-1-small-region-tl-score-false-512x512",
            ),
            (
                SVS.TCGA_CR_7395_01A_01_TS1,
                (512, 512),
                -2,
                False,
                "tiles-location-images/tcga-cr-7395-01a-01-ts1-tl-score-f-512x512",
            ),
            (
                SVS.TCGA_CR_7395_01A_01_TS1,
                (512, 530),
                0,
                False,
                "tiles-location-images/tcga-cr-7395-01a-01-ts1-tl-score-f-512x530",
            ),
            (
                SVS.CMU_1_SMALL_REGION,
                (120, 128),
                0,
                True,
                "tiles-location-images/cmu-1-small-region-tl-score-true-120x128",
            ),
            (
                SVS.TCGA_CR_7395_01A_01_TS1,
                (128, 128),
                0,
                True,
                "tiles-location-images/tcga-cr-7395-01a-01-ts1-tl-score-t-128x128",
            ),
        ],
    )
    def it_locates_tiles_on_the_slide(
        self,
        request,
        fixture_slide,
        tile_size,
        level,
        check_tissue,
        expectation,
    ):
        slide = Slide(fixture_slide, "")
        scored_tiles_extractor = ScoreTiler(
            scorer=NucleiScorer(),
            tile_size=tile_size,
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
                slide, scale_factor=8
            )

        # --- Expanding test report with actual and expected images ---
        expand_tests_report(request, expected=expected_img, actual=tiles_location_img)

        np.testing.assert_array_almost_equal(tiles_location_img, expected_img)
