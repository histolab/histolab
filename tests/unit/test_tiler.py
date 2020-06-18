import os
from unittest.mock import call

import numpy as np
import pytest
import sparse

from histolab.slide import Slide
from histolab.tiler import RandomTiler, Tiler
from histolab.types import CoordinatePair

from ..unitutil import (
    ANY,
    PILImageMock,
    SparseArrayMock,
    function_mock,
    initializer_mock,
    method_mock,
    property_mock,
)


class Describe_RandomTiler(object):
    def it_constructs_from_args(self, request):
        _init = initializer_mock(request, RandomTiler)

        random_tiler = RandomTiler((512, 512), 10, 2, 7, True, "", ".png", 1e4)

        _init.assert_called_once_with(ANY, (512, 512), 10, 2, 7, True, "", ".png", 1e4)
        assert isinstance(random_tiler, RandomTiler)
        assert isinstance(random_tiler, Tiler)

    def but_it_has_wrong_tile_size_value(self, request):
        with pytest.raises(ValueError) as err:
            RandomTiler((512, -1), 10, 0)

        assert isinstance(err.value, ValueError)
        assert str(err.value) == "Tile size must be greater than 0 ((512, -1))"

    def or_it_has_not_available_level_value(self, request, tmpdir):
        tmp_path_ = tmpdir.mkdir("myslide")
        image = PILImageMock.DIMS_50X50_RGB_RANDOM_COLOR
        image.save(os.path.join(tmp_path_, "mywsi.png"), "PNG")
        slide_path = os.path.join(tmp_path_, "mywsi.png")
        slide = Slide(slide_path, "processed")
        random_tiler = RandomTiler((512, 512), 10, 3)

        with pytest.raises(ValueError) as err:
            random_tiler.extract(slide)

        assert isinstance(err.value, ValueError)
        assert str(err.value) == "Level 3 not available. Number of available levels: 1"

    def or_it_has_negative_level_value(self, request):
        with pytest.raises(ValueError) as err:
            RandomTiler((512, 512), 10, -1)

        assert isinstance(err.value, ValueError)
        assert str(err.value) == "Level cannot be negative (-1)"

    def or_it_has_wrong_max_iter(self, request):
        with pytest.raises(ValueError) as err:
            RandomTiler((512, 512), 10, 0, max_iter=3)

        assert isinstance(err.value, ValueError)
        assert (
            str(err.value)
            == "The maximum number of iterations (3) must be grater than or equal to the maximum number of tiles (10)."
        )

    def or_it_has_wrong_seed(self, request, tmpdir):
        tmp_path_ = tmpdir.mkdir("myslide")
        image = PILImageMock.DIMS_50X50_RGB_RANDOM_COLOR
        image.save(os.path.join(tmp_path_, "mywsi.png"), "PNG")
        slide_path = os.path.join(tmp_path_, "mywsi.png")
        slide = Slide(slide_path, "processed")
        random_tiler = RandomTiler((512, 512), 10, 0, seed=-1)

        with pytest.raises(ValueError) as err:
            random_tiler.extract(slide)

        assert isinstance(err.value, ValueError)
        assert str(err.value) == "Seed must be between 0 and 2**32 - 1"

    @pytest.mark.parametrize("tile_size", ((512, 512), (128, 128), (10, 10)))
    def it_knows_its_tile_size(self, request, tile_size):
        random_tiler = RandomTiler(tile_size, 10, 0)

        tile_size_ = random_tiler.tile_size

        assert type(tile_size_) == tuple
        assert tile_size_ == tile_size

    @pytest.mark.parametrize("max_iter", (1000, 10, 3000))
    def it_knows_its_max_iter(self, request, max_iter):
        random_tiler = RandomTiler((128, 128), 10, 0, max_iter=max_iter)

        max_iter_ = random_tiler.max_iter

        assert type(max_iter_) == int
        assert max_iter_ == max_iter

    def it_knows_its_tile_filename(self, request, tile_filename_fixture):
        (
            tile_size,
            n_tiles,
            level,
            seed,
            check_tissue,
            prefix,
            suffix,
            tile_coords,
            tiles_counter,
            expected_filename,
        ) = tile_filename_fixture
        random_tiler = RandomTiler(
            tile_size, n_tiles, level, seed, check_tissue, prefix, suffix
        )

        _filename = random_tiler._tile_filename(tile_coords, tiles_counter)

        assert type(_filename) == str
        assert _filename == expected_filename

    def it_can_generate_random_coordinates(self, request, tmpdir):
        tmp_path_ = tmpdir.mkdir("myslide")
        image = PILImageMock.DIMS_500X500_RGBA_COLOR_155_249_240
        image.save(os.path.join(tmp_path_, "mywsi.png"), "PNG")
        slide_path = os.path.join(tmp_path_, "mywsi.png")
        slide = Slide(slide_path, "processed")
        _box_mask_lvl = method_mock(request, RandomTiler, "box_mask_lvl")
        _box_mask_lvl.return_value = SparseArrayMock.ONES_500X500_BOOL
        _tile_size = property_mock(request, RandomTiler, "tile_size")
        _tile_size.return_value = (128, 128)
        _np_random_choice1 = function_mock(request, "numpy.random.choice")
        _np_random_choice1.return_value = 0
        _np_random_choice2 = function_mock(request, "numpy.random.choice")
        _np_random_choice2.return_value = 0
        _scale_coordinates = function_mock(request, "histolab.tiler.scale_coordinates")
        random_tiler = RandomTiler((128, 128), 10, 0)

        random_tiler._random_tile_coordinates(slide)

        _box_mask_lvl.assert_called_once_with(random_tiler, slide)
        _tile_size.assert_has_calls([call((128, 128))])
        _scale_coordinates.assert_called_once_with(
            reference_coords=CoordinatePair(x_ul=0, y_ul=0, x_br=128, y_br=128),
            reference_size=(500, 500),
            target_size=(500, 500),
        )

    @pytest.mark.parametrize(
        "check_tissue, expected_box",
        (
            (False, SparseArrayMock.ONES_500X500_BOOL),
            (True, SparseArrayMock.RANDOM_500X500_BOOL),
        ),
    )
    def it_knows_its_box_mask(self, request, tmpdir, check_tissue, expected_box):
        tmp_path_ = tmpdir.mkdir("myslide")
        image = PILImageMock.DIMS_500X500_RGBA_COLOR_155_249_240
        image.save(os.path.join(tmp_path_, "mywsi.png"), "PNG")
        slide_path = os.path.join(tmp_path_, "mywsi.png")
        slide = Slide(slide_path, "processed")
        _biggest_tissue_box_mask = property_mock(
            request, Slide, "biggest_tissue_box_mask"
        )
        if check_tissue:
            _biggest_tissue_box_mask.return_value = expected_box
        random_tiler = RandomTiler((128, 128), 10, 0, check_tissue=check_tissue)

        box_mask = random_tiler.box_mask(slide)

        if check_tissue:
            _biggest_tissue_box_mask.assert_called_once_with()
        assert type(box_mask) == sparse._coo.core.COO
        np.testing.assert_array_almost_equal(box_mask.todense(), expected_box.todense())

    # fixtures -------------------------------------------------------

    @pytest.fixture(
        params=(
            (
                (512, 512),
                10,
                3,
                7,
                True,
                "",
                ".png",
                CoordinatePair(0, 512, 0, 512),
                3,
                "tile_3_level3_0-512-0-512.png",
            ),
            (
                (512, 512),
                10,
                0,
                7,
                True,
                "folder/",
                ".png",
                CoordinatePair(4, 127, 4, 127),
                10,
                "folder/tile_10_level0_4-127-4-127.png",
            ),
        )
    )
    def tile_filename_fixture(self, request):
        (
            tile_size,
            n_tiles,
            level,
            seed,
            check_tissue,
            prefix,
            suffix,
            tile_coords,
            tiles_counter,
            expected_filename,
        ) = request.param
        return (
            tile_size,
            n_tiles,
            level,
            seed,
            check_tissue,
            prefix,
            suffix,
            tile_coords,
            tiles_counter,
            expected_filename,
        )
