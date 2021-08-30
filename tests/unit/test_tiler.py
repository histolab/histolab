import csv
import logging
import os
import re
from unittest.mock import call

import numpy as np
import pytest

from histolab.exceptions import LevelError, TileSizeOrCoordinatesError
from histolab.masks import BiggestTissueBoxMask
from histolab.scorer import RandomScorer
from histolab.slide import Slide
from histolab.tile import Tile
from histolab.tiler import GridTiler, RandomTiler, ScoreTiler, Tiler
from histolab.types import CP

from ..base import COMPLEX_MASK4
from ..unitutil import (
    ANY,
    PILIMG,
    NpArrayMock,
    base_test_slide,
    function_mock,
    initializer_mock,
    instance_mock,
    method_mock,
    property_mock,
)


class Describe_RandomTiler:
    @pytest.mark.parametrize("level, mpp", ((2, None), (-2, None), (2, 0.5)))
    def it_constructs_from_args(self, level, mpp, request):
        _init = initializer_mock(request, RandomTiler)

        random_tiler = RandomTiler(
            tile_size=(512, 512), n_tiles=10, level=level, mpp=mpp
        )

        _init.assert_called_once_with(
            random_tiler, tile_size=(512, 512), n_tiles=10, level=level, mpp=mpp
        )
        assert isinstance(random_tiler, RandomTiler)
        assert isinstance(random_tiler, Tiler)

    @pytest.mark.parametrize(
        "level, mpp, expected_level, expected_mpp",
        ((2, None, 2, None), (None, 0.5, 0, 0.5), (2, 0.5, 0, 0.5)),
    )
    def it_knows_when_mpp_supercedes_level(
        self, level, mpp, expected_level, expected_mpp
    ):

        random_tiler = RandomTiler((512, 512), 10, level=level, mpp=mpp)

        assert random_tiler.level == expected_level
        assert random_tiler.mpp == expected_mpp

    def but_it_has_wrong_tile_size_value(self):
        with pytest.raises(ValueError) as err:
            RandomTiler((512, -1), 10, 0)

        assert isinstance(err.value, ValueError)
        assert str(err.value) == "Tile size must be greater than 0 ((512, -1))"

    def or_it_has_not_available_level_value(self, tmpdir):
        slide, _ = base_test_slide(tmpdir, PILIMG.RGB_RANDOM_COLOR_500X500)
        random_tiler = RandomTiler(tile_size=(128, 128), n_tiles=10, level=3)
        binary_mask = BiggestTissueBoxMask()

        with pytest.raises(LevelError) as err:
            random_tiler.extract(slide, binary_mask)

        assert isinstance(err.value, LevelError)
        assert str(err.value) == "Level 3 not available. Number of available levels: 1"

    def or_it_has_wrong_max_iter(self):
        with pytest.raises(ValueError) as err:
            RandomTiler((512, 512), 10, 0, max_iter=3)

        assert isinstance(err.value, ValueError)
        assert (
            str(err.value)
            == "The maximum number of iterations (3) must be grater than or equal to "
            "the maximum number of tiles (10)."
        )

    def or_it_has_wrong_seed(self, tmpdir):
        slide, _ = base_test_slide(tmpdir, PILIMG.RGB_RANDOM_COLOR_500X500)
        random_tiler = RandomTiler((128, 128), 10, 0, seed=-1)
        binary_mask = BiggestTissueBoxMask()

        with pytest.raises(ValueError) as err:
            random_tiler.extract(slide, binary_mask)

        assert isinstance(err.value, ValueError)
        assert str(err.value) == "Seed must be between 0 and 2**32 - 1"

    @pytest.mark.parametrize("tile_size", ((512, 512), (128, 128), (10, 10)))
    def it_knows_its_tile_size(self, tile_size):
        random_tiler = RandomTiler(tile_size, 10, 0)

        tile_size_ = random_tiler.tile_size

        assert type(tile_size_) == tuple
        assert tile_size_ == tile_size

    @pytest.mark.parametrize("max_iter", (1000, 10, 3000))
    def it_knows_its_max_iter(self, max_iter):
        random_tiler = RandomTiler((128, 128), 10, 0, max_iter=max_iter)

        max_iter_ = random_tiler.max_iter

        assert type(max_iter_) == int
        assert max_iter_ == max_iter

    @pytest.mark.parametrize(
        "level, prefix, suffix, tile_coords, tiles_counter, expected_filename",
        (
            (3, "", ".png", CP(0, 0, 512, 512), 3, "tile_3_level3_0-0-512-512.png"),
            (
                0,
                "folder/",
                ".png",
                CP(4, 4, 127, 127),
                10,
                "folder/tile_10_level0_4-4-127-127.png",
            ),
        ),
    )
    def it_knows_its_tile_filename(
        self, level, prefix, suffix, tile_coords, tiles_counter, expected_filename
    ):
        random_tiler = RandomTiler((512, 512), 10, level, 7, True, 80, prefix, suffix)

        _filename = random_tiler._tile_filename(tile_coords, tiles_counter)

        assert type(_filename) == str
        assert _filename == expected_filename

    @pytest.mark.parametrize(
        "tile_size, expected_result", [((512, 512), False), ((200, 200), True)]
    )
    def it_knows_if_it_has_valid_tile_size(self, tmpdir, tile_size, expected_result):
        slide, _ = base_test_slide(tmpdir, PILIMG.RGBA_COLOR_500X500_155_249_240)
        random_tiler = RandomTiler(tile_size, 10, 0, 7)

        result = random_tiler._has_valid_tile_size(slide)

        assert type(result) == bool
        assert result == expected_result

    def it_can_generate_random_coordinates(self, request, tmpdir):
        slide, _ = base_test_slide(tmpdir, PILIMG.RGBA_COLOR_500X500_155_249_240)
        _box_mask_thumb = method_mock(request, BiggestTissueBoxMask, "__call__")
        _box_mask_thumb.return_value = NpArrayMock.ONES_500X500_BOOL
        _tile_size = property_mock(request, RandomTiler, "tile_size")
        _tile_size.return_value = (128, 128)
        _random_choice_true_mask2d = function_mock(
            request, "histolab.tiler.random_choice_true_mask2d"
        )
        _random_choice_true_mask2d.return_value = (0, 0)
        _scale_coordinates = function_mock(request, "histolab.tiler.scale_coordinates")
        random_tiler = RandomTiler((128, 128), 10, 0)
        binary_mask = BiggestTissueBoxMask()

        random_tiler._random_tile_coordinates(slide, binary_mask)

        _box_mask_thumb.assert_called_once_with(binary_mask, slide)
        _tile_size.assert_has_calls([call((128, 128))])
        _random_choice_true_mask2d.assert_called_once_with(
            NpArrayMock.ONES_500X500_BOOL
        )
        _scale_coordinates.assert_called_once_with(
            reference_coords=CP(x_ul=0, y_ul=0, x_br=128, y_br=128),
            reference_size=(500, 500),
            target_size=(500, 500),
        )

    @pytest.mark.parametrize(
        "tile1, tile2, has_enough_tissue, max_iter, expected_value",
        (
            (
                Tile(PILIMG.RGBA_COLOR_500X500_155_249_240, CP(0, 0, 10, 10)),
                Tile(PILIMG.RGBA_COLOR_500X500_155_249_240, CP(0, 0, 10, 10)),
                [True, True],
                10,
                2,
            ),
            (
                Tile(PILIMG.RGBA_COLOR_500X500_155_249_240, CP(0, 0, 10, 10)),
                Tile(PILIMG.RGBA_COLOR_500X500_155_249_240, CP(0, 0, 10, 10)),
                [True, False],
                2,
                1,
            ),
            (
                Tile(PILIMG.RGBA_COLOR_500X500_155_249_240, CP(0, 0, 10, 10)),
                Tile(PILIMG.RGBA_COLOR_500X500_155_249_240, CP(5900, 5900, 6000, 6000)),
                [True, True],
                2,
                2,
            ),
        ),
    )
    def it_can_generate_random_tiles_with_check_tissue(
        self,
        request,
        tmpdir,
        tile1,
        tile2,
        has_enough_tissue,
        max_iter,
        expected_value,
        _random_tile_coordinates,
    ):
        slide, _ = base_test_slide(tmpdir, PILIMG.RGBA_COLOR_500X500_155_249_240)
        _extract_tile = method_mock(request, Slide, "extract_tile")
        _has_enough_tissue = method_mock(request, Tile, "has_enough_tissue")
        _has_enough_tissue.side_effect = has_enough_tissue * (max_iter // 2)
        binary_mask = BiggestTissueBoxMask()
        tiles = [tile1, tile2]
        _extract_tile.side_effect = tiles * (max_iter // 2)
        random_tiler = RandomTiler(
            (10, 10),
            2,
            level=0,
            max_iter=max_iter,
            check_tissue=True,
            tissue_percent=60,
        )

        generated_tiles = list(random_tiler._tiles_generator(slide, binary_mask))

        _random_tile_coordinates.assert_called_with(random_tiler, slide, binary_mask)
        assert _has_enough_tissue.call_args_list == [call(tile1, 60), call(tile2, 60)]
        assert _random_tile_coordinates.call_count <= random_tiler.max_iter
        assert len(generated_tiles) == expected_value
        for i, tile in enumerate(generated_tiles):
            assert tile[0] == tiles[i]

    def it_can_generate_random_tiles_with_check_tissue_but_tiles_without_tissue(
        self,
        request,
        tmpdir,
        _random_tile_coordinates,
    ):
        slide, _ = base_test_slide(tmpdir, PILIMG.RGBA_COLOR_500X500_155_249_240)
        _extract_tile = method_mock(request, Slide, "extract_tile")
        _has_enough_tissue = method_mock(request, Tile, "has_enough_tissue")
        _has_enough_tissue.side_effect = [False, False] * 5
        binary_mask = BiggestTissueBoxMask()
        tiles = [
            Tile(PILIMG.RGBA_COLOR_500X500_155_249_240, CP(0, 0, 10, 10)),
            Tile(PILIMG.RGBA_COLOR_500X500_155_249_240, CP(0, 0, 10, 10)),
        ]
        _extract_tile.side_effect = tiles * 5
        random_tiler = RandomTiler(
            (10, 10),
            2,
            level=0,
            max_iter=10,
            check_tissue=True,
            tissue_percent=60,
        )

        generated_tiles = list(random_tiler._tiles_generator(slide, binary_mask))

        _random_tile_coordinates.assert_called_with(random_tiler, slide, binary_mask)
        assert (
            _has_enough_tissue.call_args_list
            == [
                call(tiles[0], 60),
                call(tiles[1], 60),
            ]
            * 5
        )
        assert _random_tile_coordinates.call_count <= random_tiler.max_iter
        assert len(generated_tiles) == 0
        for i, tile in enumerate(generated_tiles):
            assert tile[0] == tiles[i]

    @pytest.mark.parametrize(
        "tile1, tile2, has_enough_tissue, max_iter, expected_value",
        (
            (
                Tile(PILIMG.RGBA_COLOR_500X500_155_249_240, CP(0, 0, 10, 10)),
                Tile(PILIMG.RGBA_COLOR_500X500_155_249_240, CP(0, 0, 10, 10)),
                [True, True],
                10,
                2,
            ),
            (
                Tile(PILIMG.RGBA_COLOR_500X500_155_249_240, CP(0, 0, 10, 10)),
                Tile(PILIMG.RGBA_COLOR_500X500_155_249_240, CP(0, 0, 10, 10)),
                [False, False],
                10,
                2,
            ),
        ),
    )
    def it_can_generate_random_tiles_with_no_check_tissue(
        self,
        request,
        tmpdir,
        tile1,
        tile2,
        has_enough_tissue,
        max_iter,
        expected_value,
        _random_tile_coordinates,
    ):
        slide, _ = base_test_slide(tmpdir, PILIMG.RGBA_COLOR_500X500_155_249_240)
        _extract_tile = method_mock(request, Slide, "extract_tile")
        _has_enough_tissue = method_mock(request, Tile, "has_enough_tissue")
        _has_enough_tissue.side_effect = has_enough_tissue * (max_iter // 2)
        tiles = [tile1, tile2]
        binary_mask = BiggestTissueBoxMask()
        _extract_tile.side_effect = tiles * (max_iter // 2)
        random_tiler = RandomTiler(
            (10, 10),
            2,
            level=0,
            max_iter=max_iter,
            check_tissue=False,
        )

        generated_tiles = list(random_tiler._tiles_generator(slide, binary_mask))

        _random_tile_coordinates.assert_called_with(random_tiler, slide, binary_mask)
        _has_enough_tissue.assert_not_called()
        assert _random_tile_coordinates.call_count <= random_tiler.max_iter
        assert len(generated_tiles) == expected_value
        for i, tile in enumerate(generated_tiles):
            assert tile[0] == tiles[i]

    def it_can_generate_random_tiles_even_when_coords_are_not_valid(
        self, tmpdir, _random_tile_coordinates
    ):
        random_tiler = RandomTiler((10, 10), 1, level=0, max_iter=1, check_tissue=False)
        _random_tile_coordinates.side_effect = [CP(-1, -1, -1, -1), CP(0, 0, 10, 10)]
        slide, _ = base_test_slide(tmpdir, PILIMG.RGBA_COLOR_500X500_155_249_240)
        binary_mask = BiggestTissueBoxMask()

        generated_tiles = list(random_tiler._tiles_generator(slide, binary_mask))

        assert len(generated_tiles) == 1
        assert generated_tiles[0][1] == CP(0, 0, 10, 10)
        assert isinstance(generated_tiles[0][0], Tile)

    def it_can_extract_random_tiles(self, request, tmpdir, caplog):
        tmp_path_ = tmpdir.mkdir("myslide")
        image = PILIMG.RGBA_COLOR_500X500_155_249_240
        image.save(os.path.join(tmp_path_, "mywsi.png"), "PNG")
        slide_path = os.path.join(tmp_path_, "mywsi.png")
        slide = Slide(slide_path, os.path.join(tmp_path_, "processed"))
        _tiles_generator = method_mock(request, RandomTiler, "_tiles_generator")
        coords = CP(0, 0, 10, 10)
        tile = Tile(image, coords)
        _tiles_generator.return_value = [(tile, coords), (tile, coords)]
        _tile_filename = method_mock(request, RandomTiler, "_tile_filename")
        _tile_filename.side_effect = [
            f"tile_{i}_level2_0-0-10-10.png" for i in range(2)
        ]
        _has_valid_tile_size = method_mock(request, RandomTiler, "_has_valid_tile_size")
        _has_valid_tile_size.return_value = True
        random_tiler = RandomTiler((10, 10), n_tiles=2, level=0)
        binary_mask = BiggestTissueBoxMask()

        with caplog.at_level(logging.INFO):
            random_tiler.extract(slide, binary_mask)

        assert re.sub(r":+\d{3}", "", caplog.text).splitlines() == [
            "INFO     tiler:tiler.py \t Tile 0 saved: tile_0_level2_0-0-10-10.png",
            "INFO     tiler:tiler.py \t Tile 1 saved: tile_1_level2_0-0-10-10.png",
            "INFO     tiler:tiler.py 2 Random Tiles have been saved.",
        ]
        assert _tile_filename.call_args_list == [
            call(random_tiler, coords, 0),
            call(random_tiler, coords, 1),
        ]
        assert os.path.exists(
            os.path.join(tmp_path_, "processed", "tile_0_level2_0-0-10-10.png")
        )
        assert os.path.exists(
            os.path.join(tmp_path_, "processed", "tile_1_level2_0-0-10-10.png")
        )
        _has_valid_tile_size.assert_called_once_with(random_tiler, slide)

    @pytest.mark.parametrize(
        "image, size",
        [
            (PILIMG.RGBA_COLOR_50X50_155_0_0, (50, 50)),
            (PILIMG.RGBA_COLOR_49X51_155_0_0, (49, 51)),
        ],
    )
    def but_it_raises_tilesizeerror_if_tilesize_larger_than_slidesize(
        self, request, tmpdir, image, size
    ):
        tmp_path_ = tmpdir.mkdir("myslide")
        image.save(os.path.join(tmp_path_, "mywsi.png"), "PNG")
        slide_path = os.path.join(tmp_path_, "mywsi.png")
        slide = Slide(slide_path, os.path.join(tmp_path_, "processed"))
        _has_valid_tile_size = method_mock(request, RandomTiler, "_has_valid_tile_size")
        _has_valid_tile_size.return_value = False
        random_tiler = RandomTiler((50, 52), n_tiles=10, level=0)
        binary_mask = BiggestTissueBoxMask()

        with pytest.raises(TileSizeOrCoordinatesError) as err:
            random_tiler.extract(slide, binary_mask)

        assert isinstance(err.value, TileSizeOrCoordinatesError)
        assert (
            str(err.value)
            == f"Tile size (50, 52) is larger than slide size {size} at level 0"
        )
        _has_valid_tile_size.assert_called_once_with(random_tiler, slide)

    # fixture components ---------------------------------------------

    @pytest.fixture
    def _random_tile_coordinates(self, request):
        return method_mock(request, RandomTiler, "_random_tile_coordinates")


class Describe_GridTiler:
    @pytest.mark.parametrize("level, mpp", ((2, None), (-2, None), (2, 0.5)))
    def it_constructs_from_args(self, level, mpp, request):
        _init = initializer_mock(request, GridTiler)

        grid_tiler = GridTiler((512, 512), level, True, 80, 0, "", ".png", mpp=mpp)

        _init.assert_called_once_with(
            ANY, (512, 512), level, True, 80, 0, "", ".png", mpp=mpp
        )
        assert isinstance(grid_tiler, GridTiler)
        assert isinstance(grid_tiler, Tiler)

    @pytest.mark.parametrize(
        "level, mpp, expected_level, expected_mpp",
        ((2, None, 2, None), (None, 0.5, 0, 0.5), (2, 0.5, 0, 0.5)),
    )
    def it_knows_when_mpp_supercedes_level(
        self, level, mpp, expected_level, expected_mpp
    ):

        tiler = GridTiler((512, 512), level=level, mpp=mpp)

        assert tiler.level == expected_level
        assert tiler.mpp == expected_mpp

    def but_it_has_wrong_tile_size_value(self):
        with pytest.raises(ValueError) as err:
            GridTiler((512, -1))

        assert isinstance(err.value, ValueError)
        assert str(err.value) == "Tile size must be greater than 0 ((512, -1))"

    def or_it_has_not_available_level_value(self, tmpdir):
        slide, _ = base_test_slide(tmpdir, PILIMG.RGB_RANDOM_COLOR_500X500)
        binary_mask = BiggestTissueBoxMask()
        grid_tiler = GridTiler((128, 128), 3)

        with pytest.raises(LevelError) as err:
            grid_tiler.extract(slide, binary_mask)

        assert isinstance(err.value, LevelError)
        assert str(err.value) == "Level 3 not available. Number of available levels: 1"

    @pytest.mark.parametrize("tile_size", ((512, 512), (128, 128), (10, 10)))
    def it_knows_its_tile_size(self, tile_size):
        grid_tiler = GridTiler(tile_size, 10, True, 0)

        tile_size_ = grid_tiler.tile_size

        assert type(tile_size_) == tuple
        assert tile_size_ == tile_size

    @pytest.mark.parametrize(
        "level, pixel_overlap, prefix, tile_coords, tiles_counter, expected_filename",
        (
            (3, 0, "", CP(0, 0, 512, 512), 3, "tile_3_level3_0-0-512-512.png"),
            (
                0,
                0,
                "folder/",
                CP(4, 4, 127, 127),
                10,
                "folder/tile_10_level0_4-4-127-127.png",
            ),
        ),
    )
    def it_knows_its_tile_filename(
        self,
        level,
        pixel_overlap,
        prefix,
        tile_coords,
        tiles_counter,
        expected_filename,
    ):
        grid_tiler = GridTiler(
            (512, 512), level, True, 80.0, pixel_overlap, prefix, ".png"
        )

        _filename = grid_tiler._tile_filename(tile_coords, tiles_counter)

        assert type(_filename) == str
        assert _filename == expected_filename

    @pytest.mark.parametrize(
        "tile_size, expected_result", [((512, 512), False), ((200, 200), True)]
    )
    def it_knows_if_it_has_valid_tile_size(self, tmpdir, tile_size, expected_result):
        slide, _ = base_test_slide(tmpdir, PILIMG.RGBA_COLOR_500X500_155_249_240)
        grid_tiler = GridTiler(tile_size, 0, True)

        result = grid_tiler._has_valid_tile_size(slide)

        assert type(result) == bool
        assert result == expected_result

    @pytest.mark.parametrize(
        "bbox_coordinates, pixel_overlap, expected_n_tiles_row",
        (
            (CP(x_ul=0, y_ul=0, x_br=6060, y_br=1917), 0, 11),
            (CP(x_ul=0, y_ul=0, x_br=1921, y_br=2187), 0, 3),
            (CP(x_ul=0, y_ul=0, x_br=1921, y_br=2187), 128, 5),
            (CP(x_ul=0, y_ul=0, x_br=1921, y_br=2187), -128, 3),
        ),
    )
    def it_can_calculate_n_tiles_row(
        self, bbox_coordinates, pixel_overlap, expected_n_tiles_row
    ):
        grid_tiler = GridTiler((512, 512), 2, True, 80.0, pixel_overlap)

        n_tiles_row = grid_tiler._n_tiles_row(bbox_coordinates)

        assert type(n_tiles_row) == int
        assert n_tiles_row == expected_n_tiles_row

    @pytest.mark.parametrize(
        "bbox_coordinates, pixel_overlap, expected_n_tiles_column",
        (
            (CP(x_ul=0, y_ul=0, x_br=6060, y_br=1917), 0, 3),
            (CP(x_ul=0, y_ul=0, x_br=6060, y_br=1917), -1, 3),
            (CP(x_ul=0, y_ul=0, x_br=1921, y_br=2187), 0, 4),
            (CP(x_ul=0, y_ul=0, x_br=1921, y_br=2187), 128, 5),
        ),
    )
    def it_can_calculate_n_tiles_column(
        self, bbox_coordinates, pixel_overlap, expected_n_tiles_column
    ):
        grid_tiler = GridTiler((512, 512), 2, True, 80, pixel_overlap)

        n_tiles_column = grid_tiler._n_tiles_column(bbox_coordinates)

        assert type(n_tiles_column) == int
        assert n_tiles_column == expected_n_tiles_column

    @pytest.mark.parametrize(
        "tile1, tile2, has_enough_tissue, expected_n_tiles",
        (
            (
                Tile(PILIMG.RGBA_COLOR_500X500_155_249_240, CP(0, 0, 10, 10)),
                Tile(PILIMG.RGBA_COLOR_500X500_155_249_240, CP(0, 0, 10, 10)),
                [True, True],
                2,
            ),
            (
                Tile(PILIMG.RGBA_COLOR_500X500_155_249_240, CP(0, 0, 10, 10)),
                Tile(PILIMG.RGBA_COLOR_500X500_155_249_240, CP(0, 0, 10, 10)),
                [False, False],
                0,
            ),
            (
                Tile(PILIMG.RGBA_COLOR_500X500_155_249_240, CP(0, 0, 10, 10)),
                Tile(PILIMG.RGBA_COLOR_500X500_155_249_240, CP(0, 0, 10, 10)),
                [True, False],
                1,
            ),
        ),
    )
    def it_can_generate_grid_tiles_with_check_tissue(
        self,
        request,
        tmpdir,
        tile1,
        tile2,
        has_enough_tissue,
        expected_n_tiles,
    ):
        slide, _ = base_test_slide(tmpdir, PILIMG.RGBA_COLOR_500X500_155_249_240)
        _extract_tile = method_mock(request, Slide, "extract_tile")
        _has_enough_tissue = method_mock(request, Tile, "has_enough_tissue")
        _has_enough_tissue.side_effect = has_enough_tissue
        _grid_coordinates_generator = method_mock(
            request, GridTiler, "_grid_coordinates_generator"
        )
        _grid_coordinates_generator.return_value = [CP(0, 0, 10, 10), CP(0, 0, 10, 10)]
        _extract_tile.side_effect = [tile1, tile2]
        grid_tiler = GridTiler((10, 10), level=0, check_tissue=True, tissue_percent=60)
        tiles = [tile1, tile2]
        binary_mask = BiggestTissueBoxMask()

        generated_tiles = list(grid_tiler._tiles_generator(slide, binary_mask))

        _grid_coordinates_generator.assert_called_once_with(
            grid_tiler, slide, binary_mask
        )

        assert _extract_tile.call_args_list == (
            [
                call(slide, CP(0, 0, 10, 10), tile_size=(10, 10), level=0, mpp=None),
                call(slide, CP(0, 0, 10, 10), tile_size=(10, 10), level=0, mpp=None),
            ]
        )
        assert _has_enough_tissue.call_args_list == [call(tile1, 60), call(tile2, 60)]
        assert len(generated_tiles) == expected_n_tiles
        for i, tile in enumerate(generated_tiles):
            assert tile[0] == tiles[i]

    @pytest.mark.parametrize(
        "tile1, tile2, has_enough_tissue, expected_n_tiles",
        (
            (
                Tile(PILIMG.RGBA_COLOR_500X500_155_249_240, CP(0, 0, 10, 10)),
                Tile(PILIMG.RGBA_COLOR_500X500_155_249_240, CP(0, 0, 10, 10)),
                [True, True],
                2,
            ),
            (
                Tile(PILIMG.RGBA_COLOR_500X500_155_249_240, CP(0, 0, 10, 10)),
                Tile(PILIMG.RGBA_COLOR_500X500_155_249_240, CP(0, 0, 10, 10)),
                [False, False],
                2,
            ),
        ),
    )
    def it_can_generate_grid_tiles_with_no_check_tissue(
        self,
        request,
        tmpdir,
        tile1,
        tile2,
        has_enough_tissue,
        expected_n_tiles,
    ):
        slide, _ = base_test_slide(tmpdir, PILIMG.RGBA_COLOR_500X500_155_249_240)
        _extract_tile = method_mock(request, Slide, "extract_tile")
        _has_enough_tissue = method_mock(request, Tile, "has_enough_tissue")
        _has_enough_tissue.side_effect = has_enough_tissue
        _grid_coordinates_generator = method_mock(
            request, GridTiler, "_grid_coordinates_generator"
        )
        _grid_coordinates_generator.return_value = [CP(0, 0, 10, 10), CP(0, 0, 10, 10)]
        _extract_tile.side_effect = [tile1, tile2]
        grid_tiler = GridTiler((10, 10), level=0, check_tissue=False)
        tiles = [tile1, tile2]
        binary_mask = BiggestTissueBoxMask()

        generated_tiles = list(grid_tiler._tiles_generator(slide, binary_mask))

        _grid_coordinates_generator.assert_called_once_with(
            grid_tiler, slide, binary_mask
        )

        assert _extract_tile.call_args_list == (
            [
                call(slide, CP(0, 0, 10, 10), tile_size=(10, 10), level=0, mpp=None),
                call(slide, CP(0, 0, 10, 10), tile_size=(10, 10), level=0, mpp=None),
            ]
        )
        _has_enough_tissue.assert_not_called()
        assert len(generated_tiles) == expected_n_tiles
        for i, tile in enumerate(generated_tiles):
            assert tile[0] == tiles[i]

    def but_with_wrong_coordinates(self, request, tmpdir):
        slide, _ = base_test_slide(tmpdir, PILIMG.RGBA_COLOR_500X500_155_249_240)
        _has_enough_tissue = method_mock(request, Tile, "has_enough_tissue")
        _has_enough_tissue.return_value = False
        _grid_coordinates_generator = method_mock(
            request, GridTiler, "_grid_coordinates_generator"
        )
        coords1 = CP(600, 600, 610, 610)
        coords2 = CP(0, 0, 10, 10)
        _grid_coordinates_generator.return_value = [coords1, coords2]
        grid_tiler = GridTiler((10, 10), level=0, check_tissue=False)
        binary_mask = BiggestTissueBoxMask()

        generated_tiles = list(grid_tiler._tiles_generator(slide, binary_mask))

        _grid_coordinates_generator.assert_called_once_with(
            grid_tiler, slide, binary_mask
        )
        assert len(generated_tiles) == 1
        # generated_tiles[0][0] is a Tile object but we don't know what object it is
        # because Slide.extract_tile is not mocked (for the exception to happen inside)
        assert isinstance(generated_tiles[0][0], Tile)
        assert generated_tiles[0][1] == coords2

    def and_it_does_not_raise_error_with_wrong_coordinates(self, request, tmpdir):
        slide, _ = base_test_slide(tmpdir, PILIMG.RGBA_COLOR_500X500_155_249_240)
        coords = CP(5800, 5800, 6000, 6000)
        _grid_coordinates_generator = method_mock(
            request, GridTiler, "_grid_coordinates_generator"
        )
        _grid_coordinates_generator.return_value = [coords]
        grid_tiler = GridTiler((10, 10))
        binary_mask = BiggestTissueBoxMask()
        generated_tiles = list(grid_tiler._tiles_generator(slide, binary_mask))

        assert len(generated_tiles) == 0
        _grid_coordinates_generator.assert_called_once_with(
            grid_tiler, slide, binary_mask
        )

    def it_can_extract_grid_tiles(self, request, tmpdir, caplog):
        tmp_path_ = tmpdir.mkdir("myslide")
        image = PILIMG.RGBA_COLOR_500X500_155_249_240
        image.save(os.path.join(tmp_path_, "mywsi.png"), "PNG")
        slide_path = os.path.join(tmp_path_, "mywsi.png")
        slide = Slide(slide_path, os.path.join(tmp_path_, "processed"))
        _tiles_generator = method_mock(request, GridTiler, "_tiles_generator")
        coords = CP(0, 0, 10, 10)
        tile = Tile(image, coords)
        _tiles_generator.return_value = [(tile, coords), (tile, coords)]
        _tile_filename = method_mock(request, GridTiler, "_tile_filename")
        _tile_filename.side_effect = [
            os.path.join(tmp_path_, "processed", f"tile_{i}_level2_0-0-10-10.png")
            for i in range(2)
        ]
        _has_valid_tile_size = method_mock(request, GridTiler, "_has_valid_tile_size")
        _has_valid_tile_size.return_value = True
        grid_tiler = GridTiler((10, 10), level=0)
        binary_mask = BiggestTissueBoxMask()

        with caplog.at_level(logging.ERROR):
            grid_tiler.extract(slide, binary_mask)

        assert caplog.text == ""
        assert _tile_filename.call_args_list == [
            call(grid_tiler, coords, 0),
            call(grid_tiler, coords, 1),
        ]
        assert os.path.exists(
            os.path.join(tmp_path_, "processed", "tile_0_level2_0-0-10-10.png")
        )
        assert os.path.exists(
            os.path.join(tmp_path_, "processed", "tile_1_level2_0-0-10-10.png")
        )
        _has_valid_tile_size.assert_called_once_with(grid_tiler, slide)
        _tiles_generator.assert_called_once_with(grid_tiler, slide, binary_mask)
        assert _tile_filename.call_args_list == [
            call(grid_tiler, coords, 0),
            call(grid_tiler, coords, 1),
        ]

    @pytest.mark.parametrize(
        "image, size",
        [
            (PILIMG.RGBA_COLOR_50X50_155_0_0, (50, 50)),
            (PILIMG.RGBA_COLOR_49X51_155_0_0, (49, 51)),
        ],
    )
    def but_it_raises_tilesizeerror_if_tilesize_larger_than_slidesize(
        self, request, tmpdir, image, size
    ):
        tmp_path_ = tmpdir.mkdir("myslide")
        image.save(os.path.join(tmp_path_, "mywsi.png"), "PNG")
        slide_path = os.path.join(tmp_path_, "mywsi.png")
        slide = Slide(slide_path, os.path.join(tmp_path_, "processed"))
        _has_valid_tile_size = method_mock(request, GridTiler, "_has_valid_tile_size")
        _has_valid_tile_size.return_value = False
        grid_tiler = GridTiler((50, 52), level=0)
        binary_mask = BiggestTissueBoxMask()

        with pytest.raises(TileSizeOrCoordinatesError) as err:
            grid_tiler.extract(slide, binary_mask)

        assert isinstance(err.value, TileSizeOrCoordinatesError)
        assert (
            str(err.value)
            == f"Tile size (50, 52) is larger than slide size {size} at level 0"
        )
        _has_valid_tile_size.assert_called_once_with(grid_tiler, slide)

    @pytest.mark.parametrize(
        "tile_coords, expected_result",
        [
            (CP(2, 6, 6, 6), False),  # bad edge coordinates from np.ceil/floor
            (CP(0, 0, 2, 2), False),  # completely outside of region
            (CP(0, 0, 8, 8), False),  # only 205
            (CP(2, 3, 6, 6), True),  # 85% in
            (CP(3, 3, 5, 5), True),  # 100% in
        ],
    )
    def it_knows_whether_coordinates_are_within_extraction_mask(
        self, tile_coords, expected_result
    ):
        grid_tiler = GridTiler((2, 2), level=0)  # tile size doens't matter here
        mask = COMPLEX_MASK4

        coords_within_extraction_mask = (
            grid_tiler._are_coordinates_within_extraction_mask(tile_coords, mask)
        )

        assert type(coords_within_extraction_mask) == bool
        assert coords_within_extraction_mask == expected_result

    @pytest.mark.parametrize(
        "outline, n_tiles, error_msg",
        (
            (
                ["yellow", "yellow"],
                3,
                "There should be as many outlines as there are tiles!",
            ),
            (
                ["yellow", "yellow"],
                1,
                "There should be as many outlines as there are tiles!",
            ),
            (
                0.5,
                2,
                "The parameter ``outline`` should be of type: "
                "str, Iterable[str], or Iterable[List[int]]",
            ),
        ),
    )
    def it_throws_error_with_invalid_tile_outline(
        self, request, tmpdir, outline, n_tiles, error_msg
    ):
        slide, _ = base_test_slide(tmpdir, PILIMG.RGBA_COLOR_500X500_155_249_240)
        coords = CP(0, 0, 10, 10)
        tile = Tile(PILIMG.RGBA_COLOR_500X500_155_249_240, coords)
        mock_tiles = [(tile, coords)] * n_tiles
        _tiles_generator = method_mock(request, GridTiler, "_tiles_generator")
        _tiles_generator.return_value = mock_tiles
        grid_tiler = GridTiler((10, 10))

        with pytest.raises(ValueError) as err:
            grid_tiler.locate_tiles(slide=slide, outline=outline)

        assert str(err.value) == error_msg

    @pytest.mark.parametrize(
        "pass_tiles, outline, expected_topleft",
        (
            (False, "red", (255, 255, 155, 155)),
            (True, ["red", "red"], (255, 255, 155, 155)),
            (True, [(255, 0, 0), (255, 0, 0)], (255, 255, 155, 155)),
        ),
    )
    def it_can_locate_tiles(
        self, request, tmpdir, pass_tiles, outline, expected_topleft
    ):
        slide, _ = base_test_slide(tmpdir, PILIMG.RGBA_COLOR_500X500_155_249_240)
        coords = CP(0, 0, 10, 10)
        tile = Tile(PILIMG.RGBA_COLOR_500X500_155_249_240, coords)
        mock_tiles = [(tile, coords), (tile, coords)]
        method_mock(request, GridTiler, "_tiles_generator", return_value=mock_tiles)
        grid_tiler = GridTiler((10, 10), level=0, check_tissue=True, tissue_percent=60)
        tiles = None if not pass_tiles else mock_tiles
        img = grid_tiler.locate_tiles(slide=slide, outline=outline, tiles=tiles)
        img = np.uint8(img)

        assert img.shape == (15, 15, 4)
        assert tuple(img[:4, 0, 0]) == expected_topleft


class Describe_ScoreTiler:
    def it_constructs_from_args(self, request):
        _init = initializer_mock(request, ScoreTiler)
        rs = RandomScorer()
        score_tiler = ScoreTiler(
            rs, (512, 512), 4, 2, True, 80, 0, "", ".png", mpp=None
        )

        _init.assert_called_once_with(
            ANY, rs, (512, 512), 4, 2, True, 80, 0, "", ".png", mpp=None
        )

        assert isinstance(score_tiler, ScoreTiler)
        assert isinstance(score_tiler, GridTiler)
        assert isinstance(score_tiler, Tiler)

    def it_knows_its_scorer(self):
        random_scorer = RandomScorer()
        score_tiler = ScoreTiler(random_scorer, (512, 512), 4, 0)

        scorer_ = score_tiler.scorer

        assert callable(scorer_)
        assert isinstance(scorer_, RandomScorer)

    def it_knows_its_n_tiles(self):
        n_tiles = 4
        score_tiler = ScoreTiler(RandomScorer(), (512, 512), n_tiles, 0)

        n_tiles_ = score_tiler.n_tiles

        assert type(n_tiles_) == int
        assert n_tiles_ == n_tiles

    @pytest.mark.parametrize(
        "tile_size, expected_result", [((512, 512), False), ((200, 200), True)]
    )
    def it_knows_if_it_has_valid_tile_size(self, tmpdir, tile_size, expected_result):
        slide, _ = base_test_slide(tmpdir, PILIMG.RGBA_COLOR_500X500_155_249_240)
        score_tiler = ScoreTiler(RandomScorer(), tile_size, 2, 0)

        result = score_tiler._has_valid_tile_size(slide)

        assert type(result) == bool
        assert result == expected_result

    def it_can_calculate_scores(self, request):
        slide = instance_mock(request, Slide)
        coords = CP(0, 0, 10, 10)
        image = PILIMG.RGB_RANDOM_COLOR_500X500
        tile = Tile(image, coords)
        _tiles_generator = method_mock(request, GridTiler, "_tiles_generator")
        # it needs to be a generator
        _tiles_generator.return_value = ((tile, coords) for i in range(3))
        _scorer = instance_mock(request, RandomScorer)
        _scorer.side_effect = [0.5, 0.7]
        score_tiler = ScoreTiler(_scorer, (10, 10), 2, 0)
        binary_mask = BiggestTissueBoxMask()

        scores = score_tiler._scores(slide, binary_mask)

        assert _tiles_generator.call_args_list == [
            call(score_tiler, slide, binary_mask),
            call(score_tiler, slide, binary_mask),
        ]
        assert _scorer.call_args_list == [call(tile), call(tile)]
        assert type(scores) == list
        assert type(scores[0]) == tuple
        assert type(scores[0][0]) == float
        assert type(scores[0][1]) == CP
        assert scores == [(0.5, coords), (0.7, coords)]

    def but_it_raises_runtimeerror_if_no_tiles_are_extracted(self, request):
        slide = instance_mock(request, Slide)
        _tiles_generator = method_mock(request, GridTiler, "_tiles_generator")
        # it needs to be an empty generator
        _tiles_generator.return_value = (n for n in [])
        score_tiler = ScoreTiler(None, (10, 10), 2, 0)
        binary_mask = BiggestTissueBoxMask()

        with pytest.raises(RuntimeError) as err:
            score_tiler._scores(slide, binary_mask)

        _tiles_generator.assert_called_once_with(score_tiler, slide, binary_mask)
        assert isinstance(err.value, RuntimeError)
        assert (
            str(err.value)
            == "No tiles have been generated. This could happen if `check_tissue=True`"
        )

    def or_it_raises_levelerror_if_has_not_available_level_value(self, tmpdir):
        slide, _ = base_test_slide(tmpdir, PILIMG.RGB_RANDOM_COLOR_500X500)
        score_tiler = ScoreTiler(None, (10, 10), 2, 3)
        binary_mask = BiggestTissueBoxMask()

        with pytest.raises(LevelError) as err:
            score_tiler.extract(slide, binary_mask)

        assert isinstance(err.value, LevelError)
        assert str(err.value) == "Level 3 not available. Number of available levels: 1"

    def it_can_scale_scores(self):
        coords = [CP(0, 0, 10 * i, 10) for i in range(3)]
        scores = [0.3, 0.4, 0.7]
        scores_ = list(zip(scores, coords))
        score_tiler = ScoreTiler(None, (10, 10), 2, 0)
        expected_scaled_coords = list(zip([0.0, 0.2500000000000001, 1.0], coords))

        scaled_scores = score_tiler._scale_scores(scores_)

        for (score, coords_), (expected_score, expected_coords) in zip(
            scaled_scores, expected_scaled_coords
        ):
            assert round(score, 5) == round(expected_score, 5)
            assert coords_ == expected_coords

    @pytest.mark.parametrize(
        "n_tiles, expected_value",
        (
            (
                0,
                (
                    [
                        (0.8, CP(0, 0, 10, 10)),
                        (0.7, CP(0, 0, 10, 10)),
                        (0.5, CP(0, 0, 10, 10)),
                        (0.2, CP(0, 0, 10, 10)),
                        (0.1, CP(0, 0, 10, 10)),
                    ],
                    [
                        (1.0, CP(0, 0, 10, 10)),
                        (0.857142857142857, CP(0, 0, 10, 10)),
                        (0.5714285714285714, CP(0, 0, 10, 10)),
                        (0.14285714285714285, CP(0, 0, 10, 10)),
                        (0.0, CP(0, 0, 10, 10)),
                    ],
                ),
            ),
            (
                2,
                (
                    [(0.8, CP(0, 0, 10, 10)), (0.7, CP(0, 0, 10, 10))],
                    [(1.0, CP(0, 0, 10, 10)), (0.857142857142857, CP(0, 0, 10, 10))],
                ),
            ),
            (
                3,
                (
                    [
                        (0.8, CP(0, 0, 10, 10)),
                        (0.7, CP(0, 0, 10, 10)),
                        (0.5, CP(0, 0, 10, 10)),
                    ],
                    [
                        (1.0, CP(0, 0, 10, 10)),
                        (0.857142857142857, CP(0, 0, 10, 10)),
                        (0.5714285714285714, CP(0, 0, 10, 10)),
                    ],
                ),
            ),
        ),
    )
    def it_can_calculate_highest_score_tiles(self, request, n_tiles, expected_value):
        slide = instance_mock(request, Slide)
        _scores = method_mock(request, ScoreTiler, "_scores")
        coords = CP(0, 0, 10, 10)
        _scores.return_value = [
            (0.7, coords),
            (0.5, coords),
            (0.2, coords),
            (0.8, coords),
            (0.1, coords),
        ]
        _scorer = instance_mock(request, RandomScorer)
        score_tiler = ScoreTiler(_scorer, (10, 10), n_tiles, 0)
        binary_mask = BiggestTissueBoxMask()

        highest_score_tiles = score_tiler._tiles_generator(slide, binary_mask)

        _scores.assert_called_once_with(score_tiler, slide, binary_mask)
        assert highest_score_tiles == expected_value

    def but_it_raises_error_with_negative_n_tiles_value(self, request, tmpdir):
        tmp_path_ = tmpdir.mkdir("myslide")
        image = PILIMG.RGBA_COLOR_500X500_155_249_240
        image.save(os.path.join(tmp_path_, "mywsi.png"), "PNG")
        slide_path = os.path.join(tmp_path_, "mywsi.png")
        slide = Slide(slide_path, os.path.join(tmp_path_, "processed"))
        _scores = method_mock(request, ScoreTiler, "_scores")
        coords = CP(0, 0, 10, 10)
        _scores.return_value = [
            (0.7, coords),
            (0.5, coords),
            (0.2, coords),
            (0.8, coords),
            (0.1, coords),
        ]
        _scorer = instance_mock(request, RandomScorer)
        score_tiler = ScoreTiler(_scorer, (10, 10), -1, 0)
        binary_mask = BiggestTissueBoxMask()

        with pytest.raises(ValueError) as err:
            score_tiler.extract(slide, binary_mask)

        assert isinstance(err.value, ValueError)
        assert str(err.value) == "'n_tiles' cannot be negative (-1)"
        _scores.assert_called_once_with(score_tiler, slide, binary_mask)

    def it_can_extract_score_tiles(self, request, tmpdir):
        _extract_tile = method_mock(request, Slide, "extract_tile")
        tmp_path_ = tmpdir.mkdir("myslide")
        image = PILIMG.RGBA_COLOR_500X500_155_249_240
        image.save(os.path.join(tmp_path_, "mywsi.png"), "PNG")
        slide_path = os.path.join(tmp_path_, "mywsi.png")
        slide = Slide(slide_path, os.path.join(tmp_path_, "processed"))
        coords = CP(0, 0, 10, 10)
        _tiles_generator = method_mock(
            request,
            ScoreTiler,
            "_tiles_generator",
        )
        tile = Tile(image, coords)
        _extract_tile.return_value = tile
        _tiles_generator.return_value = (
            [(0.8, coords), (0.7, coords)],
            [(0.8, coords), (0.7, coords)],
        )
        _tile_filename = method_mock(request, GridTiler, "_tile_filename")
        _tile_filename.side_effect = [
            f"tile_{i}_level2_0-0-10-10.png" for i in range(2)
        ]
        _save_report = method_mock(request, ScoreTiler, "_save_report")
        random_scorer = RandomScorer()
        _has_valid_tile_size = method_mock(request, ScoreTiler, "_has_valid_tile_size")
        _has_valid_tile_size.return_value = True
        score_tiler = ScoreTiler(random_scorer, (10, 10), 2, 0)
        binary_mask = BiggestTissueBoxMask()

        score_tiler.extract(slide, binary_mask)

        assert _extract_tile.call_args_list == [
            call(slide, coords, tile_size=(10, 10), level=0, mpp=None),
            call(slide, coords, tile_size=(10, 10), level=0, mpp=None),
        ]
        _tiles_generator.assert_called_with(score_tiler, slide, binary_mask)
        assert _tile_filename.call_args_list == [
            call(score_tiler, coords, 0),
            call(score_tiler, coords, 1),
        ]
        assert os.path.exists(
            os.path.join(tmp_path_, "processed", "tile_0_level2_0-0-10-10.png")
        )
        assert os.path.exists(
            os.path.join(tmp_path_, "processed", "tile_1_level2_0-0-10-10.png")
        )
        _save_report.assert_not_called()
        _has_valid_tile_size.assert_called_once_with(score_tiler, slide)

    @pytest.mark.parametrize(
        "image, size",
        [
            (PILIMG.RGBA_COLOR_50X50_155_0_0, (50, 50)),
            (PILIMG.RGBA_COLOR_49X51_155_0_0, (49, 51)),
        ],
    )
    def but_it_raises_tilesizeerror_if_tilesize_larger_than_slidesize(
        self, request, tmpdir, image, size
    ):
        tmp_path_ = tmpdir.mkdir("myslide")
        image.save(os.path.join(tmp_path_, "mywsi.png"), "PNG")
        slide_path = os.path.join(tmp_path_, "mywsi.png")
        slide = Slide(slide_path, os.path.join(tmp_path_, "processed"))
        _has_valid_tile_size = method_mock(request, ScoreTiler, "_has_valid_tile_size")
        _has_valid_tile_size.return_value = False
        score_tiler = ScoreTiler(None, (50, 52), 2, 0)
        binary_mask = BiggestTissueBoxMask()

        with pytest.raises(TileSizeOrCoordinatesError) as err:
            score_tiler.extract(slide, binary_mask)

        assert isinstance(err.value, TileSizeOrCoordinatesError)
        assert (
            str(err.value)
            == f"Tile size (50, 52) is larger than slide size {size} at level 0"
        )
        _has_valid_tile_size.assert_called_once_with(score_tiler, slide)

    def it_can_save_report(self, request, tmpdir):
        tmp_path_ = tmpdir.mkdir("path")
        coords = CP(0, 0, 10, 10)
        highest_score_tiles = [(0.8, coords), (0.7, coords)]
        highest_scaled_score_tiles = [(0.1, coords), (0.0, coords)]
        filenames = ["tile0.png", "tile1.png"]
        random_scorer_ = instance_mock(request, RandomScorer)
        score_tiler = ScoreTiler(random_scorer_, (10, 10), 2, 2)
        report_ = [
            "filename,score,scaled_score",
            "tile0.png,0.8,0.1",
            "tile1.png,0.7,0.0",
        ]

        score_tiler._save_report(
            os.path.join(tmp_path_, "report.csv"),
            highest_score_tiles,
            highest_scaled_score_tiles,
            filenames,
        )

        assert os.path.exists(os.path.join(tmp_path_, "report.csv"))
        with open(os.path.join(tmp_path_, "report.csv"), newline="") as f:
            reader = csv.reader(f)
            report = [",".join(row) for row in reader]
            assert report == report_

    def it_can_extract_score_tiles_and_save_report(self, request, tmpdir):
        _extract_tile = method_mock(request, Slide, "extract_tile")
        tmp_path_ = tmpdir.mkdir("myslide")
        image = PILIMG.RGBA_COLOR_500X500_155_249_240
        image.save(os.path.join(tmp_path_, "mywsi.png"), "PNG")
        slide_path = os.path.join(tmp_path_, "mywsi.png")
        slide = Slide(slide_path, os.path.join(tmp_path_, "processed"))
        coords = CP(0, 0, 10, 10)
        _tiles_generator = method_mock(
            request,
            ScoreTiler,
            "_tiles_generator",
        )
        tile = Tile(image, coords)
        _extract_tile.return_value = tile
        _tiles_generator.return_value = (
            [(0.8, coords), (0.7, coords)],
            [(0.8, coords), (0.7, coords)],
        )
        _tile_filename = method_mock(request, GridTiler, "_tile_filename")
        _tile_filename.side_effect = [
            f"tile_{i}_level2_0-0-10-10.png" for i in range(2)
        ]
        _save_report = method_mock(request, ScoreTiler, "_save_report", autospec=False)
        random_scorer = RandomScorer()
        score_tiler = ScoreTiler(random_scorer, (10, 10), 2, 0)
        binary_mask = BiggestTissueBoxMask()

        score_tiler.extract(slide, binary_mask, "report.csv")

        assert _extract_tile.call_args_list == [
            call(slide, coords, tile_size=(10, 10), level=0, mpp=None),
            call(slide, coords, tile_size=(10, 10), level=0, mpp=None),
        ]
        _tiles_generator.assert_called_with(score_tiler, slide, binary_mask)
        assert _tile_filename.call_args_list == [
            call(score_tiler, coords, 0),
            call(score_tiler, coords, 1),
        ]
        assert os.path.exists(
            os.path.join(tmp_path_, "processed", "tile_0_level2_0-0-10-10.png")
        )
        assert os.path.exists(
            os.path.join(tmp_path_, "processed", "tile_1_level2_0-0-10-10.png")
        )
        _save_report.assert_called_once_with(
            "report.csv",
            [(0.8, coords), (0.7, coords)],
            [(0.8, coords), (0.7, coords)],
            [f"tile_{i}_level2_0-0-10-10.png" for i in range(2)],
        )
